//! Presentation for the OpenGL backend (FIX-1.3; findings RHI-02, RGL-07).
//!
//! A swapchain is an EGL window surface plus [`SWAPCHAIN_IMAGE_COUNT`] ordinary textures of the
//! device. Applications render into the acquired texture like into any render target; `present`
//! then blits it to the window's default framebuffer and swaps buffers, on the render thread
//! after every earlier submission. The blit flips vertically: render targets store their top row
//! first (`GL_UPPER_LEFT`, ADR-013), window framebuffers their bottom row. It also copies sRGB
//! texels unchanged (`GL_FRAMEBUFFER_SRGB` off), so the window shows the encoded values just as a
//! Vulkan sRGB swapchain does.
//!
//! The image bookkeeping (which image is acquired, the requested size) lives with the device on
//! the calling thread, so acquiring and presenting do not wait for the render thread; only
//! creating, recreating and destroying swapchains do.

use crate::context::GlWindowSurface;
use crate::device::{GlThread, GlWork, OpenGlDevice};
use glow::HasContext;
use korppi_rhi::lock;
use korppi_rhi::SampleCount;
use korppi_rhi::{
    PresentMode, PresentTarget, RhiError, SwapchainDescriptor, SwapchainHandle, SwapchainImage,
    SwapchainInfo, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle, TextureUsage,
};
use raw_window_handle::RawWindowHandle;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Number of textures per swapchain.
pub const SWAPCHAIN_IMAGE_COUNT: u32 = 2;

/// Usage of swapchain textures.
const SWAPCHAIN_USAGE: TextureUsage = TextureUsage::COLOR_ATTACHMENT
    .union(TextureUsage::COPY_SRC)
    .union(TextureUsage::COPY_DST);

/// Moves raw window-system handles to the render thread.
pub(crate) struct SendHandle<T>(pub(crate) T);

// SAFETY: the wrapped values are plain handles (pointers and IDs) whose objects are kept alive by
// the caller (an `Arc` of the window or display), and EGL accepts them from any thread; on X11
// the connection was opened after `XInitThreads` (winit and the test harness do this).
unsafe impl<T> Send for SendHandle<T> {}

/// Caller-side state of a swapchain.
pub(crate) struct GlSwapchainState {
    info: SwapchainInfo,
    textures: Vec<TextureHandle>,
    requested: (u32, u32),
    needs_recreate: bool,
    next: u32,
    acquired: Option<u32>,
}

/// Render-thread side of a swapchain: the window surface and what presenting needs.
pub(crate) struct GlPresentSurface {
    /// Keeps the window alive until the surface is destroyed.
    target: Arc<dyn PresentTarget>,
    surface: GlWindowSurface,
    read_framebuffer: glow::Framebuffer,
    swap_interval: i32,
    applied_interval: Option<i32>,
}

/// EGL swap interval for a present mode, and the mode it provides. EGL has no tearing vsync
/// (FIFO_RELAXED) or mailbox mode, so those fall back to FIFO.
fn swap_interval(mode: PresentMode) -> (i32, PresentMode) {
    match mode {
        PresentMode::Immediate => (0, PresentMode::Immediate),
        PresentMode::Fifo | PresentMode::FifoRelaxed | PresentMode::Mailbox => {
            (1, PresentMode::Fifo)
        }
    }
}

impl GlThread {
    /// Creates `count` swapchain textures of `width` x `height` owned by swapchain `id`.
    fn create_swapchain_textures(
        &mut self,
        id: u32,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Vec<TextureHandle>, RhiError> {
        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }
        let desc = TextureDescriptor {
            width,
            height,
            depth: 1,
            array_layers: 1,
            mip_levels: 1,
            sample_count: SampleCount::Count1,
            format,
            usage: SWAPCHAIN_USAGE,
            dimension: TextureDimension::D2,
            label: Some("swapchain_image"),
        };
        let mut textures = Vec::with_capacity(SWAPCHAIN_IMAGE_COUNT as usize);
        for _ in 0..SWAPCHAIN_IMAGE_COUNT {
            let texture = self.create_texture(&desc, None)?;
            if let Some((entry, _)) = lock(&self.resources.textures).get_mut(texture) {
                entry.swapchain = Some(id);
            }
            textures.push(texture);
        }
        Ok(textures)
    }

    fn destroy_swapchain_textures(&mut self, textures: &[TextureHandle]) {
        for texture in textures {
            let removed = lock(&self.resources.textures).remove(texture);
            if let Some((entry, _)) = removed {
                // Earlier commands that used the texture have been issued, and GL keeps it
                // alive until they finish.
                self.delete_texture_object(Some(texture.id()), entry.texture);
            }
        }
    }

    fn create_swapchain(
        &mut self,
        id: u32,
        target: Arc<dyn PresentTarget>,
        window: SendHandle<RawWindowHandle>,
        desc: &SwapchainDescriptor,
        format: TextureFormat,
    ) -> Result<Vec<TextureHandle>, RhiError> {
        // SAFETY: `target` keeps the window alive until the surface is destroyed (it is stored
        // with the surface).
        let surface = unsafe {
            self.context
                .create_window_surface(window.0, desc.width, desc.height)
        }?;
        // SAFETY: the context is current on the render thread.
        let read_framebuffer = match unsafe { self.context.gl.create_named_framebuffer() } {
            Ok(framebuffer) => framebuffer,
            Err(error) => {
                self.context.destroy_window_surface(surface);
                return Err(RhiError::BackendError(format!(
                    "glCreateFramebuffers failed: {error}"
                )));
            }
        };
        let textures = match self.create_swapchain_textures(id, format, desc.width, desc.height) {
            Ok(textures) => textures,
            Err(error) => {
                // SAFETY: the framebuffer was created above and is unused.
                unsafe { self.context.gl.delete_framebuffer(read_framebuffer) };
                self.context.destroy_window_surface(surface);
                return Err(error);
            }
        };
        self.surfaces.insert(
            id,
            GlPresentSurface {
                target,
                surface,
                read_framebuffer,
                swap_interval: swap_interval(desc.present_mode).0,
                applied_interval: None,
            },
        );
        Ok(textures)
    }

    fn recreate_swapchain(
        &mut self,
        id: u32,
        old: &[TextureHandle],
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<Vec<TextureHandle>, RhiError> {
        self.destroy_swapchain_textures(old);
        if let Some(surface) = self.surfaces.get(&id) {
            self.context
                .resize_window_surface(&surface.surface, width, height);
        }
        self.create_swapchain_textures(id, format, width, height)
    }

    /// Destroys swapchain `id`'s surface and textures and returns its window.
    fn destroy_swapchain(
        &mut self,
        id: u32,
        textures: &[TextureHandle],
    ) -> Option<Arc<dyn PresentTarget>> {
        self.destroy_swapchain_textures(textures);
        let surface = self.surfaces.remove(&id)?;
        // SAFETY: the context is current on the render thread.
        unsafe { self.context.gl.delete_framebuffer(surface.read_framebuffer) };
        self.context.destroy_window_surface(surface.surface);
        Some(surface.target)
    }

    /// Destroys every window surface (device teardown).
    pub(crate) fn destroy_all_surfaces(&mut self) {
        let ids: Vec<u32> = self.surfaces.keys().copied().collect();
        for id in ids {
            self.destroy_swapchain(id, &[]);
        }
    }

    /// Blits `texture` to swapchain `id`'s window, flipped vertically, and swaps buffers.
    pub(crate) fn present(&mut self, id: u32, texture: TextureHandle) -> Result<(), RhiError> {
        let (gl_texture, width, height) = {
            let textures = lock(&self.resources.textures);
            let (entry, _) = textures.get(texture).ok_or_else(|| {
                RhiError::InvalidHandle(format!("swapchain texture {texture:?} not found"))
            })?;
            (entry.texture, entry.width as i32, entry.height as i32)
        };
        let surface = self
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| RhiError::InvalidHandle(format!("swapchain {id} has no surface")))?;
        self.context.make_current(Some(&surface.surface))?;
        if surface.applied_interval != Some(surface.swap_interval) {
            if let Err(error) = self.context.set_swap_interval(surface.swap_interval) {
                tracing::warn!("{error}; keeping the driver's swap interval");
            }
            surface.applied_interval = Some(surface.swap_interval);
        }
        // Blit to the surface's current size: the window may have been resized before the
        // swapchain was.
        let (dst_width, dst_height) = self
            .context
            .surface_size(&surface.surface)
            .map_or((width, height), |(w, h)| (w as i32, h as i32));
        let filter = if (dst_width, dst_height) == (width, height) {
            glow::NEAREST
        } else {
            glow::LINEAR
        };
        let read_framebuffer = surface.read_framebuffer;
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread with the window surface as its
        // draw surface; the read framebuffer and texture are live. Blits are only affected by
        // the scissor test and sRGB conversion, both disabled here (the state cache is told
        // about the scissor change; FRAMEBUFFER_SRGB is restored right after).
        unsafe {
            gl.named_framebuffer_texture(
                Some(read_framebuffer),
                glow::COLOR_ATTACHMENT0,
                Some(gl_texture),
                0,
            );
            gl.named_framebuffer_read_buffer(Some(read_framebuffer), glow::COLOR_ATTACHMENT0);
            // The default framebuffer's draw buffer is initialized when the context is first
            // made current; that was without a window (surfaceless), so it is GL_NONE.
            gl.named_framebuffer_draw_buffer(None, glow::BACK);
            self.state.disable_scissor(gl);
            gl.disable(glow::FRAMEBUFFER_SRGB);
            gl.blit_named_framebuffer(
                Some(read_framebuffer),
                None,
                0,
                0,
                width,
                height,
                0,
                dst_height,
                dst_width,
                0,
                glow::COLOR_BUFFER_BIT,
                filter,
            );
            gl.enable(glow::FRAMEBUFFER_SRGB);
        }
        let surface = &self.surfaces[&id];
        self.context.swap_buffers(&surface.surface)
    }
}

impl OpenGlDevice {
    pub(crate) fn create_swapchain_impl(
        &self,
        target: Arc<dyn PresentTarget>,
        desc: &SwapchainDescriptor,
    ) -> Result<SwapchainHandle, RhiError> {
        if self.display.is_none() {
            return Err(RhiError::Unsupported(
                "this OpenGL device was created headless; create it with \
                 OpenGlDevice::create_for_display to present"
                    .into(),
            ));
        }
        let window = target
            .window_handle()
            .map_err(|e| RhiError::SwapchainError(format!("no window handle: {e}")))?
            .as_raw();
        let window = SendHandle(window);
        let format = if desc.srgb {
            TextureFormat::Rgba8Srgb
        } else {
            TextureFormat::Rgba8Unorm
        };
        let id = self.next_swapchain_id.fetch_add(1, Ordering::Relaxed);
        let thread_target = Arc::clone(&target);
        let textures = self.thread.call(move |gl| {
            let window = window;
            gl.create_swapchain(id, thread_target, window, desc, format)
        })??;
        let (width, height) = if textures.is_empty() {
            (0, 0)
        } else {
            (desc.width, desc.height)
        };
        let state = GlSwapchainState {
            info: SwapchainInfo {
                format,
                usage: SWAPCHAIN_USAGE,
                width,
                height,
                present_mode: swap_interval(desc.present_mode).1,
                image_count: textures.len() as u32,
            },
            textures,
            requested: (desc.width, desc.height),
            needs_recreate: false,
            next: 0,
            acquired: None,
        };
        lock(&self.swapchains).insert(id, state);
        Ok(SwapchainHandle::from_parts(self.resources.device, id, 1))
    }

    /// The id of a swapchain handle of this device; a handle of another device is rejected
    /// (RHI-15), since swapchains are kept by id, not in a slot map.
    fn swapchain_key(&self, handle: SwapchainHandle) -> Result<u32, RhiError> {
        if handle.device() == self.resources.device {
            Ok(handle.id())
        } else {
            Err(RhiError::InvalidHandle(format!(
                "swapchain {handle:?} belongs to another device"
            )))
        }
    }

    pub(crate) fn destroy_swapchain_impl(&self, handle: SwapchainHandle) -> Result<(), RhiError> {
        let state = lock(&self.swapchains)
            .remove(&self.swapchain_key(handle)?)
            .ok_or_else(|| RhiError::InvalidHandle(format!("swapchain {handle:?} not found")))?;
        let target = self
            .thread
            .call(move |gl| gl.destroy_swapchain(handle.id(), &state.textures))?;
        // Release the window on the calling thread, not the render thread.
        drop(target);
        Ok(())
    }

    pub(crate) fn resize_swapchain_impl(
        &self,
        handle: SwapchainHandle,
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        let mut swapchains = lock(&self.swapchains);
        let state = swapchains
            .get_mut(&self.swapchain_key(handle)?)
            .ok_or_else(|| RhiError::InvalidHandle(format!("swapchain {handle:?} not found")))?;
        if (state.info.width, state.info.height) != (width, height) {
            state.requested = (width, height);
            state.needs_recreate = true;
        }
        Ok(())
    }

    pub(crate) fn swapchain_info_impl(
        &self,
        handle: SwapchainHandle,
    ) -> Result<SwapchainInfo, RhiError> {
        let mut swapchains = lock(&self.swapchains);
        let state = swapchains
            .get_mut(&self.swapchain_key(handle)?)
            .ok_or_else(|| RhiError::InvalidHandle(format!("swapchain {handle:?} not found")))?;
        Ok(state.info)
    }

    pub(crate) fn acquire_impl(
        &self,
        handle: SwapchainHandle,
    ) -> Result<Option<SwapchainImage>, RhiError> {
        let mut swapchains = lock(&self.swapchains);
        let state = swapchains
            .get_mut(&self.swapchain_key(handle)?)
            .ok_or_else(|| RhiError::InvalidHandle(format!("swapchain {handle:?} not found")))?;
        if state.acquired.is_some() {
            return Err(RhiError::ValidationError(format!(
                "swapchain {handle:?}: the acquired image must be presented before acquiring another"
            )));
        }
        if state.needs_recreate {
            state.needs_recreate = false;
            let (width, height) = state.requested;
            let old = std::mem::take(&mut state.textures);
            let format = state.info.format;
            let id = handle.id();
            // The render thread never takes the swapchain lock, so calling it here is safe.
            let textures = self
                .thread
                .call(move |gl| gl.recreate_swapchain(id, &old, format, width, height))??;
            state.info.image_count = textures.len() as u32;
            (state.info.width, state.info.height) = if textures.is_empty() {
                (0, 0)
            } else {
                (width, height)
            };
            state.textures = textures;
            state.next = 0;
        }
        if state.textures.is_empty() {
            return Ok(None);
        }
        let index = state.next;
        state.next = (index + 1) % state.textures.len() as u32;
        state.acquired = Some(index);
        Ok(Some(SwapchainImage {
            swapchain: handle,
            index,
            texture: state.textures[index as usize],
            width: state.info.width,
            height: state.info.height,
        }))
    }

    pub(crate) fn present_impl(&self, image: &SwapchainImage) -> Result<(), RhiError> {
        {
            let mut swapchains = lock(&self.swapchains);
            let state = swapchains
                .get_mut(&self.swapchain_key(image.swapchain)?)
                .ok_or_else(|| {
                    RhiError::InvalidHandle(format!("swapchain {:?} not found", image.swapchain))
                })?;
            let acquired = state.acquired == Some(image.index)
                && state.textures.get(image.index as usize) == Some(&image.texture);
            if !acquired {
                return Err(RhiError::ValidationError(format!(
                    "swapchain {:?} image {} is not acquired (already presented, or from before \
                     a recreation)",
                    image.swapchain, image.index
                )));
            }
            state.acquired = None;
        }
        let swapchain = image.swapchain.id();
        let texture = image.texture;
        self.post(|submission| GlWork::Present {
            submission,
            swapchain,
            texture,
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_modes_map_to_swap_intervals() {
        assert_eq!(swap_interval(PresentMode::Fifo), (1, PresentMode::Fifo));
        assert_eq!(
            swap_interval(PresentMode::Immediate),
            (0, PresentMode::Immediate)
        );
        assert_eq!(swap_interval(PresentMode::Mailbox), (1, PresentMode::Fifo));
        assert_eq!(
            swap_interval(PresentMode::FifoRelaxed),
            (1, PresentMode::Fifo)
        );
    }
}
