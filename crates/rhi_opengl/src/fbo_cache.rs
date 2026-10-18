//! Framebuffer Object (FBO) cache for dynamic rendering in OpenGL (finding RGL-14).
//!
//! FBOs are keyed by their attachments (texture, mip, layer), so a pass reuses the FBO of an
//! earlier identical pass. Keys are fixed-size, so looking one up allocates nothing. Layered
//! textures (arrays, cube maps, 3D) attach one layer with `glNamedFramebufferTextureLayer`;
//! depth/stencil formats attach at `GL_DEPTH_STENCIL_ATTACHMENT`. New FBOs are checked for
//! completeness, and FBOs unused for a while are evicted once the cache is full.

use crate::resource::GlTexture;
use glow::HasContext;
use korppi_rhi::{RenderingDescriptor, RhiError};
use std::collections::HashMap;

/// Colour attachments an FBO can have (the RHI's validated maximum).
pub const MAX_FBO_COLOR_ATTACHMENTS: usize = 8;

/// FBOs kept before unused ones are evicted.
pub const DEFAULT_FBO_CAPACITY: usize = 64;

/// One attachment: (texture ID, mip level, layer).
pub type FboAttachment = (u32, u32, u32);

/// Cache key uniquely identifying an FBO attachment configuration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FboKey {
    /// Colour attachments in order.
    pub color_attachments: [Option<FboAttachment>; MAX_FBO_COLOR_ATTACHMENTS],
    /// Depth/stencil attachment.
    pub depth_attachment: Option<FboAttachment>,
}

impl FboKey {
    /// The key of a dynamic rendering pass.
    ///
    /// # Errors
    /// `RhiError::Unsupported` for more than [`MAX_FBO_COLOR_ATTACHMENTS`] colour attachments.
    pub fn from_descriptor(desc: &RenderingDescriptor) -> Result<Self, RhiError> {
        if desc.color_attachments.len() > MAX_FBO_COLOR_ATTACHMENTS {
            return Err(RhiError::Unsupported(format!(
                "{} colour attachments; at most {MAX_FBO_COLOR_ATTACHMENTS} are supported",
                desc.color_attachments.len()
            )));
        }
        let mut color_attachments = [None; MAX_FBO_COLOR_ATTACHMENTS];
        for (slot, att) in color_attachments.iter_mut().zip(&desc.color_attachments) {
            *slot = Some((att.texture.id(), att.mip_level, att.layer));
        }
        Ok(Self {
            color_attachments,
            depth_attachment: desc
                .depth_attachment
                .as_ref()
                .map(|att| (att.texture.id(), att.mip_level, att.layer)),
        })
    }

    /// The key of an FBO with one colour attachment (resolve targets, blits).
    pub fn single_color(attachment: FboAttachment) -> Self {
        let mut color_attachments = [None; MAX_FBO_COLOR_ATTACHMENTS];
        color_attachments[0] = Some(attachment);
        Self {
            color_attachments,
            depth_attachment: None,
        }
    }

    fn references(&self, texture_id: u32) -> bool {
        self.color_attachments
            .iter()
            .flatten()
            .chain(self.depth_attachment.iter())
            .any(|(id, _, _)| *id == texture_id)
    }
}

/// Attaches mip `mip`, layer `layer` of `texture` at `point` of `fbo`.
///
/// # Safety
/// The context must be current; `fbo` and the texture must be live.
unsafe fn attach(
    gl: &glow::Context,
    fbo: glow::Framebuffer,
    point: u32,
    texture: &GlTexture,
    mip: u32,
    layer: u32,
) {
    // SAFETY: forwarded from the caller.
    unsafe {
        if texture.is_layered() {
            gl.named_framebuffer_texture_layer(
                Some(fbo),
                point,
                Some(texture.texture),
                mip as i32,
                layer as i32,
            );
        } else {
            gl.named_framebuffer_texture(Some(fbo), point, Some(texture.texture), mip as i32);
        }
    }
}

/// Framebuffer Object cache with least-recently-used eviction.
pub struct GlFboCache {
    entries: HashMap<FboKey, (glow::Framebuffer, u64)>,
    current_generation: u64,
    capacity: usize,
}

impl Default for GlFboCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            current_generation: 0,
            capacity: DEFAULT_FBO_CAPACITY,
        }
    }
}

impl GlFboCache {
    /// Creates a new empty FBO cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty cache that keeps at most `capacity` FBOs between evictions.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            ..Self::default()
        }
    }

    /// Number of active cached FBOs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache has zero entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Advances the generation (one per executed command buffer); FBOs used in the current
    /// generation are never evicted.
    pub fn advance_generation(&mut self) {
        self.current_generation += 1;
    }

    /// Deletes the least recently used FBOs beyond the capacity and returns them, so that the
    /// state cache can forget them.
    pub fn evict(&mut self, gl: &glow::Context) -> Vec<glow::Framebuffer> {
        if self.entries.len() <= self.capacity {
            return Vec::new();
        }
        let mut by_age: Vec<(u64, FboKey)> = self
            .entries
            .iter()
            .filter(|(_, (_, used))| *used < self.current_generation)
            .map(|(key, (_, used))| (*used, *key))
            .collect();
        by_age.sort_by_key(|(used, _)| *used);
        let excess = self.entries.len() - self.capacity;
        let mut evicted = Vec::with_capacity(excess);
        for (_, key) in by_age.into_iter().take(excess) {
            if let Some((fbo, _)) = self.entries.remove(&key) {
                // SAFETY: the context is current (caller); the FBO is not used by the pass
                // being recorded (it was not used in the current generation).
                unsafe { gl.delete_framebuffer(fbo) };
                evicted.push(fbo);
            }
        }
        evicted
    }

    /// Gets the FBO for `key`, creating, attaching and checking it if needed.
    ///
    /// # Errors
    /// `RhiError::InvalidHandle` for unknown textures, `RhiError::BackendError` if the
    /// framebuffer is incomplete.
    pub fn get_or_create(
        &mut self,
        gl: &glow::Context,
        key: &FboKey,
        textures: &korppi_rhi::SlotMap<GlTexture>,
    ) -> Result<glow::Framebuffer, RhiError> {
        if let Some((fbo, last_used)) = self.entries.get_mut(key) {
            *last_used = self.current_generation;
            return Ok(*fbo);
        }
        let lookup = |id: u32| {
            // Keys hold IDs of live textures: entries are invalidated when a texture dies.
            textures
                .get_by_index(id)
                .map(|(texture, _)| texture)
                .ok_or_else(|| {
                    RhiError::InvalidHandle(format!("attachment texture {id} not found"))
                })
        };
        // SAFETY: the context is current on the render thread (caller); every texture comes
        // from the registry and is live.
        let fbo = unsafe {
            let fbo = gl
                .create_named_framebuffer()
                .map_err(|e| RhiError::BackendError(format!("glCreateFramebuffers failed: {e}")))?;
            let mut draw_buffers = [glow::NONE; MAX_FBO_COLOR_ATTACHMENTS];
            let mut count = 0;
            let result = (|| {
                for (i, attachment) in key.color_attachments.iter().enumerate() {
                    if let Some((id, mip, layer)) = *attachment {
                        let point = glow::COLOR_ATTACHMENT0 + i as u32;
                        attach(gl, fbo, point, lookup(id)?, mip, layer);
                        draw_buffers[i] = point;
                        count = i + 1;
                    }
                }
                gl.named_framebuffer_draw_buffers(Some(fbo), &draw_buffers[..count.max(1)]);
                if count == 0 {
                    gl.named_framebuffer_read_buffer(Some(fbo), glow::NONE);
                }
                if let Some((id, mip, layer)) = key.depth_attachment {
                    let texture = lookup(id)?;
                    let point = if texture.format.is_stencil() {
                        glow::DEPTH_STENCIL_ATTACHMENT
                    } else {
                        glow::DEPTH_ATTACHMENT
                    };
                    attach(gl, fbo, point, texture, mip, layer);
                }
                let status = gl.check_named_framebuffer_status(Some(fbo), glow::DRAW_FRAMEBUFFER);
                if status != glow::FRAMEBUFFER_COMPLETE {
                    return Err(RhiError::BackendError(format!(
                        "framebuffer {key:?} is incomplete (status {status:#x})"
                    )));
                }
                Ok(())
            })();
            if let Err(error) = result {
                gl.delete_framebuffer(fbo);
                return Err(error);
            }
            fbo
        };
        self.entries.insert(*key, (fbo, self.current_generation));
        Ok(fbo)
    }

    /// Removes and deletes any FBOs that reference a destroyed texture ID, and returns them so
    /// that the state cache can forget them.
    pub fn invalidate_texture(
        &mut self,
        gl: &glow::Context,
        texture_id: u32,
    ) -> Vec<glow::Framebuffer> {
        let stale: Vec<FboKey> = self
            .entries
            .keys()
            .filter(|key| key.references(texture_id))
            .copied()
            .collect();
        let mut deleted = Vec::with_capacity(stale.len());
        for key in stale {
            if let Some((fbo, _)) = self.entries.remove(&key) {
                // SAFETY: the context is current on the render thread (caller).
                unsafe { gl.delete_framebuffer(fbo) };
                deleted.push(fbo);
            }
        }
        deleted
    }

    /// Purges all cached framebuffers upon device teardown.
    pub fn destroy_all(&mut self, gl: &glow::Context) {
        for (_, (fbo, _)) in self.entries.drain() {
            // SAFETY: the context is current on the render thread (caller).
            unsafe { gl.delete_framebuffer(fbo) };
        }
    }
}
