//! Presenting frames to a window (FIX-1.3; resolves ISSUE-001 together with the window crate's
//! redraw-driven event loop).
//!
//! [`WindowPresenter`] owns a graphics device that can present to the window, the window's
//! swapchain and a [`QuadBatchRenderer`]. Each frame it waits for the frame slot
//! (`GraphicsDevice::begin_frame`), acquires a swapchain image, records the caller's commands
//! into it and presents it.

use crate::backend::GraphicsBackend;
use korppi_render::render2d::{QuadBatch2d, QuadBatchRenderer};
use korppi_rhi::{
    AccessFlags, CommandBufferRecorder, DisplayTarget, FrameIndex, GraphicsDevice, ImageLayout,
    ImageMemoryBarrier, PipelineBarrier, PipelineStageFlags, PresentTarget, RhiError,
    SwapchainDescriptor, SwapchainHandle, SwapchainImage, SwapchainInfo,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn pack_size(width: u32, height: u32) -> u64 {
    ((width as u64) << 32) | (height as u64)
}

fn unpack_size(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, packed as u32)
}

fn create_presenting_device(
    backend: GraphicsBackend,
    target: &Arc<dyn PresentTarget>,
) -> Result<Arc<dyn GraphicsDevice>, RhiError> {
    match backend {
        GraphicsBackend::Vulkan => {
            Ok(Arc::new(korppi_rhi_vulkan::VulkanDevice::create_headless()?))
        }
        GraphicsBackend::OpenGl => {
            let display: Arc<dyn DisplayTarget> = target.clone();
            Ok(Arc::new(
                korppi_rhi_opengl::OpenGlDevice::create_for_display(display)?,
            ))
        }
        GraphicsBackend::Headless => Err(RhiError::Unsupported(
            "the headless backend cannot present to a window".into(),
        )),
    }
}

/// A device, a swapchain for one window, and a renderer for 2D quad batches.
pub struct WindowPresenter {
    device: Arc<dyn GraphicsDevice>,
    swapchain: Option<SwapchainHandle>,
    backend: GraphicsBackend,
    target: Arc<dyn PresentTarget>,
    desc: SwapchainDescriptor,
    requested_size: AtomicU64,
    quads: Option<QuadBatchRenderer>,
}

impl WindowPresenter {
    /// Creates a `backend` device that can present to `target`, and a swapchain for it.
    ///
    /// # Errors
    /// `RhiError::Unsupported` for [`GraphicsBackend::Headless`], or if the backend cannot
    /// present to this window system.
    pub fn new(
        backend: GraphicsBackend,
        target: Arc<dyn PresentTarget>,
        desc: &SwapchainDescriptor,
    ) -> Result<Self, RhiError> {
        let device = create_presenting_device(backend, &target)?;
        let swapchain = device.create_swapchain(target.clone(), desc)?;
        Ok(Self {
            device,
            swapchain: Some(swapchain),
            backend,
            target,
            desc: desc.clone(),
            requested_size: AtomicU64::new(pack_size(desc.width, desc.height)),
            quads: None,
        })
    }

    fn swapchain_handle(&self) -> Result<SwapchainHandle, RhiError> {
        self.swapchain
            .ok_or_else(|| RhiError::InvalidHandle("window swapchain is not active".into()))
    }

    /// Switches the presenting device and swapchain for this window to `backend` at runtime.
    ///
    /// Destroys the current backend's quad renderer and swapchain before binding a new swapchain
    /// to the same native window, avoiding `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR` / `EGL_BAD_ALLOC`.
    ///
    /// # Errors
    /// Returns `RhiError::Unsupported` if `backend` is [`GraphicsBackend::Headless`] (leaving the
    /// existing presenter untouched), or any error from device/swapchain creation.
    pub fn set_backend(&mut self, backend: GraphicsBackend) -> Result<(), RhiError> {
        if backend == GraphicsBackend::Headless {
            return Err(RhiError::Unsupported(
                "the headless backend cannot present to a window".into(),
            ));
        }
        if backend == self.backend && self.swapchain.is_some() {
            return Ok(());
        }

        let new_device = create_presenting_device(backend, &self.target)?;
        let (width, height) = unpack_size(self.requested_size.load(Ordering::Relaxed));
        self.desc.width = width;
        self.desc.height = height;

        self.device.wait_idle()?;
        if let Some(quads) = self.quads.take() {
            quads.destroy(self.device.as_ref())?;
        }
        if let Some(old_swapchain) = self.swapchain.take() {
            self.device.destroy_swapchain(old_swapchain)?;
        }

        let new_swapchain = new_device.create_swapchain(self.target.clone(), &self.desc)?;
        self.device = new_device;
        self.swapchain = Some(new_swapchain);
        self.backend = backend;
        Ok(())
    }

    /// The device presenting to the window.
    pub fn device(&self) -> &Arc<dyn GraphicsDevice> {
        &self.device
    }

    /// The backend of the device.
    pub fn backend(&self) -> GraphicsBackend {
        self.backend
    }

    /// The swapchain's current format, size and presentation mode.
    pub fn info(&self) -> Result<SwapchainInfo, RhiError> {
        self.device.swapchain_info(self.swapchain_handle()?)
    }

    /// Reports a new window size in physical pixels (applied at the next frame).
    pub fn resize(&self, width: u32, height: u32) -> Result<(), RhiError> {
        self.device
            .resize_swapchain(self.swapchain_handle()?, width, height)?;
        self.requested_size
            .store(pack_size(width, height), Ordering::Relaxed);
        Ok(())
    }

    /// Renders and presents one frame: `record` records into the acquired image, which is in
    /// the colour-attachment layout with undefined contents. Returns `false` (and calls nothing)
    /// if no image could be acquired, for example while the window is minimized.
    pub fn render_frame(
        &self,
        record: impl FnOnce(
            &mut dyn CommandBufferRecorder,
            FrameIndex,
            &SwapchainImage,
        ) -> Result<(), RhiError>,
    ) -> Result<bool, RhiError> {
        render_frame(self.device.as_ref(), self.swapchain_handle()?, record)
    }

    /// Clears the window to `clear_color` and draws `batch` (canvas coordinates: `canvas_size`
    /// units span the window, top-left origin; colours as authored sRGB values).
    pub fn present_quads(
        &mut self,
        batch: &QuadBatch2d,
        canvas_size: (f32, f32),
        clear_color: [f32; 4],
    ) -> Result<bool, RhiError> {
        let swapchain = self.swapchain_handle()?;
        let device = self.device.as_ref();
        let format = device.swapchain_info(swapchain)?.format;
        if self.quads.as_ref().map(QuadBatchRenderer::color_format) != Some(format) {
            if let Some(old) = self.quads.take() {
                device.wait_idle()?;
                old.destroy(device)?;
            }
            self.quads = Some(QuadBatchRenderer::new(device, format)?);
        }
        let quads = self.quads.as_mut().expect("created above");
        render_frame(device, swapchain, |recorder, frame, image| {
            quads.record(
                device,
                recorder,
                frame,
                image.texture,
                (image.width, image.height),
                canvas_size,
                clear_color,
                batch,
            )
        })
    }
}

/// See [`WindowPresenter::render_frame`].
fn render_frame(
    device: &dyn GraphicsDevice,
    swapchain: SwapchainHandle,
    record: impl FnOnce(
        &mut dyn CommandBufferRecorder,
        FrameIndex,
        &SwapchainImage,
    ) -> Result<(), RhiError>,
) -> Result<bool, RhiError> {
    let frame = device.begin_frame()?;
    let Some(image) = device.acquire_next_image(swapchain)? else {
        return Ok(false);
    };
    let mut recorder = device.create_command_recorder()?;
    recorder.begin()?;
    recorder.pipeline_barrier(
        &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture: image.texture,
            range: korppi_rhi::TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::ColorAttachment,
        }),
    );
    let recorded = record(recorder.as_mut(), frame, &image);
    // Present even if recording failed, so the acquired image goes back to the window system;
    // the recording error is reported afterwards.
    let submitted = recorded.and_then(|()| device.submit(recorder.finish()?).map(|_| ()));
    device.present(&image)?;
    device.end_frame()?;
    submitted.map(|()| true)
}

impl Drop for WindowPresenter {
    fn drop(&mut self) {
        if let Err(error) = self.device.wait_idle() {
            tracing::error!("waiting for the GPU before closing the window: {error}");
        }
        if let Some(quads) = self.quads.take() {
            if let Err(error) = quads.destroy(self.device.as_ref()) {
                tracing::error!("destroying the quad renderer: {error}");
            }
        }
        if let Some(swapchain) = self.swapchain.take() {
            if let Err(error) = self.device.destroy_swapchain(swapchain) {
                tracing::error!("destroying the swapchain: {error}");
            }
        }
    }
}

/// An app that draws each frame into a [`QuadBatch2d`] in logical window coordinates.
pub trait QuadBatchSource {
    /// The batch to present for the current frame.
    fn quad_batch(&self) -> &QuadBatch2d;
}

/// How [`QuadBatchPresentation`] presents frames.
#[derive(Clone, Debug)]
pub struct PresentOptions {
    /// Backend of the presenting device.
    pub backend: GraphicsBackend,
    /// Presentation mode of the swapchain.
    pub present_mode: korppi_rhi::PresentMode,
    /// Colour the window is cleared to before the batch is drawn (authored sRGB).
    pub clear_color: [f32; 4],
    /// Close the window after this many presented frames (smoke tests, benchmarks).
    pub frame_limit: Option<u32>,
}

/// Environment variable selecting the backend when `--backend` is not given.
pub const BACKEND_ENV_VAR: &str = "KORPPI_BACKEND";

impl PresentOptions {
    /// Reads `--backend vulkan|opengl` (default: `$KORPPI_BACKEND`, then Vulkan),
    /// `--present-mode fifo|mailbox|immediate|fifo-relaxed` and `--frames N` from command-line
    /// arguments; other arguments are ignored.
    ///
    /// # Errors
    /// Describes an unknown backend or present mode, or a frame count that is not a number.
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        let mut options = Self::default();
        if let Ok(name) = std::env::var(BACKEND_ENV_VAR) {
            options.backend = parse_backend(&name)?;
        }
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            let mut value = |flag: &str| {
                args.next()
                    .cloned()
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match arg.as_str() {
                "--backend" => options.backend = parse_backend(&value("--backend")?)?,
                "--frames" => {
                    let count = value("--frames")?;
                    options.frame_limit = Some(
                        count
                            .parse()
                            .map_err(|_| format!("--frames: `{count}` is not a frame count"))?,
                    );
                }
                "--present-mode" => {
                    options.present_mode = match value("--present-mode")?.as_str() {
                        "fifo" => korppi_rhi::PresentMode::Fifo,
                        "fifo-relaxed" => korppi_rhi::PresentMode::FifoRelaxed,
                        "mailbox" => korppi_rhi::PresentMode::Mailbox,
                        "immediate" => korppi_rhi::PresentMode::Immediate,
                        other => return Err(format!("unknown present mode `{other}`")),
                    }
                }
                _ => {}
            }
        }
        Ok(options)
    }
}

fn parse_backend(name: &str) -> Result<GraphicsBackend, String> {
    match name.to_ascii_lowercase().as_str() {
        "vulkan" | "vk" => Ok(GraphicsBackend::Vulkan),
        "opengl" | "gl" => Ok(GraphicsBackend::OpenGl),
        other => Err(format!(
            "unknown backend `{other}` (expected `vulkan` or `opengl`)"
        )),
    }
}

/// Runs `app` in a window (blocking the calling thread, which must be the main thread on most
/// platforms), presenting its quad batch after every update. Returns the number of frames
/// presented.
///
/// # Errors
/// The event loop or presentation failed.
pub fn run_windowed<A: korppi_window::KorppiApp + QuadBatchSource>(
    app: A,
    config: korppi_window::WindowConfig,
    options: PresentOptions,
) -> Result<u32, String> {
    let presentation = QuadBatchPresentation::new(app, options);
    let presentation = korppi_window::AppRunner::new(presentation, config)
        .run()
        .map_err(|e| e.to_string())?;
    match presentation.failure() {
        Some(failure) => Err(failure.to_owned()),
        None => Ok(presentation.presented_frames()),
    }
}

impl Default for PresentOptions {
    fn default() -> Self {
        Self {
            backend: GraphicsBackend::Vulkan,
            present_mode: korppi_rhi::PresentMode::Fifo,
            clear_color: [0.05, 0.06, 0.08, 1.0],
            frame_limit: None,
        }
    }
}

/// Wraps a [`KorppiApp`] that renders into a [`QuadBatch2d`] and presents that batch to the
/// window after every update.
pub struct QuadBatchPresentation<A> {
    app: A,
    options: PresentOptions,
    presenter: Option<WindowPresenter>,
    presented: u32,
    failure: Option<String>,
}

impl<A> QuadBatchPresentation<A> {
    /// Wraps `app`.
    pub fn new(app: A, options: PresentOptions) -> Self {
        Self {
            app,
            options,
            presenter: None,
            presented: 0,
            failure: None,
        }
    }

    /// Frames presented so far.
    pub fn presented_frames(&self) -> u32 {
        self.presented
    }

    /// The error that stopped presentation, if any.
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    fn fail(&mut self, window: &mut korppi_window::Window, error: impl std::fmt::Display) {
        let message = format!(
            "presentation with {:?} failed: {error}",
            self.options.backend
        );
        tracing::error!("{message}");
        self.failure = Some(message);
        self.presenter = None;
        window.request_close();
    }
}

impl<A: korppi_window::KorppiApp + QuadBatchSource> korppi_window::KorppiApp
    for QuadBatchPresentation<A>
{
    fn on_start(&mut self, window: &mut korppi_window::Window) {
        self.app.on_start(window);
        let Some(target) = window.shared_winit_window() else {
            return self.fail(window, "the window has no OS window to present to");
        };
        let desc = SwapchainDescriptor {
            present_mode: self.options.present_mode,
            ..SwapchainDescriptor::new(window.physical_width(), window.physical_height())
        };
        match WindowPresenter::new(self.options.backend, target, &desc) {
            Ok(presenter) => {
                if let Ok(info) = presenter.info() {
                    tracing::info!(
                        "presenting with {:?}: {:?} {}x{}, {:?}",
                        self.options.backend,
                        info.format,
                        info.width,
                        info.height,
                        info.present_mode
                    );
                }
                self.presenter = Some(presenter);
            }
            Err(error) => self.fail(window, error),
        }
    }

    fn on_update(&mut self, window: &mut korppi_window::Window, delta_time: f32) {
        self.app.on_update(window, delta_time);
        for _ in window.drain_events() {}
        let Some(presenter) = self.presenter.as_mut() else {
            return;
        };
        let canvas = (window.width() as f32, window.height() as f32);
        match presenter.present_quads(self.app.quad_batch(), canvas, self.options.clear_color) {
            Ok(true) => self.presented += 1,
            Ok(false) => {}
            Err(error) => return self.fail(window, error),
        }
        if self
            .options
            .frame_limit
            .is_some_and(|limit| self.presented >= limit)
        {
            window.request_close();
        }
    }

    fn on_event(&mut self, window: &mut korppi_window::Window, event: &korppi_window::WindowEvent) {
        self.app.on_event(window, event);
        if let korppi_window::WindowEvent::Resized { .. } = event {
            let size = (window.physical_width(), window.physical_height());
            if let Some(Err(error)) = self.presenter.as_ref().map(|p| p.resize(size.0, size.1)) {
                self.fail(window, error);
            }
        }
    }

    fn on_stop(&mut self) {
        self.app.on_stop();
        // Release the device and swapchain before the window goes away.
        self.presenter = None;
    }
}
