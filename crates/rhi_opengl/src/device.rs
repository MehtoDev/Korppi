//! Concrete OpenGL 4.5 Core Direct State Access graphics device implementation.
//!
//! Every OpenGL call runs on the device's render thread ([`crate::render_thread`]), which owns
//! the context, the state and FBO caches and the GL object names (RGL-08, RGL-11). The
//! `GraphicsDevice` methods validate their arguments on the calling thread and then run the GL
//! work there: resource calls block until done, `submit` queues the command buffer and returns.
//!
//! Each submission is followed by a `glFenceSync` fence on the render thread; submission
//! completion is tracked through those fences (RHI-02, RGL-16).

use crate::binding::{apply_bind_groups, BoundGroup, GlBindingTable};
use crate::capabilities::GlCapabilities;
use crate::command::{
    CommandStreamPool, GlCommand, GlCommandBuffer, GlCommandBufferRecorder, GlCommandStream,
};
use crate::context::{GlContext, GlPlatform};
use crate::fbo_cache::{FboKey, GlFboCache, MAX_FBO_COLOR_ATTACHMENTS};
use crate::gl_fns::GlExtraFns;
use crate::render_thread::{RenderThread, ThreadState};
use crate::resource::{
    GlBuffer, GlPipeline, GlResourceRegistry, GlSampler, GlShader, GlShaderCode, MappedPtr,
    MAX_VERTEX_BINDINGS,
};
use crate::spirv::{parse_spirv_words, SpirvShader};
use crate::state_cache::{self, GlBlend, GlDepthBias, GlStateCache, GlStencilFace};
use crate::swapchain::{GlPresentSurface, GlSwapchainState, SendHandle};
use crate::types::{
    border_color_rgba, to_gl_address_mode, to_gl_blend_factor, to_gl_blend_op, to_gl_compare_func,
    to_gl_cull_mode, to_gl_filter, to_gl_front_face, to_gl_index_type, to_gl_min_filter,
    to_gl_polygon_mode, to_gl_shader_stage, to_gl_stencil_op, to_gl_topology, to_gl_vertex_format,
};
use glow::HasContext;
use korppi_rhi::lock;
use korppi_rhi::validation::{validate_recorder_pooled, ValidatingRecorder};
use korppi_rhi::{
    check_pipeline_layouts, BindGroupDescriptor, BindGroupHandle, BindGroupLayoutDescriptor,
    BindGroupLayoutHandle, BindGroupLayoutInfo, BoundResource, BoxPool, FilterMode, MappedBuffer,
    MemoryUsage, ShaderStage, ShaderStages, MAX_BIND_GROUPS,
};
use korppi_rhi::{
    AdapterInfo, DeviceFeatures, DeviceLimits, DisplayTarget, FormatCapabilities, PresentTarget,
    TextureFormat, TextureUsage, TextureViewDescriptor, TextureViewHandle,
};
use korppi_rhi::{
    BackendDiagnostic, BufferDescriptor, BufferHandle, ColorWriteMask, CommandBuffer,
    CommandBufferRecorder, ComputePipelineDescriptor, DiagnosticLog, DiagnosticSeverity,
    FrameIndex, FrameTracker, GraphicsDevice, GraphicsPipelineDescriptor, IndexType, LoadOp,
    MultisampleState, PipelineHandle, PipelineSignature, RasterizerState, RhiError,
    SamplerDescriptor, SamplerHandle, ShaderDescriptor, ShaderHandle, ShaderSource, SubmissionId,
    SwapchainDescriptor, SwapchainHandle, SwapchainImage, SwapchainInfo, TextureDescriptor,
    TextureHandle, VertexLayoutDescriptor, VertexStepMode, MAX_PUSH_CONSTANTS_SIZE,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Concrete OpenGL 4.5 Core Direct State Access (DSA) graphics device.
///
/// The device is `Send + Sync` and may be used from any thread: the EGL context is current only
/// on the device's own render thread, which executes every GL call (RGL-08).
pub struct OpenGlDevice {
    pub(crate) thread: RenderThread<GlThread>,
    pub(crate) resources: Arc<GlResourceRegistry>,
    diagnostics: Arc<DiagnosticLog>,
    /// ID of the last submission; locked while posting so IDs follow queue order.
    last_issued: Mutex<u64>,
    /// ID of the last submission whose fence has signalled (written by the render thread).
    completed: Arc<AtomicU64>,
    pub(crate) frames: FrameTracker,
    debug_output_enabled: bool,
    /// Whether `GL_ARB_shader_draw_parameters` (core in 4.6) is available to translated SPIR-V.
    draw_parameters: bool,
    /// Adapter information, limits, features and format support (RHI-09).
    capabilities: GlCapabilities,
    /// Executed command streams, reused by new recorders (RHI-11).
    command_streams: CommandStreamPool,
    /// Finished recorders, reused by `create_command_recorder`.
    recorder_pool: Arc<BoxPool<GlCommandBufferRecorder>>,
    /// Submitted command buffer boxes, reused by `finish`.
    command_buffer_pool: Arc<BoxPool<GlCommandBuffer>>,
    /// Validating recorders (debug builds), reused likewise.
    validator_pool: Arc<BoxPool<ValidatingRecorder>>,
    /// Swapchain bookkeeping (FIX-1.3); their surfaces live on the render thread.
    pub(crate) swapchains: Mutex<HashMap<u32, GlSwapchainState>>,
    pub(crate) next_swapchain_id: AtomicU32,
    /// The display the context was created on (`None` for headless devices). Declared after
    /// `thread`, so it is released only after the render thread has destroyed the context.
    pub(crate) display: Option<Arc<dyn DisplayTarget>>,
}

/// Uniform-buffer binding the OpenGL backend reserves for push-constant emulation (RGL-04).
///
/// OpenGL has no push constants. GLSL for this backend declares them as
/// `layout(std140, binding = 64) uniform PushConstants { ... };` with the same byte offsets as
/// the Vulkan `push_constant` block. Binding 64 is the first slot after the 4 x 16 range of the
/// linear binding scheme, and `GL_MAX_UNIFORM_BUFFER_BINDINGS` is at least 84 in GL 4.5.
///
/// GLSL requires a uniform block used by several stages of one program to be declared
/// identically in each of them, so every stage must declare the full block, even where the
/// Vulkan shader for that stage declares only its own range.
pub const PUSH_CONSTANT_BINDING: u32 = 64;

/// Everything that needs the OpenGL context. Created, used and dropped on the render thread
/// only, so the context is always current when its methods run.
pub(crate) struct GlThread {
    pub(crate) context: GlContext,
    pub(crate) resources: Arc<GlResourceRegistry>,
    pub(crate) state: GlStateCache,
    pub(crate) fbo_cache: GlFboCache,
    default_vao: glow::VertexArray,
    push_constant_buffer: glow::Buffer,
    /// `GL_MAX_TEXTURE_MAX_ANISOTROPY`, or 0 if anisotropic filtering is unsupported.
    max_anisotropy: f32,
    /// `GL_MAX_TEXTURE_LOD_BIAS`.
    max_lod_bias: f32,
    /// Fences of executed submissions that have not signalled yet, oldest first.
    pending_fences: VecDeque<(SubmissionId, glow::Fence)>,
    /// Shared with `OpenGlDevice::completed`.
    completed: Arc<AtomicU64>,
    /// Window surfaces of the device's swapchains, by swapchain ID (FIX-1.3).
    pub(crate) surfaces: HashMap<u32, GlPresentSurface>,
    /// Program binaries of linked pipelines (RGL-12).
    program_binaries: crate::program_cache::ProgramBinaryCache,
}

/// Work posted to the render thread, executed in posting order.
pub(crate) enum GlWork {
    /// A submitted command buffer, and the pool its stream returns to.
    Submit(SubmissionId, GlCommandStream, CommandStreamPool),
    /// Presentation of a swapchain texture.
    Present {
        submission: SubmissionId,
        swapchain: u32,
        texture: TextureHandle,
    },
}

impl OpenGlDevice {
    /// Creates a headless OpenGL 4.5 Core device on EGL surfaceless or PBuffer context.
    ///
    /// Spawns the device's render thread, which creates the context and keeps it current.
    /// Headless devices cannot present; see [`OpenGlDevice::create_for_display`].
    pub fn create_headless() -> Result<Self, RhiError> {
        Self::spawn(None)
    }

    /// Creates a device whose context lives on `display`'s window system (EGL on X11 or
    /// Wayland), so that it can present to that display's windows. The device keeps `display`
    /// alive while it uses the connection.
    ///
    /// # Errors
    /// `RhiError::Unsupported` for window systems without an EGL path.
    pub fn create_for_display(display: Arc<dyn DisplayTarget>) -> Result<Self, RhiError> {
        Self::spawn(Some(display))
    }

    fn spawn(display: Option<Arc<dyn DisplayTarget>>) -> Result<Self, RhiError> {
        let platform = match &display {
            None => GlPlatform::Surfaceless,
            Some(display) => GlPlatform::from_display_handle(
                display
                    .display_handle()
                    .map_err(|e| RhiError::InitializationFailed(format!("no display handle: {e}")))?
                    .as_raw(),
            )?,
        };
        let platform = SendHandle(platform);
        let resources = Arc::new(GlResourceRegistry::for_device(korppi_rhi::DeviceId::next()));
        let completed = Arc::new(AtomicU64::new(0));
        let thread_resources = Arc::clone(&resources);
        let thread_completed = Arc::clone(&completed);
        let thread = RenderThread::spawn("korppi-gl-render", move || {
            let platform = platform;
            // SAFETY: the native display in `platform` belongs to `display`, which the device
            // keeps alive until after the render thread (and with it the context) is gone.
            let context = unsafe { GlContext::create_for_platform(platform.0) }?;
            GlThread::new(context, thread_resources, thread_completed)
        })?;
        let presentation = display.is_some();
        let (diagnostics, debug_output_enabled, draw_parameters, capabilities) =
            thread.call(|gl| {
                let version = gl.context.gl.version();
                let draw_parameters = (version.major, version.minor) >= (4, 6)
                    || gl
                        .context
                        .gl
                        .supported_extensions()
                        .contains("GL_ARB_shader_draw_parameters");
                // SAFETY: the context is current on the render thread; `extra` belongs to it.
                let capabilities = unsafe {
                    GlCapabilities::query(
                        &gl.context.gl,
                        &gl.context.extra,
                        gl.max_anisotropy,
                        presentation,
                    )
                };
                (
                    Arc::clone(&gl.context.diagnostics),
                    gl.context.debug_output_enabled,
                    draw_parameters,
                    capabilities,
                )
            })?;
        let _ = resources.features.set(capabilities.features);
        let _ = resources.format_capabilities.set(
            TextureFormat::ALL
                .iter()
                .map(|&format| capabilities.format(format))
                .collect(),
        );
        Ok(Self {
            thread,
            resources,
            diagnostics,
            last_issued: Mutex::new(0),
            completed,
            frames: FrameTracker::new(),
            debug_output_enabled,
            draw_parameters,
            capabilities,
            command_streams: CommandStreamPool::default(),
            recorder_pool: BoxPool::new(),
            command_buffer_pool: BoxPool::new(),
            validator_pool: BoxPool::new(),
            swapchains: Mutex::new(HashMap::new()),
            next_swapchain_id: AtomicU32::new(1),
            display,
        })
    }

    /// Posts `work` (which must complete as one submission) and returns its submission ID.
    pub(crate) fn post(
        &self,
        work: impl FnOnce(SubmissionId) -> GlWork,
    ) -> Result<SubmissionId, RhiError> {
        let submission = {
            let mut last_issued = self.lock_last_issued();
            let submission = SubmissionId::new(*last_issued + 1);
            self.thread.post(work(submission))?;
            *last_issued = submission.get();
            submission
        };
        self.frames.record(submission);
        Ok(submission)
    }

    /// Runs `f` on the render thread with the device's OpenGL context current and returns its
    /// result, for diagnostics and code that renders with OpenGL directly (UI, capture).
    ///
    /// Afterwards the device forgets the context state it cached and applies its context-wide
    /// setup again (clip control, pixel-store alignment, sRGB writes, seamless cube maps, no
    /// rasterizer discard or primitive restart, no pixel pack/unpack buffers), so `f` may leave
    /// any of that, and any pipeline, binding or fixed-function state, behind (RGL-15). It
    /// must delete the objects it creates and leave the device's objects alone.
    pub fn with_gl<R: Send>(
        &self,
        f: impl FnOnce(&glow::Context) -> R + Send,
    ) -> Result<R, RhiError> {
        self.thread.call(|thread| {
            let result = f(&thread.context.gl);
            thread.state.invalidate();
            thread.context.apply_context_state();
            result
        })
    }

    /// Number of program binaries kept for pipeline caching (RGL-12).
    pub fn program_binary_count(&self) -> usize {
        self.thread
            .call(|thread| thread.program_binaries.len())
            .unwrap_or(0)
    }

    /// Returns the number of cached OpenGL Framebuffer Objects.
    pub fn fbo_cache_len(&self) -> usize {
        self.thread
            .call(|thread| thread.fbo_cache.len())
            .unwrap_or(0)
    }

    fn lock_last_issued(&self) -> std::sync::MutexGuard<'_, u64> {
        self.last_issued
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Rejects IDs this device never returned (waiting for them would never finish).
    fn check_submission(&self, submission: SubmissionId) -> Result<(), RhiError> {
        let last_issued = *self.lock_last_issued();
        if submission.get() > last_issued {
            return Err(RhiError::ValidationError(format!(
                "{submission:?} was not returned by this device (last submission: {last_issued})"
            )));
        }
        Ok(())
    }

    fn is_completed(&self, submission: SubmissionId) -> bool {
        self.completed.load(Ordering::Acquire) >= submission.get()
    }
}

impl GlThread {
    /// Creates the device-wide GL objects. Runs on the render thread, where `context` is current.
    fn new(
        context: GlContext,
        resources: Arc<GlResourceRegistry>,
        completed: Arc<AtomicU64>,
    ) -> Result<Self, RhiError> {
        // SAFETY: `GlContext` creation made the context current on this (the render) thread,
        // where it stays current for the lifetime of `GlThread`.
        let default_vao = unsafe {
            context.gl.create_vertex_array().map_err(|e| {
                RhiError::InitializationFailed(format!("Failed to create default VAO: {e}"))
            })?
        };

        // SAFETY: as above, the context is current on this thread.
        let push_constant_buffer = unsafe {
            let buffer = context.gl.create_named_buffer().map_err(|e| {
                RhiError::InitializationFailed(format!("Failed to create push-constant UBO: {e}"))
            })?;
            context.gl.named_buffer_storage(
                buffer,
                MAX_PUSH_CONSTANTS_SIZE as i32,
                None,
                glow::DYNAMIC_STORAGE_BIT,
            );
            buffer
        };

        // Anisotropic filtering is core in GL 4.6 and an extension before that.
        let anisotropy_supported = {
            let extensions = context.gl.supported_extensions();
            context.gl.version().major > 4
                || (context.gl.version().major == 4 && context.gl.version().minor >= 6)
                || extensions.contains("GL_ARB_texture_filter_anisotropic")
                || extensions.contains("GL_EXT_texture_filter_anisotropic")
        };
        let max_anisotropy = if anisotropy_supported {
            // SAFETY: the context is current on this thread.
            unsafe {
                context
                    .gl
                    .get_parameter_f32(glow::MAX_TEXTURE_MAX_ANISOTROPY)
            }
        } else {
            0.0
        };

        // SAFETY: the context is current on this thread.
        let program_binaries =
            unsafe { crate::program_cache::ProgramBinaryCache::new(&context.gl) };
        // SAFETY: the context is current on this thread.
        let max_lod_bias = unsafe { context.gl.get_parameter_f32(glow::MAX_TEXTURE_LOD_BIAS) };
        Ok(Self {
            max_lod_bias,
            program_binaries,
            context,
            resources,
            state: GlStateCache::new(),
            fbo_cache: GlFboCache::new(),
            default_vao,
            push_constant_buffer,
            max_anisotropy,
            pending_fences: VecDeque::new(),
            completed,
            surfaces: HashMap::new(),
        })
    }

    /// Inserts a fence after the commands of `submission`.
    fn fence(&mut self, submission: SubmissionId) {
        // SAFETY: the context is current on the render thread.
        let fence = unsafe {
            self.context
                .gl
                .fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0)
        };
        match fence {
            Ok(fence) => self.pending_fences.push_back((submission, fence)),
            Err(error) => {
                // Without a fence, completion can only be observed by draining the GPU.
                tracing::error!("glFenceSync failed: {error}; waiting for the GPU instead");
                // SAFETY: the context is current on the render thread.
                unsafe { self.context.gl.finish() };
                self.retire_all(submission);
            }
        }
    }

    /// Retires the fences that have signalled, without blocking.
    fn poll_fences(&mut self) {
        while let Some(&(submission, fence)) = self.pending_fences.front() {
            // SAFETY: the context is current on the render thread; `fence` is a live sync object.
            let status = unsafe { self.context.gl.client_wait_sync(fence, 0, 0) };
            if !matches!(status, glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED) {
                break;
            }
            self.retire_front(submission, fence);
        }
    }

    /// Blocks until every submission up to and including `submission` has completed.
    fn wait_fences(&mut self, submission: SubmissionId) -> Result<(), RhiError> {
        while let Some(&(pending, fence)) = self.pending_fences.front() {
            if pending > submission {
                break;
            }
            loop {
                // SAFETY: the context is current on the render thread; `fence` is a live sync
                // object. The flush bit guarantees the fence reaches the GPU, so it signals.
                let status = unsafe {
                    self.context.gl.client_wait_sync(
                        fence,
                        glow::SYNC_FLUSH_COMMANDS_BIT,
                        1_000_000_000,
                    )
                };
                match status {
                    glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED => break,
                    glow::TIMEOUT_EXPIRED => continue,
                    _ => {
                        return Err(RhiError::BackendError(format!(
                            "glClientWaitSync failed for {pending:?} (status {status:#x})"
                        )))
                    }
                }
            }
            self.retire_front(pending, fence);
        }
        Ok(())
    }

    /// Waits for the GPU to finish everything and retires every fence.
    fn finish(&mut self) {
        // SAFETY: the context is current on the render thread.
        unsafe { self.context.gl.finish() };
        if let Some(&(last, _)) = self.pending_fences.back() {
            self.retire_all(last);
        }
    }

    fn retire_front(&mut self, submission: SubmissionId, fence: glow::Fence) {
        self.pending_fences.pop_front();
        // SAFETY: the context is current on the render thread; the fence is deleted once.
        unsafe { self.context.gl.delete_sync(fence) };
        self.completed
            .fetch_max(submission.get(), Ordering::Release);
    }

    /// Retires every pending fence and marks `last` (the newest submission) complete.
    fn retire_all(&mut self, last: SubmissionId) {
        for (_, fence) in self.pending_fences.drain(..) {
            // SAFETY: the context is current on the render thread; each fence is deleted once.
            unsafe { self.context.gl.delete_sync(fence) };
        }
        self.completed.fetch_max(last.get(), Ordering::Release);
    }
}

impl ThreadState for GlThread {
    type Posted = GlWork;

    fn handle_posted(&mut self, work: GlWork) {
        let (submission, result) = match work {
            GlWork::Submit(submission, stream, pool) => {
                let result = self
                    .execute(&stream)
                    .map_err(|e| format!("command buffer execution failed: {e}"));
                pool.give(stream);
                (submission, result)
            }
            GlWork::Present {
                submission,
                swapchain,
                texture,
            } => (
                submission,
                self.present(swapchain, texture)
                    .map_err(|e| format!("present failed: {e}")),
            ),
        };
        if let Err(message) = result {
            tracing::error!("OpenGL {message}");
            self.context
                .diagnostics
                .push(DiagnosticSeverity::Error, message);
        }
        // The submission completes (in order) even if its playback stopped early.
        self.fence(submission);
        self.poll_fences();
    }

    fn posted_panicked(&mut self, message: String) {
        tracing::error!("OpenGL command buffer execution panicked: {message}");
        self.context.diagnostics.push(
            DiagnosticSeverity::Error,
            format!("command buffer execution panicked: {message}"),
        );
    }
}

/// The GL texture, target and format of `handle`, if it is live.
fn transfer_texture(
    resources: &GlResourceRegistry,
    handle: TextureHandle,
) -> Option<crate::transfer::TransferTexture> {
    lock(&resources.textures)
        .get(handle)
        .map(|(t, _)| crate::transfer::TransferTexture {
            texture: t.texture,
            target: t.target,
            format: t.format,
        })
}

/// Applies a graphics pipeline's blend state and write mask per colour target, polygon mode,
/// depth bias and clamp, and alpha-to-coverage (RGL-03).
///
/// # Safety
/// The OpenGL context must be current on this thread and `state` must be its state cache.
unsafe fn apply_raster_output_state(
    gl: &glow::Context,
    extra: &GlExtraFns,
    state: &mut GlStateCache,
    pipe: &GlPipeline,
) {
    // SAFETY: forwarded from the caller.
    unsafe {
        state.set_alpha_to_coverage(gl, pipe.multisample.alpha_to_coverage);
        for (i, target) in pipe.color_targets.iter().enumerate() {
            let i = i as u32;
            let blend = target.blend.map(|blend| GlBlend {
                src_color: to_gl_blend_factor(blend.color.src_factor),
                dst_color: to_gl_blend_factor(blend.color.dst_factor),
                src_alpha: to_gl_blend_factor(blend.alpha.src_factor),
                dst_alpha: to_gl_blend_factor(blend.alpha.dst_factor),
                color_op: to_gl_blend_op(blend.color.operation),
                alpha_op: to_gl_blend_op(blend.alpha.operation),
            });
            state.set_blend(gl, i, blend);
            let mask = target.write_mask;
            state.set_color_mask(
                gl,
                i,
                [
                    mask.contains(ColorWriteMask::RED),
                    mask.contains(ColorWriteMask::GREEN),
                    mask.contains(ColorWriteMask::BLUE),
                    mask.contains(ColorWriteMask::ALPHA),
                ],
            );
        }
        state.set_polygon_mode(gl, to_gl_polygon_mode(pipe.rasterizer.polygon_mode));
        let r = &pipe.rasterizer;
        let bias =
            (r.depth_bias != 0.0 || r.slope_scaled_depth_bias != 0.0).then_some(GlDepthBias {
                slope: r.slope_scaled_depth_bias,
                constant: r.depth_bias,
                clamp: r.depth_bias_clamp,
            });
        state.set_depth_bias(gl, extra, bias);
        state.set_depth_clamp(gl, r.depth_clamp);
    }
}

/// Applies a graphics pipeline's fixed-function state: blending, write masks, rasterization,
/// culling, depth and stencil tests (RGL-03, RHI-04).
///
/// # Safety
/// The OpenGL context must be current on this thread and `state` must be its state cache.
unsafe fn apply_graphics_state(
    gl: &glow::Context,
    extra: &GlExtraFns,
    state: &mut GlStateCache,
    pipe: &GlPipeline,
    stencil_reference: u32,
) {
    // SAFETY: forwarded from the caller.
    unsafe {
        apply_raster_output_state(gl, extra, state, pipe);
        let cull = to_gl_cull_mode(pipe.rasterizer.cull_mode);
        let front = to_gl_front_face(pipe.rasterizer.front_face);
        state.set_cull_face(gl, cull, front);
        match &pipe.depth_stencil {
            Some(ds) => state.set_depth_test(
                gl,
                ds.depth_test_enabled,
                ds.depth_write_enabled,
                to_gl_compare_func(ds.depth_compare) as u32,
            ),
            None => state.set_depth_test(gl, false, false, glow::ALWAYS),
        }
        apply_stencil_state(gl, state, pipe, stencil_reference);
    }
}

/// Applies a graphics pipeline's stencil test with the recorder's current reference. GL's
/// front faces are the RHI's: culling and stencil faces share `glFrontFace`.
///
/// # Safety
/// The OpenGL context must be current on this thread and `state` must be its state cache.
unsafe fn apply_stencil_state(
    gl: &glow::Context,
    state: &mut GlStateCache,
    pipe: &GlPipeline,
    reference: u32,
) {
    // SAFETY: forwarded from the caller.
    unsafe {
        match &pipe.depth_stencil {
            Some(ds) if ds.stencil_enabled() => {
                state.set_stencil_test(gl, true);
                for (face, s) in [
                    (state_cache::FRONT, &ds.stencil_front),
                    (state_cache::BACK, &ds.stencil_back),
                ] {
                    state.set_stencil_face(
                        gl,
                        face,
                        GlStencilFace {
                            func: to_gl_compare_func(s.compare) as u32,
                            reference: reference as i32,
                            read_mask: ds.stencil_read_mask,
                            fail: to_gl_stencil_op(s.fail_op),
                            depth_fail: to_gl_stencil_op(s.depth_fail_op),
                            pass: to_gl_stencil_op(s.pass_op),
                        },
                    );
                    state.set_stencil_write_mask(gl, face, ds.stencil_write_mask);
                }
            }
            _ => state.set_stencil_test(gl, false),
        }
    }
}

/// Builds a vertex array object for `layout` (attribute formats, bindings and divisors).
///
/// # Safety
/// The OpenGL context must be current on this thread and `state` must be its state cache.
unsafe fn create_vertex_array(
    gl: &glow::Context,
    state: &mut GlStateCache,
    layout: &VertexLayoutDescriptor,
) -> Result<glow::VertexArray, RhiError> {
    if let Some(b) = layout
        .bindings
        .iter()
        .find(|b| b.binding as usize >= MAX_VERTEX_BINDINGS)
    {
        return Err(RhiError::Unsupported(format!(
            "vertex binding {} exceeds the {MAX_VERTEX_BINDINGS} bindings supported",
            b.binding
        )));
    }
    layout.validate()?;
    // The VAO is configured with DSA calls, except the binding divisor, which glow only exposes
    // for the bound VAO; it is bound through `state`, which stays accurate.
    // SAFETY: forwarded from the caller; every call names the VAO created here.
    unsafe {
        let vao = gl
            .create_named_vertex_array()
            .map_err(|e| RhiError::BackendError(format!("glCreateVertexArrays failed: {e}")))?;
        for attribute in &layout.attributes {
            let format = to_gl_vertex_format(attribute.format);
            gl.enable_vertex_array_attrib(vao, attribute.location);
            if format.integer {
                gl.vertex_array_attrib_format_i32(
                    vao,
                    attribute.location,
                    format.size,
                    format.data_type,
                    attribute.offset as u32,
                );
            } else {
                gl.vertex_array_attrib_format_f32(
                    vao,
                    attribute.location,
                    format.size,
                    format.data_type,
                    format.normalized,
                    attribute.offset as u32,
                );
            }
            gl.vertex_array_attrib_binding_f32(vao, attribute.location, attribute.binding);
        }
        state.set_vao(gl, Some(vao));
        for binding in &layout.bindings {
            let divisor = match binding.step_mode {
                VertexStepMode::Vertex => 0,
                VertexStepMode::Instance => 1,
            };
            gl.vertex_binding_divisor(binding.binding, divisor);
        }
        Ok(vao)
    }
}

/// Vertex input bound during command playback. Vertex and index buffers are VAO state in
/// OpenGL, so they are re-applied whenever a pipeline (and its VAO) is bound.
#[derive(Default)]
struct DrawState {
    topology: u32,
    vertex_array: Option<glow::VertexArray>,
    vertex_strides: [Option<i32>; MAX_VERTEX_BINDINGS],
    vertex_buffers: [Option<(glow::Buffer, i32)>; MAX_VERTEX_BINDINGS],
    index_buffer: Option<(glow::Buffer, u64, IndexType)>,
}

impl DrawState {
    /// Attaches the recorded vertex buffers in `bindings` to the current pipeline's VAO.
    ///
    /// # Safety
    /// The OpenGL context must be current on this thread and `state` must be its state cache.
    unsafe fn apply_vertex_buffers(
        &self,
        gl: &glow::Context,
        state: &mut GlStateCache,
        bindings: std::ops::Range<usize>,
    ) {
        let Some(vao) = self.vertex_array else {
            return;
        };
        for binding in bindings.take_while(|b| *b < MAX_VERTEX_BINDINGS) {
            if let (Some((buffer, offset)), Some(stride)) =
                (self.vertex_buffers[binding], self.vertex_strides[binding])
            {
                // SAFETY: forwarded from the caller.
                unsafe { state.set_vertex_buffer(gl, vao, binding as u32, buffer, offset, stride) };
            }
        }
    }

    /// Attaches the recorded index buffer to the current pipeline's VAO.
    ///
    /// # Safety
    /// The OpenGL context must be current on this thread and `state` must be its state cache.
    unsafe fn apply_index_buffer(&self, gl: &glow::Context, state: &mut GlStateCache) {
        if let (Some(vao), Some((buffer, _, _))) = (self.vertex_array, self.index_buffer) {
            // SAFETY: forwarded from the caller.
            unsafe { state.set_element_buffer(gl, vao, buffer) };
        }
    }
}

fn buffer_not_found(handle: BufferHandle) -> RhiError {
    RhiError::InvalidHandle(format!("Buffer handle {handle:?} not found"))
}

// Resource operations. Each runs on the render thread, where the context is current, and holds
// at most one registry lock at a time (RGL-11).
impl GlThread {
    fn create_buffer(
        &mut self,
        desc: &BufferDescriptor,
        initial_data: Option<&[u8]>,
    ) -> Result<BufferHandle, RhiError> {
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread.
        let buffer = unsafe {
            gl.create_named_buffer()
                .map_err(|e| RhiError::OutOfMemory(format!("glCreateBuffers failed: {e}")))?
        };

        // OpenGL 4.5 Core Direct State Access immutable buffer storage. Host-visible buffers
        // are mapped persistently and coherently, once (RHI-05).
        let mappable = desc.memory != MemoryUsage::GpuOnly && desc.size > 0;
        let map_flags = glow::MAP_READ_BIT
            | glow::MAP_WRITE_BIT
            | glow::MAP_PERSISTENT_BIT
            | glow::MAP_COHERENT_BIT;
        let storage_flags = if mappable {
            glow::DYNAMIC_STORAGE_BIT | map_flags
        } else {
            glow::DYNAMIC_STORAGE_BIT | glow::MAP_READ_BIT | glow::MAP_WRITE_BIT
        };

        // SAFETY: the context is current on this thread. Storage is allocated without a data
        // pointer, so GL never reads `desc.size` bytes from a shorter slice (RGL-10); the slice
        // (validated to fit) is then copied with its own length. The mapping stays valid until
        // the buffer is deleted.
        let mapped = unsafe {
            gl.named_buffer_storage(buffer, desc.size as i32, None, storage_flags);
            if let Some(data) = initial_data.filter(|d| !d.is_empty()) {
                gl.named_buffer_sub_data_u8_slice(buffer, 0, data);
            }
            if mappable {
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                let ptr =
                    gl.map_buffer_range(glow::COPY_WRITE_BUFFER, 0, desc.size as i32, map_flags);
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
                match std::ptr::NonNull::new(ptr) {
                    Some(ptr) => Some(MappedPtr(ptr)),
                    None => {
                        gl.delete_buffer(buffer);
                        return Err(RhiError::BufferMappingFailed(
                            "glMapBufferRange returned null".into(),
                        ));
                    }
                }
            } else {
                None
            }
        };

        self.label_object(glow::BUFFER, buffer.0.get(), desc.label);
        let gl_buf = GlBuffer {
            buffer,
            size: desc.size,
            usage: desc.usage,
            memory: desc.memory,
            mapped,
            host_access: korppi_rhi::HostAccessLock::new(),
        };
        let (id, generation) = lock(&self.resources.buffers).insert(gl_buf);
        Ok(BufferHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn destroy_buffer(&mut self, handle: BufferHandle) -> Result<(), RhiError> {
        let removed = {
            let mut buffers = lock(&self.resources.buffers);
            if let Some((buf, _)) = buffers.get(handle) {
                // Not while a mapping guard exists: deleting the buffer unmaps it.
                buf.host_access.acquire_for("destroy_buffer")?;
            }
            buffers.remove(handle)
        };
        let (buf, _) = removed.ok_or_else(|| buffer_not_found(handle))?;
        self.resources
            .invalidate_bind_groups(&[BoundResource::Buffer(handle)]);
        self.state.forget_buffer(buf.buffer);
        // SAFETY: the context is current on the render thread. GL keeps the storage alive until
        // commands already issued that use it have completed.
        unsafe { self.context.gl.delete_buffer(buf.buffer) };
        Ok(())
    }

    fn write_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        data: &[u8],
    ) -> Result<(), RhiError> {
        let (buffer, size, host_access) = {
            let buffers = lock(&self.resources.buffers);
            let (buf, _) = buffers
                .get(handle)
                .ok_or_else(|| buffer_not_found(handle))?;
            (buf.buffer, buf.size, buf.host_access.clone())
        };
        let _access = host_access.acquire_for("write_buffer")?;
        let offset_i32 = match i32::try_from(offset) {
            Ok(o) => o,
            Err(_) => {
                return Err(RhiError::ValidationError(
                    "Buffer write offset exceeds 2 GiB OpenGL limit".into(),
                ));
            }
        };
        if offset
            .checked_add(data.len() as u64)
            .is_none_or(|end| end > size)
        {
            return Err(RhiError::ValidationError(
                "Buffer write out of bounds".into(),
            ));
        }
        // SAFETY: the context is current on the render thread; the range was checked with
        // overflow protection above, and offset fits in i32 (RGL-21).
        unsafe {
            self.context
                .gl
                .named_buffer_sub_data_u8_slice(buffer, offset_i32, data);
        }
        Ok(())
    }

    fn read_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        let (buffer, buffer_size, host_access) = {
            let buffers = lock(&self.resources.buffers);
            let (buf, _) = buffers
                .get(handle)
                .ok_or_else(|| buffer_not_found(handle))?;
            (buf.buffer, buf.size, buf.host_access.clone())
        };
        let _access = host_access.acquire_for("read_buffer")?;
        let offset_i32 = match i32::try_from(offset) {
            Ok(o) => o,
            Err(_) => {
                return Err(RhiError::ValidationError(
                    "Buffer read offset exceeds 2 GiB OpenGL limit".into(),
                ));
            }
        };
        let size_usize = match usize::try_from(size) {
            Ok(s) => s,
            Err(_) => {
                return Err(RhiError::ValidationError(
                    "Buffer read size exceeds address space".into(),
                ));
            }
        };
        if offset.checked_add(size).is_none_or(|end| end > buffer_size) {
            return Err(RhiError::ValidationError(
                "Buffer read out of bounds".into(),
            ));
        }

        // Always read the GPU copy (RGL-06: a CPU shadow hid everything the GPU wrote).
        // Shader-storage and image writes are incoherent; make them visible to buffer reads
        // even if the caller recorded no barrier.
        let gl = &self.context.gl;
        let mut output = vec![0u8; size_usize];
        // SAFETY: the context is current on this thread; `output` is exactly `size` bytes,
        // and `offset.checked_add(size)` was checked against the buffer size above (RGL-21).
        unsafe {
            gl.memory_barrier(glow::BUFFER_UPDATE_BARRIER_BIT);
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(buffer));
            gl.get_buffer_sub_data(glow::COPY_READ_BUFFER, offset_i32, &mut output);
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        }
        Ok(output)
    }

    /// Deletes a texture's GL object and every cached reference to it: framebuffers of texture
    /// `id` (views have none) and the state cache.
    pub(crate) fn delete_texture_object(&mut self, id: Option<u32>, texture: glow::Texture) {
        let gl = &self.context.gl;
        if let Some(id) = id {
            for fbo in self.fbo_cache.invalidate_texture(gl, id) {
                self.state.forget_fbo(fbo);
            }
        }
        self.state.forget_texture(texture);
        // SAFETY: the context is current on the render thread.
        unsafe { gl.delete_texture(texture) };
    }

    fn create_sampler(&mut self, desc: &SamplerDescriptor) -> Result<SamplerHandle, RhiError> {
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread.
        let sampler = unsafe {
            gl.create_sampler()
                .map_err(|e| RhiError::OutOfMemory(format!("glCreateSamplers failed: {e}")))?
        };

        // SAFETY: the context is current on the render thread; `sampler` was created above.
        unsafe {
            gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_MIN_FILTER,
                to_gl_min_filter(desc.min_filter, desc.mipmap_filter),
            );
            gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_MAG_FILTER,
                to_gl_filter(desc.mag_filter),
            );
            gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_WRAP_S,
                to_gl_address_mode(desc.address_mode_u),
            );
            gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_WRAP_T,
                to_gl_address_mode(desc.address_mode_v),
            );
            gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_WRAP_R,
                to_gl_address_mode(desc.address_mode_w),
            );

            gl.sampler_parameter_f32_slice(
                sampler,
                glow::TEXTURE_BORDER_COLOR,
                &border_color_rgba(desc.border_color),
            );
            gl.sampler_parameter_f32(sampler, glow::TEXTURE_MIN_LOD, desc.lod_min_clamp);
            gl.sampler_parameter_f32(sampler, glow::TEXTURE_MAX_LOD, desc.lod_max_clamp);
            gl.sampler_parameter_f32(
                sampler,
                glow::TEXTURE_LOD_BIAS,
                desc.mip_lod_bias
                    .clamp(-self.max_lod_bias, self.max_lod_bias),
            );
            if desc.max_anisotropy > 1 && self.max_anisotropy >= 1.0 {
                gl.sampler_parameter_f32(
                    sampler,
                    glow::TEXTURE_MAX_ANISOTROPY,
                    f32::from(desc.max_anisotropy).min(self.max_anisotropy),
                );
            }

            if let Some(comp) = desc.compare {
                gl.sampler_parameter_i32(
                    sampler,
                    glow::TEXTURE_COMPARE_MODE,
                    glow::COMPARE_REF_TO_TEXTURE as i32,
                );
                gl.sampler_parameter_i32(
                    sampler,
                    glow::TEXTURE_COMPARE_FUNC,
                    to_gl_compare_func(comp),
                );
            }
        }

        self.label_object(glow::SAMPLER, sampler.0.get(), desc.label);
        let (id, generation) = lock(&self.resources.samplers).insert(GlSampler { sampler });
        Ok(SamplerHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn destroy_sampler(&mut self, handle: SamplerHandle) -> Result<(), RhiError> {
        let removed = lock(&self.resources.samplers).remove(handle);
        let (samp, _) = removed.ok_or_else(|| {
            RhiError::InvalidHandle(format!("Sampler handle {handle:?} not found"))
        })?;
        self.resources
            .invalidate_bind_groups(&[BoundResource::Sampler(handle)]);
        self.state.forget_sampler(samp.sampler);
        // SAFETY: the context is current on the render thread.
        unsafe { self.context.gl.delete_sampler(samp.sampler) };
        Ok(())
    }

    /// Labels a GL object for debuggers (`glObjectLabel`, core since 4.3).
    pub(crate) fn label_object(&self, identifier: u32, name: u32, label: Option<&str>) {
        if let Some(label) = label {
            // SAFETY: the context is current on the render thread; `name` is a live object of
            // kind `identifier`.
            unsafe { self.context.gl.object_label(identifier, name, Some(label)) };
        }
    }

    /// `program`, labelled.
    fn label_program(&self, program: glow::Program, label: Option<&str>) -> glow::Program {
        self.label_object(glow::PROGRAM, program.0.get(), label);
        program
    }

    fn create_query_set(
        &mut self,
        desc: &korppi_rhi::QuerySetDescriptor,
    ) -> Result<korppi_rhi::QuerySetHandle, RhiError> {
        // `glCreateQueries`, not `glGenQueries`: a generated name only becomes a query object at
        // its first use, and labelling it before then is `GL_INVALID_VALUE` (ISSUE-008).
        let create_queries = self.context.extra.create_queries.ok_or_else(|| {
            RhiError::Unsupported("glCreateQueries (OpenGL 4.5) is unavailable".into())
        })?;
        let count = i32::try_from(desc.count).map_err(|_| {
            RhiError::ValidationError(format!("{} queries exceed GLsizei", desc.count))
        })?;
        let mut names = vec![0u32; desc.count as usize];
        // SAFETY: the context is current on the render thread; `names` holds `count` elements.
        unsafe { create_queries(glow::TIMESTAMP, count, names.as_mut_ptr()) };
        let mut queries = Vec::with_capacity(names.len());
        for name in names {
            match std::num::NonZeroU32::new(name) {
                Some(name) => queries.push(glow::NativeQuery(name)),
                None => {
                    for query in queries {
                        // SAFETY: as above; the queries are unused.
                        unsafe { self.context.gl.delete_query(query) };
                    }
                    return Err(RhiError::OutOfMemory("glCreateQueries failed".into()));
                }
            }
        }
        for query in &queries {
            self.label_object(glow::QUERY, query.0.get(), desc.label);
        }
        let (id, generation) = lock(&self.resources.query_sets).insert(queries);
        Ok(korppi_rhi::QuerySetHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn destroy_query_set(&mut self, handle: korppi_rhi::QuerySetHandle) -> Result<(), RhiError> {
        let (queries, _) = lock(&self.resources.query_sets)
            .remove(handle)
            .ok_or_else(|| RhiError::InvalidHandle(format!("query set {handle:?} not found")))?;
        for query in queries {
            // SAFETY: the context is current on the render thread; earlier commands that use
            // the query were issued before this deletion.
            unsafe { self.context.gl.delete_query(query) };
        }
        Ok(())
    }

    /// Compiles GLSL `source` for `stage`.
    fn compile_glsl(&self, stage: ShaderStage, source: &str) -> Result<glow::Shader, RhiError> {
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread.
        unsafe {
            let shader = gl.create_shader(to_gl_shader_stage(stage)).map_err(|e| {
                RhiError::ShaderCompilationFailed(format!("glCreateShader failed: {e}"))
            })?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let info = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                return Err(RhiError::ShaderCompilationFailed(format!(
                    "GLSL shader compilation failed: {info}"
                )));
            }
            Ok(shader)
        }
    }

    /// Registers a shader: GLSL is compiled now; parsed SPIR-V waits for its pipelines.
    fn create_shader(
        &mut self,
        stage: ShaderStage,
        source: ShaderInput,
    ) -> Result<ShaderHandle, RhiError> {
        let code = match source {
            ShaderInput::Glsl(source) => GlShaderCode::Compiled {
                shader: self.compile_glsl(stage, &source)?,
                source_hash: crate::program_cache::source_hash(&source),
            },
            ShaderInput::Spirv(parsed) => GlShaderCode::Spirv(parsed),
        };
        let (id, generation) = lock(&self.resources.shaders).insert(GlShader { code, stage });
        Ok(ShaderHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn destroy_shader(&mut self, handle: ShaderHandle) -> Result<(), RhiError> {
        let removed = lock(&self.resources.shaders).remove(handle);
        let (shader, _) = removed.ok_or_else(|| {
            RhiError::InvalidHandle(format!("Shader handle {handle:?} not found"))
        })?;
        if let GlShaderCode::Compiled { shader: object, .. } = shader.code {
            // SAFETY: the context is current on the render thread.
            unsafe { self.context.gl.delete_shader(object) };
        }
        Ok(())
    }

    /// Compiles the pipeline's shader sources and links every stage into a program; the
    /// shader objects compiled here are deleted again.
    fn link_pipeline(
        &mut self,
        shaders: &[PipelineShader],
        what: &str,
    ) -> Result<glow::Program, RhiError> {
        // A program linked from the same sources before (this run or, through
        // `load_pipeline_cache`, an earlier one) is loaded from its binary (RGL-12).
        let key = crate::program_cache::program_key(shaders.iter().map(PipelineShader::key));
        if let Some(binary) = self.program_binaries.get(key) {
            let gl = &self.context.gl;
            // SAFETY: the context is current on the render thread; the binary came from this
            // driver (checked on load), and a rejected one only fails the link.
            let loaded = unsafe {
                gl.create_program().ok().and_then(|program| {
                    gl.program_binary(program, binary);
                    if gl.get_program_link_status(program) {
                        Some(program)
                    } else {
                        gl.delete_program(program);
                        None
                    }
                })
            };
            match loaded {
                Some(program) => return Ok(program),
                None => self.program_binaries.remove(key),
            }
        }
        let mut objects = Vec::with_capacity(shaders.len());
        let mut temporary = Vec::new();
        let mut result = Ok(());
        for shader in shaders {
            match shader {
                PipelineShader::Compiled(object, _, _) => objects.push(*object),
                PipelineShader::Source(stage, source) => match self.compile_glsl(*stage, source) {
                    Ok(object) => {
                        objects.push(object);
                        temporary.push(object);
                    }
                    Err(error) => {
                        result = Err(error);
                        break;
                    }
                },
            }
        }
        let program = result.and_then(|()| self.link_program(&objects, what));
        for object in temporary {
            // SAFETY: the context is current; a shader attached to a program is only deleted
            // once the program is.
            unsafe { self.context.gl.delete_shader(object) };
        }
        if let (Ok(program), true) = (&program, self.program_binaries.enabled()) {
            // SAFETY: the context is current; the program was linked with the retrievable hint.
            if let Some(binary) = unsafe { self.context.gl.get_program_binary(*program) } {
                self.program_binaries.insert(key, binary);
            }
        }
        program
    }

    /// Links `shaders` into a program.
    fn link_program(
        &self,
        shaders: &[glow::Shader],
        what: &str,
    ) -> Result<glow::Program, RhiError> {
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread; the shaders are live objects of
        // this context.
        unsafe {
            let program = gl
                .create_program()
                .map_err(|e| RhiError::ValidationError(format!("glCreateProgram failed: {e}")))?;
            for &shader in shaders {
                gl.attach_shader(program, shader);
            }
            if self.program_binaries.enabled() {
                gl.program_binary_retrievable_hint(program, true);
            }
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                let info = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(RhiError::ShaderCompilationFailed(format!(
                    "GLSL {what}program linking failed: {info}"
                )));
            }
            Ok(program)
        }
    }

    fn create_graphics_pipeline(
        &mut self,
        desc: &GraphicsPipelineDescriptor,
        shaders: &[PipelineShader],
        bindings: PipelineBindings,
    ) -> Result<PipelineHandle, RhiError> {
        let program = self.link_pipeline(shaders, "")?;
        self.label_object(glow::PROGRAM, program.0.get(), desc.label);

        // Vertex input layout lives in a per-pipeline vertex array object (RGL-01).
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread and `self.state` is its cache.
        let vertex_array =
            match unsafe { create_vertex_array(gl, &mut self.state, &desc.vertex_layout) } {
                Ok(vao) => vao,
                Err(error) => {
                    // SAFETY: the context is current; the program was created above and is unused.
                    unsafe { gl.delete_program(program) };
                    return Err(error);
                }
            };
        let mut vertex_strides = [None; MAX_VERTEX_BINDINGS];
        for binding in &desc.vertex_layout.bindings {
            vertex_strides[binding.binding as usize] = Some(binding.stride as i32);
        }
        let gl_pipe = GlPipeline {
            program,
            vertex_array: Some(vertex_array),
            vertex_strides,
            topology: to_gl_topology(desc.topology),
            rasterizer: desc.rasterizer.clone(),
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            color_targets: desc.color_targets.clone(),
            signature: PipelineSignature::graphics(desc).with_bind_group_layouts(&bindings.layouts),
            binding_table: bindings.table,
        };
        let (id, generation) = lock(&self.resources.pipelines).insert(gl_pipe);
        Ok(PipelineHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn create_compute_pipeline(
        &mut self,
        desc: &ComputePipelineDescriptor,
        shader: &PipelineShader,
        bindings: PipelineBindings,
    ) -> Result<PipelineHandle, RhiError> {
        let program = self.link_pipeline(std::slice::from_ref(shader), "compute ")?;
        let program = self.label_program(program, desc.label);
        let gl_pipe = GlPipeline {
            program,
            vertex_array: None,
            vertex_strides: [None; MAX_VERTEX_BINDINGS],
            topology: glow::POINTS,
            rasterizer: RasterizerState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            color_targets: Vec::new(),
            signature: PipelineSignature::compute(desc).with_bind_group_layouts(&bindings.layouts),
            binding_table: bindings.table,
        };
        let (id, generation) = lock(&self.resources.pipelines).insert(gl_pipe);
        Ok(PipelineHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn destroy_pipeline(&mut self, handle: PipelineHandle) -> Result<(), RhiError> {
        let removed = lock(&self.resources.pipelines).remove(handle);
        let (pipe, _) = removed.ok_or_else(|| {
            RhiError::InvalidHandle(format!("Pipeline handle {handle:?} not found"))
        })?;
        let gl = &self.context.gl;
        self.state.forget_program(pipe.program);
        // SAFETY: the context is current on the render thread.
        unsafe {
            gl.delete_program(pipe.program);
            if let Some(vao) = pipe.vertex_array {
                self.state.forget_vao(vao);
                gl.delete_vertex_array(vao);
            }
        }
        Ok(())
    }

    /// Plays back a recorded command buffer.
    fn execute(&mut self, stream: &GlCommandStream) -> Result<(), RhiError> {
        let gl = &self.context.gl;
        let extra = &self.context.extra;
        let state = &mut self.state;
        let fbo_cache = &mut self.fbo_cache;
        let resources = &*self.resources;
        let diagnostics = &*self.context.diagnostics;
        let push_constant_buffer = self.push_constant_buffer;

        // Ensure default VAO is bound for drawing
        // SAFETY: the context is current on the render thread.
        unsafe {
            state.set_vao(gl, Some(self.default_vao));
        }

        fbo_cache.advance_generation();
        for fbo in fbo_cache.evict(gl) {
            state.forget_fbo(fbo);
        }
        let mut current_fbo = None;
        let mut current_pass: Option<&korppi_rhi::RenderingDescriptor> = None;
        // Bind groups (RHI-07): applied before a draw or dispatch, at the bound pipeline's
        // binding points, when they or the pipeline changed.
        let mut bound_groups: [Option<BoundGroup>; MAX_BIND_GROUPS] = [None; MAX_BIND_GROUPS];
        let mut binding_table = GlBindingTable::default();
        let mut groups_dirty = false;
        // Dynamic state, per command buffer as in Vulkan: stencil reference and blend constant
        // start at 0 (RHI-04). The bound graphics pipeline's state is applied again after
        // clears, which change write masks.
        let mut stencil_reference = 0_u32;
        let mut bound_graphics: Option<PipelineHandle> = None;
        // SAFETY: the context is current on this thread and `state` is its cache.
        unsafe { state.set_blend_color(gl, [0.0; 4]) };
        let mut draw = DrawState::default();
        // Push constants are per command buffer (as in Vulkan): start zeroed, upload lazily
        // before the next draw or dispatch that follows a change (RGL-04).
        let mut push_data = [0_u8; MAX_PUSH_CONSTANTS_SIZE];
        let mut push_dirty = true;
        // SAFETY: the context is current on this thread and `state` is its cache.
        unsafe {
            state.bind_buffer_range(
                gl,
                glow::UNIFORM_BUFFER,
                PUSH_CONSTANT_BINDING,
                push_constant_buffer,
                0,
                MAX_PUSH_CONSTANTS_SIZE as i32,
            );
        }
        let flush_push_constants = |gl: &glow::Context, data: &[u8], dirty: &mut bool| {
            if *dirty {
                // SAFETY: the context is current on this thread; `data` is the 128-byte
                // staging array and the buffer was allocated with exactly that size.
                unsafe {
                    gl.named_buffer_sub_data_u8_slice(push_constant_buffer, 0, data);
                }
                *dirty = false;
            }
        };
        // Push constants and bind groups are applied lazily, before each draw or dispatch.
        macro_rules! flush_draw_state {
            () => {{
                flush_push_constants(gl, &push_data, &mut push_dirty);
                if std::mem::take(&mut groups_dirty) {
                    let groups = lock(&resources.bind_groups);
                    // SAFETY: the context is current on this thread and `state` is its cache.
                    unsafe {
                        apply_bind_groups(
                            gl,
                            state,
                            &binding_table,
                            &bound_groups,
                            &groups,
                            &mut |message| diagnostics.push(DiagnosticSeverity::Error, message),
                        );
                    }
                }
            }};
        }

        // Commands hold no heap data, so copying one out of the stream is cheap.
        for cmd in stream.commands.iter().cloned() {
            match cmd {
                GlCommand::BeginRendering(pass) => {
                    let desc = &stream.passes[pass as usize];
                    let key = FboKey::from_descriptor(desc)?;
                    let (fbo, color_integer, depth_stencil) = {
                        let textures = lock(&resources.textures);
                        let fbo = fbo_cache.get_or_create(gl, &key, &textures)?;
                        let format = |id: u32| textures.get_by_index(id).map(|(t, _)| t.format);
                        let mut color_integer = [false; MAX_FBO_COLOR_ATTACHMENTS];
                        for (slot, att) in color_integer.iter_mut().zip(&desc.color_attachments) {
                            *slot = format(att.texture.id()).is_some_and(|f| f.is_integer());
                        }
                        let depth_stencil = desc
                            .depth_attachment
                            .as_ref()
                            .and_then(|d| format(d.texture.id()))
                            .is_some_and(|f| f.is_stencil());
                        (fbo, color_integer, depth_stencil)
                    };

                    current_fbo = Some(fbo);
                    // LoadOp::Clear clears the render area (desc.offset_x..+width,
                    // desc.offset_y..+height) regardless of state left by earlier pipelines or
                    // passes: GL clears honour the scissor test and the colour/depth write masks
                    // (GL 4.6 §17.4.3), Vulkan clears the renderArea ignoring write masks
                    // (RGL-09, RGL-25). Pipelines re-apply their masks when bound, and draws
                    // require a fresh `set_scissor`, so resetting here is safe.
                    // SAFETY: the context is current on this thread and `state` is its cache.
                    unsafe {
                        state.set_fbo(gl, Some(fbo));
                        state.set_viewport(
                            gl,
                            desc.offset_x,
                            desc.offset_y,
                            desc.width as i32,
                            desc.height as i32,
                        );
                        state.set_scissor(
                            gl,
                            desc.offset_x,
                            desc.offset_y,
                            desc.width as i32,
                            desc.height as i32,
                        );
                        state.set_depth_write(gl, true);
                        for i in 0..desc.color_attachments.len() as u32 {
                            state.set_color_mask(gl, i, [true; 4]);
                        }
                    }

                    // Process load operations (Clear)
                    for (i, att) in desc.color_attachments.iter().enumerate() {
                        if att.load_op == LoadOp::Clear {
                            let clear_color = att.clear_color.unwrap_or([0.0, 0.0, 0.0, 1.0]);
                            // SAFETY: the context is current on this thread; integer
                            // attachments are cleared with integer values.
                            unsafe {
                                if color_integer[i] {
                                    gl.clear_named_framebuffer_u32_slice(
                                        Some(fbo),
                                        glow::COLOR,
                                        i as u32,
                                        &clear_color.map(|c| c as u32),
                                    );
                                } else {
                                    gl.clear_named_framebuffer_f32_slice(
                                        Some(fbo),
                                        glow::COLOR,
                                        i as u32,
                                        &clear_color,
                                    );
                                }
                            }
                        }
                    }

                    if let Some(ref depth) = desc.depth_attachment {
                        let d_val = depth.clear_depth.unwrap_or(0.0);
                        let s_val = depth.clear_stencil.unwrap_or(0) as i32;
                        let clear_depth = depth.load_op == LoadOp::Clear;
                        let clear_stencil = depth_stencil && depth.stencil_load_op == LoadOp::Clear;
                        // SAFETY: the context is current on this thread. Stencil clears honour
                        // the stencil write mask, which pipelines may have narrowed.
                        unsafe {
                            if clear_stencil {
                                state.set_stencil_write_mask(gl, state_cache::FRONT, 0xFF);
                                state.set_stencil_write_mask(gl, state_cache::BACK, 0xFF);
                            }
                            match (clear_depth, clear_stencil) {
                                (true, true) => gl.clear_named_framebuffer_depth_stencil(
                                    Some(fbo),
                                    glow::DEPTH_STENCIL,
                                    0,
                                    d_val,
                                    s_val,
                                ),
                                (true, false) => gl.clear_named_framebuffer_f32_slice(
                                    Some(fbo),
                                    glow::DEPTH,
                                    0,
                                    &[d_val],
                                ),
                                (false, true) => gl.clear_named_framebuffer_i32_slice(
                                    Some(fbo),
                                    glow::STENCIL,
                                    0,
                                    &[s_val],
                                ),
                                (false, false) => {}
                            }
                        }
                    }
                    // Leave the scissor test disabled for the pass until `set_scissor` enables it.
                    // SAFETY: the context is current on this thread and `state` is its cache.
                    unsafe {
                        state.disable_scissor(gl);
                    }
                    // Clears reset write masks; a pipeline bound in an earlier pass stays bound.
                    if let Some(handle) = bound_graphics {
                        let pipelines = lock(&resources.pipelines);
                        if let Some((pipe, _)) = pipelines.get(handle) {
                            // SAFETY: the context is current on this thread and `state` is its
                            // cache.
                            unsafe {
                                apply_graphics_state(gl, extra, state, pipe, stencil_reference)
                            };
                        }
                    }
                    current_pass = Some(desc);
                }
                GlCommand::EndRendering => {
                    let pass = current_pass.take();
                    if let (Some(fbo), Some(pass)) = (current_fbo.take(), pass) {
                        // Multisampled attachments resolve into their resolve targets.
                        for (i, att) in pass.color_attachments.iter().enumerate() {
                            let Some(resolve) = att.resolve_target else {
                                continue;
                            };
                            let target = {
                                let textures = lock(&resources.textures);
                                fbo_cache.get_or_create(
                                    gl,
                                    &FboKey::single_color((resolve.id(), 0, 0)),
                                    &textures,
                                )?
                            };
                            let (x0, y0) = (pass.offset_x, pass.offset_y);
                            let (x1, y1) = (x0 + pass.width as i32, y0 + pass.height as i32);
                            // SAFETY: the context is current on this thread; both framebuffers
                            // are complete; blits only honour the scissor test, disabled here.
                            unsafe {
                                state.disable_scissor(gl);
                                gl.named_framebuffer_read_buffer(
                                    Some(fbo),
                                    glow::COLOR_ATTACHMENT0 + i as u32,
                                );
                                gl.blit_named_framebuffer(
                                    Some(fbo),
                                    Some(target),
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                    glow::COLOR_BUFFER_BIT,
                                    glow::NEAREST,
                                );
                                gl.named_framebuffer_read_buffer(
                                    Some(fbo),
                                    glow::COLOR_ATTACHMENT0,
                                );
                            }
                        }
                        // SAFETY: the context is current on this thread.
                        unsafe {
                            state.set_fbo(gl, None);
                        }
                    }
                }
                GlCommand::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => {
                    // SAFETY: the context is current on this thread and `state` is its cache.
                    unsafe {
                        state.set_viewport(gl, x as i32, y as i32, width as i32, height as i32);
                        // With `GL_ZERO_TO_ONE` clip control, `glDepthRangef` maps NDC depth exactly
                        // as Vulkan's `minDepth`/`maxDepth` do (RGL-15).
                        state.set_depth_range(gl, min_depth, max_depth);
                    }
                }
                GlCommand::SetScissor {
                    x,
                    y,
                    width,
                    height,
                } => {
                    // SAFETY: the context is current on this thread and `state` is its cache.
                    unsafe { state.set_scissor(gl, x, y, width as i32, height as i32) };
                }
                GlCommand::BindPipeline(pipe_handle) => {
                    let pipelines = lock(&resources.pipelines);
                    if let Some((pipe, _)) = pipelines.get(pipe_handle) {
                        draw.topology = pipe.topology;
                        if pipe.binding_table != binding_table {
                            binding_table = pipe.binding_table;
                            groups_dirty = true;
                        }
                        if pipe.vertex_array.is_some() {
                            bound_graphics = Some(pipe_handle);
                            // SAFETY: the context is current on this thread and `state` is its
                            // cache.
                            unsafe {
                                apply_graphics_state(gl, extra, state, pipe, stencil_reference)
                            };
                        }
                        if let Some(vao) = pipe.vertex_array {
                            draw.vertex_array = Some(vao);
                            draw.vertex_strides = pipe.vertex_strides;
                            // SAFETY: the context is current on this thread; `vao` belongs to
                            // this context.
                            unsafe {
                                state.set_vao(gl, Some(vao));
                                draw.apply_vertex_buffers(gl, state, 0..MAX_VERTEX_BINDINGS);
                                draw.apply_index_buffer(gl, state);
                            }
                        }
                        // SAFETY: the context is current on this thread.
                        unsafe { state.set_program(gl, Some(pipe.program)) };
                    }
                }
                GlCommand::BindVertexBuffers {
                    first_binding,
                    start,
                    count,
                } => {
                    let reg = lock(&resources.buffers);
                    let first = first_binding as usize;
                    let bindings = &stream.vertex_buffers[start as usize..(start + count) as usize];
                    for (i, &(b_handle, offset)) in bindings.iter().enumerate() {
                        if let (Some(slot), Some((buf, _))) =
                            (draw.vertex_buffers.get_mut(first + i), reg.get(b_handle))
                        {
                            *slot = Some((buf.buffer, offset as i32));
                        }
                    }
                    // SAFETY: the context is current on this thread.
                    unsafe {
                        draw.apply_vertex_buffers(gl, state, first..first + bindings.len());
                    }
                }
                GlCommand::BindIndexBuffer {
                    buffer,
                    offset,
                    index_type,
                } => {
                    let reg = lock(&resources.buffers);
                    if let Some((buf, _)) = reg.get(buffer) {
                        draw.index_buffer = Some((buf.buffer, offset, index_type));
                        // SAFETY: the context is current on this thread.
                        unsafe { draw.apply_index_buffer(gl, state) };
                    }
                }
                GlCommand::SetStencilReference(reference) => {
                    stencil_reference = reference;
                    if let Some(handle) = bound_graphics {
                        let pipelines = lock(&resources.pipelines);
                        if let Some((pipe, _)) = pipelines.get(handle) {
                            // SAFETY: the context is current on this thread and `state` is its
                            // cache.
                            unsafe { apply_stencil_state(gl, state, pipe, reference) };
                        }
                    }
                }
                GlCommand::SetBlendConstant(c) => {
                    // SAFETY: the context is current on this thread and `state` is its cache.
                    unsafe { state.set_blend_color(gl, c) };
                }
                GlCommand::SetBindGroup {
                    index,
                    group,
                    offsets,
                    count,
                } => {
                    if let Some(slot) = bound_groups.get_mut(index as usize) {
                        *slot = Some(BoundGroup {
                            group,
                            offsets,
                            count,
                        });
                        groups_dirty = true;
                    }
                }
                GlCommand::PushConstants { offset, start, len } => {
                    let data = stream.bytes(start, len);
                    let at = offset as usize;
                    if let Some(dst) = push_data.get_mut(at..at + data.len()) {
                        dst.copy_from_slice(data);
                        push_dirty = true;
                    }
                }
                GlCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    flush_draw_state!();
                    if instance_count > 0 {
                        // SAFETY: the context is current on this thread; topology, vertex array
                        // and buffers were set up by the bound pipeline (RGL-02).
                        unsafe {
                            gl.draw_arrays_instanced_base_instance(
                                draw.topology,
                                first_vertex as i32,
                                vertex_count as i32,
                                instance_count as i32,
                                first_instance,
                            );
                        }
                    }
                }
                GlCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                } => {
                    flush_draw_state!();
                    if let (Some((_, offset, index_type)), true) =
                        (draw.index_buffer, instance_count > 0)
                    {
                        let (gl_type, stride) = to_gl_index_type(index_type);
                        let byte_offset = offset + u64::from(first_index) * stride as u64;
                        // SAFETY: the context is current on this thread; the index buffer is
                        // attached to the bound vertex array (RGL-02).
                        unsafe {
                            gl.draw_elements_instanced_base_vertex_base_instance(
                                draw.topology,
                                index_count as i32,
                                gl_type,
                                byte_offset as i32,
                                instance_count as i32,
                                vertex_offset,
                                first_instance,
                            );
                        }
                    }
                }
                GlCommand::Dispatch {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                } => {
                    flush_draw_state!();
                    // SAFETY: the context is current on this thread.
                    unsafe { gl.dispatch_compute(group_count_x, group_count_y, group_count_z) };
                }
                GlCommand::PipelineBarrier(bits) => {
                    // Layout transitions are no-ops in GL; only shader writes need a barrier
                    // (the bits were derived when the barrier was recorded).
                    if bits != 0 {
                        // SAFETY: the context is current on this thread.
                        unsafe { gl.memory_barrier(bits) };
                    }
                }
                GlCommand::CopyBufferToBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let reg = lock(&resources.buffers);
                    if let (Some((s_buf, _)), Some((d_buf, _))) = (reg.get(src), reg.get(dst)) {
                        // SAFETY: the context is current on this thread; both buffers are live
                        // (looked up above) and the copy targets are unbound afterwards. GL
                        // reports ranges outside the buffers as errors.
                        unsafe {
                            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(s_buf.buffer));
                            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(d_buf.buffer));
                            gl.copy_buffer_sub_data(
                                glow::COPY_READ_BUFFER,
                                glow::COPY_WRITE_BUFFER,
                                src_offset as i32,
                                dst_offset as i32,
                                size as i32,
                            );
                            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
                            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
                        }
                    }
                }
                GlCommand::FillBuffer {
                    buffer,
                    offset,
                    size,
                    value,
                } => {
                    let found = lock(&resources.buffers)
                        .get(buffer)
                        .map(|(b, _)| (b.buffer, b.size));
                    // Checked here too: release builds do not wrap recorders in the validator.
                    let checked = found
                        .ok_or_else(|| format!("unknown buffer {buffer:?}"))
                        .and_then(|(gl_buffer, buffer_size)| {
                            korppi_rhi::transfer::check_fill_buffer(buffer_size, offset, size)
                                .map(|()| gl_buffer)
                        });
                    match checked {
                        Ok(gl_buffer) => {
                            // SAFETY: the context is current on this thread; the buffer is live
                            // and the range was checked above.
                            let result = unsafe {
                                crate::transfer::fill_buffer(extra, gl_buffer, offset, size, value)
                            };
                            if let Err(missing) = result {
                                diagnostics.push(
                                    DiagnosticSeverity::Error,
                                    format!("fill_buffer: {missing} is unavailable"),
                                );
                            }
                        }
                        Err(problem) => diagnostics.push(
                            DiagnosticSeverity::Error,
                            format!("fill_buffer: {problem}; skipped"),
                        ),
                    }
                }
                GlCommand::ClearTexture {
                    texture,
                    range,
                    value,
                } => {
                    let found = lock(&resources.textures)
                        .get(texture)
                        .map(|(t, _)| (t.texture, t.info));
                    let checked = found
                        .ok_or_else(|| format!("unknown texture {texture:?}"))
                        .and_then(|(gl_texture, info)| {
                            korppi_rhi::transfer::check_clear_texture(&info, &range, &value)
                                .map(|()| (gl_texture, info))
                        });
                    match checked {
                        Ok((gl_texture, info)) => {
                            // SAFETY: the context is current on this thread; the texture is live
                            // and the range and value were checked above.
                            let result = unsafe {
                                crate::transfer::clear_texture(
                                    extra, gl_texture, &info, &range, value,
                                )
                            };
                            if let Err(missing) = result {
                                diagnostics.push(
                                    DiagnosticSeverity::Error,
                                    format!("clear_texture: {missing} is unavailable"),
                                );
                            }
                        }
                        Err(problem) => diagnostics.push(
                            DiagnosticSeverity::Error,
                            format!("clear_texture: {problem}; skipped"),
                        ),
                    }
                }
                GlCommand::CopyBufferToTexture { src, dst, copy } => {
                    // One registry lock at a time (RGL-11).
                    let buffer = lock(&resources.buffers).get(src).map(|(b, _)| b.buffer);
                    if let (Some(buffer), Some(texture)) =
                        (buffer, transfer_texture(resources, dst))
                    {
                        // SAFETY: the context is current on this thread; both objects are live.
                        let result = unsafe {
                            crate::transfer::buffer_to_texture(gl, extra, texture, buffer, &copy)
                        };
                        if let Err(missing) = result {
                            diagnostics.push(
                                DiagnosticSeverity::Error,
                                format!("copy_buffer_to_texture: {missing} is unavailable"),
                            );
                        }
                    }
                }
                GlCommand::CopyTextureToBuffer { src, dst, copy } => {
                    let buffer = lock(&resources.buffers)
                        .get(dst)
                        .map(|(b, _)| (b.buffer, b.size));
                    if let (Some((buffer, size)), Some(texture)) =
                        (buffer, transfer_texture(resources, src))
                    {
                        // SAFETY: the context is current on this thread; both objects are live
                        // and `size` is the buffer's size.
                        let result = unsafe {
                            crate::transfer::texture_to_buffer(
                                gl, extra, texture, buffer, size, &copy,
                            )
                        };
                        if let Err(missing) = result {
                            diagnostics.push(
                                DiagnosticSeverity::Error,
                                format!("copy_texture_to_buffer: {missing} is unavailable"),
                            );
                        }
                    }
                }
                GlCommand::CopyTextureToTexture { src, dst, copy } => {
                    if let (Some(s), Some(d)) = (
                        transfer_texture(resources, src),
                        transfer_texture(resources, dst),
                    ) {
                        // Layers (and cube faces) are z; 3D textures copy slices.
                        let z = |t: &crate::transfer::TransferTexture, layer: u32, origin: u32| {
                            if t.target == glow::TEXTURE_3D {
                                origin as i32
                            } else {
                                layer as i32
                            }
                        };
                        let depth = if s.target == glow::TEXTURE_3D {
                            copy.extent[2]
                        } else {
                            copy.layer_count
                        } as i32;
                        // SAFETY: the context is current on this thread; both textures are live
                        // and the regions were validated (debug builds).
                        unsafe {
                            gl.copy_image_sub_data(
                                s.texture,
                                s.target,
                                copy.src_mip_level as i32,
                                copy.src_origin[0] as i32,
                                copy.src_origin[1] as i32,
                                z(&s, copy.src_layer, copy.src_origin[2]),
                                d.texture,
                                d.target,
                                copy.dst_mip_level as i32,
                                copy.dst_origin[0] as i32,
                                copy.dst_origin[1] as i32,
                                z(&d, copy.dst_layer, copy.dst_origin[2]),
                                copy.extent[0] as i32,
                                copy.extent[1] as i32,
                                depth,
                            );
                        }
                    }
                }
                GlCommand::BlitTexture { src, dst, blit } => {
                    let Some(format) = transfer_texture(resources, src).map(|t| t.format) else {
                        continue;
                    };
                    // Blits go through framebuffers: one per layer (or 3D slice) and side.
                    let key = |id: u32, mip: u32, layer: u32| {
                        if format.is_depth() || format.is_stencil() {
                            FboKey {
                                color_attachments: [None; MAX_FBO_COLOR_ATTACHMENTS],
                                depth_attachment: Some((id, mip, layer)),
                            }
                        } else {
                            FboKey::single_color((id, mip, layer))
                        }
                    };
                    let mask = if format.is_depth() || format.is_stencil() {
                        let mut mask = glow::DEPTH_BUFFER_BIT;
                        if format.is_stencil() {
                            mask |= glow::STENCIL_BUFFER_BIT;
                        }
                        mask
                    } else {
                        glow::COLOR_BUFFER_BIT
                    };
                    let filter = match blit.filter {
                        FilterMode::Nearest => glow::NEAREST,
                        FilterMode::Linear => glow::LINEAR,
                    };
                    for i in 0..blit.layer_count {
                        let fbos = {
                            let textures = lock(&resources.textures);
                            fbo_cache
                                .get_or_create(
                                    gl,
                                    &key(src.id(), blit.src_mip_level, blit.src_layer + i),
                                    &textures,
                                )
                                .and_then(|read| {
                                    fbo_cache
                                        .get_or_create(
                                            gl,
                                            &key(dst.id(), blit.dst_mip_level, blit.dst_layer + i),
                                            &textures,
                                        )
                                        .map(|draw| (read, draw))
                                })
                        };
                        let (read, draw) = match fbos {
                            Ok(fbos) => fbos,
                            Err(error) => {
                                diagnostics.push(
                                    DiagnosticSeverity::Error,
                                    format!("blit_texture: {error}"),
                                );
                                break;
                            }
                        };
                        let [sx0, sy0, sx1, sy1] = blit.src_region.map(|v| v as i32);
                        let [dx0, dy0, dx1, dy1] = blit.dst_region.map(|v| v as i32);
                        // SAFETY: the context is current on this thread; both framebuffers are
                        // complete. Blits honour the scissor test, which is disabled here.
                        unsafe {
                            state.disable_scissor(gl);
                            gl.blit_named_framebuffer(
                                Some(read),
                                Some(draw),
                                sx0,
                                sy0,
                                sx1,
                                sy1,
                                dx0,
                                dy0,
                                dx1,
                                dy1,
                                mask,
                                filter,
                            );
                        }
                    }
                }
                GlCommand::GenerateMipmaps(texture) => {
                    if let Some(texture) = transfer_texture(resources, texture) {
                        // SAFETY: the context is current on this thread; the texture is live.
                        unsafe { gl.generate_texture_mipmap(texture.texture) };
                    }
                }
                GlCommand::DrawIndirect {
                    buffer,
                    offset,
                    draw_count,
                    stride,
                    indexed,
                } => {
                    flush_draw_state!();
                    let Some(args) = lock(&resources.buffers).get(buffer).map(|(b, _)| b.buffer)
                    else {
                        continue;
                    };
                    let index_type = draw.index_buffer.map(|(_, _, ty)| to_gl_index_type(ty).0);
                    let pointer = offset as usize as *const std::ffi::c_void;
                    // SAFETY: the context is current on this thread; the argument buffer is
                    // bound as the draw-indirect buffer, so `pointer` is an offset into it, and
                    // the records lie in range (validated in debug builds). Indexed draws use
                    // the element buffer of the bound vertex array, at offset 0.
                    let drawn = unsafe {
                        gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(args));
                        let drawn = match (indexed, index_type) {
                            (false, _) => extra.multi_draw_arrays_indirect.map(|f| {
                                f(draw.topology, pointer, draw_count as i32, stride as i32)
                            }),
                            (true, Some(ty)) => extra.multi_draw_elements_indirect.map(|f| {
                                f(draw.topology, ty, pointer, draw_count as i32, stride as i32)
                            }),
                            (true, None) => Some(()),
                        };
                        gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, None);
                        drawn
                    };
                    if drawn.is_none() {
                        diagnostics.push(
                            DiagnosticSeverity::Error,
                            "glMultiDraw*Indirect is unavailable",
                        );
                    }
                }
                GlCommand::DispatchIndirect { buffer, offset } => {
                    flush_draw_state!();
                    let args = lock(&resources.buffers).get(buffer).map(|(b, _)| b.buffer);
                    if let Some(args) = args {
                        // SAFETY: the context is current on this thread; the record lies in
                        // the bound dispatch-indirect buffer.
                        unsafe {
                            gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, Some(args));
                            gl.dispatch_compute_indirect(offset as i32);
                            gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, None);
                        }
                    }
                }
                GlCommand::WriteTimestamp { query_set, index } => {
                    let query = lock(&resources.query_sets)
                        .get(query_set)
                        .and_then(|(queries, _)| queries.get(index as usize).copied());
                    if let Some(query) = query {
                        // SAFETY: the context is current on this thread; the query is live.
                        unsafe { gl.query_counter(query, glow::TIMESTAMP) };
                    }
                }
                GlCommand::ResolveQuerySet {
                    query_set,
                    first,
                    count,
                    dst,
                    dst_offset,
                } => {
                    let queries: Vec<glow::Query> = lock(&resources.query_sets)
                        .get(query_set)
                        .map(|(queries, _)| {
                            queries
                                .iter()
                                .skip(first as usize)
                                .take(count as usize)
                                .copied()
                                .collect()
                        })
                        .unwrap_or_default();
                    let dst = lock(&resources.buffers).get(dst).map(|(b, _)| b.buffer);
                    if let Some(dst) = dst {
                        // SAFETY: the context is current on this thread; with the destination
                        // bound as the query buffer, each result is written (waiting for it on
                        // the GPU) at an offset into it, in range (validated in debug builds).
                        unsafe {
                            gl.bind_buffer(glow::QUERY_BUFFER, Some(dst));
                            for (i, query) in queries.into_iter().enumerate() {
                                gl.get_query_parameter_u64_with_offset(
                                    query,
                                    glow::QUERY_RESULT,
                                    (dst_offset + 8 * i as u64) as usize,
                                );
                            }
                            gl.bind_buffer(glow::QUERY_BUFFER, None);
                        }
                    }
                }
                GlCommand::PushDebugGroup { start, len } => {
                    let label = std::str::from_utf8(stream.bytes(start, len)).unwrap_or("?");
                    // SAFETY: the context is current on this thread.
                    unsafe { gl.push_debug_group(glow::DEBUG_SOURCE_APPLICATION, 0, label) };
                }
                GlCommand::PopDebugGroup => {
                    // SAFETY: the context is current on this thread; a group is open
                    // (validated in debug builds).
                    unsafe { gl.pop_debug_group() };
                }
                GlCommand::InsertDebugMarker { start, len } => {
                    let label = std::str::from_utf8(stream.bytes(start, len)).unwrap_or("?");
                    // SAFETY: the context is current on this thread.
                    unsafe {
                        gl.debug_message_insert(
                            glow::DEBUG_SOURCE_APPLICATION,
                            glow::DEBUG_TYPE_MARKER,
                            0,
                            glow::DEBUG_SEVERITY_NOTIFICATION,
                            label,
                        )
                    };
                }
            }
        }

        Ok(())
    }
}

impl Drop for GlThread {
    /// Deletes every remaining GL object. Runs on the render thread, before the fields (and with
    /// them the context) are dropped.
    fn drop(&mut self) {
        // Window surfaces first: they are released while the context is still alive.
        self.destroy_all_surfaces();
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread; every name below belongs to it
        // and is deleted exactly once (the registry entries are drained).
        unsafe {
            gl.finish();
            for (_, fence) in self.pending_fences.drain(..) {
                gl.delete_sync(fence);
            }
            gl.delete_vertex_array(self.default_vao);
            gl.delete_buffer(self.push_constant_buffer);

            self.fbo_cache.destroy_all(gl);

            for (_, (p, _)) in lock(&self.resources.pipelines).drain() {
                gl.delete_program(p.program);
                if let Some(vao) = p.vertex_array {
                    gl.delete_vertex_array(vao);
                }
            }
            for (_, (s, _)) in lock(&self.resources.shaders).drain() {
                if let GlShaderCode::Compiled { shader: object, .. } = s.code {
                    gl.delete_shader(object);
                }
            }
            for (_, (queries, _)) in lock(&self.resources.query_sets).drain() {
                for query in queries {
                    gl.delete_query(query);
                }
            }
            lock(&self.resources.bind_groups).clear();
            lock(&self.resources.bind_group_layouts).clear();
            for (_, (v, _)) in lock(&self.resources.texture_views).drain() {
                gl.delete_texture(v.texture);
            }
            for (_, (s, _)) in lock(&self.resources.samplers).drain() {
                gl.delete_sampler(s.sampler);
            }
            for (_, (t, _)) in lock(&self.resources.textures).drain() {
                gl.delete_texture(t.texture);
            }
            for (_, (b, _)) in lock(&self.resources.buffers).drain() {
                gl.delete_buffer(b.buffer);
            }
        }
    }
}

/// A shader as given to the render thread.
pub(crate) enum ShaderInput {
    /// GLSL source.
    Glsl(String),
    /// Parsed SPIR-V.
    Spirv(Arc<SpirvShader>),
}

/// A pipeline stage as given to the render thread: a compiled shader object, or GLSL
/// translated for the pipeline.
pub(crate) enum PipelineShader {
    /// A compiled GLSL shader, its stage and source hash.
    Compiled(glow::Shader, ShaderStage, u64),
    /// GLSL translated for the pipeline.
    Source(ShaderStage, String),
}

impl PipelineShader {
    /// (stage, source hash), for program-binary keys.
    fn key(&self) -> (u32, u64) {
        match self {
            Self::Compiled(_, stage, hash) => (to_gl_shader_stage(*stage), *hash),
            Self::Source(stage, source) => (
                to_gl_shader_stage(*stage),
                crate::program_cache::source_hash(source),
            ),
        }
    }
}

/// A pipeline's binding points and bind group layout summaries.
pub(crate) struct PipelineBindings {
    table: GlBindingTable,
    layouts: Vec<BindGroupLayoutInfo>,
}

impl OpenGlDevice {
    /// The binding table of a pipeline's bind group layouts, checked against the per-stage
    /// limits.
    fn pipeline_bindings(
        &self,
        handles: &[BindGroupLayoutHandle],
        stages: ShaderStages,
    ) -> Result<PipelineBindings, RhiError> {
        let registry = lock(&self.resources.bind_group_layouts);
        let mut layouts = Vec::with_capacity(handles.len());
        for handle in handles {
            let (layout, _) = registry.get(handle).ok_or_else(|| {
                RhiError::InvalidHandle(format!("bind group layout {handle:?} not found"))
            })?;
            layouts.push(layout);
        }
        let descs: Vec<&BindGroupLayoutDescriptor> = layouts.iter().map(|l| &l.desc).collect();
        check_pipeline_layouts(&descs, stages, &self.capabilities.limits)?;
        Ok(PipelineBindings {
            table: GlBindingTable::new(&descs)?,
            layouts: layouts.iter().map(|l| l.info).collect(),
        })
    }

    /// The `stage` shader `handle` for a pipeline with binding points `table`.
    fn pipeline_shader(
        &self,
        handle: ShaderHandle,
        stage: ShaderStage,
        table: &GlBindingTable,
    ) -> Result<PipelineShader, RhiError> {
        let code = {
            let shaders = lock(&self.resources.shaders);
            let (shader, _) = shaders.get(handle).ok_or_else(|| {
                RhiError::InvalidHandle(format!("{stage:?} shader {handle:?} not found"))
            })?;
            if shader.stage != stage {
                return Err(RhiError::ValidationError(format!(
                    "shader {handle:?} is a {:?} shader, not a {stage:?} shader",
                    shader.stage
                )));
            }
            match &shader.code {
                GlShaderCode::Compiled {
                    shader: object,
                    source_hash,
                } => return Ok(PipelineShader::Compiled(*object, stage, *source_hash)),
                GlShaderCode::Spirv(parsed) => Arc::clone(parsed),
            }
        };
        let glsl = code.to_glsl(
            &|set, binding, kind| table.point(set, binding, kind),
            self.draw_parameters,
        )?;
        Ok(PipelineShader::Source(stage, glsl))
    }
}

impl GraphicsDevice for OpenGlDevice {
    fn adapter_info(&self) -> AdapterInfo {
        self.capabilities.adapter.clone()
    }

    fn limits(&self) -> DeviceLimits {
        self.capabilities.limits.clone()
    }

    fn features(&self) -> DeviceFeatures {
        self.capabilities.features
    }

    fn format_capabilities(&self, format: TextureFormat) -> FormatCapabilities {
        self.capabilities.format(format)
    }

    fn create_buffer(
        &self,
        desc: &BufferDescriptor,
        initial_data: Option<&[u8]>,
    ) -> Result<BufferHandle, RhiError> {
        desc.validate()?;
        if let Some(data) = initial_data {
            desc.validate_initial_data(data)?;
        }
        // Sizes and offsets are passed to GL as 32-bit signed values (RGL-10).
        if desc.size > i32::MAX as u64 {
            return Err(RhiError::Unsupported(format!(
                "OpenGL buffers are limited to {} bytes; {} requested",
                i32::MAX,
                desc.size
            )));
        }
        self.thread
            .call(|gl| gl.create_buffer(desc, initial_data))?
    }

    fn destroy_buffer(&self, handle: BufferHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_buffer(handle))?
    }

    fn write_buffer(&self, handle: BufferHandle, offset: u64, data: &[u8]) -> Result<(), RhiError> {
        self.thread
            .call(|gl| gl.write_buffer(handle, offset, data))?
    }

    fn read_buffer(
        &self,
        handle: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        self.thread
            .call(|gl| gl.read_buffer(handle, offset, size))?
    }

    fn map_buffer(&self, handle: BufferHandle) -> Result<MappedBuffer<'_>, RhiError> {
        let buffers = lock(&self.resources.buffers);
        let (buf, _) = buffers
            .get(handle)
            .ok_or_else(|| buffer_not_found(handle))?;
        let ptr = buf.mapped.ok_or_else(|| {
            RhiError::ValidationError(format!(
                "{handle:?} is GpuOnly; only CpuToGpu and GpuToCpu buffers can be mapped"
            ))
        })?;
        let guard = buf.host_access.acquire_for("map_buffer")?;
        // SAFETY: the persistent mapping lives until the buffer is deleted, which
        // `destroy_buffer` refuses while the guard's lock is held, and the device (borrowed for
        // the guard's lifetime) deletes buffers only when it drops. It covers `size` bytes.
        Ok(unsafe { MappedBuffer::new(ptr.0, buf.size, guard) })
    }

    fn create_texture(
        &self,
        desc: &TextureDescriptor,
        initial_data: Option<&[u8]>,
    ) -> Result<TextureHandle, RhiError> {
        desc.validate()?;
        let levels = initial_data
            .map(|data| desc.validate_initial_data(data))
            .transpose()?;
        // Usage the format must support on this device (RHI-09); initial data is copied in.
        let mut required_usage = desc.usage;
        if initial_data.is_some() {
            required_usage |= TextureUsage::COPY_DST;
        }
        self.capabilities.format(desc.format).check(
            desc.format,
            required_usage,
            desc.sample_count,
        )?;
        let upload = initial_data.zip(levels);
        self.thread.call(|gl| gl.create_texture(desc, upload))?
    }

    fn destroy_texture(&self, handle: TextureHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_texture(handle))?
    }

    fn create_texture_view(
        &self,
        desc: &TextureViewDescriptor,
    ) -> Result<TextureViewHandle, RhiError> {
        self.thread.call(|gl| gl.create_texture_view(desc))?
    }

    fn destroy_texture_view(&self, view: TextureViewHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_texture_view(view))?
    }

    fn read_texture(&self, handle: TextureHandle, mip_level: u32) -> Result<Vec<u8>, RhiError> {
        self.thread.call(|gl| gl.read_texture(handle, mip_level))?
    }

    fn create_sampler(&self, desc: &SamplerDescriptor) -> Result<SamplerHandle, RhiError> {
        desc.validate()?;
        self.thread.call(|gl| gl.create_sampler(desc))?
    }

    fn destroy_sampler(&self, handle: SamplerHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_sampler(handle))?
    }

    /// GLSL is compiled as given, with its own binding points. SPIR-V is parsed and validated
    /// on the calling thread, and translated to GLSL 4.50 for each pipeline that uses it, with
    /// that pipeline's binding points (RHI-07, RHI-08, RGL-12; see [`crate::spirv`]).
    fn create_shader(&self, desc: &ShaderDescriptor) -> Result<ShaderHandle, RhiError> {
        // SPIR-V arrives in native byte order with its framing checked (RVK-18).
        let input = match (&desc.source, desc.spirv()?) {
            (ShaderSource::Glsl(code), _) => ShaderInput::Glsl((*code).to_owned()),
            (_, Some(words)) => ShaderInput::Spirv(Arc::new(parse_spirv_words(
                &words,
                desc.stage,
                desc.entry_point,
            )?)),
            (_, None) => unreachable!("only GLSL has no SPIR-V words"),
        };
        let stage = desc.stage;
        self.thread.call(move |gl| gl.create_shader(stage, input))?
    }

    fn destroy_shader(&self, handle: ShaderHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_shader(handle))?
    }

    fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        desc.validate()?;
        let missing = desc.rasterizer.required_features() - self.capabilities.features;
        if !missing.is_empty() {
            return Err(RhiError::Unsupported(format!(
                "the rasterizer state needs {missing:?}, which this device lacks"
            )));
        }
        let stages = if desc.fragment_shader.is_some() {
            ShaderStages::VERTEX_FRAGMENT
        } else {
            ShaderStages::VERTEX
        };
        let bindings = self.pipeline_bindings(&desc.bind_group_layouts, stages)?;
        let mut shaders =
            vec![self.pipeline_shader(desc.vertex_shader, ShaderStage::Vertex, &bindings.table)?];
        if let Some(fragment) = desc.fragment_shader {
            shaders.push(self.pipeline_shader(fragment, ShaderStage::Fragment, &bindings.table)?);
        }
        self.thread
            .call(|gl| gl.create_graphics_pipeline(desc, &shaders, bindings))?
    }

    fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        desc.validate()?;
        let bindings = self.pipeline_bindings(&desc.bind_group_layouts, ShaderStages::COMPUTE)?;
        let shader =
            self.pipeline_shader(desc.compute_shader, ShaderStage::Compute, &bindings.table)?;
        self.thread
            .call(|gl| gl.create_compute_pipeline(desc, &shader, bindings))?
    }

    fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayoutHandle, RhiError> {
        self.thread.call(|gl| gl.create_bind_group_layout(desc))?
    }

    fn create_query_set(
        &self,
        desc: &korppi_rhi::QuerySetDescriptor,
    ) -> Result<korppi_rhi::QuerySetHandle, RhiError> {
        if !self
            .capabilities
            .features
            .contains(korppi_rhi::DeviceFeatures::TIMESTAMP_QUERY)
        {
            return Err(RhiError::Unsupported(
                "timestamp queries are unavailable".into(),
            ));
        }
        if desc.count == 0 {
            return Err(RhiError::ValidationError(
                "query sets need at least one query".into(),
            ));
        }
        self.thread.call(|gl| gl.create_query_set(desc))?
    }

    fn destroy_query_set(&self, query_set: korppi_rhi::QuerySetHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_query_set(query_set))?
    }

    fn destroy_bind_group_layout(&self, layout: BindGroupLayoutHandle) -> Result<(), RhiError> {
        self.thread
            .call(|gl| gl.destroy_bind_group_layout(layout))?
    }

    fn create_bind_group(&self, desc: &BindGroupDescriptor) -> Result<BindGroupHandle, RhiError> {
        let limits = &self.capabilities.limits;
        self.thread.call(|gl| gl.create_bind_group(desc, limits))?
    }

    fn destroy_bind_group(&self, group: BindGroupHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_bind_group(group))?
    }

    fn destroy_pipeline(&self, handle: PipelineHandle) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.destroy_pipeline(handle))?
    }

    fn pipeline_cache_data(&self) -> Result<Vec<u8>, RhiError> {
        self.thread.call(|gl| gl.program_binaries.serialize())
    }

    fn load_pipeline_cache(&self, data: &[u8]) -> Result<bool, RhiError> {
        self.thread.call(|gl| gl.program_binaries.load(data))
    }

    fn create_command_recorder(&self) -> Result<Box<dyn CommandBufferRecorder>, RhiError> {
        // Recorders, command buffers, validators and command streams are pooled: steady-state
        // frames allocate none of them (RHI-11).
        let recorder: Box<dyn CommandBufferRecorder> = GlCommandBufferRecorder::pooled(
            self.command_streams.clone(),
            self.resources.device,
            &self.recorder_pool,
            &self.command_buffer_pool,
        );
        // Every recorder is validated, in every build (RHI-16).
        Ok(validate_recorder_pooled(
            recorder,
            self.resources.clone(),
            &self.validator_pool,
        ))
    }

    /// Queues the command buffer on the render thread and returns. Commands run in submission
    /// order, before any later device call; execution errors are reported as diagnostics.
    fn submit(&self, command_buffer: Box<dyn CommandBuffer>) -> Result<SubmissionId, RhiError> {
        let command_buffer: Box<dyn std::any::Any> = command_buffer;
        let mut buffer = command_buffer.downcast::<GlCommandBuffer>().map_err(|_| {
            RhiError::InvalidHandle("command buffer was not recorded by an OpenGL device".into())
        })?;
        // Its commands name the recording device's handles (RGL-22).
        if buffer.device != self.resources.device {
            return Err(RhiError::InvalidHandle(
                "command buffer was recorded by another OpenGL device".into(),
            ));
        }
        let stream = buffer.stream.take().unwrap_or_default();
        let pool = buffer.pool.clone();
        self.command_buffer_pool.give(buffer);
        self.post(|submission| GlWork::Submit(submission, stream, pool))
    }

    fn is_submission_complete(&self, submission: SubmissionId) -> Result<bool, RhiError> {
        self.check_submission(submission)?;
        if !self.is_completed(submission) {
            self.thread.call(|gl| gl.poll_fences())?;
        }
        Ok(self.is_completed(submission))
    }

    fn wait_for_submission(&self, submission: SubmissionId) -> Result<(), RhiError> {
        self.check_submission(submission)?;
        if self.is_completed(submission) {
            return Ok(());
        }
        self.thread.call(|gl| gl.wait_fences(submission))?
    }

    fn wait_idle(&self) -> Result<(), RhiError> {
        self.thread.call(|gl| gl.finish())
    }

    fn begin_frame(&self) -> Result<FrameIndex, RhiError> {
        let (frame, wait_for) = self.frames.begin_frame();
        if let Some(submission) = wait_for {
            self.wait_for_submission(submission)?;
        }
        Ok(frame)
    }

    fn create_swapchain(
        &self,
        target: Arc<dyn PresentTarget>,
        desc: &SwapchainDescriptor,
    ) -> Result<SwapchainHandle, RhiError> {
        self.create_swapchain_impl(target, desc)
    }

    fn destroy_swapchain(&self, swapchain: SwapchainHandle) -> Result<(), RhiError> {
        self.destroy_swapchain_impl(swapchain)
    }

    fn resize_swapchain(
        &self,
        swapchain: SwapchainHandle,
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        self.resize_swapchain_impl(swapchain, width, height)
    }

    fn swapchain_info(&self, swapchain: SwapchainHandle) -> Result<SwapchainInfo, RhiError> {
        self.swapchain_info_impl(swapchain)
    }

    fn acquire_next_image(
        &self,
        swapchain: SwapchainHandle,
    ) -> Result<Option<SwapchainImage>, RhiError> {
        self.acquire_impl(swapchain)
    }

    fn present(&self, image: &SwapchainImage) -> Result<(), RhiError> {
        self.present_impl(image)
    }

    fn api_validation_enabled(&self) -> bool {
        self.debug_output_enabled
    }

    /// Drains the render thread's queue first, so errors raised by every command buffer
    /// submitted before this call are included.
    fn take_diagnostics(&self) -> Vec<BackendDiagnostic> {
        let _ = self.thread.call(|_| ());
        self.diagnostics.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glow::HasContext;
    use korppi_rhi::{QuerySetDescriptor, QueryType};

    /// ISSUE-008: a query set's queries are query objects as soon as the set exists, so they
    /// can be labelled. A name from `glGenQueries` is not an object until its first use, and
    /// `glObjectLabel` or `glGetQueryObject` on it is an error (NVIDIA reports the label).
    #[test]
    fn query_set_queries_exist_before_their_first_use() {
        let device = OpenGlDevice::create_headless().expect("OpenGL device");
        if !device.api_validation_enabled() {
            eprintln!("skipped: API validation is disabled for this process");
            return;
        }
        let handle = device
            .create_query_set(&QuerySetDescriptor {
                ty: QueryType::Timestamp,
                count: 2,
                label: Some("unused_timestamps"),
            })
            .expect("query set");
        let queries = lock(&device.resources.query_sets)
            .get(handle)
            .expect("registered")
            .0
            .clone();
        device
            .with_gl(|gl| {
                for query in queries {
                    // SAFETY: `with_gl` runs on the render thread, where the context is current;
                    // the queries are live and not active.
                    unsafe { gl.get_query_parameter_u32(query, glow::QUERY_RESULT_AVAILABLE) };
                }
            })
            .expect("render thread");
        let errors: Vec<_> = device
            .take_diagnostics()
            .into_iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();
        assert!(errors.is_empty(), "GL errors: {errors:#?}");
        device.destroy_query_set(handle).expect("destroy");
    }
}
