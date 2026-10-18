use crate::capabilities::{AdapterInfo, DeviceFeatures, DeviceLimits, FormatCapabilities};
use crate::command::{CommandBuffer, CommandBufferRecorder};
use crate::descriptor::{
    BufferDescriptor, ComputePipelineDescriptor, GraphicsPipelineDescriptor, SamplerDescriptor,
    ShaderDescriptor, TextureDescriptor,
};
use crate::diagnostics::BackendDiagnostic;
use crate::error::RhiError;
use crate::handle::{
    BufferHandle, PipelineHandle, SamplerHandle, ShaderHandle, SwapchainHandle, TextureHandle,
};
use crate::submission::{FrameIndex, SubmissionId};
use crate::swapchain::{PresentTarget, SwapchainDescriptor, SwapchainImage, SwapchainInfo};
use std::sync::Arc;

/// Core abstraction trait implemented by graphics hardware backends (Vulkan, OpenGL).
pub trait GraphicsDevice: Send + Sync + 'static {
    /// Identifies the device and driver.
    fn adapter_info(&self) -> AdapterInfo;

    /// Numeric limits of the device (RHI-09).
    fn limits(&self) -> DeviceLimits;

    /// Optional features the device supports and has enabled.
    fn features(&self) -> DeviceFeatures;

    /// What `format` supports on this device. `create_texture` rejects usages outside it.
    fn format_capabilities(&self, format: crate::types::TextureFormat) -> FormatCapabilities;

    /// Allocates and initializes a GPU buffer.
    fn create_buffer(
        &self,
        desc: &BufferDescriptor,
        initial_data: Option<&[u8]>,
    ) -> Result<BufferHandle, RhiError>;

    /// Destroys a previously allocated buffer.
    fn destroy_buffer(&self, handle: BufferHandle) -> Result<(), RhiError>;

    /// Updates bytes in an existing buffer.
    fn write_buffer(&self, handle: BufferHandle, offset: u64, data: &[u8]) -> Result<(), RhiError>;

    /// Reads back bytes from a GPU buffer to the CPU.
    ///
    /// Reads the buffer's current contents: wait for submissions that write it first
    /// ([`GraphicsDevice::wait_for_submission`]).
    fn read_buffer(
        &self,
        handle: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError>;

    /// Host access to the memory of a `CpuToGpu` or `GpuToCpu` buffer, which stays mapped for
    /// its lifetime (see `crate::mapping` for the safety model). `GpuOnly` buffers cannot be
    /// mapped (`ValidationError`); neither can a buffer whose guard still exists.
    fn map_buffer(
        &self,
        handle: BufferHandle,
    ) -> Result<crate::mapping::MappedBuffer<'_>, RhiError> {
        Err(RhiError::Unsupported(format!(
            "{handle:?}: this device cannot map buffers"
        )))
    }

    /// Allocates a GPU texture.
    ///
    /// The format must support the usage on this device ([`GraphicsDevice::format_capabilities`];
    /// `RhiError::Unsupported` otherwise; initial data also needs `COPY_DST` support). A texture
    /// created with initial data is left in the layout of its most specific use:
    /// `ShaderReadOnly` if sampled, else `ColorAttachment`, `DepthStencilAttachment`, `General`
    /// (storage), `TransferSrc` or `TransferDst`; without initial data it is `Undefined`.
    fn create_texture(
        &self,
        desc: &TextureDescriptor,
        initial_data: Option<&[u8]>,
    ) -> Result<TextureHandle, RhiError>;

    /// Destroys a previously allocated texture.
    fn destroy_texture(&self, handle: TextureHandle) -> Result<(), RhiError>;

    /// Creates a view of a range of `desc.texture`'s mips and layers (RHI-05). Views are
    /// destroyed with their texture.
    fn create_texture_view(
        &self,
        desc: &crate::descriptor::TextureViewDescriptor,
    ) -> Result<crate::handle::TextureViewHandle, RhiError>;

    /// Destroys a texture view once the GPU no longer uses it.
    fn destroy_texture_view(&self, view: crate::handle::TextureViewHandle) -> Result<(), RhiError>;

    /// Reads back one mip level of a texture, tightly packed, row 0 first (whole blocks for
    /// compressed formats, the depth aspect for depth/stencil formats). The texture keeps its
    /// layout. `Depth24PlusStencil8` cannot be read back (`Unsupported`).
    fn read_texture(&self, handle: TextureHandle, mip_level: u32) -> Result<Vec<u8>, RhiError>;

    /// Creates a bind group layout: the declaration of one descriptor set (RHI-07).
    fn create_bind_group_layout(
        &self,
        desc: &crate::binding::BindGroupLayoutDescriptor,
    ) -> Result<crate::handle::BindGroupLayoutHandle, RhiError>;

    /// Destroys a bind group layout. Pipelines and groups created with it stay valid.
    fn destroy_bind_group_layout(
        &self,
        layout: crate::handle::BindGroupLayoutHandle,
    ) -> Result<(), RhiError>;

    /// Creates a bind group: `desc.layout` filled with resources. Destroying a resource it
    /// refers to makes the group unusable.
    fn create_bind_group(
        &self,
        desc: &crate::binding::BindGroupDescriptor,
    ) -> Result<crate::handle::BindGroupHandle, RhiError>;

    /// Destroys a bind group once the GPU no longer uses it.
    fn destroy_bind_group(&self, group: crate::handle::BindGroupHandle) -> Result<(), RhiError>;

    /// Creates a set of queries (RHI-05; see `crate::query`).
    fn create_query_set(
        &self,
        desc: &crate::query::QuerySetDescriptor,
    ) -> Result<crate::handle::QuerySetHandle, RhiError> {
        Err(RhiError::Unsupported(format!(
            "{:?} query sets are not supported by this device",
            desc.ty
        )))
    }

    /// Destroys a query set once the GPU no longer uses it.
    fn destroy_query_set(&self, query_set: crate::handle::QuerySetHandle) -> Result<(), RhiError> {
        Err(RhiError::InvalidHandle(format!("{query_set:?}")))
    }

    /// Creates a texture sampler.
    fn create_sampler(&self, desc: &SamplerDescriptor) -> Result<SamplerHandle, RhiError>;

    /// Destroys a sampler.
    fn destroy_sampler(&self, handle: SamplerHandle) -> Result<(), RhiError>;

    /// Compiles or loads a shader module.
    fn create_shader(&self, desc: &ShaderDescriptor) -> Result<ShaderHandle, RhiError>;

    /// Destroys a shader module.
    fn destroy_shader(&self, handle: ShaderHandle) -> Result<(), RhiError>;

    /// Compiles a graphics pipeline state object.
    fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError>;

    /// Compiles a compute pipeline state object.
    fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError>;

    /// Destroys a pipeline state object.
    fn destroy_pipeline(&self, handle: PipelineHandle) -> Result<(), RhiError>;

    /// The driver's pipeline caches serialized for a later run (Vulkan `VkPipelineCache`
    /// data, OpenGL program binaries), tagged with the backend and driver they belong to.
    /// Empty if the device keeps none (RVK-16, RGL-12).
    fn pipeline_cache_data(&self) -> Result<Vec<u8>, RhiError> {
        Ok(Vec::new())
    }

    /// Seeds the pipeline caches with `pipeline_cache_data` of an earlier run, so pipelines
    /// created afterwards skip compilation. Data of another backend, driver or device, and data
    /// the driver refuses (a truncated or damaged file), is ignored; returns whether `data` was
    /// used.
    fn load_pipeline_cache(&self, data: &[u8]) -> Result<bool, RhiError> {
        let _ = data;
        Ok(false)
    }

    /// Creates a command recorder for building command buffers.
    fn create_command_recorder(&self) -> Result<Box<dyn CommandBufferRecorder>, RhiError>;

    /// Queues a finalized command buffer for execution and returns without waiting for the GPU.
    ///
    /// Resources the command buffer uses must stay alive until the returned submission has
    /// completed ([`GraphicsDevice::is_submission_complete`]), and the CPU must not write memory
    /// the GPU may still be reading; backends defer their own internal recycling accordingly.
    fn submit(&self, command_buffer: Box<dyn CommandBuffer>) -> Result<SubmissionId, RhiError>;

    /// Returns whether the GPU has finished `submission` (and therefore every earlier one).
    ///
    /// # Errors
    /// `RhiError::ValidationError` if `submission` was not returned by this device.
    fn is_submission_complete(&self, submission: SubmissionId) -> Result<bool, RhiError>;

    /// Blocks until the GPU has finished `submission` and every earlier submission.
    ///
    /// # Errors
    /// `RhiError::ValidationError` if `submission` was not returned by this device.
    fn wait_for_submission(&self, submission: SubmissionId) -> Result<(), RhiError>;

    /// Submits `command_buffer` and waits for it to complete (tests, readback, tools).
    fn submit_and_wait(
        &self,
        command_buffer: Box<dyn CommandBuffer>,
    ) -> Result<SubmissionId, RhiError> {
        let submission = self.submit(command_buffer)?;
        self.wait_for_submission(submission)?;
        Ok(submission)
    }

    /// Blocks CPU execution until all pending GPU operations have completed.
    fn wait_idle(&self) -> Result<(), RhiError>;

    /// Starts a new frame and returns its slot in `0..FRAMES_IN_FLIGHT`.
    ///
    /// Waits until the GPU has finished the work submitted in the frame that last used this slot
    /// ([`crate::FRAMES_IN_FLIGHT`] frames ago), then recycles that frame's transient resources.
    /// The CPU therefore runs at most `FRAMES_IN_FLIGHT` frames ahead of the GPU.
    fn begin_frame(&self) -> Result<FrameIndex, RhiError>;

    /// Signals the completion of a rendering frame.
    fn end_frame(&self) -> Result<(), RhiError> {
        Ok(())
    }

    /// Creates a swapchain that presents to `target` (see [`crate::swapchain`]).
    ///
    /// # Errors
    /// `RhiError::Unsupported` if the device cannot present to this kind of window (for example
    /// an OpenGL device created headless, or a window system the backend does not support).
    fn create_swapchain(
        &self,
        target: Arc<dyn PresentTarget>,
        desc: &SwapchainDescriptor,
    ) -> Result<SwapchainHandle, RhiError> {
        let _ = (target, desc);
        Err(RhiError::Unsupported(
            "this device cannot present to windows".into(),
        ))
    }

    /// Destroys a swapchain and its textures once the GPU no longer uses them.
    fn destroy_swapchain(&self, swapchain: SwapchainHandle) -> Result<(), RhiError> {
        Err(RhiError::InvalidHandle(format!(
            "swapchain {swapchain:?} not found"
        )))
    }

    /// Requests a new size (in physical pixels), applied at the next acquisition. Call it when
    /// the window is resized; a size of 0 pauses presentation until a non-zero size is set.
    fn resize_swapchain(
        &self,
        swapchain: SwapchainHandle,
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        let _ = (width, height);
        Err(RhiError::InvalidHandle(format!(
            "swapchain {swapchain:?} not found"
        )))
    }

    /// Returns the current format, size and presentation mode of a swapchain.
    fn swapchain_info(&self, swapchain: SwapchainHandle) -> Result<SwapchainInfo, RhiError> {
        Err(RhiError::InvalidHandle(format!(
            "swapchain {swapchain:?} not found"
        )))
    }

    /// Acquires the next image to render into. Returns `None` while the window has no area
    /// (minimized); skip rendering that frame. Recreates the swapchain first if it was resized
    /// or reported out of date.
    ///
    /// # Errors
    /// `RhiError::ValidationError` if the previously acquired image was not presented yet.
    fn acquire_next_image(
        &self,
        swapchain: SwapchainHandle,
    ) -> Result<Option<SwapchainImage>, RhiError> {
        Err(RhiError::InvalidHandle(format!(
            "swapchain {swapchain:?} not found"
        )))
    }

    /// Queues `image` for display after every submission made so far, and returns without
    /// waiting for it. An out-of-date swapchain is not an error: it is recreated at the next
    /// acquisition.
    fn present(&self, image: &SwapchainImage) -> Result<(), RhiError> {
        Err(RhiError::InvalidHandle(format!(
            "swapchain {:?} not found",
            image.swapchain
        )))
    }

    /// Returns true when API-level validation is active for this device (Vulkan validation
    /// layers, OpenGL debug output). See [`crate::api_validation_requested`].
    fn api_validation_enabled(&self) -> bool {
        false
    }

    /// Removes and returns the diagnostics the graphics API reported since the last call.
    fn take_diagnostics(&self) -> Vec<BackendDiagnostic> {
        Vec::new()
    }
}
