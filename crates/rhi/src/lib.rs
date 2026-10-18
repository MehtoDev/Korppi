//! Render Hardware Interface (RHI) abstraction crate for the game engine.
//!
//! Provides backend-agnostic hardware abstractions, generational resource handles,
//! dynamic rendering pass descriptors, pipeline state objects, synchronization primitives,
//! and command buffer recording interfaces.

/// Explicit resource binding: bind group layouts and bind groups.
pub mod binding;
/// Device capabilities: adapter information, limits, features and per-format support.
pub mod capabilities;
/// Command buffer and command recorder traits.
pub mod command;
/// Descriptors for buffers, textures, samplers, pipelines, and passes.
pub mod descriptor;
/// Core graphics device abstraction trait.
pub mod device;
/// API-level validation messages reported by backends.
pub mod diagnostics;
/// RHI error types.
pub mod error;
/// Generational resource handles.
pub mod handle;
/// Mutex locking that survives poisoning.
pub mod lock;
/// Host access to buffer memory: persistent mapping.
pub mod mapping;
/// Reuse of boxed per-frame objects.
pub mod pool;
/// GPU queries (timestamps) and indirect command arguments.
pub mod query;
/// Generational slot maps for resource registries.
pub mod slot_map;
/// SPIR-V module framing checks shared by the backends.
pub mod spirv;
/// Queue submissions and frames in flight.
pub mod submission;
/// Presentation to OS windows: swapchain descriptors and window targets.
pub mod swapchain;
/// Pipeline stages, access flags, memory barriers, and synchronization.
pub mod sync;
/// Recorded copies, blits and mip generation.
pub mod transfer;
/// Hardware formats, bitflags, and alignment constants.
pub mod types;
/// Debug-build validation of command recording.
pub mod validation;

// Re-export primary types and traits at crate root
pub use binding::{
    check_pipeline_layouts, validate_bind_group, validate_bind_group_resources,
    BindGroupDescriptor, BindGroupEntry, BindGroupInfo, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutInfo, BindingLookup, BindingResource, BindingType,
    BoundResource, DynamicBinding, ShaderStages, StorageTextureAccess, TextureBinding,
    MAX_BINDINGS_PER_GROUP, MAX_BIND_GROUPS, MAX_DYNAMIC_OFFSETS_PER_GROUP,
};
pub use capabilities::{
    AdapterInfo, DeviceFeatures, DeviceLimits, DeviceType, FormatCapabilities, GraphicsApi,
};
pub use command::{CommandBuffer, CommandBufferRecorder};
pub use descriptor::{
    BlendComponent, BlendFactor, BlendOp, BlendState, BufferDescriptor, BufferTextureCopy,
    ColorTargetState, ComputePipelineDescriptor, DepthAttachment, DepthStencilState,
    GraphicsPipelineDescriptor, MultisampleState, PushConstantRange, RasterizerState,
    RenderingAttachment, RenderingDescriptor, SamplerDescriptor, ShaderDescriptor, ShaderSource,
    ShaderStage, StencilFaceState, StencilOperation, TextureDescriptor, TextureViewDescriptor,
    VertexAttribute, VertexBinding, VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
pub use descriptor::{ColorAttachments, MAX_COLOR_ATTACHMENTS};
pub use device::GraphicsDevice;
pub use diagnostics::{
    api_validation_requested, BackendDiagnostic, DiagnosticLog, DiagnosticSeverity,
    VALIDATION_ENV_VAR,
};
pub use error::RhiError;
pub use handle::{
    BindGroupHandle, BindGroupLayoutHandle, BufferHandle, DeviceId, PipelineHandle, QuerySetHandle,
    SamplerHandle, ShaderHandle, SwapchainHandle, TextureHandle, TextureViewHandle,
};
pub use korppi_core::{smallvec, SmallVec};
pub use lock::lock;
pub use mapping::{HostAccessGuard, HostAccessLock, MappedBuffer};
pub use pool::BoxPool;
pub use query::{
    DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs, QuerySetDescriptor, QueryType,
};
pub use slot_map::{SlotHandle, SlotKey, SlotMap};
pub use spirv::{
    spirv_native_words, spirv_words_from_bytes, validate_spirv_structure, SPIRV_MAGIC,
};
pub use submission::{FrameIndex, FrameTracker, SubmissionId, FRAMES_IN_FLIGHT};
pub use swapchain::{
    DisplayTarget, PresentMode, PresentTarget, SwapchainDescriptor, SwapchainImage, SwapchainInfo,
};
pub use sync::{
    AccessFlags, BufferMemoryBarrier, ImageLayout, ImageMemoryBarrier, MemoryBarrier,
    PipelineBarrier, PipelineStageFlags, TextureSubresourceRange,
};
pub use transfer::{TextureBlit, TextureClear, TextureCopy};
pub use types::{
    align_to_ubo_offset, BorderColor, BufferUsage, ColorWriteMask, CompareFunction, CullMode,
    FilterMode, FrontFace, IndexType, LoadOp, MemoryUsage, PolygonMode, PrimitiveTopology,
    SampleCount, SamplerAddressMode, StoreOp, TextureAspect, TextureDimension, TextureFormat,
    TextureUsage, MAX_PUSH_CONSTANTS_SIZE, STANDARD_SSBO_ALIGNMENT, STANDARD_UBO_ALIGNMENT,
};
pub use validation::{
    validate_recorder, BufferSummary, PipelineKind, PipelineSignature, ResourceOracle, TextureInfo,
    ValidatingRecorder,
};

/// Type alias for 4-component float RGBA clear color.
pub type ClearColor = [f32; 4];
