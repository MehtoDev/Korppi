/// Physical GPU memory aliasing and interval bin-packing allocator.
pub mod aliasing;
/// Graph compiler executing topological sort, dead-pass culling, and barrier synthesis.
pub mod compiler;
/// Handle types for virtual graph textures and buffers.
pub mod handle;
/// Pass descriptions, contexts, and callback builders.
pub mod pass;
/// Primary RenderGraph builder and compiled graph executor.
pub mod render_graph;
/// Virtual resource definitions, access flags, and transient resource pool.
pub mod resource;

pub use aliasing::{
    compute_transient_texture_byte_size, MemoryAliasingReport, PhysicalResourceAllocation,
    TransientMemoryHeapAllocator,
};
pub use compiler::CompiledGraphOutput;
pub use handle::{RenderGraphBufferHandle, RenderGraphTextureHandle};
pub use pass::{
    ColorAttachmentDesc, CompiledPass, DepthAttachmentDesc, PassBuilder, PassCallback, PassContext,
    PassType,
};
pub use render_graph::{CompiledRenderGraph, RenderGraph};
pub use resource::{
    BufferAccess, TextureAccess, TransientBufferDesc, TransientResourcePool, TransientTextureDesc,
};
