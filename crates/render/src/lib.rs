//! # Engine Render
//!
//! Modern Clustered Forward rendering pipeline and Render Graph pass scheduler.

/// Implements [`korppi_core::Pod`] for GPU-facing structs, whose bytes are uploaded as they are.
///
/// Each type must be `repr(C)` with only 4-byte scalar fields (`f32`, `u32` or arrays of them),
/// and is listed with the sum of its field sizes: the compile-time check that the type is
/// exactly that size rules out padding, so every byte is initialized and any bit pattern is a
/// valid value.
macro_rules! gpu_pod {
    ($($ty:ty = $field_bytes:expr),* $(,)?) => {
        korppi_core::impl_pod!($($ty = $field_bytes),*);
    };
}

/// Clustered forward lighting, froxel culling, and Reverse-Z depth prepass.
pub mod clustered;
/// 3D physics and world debug drawing utilities.
pub mod debug3d;
/// Error types for render pipeline and graph execution.
pub mod error;
/// Frame graph / render graph DAG scheduler, transient caching, and execution.
pub mod graph;
/// Physically Based Rendering (PBR), Microfacet Models, and Specular Anti-Aliasing.
pub mod pbr;
/// 3D Clustered Forward Rendering Pipeline on the GPU.
pub mod pipeline;
/// Post-processing, HDR pipeline, tonemapping, progressive bloom, and SSAO.
pub mod postprocess;
/// 2D rendering pipeline: batched quads, texture atlasing, 9-patch scaling, and canvas modes.
pub mod render2d;
/// Real-time shadow mapping (CSM, texel snapping, normal offset bias, octahedral atlas).
pub mod shadows;

pub use debug3d::{DebugDraw3d, LineVertex3d};
pub use error::RenderError;
pub use graph::{
    BufferAccess, ColorAttachmentDesc, CompiledGraphOutput, CompiledRenderGraph,
    DepthAttachmentDesc, PassBuilder, PassContext, PassType, RenderGraph, RenderGraphBufferHandle,
    RenderGraphTextureHandle, TextureAccess, TransientBufferDesc, TransientResourcePool,
    TransientTextureDesc,
};
pub use pipeline::{
    Camera3d, ForwardMaterial, ForwardPipeline, ForwardPipelineConfig, ForwardScene, GpuMesh,
    MeshData, MeshDraw, Vertex3d, WorldDirectionalLight, WorldPointLight, WorldSpotLight,
};
