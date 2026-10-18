//! 3D Clustered Forward Rendering Pipeline (`FIX-6.4`, `RND-01`).

/// GPU-backed forward renderer orchestrating depth prepass, shadow atlas, compute light culling,
/// SSAO, MSAA forward PBR shading, progressive bloom, and tonemapping.
pub mod forward;
/// Scene, mesh, material, camera, and light descriptors.
pub mod scene;

pub use forward::{ForwardPipeline, ForwardPipelineConfig};
pub use scene::{
    Camera3d, ForwardMaterial, ForwardScene, GpuMesh, MeshData, MeshDraw, Vertex3d,
    WorldDirectionalLight, WorldPointLight, WorldSpotLight,
};
