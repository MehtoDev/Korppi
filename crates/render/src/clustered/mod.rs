//! Clustered forward lighting, froxel culling, and Reverse-Z depth prepass pipeline.

/// CPU/GPU light culler allocating light indices into froxel records.
pub mod culler;
/// Decals and OBB intersection tests.
pub mod decal;
/// Declarative depth prepass helper for Render Graph.
pub mod depth_prepass;
/// GPU compute shader light culler pipeline and structures.
pub mod gpu_culler;
/// Clustered frustum spatial partitioning and view-space froxel AABB construction.
pub mod grid;
/// Point, spot, and directional light models and view-space intersection algorithms.
pub mod light;
/// Reflection probes and bounding volume intersection tests.
pub mod probe;
/// Reverse-Z depth buffer mathematics and conventions.
pub mod reverse_z;

pub use culler::{ClusterRecord, ClusteredLightCuller, CullingOutput};
pub use decal::{test_decal_vs_aabb, Decal};
pub use depth_prepass::{add_depth_prepass, DepthPrepassConfig};
pub use gpu_culler::{
    CullingParamsGpu, FroxelAabbGpu, GpuCullingBuffers, GpuLightCuller, PointLightGpu,
    SpotLightGpu, CLUSTERED_LIGHT_CULL_GLSL, CLUSTERED_LIGHT_CULL_SPIRV,
};
pub use grid::{ClusterGridConfig, FroxelAabb};
pub use light::{
    test_cone_vs_sphere, test_sphere_vs_aabb, DirectionalLight, PointLight, SpotLight,
};
pub use probe::{test_probe_vs_aabb, ProbeShape, ReflectionProbe};
pub use reverse_z::ReverseZ;
