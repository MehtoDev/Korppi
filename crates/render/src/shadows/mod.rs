//! Real-time shadow mapping pipeline (Cascaded Shadow Maps, Texel Snapping, Normal Offset Bias, Octahedral Omni Shadows, and Unified Atlas).

/// Unified 2D shadow atlas allocator.
pub mod atlas;
/// Adaptive normal offset bias and depth biasing.
pub mod bias;
/// Directional Cascaded Shadow Maps (CSM) and texel snapping algorithms.
pub mod csm;
/// Octahedral projection mapping for omnidirectional point lights.
pub mod octahedral;
/// Poisson-disk sampling kernels for Percentage-Closer Filtering (PCF).
pub mod pcf;

pub use atlas::{
    AtlasSlot, CachedShadowAtlas, CachedTile, CasterType, FrameShadowPlan, ShadowAtlas,
    ShadowCaster, TileCopyCommand,
};
pub use bias::normal_offset_bias;
pub use csm::{compute_cascades, compute_cascades_into, Cascade, CascadeConfig};
pub use octahedral::{decode_octahedral, encode_octahedral};
pub use pcf::POISSON_DISK_16;
