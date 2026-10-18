//! Physically Based Rendering (PBR), Microfacet Models, and Specular Anti-Aliasing.

pub mod brdf;
pub mod dfg;
pub mod jhfaa;
pub mod material;

pub use brdf::{
    compute_energy_compensation, d_charlie, d_ggx, evaluate_pbr, fresnel_schlick, v_neubelt,
    v_smith_heitz, MIN_PERCEPTUAL_ROUGHNESS,
};
pub use dfg::{approximate_dfg, dfg_lut_packed, integrate_dfg, DFG_LUT_DIM};
pub use jhfaa::{
    compute_effective_roughness_toksvig, compute_geometric_roughness, compute_toksvig_roughness,
    compute_toksvig_variance, rectify_silhouette_normal,
};
pub use material::{CharlieSheen, ClearCoat, PbrMaterial, PbrSurface};
