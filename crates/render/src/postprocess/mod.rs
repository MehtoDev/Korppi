//! Post-processing, HDR pipeline, tonemapping, progressive bloom, and SSAO.

pub mod bloom;
pub mod pipeline;
pub mod ssao;
pub mod tonemap;

pub use bloom::{
    bloom_soft_knee, compute_karis_average_weight, BloomConfig, DOWNSAMPLE_13TAP_WEIGHTS,
    UPSAMPLE_9TAP_WEIGHTS,
};
pub use pipeline::PostProcessPipeline;
pub use ssao::{
    compute_bilateral_depth_weight, generate_ssao_kernel, generate_ssao_noise_table, SsaoConfig,
};
pub use tonemap::{
    aces_fitted, agx_tonemap, apply_exposure, evaluate_tonemapping, filmic_hable,
    reinhard_luminance, TonemapConfig, TonemapOperator,
};
