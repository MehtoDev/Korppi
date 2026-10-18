//! Unit tests for HDR Post-Processing, Tonemapping, Bloom Pyramid, and SSAO.

use korppi_math::Vec3;
use korppi_render::postprocess::{
    aces_fitted, agx_tonemap, apply_exposure, bloom_soft_knee, compute_bilateral_depth_weight,
    compute_karis_average_weight, evaluate_tonemapping, filmic_hable, generate_ssao_kernel,
    reinhard_luminance, BloomConfig, SsaoConfig, TonemapConfig, TonemapOperator,
};
use korppi_render::{PassType, RenderGraph, TextureAccess, TransientTextureDesc};
use korppi_rhi::{
    AccessFlags, ImageLayout, LoadOp, PipelineStageFlags, SampleCount, StoreOp, TextureDimension,
    TextureFormat, TextureUsage,
};

#[test]
fn test_reinhard_luminance_tonemapping() {
    let dark = Vec3::new(0.05, 0.05, 0.05);
    let mapped_dark = reinhard_luminance(dark, 10.0);
    // Dark values remain nearly linear: x / (1 + x) ~ x
    assert!((mapped_dark.x - 0.05).abs() < 0.01);

    let bright = Vec3::new(6.0, 3.0, 1.0);
    let mapped_bright = reinhard_luminance(bright, 10.0);
    // Bright values are compressed into [0.0, 1.0]
    assert!((0.0..=1.0).contains(&mapped_bright.x));
    assert!((0.0..=1.0).contains(&mapped_bright.y));
    assert!((0.0..=1.0).contains(&mapped_bright.z));
    assert!(mapped_bright.x > mapped_bright.y);
    assert!(mapped_bright.y > mapped_bright.z);
}

#[test]
fn test_aces_fitted_s_curve() {
    let values = [0.0, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0];
    let mut prev_mapped = -1.0;

    for &val in &values {
        let mapped = aces_fitted(Vec3::splat(val)).x;
        assert!((0.0..=1.0).contains(&mapped));
        assert!(
            mapped >= prev_mapped,
            "ACES tone mapper must be monotonically increasing"
        );
        prev_mapped = mapped;
    }

    // Zero maps to zero
    assert_eq!(aces_fitted(Vec3::ZERO), Vec3::ZERO);
    // Extreme highlight compresses smoothly near 1.0
    let extreme = aces_fitted(Vec3::splat(1000.0));
    assert!(extreme.x > 0.95 && extreme.x <= 1.0);
}

#[test]
fn test_agx_and_filmic_hable_tonemapping() {
    let hdr_color = Vec3::new(4.0, 2.0, 0.5);

    let agx = agx_tonemap(hdr_color);
    assert!((0.0..=1.0).contains(&agx.x));
    assert!((0.0..=1.0).contains(&agx.y));
    assert!((0.0..=1.0).contains(&agx.z));
    assert!(agx.is_finite());

    let filmic = filmic_hable(hdr_color);
    assert!((0.0..=1.0).contains(&filmic.x));
    assert!((0.0..=1.0).contains(&filmic.y));
    assert!((0.0..=1.0).contains(&filmic.z));
    assert!(filmic.is_finite());
}

#[test]
fn test_tonemapping_config_and_evaluation() {
    let config = TonemapConfig {
        operator: TonemapOperator::AgX,
        exposure: 1.5,
        gamma: 2.2,
        white_point: 10.0,
    };
    let color = Vec3::new(2.0, 1.0, 0.5);
    let exposed = apply_exposure(color, config.exposure);
    assert!((exposed.x - 3.0).abs() < 1e-4);

    let ldr = evaluate_tonemapping(color, &config);
    assert!((0.0..=1.0).contains(&ldr.x));
    assert!((0.0..=1.0).contains(&ldr.y));
    assert!((0.0..=1.0).contains(&ldr.z));
    assert!(ldr.is_finite());

    let bloom = BloomConfig::default();
    assert_eq!(bloom.mip_levels, 5);

    let ssao = SsaoConfig::default();
    assert_eq!(ssao.kernel_size, 32);
}

#[test]
fn test_bloom_soft_knee_thresholding() {
    let threshold = 1.0;
    let soft_knee = 0.5;

    // Sub-threshold values are smoothly extinguished
    let dark = Vec3::new(0.2, 0.2, 0.2);
    let bloom_dark = bloom_soft_knee(dark, threshold, soft_knee);
    assert_eq!(bloom_dark, Vec3::ZERO);

    // In the knee zone [threshold - knee, threshold + knee]
    let mid = Vec3::new(1.0, 1.0, 1.0);
    let bloom_mid = bloom_soft_knee(mid, threshold, soft_knee);
    assert!(bloom_mid.x > 0.0 && bloom_mid.x < 1.0);

    // Bright highlights pass through with subtractive knee
    let bright = Vec3::new(10.0, 10.0, 10.0);
    let bloom_bright = bloom_soft_knee(bright, threshold, soft_knee);
    assert!(bloom_bright.x > 8.0);
}

#[test]
fn test_karis_average_firefly_suppression() {
    let normal_pixel = Vec3::new(1.0, 1.0, 1.0);
    let firefly_pixel = Vec3::new(1000.0, 1000.0, 1000.0);

    let w_normal = compute_karis_average_weight(normal_pixel);
    let w_firefly = compute_karis_average_weight(firefly_pixel);

    assert!(w_normal > 0.0);
    assert!(w_firefly > 0.0);
    // Firefly weight must be drastically lower than normal pixel weight
    assert!(w_firefly < w_normal * 0.01);
}

#[test]
fn test_ssao_hemisphere_kernel_generation() {
    let kernel_size = 32;
    let kernel = generate_ssao_kernel(kernel_size);

    assert_eq!(kernel.len(), kernel_size);
    for sample in &kernel {
        assert!(
            sample.length() <= 1.0001,
            "Sample must be inside unit sphere"
        );
        assert!(sample.z >= 0.0, "Sample must lie in upper hemisphere (+Z)");
    }

    // Verify samples are non-trivially distributed (not all stacked at origin or surface)
    let avg_length: f32 = kernel.iter().map(|s| s.length()).sum::<f32>() / (kernel_size as f32);
    assert!(avg_length > 0.1 && avg_length < 0.9);
}

#[test]
fn test_ssao_bilateral_depth_blur_weight() {
    let depth_sigma = 0.05;

    // Same depth -> maximum bilateral weight (1.0)
    let weight_same = compute_bilateral_depth_weight(1.0, 1.0, depth_sigma);
    assert!((weight_same - 1.0).abs() < 1e-5);

    // Minor depth difference within same planar surface
    let weight_close = compute_bilateral_depth_weight(1.0, 1.01, depth_sigma);
    assert!(weight_close > 0.7);

    // Large depth jump (silhouette edge / background)
    let weight_edge = compute_bilateral_depth_weight(1.0, 1.5, depth_sigma);
    assert!(
        weight_edge < 0.01,
        "Bilateral weight must reject cross-edge depth bleed"
    );
}

#[test]
fn test_hdr_postprocess_render_graph_pipeline() {
    let mut graph = RenderGraph::new();

    // 1. HDR Color Buffer (16-bit float)
    let hdr_color = graph.create_transient_texture(TransientTextureDesc {
        name: "HdrSceneColor".to_string(),
        width: 1920,
        height: 1080,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::Count1,
    });

    // 2. Bloom Downsample Target (Quarter res)
    let bloom_down = graph.create_transient_texture(TransientTextureDesc {
        name: "BloomDownsample0".to_string(),
        width: 960,
        height: 540,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::Count1,
    });

    // 3. Final LDR Target (8-bit sRGB swapchain)
    let ldr_target = graph.create_transient_texture(TransientTextureDesc {
        name: "LdrSwapchainTarget".to_string(),
        width: 1920,
        height: 1080,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::Count1,
    });

    // Forward Scene Pass
    graph
        .add_pass("HdrScenePass", PassType::Graphics)
        .write_color(
            hdr_color,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.1, 0.2, 0.5, 1.0],
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 1920, 1080);
        });

    // Bloom Downsample Pass
    graph
        .add_pass("BloomDownsamplePass", PassType::Graphics)
        .read_texture(
            hdr_color,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(
            bloom_down,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.0, 0.0, 0.0, 1.0],
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 960.0, 540.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 960, 540);
        });

    // Post-Process Composite & Tonemapping Pass
    graph
        .add_pass("PostProcessTonemapPass", PassType::Graphics)
        .read_texture(
            hdr_color,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .read_texture(
            bloom_down,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(
            ldr_target,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.0, 0.0, 0.0, 1.0],
        )
        .mark_as_root()
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 1920, 1080);
        });

    let compiled = graph.compile().expect("Post-processing graph must compile");
    assert_eq!(compiled.passes().len(), 3);
}

#[test]
fn test_g13_agx_matches_reference_curve_without_double_gamma() {
    fn agx_reference(x: f32) -> f32 {
        let v = x.max(1e-10);
        let min_ev = -12.473_93f32;
        let max_ev = 4.026_069f32;
        let t = ((v.log2() - min_ev) / (max_ev - min_ev)).clamp(0.0, 1.0);
        let t2 = t * t;
        let t4 = t2 * t2;
        15.5 * t4 * t2 - 40.14 * t4 * t + 31.96 * t4 - 6.868 * t2 * t + 0.4298 * t2 + 0.1191 * t
            - 0.00232
    }

    let config = TonemapConfig {
        operator: TonemapOperator::AgX,
        exposure: 1.0,
        gamma: 2.2,
        white_point: 10.0,
    };

    for x in [0.01f32, 0.05, 0.18, 0.5, 1.0, 4.0, 16.0] {
        let k = evaluate_tonemapping(Vec3::splat(x), &config).x;
        let expected = agx_reference(x);
        assert!(
            (k - expected).abs() < 0.002,
            "G13: AgX at scene grey {x} evaluated to {k:.4}, expected {expected:.4}"
        );
    }
}

#[test]
fn test_rnd15_jimenez_13tap_downsample_weights() {
    use korppi_render::postprocess::bloom::DOWNSAMPLE_13TAP_WEIGHTS;

    // 4 inner box samples = 0.125 each
    assert_eq!(&DOWNSAMPLE_13TAP_WEIGHTS[0..4], &[0.125; 4]);
    // 4 outer corner samples = 0.03125 each
    assert_eq!(&DOWNSAMPLE_13TAP_WEIGHTS[4..8], &[0.03125; 4]);
    // 4 cross edge samples = 0.0625 each
    assert_eq!(&DOWNSAMPLE_13TAP_WEIGHTS[8..12], &[0.0625; 4]);
    // 1 center sample = 0.125
    assert_eq!(DOWNSAMPLE_13TAP_WEIGHTS[12], 0.125);

    let sum: f32 = DOWNSAMPLE_13TAP_WEIGHTS.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "13-tap Jimenez weights must sum to 1.0, got {sum}"
    );
}
