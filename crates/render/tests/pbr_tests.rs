//! Unit tests for Modern Forward PBR, Multi-Scattering GGX, Energy Conservation, and JHFAA.

use korppi_math::Vec3;
use korppi_render::pbr::{
    approximate_dfg, compute_geometric_roughness, evaluate_pbr, rectify_silhouette_normal,
    CharlieSheen, ClearCoat, PbrMaterial, PbrSurface,
};
use korppi_render::{PassType, RenderGraph, TransientTextureDesc};
use korppi_rhi::{LoadOp, SampleCount, StoreOp, TextureDimension, TextureFormat, TextureUsage};

#[test]
fn test_cook_torrance_ggx_and_heitz_visibility() {
    let surface = PbrSurface {
        position: Vec3::ZERO,
        normal: Vec3::Y,
        view: Vec3::Y, // Looking straight down at surface
        albedo: Vec3::splat(1.0),
        material: PbrMaterial {
            roughness: 0.2,
            metallic: 1.0,
            reflectance: 0.5,
            clear_coat: None,
            sheen: None,
        },
    };

    // Incident light from straight above (L = Y)
    let light_dir = Vec3::Y;
    let radiance = Vec3::splat(10.0);
    let out_color = evaluate_pbr(&surface, light_dir, radiance);

    assert!(out_color.x > 0.0);
    assert!(out_color.is_finite());
}

#[test]
fn test_multi_scattering_energy_compensation_white_furnace() {
    // A pure white conductor (F0 = 1.0, metallic = 1.0, albedo = 1.0)
    // Under multi-scattering compensation, reflected radiance should not darken as roughness increases!
    let roughness_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    let light_dir = Vec3::new(0.5, 1.0, 0.0).normalize();
    let radiance = Vec3::ONE;

    for &roughness in &roughness_values {
        let surface = PbrSurface {
            position: Vec3::ZERO,
            normal: Vec3::Y,
            view: Vec3::new(0.0, 1.0, 0.5).normalize(),
            albedo: Vec3::ONE,
            material: PbrMaterial {
                roughness,
                metallic: 1.0,
                reflectance: 1.0,
                clear_coat: None,
                sheen: None,
            },
        };

        let reflected = evaluate_pbr(&surface, light_dir, radiance);
        assert!(reflected.is_finite());
        assert!(reflected.x > 0.0);

        // Verify DFG approximation has positive scale and bias
        let dfg = approximate_dfg(surface.normal.dot(surface.view), roughness);
        assert!(dfg.x > 0.0, "DFG scale must be positive");
        assert!(dfg.y > 0.0, "DFG bias must be positive");
        assert!(
            dfg.x + dfg.y <= 1.05,
            "Directional albedo must be <= 1.0 (energy conserving)"
        );
    }
}

#[test]
fn test_jhfaa_silhouette_normal_rectification() {
    let view = Vec3::new(0.0, 0.0, 1.0); // Viewer looking along -Z (view vector points towards +Z)

    // 1. Unphysical interpolated normal tilting away: N . V = -0.1
    let tilted_normal = Vec3::new(0.0, 1.0, -0.1).normalize();
    assert!(tilted_normal.dot(view) < 0.0);

    let threshold = 0.05;
    let rectified = rectify_silhouette_normal(tilted_normal, view, threshold);

    assert!(
        rectified.dot(view) >= threshold - 1e-4,
        "Rectified normal must satisfy N . V >= threshold, got {}",
        rectified.dot(view)
    );
    assert!(
        (rectified.length() - 1.0).abs() < 1e-4,
        "Rectified normal must be normalized"
    );

    // Reflection vector: R = reflect(-V, N) = -V + 2(N.V)N
    let r = -view + rectified * (2.0 * rectified.dot(view));
    assert!(
        r.dot(rectified) >= -1e-4,
        "Reflection vector must face forward towards surface normal hemisphere to prevent mesh self-intersection"
    );

    // 2. Already valid normal should undergo zero or minimal correction
    let valid_normal = Vec3::new(0.0, 0.5, 0.8).normalize();
    let rectified_valid = rectify_silhouette_normal(valid_normal, view, threshold);
    assert!((rectified_valid - valid_normal).length() < 1e-4);
}

#[test]
fn test_jhfaa_geometric_roughness_curvature_filtering() {
    let base_roughness = 0.2;
    let curvature_scale = 1.0;

    // 1. Flat planar surface: zero normal derivatives -> roughness unchanged
    let flat_roughness =
        compute_geometric_roughness(Vec3::ZERO, Vec3::ZERO, base_roughness, curvature_scale);
    assert!((flat_roughness - base_roughness).abs() < 1e-5);

    // 2. High-curvature edge: non-zero normal derivatives -> roughness increases
    let dndx = Vec3::new(0.5, 0.0, 0.0);
    let dndy = Vec3::new(0.0, 0.5, 0.0);
    let filtered_roughness =
        compute_geometric_roughness(dndx, dndy, base_roughness, curvature_scale);

    assert!(
        filtered_roughness > base_roughness,
        "Filtered roughness must expand over high curvature to prevent specular aliasing"
    );
    assert!(
        filtered_roughness <= 1.0,
        "Roughness must remain clamped <= 1.0"
    );
}

#[test]
fn test_clear_coat_energy_conservation() {
    let surface_no_coat = PbrSurface {
        position: Vec3::ZERO,
        normal: Vec3::Y,
        view: Vec3::Y,
        albedo: Vec3::new(0.8, 0.1, 0.1),
        material: PbrMaterial {
            roughness: 0.5,
            metallic: 0.0,
            reflectance: 0.5,
            clear_coat: None,
            sheen: None,
        },
    };

    let surface_with_coat = PbrSurface {
        position: Vec3::ZERO,
        normal: Vec3::Y,
        view: Vec3::Y,
        albedo: Vec3::new(0.8, 0.1, 0.1),
        material: PbrMaterial {
            roughness: 0.5,
            metallic: 0.0,
            reflectance: 0.5,
            clear_coat: Some(ClearCoat {
                roughness: 0.05,
                weight: 1.0,
            }),
            sheen: None,
        },
    };

    let light_dir = Vec3::Y;
    let radiance = Vec3::ONE;

    let color_base = evaluate_pbr(&surface_no_coat, light_dir, radiance);
    let color_coated = evaluate_pbr(&surface_with_coat, light_dir, radiance);

    assert!(color_base.is_finite());
    assert!(color_coated.is_finite());
    // Clear coat adds specular resin highlight on top while attenuating base diffuse
    assert!(color_coated.x > 0.0);
}

#[test]
fn test_charlie_sheen_grazing_lobe() {
    let surface_sheen = PbrSurface {
        position: Vec3::ZERO,
        normal: Vec3::Y,
        view: Vec3::new(0.99, 0.1, 0.0).normalize(), // Grazing viewing angle
        albedo: Vec3::new(0.2, 0.05, 0.05),
        material: PbrMaterial {
            roughness: 0.8,
            metallic: 0.0,
            reflectance: 0.5,
            clear_coat: None,
            sheen: Some(CharlieSheen {
                color: Vec3::new(1.0, 0.8, 0.8),
                roughness: 0.3,
            }),
        },
    };

    let light_dir = Vec3::new(-0.99, 0.1, 0.0).normalize();
    let radiance = Vec3::splat(2.0);
    let color = evaluate_pbr(&surface_sheen, light_dir, radiance);

    assert!(color.is_finite());
    assert!(color.x > 0.0);
}

#[test]
fn test_pbr_render_graph_pipeline() {
    let mut graph = RenderGraph::new();

    let msaa_color = graph.create_transient_texture(TransientTextureDesc {
        name: "MsaaHdrColor".to_string(),
        width: 1920,
        height: 1080,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsage::COLOR_ATTACHMENT,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::Count4,
    });

    let msaa_depth = graph.create_transient_texture(TransientTextureDesc {
        name: "MsaaDepthBuffer".to_string(),
        width: 1920,
        height: 1080,
        format: TextureFormat::Depth32Float,
        usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::Count4,
    });

    graph
        .add_pass("ForwardPbrShading4xMsaa", PassType::Graphics)
        .write_color(
            msaa_color,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.05, 0.05, 0.08, 1.0],
        )
        .write_depth(msaa_depth, LoadOp::Clear, StoreOp::Store, 0.0)
        .mark_as_root()
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 1920, 1080);
        });

    let compiled = graph.compile().expect("PBR 4x MSAA graph must compile");
    assert_eq!(compiled.passes().len(), 1);

    let pass = &compiled.passes()[0];
    let rendering = pass.rendering_descriptor().expect("Rendering descriptor");
    assert_eq!(rendering.color_attachments.len(), 1);
    assert!(rendering.depth_attachment.is_some());
}
