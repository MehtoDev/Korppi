//! Unit tests for Cascaded Shadow Maps (CSM), Texel Snapping, Normal Offset Bias, Octahedral Mapping, and Shadow Atlas.

use korppi_math::{Mat4, Vec2, Vec3};
use korppi_render::shadows::{
    compute_cascades, decode_octahedral, encode_octahedral, normal_offset_bias, CascadeConfig,
    ShadowAtlas, POISSON_DISK_16,
};
use korppi_render::{PassType, RenderGraph, TextureAccess, TransientTextureDesc};
use korppi_rhi::{
    AccessFlags, ImageLayout, LoadOp, PipelineStageFlags, StoreOp, TextureFormat, TextureUsage,
};

#[test]
fn test_csm_split_depths_practical_split_scheme() {
    let config = CascadeConfig {
        cascade_count: 4,
        split_lambda: 0.85,
        shadow_map_resolution: 2048,
        near_plane: 0.5,
        far_plane: 100.0,
    };

    let splits = config.compute_split_depths();
    assert_eq!(splits.len(), 5); // [z0=near, z1, z2, z3, z4=far]
    assert_eq!(splits[0], 0.5);
    assert_eq!(splits[4], 100.0);

    // Splits must be strictly monotonic
    for i in 1..splits.len() {
        assert!(
            splits[i] > splits[i - 1],
            "Cascade split {} ({}) must be greater than split {} ({})",
            i,
            splits[i],
            i - 1,
            splits[i - 1]
        );
    }

    // Practical split blends log and uniform: split 1 should be near (e.g. 2-5m), not linear 25m
    assert!(
        splits[1] < 10.0,
        "Cascade 1 split should be near camera for high resolution"
    );
}

#[test]
fn test_csm_bounding_sphere_and_texel_snapping() {
    let config = CascadeConfig {
        cascade_count: 4,
        split_lambda: 0.85,
        shadow_map_resolution: 1024,
        near_plane: 0.5,
        far_plane: 100.0,
    };

    let cam_view = Mat4::look_at_rh(Vec3::new(0.0, 2.0, 5.0), Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
    let cam_proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.5, 100.0);
    let light_dir = Vec3::new(0.5, -1.0, 0.3).normalize();

    let cascades = compute_cascades(cam_view, cam_proj, light_dir, &config);
    assert_eq!(cascades.len(), 4);

    for (i, cascade) in cascades.iter().enumerate() {
        assert!(cascade.texel_size_world > 0.0);
        assert!(cascade.sphere_radius > 0.0);
        assert!(cascade.view_proj_matrix.is_finite());

        // Successive cascades must cover larger radii
        if i > 0 {
            assert!(
                cascade.sphere_radius >= cascades[i - 1].sphere_radius,
                "Outer cascade must have larger or equal radius"
            );
            assert!(
                cascade.texel_size_world >= cascades[i - 1].texel_size_world,
                "Outer cascade must have larger world texel size"
            );
        }
    }

    // Verify Texel Snapping:
    // Moving the camera by a fraction of a texel should result in a snapped light projection origin
    let texel_size = cascades[0].texel_size_world;
    let subtexel_shift = Vec3::new(texel_size * 0.25, 0.0, 0.0);
    let cam_view_shifted = Mat4::look_at_rh(
        Vec3::new(0.0, 2.0, 5.0) + subtexel_shift,
        Vec3::new(0.0, 0.0, 0.0) + subtexel_shift,
        Vec3::Y,
    );
    let cascades_shifted = compute_cascades(cam_view_shifted, cam_proj, light_dir, &config);

    // Light space projection center should be snapped to integer multiples of texel_size
    let snap_diff = (cascades[0].snapped_light_center.x / texel_size)
        .fract()
        .abs();
    assert!(
        snap_diff < 1e-3 || (snap_diff - 1.0).abs() < 1e-3,
        "Light center must snap to integer multiples of texel size, got frac {}",
        snap_diff
    );
    let snap_diff_shifted = (cascades_shifted[0].snapped_light_center.x / texel_size)
        .fract()
        .abs();
    assert!(
        snap_diff_shifted < 1e-3 || (snap_diff_shifted - 1.0).abs() < 1e-3,
        "Shifted light center must also snap to integer multiples of texel size"
    );
}

#[test]
fn test_normal_offset_bias_computation() {
    let normal = Vec3::Y;
    let texel_size = 0.05;
    let bias_scale = 1.0;

    // 1. Light perpendicular to surface (N . L = 1.0) -> offset is 0.0 (no acne, perfect contact)
    let light_dir_perp = Vec3::Y;
    let offset_perp = normal_offset_bias(normal, light_dir_perp, texel_size, bias_scale);
    assert!(
        offset_perp.length() < 1e-5,
        "Perpendicular light must have zero normal offset bias"
    );

    // 2. Light at grazing angle (45 degrees) -> offset expands along normal
    let light_dir_grazing = Vec3::new(1.0, 1.0, 0.0).normalize();
    let offset_grazing = normal_offset_bias(normal, light_dir_grazing, texel_size, bias_scale);
    assert!(
        offset_grazing.y > 0.0,
        "Grazing offset must be along positive normal"
    );
    assert_eq!(offset_grazing.x, 0.0);
    assert_eq!(offset_grazing.z, 0.0);

    // 3. Maximum offset at parallel light (90 degrees, N . L = 0.0)
    let light_dir_parallel = Vec3::X;
    let offset_parallel = normal_offset_bias(normal, light_dir_parallel, texel_size, bias_scale);
    let expected_max = texel_size * bias_scale;
    assert!(
        (offset_parallel.y - expected_max).abs() < 1e-5,
        "Parallel light should have maximum offset equal to texel_size * bias_scale"
    );
}

#[test]
fn test_octahedral_mapping_roundtrip() {
    let test_dirs = [
        Vec3::X,
        -Vec3::X,
        Vec3::Y,
        -Vec3::Y,
        Vec3::Z,
        -Vec3::Z,
        Vec3::new(1.0, 1.0, 1.0).normalize(),
        Vec3::new(-1.0, 2.0, -0.5).normalize(),
        Vec3::new(0.3, -0.8, 0.5).normalize(),
        Vec3::new(-0.7, -0.1, -0.7).normalize(),
    ];

    for dir in test_dirs {
        let uv = encode_octahedral(dir);
        assert!(uv.x >= 0.0 && uv.x <= 1.0, "UV.x out of bounds: {}", uv.x);
        assert!(uv.y >= 0.0 && uv.y <= 1.0, "UV.y out of bounds: {}", uv.y);

        let decoded = decode_octahedral(uv);
        assert!(
            (decoded.length() - 1.0).abs() < 1e-4,
            "Decoded vector must be normalized"
        );

        let diff = (decoded - dir).length();
        assert!(
            diff < 1e-3,
            "Octahedral roundtrip failed for dir {:?}, got {:?}, diff {}",
            dir,
            decoded,
            diff
        );
    }
}

#[test]
fn test_shadow_atlas_allocation_and_tile_remapping() {
    let mut atlas = ShadowAtlas::new(4096);

    // Allocate 4 CSM cascade slots (1024x1024 each)
    let csm0 = atlas.allocate(1024).expect("Cascade 0 allocation");
    let csm1 = atlas.allocate(1024).expect("Cascade 1 allocation");
    let csm2 = atlas.allocate(1024).expect("Cascade 2 allocation");
    let csm3 = atlas.allocate(1024).expect("Cascade 3 allocation");

    // Allocate 2 spotlight slots (512x512 each)
    let spot0 = atlas.allocate(512).expect("Spotlight 0 allocation");
    let spot1 = atlas.allocate(512).expect("Spotlight 1 allocation");

    // Allocate 2 octahedral point light slots (512x512 each)
    let omni0 = atlas.allocate(512).expect("Omni 0 allocation");
    let omni1 = atlas.allocate(512).expect("Omni 1 allocation");

    let all_slots = [&csm0, &csm1, &csm2, &csm3, &spot0, &spot1, &omni0, &omni1];

    // Verify all slots are within [0, 1] UV space and do not overlap
    for (i, s1) in all_slots.iter().enumerate() {
        assert!(s1.uv_min.x >= 0.0 && s1.uv_max.x <= 1.0);
        assert!(s1.uv_min.y >= 0.0 && s1.uv_max.y <= 1.0);
        assert!(s1.uv_min.x < s1.uv_max.x);
        assert!(s1.uv_min.y < s1.uv_max.y);

        for (j, s2) in all_slots.iter().enumerate() {
            if i != j {
                let overlap_x = s1.uv_min.x < s2.uv_max.x && s1.uv_max.x > s2.uv_min.x;
                let overlap_y = s1.uv_min.y < s2.uv_max.y && s1.uv_max.y > s2.uv_min.y;
                assert!(
                    !(overlap_x && overlap_y),
                    "Slot {} and Slot {} overlap in atlas UV space!",
                    i,
                    j
                );
            }
        }
    }

    // Verify UV remapping: local UV (0.5, 0.5) should map to center of slot
    let local_uv = Vec2::new(0.5, 0.5);
    let atlas_uv = csm0.remap_uv(local_uv);
    let expected = (csm0.uv_min + csm0.uv_max) * 0.5;
    assert!((atlas_uv - expected).length() < 1e-5);
}

#[test]
fn test_poisson_disk_samples() {
    assert_eq!(POISSON_DISK_16.len(), 16);
    for sample in POISSON_DISK_16 {
        assert!(
            sample.length() <= 1.0 + 1e-4,
            "Poisson sample must be inside unit disk"
        );
    }
}

#[test]
fn test_shadow_render_graph_integration() {
    let mut graph = RenderGraph::new();

    let shadow_atlas_desc = TransientTextureDesc::new(
        "ShadowAtlasDepth",
        4096,
        4096,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let shadow_handle = graph.create_transient_texture(shadow_atlas_desc);

    let output_color_desc = TransientTextureDesc::new(
        "OutputColor",
        1920,
        1080,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let output_handle = graph.create_transient_texture(output_color_desc);

    // Pass 1: Shadow Atlas Depth Pass
    graph
        .add_pass("ShadowAtlasPass", PassType::Graphics)
        .write_depth(
            shadow_handle,
            LoadOp::Clear,
            StoreOp::Store,
            0.0, // Reverse-Z clear
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 4096.0, 4096.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 4096, 4096);
        });

    // Pass 2: Forward Shading Pass (reads shadow atlas)
    graph
        .add_pass("ForwardShadingPass", PassType::Graphics)
        .read_texture(
            shadow_handle,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(
            output_handle,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.1, 0.1, 0.1, 1.0],
        )
        .mark_as_root()
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 1920, 1080);
        });

    let compiled = graph.compile().expect("Compile shadow graph failed");
    assert_eq!(compiled.passes().len(), 2);
    assert_eq!(compiled.passes()[0].name(), "ShadowAtlasPass");
    assert_eq!(compiled.passes()[1].name(), "ForwardShadingPass");

    // Pass 1 must transition shadow atlas to ShaderReadOnly and output to ColorAttachment
    let p1_barrier = compiled.passes()[1].pre_pass_barrier();
    assert_eq!(p1_barrier.image_barriers.len(), 2);
    let shadow_barrier = p1_barrier
        .image_barriers
        .iter()
        .find(|b| b.new_layout == ImageLayout::ShaderReadOnly)
        .expect("Must find barrier for shadow map in ShaderReadOnly layout");
    assert_eq!(shadow_barrier.new_layout, ImageLayout::ShaderReadOnly);
}

#[test]
fn test_g9_csm_sub_texel_camera_move_shifts_shadow_map_by_whole_texels() {
    use korppi_math::Vec4;
    use korppi_render::clustered::ReverseZ;

    let proj = ReverseZ::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
    let light = Vec3::new(-0.3, -1.0, -0.2);
    let config = CascadeConfig::default();
    let world_point = Vec4::new(1.0, 0.0, 1.0, 1.0);

    let texel_of = |eye: Vec3| -> (Vec2, f32, Vec3) {
        let view = Mat4::look_at_rh(eye, eye + Vec3::new(0.0, -0.5, -1.0), Vec3::Y);
        let c = compute_cascades(view, proj, light, &config)[0];
        let clip = c.view_proj_matrix * world_point;
        let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
        let res = config.shadow_map_resolution as f32;
        (
            (ndc * 0.5 + Vec2::splat(0.5)) * res,
            c.texel_size_world,
            c.snapped_light_center,
        )
    };

    let eye = Vec3::new(0.0, 5.0, 10.0);
    let (t0, texel, snapped) = texel_of(eye);
    assert!(
        snapped.x.abs() > 1e-4 || snapped.y.abs() > 1e-4,
        "G9: snapped_light_center must be evaluated in a fixed-origin light space, got {snapped:?}"
    );

    for step in [0.1f32, 0.3, 0.5, 0.7] {
        let (t1, _, _) = texel_of(eye + Vec3::new(step * texel, 0.0, 0.0));
        let d = t1 - t0;
        let err_x = (d.x - d.x.round()).abs();
        let err_y = (d.y - d.y.round()).abs();
        assert!(
            err_x < 1e-3 && err_y < 1e-3,
            "G9: camera moved {step} texel, fixed world point shifted by fractional texels ({:+.4}, {:+.4}) instead of integer texels",
            d.x,
            d.y
        );
    }
}
