//! Unit tests for Clustered Light Culling, Froxel AABBs, Wronski Cone Intersection, and Reverse-Z Depth.

use korppi_math::{Mat4, Vec2, Vec3};
use korppi_render::clustered::{
    test_cone_vs_sphere, test_sphere_vs_aabb, ClusterGridConfig, ClusteredLightCuller, PointLight,
    ReverseZ, SpotLight,
};
use korppi_render::{PassType, RenderGraph, TransientTextureDesc};
use korppi_rhi::{CompareFunction, LoadOp, StoreOp, TextureFormat, TextureUsage};

#[test]
fn test_cluster_grid_dimensions_and_depth_slicing() {
    // 1920x1080 with 64x64 pixel tiles and 24 depth slices from 0.5m to 100.0m
    let config = ClusterGridConfig::new(1920, 1080, 64, 24, 0.5, 100.0);

    assert_eq!(config.dim_x(), 30);
    assert_eq!(config.dim_y(), 17);
    assert_eq!(config.dim_z(), 24);
    assert_eq!(config.total_clusters(), 30 * 17 * 24);

    // Near plane depth slice should be 0
    assert_eq!(config.compute_slice_z(0.5), 0);
    assert_eq!(config.compute_slice_z(0.6), 0);

    // Far plane depth slice should be 23 (max slice = dim_z - 1)
    assert_eq!(config.compute_slice_z(99.9), 23);
    assert_eq!(config.compute_slice_z(100.0), 23);

    // Clamped outside bounds
    assert_eq!(config.compute_slice_z(0.1), 0);
    assert_eq!(config.compute_slice_z(200.0), 23);

    // Exponential progression: slice indices must be strictly monotonic non-decreasing with depth
    let mut prev_slice = 0;
    for step in 1..100 {
        let depth = 0.5 + (step as f32) * 1.0;
        let slice = config.compute_slice_z(depth);
        assert!(slice >= prev_slice, "Slice index must be monotonic");
        prev_slice = slice;
    }

    // Cluster linear index calculation
    let idx = config.compute_cluster_index(Vec2::new(10.0, 10.0), 0.5);
    assert_eq!(idx, 0); // (0, 0, 0) -> 0

    let idx_far = config.compute_cluster_index(Vec2::new(1919.0, 1079.0), 100.0);
    assert_eq!(
        idx_far,
        config.dim_x() - 1
            + (config.dim_y() - 1) * config.dim_x()
            + 23 * (config.dim_x() * config.dim_y())
    );
}

#[test]
fn test_froxel_view_space_aabb_generation() {
    let config = ClusterGridConfig::new(640, 480, 64, 8, 1.0, 50.0);
    let aspect = 640.0 / 480.0;
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 1.0, 50.0);
    let inv_proj = proj.inverse();

    let froxels = config.build_froxel_aabbs(inv_proj);
    assert_eq!(froxels.len(), config.total_clusters() as usize);

    for froxel in &froxels {
        // Froxel AABB must have positive volume
        assert!(froxel.aabb_min.x < froxel.aabb_max.x);
        assert!(froxel.aabb_min.y < froxel.aabb_max.y);
        assert!(froxel.aabb_min.z < froxel.aabb_max.z);

        // Circumscribed sphere must cover the AABB
        assert!(froxel.sphere_radius > 0.0);
        let center = (froxel.aabb_min + froxel.aabb_max) * 0.5;
        assert!((froxel.sphere_center - center).length() < 1e-4);

        // Distance from center to any corner must equal sphere_radius
        let half_extents = (froxel.aabb_max - froxel.aabb_min) * 0.5;
        assert!((half_extents.length() - froxel.sphere_radius).abs() < 1e-4);
    }
}

#[test]
fn test_point_light_sphere_vs_aabb_culling() {
    let aabb_min = Vec3::new(-1.0, -1.0, -5.0);
    let aabb_max = Vec3::new(1.0, 1.0, -3.0);

    // 1. Point light strictly inside AABB
    assert!(test_sphere_vs_aabb(
        Vec3::new(0.0, 0.0, -4.0),
        0.5,
        aabb_min,
        aabb_max
    ));

    // 2. Point light outside AABB, but sphere intersects edge
    assert!(test_sphere_vs_aabb(
        Vec3::new(1.5, 0.0, -4.0),
        0.6,
        aabb_min,
        aabb_max
    ));

    // 3. Point light outside AABB touching corner
    assert!(test_sphere_vs_aabb(
        Vec3::new(2.0, 2.0, -2.0),
        2.0,
        aabb_min,
        aabb_max
    ));

    // 4. Point light completely outside AABB
    assert!(!test_sphere_vs_aabb(
        Vec3::new(5.0, 0.0, -4.0),
        1.0,
        aabb_min,
        aabb_max
    ));

    // 5. Point light behind AABB along Z
    assert!(!test_sphere_vs_aabb(
        Vec3::new(0.0, 0.0, -10.0),
        2.0,
        aabb_min,
        aabb_max
    ));
}

#[test]
fn test_spot_light_wronski_cone_vs_sphere_culling() {
    let sphere_center = Vec3::new(0.0, 0.0, -10.0);
    let sphere_radius = 2.0;

    let cone_origin = Vec3::new(0.0, 0.0, -2.0);
    let cone_forward = Vec3::new(0.0, 0.0, -1.0); // Facing towards sphere along -Z
    let cone_range = 15.0;
    let cone_half_angle = std::f32::consts::FRAC_PI_6; // 30 degrees

    // 1. Sphere directly in front of spotlight within range -> Intersects
    assert!(test_cone_vs_sphere(
        cone_origin,
        cone_forward,
        cone_range,
        cone_half_angle,
        sphere_center,
        sphere_radius,
    ));

    // 2. Spotlight pointing in opposite direction -> Back apex culled
    let cone_opposite = Vec3::new(0.0, 0.0, 1.0);
    assert!(!test_cone_vs_sphere(
        cone_origin,
        cone_opposite,
        cone_range,
        cone_half_angle,
        sphere_center,
        sphere_radius,
    ));

    // 3. Spotlight pointing perpendicular (away from sphere) -> Angular culled
    let cone_perp = Vec3::new(1.0, 0.0, 0.0);
    assert!(!test_cone_vs_sphere(
        cone_origin,
        cone_perp,
        cone_range,
        cone_half_angle,
        sphere_center,
        sphere_radius,
    ));

    // 4. Spotlight range too short to reach sphere -> Range culled
    let short_range = 5.0; // Distance to center is 8.0, sphere boundary is at 6.0
    assert!(!test_cone_vs_sphere(
        cone_origin,
        cone_forward,
        short_range,
        cone_half_angle,
        sphere_center,
        sphere_radius,
    ));
}

#[test]
fn test_clustered_light_culling_execution() {
    let config = ClusterGridConfig::new(320, 240, 64, 8, 1.0, 40.0);
    let aspect = 320.0 / 240.0;
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 1.0, 40.0);
    let inv_proj = proj.inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);

    let point_lights = vec![
        PointLight {
            position_vs: Vec3::new(0.0, 0.0, -5.0),
            radius: 3.0,
            color: Vec3::new(1.0, 1.0, 1.0),
            intensity: 100.0,
        },
        PointLight {
            position_vs: Vec3::new(100.0, 100.0, -5.0), // Offscreen light
            radius: 2.0,
            color: Vec3::new(1.0, 0.0, 0.0),
            intensity: 50.0,
        },
    ];

    let spot_lights = vec![SpotLight {
        position_vs: Vec3::new(0.0, 0.0, -2.0),
        direction_vs: Vec3::new(0.0, 0.0, -1.0),
        range: 10.0,
        half_angle: std::f32::consts::FRAC_PI_6,
        color: Vec3::new(0.0, 1.0, 0.0),
        intensity: 200.0,
    }];

    let mut culler = ClusteredLightCuller::new(&config);
    let result = culler.cull_lights(&froxels, &point_lights, &spot_lights);

    assert_eq!(
        result.cluster_records.len(),
        config.total_clusters() as usize
    );

    // Center froxel near z = -5.0 must contain point light 0 and spot light 0
    let center_idx = config.compute_cluster_index(Vec2::new(160.0, 120.0), 5.0) as usize;
    let center_record = &result.cluster_records[center_idx];

    assert!(center_record.point_count > 0);
    assert!(center_record.spot_count > 0);

    // Verify point light 0 was culled into center froxel
    let point_slice = &result.global_light_indices[center_record.point_offset as usize
        ..(center_record.point_offset + center_record.point_count) as usize];
    assert!(point_slice.contains(&0));
    assert!(!point_slice.contains(&1)); // Offscreen light 1 must not be in center froxel!

    // Verify spot light 0 was culled into center froxel
    let spot_slice = &result.global_light_indices[center_record.spot_offset as usize
        ..(center_record.spot_offset + center_record.spot_count) as usize];
    assert!(spot_slice.contains(&0));
}

#[test]
fn test_reverse_z_perspective_and_depth_invariants() {
    let fov = std::f32::consts::FRAC_PI_4;
    let aspect = 16.0 / 9.0;
    let z_near = 0.5;
    let z_far = 100.0;

    let proj = ReverseZ::perspective_rh(fov, aspect, z_near, z_far);

    // In a right-handed camera, viewing direction is -Z.
    // Projected NDC point: p_clip = proj * p_view, p_ndc = p_clip.xyz / p_clip.w
    // Near plane point: (0, 0, -z_near, 1) -> NDC Z should be 1.0 (Reverse-Z near)
    let p_near = proj.project_point3(Vec3::new(0.0, 0.0, -z_near));
    assert!(
        (p_near.z - 1.0).abs() < 1e-4,
        "Reverse-Z near plane must map to 1.0, got {}",
        p_near.z
    );

    // Far plane point: (0, 0, -z_far, 1) -> NDC Z should be 0.0 (Reverse-Z far)
    let p_far = proj.project_point3(Vec3::new(0.0, 0.0, -z_far));
    assert!(
        (p_far.z - 0.0).abs() < 1e-4,
        "Reverse-Z far plane must map to 0.0, got {}",
        p_far.z
    );

    // Intermediate point: (0, 0, -5.0, 1) -> NDC Z must be strictly in (0.0, 1.0)
    let p_mid = proj.project_point3(Vec3::new(0.0, 0.0, -5.0));
    assert!(p_mid.z > 0.0 && p_mid.z < 1.0);
    assert!(
        p_mid.z < p_near.z,
        "NDC depth must decrease as distance increases"
    );
    assert!(p_mid.z > p_far.z);

    // Reverse-Z compare function is GreaterEqual, clear value is 0.0
    assert_eq!(ReverseZ::DEPTH_COMPARE, CompareFunction::GreaterEqual);
    assert_eq!(ReverseZ::CLEAR_DEPTH, 0.0);
}

#[test]
fn test_depth_prepass_render_graph_integration() {
    let mut graph = RenderGraph::new();

    let depth_desc = TransientTextureDesc::new(
        "DepthPrepassBuffer",
        1920,
        1080,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let depth_handle = graph.create_transient_texture(depth_desc);

    let color_desc = TransientTextureDesc::new(
        "HdrColorBuffer",
        1920,
        1080,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let color_handle = graph.create_transient_texture(color_desc);

    // Pass 1: Depth Prepass (Reverse-Z clear to 0.0, write depth)
    graph
        .add_pass("DepthPrepass", PassType::Graphics)
        .write_depth(
            depth_handle,
            LoadOp::Clear,
            StoreOp::Store,
            ReverseZ::CLEAR_DEPTH,
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 1920, 1080);
        });

    // Pass 2: Forward Shading (Reads prepass depth, writes color, tests depth with LoadOp::Load)
    graph
        .add_pass("ForwardClusteredShading", PassType::Graphics)
        .write_color(
            color_handle,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.1, 0.1, 0.1, 1.0],
        )
        .write_depth(depth_handle, LoadOp::Load, StoreOp::Store, 0.0)
        .mark_as_root()
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 1920, 1080);
        });

    let compiled = graph.compile().expect("Graph compilation must succeed");
    assert_eq!(compiled.passes().len(), 2);

    let passes = compiled.passes();
    assert_eq!(passes[0].name(), "DepthPrepass");
    assert_eq!(passes[1].name(), "ForwardClusteredShading");

    // Depth attachment on pass 0 must have clear_depth 0.0 and load_op Clear
    let p0_desc = passes[0]
        .rendering_descriptor()
        .expect("Prepass rendering descriptor");
    let p0_depth = p0_desc
        .depth_attachment
        .as_ref()
        .expect("Prepass depth attachment");
    assert_eq!(p0_depth.load_op, LoadOp::Clear);
    assert_eq!(p0_depth.clear_depth, Some(0.0));

    // Depth attachment on pass 1 must have load_op Load (Early-Z / Depth testing against prepass)
    let p1_desc = passes[1]
        .rendering_descriptor()
        .expect("Forward rendering descriptor");
    let p1_depth = p1_desc
        .depth_attachment
        .as_ref()
        .expect("Forward depth attachment");
    assert_eq!(p1_depth.load_op, LoadOp::Load);
}

#[test]
fn test_g10_froxel_rows_match_top_down_rhi_pixel_rows() {
    use korppi_math::Vec4;

    let (w, h) = (1920u32, 1080u32);
    let config = ClusterGridConfig::new(w, h, 64, 24, 0.1, 100.0);
    let proj = ReverseZ::perspective_rh(60f32.to_radians(), w as f32 / h as f32, 0.1, 100.0);
    let froxels = config.build_froxel_aabbs(proj.inverse());
    let mut culler = ClusteredLightCuller::new(&config);
    let light = PointLight {
        position_vs: Vec3::new(0.0, 3.0, -10.0),
        radius: 0.5,
        color: Vec3::ONE,
        intensity: 1.0,
    };
    let out = culler.cull_lights(&froxels, &[light], &[]);
    let clip = proj * Vec4::new(0.0, 3.0, -10.0, 1.0);
    let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
    let pixel_top_left = Vec2::new(
        (ndc.x + 1.0) * 0.5 * w as f32,
        (1.0 - ndc.y) * 0.5 * h as f32,
    );
    let idx = config.compute_cluster_index(pixel_top_left, 10.0) as usize;
    let mirrored = config.compute_cluster_index(
        Vec2::new(pixel_top_left.x, h as f32 - pixel_top_left.y),
        10.0,
    ) as usize;

    assert_eq!(
        out.cluster_records[idx].point_count, 1,
        "G10: light above view axis must be culled into its top-down pixel cluster"
    );
    assert_eq!(
        out.cluster_records[mirrored].point_count, 0,
        "G10: light above view axis must not be culled into the vertically mirrored bottom cluster"
    );
}

#[test]
fn test_rnd15_orthographic_froxel_aabbs_and_culling() {
    use korppi_math::Vec4;

    let (w, h) = (640u32, 480u32);
    let config = ClusterGridConfig::new(w, h, 64, 8, 1.0, 50.0);
    let proj = Mat4::orthographic_rh(-10.0, 10.0, -7.5, 7.5, 1.0, 50.0);
    let froxels = config.build_froxel_aabbs(proj.inverse());

    // In an orthographic projection, X/Y bounds of a tile do not diverge with Z slice
    let near_top_left = &froxels[0]; // (x=0, y=0, z=0)
    let far_top_left = &froxels[(7 * config.dim_x() * config.dim_y()) as usize]; // (x=0, y=0, z=7)
    assert!(
        (near_top_left.aabb_min.x - (-10.0)).abs() < 1e-3
            && (far_top_left.aabb_min.x - (-10.0)).abs() < 1e-3,
        "Orthographic froxel X min must stay at -10.0 across all Z slices, got near={} far={}",
        near_top_left.aabb_min.x,
        far_top_left.aabb_min.x
    );
    assert!(
        (near_top_left.aabb_max.y - 7.5).abs() < 1e-3
            && (far_top_left.aabb_max.y - 7.5).abs() < 1e-3,
        "Orthographic top-row froxel Y max must stay at +7.5 across all Z slices, got near={} far={}",
        near_top_left.aabb_max.y,
        far_top_left.aabb_max.y
    );

    // Off-axis light in orthographic view space must be found in its pixel's cluster
    let light = PointLight {
        position_vs: Vec3::new(-7.0, 5.0, -25.0),
        radius: 0.4,
        color: Vec3::ONE,
        intensity: 1.0,
    };
    let mut culler = ClusteredLightCuller::new(&config);
    let out = culler.cull_lights(&froxels, &[light], &[]);
    let clip = proj * Vec4::new(-7.0, 5.0, -25.0, 1.0);
    let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
    let pixel = Vec2::new(
        (ndc.x + 1.0) * 0.5 * w as f32,
        (1.0 - ndc.y) * 0.5 * h as f32,
    );
    let idx = config.compute_cluster_index(pixel, 25.0) as usize;
    assert_eq!(out.cluster_records[idx].point_count, 1);
}
