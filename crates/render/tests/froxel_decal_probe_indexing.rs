//! Verification test for Decal & Reflection Probe Froxel Indexing (AUDIT-4.2 / Finding 3.2).
//!
//! Research reference: `research/render/clustered_shading_and_light_culling.md:183-200`
//! and `comprehensive_forward_renderer_synthesis.md:105-107`.
//! Froxels index point lights, spotlights, un-deferred decals, and reflection probes in a unified pass.

use korppi_math::{Quat, Vec3};
use korppi_render::clustered::{
    test_decal_vs_aabb, test_probe_vs_aabb, ClusterGridConfig, ClusterRecord, ClusteredLightCuller,
    Decal, PointLight, ReflectionProbe,
};

#[test]
fn test_cluster_record_layout_and_size() {
    assert_eq!(
        std::mem::size_of::<ClusterRecord>(),
        32,
        "ClusterRecord must contain 8 u32 fields (32 bytes)"
    );

    let record = ClusterRecord {
        point_offset: 0,
        point_count: 2,
        spot_offset: 2,
        spot_count: 1,
        decal_offset: 0,
        decal_count: 3,
        probe_offset: 0,
        probe_count: 1,
    };

    assert_eq!(record.decal_count, 3);
    assert_eq!(record.probe_count, 1);
}

#[test]
fn test_decal_obb_vs_aabb_intersection() {
    let aabb_min = Vec3::new(-1.0, -1.0, -5.0);
    let aabb_max = Vec3::new(1.0, 1.0, -3.0);

    // 1. Decal inside AABB
    let decal_inside = Decal::axis_aligned(Vec3::new(0.0, 0.0, -4.0), Vec3::new(0.5, 0.5, 0.5));
    assert!(test_decal_vs_aabb(&decal_inside, aabb_min, aabb_max));

    // 2. Decal clearly outside AABB
    let decal_outside = Decal::axis_aligned(Vec3::new(10.0, 0.0, -4.0), Vec3::new(0.5, 0.5, 0.5));
    assert!(!test_decal_vs_aabb(&decal_outside, aabb_min, aabb_max));

    // 3. Rotated decal intersecting corner of AABB
    let rotation_45_y = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);
    let decal_rotated = Decal::new(
        Vec3::new(1.2, 0.0, -3.0),
        Vec3::new(0.8, 0.5, 0.2),
        rotation_45_y,
    );
    assert!(test_decal_vs_aabb(&decal_rotated, aabb_min, aabb_max));
}

#[test]
fn test_reflection_probe_vs_aabb_intersection() {
    let aabb_min = Vec3::new(-2.0, -2.0, -10.0);
    let aabb_max = Vec3::new(2.0, 2.0, -5.0);

    // 1. Spherical probe intersecting
    let sphere_probe = ReflectionProbe::sphere(Vec3::new(0.0, 0.0, -7.5), 1.0, 1.0);
    assert!(test_probe_vs_aabb(&sphere_probe, aabb_min, aabb_max));

    // 2. Spherical probe outside
    let sphere_outside = ReflectionProbe::sphere(Vec3::new(10.0, 0.0, -7.5), 1.0, 1.0);
    assert!(!test_probe_vs_aabb(&sphere_outside, aabb_min, aabb_max));

    // 3. Box probe intersecting
    let box_probe =
        ReflectionProbe::box_projected(Vec3::new(1.5, 0.0, -6.0), Vec3::new(1.0, 1.0, 1.0), 1.0);
    assert!(test_probe_vs_aabb(&box_probe, aabb_min, aabb_max));

    // 4. Box probe outside
    let box_outside =
        ReflectionProbe::box_projected(Vec3::new(5.0, 0.0, -6.0), Vec3::new(1.0, 1.0, 1.0), 1.0);
    assert!(!test_probe_vs_aabb(&box_outside, aabb_min, aabb_max));
}

#[test]
fn test_froxel_decal_and_probe_culling_integration() {
    let config = ClusterGridConfig::new(1920, 1080, 64, 16, 0.1, 100.0);
    let inv_proj = korppi_render::clustered::ReverseZ::perspective_rh(
        60.0f32.to_radians(),
        1920.0 / 1080.0,
        0.1,
        100.0,
    )
    .inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);

    let point_lights = vec![PointLight {
        position_vs: Vec3::new(0.0, 0.0, -5.0),
        radius: 2.0,
        color: Vec3::ONE,
        intensity: 100.0,
    }];
    let spot_lights = vec![];

    let decals = vec![
        Decal::axis_aligned(Vec3::new(0.0, 0.0, -5.0), Vec3::new(1.0, 1.0, 1.0)),
        Decal::axis_aligned(Vec3::new(50.0, 50.0, -50.0), Vec3::new(1.0, 1.0, 1.0)), // Outside
    ];

    let probes = vec![ReflectionProbe::box_projected(
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(3.0, 3.0, 3.0),
        1.0,
    )];

    let mut culler = ClusteredLightCuller::new(&config);
    let output = culler.cull_features(&froxels, &point_lights, &spot_lights, &decals, &probes);

    assert_eq!(
        output.cluster_records.len(),
        config.total_clusters() as usize
    );
    assert!(!output.global_light_indices.is_empty());
    assert!(!output.global_decal_indices.is_empty());
    assert!(!output.global_probe_indices.is_empty());

    // Decal 0 should be visible in central froxels around Z = -5.0
    let has_decal = output.cluster_records.iter().any(|r| r.decal_count > 0);
    assert!(
        has_decal,
        "At least one froxel must index the overlapping decal"
    );

    // Probe 0 should be visible
    let has_probe = output.cluster_records.iter().any(|r| r.probe_count > 0);
    assert!(
        has_probe,
        "At least one froxel must index the overlapping reflection probe"
    );

    // Backward-compatible cull_lights call should also succeed
    let lights_only = culler.cull_lights(&froxels, &point_lights, &spot_lights);
    assert_eq!(
        lights_only.cluster_records.len(),
        config.total_clusters() as usize
    );
    assert!(lights_only
        .cluster_records
        .iter()
        .all(|r| r.decal_count == 0 && r.probe_count == 0));
}
