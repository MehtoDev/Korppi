//! Integration and unit tests for Sub-Milestone 10.4:
//! Raytraced Geometric Acoustics & Path Tracing.

use korppi_audio::raytrace::{
    energy_decay::{eyring_rt60, sabine_rt60, EnergyHistogram},
    geometry::{AcousticBox, AcousticRay, AcousticScene, AcousticTriangle},
    material::AcousticMaterial,
    tracer::{fibonacci_sphere_rays, PathTraceConfig, RaytracedAcoustics},
};
use korppi_math::Vec3;

#[test]
fn test_acoustic_material_presets_and_properties() {
    let concrete = AcousticMaterial::concrete();
    let carpet = AcousticMaterial::carpet();
    let drywall = AcousticMaterial::drywall();

    // Concrete should have very low absorption (highly reverberant)
    assert!(concrete.absorption[0] < 0.05);
    assert!(concrete.absorption[1] < 0.05);
    assert!(concrete.absorption[2] < 0.06);

    // Carpet should have high absorption at high frequencies
    assert!(carpet.absorption[2] > 0.6);
    assert!(carpet.absorption[2] > carpet.absorption[0]);

    // Reflection = 1.0 - absorption
    let refl = concrete.reflection();
    for (&r, &a) in refl.iter().zip(concrete.absorption.iter()) {
        assert!((r + a - 1.0).abs() < 1e-5);
    }

    // Drywall transmission: bass passes through much more than treble
    assert!(drywall.transmission[0] > drywall.transmission[2]);
}

#[test]
fn test_fibonacci_sphere_ray_distribution() {
    let count = 256;
    let rays = fibonacci_sphere_rays(count);
    assert_eq!(rays.len(), count);

    let mut sum = Vec3::ZERO;
    let mut octant_counts = [0usize; 8];

    for r in &rays {
        // Must be unit vectors
        assert!((r.length() - 1.0).abs() < 1e-4);
        sum += *r;

        // Determine octant
        let ox = if r.x >= 0.0 { 1 } else { 0 };
        let oy = if r.y >= 0.0 { 2 } else { 0 };
        let oz = if r.z >= 0.0 { 4 } else { 0 };
        octant_counts[ox | oy | oz] += 1;
    }

    // Mean vector should be very close to zero due to spherical symmetry
    let mean = sum / count as f32;
    assert!(
        mean.length() < 0.05,
        "Mean vector should be near origin: {:?}",
        mean
    );

    // Each octant should contain roughly count / 8 = 32 rays (+/- 10)
    for (i, &cnt) in octant_counts.iter().enumerate() {
        assert!(
            (cnt as i32 - 32).abs() <= 10,
            "Octant {} has {} rays, expected ~32",
            i,
            cnt
        );
    }
}

#[test]
fn test_acoustic_scene_ray_intersection_and_box() {
    let mut scene = AcousticScene::new();

    // Box room of 10m x 4m x 8m centered at (0, 0, 0)
    scene.add_box(AcousticBox::new(
        Vec3::ZERO,
        Vec3::new(10.0, 4.0, 8.0),
        AcousticMaterial::concrete(),
    ));

    // Ray from center pointing along +X
    let ray_x = AcousticRay::new(Vec3::ZERO, Vec3::X);
    let hit_x = scene
        .raycast_closest(&ray_x)
        .expect("Should hit right wall");
    assert!((hit_x.distance - 5.0).abs() < 1e-4);
    assert!((hit_x.point - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4);
    assert!(
        (hit_x.normal - Vec3::NEG_X).length() < 1e-4,
        "Normal should face inward"
    );

    // Ray from center pointing along +Y (ceiling)
    let ray_y = AcousticRay::new(Vec3::ZERO, Vec3::Y);
    let hit_y = scene.raycast_closest(&ray_y).expect("Should hit ceiling");
    assert!((hit_y.distance - 2.0).abs() < 1e-4);
    assert!((hit_y.normal - Vec3::NEG_Y).length() < 1e-4);
}

#[test]
fn test_multi_wall_transmission_loss() {
    let mut scene = AcousticScene::new();

    // Two parallel drywall panels at x = 2.0 and x = 4.0
    // Triangle 1: x = 2.0
    scene.add_triangle(AcousticTriangle::new(
        Vec3::new(2.0, -5.0, -5.0),
        Vec3::new(2.0, 5.0, -5.0),
        Vec3::new(2.0, -5.0, 5.0),
        AcousticMaterial::drywall(),
    ));
    scene.add_triangle(AcousticTriangle::new(
        Vec3::new(2.0, 5.0, -5.0),
        Vec3::new(2.0, 5.0, 5.0),
        Vec3::new(2.0, -5.0, 5.0),
        AcousticMaterial::drywall(),
    ));

    // Triangle 2: x = 4.0
    scene.add_triangle(AcousticTriangle::new(
        Vec3::new(4.0, -5.0, -5.0),
        Vec3::new(4.0, 5.0, -5.0),
        Vec3::new(4.0, -5.0, 5.0),
        AcousticMaterial::drywall(),
    ));
    scene.add_triangle(AcousticTriangle::new(
        Vec3::new(4.0, 5.0, -5.0),
        Vec3::new(4.0, 5.0, 5.0),
        Vec3::new(4.0, -5.0, 5.0),
        AcousticMaterial::drywall(),
    ));

    let ray = AcousticRay::new(Vec3::new(0.0, 1.0, 1.0), Vec3::X);
    let max_dist = 6.0;

    let transmission = scene.raycast_transmission(&ray, max_dist);
    let single_drywall_trans = AcousticMaterial::drywall().transmission;

    // With 2 walls, transmission should be roughly tau^2
    let expected_low = single_drywall_trans[0] * single_drywall_trans[0];
    let expected_high = single_drywall_trans[2] * single_drywall_trans[2];

    assert!((transmission[0] - expected_low).abs() < 1e-3);
    assert!((transmission[2] - expected_high).abs() < 1e-3);
    // Bass passes significantly better than treble through multiple walls
    assert!(transmission[0] > transmission[2] * 5.0);
}

#[test]
fn test_next_event_estimation_shadow_rays() {
    let mut scene = AcousticScene::new();
    // Wall at x = 5.0
    scene.add_triangle(AcousticTriangle::new(
        Vec3::new(5.0, -10.0, -10.0),
        Vec3::new(5.0, 10.0, -10.0),
        Vec3::new(5.0, -10.0, 10.0),
        AcousticMaterial::concrete(),
    ));
    scene.add_triangle(AcousticTriangle::new(
        Vec3::new(5.0, 10.0, -10.0),
        Vec3::new(5.0, 10.0, 10.0),
        Vec3::new(5.0, -10.0, 10.0),
        AcousticMaterial::concrete(),
    ));

    let tracer = RaytracedAcoustics::new(PathTraceConfig {
        ray_count: 64,
        max_bounces: 2,
        energy_threshold: 1e-4,
    });

    let emitter = Vec3::new(0.0, 0.0, 0.0);
    let listener = Vec3::new(0.0, 2.0, 0.0);

    let reflections = tracer.trace_early_reflections(&scene, emitter, listener);

    // There should be valid early reflection hits gathered via NEE shadow rays
    assert!(
        !reflections.is_empty(),
        "Should gather early reflections from wall"
    );

    for r in &reflections {
        assert!(r.arrival_time_sec > 0.0);
        // Distance emitter -> wall (5m) + wall -> listener (~5.38m) >= 10m
        // Travel time >= 10 / 343 = ~0.029 s
        assert!(r.arrival_time_sec >= 0.025);
        assert!(r.energy[0] > 0.0);
        assert!((r.direction.length() - 1.0).abs() < 1e-3);
    }
}

#[test]
fn test_energy_decay_curve_and_rt60_matches_sabine_eyring() {
    // Test Sabine and Eyring formula implementations
    let volume = 300.0; // 10 x 10 x 3 m
    let surface_area = 320.0; // 2*(100 + 30 + 30) = 320 m^2
    let concrete_alpha = 0.03;

    let sabine = sabine_rt60(volume, surface_area, concrete_alpha);
    let eyring = eyring_rt60(volume, surface_area, concrete_alpha);

    // For low absorption (0.03), Sabine and Eyring are very close (~5.0 seconds)
    assert!((sabine - eyring).abs() < 0.2);
    assert!((sabine - 5.03).abs() < 0.2);

    // For high absorption (0.6), Eyring is physically correct while Sabine diverges
    let high_alpha = 0.6;
    let eyring_dead = eyring_rt60(volume, surface_area, high_alpha);
    assert!(
        eyring_dead < 0.2,
        "Dead room should have short RT60 (< 0.2s)"
    );

    // Test EnergyHistogram accumulation and Schroeder backward integration
    let mut hist = EnergyHistogram::new(0.01, 300); // 10ms bins, 3.0s duration
                                                    // Inject exponential decay signal: E(t) = E0 * exp(-13.8 * t / RT60_target) where RT60 = 1.0s
    let target_rt60 = 1.0f32;
    let decay_rate = 13.8155 / target_rt60; // ln(10^6) = 13.8155

    for bin in 0..200 {
        let t = bin as f32 * 0.01;
        let energy = (-decay_rate * t).exp();
        hist.add_energy(t, [energy, energy, energy]);
    }

    let extracted_rt60 = hist.extract_rt60();
    // Should match target RT60 within 10%
    assert!(
        (extracted_rt60[0] - target_rt60).abs() < 0.15,
        "Extracted RT60 was {}, expected ~{}",
        extracted_rt60[0],
        target_rt60
    );
}
