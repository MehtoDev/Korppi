//! Verification test for Toksvig Normal Map Variance Roughness Rectification (AUDIT-4.4 / Finding 3.4).
//!
//! Research reference: `research/render/anti_aliasing_and_specular_filtering.md:141-176`.
//! Toksvig variance formula:
//! $$\alpha_{eff} = \sqrt{\text{clamp}\left( \alpha_{base}^2 + (2 \sigma_{geom})^2 + \frac{1.0 - L}{2 L}, 0.0001, 1.0 \right)}$$

use korppi_math::Vec3;
use korppi_render::pbr::{
    compute_effective_roughness_toksvig, compute_toksvig_roughness, compute_toksvig_variance,
};

#[test]
fn test_toksvig_variance_unit_length_is_zero() {
    let variance = compute_toksvig_variance(1.0, 0.5);
    assert!(
        variance.abs() < 1e-6,
        "Unit normal length (L = 1.0) must produce zero Toksvig variance"
    );
}

#[test]
fn test_toksvig_variance_shortened_normal_increases_monotonically() {
    let var_95 = compute_toksvig_variance(0.95, 0.5);
    let var_80 = compute_toksvig_variance(0.80, 0.5);
    let var_50 = compute_toksvig_variance(0.50, 0.5);

    assert!(var_95 > 0.0);
    assert!(
        var_80 > var_95,
        "Shorter normal length must produce greater variance"
    );
    assert!(
        var_50 > var_80,
        "Variance must increase monotonically as normal length shortens"
    );

    // Analytical check: for L = 0.5, variance = 0.5 * (1.0 - 0.5) / 0.5 = 0.5
    assert!(
        (var_50 - 0.5).abs() < 1e-5,
        "Expected variance 0.5 for L = 0.5, got {}",
        var_50
    );
}

#[test]
fn test_toksvig_roughness_widening() {
    let base_roughness = 0.1; // Smooth mirror-like surface

    // Uniform normal (L = 1.0) -> roughness unchanged
    let roughness_uniform = compute_toksvig_roughness(base_roughness, 1.0);
    assert!(
        (roughness_uniform - base_roughness).abs() < 1e-5,
        "Uniform normal must not widen base roughness"
    );

    // Filtered/turbulent normal map (L = 0.7) -> roughness expands
    let roughness_turbulent = compute_toksvig_roughness(base_roughness, 0.7);
    assert!(
        roughness_turbulent > base_roughness,
        "High-frequency normal map variance must widen effective roughness to eliminate specular shimmering"
    );
    assert!(
        roughness_turbulent <= 1.0,
        "Effective roughness must remain <= 1.0"
    );
}

#[test]
fn test_combined_geometric_and_toksvig_effective_roughness() {
    let base_roughness = 0.15;
    let dndx = Vec3::new(0.2, 0.0, 0.0);
    let dndy = Vec3::new(0.0, 0.2, 0.0);
    let curvature_scale = 1.0;
    let normal_length = 0.65; // Shortened from high-frequency normal MIP filtering

    let eff_roughness = compute_effective_roughness_toksvig(
        base_roughness,
        dndx,
        dndy,
        curvature_scale,
        normal_length,
    );

    // Roughness with both geometric curvature and Toksvig variance must exceed either alone
    let only_toksvig = compute_toksvig_roughness(base_roughness, normal_length);
    assert!(
        eff_roughness > only_toksvig,
        "Combined roughness must include both geometric curvature and normal map variance"
    );
    assert!(
        eff_roughness <= 1.0,
        "Effective roughness must remain bounded <= 1.0"
    );
}
