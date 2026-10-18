//! Verification tests for Multi-Scattering Energy Compensation, DFG Split-Sum Integration,
//! White Furnace Conservation, Charlie Sheen Attenuation, Roughness Floor, and Specular AA
//! (`FIX-6.2`, `RND-06`, `RND-15`, Probe `G1`).

use std::f32::consts::PI;

use korppi_math::{Vec2, Vec3};
use korppi_render::pbr::{
    approximate_dfg, compute_effective_roughness_toksvig, compute_energy_compensation,
    compute_geometric_roughness, compute_toksvig_roughness, compute_toksvig_variance, d_ggx,
    evaluate_pbr, v_smith_heitz, CharlieSheen, PbrMaterial, PbrSurface,
};

/// Numerical split-sum integration of `(scale, bias)` = `(A, B)` over the hemisphere
/// using midpoint quadrature in `(cos_theta, phi)` (`Probe G1`).
fn numerical_split_sum(roughness: f32, n_dot_v: f32) -> (f32, f32) {
    let v = Vec3::new((1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), n_dot_v, 0.0);
    let (nu, nphi) = (1024, 128);
    let (mut a, mut b) = (0.0f64, 0.0f64);
    for i in 0..nu {
        let cos_t = (i as f32 + 0.5) / nu as f32;
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        for j in 0..nphi {
            let phi = (j as f32 + 0.5) / nphi as f32 * 2.0 * PI;
            let l = Vec3::new(sin_t * phi.cos(), cos_t, sin_t * phi.sin());
            let h = (v + l).normalize();
            let dv = d_ggx(h.y, roughness) * v_smith_heitz(cos_t, n_dot_v, roughness) * cos_t;
            let fc = (1.0 - v.dot(h).max(0.0)).powi(5);
            a += f64::from(dv * (1.0 - fc));
            b += f64::from(dv * fc);
        }
    }
    let w = (1.0 / nu as f64) * (2.0 * std::f64::consts::PI / nphi as f64);
    ((a * w) as f32, (b * w) as f32)
}

/// Directional albedo of `evaluate_pbr` for unit incident radiance by midpoint quadrature
/// over the upper hemisphere (`Probe G1`).
fn white_furnace_albedo(surface: &PbrSurface) -> f32 {
    let (nu, nphi) = (1024, 128);
    let mut sum = 0.0f64;
    for i in 0..nu {
        let cos_t = (i as f32 + 0.5) / nu as f32;
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        for j in 0..nphi {
            let phi = (j as f32 + 0.5) / nphi as f32 * 2.0 * PI;
            let l = Vec3::new(sin_t * phi.cos(), cos_t, sin_t * phi.sin());
            sum += f64::from(evaluate_pbr(surface, l, Vec3::ONE).x);
        }
    }
    (sum * (1.0 / nu as f64) * (2.0 * std::f64::consts::PI / nphi as f64)) as f32
}

#[test]
fn test_energy_compensation_formula_uses_directional_albedo_denominator() {
    let f0 = Vec3::ONE;
    let dfg = Vec2::new(0.5, 0.25);

    let factor = compute_energy_compensation(f0, dfg);

    // Directional albedo for F0 = 1 is E = scale + bias = dfg.x + dfg.y = 0.75.
    // Expected: 1.0 + 1.0 * (1.0 / 0.75 - 1.0) = 1.333333...
    let expected = 1.0 + (1.0 / (dfg.x + dfg.y) - 1.0);
    assert!(
        (factor.x - expected).abs() < 1e-3,
        "Energy compensation must divide by directional albedo (dfg.x + dfg.y). Expected {}, got {}",
        expected,
        factor.x
    );
}

#[test]
fn test_energy_compensation_zero_f0_is_identity() {
    let f0 = Vec3::ZERO;
    let dfg = Vec2::new(0.6, 0.3);

    let factor = compute_energy_compensation(f0, dfg);
    assert_eq!(factor, Vec3::ONE, "When F0 is 0, energy comp must be 1.0");
}

#[test]
fn test_g1_dfg_matches_numerical_split_sum_table() {
    use korppi_render::pbr::integrate_dfg;

    for &(r, nv) in &[
        (0.3f32, 1.0f32),
        (0.4, 0.7),
        (0.6, 1.0),
        (0.6, 0.5),
        (0.8, 0.4),
        (1.0, 1.0),
        (1.0, 0.3),
    ] {
        let (num_a, num_b) = numerical_split_sum(r, nv);
        let dfg = approximate_dfg(nv, r);
        assert!(
            (dfg.x - num_a).abs() < 0.015 && (dfg.y - num_b).abs() < 0.015,
            "approximate_dfg(nv={nv}, r={r}) = ({:.4}, {:.4}) diverged from numerical split-sum ({num_a:.4}, {num_b:.4})",
            dfg.x,
            dfg.y
        );
    }

    // For smooth surfaces (r = 0.1), compare against GGX importance-sampled integration
    let imp = integrate_dfg(0.2, 0.1, 2048);
    let dfg_smooth = approximate_dfg(0.2, 0.1);
    assert!(
        (dfg_smooth.x - imp.x).abs() < 0.015 && (dfg_smooth.y - imp.y).abs() < 0.015,
        "approximate_dfg(0.2, 0.1) = {dfg_smooth:?} diverged from importance-sampled {imp:?}"
    );
}

#[test]
fn test_g1_white_furnace_energy_conservation() {
    for &(r, nv) in &[
        (0.3f32, 1.0f32),
        (0.6, 1.0),
        (1.0, 1.0),
        (0.6, 0.5),
        (1.0, 0.3),
    ] {
        let surface = PbrSurface {
            position: Vec3::ZERO,
            normal: Vec3::Y,
            view: Vec3::new((1.0 - nv * nv).max(0.0).sqrt(), nv, 0.0),
            albedo: Vec3::ONE,
            material: PbrMaterial {
                roughness: r,
                metallic: 1.0,
                reflectance: 0.5,
                clear_coat: None,
                sheen: None,
            },
        };
        let albedo = white_furnace_albedo(&surface);
        assert!(
            (0.97..=1.0).contains(&albedo),
            "G1 white furnace failed at roughness={r}, NoV={nv}: directional albedo={albedo:.4} (expected in [0.97, 1.0], never above 1.0)"
        );
    }
}

#[test]
fn test_rnd15_sheen_visibility_and_substrate_attenuation_conserves_energy() {
    // White dielectric with white Charlie sheen in a white furnace must not exceed 1.0 directional albedo
    for &nv in &[1.0f32, 0.5, 0.2] {
        let surface = PbrSurface {
            position: Vec3::ZERO,
            normal: Vec3::Y,
            view: Vec3::new((1.0 - nv * nv).max(0.0).sqrt(), nv, 0.0),
            albedo: Vec3::ONE,
            material: PbrMaterial {
                roughness: 0.8,
                metallic: 0.0,
                reflectance: 0.5,
                clear_coat: None,
                sheen: Some(CharlieSheen {
                    color: Vec3::ONE,
                    roughness: 0.5,
                }),
            },
        };
        let albedo = white_furnace_albedo(&surface);
        assert!(
            albedo <= 1.0,
            "RND-15: white dielectric + white Charlie sheen exceeded unit furnace energy at NoV={nv}: {albedo:.4}"
        );
    }
}

#[test]
fn test_rnd15_roughness_floor_and_specular_aa_alpha_squared() {
    // 1. Roughness floor: perceptual roughness 0.0 clamps to >= 0.045 so D_GGX * V_Smith fits in RGBA16F (65,504)
    let d_zero = d_ggx(1.0, 0.0);
    let d_floor = d_ggx(1.0, 0.045);
    assert!(
        (d_zero - d_floor).abs() < 1e-3,
        "d_ggx must clamp perceptual roughness at 0.045 (got d_zero={d_zero}, d_floor={d_floor})"
    );
    let dv_peak = d_ggx(1.0, 0.0) * v_smith_heitz(1.0, 1.0, 0.0);
    assert!(
        dv_peak < 65_504.0,
        "Peak specular D * V at roughness 0 must fit within RGBA16F max (65,504), got {dv_peak}"
    );

    // 2. Specular AA adds variance to alpha^2 (where alpha = roughness^2), not to alpha
    let base_r = 0.5f32;
    let alpha = base_r * base_r; // 0.25
    let dndx = Vec3::new(0.2, 0.0, 0.0);
    let dndy = Vec3::new(0.0, 0.1, 0.0);
    let curvature_scale = 1.0;
    let geom_var = curvature_scale * (dndx.length_squared() + dndy.length_squared()); // 0.05
    let expected_geom_r = (alpha * alpha + geom_var).clamp(0.0, 1.0).sqrt().sqrt();
    let actual_geom_r = compute_geometric_roughness(dndx, dndy, base_r, curvature_scale);
    assert!(
        (actual_geom_r - expected_geom_r).abs() < 1e-5,
        "compute_geometric_roughness must add variance to alpha^2: expected {expected_geom_r}, got {actual_geom_r}"
    );

    let normal_len = 0.8f32;
    let tok_var = compute_toksvig_variance(normal_len, 0.5);
    let expected_tok_r = (alpha * alpha + tok_var).clamp(0.0, 1.0).sqrt().sqrt();
    let actual_tok_r = compute_toksvig_roughness(base_r, normal_len);
    assert!(
        (actual_tok_r - expected_tok_r).abs() < 1e-5,
        "compute_toksvig_roughness must add variance to alpha^2: expected {expected_tok_r}, got {actual_tok_r}"
    );

    let expected_comb_r = (alpha * alpha + geom_var + tok_var)
        .clamp(0.0, 1.0)
        .sqrt()
        .sqrt();
    let actual_comb_r =
        compute_effective_roughness_toksvig(base_r, dndx, dndy, curvature_scale, normal_len);
    assert!(
        (actual_comb_r - expected_comb_r).abs() < 1e-5,
        "compute_effective_roughness_toksvig must add variances to alpha^2: expected {expected_comb_r}, got {actual_comb_r}"
    );
}

#[test]
fn test_evaluate_pbr_conductor_grazing_energy_restoration() {
    let surface_smooth = PbrSurface {
        position: Vec3::ZERO,
        normal: Vec3::Y,
        view: Vec3::new(0.8, 0.2, 0.0).normalize(),
        albedo: Vec3::ONE,
        material: PbrMaterial {
            roughness: 0.1,
            metallic: 1.0,
            reflectance: 1.0,
            clear_coat: None,
            sheen: None,
        },
    };

    let surface_rough = PbrSurface {
        position: Vec3::ZERO,
        normal: Vec3::Y,
        view: Vec3::new(0.8, 0.2, 0.0).normalize(),
        albedo: Vec3::ONE,
        material: PbrMaterial {
            roughness: 0.8,
            metallic: 1.0,
            reflectance: 1.0,
            clear_coat: None,
            sheen: None,
        },
    };

    let light_dir = Vec3::new(-0.8, 0.2, 0.0).normalize();
    let radiance = Vec3::splat(1.0);

    let color_smooth = evaluate_pbr(&surface_smooth, light_dir, radiance);
    let color_rough = evaluate_pbr(&surface_rough, light_dir, radiance);

    assert!(color_smooth.is_finite() && color_smooth.x > 0.0);
    assert!(color_rough.is_finite() && color_rough.x > 0.0);

    let dfg_rough = approximate_dfg(surface_rough.normal.dot(surface_rough.view), 0.8);
    let comp = compute_energy_compensation(Vec3::ONE, dfg_rough);
    assert!(
        comp.x >= 1.0,
        "Multi-scattering must restore energy (factor >= 1.0) on rough surfaces"
    );
}
