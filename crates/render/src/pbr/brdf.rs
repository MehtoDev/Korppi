//! Cook-Torrance microfacet BRDF, multi-scattering GGX, and energy conservation (`RND-06`, `RND-15`).

use crate::pbr::dfg::{approximate_dfg, approximate_sheen_dfg};
use crate::pbr::material::PbrSurface;
use korppi_math::{Vec2, Vec3};

const PI: f32 = std::f32::consts::PI;

/// Minimum perceptual roughness clamp matching Google Filament (`RND-15`),
/// preventing `D_GGX * V_Smith` from overflowing `RGBA16Float` (max `65,504.0`).
pub const MIN_PERCEPTUAL_ROUGHNESS: f32 = 0.045;

/// Trowbridge-Reitz / GGX normal distribution function (NDF).
#[inline]
pub fn d_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let r = roughness.clamp(MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    let alpha = r * r;
    let a2 = alpha * alpha;
    let nh = n_dot_h.clamp(0.0, 1.0);
    let denom_part = nh * nh * (a2 - 1.0) + 1.0;
    let denom = PI * denom_part * denom_part;
    a2 / denom.max(1e-7)
}

/// Eric Heitz height-correlated Smith masking-shadowing visibility function $V = \frac{G_2}{4 (N \cdot L)(N \cdot V)}$.
#[inline]
pub fn v_smith_heitz(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness.clamp(MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    let alpha = r * r;
    let a2 = alpha * alpha;
    let nl = n_dot_l.clamp(0.0, 1.0);
    let nv = n_dot_v.clamp(0.0, 1.0);

    let lambda_v = nl * (nv * nv * (1.0 - a2) + a2).sqrt();
    let lambda_l = nv * (nl * nl * (1.0 - a2) + a2).sqrt();
    0.5 / (lambda_v + lambda_l).max(1e-7)
}

/// Neubelt & Pettineo smooth cloth visibility term used with the Charlie sheen NDF (`RND-15`).
#[inline]
pub fn v_neubelt(n_dot_l: f32, n_dot_v: f32) -> f32 {
    let nl = n_dot_l.clamp(0.0, 1.0);
    let nv = n_dot_v.clamp(0.0, 1.0);
    1.0 / (4.0 * (nl + nv - nl * nv)).max(1e-5)
}

/// Schlick approximation of the Fresnel reflectance equation.
#[inline]
pub fn fresnel_schlick(v_dot_h: f32, f0: Vec3) -> Vec3 {
    let vh = v_dot_h.clamp(0.0, 1.0);
    let factor = (1.0 - vh).powi(5);
    f0 + (Vec3::ONE - f0) * factor
}

/// Inverted exponential Charlie distribution function for fabric and velvet sheen.
#[inline]
pub fn d_charlie(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness.clamp(MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    let nh = n_dot_h.clamp(0.0, 1.0);
    let sin2 = (1.0 - nh * nh).max(0.0);
    let cos2 = nh * nh;
    let denom = sin2 + alpha * alpha * cos2;
    if denom <= 0.0 {
        return 0.0;
    }
    let inv_alpha = 1.0 / alpha;
    (2.0 + inv_alpha) * sin2.powf(inv_alpha * 0.5) / (2.0 * PI)
}

/// Evaluates the Fdez-Agüera / Filament multi-scattering energy compensation factor for GGX specular reflection (`RND-06`).
///
/// Given `dfg = Vec2(scale, bias)` from [`approximate_dfg`], the single-scattering directional albedo
/// for a white conductor ($F_0 = 1$) is $E = \text{dfg.x} + \text{dfg.y}$:
/// $$C_{energy} = 1.0 + F_0 \cdot \left( \frac{1.0}{\text{dfg.x} + \text{dfg.y}} - 1.0 \right)$$
#[inline]
pub fn compute_energy_compensation(f0: Vec3, dfg: Vec2) -> Vec3 {
    let directional_albedo = (dfg.x + dfg.y).max(1e-4);
    Vec3::ONE + f0 * (1.0 / directional_albedo - 1.0)
}

/// Evaluates the complete multi-scattering PBR BRDF for a surface point, light direction, and radiance.
pub fn evaluate_pbr(surface: &PbrSurface, light_dir: Vec3, radiance: Vec3) -> Vec3 {
    let n = surface.normal.normalize();
    let v = surface.view.normalize();
    let l = light_dir.normalize();

    let n_dot_l = n.dot(l).max(0.0);
    let n_dot_v = n.dot(v).max(1e-4);

    if n_dot_l <= 0.0 {
        return Vec3::ZERO;
    }

    let h = (v + l).normalize();
    let n_dot_h = n.dot(h).max(0.0);
    let v_dot_h = v.dot(h).max(0.0);

    // Compute characteristic specular reflectance at normal incidence F0
    // Dielectric base F0 is 0.04 * reflectance * 2.0; Conductors use base albedo
    let dielectric_f0 = Vec3::splat(0.08 * surface.material.reflectance);
    let f0 = dielectric_f0.lerp(surface.albedo, surface.material.metallic);

    // 1. Specular microfacet evaluation
    let d = d_ggx(n_dot_h, surface.material.roughness);
    let vis = v_smith_heitz(n_dot_l, n_dot_v, surface.material.roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    // 2. Fdez-Agüera / Filament Multi-Scattering Energy Compensation (RND-06)
    let dfg = approximate_dfg(n_dot_v, surface.material.roughness);
    let energy_comp = compute_energy_compensation(f0, dfg);
    let specular_single = f * (d * vis);
    let specular_compensated = specular_single * energy_comp;

    // 3. Diffuse Lambertian with directional specular energy conservation
    let spec_directional_albedo =
        ((f0 * dfg.x + Vec3::splat(dfg.y)) * energy_comp).clamp(Vec3::ZERO, Vec3::ONE);
    let kd = (Vec3::ONE - f).min(Vec3::ONE - spec_directional_albedo)
        * (1.0 - surface.material.metallic);
    let diffuse = (surface.albedo / PI) * kd;

    let mut brdf_composite = diffuse + specular_compensated;

    // 4. Multi-Layer Extensions: Clear Coat
    if let Some(coat) = surface.material.clear_coat {
        let coat_f0 = Vec3::splat(0.04);
        let coat_f = fresnel_schlick(v_dot_h, coat_f0);
        let coat_d = d_ggx(n_dot_h, coat.roughness);
        let coat_vis = v_smith_heitz(n_dot_l, n_dot_v, coat.roughness);
        let coat_specular = coat_f * (coat_d * coat_vis * coat.weight);

        // Attenuate substrate by clear coat transmission
        brdf_composite = brdf_composite * (Vec3::ONE - coat_f * coat.weight) + coat_specular;
    }

    // 5. Multi-Layer Extensions: Charlie Sheen with Neubelt visibility and substrate attenuation (RND-15)
    if let Some(sheen) = surface.material.sheen {
        let sheen_d = d_charlie(n_dot_h, sheen.roughness);
        let sheen_vis = v_neubelt(n_dot_l, n_dot_v);
        let sheen_lobe = sheen.color * (sheen_d * sheen_vis);
        let max_sheen = sheen
            .color
            .x
            .max(sheen.color.y)
            .max(sheen.color.z)
            .clamp(0.0, 1.0);
        let sheen_albedo = approximate_sheen_dfg(n_dot_v, sheen.roughness);
        let sheen_scaling = (1.0 - max_sheen * sheen_albedo).clamp(0.0, 1.0);
        brdf_composite = brdf_composite * sheen_scaling + sheen_lobe;
    }

    brdf_composite * radiance * n_dot_l
}
