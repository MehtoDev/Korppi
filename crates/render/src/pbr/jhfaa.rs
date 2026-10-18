//! Johan Hammes Fresnel Anti-Aliasing (JHFAA) and geometric curvature roughness.

use korppi_math::Vec3;

/// Rectifies silhouette interpolated normals that tilt away from the eye vector.
///
/// Prevents unphysical back-facing reflection rays and runaway Schlick Fresnel spikes
/// at polygon silhouettes by projecting and clamping $\mathbf{N} \cdot \mathbf{V} \ge \text{threshold}$.
#[inline]
pub fn rectify_silhouette_normal(normal: Vec3, view_dir: Vec3, threshold: f32) -> Vec3 {
    let n = normal.normalize();
    let v = view_dir.normalize();
    let n_dot_v = n.dot(v);

    if n_dot_v < threshold {
        let correction = threshold - n_dot_v;
        (n + v * correction).normalize()
    } else {
        n
    }
}

/// Derives effective perceptual roughness from screen-space surface normal derivatives (`RND-15`).
///
/// Following Kaplanyan & Tokuyoshi (2019) and Google Filament, normal variance is added to $\alpha^2$
/// (where microfacet roughness $\alpha = r_{\text{perceptual}}^2$):
/// $$\alpha_{\text{eff}}^2 = \text{clamp}\left(\alpha_{\text{mat}}^2 + \kappa (\|\partial_x \mathbf{N}\|^2 + \|\partial_y \mathbf{N}\|^2), 0, 1\right), \quad r_{\text{eff}} = (\alpha_{\text{eff}}^2)^{1/4}$$
#[inline]
pub fn compute_geometric_roughness(
    dndx: Vec3,
    dndy: Vec3,
    material_roughness: f32,
    curvature_scale: f32,
) -> f32 {
    let normal_variance = dndx.length_squared() + dndy.length_squared();
    let geom_variance = curvature_scale * normal_variance;
    let alpha = material_roughness * material_roughness;
    let alpha_sq = alpha * alpha;

    (alpha_sq + geom_variance)
        .clamp(0.0, 1.0)
        .sqrt()
        .sqrt()
        .clamp(material_roughness, 1.0)
}

/// Computes the Toksvig normal map variance expansion factor from the length of a sampled/filtered normal vector.
///
/// For an unnormalized normal vector sampled from a normal map MIP level, length $L \in (0, 1]$
/// measures the statistical alignment of micro-facets within the texel footprint.
///
/// Toksvig variance:
/// $$\sigma_{toksvig}^2 = \kappa \cdot \frac{1.0 - L_{clamped}}{L_{clamped}}$$
/// with default $\kappa = 0.5$.
#[inline]
pub fn compute_toksvig_variance(normal_length: f32, factor: f32) -> f32 {
    let l = normal_length.clamp(0.001, 1.0);
    factor * ((1.0 - l) / l)
}

/// Derives effective perceptual roughness from base perceptual roughness and Toksvig normal map variance (`RND-15`).
///
/// Adds Toksvig variance to $\alpha^2 = (r_{\text{base}}^2)^2$ and converts back to perceptual roughness.
#[inline]
pub fn compute_toksvig_roughness(base_roughness: f32, normal_length: f32) -> f32 {
    let toksvig_variance = compute_toksvig_variance(normal_length, 0.5);
    let alpha = base_roughness * base_roughness;
    let alpha_sq = alpha * alpha;
    (alpha_sq + toksvig_variance)
        .clamp(0.0, 1.0)
        .sqrt()
        .sqrt()
        .clamp(base_roughness, 1.0)
}

/// Derives effective perceptual roughness incorporating base material roughness, screen-space geometric curvature,
/// and Toksvig normal map texture variance filtering (`RND-15`).
///
/// Formula:
/// $$\alpha_{\text{eff}}^2 = \text{clamp}\left(\alpha_{\text{base}}^2 + \sigma_{\text{geom}}^2 + \frac{1.0 - L}{2 L}, 0.0, 1.0\right), \quad r_{\text{eff}} = (\alpha_{\text{eff}}^2)^{1/4}$$
#[inline]
pub fn compute_effective_roughness_toksvig(
    base_roughness: f32,
    dndx: Vec3,
    dndy: Vec3,
    curvature_scale: f32,
    normal_length: f32,
) -> f32 {
    let normal_variance = dndx.length_squared() + dndy.length_squared();
    let geom_variance = curvature_scale * normal_variance;
    let toksvig_variance = compute_toksvig_variance(normal_length, 0.5);
    let alpha = base_roughness * base_roughness;
    let alpha_sq = alpha * alpha;

    (alpha_sq + geom_variance + toksvig_variance)
        .clamp(0.0, 1.0)
        .sqrt()
        .sqrt()
        .clamp(base_roughness, 1.0)
}
