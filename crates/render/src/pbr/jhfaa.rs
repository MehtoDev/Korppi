//! Johan Hammes Fresnel Anti-Aliasing (JHFAA) and geometric curvature roughness.

use korppi_math::Vec3;

/// Rectifies silhouette interpolated normals that tilt away from the eye vector.
///
/// Prevents unphysical back-facing reflection rays and runaway Schlick Fresnel spikes
/// at polygon silhouettes by projecting and clamping $\mathbf{N} \cdot \mathbf{V} \ge \text{threshold}$.
#[inline]
pub fn rectify_silhouette_normal(normal: Vec3, view_dir: Vec3, threshold: f32) -> Vec3 {
    let n = if normal.is_finite() {
        normal.try_normalize().unwrap_or(Vec3::Y)
    } else {
        Vec3::Y
    };
    let v = if view_dir.is_finite() {
        view_dir.try_normalize().unwrap_or(n)
    } else {
        n
    };
    let thresh = if threshold.is_finite() {
        threshold.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let n_dot_v = n.dot(v);

    if n_dot_v < thresh {
        let correction = thresh - n_dot_v;
        let candidate = n + v * correction;
        if let Some(rectified) = candidate.try_normalize() {
            rectified
        } else {
            // `n` and `v` are antiparallel and `correction == 1.0`: pick an orthogonal tangent tilted toward `v`.
            let ortho = if v.y.abs() < 0.99 {
                v.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X)
            } else {
                v.cross(Vec3::X).try_normalize().unwrap_or(Vec3::Z)
            };
            (ortho + v * thresh.max(1e-4)).try_normalize().unwrap_or(v)
        }
    } else {
        n
    }
}

/// Derives effective perceptual roughness from screen-space surface normal derivatives (`RND-15`, `RND-25`).
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
    let base_r = if material_roughness.is_finite() {
        material_roughness.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scale = if curvature_scale.is_finite() {
        curvature_scale.max(0.0)
    } else {
        0.0
    };
    let normal_variance = dndx.length_squared() + dndy.length_squared();
    let geom_variance = if normal_variance.is_finite() {
        scale * normal_variance
    } else {
        0.0
    };
    let alpha = base_r * base_r;
    let alpha_sq = alpha * alpha;

    (alpha_sq + geom_variance)
        .clamp(0.0, 1.0)
        .sqrt()
        .sqrt()
        .clamp(base_r, 1.0)
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
    let l = if normal_length.is_finite() {
        normal_length.clamp(0.001, 1.0)
    } else {
        1.0
    };
    let f = if factor.is_finite() {
        factor.max(0.0)
    } else {
        0.0
    };
    f * ((1.0 - l) / l)
}

/// Derives effective perceptual roughness from base perceptual roughness and Toksvig normal map variance (`RND-15`, `RND-25`).
///
/// Adds Toksvig variance to $\alpha^2 = (r_{\text{base}}^2)^2$ and converts back to perceptual roughness.
#[inline]
pub fn compute_toksvig_roughness(base_roughness: f32, normal_length: f32) -> f32 {
    let base_r = if base_roughness.is_finite() {
        base_roughness.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let toksvig_variance = compute_toksvig_variance(normal_length, 0.5);
    let alpha = base_r * base_r;
    let alpha_sq = alpha * alpha;
    (alpha_sq + toksvig_variance)
        .clamp(0.0, 1.0)
        .sqrt()
        .sqrt()
        .clamp(base_r, 1.0)
}

/// Derives effective perceptual roughness incorporating base material roughness, screen-space geometric curvature,
/// and Toksvig normal map texture variance filtering (`RND-15`, `RND-25`).
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
    let base_r = if base_roughness.is_finite() {
        base_roughness.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scale = if curvature_scale.is_finite() {
        curvature_scale.max(0.0)
    } else {
        0.0
    };
    let normal_variance = dndx.length_squared() + dndy.length_squared();
    let geom_variance = if normal_variance.is_finite() {
        scale * normal_variance
    } else {
        0.0
    };
    let toksvig_variance = compute_toksvig_variance(normal_length, 0.5);
    let alpha = base_r * base_r;
    let alpha_sq = alpha * alpha;

    (alpha_sq + geom_variance + toksvig_variance)
        .clamp(0.0, 1.0)
        .sqrt()
        .sqrt()
        .clamp(base_r, 1.0)
}
