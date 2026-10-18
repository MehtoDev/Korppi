//! Light representations and view-space intersection tests for clustered culling.

use korppi_math::Vec3;

/// Point (omnidirectional) light source in view space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    /// View-space position coordinates.
    pub position_vs: Vec3,
    /// Light attenuation cutoff radius.
    pub radius: f32,
    /// Linear RGB color.
    pub color: Vec3,
    /// Luminous intensity (lumens).
    pub intensity: f32,
}

/// Spot (conical) light source in view space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpotLight {
    /// Apex position in view space.
    pub position_vs: Vec3,
    /// Unit forward direction vector in view space.
    pub direction_vs: Vec3,
    /// Maximum cutoff range.
    pub range: f32,
    /// Half-aperture outer cone angle in radians.
    pub half_angle: f32,
    /// Linear RGB color.
    pub color: Vec3,
    /// Luminous intensity (lumens).
    pub intensity: f32,
}

/// Directional (sun/moon) light source in view space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Unit direction vector from the light towards the scene in view space.
    pub direction_vs: Vec3,
    /// Linear RGB color.
    pub color: Vec3,
    /// Illuminance (lux).
    pub illuminance: f32,
}

/// Tests intersection between a view-space sphere and an Axis-Aligned Bounding Box (AABB) using Arvo's algorithm.
#[inline]
pub fn test_sphere_vs_aabb(center: Vec3, radius: f32, aabb_min: Vec3, aabb_max: Vec3) -> bool {
    let mut dist_sq = 0.0f32;

    for i in 0..3 {
        let c = center[i];
        let min_val = aabb_min[i];
        let max_val = aabb_max[i];

        if c < min_val {
            let d = min_val - c;
            dist_sq += d * d;
        } else if c > max_val {
            let d = c - max_val;
            dist_sq += d * d;
        }
    }

    dist_sq <= radius * radius
}

/// Tests intersection between a spotlight cone and a bounding sphere using Bart Wronski's formulation.
#[inline]
pub fn test_cone_vs_sphere(
    cone_origin: Vec3,
    cone_forward: Vec3,
    cone_range: f32,
    cone_half_angle: f32,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> bool {
    let v = sphere_center - cone_origin;
    let v_len_sq = v.length_squared();
    let norm_forward = cone_forward.normalize();
    let v1_len = v.dot(norm_forward);

    // Orthogonal distance squared to cone axis
    let v_perp_len_sq = (v_len_sq - v1_len * v1_len).max(0.0);
    let v_perp_len = v_perp_len_sq.sqrt();

    // Distance from closest point on cone mantle to sphere center
    let (sin_a, cos_a) = cone_half_angle.sin_cos();
    let dist_closest = cos_a * v_perp_len - v1_len * sin_a;

    let angle_cull = dist_closest > sphere_radius;
    let front_cull = v1_len > (sphere_radius + cone_range);
    let back_cull = v1_len < -sphere_radius;

    !(angle_cull || front_cull || back_cull)
}
