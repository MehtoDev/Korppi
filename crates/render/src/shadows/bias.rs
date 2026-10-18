//! Normal offset bias and adaptive depth biasing algorithms.

use korppi_math::Vec3;

/// Computes the world-space normal offset vector to prevent shadow acne and peter-panning.
///
/// Displaces the surface position along the surface geometric normal $\mathbf{N}$ proportionally
/// to the incidence angle of light direction $\mathbf{L}$:
/// $$\mathbf{P}_{biased} = \mathbf{P} + \mathbf{N} \cdot (\Delta \cdot \beta \cdot \sqrt{1.0 - (\mathbf{N} \cdot \mathbf{L})^2})$$
#[inline]
pub fn normal_offset_bias(normal: Vec3, light_dir: Vec3, texel_size: f32, bias_scale: f32) -> Vec3 {
    let n = normal.normalize();
    let l = light_dir.normalize();

    let n_dot_l = n.dot(l).clamp(0.0, 1.0);
    let sin_theta = (1.0 - n_dot_l * n_dot_l).max(0.0).sqrt();

    n * (texel_size * bias_scale * sin_theta)
}
