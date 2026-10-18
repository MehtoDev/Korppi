//! Screen-Space Ambient Occlusion (SSAO) with hemisphere sampling and bilateral depth blur.

use korppi_math::Vec3;

/// Configuration settings for the Screen-Space Ambient Occlusion (SSAO) pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsaoConfig {
    /// World-space sampling sphere radius.
    pub sample_radius: f32,
    /// Depth comparison bias to eliminate self-shadowing on flat surfaces.
    pub bias: f32,
    /// Occlusion strength multiplier.
    pub intensity: f32,
    /// Non-linear contrast power exponent.
    pub power: f32,
    /// Total number of hemisphere samples evaluated per fragment.
    pub kernel_size: usize,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            sample_radius: 0.5,
            bias: 0.025,
            intensity: 1.0,
            power: 1.5,
            kernel_size: 32,
        }
    }
}

/// Generates a tangent-space hemisphere sample kernel with quadratic origin falloff.
///
/// Samples are concentrated closer to the origin to capture precise micro-contact shadows:
/// $$\text{scale}_i = \text{lerp}(0.1, 1.0, (i / N)^2)$$
pub fn generate_ssao_kernel(kernel_size: usize) -> Vec<Vec3> {
    let mut kernel = Vec::with_capacity(kernel_size);
    let pi2 = std::f32::consts::PI * 2.0;

    for i in 0..kernel_size {
        // Deterministic low-discrepancy Hammersley distribution
        let u1 = (i as f32 + 0.5) / (kernel_size as f32);
        let mut u2 = 0.0f32;
        let mut bit = 0.5f32;
        let mut n = i;
        while n > 0 {
            if n & 1 == 1 {
                u2 += bit;
            }
            n >>= 1;
            bit *= 0.5;
        }

        let phi = u1 * pi2;
        let cos_theta = (1.0 - u2).sqrt();
        let sin_theta = u2.sqrt();

        let x = phi.cos() * sin_theta;
        let y = phi.sin() * sin_theta;
        let z = cos_theta.max(0.001); // strictly in upper hemisphere

        let dir = Vec3::new(x, y, z).normalize();

        // Quadratic scale factor to cluster samples near origin
        let fraction = (i as f32) / (kernel_size as f32);
        let scale = 0.1 + 0.9 * (fraction * fraction);

        kernel.push(dir * scale);
    }

    kernel
}

/// Generates a 4x4 tiled pseudo-random rotation vector table for SSAO kernel jittering.
///
/// Returns 16 RGBA float vectors containing $(\cos\theta, \sin\theta, 0.0, 0.0)$.
pub fn generate_ssao_noise_table() -> [[f32; 4]; 16] {
    let mut noise = [[0.0; 4]; 16];
    let pi2 = std::f32::consts::PI * 2.0;

    for (i, slot) in noise.iter_mut().enumerate() {
        // High-frequency pseudo-random angle
        let angle = ((i as f32 * 0.618_034) % 1.0) * pi2;
        *slot = [angle.cos(), angle.sin(), 0.0, 0.0];
    }

    noise
}

/// Computes depth similarity weight for bilateral edge-preserving SSAO blur.
///
/// Rejects occlusion bleeding across geometric silhouette edges and depth discontinuities:
/// $$w = \exp\left(-\frac{|z_{center} - z_{sample}|}{\sigma_z}\right)$$
#[inline]
pub fn compute_bilateral_depth_weight(
    center_depth: f32,
    sample_depth: f32,
    depth_sigma: f32,
) -> f32 {
    let diff = (center_depth - sample_depth).abs();
    let sigma = depth_sigma.max(1e-4);
    (-diff / sigma).exp()
}
