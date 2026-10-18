//! Stochastic Monte Carlo geometric acoustics path tracer with Next Event Estimation (NEE).

use super::geometry::{AcousticRay, AcousticScene};
use korppi_math::Vec3;
use std::f32::consts::PI;

/// Speed of sound in dry air at 20°C in meters per second.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Atmospheric air absorption attenuation coefficients $\gamma_{\text{air}}(b)$ per meter:
/// Low (500 Hz), Mid (1 kHz), High (8 kHz).
pub const AIR_ABSORPTION_COEFFS: [f32; 3] = [0.0001, 0.001, 0.015];

/// Golden angle constant: $\pi (3 - \sqrt{5}) \approx 2.39996323\text{ rad}$.
pub const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// Generates $N$ stratified, uniformly distributed ray direction vectors on the unit sphere
/// using the low-discrepancy Fibonacci spherical lattice.
#[must_use]
pub fn fibonacci_sphere_rays(count: usize) -> Vec<Vec3> {
    if count == 0 {
        return Vec::new();
    }

    let mut rays = Vec::with_capacity(count);
    let n = count as f32;

    for i in 0..count {
        let i_f = i as f32;
        // z varies uniformly from 1 - 1/n to -1 + 1/n
        let z = 1.0 - (2.0 * i_f + 1.0) / n;
        let radius = (1.0 - z * z).max(0.0).sqrt();
        let phi = i_f * GOLDEN_ANGLE;

        let x = radius * phi.cos();
        let y = radius * phi.sin();

        rays.push(Vec3::new(x, y, z).normalize_or_zero());
    }

    rays
}

/// Configuration settings for the acoustic path tracer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathTraceConfig {
    /// Total number of stochastic rays emitted uniformly per simulation step (e.g. 64 to 256).
    pub ray_count: usize,
    /// Maximum reflection bounce depth before terminating ray paths.
    pub max_bounces: usize,
    /// Minimum energy threshold below which paths are terminated.
    pub energy_threshold: f32,
}

impl Default for PathTraceConfig {
    fn default() -> Self {
        Self {
            ray_count: 128,
            max_bounces: 4,
            energy_threshold: 1e-4,
        }
    }
}

/// Discrete acoustic arrival event reaching the listener from a reflection path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyReflectionHit {
    /// Time delay in seconds from sound emission to arrival at listener.
    pub arrival_time_sec: f32,
    /// 3-band acoustic energy received $[E_{\text{low}}, E_{\text{mid}}, E_{\text{high}}]$.
    pub energy: [f32; 3],
    /// Direction of arrival at listener (unit vector pointing towards reflection source).
    pub direction: Vec3,
}

/// Stochastic Monte Carlo acoustic path tracer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaytracedAcoustics {
    /// Path tracer configuration.
    pub config: PathTraceConfig,
}

impl RaytracedAcoustics {
    /// Creates a new acoustic path tracer with specified configuration.
    #[must_use]
    pub const fn new(config: PathTraceConfig) -> Self {
        Self { config }
    }

    /// Traces stochastic paths and gathers discrete early reflections reaching the listener
    /// using Next Event Estimation (NEE) shadow rays.
    #[must_use]
    pub fn trace_early_reflections(
        &self,
        scene: &AcousticScene,
        emitter: Vec3,
        listener: Vec3,
    ) -> Vec<EarlyReflectionHit> {
        let count = self.config.ray_count;
        if count == 0 {
            return Vec::new();
        }

        let rays = fibonacci_sphere_rays(count);
        let initial_energy = 1.0 / count as f32;
        let mut reflections = Vec::new();

        for (ray_idx, &ray_dir) in rays.iter().enumerate() {
            let mut current_origin = emitter;
            let mut current_dir = ray_dir;
            let mut current_energy = [initial_energy; 3];
            let mut path_distance = 0.0f32;

            for bounce in 0..self.config.max_bounces {
                let ray = AcousticRay::new(current_origin, current_dir);
                let Some(hit) = scene.raycast_closest(&ray) else {
                    break;
                };

                let dist = hit.distance;
                path_distance += dist;

                // 1. Atmospheric air absorption along path segment
                current_energy[0] *= (-AIR_ABSORPTION_COEFFS[0] * dist).exp();
                current_energy[1] *= (-AIR_ABSORPTION_COEFFS[1] * dist).exp();
                current_energy[2] *= (-AIR_ABSORPTION_COEFFS[2] * dist).exp();

                // 2. Surface absorption
                current_energy[0] *= (1.0 - hit.material.absorption[0]).max(0.0);
                current_energy[1] *= (1.0 - hit.material.absorption[1]).max(0.0);
                current_energy[2] *= (1.0 - hit.material.absorption[2]).max(0.0);

                if current_energy[0] < self.config.energy_threshold
                    && current_energy[1] < self.config.energy_threshold
                    && current_energy[2] < self.config.energy_threshold
                {
                    break;
                }

                // 3. Next Event Estimation (NEE): direct shadow ray to listener
                let shadow_vec = listener - hit.point;
                let shadow_dist = shadow_vec.length();

                if shadow_dist > 1e-4 {
                    let shadow_dir = shadow_vec / shadow_dist;
                    let n_dot_l = hit.normal.dot(shadow_dir).max(0.0);

                    if n_dot_l > 0.0 {
                        // Cast shadow ray slightly offset from surface
                        let shadow_ray =
                            AcousticRay::new(hit.point + hit.normal * 1e-4, shadow_dir);
                        let transmission = scene.raycast_transmission(&shadow_ray, shadow_dist);

                        if transmission[0] > 1e-4
                            || transmission[1] > 1e-4
                            || transmission[2] > 1e-4
                        {
                            let total_distance = path_distance + shadow_dist;
                            let arrival_time = total_distance / SPEED_OF_SOUND;
                            let geom_spreading = 1.0 / (4.0 * PI * total_distance.max(1.0).powi(2));

                            let recv_low = current_energy[0]
                                * geom_spreading
                                * n_dot_l
                                * (-AIR_ABSORPTION_COEFFS[0] * shadow_dist).exp()
                                * transmission[0];
                            let recv_mid = current_energy[1]
                                * geom_spreading
                                * n_dot_l
                                * (-AIR_ABSORPTION_COEFFS[1] * shadow_dist).exp()
                                * transmission[1];
                            let recv_high = current_energy[2]
                                * geom_spreading
                                * n_dot_l
                                * (-AIR_ABSORPTION_COEFFS[2] * shadow_dist).exp()
                                * transmission[2];

                            if recv_low > 1e-8 || recv_mid > 1e-8 || recv_high > 1e-8 {
                                let arrival_dir = (hit.point - listener).normalize_or_zero();
                                reflections.push(EarlyReflectionHit {
                                    arrival_time_sec: arrival_time,
                                    energy: [recv_low, recv_mid, recv_high],
                                    direction: arrival_dir,
                                });
                            }
                        }
                    }
                }

                // 4. Compute next bounce direction (specular vs diffuse scattering)
                let spec_dir = (current_dir - 2.0 * current_dir.dot(hit.normal) * hit.normal)
                    .normalize_or_zero();
                let s = hit.material.scattering;

                let next_dir = if s > 0.001 {
                    // Generate orthogonal basis around normal
                    let tangent = if hit.normal.x.abs() > 0.9 {
                        Vec3::Y.cross(hit.normal).normalize_or_zero()
                    } else {
                        Vec3::X.cross(hit.normal).normalize_or_zero()
                    };
                    let bitangent = hit.normal.cross(tangent);

                    // Cosine-weighted angle variation based on bounce and ray index
                    let angle = ((ray_idx + bounce) as f32 + 1.0) * GOLDEN_ANGLE;
                    let diff_dir = (hit.normal * std::f32::consts::FRAC_1_SQRT_2
                        + tangent * (std::f32::consts::FRAC_1_SQRT_2 * angle.cos())
                        + bitangent * (std::f32::consts::FRAC_1_SQRT_2 * angle.sin()))
                    .normalize_or_zero();

                    ((1.0 - s) * spec_dir + s * diff_dir).normalize_or_zero()
                } else {
                    spec_dir
                };

                current_origin = hit.point + hit.normal * 1e-4;
                current_dir = next_dir;
            }
        }

        // Sort reflections chronologically by arrival time
        reflections.sort_by(|a, b| {
            a.arrival_time_sec
                .partial_cmp(&b.arrival_time_sec)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        reflections
    }
}
