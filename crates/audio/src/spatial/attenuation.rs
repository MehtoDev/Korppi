//! Distance attenuation models for 3D sound emitters.

use serde::{Deserialize, Serialize};

/// Distance attenuation models for 3D spatial sound emitters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceAttenuation {
    /// Inverse distance model with rolloff factor: G(d) = d_min / (d_min + rolloff * (d - d_min)).
    InverseClamped {
        /// Distance under which sound is played at unity gain (1.0).
        min_distance: f32,
        /// Maximum cutoff distance past which sound is completely silent (0.0).
        max_distance: f32,
        /// Attenuation curve rolloff slope (default 1.0).
        rolloff: f32,
    },

    /// Linear attenuation model: G(d) = 1.0 - (d - d_min) / (d_max - d_min).
    Linear {
        /// Distance under which sound is played at unity gain (1.0).
        min_distance: f32,
        /// Distance at which sound drops to silence (0.0).
        max_distance: f32,
    },

    /// Exponential distance attenuation model: G(d) = (d / d_min)^(-rolloff).
    Exponential {
        /// Minimum reference distance.
        min_distance: f32,
        /// Maximum cutoff distance.
        max_distance: f32,
        /// Exponential decay power.
        rolloff: f32,
    },
}

/// Alias for [`DistanceAttenuation`].
pub type AttenuationModel = DistanceAttenuation;

impl Default for DistanceAttenuation {
    fn default() -> Self {
        Self::InverseClamped {
            min_distance: 1.0,
            max_distance: 100.0,
            rolloff: 1.0,
        }
    }
}

impl DistanceAttenuation {
    /// Calculates the linear gain multiplier for an emitter at the given distance in meters.
    #[must_use]
    pub fn calculate_gain(&self, distance: f32) -> f32 {
        let d = distance.max(0.0);

        match *self {
            Self::InverseClamped {
                min_distance,
                max_distance,
                rolloff,
            } => {
                if d <= min_distance {
                    1.0
                } else if d >= max_distance {
                    0.0
                } else {
                    let min_dist = min_distance.max(0.001);
                    min_dist / (min_dist + rolloff * (d - min_dist))
                }
            }
            Self::Linear {
                min_distance,
                max_distance,
            } => {
                if d <= min_distance {
                    1.0
                } else if d >= max_distance {
                    0.0
                } else {
                    let range = (max_distance - min_distance).max(0.001);
                    (1.0 - (d - min_distance) / range).clamp(0.0, 1.0)
                }
            }
            Self::Exponential {
                min_distance,
                max_distance,
                rolloff,
            } => {
                if d <= min_distance {
                    1.0
                } else if d >= max_distance {
                    0.0
                } else {
                    let min_dist = min_distance.max(0.001);
                    (d / min_dist).powf(-rolloff)
                }
            }
        }
    }
}
