//! Psychoacoustic binaural spatialization cues (ITD & ILD).

use std::f32::consts::{FRAC_PI_2, PI};

/// Woodworth spherical head model for computing Interaural Time Difference (ITD)
/// and contralateral head-shadow acoustic filtering.
#[derive(Debug, Clone, Copy)]
pub struct WoodworthBinaural {
    /// Average human head radius in meters (default 0.0875 m).
    pub head_radius: f32,
    /// Speed of sound in air at 20 degrees Celsius in meters per second (default 343.0 m/s).
    pub speed_of_sound: f32,
}

impl Default for WoodworthBinaural {
    fn default() -> Self {
        Self {
            head_radius: 0.0875,
            speed_of_sound: 343.0,
        }
    }
}

impl WoodworthBinaural {
    /// Creates a new Woodworth binaural model with custom physical parameters.
    #[must_use]
    pub const fn new(head_radius: f32, speed_of_sound: f32) -> Self {
        Self {
            head_radius,
            speed_of_sound,
        }
    }

    /// Computes the Interaural Time Difference (ITD) in seconds for left and right ears.
    ///
    /// Returns `(delay_left_sec, delay_right_sec)`. The ipsilateral (closer) ear has 0.0 delay,
    /// while the contralateral (farther) ear has a positive propagation delay up to ~0.655 ms.
    #[must_use]
    pub fn calculate_itd_seconds(&self, azimuth_rad: f32) -> (f32, f32) {
        if azimuth_rad.abs() < 1e-5 {
            return (0.0, 0.0);
        }

        let abs_theta = azimuth_rad.abs().min(PI);
        let delta_t = if abs_theta <= FRAC_PI_2 {
            (self.head_radius / self.speed_of_sound) * (abs_theta + abs_theta.sin())
        } else {
            (self.head_radius / self.speed_of_sound) * (PI - abs_theta + abs_theta.sin())
        };

        if azimuth_rad > 0.0 {
            // Source is to the right: left ear is farther away
            (delta_t, 0.0)
        } else {
            // Source is to the left: right ear is farther away
            (0.0, delta_t)
        }
    }

    /// Computes the head-shadow low-pass cutoff frequency in Hz for the contralateral ear.
    #[must_use]
    pub fn calculate_head_shadow_lpf(&self, azimuth_rad: f32) -> f32 {
        let lateral = azimuth_rad.sin().abs();
        // High frequencies on the contralateral ear roll off from 22 kHz down to ~2.5 kHz
        22_000.0 - lateral * 19_500.0
    }

    /// The head-shadow cutoffs of the left and right ears for a source at `azimuth_rad`
    /// (positive to the right): the far ear's from [`Self::calculate_head_shadow_lpf`], 22 kHz
    /// for the near one.
    #[must_use]
    pub fn head_shadow_cutoffs(&self, azimuth_rad: f32) -> (f32, f32) {
        let shadow = self.calculate_head_shadow_lpf(azimuth_rad);
        if azimuth_rad > 0.0 {
            (shadow, 22_000.0)
        } else if azimuth_rad < 0.0 {
            (22_000.0, shadow)
        } else {
            (22_000.0, 22_000.0)
        }
    }
}
