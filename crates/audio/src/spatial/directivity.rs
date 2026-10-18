//! Directional emitter cones and acoustic directivity patterns.

use serde::{Deserialize, Serialize};

/// Directional emission cone modeling focused sound sources (e.g. human voice, megaphones).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EmitterCone {
    /// Full inner cone angle in radians where sound is unattenuated.
    pub inner_angle_rad: f32,
    /// Full outer cone angle in radians past which sound is attenuated to `outer_gain`.
    pub outer_angle_rad: f32,
    /// Attenuation gain factor outside the outer cone (0.0 to 1.0).
    pub outer_gain: f32,
    /// Low-pass filter cutoff in Hz applied outside the outer cone.
    pub outer_lpf_cutoff: f32,
}

impl Default for EmitterCone {
    fn default() -> Self {
        Self {
            inner_angle_rad: std::f32::consts::PI * 0.5,
            outer_angle_rad: std::f32::consts::PI,
            outer_gain: 0.25,
            outer_lpf_cutoff: 2500.0,
        }
    }
}

impl EmitterCone {
    /// Calculates the directional `(gain, lpf_cutoff)` for a given off-axis angle in radians.
    #[must_use]
    pub fn calculate_attenuation(&self, off_axis_angle_rad: f32) -> (f32, f32) {
        let half_inner = self.inner_angle_rad * 0.5;
        let half_outer = self.outer_angle_rad * 0.5;
        let angle = off_axis_angle_rad.abs();

        if angle <= half_inner {
            (1.0, 22_000.0)
        } else if angle >= half_outer {
            (self.outer_gain, self.outer_lpf_cutoff)
        } else {
            let t = (angle - half_inner) / (half_outer - half_inner).max(0.001);
            let gain = 1.0 + t * (self.outer_gain - 1.0);
            let cutoff = 22_000.0 + t * (self.outer_lpf_cutoff - 22_000.0);
            (gain, cutoff)
        }
    }
}
