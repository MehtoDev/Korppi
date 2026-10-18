//! Equal-power stereo panning and multichannel speaker distribution.

use std::f32::consts::FRAC_PI_2;

/// Equal-power stereo panner preserving total perceived acoustic loudness (G_L^2 + G_R^2 = 1.0).
#[derive(Debug, Clone, Copy, Default)]
pub struct EqualPowerPanner;

impl EqualPowerPanner {
    /// Creates a new equal-power stereo panner.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Calculates normalized stereo gains `(left_gain, right_gain)` for a given azimuth angle in radians.
    ///
    /// - `0.0` rad: Front center (Left = ~0.7071, Right = ~0.7071)
    /// - `-PI/2` rad: Hard left (Left = 1.0, Right = 0.0)
    /// - `+PI/2` rad: Hard right (Left = 0.0, Right = 1.0)
    #[must_use]
    pub fn calculate_gains(&self, azimuth_rad: f32) -> (f32, f32) {
        // Lateral projection p in [-1.0, 1.0]
        let p = azimuth_rad.sin().clamp(-1.0, 1.0);
        let pan_angle = (p + 1.0) * 0.5 * FRAC_PI_2;

        let left_gain = pan_angle.cos();
        let right_gain = pan_angle.sin();

        (left_gain, right_gain)
    }
}
