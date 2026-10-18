//! Audio frame representation and arithmetic for multichannel mixing.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// A stereo audio frame containing left and right floating-point PCM samples.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[repr(C)]
pub struct AudioFrame {
    /// Left audio channel sample (-1.0 to 1.0).
    pub left: f32,
    /// Right audio channel sample (-1.0 to 1.0).
    pub right: f32,
}

impl AudioFrame {
    /// Absolute silence (both channels at 0.0).
    pub const SILENT: Self = Self {
        left: 0.0,
        right: 0.0,
    };

    /// Absolute silence constant alias.
    pub const SILENCE: Self = Self::SILENT;

    /// Creates a new stereo frame with explicit left and right channel samples.
    #[must_use]
    pub const fn new(left: f32, right: f32) -> Self {
        Self { left, right }
    }

    /// Creates a stereo frame from a single mono sample (identical in both channels).
    #[must_use]
    pub const fn from_mono(sample: f32) -> Self {
        Self {
            left: sample,
            right: sample,
        }
    }

    /// Applies equal-power stereo panning to this frame.
    ///
    /// `pan` must be in the range `[-1.0, 1.0]`:
    /// - `-1.0`: Hard left (Left = 1.0, Right = 0.0)
    /// - `0.0`: Center (Left = ~0.7071, Right = ~0.7071)
    /// - `1.0`: Hard right (Left = 0.0, Right = 1.0)
    #[must_use]
    pub fn pan(self, pan: f32) -> Self {
        let clamped = pan.clamp(-1.0, 1.0);
        let theta = (clamped + 1.0) * 0.5 * (std::f32::consts::FRAC_PI_2);
        let gain_left = theta.cos();
        let gain_right = theta.sin();

        Self {
            left: self.left * gain_left,
            right: self.right * gain_right,
        }
    }

    /// Converts this stereo frame into a single mono sample by averaging channels.
    #[must_use]
    pub fn to_mono(self) -> f32 {
        (self.left + self.right) * 0.5
    }

    /// Clamps both channel values to the specified `[min, max]` range.
    #[must_use]
    pub fn clamp(self, min: f32, max: f32) -> Self {
        Self {
            left: self.left.clamp(min, max),
            right: self.right.clamp(min, max),
        }
    }
}

impl Add for AudioFrame {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            left: self.left + rhs.left,
            right: self.right + rhs.right,
        }
    }
}

impl AddAssign for AudioFrame {
    fn add_assign(&mut self, rhs: Self) {
        self.left += rhs.left;
        self.right += rhs.right;
    }
}

impl Sub for AudioFrame {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            left: self.left - rhs.left,
            right: self.right - rhs.right,
        }
    }
}

impl SubAssign for AudioFrame {
    fn sub_assign(&mut self, rhs: Self) {
        self.left -= rhs.left;
        self.right -= rhs.right;
    }
}

impl Mul<f32> for AudioFrame {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            left: self.left * rhs,
            right: self.right * rhs,
        }
    }
}

impl MulAssign<f32> for AudioFrame {
    fn mul_assign(&mut self, rhs: f32) {
        self.left *= rhs;
        self.right *= rhs;
    }
}

impl Div<f32> for AudioFrame {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let inv = 1.0 / rhs;
        Self {
            left: self.left * inv,
            right: self.right * inv,
        }
    }
}

impl DivAssign<f32> for AudioFrame {
    fn div_assign(&mut self, rhs: f32) {
        let inv = 1.0 / rhs;
        self.left *= inv;
        self.right *= inv;
    }
}
