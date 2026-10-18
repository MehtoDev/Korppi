//! 2D complex number rotation (cosine, sine).

use crate::Vec2;
use serde::{Deserialize, Serialize};
use std::ops::Mul;

/// A 2-dimensional rotation represented by a unit complex number (cosine and sine).
///
/// This avoids trigonometry during vector transformation:
/// `rotate(v) = (c * x - s * y, s * x + c * y)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rot2 {
    /// Cosine of the rotation angle.
    pub c: f32,
    /// Sine of the rotation angle.
    pub s: f32,
}

impl Default for Rot2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Rot2 {
    /// Identity rotation (0 radians: c = 1, s = 0).
    pub const IDENTITY: Self = Self { c: 1.0, s: 0.0 };

    /// Creates a rotation from cosine and sine components.
    #[must_use]
    pub const fn new(c: f32, s: f32) -> Self {
        Self { c, s }
    }

    /// Creates a rotation from an angle in radians.
    #[must_use]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self { c, s }
    }

    /// Returns the rotation angle in radians within `[-PI, PI]`.
    #[must_use]
    pub fn angle(self) -> f32 {
        self.s.atan2(self.c)
    }

    /// Returns the angle of `other` relative to `self` in `[-PI, PI]`: the angle of
    /// `self⁻¹ · other` (Box2D `b2RelativeAngle`).
    ///
    /// Unlike `other.angle() - self.angle()`, it does not jump by 2π when either rotation
    /// crosses ±π.
    #[must_use]
    pub fn angle_to(self, other: Self) -> f32 {
        let s = self.c * other.s - self.s * other.c;
        let c = self.c * other.c + self.s * other.s;
        s.atan2(c)
    }

    /// Advances the rotation by `delta_angle` radians to first order and renormalizes
    /// (Box2D `b2IntegrateRotation`): `q + delta_angle * perp(q)`, without sine or cosine.
    ///
    /// The turn falls short of `delta_angle` by about `delta_angle^3 / 3`, which is negligible
    /// for the small per-substep angles of a physics step.
    #[must_use]
    pub fn integrate(self, delta_angle: f32) -> Self {
        Self {
            c: self.c - delta_angle * self.s,
            s: self.s + delta_angle * self.c,
        }
        .normalize()
    }

    /// Returns the inverse rotation (conjugate: c, -s).
    #[must_use]
    pub const fn inverse(self) -> Self {
        Self {
            c: self.c,
            s: -self.s,
        }
    }

    /// Rotates a 2D vector by this rotation.
    #[must_use]
    pub fn rotate(self, v: Vec2) -> Vec2 {
        Vec2::new(self.c * v.x - self.s * v.y, self.s * v.x + self.c * v.y)
    }

    /// Rotates a 2D vector by the inverse of this rotation.
    #[must_use]
    pub fn inv_rotate(self, v: Vec2) -> Vec2 {
        Vec2::new(self.c * v.x + self.s * v.y, -self.s * v.x + self.c * v.y)
    }

    /// Returns the local X-axis (cosine, sine).
    #[must_use]
    pub const fn x_axis(self) -> Vec2 {
        Vec2::new(self.c, self.s)
    }

    /// Returns the local Y-axis (-sine, cosine).
    #[must_use]
    pub const fn y_axis(self) -> Vec2 {
        Vec2::new(-self.s, self.c)
    }

    /// Normalizes this rotation to ensure unit length.
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = (self.c * self.c + self.s * self.s).sqrt();
        if len > 0.0 {
            let inv_len = 1.0 / len;
            Self {
                c: self.c * inv_len,
                s: self.s * inv_len,
            }
        } else {
            Self::IDENTITY
        }
    }
}

/// Wraps an angle in radians into `[-PI, PI]` (Box2D `b2UnwindAngle`).
///
/// Works in `f64`, so angles up to about 1e6 radians keep their precision.
#[must_use]
pub fn unwind_angle(radians: f32) -> f32 {
    let x = f64::from(radians);
    let turns = (x / std::f64::consts::TAU).round();
    (x - turns * std::f64::consts::TAU) as f32
}

impl Mul for Rot2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            c: self.c * rhs.c - self.s * rhs.s,
            s: self.s * rhs.c + self.c * rhs.s,
        }
    }
}
