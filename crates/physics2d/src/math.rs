//! Box2D's vector helpers (`math_functions.h`), with its order of float operations, so that
//! ported code rounds as Box2D does. `normalize` returns zero for vectors shorter than
//! `f32::EPSILON` instead of failing.

use korppi_math::{Rot2, Vec2};
use serde::{Deserialize, Serialize};

/// Box2D `b2Cross` of two vectors.
pub(crate) fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Box2D `b2CrossSV`: scalar `s` crossed with `v`.
pub(crate) fn cross_sv(s: f32, v: Vec2) -> Vec2 {
    Vec2::new(-s * v.y, s * v.x)
}

/// Box2D `b2CrossVS`: `v` crossed with scalar `s`.
pub(crate) fn cross_vs(v: Vec2, s: f32) -> Vec2 {
    Vec2::new(s * v.y, -s * v.x)
}

/// Box2D `b2RightPerp`.
pub(crate) fn right_perp(v: Vec2) -> Vec2 {
    Vec2::new(v.y, -v.x)
}

/// Box2D `b2LeftPerp`.
pub(crate) fn left_perp(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

/// Box2D `b2Lerp`.
pub(crate) fn lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
    Vec2::new((1.0 - t) * a.x + t * b.x, (1.0 - t) * a.y + t * b.y)
}

/// Box2D `b2DistanceSquared`.
pub(crate) fn distance_squared(a: Vec2, b: Vec2) -> f32 {
    let c = Vec2::new(b.x - a.x, b.y - a.y);
    c.x * c.x + c.y * c.y
}

/// Box2D `b2Normalize`.
pub(crate) fn normalize(v: Vec2) -> Vec2 {
    length_and_normalize(v).1
}

/// Box2D `b2GetLengthAndNormalize`: the length and the unit vector.
pub(crate) fn length_and_normalize(v: Vec2) -> (f32, Vec2) {
    let length = (v.x * v.x + v.y * v.y).sqrt();
    if length < f32::EPSILON {
        return (length, Vec2::ZERO);
    }
    let inv_length = 1.0 / length;
    (length, Vec2::new(inv_length * v.x, inv_length * v.y))
}

/// Box2D `B2_PI` (the same single-precision value).
pub(crate) const PI: f32 = std::f32::consts::PI;

/// Box2D `b2ClampFloat`: `a` within `[lower, upper]`, without panicking when `lower > upper`.
pub(crate) fn clamp(a: f32, lower: f32, upper: f32) -> f32 {
    if a < lower {
        lower
    } else if a > upper {
        upper
    } else {
        a
    }
}

/// Box2D `b2Atan2`: a minimax polynomial approximation of `atan2(y, x)`, the same on every
/// platform (`f32::atan2` depends on the C library).
pub(crate) fn atan2(y: f32, x: f32) -> f32 {
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }
    let (ax, ay) = (x.abs(), y.abs());
    let mx = ay.max(ax);
    let mn = ay.min(ax);
    let a = mn / mx;
    // Minimax polynomial approximation to atan(a) on [0, 1].
    let s = a * a;
    let c = s * a;
    let q = s * s;
    let mut r = 0.024_840_285 * q + 0.186_814_18;
    let t = -0.094_097_95 * q - 0.332_130_72;
    r = r * s + t;
    r = r * c + a;
    // Map to the full circle.
    if ay > ax {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = PI - r;
    }
    if y < 0.0 {
        r = -r;
    }
    r
}

/// Box2D `b2Rot_GetAngle`.
pub(crate) fn rot_angle(q: Rot2) -> f32 {
    atan2(q.s, q.c)
}

/// Box2D `b2InvMulRot`: `inv(a) * b`.
pub(crate) fn inv_mul_rot(a: Rot2, b: Rot2) -> Rot2 {
    Rot2::new(a.c * b.c + a.s * b.s, a.c * b.s - a.s * b.c)
}

/// Box2D `b2RelativeAngle`: the angle of `b` relative to `a`.
pub(crate) fn relative_angle(a: Rot2, b: Rot2) -> f32 {
    rot_angle(inv_mul_rot(a, b))
}

/// Box2D `b2UnwindAngle`: `radians` in `[-π, π]`, rounding to the nearest turn in double
/// precision (ties to even) with Box2D's single-precision 2π.
pub(crate) fn unwind_angle(radians: f32) -> f32 {
    let x = f64::from(clamp(radians, -1.0e6, 1.0e6));
    let two_pi = f64::from(2.0 * PI);
    let round_to_nearest = 6_755_399_441_055_744.0;
    let k = (x / two_pi + round_to_nearest) - round_to_nearest;
    (x - k * two_pi) as f32
}

/// Box2D `b2SpringDamper`: the velocity of a damped spring after `time_step`.
pub(crate) fn spring_damper(
    hertz: f32,
    damping_ratio: f32,
    position: f32,
    velocity: f32,
    time_step: f32,
) -> f32 {
    let omega = 2.0 * PI * hertz;
    let omega_h = omega * time_step;
    (velocity - omega * omega_h * position)
        / (1.0 + 2.0 * damping_ratio * omega_h + omega_h * omega_h)
}

/// Box2D `b2Mat22`, by columns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Mat22 {
    pub cx: Vec2,
    pub cy: Vec2,
}

impl Mat22 {
    /// Box2D `b2GetInverse22`; the zero matrix stays zero.
    pub(crate) fn inverse(self) -> Self {
        let (a, b, c, d) = (self.cx.x, self.cy.x, self.cx.y, self.cy.y);
        let mut det = a * d - b * c;
        if det != 0.0 {
            det = 1.0 / det;
        }
        Self {
            cx: Vec2::new(det * d, -det * c),
            cy: Vec2::new(-det * b, det * a),
        }
    }

    /// Box2D `b2MulMV`.
    pub(crate) fn mul_vec(self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.cx.x * v.x + self.cy.x * v.y,
            self.cx.y * v.x + self.cy.y * v.y,
        )
    }

    /// Box2D `b2Solve22`: `x` with `self * x = b`, zero for a singular matrix.
    pub(crate) fn solve(self, b: Vec2) -> Vec2 {
        let (a11, a12, a21, a22) = (self.cx.x, self.cy.x, self.cx.y, self.cy.y);
        let mut det = a11 * a22 - a12 * a21;
        if det != 0.0 {
            det = 1.0 / det;
        }
        Vec2::new(det * (a22 * b.x - a12 * b.y), det * (a11 * b.y - a21 * b.x))
    }
}
