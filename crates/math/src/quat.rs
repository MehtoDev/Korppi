//! 3D orientation quaternion (`Quat`).
//!
//! # Inputs / Outputs
//! Provides a right-handed Hamilton quaternion `(x, y, z, w)` where `(x, y, z)` is the vector
//! imaginary part and `w` is the scalar real part (`Quat::IDENTITY = (0, 0, 0, 1)`).

use crate::{Mat3, Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, MulAssign, Neg, Sub};

/// A right-handed quaternion representing 3D orientation `(x, y, z, w)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    /// Vector X imaginary component.
    pub x: f32,
    /// Vector Y imaginary component.
    pub y: f32,
    /// Vector Z imaginary component.
    pub z: f32,
    /// Scalar W real component.
    pub w: f32,
}

impl Default for Quat {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    /// Identity quaternion `(0.0, 0.0, 0.0, 1.0)` representing no rotation.
    pub const IDENTITY: Self = Self::from_xyzw(0.0, 0.0, 0.0, 1.0);
    /// All `f32::NAN`.
    pub const NAN: Self = Self::from_xyzw(f32::NAN, f32::NAN, f32::NAN, f32::NAN);

    /// Creates a quaternion from explicit `(x, y, z, w)` components without normalizing.
    #[inline(always)]
    #[must_use]
    pub const fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a quaternion from `[x, y, z, w]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::from_xyzw(a[0], a[1], a[2], a[3])
    }

    /// Converts this quaternion into `[x, y, z, w]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Creates a quaternion from a [`Vec4`] `(x, y, z, w)`.
    #[inline(always)]
    #[must_use]
    pub const fn from_vec4(v: Vec4) -> Self {
        Self::from_xyzw(v.x, v.y, v.z, v.w)
    }

    /// Creates a unit quaternion representing a rotation of `angle` radians around unit `axis`.
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        let v = axis * s;
        Self::from_xyzw(v.x, v.y, v.z, c)
    }

    /// Creates a unit quaternion for a rotation of `angle` radians around the +X axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::from_xyzw(s, 0.0, 0.0, c)
    }

    /// Creates a unit quaternion for a rotation of `angle` radians around the +Y axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::from_xyzw(0.0, s, 0.0, c)
    }

    /// Creates a unit quaternion for a rotation of `angle` radians around the +Z axis.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::from_xyzw(0.0, 0.0, s, c)
    }

    /// Computes the minimal shortest-arc rotation transforming unit vector `from` into unit vector `to`.
    #[inline]
    #[must_use]
    pub fn from_rotation_arc(from: Vec3, to: Vec3) -> Self {
        const ONE_MINUS_EPS: f32 = 1.0 - 2.0 * f32::EPSILON;
        let dot = from.dot(to);
        if dot > ONE_MINUS_EPS {
            Self::IDENTITY
        } else if dot < -ONE_MINUS_EPS {
            Self::from_axis_angle(from.any_orthonormal_vector(), std::f32::consts::PI)
        } else {
            let c = from.cross(to);
            Self::from_xyzw(c.x, c.y, c.z, 1.0 + dot).normalize()
        }
    }

    /// Extracts a unit quaternion from the orthonormal columns `(x_axis, y_axis, z_axis)` of a rotation matrix.
    #[inline]
    #[must_use]
    pub(crate) fn from_rotation_axes(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        let (m00, m01, m02) = (x_axis.x, x_axis.y, x_axis.z);
        let (m10, m11, m12) = (y_axis.x, y_axis.y, y_axis.z);
        let (m20, m21, m22) = (z_axis.x, z_axis.y, z_axis.z);
        if m22 <= 0.0 {
            let dif10 = m11 - m00;
            let omm22 = 1.0 - m22;
            if dif10 <= 0.0 {
                let four_xsq = omm22 - dif10;
                let inv4x = 0.5 / four_xsq.sqrt();
                Self::from_xyzw(
                    four_xsq * inv4x,
                    (m01 + m10) * inv4x,
                    (m02 + m20) * inv4x,
                    (m12 - m21) * inv4x,
                )
            } else {
                let four_ysq = omm22 + dif10;
                let inv4y = 0.5 / four_ysq.sqrt();
                Self::from_xyzw(
                    (m01 + m10) * inv4y,
                    four_ysq * inv4y,
                    (m12 + m21) * inv4y,
                    (m20 - m02) * inv4y,
                )
            }
        } else {
            let sum10 = m11 + m00;
            let opm22 = 1.0 + m22;
            if sum10 <= 0.0 {
                let four_zsq = opm22 - sum10;
                let inv4z = 0.5 / four_zsq.sqrt();
                Self::from_xyzw(
                    (m02 + m20) * inv4z,
                    (m12 + m21) * inv4z,
                    four_zsq * inv4z,
                    (m01 - m10) * inv4z,
                )
            } else {
                let four_wsq = opm22 + sum10;
                let inv4w = 0.5 / four_wsq.sqrt();
                Self::from_xyzw(
                    (m12 - m21) * inv4w,
                    (m20 - m02) * inv4w,
                    (m01 - m10) * inv4w,
                    four_wsq * inv4w,
                )
            }
        }
    }

    /// Creates a quaternion from a 3x3 rotation matrix [`Mat3`].
    #[inline]
    #[must_use]
    pub fn from_mat3(mat: &Mat3) -> Self {
        Self::from_rotation_axes(mat.x_axis, mat.y_axis, mat.z_axis)
    }

    /// Creates a quaternion from the upper-left 3x3 rotation matrix of a [`Mat4`].
    #[inline]
    #[must_use]
    pub fn from_mat4(mat: &Mat4) -> Self {
        Self::from_rotation_axes(
            mat.x_axis.truncate(),
            mat.y_axis.truncate(),
            mat.z_axis.truncate(),
        )
    }

    /// Returns the unit rotation axis and angle in radians `(axis, angle)`.
    #[inline]
    #[must_use]
    pub fn to_axis_angle(self) -> (Vec3, f32) {
        let v = Vec3::new(self.x, self.y, self.z);
        let length = v.length();
        if length >= 1.0e-8 {
            let angle = 2.0 * length.atan2(self.w);
            (v / length, angle)
        } else {
            (Vec3::X, 0.0)
        }
    }

    /// Computes the 4D dot product of two quaternions.
    #[inline(always)]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z) + (self.w * rhs.w)
    }

    /// Computes the squared norm `self.dot(self)`.
    #[inline(always)]
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Computes the norm `self.length_squared().sqrt()`.
    #[inline(always)]
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Computes `1.0 / self.length()`.
    #[inline(always)]
    #[must_use]
    pub fn length_recip(self) -> f32 {
        self.length().recip()
    }

    /// Normalizes this quaternion to unit length.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let inv = self.length_recip();
        Self::from_xyzw(self.x * inv, self.y * inv, self.z * inv, self.w * inv)
    }

    /// Returns `true` if `self` is unit length within tolerance `1e-4`.
    #[inline]
    #[must_use]
    pub fn is_normalized(self) -> bool {
        (self.length_squared() - 1.0).abs() <= 2e-4
    }

    /// Returns the quaternion conjugate `(-x, -y, -z, w)`.
    #[inline(always)]
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self::from_xyzw(-self.x, -self.y, -self.z, self.w)
    }

    /// Returns the inverse of a unit quaternion (its conjugate).
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        self.conjugate()
    }

    /// Normalized linear interpolation between `self` and `end` along the shortest path.
    #[inline]
    #[must_use]
    pub fn lerp(self, end: Self, s: f32) -> Self {
        let end = if self.dot(end) < 0.0 { -end } else { end };
        Self::from_xyzw(
            self.x + (end.x - self.x) * s,
            self.y + (end.y - self.y) * s,
            self.z + (end.z - self.z) * s,
            self.w + (end.w - self.w) * s,
        )
        .normalize()
    }

    /// Spherical linear interpolation between `self` and `end` along the shortest path.
    #[inline]
    #[must_use]
    pub fn slerp(self, mut end: Self, s: f32) -> Self {
        let mut dot = self.dot(end);
        if dot < 0.0 {
            end = -end;
            dot = -dot;
        }
        if dot > 0.9995 {
            return self.lerp(end, s);
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let s0 = ((1.0 - s) * theta).sin() / sin_theta;
        let s1 = (s * theta).sin() / sin_theta;
        Self::from_xyzw(
            self.x * s0 + end.x * s1,
            self.y * s0 + end.y * s1,
            self.z * s0 + end.z * s1,
            self.w * s0 + end.w * s1,
        )
    }

    /// Rotates a 3D vector by this unit quaternion.
    #[inline]
    #[must_use]
    pub fn mul_vec3(self, rhs: Vec3) -> Vec3 {
        let w = self.w;
        let b = Vec3::new(self.x, self.y, self.z);
        let b2 = b.dot(b);
        rhs * (w * w - b2) + b * (rhs.dot(b) * 2.0) + b.cross(rhs) * (w * 2.0)
    }

    /// Multiplies two quaternions using the Hamilton product.
    #[inline]
    #[must_use]
    pub fn mul_quat(self, rhs: Self) -> Self {
        let (x0, y0, z0, w0) = (self.x, self.y, self.z, self.w);
        let (x1, y1, z1, w1) = (rhs.x, rhs.y, rhs.z, rhs.w);
        Self::from_xyzw(
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        )
    }

    /// Returns `true` if all components are finite.
    #[inline(always)]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// Returns `true` if any component is `NaN`.
    #[inline(always)]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    /// Returns `true` if `|self - rhs| <= max_abs_diff` for every component.
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        (self.x - rhs.x).abs() <= max_abs_diff
            && (self.y - rhs.y).abs() <= max_abs_diff
            && (self.z - rhs.z).abs() <= max_abs_diff
            && (self.w - rhs.w).abs() <= max_abs_diff
    }
}

impl fmt::Display for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}

impl Neg for Quat {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_xyzw(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Add for Quat {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_xyzw(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Sub for Quat {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_xyzw(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl Mul for Quat {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul_quat(rhs)
    }
}

impl MulAssign for Quat {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_quat(rhs);
    }
}

impl Mul<Vec3> for Quat {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Vec3 {
        self.mul_vec3(rhs)
    }
}

impl Mul<f32> for Quat {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::from_xyzw(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Div<f32> for Quat {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::from_xyzw(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl From<[f32; 4]> for Quat {
    #[inline(always)]
    fn from(a: [f32; 4]) -> Self {
        Self::from_array(a)
    }
}

impl From<Quat> for [f32; 4] {
    #[inline(always)]
    fn from(q: Quat) -> Self {
        q.to_array()
    }
}

impl From<(f32, f32, f32, f32)> for Quat {
    #[inline(always)]
    fn from((x, y, z, w): (f32, f32, f32, f32)) -> Self {
        Self::from_xyzw(x, y, z, w)
    }
}

impl From<Quat> for (f32, f32, f32, f32) {
    #[inline(always)]
    fn from(q: Quat) -> Self {
        (q.x, q.y, q.z, q.w)
    }
}
