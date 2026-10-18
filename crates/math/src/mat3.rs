//! 3x3 column-major single-precision matrix (`Mat3`).
//!
//! # Inputs / Outputs
//! Provides a contiguous `#[repr(C)]` 3x3 column-major matrix (`x_axis`, `y_axis`, `z_axis`)
//! for 3D rotations, inertia tensors, normal transforms, determinants, and matrix inversion.

use crate::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 3x3 column-major matrix of `f32` elements.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mat3 {
    /// Column 0 (`x` basis vector).
    pub x_axis: Vec3,
    /// Column 1 (`y` basis vector).
    pub y_axis: Vec3,
    /// Column 2 (`z` basis vector).
    pub z_axis: Vec3,
}

impl Default for Mat3 {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat3 {
    /// All zeroes.
    pub const ZERO: Self = Self::from_cols(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    /// Identity matrix with `1.0` on the main diagonal.
    pub const IDENTITY: Self = Self::from_cols(Vec3::X, Vec3::Y, Vec3::Z);
    /// All `f32::NAN`.
    pub const NAN: Self = Self::from_cols(Vec3::NAN, Vec3::NAN, Vec3::NAN);

    /// Creates a 3x3 matrix from its three column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    /// Creates a 3x3 matrix from a flat column-major 9-element array.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_array(m: &[f32; 9]) -> Self {
        Self::from_cols(
            Vec3::new(m[0], m[1], m[2]),
            Vec3::new(m[3], m[4], m[5]),
            Vec3::new(m[6], m[7], m[8]),
        )
    }

    /// Converts this matrix into a flat column-major 9-element array.
    #[inline(always)]
    #[must_use]
    pub const fn to_cols_array(&self) -> [f32; 9] {
        [
            self.x_axis.x,
            self.x_axis.y,
            self.x_axis.z,
            self.y_axis.x,
            self.y_axis.y,
            self.y_axis.z,
            self.z_axis.x,
            self.z_axis.y,
            self.z_axis.z,
        ]
    }

    /// Creates a 3x3 matrix from a 2D array of column vectors `[[f32; 3]; 3]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[f32; 3]; 3]) -> Self {
        Self::from_cols(
            Vec3::from_array(m[0]),
            Vec3::from_array(m[1]),
            Vec3::from_array(m[2]),
        )
    }

    /// Converts this matrix into a 2D array of column vectors `[[f32; 3]; 3]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[f32; 3]; 3] {
        [
            self.x_axis.to_array(),
            self.y_axis.to_array(),
            self.z_axis.to_array(),
        ]
    }

    /// Creates a diagonal 3x3 matrix with `diagonal` along the main diagonal.
    #[inline(always)]
    #[must_use]
    pub const fn from_diagonal(diagonal: Vec3) -> Self {
        Self::from_cols(
            Vec3::new(diagonal.x, 0.0, 0.0),
            Vec3::new(0.0, diagonal.y, 0.0),
            Vec3::new(0.0, 0.0, diagonal.z),
        )
    }

    /// Creates a 3x3 rotation matrix from a unit quaternion [`Quat`].
    #[inline]
    #[must_use]
    pub fn from_quat(rotation: Quat) -> Self {
        let x2 = rotation.x + rotation.x;
        let y2 = rotation.y + rotation.y;
        let z2 = rotation.z + rotation.z;
        let xx = rotation.x * x2;
        let xy = rotation.x * y2;
        let xz = rotation.x * z2;
        let yy = rotation.y * y2;
        let yz = rotation.y * z2;
        let zz = rotation.z * z2;
        let wx = rotation.w * x2;
        let wy = rotation.w * y2;
        let wz = rotation.w * z2;

        Self::from_cols(
            Vec3::new(1.0 - (yy + zz), xy + wz, xz - wy),
            Vec3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
            Vec3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
        )
    }

    /// Creates a 3x3 rotation matrix around unit `axis` by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        Self::from_quat(Quat::from_axis_angle(axis, angle))
    }

    /// Creates a 3x3 rotation matrix around the +X axis by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(Vec3::X, Vec3::new(0.0, c, s), Vec3::new(0.0, -s, c))
    }

    /// Creates a 3x3 rotation matrix around the +Y axis by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(Vec3::new(c, 0.0, -s), Vec3::Y, Vec3::new(s, 0.0, c))
    }

    /// Creates a 3x3 rotation matrix around the +Z axis by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(Vec3::new(c, s, 0.0), Vec3::new(-s, c, 0.0), Vec3::Z)
    }

    /// Extracts the upper-left 3x3 submatrix of a [`Mat4`].
    #[inline(always)]
    #[must_use]
    pub const fn from_mat4(m: Mat4) -> Self {
        Self::from_cols(
            m.x_axis.truncate(),
            m.y_axis.truncate(),
            m.z_axis.truncate(),
        )
    }

    /// Returns column `index` (`0..3`).
    ///
    /// # Panics
    /// Panics if `index >= 3`.
    #[inline(always)]
    #[must_use]
    pub const fn col(&self, index: usize) -> Vec3 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            _ => panic!("Mat3 column index out of bounds"),
        }
    }

    /// Returns row `index` (`0..3`).
    ///
    /// # Panics
    /// Panics if `index >= 3`.
    #[inline(always)]
    #[must_use]
    pub const fn row(&self, index: usize) -> Vec3 {
        match index {
            0 => Vec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            1 => Vec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            2 => Vec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
            _ => panic!("Mat3 row index out of bounds"),
        }
    }

    /// Returns the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub const fn transpose(&self) -> Self {
        Self::from_cols(self.row(0), self.row(1), self.row(2))
    }

    /// Computes the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
    }

    /// Computes the inverse of `self`.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let tmp0 = self.y_axis.cross(self.z_axis);
        let tmp1 = self.z_axis.cross(self.x_axis);
        let tmp2 = self.x_axis.cross(self.y_axis);
        let det = self.z_axis.dot(tmp2);
        let inv_det = Vec3::splat(det.recip());
        Self::from_cols(tmp0 * inv_det, tmp1 * inv_det, tmp2 * inv_det).transpose()
    }

    /// Multiplies this 3x3 matrix by a 3D column vector `rhs`.
    #[inline(always)]
    #[must_use]
    pub fn mul_vec3(&self, rhs: Vec3) -> Vec3 {
        (self.x_axis * rhs.x) + (self.y_axis * rhs.y) + (self.z_axis * rhs.z)
    }

    /// Multiplies two 3x3 matrices `self * rhs`.
    #[inline]
    #[must_use]
    pub fn mul_mat3(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.mul_vec3(rhs.x_axis),
            self.mul_vec3(rhs.y_axis),
            self.mul_vec3(rhs.z_axis),
        )
    }

    /// Multiplies every element by scalar `rhs`.
    #[inline(always)]
    #[must_use]
    pub fn mul_scalar(&self, rhs: f32) -> Self {
        Self::from_cols(self.x_axis * rhs, self.y_axis * rhs, self.z_axis * rhs)
    }

    /// Returns `true` if every element of `self` is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite() && self.y_axis.is_finite() && self.z_axis.is_finite()
    }

    /// Returns `true` if any element of `self` is `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan() || self.z_axis.is_nan()
    }

    /// Returns `true` if every column is within `max_abs_diff` of `rhs`.
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
            && self.z_axis.abs_diff_eq(rhs.z_axis, max_abs_diff)
    }
}

impl fmt::Display for Mat3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x_axis, self.y_axis, self.z_axis)
    }
}

impl Neg for Mat3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_cols(-self.x_axis, -self.y_axis, -self.z_axis)
    }
}

impl Add for Mat3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis + rhs.x_axis,
            self.y_axis + rhs.y_axis,
            self.z_axis + rhs.z_axis,
        )
    }
}

impl AddAssign for Mat3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x_axis += rhs.x_axis;
        self.y_axis += rhs.y_axis;
        self.z_axis += rhs.z_axis;
    }
}

impl Sub for Mat3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis - rhs.x_axis,
            self.y_axis - rhs.y_axis,
            self.z_axis - rhs.z_axis,
        )
    }
}

impl SubAssign for Mat3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x_axis -= rhs.x_axis;
        self.y_axis -= rhs.y_axis;
        self.z_axis -= rhs.z_axis;
    }
}

impl Mul for Mat3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat3(&rhs)
    }
}

impl MulAssign for Mat3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_mat3(&rhs);
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Vec3 {
        self.mul_vec3(rhs)
    }
}

impl Mul<f32> for Mat3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        self.mul_scalar(rhs)
    }
}

impl Mul<Mat3> for f32 {
    type Output = Mat3;
    #[inline(always)]
    fn mul(self, rhs: Mat3) -> Mat3 {
        rhs.mul_scalar(self)
    }
}
