//! 4x4 column-major single-precision matrix (`Mat4`) and 3D affine transform (`Affine3A`).
//!
//! # Inputs / Outputs
//! Provides a contiguous `#[repr(C)]` 4x4 column-major matrix (`x_axis`, `y_axis`, `z_axis`, `w_axis`)
//! for 3D affine model/view transforms and right-handed `[0, 1]` clip-space projections
//! (perspective, Reverse-Z infinite perspective, and orthographic), plus `Affine3A`.

use crate::{Mat3, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 4x4 column-major matrix of `f32` elements.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mat4 {
    /// Column 0 (`x` basis vector).
    pub x_axis: Vec4,
    /// Column 1 (`y` basis vector).
    pub y_axis: Vec4,
    /// Column 2 (`z` basis vector).
    pub z_axis: Vec4,
    /// Column 3 (`w` translation/homogeneous column).
    pub w_axis: Vec4,
}

impl Default for Mat4 {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat4 {
    /// All zeroes.
    pub const ZERO: Self = Self::from_cols(Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO);
    /// Identity 4x4 matrix with `1.0` on the main diagonal.
    pub const IDENTITY: Self = Self::from_cols(Vec4::X, Vec4::Y, Vec4::Z, Vec4::W);
    /// All `f32::NAN`.
    pub const NAN: Self = Self::from_cols(Vec4::NAN, Vec4::NAN, Vec4::NAN, Vec4::NAN);

    /// Creates a 4x4 matrix from its four column vectors.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    /// Creates a 4x4 matrix from a flat column-major 16-element array.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_array(m: &[f32; 16]) -> Self {
        Self::from_cols(
            Vec4::new(m[0], m[1], m[2], m[3]),
            Vec4::new(m[4], m[5], m[6], m[7]),
            Vec4::new(m[8], m[9], m[10], m[11]),
            Vec4::new(m[12], m[13], m[14], m[15]),
        )
    }

    /// Converts this matrix into a flat column-major 16-element array.
    #[inline(always)]
    #[must_use]
    pub const fn to_cols_array(&self) -> [f32; 16] {
        [
            self.x_axis.x,
            self.x_axis.y,
            self.x_axis.z,
            self.x_axis.w,
            self.y_axis.x,
            self.y_axis.y,
            self.y_axis.z,
            self.y_axis.w,
            self.z_axis.x,
            self.z_axis.y,
            self.z_axis.z,
            self.z_axis.w,
            self.w_axis.x,
            self.w_axis.y,
            self.w_axis.z,
            self.w_axis.w,
        ]
    }

    /// Creates a 4x4 matrix from a 2D array of column vectors `[[f32; 4]; 4]`.
    #[inline(always)]
    #[must_use]
    pub const fn from_cols_array_2d(m: &[[f32; 4]; 4]) -> Self {
        Self::from_cols(
            Vec4::from_array(m[0]),
            Vec4::from_array(m[1]),
            Vec4::from_array(m[2]),
            Vec4::from_array(m[3]),
        )
    }

    /// Converts this matrix into a 2D array of column vectors `[[f32; 4]; 4]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_cols_array_2d(&self) -> [[f32; 4]; 4] {
        [
            self.x_axis.to_array(),
            self.y_axis.to_array(),
            self.z_axis.to_array(),
            self.w_axis.to_array(),
        ]
    }

    /// Creates a diagonal 4x4 matrix with `diagonal` along the main diagonal.
    #[inline(always)]
    #[must_use]
    pub const fn from_diagonal(diagonal: Vec4) -> Self {
        Self::from_cols(
            Vec4::new(diagonal.x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, diagonal.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, diagonal.z, 0.0),
            Vec4::new(0.0, 0.0, 0.0, diagonal.w),
        )
    }

    /// Creates a 4x4 affine translation matrix.
    #[inline(always)]
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self::from_cols(
            Vec4::X,
            Vec4::Y,
            Vec4::Z,
            Vec4::new(translation.x, translation.y, translation.z, 1.0),
        )
    }

    /// Creates a 4x4 non-uniform scale matrix.
    #[inline(always)]
    #[must_use]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self::from_cols(
            Vec4::new(scale.x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, scale.y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, scale.z, 0.0),
            Vec4::W,
        )
    }

    /// Creates a 4x4 rotation matrix from a 3x3 matrix [`Mat3`].
    #[inline(always)]
    #[must_use]
    pub const fn from_mat3(m: Mat3) -> Self {
        Self::from_cols(
            m.x_axis.extend(0.0),
            m.y_axis.extend(0.0),
            m.z_axis.extend(0.0),
            Vec4::W,
        )
    }

    /// Creates a 4x4 rotation matrix from a unit quaternion [`Quat`].
    #[inline]
    #[must_use]
    pub fn from_quat(rotation: Quat) -> Self {
        Self::from_mat3(Mat3::from_quat(rotation))
    }

    /// Creates a 4x4 rotation matrix around unit `axis` by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        Self::from_mat3(Mat3::from_axis_angle(axis, angle))
    }

    /// Creates a 4x4 rotation matrix around the +X axis by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_rotation_x(angle: f32) -> Self {
        Self::from_mat3(Mat3::from_rotation_x(angle))
    }

    /// Creates a 4x4 rotation matrix around the +Y axis by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_rotation_y(angle: f32) -> Self {
        Self::from_mat3(Mat3::from_rotation_y(angle))
    }

    /// Creates a 4x4 rotation matrix around the +Z axis by `angle` radians.
    #[inline]
    #[must_use]
    pub fn from_rotation_z(angle: f32) -> Self {
        Self::from_mat3(Mat3::from_rotation_z(angle))
    }

    /// Creates a 4x4 affine transformation matrix from `scale`, `rotation`, and `translation` (`T * R * S`).
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        let rot = Mat3::from_quat(rotation);
        Self::from_cols(
            (rot.x_axis * scale.x).extend(0.0),
            (rot.y_axis * scale.y).extend(0.0),
            (rot.z_axis * scale.z).extend(0.0),
            translation.extend(1.0),
        )
    }

    /// Extracts `(scale, rotation, translation)` from a 4x4 affine transformation matrix with non-zero scale.
    #[inline]
    #[must_use]
    pub fn to_scale_rotation_translation(&self) -> (Vec3, Quat, Vec3) {
        let det = self.determinant();
        let scale = Vec3::new(
            self.x_axis.truncate().length() * det.signum(),
            self.y_axis.truncate().length(),
            self.z_axis.truncate().length(),
        );
        let inv_scale = Vec3::new(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
        let rotation = Quat::from_rotation_axes(
            self.x_axis.truncate() * inv_scale.x,
            self.y_axis.truncate() * inv_scale.y,
            self.z_axis.truncate() * inv_scale.z,
        );
        let translation = self.w_axis.truncate();
        (scale, rotation, translation)
    }

    /// Creates a right-handed view matrix from camera `eye` position, forward `dir`, and `up` vector.
    #[inline]
    #[must_use]
    pub fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self::from_cols(
            Vec4::new(s.x, u.x, -f.x, 0.0),
            Vec4::new(s.y, u.y, -f.y, 0.0),
            Vec4::new(s.z, u.z, -f.z, 0.0),
            Vec4::new(-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0),
        )
    }

    /// Creates a right-handed view matrix looking from `eye` toward `center` with `up` direction.
    #[inline]
    #[must_use]
    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        Self::look_to_rh(eye, center - eye, up)
    }

    /// Creates a right-handed perspective projection matrix with `[0, 1]` depth range.
    ///
    /// View-space `z = -z_near` maps to NDC `z = 0.0` and `z = -z_far` maps to NDC `z = 1.0`.
    /// Swapping `z_near` and `z_far` produces a finite Reverse-Z projection.
    #[inline]
    #[must_use]
    pub fn perspective_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self {
        let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
        let h = cos_fov / sin_fov;
        let w = h / aspect_ratio;
        let r = z_far / (z_near - z_far);
        Self::from_cols(
            Vec4::new(w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, -1.0),
            Vec4::new(0.0, 0.0, r * z_near, 0.0),
        )
    }

    /// Creates a right-handed infinite far-plane Reverse-Z perspective projection matrix (`[0, 1]` depth).
    ///
    /// View-space `z = -z_near` maps to NDC `z = 1.0` and `z -> -∞` maps to NDC `z = 0.0`.
    #[inline]
    #[must_use]
    pub fn perspective_infinite_reverse_rh(
        fov_y_radians: f32,
        aspect_ratio: f32,
        z_near: f32,
    ) -> Self {
        let (sin_fov, cos_fov) = (0.5 * fov_y_radians).sin_cos();
        let f = cos_fov / sin_fov;
        Self::from_cols(
            Vec4::new(f / aspect_ratio, 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
            Vec4::new(0.0, 0.0, z_near, 0.0),
        )
    }

    /// Creates a right-handed orthographic projection matrix with `[0, 1]` depth range.
    #[inline]
    #[must_use]
    pub fn orthographic_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let rcp_width = 1.0 / (right - left);
        let rcp_height = 1.0 / (top - bottom);
        let r = 1.0 / (near - far);
        Self::from_cols(
            Vec4::new(rcp_width + rcp_width, 0.0, 0.0, 0.0),
            Vec4::new(0.0, rcp_height + rcp_height, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 0.0),
            Vec4::new(
                -(left + right) * rcp_width,
                -(top + bottom) * rcp_height,
                r * near,
                1.0,
            ),
        )
    }

    /// Returns column `index` (`0..4`).
    ///
    /// # Panics
    /// Panics if `index >= 4`.
    #[inline(always)]
    #[must_use]
    pub const fn col(&self, index: usize) -> Vec4 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            3 => self.w_axis,
            _ => panic!("Mat4 column index out of bounds"),
        }
    }

    /// Returns row `index` (`0..4`).
    ///
    /// # Panics
    /// Panics if `index >= 4`.
    #[inline(always)]
    #[must_use]
    pub const fn row(&self, index: usize) -> Vec4 {
        match index {
            0 => Vec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
            1 => Vec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
            2 => Vec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
            3 => Vec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
            _ => panic!("Mat4 row index out of bounds"),
        }
    }

    /// Returns the transpose of `self`.
    #[inline(always)]
    #[must_use]
    pub const fn transpose(&self) -> Self {
        Self::from_cols(self.row(0), self.row(1), self.row(2), self.row(3))
    }

    /// Computes the determinant of `self`.
    #[inline]
    #[must_use]
    pub fn determinant(&self) -> f32 {
        let (m00, m01, m02, m03) = self.x_axis.into();
        let (m10, m11, m12, m13) = self.y_axis.into();
        let (m20, m21, m22, m23) = self.z_axis.into();
        let (m30, m31, m32, m33) = self.w_axis.into();

        let a2323 = m22 * m33 - m23 * m32;
        let a1323 = m21 * m33 - m23 * m31;
        let a1223 = m21 * m32 - m22 * m31;
        let a0323 = m20 * m33 - m23 * m30;
        let a0223 = m20 * m32 - m22 * m30;
        let a0123 = m20 * m31 - m21 * m30;

        m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223)
            - m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223)
            + m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123)
            - m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123)
    }

    /// Computes the inverse of `self` using 2x2 sub-determinant cofactor expansion.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let (m00, m01, m02, m03) = self.x_axis.into();
        let (m10, m11, m12, m13) = self.y_axis.into();
        let (m20, m21, m22, m23) = self.z_axis.into();
        let (m30, m31, m32, m33) = self.w_axis.into();

        let coef00 = m22 * m33 - m32 * m23;
        let coef02 = m12 * m33 - m32 * m13;
        let coef03 = m12 * m23 - m22 * m13;

        let coef04 = m21 * m33 - m31 * m23;
        let coef06 = m11 * m33 - m31 * m13;
        let coef07 = m11 * m23 - m21 * m13;

        let coef08 = m21 * m32 - m31 * m22;
        let coef10 = m11 * m32 - m31 * m12;
        let coef11 = m11 * m22 - m21 * m12;

        let coef12 = m20 * m33 - m30 * m23;
        let coef14 = m10 * m33 - m30 * m13;
        let coef15 = m10 * m23 - m20 * m13;

        let coef16 = m20 * m32 - m30 * m22;
        let coef18 = m10 * m32 - m30 * m12;
        let coef19 = m10 * m22 - m20 * m12;

        let coef20 = m20 * m31 - m30 * m21;
        let coef22 = m10 * m31 - m30 * m11;
        let coef23 = m10 * m21 - m20 * m11;

        let fac0 = Vec4::new(coef00, coef00, coef02, coef03);
        let fac1 = Vec4::new(coef04, coef04, coef06, coef07);
        let fac2 = Vec4::new(coef08, coef08, coef10, coef11);
        let fac3 = Vec4::new(coef12, coef12, coef14, coef15);
        let fac4 = Vec4::new(coef16, coef16, coef18, coef19);
        let fac5 = Vec4::new(coef20, coef20, coef22, coef23);

        let vec0 = Vec4::new(m10, m00, m00, m00);
        let vec1 = Vec4::new(m11, m01, m01, m01);
        let vec2 = Vec4::new(m12, m02, m02, m02);
        let vec3 = Vec4::new(m13, m03, m03, m03);

        let inv0 = vec1 * fac0 - vec2 * fac1 + vec3 * fac2;
        let inv1 = vec0 * fac0 - vec2 * fac3 + vec3 * fac4;
        let inv2 = vec0 * fac1 - vec1 * fac3 + vec3 * fac5;
        let inv3 = vec0 * fac2 - vec1 * fac4 + vec2 * fac5;

        let sign_a = Vec4::new(1.0, -1.0, 1.0, -1.0);
        let sign_b = Vec4::new(-1.0, 1.0, -1.0, 1.0);

        let inverse = Self::from_cols(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b);

        let col0 = Vec4::new(
            inverse.x_axis.x,
            inverse.y_axis.x,
            inverse.z_axis.x,
            inverse.w_axis.x,
        );
        let dot = self.x_axis.dot(col0);
        let rcp_det = dot.recip();
        inverse.mul_scalar(rcp_det)
    }

    /// Multiplies this 4x4 matrix by a 4D column vector `rhs`.
    #[inline(always)]
    #[must_use]
    pub fn mul_vec4(&self, rhs: Vec4) -> Vec4 {
        (self.x_axis * rhs.x)
            + (self.y_axis * rhs.y)
            + (self.z_axis * rhs.z)
            + (self.w_axis * rhs.w)
    }

    /// Multiplies two 4x4 matrices `self * rhs`.
    #[inline]
    #[must_use]
    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.mul_vec4(rhs.x_axis),
            self.mul_vec4(rhs.y_axis),
            self.mul_vec4(rhs.z_axis),
            self.mul_vec4(rhs.w_axis),
        )
    }

    /// Multiplies every element by scalar `rhs`.
    #[inline(always)]
    #[must_use]
    pub fn mul_scalar(&self, rhs: f32) -> Self {
        Self::from_cols(
            self.x_axis * rhs,
            self.y_axis * rhs,
            self.z_axis * rhs,
            self.w_axis * rhs,
        )
    }

    /// Transforms a 3D affine point `(rhs.x, rhs.y, rhs.z, 1.0)` without perspective division.
    #[inline(always)]
    #[must_use]
    pub fn transform_point3(&self, rhs: Vec3) -> Vec3 {
        (self.x_axis.truncate() * rhs.x)
            + (self.y_axis.truncate() * rhs.y)
            + (self.z_axis.truncate() * rhs.z)
            + self.w_axis.truncate()
    }

    /// Transforms a 3D direction vector `(rhs.x, rhs.y, rhs.z, 0.0)` ignoring translation.
    #[inline(always)]
    #[must_use]
    pub fn transform_vector3(&self, rhs: Vec3) -> Vec3 {
        (self.x_axis.truncate() * rhs.x)
            + (self.y_axis.truncate() * rhs.y)
            + (self.z_axis.truncate() * rhs.z)
    }

    /// Projects a 3D point `(rhs.x, rhs.y, rhs.z, 1.0)` with homogeneous perspective division by `w`.
    #[inline]
    #[must_use]
    pub fn project_point3(&self, rhs: Vec3) -> Vec3 {
        let res = self.mul_vec4(rhs.extend(1.0));
        res.truncate() * res.w.recip()
    }

    /// Returns `true` if every element of `self` is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x_axis.is_finite()
            && self.y_axis.is_finite()
            && self.z_axis.is_finite()
            && self.w_axis.is_finite()
    }

    /// Returns `true` if any element of `self` is `NaN`.
    #[inline]
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.x_axis.is_nan() || self.y_axis.is_nan() || self.z_axis.is_nan() || self.w_axis.is_nan()
    }

    /// Returns `true` if every column is within `max_abs_diff` of `rhs`.
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(&self, rhs: Self, max_abs_diff: f32) -> bool {
        self.x_axis.abs_diff_eq(rhs.x_axis, max_abs_diff)
            && self.y_axis.abs_diff_eq(rhs.y_axis, max_abs_diff)
            && self.z_axis.abs_diff_eq(rhs.z_axis, max_abs_diff)
            && self.w_axis.abs_diff_eq(rhs.w_axis, max_abs_diff)
    }
}

impl fmt::Display for Mat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.x_axis, self.y_axis, self.z_axis, self.w_axis
        )
    }
}

impl Neg for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::from_cols(-self.x_axis, -self.y_axis, -self.z_axis, -self.w_axis)
    }
}

impl Add for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis + rhs.x_axis,
            self.y_axis + rhs.y_axis,
            self.z_axis + rhs.z_axis,
            self.w_axis + rhs.w_axis,
        )
    }
}

impl AddAssign for Mat4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x_axis += rhs.x_axis;
        self.y_axis += rhs.y_axis;
        self.z_axis += rhs.z_axis;
        self.w_axis += rhs.w_axis;
    }
}

impl Sub for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_cols(
            self.x_axis - rhs.x_axis,
            self.y_axis - rhs.y_axis,
            self.z_axis - rhs.z_axis,
            self.w_axis - rhs.w_axis,
        )
    }
}

impl SubAssign for Mat4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x_axis -= rhs.x_axis;
        self.y_axis -= rhs.y_axis;
        self.z_axis -= rhs.z_axis;
        self.w_axis -= rhs.w_axis;
    }
}

impl Mul for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul_mat4(&rhs)
    }
}

impl MulAssign for Mat4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_mat4(&rhs);
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, rhs: Vec4) -> Vec4 {
        self.mul_vec4(rhs)
    }
}

impl Mul<f32> for Mat4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        self.mul_scalar(rhs)
    }
}

impl Mul<Mat4> for f32 {
    type Output = Mat4;
    #[inline(always)]
    fn mul(self, rhs: Mat4) -> Mat4 {
        rhs.mul_scalar(self)
    }
}

/// A 3D affine transformation stored as a 3x3 linear basis (`matrix3`) and 3D `translation`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Affine3A {
    /// 3x3 linear basis (rotation and scale).
    pub matrix3: Mat3,
    /// 3D translation offset.
    pub translation: Vec3,
}

impl Default for Affine3A {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine3A {
    /// Identity 3D affine transform.
    pub const IDENTITY: Self = Self {
        matrix3: Mat3::IDENTITY,
        translation: Vec3::ZERO,
    };

    /// Creates an affine transform from `scale`, `rotation`, and `translation`.
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        let r = Mat3::from_quat(rotation);
        Self {
            matrix3: Mat3::from_cols(r.x_axis * scale.x, r.y_axis * scale.y, r.z_axis * scale.z),
            translation,
        }
    }

    /// Creates an affine transform from a translation offset.
    #[inline(always)]
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            matrix3: Mat3::IDENTITY,
            translation,
        }
    }

    /// Transforms a 3D point.
    #[inline(always)]
    #[must_use]
    pub fn transform_point3(&self, rhs: Vec3) -> Vec3 {
        self.matrix3.mul_vec3(rhs) + self.translation
    }

    /// Transforms a 3D direction vector.
    #[inline(always)]
    #[must_use]
    pub fn transform_vector3(&self, rhs: Vec3) -> Vec3 {
        self.matrix3.mul_vec3(rhs)
    }

    /// Computes the inverse of this affine transform.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let matrix3 = self.matrix3.inverse();
        let translation = -(matrix3 * self.translation);
        Self {
            matrix3,
            translation,
        }
    }
}

impl From<Affine3A> for Mat4 {
    #[inline]
    fn from(a: Affine3A) -> Self {
        Self::from_cols(
            a.matrix3.x_axis.extend(0.0),
            a.matrix3.y_axis.extend(0.0),
            a.matrix3.z_axis.extend(0.0),
            a.translation.extend(1.0),
        )
    }
}
