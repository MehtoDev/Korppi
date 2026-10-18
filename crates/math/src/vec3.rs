//! 3-dimensional single-precision floating-point vector (`Vec3`).
//!
//! # Inputs / Outputs
//! Provides a contiguous `#[repr(C)]` `(x, y, z)` 3D vector with Euclidean dot/cross products,
//! Pixar orthonormal basis generation, swizzles, component-wise arithmetic, and `serde` support.

use crate::{Vec2, Vec4};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A 3-dimensional vector of `f32` components.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    /// X coordinate component.
    pub x: f32,
    /// Y coordinate component.
    pub y: f32,
    /// Z coordinate component.
    pub z: f32,
}

impl Vec3 {
    /// All zeroes `(0.0, 0.0, 0.0)`.
    pub const ZERO: Self = Self::splat(0.0);
    /// All ones `(1.0, 1.0, 1.0)`.
    pub const ONE: Self = Self::splat(1.0);
    /// All negative ones `(-1.0, -1.0, -1.0)`.
    pub const NEG_ONE: Self = Self::splat(-1.0);
    /// All `f32::MIN`.
    pub const MIN: Self = Self::splat(f32::MIN);
    /// All `f32::MAX`.
    pub const MAX: Self = Self::splat(f32::MAX);
    /// All `f32::INFINITY`.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);
    /// All `f32::NEG_INFINITY`.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);
    /// All `f32::NAN`.
    pub const NAN: Self = Self::splat(f32::NAN);
    /// Unit vector along positive X `(1.0, 0.0, 0.0)`.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    /// Unit vector along positive Y `(0.0, 1.0, 0.0)`.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    /// Unit vector along positive Z `(0.0, 0.0, 1.0)`.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);
    /// Unit vector along negative X `(-1.0, 0.0, 0.0)`.
    pub const NEG_X: Self = Self::new(-1.0, 0.0, 0.0);
    /// Unit vector along negative Y `(0.0, -1.0, 0.0)`.
    pub const NEG_Y: Self = Self::new(0.0, -1.0, 0.0);
    /// Unit vector along negative Z `(0.0, 0.0, -1.0)`.
    pub const NEG_Z: Self = Self::new(0.0, 0.0, -1.0);
    /// Coordinate axes `[X, Y, Z]`.
    pub const AXES: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Creates a new 3D vector from `x`, `y`, and `z` components.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Creates a 3D vector with all components set to `v`.
    #[inline(always)]
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Creates a 3D vector from a 3-element array.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    /// Converts this vector into a 3-element array `[x, y, z]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Creates a vector from the first 3 elements of `slice`.
    ///
    /// # Panics
    /// Panics if `slice.len() < 3`.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[f32]) -> Self {
        Self::new(slice[0], slice[1], slice[2])
    }

    /// Writes the components of `self` into the first 3 elements of `slice`.
    ///
    /// # Panics
    /// Panics if `slice.len() < 3`.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
    }

    /// Extends this 3D vector with a `w` component to form a [`Vec4`].
    #[inline(always)]
    #[must_use]
    pub const fn extend(self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }

    /// Truncates this 3D vector to a [`Vec2`] by dropping `z`.
    #[inline(always)]
    #[must_use]
    pub const fn truncate(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Returns the `(x, y)` sub-vector as a [`Vec2`].
    #[inline(always)]
    #[must_use]
    pub const fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Returns the `(x, z)` sub-vector as a [`Vec2`].
    #[inline(always)]
    #[must_use]
    pub const fn xz(self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }

    /// Returns the `(y, z)` sub-vector as a [`Vec2`].
    #[inline(always)]
    #[must_use]
    pub const fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    /// Computes the dot product `self.x * rhs.x + self.y * rhs.y + self.z * rhs.z`.
    #[inline(always)]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    /// Computes the right-handed 3D cross product `self × rhs`.
    #[inline(always)]
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            (self.y * rhs.z) - (rhs.y * self.z),
            (self.z * rhs.x) - (rhs.z * self.x),
            (self.x * rhs.y) - (rhs.x * self.y),
        )
    }

    /// Computes the squared Euclidean length `self.dot(self)`.
    #[inline(always)]
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Computes the Euclidean length.
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

    /// Computes the Euclidean distance between `self` and `rhs`.
    #[inline(always)]
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }

    /// Computes the squared Euclidean distance between `self` and `rhs`.
    #[inline(always)]
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> f32 {
        (self - rhs).length_squared()
    }

    /// Returns `self` normalized to unit length.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        self * self.length_recip()
    }

    /// Returns `Some(normalized)` if length is finite and positive, otherwise `None`.
    #[inline]
    #[must_use]
    pub fn try_normalize(self) -> Option<Self> {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 {
            Some(self * rcp)
        } else {
            None
        }
    }

    /// Returns `self` normalized to unit length, or [`Self::ZERO`] if length is zero or non-finite.
    #[inline]
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 {
            self * rcp
        } else {
            Self::ZERO
        }
    }

    /// Returns `self` normalized to unit length, or `fallback` if length is zero or non-finite.
    #[inline]
    #[must_use]
    pub fn normalize_or(self, fallback: Self) -> Self {
        let rcp = self.length_recip();
        if rcp.is_finite() && rcp > 0.0 {
            self * rcp
        } else {
            fallback
        }
    }

    /// Returns `true` if the vector length is within `1e-4` of `1.0`.
    #[inline]
    #[must_use]
    pub fn is_normalized(self) -> bool {
        (self.length_squared() - 1.0).abs() <= 2e-4
    }

    /// Returns any non-zero vector orthogonal to `self`.
    #[inline]
    #[must_use]
    pub fn any_orthogonal_vector(self) -> Self {
        if self.x.abs() > self.y.abs() {
            Self::new(-self.z, 0.0, self.x)
        } else {
            Self::new(0.0, self.z, -self.y)
        }
    }

    /// Returns a unit vector orthogonal to unit vector `self` using Duff et al.'s Pixar basis.
    #[inline]
    #[must_use]
    pub fn any_orthonormal_vector(self) -> Self {
        let sign = self.z.signum();
        let a = -1.0 / (sign + self.z);
        let b = self.x * self.y * a;
        Self::new(b, sign + self.y * self.y * a, -self.y)
    }

    /// Returns two unit vectors that together with unit vector `self` form an orthonormal basis.
    #[inline]
    #[must_use]
    pub fn any_orthonormal_pair(self) -> (Self, Self) {
        let sign = self.z.signum();
        let a = -1.0 / (sign + self.z);
        let b = self.x * self.y * a;
        (
            Self::new(1.0 + sign * self.x * self.x * a, sign * b, -sign * self.x),
            Self::new(b, sign + self.y * self.y * a, -self.y),
        )
    }

    /// Component-wise minimum.
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    /// Component-wise maximum.
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    /// Component-wise clamp between `min` and `max`.
    #[inline(always)]
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Returns the smallest component.
    #[inline(always)]
    #[must_use]
    pub fn min_element(self) -> f32 {
        self.x.min(self.y.min(self.z))
    }

    /// Returns the largest component.
    #[inline(always)]
    #[must_use]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y.max(self.z))
    }

    /// Returns the sum of all components.
    #[inline(always)]
    #[must_use]
    pub fn element_sum(self) -> f32 {
        self.x + self.y + self.z
    }

    /// Component-wise absolute value.
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Component-wise signum.
    #[inline(always)]
    #[must_use]
    pub fn signum(self) -> Self {
        Self::new(self.x.signum(), self.y.signum(), self.z.signum())
    }

    /// Component-wise `copysign`.
    #[inline(always)]
    #[must_use]
    pub fn copysign(self, rhs: Self) -> Self {
        Self::new(
            self.x.copysign(rhs.x),
            self.y.copysign(rhs.y),
            self.z.copysign(rhs.z),
        )
    }

    /// Component-wise floor.
    #[inline(always)]
    #[must_use]
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Component-wise ceil.
    #[inline(always)]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Component-wise round.
    #[inline(always)]
    #[must_use]
    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Component-wise trunc.
    #[inline(always)]
    #[must_use]
    pub fn trunc(self) -> Self {
        Self::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Component-wise fractional part `self - self.floor()`.
    #[inline(always)]
    #[must_use]
    pub fn fract(self) -> Self {
        self - self.floor()
    }

    /// Component-wise reciprocal `1.0 / self`.
    #[inline(always)]
    #[must_use]
    pub fn recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip(), self.z.recip())
    }

    /// Linearly interpolates between `self` and `rhs` by `s`: `self * (1.0 - s) + rhs * s`.
    #[inline(always)]
    #[must_use]
    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self * (1.0 - s) + rhs * s
    }

    /// Component-wise `(self * a) + b` (unfused for cross-platform determinism).
    #[inline(always)]
    #[must_use]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    /// Clamps the vector's length to at most `max`.
    #[inline]
    #[must_use]
    pub fn clamp_length_max(self, max: f32) -> Self {
        let len_sq = self.length_squared();
        if len_sq > max * max {
            self * (max / len_sq.sqrt())
        } else {
            self
        }
    }

    /// Clamps the vector's length to at least `min`.
    #[inline]
    #[must_use]
    pub fn clamp_length_min(self, min: f32) -> Self {
        let len_sq = self.length_squared();
        if len_sq < min * min {
            self * (min / len_sq.sqrt())
        } else {
            self
        }
    }

    /// Clamps the vector's length to `[min, max]`.
    #[inline]
    #[must_use]
    pub fn clamp_length(self, min: f32, max: f32) -> Self {
        let len_sq = self.length_squared();
        if len_sq < min * min {
            self * (min / len_sq.sqrt())
        } else if len_sq > max * max {
            self * (max / len_sq.sqrt())
        } else {
            self
        }
    }

    /// Projects `self` onto `rhs`.
    #[inline]
    #[must_use]
    pub fn project_onto(self, rhs: Self) -> Self {
        rhs * (self.dot(rhs) / rhs.dot(rhs))
    }

    /// Rejects `self` from `rhs` (component of `self` orthogonal to `rhs`).
    #[inline]
    #[must_use]
    pub fn reject_from(self, rhs: Self) -> Self {
        self - self.project_onto(rhs)
    }

    /// Returns `true` if all components are finite.
    #[inline(always)]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Returns `true` if any component is `NaN`.
    #[inline(always)]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Returns `true` if `|self - rhs| <= max_abs_diff` for every component.
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        let diff = (self - rhs).abs();
        diff.x <= max_abs_diff && diff.y <= max_abs_diff && diff.z <= max_abs_diff
    }

    /// Component-wise equality mask `self == rhs`.
    #[inline(always)]
    #[must_use]
    pub fn cmpeq(self, rhs: Self) -> BVec3 {
        BVec3::new(self.x == rhs.x, self.y == rhs.y, self.z == rhs.z)
    }

    /// Component-wise inequality mask `self != rhs`.
    #[inline(always)]
    #[must_use]
    pub fn cmpne(self, rhs: Self) -> BVec3 {
        BVec3::new(self.x != rhs.x, self.y != rhs.y, self.z != rhs.z)
    }

    /// Component-wise `self < rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmplt(self, rhs: Self) -> BVec3 {
        BVec3::new(self.x < rhs.x, self.y < rhs.y, self.z < rhs.z)
    }

    /// Component-wise `self <= rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmple(self, rhs: Self) -> BVec3 {
        BVec3::new(self.x <= rhs.x, self.y <= rhs.y, self.z <= rhs.z)
    }

    /// Component-wise `self > rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmpgt(self, rhs: Self) -> BVec3 {
        BVec3::new(self.x > rhs.x, self.y > rhs.y, self.z > rhs.z)
    }

    /// Component-wise `self >= rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmpge(self, rhs: Self) -> BVec3 {
        BVec3::new(self.x >= rhs.x, self.y >= rhs.y, self.z >= rhs.z)
    }
}

/// 3-lane boolean mask returned by [`Vec3`] component-wise comparisons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BVec3 {
    /// X lane boolean result.
    pub x: bool,
    /// Y lane boolean result.
    pub y: bool,
    /// Z lane boolean result.
    pub z: bool,
}

impl BVec3 {
    /// Creates a new 3-lane boolean mask.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: bool, y: bool, z: bool) -> Self {
        Self { x, y, z }
    }

    /// Returns `true` if any lane is `true`.
    #[inline(always)]
    #[must_use]
    pub const fn any(self) -> bool {
        self.x || self.y || self.z
    }

    /// Returns `true` if all lanes are `true`.
    #[inline(always)]
    #[must_use]
    pub const fn all(self) -> bool {
        self.x && self.y && self.z
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index {index} out of bounds (len is 3)"),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index {index} out of bounds (len is 3)"),
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl AddAssign for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl AddAssign<f32> for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl SubAssign for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl SubAssign<f32> for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
    }
}

impl Mul for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self * rhs.x, self * rhs.y, self * rhs.z)
    }
}

impl MulAssign for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Div<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn div(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self / rhs.x, self / rhs.y, self / rhs.z)
    }
}

impl DivAssign for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl DivAssign<f32> for Vec3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl From<[f32; 3]> for Vec3 {
    #[inline(always)]
    fn from(a: [f32; 3]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vec3> for [f32; 3] {
    #[inline(always)]
    fn from(v: Vec3) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32, f32)> for Vec3 {
    #[inline(always)]
    fn from((x, y, z): (f32, f32, f32)) -> Self {
        Self::new(x, y, z)
    }
}

impl From<Vec3> for (f32, f32, f32) {
    #[inline(always)]
    fn from(v: Vec3) -> Self {
        (v.x, v.y, v.z)
    }
}

impl Sum for Vec3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<'a> Sum<&'a Self> for Vec3 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl Product for Vec3 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}
