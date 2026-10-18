//! 4-dimensional single-precision floating-point vector (`Vec4`).
//!
//! # Inputs / Outputs
//! Provides a contiguous `#[repr(C)]` `(x, y, z, w)` 4D vector for homogeneous clip-space
//! coordinates, colors, and uniform packing, with swizzles, indexing, and `serde` support.

use crate::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A 4-dimensional vector of `f32` components.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec4 {
    /// X coordinate component.
    pub x: f32,
    /// Y coordinate component.
    pub y: f32,
    /// Z coordinate component.
    pub z: f32,
    /// W coordinate component.
    pub w: f32,
}

impl Vec4 {
    /// All zeroes `(0.0, 0.0, 0.0, 0.0)`.
    pub const ZERO: Self = Self::splat(0.0);
    /// All ones `(1.0, 1.0, 1.0, 1.0)`.
    pub const ONE: Self = Self::splat(1.0);
    /// All negative ones `(-1.0, -1.0, -1.0, -1.0)`.
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
    /// Unit vector along positive X `(1.0, 0.0, 0.0, 0.0)`.
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    /// Unit vector along positive Y `(0.0, 1.0, 0.0, 0.0)`.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    /// Unit vector along positive Z `(0.0, 0.0, 1.0, 0.0)`.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    /// Unit vector along positive W `(0.0, 0.0, 0.0, 1.0)`.
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    /// Unit vector along negative X `(-1.0, 0.0, 0.0, 0.0)`.
    pub const NEG_X: Self = Self::new(-1.0, 0.0, 0.0, 0.0);
    /// Unit vector along negative Y `(0.0, -1.0, 0.0, 0.0)`.
    pub const NEG_Y: Self = Self::new(0.0, -1.0, 0.0, 0.0);
    /// Unit vector along negative Z `(0.0, 0.0, -1.0, 0.0)`.
    pub const NEG_Z: Self = Self::new(0.0, 0.0, -1.0, 0.0);
    /// Unit vector along negative W `(0.0, 0.0, 0.0, -1.0)`.
    pub const NEG_W: Self = Self::new(0.0, 0.0, 0.0, -1.0);
    /// Coordinate axes `[X, Y, Z, W]`.
    pub const AXES: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    /// Creates a new 4D vector from `x`, `y`, `z`, and `w` components.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a 4D vector with all components set to `v`.
    #[inline(always)]
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }

    /// Creates a 4D vector from a 4-element array.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// Converts this vector into a 4-element array `[x, y, z, w]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Creates a vector from the first 4 elements of `slice`.
    ///
    /// # Panics
    /// Panics if `slice.len() < 4`.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[f32]) -> Self {
        Self::new(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the components of `self` into the first 4 elements of `slice`.
    ///
    /// # Panics
    /// Panics if `slice.len() < 4`.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
        slice[3] = self.w;
    }

    /// Truncates this 4D vector to a [`Vec3`] by dropping `w`.
    #[inline(always)]
    #[must_use]
    pub const fn truncate(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Returns the `(x, y, z)` sub-vector as a [`Vec3`].
    #[inline(always)]
    #[must_use]
    pub const fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Returns the `(x, y)` sub-vector as a [`Vec2`].
    #[inline(always)]
    #[must_use]
    pub const fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Computes the dot product `x1*x2 + y1*y2 + z1*z2 + w1*w2`.
    #[inline(always)]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z) + (self.w * rhs.w)
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

    /// Returns `true` if the vector length is within `1e-4` of `1.0`.
    #[inline]
    #[must_use]
    pub fn is_normalized(self) -> bool {
        (self.length_squared() - 1.0).abs() <= 2e-4
    }

    /// Component-wise minimum.
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(
            self.x.min(rhs.x),
            self.y.min(rhs.y),
            self.z.min(rhs.z),
            self.w.min(rhs.w),
        )
    }

    /// Component-wise maximum.
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(
            self.x.max(rhs.x),
            self.y.max(rhs.y),
            self.z.max(rhs.z),
            self.w.max(rhs.w),
        )
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
        self.x.min(self.y.min(self.z.min(self.w)))
    }

    /// Returns the largest component.
    #[inline(always)]
    #[must_use]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y.max(self.z.max(self.w)))
    }

    /// Returns the sum of all components.
    #[inline(always)]
    #[must_use]
    pub fn element_sum(self) -> f32 {
        self.x + self.y + self.z + self.w
    }

    /// Component-wise absolute value.
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
    }

    /// Component-wise signum.
    #[inline(always)]
    #[must_use]
    pub fn signum(self) -> Self {
        Self::new(
            self.x.signum(),
            self.y.signum(),
            self.z.signum(),
            self.w.signum(),
        )
    }

    /// Component-wise floor.
    #[inline(always)]
    #[must_use]
    pub fn floor(self) -> Self {
        Self::new(
            self.x.floor(),
            self.y.floor(),
            self.z.floor(),
            self.w.floor(),
        )
    }

    /// Component-wise ceil.
    #[inline(always)]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil(), self.w.ceil())
    }

    /// Component-wise round.
    #[inline(always)]
    #[must_use]
    pub fn round(self) -> Self {
        Self::new(
            self.x.round(),
            self.y.round(),
            self.z.round(),
            self.w.round(),
        )
    }

    /// Component-wise trunc.
    #[inline(always)]
    #[must_use]
    pub fn trunc(self) -> Self {
        Self::new(
            self.x.trunc(),
            self.y.trunc(),
            self.z.trunc(),
            self.w.trunc(),
        )
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
        Self::new(
            self.x.recip(),
            self.y.recip(),
            self.z.recip(),
            self.w.recip(),
        )
    }

    /// Linearly interpolates between `self` and `rhs` by `s`: `self * (1.0 - s) + rhs * s`.
    #[inline(always)]
    #[must_use]
    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self * (1.0 - s) + rhs * s
    }

    /// Component-wise `(self * a) + b`.
    #[inline(always)]
    #[must_use]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
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
        let diff = (self - rhs).abs();
        diff.x <= max_abs_diff
            && diff.y <= max_abs_diff
            && diff.z <= max_abs_diff
            && diff.w <= max_abs_diff
    }

    /// Component-wise equality mask `self == rhs`.
    #[inline(always)]
    #[must_use]
    pub fn cmpeq(self, rhs: Self) -> BVec4 {
        BVec4::new(
            self.x == rhs.x,
            self.y == rhs.y,
            self.z == rhs.z,
            self.w == rhs.w,
        )
    }

    /// Component-wise inequality mask `self != rhs`.
    #[inline(always)]
    #[must_use]
    pub fn cmpne(self, rhs: Self) -> BVec4 {
        BVec4::new(
            self.x != rhs.x,
            self.y != rhs.y,
            self.z != rhs.z,
            self.w != rhs.w,
        )
    }

    /// Component-wise `self < rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmplt(self, rhs: Self) -> BVec4 {
        BVec4::new(
            self.x < rhs.x,
            self.y < rhs.y,
            self.z < rhs.z,
            self.w < rhs.w,
        )
    }

    /// Component-wise `self <= rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmple(self, rhs: Self) -> BVec4 {
        BVec4::new(
            self.x <= rhs.x,
            self.y <= rhs.y,
            self.z <= rhs.z,
            self.w <= rhs.w,
        )
    }

    /// Component-wise `self > rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmpgt(self, rhs: Self) -> BVec4 {
        BVec4::new(
            self.x > rhs.x,
            self.y > rhs.y,
            self.z > rhs.z,
            self.w > rhs.w,
        )
    }

    /// Component-wise `self >= rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmpge(self, rhs: Self) -> BVec4 {
        BVec4::new(
            self.x >= rhs.x,
            self.y >= rhs.y,
            self.z >= rhs.z,
            self.w >= rhs.w,
        )
    }
}

/// 4-lane boolean mask returned by [`Vec4`] component-wise comparisons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BVec4 {
    /// X lane boolean result.
    pub x: bool,
    /// Y lane boolean result.
    pub y: bool,
    /// Z lane boolean result.
    pub z: bool,
    /// W lane boolean result.
    pub w: bool,
}

impl BVec4 {
    /// Creates a new 4-lane boolean mask.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
        Self { x, y, z, w }
    }

    /// Returns `true` if any lane is `true`.
    #[inline(always)]
    #[must_use]
    pub const fn any(self) -> bool {
        self.x || self.y || self.z || self.w
    }

    /// Returns `true` if all lanes are `true`.
    #[inline(always)]
    #[must_use]
    pub const fn all(self) -> bool {
        self.x && self.y && self.z && self.w
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index {index} out of bounds (len is 4)"),
        }
    }
}

impl IndexMut<usize> for Vec4 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index {index} out of bounds (len is 4)"),
        }
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}

impl Add<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs, self.w + rhs)
    }
}

impl AddAssign for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl AddAssign<f32> for Vec4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.z += rhs;
        self.w += rhs;
    }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}

impl Sub<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs, self.w - rhs)
    }
}

impl SubAssign for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl SubAssign<f32> for Vec4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.z -= rhs;
        self.w -= rhs;
    }
}

impl Mul for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.x * rhs.x,
            self.y * rhs.y,
            self.z * rhs.z,
            self.w * rhs.w,
        )
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn mul(self, rhs: Vec4) -> Vec4 {
        Vec4::new(self * rhs.x, self * rhs.y, self * rhs.z, self * rhs.w)
    }
}

impl MulAssign for Vec4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
        self.w *= rhs.w;
    }
}

impl MulAssign<f32> for Vec4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl Div for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new(
            self.x / rhs.x,
            self.y / rhs.y,
            self.z / rhs.z,
            self.w / rhs.w,
        )
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}

impl Div<Vec4> for f32 {
    type Output = Vec4;
    #[inline(always)]
    fn div(self, rhs: Vec4) -> Vec4 {
        Vec4::new(self / rhs.x, self / rhs.y, self / rhs.z, self / rhs.w)
    }
}

impl DivAssign for Vec4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
        self.w /= rhs.w;
    }
}

impl DivAssign<f32> for Vec4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline(always)]
    fn from(a: [f32; 4]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vec4> for [f32; 4] {
    #[inline(always)]
    fn from(v: Vec4) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32, f32, f32)> for Vec4 {
    #[inline(always)]
    fn from((x, y, z, w): (f32, f32, f32, f32)) -> Self {
        Self::new(x, y, z, w)
    }
}

impl From<Vec4> for (f32, f32, f32, f32) {
    #[inline(always)]
    fn from(v: Vec4) -> Self {
        (v.x, v.y, v.z, v.w)
    }
}

impl Sum for Vec4 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl Product for Vec4 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}
