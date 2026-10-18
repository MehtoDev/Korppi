//! 2-dimensional single-precision floating-point vector (`Vec2`).
//!
//! # Inputs / Outputs
//! Provides a contiguous `#[repr(C)]` `(x, y)` 2D vector with standard Euclidean geometry,
//! component-wise arithmetic, swizzle/extension helpers, indexing, and `serde` support.

use crate::Vec3;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A 2-dimensional vector of `f32` components.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    /// X coordinate component.
    pub x: f32,
    /// Y coordinate component.
    pub y: f32,
}

impl Vec2 {
    /// All zeroes `(0.0, 0.0)`.
    pub const ZERO: Self = Self::splat(0.0);
    /// All ones `(1.0, 1.0)`.
    pub const ONE: Self = Self::splat(1.0);
    /// All negative ones `(-1.0, -1.0)`.
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
    /// Unit vector along positive X `(1.0, 0.0)`.
    pub const X: Self = Self::new(1.0, 0.0);
    /// Unit vector along positive Y `(0.0, 1.0)`.
    pub const Y: Self = Self::new(0.0, 1.0);
    /// Unit vector along negative X `(-1.0, 0.0)`.
    pub const NEG_X: Self = Self::new(-1.0, 0.0);
    /// Unit vector along negative Y `(0.0, -1.0)`.
    pub const NEG_Y: Self = Self::new(0.0, -1.0);
    /// Coordinate axes `[X, Y]`.
    pub const AXES: [Self; 2] = [Self::X, Self::Y];

    /// Creates a new 2D vector from `x` and `y` components.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Creates a 2D vector with both components set to `v`.
    #[inline(always)]
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }

    /// Creates a 2D vector from a 2-element array.
    #[inline(always)]
    #[must_use]
    pub const fn from_array(a: [f32; 2]) -> Self {
        Self::new(a[0], a[1])
    }

    /// Converts this vector into a 2-element array `[x, y]`.
    #[inline(always)]
    #[must_use]
    pub const fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Creates a vector from the first 2 elements of `slice`.
    ///
    /// # Panics
    /// Panics if `slice.len() < 2`.
    #[inline]
    #[must_use]
    pub const fn from_slice(slice: &[f32]) -> Self {
        Self::new(slice[0], slice[1])
    }

    /// Writes the components of `self` into the first 2 elements of `slice`.
    ///
    /// # Panics
    /// Panics if `slice.len() < 2`.
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x;
        slice[1] = self.y;
    }

    /// Extends this 2D vector with a `z` component to form a [`Vec3`].
    #[inline(always)]
    #[must_use]
    pub const fn extend(self, z: f32) -> Vec3 {
        Vec3::new(self.x, self.y, z)
    }

    /// Computes the dot product `self.x * rhs.x + self.y * rhs.y`.
    #[inline(always)]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y)
    }

    /// Returns the 2D vector rotated 90 degrees counter-clockwise `(-y, x)`.
    #[inline(always)]
    #[must_use]
    pub fn perp(self) -> Self {
        Self::new(-self.y, self.x)
    }

    /// Computes the 2D perpendicular dot (2D cross/wedge) product `self.x * rhs.y - self.y * rhs.x`.
    #[inline(always)]
    #[must_use]
    pub fn perp_dot(self, rhs: Self) -> f32 {
        (self.x * rhs.y) - (self.y * rhs.x)
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

    /// Component-wise minimum.
    #[inline(always)]
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }

    /// Component-wise maximum.
    #[inline(always)]
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
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
        self.x.min(self.y)
    }

    /// Returns the largest component.
    #[inline(always)]
    #[must_use]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y)
    }

    /// Returns the sum of all components.
    #[inline(always)]
    #[must_use]
    pub fn element_sum(self) -> f32 {
        self.x + self.y
    }

    /// Component-wise absolute value.
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    /// Component-wise signum.
    #[inline(always)]
    #[must_use]
    pub fn signum(self) -> Self {
        Self::new(self.x.signum(), self.y.signum())
    }

    /// Component-wise `copysign`.
    #[inline(always)]
    #[must_use]
    pub fn copysign(self, rhs: Self) -> Self {
        Self::new(self.x.copysign(rhs.x), self.y.copysign(rhs.y))
    }

    /// Component-wise floor.
    #[inline(always)]
    #[must_use]
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor())
    }

    /// Component-wise ceil.
    #[inline(always)]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil())
    }

    /// Component-wise round.
    #[inline(always)]
    #[must_use]
    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round())
    }

    /// Component-wise trunc.
    #[inline(always)]
    #[must_use]
    pub fn trunc(self) -> Self {
        Self::new(self.x.trunc(), self.y.trunc())
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
        Self::new(self.x.recip(), self.y.recip())
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

    /// Rotates `self` by unit vector `rhs = (cos, sin)`.
    #[inline]
    #[must_use]
    pub fn rotate(self, rhs: Self) -> Self {
        Self::new(
            self.x * rhs.x - self.y * rhs.y,
            self.y * rhs.x + self.x * rhs.y,
        )
    }

    /// Returns `true` if both components are finite.
    #[inline(always)]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Returns `true` if any component is `NaN`.
    #[inline(always)]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    /// Returns `true` if `|self - rhs| <= max_abs_diff` for every component.
    #[inline]
    #[must_use]
    pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: f32) -> bool {
        let diff = (self - rhs).abs();
        diff.x <= max_abs_diff && diff.y <= max_abs_diff
    }

    /// Component-wise equality mask `self == rhs`.
    #[inline(always)]
    #[must_use]
    pub fn cmpeq(self, rhs: Self) -> BVec2 {
        BVec2::new(self.x == rhs.x, self.y == rhs.y)
    }

    /// Component-wise inequality mask `self != rhs`.
    #[inline(always)]
    #[must_use]
    pub fn cmpne(self, rhs: Self) -> BVec2 {
        BVec2::new(self.x != rhs.x, self.y != rhs.y)
    }

    /// Component-wise `self < rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmplt(self, rhs: Self) -> BVec2 {
        BVec2::new(self.x < rhs.x, self.y < rhs.y)
    }

    /// Component-wise `self <= rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmple(self, rhs: Self) -> BVec2 {
        BVec2::new(self.x <= rhs.x, self.y <= rhs.y)
    }

    /// Component-wise `self > rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmpgt(self, rhs: Self) -> BVec2 {
        BVec2::new(self.x > rhs.x, self.y > rhs.y)
    }

    /// Component-wise `self >= rhs` mask.
    #[inline(always)]
    #[must_use]
    pub fn cmpge(self, rhs: Self) -> BVec2 {
        BVec2::new(self.x >= rhs.x, self.y >= rhs.y)
    }
}

/// 2-lane boolean mask returned by [`Vec2`] component-wise comparisons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BVec2 {
    /// X lane boolean result.
    pub x: bool,
    /// Y lane boolean result.
    pub y: bool,
}

impl BVec2 {
    /// Creates a new 2-lane boolean mask.
    #[inline(always)]
    #[must_use]
    pub const fn new(x: bool, y: bool) -> Self {
        Self { x, y }
    }

    /// Returns `true` if any lane is `true`.
    #[inline(always)]
    #[must_use]
    pub const fn any(self) -> bool {
        self.x || self.y
    }

    /// Returns `true` if all lanes are `true`.
    #[inline(always)]
    #[must_use]
    pub const fn all(self) -> bool {
        self.x && self.y
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
}

impl Index<usize> for Vec2 {
    type Output = f32;
    #[inline(always)]
    fn index(&self, index: usize) -> &f32 {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec2 index {index} out of bounds (len is 2)"),
        }
    }
}

impl IndexMut<usize> for Vec2 {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Vec2 index {index} out of bounds (len is 2)"),
        }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Add<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: f32) -> Self {
        Self::new(self.x + rhs, self.y + rhs)
    }
}

impl AddAssign for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl AddAssign<f32> for Vec2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Sub<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: f32) -> Self {
        Self::new(self.x - rhs, self.y - rhs)
    }
}

impl SubAssign for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl SubAssign<f32> for Vec2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
    }
}

impl Mul for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn mul(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self * rhs.x, self * rhs.y)
    }
}

impl MulAssign for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl MulAssign<f32> for Vec2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl Div<Vec2> for f32 {
    type Output = Vec2;
    #[inline(always)]
    fn div(self, rhs: Vec2) -> Vec2 {
        Vec2::new(self / rhs.x, self / rhs.y)
    }
}

impl DivAssign for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl DivAssign<f32> for Vec2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl From<[f32; 2]> for Vec2 {
    #[inline(always)]
    fn from(a: [f32; 2]) -> Self {
        Self::from_array(a)
    }
}

impl From<Vec2> for [f32; 2] {
    #[inline(always)]
    fn from(v: Vec2) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32)> for Vec2 {
    #[inline(always)]
    fn from((x, y): (f32, f32)) -> Self {
        Self::new(x, y)
    }
}

impl From<Vec2> for (f32, f32) {
    #[inline(always)]
    fn from(v: Vec2) -> Self {
        (v.x, v.y)
    }
}

impl Sum for Vec2 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<'a> Sum<&'a Self> for Vec2 {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, &b| a + b)
    }
}

impl Product for Vec2 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}
