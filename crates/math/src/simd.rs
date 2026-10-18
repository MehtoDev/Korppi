//! Four-lane `f32` vectors for wide solvers (Box2D `b2FloatW`, `contact_solver_wide.inl`).
//!
//! [`FloatW`] holds four floats and does the same IEEE operation on each lane: SSE2 on x86-64,
//! NEON on AArch64, and plain arrays elsewhere ([`portable::FloatW`]). Every backend gives the
//! same bits as SSE2, so a wide solver's results do not depend on the platform:
//!
//! - `mul_add` and `mul_sub` round the product before the sum, as Box2D's `b2MulAddW`; they
//!   never fuse.
//! - `min` and `max` follow SSE2's `minps` and `maxps`: `a.min(b)` is `a` only where `a < b`,
//!   else `b`, so the second operand wins for equal values (`-0.0` and `0.0`) and for NaN.
//!   Box2D's NEON and scalar fallbacks differ there; Korppi's do not.
//! - Comparisons give masks with every bit of a lane set or clear; `blend`, `and`, `any_true`
//!   take such masks.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Lanes in a [`FloatW`].
pub const LANES: usize = 4;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod native {
    use std::arch::x86_64::{
        __m128, _mm_add_ps, _mm_and_ps, _mm_andnot_ps, _mm_cmpeq_ps, _mm_cmpgt_ps, _mm_cmplt_ps,
        _mm_div_ps, _mm_max_ps, _mm_min_ps, _mm_movemask_ps, _mm_mul_ps, _mm_or_ps, _mm_set1_ps,
        _mm_setr_ps, _mm_setzero_ps, _mm_sqrt_ps, _mm_storeu_ps, _mm_sub_ps, _mm_xor_ps,
    };

    /// Runs SSE2 intrinsics that take and return values.
    macro_rules! sse2 {
        ($e:expr) => {
            // SAFETY: this backend is compiled only with `target_feature = "sse2"`, so the
            // CPU running it has every SSE2 instruction; the intrinsics read no memory.
            unsafe { $e }
        };
    }

    /// Four lanes in an SSE2 register.
    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct FloatW(__m128);

    impl FloatW {
        #[inline(always)]
        pub fn zero() -> Self {
            Self(sse2!(_mm_setzero_ps()))
        }

        #[inline(always)]
        pub fn splat(value: f32) -> Self {
            Self(sse2!(_mm_set1_ps(value)))
        }

        #[inline(always)]
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            Self(sse2!(_mm_setr_ps(a, b, c, d)))
        }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut lanes = [0.0; 4];
            // SAFETY: SSE2 is enabled (see `sse2!`), and `lanes` is four writable floats;
            // `storeu` needs no alignment.
            unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), self.0) };
            lanes
        }

        #[inline(always)]
        pub fn neg_w(self) -> Self {
            Self(sse2!(_mm_xor_ps(self.0, _mm_set1_ps(-0.0))))
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self(sse2!(_mm_andnot_ps(_mm_set1_ps(-0.0), self.0)))
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self(sse2!(_mm_sqrt_ps(self.0)))
        }

        #[inline(always)]
        pub fn add_w(self, b: Self) -> Self {
            Self(sse2!(_mm_add_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn sub_w(self, b: Self) -> Self {
            Self(sse2!(_mm_sub_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn mul_w(self, b: Self) -> Self {
            Self(sse2!(_mm_mul_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn div_w(self, b: Self) -> Self {
            Self(sse2!(_mm_div_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn min(self, b: Self) -> Self {
            Self(sse2!(_mm_min_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn max(self, b: Self) -> Self {
            Self(sse2!(_mm_max_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn and(self, b: Self) -> Self {
            Self(sse2!(_mm_and_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn or(self, b: Self) -> Self {
            Self(sse2!(_mm_or_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn greater_than(self, b: Self) -> Self {
            Self(sse2!(_mm_cmpgt_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn less_than(self, b: Self) -> Self {
            Self(sse2!(_mm_cmplt_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn equals(self, b: Self) -> Self {
            Self(sse2!(_mm_cmpeq_ps(self.0, b.0)))
        }

        #[inline(always)]
        pub fn any_true(self) -> bool {
            sse2!(_mm_movemask_ps(self.0)) != 0
        }

        #[inline(always)]
        pub fn all_zero(self) -> bool {
            sse2!(_mm_movemask_ps(_mm_cmpeq_ps(self.0, _mm_setzero_ps()))) == 0xF
        }

        #[inline(always)]
        pub fn blend(self, b: Self, mask: Self) -> Self {
            Self(sse2!(_mm_or_ps(
                _mm_and_ps(mask.0, b.0),
                _mm_andnot_ps(mask.0, self.0)
            )))
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod native {
    use std::arch::aarch64::{
        float32x4_t, vabsq_f32, vaddq_f32, vandq_u32, vbslq_f32, vceqq_f32, vcgtq_f32, vcltq_f32,
        vdivq_f32, vdupq_n_f32, vld1q_f32, vmaxvq_u32, vminvq_u32, vmulq_f32, vnegq_f32, vorrq_u32,
        vreinterpretq_f32_u32, vreinterpretq_u32_f32, vshrq_n_u32, vsqrtq_f32, vst1q_f32,
        vsubq_f32,
    };

    /// Runs NEON intrinsics that take and return values.
    macro_rules! neon {
        ($e:expr) => {
            // SAFETY: this backend is compiled only with `target_feature = "neon"`, so the
            // CPU running it has every NEON instruction; the intrinsics read no memory.
            unsafe { $e }
        };
    }

    /// Four lanes in a NEON register.
    #[derive(Clone, Copy, Debug)]
    #[repr(transparent)]
    pub struct FloatW(float32x4_t);

    impl FloatW {
        #[inline(always)]
        pub fn zero() -> Self {
            Self(neon!(vdupq_n_f32(0.0)))
        }

        #[inline(always)]
        pub fn splat(value: f32) -> Self {
            Self(neon!(vdupq_n_f32(value)))
        }

        #[inline(always)]
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            let lanes = [a, b, c, d];
            // SAFETY: NEON is enabled (see `neon!`), and `lanes` is four readable floats;
            // `vld1q_f32` needs no alignment.
            Self(unsafe { vld1q_f32(lanes.as_ptr()) })
        }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 4] {
            let mut lanes = [0.0; 4];
            // SAFETY: NEON is enabled (see `neon!`), and `lanes` is four writable floats;
            // `vst1q_f32` needs no alignment.
            unsafe { vst1q_f32(lanes.as_mut_ptr(), self.0) };
            lanes
        }

        #[inline(always)]
        pub fn neg_w(self) -> Self {
            Self(neon!(vnegq_f32(self.0)))
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            Self(neon!(vabsq_f32(self.0)))
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            Self(neon!(vsqrtq_f32(self.0)))
        }

        #[inline(always)]
        pub fn add_w(self, b: Self) -> Self {
            Self(neon!(vaddq_f32(self.0, b.0)))
        }

        #[inline(always)]
        pub fn sub_w(self, b: Self) -> Self {
            Self(neon!(vsubq_f32(self.0, b.0)))
        }

        #[inline(always)]
        pub fn mul_w(self, b: Self) -> Self {
            Self(neon!(vmulq_f32(self.0, b.0)))
        }

        #[inline(always)]
        pub fn div_w(self, b: Self) -> Self {
            Self(neon!(vdivq_f32(self.0, b.0)))
        }

        /// SSE2's `minps`: `a` where `a < b`, else `b` (NEON's `vminq` differs on zeros and NaN).
        #[inline(always)]
        pub fn min(self, b: Self) -> Self {
            Self(neon!(vbslq_f32(vcltq_f32(self.0, b.0), self.0, b.0)))
        }

        /// SSE2's `maxps`: `a` where `a > b`, else `b`.
        #[inline(always)]
        pub fn max(self, b: Self) -> Self {
            Self(neon!(vbslq_f32(vcgtq_f32(self.0, b.0), self.0, b.0)))
        }

        #[inline(always)]
        pub fn and(self, b: Self) -> Self {
            Self(neon!(vreinterpretq_f32_u32(vandq_u32(
                vreinterpretq_u32_f32(self.0),
                vreinterpretq_u32_f32(b.0)
            ))))
        }

        #[inline(always)]
        pub fn or(self, b: Self) -> Self {
            Self(neon!(vreinterpretq_f32_u32(vorrq_u32(
                vreinterpretq_u32_f32(self.0),
                vreinterpretq_u32_f32(b.0)
            ))))
        }

        #[inline(always)]
        pub fn greater_than(self, b: Self) -> Self {
            Self(neon!(vreinterpretq_f32_u32(vcgtq_f32(self.0, b.0))))
        }

        #[inline(always)]
        pub fn less_than(self, b: Self) -> Self {
            Self(neon!(vreinterpretq_f32_u32(vcltq_f32(self.0, b.0))))
        }

        #[inline(always)]
        pub fn equals(self, b: Self) -> Self {
            Self(neon!(vreinterpretq_f32_u32(vceqq_f32(self.0, b.0))))
        }

        /// Whether a lane's sign bit is set, as SSE2's `movemask`.
        #[inline(always)]
        pub fn any_true(self) -> bool {
            neon!(vmaxvq_u32(vshrq_n_u32::<31>(vreinterpretq_u32_f32(self.0)))) != 0
        }

        #[inline(always)]
        pub fn all_zero(self) -> bool {
            neon!(vminvq_u32(vceqq_f32(self.0, vdupq_n_f32(0.0)))) != 0
        }

        #[inline(always)]
        pub fn blend(self, b: Self, mask: Self) -> Self {
            Self(neon!(vbslq_f32(vreinterpretq_u32_f32(mask.0), b.0, self.0)))
        }
    }
}

/// The plain-array backend: the same results as SSE2, lane by lane. It is the [`FloatW`] of
/// targets without SSE2 or NEON, and is public so tests can hold the native backend to it.
pub mod portable {
    /// Four lanes in an array, aligned as a SIMD register.
    #[derive(Clone, Copy, Debug)]
    #[repr(C, align(16))]
    pub struct FloatW([f32; 4]);

    /// A lane of a mask: every bit set or clear.
    const fn mask_lane(set: bool) -> f32 {
        f32::from_bits(if set { u32::MAX } else { 0 })
    }

    impl FloatW {
        #[inline(always)]
        fn map(self, b: Self, f: impl Fn(f32, f32) -> f32) -> Self {
            let (a, b) = (self.0, b.0);
            Self([f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])])
        }

        #[inline(always)]
        fn bits(self, b: Self, f: impl Fn(u32, u32) -> u32) -> Self {
            self.map(b, |x, y| f32::from_bits(f(x.to_bits(), y.to_bits())))
        }

        /// All lanes zero.
        #[inline(always)]
        #[must_use]
        pub fn zero() -> Self {
            Self([0.0; 4])
        }

        /// All lanes `value`.
        #[inline(always)]
        #[must_use]
        pub fn splat(value: f32) -> Self {
            Self([value; 4])
        }

        /// Lanes `a`, `b`, `c`, `d`.
        #[inline(always)]
        #[must_use]
        pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
            Self([a, b, c, d])
        }

        /// The lanes.
        #[inline(always)]
        #[must_use]
        pub fn to_array(self) -> [f32; 4] {
            self.0
        }

        /// Flips each lane's sign bit.
        #[inline(always)]
        #[must_use]
        pub fn neg_w(self) -> Self {
            Self(self.0.map(|x| -x))
        }

        /// Clears each lane's sign bit.
        #[inline(always)]
        #[must_use]
        pub fn abs(self) -> Self {
            Self(self.0.map(f32::abs))
        }

        /// Lane-wise square root.
        #[inline(always)]
        #[must_use]
        pub fn sqrt(self) -> Self {
            Self(self.0.map(f32::sqrt))
        }

        /// Lane-wise `a + b`.
        #[inline(always)]
        #[must_use]
        pub fn add_w(self, b: Self) -> Self {
            self.map(b, |x, y| x + y)
        }

        /// Lane-wise `a - b`.
        #[inline(always)]
        #[must_use]
        pub fn sub_w(self, b: Self) -> Self {
            self.map(b, |x, y| x - y)
        }

        /// Lane-wise `a * b`.
        #[inline(always)]
        #[must_use]
        pub fn mul_w(self, b: Self) -> Self {
            self.map(b, |x, y| x * y)
        }

        /// Lane-wise `a / b`.
        #[inline(always)]
        #[must_use]
        pub fn div_w(self, b: Self) -> Self {
            self.map(b, |x, y| x / y)
        }

        /// Lane-wise `a < b ? a : b`, as SSE2's `minps`.
        #[inline(always)]
        #[must_use]
        pub fn min(self, b: Self) -> Self {
            self.map(b, |x, y| if x < y { x } else { y })
        }

        /// Lane-wise `a > b ? a : b`, as SSE2's `maxps`.
        #[inline(always)]
        #[must_use]
        pub fn max(self, b: Self) -> Self {
            self.map(b, |x, y| if x > y { x } else { y })
        }

        /// Lane-wise bitwise and.
        #[inline(always)]
        #[must_use]
        pub fn and(self, b: Self) -> Self {
            self.bits(b, |x, y| x & y)
        }

        /// Lane-wise bitwise or.
        #[inline(always)]
        #[must_use]
        pub fn or(self, b: Self) -> Self {
            self.bits(b, |x, y| x | y)
        }

        /// Mask of the lanes where `a > b`.
        #[inline(always)]
        #[must_use]
        pub fn greater_than(self, b: Self) -> Self {
            self.map(b, |x, y| mask_lane(x > y))
        }

        /// Mask of the lanes where `a < b`.
        #[inline(always)]
        #[must_use]
        pub fn less_than(self, b: Self) -> Self {
            self.map(b, |x, y| mask_lane(x < y))
        }

        /// Mask of the lanes where `a == b`.
        #[inline(always)]
        #[must_use]
        pub fn equals(self, b: Self) -> Self {
            self.map(b, |x, y| mask_lane(x == y))
        }

        /// Whether any lane has its sign bit set, as SSE2's `movemask`.
        #[inline(always)]
        #[must_use]
        pub fn any_true(self) -> bool {
            self.0.iter().any(|x| x.is_sign_negative())
        }

        /// Whether every lane equals zero (either sign).
        #[inline(always)]
        #[must_use]
        pub fn all_zero(self) -> bool {
            self.0.iter().all(|&x| x == 0.0)
        }

        /// `b` in the lanes `mask` sets, `self` in the others, bit by bit.
        #[inline(always)]
        #[must_use]
        pub fn blend(self, b: Self, mask: Self) -> Self {
            let picked = mask.bits(b, |m, y| m & y);
            let kept = mask.bits(self, |m, x| !m & x);
            picked.bits(kept, |x, y| x | y)
        }
    }

    impl std::ops::Neg for FloatW {
        type Output = Self;

        fn neg(self) -> Self {
            self.neg_w()
        }
    }

    impl std::ops::Add for FloatW {
        type Output = Self;

        fn add(self, b: Self) -> Self {
            self.add_w(b)
        }
    }

    impl std::ops::Sub for FloatW {
        type Output = Self;

        fn sub(self, b: Self) -> Self {
            self.sub_w(b)
        }
    }

    impl std::ops::Mul for FloatW {
        type Output = Self;

        fn mul(self, b: Self) -> Self {
            self.mul_w(b)
        }
    }

    impl std::ops::Div for FloatW {
        type Output = Self;

        fn div(self, b: Self) -> Self {
            self.div_w(b)
        }
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
mod native {
    pub use super::portable::FloatW;
}

/// Four `f32` lanes (Box2D `b2FloatW`), with SSE2 results on every platform.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct FloatW(native::FloatW);

impl Default for FloatW {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl FloatW {
    /// All lanes zero (Box2D `b2ZeroW`).
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self(native::FloatW::zero())
    }

    /// All lanes `value` (Box2D `b2SplatW`).
    #[inline(always)]
    #[must_use]
    pub fn splat(value: f32) -> Self {
        Self(native::FloatW::splat(value))
    }

    /// Lanes `a`, `b`, `c`, `d` (Box2D `b2SetW`).
    #[inline(always)]
    #[must_use]
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self(native::FloatW::new(a, b, c, d))
    }

    /// Lanes from an array.
    #[inline(always)]
    #[must_use]
    pub fn from_array(lanes: [f32; 4]) -> Self {
        Self::new(lanes[0], lanes[1], lanes[2], lanes[3])
    }

    /// The lanes (Box2D `b2StoreW`).
    #[inline(always)]
    #[must_use]
    pub fn to_array(self) -> [f32; 4] {
        self.0.to_array()
    }

    /// Clears each lane's sign bit (Box3D `b3AbsW`).
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Lane-wise square root (Box3D `b3SqrtW`).
    #[inline(always)]
    #[must_use]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// `self + b * c`, the product rounded first (Box2D `b2MulAddW`).
    #[inline(always)]
    #[must_use]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.add_w(b.0.mul_w(c.0)))
    }

    /// `self - b * c`, the product rounded first (Box2D `b2MulSubW`).
    #[inline(always)]
    #[must_use]
    pub fn mul_sub(self, b: Self, c: Self) -> Self {
        Self(self.0.sub_w(b.0.mul_w(c.0)))
    }

    /// Lane-wise `self < b ? self : b` (Box2D `b2MinW` on SSE2).
    #[inline(always)]
    #[must_use]
    pub fn min(self, b: Self) -> Self {
        Self(self.0.min(b.0))
    }

    /// Lane-wise `self > b ? self : b` (Box2D `b2MaxW` on SSE2).
    #[inline(always)]
    #[must_use]
    pub fn max(self, b: Self) -> Self {
        Self(self.0.max(b.0))
    }

    /// `max(-b, min(self, b))` (Box2D `b2SymClampW`).
    #[inline(always)]
    #[must_use]
    pub fn sym_clamp(self, b: Self) -> Self {
        Self(b.0.neg_w().max(self.0.min(b.0)))
    }

    /// `max(low, min(self, high))` (Box3D `b3ClampW`).
    #[inline(always)]
    #[must_use]
    pub fn clamp_w(self, low: Self, high: Self) -> Self {
        Self(low.0.max(self.0.min(high.0)))
    }

    /// Lane-wise bitwise and of two masks (Box2D `b2AndW`).
    #[inline(always)]
    #[must_use]
    pub fn and(self, b: Self) -> Self {
        Self(self.0.and(b.0))
    }

    /// Lane-wise bitwise or of two masks (Box3D `b3OrW`).
    #[inline(always)]
    #[must_use]
    pub fn or(self, b: Self) -> Self {
        Self(self.0.or(b.0))
    }

    /// Mask of the lanes where `self > b` (Box2D `b2GreaterThanW`).
    #[inline(always)]
    #[must_use]
    pub fn greater_than(self, b: Self) -> Self {
        Self(self.0.greater_than(b.0))
    }

    /// Mask of the lanes where `self < b` (Box2D `b2LessThanW`).
    #[inline(always)]
    #[must_use]
    pub fn less_than(self, b: Self) -> Self {
        Self(self.0.less_than(b.0))
    }

    /// Mask of the lanes where `self == b` (Box3D `b3EqualsW`).
    #[inline(always)]
    #[must_use]
    pub fn equals(self, b: Self) -> Self {
        Self(self.0.equals(b.0))
    }

    /// Whether any lane of a mask is set (Box2D `b2AnyTrueW`).
    #[inline(always)]
    #[must_use]
    pub fn any_true(self) -> bool {
        self.0.any_true()
    }

    /// Whether every lane equals zero (Box2D `b2AllZeroW`).
    #[inline(always)]
    #[must_use]
    pub fn all_zero(self) -> bool {
        self.0.all_zero()
    }

    /// `b` in the lanes `mask` sets, `self` in the others (Box2D `b2BlendW(self, b, mask)`).
    #[inline(always)]
    #[must_use]
    pub fn blend(self, b: Self, mask: Self) -> Self {
        Self(self.0.blend(b.0, mask.0))
    }
}

impl Neg for FloatW {
    type Output = Self;

    /// Flips each lane's sign bit (Box2D `b2NegW`).
    #[inline(always)]
    fn neg(self) -> Self {
        Self(self.0.neg_w())
    }
}

impl Add for FloatW {
    type Output = Self;

    #[inline(always)]
    fn add(self, b: Self) -> Self {
        Self(self.0.add_w(b.0))
    }
}

impl Sub for FloatW {
    type Output = Self;

    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        Self(self.0.sub_w(b.0))
    }
}

impl Mul for FloatW {
    type Output = Self;

    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        Self(self.0.mul_w(b.0))
    }
}

impl Div for FloatW {
    type Output = Self;

    #[inline(always)]
    fn div(self, b: Self) -> Self {
        Self(self.0.div_w(b.0))
    }
}
