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
//! - Arithmetic NaN propagation and invalid-operation NaNs (`0.0 / 0.0`, `∞ - ∞`, `0.0 * ∞`,
//!   `sqrt(-x)`) follow SSE2's exact bit rules (`0xffc0_0000` for generated NaNs, quieted first
//!   operand NaN, then quieted second operand NaN).
//! - Comparisons give masks with every bit of a lane set or clear; `blend`, `and`, `any_true`
//!   take such masks.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Lanes in a [`FloatW`].
pub const LANES: usize = 4;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(miri)))]
mod native {
    use std::arch::x86_64::{
        __m128, _mm_and_ps, _mm_andnot_ps, _mm_cmpeq_ps, _mm_cmpgt_ps, _mm_cmplt_ps, _mm_max_ps,
        _mm_min_ps, _mm_movemask_ps, _mm_or_ps, _mm_set1_ps, _mm_setr_ps, _mm_setzero_ps,
        _mm_storeu_ps, _mm_xor_ps,
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
            let mut out = self.0;
            // SAFETY: `target_feature = "sse2"` is enabled; `sqrtps` operates purely on `xmm`
            // registers without touching memory, stack, or RFLAGS.
            unsafe {
                std::arch::asm!(
                    "sqrtps {0}, {0}",
                    inout(xmm_reg) out,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(out)
        }

        #[inline(always)]
        pub fn add_w(self, b: Self) -> Self {
            let mut out = self.0;
            // SAFETY: `target_feature = "sse2"` is enabled; `addps` operates purely on `xmm`
            // registers without touching memory, stack, or RFLAGS, avoiding LLVM target-independent
            // IR constant-folding that would alter SSE2 NaN sign bits on constant inputs.
            unsafe {
                std::arch::asm!(
                    "addps {0}, {1}",
                    inout(xmm_reg) out,
                    in(xmm_reg) b.0,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(out)
        }

        #[inline(always)]
        pub fn sub_w(self, b: Self) -> Self {
            let mut out = self.0;
            // SAFETY: `target_feature = "sse2"` is enabled; `subps` operates purely on `xmm`
            // registers without touching memory, stack, or RFLAGS.
            unsafe {
                std::arch::asm!(
                    "subps {0}, {1}",
                    inout(xmm_reg) out,
                    in(xmm_reg) b.0,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(out)
        }

        #[inline(always)]
        pub fn mul_w(self, b: Self) -> Self {
            let mut out = self.0;
            // SAFETY: `target_feature = "sse2"` is enabled; `mulps` operates purely on `xmm`
            // registers without touching memory, stack, or RFLAGS.
            unsafe {
                std::arch::asm!(
                    "mulps {0}, {1}",
                    inout(xmm_reg) out,
                    in(xmm_reg) b.0,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(out)
        }

        #[inline(always)]
        pub fn div_w(self, b: Self) -> Self {
            let mut out = self.0;
            // SAFETY: `target_feature = "sse2"` is enabled; `divps` operates purely on `xmm`
            // registers without touching memory, stack, or RFLAGS.
            unsafe {
                std::arch::asm!(
                    "divps {0}, {1}",
                    inout(xmm_reg) out,
                    in(xmm_reg) b.0,
                    options(pure, nomem, nostack, preserves_flags),
                );
            }
            Self(out)
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

#[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
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
    /// SSE2 floating-point indefinite value (`-qNaN`, `0xffc0_0000`) returned when an arithmetic
    /// instruction raises an Invalid Operation exception on non-NaN operands.
    const SSE2_INDEFINITE_NAN: u32 = 0xffc0_0000;
    /// IEEE-754 binary32 quiet NaN bit.
    const QUIET_NAN_BIT: u32 = 0x0040_0000;

    #[inline(always)]
    fn sse2_unary_nan(x: f32, res: f32) -> f32 {
        if res.is_nan() {
            if x.is_nan() {
                f32::from_bits(x.to_bits() | QUIET_NAN_BIT)
            } else {
                f32::from_bits(SSE2_INDEFINITE_NAN)
            }
        } else {
            res
        }
    }

    #[inline(always)]
    fn sse2_binary_nan(x: f32, y: f32, res: f32) -> f32 {
        if res.is_nan() {
            if x.is_nan() {
                f32::from_bits(x.to_bits() | QUIET_NAN_BIT)
            } else if y.is_nan() {
                f32::from_bits(y.to_bits() | QUIET_NAN_BIT)
            } else {
                f32::from_bits(SSE2_INDEFINITE_NAN)
            }
        } else {
            res
        }
    }

    /// Four lanes in an array, aligned as a SIMD register.
    #[derive(Clone, Copy, Debug)]
    #[repr(C, align(16))]
    pub struct FloatW([f32; 4]);

    /// A lane of a mask: every bit set or clear.
    #[inline(always)]
    const fn mask_lane(set: bool) -> f32 {
        f32::from_bits(if set { u32::MAX } else { 0 })
    }

    impl FloatW {
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

        /// Lanes from an array.
        #[inline(always)]
        #[must_use]
        pub fn from_array(lanes: [f32; 4]) -> Self {
            Self(lanes)
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
            let a = self.0;
            Self([-a[0], -a[1], -a[2], -a[3]])
        }

        /// Clears each lane's sign bit.
        #[inline(always)]
        #[must_use]
        pub fn abs(self) -> Self {
            let a = self.0;
            Self([a[0].abs(), a[1].abs(), a[2].abs(), a[3].abs()])
        }

        /// Lane-wise square root.
        #[inline(always)]
        #[must_use]
        pub fn sqrt(self) -> Self {
            let a = self.0;
            Self([
                sse2_unary_nan(a[0], a[0].sqrt()),
                sse2_unary_nan(a[1], a[1].sqrt()),
                sse2_unary_nan(a[2], a[2].sqrt()),
                sse2_unary_nan(a[3], a[3].sqrt()),
            ])
        }

        /// Lane-wise `a + b`.
        #[inline(always)]
        #[must_use]
        pub fn add_w(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                sse2_binary_nan(a[0], b[0], a[0] + b[0]),
                sse2_binary_nan(a[1], b[1], a[1] + b[1]),
                sse2_binary_nan(a[2], b[2], a[2] + b[2]),
                sse2_binary_nan(a[3], b[3], a[3] + b[3]),
            ])
        }

        /// Lane-wise `a - b`.
        #[inline(always)]
        #[must_use]
        pub fn sub_w(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                sse2_binary_nan(a[0], b[0], a[0] - b[0]),
                sse2_binary_nan(a[1], b[1], a[1] - b[1]),
                sse2_binary_nan(a[2], b[2], a[2] - b[2]),
                sse2_binary_nan(a[3], b[3], a[3] - b[3]),
            ])
        }

        /// Lane-wise `a * b`.
        #[inline(always)]
        #[must_use]
        pub fn mul_w(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                sse2_binary_nan(a[0], b[0], a[0] * b[0]),
                sse2_binary_nan(a[1], b[1], a[1] * b[1]),
                sse2_binary_nan(a[2], b[2], a[2] * b[2]),
                sse2_binary_nan(a[3], b[3], a[3] * b[3]),
            ])
        }

        /// Lane-wise `a / b`.
        #[inline(always)]
        #[must_use]
        pub fn div_w(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                sse2_binary_nan(a[0], b[0], a[0] / b[0]),
                sse2_binary_nan(a[1], b[1], a[1] / b[1]),
                sse2_binary_nan(a[2], b[2], a[2] / b[2]),
                sse2_binary_nan(a[3], b[3], a[3] / b[3]),
            ])
        }

        /// Lane-wise `a < b ? a : b`, as SSE2's `minps`.
        #[inline(always)]
        #[must_use]
        pub fn min(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                if a[0] < b[0] { a[0] } else { b[0] },
                if a[1] < b[1] { a[1] } else { b[1] },
                if a[2] < b[2] { a[2] } else { b[2] },
                if a[3] < b[3] { a[3] } else { b[3] },
            ])
        }

        /// Lane-wise `a > b ? a : b`, as SSE2's `maxps`.
        #[inline(always)]
        #[must_use]
        pub fn max(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                if a[0] > b[0] { a[0] } else { b[0] },
                if a[1] > b[1] { a[1] } else { b[1] },
                if a[2] > b[2] { a[2] } else { b[2] },
                if a[3] > b[3] { a[3] } else { b[3] },
            ])
        }

        /// Lane-wise bitwise and.
        #[inline(always)]
        #[must_use]
        pub fn and(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                f32::from_bits(a[0].to_bits() & b[0].to_bits()),
                f32::from_bits(a[1].to_bits() & b[1].to_bits()),
                f32::from_bits(a[2].to_bits() & b[2].to_bits()),
                f32::from_bits(a[3].to_bits() & b[3].to_bits()),
            ])
        }

        /// Lane-wise bitwise or.
        #[inline(always)]
        #[must_use]
        pub fn or(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                f32::from_bits(a[0].to_bits() | b[0].to_bits()),
                f32::from_bits(a[1].to_bits() | b[1].to_bits()),
                f32::from_bits(a[2].to_bits() | b[2].to_bits()),
                f32::from_bits(a[3].to_bits() | b[3].to_bits()),
            ])
        }

        /// Mask of the lanes where `a > b`.
        #[inline(always)]
        #[must_use]
        pub fn greater_than(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                mask_lane(a[0] > b[0]),
                mask_lane(a[1] > b[1]),
                mask_lane(a[2] > b[2]),
                mask_lane(a[3] > b[3]),
            ])
        }

        /// Mask of the lanes where `a < b`.
        #[inline(always)]
        #[must_use]
        pub fn less_than(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                mask_lane(a[0] < b[0]),
                mask_lane(a[1] < b[1]),
                mask_lane(a[2] < b[2]),
                mask_lane(a[3] < b[3]),
            ])
        }

        /// Mask of the lanes where `a == b`.
        #[inline(always)]
        #[must_use]
        pub fn equals(self, b: Self) -> Self {
            let (a, b) = (self.0, b.0);
            Self([
                mask_lane(a[0] == b[0]),
                mask_lane(a[1] == b[1]),
                mask_lane(a[2] == b[2]),
                mask_lane(a[3] == b[3]),
            ])
        }

        /// Whether any lane has its sign bit set, as SSE2's `movemask`.
        #[inline(always)]
        #[must_use]
        pub fn any_true(self) -> bool {
            let a = self.0;
            a[0].is_sign_negative()
                || a[1].is_sign_negative()
                || a[2].is_sign_negative()
                || a[3].is_sign_negative()
        }

        /// Whether every lane equals zero (either sign).
        #[inline(always)]
        #[must_use]
        pub fn all_zero(self) -> bool {
            let a = self.0;
            a[0] == 0.0 && a[1] == 0.0 && a[2] == 0.0 && a[3] == 0.0
        }

        /// `b` in the lanes `mask` sets, `self` in the others, bit by bit.
        #[inline(always)]
        #[must_use]
        pub fn blend(self, b: Self, mask: Self) -> Self {
            let (a, b, m) = (self.0, b.0, mask.0);
            Self([
                f32::from_bits(
                    (m[0].to_bits() & b[0].to_bits()) | (!m[0].to_bits() & a[0].to_bits()),
                ),
                f32::from_bits(
                    (m[1].to_bits() & b[1].to_bits()) | (!m[1].to_bits() & a[1].to_bits()),
                ),
                f32::from_bits(
                    (m[2].to_bits() & b[2].to_bits()) | (!m[2].to_bits() & a[2].to_bits()),
                ),
                f32::from_bits(
                    (m[3].to_bits() & b[3].to_bits()) | (!m[3].to_bits() & a[3].to_bits()),
                ),
            ])
        }
    }

    impl std::ops::Neg for FloatW {
        type Output = Self;

        #[inline(always)]
        fn neg(self) -> Self {
            self.neg_w()
        }
    }

    impl std::ops::Add for FloatW {
        type Output = Self;

        #[inline(always)]
        fn add(self, b: Self) -> Self {
            self.add_w(b)
        }
    }

    impl std::ops::Sub for FloatW {
        type Output = Self;

        #[inline(always)]
        fn sub(self, b: Self) -> Self {
            self.sub_w(b)
        }
    }

    impl std::ops::Mul for FloatW {
        type Output = Self;

        #[inline(always)]
        fn mul(self, b: Self) -> Self {
            self.mul_w(b)
        }
    }

    impl std::ops::Div for FloatW {
        type Output = Self;

        #[inline(always)]
        fn div(self, b: Self) -> Self {
            self.div_w(b)
        }
    }
}

#[cfg(any(
    miri,
    not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))
))]
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
