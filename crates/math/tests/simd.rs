//! `FloatW` gives SSE2's results on every backend (Box2D `b2FloatW`): non-fused multiply-add,
//! `minps`/`maxps` operand order for equal values and NaN, and all-bits masks. The native
//! backend (SSE2 on x86-64, NEON on AArch64) must match the portable one bit for bit, so a wide
//! solver steps the same on every platform.

use korppi_math::simd::{portable, FloatW};

/// Inputs that tell rounding, signed zeros, NaN and infinities apart.
const SPECIAL: [f32; 12] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    -3.75,
    1.0e-38,
    f32::MAX,
    f32::MIN_POSITIVE,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

/// A small deterministic generator (xorshift32) of finite and special floats.
struct Floats(u32);

impl Floats {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        match self.0 % 8 {
            0 => SPECIAL[(self.0 >> 8) as usize % SPECIAL.len()],
            1 => f32::from_bits(self.0),
            _ => (self.0 >> 8) as f32 / 65536.0 - 128.0,
        }
    }

    fn lanes(&mut self) -> [f32; 4] {
        [self.next(), self.next(), self.next(), self.next()]
    }
}

fn bits(lanes: [f32; 4]) -> [u32; 4] {
    lanes.map(f32::to_bits)
}

fn native(lanes: [f32; 4]) -> FloatW {
    FloatW::from_array(lanes)
}

fn port(lanes: [f32; 4]) -> portable::FloatW {
    portable::FloatW::new(lanes[0], lanes[1], lanes[2], lanes[3])
}

#[test]
fn the_native_backend_matches_the_portable_one_bit_for_bit() {
    let mut floats = Floats(0x1234_5678);
    for _ in 0..20_000 {
        let (a, b, c) = (floats.lanes(), floats.lanes(), floats.lanes());
        let (na, nb, nc) = (native(a), native(b), native(c));
        let (pa, pb, pc) = (port(a), port(b), port(c));
        let pairs = [
            (na + nb, pa + pb),
            (na - nb, pa - pb),
            (na * nb, pa * pb),
            (na / nb, pa / pb),
            (-na, -pa),
            (na.mul_add(nb, nc), pa + pb * pc),
            (na.mul_sub(nb, nc), pa - pb * pc),
            (na.min(nb), pa.min(pb)),
            (na.max(nb), pa.max(pb)),
            (na.sym_clamp(nb), (-pb).max(pa.min(pb))),
            (na.greater_than(nb), pa.greater_than(pb)),
            (na.less_than(nb), pa.less_than(pb)),
            (
                na.greater_than(nb).and(nb.less_than(nc)),
                pa.greater_than(pb).and(pb.less_than(pc)),
            ),
            (
                na.blend(nb, nc.greater_than(na)),
                pa.blend(pb, pc.greater_than(pa)),
            ),
        ];
        for (index, (n, p)) in pairs.into_iter().enumerate() {
            assert_eq!(
                bits(n.to_array()),
                bits(p.to_array()),
                "operation {index} on {a:?} {b:?} {c:?}"
            );
        }
        let mask = na.greater_than(nb);
        assert_eq!(mask.any_true(), pa.greater_than(pb).any_true());
        assert_eq!(na.all_zero(), pa.all_zero());
    }
}

#[test]
fn operations_follow_sse2() {
    // Non-fused: 1 + 3 * (1/3) rounds the product first; a fused multiply-add would not.
    let third = 1.0f32 / 3.0;
    let fused = 3.0f32.mul_add(third, -1.0);
    let wide = FloatW::splat(-1.0).mul_add(FloatW::splat(3.0), FloatW::splat(third));
    assert_eq!(wide.to_array()[0], 3.0 * third - 1.0);
    assert_ne!(wide.to_array()[0], fused);

    // minps/maxps: the second operand for equal values and NaN.
    let a = FloatW::new(-0.0, 0.0, f32::NAN, 1.0);
    let b = FloatW::new(0.0, -0.0, 2.0, f32::NAN);
    assert_eq!(bits(a.min(b).to_array()), bits([0.0, -0.0, 2.0, f32::NAN]));
    assert_eq!(bits(a.max(b).to_array()), bits([0.0, -0.0, 2.0, f32::NAN]));

    // Masks set every bit; blend picks bit by bit.
    let mask = FloatW::new(1.0, 2.0, 3.0, 4.0).greater_than(FloatW::splat(2.5));
    assert_eq!(bits(mask.to_array()), [0, 0, u32::MAX, u32::MAX]);
    assert!(mask.any_true());
    assert!(!FloatW::zero().greater_than(FloatW::splat(1.0)).any_true());
    let picked = FloatW::splat(1.0).blend(FloatW::splat(2.0), mask);
    assert_eq!(picked.to_array(), [1.0, 1.0, 2.0, 2.0]);

    // Negation flips the sign bit only; zeros of either sign are zero.
    assert_eq!(bits((-FloatW::zero()).to_array()), [0x8000_0000; 4]);
    assert!(FloatW::new(0.0, -0.0, 0.0, -0.0).all_zero());
    assert!(!FloatW::new(0.0, 0.0, 1.0e-45, 0.0).all_zero());
    assert_eq!(
        FloatW::new(-5.0, 5.0, 0.5, -0.5)
            .sym_clamp(FloatW::splat(1.0))
            .to_array(),
        [-1.0, 1.0, 0.5, -0.5]
    );
}
