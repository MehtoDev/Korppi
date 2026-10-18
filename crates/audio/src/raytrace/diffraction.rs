//! Biot-Tolstoy-Medwin edge diffraction (FIX-7.5, ADR-051; findings AUD-08 and AUD-10).
//!
//! Sound bends around a wedge's edge. Biot and Tolstoy solved a point source near an infinite
//! wedge exactly; Medwin and Svensson made it a line integral along the edge, which holds for
//! finite edges too:
//!
//! `h(τ) = −(ν / 4π) ∫ δ(τ − (m + l)/c) β / (m l) dz`,
//!
//! with `ν = π/θ_W` for the wedge's exterior angle `θ_W`, `m` and `l` the distances from the
//! source to the edge point at `z` and from it to the receiver, and
//!
//! `β = Σ sin(νφ) / (cosh(νη) − cos(νφ))` over `φ = π ± θ_S ± θ_R`,
//! `η = acosh((m l + (z − z_S)(z − z_R)) / (r_S r_R))`,
//!
//! in cylindrical coordinates `(r, θ, z)` about the edge, `θ` from the first face through the air.
//! The direct sound is `δ(τ − d/c)/d` on the same scale. A sample of the response is the integral
//! over the edge points whose path takes a time within the sample's interval: [`Wedge`] finds them
//! by solving `m + l = cτ` at the interval's ends (a quadratic along the edge) and integrates
//! `β / (m l)` between, adaptively, since near the shadow and reflection boundaries the integrand
//! peaks sharply at the edge point of least time.
//!
//! Behind a thick obstacle no edge sees both the source and the receiver: the sound bends over
//! one edge, runs along the face to the next and bends again. Second-order diffraction is the
//! product of two edges' secondary sources (Svensson's edge diffraction toolbox):
//!
//! `h₂(τ) = ½ (ν₁/4π)(ν₂/4π) ∬ δ(τ − (m + n + l)/c) β₁ β₂ / (m n l) dz₁ dz₂`,
//!
//! with `n` the distance between the two edge points, `β₁` taking the point on the second edge as
//! the first edge's receiver and `β₂` the point on the first edge as the second's source, and the
//! ½ for the leg along the shared face. [`Wedge::second_order`] integrates it over element pairs,
//! each pair's contribution shared by linear interpolation between the two samples about its
//! time, as Svensson does.
//!
//! Inputs: a wedge (or two sharing a face), a source and a receiver. Outputs: the sampled impulse
//! response ([`EdgeResponse`]) and the diffracted sound's gain per band relative to the direct
//! sound over the same distance ([`Wedge::band_gains`], [`Wedge::second_order`]).

use super::acoustic_geometry::AcousticGeometry;
use super::geometry::AcousticRay;
use super::tracer::SPEED_OF_SOUND;
use korppi_math::Vec3;
use std::f64::consts::PI;

/// The bands' centre frequencies, in Hz (as the reverb's decay times).
pub const DIFFRACTION_BANDS: [f32; 3] = [250.0, 1000.0, 4000.0];
/// The rate [`Wedge::band_gains`] samples the response at: high enough that the response above
/// the bands aliases into them by less than 0.3 % (16 kHz would alias 2 % into the 1 kHz band).
const BAND_RATE: f32 = 32_000.0;
/// The response [`Wedge::band_gains`] takes: 20 ms, which leaves out less than 0.3 % of each
/// band (10 ms would leave out 1 % of the lowest).
const BAND_FRAMES: usize = 640;
/// Interval halvings of the adaptive integration at most.
const MAX_DEPTH: u32 = 40;
/// The rate [`Wedge::second_order`] samples its response at.
const SECOND_RATE: f32 = 32_000.0;
/// The second-order response taken: 10 ms from the least time.
const SECOND_FRAMES: usize = 320;
/// Edge elements per edge at most, for the second order.
const MAX_ELEMENTS: usize = 2000;

/// A wedge's edge: from `start` to `end`, its first face running from it along `face` with the air
/// toward `normal`, and the second face `angle` around through the air.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Wedge {
    /// One end of the edge.
    pub start: Vec3,
    /// The other end.
    pub end: Vec3,
    /// The unit direction the first face runs in from the edge, perpendicular to it.
    pub face: Vec3,
    /// The first face's unit normal, toward the air.
    pub normal: Vec3,
    /// The exterior angle through the air from the first face to the second, in radians: 2π for
    /// a thin screen, 3π/2 for a box's edge.
    pub angle: f32,
}

/// A sampled edge diffraction impulse response on the scale where a direct sound over `d` is
/// `1/d`: for first order, sample `i` is the response's integral over
/// `[(first + i − ½)/fs, (first + i + ½)/fs)`; for second order, each element pair is shared
/// between the two samples about its time by linear interpolation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EdgeResponse {
    /// The absolute index of the first sample: that of the least time over the edge.
    pub first: usize,
    /// The samples.
    pub samples: Vec<f32>,
    /// The power of the sinc the sampling multiplies the response by: 1 for the integral over a
    /// sample's interval (a boxcar), 2 for linear interpolation (a triangle).
    pub smoothing: i32,
}

impl EdgeResponse {
    /// The diffraction's response at `frequency` (real and imaginary parts) from samples at
    /// `sample_rate`: `Σ h[n] e^(−i 2π f n / fs)`, divided by the sinc the sampling multiplies it
    /// by.
    #[must_use]
    pub fn at(&self, frequency: f32, sample_rate: f32) -> (f32, f32) {
        let step = -2.0 * PI * f64::from(frequency) / f64::from(sample_rate);
        let half = 0.5 * step.abs();
        let boxcar = if half > 1e-12 { half.sin() / half } else { 1.0 };
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (index, &sample) in self.samples.iter().enumerate() {
            let phase = step * (self.first + index) as f64;
            re += f64::from(sample) * phase.cos();
            im += f64::from(sample) * phase.sin();
        }
        let kernel = boxcar.powi(self.smoothing);
        ((re / kernel) as f32, (im / kernel) as f32)
    }
}

/// The geometry of one source and receiver about an edge, shifted so that the source is at
/// `z = 0`: everything the integrand needs.
struct Configuration {
    r_s: f64,
    r_r: f64,
    /// The receiver's `z`; the source's is 0.
    z_r: f64,
    /// The edge's ends.
    from: f64,
    to: f64,
    nu: f64,
    /// `sin(νφ)` and `2 sin²(νφ/2)` for the four `φ`.
    sines: [f64; 4],
    half_sines: [f64; 4],
}

/// `β = Σ sin(νφ) / (cosh(νη) − cos(νφ))` from `sin(νφ)` and `2 sin²(νφ/2)` of the four `φ` and
/// `x = cosh η − 1`, stably near the edge point of least time (where `x` vanishes).
fn beta(nu: f64, sines: &[f64; 4], half_sines: &[f64; 4], x: f64) -> f64 {
    let x = x.max(0.0);
    let eta = (x + (x * (x + 2.0)).sqrt()).ln_1p();
    let shift = 2.0 * (0.5 * nu * eta).sinh().powi(2);
    let mut beta = 0.0;
    for (sine, half) in sines.iter().zip(half_sines) {
        // cosh(νη) − cos(νφ) = 2 sinh²(νη/2) + 2 sin²(νφ/2).
        let denominator = shift + half;
        if denominator > 1e-300 {
            beta += sine / denominator;
        }
    }
    beta
}

/// `sin(νφ)` and `2 sin²(νφ/2)` for `φ = π ± a ± b`.
fn angle_terms(nu: f64, a: f64, b: f64) -> ([f64; 4], [f64; 4]) {
    let phis = [PI + a + b, PI + a - b, PI - a + b, PI - a - b];
    (
        phis.map(|phi| (nu * phi).sin()),
        phis.map(|phi| 2.0 * (0.5 * nu * phi).sin().powi(2)),
    )
}

impl Configuration {
    /// `β / (m l)` at the edge point `z`.
    fn integrand(&self, z: f64) -> f64 {
        let m = (self.r_s * self.r_s + z * z).sqrt();
        let l = (self.r_r * self.r_r + (z - self.z_r).powi(2)).sqrt();
        let rr = self.r_s * self.r_r;
        let x = ((m * l - rr) + z * (z - self.z_r)) / rr;
        beta(self.nu, &self.sines, &self.half_sines, x) / (m * l)
    }

    /// The path length via the edge point `z`.
    fn length(&self, z: f64) -> f64 {
        (self.r_s * self.r_s + z * z).sqrt() + (self.r_r * self.r_r + (z - self.z_r).powi(2)).sqrt()
    }

    /// The edge point of least path length, on the infinite edge.
    fn apex(&self) -> f64 {
        self.z_r * self.r_s / (self.r_s + self.r_r)
    }

    /// The two edge points (on the infinite edge) whose path is `length` long, the lower first;
    /// both the apex for the least length or less. `m + l = D` squared twice is a quadratic.
    fn points_at(&self, length: f64) -> (f64, f64) {
        let least = self.length(self.apex());
        if length <= least {
            return (self.apex(), self.apex());
        }
        let (d2, q) = (length * length, 2.0 * self.z_r);
        let p = d2 + self.r_s * self.r_s - self.r_r * self.r_r - self.z_r * self.z_r;
        let a = 4.0 * d2 - q * q;
        let b = -2.0 * p * q;
        let c = 4.0 * d2 * self.r_s * self.r_s - p * p;
        let root = (b * b - 4.0 * a * c).max(0.0).sqrt();
        // The stable pair of roots.
        let t = -0.5 * (b + b.signum() * root);
        let (one, other) = if t.abs() > 1e-300 {
            (t / a, c / t)
        } else {
            (0.0, 0.0)
        };
        (one.min(other), one.max(other))
    }

    /// The integral of the integrand over `[low, high]` clipped to the edge.
    fn integral(&self, low: f64, high: f64) -> f64 {
        let (low, high) = (low.max(self.from), high.min(self.to));
        if high <= low {
            return 0.0;
        }
        let scale = 1e-14 / (self.r_s * self.r_r);
        self.adaptive(
            low,
            high,
            gauss(|z| self.integrand(z), low, high),
            scale,
            MAX_DEPTH,
        )
    }

    /// Adaptive Gauss-Legendre: an interval is halved until its halves agree with it.
    fn adaptive(&self, low: f64, high: f64, whole: f64, scale: f64, depth: u32) -> f64 {
        let middle = 0.5 * (low + high);
        let (left, right) = (
            gauss(|z| self.integrand(z), low, middle),
            gauss(|z| self.integrand(z), middle, high),
        );
        let halves = left + right;
        if depth == 0 || (halves - whole).abs() <= 1e-9 * halves.abs() + scale * (high - low) {
            return halves;
        }
        self.adaptive(low, middle, left, scale, depth - 1)
            + self.adaptive(middle, high, right, scale, depth - 1)
    }
}

/// Four-point Gauss-Legendre quadrature of `f` over `[low, high]`.
fn gauss(f: impl Fn(f64) -> f64, low: f64, high: f64) -> f64 {
    const NODES: [f64; 2] = [0.339_981_043_584_856_3, 0.861_136_311_594_052_6];
    const WEIGHTS: [f64; 2] = [0.652_145_154_862_546_1, 0.347_854_845_137_453_8];
    let (middle, half) = (0.5 * (low + high), 0.5 * (high - low));
    let mut sum = 0.0;
    for (node, weight) in NODES.iter().zip(WEIGHTS) {
        sum += weight * (f(middle - half * node) + f(middle + half * node));
    }
    sum * half
}

/// A vector in double precision.
fn wide(v: Vec3) -> [f64; 3] {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Wedge {
    /// A point's cylindrical coordinates `(r, θ, z)` about the edge, `z` from `start`.
    fn coordinates(&self, point: Vec3) -> (f64, f64, f64) {
        let (start, end) = (wide(self.start), wide(self.end));
        let length = (0..3)
            .map(|i| (end[i] - start[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let axis: [f64; 3] = std::array::from_fn(|i| (end[i] - start[i]) / length.max(1e-300));
        let relative: [f64; 3] = std::array::from_fn(|i| wide(point)[i] - start[i]);
        let z = dot(relative, axis);
        let across: [f64; 3] = std::array::from_fn(|i| relative[i] - z * axis[i]);
        let r = dot(across, across).sqrt();
        let theta = dot(across, wide(self.normal))
            .atan2(dot(across, wide(self.face)))
            .rem_euclid(2.0 * PI);
        let angle = f64::from(self.angle);
        let clamped = if theta <= angle {
            theta
        } else if theta - angle < 2.0 * PI - theta {
            angle
        } else {
            0.0
        };
        (r, clamped, z)
    }

    /// Whether `point` lies in the wedge's exterior air sector rather than inside the solid.
    #[must_use]
    pub fn in_air(&self, point: Vec3) -> bool {
        let (start, end) = (wide(self.start), wide(self.end));
        let length = (0..3)
            .map(|i| (end[i] - start[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let axis: [f64; 3] = std::array::from_fn(|i| (end[i] - start[i]) / length.max(1e-300));
        let relative: [f64; 3] = std::array::from_fn(|i| wide(point)[i] - start[i]);
        let z = dot(relative, axis);
        let across: [f64; 3] = std::array::from_fn(|i| relative[i] - z * axis[i]);
        let r = dot(across, across).sqrt();
        if r < 1e-6 {
            return true;
        }
        let theta = dot(across, wide(self.normal))
            .atan2(dot(across, wide(self.face)))
            .rem_euclid(2.0 * PI);
        let angle = f64::from(self.angle);
        theta <= angle + 1e-4 || theta >= 2.0 * PI - 1e-4
    }

    /// The point on the finite edge segment of least path length between `source` and `receiver`.
    #[must_use]
    pub fn apex_point(&self, source: Vec3, receiver: Vec3) -> Vec3 {
        let ((r_s, _, z_s), (r_r, _, z_r)) = (self.coordinates(source), self.coordinates(receiver));
        let length = f64::from((self.end - self.start).length());
        if length <= 1e-9 {
            return self.start;
        }
        let denom = r_s + r_r;
        let z_apex = if denom > 1e-12 {
            (z_s * r_r + z_r * r_s) / denom
        } else {
            0.5 * (z_s + z_r)
        }
        .clamp(0.0, length);
        let t = (z_apex / length) as f32;
        self.start + (self.end - self.start) * t
    }

    /// The edge diffraction impulse response from `source` to `receiver`: `frames` samples at
    /// `sample_rate` from the least time over the edge (see the module documentation).
    #[must_use]
    pub fn impulse_response(
        &self,
        source: Vec3,
        receiver: Vec3,
        sample_rate: f32,
        frames: usize,
    ) -> EdgeResponse {
        let mut response = EdgeResponse {
            first: 0,
            samples: vec![0.0; frames],
            smoothing: 1,
        };
        let ((r_s, theta_s, z_s), (r_r, theta_r, z_r)) =
            (self.coordinates(source), self.coordinates(receiver));
        let length = f64::from((self.end - self.start).length());
        if r_s < 1e-9 || r_r < 1e-9 || length <= 0.0 || self.angle <= 0.0 {
            return response;
        }
        let nu = PI / f64::from(self.angle);
        let (sines, half_sines) = angle_terms(nu, theta_s, theta_r);
        let setup = Configuration {
            r_s,
            r_r,
            z_r: z_r - z_s,
            from: -z_s,
            to: length - z_s,
            nu,
            sines,
            half_sines,
        };
        let (c, fs) = (f64::from(SPEED_OF_SOUND), f64::from(sample_rate));
        let earliest = setup.length(setup.apex().clamp(setup.from, setup.to));
        response.first = (earliest / c * fs).round() as usize;
        let prefactor = -nu / (4.0 * PI);
        for (index, sample) in response.samples.iter_mut().enumerate() {
            let n = (response.first + index) as f64;
            let (low, high) = (c * (n - 0.5) / fs, c * (n + 0.5) / fs);
            let (left_high, right_high) = setup.points_at(high);
            let (left_low, right_low) = setup.points_at(low);
            *sample = (prefactor
                * (setup.integral(left_high, left_low) + setup.integral(right_low, right_high)))
                as f32;
            if left_high <= setup.from && right_high >= setup.to {
                // Every edge point has been reached: the rest is silence.
                break;
            }
        }
        response
    }

    /// The diffracted sound's amplitude at [`DIFFRACTION_BANDS`], relative to the direct sound
    /// over the straight distance from `source` to `receiver`.
    #[must_use]
    pub fn band_gains(&self, source: Vec3, receiver: Vec3) -> [f32; 3] {
        let response = self.impulse_response(source, receiver, BAND_RATE, BAND_FRAMES);
        let distance = (receiver - source).length();
        DIFFRACTION_BANDS.map(|frequency| {
            let (re, im) = response.at(frequency, BAND_RATE);
            (re * re + im * im).sqrt() * distance
        })
    }

    /// The edge as a line in double precision: its start, unit direction and length.
    fn line(&self) -> ([f64; 3], [f64; 3], f64) {
        let (start, end) = (wide(self.start), wide(self.end));
        let length = (0..3)
            .map(|i| (end[i] - start[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let axis = std::array::from_fn(|i| (end[i] - start[i]) / length.max(1e-300));
        (start, axis, length)
    }

    /// The second-order diffraction from `source` over this edge, along the face it shares with
    /// `next`, then over `next` to `receiver`: its amplitude at [`DIFFRACTION_BANDS`] relative to
    /// the direct sound over the straight distance (see the module documentation).
    #[must_use]
    pub fn second_order(&self, next: &Wedge, source: Vec3, receiver: Vec3) -> [f32; 3] {
        let response =
            self.second_order_response(next, source, receiver, SECOND_RATE, SECOND_FRAMES);
        let distance = (receiver - source).length();
        DIFFRACTION_BANDS.map(|frequency| {
            let (re, im) = response.at(frequency, SECOND_RATE);
            (re * re + im * im).sqrt() * distance
        })
    }

    /// The second-order impulse response over this edge and `next` (sharing a face): `frames`
    /// samples at `sample_rate` from the least time.
    #[must_use]
    pub fn second_order_response(
        &self,
        next: &Wedge,
        source: Vec3,
        receiver: Vec3,
        sample_rate: f32,
        frames: usize,
    ) -> EdgeResponse {
        let mut response = EdgeResponse {
            first: 0,
            samples: vec![0.0; frames + 2],
            smoothing: 2,
        };
        let ((first_start, first_axis, first_length), (next_start, next_axis, next_length)) =
            (self.line(), next.line());
        let (r_s, theta_s, z_s) = self.coordinates(source);
        let (r_r, theta_r, z_r) = next.coordinates(receiver);
        if r_s < 1e-9 || r_r < 1e-9 || first_length <= 0.0 || next_length <= 0.0 {
            response.samples.truncate(frames);
            return response;
        }
        let (nu1, nu2) = (PI / f64::from(self.angle), PI / f64::from(next.angle));
        // Each edge lies in a face plane of the other: one angle about each.
        let (theta_next, theta_first) = (
            self.coordinates((next.start + next.end) * 0.5).1,
            next.coordinates((self.start + self.end) * 0.5).1,
        );
        let (sines1, halves1) = angle_terms(nu1, theta_s, theta_next);
        let (sines2, halves2) = angle_terms(nu2, theta_first, theta_r);
        let (s, r) = (wide(source), wide(receiver));
        let at = |start: [f64; 3], axis: [f64; 3], z: f64| -> [f64; 3] {
            std::array::from_fn(|i| start[i] + axis[i] * z)
        };
        let distance = |a: [f64; 3], b: [f64; 3]| -> f64 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let path = |z1: f64, z2: f64| {
            let (p1, p2) = (
                at(first_start, first_axis, z1),
                at(next_start, next_axis, z2),
            );
            distance(s, p1) + distance(p1, p2) + distance(p2, r)
        };
        // The least path: the path length is convex in (z1, z2); coordinate descent finds it.
        let golden = |f: &dyn Fn(f64) -> f64, mut low: f64, mut high: f64| {
            let ratio = 0.5 * (5f64.sqrt() - 1.0);
            for _ in 0..60 {
                let (a, b) = (high - ratio * (high - low), low + ratio * (high - low));
                if f(a) < f(b) {
                    high = b;
                } else {
                    low = a;
                }
            }
            0.5 * (low + high)
        };
        let (mut z1, mut z2) = (0.5 * first_length, 0.5 * next_length);
        for _ in 0..30 {
            z1 = golden(&|z| path(z, z2), 0.0, first_length);
            z2 = golden(&|z| path(z1, z), 0.0, next_length);
        }
        let (c, fs) = (f64::from(SPEED_OF_SOUND), f64::from(sample_rate));
        let least = path(z1, z2);
        response.first = (least / c * fs).floor() as usize;
        let longest = (response.first + frames) as f64 * c / fs;
        // The parts of each edge a path within the window can cross: `m + |P − R|` (or
        // `|S − P| + l`) bounds the path through a point from below and is convex along the edge.
        let span = |f: &dyn Fn(f64) -> f64, apex: f64, length: f64| -> (f64, f64) {
            let edge = |mut inside: f64, mut outside: f64| {
                if f(outside) <= longest {
                    return outside;
                }
                for _ in 0..60 {
                    let middle = 0.5 * (inside + outside);
                    if f(middle) <= longest {
                        inside = middle;
                    } else {
                        outside = middle;
                    }
                }
                outside
            };
            (edge(apex, 0.0), edge(apex, length))
        };
        let (from1, to1) = span(
            &|z| {
                let p = at(first_start, first_axis, z);
                distance(s, p) + distance(p, r)
            },
            z1,
            first_length,
        );
        let (from2, to2) = span(
            &|z| {
                let p = at(next_start, next_axis, z);
                distance(s, p) + distance(p, r)
            },
            z2,
            next_length,
        );
        // Elements of at most 2 c/fs, and at most a twelfth of the edges' distance apart (as
        // Svensson refines close edges).
        let gap = distance(
            at(first_start, first_axis, z1),
            at(next_start, next_axis, z2),
        );
        let size = (2.0 * c / fs).min(gap / 12.0).max(1e-4);
        let count =
            |from: f64, to: f64| (((to - from) / size).ceil() as usize).clamp(1, MAX_ELEMENTS);
        let (count1, count2) = (count(from1, to1), count(from2, to2));
        let (dz1, dz2) = ((to1 - from1) / count1 as f64, (to2 - from2) / count2 as f64);
        // Per element of the first edge: its point, distance from the source, and coordinates
        // about the second edge; per element of the second: its point, distance to the receiver,
        // and coordinates about the first.
        let about = |start: [f64; 3], axis: [f64; 3], p: [f64; 3]| -> (f64, f64) {
            let relative: [f64; 3] = std::array::from_fn(|i| p[i] - start[i]);
            let z = dot(relative, axis);
            let across: [f64; 3] = std::array::from_fn(|i| relative[i] - z * axis[i]);
            (dot(across, across).sqrt(), z)
        };
        let firsts: Vec<([f64; 3], f64, f64, f64, f64)> = (0..count1)
            .map(|i| {
                let z = from1 + (i as f64 + 0.5) * dz1;
                let p = at(first_start, first_axis, z);
                let (r2, z2) = about(next_start, next_axis, p);
                (p, z, distance(s, p), r2, z2)
            })
            .collect();
        let nexts: Vec<([f64; 3], f64, f64, f64, f64)> = (0..count2)
            .map(|j| {
                let z = from2 + (j as f64 + 0.5) * dz2;
                let p = at(next_start, next_axis, z);
                let (r1, z1) = about(first_start, first_axis, p);
                (p, z, distance(p, r), r1, z1)
            })
            .collect();
        let scale = 0.5 * nu1 * nu2 / (16.0 * PI * PI) * dz1 * dz2;
        let to_samples = fs / c;
        let offset = response.first as f64;
        for &(p1, z1, m, r1_about_next, z1_along_next) in &firsts {
            for &(p2, z2, l, r2_about_first, z2_along_first) in &nexts {
                let n = distance(p1, p2);
                let total = m + n + l;
                if total > longest || n < 1e-9 {
                    continue;
                }
                let x1 = ((z1 - z_s) * (z1 - z2_along_first) + (m * n - r_s * r2_about_first))
                    / (r_s * r2_about_first);
                let x2 = ((z2 - z1_along_next) * (z2 - z_r) + (n * l - r1_about_next * r_r))
                    / (r1_about_next * r_r);
                let contribution =
                    scale * beta(nu1, &sines1, &halves1, x1) * beta(nu2, &sines2, &halves2, x2)
                        / (m * n * l);
                let position = (total * to_samples - offset).max(0.0);
                let index = position.floor() as usize;
                let fraction = position - index as f64;
                if index + 1 < response.samples.len() {
                    response.samples[index] += (contribution * (1.0 - fraction)) as f32;
                    response.samples[index + 1] += (contribution * fraction) as f32;
                }
            }
        }
        response.samples.truncate(frames);
        response
    }
}

/// The most diffraction paths evaluated per occluded source-receiver pair.
const MAX_CANDIDATES: usize = 4;

enum DiffractionCandidate {
    First(Wedge, f32),
    Second(Wedge, Wedge, f32),
}

impl DiffractionCandidate {
    fn detour(&self) -> f32 {
        match *self {
            Self::First(_, detour) | Self::Second(_, _, detour) => detour,
        }
    }
}

fn clear_to_edge(geometry: &AcousticGeometry, from: Vec3, edge_point: Vec3) -> bool {
    let delta = edge_point - from;
    let dist = delta.length();
    if dist <= 2e-3 {
        return true;
    }
    let ray = AcousticRay::new(from, delta / dist);
    geometry.raycast_closest(&ray, dist - 1e-3).is_none()
}

/// The diffracted amplitude at [`DIFFRACTION_BANDS`] from `source` to `receiver` around the
/// edges of the surfaces in `geometry` that block the direct path, relative to the direct sound
/// over the straight distance. Evaluates up to [`MAX_CANDIDATES`] shortest-detour first- and
/// second-order edge paths and sums their energies in quadrature; returns `[0.0; 3]` when the
/// direct line of sight is clear.
#[must_use]
pub fn diffracted_sound(geometry: &AcousticGeometry, source: Vec3, receiver: Vec3) -> [f32; 3] {
    let direct_vec = receiver - source;
    let direct_dist = direct_vec.length();
    if direct_dist < 1e-4 {
        return [0.0; 3];
    }
    let crossed = geometry.crossed_surfaces(source, receiver);
    if crossed.is_empty() {
        return [0.0; 3];
    }

    let mut candidates: Vec<DiffractionCandidate> = Vec::new();

    for &surf_idx in &crossed {
        let wedges = geometry.surface_wedges(surf_idx);
        let mut first_order_edges = vec![false; wedges.len()];

        for (i, wedge) in wedges.iter().enumerate() {
            if !wedge.in_air(source) || !wedge.in_air(receiver) {
                continue;
            }
            let apex = wedge.apex_point(source, receiver);
            if clear_to_edge(geometry, source, apex) && clear_to_edge(geometry, receiver, apex) {
                let detour = (apex - source).length() + (receiver - apex).length() - direct_dist;
                candidates.push(DiffractionCandidate::First(*wedge, detour));
                first_order_edges[i] = true;
            }
        }

        for &(i, j) in geometry.surface_wedge_pairs(surf_idx) {
            if i >= wedges.len() || j >= wedges.len() {
                continue;
            }
            if first_order_edges[i] || first_order_edges[j] {
                continue;
            }
            for (w1, w2) in [(wedges[i], wedges[j]), (wedges[j], wedges[i])] {
                if !w1.in_air(source) || !w2.in_air(receiver) {
                    continue;
                }
                let mid2 = (w2.start + w2.end) * 0.5;
                let a1 = w1.apex_point(source, mid2);
                let a2 = w2.apex_point(a1, receiver);
                let a1 = w1.apex_point(source, a2);
                let to_next = mid2 - (w1.start + w1.end) * 0.5;
                let shared_normal = if to_next.dot(w1.face) > 1e-4 {
                    w1.normal
                } else {
                    -w1.face
                };
                if clear_to_edge(geometry, source, a1)
                    && clear_to_edge(geometry, receiver, a2)
                    && clear_to_edge(
                        geometry,
                        a1 + shared_normal * 1e-3,
                        a2 + shared_normal * 1e-3,
                    )
                {
                    let detour =
                        (a1 - source).length() + (a2 - a1).length() + (receiver - a2).length()
                            - direct_dist;
                    candidates.push(DiffractionCandidate::Second(w1, w2, detour));
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.detour().total_cmp(&b.detour()));
    candidates.truncate(MAX_CANDIDATES);

    let mut energy = [0.0f32; 3];
    for candidate in candidates {
        let gains = match candidate {
            DiffractionCandidate::First(w, _) => w.band_gains(source, receiver),
            DiffractionCandidate::Second(w1, w2, _) => {
                let to_next = (w2.start + w2.end - w1.start - w1.end) * 0.5;
                let gap = to_next.length();
                if gap < 0.01 {
                    let (face, normal) = if to_next.dot(w1.face) > 1e-4 {
                        (-w1.normal, -w1.face)
                    } else {
                        (w1.face, w1.normal)
                    };
                    let screen = Wedge {
                        start: (w1.start + w2.start) * 0.5,
                        end: (w1.end + w2.end) * 0.5,
                        face,
                        normal,
                        angle: 2.0 * std::f32::consts::PI,
                    };
                    screen.band_gains(source, receiver)
                } else {
                    w1.second_order(&w2, source, receiver)
                }
            }
        };
        for band in 0..3 {
            energy[band] += gains[band] * gains[band];
        }
    }

    energy.map(f32::sqrt)
}
