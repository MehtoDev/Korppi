//! Biot-Tolstoy-Medwin edge diffraction (FIX-7.5, ADR-051; findings AUD-08 and AUD-10).
//!
//! Before FIX-7.5 `BtmDiffraction` was a Maekawa-style detour heuristic under the BTM name, and
//! nothing used it. Now a wedge's edge diffraction impulse response is Svensson's discrete
//! formulation of the Biot-Tolstoy-Medwin solution: the line integral along the edge of
//! `β / (m l)`, sampled by the time each edge point's path takes.
//!
//! The reference is Biot and Tolstoy's exact solution for an infinite wedge and a point source,
//! `h(τ) = −(c ν / 2π) β / (r_S r_R sinh η)` after the least time `τ₀`. The half-plane is the
//! wedge of exterior angle 2π. The test integrates it over each sample's interval in time,
//! independently of the line integral the implementation evaluates along the edge.

use korppi_audio::raytrace::{Wedge, DIFFRACTION_BANDS, SPEED_OF_SOUND};
use korppi_math::Vec3;
use std::f64::consts::PI;

const RATE: f32 = 48_000.0;

/// A point at cylindrical coordinates (`r`, `theta`, `z`) about `wedge`'s edge: `theta` from its
/// first face toward the air, `z` from the edge's middle.
fn around(wedge: &Wedge, r: f32, theta: f32, z: f32) -> Vec3 {
    let axis = (wedge.end - wedge.start).normalize();
    let middle = (wedge.start + wedge.end) * 0.5;
    middle + axis * z + (wedge.face * theta.cos() + wedge.normal * theta.sin()) * r
}

/// An edge along +Z from z = −200 m to 200 m, its first face running along −X with the air
/// above it (+Y), of exterior angle `angle`.
fn wedge(angle: f32) -> Wedge {
    Wedge {
        start: Vec3::new(0.0, 0.0, -200.0),
        end: Vec3::new(0.0, 0.0, 200.0),
        face: Vec3::NEG_X,
        normal: Vec3::Y,
        angle,
    }
}

/// Biot and Tolstoy's exact response of an infinite wedge, integrated over sample `n`'s interval
/// `[(n − ½)/fs, (n + ½)/fs)` (the substitution τ = τ₀ + s² removes the singularity at τ₀).
#[allow(clippy::too_many_arguments)]
fn biot_tolstoy(
    nu: f64,
    (r_s, theta_s, z_s): (f64, f64, f64),
    (r_r, theta_r, z_r): (f64, f64, f64),
    n: usize,
) -> f64 {
    let c = f64::from(SPEED_OF_SOUND);
    let fs = f64::from(RATE);
    let dz = z_r - z_s;
    let tau0 = ((r_s + r_r).powi(2) + dz * dz).sqrt() / c;
    let h = |tau: f64| {
        let cosh_eta = (c * c * tau * tau - (r_s * r_s + r_r * r_r + dz * dz)) / (2.0 * r_s * r_r);
        let eta = cosh_eta.max(1.0).acosh();
        let beta: f64 = [
            PI + theta_s + theta_r,
            PI + theta_s - theta_r,
            PI - theta_s + theta_r,
            PI - theta_s - theta_r,
        ]
        .iter()
        .map(|phi| (nu * phi).sin() / ((nu * eta).cosh() - (nu * phi).cos()))
        .sum();
        -(c * nu / (2.0 * PI)) * beta / (r_s * r_r * eta.sinh())
    };
    let (low, high) = ((n as f64 - 0.5) / fs, (n as f64 + 0.5) / fs);
    if high <= tau0 {
        return 0.0;
    }
    let (s0, s1) = ((low - tau0).max(0.0).sqrt(), (high - tau0).sqrt());
    // 64 intervals of 8-point Gauss-Legendre in s.
    const X: [f64; 4] = [
        0.183_434_642_495_650,
        0.525_532_409_916_329,
        0.796_666_477_413_627,
        0.960_289_856_497_536,
    ];
    const W: [f64; 4] = [
        0.362_683_783_378_362,
        0.313_706_645_877_887,
        0.222_381_034_453_374,
        0.101_228_536_290_376,
    ];
    let parts = 64;
    let width = (s1 - s0) / f64::from(parts);
    let mut sum = 0.0;
    for part in 0..parts {
        let middle = s0 + width * (f64::from(part) + 0.5);
        for (x, w) in X.iter().zip(W) {
            for s in [middle - x * width / 2.0, middle + x * width / 2.0] {
                if s > 0.0 {
                    sum += w * width / 2.0 * 2.0 * s * h(tau0 + s * s);
                }
            }
        }
    }
    sum
}

/// For a half-plane and a right-angled wedge (a box's edge), with the receiver in the shadow and
/// off the source's plane across the edge, the sampled response is Biot and Tolstoy's exact
/// solution: each of its first 10 ms of samples within 0.2 % of the first sample's size, and their
/// sum within 0.2 %.
#[test]
fn a_wedge_diffracts_as_biot_and_tolstoy_solved() {
    for (angle, theta_r) in [
        (2.0 * std::f32::consts::PI, 5.2),
        (1.5 * std::f32::consts::PI, 4.2),
    ] {
        let wedge = wedge(angle);
        let nu = PI / f64::from(angle);
        let (source, receiver) = ((2.0f32, 1.0f32, 0.0f32), (3.0f32, theta_r, 1.5f32));
        let response = wedge.impulse_response(
            around(&wedge, source.0, source.1, source.2),
            around(&wedge, receiver.0, receiver.1, receiver.2),
            RATE,
            480,
        );
        let exact: Vec<f64> = (response.first..response.first + 480)
            .map(|n| {
                biot_tolstoy(
                    nu,
                    (
                        f64::from(source.0),
                        f64::from(source.1),
                        f64::from(source.2),
                    ),
                    (
                        f64::from(receiver.0),
                        f64::from(receiver.1),
                        f64::from(receiver.2),
                    ),
                    n,
                )
            })
            .collect();
        assert_eq!(response.samples.len(), 480);
        let scale = exact[0].abs().max(exact[1].abs());
        for (index, (&got, &want)) in response.samples.iter().zip(&exact).enumerate() {
            assert!(
                (f64::from(got) - want).abs() < 2e-3 * scale,
                "angle {angle}, sample {index}: {got}, exactly {want}"
            );
        }
        let (got, want) = (
            response.samples.iter().map(|&s| f64::from(s)).sum::<f64>(),
            exact.iter().sum::<f64>(),
        );
        eprintln!(
            "angle {angle}: first {} vs {}, sum {got:.6e} vs {want:.6e}",
            response.samples[0], exact[0]
        );
        assert!(
            (got / want - 1.0).abs() < 2e-3,
            "angle {angle}: sum {got}, exactly {want}"
        );
        // The response starts at the least time over the edge.
        let least = ((source.0 + receiver.0).powi(2) + 1.5f32.powi(2)).sqrt() / SPEED_OF_SOUND;
        assert_eq!(response.first, (least * RATE).round() as usize);
    }
}

/// The field is continuous across the shadow boundary: just in the light, the direct sound plus
/// the diffracted one equals, just in the shadow, the diffracted sound alone (the diffraction
/// makes up for the direct sound's step), at each band.
#[test]
fn the_shadow_boundary_is_continuous() {
    let wedge = wedge(2.0 * std::f32::consts::PI);
    let source = around(&wedge, 2.0, 1.2, 0.0);
    let boundary = 1.2 + std::f32::consts::PI;
    for frequency in DIFFRACTION_BANDS {
        let field = |theta: f32, with_direct: bool| {
            let receiver = around(&wedge, 2.5, theta, 0.4);
            let response = wedge.impulse_response(source, receiver, RATE, 4800);
            let (mut re, mut im) = response.at(frequency, RATE);
            if with_direct {
                let distance = (receiver - source).length();
                let phase = -2.0 * std::f32::consts::PI * frequency * distance / SPEED_OF_SOUND;
                re += phase.cos() / distance;
                im += phase.sin() / distance;
            }
            (re * re + im * im).sqrt()
        };
        let (lit, shadowed) = (
            field(boundary - 0.001, true),
            field(boundary + 0.001, false),
        );
        eprintln!("{frequency} Hz: lit {lit:.5}, shadowed {shadowed:.5}");
        assert!(
            (lit / shadowed - 1.0).abs() < 0.02,
            "{frequency} Hz: {lit} in the light, {shadowed} in the shadow"
        );
    }
}

/// In the shadow, the diffracted sound is quieter the deeper the receiver and the higher the
/// band. The band gains, from a short response sampled at 16 kHz, are those of a long response
/// at 48 kHz within 1 %.
#[test]
fn deeper_shadow_and_higher_bands_are_quieter() {
    let wedge = wedge(1.5 * std::f32::consts::PI);
    let source = around(&wedge, 2.0, 0.6, 0.0);
    let gains = |theta: f32| wedge.band_gains(source, around(&wedge, 3.0, theta, 0.0));
    let (shallow, deep) = (gains(0.6 + std::f32::consts::PI + 0.2), gains(4.6));
    eprintln!("shallow {shallow:?}, deep {deep:?}");
    for band in 0..3 {
        assert!(
            deep[band] < shallow[band],
            "band {band}: {shallow:?}, {deep:?}"
        );
    }
    for g in [shallow, deep] {
        assert!(g[0] > g[1] && g[1] > g[2], "{g:?}");
    }
    let receiver = around(&wedge, 3.0, 4.6, 0.0);
    let long = wedge.impulse_response(source, receiver, RATE, 9600);
    let distance = (receiver - source).length();
    for (band, frequency) in DIFFRACTION_BANDS.into_iter().enumerate() {
        let (re, im) = long.at(frequency, RATE);
        let exact = (re * re + im * im).sqrt() * distance;
        eprintln!(
            "{frequency} Hz: {} from the short response, {exact} from the long",
            deep[band]
        );
        assert!(
            (deep[band] / exact - 1.0).abs() < 0.01,
            "{frequency} Hz: {}, not {exact}",
            deep[band]
        );
    }
}

/// Second-order diffraction (Svensson's double line integral over two edges sharing a face,
/// halved for the leg along the face) of a thin plate approaches the half-plane's first-order
/// diffraction as the plate thins. A plate 5 mm thick has two right-angled top edges sharing its
/// top face. From a source in front to a receiver behind, neither edge sees both, so only the
/// second order reaches the receiver, and it diffracts as a half-plane at the plate's middle:
/// within 3 % at 250 Hz and 5 % at 1 kHz, where the plate is thin against the wavelength.
#[test]
fn a_thin_plate_diffracts_as_a_half_plane() {
    use std::f32::consts::PI as PI32;
    let (half, reach) = (0.0025, 50.0);
    // The plate fills x in [-2.5 mm, 2.5 mm] below y = 0; its edges run along z.
    let front = Wedge {
        start: Vec3::new(-half, 0.0, -reach),
        end: Vec3::new(-half, 0.0, reach),
        face: Vec3::NEG_Y,
        normal: Vec3::NEG_X,
        angle: 1.5 * PI32,
    };
    let back = Wedge {
        start: Vec3::new(half, 0.0, -reach),
        end: Vec3::new(half, 0.0, reach),
        face: Vec3::NEG_X,
        normal: Vec3::Y,
        angle: 1.5 * PI32,
    };
    let screen = Wedge {
        start: Vec3::new(0.0, 0.0, -reach),
        end: Vec3::new(0.0, 0.0, reach),
        face: Vec3::NEG_Y,
        normal: Vec3::NEG_X,
        angle: 2.0 * PI32,
    };
    let (source, receiver) = (Vec3::new(-2.0, -1.0, 0.3), Vec3::new(1.5, -1.2, -0.4));
    let plate = front.second_order(&back, source, receiver);
    let thin = screen.band_gains(source, receiver);
    for (band, tolerance) in [(0, 0.03), (1, 0.05)] {
        assert!(
            (plate[band] / thin[band] - 1.0).abs() < tolerance,
            "{} Hz: the plate {}, the half-plane {}",
            DIFFRACTION_BANDS[band],
            plate[band],
            thin[band]
        );
    }
}
