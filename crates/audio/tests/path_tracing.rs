//! Monte Carlo acoustic path tracing (FIX-7.5, ADR-051; finding AUD-08).
//!
//! Before FIX-7.5 the "diffuse" bounce was a fixed direction rotated by the golden angle, nothing
//! was random, energy was not split between specular and diffuse paths, and every hit was weighted
//! by 1/(4π d²). Now rays leave the listener in random directions; each hit absorbs its share per
//! band and scatters diffusely (cosine-weighted) or specularly by the material's scattering
//! coefficient; Russian roulette ends weak paths; and a receiver sphere around each emitter
//! collects the energy that crosses it. A shoebox room checks the result against Eyring's formula
//! with Kuttruff's correction for the spread of free path lengths.

use korppi_audio::raytrace::{
    AcousticGeometry, AcousticMaterial, PathTracer, TraceSettings, AIR_ABSORPTION_COEFFS,
    SPEED_OF_SOUND,
};
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

/// A material absorbing `alpha` in every band, scattering `scattering`, transmitting nothing.
fn material(alpha: f32, scattering: f32) -> AcousticMaterial {
    AcousticMaterial {
        absorption: [alpha; 3],
        scattering,
        transmission: [0.0; 3],
    }
}

fn add_box(world: &mut PhysicsWorld, center: Vec3, half_extents: Vec3) {
    let body = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: center,
        rotation: Quat::IDENTITY,
        ..RigidBody::default()
    });
    world.create_collider(body, Collider::new(Shape::Box { half_extents }));
}

/// A closed room of interior `size` (x, y up, z) from the origin, walls 0.5 m thick, all of
/// `wall`.
fn shoebox(size: Vec3, wall: AcousticMaterial) -> AcousticGeometry {
    let mut world = PhysicsWorld::new();
    let (half, t) = (size * 0.5, 0.25);
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            let mut center = half;
            let mut extents = half + Vec3::splat(2.0 * t);
            center[axis] = half[axis] + side * (half[axis] + t);
            extents[axis] = t;
            add_box(&mut world, center, extents);
        }
    }
    AcousticGeometry::from_physics(&world, |_| Some(wall))
}

/// The variance of a rectangular room's free path lengths relative to their squared mean
/// (Kuttruff, Room Acoustics: about 0.4 for rectangular rooms).
const PATH_VARIANCE: f32 = 0.4;

/// Eyring's RT60 with air absorption and Kuttruff's correction for the spread of free path
/// lengths, per band: 0.161 V / (-S ln(1 - α) (1 + γ²/2 ln(1 - α)) + 4 m V). Eyring alone assumes
/// a perfectly diffuse field and comes out short as absorption grows (16 % at α = 0.5 here).
fn eyring(size: Vec3, alpha: f32) -> [f32; 3] {
    let volume = size.x * size.y * size.z;
    let surface = 2.0 * (size.x * size.y + size.x * size.z + size.y * size.z);
    let constant = 24.0 * std::f32::consts::LN_10 / SPEED_OF_SOUND;
    let ln = (1.0 - alpha).ln();
    AIR_ABSORPTION_COEFFS.map(|m| {
        constant * volume / (-surface * ln * (1.0 + PATH_VARIANCE / 2.0 * ln) + 4.0 * m * volume)
    })
}

fn settings(rays: usize) -> TraceSettings {
    TraceSettings {
        rays,
        max_bounces: 400,
        receiver_radius: 1.0,
        ..TraceSettings::default()
    }
}

/// A diffuse shoebox room decays as Eyring's formula, corrected for its path lengths' spread,
/// predicts, in each band within 10 %.
#[test]
fn a_shoebox_room_decays_as_eyring_predicts() {
    let size = Vec3::new(10.0, 4.0, 6.0);
    for alpha in [0.2, 0.5] {
        let room = shoebox(size, material(alpha, 1.0));
        let mut tracer = PathTracer::new(7);
        let responses = tracer.trace(
            &room,
            Vec3::new(2.5, 1.7, 2.0),
            &[Vec3::new(7.0, 1.2, 4.0)],
            &settings(4000),
        );
        let measured = responses[0].rt60();
        let expected = eyring(size, alpha);
        for band in 0..3 {
            eprintln!(
                "α {alpha}, band {band}: RT60 {:.3} s, Eyring {:.3} s",
                measured[band], expected[band]
            );
            assert!(
                (measured[band] / expected[band] - 1.0).abs() < 0.10,
                "α {alpha}, band {band}: {} s, Eyring {} s",
                measured[band],
                expected[band]
            );
        }
    }
}

/// The settings take effect: every ray is traced, and paths bounce as often as allowed, no
/// more.
#[test]
fn the_ray_count_and_bounce_limit_take_effect() {
    let room = shoebox(Vec3::new(8.0, 3.0, 5.0), material(0.1, 0.5));
    let mut tracer = PathTracer::new(1);
    let listener = Vec3::new(2.0, 1.5, 2.0);
    let emitter = [Vec3::new(6.0, 1.5, 3.0)];
    for (rays, bounces) in [(100, 3), (700, 12)] {
        let response = &tracer.trace(
            &room,
            listener,
            &emitter,
            &TraceSettings {
                rays,
                max_bounces: bounces,
                ..TraceSettings::default()
            },
        )[0];
        assert_eq!(response.rays, rays);
        assert_eq!(
            response.deepest_bounce, bounces,
            "paths use all {bounces} bounces and no more"
        );
    }
}

/// Paths are random: two seeds give different responses, one seed the same.
#[test]
fn paths_are_random_and_reproducible() {
    let room = shoebox(Vec3::new(8.0, 3.0, 5.0), material(0.2, 1.0));
    let trace = |seed: u64| {
        PathTracer::new(seed).trace(
            &room,
            Vec3::new(2.0, 1.5, 2.0),
            &[Vec3::new(6.0, 1.5, 3.0)],
            &settings(300),
        )
    };
    let (a, b, again) = (trace(1), trace(2), trace(1));
    assert_ne!(
        a[0].histogram, b[0].histogram,
        "different seeds, different paths"
    );
    assert_ne!(
        trace(2)[0].histogram,
        trace(3)[0].histogram,
        "seeds differing in their lowest bit too"
    );
    assert_eq!(
        a[0].histogram, again[0].histogram,
        "the same seed, the same paths"
    );
}

/// Above a hard floor, the direct sound and the floor's reflection arrive at their geometric
/// delays: 4 m and the 5 m image-source path (to within 0.1 ms), from their directions.
#[test]
fn a_floor_reflection_arrives_from_its_image_source() {
    let mut world = PhysicsWorld::new();
    add_box(
        &mut world,
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(50.0, 0.5, 50.0),
    );
    let floor = AcousticGeometry::from_physics(&world, |_| Some(material(0.0, 0.0)));
    let (listener, emitter) = (Vec3::new(0.0, 1.5, 0.0), Vec3::new(4.0, 1.5, 0.0));
    let response = &PathTracer::new(3).trace(
        &floor,
        listener,
        &[emitter],
        &TraceSettings {
            rays: 20_000,
            max_bounces: 2,
            receiver_radius: 0.25,
            ..TraceSettings::default()
        },
    )[0];
    let first = |order: usize| {
        response
            .paths()
            .iter()
            .filter(|path| path.order == order)
            .min_by(|a, b| a.delay.total_cmp(&b.delay))
            .copied()
            .unwrap_or_else(|| panic!("no path of order {order}"))
    };
    let (direct, reflected) = (first(0), first(1));
    let close = |delay: f32, metres: f32| (delay - metres / SPEED_OF_SOUND).abs() < 1e-4;
    assert!(
        close(direct.delay, 4.0),
        "direct sound at {} s",
        direct.delay
    );
    assert!(
        close(reflected.delay, 5.0),
        "floor reflection at {} s",
        reflected.delay
    );
    assert!(
        direct.direction.dot(Vec3::X) > 0.99,
        "{:?}",
        direct.direction
    );
    let image = (Vec3::new(4.0, -1.5, 0.0) - listener).normalize();
    assert!(
        reflected.direction.dot(image) > 0.99,
        "{:?}",
        reflected.direction
    );
}

/// The receiver's estimate has the scale of a unit point source: in free field, the direct sound
/// at 4 m carries 1 / (4π d²) (less the air's share), to within 5 % (about 3 standard errors
/// with a million rays).
#[test]
fn direct_sound_has_the_intensity_of_a_point_source() {
    let free = AcousticGeometry::from_physics(&PhysicsWorld::new(), |_| None);
    let response = &PathTracer::new(11).trace(
        &free,
        Vec3::ZERO,
        &[Vec3::new(0.0, 0.0, -4.0)],
        &TraceSettings {
            rays: 1_000_000,
            max_bounces: 0,
            receiver_radius: 0.5,
            ..TraceSettings::default()
        },
    )[0];
    let received: f32 = response.paths().iter().map(|path| path.energy[1]).sum();
    let expected = (-AIR_ABSORPTION_COEFFS[1] * 4.0).exp() / (4.0 * std::f32::consts::PI * 16.0);
    eprintln!("direct intensity {received:.6}, point source {expected:.6}");
    assert!(
        (received / expected - 1.0).abs() < 0.05,
        "{received}, not {expected}"
    );
}
