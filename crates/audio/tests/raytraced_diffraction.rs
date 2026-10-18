//! Integration tests for raytraced edge diffraction and bounce rays (FIX-7.5, ADR-051;
//! findings AUD-08, AUD-10, AUD-11).
//!
//! Sound occluded by static physics colliders reaches the listener by wave diffraction around
//! geometric edges (BTM first and second order) and reflections around corners (next-event
//! estimation shadow rays). In raytraced mode, voices are attenuated and filtered by
//! sqrt(transmission² + diffraction²), replacing heuristic occlusion.

mod common;

use common::{channel, Game, RATE};
use korppi_audio::ecs::{AudioListener, AudioSettings};
use korppi_audio::player::{AudioPlayer, AudioRoom, AudioSource};
use korppi_audio::raytrace::{
    diffracted_sound, AcousticGeometry, AcousticMaterial, PathTracer, TraceSettings,
};
use korppi_audio::spatial::SpatializerMode;
use korppi_audio::AudioClip;
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, ColliderId3d, Mesh3d, PhysicsWorld, RigidBody, Shape};
use korppi_scene::GlobalTransform;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn add_body(
    world: &mut PhysicsWorld,
    body_type: BodyType,
    position: Vec3,
    rotation: Quat,
    shape: Shape,
) -> ColliderId3d {
    let body = world.create_body(RigidBody {
        body_type,
        position,
        rotation,
        ..RigidBody::default()
    });
    world.create_collider(body, Collider::new(shape))
}

fn box_shape(half_extents: Vec3) -> Shape {
    Shape::Box { half_extents }
}

/// Static box colliders precompute their 12 wedges and face-sharing pairs in world space,
/// and triangle meshes precompute boundary and crease edges.
#[test]
fn acoustic_geometry_precomputes_edges_from_colliders() {
    let mut world = PhysicsWorld::new();
    let pos = Vec3::new(1.0, 2.0, 3.0);
    let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    add_body(
        &mut world,
        BodyType::Static,
        pos,
        rot,
        box_shape(Vec3::new(2.0, 1.0, 0.5)),
    );

    let quad = Mesh3d::new(
        &[
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ],
        &[[0, 1, 2], [0, 2, 3]],
    )
    .expect("quad mesh");
    add_body(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 10.0, 0.0),
        Quat::IDENTITY,
        Shape::TriangleMesh(quad),
    );

    let geometry = AcousticGeometry::from_physics(&world, |_| Some(AcousticMaterial::concrete()));
    assert_eq!(geometry.surface_count(), 2);

    let box_edges = geometry.surface_wedges(0);
    assert_eq!(box_edges.len(), 12, "a box must have 12 diffraction edges");
    for edge in box_edges {
        assert!(
            (edge.angle - 1.5 * std::f32::consts::PI).abs() < 1e-4,
            "box edge exterior angle is 3pi/2"
        );
        assert!(
            (edge.face.length() - 1.0).abs() < 1e-4,
            "face direction is normalized"
        );
        assert!(
            (edge.normal.length() - 1.0).abs() < 1e-4,
            "normal is normalized"
        );
        assert!(
            edge.face.dot(edge.normal).abs() < 1e-4,
            "face and normal are orthogonal"
        );
    }

    let mesh_edges = geometry.surface_wedges(1);
    // The quad mesh has 4 boundary edges
    assert!(
        mesh_edges.len() >= 4,
        "mesh must have at least boundary edges"
    );
    for edge in mesh_edges {
        assert!(
            edge.angle > std::f32::consts::PI,
            "mesh exterior diffraction angle > pi"
        );
    }
}

/// Sound bending around an edge of a blocking box produces first-order BTM diffraction.
#[test]
fn first_order_diffraction_around_a_corner() {
    let mut world = PhysicsWorld::new();
    // A wall 4m wide along X, 4m high along Y, 0.2m thick along Z, centered at origin
    add_body(
        &mut world,
        BodyType::Static,
        Vec3::ZERO,
        Quat::IDENTITY,
        box_shape(Vec3::new(2.0, 2.0, 0.1)),
    );
    let geometry = AcousticGeometry::from_physics(&world, |_| Some(AcousticMaterial::concrete()));

    // Source is on one side, receiver is in shadow on the other side
    let source = Vec3::new(-3.0, 0.0, 2.0);
    let receiver = Vec3::new(-1.0, 0.0, -2.0);

    let gains = diffracted_sound(&geometry, source, receiver);
    eprintln!("diffracted gains around corner: {gains:?}");

    assert!(gains[0] > 0.0, "low frequencies diffract around the edge");
    assert!(gains[1] > 0.0, "mid frequencies diffract around the edge");
    assert!(
        gains[0] > gains[1],
        "bass diffracts more strongly than midrange: {} > {}",
        gains[0],
        gains[1]
    );
    assert!(
        gains[1] > gains[2],
        "midrange diffracts more strongly than treble: {} > {}",
        gains[1],
        gains[2]
    );

    // Deeper into the shadow (further behind the wall) must be quieter
    let deep_receiver = Vec3::new(0.0, 0.0, -2.0);
    let deep_gains = diffracted_sound(&geometry, source, deep_receiver);
    assert!(
        deep_gains[0] < gains[0],
        "deeper shadow is quieter: {} < {}",
        deep_gains[0],
        gains[0]
    );
}

/// Behind a thick wall, neither front nor back edge sees both source and receiver;
/// second-order BTM over the face-sharing pair carries sound.
#[test]
fn second_order_diffraction_behind_a_thick_wall() {
    let mut world = PhysicsWorld::new();
    // Wall 10m wide along X, 2m high (y in [-1, 1]), 1m thick (z in [-0.5, 0.5])
    add_body(
        &mut world,
        BodyType::Static,
        Vec3::ZERO,
        Quat::IDENTITY,
        box_shape(Vec3::new(5.0, 1.0, 0.5)),
    );
    let geometry = AcousticGeometry::from_physics(&world, |_| Some(AcousticMaterial::concrete()));

    // Source in front below top edge, receiver behind below top edge
    let source = Vec3::new(0.0, 0.0, 3.0);
    let receiver = Vec3::new(0.0, 0.0, -3.0);

    let gains = diffracted_sound(&geometry, source, receiver);
    eprintln!("second-order diffracted gains: {gains:?}");

    assert!(
        gains[0] > 0.0,
        "second-order diffraction reaches behind thick wall: {}",
        gains[0]
    );
    assert!(
        gains[0] > gains[1],
        "bass diffracts more than mid: {} > {}",
        gains[0],
        gains[1]
    );
    assert!(
        gains[1] > gains[2],
        "mid diffracts more than treble: {} > {}",
        gains[1],
        gains[2]
    );

    // A plate thinner than 1 cm (gap < 0.01) collapses its edge pair to a half-plane (2pi).
    let mut thin_world = PhysicsWorld::new();
    add_body(
        &mut thin_world,
        BodyType::Static,
        Vec3::ZERO,
        Quat::IDENTITY,
        box_shape(Vec3::new(5.0, 1.0, 0.002)),
    );
    let thin_geom =
        AcousticGeometry::from_physics(&thin_world, |_| Some(AcousticMaterial::concrete()));
    let thin_gains = diffracted_sound(&thin_geom, source, receiver);
    assert!(
        thin_gains[0] > gains[0],
        "thin plate (< 1 cm) diffracts more strongly than 1 m thick wall: {} > {}",
        thin_gains[0],
        gains[0]
    );
}

/// Next-event estimation shadow rays in PathTracer gather energy reflected around a corner
/// from an adjacent reflecting surface.
#[test]
fn next_event_estimation_shadow_rays_reach_around_corners() {
    let listener = Vec3::new(-2.0, 1.0, 0.0);
    let emitter = Vec3::new(2.0, 1.0, 0.0);
    let settings = TraceSettings {
        rays: 128,
        max_bounces: 3,
        ..TraceSettings::default()
    };

    // With only the blocking wall, no bounce can see the emitter.
    let mut wall_only = PhysicsWorld::new();
    add_body(
        &mut wall_only,
        BodyType::Static,
        Vec3::new(0.0, 2.0, 0.0),
        Quat::IDENTITY,
        box_shape(Vec3::new(0.1, 2.0, 2.0)),
    );
    let wall_geom =
        AcousticGeometry::from_physics(&wall_only, |_| Some(AcousticMaterial::concrete()));
    let mut tracer = PathTracer::new(0xB00B5);
    let wall_resp = tracer.trace(&wall_geom, listener, &[emitter], &settings);
    assert_eq!(
        wall_resp[0].bounce_energy(),
        [0.0; 3],
        "without a reflecting surface around the wall, bounce energy is zero"
    );

    // Adding a reflecting floor at y = 0 lets bounce shadow rays reach the emitter around the
    // bottom of the wall (Wait: the wall is y in [0, 4], z in [-2, 2], so add a side wall at z = -3
    // that sees both listener and emitter around z = -2!).
    let mut world = wall_only;
    add_body(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 2.0, -3.0),
        Quat::IDENTITY,
        box_shape(Vec3::new(10.0, 4.0, 0.1)),
    );
    let geometry = AcousticGeometry::from_physics(&world, |_| Some(AcousticMaterial::concrete()));

    let mut tracer = PathTracer::new(0xB00B5);
    let responses = tracer.trace(&geometry, listener, &[emitter], &settings);
    assert_eq!(responses.len(), 1);
    let bounce = responses[0].bounce_energy();
    eprintln!("bounce energy around corner: {bounce:?}");
    assert!(
        bounce[0] > 0.0 && bounce[1] > 0.0 && bounce[2] > 0.0,
        "bounce shadow rays gather reflected energy around the corner: {bounce:?}"
    );
    assert!(
        bounce[0] > bounce[2],
        "low frequencies retain more reflected bounce energy than treble: {} > {}",
        bounce[0],
        bounce[2]
    );
}

/// Measures the single-frequency DFT amplitude of `samples` at `freq_hz`.
fn tone_amplitude(samples: &[f32], freq_hz: f32) -> f32 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let step = -2.0 * std::f64::consts::PI * f64::from(freq_hz) / f64::from(RATE);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &s) in samples.iter().enumerate() {
        let phase = step * i as f64;
        re += f64::from(s) * phase.cos();
        im += f64::from(s) * phase.sin();
    }
    (2.0 * (re * re + im * im).sqrt() / n as f64) as f32
}

/// In raytraced mode, an occluded player's voice receives broadband gain and 2-pole Butterworth
/// low-pass filtering from sqrt(transmission² + diffraction²), replacing heuristic occlusion.
#[test]
fn raytraced_player_uses_physical_diffraction_and_transmission() {
    let listener_pos = Vec3::new(0.0, 0.0, 2.0);
    let emitter_pos = Vec3::new(-1.0, 0.0, -2.0);

    // Measure the rendered 250 Hz and 4000 Hz amplitudes for a wall with `transmission`.
    let measure = |transmission: [f32; 3]| -> (f32, f32, [f32; 3]) {
        let mut world = PhysicsWorld::new();
        add_body(
            &mut world,
            BodyType::Static,
            Vec3::ZERO,
            Quat::IDENTITY,
            box_shape(Vec3::new(2.0, 2.0, 0.1)),
        );
        let material = AcousticMaterial {
            absorption: [1.0; 3],
            scattering: 0.0,
            transmission,
        };
        let geometry = Arc::new(AcousticGeometry::from_physics(&world, |_| Some(material)));
        let expected_diff = diffracted_sound(&geometry, emitter_pos, listener_pos);

        let frames = RATE as usize;
        let samples: Vec<f32> = (0..frames)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                0.4 * (2.0 * std::f32::consts::PI * 250.0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * 4000.0 * t).sin()
            })
            .collect();

        let mut game = Game::new();
        game.world
            .resource_mut::<AudioSettings>()
            .unwrap()
            .spatializer_mode = SpatializerMode::Raytraced;
        game.world.resource_mut::<AudioRoom>().unwrap().geometry = Some(geometry);
        game.clip("two_tone", AudioClip::new(1, RATE, samples));
        let clip = game.start().remove(0);

        game.world.spawn((
            AudioListener::default(),
            GlobalTransform::from_translation(listener_pos),
        ));
        let player = game.world.spawn((
            AudioPlayer::new(AudioSource::clip(clip).looping()),
            GlobalTransform::from_translation(emitter_pos),
        ));

        // Wait for the acoustics worker to trace the player's diffraction before playing.
        let start = Instant::now();
        while game.world.resource::<AudioRoom>().unwrap().traces() < 3 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "worker did not complete traces"
            );
            game.tick();
            std::thread::sleep(Duration::from_millis(5));
        }

        // Start playback: the first tick (800 frames = 16.7 ms) is before the FDN late reverb's
        // shortest delay line (1499 frames = 31.2 ms), isolating the direct/diffracted voice.
        // Skip the first 224 frames for the 2-pole Butterworth filter transient, then take 576
        // frames (exact integer cycles of both 250 Hz and 4000 Hz at 48 kHz).
        game.player(player).play();
        let out = channel(&game.tick(), 0);
        let window = &out[224..800];
        (
            tone_amplitude(window, 250.0),
            tone_amplitude(window, 4000.0),
            expected_diff,
        )
    };

    let (amp_250, amp_4000, expected_diff) = measure([0.0; 3]);
    assert!(
        expected_diff[0] > expected_diff[2] && expected_diff[2] > 0.0,
        "expected diffraction must low-pass the signal: {expected_diff:?}"
    );
    eprintln!(
        "diffracted player output: 250 Hz = {amp_250:.5}, 4000 Hz = {amp_4000:.5}, ratio = {:.4} (expected {:.4})",
        amp_4000 / amp_250,
        expected_diff[2] / expected_diff[0]
    );

    assert!(
        amp_250 > 1e-3,
        "diffraction carries sound around an opaque wall even with zero transmission: {amp_250}"
    );
    let measured_ratio = amp_4000 / amp_250;
    let target_ratio = expected_diff[2] / expected_diff[0];
    assert!(
        (measured_ratio - target_ratio).abs() < 0.02,
        "2-pole Butterworth filter matches the 4 kHz / 250 Hz diffraction ratio: got {measured_ratio:.4}, expected {target_ratio:.4}"
    );

    // With transmission = [0.5, 0.25, 0.1], transmission and diffraction combine in quadrature:
    // sqrt(0.5² + diff[0]²) / diff[0].
    let (trans_amp_250, _, _) = measure([0.5, 0.25, 0.1]);
    let expected_scale =
        (0.5f32 * 0.5 + expected_diff[0] * expected_diff[0]).sqrt() / expected_diff[0];
    let measured_scale = trans_amp_250 / amp_250;
    eprintln!(
        "quadrature transmission + diffraction scale: got {measured_scale:.4}, expected {expected_scale:.4}"
    );
    assert!(
        (measured_scale - expected_scale).abs() < 0.02,
        "transmission and diffraction combine in quadrature: got {measured_scale:.4}, expected {expected_scale:.4}"
    );
}
