//! Raytraced mode's room (FIX-7.5, ADR-051; findings AUD-08 and AUD-11): the acoustics worker
//! traces the level's geometry at 20 Hz with the settings' ray count and bounce limit, and the
//! room's decay time drives the late reverb the players send to.
//!
//! With few bounces per ray, the decay time comes from the trace's statistics: the mean free path
//! between reflections and its spread, the mean absorption per reflection in each band, and the
//! air's share. In a diffuse room that is Eyring's formula with Kuttruff's correction, computed
//! from the room as traced.

mod common;

use common::{channel, Game, RATE};
use korppi_audio::player::{AudioPlayer, AudioRoom, AudioSource};
use korppi_audio::raytrace::{
    AcousticGeometry, AcousticMaterial, AIR_ABSORPTION_COEFFS, SPEED_OF_SOUND,
};
use korppi_audio::spatial::SpatializerMode;
use korppi_audio::{AudioClip, AudioListener, AudioSettings};
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};
use korppi_scene::GlobalTransform;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A closed room of interior `size` from the origin, walls of `alpha` absorption, diffuse.
fn shoebox(size: Vec3, alpha: f32) -> AcousticGeometry {
    let mut world = PhysicsWorld::new();
    let (half, t) = (size * 0.5, 0.25);
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            let (mut center, mut extents) = (half, half + Vec3::splat(2.0 * t));
            center[axis] = half[axis] + side * (half[axis] + t);
            extents[axis] = t;
            let body = world.create_body(RigidBody {
                body_type: BodyType::Static,
                position: center,
                rotation: Quat::IDENTITY,
                ..RigidBody::default()
            });
            world.create_collider(
                body,
                Collider::new(Shape::Box {
                    half_extents: extents,
                }),
            );
        }
    }
    let wall = AcousticMaterial {
        absorption: [alpha; 3],
        scattering: 1.0,
        transmission: [0.0; 3],
    };
    AcousticGeometry::from_physics(&world, |_| Some(wall))
}

/// Eyring's RT60 with Kuttruff's correction (γ² = 0.4) and air absorption, per band.
fn eyring(size: Vec3, alpha: f32) -> [f32; 3] {
    let volume = size.x * size.y * size.z;
    let surface = 2.0 * (size.x * size.y + size.x * size.z + size.y * size.z);
    let ln = (1.0 - alpha).ln();
    let constant = 24.0 * std::f32::consts::LN_10 / SPEED_OF_SOUND;
    AIR_ABSORPTION_COEFFS
        .map(|m| constant * volume / (-surface * ln * (1.0 + 0.2 * ln) + 4.0 * m * volume))
}

/// A game in raytraced mode inside `room`, a listener and a spatialized player of `clip` in it.
fn game(room: AcousticGeometry, clip: AudioClip, looping: bool) -> Game {
    let mut game = Game::new();
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;
    game.world.resource_mut::<AudioRoom>().unwrap().geometry = Some(Arc::new(room));
    game.clip("clip", clip);
    let handle = game.start().remove(0);
    game.world.spawn((
        AudioListener::default(),
        GlobalTransform::from_translation(Vec3::new(2.5, 1.7, 2.0)),
    ));
    let source = if looping {
        AudioSource::clip(handle).looping()
    } else {
        AudioSource::clip(handle)
    };
    game.world.spawn((
        AudioPlayer::new(source),
        GlobalTransform::from_translation(Vec3::new(7.0, 1.2, 4.0)),
    ));
    game
}

/// Ticks until the worker has reported the room's decay time, for at most 10 s.
fn room_rt60(game: &mut Game) -> [f32; 3] {
    let start = Instant::now();
    loop {
        game.tick();
        if let Some(rt60) = game.world.resource::<AudioRoom>().unwrap().rt60() {
            // A few more reports, so the estimate averages several traces.
            if start.elapsed() > Duration::from_millis(400) {
                return rt60;
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "no decay time from the worker"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The worker's decay time for a shoebox room, with the default 128 rays and 4 bounces, is the
/// corrected Eyring prediction within 7 % in each band, for two absorptions (without the
/// correction the estimate falls 10 % short at α = 0.5).
#[test]
fn the_worker_measures_the_room() {
    let size = Vec3::new(10.0, 4.0, 6.0);
    for alpha in [0.2, 0.5] {
        let mut game = game(
            shoebox(size, alpha),
            AudioClip::sine(500.0, 0.1, RATE),
            true,
        );
        let measured = room_rt60(&mut game);
        let expected = eyring(size, alpha);
        for band in 0..3 {
            eprintln!(
                "α {alpha}, band {band}: worker {:.3} s, Eyring {:.3} s",
                measured[band], expected[band]
            );
            assert!(
                (measured[band] / expected[band] - 1.0).abs() < 0.07,
                "α {alpha}, band {band}: {} s, not {} s",
                measured[band],
                expected[band]
            );
        }
    }
}

/// The late reverb takes the room's decay time: after a knock, the more absorbing room's tail
/// dies sooner.
#[test]
fn the_reverb_decays_with_the_room() {
    let tail = |alpha: f32| -> f32 {
        let mut game = game(
            shoebox(Vec3::new(10.0, 4.0, 6.0), alpha),
            AudioClip::sine(500.0, 0.02, RATE),
            false,
        );
        room_rt60(&mut game);
        let player = game
            .world
            .query_mut::<(korppi_ecs::Entity, &AudioPlayer)>()
            .next()
            .map(|(entity, _)| entity)
            .unwrap();
        game.player(player).play();
        // From 0.1 s to 0.3 s after the knock: only the reverb sounds.
        let out = channel(&game.ticks(24), 0);
        let energy = |range: std::ops::Range<usize>| -> f32 {
            out[range].iter().map(|s| s * s).sum::<f32>()
        };
        // dB lost from the first to the second 0.1 s of the tail.
        10.0 * (energy(4_800..9_600) / energy(9_600..14_400)).log10()
    };
    let (live, dead) = (tail(0.2), tail(0.5));
    eprintln!("tail decay over 0.1 s: {live:.1} dB at α 0.2, {dead:.1} dB at α 0.5");
    assert!(
        dead > live * 2.0,
        "the absorbing room's tail dies faster: {live} dB, {dead} dB"
    );
}
