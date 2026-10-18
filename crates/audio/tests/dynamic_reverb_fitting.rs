//! Verification test for dynamic reverb parameter fitting and reflection taps from acoustic geometry (AUDIT-5.3 / Finding 4.3).
//!
//! Research reference: `research/audio/04_dsp_synthesis_for_raytraced_acoustics.md:40-100`.
//! Late reverberation RT60 and early reflection delay taps must adapt dynamically to room geometry
//! and surface materials rather than using static constants (room_size = 0.7, taps = [15ms, 28ms]).
//! Moved to the player path in FIX-7.5 (the owner's decision).

mod common;

use common::{Game, RATE};
use korppi_audio::asset::AudioClip;
use korppi_audio::ecs::{AudioListener, AudioSettings};
use korppi_audio::player::{AudioPlayer, AudioRoom, AudioSource};
use korppi_audio::raytrace::{AcousticGeometry, AcousticMaterial};
use korppi_audio::spatial::SpatializerMode;
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};
use korppi_scene::GlobalTransform;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Builds a closed box room of interior `size` centered at the origin with `material` surfaces.
fn shoebox_room(size: Vec3, material: AcousticMaterial) -> Arc<AcousticGeometry> {
    let mut world = PhysicsWorld::new();
    let (half, t) = (size * 0.5, 0.25);
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            let (mut center, mut extents) = (Vec3::ZERO, half + Vec3::splat(2.0 * t));
            center[axis] = side * (half[axis] + t);
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
    Arc::new(AcousticGeometry::from_physics(&world, |_| Some(material)))
}

/// Measures the late reverberant tail energy after ~107 ms (frame 5120) of an impulse played in a
/// 15 x 10 x 15 m room with `material` walls.
fn late_tail_energy(material: AcousticMaterial) -> f32 {
    let mut game = Game::new();
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;
    game.world.resource_mut::<AudioRoom>().unwrap().geometry =
        Some(shoebox_room(Vec3::new(15.0, 10.0, 15.0), material));

    game.clip("impulse", AudioClip::impulse(RATE));
    let impulse = game.start().remove(0);

    game.world.spawn((
        AudioListener {
            active: true,
            is_primary: true,
            volume: 1.0,
        },
        GlobalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    ));

    let emitter = game.world.spawn((
        AudioPlayer::new(AudioSource::clip(impulse)),
        GlobalTransform::from_translation(Vec3::new(0.0, 1.0, 2.0)),
    ));

    // Wait for the background worker to publish the room's RT60 and for `update_audio_players`
    // to apply it to the "Reverb" bus.
    let start = Instant::now();
    loop {
        game.tick();
        let room = game.world.resource::<AudioRoom>().unwrap();
        if room.rt60().is_some() && room.traces() >= 4 {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "Timed out waiting for RoomWorker to report RT60"
        );
        std::thread::sleep(Duration::from_millis(5));
    }

    game.player(emitter).play();

    // Render 20 ticks (16,000 frames ~ 333 ms) and accumulate late reverberant energy after
    // 5,120 frames (~107 ms, well past the direct sound and early reflections).
    let samples = game.ticks(20);
    samples[5_120 * 2..].iter().map(|s| s * s).sum()
}

#[test]
fn test_dynamic_reverb_fitting_live_vs_anechoic_room() {
    let live_late_energy = late_tail_energy(AcousticMaterial::concrete());
    let dead_late_energy = late_tail_energy(AcousticMaterial::acoustic_foam());

    assert!(
        live_late_energy > dead_late_energy * 2.0,
        "Live room must sustain significantly higher late reverberant energy than dead room: live={}, dead={}",
        live_late_energy,
        dead_late_energy
    );
}
