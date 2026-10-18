//! Verification tests for Milestone REM-5 (Audio Subsystem Remediation).
//!
//! Tests:
//! 1. Auto-starting `RoomWorker` by default when raytraced mode is active (Finding 4.2 part 1).
//! 2. Zero scene deep-cloning in worker using `Arc<AcousticGeometry>` (Finding 4.2 part 2).
//! 3. Consumption of reflection metadata in `update_audio_players` from `RoomWorker`'s
//!    `TripleBuffer` without synchronous main thread tracing (Finding 4.1).
//!
//! Moved to the player path in FIX-7.5 (the owner's decision).

mod common;

use common::{Game, RATE};
use korppi_audio::asset::AudioClip;
use korppi_audio::ecs::{AudioListener, AudioSettings};
use korppi_audio::player::{AudioPlayer, AudioRoom, AudioSource};
use korppi_audio::raytrace::{
    AcousticGeometry, AcousticMaterial, RoomInput, RoomOutput, RoomWorker,
};
use korppi_audio::spatial::SpatializerMode;
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};
use korppi_scene::GlobalTransform;
use std::sync::Arc;
use std::thread;
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

/// Reads the worker's latest result until `done` accepts one, for at most 10 s, and returns the
/// last one read: a loaded machine can take longer than a few worker periods to publish
/// (ISSUE-020), so the test waits for the result instead of a fixed time.
fn wait_for(worker: &mut RoomWorker, done: impl Fn(&RoomOutput) -> bool) -> RoomOutput {
    let start = Instant::now();
    loop {
        let data = worker.output();
        if done(&data) || start.elapsed() > Duration::from_secs(10) {
            return data;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn test_raytraced_mode_auto_starts_async_worker() {
    let mut game = Game::new();
    game.world.resource_mut::<AudioRoom>().unwrap().geometry = Some(shoebox_room(
        Vec3::new(10.0, 6.0, 10.0),
        AcousticMaterial::brick(),
    ));
    game.clip("tone", AudioClip::sine(440.0, 0.2, RATE));
    let tone = game.start().remove(0);
    game.world.spawn((
        AudioListener::default(),
        GlobalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    ));
    game.world.spawn((
        AudioPlayer::new(AudioSource::clip(tone).looping()).autoplay(),
        GlobalTransform::from_translation(Vec3::new(2.0, 1.0, 2.0)),
    ));

    // Initially in Standard mode: worker must stay inactive
    game.tick();
    assert_eq!(
        game.world.resource::<AudioRoom>().unwrap().traces(),
        0,
        "Worker must be inactive initially in Standard mode"
    );

    // 1. Switching AudioSettings resource to Raytraced mode must auto-start worker
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;
    let start = Instant::now();
    while game.world.resource::<AudioRoom>().unwrap().traces() == 0
        && start.elapsed() < Duration::from_secs(10)
    {
        thread::sleep(Duration::from_millis(5));
        game.tick();
    }
    assert!(
        game.world.resource::<AudioRoom>().unwrap().traces() > 0,
        "Worker must auto-start via AudioSettings resource in Raytraced mode"
    );

    // 2. Switching back to Standard must stop worker to conserve CPU
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Standard;
    game.tick();
    assert_eq!(
        game.world.resource::<AudioRoom>().unwrap().traces(),
        0,
        "Worker must stop when switched to Standard mode"
    );
    assert!(
        game.world.resource::<AudioRoom>().unwrap().rt60().is_none(),
        "Room RT60 must clear when switched to Standard mode"
    );
}

#[test]
fn test_worker_uses_arc_acoustic_scene_without_deep_cloning() {
    let scene_arc = shoebox_room(Vec3::new(10.0, 6.0, 10.0), AcousticMaterial::brick());
    let initial_strong_count = Arc::strong_count(&scene_arc);

    let mut worker = RoomWorker::spawn(Arc::clone(&scene_arc));

    // Strong count must increase because worker holds an Arc reference without cloning geometry
    assert!(
        Arc::strong_count(&scene_arc) > initial_strong_count,
        "RoomWorker must share geometry via Arc without cloning geometry"
    );
    assert!(
        Arc::ptr_eq(worker.geometry(), &scene_arc),
        "RoomWorker must point to the exact same AcousticGeometry allocation"
    );

    let mut input = RoomInput {
        listener: Vec3::new(0.0, 1.0, 0.0),
        receiver_count: 1,
        rays: 128,
        max_bounces: 4,
        ..RoomInput::default()
    };
    input.receivers[0] = Vec3::new(2.0, 1.0, 2.0);
    input.ids[0] = 1;
    worker.update(input);

    let frame = wait_for(&mut worker, |out| {
        out.rt60.is_some() && out.reflections_of(1).is_some_and(|r| r.count > 0)
    });
    assert!(
        frame.rt60.unwrap()[1] > 0.0,
        "Worker must successfully compute RT60 using Arc<AcousticGeometry>"
    );
    assert!(
        frame.reflections_of(1).unwrap().count > 0,
        "Worker must successfully trace early reflection taps using Arc<AcousticGeometry>"
    );

    drop(worker);
    assert_eq!(
        Arc::strong_count(&scene_arc),
        initial_strong_count,
        "Dropping RoomWorker must join its thread and release its Arc<AcousticGeometry>"
    );
}

#[test]
fn test_raytraced_spatializer_consumes_triple_buffer_without_synchronous_tracing() {
    // Compare an AudioPlayer's early reflection energy in free field vs an enclosed reflective
    // room after consuming RoomWorker's triple-buffered output.
    let render_early_energy = |enclosed: bool| -> f32 {
        let mut game = Game::new();
        game.world
            .resource_mut::<AudioSettings>()
            .unwrap()
            .spatializer_mode = SpatializerMode::Raytraced;

        let geometry = if enclosed {
            shoebox_room(Vec3::new(10.0, 4.0, 10.0), AcousticMaterial::concrete())
        } else {
            let empty = PhysicsWorld::new();
            Arc::new(AcousticGeometry::from_physics(&empty, |_| None))
        };
        game.world.resource_mut::<AudioRoom>().unwrap().geometry = Some(geometry);

        game.clip("impulse", AudioClip::impulse(RATE));
        let impulse = game.start().remove(0);

        game.world.spawn((
            AudioListener::default(),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        ));
        let player = game.world.spawn((
            AudioPlayer::new(AudioSource::clip(impulse)),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -3.0)),
        ));

        // Wait for RoomWorker to publish into its triple buffer and for update_audio_players
        // to consume it on the game thread.
        let start = Instant::now();
        while game.world.resource::<AudioRoom>().unwrap().traces() < 3 {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "Timed out waiting for RoomWorker"
            );
            thread::sleep(Duration::from_millis(5));
            game.tick();
        }

        game.player(player).play();
        // First 1400 frames (~29 ms): after the direct arrival (at 3m / 343 ≈ 420 frames),
        // before the FDN late reverb (shortest delay 1499 frames), only the early reflection
        // taps consumed from the worker's triple buffer sound.
        let samples = game.ticks(2);
        samples[500 * 2..1400 * 2].iter().map(|s| s * s).sum()
    };

    let free_early = render_early_energy(false);
    let room_early = render_early_energy(true);

    assert!(
        room_early > free_early + 1e-4,
        "Early reflection energy consumed from worker's triple buffer ({room_early}) must exceed free field ({free_early})"
    );
}
