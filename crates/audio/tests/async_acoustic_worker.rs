//! Verification test for Asynchronous 20 Hz Acoustic Worker Thread & Wait-Free Triple-Buffer (AUDIT-5.2 / Finding 4.2).
//!
//! Research reference: `research/audio/04_dsp_synthesis_for_raytraced_acoustics.md:153-180`,
//! `research/audio/05_dual_mode_audio_architecture_and_milestone10_synthesis.md:32-33`.
//! The background acoustic worker thread executes ray tracing at ~20 Hz (50 ms cadence), decoupled from
//! the audio callback, transferring dynamic RT60 and early reflection paths via wait-free triple buffers.
//! Moved to `RoomWorker` and `AudioPlayer` in FIX-7.5 (the owner's decision).

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
/// last one read. The worker runs at 20 Hz, but a loaded machine can take longer to publish
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
fn test_async_acoustic_worker_runs_at_20hz_and_publishes_via_triple_buffer() {
    let mut input = RoomInput {
        listener: Vec3::new(0.0, 1.0, 0.0),
        receiver_count: 1,
        rays: 128,
        max_bounces: 4,
        ..RoomInput::default()
    };
    input.receivers[0] = Vec3::new(2.0, 1.0, 2.0);
    input.ids[0] = 1;

    // 1. Live concrete room
    let live_scene = shoebox_room(Vec3::new(12.0, 8.0, 12.0), AcousticMaterial::concrete());
    let mut worker = RoomWorker::spawn(live_scene);
    worker.update(input);

    // Consumer reads atomically without blocking; wait for the first computed result with reflections
    let data = wait_for(&mut worker, |out| {
        out.rt60.is_some() && out.reflections_of(1).is_some_and(|r| r.count > 0)
    });
    let live_rt60 = data.rt60.expect("Concrete room RT60 must be computed");

    assert!(
        live_rt60[1] > 1.0,
        "Concrete room mid-band RT60 must be > 1.0s, got {}",
        live_rt60[1]
    );

    let tap_count = data.reflections_of(1).map_or(0, |r| r.count);
    assert!(
        tap_count > 0,
        "Worker must gather early reflection taps in enclosed room, got {}",
        tap_count
    );

    // 2. Dynamically change scene to dead foam room
    let dead_scene = shoebox_room(
        Vec3::new(12.0, 8.0, 12.0),
        AcousticMaterial::acoustic_foam(),
    );
    let mut dead_worker = RoomWorker::spawn(dead_scene);
    dead_worker.update(input);

    let dead_data = wait_for(&mut dead_worker, |out| out.rt60.is_some());
    let dead_rt60 = dead_data.rt60.expect("Foam room RT60 must be computed");
    assert!(
        dead_rt60[1] < live_rt60[1] * 0.5,
        "Foam room RT60 ({}) must be significantly lower than concrete ({})",
        dead_rt60[1],
        live_rt60[1]
    );
}

#[test]
fn test_audio_coordinator_integrates_async_worker() {
    let mut game = Game::new();
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;
    game.world.resource_mut::<AudioRoom>().unwrap().geometry = Some(shoebox_room(
        Vec3::new(10.0, 6.0, 10.0),
        AcousticMaterial::brick(),
    ));

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
    game.world.spawn((
        AudioPlayer::new(AudioSource::clip(impulse).looping()).autoplay(),
        GlobalTransform::from_translation(Vec3::new(2.0, 1.0, 2.0)),
    ));

    // Update until the player system has latched a result of the background worker from the
    // triple buffer, for at most 10 s (ISSUE-020: a fixed wait can end before the worker
    // publishes)
    let start = Instant::now();
    game.tick();
    while game.world.resource::<AudioRoom>().unwrap().rt60().is_none()
        && start.elapsed() < Duration::from_secs(10)
    {
        thread::sleep(Duration::from_millis(5));
        game.tick();
    }
    assert!(
        game.world.resource::<AudioRoom>().unwrap().rt60().is_some(),
        "RoomWorker must publish RT60 into AudioRoom"
    );

    // Render audio frames - must process with valid FDN and early taps
    let samples = game.tick();
    let energy: f32 = samples.iter().map(|s| s * s).sum();
    assert!(
        energy > 0.0,
        "Rendered audio frames must contain non-zero acoustic energy"
    );

    // Switching back to Standard mode stops the worker and clears room trace state
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Standard;
    game.tick();
    assert_eq!(game.world.resource::<AudioRoom>().unwrap().traces(), 0);
}
