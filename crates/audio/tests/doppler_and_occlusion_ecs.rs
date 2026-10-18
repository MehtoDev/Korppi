//! Verification test for velocity tracking (Doppler) and line-of-sight occlusion (AUDIT-5.6 /
//! Finding 4.6).
//!
//! Research reference: `research/audio/02_spatial_audio_and_psychoacoustics.md:80-110`.
//! Listener and emitter velocities from physics RigidBody components must drive Doppler pitch
//! shifts, and acoustic scene obstacles must drive line-of-sight occlusion muffling and
//! attenuation. Moved to the player path in FIX-7.2 (the owner's decision): before, it read the
//! old coordinator's voice parameters; now it measures what an `AudioPlayer` sounds like.

mod common;

use common::{channel, frequency, rms_db, Game, RATE, TICK};
use korppi_audio::player::{AudioOcclusion, AudioPlayer, AudioSource};
use korppi_audio::raytrace::{AcousticBox, AcousticMaterial, AcousticScene};
use korppi_audio::{AudioClip, AudioListener};
use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_scene::GlobalTransform;
use std::sync::Arc;

/// A game with a listener at the origin and a looping tone of `hz` on an entity at `at`.
fn tone_at(hz: f32, seconds: f32, at: Vec3) -> (Game, korppi_ecs::Entity) {
    let mut game = Game::new();
    game.clip("tone", AudioClip::sine(hz, seconds, RATE));
    let tone = game.start().remove(0);
    game.world.spawn((
        AudioListener::default(),
        GlobalTransform::from_translation(Vec3::ZERO),
    ));
    let emitter = game.world.spawn((
        AudioPlayer::new(AudioSource::clip(tone).looping()).autoplay(),
        GlobalTransform::from_translation(at),
    ));
    (game, emitter)
}

#[test]
fn test_ecs_rigid_body_velocities_drive_doppler_pitch_shift() {
    let (mut game, emitter) = tone_at(440.0, 1.0, Vec3::new(0.0, 0.0, 50.0));
    // Approaching the listener at 34.3 m/s (10 % of the speed of sound): about 343 / 308.7 ≈ 1.11.
    game.world.add_component(
        emitter,
        RigidBody {
            body_type: BodyType::Dynamic,
            linear_velocity: Vec3::new(0.0, 0.0, -34.3),
            ..Default::default()
        },
    );
    let out = channel(&game.ticks(8), 0);
    let approaching = frequency(&out[4 * TICK..]);
    assert!(
        approaching > 440.0 * 1.08,
        "an approaching emitter must sound higher than 1.08 × 440 Hz, got {approaching} Hz"
    );

    // Receding at 34.3 m/s: lower.
    game.world
        .get_mut::<RigidBody>(emitter)
        .unwrap()
        .linear_velocity = Vec3::new(0.0, 0.0, 34.3);
    let out = channel(&game.ticks(8), 0);
    let receding = frequency(&out[4 * TICK..]);
    assert!(
        receding < 440.0 * 0.95,
        "a receding emitter must sound lower than 0.95 × 440 Hz, got {receding} Hz"
    );
}

/// The level in dB of a looping tone of `hz` 10 m in front of the listener, with or without a
/// concrete wall halfway.
fn level(hz: f32, wall: bool) -> f32 {
    // Whole cycles in the clip, so that it loops without a seam.
    let (mut game, _) = tone_at(hz, 1.0, Vec3::new(0.0, 0.0, 10.0));
    if wall {
        let mut scene = AcousticScene::new();
        scene.add_box(AcousticBox::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(5.0, 5.0, 0.5),
            AcousticMaterial::concrete(),
        ));
        game.world.insert_resource(AudioOcclusion {
            raycaster: Some(Arc::new(scene)),
        });
    }
    let out = channel(&game.ticks(8), 0);
    rms_db(&out[4 * TICK..])
}

#[test]
fn test_ecs_occlusion_query_muffles_cutoff_and_attenuates_gain() {
    let (open_low, walled_low) = (level(440.0, false), level(440.0, true));
    assert!(
        walled_low < open_low - 1.0,
        "the wall must attenuate the tone: {walled_low} dB behind it, {open_low} dB without"
    );
    // Muffling: the wall's low-pass cut (below 10 kHz) takes much more off a 15 kHz tone.
    let (open_high, walled_high) = (level(15_000.0, false), level(15_000.0, true));
    let extra = (open_high - walled_high) - (open_low - walled_low);
    assert!(
        extra > 6.0,
        "behind the wall 15 kHz must lose 6 dB more than 440 Hz, lost {extra} dB more"
    );
}
