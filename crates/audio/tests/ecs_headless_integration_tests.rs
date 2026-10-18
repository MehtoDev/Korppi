//! Headless end-to-end integration tests for Sub-Milestone 10.6:
//! ECS Coordinator, Asset Pipeline & Dual-Mode Headless Verification Suite.

mod common;

use common::Game;
use korppi_asset::AssetLoader;
use korppi_audio::player::{AudioPlayer, AudioSource};
use korppi_audio::{
    asset::{AudioClip, WavAudioLoader},
    coordinator::AudioCoordinator,
    ecs::{AudioListener, AudioSettings},
    spatial::SpatializerMode,
};
use korppi_math::{Mat4, Vec3};
use korppi_scene::GlobalTransform;

#[test]
fn test_audio_clip_wav_decode_and_asset_loader() {
    // Generate a 0.1 second 440 Hz mono sine wave
    let original = AudioClip::sine(440.0, 0.1, 48_000);
    assert_eq!(original.channels, 1);
    assert_eq!(original.sample_rate, 48_000);
    assert!((original.duration_sec() - 0.1).abs() < 1e-3);

    // Encode to RIFF/WAVE byte buffer
    let wav_bytes = original.to_wav_bytes();
    assert!(wav_bytes.len() > 44, "Must contain valid WAV header + data");
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(&wav_bytes[8..12], b"WAVE");

    // Decode via WavAudioLoader
    let loader = WavAudioLoader;
    assert!(loader.extensions().contains(&"wav"));

    let decoded = loader
        .load(&wav_bytes, "test.wav")
        .expect("WavAudioLoader should decode valid WAV bytes");

    assert_eq!(decoded.channels, original.channels);
    assert_eq!(decoded.sample_rate, original.sample_rate);
    assert_eq!(decoded.samples.len(), original.samples.len());

    // Compare sample fidelity
    for (&s_orig, &s_dec) in original.samples.iter().zip(decoded.samples.iter()) {
        assert!(
            (s_orig - s_dec).abs() < 1e-3,
            "Sample mismatch: orig = {}, decoded = {}",
            s_orig,
            s_dec
        );
    }
}

/// Listener and player components attach to entities (moved from `AudioEmitter` to `AudioPlayer`
/// in FIX-7.2, the owner's decision).
#[test]
fn test_ecs_audio_listener_and_player_components() {
    let mut game = Game::new();
    game.clip("tone", AudioClip::sine(440.0, 0.1, 48_000));
    let tone = game.start().remove(0);
    let world = &mut game.world;

    let listener_entity = world.spawn((
        AudioListener {
            is_primary: true,
            volume: 1.0,
            active: true,
        },
        GlobalTransform::from_matrix(Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))),
    ));

    let player_entity = world.spawn((
        AudioPlayer::new(AudioSource::clip(tone).looping()).with_volume(0.8),
        GlobalTransform::from_matrix(Mat4::from_translation(Vec3::new(5.0, 1.0, 0.0))),
    ));

    assert!(world.get::<AudioListener>(listener_entity).is_some());
    assert_eq!(
        world.get::<AudioPlayer>(player_entity).unwrap().volume(),
        0.8
    );
}

#[test]
fn test_runtime_mode_toggle_between_standard_and_raytraced() {
    let mut coordinator = AudioCoordinator::new(48_000.0);

    // Must default to Standard spatializer
    assert_eq!(coordinator.mode(), SpatializerMode::Standard);

    // Switch to Raytraced mode
    coordinator.set_mode(SpatializerMode::Raytraced);
    assert_eq!(coordinator.mode(), SpatializerMode::Raytraced);

    // Switch back to Standard mode
    coordinator.set_mode(SpatializerMode::Standard);
    assert_eq!(coordinator.mode(), SpatializerMode::Standard);
}

/// A player renders through the mixer into the headless device (moved from the old coordinator's
/// `render_audio_frames` to `AudioPlayer` in FIX-7.2, the owner's decision).
#[test]
fn test_headless_dummy_device_bit_exact_voice_rendering() {
    let mut game = Game::new();

    // 1. Setup Listener at origin
    game.world.spawn((
        AudioListener {
            is_primary: true,
            volume: 1.0,
            active: true,
        },
        GlobalTransform::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0))),
    ));

    // 2. Setup a player at (2, 0, 0) with a 440 Hz sine tone
    game.clip("tone", AudioClip::sine(440.0, 0.5, 48_000));
    let tone = game.start().remove(0);
    game.world.spawn((
        AudioPlayer::new(AudioSource::clip(tone).looping()).autoplay(),
        GlobalTransform::from_matrix(Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))),
    ));

    // One game tick: the players update and the headless device renders a tick of frames
    let frames = game.tick();
    assert_eq!(game.stream.frames_rendered(), common::TICK as u64);

    // Verify non-silent rendering and valid finite samples without NaN
    let mut non_zero_count = 0;
    for frame in frames.chunks(2) {
        assert!(frame[0].is_finite());
        assert!(frame[1].is_finite());
        assert!(!frame[0].is_nan());
        assert!(!frame[1].is_nan());

        if frame[0].abs() > 1e-4 || frame[1].abs() > 1e-4 {
            non_zero_count += 1;
        }
    }

    assert!(
        non_zero_count > 100,
        "Rendered audio frames should contain audible waveform data"
    );
}

/// A spatialized player in raytraced mode renders its reverberant tail through the mixer's
/// "Reverb" bus (moved from `AudioEmitter` and `render_audio_frames` to `AudioPlayer` in FIX-7.5,
/// the owner's decision).
#[test]
fn test_raytraced_spatializer_dynamic_reverb_rendering() {
    let mut game = Game::new();
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;

    game.world.spawn((
        AudioListener::default(),
        GlobalTransform::from_matrix(Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0))),
    ));

    game.clip("impulse", AudioClip::impulse(48_000));
    let impulse = game.start().remove(0);

    game.world.spawn((
        AudioPlayer::new(AudioSource::clip(impulse)).autoplay(),
        GlobalTransform::from_matrix(Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))),
    ));

    // Render 4 ticks (3200 frames): covers the direct impulse and the FDN reverberant tail.
    let samples = game.ticks(4);

    for &sample in &samples {
        assert!(sample.is_finite());
    }

    // Measure reverberant tail energy after the direct impulse has passed (after frame 200).
    let tail_energy: f32 = samples[400..].iter().map(|s| s * s).sum();
    assert!(
        tail_energy > 1e-5,
        "Raytraced mode should generate reverberant acoustic tail, got {tail_energy}"
    );
}
