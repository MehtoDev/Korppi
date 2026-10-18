//! The engine's audio output (FIX-7.2, ADR-050; finding AUD-01): the engine opens an output, puts
//! the voice API in the world, and runs the audio players after transforms propagate. Headless,
//! the output follows game time, so a one-shot finishes after its length of updates.

use korppi_asset::{AssetServer, MemoryFileSystem, Vfs};
use korppi_audio::mixer::AudioVoices;
use korppi_audio::output::AudioBackend;
use korppi_audio::{
    AudioClip, AudioPlayer, AudioSettings, AudioSource, PlayerState, WavAudioLoader,
};
use korppi_engine::{EngineApp, EngineConfig};
use std::time::Duration;

#[test]
fn players_run_through_the_engine() {
    let mut app = EngineApp::new().expect("engine");
    assert!(app.audio_output.is_headless(), "headless by default");
    assert!(app.world.contains_resource::<AudioVoices>());
    assert!(app.world.contains_resource::<AudioSettings>());

    let mut files = MemoryFileSystem::new();
    files.insert(
        "shot.wav",
        AudioClip::sine(440.0, 0.1, 48_000).to_wav_bytes(),
    );
    let mut vfs = Vfs::new();
    vfs.mount("", Box::new(files));
    let assets = AssetServer::new(vfs);
    assets.register_loader(WavAudioLoader);
    let shot = assets.load_sync::<WavAudioLoader>("shot.wav").unwrap();
    app.world.insert_resource(assets);
    let player = app
        .world
        .spawn((AudioPlayer::new(AudioSource::clip(shot)).autoplay(),));

    let tick = Duration::from_secs_f64(1.0 / 60.0);
    app.update(tick);
    assert_eq!(
        app.world.get::<AudioPlayer>(player).unwrap().state(),
        PlayerState::Playing
    );
    // 0.1 s is six ticks; the seventh update sees it ended.
    for _ in 0..7 {
        app.update(tick);
    }
    assert_eq!(
        app.world.get::<AudioPlayer>(player).unwrap().state(),
        PlayerState::Finished
    );
}

/// A game asks for the device; where there is none, the engine falls back to headless output
/// (unless `KORPPI_REQUIRE_AUDIO=1`).
#[test]
fn a_game_asks_for_the_device() {
    let config = EngineConfig {
        audio: AudioBackend::Device,
        ..EngineConfig::default()
    };
    let app = EngineApp::with_config(config).expect("engine");
    let required = std::env::var("KORPPI_REQUIRE_AUDIO").is_ok_and(|v| v == "1");
    if required {
        assert!(!app.audio_output.is_headless(), "no output device opened");
    }
    eprintln!(
        "audio output: {} ({})",
        app.audio_output.device_name(),
        if app.audio_output.is_headless() {
            "headless"
        } else {
            "device"
        }
    );
}

/// The level's static colliders are the acoustic scene: a static concrete wall entity between
/// the listener and a player muffles and attenuates it, with nothing else set up, and stops once
/// the wall is despawned (FIX-7.5, ADR-051).
#[test]
fn static_walls_occlude_players() {
    use korppi_audio::raytrace::AcousticMaterial;
    use korppi_audio::AudioListener;
    use korppi_math::Vec3;
    use korppi_physics::{BodyType, Collider, RigidBody, Shape};
    use korppi_scene::LocalTransform;

    let mut app = EngineApp::new().expect("engine");
    let mut files = MemoryFileSystem::new();
    files.insert(
        "hum.wav",
        // 375 Hz: exactly two cycles in each 256-frame block, so a block's level does not depend
        // on where the tone's cycle falls in it.
        AudioClip::sine(375.0, 1.0, 48_000).to_wav_bytes(),
    );
    let mut vfs = Vfs::new();
    vfs.mount("", Box::new(files));
    let assets = AssetServer::new(vfs);
    assets.register_loader(WavAudioLoader);
    let hum = assets.load_sync::<WavAudioLoader>("hum.wav").unwrap();
    app.world.insert_resource(assets);
    app.world
        .spawn((AudioListener::default(), LocalTransform::default()));
    app.world.spawn((
        AudioPlayer::new(AudioSource::clip(hum).looping()).autoplay(),
        LocalTransform {
            translation: Vec3::new(0.0, 0.0, -10.0),
            ..LocalTransform::default()
        },
    ));
    // Each update renders one whole 256-frame block, the one `last_output` holds.
    let level = |app: &mut EngineApp| -> f32 {
        for _ in 0..60 {
            app.update(Duration::from_secs_f64(256.0 / 48_000.0));
        }
        let out = app.audio_output.last_output();
        let mean = out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32;
        10.0 * mean.max(1e-20).log10()
    };
    let open = level(&mut app);
    let wall = app.world.spawn((
        RigidBody {
            body_type: BodyType::Static,
            position: Vec3::new(0.0, 0.0, -5.0),
            ..RigidBody::default()
        },
        Collider::new(Shape::Box {
            half_extents: Vec3::new(5.0, 5.0, 0.25),
        }),
        AcousticMaterial::concrete(),
    ));
    let walled = level(&mut app);
    assert!(
        walled < open - 1.0,
        "the wall attenuates the player: {walled} dB behind it, {open} dB without"
    );
    app.world.despawn(wall);
    let cleared = level(&mut app);
    assert!(
        (cleared - open).abs() < 0.5,
        "without the wall it is heard as before: {cleared} dB, {open} dB"
    );
}
