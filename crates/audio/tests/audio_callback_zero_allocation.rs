//! Verification test for zero heap allocations during audio frame rendering (AUDIT-5.5 / Finding 4.5).
//!
//! Real-time audio rendering must NEVER allocate on the heap during callback execution.

use korppi_audio::asset::AudioClip;
use korppi_audio::ecs::AudioListener;
use korppi_ecs::World;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static DEALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract. The counter only touches const-initialized `Cell` thread-locals, which
// never allocate, so counting cannot recurse into the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = TRACKING.try_with(|tracking| {
            if tracking.get() {
                let _ = ALLOC_COUNT.try_with(|count| {
                    count.set(count.get() + 1);
                });
            }
        });
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = TRACKING.try_with(|tracking| {
            if tracking.get() {
                let _ = DEALLOC_COUNT.try_with(|count| {
                    count.set(count.get() + 1);
                });
            }
        });
        // SAFETY: forwarded unchanged; the caller upholds `dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn set_tracking(enable: bool) {
    TRACKING.with(|t| t.set(enable));
}

fn reset_alloc_count() {
    ALLOC_COUNT.with(|c| c.set(0));
    DEALLOC_COUNT.with(|c| c.set(0));
}

fn get_dealloc_count() -> usize {
    DEALLOC_COUNT.with(|c| c.get())
}

fn get_alloc_count() -> usize {
    ALLOC_COUNT.with(|c| c.get())
}

/// The mixer neither allocates nor frees while rendering (rule 4, FIX-7.2): voices start, stop,
/// loop and finish, and released clips go back to the game thread inside events instead of
/// being dropped on the audio thread.
#[test]
fn the_mixer_renders_without_allocating_or_freeing() {
    use korppi_audio::mixer::{AudioMixer, ClipData, MixerConfig, VoiceParams};
    use korppi_audio::{AudioCallback, AudioCallbackInfo, AudioStreamConfig, ChannelCount};
    use std::sync::Arc;

    let config = AudioStreamConfig::default();
    let (mut control, mut mixer) = AudioMixer::new(MixerConfig::for_stream(&config));
    let info = AudioCallbackInfo {
        sample_rate: 48_000,
        channels: ChannelCount::Stereo,
        frame_count: 256,
        frames_rendered: 0,
    };
    let mut output = vec![0.0f32; 512];
    let short = Arc::new(ClipData::from(AudioClip::sine(440.0, 0.01, 44_100)));
    let long = Arc::new(ClipData::from(AudioClip::sine(220.0, 0.5, 48_000)));
    let mut tracked = |mixer: &mut AudioMixer| {
        set_tracking(true);
        mixer.render(&mut output, &info);
        set_tracking(false);
    };

    for round in 0..20 {
        let a = control.register_clip(Arc::clone(&short)).unwrap();
        let b = control.register_clip(Arc::clone(&long)).unwrap();
        let shot = control.start(a, VoiceParams::default()).unwrap();
        let looped = control
            .start(
                b,
                VoiceParams {
                    looping: true,
                    pitch: 1.5,
                    ..VoiceParams::default()
                },
            )
            .unwrap();
        reset_alloc_count();
        tracked(&mut mixer);
        // Released while their voices play: handed back when the voices end.
        control.release_clip(a).unwrap();
        control.release_clip(b).unwrap();
        for _ in 0..4 {
            tracked(&mut mixer);
        }
        control.stop(looped).unwrap();
        tracked(&mut mixer);
        assert_eq!(
            (get_alloc_count(), get_dealloc_count()),
            (0, 0),
            "round {round}: the mixer allocated or freed while rendering"
        );
        // The game thread receives the voices' ends and drops the clips' references.
        control.update();
        assert!(!control.is_playing(shot) && !control.is_playing(looped));
    }
    assert_eq!(Arc::strong_count(&short), 1);
    assert_eq!(Arc::strong_count(&long), 1);
}

/// The voice pool allocates nothing on either thread (rule 4): starting, replacing, refusing,
/// ramping, filtering, pausing and stopping voices on the game thread, and ranking,
/// virtualizing, filtering and fading out replaced voices on the audio thread.
#[test]
fn the_voice_pool_neither_allocates_nor_frees() {
    use korppi_audio::mixer::{
        AudioMixer, BusEffect, ClipData, MixerConfig, MixerControl, VoiceHandle, VoiceParams,
        VoiceSpatial, MASTER,
    };
    use korppi_audio::{
        AudioCallback, AudioCallbackInfo, AudioError, AudioStreamConfig, ChannelCount,
    };
    use std::sync::Arc;

    let config = AudioStreamConfig::default();
    let (mut control, mut mixer) = AudioMixer::new(MixerConfig {
        max_voices: 4,
        max_virtual_voices: 4,
        ..MixerConfig::for_stream(&config)
    });
    let info = AudioCallbackInfo {
        sample_rate: 48_000,
        channels: ChannelCount::Stereo,
        frame_count: 256,
        frames_rendered: 0,
    };
    let mut output = vec![0.0f32; 512];
    let tone = control
        .register_clip(Arc::new(ClipData::from(AudioClip::sine(
            330.0, 0.05, 48_000,
        ))))
        .unwrap();
    // A reverb bus every voice sends to: its effect runs while rendering.
    let reverb = control.add_bus("Reverb", MASTER).unwrap();
    control
        .set_bus_effect(reverb, Some(BusEffect::late([1.2, 1.0, 0.6], 48_000)))
        .unwrap();
    // Every voice started, to see replacements: reserved, so pushing does not allocate.
    let mut all = Vec::with_capacity(256);
    let (mut refused, mut replaced) = (0, 0);

    for round in 0..30u8 {
        let first = all.len();
        reset_alloc_count();
        set_tracking(true);
        for n in 0..6u8 {
            let params = VoiceParams {
                gain: 0.1 * f32::from(n + 1),
                priority: (n * 40) % 200 + round % 3,
                looping: n % 2 == 0,
                ..VoiceParams::default()
            };
            let playing = |control: &MixerControl, all: &[VoiceHandle]| {
                all.iter().filter(|&&v| control.is_playing(v)).count()
            };
            let before = playing(&control, &all);
            match control.start(tone, params) {
                Ok(voice) => {
                    if playing(&control, &all) < before {
                        replaced += 1;
                    }
                    control.set_gain(voice, 0.05, 480).unwrap();
                    let spatial = VoiceSpatial {
                        gain_left: 0.5,
                        gain_right: 1.0,
                        cutoff_hz: 800.0 + 400.0 * f32::from(n),
                        delay_left: 0.0002 * f32::from(n),
                        ..VoiceSpatial::default()
                    };
                    control.set_spatial(voice, spatial, 480).unwrap();
                    control.set_send(voice, reverb, 0.3, 480).unwrap();
                    all.push(voice);
                }
                Err(AudioError::VoicePoolFull) => refused += 1,
                Err(_) => unreachable!("the command queue has room"),
            }
        }
        mixer.render(&mut output, &info);
        if let Some(&voice) = all.get(first) {
            control.stop(voice).unwrap();
        }
        if let Some(&voice) = all.get(first + 1) {
            control.pause(voice).unwrap();
        }
        mixer.render(&mut output, &info);
        if let Some(&voice) = all.get(first + 1) {
            control.resume(voice).unwrap();
        }
        mixer.render(&mut output, &info);
        control.update();
        set_tracking(false);
        assert_eq!(
            (get_alloc_count(), get_dealloc_count()),
            (0, 0),
            "round {round}: the voice pool allocated or freed"
        );
    }
    assert!(
        refused > 0 && replaced > 0,
        "refused {refused}, replaced {replaced}"
    );
}

/// Audio players allocate nothing per tick once warmed up (rule 4): one-shots replaying over
/// their instance limit, a spatialized child following its moving parent behind a wall at times
/// (occlusion rays), a looping playlist scheduling its crossfades and transitioning, and stems,
/// with the mixer rendering between ticks.
#[test]
fn audio_players_tick_without_allocating() {
    players_tick_without_allocating(false);
}

/// The same in raytraced mode, inside a room the acoustics worker traces: handing it the
/// listener and emitters and taking its decay time allocates nothing on the game thread
/// (AUD-11; the worker's own allocations are on its thread).
#[test]
fn raytraced_players_tick_without_allocating() {
    players_tick_without_allocating(true);
}

fn players_tick_without_allocating(raytraced: bool) {
    use korppi_asset::{AssetServer, MemoryFileSystem, Vfs};
    use korppi_audio::mixer::{AudioMixer, AudioVoices, MixerConfig};
    use korppi_audio::player::{update_audio_players, AudioPlayer, AudioSource};
    use korppi_audio::{
        AudioCallback, AudioCallbackInfo, AudioSettings, AudioStreamConfig, ChannelCount,
        WavAudioLoader,
    };
    use korppi_ecs::{ChildOf, Schedule, Stage};
    use korppi_math::Vec3;
    use korppi_scene::{propagate_transforms, LocalTransform};

    let config = AudioStreamConfig::default();
    let (control, mut mixer) = AudioMixer::new(MixerConfig::for_stream(&config));
    let info = AudioCallbackInfo {
        sample_rate: 48_000,
        channels: ChannelCount::Stereo,
        frame_count: 256,
        frames_rendered: 0,
    };
    let mut output = vec![0.0f32; 512];

    let mut files = MemoryFileSystem::new();
    for (name, seconds) in [("shot", 0.05), ("hum", 1.0), ("a", 0.3), ("b", 0.4)] {
        files.insert(
            &format!("{name}.wav"),
            AudioClip::sine(440.0, seconds, 48_000).to_wav_bytes(),
        );
    }
    let mut vfs = Vfs::new();
    vfs.mount("", Box::new(files));
    let assets = AssetServer::new(vfs);
    assets.register_loader(WavAudioLoader);
    let load = |name: &str| {
        assets
            .load_sync::<WavAudioLoader>(&format!("{name}.wav"))
            .unwrap()
    };
    let (shot, hum, a, b) = (load("shot"), load("hum"), load("a"), load("b"));

    let mut world = World::new();
    world.insert_resource(AudioVoices::new(control));
    world.insert_resource(AudioSettings::default());
    // A wall the moving child's line of sight crosses at times.
    let mut walls = korppi_audio::raytrace::AcousticScene::new();
    walls.add_box(korppi_audio::raytrace::AcousticBox::new(
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(4.0, 4.0, 0.2),
        korppi_audio::raytrace::AcousticMaterial::concrete(),
    ));
    let walls = std::sync::Arc::new(walls);
    world.insert_resource(korppi_audio::AudioOcclusion {
        raycaster: Some(walls.clone()),
    });
    let mut room = korppi_audio::AudioRoom::default();
    if raytraced {
        world
            .resource_mut::<AudioSettings>()
            .unwrap()
            .spatializer_mode = korppi_audio::spatial::SpatializerMode::Raytraced;
        room.geometry = Some(shoebox_geometry());
    }
    world.insert_resource(room);
    world.spawn((AudioListener::default(), LocalTransform::default()));
    let one_shot = world.spawn((AudioPlayer::new(AudioSource::clip(shot)).max_instances(2),));
    let parent = world.spawn((LocalTransform::default(),));
    let child = world.spawn((
        AudioPlayer::new(AudioSource::clip(hum.clone()).looping()).autoplay(),
        LocalTransform::default(),
    ));
    world.add_pair::<ChildOf>(child, parent);
    let playlist = world.spawn((AudioPlayer::new(
        AudioSource::playlist(vec![a.clone(), b.clone()], 0.05).looping(),
    )
    .autoplay(),));
    world.spawn((AudioPlayer::new(AudioSource::stems(vec![hum, a]).looping()).autoplay(),));
    world.insert_resource(assets);
    let mut schedule = Schedule::new();
    schedule.add(Stage::PostUpdate, update_audio_players);

    let mut tick = |world: &mut World, n: u32| {
        if n.is_multiple_of(7) {
            world.get_mut::<AudioPlayer>(one_shot).unwrap().play();
        }
        if n.is_multiple_of(23) {
            world
                .get_mut::<AudioPlayer>(playlist)
                .unwrap()
                .transition_to((n as usize / 23) % 2, 0.02);
        }
        world.get_mut::<LocalTransform>(parent).unwrap().translation =
            Vec3::new((n as f32 * 0.1).sin() * 10.0, 0.0, -2.0);
        propagate_transforms(world);
        schedule.run(world);
        for _ in 0..3 {
            mixer.render(&mut output, &info);
        }
    };
    for n in 0..120 {
        tick(&mut world, n);
    }
    if raytraced {
        let mut warmup_tick = 120;
        while world
            .resource::<korppi_audio::AudioRoom>()
            .unwrap()
            .traces()
            < 2
            && warmup_tick < 600
        {
            std::thread::sleep(std::time::Duration::from_millis(2));
            tick(&mut world, warmup_tick);
            warmup_tick += 1;
        }
        assert!(
            world
                .resource::<korppi_audio::AudioRoom>()
                .unwrap()
                .traces()
                >= 2,
            "raytraced room worker must complete traces before zero-allocation verification"
        );
    }
    reset_alloc_count();
    set_tracking(true);
    for n in 120..360 {
        tick(&mut world, n);
    }
    set_tracking(false);
    assert_eq!(
        (get_alloc_count(), get_dealloc_count()),
        (0, 0),
        "audio players allocated or freed on a tick"
    );
}

/// A headless output advances the mixer by game time without allocating (rule 4): the engine
/// calls it every frame.
#[test]
fn the_headless_output_advances_without_allocating() {
    use korppi_audio::mixer::{ClipData, VoiceParams};
    use korppi_audio::output::{AudioBackend, AudioOutput};
    use korppi_audio::AudioSettings;
    use std::sync::Arc;
    use std::time::Duration;

    let (mut output, mut voices) =
        AudioOutput::open(AudioBackend::Headless, &AudioSettings::default()).unwrap();
    let clip = voices
        .control()
        .register_clip(Arc::new(ClipData::from(AudioClip::sine(
            220.0, 0.5, 48_000,
        ))))
        .unwrap();
    voices
        .start(
            clip,
            VoiceParams {
                looping: true,
                ..VoiceParams::default()
            },
        )
        .unwrap();
    output.advance(Duration::from_millis(100));
    reset_alloc_count();
    set_tracking(true);
    for _ in 0..120 {
        output.advance(Duration::from_secs_f64(1.0 / 60.0));
    }
    set_tracking(false);
    assert_eq!((get_alloc_count(), get_dealloc_count()), (0, 0));
}

/// A closed 8 x 3 x 5 m room of concrete walls with an interior partition, as the acoustics see it.
fn shoebox_geometry() -> std::sync::Arc<korppi_audio::raytrace::AcousticGeometry> {
    use korppi_audio::raytrace::{AcousticGeometry, AcousticMaterial};
    use korppi_math::{Quat, Vec3};
    use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};
    let mut world = PhysicsWorld::new();
    let (half, t) = (Vec3::new(4.0, 1.5, 2.5), 0.25);
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            let (mut center, mut extents) = (half, half + Vec3::splat(2.0 * t));
            center[axis] = half[axis] + side * (half[axis] + t);
            extents[axis] = t;
            let body = world.create_body(RigidBody {
                body_type: BodyType::Static,
                position: center - half,
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
    let pillar = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, 0.0, -1.0),
        rotation: Quat::IDENTITY,
        ..RigidBody::default()
    });
    world.create_collider(
        pillar,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.8, 1.2, 0.25),
        }),
    );
    std::sync::Arc::new(AcousticGeometry::from_physics(&world, |_| {
        Some(AcousticMaterial::concrete())
    }))
}
