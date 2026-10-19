//! Primary regression test suite for FIX-14.7 (`AUD-14` through `AUD-20`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use korppi_asset::{AssetHandle, AssetServer, MemoryFileSystem, Vfs};
use korppi_audio::mixer::{
    AudioMixer, AudioVoices, BusEffect, ClipData, MixerConfig, RoomReverb, VoiceParams,
    VoiceSpatial, MASTER,
};
use korppi_audio::player::{
    update_audio_players, AudioOcclusion, AudioPlayer, AudioRoom, AudioSource, PlayerState,
};
use korppi_audio::raytrace::{
    AcousticGeometry, AcousticMaterial, PathTracer, TraceSettings, TraceStatistics,
};
use korppi_audio::synthesis::{FdnConfig, FeedbackDelayNetwork};
use korppi_audio::{
    AudioCallback, AudioCallbackInfo, AudioClip, AudioClipLoader, AudioDevice, AudioFrame,
    AudioListener, AudioSettings, AudioStream, AudioStreamConfig, BiquadFilter, BiquadType,
    BufferSize, ChannelCount, DistanceAttenuation, EmitterCone, EmitterState, Freeverb,
    HeadlessDummyDevice, ListenerState, PeakLimiter, SpatialAudioModel, StandardSpatializer,
    VoiceSpatialParams,
};
use korppi_ecs::{Entity, Schedule, Stage, World};
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};
use korppi_scene::{propagate_transforms, GlobalTransform, LocalTransform};

struct CountingAlloc;

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to the system allocator unchanged; only increments atomic counters.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACKING.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn count_allocs_deallocs(f: impl FnOnce()) -> (usize, usize) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::Relaxed);
    f();
    TRACKING.store(false, Ordering::Relaxed);
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        DEALLOCATIONS.load(Ordering::Relaxed),
    )
}

const SR: u32 = 48_000;
const TICK_FRAMES: usize = 800; // 60 Hz at 48 kHz

fn rms_interleaved(samples: &[f32]) -> (f32, f32) {
    let frames = (samples.len() / 2).max(1) as f32;
    let mut sum_l = 0.0f32;
    let mut sum_r = 0.0f32;
    for &[l, r] in samples.as_chunks::<2>().0 {
        sum_l += l * l;
        sum_r += r * r;
    }
    ((sum_l / frames).sqrt(), (sum_r / frames).sqrt())
}

struct TestEnv {
    world: World,
    schedule: Schedule,
    stream: Box<dyn AudioStream>,
    files: Vec<(String, Vec<u8>)>,
}

impl TestEnv {
    fn new(settings: AudioSettings) -> Self {
        Self::with_max_clips(settings, 256)
    }

    fn with_max_clips(settings: AudioSettings, max_clips: usize) -> Self {
        let config = AudioStreamConfig {
            sample_rate: SR,
            channels: ChannelCount::Stereo,
            buffer_size: BufferSize::Fixed(256),
        };
        let max_voices = settings.max_voices.max(1);
        let mixer_cfg = MixerConfig {
            max_voices,
            max_virtual_voices: max_voices * 3,
            max_clips,
            ..MixerConfig::for_stream(&config)
        };
        let (control, mixer) = AudioMixer::new(mixer_cfg);
        let mut stream = HeadlessDummyDevice::new()
            .create_output_stream(config, Box::new(mixer))
            .expect("create headless stream");
        stream.play().expect("play stream");

        let mut world = World::new();
        world.insert_resource(AudioVoices::new(control));
        world.insert_resource(settings);
        world.insert_resource(AudioOcclusion::default());
        world.insert_resource(AudioRoom::default());

        let mut schedule = Schedule::new();
        schedule.add(Stage::PostUpdate, update_audio_players);

        Self {
            world,
            schedule,
            stream,
            files: Vec::new(),
        }
    }

    fn add_clip(&mut self, name: &str, clip: AudioClip) {
        self.files
            .push((format!("{name}.wav"), clip.to_wav_bytes()));
    }

    fn build_assets(&mut self) -> Vec<AssetHandle<AudioClip>> {
        let mut mem = MemoryFileSystem::new();
        for (path, bytes) in &self.files {
            mem.insert(path, bytes.clone());
        }
        let mut vfs = Vfs::new();
        vfs.mount("", Box::new(mem));
        let assets = AssetServer::new(vfs);
        assets.register_loader(AudioClipLoader);
        let handles = self
            .files
            .iter()
            .map(|(path, _)| {
                assets
                    .load_sync::<AudioClipLoader>(path)
                    .expect("load clip")
            })
            .collect();
        self.world.insert_resource(assets);
        handles
    }

    fn spawn_listener(&mut self, translation: Vec3, rotation: Quat) -> Entity {
        self.world.spawn((
            AudioListener::default(),
            LocalTransform {
                translation,
                rotation,
                ..LocalTransform::default()
            },
            GlobalTransform::default(),
        ))
    }

    fn tick(&mut self, frames: usize) -> Vec<f32> {
        propagate_transforms(&mut self.world);
        self.schedule.run(&mut self.world);
        self.stream.render_frames(frames)
    }

    fn ticks(&mut self, count: usize, frames_per_tick: usize) -> Vec<f32> {
        (0..count)
            .flat_map(|_| self.tick(frames_per_tick))
            .collect()
    }

    fn player(&self, entity: Entity) -> &AudioPlayer {
        self.world.get::<AudioPlayer>(entity).expect("player")
    }

    fn player_mut(&mut self, entity: Entity) -> &mut AudioPlayer {
        self.world.get_mut::<AudioPlayer>(entity).expect("player")
    }
}

#[test]
fn aud_14_paused_voice_can_be_stopped_and_releases_slot_and_clip() {
    let mut env = TestEnv::new(AudioSettings::default());
    env.add_clip("loop", AudioClip::new(1, SR, vec![0.5; SR as usize]));
    let handle = env.build_assets().remove(0);
    let entity = env
        .world
        .spawn((AudioPlayer::new(AudioSource::clip(handle).looping())
            .autoplay()
            .despawn_on_finish(),));

    let _ = env.tick(256);
    assert_eq!(env.player(entity).state(), PlayerState::Playing);

    // Pause and let the 5 ms pause ramp settle (> 240 frames at 48 kHz) so `Voice::is_held()` is true.
    env.player_mut(entity).pause();
    let _ = env.tick(512);
    assert_eq!(env.player(entity).state(), PlayerState::Paused);

    // Stop the paused voice and advance 10 blocks.
    env.player_mut(entity).stop(0.0);
    for _ in 0..10 {
        let _ = env.tick(256);
    }
    let entity_alive = env.world.get::<AudioPlayer>(entity).is_some();
    assert!(
        !entity_alive,
        "AUD-14: stopping a settled paused voice with despawn_on_finish must terminate the voice and despawn the entity"
    );

    // Also verify directly on MixerControl + AudioMixer that the voice finishes and releases its clip.
    let config = AudioStreamConfig::default();
    let (mut control, mut mixer) = AudioMixer::new(MixerConfig::for_stream(&config));
    let clip_id = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            SR,
            vec![0.5; 4800],
        ))))
        .unwrap();
    let voice = control.start(clip_id, VoiceParams::default()).unwrap();
    let mut buf = vec![0.0f32; 512];
    let mut rendered = 0u64;
    let mut render_block = |mixer: &mut AudioMixer| {
        let info = AudioCallbackInfo {
            sample_rate: SR,
            channels: ChannelCount::Stereo,
            frame_count: 256,
            frames_rendered: rendered,
        };
        mixer.render(&mut buf, &info);
        rendered += 256;
    };
    render_block(&mut mixer);
    control.pause(voice).unwrap();
    render_block(&mut mixer);
    render_block(&mut mixer); // pause ramp settled -> is_held() == true
    control.stop(voice).unwrap();
    control.release_clip(clip_id).unwrap();
    for _ in 0..10 {
        render_block(&mut mixer);
    }
    control.update();
    assert!(
        !control.is_playing(voice),
        "AUD-14: MixerControl::is_playing must be false after stopping a paused voice"
    );
}

#[test]
fn aud_15_voice_ranking_and_stealing_account_for_spatial_gain() {
    // Part 1: 64 silent distant players (500 m, max_distance 50 m -> spatial gain 0.0)
    // vs 10 loud near players (0.5 m -> spatial gain ~0.7071), with max_voices = 64.
    let mut env = TestEnv::new(AudioSettings {
        max_voices: 64,
        ..AudioSettings::default()
    });
    env.add_clip("tone", AudioClip::new(1, SR, vec![0.5; SR as usize]));
    let handle = env.build_assets().remove(0);
    env.spawn_listener(Vec3::ZERO, Quat::IDENTITY);

    let atten = DistanceAttenuation::Linear {
        min_distance: 1.0,
        max_distance: 50.0,
    };
    for _ in 0..64 {
        let mut p = AudioPlayer::new(AudioSource::clip(handle.clone()).looping()).autoplay();
        p.spatial.attenuation = atten;
        p.spatial.doppler_factor = 0.0;
        env.world.spawn((
            p,
            LocalTransform::from_translation(Vec3::new(0.0, 0.0, -500.0)),
            GlobalTransform::default(),
        ));
    }
    for _ in 0..10 {
        let mut p = AudioPlayer::new(AudioSource::clip(handle.clone()).looping()).autoplay();
        p.spatial.attenuation = atten;
        p.spatial.doppler_factor = 0.0;
        env.world.spawn((
            p,
            LocalTransform::from_translation(Vec3::new(0.0, 0.0, -0.5)),
            GlobalTransform::default(),
        ));
    }
    for _ in 0..3 {
        let _ = env.tick(256);
    }
    let out = env.tick(256);
    let (rms_l, rms_r) = rms_interleaved(&out);
    assert!(
        rms_l > 0.1 && rms_r > 0.1,
        "AUD-15: 10 near voices must win audible slots over 64 silent distant voices, got RMS=({rms_l}, {rms_r})"
    );

    // Part 2: Voice stealing in MixerControl when pool is full (4 slots: 1 audible + 3 virtual).
    let config = AudioStreamConfig::default();
    let (mut control, _mixer) = AudioMixer::new(MixerConfig {
        max_voices: 1,
        max_virtual_voices: 3,
        ..MixerConfig::for_stream(&config)
    });
    let clip_id = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            SR,
            vec![0.5; 4800],
        ))))
        .unwrap();
    let near_voice = control.start(clip_id, VoiceParams::default()).unwrap();
    control
        .set_spatial(
            near_voice,
            VoiceSpatial {
                gain_left: 1.0,
                gain_right: 1.0,
                ..VoiceSpatial::default()
            },
            0,
        )
        .unwrap();
    let mut far_voices = Vec::new();
    for _ in 0..3 {
        let v = control.start(clip_id, VoiceParams::default()).unwrap();
        control
            .set_spatial(
                v,
                VoiceSpatial {
                    gain_left: 0.0,
                    gain_right: 0.0,
                    ..VoiceSpatial::default()
                },
                0,
            )
            .unwrap();
        far_voices.push(v);
    }
    let _fifth = control.start(clip_id, VoiceParams::default()).unwrap();
    assert!(
        control.is_playing(near_voice),
        "AUD-15: voice stealing must steal a silent far voice (gain 0.0), not the loud near voice (gain 1.0)"
    );
    let far_survived = far_voices
        .iter()
        .filter(|&&v| control.is_playing(v))
        .count();
    assert_eq!(far_survived, 2);
}

#[test]
fn aud_16_fdn_band_calibration_stability_and_kuttruff_rt60_high_absorption() {
    // 1a. Low-heavy RT60 ([4.0, 0.15, 0.15] s at 48 kHz) must not blow up DC loop gain > 1.0.
    let mut fdn = FeedbackDelayNetwork::new(FdnConfig::default(), SR as f32);
    fdn.set_rt60([4.0, 0.15, 0.15], SR as f32);
    let (mut peak_early, mut peak_late) = (0.0f32, 0.0f32);
    for i in 0..(SR as usize * 2) {
        let (l, r) = fdn.process_sample(if i == 0 { 1.0 } else { 0.0 });
        assert!(
            l.is_finite() && r.is_finite(),
            "AUD-16: FDN sample {i} non-finite ({l}, {r})"
        );
        let m = l.abs().max(r.abs());
        if i < 4_800 {
            peak_early = peak_early.max(m);
        }
        if i >= SR as usize * 2 - 4_800 {
            peak_late = peak_late.max(m);
        }
    }
    assert!(
        peak_early < 2.0,
        "AUD-16: FDN early peak must not blow up, got {peak_early}"
    );
    assert!(
        peak_late < peak_early,
        "AUD-16: FDN late peak ({peak_late}) must decay below early peak ({peak_early})"
    );

    // 1b. Extreme band ratio [5.0, 0.05, 0.05] at 48 kHz and low sample rate (2000 Hz) must stay finite.
    for &sr in &[48_000.0f32, 2_000.0] {
        let mut fdn_ext = FeedbackDelayNetwork::new(FdnConfig::default(), sr);
        fdn_ext.set_rt60([5.0, 0.05, 0.05], sr);
        for i in 0..4_800 {
            let (l, r) = fdn_ext.process_sample(if i == 0 { 1.0 } else { 0.0 });
            assert!(
                l.is_finite() && r.is_finite(),
                "AUD-16: FDN produced non-finite output ({l}, {r}) at sr={sr}, sample={i}"
            );
        }
    }

    // 2. Kuttruff correction in TraceStatistics::rt60() must not collapse at high absorption (alpha = 0.95).
    let foam = AcousticMaterial::acoustic_foam(); // absorption = [0.20, 0.75, 0.95]
    let segments = 100usize;
    let mean = 4.0f64;
    let spread = 0.8f64;
    let stats = TraceStatistics {
        segments,
        length: segments as f64 * mean,
        length_squared: segments as f64 * mean * mean * (1.0 + spread),
        reflections: segments,
        absorption: foam.absorption.map(|a| f64::from(a) * segments as f64),
    };
    let rt60 = stats.rt60().expect("rt60");
    assert!(
        rt60[2] < rt60[1] && rt60[2] < 0.15,
        "AUD-16: 95%-absorbing foam high band must have shorter RT60 than 75% mid band, got {rt60:?}"
    );
}

#[test]
fn aud_17_overflow_release_events_never_deallocate_on_audio_thread_or_leak_clips_and_voices_evict()
{
    let (mut control, mut mixer) = AudioMixer::new(MixerConfig {
        max_voices: 4,
        max_virtual_voices: 4,
        max_clips: 2,
        sample_rate: SR,
        max_buses: 1,
        queue_capacity: 4,
        max_reflecting_voices: 0,
    });
    let mut buf = vec![0.0f32; 512];
    let mut rendered = 0u64;
    let mut render_block = |mixer: &mut AudioMixer| {
        let info = AudioCallbackInfo {
            sample_rate: SR,
            channels: ChannelCount::Stereo,
            frame_count: 256,
            frames_rendered: rendered,
        };
        mixer.render(&mut buf, &info);
        rendered += 256;
    };

    let c0 = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            SR,
            vec![0.1; 64],
        ))))
        .unwrap();
    let c1 = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            SR,
            vec![0.2; 64],
        ))))
        .unwrap();
    control
        .set_bus_effect(
            MASTER,
            Some(BusEffect::room(
                RoomReverb {
                    size: 0.5,
                    damping: 0.5,
                },
                SR,
            )),
        )
        .unwrap();
    render_block(&mut mixer);

    for _ in 0..4 {
        control
            .set_bus_effect(
                MASTER,
                Some(BusEffect::room(
                    RoomReverb {
                        size: 0.6,
                        damping: 0.4,
                    },
                    SR,
                )),
            )
            .unwrap();
    }
    render_block(&mut mixer);
    for _ in 0..3 {
        control
            .set_bus_effect(
                MASTER,
                Some(BusEffect::room(
                    RoomReverb {
                        size: 0.7,
                        damping: 0.3,
                    },
                    SR,
                )),
            )
            .unwrap();
    }
    render_block(&mut mixer);

    control.release_clip(c0).unwrap();
    control.release_clip(c1).unwrap();
    let (_, audio_thread_deallocs) = count_allocs_deallocs(|| {
        render_block(&mut mixer);
    });
    assert_eq!(
        audio_thread_deallocs, 0,
        "AUD-17: audio thread must never deallocate ClipData or BusEffect on event overflow"
    );

    for _ in 0..6 {
        control.update();
        render_block(&mut mixer);
    }
    control.update();

    let c2 = control.register_clip(Arc::new(ClipData::from(AudioClip::new(
        1,
        SR,
        vec![0.3; 64],
    ))));
    let c3 = control.register_clip(Arc::new(ClipData::from(AudioClip::new(
        1,
        SR,
        vec![0.4; 64],
    ))));
    assert!(
        c2.is_ok() && c3.is_ok(),
        "AUD-17: both ClipId slots must be reclaimed after draining overflowed releases, got c2={c2:?}, c3={c3:?}"
    );

    // Part 2: AudioVoices evicts unused cached clips when max_clips is reached.
    let mut env = TestEnv::with_max_clips(AudioSettings::default(), 4);
    for i in 0..10 {
        env.add_clip(&format!("clip_{i}"), AudioClip::new(1, SR, vec![0.2; 128]));
    }
    let handles = env.build_assets();
    for handle in handles {
        let entity = env.world.spawn((AudioPlayer::new(AudioSource::clip(handle))
            .autoplay()
            .despawn_on_finish(),));
        for _ in 0..4 {
            let _ = env.tick(256);
        }
        assert!(
            env.world.get::<AudioPlayer>(entity).is_none(),
            "AUD-17: AudioVoices with max_clips=4 must evict finished clips and play all 10 sequential clips"
        );
    }
}

#[test]
fn aud_18_playlist_and_scheduled_stop_preserve_remaining_duration_across_pause_and_resume() {
    let mut env = TestEnv::new(AudioSettings::default());
    env.add_clip("track0", AudioClip::new(1, SR, vec![0.25; SR as usize]));
    env.add_clip("track1", AudioClip::new(1, SR, vec![0.75; SR as usize]));
    let items = env.build_assets();
    let entity = env
        .world
        .spawn((AudioPlayer::new(AudioSource::playlist(items, 0.1)).autoplay(),));

    // Play for 0.20 s (12 ticks of 800 frames = 9,600 frames).
    let _ = env.ticks(12, TICK_FRAMES);
    assert_eq!(env.player(entity).current_item(), 0);
    let pos_before_pause = env.player(entity).position();
    assert!(pos_before_pause > 0.15 && pos_before_pause < 0.25);

    // Pause for 1.00 s (60 ticks = 48,000 frames) while the mixer clock advances.
    env.player_mut(entity).pause();
    let _ = env.ticks(60, TICK_FRAMES);
    assert_eq!(env.player(entity).current_item(), 0);

    // Resume and tick twice (0.033 s): must still be on item 0 around ~0.23 s!
    env.player_mut(entity).resume();
    let _ = env.ticks(2, TICK_FRAMES);
    let item_after_resume = env.player(entity).current_item();
    let pos_after_resume = env.player(entity).position();
    assert_eq!(
        item_after_resume, 0,
        "AUD-18: resuming a paused playlist must not cut early to track 1 (pos={pos_after_resume})"
    );
    assert!(
        pos_after_resume > 0.18 && pos_after_resume < 0.35,
        "AUD-18: expected position ~0.23 s after resume, got {pos_after_resume}"
    );
}

fn wav_bytes(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let body_len: usize = chunks
        .iter()
        .map(|(_, data)| 8 + data.len() + (data.len() & 1))
        .sum();
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((4 + body_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    for (id, data) in chunks {
        out.extend_from_slice(*id);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if data.len() & 1 == 1 {
            out.push(0);
        }
    }
    out
}

fn fmt_chunk(format: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    let block = channels * bits / 8;
    let mut v = Vec::new();
    v.extend_from_slice(&format.to_le_bytes());
    v.extend_from_slice(&channels.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    v.extend_from_slice(&block.to_le_bytes());
    v.extend_from_slice(&bits.to_le_bytes());
    v
}

#[test]
fn aud_19_nan_and_panic_guards_in_spatializer_wav_dsp_and_peak_limiter_attack_and_threshold() {
    // 1. StandardSpatializer with parallel forward/up and zero emitter forward with cone.
    let mut sp = StandardSpatializer::new(SR as f32);
    let degenerate_listener = ListenerState {
        position: Vec3::ZERO,
        forward: Vec3::Y,
        up: Vec3::Y,
        velocity: Vec3::ZERO,
    };
    let emitter = EmitterState {
        position: Vec3::new(0.0, 0.0, -5.0),
        ..EmitterState::default()
    };
    let mut out_parallel = VoiceSpatialParams::default();
    sp.spatialize_emitter(&degenerate_listener, &emitter, None, &mut out_parallel);
    assert!(
        out_parallel.gain_left.is_finite()
            && out_parallel.gain_right.is_finite()
            && out_parallel.delay_left_sec.is_finite()
            && out_parallel.delay_right_sec.is_finite()
            && out_parallel.head_shadow_left.is_finite()
            && out_parallel.head_shadow_right.is_finite()
            && out_parallel.low_pass_cutoff.is_finite(),
        "AUD-19: StandardSpatializer produced non-finite params for parallel forward/up: {out_parallel:?}"
    );

    let zero_fwd_emitter = EmitterState {
        position: Vec3::new(0.0, 0.0, -5.0),
        forward: Vec3::ZERO,
        cone: Some(EmitterCone::default()),
        ..EmitterState::default()
    };
    let mut out_zero_cone = VoiceSpatialParams::default();
    sp.spatialize_emitter(
        &ListenerState::default(),
        &zero_fwd_emitter,
        None,
        &mut out_zero_cone,
    );
    assert!(
        out_zero_cone.gain_left.is_finite()
            && out_zero_cone.gain_right.is_finite()
            && out_zero_cone.low_pass_cutoff.is_finite(),
        "AUD-19: StandardSpatializer produced non-finite params for zero cone forward: {out_zero_cone:?}"
    );

    // 2. Float32 WAV decoder sanitizes NaN / +-Inf to 0.0.
    let float_samples: [f32; 4] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];
    let float_bytes: Vec<u8> = float_samples.iter().flat_map(|f| f.to_le_bytes()).collect();
    let wav_nan = wav_bytes(&[(b"fmt ", fmt_chunk(3, 1, SR, 32)), (b"data", float_bytes)]);
    let decoded = AudioClip::from_wav_bytes(&wav_nan).unwrap();
    assert_eq!(
        decoded.samples,
        vec![0.0, 0.0, 0.0, 0.5],
        "AUD-19: non-finite float WAV samples must be sanitized to 0.0"
    );

    // 3. BiquadFilter at sr=16.0 and Freeverb at sr=0.0 must not panic.
    let biquad_ok = catch_unwind(AssertUnwindSafe(|| {
        let mut f = BiquadFilter::new(BiquadType::LowPass, 100.0, 0.707, 16.0);
        f.process_sample(0.5).is_finite()
    }));
    assert_eq!(biquad_ok.ok(), Some(true), "AUD-19: BiquadFilter panicked");

    let freeverb_ok = catch_unwind(AssertUnwindSafe(|| {
        let mut fv = Freeverb::new(0.0);
        let (l, r) = fv.process_mono(1.0);
        l.is_finite() && r.is_finite()
    }));
    assert_eq!(freeverb_ok.ok(), Some(true), "AUD-19: Freeverb panicked");

    // 4. PeakLimiter honours attack_ms and positive threshold_db.
    let mut lim_fast = PeakLimiter::new(SR as f32);
    lim_fast.set_lookahead_ms(0.0);
    lim_fast.set_attack_ms(0.1);
    let mut lim_slow = PeakLimiter::new(SR as f32);
    lim_slow.set_lookahead_ms(0.0);
    lim_slow.set_attack_ms(100.0);
    let mut frames_fast = [AudioFrame::new(0.5, 0.5), AudioFrame::new(2.0, 2.0)];
    let mut frames_slow = frames_fast;
    lim_fast.process(&mut frames_fast);
    lim_slow.process(&mut frames_slow);
    assert_ne!(
        frames_fast, frames_slow,
        "AUD-19: PeakLimiter must honour attack_ms (0.1 ms vs 100.0 ms)"
    );

    let mut lim_high_thresh = PeakLimiter::new(SR as f32);
    lim_high_thresh.set_lookahead_ms(0.0);
    lim_high_thresh.set_threshold_db(6.0); // ~1.995 linear
    let mut high_frame = [AudioFrame::new(1.5, -1.5)];
    lim_high_thresh.process(&mut high_frame);
    assert!(
        (high_frame[0].left - 1.5).abs() < 1e-3 && (high_frame[0].right + 1.5).abs() < 1e-3,
        "AUD-19: PeakLimiter with +6 dB threshold must not hard-clip 1.5 to 1.0, got {:?}",
        high_frame[0]
    );
}

#[test]
fn aud_20_path_tracer_nee_scales_by_scattering() {
    let mut phys = PhysicsWorld::new();
    let mirror = AcousticMaterial::new([0.05, 0.05, 0.05], 0.0, [0.0; 3]);
    for (pos, half) in [
        (Vec3::new(0.0, -2.0, 0.0), Vec3::new(6.0, 0.1, 6.0)),
        (Vec3::new(0.0, 2.0, 0.0), Vec3::new(6.0, 0.1, 6.0)),
        (Vec3::new(-6.0, 0.0, 0.0), Vec3::new(0.1, 2.0, 6.0)),
        (Vec3::new(6.0, 0.0, 0.0), Vec3::new(0.1, 2.0, 6.0)),
        (Vec3::new(0.0, 0.0, -6.0), Vec3::new(6.0, 2.0, 0.1)),
        (Vec3::new(0.0, 0.0, 6.0), Vec3::new(6.0, 2.0, 0.1)),
    ] {
        let body = phys.create_body(RigidBody {
            body_type: BodyType::Static,
            position: pos,
            ..RigidBody::default()
        });
        phys.create_collider(body, Collider::new(Shape::Box { half_extents: half }));
    }
    let geometry_mirror = AcousticGeometry::from_physics(&phys, |_| Some(mirror));
    let mut tracer = PathTracer::new(42);
    let responses_mirror = tracer.trace(
        &geometry_mirror,
        Vec3::new(-2.0, 0.0, 0.0),
        &[Vec3::new(2.0, 0.0, 0.0)],
        &TraceSettings::default(),
    );
    assert_eq!(
        responses_mirror[0].bounce_energy(),
        [0.0, 0.0, 0.0],
        "AUD-20: 100% specular surfaces (scattering = 0.0) must contribute 0 diffuse NEE bounce energy"
    );

    let diffuse = AcousticMaterial::new([0.05, 0.05, 0.05], 0.5, [0.0; 3]);
    let geometry_diffuse = AcousticGeometry::from_physics(&phys, |_| Some(diffuse));
    let mut tracer2 = PathTracer::new(42);
    let responses_diffuse = tracer2.trace(
        &geometry_diffuse,
        Vec3::new(-2.0, 0.0, 0.0),
        &[Vec3::new(2.0, 0.0, 0.0)],
        &TraceSettings::default(),
    );
    assert!(
        responses_diffuse[0].bounce_energy()[0] > 0.0,
        "AUD-20: scattering = 0.5 must contribute positive diffuse NEE bounce energy"
    );
}
