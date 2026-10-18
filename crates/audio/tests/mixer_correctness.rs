//! What the mixer does to a clip on the way out (FIX-7.2, finding AUD-03, probes A1 to A3).
//!
//! Before FIX-7.2 the mixer ignored a clip's sample rate (a 44.1 kHz clip ended after 0.92 s at
//! 48 kHz, 8.8 % sharp), interpolated a clip's last frame with its first, dropped a loop's
//! fractional overshoot, and ignored master volume and buses (output at master volume 0 equalled
//! output at 1).

use korppi_audio::mixer::{AudioMixer, ClipData, MixerConfig, MixerControl, VoiceParams, MASTER};
use korppi_audio::{
    AudioClip, AudioDevice, AudioStream, AudioStreamConfig, BufferSize, ChannelCount,
    HeadlessDummyDevice,
};
use std::f32::consts::TAU;
use std::sync::Arc;

const RATE: u32 = 48_000;
/// Gain checks skip the first 10 ms, so that ramps towards a newly set gain have settled.
const SETTLED: usize = 480;

fn open() -> (MixerControl, Box<dyn AudioStream>) {
    let config = AudioStreamConfig {
        sample_rate: RATE,
        channels: ChannelCount::Stereo,
        buffer_size: BufferSize::Fixed(256),
    };
    let (control, mixer) = AudioMixer::new(MixerConfig::for_stream(&config));
    let mut stream = HeadlessDummyDevice::new()
        .create_output_stream(config, Box::new(mixer))
        .expect("stream");
    stream.play().expect("play");
    (control, stream)
}

fn play(control: &mut MixerControl, clip: AudioClip, params: VoiceParams) {
    let clip = control
        .register_clip(Arc::new(ClipData::from(clip)))
        .expect("register_clip");
    control.start(clip, params).expect("start");
}

fn left(out: &[f32]) -> Vec<f32> {
    out.iter().step_by(2).copied().collect()
}

/// A 1-second 44.1 kHz clip lasts one second at 48 kHz, and its 440 Hz tone stays 440 Hz.
#[test]
fn clips_resample_to_the_device_rate() {
    let (mut control, mut stream) = open();
    play(
        &mut control,
        AudioClip::sine(440.0, 1.0, 44_100),
        VoiceParams::default(),
    );
    let out = left(&stream.render_frames(RATE as usize + 1000));
    let last = out.iter().rposition(|&s| s != 0.0).expect("sound");
    assert!(
        (RATE as usize - 2..=RATE as usize).contains(&last),
        "the clip ended at output frame {last}, not after one second"
    );
    // Against the ideal 440 Hz sine at the device rate (the clip starts at phase 0).
    let worst = out[..RATE as usize - 4]
        .iter()
        .enumerate()
        .map(|(n, &s)| (s - (TAU * 440.0 * n as f32 / RATE as f32).sin()).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 2e-3, "resampled tone deviates by {worst}");
}

/// A non-looping clip does not interpolate its last frames with its first.
#[test]
fn a_clip_ends_without_wrapping_into_its_start() {
    let (mut control, mut stream) = open();
    // An impulse at the first frame, silence after: at half speed, the end must stay silent.
    let mut samples = vec![0.0; 200];
    samples[0] = 1.0;
    play(
        &mut control,
        AudioClip::new(1, RATE, samples),
        VoiceParams {
            pitch: 0.5,
            ..VoiceParams::default()
        },
    );
    let out = left(&stream.render_frames(500));
    assert!(
        out[10..].iter().all(|&s| s == 0.0),
        "the end of the clip picked up its first frame: {:?}",
        &out[380..400]
    );
}

/// A loop carries its fractional overshoot into the next pass: a looped one-cycle sine at a
/// pitch that does not divide its length stays one continuous sine.
#[test]
fn loops_keep_their_phase() {
    let (mut control, mut stream) = open();
    let period = 48;
    let cycle: Vec<f32> = (0..period)
        .map(|n| (TAU * n as f32 / period as f32).sin())
        .collect();
    let pitch = 1.1;
    play(
        &mut control,
        AudioClip::new(1, RATE, cycle),
        VoiceParams {
            pitch,
            looping: true,
            ..VoiceParams::default()
        },
    );
    let out = left(&stream.render_frames(4800));
    let worst = out
        .iter()
        .enumerate()
        .map(|(n, &s)| (s - (TAU * pitch * n as f32 / period as f32).sin()).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 2e-3, "the looped sine drifts or jumps by {worst}");
}

/// Master volume and buses are in the signal path: a voice routed to a child bus is scaled by
/// that bus and by the master, and muting either silences it.
#[test]
fn buses_and_master_volume_scale_the_output() {
    let constant = || AudioClip::new(1, RATE, vec![0.8; RATE as usize]);
    let rendered = |setup: &dyn Fn(&mut MixerControl) -> korppi_audio::mixer::BusId| {
        let (mut control, mut stream) = open();
        let bus = setup(&mut control);
        play(
            &mut control,
            constant(),
            VoiceParams {
                bus,
                ..VoiceParams::default()
            },
        );
        left(&stream.render_frames(2000))[SETTLED..].to_vec()
    };
    let all = |samples: &[f32], value: f32| samples.iter().all(|&s| (s - value).abs() < 1e-6);

    assert!(all(&rendered(&|_| MASTER), 0.8));
    let half_master = rendered(&|c| {
        c.set_bus_gain(MASTER, 0.5).unwrap();
        MASTER
    });
    assert!(
        all(&half_master, 0.4),
        "master volume 0.5: {}",
        half_master[0]
    );
    let silent_master = rendered(&|c| {
        c.set_bus_gain(MASTER, 0.0).unwrap();
        MASTER
    });
    assert!(
        all(&silent_master, 0.0),
        "master volume 0: {}",
        silent_master[0]
    );
    let music = rendered(&|c| {
        let music = c.add_bus("Music", MASTER).unwrap();
        c.set_bus_gain(music, 0.25).unwrap();
        c.set_bus_gain(MASTER, 0.5).unwrap();
        music
    });
    assert!(
        all(&music, 0.1),
        "Music 0.25 under master 0.5: {}",
        music[0]
    );
    let muted = rendered(&|c| {
        let music = c.add_bus("Music", MASTER).unwrap();
        c.set_bus_muted(music, true).unwrap();
        music
    });
    assert!(all(&muted, 0.0), "muted bus: {}", muted[0]);
}

/// Buses are found by name on the game thread.
#[test]
fn buses_are_named() {
    let (mut control, _stream) = open();
    let sfx = control.add_bus("SFX", MASTER).unwrap();
    let footsteps = control.add_bus("Footsteps", sfx).unwrap();
    assert_eq!(control.bus("SFX"), Some(sfx));
    assert_eq!(control.bus("Footsteps"), Some(footsteps));
    assert_eq!(control.bus("Master"), Some(MASTER));
    assert_eq!(control.bus("Music"), None);
    assert!(control.add_bus("SFX", MASTER).is_err(), "names are unique");
}
