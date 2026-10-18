//! Sound reaches the device through the mixer (FIX-7.2, finding AUD-01, ADR-050).
//!
//! Before FIX-7.2 nothing rendered the mix into a stream: the coordinator's mixer was called only
//! by tests, and it shared all voice state with the game thread. Now the game thread drives an
//! `AudioMixer` through a `MixerControl` (commands in, events out), and the mixer runs as the
//! stream's `AudioCallback`. These tests play clips through `HeadlessDummyDevice` and check the
//! samples that come out.

use korppi_audio::mixer::{AudioMixer, ClipData, MixerConfig, VoiceParams};
use korppi_audio::{
    AudioClip, AudioDevice, AudioStream, AudioStreamConfig, BufferSize, ChannelCount,
    HeadlessDummyDevice,
};
use std::sync::Arc;

const RATE: u32 = 48_000;

fn open(channels: ChannelCount) -> (korppi_audio::mixer::MixerControl, Box<dyn AudioStream>) {
    let config = AudioStreamConfig {
        sample_rate: RATE,
        channels,
        buffer_size: BufferSize::Fixed(256),
    };
    let (control, mixer) = AudioMixer::new(MixerConfig::for_stream(&config));
    let mut stream = HeadlessDummyDevice::new()
        .create_output_stream(config, Box::new(mixer))
        .expect("stream");
    stream.play().expect("play");
    (control, stream)
}

/// A mono clip of `frames` distinct values at the device rate.
fn ramp(frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|i| (i + 1) as f32 / frames as f32)
        .collect()
}

#[test]
fn a_clip_reaches_the_output_sample_for_sample_and_then_silence() {
    let (mut control, mut stream) = open(ChannelCount::Stereo);
    let samples = ramp(1000);
    let clip = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            RATE,
            samples.clone(),
        ))))
        .expect("register_clip");
    let voice = control.start(clip, VoiceParams::default()).expect("start");
    assert!(control.is_playing(voice));

    let out = stream.render_frames(2000);
    for (frame, &expected) in samples.iter().enumerate() {
        assert_eq!(
            (out[2 * frame], out[2 * frame + 1]),
            (expected, expected),
            "frame {frame}: a mono clip plays in both channels"
        );
    }
    assert!(
        out[2 * samples.len()..].iter().all(|&s| s == 0.0),
        "the output is silent after the clip"
    );
    control.update();
    assert!(
        !control.is_playing(voice),
        "the mixer reports the voice finished"
    );
}

#[test]
fn a_stereo_clip_keeps_its_channels() {
    let (mut control, mut stream) = open(ChannelCount::Stereo);
    // Left carries a ramp, right is silent: the sound must stay on the left.
    let frames = 500;
    let interleaved: Vec<f32> = ramp(frames).iter().flat_map(|&l| [l, 0.0]).collect();
    let clip = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            2,
            RATE,
            interleaved,
        ))))
        .expect("register_clip");
    control.start(clip, VoiceParams::default()).expect("start");
    let out = stream.render_frames(frames);
    let left: Vec<f32> = out.iter().step_by(2).copied().collect();
    let right: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
    assert_eq!(left, ramp(frames));
    assert!(
        right.iter().all(|&s| s == 0.0),
        "the right channel stays silent"
    );
}

#[test]
fn surround_devices_get_the_stereo_mix_in_front_and_mono_devices_its_average() {
    let interleaved: Vec<f32> = (0..64).flat_map(|_| [0.5, 0.25]).collect();
    let (mut control, mut stream) = open(ChannelCount::Surround51);
    let clip = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            2,
            RATE,
            interleaved.clone(),
        ))))
        .expect("register_clip");
    control.start(clip, VoiceParams::default()).expect("start");
    let out = stream.render_frames(64);
    for frame in out.as_chunks::<6>().0 {
        assert_eq!(*frame, [0.5, 0.25, 0.0, 0.0, 0.0, 0.0]);
    }

    let (mut control, mut stream) = open(ChannelCount::Mono);
    let clip = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            2,
            RATE,
            interleaved,
        ))))
        .expect("register_clip");
    control.start(clip, VoiceParams::default()).expect("start");
    let out = stream.render_frames(64);
    assert!(out.iter().all(|&s| s == 0.375), "{:?}", &out[..4]);
}

#[test]
fn a_released_clip_is_freed_on_the_game_thread_after_its_last_voice() {
    let (mut control, mut stream) = open(ChannelCount::Stereo);
    let data = Arc::new(ClipData::from(AudioClip::new(1, RATE, ramp(1000))));
    let weak = Arc::downgrade(&data);
    let clip = control.register_clip(data).expect("register_clip");
    let voice = control.start(clip, VoiceParams::default()).expect("start");

    // Released while the voice plays: the mixer keeps the clip until the voice ends.
    control.release_clip(clip).expect("release_clip");
    stream.render_frames(256);
    control.update();
    assert!(weak.upgrade().is_some(), "the playing voice keeps the clip");
    assert!(control.is_playing(voice));

    stream.render_frames(1000);
    // The mixer has handed the clip back, but the game thread has not received it yet.
    assert!(weak.upgrade().is_some());
    control.update();
    assert!(!control.is_playing(voice));
    assert!(
        weak.upgrade().is_none(),
        "the game thread dropped the released clip"
    );
}
