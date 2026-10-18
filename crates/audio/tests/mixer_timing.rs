//! Ramps, fades and scheduling in the mixer (FIX-7.2, finding AUD-06, ADR-050).
//!
//! Before FIX-7.2, gains and pitch changed in one step at the game tick (zipper noise and
//! clicks), and nothing could start or stop a sound between ticks. Now every gain and pitch change
//! ramps per sample over at least 5 ms, fades follow linear or equal-power curves, and starts and
//! stops can be scheduled on an output frame.

use korppi_audio::mixer::{
    At, AudioMixer, ClipData, Fade, MixerConfig, MixerControl, VoiceHandle, VoiceParams,
};
use korppi_audio::{
    AudioClip, AudioDevice, AudioStream, AudioStreamConfig, BufferSize, ChannelCount,
    HeadlessDummyDevice,
};
use std::sync::Arc;

const RATE: u32 = 48_000;
/// 5 ms at 48 kHz: the shortest ramp.
const MIN_RAMP: usize = 240;

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

fn clip(
    control: &mut MixerControl,
    channels: u16,
    samples: Vec<f32>,
) -> korppi_audio::mixer::ClipId {
    control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            channels, RATE, samples,
        ))))
        .expect("register_clip")
}

/// A looping clip of constant 1.0.
fn ones(control: &mut MixerControl) -> korppi_audio::mixer::ClipId {
    clip(control, 1, vec![1.0; 64])
}

fn looping() -> VoiceParams {
    VoiceParams {
        looping: true,
        ..VoiceParams::default()
    }
}

fn left(out: &[f32]) -> Vec<f32> {
    out.iter().step_by(2).copied().collect()
}

fn largest_step(samples: &[f32]) -> f32 {
    samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max)
}

/// `out` after the last frame of `before`: a change at the start of `out` shows as a step.
fn joined(before: &[f32], out: &[f32]) -> Vec<f32> {
    before.last().into_iter().chain(out).copied().collect()
}

#[test]
fn gain_changes_ramp_without_steps() {
    let (mut control, mut stream) = open();
    let ones = ones(&mut control);
    let voice = control.start(ones, looping()).unwrap();
    let before = left(&stream.render_frames(512));
    control.set_gain(voice, 0.0, 480).unwrap();
    let out = left(&stream.render_frames(1024));
    let ramp = joined(&before, &out);
    assert!(
        largest_step(&ramp) <= 1.0 / 480.0 + 1e-6,
        "a step of {} in a 480-frame ramp",
        largest_step(&ramp)
    );
    assert!(
        ramp.windows(2).all(|w| w[1] <= w[0]),
        "the ramp falls monotonically"
    );
    assert!(
        out[480..].iter().all(|&s| s == 0.0),
        "the ramp ends at its target"
    );
}

#[test]
fn instant_changes_still_ramp_over_five_milliseconds() {
    let (mut control, mut stream) = open();
    let ones = ones(&mut control);
    let voice = control.start(ones, looping()).unwrap();
    let before = left(&stream.render_frames(512));
    control.set_gain(voice, 0.0, 0).unwrap();
    let out = left(&stream.render_frames(1024));
    let step = largest_step(&joined(&before, &out));
    assert!(
        step <= 1.0 / MIN_RAMP as f32 + 1e-6,
        "an instant change stepped by {step}"
    );
    assert!(out[MIN_RAMP..].iter().all(|&s| s == 0.0));
}

#[test]
fn pitch_changes_ramp() {
    let (mut control, mut stream) = open();
    // A clip whose value is its own frame index (scaled): the output's slope is the pitch.
    let frames = 48_000;
    let scale = 1.0 / frames as f32;
    let index = clip(
        &mut control,
        1,
        (0..frames).map(|i| i as f32 * scale).collect(),
    );
    let voice = control.start(index, VoiceParams::default()).unwrap();
    stream.render_frames(512);
    control.set_pitch(voice, 2.0, 480).unwrap();
    let out = left(&stream.render_frames(2048));
    let slopes: Vec<f32> = out.windows(2).map(|w| (w[1] - w[0]) / scale).collect();
    assert!(
        (slopes[2] - 1.0).abs() < 0.01,
        "pitch 1 before the ramp: {}",
        slopes[2]
    );
    assert!(
        (slopes[1000] - 2.0).abs() < 0.01,
        "pitch 2 after it: {}",
        slopes[1000]
    );
    let jump = slopes
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max);
    assert!(jump < 0.01, "the pitch jumped by {jump} between frames");
}

#[test]
fn a_scheduled_start_begins_on_its_frame() {
    let (mut control, mut stream) = open();
    let ones = ones(&mut control);
    let at = control.clock().frames() + 300;
    control
        .start_with(ones, looping(), At::Frame(at), Fade::NONE)
        .unwrap();
    let out = left(&stream.render_frames(1024));
    assert!(
        out[..300].iter().all(|&s| s == 0.0),
        "silent before frame 300"
    );
    assert!(
        out[300..].iter().all(|&s| s == 1.0),
        "sounding from frame 300 on"
    );

    // A start scheduled in the past begins with the next block.
    let late = control
        .start_with(ones, looping(), At::Frame(0), Fade::NONE)
        .unwrap();
    let out = left(&stream.render_frames(256));
    assert!(
        out.iter().all(|&s| s == 2.0),
        "both voices sound: {}",
        out[0]
    );
    assert!(control.is_playing(late));
}

#[test]
fn a_scheduled_stop_fades_out_and_ends_the_voice() {
    let (mut control, mut stream) = open();
    let ones = ones(&mut control);
    let voice: VoiceHandle = control.start(ones, looping()).unwrap();
    let at = control.clock().frames() + 1000;
    control
        .stop_with(voice, At::Frame(at), Fade::linear(200))
        .unwrap();
    let out = left(&stream.render_frames(2048));
    assert!(out[..1000].iter().all(|&s| s == 1.0), "full until the stop");
    for (offset, &s) in out[1000..1200].iter().enumerate() {
        let expected = 1.0 - offset as f32 / 200.0;
        assert!((s - expected).abs() < 1e-5, "fade frame {offset}: {s}");
    }
    assert!(
        out[1200..].iter().all(|&s| s == 0.0),
        "silent after the fade"
    );
    control.update();
    assert!(!control.is_playing(voice), "the voice ended with its fade");
}

/// An equal-power crossfade keeps the summed power constant: one voice on the left fades out,
/// another on the right fades in, and left² + right² stays 1.
#[test]
fn an_equal_power_crossfade_keeps_the_power() {
    let (mut control, mut stream) = open();
    let left_only = clip(&mut control, 2, (0..64).flat_map(|_| [1.0, 0.0]).collect());
    let right_only = clip(&mut control, 2, (0..64).flat_map(|_| [0.0, 1.0]).collect());
    let outgoing = control.start(left_only, looping()).unwrap();
    let at = control.clock().frames() + 100;
    control
        .stop_with(outgoing, At::Frame(at), Fade::equal_power(1000))
        .unwrap();
    control
        .start_with(
            right_only,
            looping(),
            At::Frame(at),
            Fade::equal_power(1000),
        )
        .unwrap();
    let out = stream.render_frames(1200);
    for frame in 100..1100 {
        let (l, r) = (out[2 * frame], out[2 * frame + 1]);
        assert!(
            (l * l + r * r - 1.0).abs() < 1e-4,
            "frame {frame}: power {} ({l}, {r})",
            l * l + r * r
        );
    }
    assert!((out[2 * 600] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-2);
}

/// The game thread learns the output frame a voice first sounded on, to schedule what follows
/// it to the sample.
#[test]
fn the_start_frame_comes_back() {
    let (mut control, mut stream) = open();
    let ones = ones(&mut control);
    let at = control.clock().frames() + 300;
    let voice = control
        .start_with(ones, looping(), At::Frame(at), Fade::NONE)
        .unwrap();
    assert_eq!(control.started_at(voice), None, "not rendered yet");
    stream.render_frames(1024);
    assert_eq!(control.started_at(voice), Some(at));
    control.stop(voice).unwrap();
    stream.render_frames(1024);
    assert_eq!(control.started_at(voice), None, "ended");
}

/// A voice stopped before it first sounds ends without sounding, fade or not: cancelling a
/// scheduled start leaves no blip.
#[test]
fn a_voice_stopped_before_it_sounds_stays_silent() {
    let (mut control, mut stream) = open();
    let ones = ones(&mut control);
    let at = control.clock().frames() + 500;
    let voice = control
        .start_with(ones, looping(), At::Frame(at), Fade::NONE)
        .unwrap();
    control
        .stop_with(voice, At::Frame(at - 100), Fade::linear(1000))
        .unwrap();
    let out = stream.render_frames(2048);
    assert!(out.iter().all(|&s| s == 0.0), "the cancelled voice sounded");
    assert!(!control.is_playing(voice));
}
