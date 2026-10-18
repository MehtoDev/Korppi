//! What a voice does beyond gain and pitch, for players to build on (FIX-7.2, ADR-050):
//! stereo gains and a low-pass filter whose cutoff moves smoothly (spatialization and occlusion,
//! AUD-06), an intro followed by a loop region, pause and resume, and the playback position
//! reported back to the game thread.

use korppi_audio::mixer::{
    AudioMixer, ClipData, ClipId, MixerConfig, MixerControl, VoiceParams, VoiceSpatial,
};
use korppi_audio::{
    AudioClip, AudioDevice, AudioStream, AudioStreamConfig, BufferSize, ChannelCount,
    HeadlessDummyDevice,
};
use std::f32::consts::TAU;
use std::sync::Arc;

const RATE: u32 = 48_000;
/// 5 ms at 48 kHz.
const FADE: usize = 240;

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

fn clip(control: &mut MixerControl, rate: u32, samples: Vec<f32>) -> ClipId {
    control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(1, rate, samples))))
        .expect("register_clip")
}

fn sine(frequency: f32, frames: usize) -> Vec<f32> {
    (0..frames)
        .map(|n| (TAU * frequency * n as f32 / RATE as f32).sin())
        .collect()
}

fn looping() -> VoiceParams {
    VoiceParams {
        looping: true,
        ..VoiceParams::default()
    }
}

fn channel(out: &[f32], channel: usize) -> Vec<f32> {
    out.iter().skip(channel).step_by(2).copied().collect()
}

fn rms_db(samples: &[f32]) -> f32 {
    let mean = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    10.0 * mean.max(1e-20).log10()
}

/// The level of a looping sine of `frequency` through a filter at `cutoff_hz`, relative to a
/// full-scale sine (-3.01 dB RMS).
fn filtered_level(frequency: f32, cutoff_hz: f32) -> f32 {
    let (mut control, mut stream) = open();
    // 4800 frames hold a whole number of cycles of every frequency tested.
    let tone = clip(&mut control, RATE, sine(frequency, 4800));
    let voice = control.start(tone, looping()).unwrap();
    control
        .set_spatial(
            voice,
            VoiceSpatial {
                cutoff_hz,
                ..VoiceSpatial::default()
            },
            0,
        )
        .unwrap();
    let out = channel(&stream.render_frames(9600), 0);
    rms_db(&out[4800..]) - rms_db(&sine(frequency, 4800))
}

#[test]
fn stereo_gains_pan_a_voice_smoothly() {
    let (mut control, mut stream) = open();
    let ones = clip(&mut control, RATE, vec![1.0; 64]);
    let voice = control.start(ones, looping()).unwrap();
    let before = stream.render_frames(512);
    control
        .set_spatial(
            voice,
            VoiceSpatial {
                gain_left: 0.25,
                gain_right: 1.0,
                ..VoiceSpatial::default()
            },
            480,
        )
        .unwrap();
    let out = stream.render_frames(1024);
    let left: Vec<f32> = [before[before.len() - 2]]
        .into_iter()
        .chain(channel(&out, 0))
        .collect();
    let step = left
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max);
    assert!(
        step <= 0.75 / 480.0 + 1e-6,
        "the left gain stepped by {step}"
    );
    assert!((left[left.len() - 1] - 0.25).abs() < 1e-6);
    assert!(channel(&out, 1).iter().all(|&s| (s - 1.0).abs() < 1e-6));
}

#[test]
fn the_low_pass_filter_passes_lows_and_cuts_highs() {
    let pass = filtered_level(100.0, 500.0);
    let corner = filtered_level(500.0, 500.0);
    let stop = filtered_level(6000.0, 500.0);
    assert!(pass.abs() < 0.1, "100 Hz under a 500 Hz cutoff: {pass} dB");
    assert!(
        (corner + 3.0).abs() < 0.3,
        "500 Hz at the cutoff: {corner} dB"
    );
    assert!(stop < -30.0, "6 kHz under a 500 Hz cutoff: {stop} dB");
}

/// An open filter leaves the voice untouched, sample for sample.
#[test]
fn an_open_filter_changes_nothing() {
    let (mut control, mut stream) = open();
    let samples = sine(9000.0, 4800);
    let tone = clip(&mut control, RATE, samples.clone());
    control.start(tone, VoiceParams::default()).unwrap();
    let out = channel(&stream.render_frames(4800), 0);
    assert_eq!(out, samples);
}

/// A cutoff change moves over its ramp: the level of a 4 kHz tone falls window by window, not
/// at once.
#[test]
fn the_cutoff_moves_smoothly() {
    let (mut control, mut stream) = open();
    let tone = clip(&mut control, RATE, sine(4000.0, 4800));
    let voice = control.start(tone, looping()).unwrap();
    stream.render_frames(1024);
    control
        .set_spatial(
            voice,
            VoiceSpatial {
                cutoff_hz: 500.0,
                ..VoiceSpatial::default()
            },
            4800,
        )
        .unwrap();
    let out = channel(&stream.render_frames(9600), 0);
    let levels: Vec<f32> = out.chunks(480).map(rms_db).collect();
    let full = rms_db(&sine(4000.0, 4800));
    assert!(
        levels[0] - full > -1.0,
        "the first 10 ms of the ramp are barely filtered: {} dB",
        levels[0] - full
    );
    assert!(
        levels.windows(2).take(10).all(|w| w[1] < w[0] + 0.1),
        "the level falls through the ramp: {levels:?}"
    );
    assert!(
        levels[12] - full < -30.0,
        "the 500 Hz cutoff is reached: {} dB",
        levels[12] - full
    );
}

/// A voice plays its intro once, then repeats its loop region.
#[test]
fn an_intro_plays_once_then_the_loop_region_repeats() {
    let (mut control, mut stream) = open();
    let samples: Vec<f32> = [1.0, 2.0, 3.0]
        .into_iter()
        .flat_map(|v| std::iter::repeat_n(v, 100))
        .collect();
    let music = clip(&mut control, RATE, samples);
    control
        .start(
            music,
            VoiceParams {
                looping: true,
                loop_start: 100,
                loop_end: Some(200),
                ..VoiceParams::default()
            },
        )
        .unwrap();
    let out = channel(&stream.render_frames(1000), 0);
    assert!(out[..99].iter().all(|&s| s == 1.0), "the intro");
    assert!(
        out[100..].iter().all(|&s| s == 2.0),
        "the loop region, never the tail after it: {:?}",
        &out[195..210]
    );
}

/// A paused voice fades out and holds its place; resumed, it fades in and continues from there.
#[test]
fn pause_holds_the_place_and_resume_continues() {
    let (mut control, mut stream) = open();
    let frames = 48_000;
    let scale = 1.0 / frames as f32;
    let index = clip(
        &mut control,
        RATE,
        (0..frames).map(|i| i as f32 * scale).collect(),
    );
    let voice = control.start(index, VoiceParams::default()).unwrap();
    stream.render_frames(1000);
    control.pause(voice).unwrap();
    let paused = channel(&stream.render_frames(2000), 0);
    assert!(
        paused[FADE..].iter().all(|&s| s == 0.0),
        "silent once the pause has faded"
    );
    assert!(control.is_playing(voice), "a paused voice still plays");
    control.resume(voice).unwrap();
    let resumed = channel(&stream.render_frames(1000), 0);
    // It paused at frame 1000 + FADE of the clip and resumes from there, fading in.
    for (offset, &s) in resumed[FADE..].iter().enumerate() {
        let expected = (1000 + 2 * FADE + offset) as f32 * scale;
        assert!(
            (s - expected).abs() < 1e-4,
            "resumed frame {offset}: {s}, not {expected}"
        );
    }
}

/// The game thread reads a voice's position in its clip, in seconds of the clip.
#[test]
fn the_position_comes_back_to_the_game_thread() {
    let (mut control, mut stream) = open();
    let slow = clip(&mut control, 44_100, vec![0.5; 88_200]);
    let voice = control.start(slow, VoiceParams::default()).unwrap();
    assert_eq!(control.position(voice), Some(0.0), "not started yet");
    stream.render_frames(24_000);
    let position = control.position(voice).expect("playing");
    assert!(
        (position - 0.5).abs() < 1e-4,
        "half a second of output is half a second of the clip: {position}"
    );
    control.stop(voice).unwrap();
    stream.render_frames(1024);
    assert_eq!(control.position(voice), None, "an ended voice has none");
}

/// Settings sent before a voice first sounds apply from its first frame: a voice can start
/// already panned, filtered and at its gain, since it cannot click before it sounds.
#[test]
fn settings_before_a_voice_sounds_apply_at_once() {
    let (mut control, mut stream) = open();
    let ones = clip(&mut control, RATE, vec![1.0; 64]);
    let voice = control.start(ones, looping()).unwrap();
    control
        .set_spatial(
            voice,
            VoiceSpatial {
                gain_left: 0.0,
                gain_right: 0.5,
                ..VoiceSpatial::default()
            },
            0,
        )
        .unwrap();
    control.set_gain(voice, 0.5, 0).unwrap();
    let out = stream.render_frames(256);
    assert!(channel(&out, 0).iter().all(|&s| s == 0.0));
    assert!(channel(&out, 1).iter().all(|&s| s == 0.25));
}

/// Each ear's delay shifts its channel by exactly that many frames, fractions included: on a
/// ramp, which cubic interpolation reproduces exactly, a 10-frame delay reads 10 frames back and
/// a 10.5-frame delay 10.5.
#[test]
fn ear_delays_shift_each_channel() {
    for frames in [10.0f32, 10.5] {
        let (mut control, mut stream) = open();
        let scale = 1.0 / 48_000.0;
        let ramp = clip(
            &mut control,
            RATE,
            (0..48_000).map(|i| i as f32 * scale).collect(),
        );
        let voice = control.start(ramp, VoiceParams::default()).unwrap();
        control
            .set_spatial(
                voice,
                VoiceSpatial {
                    delay_left: frames / RATE as f32,
                    ..VoiceSpatial::default()
                },
                0,
            )
            .unwrap();
        let out = stream.render_frames(1024);
        let (left, right) = (channel(&out, 0), channel(&out, 1));
        for n in 20..1024 {
            assert_eq!(right[n], n as f32 * scale, "the right ear is not delayed");
            let expected = (n as f32 - frames) * scale;
            assert!(
                (left[n] - expected).abs() < 1e-6,
                "{frames} frames: left[{n}] = {}, not {expected}",
                left[n]
            );
        }
    }
}
