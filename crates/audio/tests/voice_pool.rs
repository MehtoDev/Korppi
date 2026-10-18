//! The voice pool and the voice API (FIX-7.2, ADR-050).
//!
//! The mixer mixes at most `max_voices` voices: the playing voices that rank highest by priority,
//! then by audibility (gain, fades and buses). The others are virtual: their cursors advance but
//! they are not mixed, and voices move between the two with a 5 ms fade. When every slot,
//! audible or virtual, is in use, a new voice replaces the lowest-priority voice (then the
//! quietest, then the oldest) unless every voice outranks it. Handles are generational: a handle
//! to a voice that ended names nothing, even after its slot is reused.

use korppi_audio::mixer::{
    At, AudioMixer, AudioVoices, ClipData, ClipId, Fade, MixerConfig, MixerControl, VoiceParams,
};
use korppi_audio::{
    AudioClip, AudioDevice, AudioError, AudioStream, AudioStreamConfig, BufferSize, ChannelCount,
    HeadlessDummyDevice,
};
use korppi_ecs::{ResMut, Schedule, Stage, World};
use std::sync::Arc;

const RATE: u32 = 48_000;
/// 5 ms at 48 kHz: the fade between audible and virtual.
const FADE: usize = 240;

fn open(max_voices: usize, max_virtual_voices: usize) -> (MixerControl, Box<dyn AudioStream>) {
    let config = AudioStreamConfig {
        sample_rate: RATE,
        channels: ChannelCount::Stereo,
        buffer_size: BufferSize::Fixed(256),
    };
    let (control, mixer) = AudioMixer::new(MixerConfig {
        max_voices,
        max_virtual_voices,
        ..MixerConfig::for_stream(&config)
    });
    let mut stream = HeadlessDummyDevice::new()
        .create_output_stream(config, Box::new(mixer))
        .expect("stream");
    stream.play().expect("play");
    (control, stream)
}

fn clip(control: &mut MixerControl, samples: Vec<f32>) -> ClipId {
    control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(1, RATE, samples))))
        .expect("register_clip")
}

fn looping(gain: f32, priority: u8) -> VoiceParams {
    VoiceParams {
        gain,
        priority,
        looping: true,
        ..VoiceParams::default()
    }
}

fn left(out: &[f32]) -> Vec<f32> {
    out.iter().step_by(2).copied().collect()
}

fn settled(out: &[f32]) -> f32 {
    let tail = &out[out.len() - 64..];
    assert!(
        tail.iter().all(|&s| (s - tail[0]).abs() < 1e-5),
        "the output has not settled: {tail:?}"
    );
    tail[0]
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

/// With room for two audible voices, the two loudest of three are mixed, and a voice of higher
/// priority is mixed before louder ones.
#[test]
fn only_the_highest_ranked_voices_are_mixed() {
    let (mut control, mut stream) = open(2, 8);
    let ones = clip(&mut control, vec![1.0; 64]);
    for gain in [0.1, 0.2, 0.4] {
        control.start(ones, looping(gain, 128)).unwrap();
    }
    let out = left(&stream.render_frames(1024));
    assert!(
        out.iter().all(|&s| (s - 0.6).abs() < 1e-5),
        "the 0.4 and 0.2 voices from the first frame: {}",
        out[0]
    );

    control.start(ones, looping(0.05, 200)).unwrap();
    let out = left(&stream.render_frames(1024));
    assert!(
        (settled(&out) - 0.45).abs() < 1e-5,
        "the priority-200 voice displaces the 0.2 voice: {}",
        settled(&out)
    );
}

/// A voice moving between audible and virtual fades over 5 ms instead of cutting.
#[test]
fn voices_fade_between_audible_and_virtual() {
    let (mut control, mut stream) = open(2, 8);
    let ones = clip(&mut control, vec![1.0; 64]);
    let silence = clip(&mut control, vec![0.0; 64]);
    control.start(ones, looping(0.2, 128)).unwrap();
    control.start(ones, looping(0.4, 128)).unwrap();
    let before = left(&stream.render_frames(512));

    // A silent voice of higher priority takes the 0.2 voice's place.
    let silent = control.start(silence, looping(1.0, 200)).unwrap();
    let virtual_out = left(&stream.render_frames(1024));
    let step = largest_step(&joined(&before, &virtual_out));
    assert!(
        step <= 0.2 / FADE as f32 + 1e-6,
        "going virtual stepped by {step}"
    );
    assert!(
        (settled(&virtual_out) - 0.4).abs() < 1e-5,
        "{}",
        settled(&virtual_out)
    );

    // When it stops, the 0.2 voice fades back in.
    control.stop(silent).unwrap();
    let out = left(&stream.render_frames(2048));
    let step = largest_step(&joined(&virtual_out, &out));
    assert!(
        step <= 0.2 / FADE as f32 + 1e-6,
        "becoming audible stepped by {step}"
    );
    assert!((settled(&out) - 0.6).abs() < 1e-5, "{}", settled(&out));
}

/// A virtual voice keeps its place in the clip: when it becomes audible again, it plays what it
/// would have played had it been mixed all along.
#[test]
fn virtual_voices_keep_their_place() {
    let (mut control, mut stream) = open(2, 8);
    let frames = 48_000;
    let scale = 1.0 / frames as f32;
    // The clip's value is its frame index, scaled: the output shows the voice's cursor.
    let index = clip(
        &mut control,
        (0..frames).map(|i| i as f32 * scale).collect(),
    );
    let ones = clip(&mut control, vec![1.0; 64]);
    control
        .start(
            index,
            VoiceParams {
                priority: 0,
                ..VoiceParams::default()
            },
        )
        .unwrap();
    stream.render_frames(1000);
    let loud = [
        control.start(ones, looping(1.0, 200)).unwrap(),
        control.start(ones, looping(1.0, 200)).unwrap(),
    ];
    let out = left(&stream.render_frames(4000));
    assert!(
        (settled(&out) - 2.0).abs() < 1e-5,
        "only the two loud voices are mixed: {}",
        settled(&out)
    );
    for voice in loud {
        control.stop(voice).unwrap();
    }
    let out = left(&stream.render_frames(2000));
    // Output frames 6500..7000 come from the index clip alone, at its own frames 6500..7000.
    for (offset, &s) in out[1500..].iter().enumerate() {
        let expected = (6500 + offset) as f32 * scale;
        assert!(
            (s - expected).abs() < 1e-4,
            "output frame {}: {s}, not {expected}",
            6500 + offset
        );
    }
}

/// A full pool replaces its lowest-priority voice, then its quietest; a voice that every playing
/// voice outranks is refused.
#[test]
fn a_full_pool_replaces_the_lowest_priority_voice() {
    let (mut control, mut stream) = open(4, 0);
    let ones = clip(&mut control, vec![1.0; 64]);
    let a = control.start(ones, looping(2.0, 100)).unwrap();
    let b = control.start(ones, looping(1.0, 100)).unwrap();
    let c = control.start(ones, looping(4.0, 50)).unwrap();
    let d = control.start(ones, looping(8.0, 150)).unwrap();
    let out = left(&stream.render_frames(512));
    assert!((settled(&out) - 15.0).abs() < 1e-4);

    let e = control.start(ones, looping(16.0, 100)).unwrap();
    assert!(!control.is_playing(c), "the priority-50 voice was replaced");
    assert!(control.is_playing(e));
    let out = left(&stream.render_frames(1024));
    assert!((settled(&out) - 27.0).abs() < 1e-4, "{}", settled(&out));

    assert!(
        matches!(
            control.start(ones, looping(1.0, 40)),
            Err(AudioError::VoicePoolFull)
        ),
        "every playing voice outranks a priority-40 voice"
    );

    let g = control.start(ones, looping(32.0, 100)).unwrap();
    assert!(!control.is_playing(b), "the quietest priority-100 voice");
    assert!(control.is_playing(a) && control.is_playing(d) && control.is_playing(g));
    let out = left(&stream.render_frames(1024));
    assert!((settled(&out) - 58.0).abs() < 1e-4, "{}", settled(&out));
}

/// A replaced audible voice fades out instead of cutting off.
#[test]
fn a_replaced_voice_fades_out() {
    let (mut control, mut stream) = open(1, 0);
    let ones = clip(&mut control, vec![1.0; 64]);
    let silence = clip(&mut control, vec![0.0; 64]);
    control.start(ones, looping(1.0, 100)).unwrap();
    let before = left(&stream.render_frames(512));
    control.start(silence, looping(1.0, 100)).unwrap();
    let out = left(&stream.render_frames(1024));
    let step = largest_step(&joined(&before, &out));
    assert!(
        step <= 1.0 / FADE as f32 + 1e-6,
        "the replaced voice stepped by {step}"
    );
    assert_eq!(settled(&out), 0.0);
}

/// A handle to a voice that ended names nothing once its slot holds another voice.
#[test]
fn stale_handles_name_nothing() {
    let (mut control, mut stream) = open(1, 0);
    let short = clip(&mut control, vec![1.0; 100]);
    let ones = clip(&mut control, vec![1.0; 64]);
    let first = control.start(short, VoiceParams::default()).unwrap();
    stream.render_frames(512);
    assert!(!control.is_playing(first), "the 100-frame voice ended");
    control.update();

    let second = control.start(ones, looping(1.0, 128)).unwrap();
    assert_ne!(first, second);
    control.set_gain(first, 0.0, 0).unwrap();
    control.stop(first).unwrap();
    let out = left(&stream.render_frames(1024));
    assert!(
        out.iter().all(|&s| s == 1.0),
        "commands to the ended voice reached its slot's new voice"
    );
    assert!(!control.is_playing(first) && control.is_playing(second));
}

/// `AudioVoices` is a world resource, so gameplay systems start voices on the audio clock.
#[test]
fn systems_start_voices_through_the_resource() {
    let (mut control, mut stream) = open(4, 4);
    let ones = clip(&mut control, vec![1.0; 64]);
    let mut world = World::new();
    world.insert_resource(AudioVoices::new(control));
    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, move |mut voices: ResMut<AudioVoices>| {
        let at = voices.clock().frames() + 100;
        voices
            .start_with(ones, looping(1.0, 128), At::Frame(at), Fade::NONE)
            .unwrap();
    });
    schedule.run(&mut world);
    let out = left(&stream.render_frames(512));
    assert!(out[..100].iter().all(|&s| s == 0.0));
    assert!(out[100..].iter().all(|&s| s == 1.0));
}
