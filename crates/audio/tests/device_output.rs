//! Opening the output (FIX-7.2, ADR-050; finding AUD-01): the default device through `cpal`, with
//! its sample format and rate negotiated, or a headless output that runs the mixer in step with
//! game time.
//!
//! The device test needs an output device. Without one it skips itself; `KORPPI_REQUIRE_AUDIO=1`
//! makes it fail instead, as on the owner's machine.

use korppi_audio::mixer::{AudioVoices, ClipData, VoiceParams};
use korppi_audio::output::{AudioBackend, AudioOutput};
use korppi_audio::{AudioClip, AudioSettings};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn tone(
    voices: &mut AudioVoices,
    rate: u32,
    seconds: f32,
    looping: bool,
) -> korppi_audio::mixer::VoiceHandle {
    let clip = voices
        .control()
        .register_clip(Arc::new(ClipData::from(AudioClip::sine(
            440.0, seconds, rate,
        ))))
        .expect("register_clip");
    voices
        .start(
            clip,
            VoiceParams {
                gain: 0.1,
                looping,
                ..VoiceParams::default()
            },
        )
        .expect("start")
}

/// The rate the device pulls frames at over `window`: the least-squares slope of the mixer's
/// clock against wall time, sampled every 5 ms. The clock advances a device period at a time, so
/// the difference of two readings is off by up to a period at each end; the slope is not.
fn pull_rate(voices: &AudioVoices, window: Duration) -> f64 {
    let start = Instant::now();
    let mut samples = Vec::new();
    while start.elapsed() < window {
        samples.push((
            start.elapsed().as_secs_f64(),
            voices.clock().frames() as f64,
        ));
        std::thread::sleep(Duration::from_millis(5));
    }
    let n = samples.len() as f64;
    let (mean_t, mean_f) = samples
        .iter()
        .fold((0.0, 0.0), |(t, f), &(ti, fi)| (t + ti / n, f + fi / n));
    let (covariance, variance) = samples.iter().fold((0.0, 0.0), |(c, v), &(t, f)| {
        (
            c + (t - mean_t) * (f - mean_f),
            v + (t - mean_t) * (t - mean_t),
        )
    });
    covariance / variance
}

fn audio_required() -> bool {
    std::env::var("KORPPI_REQUIRE_AUDIO").is_ok_and(|v| v == "1")
}

/// The default device pulls the mixer's output in real time: over a steady stretch its clock
/// advances at the device's rate, and a one-shot ends after its length, less the device's buffer
/// (the mixer runs ahead of the speaker by the buffer it fills when the stream starts).
#[test]
fn the_default_device_plays_the_mixer() {
    let (output, mut voices) =
        AudioOutput::open(AudioBackend::Device, &AudioSettings::default()).expect("open");
    if output.is_headless() {
        assert!(
            !audio_required(),
            "KORPPI_REQUIRE_AUDIO=1, but no output device opened: {}",
            output.fallback_reason().unwrap_or("headless")
        );
        eprintln!(
            "skipped: no audio output device ({})",
            output.fallback_reason().unwrap_or("headless")
        );
        return;
    }
    let config = output.config();
    let rate = f64::from(config.sample_rate);
    assert!(config.sample_rate >= 8_000);

    // A quiet hum while the stream settles, then a steady stretch.
    let opened = Instant::now();
    let hum = tone(&mut voices, config.sample_rate, 1.0, true);
    std::thread::sleep(Duration::from_millis(300));
    let lead = voices.clock().frames() as f64 / rate - opened.elapsed().as_secs_f64();
    let pulled = pull_rate(&voices, Duration::from_millis(1500));
    eprintln!(
        "device {:?}: {} Hz, {:?}, {}; pulls {pulled:.0} frames/s; mixer {:.0} ms ahead",
        output.device_name(),
        config.sample_rate,
        config.channels,
        output.sample_format(),
        lead * 1000.0
    );
    assert!(
        (pulled / rate - 1.0).abs() < 0.05,
        "the device pulled {pulled:.0} frames per second at {rate} Hz"
    );

    voices.stop(hum).unwrap();
    let shot = tone(&mut voices, config.sample_rate, 0.3, false);
    let started = Instant::now();
    while voices.is_playing(shot) && started.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(5));
        voices.update();
    }
    let elapsed = started.elapsed().as_secs_f64();
    assert!(!voices.is_playing(shot), "the one-shot never ended");
    assert!(
        elapsed > 0.3 - lead.max(0.0) - 0.05 && elapsed < 0.3 + 0.2,
        "a 0.3 s one-shot ended after {elapsed:.3} s, with the mixer {lead:.3} s ahead"
    );
    drop(output);
}

/// A headless output renders the mixer by game time: half a second of updates is half a second
/// of frames, and a quarter-second clip ends within it.
#[test]
fn the_headless_output_follows_game_time() {
    let (mut output, mut voices) =
        AudioOutput::open(AudioBackend::Headless, &AudioSettings::default()).expect("open");
    assert!(output.is_headless());
    let rate = output.config().sample_rate;
    let voice = tone(&mut voices, rate, 0.25, false);
    for _ in 0..30 {
        output.advance(Duration::from_secs_f64(1.0 / 60.0));
    }
    voices.update();
    assert_eq!(voices.clock().frames(), u64::from(rate) / 2);
    assert!(!voices.is_playing(voice), "the clip ended");
}

/// `AudioSettings::max_voices` sizes the mixer: with 2, the quietest of three voices is virtual.
#[test]
fn max_voices_comes_from_the_settings() {
    let settings = AudioSettings {
        max_voices: 2,
        ..AudioSettings::default()
    };
    let (mut output, mut voices) =
        AudioOutput::open(AudioBackend::Headless, &settings).expect("open");
    let rate = output.config().sample_rate;
    let clip = voices
        .control()
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            rate,
            vec![1.0; 64],
        ))))
        .unwrap();
    for gain in [0.1, 0.2, 0.4] {
        voices
            .start(
                clip,
                VoiceParams {
                    gain,
                    looping: true,
                    ..VoiceParams::default()
                },
            )
            .unwrap();
    }
    output.advance(Duration::from_millis(20));
    let last = output.last_output();
    assert!(
        (last[last.len() - 1] - 0.6).abs() < 1e-5,
        "two voices mixed: {}",
        last[last.len() - 1]
    );
}
