//! Unit and integration tests for Sub-Milestone 10.2:
//! Core DSP Pipeline (Biquads, Limiter, Freeverb). The bus graph's tests went with it in FIX-7.5
//! (AUD-10, the owner's decision): the mixer's buses replaced it (`tests/mixer_correctness.rs`).

use korppi_audio::dsp::biquad::{BiquadFilter, BiquadType};
use korppi_audio::dsp::freeverb::Freeverb;
use korppi_audio::dsp::limiter::PeakLimiter;
use korppi_audio::frame::AudioFrame;
use std::f32::consts::PI;

fn generate_sine_wave(freq: f32, sample_rate: f32, num_samples: usize) -> Vec<f32> {
    let mut buffer = Vec::with_capacity(num_samples);
    let phase_inc = 2.0 * PI * freq / sample_rate;
    let mut phase = 0.0f32;
    for _ in 0..num_samples {
        buffer.push(phase.sin());
        phase = (phase + phase_inc) % (2.0 * PI);
    }
    buffer
}

fn compute_rms(samples: &[f32]) -> f32 {
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

#[test]
fn test_biquad_lowpass_attenuates_high_frequencies() {
    let sample_rate = 48_000.0;
    let mut lpf = BiquadFilter::new(
        BiquadType::LowPass,
        1_000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        sample_rate,
    );

    let low_freq_sine = generate_sine_wave(100.0, sample_rate, 2400);
    let high_freq_sine = generate_sine_wave(10_000.0, sample_rate, 2400);

    // Process low frequency (100 Hz, well below 1 kHz cutoff)
    let mut low_out = Vec::with_capacity(2400);
    for &sample in &low_freq_sine {
        low_out.push(lpf.process_sample(sample));
    }

    // Reset filter state for fair comparison
    lpf.reset();

    // Process high frequency (10 kHz, well above 1 kHz cutoff)
    let mut high_out = Vec::with_capacity(2400);
    for &sample in &high_freq_sine {
        high_out.push(lpf.process_sample(sample));
    }

    let low_rms = compute_rms(&low_out[1200..]); // Skip initial filter transient
    let high_rms = compute_rms(&high_out[1200..]);

    assert!(
        low_rms > 0.6,
        "100 Hz signal must pass through 1 kHz LPF with minimal loss (got {low_rms})"
    );
    assert!(
        high_rms < 0.1,
        "10 kHz signal must be heavily attenuated by 1 kHz LPF (got {high_rms})"
    );
}

#[test]
fn test_biquad_highpass_attenuates_low_frequencies() {
    let sample_rate = 48_000.0;
    let mut hpf = BiquadFilter::new(
        BiquadType::HighPass,
        2_000.0,
        std::f32::consts::FRAC_1_SQRT_2,
        sample_rate,
    );

    let low_freq_sine = generate_sine_wave(100.0, sample_rate, 2400);
    let high_freq_sine = generate_sine_wave(8_000.0, sample_rate, 2400);

    let mut low_out = Vec::with_capacity(2400);
    for &sample in &low_freq_sine {
        low_out.push(hpf.process_sample(sample));
    }

    hpf.reset();

    let mut high_out = Vec::with_capacity(2400);
    for &sample in &high_freq_sine {
        high_out.push(hpf.process_sample(sample));
    }

    let low_rms = compute_rms(&low_out[1200..]);
    let high_rms = compute_rms(&high_out[1200..]);

    assert!(
        low_rms < 0.05,
        "100 Hz signal must be heavily attenuated by 2 kHz HPF (got {low_rms})"
    );
    assert!(
        high_rms > 0.65,
        "8 kHz signal must pass through 2 kHz HPF with minimal loss (got {high_rms})"
    );
}

#[test]
fn test_limiter_clamps_loud_signals_without_digital_overflow() {
    let sample_rate = 48_000.0;
    let mut limiter = PeakLimiter::new(sample_rate);
    limiter.set_threshold_db(-0.5); // Peak threshold approx 0.944 linear

    // Generate extreme amplitude signal (overshooting by 5x)
    let loud_input = vec![5.0f32; 1000];
    let mut frames: Vec<AudioFrame> = loud_input.iter().map(|&s| AudioFrame::new(s, s)).collect();

    limiter.process(&mut frames);

    // Verify all frames are strictly clamped within [-1.0, 1.0]
    for frame in &frames {
        assert!(
            frame.left.abs() <= 1.0,
            "Left sample {} exceeded 1.0",
            frame.left
        );
        assert!(
            frame.right.abs() <= 1.0,
            "Right sample {} exceeded 1.0",
            frame.right
        );
    }
}

#[test]
fn test_limiter_lookahead_prevents_attack_overshoot_and_ramps_ahead_of_transient() {
    let sample_rate = 48_000.0;
    let mut limiter = PeakLimiter::new(sample_rate);
    let threshold_db = -1.0;
    let threshold_linear = 10.0f32.powf(threshold_db / 20.0); // ~0.89125
    limiter.set_threshold_db(threshold_db);

    let lookahead_frames = limiter.lookahead_frames();
    assert_eq!(
        lookahead_frames, 96,
        "Default 2 ms lookahead at 48 kHz must be 96 frames"
    );

    // 200 frames of sub-threshold signal (0.5), then a sudden 4.0 transient step.
    let transient_start = 200;
    let mut frames = vec![AudioFrame::new(0.5, -0.5); 600];
    for frame in &mut frames[transient_start..] {
        *frame = AudioFrame::new(4.0, -4.0);
    }

    limiter.process(&mut frames);

    // 1. No sample may ever overshoot threshold_linear (no hard-clipped 1.0 attack spike).
    for (i, frame) in frames.iter().enumerate() {
        assert!(
            frame.left.abs() <= threshold_linear + 1e-4
                && frame.right.abs() <= threshold_linear + 1e-4,
            "Sample {i} overshot threshold {threshold_linear}: ({}, {})",
            frame.left,
            frame.right
        );
    }

    // 2. The 0.5 signal exits the delay line starting at `lookahead_frames`, and the transient
    // exits at `transient_start + lookahead_frames`. Before the lookahead window sees the
    // transient, the 0.5 signal passes at unity gain (0.5). Just before the transient exits,
    // the lookahead ramp has already reduced the 0.5 signal smoothly toward `0.5 * (threshold / 4.0)`.
    let steady_before_ramp = frames[lookahead_frames + 10].left.abs();
    let pre_transient_ramped = frames[transient_start + lookahead_frames - 1].left.abs();
    let at_transient = frames[transient_start + lookahead_frames].left.abs();

    assert!(
        (steady_before_ramp - 0.5).abs() < 1e-4,
        "Sub-threshold signal before lookahead window must pass at unity (got {steady_before_ramp})"
    );
    assert!(
        pre_transient_ramped < 0.2,
        "Lookahead must ramp gain down ahead of the transient (got {pre_transient_ramped})"
    );
    assert!(
        (at_transient - threshold_linear).abs() < 1e-3,
        "Transient arrival must be limited to threshold {threshold_linear} (got {at_transient})"
    );
}

#[test]
fn test_freeverb_creates_reverberant_tail_from_impulse() {
    let sample_rate = 48_000.0;
    let mut reverb = Freeverb::new(sample_rate);
    reverb.set_room_size(0.8);
    reverb.set_damp(0.2);
    reverb.set_wet(1.0);
    reverb.set_dry(0.0);

    // Single impulse at sample 0 followed by 48,000 samples of silence (1 second at 48 kHz)
    let mut frames = vec![AudioFrame::SILENT; 48_000];
    frames[0] = AudioFrame::new(1.0, 1.0);

    reverb.process(&mut frames);

    // Initial buildup of diffuse reflections around 2,000 to 5,000 samples
    let peak_rms = compute_rms(
        &frames[2000..5000]
            .iter()
            .map(|f| f.to_mono())
            .collect::<Vec<_>>(),
    );

    assert!(
        peak_rms > 0.01,
        "Freeverb must generate a sustained reverberant tail from an impulse (got {peak_rms})"
    );

    // Later tail around 35,000 to 45,000 samples should be decaying smoothly
    let late_rms = compute_rms(
        &frames[35000..45000]
            .iter()
            .map(|f| f.to_mono())
            .collect::<Vec<_>>(),
    );
    assert!(
        late_rms < peak_rms,
        "Reverberant tail must decay over time (peak {peak_rms} vs late {late_rms})"
    );
}
