//! Integration and unit tests for Sub-Milestone 10.5:
//! Hybrid Acoustic Synthesis & Dynamic Reverb.

use korppi_audio::synthesis::{
    fdn::{householder_diffuse_8, FdnConfig, FeedbackDelayNetwork},
    tapped_delay::{EarlyReflectionTap, TappedDelayLine},
    triple_buffer::triple_buffer,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

#[test]
fn test_tapped_delay_line_impulse_response() {
    let mut delay_line = TappedDelayLine::new(4800, 48_000.0);

    // Setup 2 discrete early reflection taps
    // Tap 0: 10 samples delay (10 / 48000 s ≈ 0.0002083 s), gain = 0.8, pan = 0.0 (Left: cos(0)=1, Right: sin(0)=0)
    // Tap 1: 25 samples delay (25 / 48000 s ≈ 0.0005208 s), gain = 0.5, pan = PI/2 (Left: 0, Right: 1)
    let taps = vec![
        EarlyReflectionTap {
            delay_samples: 10.0,
            gain_left: 0.8,
            gain_right: 0.0,
            damping: 0.0,
        },
        EarlyReflectionTap {
            delay_samples: 25.0,
            gain_left: 0.0,
            gain_right: 0.5,
            damping: 0.0,
        },
    ];

    delay_line.set_taps(&taps);

    // Feed impulse: 1.0 at n=0, followed by zeroes
    let mut out_left = Vec::with_capacity(50);
    let mut out_right = Vec::with_capacity(50);

    for n in 0..50 {
        let input = if n == 0 { 1.0 } else { 0.0 };
        let (l, r) = delay_line.process_sample(input);
        out_left.push(l);
        out_right.push(r);
    }

    // Tap 0 at n=10 should have output on Left channel
    assert!(
        (out_left[10] - 0.8).abs() < 0.05,
        "Left tap 0 expected ~0.8, got {}",
        out_left[10]
    );
    assert!(out_right[10].abs() < 1e-4);

    // Tap 1 at n=25 should have output on Right channel
    assert!(out_left[25].abs() < 1e-4);
    assert!(
        (out_right[25] - 0.5).abs() < 0.05,
        "Right tap 1 expected ~0.5, got {}",
        out_right[25]
    );

    // Sample 5 and 35 should be zero
    assert!(out_left[5].abs() < 1e-4);
    assert!(out_right[5].abs() < 1e-4);
    assert!(out_left[35].abs() < 1e-4);
    assert!(out_right[35].abs() < 1e-4);
}

#[test]
fn test_tapped_delay_line_raised_cosine_crossfade_no_clicks() {
    let mut delay_line = TappedDelayLine::new(4800, 48_000.0);

    // Initial tap at 20 samples
    delay_line.set_taps(&[EarlyReflectionTap {
        delay_samples: 20.0,
        gain_left: 0.7,
        gain_right: 0.7,
        damping: 0.0,
    }]);

    // Feed continuous DC signal 0.5
    for _ in 0..100 {
        let _ = delay_line.process_sample(0.5);
    }

    // Move tap to 40 samples (triggers raised-cosine crossfade)
    delay_line.set_taps(&[EarlyReflectionTap {
        delay_samples: 40.0,
        gain_left: 0.7,
        gain_right: 0.7,
        damping: 0.0,
    }]);

    let mut prev_l = 0.5 * 0.7;
    for _ in 0..500 {
        let (l, _r) = delay_line.process_sample(0.5);
        let delta = (l - prev_l).abs();
        // Should not have abrupt discontinuity or click
        assert!(
            delta < 0.1,
            "Discontinuity detected during crossfade: delta = {}",
            delta
        );
        prev_l = l;
    }
}

#[test]
fn test_householder_fdn_energy_conservation_and_lossless_matrix() {
    // Householder matrix A = I - (2/8) * 1 * 1^T
    // For any u, ||A u||^2 == ||u||^2 (unitary / energy preserving)
    let test_vectors: [[f32; 8]; 3] = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        [1.2, 3.4, -0.8, 2.1, -1.5, 0.7, -2.2, 1.9],
    ];

    for u in test_vectors {
        let v = householder_diffuse_8(u);

        let energy_u: f32 = u.iter().map(|x| x * x).sum();
        let energy_v: f32 = v.iter().map(|x| x * x).sum();

        assert!(
            (energy_u - energy_v).abs() < 1e-4,
            "Householder must be lossless: u energy = {}, v energy = {}",
            energy_u,
            energy_v
        );
    }
}

#[test]
fn test_fdn_reverb_decay_calibrated_to_rt60() {
    let sample_rate = 48_000.0;
    let mut fdn_short = FeedbackDelayNetwork::new(FdnConfig::default(), sample_rate);
    let mut fdn_long = FeedbackDelayNetwork::new(FdnConfig::default(), sample_rate);

    // Short RT60: 0.4 seconds across all bands
    fdn_short.set_rt60([0.4, 0.4, 0.4], sample_rate);
    // Long RT60: 2.5 seconds across all bands
    fdn_long.set_rt60([2.5, 2.5, 2.5], sample_rate);

    // Feed impulse into both
    let total_samples = (sample_rate * 1.0) as usize; // 1 second of simulation
    let mut energy_short = 0.0f32;
    let mut energy_long = 0.0f32;

    for n in 0..total_samples {
        let input = if n == 0 { 1.0 } else { 0.0 };
        let (sl, sr) = fdn_short.process_sample(input);
        let (ll, lr) = fdn_long.process_sample(input);

        // Measure remaining energy in the tail between 0.6s and 1.0s
        if n > (sample_rate * 0.6) as usize {
            energy_short += sl * sl + sr * sr;
            energy_long += ll * ll + lr * lr;
        }
    }

    assert!(
        energy_long > energy_short * 10.0,
        "Long RT60 must have much higher tail energy than short RT60: long = {}, short = {}",
        energy_long,
        energy_short
    );
}

#[test]
fn test_wait_free_triple_buffer_producer_consumer() {
    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    struct AcousticState {
        room_rt60: [f32; 3],
        tap_count: u32,
    }

    let (mut producer, mut consumer) = triple_buffer::<AcousticState>();

    // Initial state
    let state = consumer.read();
    assert_eq!(state.tap_count, 0);

    // Write update 1
    producer.write(AcousticState {
        room_rt60: [1.2, 1.0, 0.8],
        tap_count: 8,
    });

    let state = consumer.read();
    assert_eq!(state.tap_count, 8);
    assert_eq!(state.room_rt60, [1.2, 1.0, 0.8]);

    // Concurrency test across threads
    let (mut prod, mut cons) = triple_buffer::<u64>();
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    let writer_handle = thread::spawn(move || {
        let mut counter = 1u64;
        while running_clone.load(Ordering::Relaxed) {
            prod.write(counter);
            counter = counter.wrapping_add(1);
        }
    });

    let mut prev_val = 0u64;
    for _ in 0..10_000 {
        let current = *cons.read();
        assert!(
            current >= prev_val,
            "Triple buffer values must be monotonically non-decreasing"
        );
        prev_val = current;
    }

    running.store(false, Ordering::Relaxed);
    writer_handle.join().unwrap();
}
