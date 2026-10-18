//! Unit and integration tests for Sub-Milestone 10.1:
//! Audio Device Abstraction, Stream Output & Wait-Free Ring Buffers. The command queue's test went
//! with it in FIX-7.5 (AUD-10, the owner's decision): the mixer's commands replaced it.

use korppi_audio::device::{
    AudioCallback, AudioCallbackInfo, AudioDevice, AudioStreamConfig, BufferSize, ChannelCount,
    HeadlessDummyDevice,
};
use korppi_audio::frame::AudioFrame;
use korppi_audio::ring_buffer::SpscRingBuffer;
use std::thread;

/// Simple sine-wave generator callback for testing audio streams.
struct SineWaveCallback {
    phase: f32,
    frequency: f32,
    sample_rate: f32,
    amplitude: f32,
}

impl SineWaveCallback {
    fn new(frequency: f32, sample_rate: f32, amplitude: f32) -> Self {
        Self {
            phase: 0.0,
            frequency,
            sample_rate,
            amplitude,
        }
    }
}

impl AudioCallback for SineWaveCallback {
    fn render(&mut self, output: &mut [f32], info: &AudioCallbackInfo) {
        let phase_inc = 2.0 * std::f32::consts::PI * self.frequency / self.sample_rate;
        let channels = info.channels as usize;

        for frame in output.chunks_exact_mut(channels) {
            let sample = self.phase.sin() * self.amplitude;
            for ch in frame.iter_mut() {
                *ch = sample;
            }
            self.phase = (self.phase + phase_inc) % (2.0 * std::f32::consts::PI);
        }
    }
}

#[test]
fn test_audio_frame_math_and_conversions() {
    let silent = AudioFrame::SILENT;
    assert_eq!(silent.left, 0.0);
    assert_eq!(silent.right, 0.0);

    let mono = AudioFrame::from_mono(0.5);
    assert_eq!(mono.left, 0.5);
    assert_eq!(mono.right, 0.5);

    let frame_a = AudioFrame::new(0.2, 0.8);
    let frame_b = AudioFrame::new(0.3, 0.1);
    let sum = frame_a + frame_b;
    assert!((sum.left - 0.5).abs() < 1e-6);
    assert!((sum.right - 0.9).abs() < 1e-6);

    let scaled = frame_a * 2.0;
    assert!((scaled.left - 0.4).abs() < 1e-6);
    assert!((scaled.right - 1.6).abs() < 1e-6);

    // Test equal-power panning
    let center = AudioFrame::from_mono(1.0).pan(0.0);
    // At center, each channel is scaled by sqrt(0.5) approx 0.7071
    assert!((center.left - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
    assert!((center.right - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
    assert!(((center.left * center.left + center.right * center.right) - 1.0).abs() < 1e-4);

    let hard_left = AudioFrame::from_mono(1.0).pan(-1.0);
    assert!((hard_left.left - 1.0).abs() < 1e-5);
    assert!(hard_left.right.abs() < 1e-5);

    let hard_right = AudioFrame::from_mono(1.0).pan(1.0);
    assert!(hard_right.left.abs() < 1e-5);
    assert!((hard_right.right - 1.0).abs() < 1e-5);
}

#[test]
fn test_stream_config_and_channel_counts() {
    let default_config = AudioStreamConfig::default();
    assert_eq!(default_config.sample_rate, 48_000);
    assert_eq!(default_config.channels, ChannelCount::Stereo);
    assert_eq!(default_config.buffer_size, BufferSize::Default);
    assert_eq!(default_config.channels.count(), 2);

    assert_eq!(ChannelCount::Mono.count(), 1);
    assert_eq!(ChannelCount::Stereo.count(), 2);
    assert_eq!(ChannelCount::Surround51.count(), 6);
    assert_eq!(ChannelCount::Surround71.count(), 8);

    let custom_config = AudioStreamConfig {
        sample_rate: 44_100,
        channels: ChannelCount::Mono,
        buffer_size: BufferSize::Fixed(256),
    };
    assert_eq!(custom_config.sample_rate, 44_100);
    assert_eq!(custom_config.channels.count(), 1);
    assert_eq!(custom_config.buffer_size.frame_count(), 256);
}

#[test]
fn test_spsc_ring_buffer_fifo_ordering_and_saturation() {
    let (mut producer, mut consumer) = SpscRingBuffer::<u32>::new(4);
    assert_eq!(producer.capacity(), 4);
    assert!(consumer.is_empty());
    assert_eq!(consumer.len(), 0);

    // Push 4 items (fills capacity)
    assert!(producer.push(10).is_ok());
    assert!(producer.push(20).is_ok());
    assert!(producer.push(30).is_ok());
    assert!(producer.push(40).is_ok());

    // 5th push must fail (wait-free, zero allocation)
    assert!(producer.push(50).is_err());
    assert_eq!(consumer.len(), 4);
    assert!(!consumer.is_empty());

    // Read back in FIFO order
    assert_eq!(consumer.pop(), Some(10));
    assert_eq!(consumer.pop(), Some(20));
    assert_eq!(consumer.len(), 2);

    // Now push 2 more items
    assert!(producer.push(50).is_ok());
    assert!(producer.push(60).is_ok());
    assert!(producer.push(70).is_err()); // Full again

    assert_eq!(consumer.pop(), Some(30));
    assert_eq!(consumer.pop(), Some(40));
    assert_eq!(consumer.pop(), Some(50));
    assert_eq!(consumer.pop(), Some(60));
    assert_eq!(consumer.pop(), None);
    assert!(consumer.is_empty());
}

#[test]
fn test_spsc_ring_buffer_concurrency_across_threads() {
    let (mut producer, mut consumer) = SpscRingBuffer::<usize>::new(128);
    const TOTAL_ITEMS: usize = 50_000;

    let producer_handle = thread::spawn(move || {
        for i in 0..TOTAL_ITEMS {
            while producer.push(i).is_err() {
                // Yield thread if full
                std::hint::spin_loop();
            }
        }
    });

    let consumer_handle = thread::spawn(move || {
        let mut count = 0;
        let mut next_expected = 0;
        while count < TOTAL_ITEMS {
            if let Some(item) = consumer.pop() {
                assert_eq!(item, next_expected, "FIFO order must be strictly preserved");
                next_expected += 1;
                count += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        count
    });

    producer_handle.join().unwrap();
    let total_consumed = consumer_handle.join().unwrap();
    assert_eq!(total_consumed, TOTAL_ITEMS);
}

#[test]
fn test_headless_dummy_device_stream_lifecycle() {
    let device = HeadlessDummyDevice::new();
    assert_eq!(device.name(), "Headless Dummy Audio Device");

    let config = AudioStreamConfig {
        sample_rate: 48_000,
        channels: ChannelCount::Stereo,
        buffer_size: BufferSize::Fixed(128),
    };

    let callback = SineWaveCallback::new(440.0, 48_000.0, 0.8);
    let mut stream = device
        .create_output_stream(config, Box::new(callback))
        .unwrap();

    assert_eq!(stream.config().sample_rate, 48_000);
    assert_eq!(stream.config().channels, ChannelCount::Stereo);
    assert_eq!(stream.frames_rendered(), 0);
    assert!(!stream.is_playing());

    // Start playback
    stream.play().unwrap();
    assert!(stream.is_playing());

    // Step forward 256 frames (2 buffers of 128 frames)
    let rendered_pcm = stream.render_frames(256);
    assert_eq!(rendered_pcm.len(), 256 * 2); // 256 stereo frames = 512 samples
    assert_eq!(stream.frames_rendered(), 256);

    // Verify non-silent samples
    let max_sample = rendered_pcm.iter().copied().fold(0.0f32, f32::max);
    assert!(
        max_sample > 0.5,
        "Rendered sine wave should have significant amplitude"
    );

    // Pause playback
    stream.pause().unwrap();
    assert!(!stream.is_playing());

    // Stepping while paused outputs silence (zeros)
    let paused_pcm = stream.render_frames(128);
    assert_eq!(paused_pcm.len(), 256);
    assert!(
        paused_pcm.iter().all(|&s| s == 0.0),
        "Paused stream must render absolute silence"
    );
}
