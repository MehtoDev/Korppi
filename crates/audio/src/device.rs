//! Audio device abstraction, stream configuration, and headless testing driver.

use crate::error::AudioResult;
use serde::{Deserialize, Serialize};

/// Number of discrete audio channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ChannelCount {
    /// Single-channel monaural audio.
    Mono = 1,
    /// Dual-channel stereo audio (Default).
    #[default]
    Stereo = 2,
    /// 5.1 channel surround sound (Front L/R, Center, LFE, Surround L/R).
    Surround51 = 6,
    /// 7.1 channel surround sound (Front L/R, Center, LFE, Side L/R, Rear L/R).
    Surround71 = 8,
}

impl ChannelCount {
    /// Returns the integer count of audio channels.
    #[must_use]
    pub const fn count(self) -> usize {
        self as usize
    }
}

/// Buffer size specification for audio streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BufferSize {
    /// Platform default buffer size (typically 256 or 512 frames).
    #[default]
    Default,
    /// Explicit fixed buffer size in frames.
    Fixed(usize),
}

impl BufferSize {
    /// Returns the effective buffer size in frames (defaulting to 256 frames if unspecified).
    #[must_use]
    pub const fn frame_count(self) -> usize {
        match self {
            Self::Default => 256,
            Self::Fixed(frames) => frames,
        }
    }
}

/// Configuration settings for opening an audio output stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioStreamConfig {
    /// Sample rate in Hertz (e.g., 44100, 48000, 96000). Default is 48000 Hz.
    pub sample_rate: u32,
    /// Channel layout. Default is Stereo.
    pub channels: ChannelCount,
    /// Buffer frame count.
    pub buffer_size: BufferSize,
}

impl Default for AudioStreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: ChannelCount::Stereo,
            buffer_size: BufferSize::Default,
        }
    }
}

/// Information passed to the real-time audio callback during render calls.
#[derive(Debug, Clone, Copy)]
pub struct AudioCallbackInfo {
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Active channel layout.
    pub channels: ChannelCount,
    /// Number of audio frames to render in this block.
    pub frame_count: usize,
    /// Monotonically increasing count of total frames rendered so far.
    pub frames_rendered: u64,
}

/// Trait implemented by real-time audio renderers to fill hardware buffers.
pub trait AudioCallback: Send + 'static {
    /// Renders interleaved PCM audio samples into `output`.
    ///
    /// The length of `output` is guaranteed to be `info.frame_count * info.channels.count()`.
    /// This method executes within a high-priority real-time audio callback;
    /// it MUST NOT allocate heap memory or perform blocking synchronization.
    fn render(&mut self, output: &mut [f32], info: &AudioCallbackInfo);
}

/// Handle representing an active or paused audio output stream.
pub trait AudioStream: Send {
    /// Begins or resumes audio playback.
    fn play(&mut self) -> AudioResult<()>;

    /// Pauses audio playback.
    fn pause(&mut self) -> AudioResult<()>;

    /// Returns `true` if the stream is currently playing.
    fn is_playing(&self) -> bool;

    /// Returns the stream's configuration parameters.
    fn config(&self) -> AudioStreamConfig;

    /// Returns the total number of audio frames rendered by this stream.
    fn frames_rendered(&self) -> u64;

    /// Manually triggers rendering of `frames` audio frames, returning the interleaved PCM output.
    ///
    /// Primarily used for headless CI verification and offscreen audio simulation.
    fn render_frames(&mut self, frames: usize) -> Vec<f32>;
}

/// Abstract audio hardware device interface.
pub trait AudioDevice: Send + Sync {
    /// Returns the human-readable display name of the audio device.
    fn name(&self) -> &str;

    /// Returns the device's preferred default stream configuration.
    fn default_config(&self) -> AudioStreamConfig;

    /// Returns the list of stream configurations supported by the device.
    fn supported_configs(&self) -> Vec<AudioStreamConfig>;

    /// Opens an output stream with the specified configuration and render callback.
    fn create_output_stream(
        &self,
        config: AudioStreamConfig,
        callback: Box<dyn AudioCallback>,
    ) -> AudioResult<Box<dyn AudioStream>>;
}

/// Headless dummy audio device for CI testing and deterministic offscreen execution.
#[derive(Debug, Default, Clone)]
pub struct HeadlessDummyDevice {
    recorded_frames: Vec<crate::frame::AudioFrame>,
}

impl HeadlessDummyDevice {
    /// Creates a new headless dummy audio device.
    #[must_use]
    pub fn new() -> Self {
        Self {
            recorded_frames: Vec::new(),
        }
    }

    /// Appends audio frames into the dummy device's recording buffer.
    pub fn push_frames(&mut self, frames: &[crate::frame::AudioFrame]) {
        self.recorded_frames.extend_from_slice(frames);
    }

    /// Returns the total number of frames rendered/pushed into this device.
    #[must_use]
    pub fn rendered_frame_count(&self) -> usize {
        self.recorded_frames.len()
    }

    /// Returns the slice of recorded audio frames.
    #[must_use]
    pub fn frames(&self) -> &[crate::frame::AudioFrame] {
        &self.recorded_frames
    }
}

impl AudioDevice for HeadlessDummyDevice {
    fn name(&self) -> &str {
        "Headless Dummy Audio Device"
    }

    fn default_config(&self) -> AudioStreamConfig {
        AudioStreamConfig::default()
    }

    fn supported_configs(&self) -> Vec<AudioStreamConfig> {
        vec![
            AudioStreamConfig {
                sample_rate: 44_100,
                channels: ChannelCount::Stereo,
                buffer_size: BufferSize::Fixed(128),
            },
            AudioStreamConfig {
                sample_rate: 48_000,
                channels: ChannelCount::Stereo,
                buffer_size: BufferSize::Fixed(128),
            },
            AudioStreamConfig {
                sample_rate: 48_000,
                channels: ChannelCount::Surround51,
                buffer_size: BufferSize::Fixed(256),
            },
        ]
    }

    fn create_output_stream(
        &self,
        config: AudioStreamConfig,
        callback: Box<dyn AudioCallback>,
    ) -> AudioResult<Box<dyn AudioStream>> {
        Ok(Box::new(HeadlessDummyStream::new(config, callback)))
    }
}

/// Headless stream implementation that executes rendering synchronously in memory.
pub struct HeadlessDummyStream {
    config: AudioStreamConfig,
    callback: Box<dyn AudioCallback>,
    is_playing: bool,
    frames_rendered: u64,
}

impl HeadlessDummyStream {
    /// Creates a new headless audio stream.
    #[must_use]
    pub fn new(config: AudioStreamConfig, callback: Box<dyn AudioCallback>) -> Self {
        Self {
            config,
            callback,
            is_playing: false,
            frames_rendered: 0,
        }
    }
}

impl AudioStream for HeadlessDummyStream {
    fn play(&mut self) -> AudioResult<()> {
        self.is_playing = true;
        Ok(())
    }

    fn pause(&mut self) -> AudioResult<()> {
        self.is_playing = false;
        Ok(())
    }

    fn is_playing(&self) -> bool {
        self.is_playing
    }

    fn config(&self) -> AudioStreamConfig {
        self.config
    }

    fn frames_rendered(&self) -> u64 {
        self.frames_rendered
    }

    fn render_frames(&mut self, frames: usize) -> Vec<f32> {
        let total_samples = frames * self.config.channels.count();
        let mut buffer = vec![0.0f32; total_samples];

        if self.is_playing {
            let info = AudioCallbackInfo {
                sample_rate: self.config.sample_rate,
                channels: self.config.channels,
                frame_count: frames,
                frames_rendered: self.frames_rendered,
            };
            self.callback.render(&mut buffer, &info);
            self.frames_rendered += frames as u64;
        }

        buffer
    }
}
