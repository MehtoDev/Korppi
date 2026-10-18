//! The hardware output through `cpal` (FIX-7.2, ADR-050).
//!
//! [`CpalAudioDevice::new_default`] opens the default output device and negotiates its stream:
//! the device's default sample rate (the mixer runs at it), a channel layout the mixer writes
//! (mono, stereo, 5.1 or 7.1), and a sample format among `f32`, `i16` and `u16`, preferring
//! the device's default. The callback converts the mixer's `f32` into the device's format through
//! a buffer allocated before the stream starts.
//!
//! `cpal` streams are not `Send` (some hosts, such as Android's AAudio, have stream APIs that
//! are not thread-safe), while [`AudioStream`] is. Each stream therefore lives on a control thread
//! of its own, which builds it, plays and pauses it on request, and drops it; the handle only
//! holds channels to that thread, so it moves between threads without `unsafe`.

use crate::device::{
    AudioCallback, AudioCallbackInfo, AudioDevice, AudioStream, AudioStreamConfig, BufferSize,
    ChannelCount,
};
use crate::error::{AudioError, AudioResult};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Frames converted at a time for a device whose format is not `f32`.
const CONVERSION_FRAMES: usize = 4096;

/// The default output device and the stream format negotiated for it.
pub struct CpalAudioDevice {
    device: cpal::Device,
    name: String,
    channels: ChannelCount,
    sample_rate: u32,
    format: SampleFormat,
}

/// A channel count the mixer writes.
fn channel_count(channels: u16) -> Option<ChannelCount> {
    match channels {
        1 => Some(ChannelCount::Mono),
        2 => Some(ChannelCount::Stereo),
        6 => Some(ChannelCount::Surround51),
        8 => Some(ChannelCount::Surround71),
        _ => None,
    }
}

fn is_convertible(format: SampleFormat) -> bool {
    matches!(
        format,
        SampleFormat::F32 | SampleFormat::I16 | SampleFormat::U16
    )
}

impl CpalAudioDevice {
    /// Opens the host's default output device and negotiates its stream format.
    ///
    /// # Errors
    /// [`AudioError::DeviceInitialization`] if there is no default output device, or it offers
    /// no format the mixer can write.
    pub fn new_default() -> AudioResult<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or_else(|| {
            AudioError::DeviceInitialization("no default audio output device".into())
        })?;
        let name = device
            .name()
            .unwrap_or_else(|_| "unnamed audio device".to_string());
        let default = device
            .default_output_config()
            .map_err(|error| AudioError::DeviceInitialization(error.to_string()))?;
        let rate = default.sample_rate();
        if let (Some(channels), true) = (
            channel_count(default.channels()),
            is_convertible(default.sample_format()),
        ) {
            return Ok(Self {
                device,
                name,
                channels,
                sample_rate: rate.0,
                format: default.sample_format(),
            });
        }
        // Otherwise a stereo (or any writable) layout in a writable format at the default rate,
        // preferring f32.
        let mut best: Option<(u16, SampleFormat)> = None;
        let supported = device
            .supported_output_configs()
            .map_err(|error| AudioError::DeviceInitialization(error.to_string()))?;
        for range in supported {
            if range.min_sample_rate() > rate || range.max_sample_rate() < rate {
                continue;
            }
            let (channels, format) = (range.channels(), range.sample_format());
            if channel_count(channels).is_none() || !is_convertible(format) {
                continue;
            }
            let rank = |(channels, format): (u16, SampleFormat)| {
                (channels == 2, format == SampleFormat::F32)
            };
            if best.is_none_or(|current| rank((channels, format)) > rank(current)) {
                best = Some((channels, format));
            }
        }
        let (channels, format) = best.ok_or_else(|| {
            AudioError::DeviceInitialization(format!(
                "{name} offers no format the mixer writes (default {} channels, {:?})",
                default.channels(),
                default.sample_format()
            ))
        })?;
        Ok(Self {
            device,
            name,
            channels: channel_count(channels).unwrap_or(ChannelCount::Stereo),
            sample_rate: rate.0,
            format,
        })
    }

    /// The negotiated sample format's name.
    #[must_use]
    pub fn sample_format_name(&self) -> &'static str {
        match self.format {
            SampleFormat::F32 => "f32",
            SampleFormat::I16 => "i16",
            SampleFormat::U16 => "u16",
            _ => "unsupported",
        }
    }
}

impl AudioDevice for CpalAudioDevice {
    fn name(&self) -> &str {
        &self.name
    }

    fn default_config(&self) -> AudioStreamConfig {
        AudioStreamConfig {
            sample_rate: self.sample_rate,
            channels: self.channels,
            buffer_size: BufferSize::Default,
        }
    }

    fn supported_configs(&self) -> Vec<AudioStreamConfig> {
        vec![self.default_config()]
    }

    fn create_output_stream(
        &self,
        config: AudioStreamConfig,
        callback: Box<dyn AudioCallback>,
    ) -> AudioResult<Box<dyn AudioStream>> {
        let is_playing = Arc::new(AtomicBool::new(false));
        let frames_rendered = Arc::new(AtomicU64::new(0));
        let (control, requests) = sync_channel::<Control>(1);
        let (reply_to, replies) = sync_channel::<AudioResult<()>>(1);
        let stream = StreamSetup {
            device: self.device.clone(),
            config,
            format: self.format,
            callback,
            is_playing: Arc::clone(&is_playing),
            frames_rendered: Arc::clone(&frames_rendered),
        };
        let thread = std::thread::Builder::new()
            .name("korppi-audio-device".into())
            .spawn(move || stream.run(&requests, &reply_to))
            .map_err(|error| AudioError::StreamCreation(error.to_string()))?;
        let built = replies.recv().unwrap_or_else(|_| {
            Err(AudioError::StreamCreation(
                "the audio device thread ended".into(),
            ))
        });
        let handle = CpalAudioStream {
            control,
            replies,
            thread: Some(thread),
            config,
            is_playing,
            frames_rendered,
        };
        built.map(|()| Box::new(handle) as Box<dyn AudioStream>)
    }
}

/// A request to a stream's control thread.
enum Control {
    Play,
    Pause,
}

/// What a control thread needs to build and drive its stream.
struct StreamSetup {
    device: cpal::Device,
    config: AudioStreamConfig,
    format: SampleFormat,
    callback: Box<dyn AudioCallback>,
    is_playing: Arc<AtomicBool>,
    frames_rendered: Arc<AtomicU64>,
}

impl StreamSetup {
    /// The control thread: builds the stream, reports whether that worked, then plays and
    /// pauses it until the handle is dropped.
    fn run(self, requests: &Receiver<Control>, reply_to: &SyncSender<AudioResult<()>>) {
        let stream = match self.build() {
            Ok(stream) => {
                let _ = reply_to.send(Ok(()));
                stream
            }
            Err(error) => {
                let _ = reply_to.send(Err(error));
                return;
            }
        };
        // The handle's drop closes the channel, which ends the loop and drops the stream here.
        while let Ok(request) = requests.recv() {
            let result = match request {
                Control::Play => stream.play().map_err(|error| error.to_string()),
                Control::Pause => stream.pause().map_err(|error| error.to_string()),
            };
            let _ = reply_to.send(result.map_err(AudioError::BackendError));
        }
    }

    fn build(self) -> AudioResult<cpal::Stream> {
        let StreamSetup {
            device,
            config,
            format,
            mut callback,
            is_playing,
            frames_rendered,
        } = self;
        let channels = config.channels;
        let stream_config = cpal::StreamConfig {
            channels: channels.count() as u16,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: match config.buffer_size {
                BufferSize::Default => cpal::BufferSize::Default,
                BufferSize::Fixed(frames) => cpal::BufferSize::Fixed(frames as u32),
            },
        };
        let on_error = |error| tracing::error!("audio stream error: {error:?}");
        // Renders `frames` frames of the mixer into `samples`.
        let mut render = move |samples: &mut [f32]| {
            if !is_playing.load(Ordering::Relaxed) {
                samples.fill(0.0);
                return;
            }
            let frame_count = samples.len() / channels.count();
            let info = AudioCallbackInfo {
                sample_rate: config.sample_rate,
                channels,
                frame_count,
                frames_rendered: frames_rendered.load(Ordering::Relaxed),
            };
            callback.render(samples, &info);
            frames_rendered.fetch_add(frame_count as u64, Ordering::Relaxed);
        };
        let chunk = CONVERSION_FRAMES * channels.count();
        let stream = match format {
            SampleFormat::F32 => device.build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| render(data),
                on_error,
                None,
            ),
            SampleFormat::I16 => {
                let mut scratch = vec![0.0f32; chunk];
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        for out in data.chunks_mut(chunk) {
                            let mixed = &mut scratch[..out.len()];
                            render(mixed);
                            for (out, &sample) in out.iter_mut().zip(mixed.iter()) {
                                *out = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                            }
                        }
                    },
                    on_error,
                    None,
                )
            }
            SampleFormat::U16 => {
                let mut scratch = vec![0.0f32; chunk];
                device.build_output_stream(
                    &stream_config,
                    move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                        for out in data.chunks_mut(chunk) {
                            let mixed = &mut scratch[..out.len()];
                            render(mixed);
                            for (out, &sample) in out.iter_mut().zip(mixed.iter()) {
                                let centred = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i32;
                                *out = (centred + 32768) as u16;
                            }
                        }
                    },
                    on_error,
                    None,
                )
            }
            other => {
                return Err(AudioError::StreamCreation(format!(
                    "unsupported sample format {other:?}"
                )))
            }
        };
        stream.map_err(|error| AudioError::StreamCreation(error.to_string()))
    }
}

/// A playing `cpal` stream, driven through its control thread (see the module documentation).
pub struct CpalAudioStream {
    control: SyncSender<Control>,
    replies: Receiver<AudioResult<()>>,
    thread: Option<JoinHandle<()>>,
    config: AudioStreamConfig,
    is_playing: Arc<AtomicBool>,
    frames_rendered: Arc<AtomicU64>,
}

impl CpalAudioStream {
    fn request(&mut self, request: Control) -> AudioResult<()> {
        let gone = || AudioError::BackendError("the audio device thread ended".into());
        self.control.send(request).map_err(|_| gone())?;
        self.replies.recv().map_err(|_| gone())?
    }
}

impl AudioStream for CpalAudioStream {
    fn play(&mut self) -> AudioResult<()> {
        self.request(Control::Play)?;
        self.is_playing.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn pause(&mut self) -> AudioResult<()> {
        self.request(Control::Pause)?;
        self.is_playing.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::SeqCst)
    }

    fn config(&self) -> AudioStreamConfig {
        self.config
    }

    fn frames_rendered(&self) -> u64 {
        self.frames_rendered.load(Ordering::Relaxed)
    }

    fn render_frames(&mut self, _frames: usize) -> Vec<f32> {
        // The host pulls a device stream on its own audio thread.
        Vec::new()
    }
}

impl Drop for CpalAudioStream {
    fn drop(&mut self) {
        // Closing the channel ends the control thread, which drops the stream.
        let (closed, _) = sync_channel(1);
        drop(std::mem::replace(&mut self.control, closed));
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
