//! Opening the audio output (FIX-7.2, ADR-050; finding AUD-01): the default device through
//! `cpal`, or a headless output that runs the mixer in step with game time.
//!
//! [`AudioOutput::open`] creates the mixer for the output's sample rate and channel layout,
//! sized by [`AudioSettings::max_voices`], and returns the game thread's [`AudioVoices`] with it.
//! A device output's mixer runs on the OS audio thread; a headless one renders when
//! [`AudioOutput::advance`] is called, so tests, the editor's headless play mode and machines
//! without a device still hear time pass: voices progress and end, and players finish.
//!
//! Asking for the device where there is none (or in a build without the `cpal` feature) falls
//! back to headless output, unless `KORPPI_REQUIRE_AUDIO=1` is set, which makes it an error.

use crate::device::{
    AudioCallback, AudioCallbackInfo, AudioStreamConfig, BufferSize, ChannelCount,
};
use crate::ecs::AudioSettings;
use crate::error::AudioResult;
use crate::mixer::{AudioMixer, AudioVoices, MixerConfig};
use std::time::Duration;

/// Where the output goes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum AudioBackend {
    /// No device: the mixer renders as the game advances ([`AudioOutput::advance`]).
    #[default]
    Headless,
    /// The default output device, falling back to headless output where there is none.
    Device,
}

/// Frames a headless output renders per mixer call.
const HEADLESS_BLOCK: usize = 256;

enum Sink {
    Headless {
        mixer: Box<AudioMixer>,
        /// The last block rendered, interleaved: its first `last` samples.
        block: Vec<f32>,
        last: usize,
        /// Frames of game time not yet rendered.
        owed: f64,
        rendered: u64,
    },
    #[cfg(feature = "cpal")]
    Device {
        /// Held for the output's life: dropping it closes the device.
        _stream: Box<dyn crate::device::AudioStream>,
        name: String,
        format: &'static str,
    },
}

/// An open audio output: a device stream or a headless renderer (see the module documentation).
pub struct AudioOutput {
    sink: Sink,
    config: AudioStreamConfig,
    /// Why the device was not opened, when it was asked for and the output fell back.
    fallback: Option<String>,
}

/// Whether a missing device is an error (`KORPPI_REQUIRE_AUDIO=1`).
fn audio_required() -> bool {
    std::env::var("KORPPI_REQUIRE_AUDIO").is_ok_and(|value| value == "1")
}

/// The mixer for an output of `config`, sized by `settings`.
fn mixer_config(config: &AudioStreamConfig, settings: &AudioSettings) -> MixerConfig {
    let max_voices = settings.max_voices.max(1);
    MixerConfig {
        max_voices,
        max_virtual_voices: max_voices * 3,
        ..MixerConfig::for_stream(config)
    }
}

impl AudioOutput {
    /// Opens `backend` with a mixer sized by `settings`, and returns the voice API for it.
    ///
    /// # Errors
    /// [`crate::AudioError::DeviceInitialization`] or [`crate::AudioError::StreamCreation`] if the device was
    /// asked for, could not be opened, and `KORPPI_REQUIRE_AUDIO=1` is set.
    pub fn open(
        backend: AudioBackend,
        settings: &AudioSettings,
    ) -> AudioResult<(AudioOutput, AudioVoices)> {
        match backend {
            AudioBackend::Headless => Ok(Self::headless(settings)),
            AudioBackend::Device => match Self::device(settings) {
                Ok(opened) => Ok(opened),
                Err(error) if audio_required() => Err(error),
                Err(error) => {
                    tracing::warn!("no audio output device ({error}); audio runs headless");
                    let (mut output, voices) = Self::headless(settings);
                    output.fallback = Some(error.to_string());
                    Ok((output, voices))
                }
            },
        }
    }

    /// A headless output at 48 kHz stereo.
    fn headless(settings: &AudioSettings) -> (AudioOutput, AudioVoices) {
        let config = AudioStreamConfig {
            sample_rate: 48_000,
            channels: ChannelCount::Stereo,
            buffer_size: BufferSize::Fixed(HEADLESS_BLOCK),
        };
        let (control, mixer) = AudioMixer::new(mixer_config(&config, settings));
        let output = AudioOutput {
            sink: Sink::Headless {
                mixer: Box::new(mixer),
                block: vec![0.0; HEADLESS_BLOCK * config.channels.count()],
                last: 0,
                owed: 0.0,
                rendered: 0,
            },
            config,
            fallback: None,
        };
        (output, AudioVoices::new(control))
    }

    #[cfg(feature = "cpal")]
    fn device(settings: &AudioSettings) -> AudioResult<(AudioOutput, AudioVoices)> {
        use crate::backend::CpalAudioDevice;
        use crate::device::AudioDevice;
        let device = CpalAudioDevice::new_default()?;
        let config = device.default_config();
        let (control, mixer) = AudioMixer::new(mixer_config(&config, settings));
        let name = device.name().to_string();
        let format = device.sample_format_name();
        let mut stream = device.create_output_stream(config, Box::new(mixer))?;
        stream.play()?;
        let output = AudioOutput {
            sink: Sink::Device {
                _stream: stream,
                name,
                format,
            },
            config,
            fallback: None,
        };
        Ok((output, AudioVoices::new(control)))
    }

    #[cfg(not(feature = "cpal"))]
    fn device(_settings: &AudioSettings) -> AudioResult<(AudioOutput, AudioVoices)> {
        Err(crate::error::AudioError::DeviceInitialization(
            "built without the `cpal` feature".to_string(),
        ))
    }

    /// Why the device was not opened, if it was asked for and the output fell back to headless.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback.as_deref()
    }

    /// Whether the output is headless.
    #[must_use]
    pub fn is_headless(&self) -> bool {
        matches!(self.sink, Sink::Headless { .. })
    }

    /// The output's sample rate, channels and buffer size; the mixer runs at its rate.
    #[must_use]
    pub fn config(&self) -> AudioStreamConfig {
        self.config
    }

    /// The device's name, or "headless".
    #[must_use]
    pub fn device_name(&self) -> &str {
        match &self.sink {
            Sink::Headless { .. } => "headless",
            #[cfg(feature = "cpal")]
            Sink::Device { name, .. } => name,
        }
    }

    /// The device's sample format ("f32", "i16" or "u16"; converted from the mixer's `f32` in
    /// the callback), or "f32" headless.
    #[must_use]
    pub fn sample_format(&self) -> &'static str {
        match &self.sink {
            Sink::Headless { .. } => "f32",
            #[cfg(feature = "cpal")]
            Sink::Device { format, .. } => format,
        }
    }

    /// Advances a headless output by `delta` of game time, rendering that many frames; a device
    /// output renders on its own and ignores it. Nothing allocates.
    // Without the `cpal` feature, headless is the only sink.
    #[allow(irrefutable_let_patterns)]
    pub fn advance(&mut self, delta: Duration) {
        let rate = f64::from(self.config.sample_rate);
        let channels = self.config.channels;
        if let Sink::Headless {
            mixer,
            block,
            last,
            owed,
            rendered,
        } = &mut self.sink
        {
            *owed += delta.as_secs_f64() * rate;
            while *owed >= 1.0 {
                let frames = (owed.floor() as usize).min(HEADLESS_BLOCK);
                let samples = &mut block[..frames * channels.count()];
                let info = AudioCallbackInfo {
                    sample_rate: self.config.sample_rate,
                    channels,
                    frame_count: frames,
                    frames_rendered: *rendered,
                };
                mixer.render(samples, &info);
                *last = samples.len();
                *rendered += frames as u64;
                *owed -= frames as f64;
            }
        }
    }

    /// The last block a headless output rendered, interleaved (empty for a device).
    #[must_use]
    pub fn last_output(&self) -> &[f32] {
        match &self.sink {
            Sink::Headless { block, last, .. } => &block[..*last],
            #[cfg(feature = "cpal")]
            Sink::Device { .. } => &[],
        }
    }
}
