//! Decoded clips shared between the game and audio threads.

use crate::asset::AudioClip;

/// An immutable decoded clip: interleaved `f32` samples, a channel count and a sample rate.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipData {
    samples: Box<[f32]>,
    channels: u16,
    sample_rate: u32,
}

impl ClipData {
    /// Interleaved samples.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Channels per frame (at least 1).
    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Frames per second.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of frames.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }

    /// Sample `channel` of `frame`; channels beyond the clip's repeat its last one, frames
    /// outside the clip are silent.
    #[inline]
    pub(crate) fn sample(&self, frame: isize, channel: usize) -> f32 {
        if frame < 0 || frame as usize >= self.frames() {
            return 0.0;
        }
        let channels = usize::from(self.channels);
        self.samples[frame as usize * channels + channel.min(channels - 1)]
    }
}

impl From<AudioClip> for ClipData {
    fn from(clip: AudioClip) -> Self {
        let channels = clip.channels.max(1);
        let mut samples = clip.samples;
        // A trailing partial frame is not a frame.
        samples.truncate(samples.len() / usize::from(channels) * usize::from(channels));
        Self {
            samples: samples.into_boxed_slice(),
            channels,
            sample_rate: clip.sample_rate.max(1),
        }
    }
}

/// A clip registered with a mixer ([`super::MixerControl::register_clip`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ClipId(pub(crate) u32);
