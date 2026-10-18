//! The voice API for gameplay systems (ADR-050).

use super::clip::{ClipData, ClipId};
use super::control::{At, AudioClock, Fade, MixerControl, VoiceHandle, VoiceParams, VoiceSpatial};
use crate::asset::AudioClip;
use crate::error::AudioResult;
use korppi_asset::{AssetHandle, AssetId, AssetServer};
use std::collections::HashMap;
use std::sync::Arc;

/// A clip registered with the mixer, and what scheduling needs of it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RegisteredClip {
    /// The mixer's ID for the clip.
    pub id: ClipId,
    /// Frames in the clip.
    pub frames: usize,
    /// The clip's sample rate.
    pub sample_rate: u32,
}

/// Single mixer voices for custom systems, such as adaptive music or granular effects: a world
/// resource wrapping the game thread's [`MixerControl`].
///
/// Voices start and stop on output frames of the [`AudioClock`], fade in and out, and ramp their
/// gain and pitch, all to the sample; [`VoiceHandle`]s are generational. The mixer mixes the
/// voices of highest priority and audibility, keeps the others virtual, and a full pool
/// replaces its lowest-priority voice (see [`MixerControl::start_with`]).
pub struct AudioVoices {
    control: MixerControl,
    /// Clips from the asset server registered with the mixer, by asset.
    clips: HashMap<AssetId, RegisteredClip>,
}

impl AudioVoices {
    /// The voice API over `control`.
    #[must_use]
    pub fn new(control: MixerControl) -> Self {
        Self {
            control,
            clips: HashMap::new(),
        }
    }

    /// The mixer's clip for the asset `handle` names, registered the first time it is asked
    /// for (its samples are copied once); `None` while the asset is not loaded.
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if every clip slot is in use or the command queue is
    /// full.
    pub fn clip(
        &mut self,
        assets: &AssetServer,
        handle: &AssetHandle<AudioClip>,
    ) -> AudioResult<Option<RegisteredClip>> {
        if let Some(&clip) = self.clips.get(&handle.id()) {
            return Ok(Some(clip));
        }
        let Some(asset) = assets.get(handle) else {
            return Ok(None);
        };
        let data = Arc::new(ClipData::from(AudioClip::clone(&asset)));
        let (frames, sample_rate) = (data.frames(), data.sample_rate());
        let clip = RegisteredClip {
            id: self.control.register_clip(data)?,
            frames,
            sample_rate,
        };
        self.clips.insert(handle.id(), clip);
        Ok(Some(clip))
    }

    /// The mixer's frame clock, for scheduling.
    #[must_use]
    pub fn clock(&self) -> &AudioClock {
        self.control.clock()
    }

    /// Starts a voice playing `clip` at full gain at the start of the mixer's next block.
    ///
    /// # Errors
    /// As [`MixerControl::start_with`].
    pub fn start(&mut self, clip: ClipId, params: VoiceParams) -> AudioResult<VoiceHandle> {
        self.control.start(clip, params)
    }

    /// Starts a voice playing `clip` at `at`, fading in over `fade_in`.
    ///
    /// # Errors
    /// As [`MixerControl::start_with`].
    pub fn start_with(
        &mut self,
        clip: ClipId,
        params: VoiceParams,
        at: At,
        fade_in: Fade,
    ) -> AudioResult<VoiceHandle> {
        self.control.start_with(clip, params, at, fade_in)
    }

    /// Stops `voice` at the start of the mixer's next block, fading out over 5 ms.
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn stop(&mut self, voice: VoiceHandle) -> AudioResult<()> {
        self.control.stop(voice)
    }

    /// Stops `voice` at `at`, fading out over `fade_out`.
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn stop_with(&mut self, voice: VoiceHandle, at: At, fade_out: Fade) -> AudioResult<()> {
        self.control.stop_with(voice, at, fade_out)
    }

    /// Ramps `voice`'s gain to `gain` over `frames` output frames (at least 5 ms).
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn set_gain(&mut self, voice: VoiceHandle, gain: f32, frames: u32) -> AudioResult<()> {
        self.control.set_gain(voice, gain, frames)
    }

    /// Ramps `voice`'s pitch to `pitch` over `frames` output frames (at least 5 ms).
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn set_pitch(&mut self, voice: VoiceHandle, pitch: f32, frames: u32) -> AudioResult<()> {
        self.control.set_pitch(voice, pitch, frames)
    }

    /// Ramps `voice`'s stereo gains and filter cutoff over `frames` output frames.
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn set_spatial(
        &mut self,
        voice: VoiceHandle,
        spatial: VoiceSpatial,
        frames: u32,
    ) -> AudioResult<()> {
        self.control.set_spatial(voice, spatial, frames)
    }

    /// Ramps `voice`'s send to `bus` to `level` over `frames` output frames.
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn set_send(
        &mut self,
        voice: VoiceHandle,
        bus: super::BusId,
        level: f32,
        frames: u32,
    ) -> AudioResult<()> {
        self.control.set_send(voice, bus, level, frames)
    }

    /// Gives `voice` early reflections ([`super::MixerControl::set_reflections`]).
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn set_reflections(
        &mut self,
        voice: VoiceHandle,
        reflections: super::Reflections,
        frames: u32,
    ) -> AudioResult<()> {
        self.control.set_reflections(voice, reflections, frames)
    }

    /// Pauses `voice`, fading out over 5 ms; it holds its place.
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn pause(&mut self, voice: VoiceHandle) -> AudioResult<()> {
        self.control.pause(voice)
    }

    /// Resumes a paused `voice`, fading in over 5 ms.
    ///
    /// # Errors
    /// [`crate::AudioError::QueueFull`] if the command queue is full.
    pub fn resume(&mut self, voice: VoiceHandle) -> AudioResult<()> {
        self.control.resume(voice)
    }

    /// Whether `voice` still plays (a paused voice does).
    #[must_use]
    pub fn is_playing(&self, voice: VoiceHandle) -> bool {
        self.control.is_playing(voice)
    }

    /// `voice`'s position in its clip, in seconds; `None` once it has ended.
    #[must_use]
    pub fn position(&self, voice: VoiceHandle) -> Option<f64> {
        self.control.position(voice)
    }

    /// The output frame `voice` first sounded on, once the mixer has rendered it.
    #[must_use]
    pub fn started_at(&self, voice: VoiceHandle) -> Option<u64> {
        self.control.started_at(voice)
    }

    /// The output stream's sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.control.sample_rate()
    }

    /// The control, for registering clips and adding buses.
    pub fn control(&mut self) -> &mut MixerControl {
        &mut self.control
    }

    /// Frees ended voices' slots and drops released clips; the engine calls it every tick.
    pub fn update(&mut self) {
        self.control.update();
    }
}
