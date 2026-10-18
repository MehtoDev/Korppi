//! The game thread's side of the mixer.

use super::clip::{ClipData, ClipId};
use super::effects::{BusEffect, RoomReverb};
use super::engine::MixerConfig;
use super::message::{MixerCommand, MixerEvent, VoiceStates};
use crate::error::{AudioError, AudioResult};
use crate::ring_buffer::{Consumer, Producer};
use korppi_core::{Handle, IdPool};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The slots of a mixer's voice pool, for [`VoiceHandle`]s.
pub(crate) struct VoiceSlot;

/// A voice started through [`MixerControl::start`]: a generational handle to a slot of the
/// mixer's voice pool. When the voice ends, its slot's generation advances, so the handle names
/// nothing even after the slot holds another voice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VoiceHandle(pub(crate) Handle<VoiceSlot>);

impl VoiceHandle {
    pub(crate) fn slot(self) -> usize {
        self.0.index() as usize
    }

    pub(crate) fn generation(self) -> u32 {
        self.0.generation().get()
    }
}

/// A mixing bus of a mixer: [`MASTER`], or one added with [`MixerControl::add_bus`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BusId(pub(crate) u16);

/// The bus every mixer starts with, named "Master"; it feeds the device.
pub const MASTER: BusId = BusId(0);

/// The priority of a voice started with [`VoiceParams::default`].
pub const DEFAULT_PRIORITY: u8 = 128;

/// How a voice plays.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VoiceParams {
    /// Linear gain.
    pub gain: f32,
    /// Playback rate relative to the clip's own (2.0 is an octave up).
    pub pitch: f32,
    /// Whether the clip repeats from its start when it ends.
    pub looping: bool,
    /// The bus the voice mixes into.
    pub bus: BusId,
    /// Higher is more important: the mixer mixes voices by priority first and audibility second,
    /// and a full pool replaces its lowest-priority voice.
    pub priority: u8,
    /// Where a looping voice returns to, in clip frames: the frames before it are an intro,
    /// played once.
    pub loop_start: u32,
    /// Where a looping voice returns from, in clip frames; `None` is the clip's end.
    pub loop_end: Option<u32>,
}

impl Default for VoiceParams {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pitch: 1.0,
            looping: false,
            bus: MASTER,
            priority: DEFAULT_PRIORITY,
            loop_start: 0,
            loop_end: None,
        }
    }
}

/// Where a voice sits between the left and right speakers, and how muffled it sounds: what
/// spatialization and occlusion set.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VoiceSpatial {
    /// Linear gain of the left channel.
    pub gain_left: f32,
    /// Linear gain of the right channel.
    pub gain_right: f32,
    /// Cutoff of the voice's low-pass filter in Hz; [`VoiceSpatial::OPEN`] and above leave the
    /// voice unfiltered.
    pub cutoff_hz: f32,
    /// How much later the left ear hears the voice, in seconds (the interaural time difference;
    /// 0 for the near ear).
    pub delay_left: f32,
    /// How much later the right ear hears the voice, in seconds.
    pub delay_right: f32,
    /// The head's shadow on the left ear: a further low-pass cutoff in Hz for that ear alone
    /// ([`VoiceSpatial::OPEN`] for none).
    pub shadow_left_hz: f32,
    /// The head's shadow on the right ear.
    pub shadow_right_hz: f32,
}

/// The taps a voice's early reflections have at most.
pub const MAX_TAPS: usize = 8;

/// One early reflection of a voice: how long after its direct sound it arrives, and its gain at
/// each ear.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ReflectionTap {
    /// Seconds after the direct sound.
    pub delay: f32,
    /// Linear gain at the left ear.
    pub gain_left: f32,
    /// Linear gain at the right ear.
    pub gain_right: f32,
}

/// A voice's early reflections: its first `count` taps.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Reflections {
    /// The taps; the first `count` count.
    pub taps: [ReflectionTap; MAX_TAPS],
    /// How many of `taps` count; 0 takes the voice's reflections away.
    pub count: usize,
}

impl VoiceSpatial {
    /// The cutoff at and above which the filter is open.
    pub const OPEN: f32 = 20_000.0;
}

impl Default for VoiceSpatial {
    fn default() -> Self {
        Self {
            gain_left: 1.0,
            gain_right: 1.0,
            cutoff_hz: Self::OPEN,
            delay_left: 0.0,
            delay_right: 0.0,
            shadow_left_hz: Self::OPEN,
            shadow_right_hz: Self::OPEN,
        }
    }
}

/// When a scheduled command takes effect.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum At {
    /// At the start of the mixer's next block.
    Now,
    /// At this output frame of the [`AudioClock`]; a frame already rendered means the next block.
    Frame(u64),
}

/// The shape of a fade.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FadeCurve {
    /// Gain changes linearly.
    Linear,
    /// Gain follows a quarter sine, so a fade-out and a fade-in of the same length keep their
    /// summed power constant (a crossfade).
    EqualPower,
}

/// A fade in or out over a number of output frames.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Fade {
    /// Length in output frames.
    pub frames: u32,
    /// Shape.
    pub curve: FadeCurve,
}

impl Fade {
    /// No fade: a start begins at full gain; a stop still fades over the 5 ms minimum, so that
    /// it does not click.
    pub const NONE: Self = Self {
        frames: 0,
        curve: FadeCurve::Linear,
    };

    /// A linear fade over `frames` output frames.
    #[must_use]
    pub const fn linear(frames: u32) -> Self {
        Self {
            frames,
            curve: FadeCurve::Linear,
        }
    }

    /// An equal-power fade over `frames` output frames.
    #[must_use]
    pub const fn equal_power(frames: u32) -> Self {
        Self {
            frames,
            curve: FadeCurve::EqualPower,
        }
    }
}

/// The number of frames the mixer has rendered, read without waiting.
#[derive(Clone, Debug)]
pub struct AudioClock(pub(crate) Arc<AtomicU64>);

impl AudioClock {
    /// Output frames rendered so far: the frame the next block starts at.
    #[must_use]
    pub fn frames(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}

/// What the control knows of a slot's voice: what a full pool compares to choose the voice a
/// new one replaces, and its clip's rate for positions.
#[derive(Copy, Clone, Debug, Default)]
struct VoiceRank {
    priority: u8,
    /// The gain last set.
    gain: f32,
    /// When the voice started, in starts.
    started: u64,
    /// The sample rate of the voice's clip.
    clip_rate: u32,
}

/// Drives an [`super::AudioMixer`] from the game thread (see the module documentation).
///
/// None of the voice calls allocate: the pool's slots and the command queue are allocated when
/// the mixer is created.
pub struct MixerControl {
    commands: Producer<MixerCommand>,
    events: Consumer<MixerEvent>,
    clock: AudioClock,
    /// The voice slots: a slot is live from its voice's start until the mixer ends the voice.
    voices: IdPool<VoiceSlot>,
    ranks: Box<[VoiceRank]>,
    states: Arc<VoiceStates>,
    starts: u64,
    /// Clip slots free to register into.
    free_clips: Vec<u32>,
    /// The sample rate of each registered clip, by `ClipId`.
    clip_rates: Box<[u32]>,
    /// Bus names, indexed by `BusId`.
    buses: Vec<String>,
    max_buses: usize,
    sample_rate: u32,
}

impl MixerControl {
    /// The control of a mixer of `config`'s capacities.
    pub(crate) fn new(
        commands: Producer<MixerCommand>,
        events: Consumer<MixerEvent>,
        clock: AudioClock,
        states: Arc<VoiceStates>,
        config: &MixerConfig,
    ) -> Self {
        let voice_slots = config.max_voices + config.max_virtual_voices;
        let max_clips = config.max_clips;
        // Every slot is allocated once and freed, so starting voices only reuses slots.
        let mut voices = IdPool::new();
        let handles: Vec<_> = (0..voice_slots).map(|_| voices.alloc()).collect();
        for handle in handles.into_iter().rev() {
            voices.free(handle);
        }
        Self {
            commands,
            events,
            clock,
            voices,
            ranks: vec![VoiceRank::default(); voice_slots].into_boxed_slice(),
            states,
            starts: 0,
            free_clips: (0..max_clips as u32).rev().collect(),
            clip_rates: vec![0; max_clips].into_boxed_slice(),
            buses: vec!["Master".to_string()],
            max_buses: config.max_buses.max(1),
            sample_rate: config.sample_rate,
        }
    }

    /// The output stream's sample rate: output frames per second of the [`AudioClock`].
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Adds a bus named `name` that feeds `parent`.
    ///
    /// # Errors
    /// [`AudioError::InvalidConfiguration`] if the name is taken, `parent` does not exist or
    /// every bus slot is in use; [`AudioError::QueueFull`] if the command queue is full.
    pub fn add_bus(&mut self, name: &str, parent: BusId) -> AudioResult<BusId> {
        if self.bus(name).is_some() {
            return Err(AudioError::InvalidConfiguration(format!(
                "a bus named {name:?} exists"
            )));
        }
        if usize::from(parent.0) >= self.buses.len() {
            return Err(AudioError::InvalidConfiguration(format!(
                "parent bus {parent:?} does not exist"
            )));
        }
        if self.buses.len() >= self.max_buses {
            return Err(AudioError::InvalidConfiguration(format!(
                "the mixer has room for {} buses",
                self.max_buses
            )));
        }
        let bus = BusId(self.buses.len() as u16);
        self.commands
            .push(MixerCommand::AddBus { bus, parent })
            .map_err(|_| AudioError::QueueFull)?;
        self.buses.push(name.to_string());
        Ok(bus)
    }

    /// The bus named `name`.
    #[must_use]
    pub fn bus(&self, name: &str) -> Option<BusId> {
        self.buses
            .iter()
            .position(|n| n == name)
            .map(|i| BusId(i as u16))
    }

    /// Ramps `bus`'s linear gain to `gain` over the 5 ms minimum; [`MASTER`]'s is the master
    /// volume. Before a bus mixes its first block, gain and mute changes apply at once.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_bus_gain(&mut self, bus: BusId, gain: f32) -> AudioResult<()> {
        self.ramp_bus_gain(bus, gain, 0)
    }

    /// Ramps `bus`'s linear gain to `gain` over `frames` output frames (at least 5 ms).
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn ramp_bus_gain(&mut self, bus: BusId, gain: f32, frames: u32) -> AudioResult<()> {
        self.commands
            .push(MixerCommand::SetBusGain {
                bus,
                gain: gain.max(0.0),
                frames,
            })
            .map_err(|_| AudioError::QueueFull)
    }

    /// Gives `bus` an effect (or none), which processes the mono sum of the voices' sends to it
    /// before the bus's gain; an effect it replaces is dropped here, at a later [`Self::update`].
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_bus_effect(&mut self, bus: BusId, effect: Option<BusEffect>) -> AudioResult<()> {
        self.send(MixerCommand::SetBusEffect { bus, effect })
    }

    /// Changes the size and damping of `bus`'s room reverb, without replacing it.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_room(&mut self, bus: BusId, room: RoomReverb) -> AudioResult<()> {
        self.send(MixerCommand::SetRoom { bus, room })
    }

    /// Changes the decay times of `bus`'s late reverb in each band, without replacing it.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_reverb_time(&mut self, bus: BusId, rt60: [f32; 3]) -> AudioResult<()> {
        self.send(MixerCommand::SetReverbTime { bus, rt60 })
    }

    /// Ramps `voice`'s send to `bus` to `level` over `frames` output frames (at least 5 ms once it
    /// sounds): the voice's signal after its gain and filter, before its panning, reaches `bus`'s
    /// effect at that level. A voice has one send; another bus moves it.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_send(
        &mut self,
        voice: VoiceHandle,
        bus: BusId,
        level: f32,
        frames: u32,
    ) -> AudioResult<()> {
        self.send(MixerCommand::SetSend {
            voice,
            bus,
            level: level.max(0.0),
            frames,
        })
    }

    /// Mutes or unmutes `bus` (ramping over the 5 ms minimum).
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_bus_muted(&mut self, bus: BusId, muted: bool) -> AudioResult<()> {
        self.commands
            .push(MixerCommand::SetBusMuted { bus, muted })
            .map_err(|_| AudioError::QueueFull)
    }

    /// The mixer's frame clock.
    #[must_use]
    pub fn clock(&self) -> &AudioClock {
        &self.clock
    }

    /// Makes `data` playable. The mixer keeps it until [`Self::release_clip`].
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if every clip slot is in use or the command queue is full.
    pub fn register_clip(&mut self, data: Arc<ClipData>) -> AudioResult<ClipId> {
        let slot = self.free_clips.pop().ok_or(AudioError::QueueFull)?;
        let clip = ClipId(slot);
        self.clip_rates[slot as usize] = data.sample_rate();
        if self
            .commands
            .push(MixerCommand::RegisterClip { clip, data })
            .is_err()
        {
            self.free_clips.push(slot);
            return Err(AudioError::QueueFull);
        }
        Ok(clip)
    }

    /// Releases `clip`: the mixer hands it back once no voice plays it, and [`Self::update`]
    /// drops it here. Voices playing it finish first.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn release_clip(&mut self, clip: ClipId) -> AudioResult<()> {
        self.commands
            .push(MixerCommand::ReleaseClip { clip })
            .map_err(|_| AudioError::QueueFull)
    }

    /// Starts a voice playing `clip` at full gain at the start of the mixer's next block.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn start(&mut self, clip: ClipId, params: VoiceParams) -> AudioResult<VoiceHandle> {
        self.start_with(clip, params, At::Now, Fade::NONE)
    }

    /// Starts a voice playing `clip` at `at`, fading in over `fade_in`.
    ///
    /// When every slot of the pool is in use, the new voice replaces the voice of lowest
    /// priority, then the quietest, then the oldest, unless that voice's priority is higher than
    /// the new one's. The replaced voice fades out over 5 ms.
    ///
    /// # Errors
    /// [`AudioError::VoicePoolFull`] if every playing voice has a higher priority;
    /// [`AudioError::QueueFull`] if the command queue is full. Neither changes anything.
    pub fn start_with(
        &mut self,
        clip: ClipId,
        params: VoiceParams,
        at: At,
        fade_in: Fade,
    ) -> AudioResult<VoiceHandle> {
        if self.commands.is_full() {
            return Err(AudioError::QueueFull);
        }
        if self.voices.len() == self.voices.slot_count() {
            self.collect_ends();
        }
        if self.voices.len() == self.voices.slot_count() {
            let victim = self
                .lowest_ranked()
                .filter(|&victim| self.ranks[victim.slot()].priority <= params.priority)
                .ok_or(AudioError::VoicePoolFull)?;
            // The mixer ends the replaced voice when the new one takes its slot.
            self.voices.free(victim.0);
        }
        let voice = VoiceHandle(self.voices.alloc());
        self.ranks[voice.slot()] = VoiceRank {
            priority: params.priority,
            gain: params.gain,
            started: self.starts,
            clip_rate: self.clip_rates.get(clip.0 as usize).copied().unwrap_or(0),
        };
        self.starts += 1;
        // Only this control pushes, so a queue that was not full has room.
        self.send(MixerCommand::Start {
            voice,
            clip,
            params,
            at: frame(at),
            fade_in,
        })?;
        Ok(voice)
    }

    /// The live voice a full pool replaces first: lowest priority, then quietest, then oldest.
    fn lowest_ranked(&self) -> Option<VoiceHandle> {
        let rank = |voice: &Handle<VoiceSlot>| self.ranks[voice.index() as usize];
        self.voices
            .iter()
            .min_by(|a, b| {
                let (a, b) = (rank(a), rank(b));
                a.priority
                    .cmp(&b.priority)
                    .then(a.gain.total_cmp(&b.gain))
                    .then(a.started.cmp(&b.started))
            })
            .map(VoiceHandle)
    }

    /// Frees the slots of the voices the mixer has ended.
    fn collect_ends(&mut self) {
        for index in 0..self.voices.slot_count() as u32 {
            if let Some(handle) = self.voices.handle_at(index) {
                if self.states.has_ended(VoiceHandle(handle)) {
                    self.voices.free(handle);
                }
            }
        }
    }

    fn send(&mut self, command: MixerCommand) -> AudioResult<()> {
        self.commands
            .push(command)
            .map_err(|_| AudioError::QueueFull)
    }

    /// Stops `voice` at the start of the mixer's next block, fading out over the 5 ms minimum.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn stop(&mut self, voice: VoiceHandle) -> AudioResult<()> {
        self.stop_with(voice, At::Now, Fade::NONE)
    }

    /// Stops `voice` at `at`, fading out over `fade_out` ([`Fade::NONE`]: the 5 ms minimum); the
    /// voice ends when the fade does. A voice stopped before it first sounds ends without
    /// sounding.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn stop_with(&mut self, voice: VoiceHandle, at: At, fade_out: Fade) -> AudioResult<()> {
        self.commands
            .push(MixerCommand::Stop {
                voice,
                at: frame(at),
                fade_out,
            })
            .map_err(|_| AudioError::QueueFull)
    }

    /// Ramps `voice`'s gain to `gain` over `frames` output frames (at least 5 ms).
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_gain(&mut self, voice: VoiceHandle, gain: f32, frames: u32) -> AudioResult<()> {
        let gain = gain.max(0.0);
        self.send(MixerCommand::SetGain {
            voice,
            gain,
            frames,
        })?;
        if self.voices.contains(voice.0) {
            self.ranks[voice.slot()].gain = gain;
        }
        Ok(())
    }

    /// Ramps `voice`'s pitch to `pitch` over `frames` output frames (at least 5 ms).
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_pitch(&mut self, voice: VoiceHandle, pitch: f32, frames: u32) -> AudioResult<()> {
        self.commands
            .push(MixerCommand::SetPitch {
                voice,
                pitch: pitch.max(0.0),
                frames,
            })
            .map_err(|_| AudioError::QueueFull)
    }

    /// True from [`Self::start`] until the mixer ends the voice: at the end of its clip, at the
    /// end of its stop's fade, or when a full pool replaces it.
    #[must_use]
    pub fn is_playing(&self, voice: VoiceHandle) -> bool {
        self.voices.contains(voice.0) && !self.states.has_ended(voice)
    }

    /// The output frame `voice` first sounded on (or, while virtual, would have): known once
    /// the mixer has rendered it; `None` before, and once it has ended.
    #[must_use]
    pub fn started_at(&self, voice: VoiceHandle) -> Option<u64> {
        if !self.is_playing(voice) {
            return None;
        }
        self.states.started(voice)
    }

    /// `voice`'s position in its clip, in seconds of the clip, as of the mixer's last block: 0
    /// until the mixer starts it, and `None` once it has ended.
    #[must_use]
    pub fn position(&self, voice: VoiceHandle) -> Option<f64> {
        if !self.is_playing(voice) {
            return None;
        }
        let frame = self.states.position(voice).unwrap_or(0);
        let rate = self.ranks[voice.slot()].clip_rate.max(1);
        Some(f64::from(frame) / f64::from(rate))
    }

    /// Ramps `voice`'s stereo gains, filter cutoffs and ear delays to `spatial` over `frames`
    /// output frames (at least 5 ms once it sounds); the cutoffs move in octaves.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_spatial(
        &mut self,
        voice: VoiceHandle,
        spatial: VoiceSpatial,
        frames: u32,
    ) -> AudioResult<()> {
        self.send(MixerCommand::SetSpatial {
            voice,
            spatial,
            frames,
        })
    }

    /// Gives `voice` early reflections, ramping each tap's delay and gains over `frames` output
    /// frames (at once for a voice new to its reflections). The mixer renders reflections for a
    /// fixed number of voices (`MixerConfig::max_reflecting_voices`); a voice beyond them has none
    /// until a slot is free, and a voice's end frees its slot. `count` 0 frees it at once.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn set_reflections(
        &mut self,
        voice: VoiceHandle,
        reflections: Reflections,
        frames: u32,
    ) -> AudioResult<()> {
        self.send(MixerCommand::SetReflections {
            voice,
            reflections,
            frames,
        })
    }

    /// Pauses `voice` at the start of the mixer's next block: it fades out over 5 ms, then holds
    /// its place. It still plays.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn pause(&mut self, voice: VoiceHandle) -> AudioResult<()> {
        self.send(MixerCommand::SetPaused {
            voice,
            paused: true,
        })
    }

    /// Resumes a paused `voice` from where it holds, fading in over 5 ms.
    ///
    /// # Errors
    /// [`AudioError::QueueFull`] if the command queue is full.
    pub fn resume(&mut self, voice: VoiceHandle) -> AudioResult<()> {
        self.send(MixerCommand::SetPaused {
            voice,
            paused: false,
        })
    }

    /// Frees the slots of ended voices and drops the clips the mixer released. Call it every
    /// game tick.
    pub fn update(&mut self) {
        while let Some(event) = self.events.pop() {
            match event {
                MixerEvent::ClipReleased { clip, data } => {
                    drop(data);
                    self.free_clips.push(clip.0);
                }
                MixerEvent::EffectReleased { effect } => drop(effect),
            }
        }
        self.collect_ends();
    }
}

/// The frame a scheduled command names: 0 for [`At::Now`] (any frame already rendered means the
/// next block).
fn frame(at: At) -> u64 {
    match at {
        At::Now => 0,
        At::Frame(frame) => frame,
    }
}
