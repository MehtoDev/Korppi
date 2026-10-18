//! Commands from the game thread, and events and voice ends from the audio thread.

use super::clip::{ClipData, ClipId};
use super::control::{BusId, Fade, Reflections, VoiceHandle, VoiceParams, VoiceSpatial};
use super::effects::{BusEffect, RoomReverb};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// A command the mixer applies at the start of its next block.
pub(crate) enum MixerCommand {
    /// Makes `data` playable as `clip`.
    RegisterClip { clip: ClipId, data: Arc<ClipData> },
    /// Hands `clip` back once no voice plays it.
    ReleaseClip { clip: ClipId },
    /// Starts `voice` playing `clip` at output frame `at`, fading in.
    Start {
        voice: VoiceHandle,
        clip: ClipId,
        params: VoiceParams,
        at: u64,
        fade_in: Fade,
    },
    /// Stops `voice` at output frame `at`, fading out.
    Stop {
        voice: VoiceHandle,
        at: u64,
        fade_out: Fade,
    },
    /// Ramps a voice's gain.
    SetGain {
        voice: VoiceHandle,
        gain: f32,
        frames: u32,
    },
    /// Ramps a voice's pitch.
    SetPitch {
        voice: VoiceHandle,
        pitch: f32,
        frames: u32,
    },
    /// Ramps a voice's stereo gains and filter cutoff.
    SetSpatial {
        voice: VoiceHandle,
        spatial: VoiceSpatial,
        frames: u32,
    },
    /// Gives a voice early reflections, or takes them away.
    SetReflections {
        voice: VoiceHandle,
        reflections: Reflections,
        frames: u32,
    },
    /// Pauses (fading out, then holding its place) or resumes (fading in) a voice.
    SetPaused { voice: VoiceHandle, paused: bool },
    /// Adds `bus`, feeding `parent` (added before it).
    AddBus { bus: BusId, parent: BusId },
    /// Ramps a bus's linear gain.
    SetBusGain { bus: BusId, gain: f32, frames: u32 },
    /// Mutes or unmutes a bus.
    SetBusMuted { bus: BusId, muted: bool },
    /// Gives a bus an effect, or takes its effect away; a replaced effect comes back in an event.
    SetBusEffect {
        bus: BusId,
        effect: Option<BusEffect>,
    },
    /// Changes a bus's room reverb in place.
    SetRoom { bus: BusId, room: RoomReverb },
    /// Changes a bus's late reverb's decay times in place.
    SetReverbTime { bus: BusId, rt60: [f32; 3] },
    /// Ramps a voice's send to `bus` (one send per voice).
    SetSend {
        voice: VoiceHandle,
        bus: BusId,
        level: f32,
        frames: u32,
    },
}

/// What the mixer reports back through the event queue.
pub(crate) enum MixerEvent {
    /// `clip` was released and no voice plays it: its data, to drop on the game thread.
    ClipReleased { clip: ClipId, data: Arc<ClipData> },
    /// A bus's effect was replaced or removed: the effect, to drop on the game thread.
    EffectReleased { effect: BusEffect },
}

/// What the mixer reports about each voice slot, shared by the control and the mixer: the
/// generation of the voice it last ended, and its voice's position.
///
/// The mixer records a voice's end here (it reached the end of its clip, was stopped or
/// replaced, or could not start), and the control frees the voice's slot when it sees it. Unlike
/// an event, a record cannot be lost to a full queue, so a slot never leaks. A position packs
/// the voice's generation with its clip frame, so that it is never read for another voice.
pub(crate) struct VoiceStates {
    ended: Box<[AtomicU32]>,
    positions: Box<[AtomicU64]>,
    /// The output frame each slot's voice first rendered, valid with its position.
    starts: Box<[AtomicU64]>,
}

impl VoiceStates {
    pub(crate) fn new(slots: usize) -> Self {
        // Generations are never 0, so no slot starts out ended or with a position.
        Self {
            ended: (0..slots).map(|_| AtomicU32::new(0)).collect(),
            positions: (0..slots).map(|_| AtomicU64::new(0)).collect(),
            starts: (0..slots).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Records that `voice` ended.
    pub(crate) fn record_end(&self, voice: VoiceHandle) {
        if let Some(slot) = self.ended.get(voice.slot()) {
            slot.store(voice.generation(), Ordering::Release);
        }
    }

    /// Whether the mixer has ended `voice`.
    pub(crate) fn has_ended(&self, voice: VoiceHandle) -> bool {
        self.ended
            .get(voice.slot())
            .is_some_and(|slot| slot.load(Ordering::Acquire) == voice.generation())
    }

    /// Publishes `voice`'s position, in whole clip frames, and the output frame it first
    /// rendered.
    pub(crate) fn publish(&self, voice: VoiceHandle, frame: u32, started: Option<u64>) {
        let slot = voice.slot();
        if slot >= self.positions.len() {
            return;
        }
        // The start is written first: whoever reads the position with this generation (Acquire)
        // also reads this voice's start.
        self.starts[slot].store(started.unwrap_or(u64::MAX), Ordering::Relaxed);
        let packed = (u64::from(voice.generation()) << 32) | u64::from(frame);
        self.positions[slot].store(packed, Ordering::Release);
    }

    /// `voice`'s last published position in clip frames, if the mixer has published one.
    pub(crate) fn position(&self, voice: VoiceHandle) -> Option<u32> {
        let packed = self.positions.get(voice.slot())?.load(Ordering::Acquire);
        ((packed >> 32) as u32 == voice.generation()).then_some(packed as u32)
    }

    /// The output frame `voice` first rendered, once the mixer has published it. Only the
    /// control starts voices in a slot, so the slot's start cannot change between reading the
    /// position and the start for a voice the control still holds.
    pub(crate) fn started(&self, voice: VoiceHandle) -> Option<u64> {
        self.position(voice)?;
        let frame = self.starts[voice.slot()].load(Ordering::Relaxed);
        (frame != u64::MAX).then_some(frame)
    }
}
