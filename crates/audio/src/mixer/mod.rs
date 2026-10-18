//! The real-time mixer and its game-thread control (FIX-7.2, ADR-050; finding AUD-01).
//!
//! [`AudioMixer::new`] returns a pair: the [`MixerControl`] stays on the game thread, and the
//! [`AudioMixer`] becomes an output stream's [`crate::AudioCallback`] on the audio thread. They
//! share no lock: the control sends commands through one wait-free queue and receives events
//! through another, and reads the mixer's frame count ([`AudioClock`]) and which voices ended from
//! atomics. [`AudioVoices`] is the control as a world resource, for gameplay systems.
//!
//! The mixer owns the voices and the registered clips. A clip ([`ClipData`]) is immutable and
//! shared as an `Arc`; the mixer never frees one. When a clip is released and no voice plays it
//! any more, the mixer hands it back in an event, and the control drops it on the game thread.
//! Everything the mixer touches while rendering is allocated when it is created.
//!
//! Voices mix into buses ([`BusId`]): a bus scales and mutes its input and feeds its parent, and
//! [`MASTER`] feeds the device. The control names the buses and the mixer only sees their
//! indices; a bus is always added after its parent, so the mixer processes buses from the last
//! one down to the master.
//!
//! The voice pool has a fixed number of slots, `max_voices + max_virtual_voices`. Each block, the
//! mixer mixes the `max_voices` playing voices that rank highest by priority, then by audibility
//! (gain, fade and bus gains); the others are virtual, advancing without being mixed, and voices
//! move between the two over 5 ms. [`VoiceHandle`]s are generational handles to the slots, given
//! out by the control; when every slot is in use, a new voice replaces the lowest-ranked voice of
//! no higher priority, which fades out in one of a few spare tail slots.

mod clip;
mod control;
mod effects;
mod engine;
mod filter;
mod message;
mod render;
mod voices;

pub use clip::{ClipData, ClipId};
pub use control::{
    At, AudioClock, BusId, Fade, FadeCurve, MixerControl, ReflectionTap, Reflections, VoiceHandle,
    VoiceParams, VoiceSpatial, DEFAULT_PRIORITY, MASTER, MAX_TAPS,
};
pub use effects::{BusEffect, RoomReverb, REVERB_BANDS};
pub use engine::{AudioMixer, MixerConfig, REFLECTION_SECONDS};
pub use voices::{AudioVoices, RegisteredClip};
