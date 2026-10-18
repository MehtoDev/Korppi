//! Effects a bus runs on what voices send to it: reverbs (FIX-7.5, ADR-051; finding AUD-10).
//!
//! An effect is built on the game thread, where its delay lines are allocated, and moves to the
//! mixer in a command; one it replaces comes back in an event and is dropped on the game thread,
//! as clips are. Processing allocates nothing.

use crate::dsp::freeverb::Freeverb;
use crate::synthesis::fdn::{FdnConfig, FeedbackDelayNetwork, RT60_BANDS};
use serde::{Deserialize, Serialize};

/// The centre frequencies, in Hz, of the bands the late reverb takes decay times for: low, mid and
/// high.
pub const REVERB_BANDS: [f32; 3] = RT60_BANDS;

/// Freeverb's room: its size and how much it damps the highs, both from 0 to 1.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoomReverb {
    /// The room's size: longer decays as it grows.
    pub size: f32,
    /// High-frequency damping.
    pub damping: f32,
}

/// What a bus does to the mono sum of the sends it receives, before its gain: a reverb, whose wet
/// output joins the bus's mix.
pub enum BusEffect {
    /// Freeverb's room: standard mode's reverb.
    Room(Box<Freeverb>),
    /// The feedback delay network with a decay time per band: raytraced mode's late reverb.
    Late(Box<FeedbackDelayNetwork>),
}

impl BusEffect {
    /// Freeverb's `room` for a stream at `sample_rate`.
    #[must_use]
    pub fn room(room: RoomReverb, sample_rate: u32) -> Self {
        let mut reverb = Freeverb::new(sample_rate as f32);
        reverb.set_room_size(room.size);
        reverb.set_damp(room.damping);
        reverb.set_wet(1.0);
        reverb.set_dry(0.0);
        Self::Room(Box::new(reverb))
    }

    /// A late reverb decaying in `rt60` seconds in each of [`REVERB_BANDS`], for a stream at
    /// `sample_rate`.
    #[must_use]
    pub fn late(rt60: [f32; 3], sample_rate: u32) -> Self {
        let rate = sample_rate as f32;
        let mut network = FeedbackDelayNetwork::new(scaled_config(rate), rate);
        network.set_rt60(rt60, rate);
        Self::Late(Box::new(network))
    }

    /// Reverberates one mono sample.
    #[inline]
    pub(super) fn process(&mut self, input: f32) -> [f32; 2] {
        let (left, right) = match self {
            Self::Room(reverb) => reverb.process_mono(input),
            Self::Late(network) => network.process_sample(input),
        };
        [left, right]
    }

    /// Changes a room's size and damping in place; other effects are unchanged.
    pub(super) fn set_room(&mut self, room: RoomReverb) {
        if let Self::Room(reverb) = self {
            reverb.set_room_size(room.size);
            reverb.set_damp(room.damping);
        }
    }

    /// Changes a late reverb's decay times in place; other effects are unchanged.
    pub(super) fn set_rt60(&mut self, rt60: [f32; 3], sample_rate: f32) {
        if let Self::Late(network) = self {
            network.set_rt60(rt60, sample_rate);
        }
    }
}

/// The network's delay lines scaled from their 48 kHz lengths to `sample_rate`, so that its
/// echo density is the same at any rate.
fn scaled_config(sample_rate: f32) -> FdnConfig {
    let mut config = FdnConfig::default();
    for length in &mut config.delay_lengths {
        *length = ((*length as f32) * sample_rate / 48_000.0).round().max(1.0) as usize;
    }
    config
}
