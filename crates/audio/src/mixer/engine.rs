//! The audio thread's side: the voice pool, clips, buses and the render loop.

use super::clip::{ClipData, ClipId};
use super::control::{AudioClock, BusId, MixerControl, Reflections, VoiceHandle, VoiceSpatial};
use super::effects::BusEffect;
use super::filter::{EarDelay, LowPass, ReflectionSlot};
use super::message::{MixerCommand, MixerEvent, VoiceStates};
use super::render::{Envelope, Ramp, Voice};
use crate::device::{AudioCallback, AudioCallbackInfo, AudioStreamConfig};
use crate::ring_buffer::{Consumer, Producer, SpscRingBuffer};
use std::cmp::Ordering as Rank;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Capacities of a mixer, fixed when it is created.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MixerConfig {
    /// Voices mixed at once: the playing voices that rank highest by priority, then audibility.
    pub max_voices: usize,
    /// Further voices that can play while virtual: their cursors advance, but they are not
    /// mixed. The pool holds `max_voices + max_virtual_voices` voices.
    pub max_virtual_voices: usize,
    /// Clips that can be registered at once.
    pub max_clips: usize,
    /// The output stream's sample rate, for scheduling on the game thread.
    pub sample_rate: u32,
    /// Buses, the master included.
    pub max_buses: usize,
    /// Commands that can wait for the next block, and events for the next game tick.
    pub queue_capacity: usize,
    /// Voices with early reflections at once ([`MixerControl::set_reflections`]), each with a
    /// line of [`REFLECTION_SECONDS`].
    pub max_reflecting_voices: usize,
}

impl MixerConfig {
    /// Capacities for a mixer feeding a stream of `config`: 64 voices mixed of 256 playing,
    /// 256 clips, 32 buses, 16 voices with early reflections.
    #[must_use]
    pub fn for_stream(config: &AudioStreamConfig) -> Self {
        Self {
            max_voices: 64,
            max_virtual_voices: 192,
            max_clips: 256,
            sample_rate: config.sample_rate,
            max_buses: 32,
            queue_capacity: 1024,
            max_reflecting_voices: 16,
        }
    }
}

/// The longest delay of an early reflection after its voice's direct sound, in seconds.
pub const REFLECTION_SECONDS: f32 = 0.1;

/// Frames the mixer renders at a time; a device block longer than this is rendered in parts.
const BLOCK: usize = 1024;
/// The shortest ramp of a gain or pitch change, and the fade of a stop given none (AUD-06).
const MIN_RAMP_SECONDS: f64 = 0.005;
/// Voices replaced while audible that can fade out at once; beyond these, they stop at once.
const TAILS: usize = 8;
/// How much louder an audible voice counts when voices are ranked, so that voices of similar
/// audibility do not trade places every block.
const AUDIBLE_BIAS: f32 = 2.0;

/// The mix of bus `own` and the sends of bus `send`, if any, for `frames` frames; they may be the
/// same bus.
fn mix_and_send(
    buses: &mut [Bus],
    own: usize,
    send: Option<usize>,
    frames: usize,
) -> (&mut [f32], Option<&mut [f32]>) {
    match send {
        None => (&mut buses[own].mix[..2 * frames], None),
        Some(send) if send == own => {
            let Bus { mix, sends, .. } = &mut buses[own];
            (&mut mix[..2 * frames], Some(&mut sends[..frames]))
        }
        Some(send) => {
            let (low, high) = buses.split_at_mut(own.max(send));
            let (own_bus, send_bus) = if own < send {
                (&mut low[own], &mut high[0])
            } else {
                (&mut high[0], &mut low[send])
            };
            (
                &mut own_bus.mix[..2 * frames],
                Some(&mut send_bus.sends[..frames]),
            )
        }
    }
}

/// The length of a ramp of `frames` frames requested for `voice`: at least `min_ramp` once the
/// voice sounds, so that it does not click; as requested before, since it cannot.
fn ramp(voice: &Voice, frames: u32, min_ramp: u32) -> u32 {
    if voice.fresh {
        frames
    } else {
        frames.max(min_ramp)
    }
}

/// A mixing bus: its stereo accumulator for the current block, and how it feeds its parent.
struct Bus {
    parent: usize,
    gain: Ramp,
    /// 1 when audible, 0 when muted; ramps between them.
    mute: Ramp,
    /// Whether the bus exists (the slot was added).
    active: bool,
    /// Whether the bus has not mixed a block yet: its gain and mute change at once, since it
    /// cannot click before it carries sound.
    fresh: bool,
    /// The bus's gain times its parents', for ranking voices.
    level: f32,
    /// Interleaved stereo, `2 * BLOCK` samples.
    mix: Box<[f32]>,
    /// The mono sum of the sends voices make to the bus, `BLOCK` samples, for its effect.
    sends: Box<[f32]>,
    effect: Option<BusEffect>,
}

/// A registered clip and the voices playing it.
#[derive(Default)]
struct ClipSlot {
    data: Option<Arc<ClipData>>,
    voices: u32,
    released: bool,
}

/// The real-time mixer: an [`AudioCallback`] that renders the voices a [`MixerControl`] starts.
pub struct AudioMixer {
    commands: Consumer<MixerCommand>,
    events: Producer<MixerEvent>,
    /// Clip releases the queue had no room for, sent first at the next block: they are never
    /// dropped, since the mixer must not free a clip.
    pending_releases: Vec<MixerEvent>,
    clock: Arc<AtomicU64>,
    states: Arc<VoiceStates>,
    /// The pool's slots, indexed by [`VoiceHandle`], then the tails: replaced voices fading out.
    voices: Vec<Option<Voice>>,
    slots: usize,
    max_audible: usize,
    /// Scratch for ranking the slots each block.
    order: Vec<u32>,
    clips: Vec<ClipSlot>,
    buses: Vec<Bus>,
    /// Early reflection lines, lent to voices.
    reflections: Vec<ReflectionSlot>,
    /// Output frames rendered: the first frame of the next block.
    rendered: u64,
    /// The shortest ramp at the stream's rate, in frames.
    min_ramp: u32,
    /// The stream's rate, as of the last block.
    device_rate: f64,
}

impl AudioMixer {
    /// Creates a mixer and the control that drives it. Every buffer the mixer uses while
    /// rendering is allocated here.
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(config: MixerConfig) -> (MixerControl, AudioMixer) {
        let (command_producer, command_consumer) = SpscRingBuffer::new(config.queue_capacity);
        let (event_producer, event_consumer) = SpscRingBuffer::new(config.queue_capacity);
        let clock = Arc::new(AtomicU64::new(0));
        let slots = config.max_voices + config.max_virtual_voices;
        let states = Arc::new(VoiceStates::new(slots));
        let control = MixerControl::new(
            command_producer,
            event_consumer,
            AudioClock(Arc::clone(&clock)),
            Arc::clone(&states),
            &config,
        );
        let mixer = AudioMixer {
            commands: command_consumer,
            events: event_producer,
            pending_releases: Vec::with_capacity(config.max_clips + config.max_buses.max(1)),
            clock,
            states,
            voices: (0..slots + TAILS).map(|_| None).collect(),
            slots,
            max_audible: config.max_voices,
            order: Vec::with_capacity(slots),
            clips: (0..config.max_clips).map(|_| ClipSlot::default()).collect(),
            buses: (0..config.max_buses.max(1))
                .map(|index| Bus {
                    parent: 0,
                    gain: Ramp::at(1.0),
                    mute: Ramp::at(1.0),
                    active: index == 0,
                    fresh: true,
                    level: 1.0,
                    mix: vec![0.0; 2 * BLOCK].into_boxed_slice(),
                    sends: vec![0.0; BLOCK].into_boxed_slice(),
                    effect: None,
                })
                .collect(),
            reflections: {
                // Up to 192 kHz streams; the line also holds the interpolation's neighbours.
                let frames =
                    (REFLECTION_SECONDS * config.sample_rate.max(1) as f32).ceil() as usize;
                (0..config.max_reflecting_voices)
                    .map(|_| ReflectionSlot::new(frames + 4))
                    .collect()
            },
            rendered: 0,
            min_ramp: 240,
            device_rate: f64::from(config.sample_rate.max(1)),
        };
        (control, mixer)
    }

    fn apply(&mut self, command: MixerCommand) {
        match command {
            MixerCommand::RegisterClip { clip, data } => {
                let Some(slot) = self.clips.get_mut(clip.0 as usize) else {
                    // A slot the control never hands out: give the data back.
                    self.emit(MixerEvent::ClipReleased { clip, data });
                    return;
                };
                let previous = slot.data.replace(data);
                slot.voices = 0;
                slot.released = false;
                if let Some(previous) = previous {
                    self.emit(MixerEvent::ClipReleased {
                        clip,
                        data: previous,
                    });
                }
            }
            MixerCommand::ReleaseClip { clip } => {
                if let Some(slot) = self.clips.get_mut(clip.0 as usize) {
                    slot.released = true;
                    self.release_if_unused(clip.0 as usize);
                }
            }
            MixerCommand::Start {
                voice,
                clip,
                params,
                at,
                fade_in,
            } => {
                let slot = voice.slot();
                if slot >= self.slots {
                    return;
                }
                // The control reuses a slot only after its voice ended or to replace it.
                self.replace(slot);
                let index = clip.0 as usize;
                let playable = self
                    .clips
                    .get(index)
                    .is_some_and(|slot| slot.data.is_some() && !slot.released);
                if !playable {
                    // An unknown or released clip: the voice ends at once.
                    self.states.record_end(voice);
                    return;
                }
                self.clips[index].voices += 1;
                let frames = self.clips[index]
                    .data
                    .as_ref()
                    .map_or(0, |data| data.frames());
                let loop_end = params
                    .loop_end
                    .map_or(frames, |end| (end as usize).min(frames))
                    .max(1);
                let loop_start = (params.loop_start as usize).min(loop_end - 1);
                self.voices[slot] = Some(Voice {
                    handle: voice,
                    clip: index,
                    cursor: 0.0,
                    gain: Ramp::at(params.gain),
                    pitch: Ramp::at(params.pitch.max(0.0)),
                    pan: [Ramp::at(1.0); 2],
                    filter: LowPass::new(VoiceSpatial::OPEN),
                    delay: EarDelay::new(),
                    send_bus: usize::MAX,
                    send: Ramp::at(0.0),
                    reflection: usize::MAX,
                    looping: params.looping,
                    loop_start: loop_start as f64,
                    loop_end: loop_end as f64,
                    wrapped: false,
                    bus: usize::from(params.bus.0),
                    priority: params.priority,
                    start_at: at,
                    stop_at: None,
                    envelope: Envelope::fade_in(fade_in),
                    audible: false,
                    presence: Ramp::at(0.0),
                    fresh: true,
                    started: None,
                    paused: false,
                    run: Ramp::at(1.0),
                    rank_level: 0.0,
                });
            }
            MixerCommand::AddBus { bus, parent } => {
                let (index, parent) = (usize::from(bus.0), usize::from(parent.0));
                if index < self.buses.len() && parent < index {
                    let bus = &mut self.buses[index];
                    bus.parent = parent;
                    bus.gain = Ramp::at(1.0);
                    bus.mute = Ramp::at(1.0);
                    bus.active = true;
                    bus.fresh = true;
                }
            }
            MixerCommand::SetBusGain { bus, gain, frames } => {
                let min_ramp = self.min_ramp;
                if let Some(bus) = self.bus_mut(bus) {
                    let frames = if bus.fresh { 0 } else { frames.max(min_ramp) };
                    bus.gain.to(gain, frames);
                }
            }
            MixerCommand::SetBusMuted { bus, muted } => {
                let min_ramp = self.min_ramp;
                if let Some(bus) = self.bus_mut(bus) {
                    let frames = if bus.fresh { 0 } else { min_ramp };
                    bus.mute.to(if muted { 0.0 } else { 1.0 }, frames);
                }
            }
            MixerCommand::Stop {
                voice,
                at,
                fade_out,
            } => {
                let first_frame = self.rendered;
                let Some(found) = self.voice_mut(voice) else {
                    return;
                };
                if found.fresh && at <= found.start_at.max(first_frame) {
                    // Stopped before it first sounds: it ends without sounding.
                    self.finish(voice.slot());
                } else {
                    found.stop_at = Some((at, fade_out));
                }
            }
            MixerCommand::SetGain {
                voice,
                gain,
                frames,
            } => {
                let min_ramp = self.min_ramp;
                if let Some(voice) = self.voice_mut(voice) {
                    voice.gain.to(gain, ramp(voice, frames, min_ramp));
                }
            }
            MixerCommand::SetPitch {
                voice,
                pitch,
                frames,
            } => {
                let min_ramp = self.min_ramp;
                if let Some(voice) = self.voice_mut(voice) {
                    voice.pitch.to(pitch, ramp(voice, frames, min_ramp));
                }
            }
            MixerCommand::SetSpatial {
                voice,
                spatial,
                frames,
            } => {
                let (min_ramp, rate) = (self.min_ramp, self.device_rate as f32);
                if let Some(voice) = self.voice_mut(voice) {
                    let frames = ramp(voice, frames, min_ramp);
                    voice.pan[0].to(spatial.gain_left.max(0.0), frames);
                    voice.pan[1].to(spatial.gain_right.max(0.0), frames);
                    voice.filter.set_cutoffs(
                        [
                            spatial.cutoff_hz.min(spatial.shadow_left_hz),
                            spatial.cutoff_hz.min(spatial.shadow_right_hz),
                        ],
                        frames,
                    );
                    voice.delay.set(
                        [spatial.delay_left * rate, spatial.delay_right * rate],
                        frames,
                    );
                }
            }
            MixerCommand::SetBusEffect { bus, effect } => {
                let previous = match self.bus_mut(bus) {
                    Some(bus) => std::mem::replace(&mut bus.effect, effect),
                    // A bus that does not exist: give the effect back.
                    None => effect,
                };
                if let Some(effect) = previous {
                    self.emit(MixerEvent::EffectReleased { effect });
                }
            }
            MixerCommand::SetRoom { bus, room } => {
                if let Some(effect) = self.bus_mut(bus).and_then(|bus| bus.effect.as_mut()) {
                    effect.set_room(room);
                }
            }
            MixerCommand::SetReverbTime { bus, rt60 } => {
                let rate = self.device_rate as f32;
                if let Some(effect) = self.bus_mut(bus).and_then(|bus| bus.effect.as_mut()) {
                    effect.set_rt60(rt60, rate);
                }
            }
            MixerCommand::SetSend {
                voice,
                bus,
                level,
                frames,
            } => {
                let min_ramp = self.min_ramp;
                if let Some(voice) = self.voice_mut(voice) {
                    let frames = ramp(voice, frames, min_ramp);
                    let target = usize::from(bus.0);
                    if voice.send_bus != target {
                        // Moving the send starts it from silence at its new bus.
                        voice.send_bus = target;
                        voice.send = Ramp::at(0.0);
                    }
                    voice.send.to(level, frames);
                }
            }
            MixerCommand::SetReflections {
                voice,
                reflections,
                frames,
            } => self.set_reflections(voice, &reflections, frames),
            MixerCommand::SetPaused { voice, paused } => {
                let min_ramp = self.min_ramp;
                if let Some(voice) = self.voice_mut(voice) {
                    voice.paused = paused;
                    let frames = ramp(voice, 0, min_ramp);
                    voice.run.to(if paused { 0.0 } else { 1.0 }, frames);
                }
            }
        }
    }

    /// Gives a voice its early reflections, lending it a free line (a quiet one first) if it has
    /// none; `count` 0 lets its line go.
    fn set_reflections(&mut self, handle: VoiceHandle, reflections: &Reflections, frames: u32) {
        let (min_ramp, rate) = (self.min_ramp, self.device_rate as f32);
        let Some(voice) = self.voice_mut(handle) else {
            return;
        };
        let frames = ramp(voice, frames, min_ramp);
        let lent = voice.reflection;
        if reflections.count == 0 {
            if let Some(slot) = self.reflections.get_mut(lent) {
                slot.release();
            }
            if let Some(voice) = self.voice_mut(handle) {
                voice.reflection = usize::MAX;
            }
            return;
        }
        let index = if lent < self.reflections.len() {
            lent
        } else {
            let free = |quiet: bool| {
                self.reflections
                    .iter()
                    .position(|slot| !slot.in_use && slot.is_quiet() == quiet)
            };
            // Every line is lent: the voice has none until one is free.
            let Some(free) = free(true).or_else(|| free(false)) else {
                return;
            };
            self.reflections[free].assign();
            if let Some(voice) = self.voice_mut(handle) {
                voice.reflection = free;
            }
            free
        };
        self.reflections[index].set(reflections, rate, frames);
    }

    /// The voice `handle` names, if it still plays: its slot holds it, not a later voice.
    fn voice_mut(&mut self, handle: VoiceHandle) -> Option<&mut Voice> {
        self.voices
            .get_mut(handle.slot())
            .filter(|_| handle.slot() < self.slots)?
            .as_mut()
            .filter(|voice| voice.handle == handle)
    }

    fn bus_mut(&mut self, bus: BusId) -> Option<&mut Bus> {
        self.buses
            .get_mut(usize::from(bus.0))
            .filter(|bus| bus.active)
    }

    /// Ends the voice in pool slot `slot` to make room for another: one in the mix fades out in
    /// a free tail, any other ends at once.
    fn replace(&mut self, slot: usize) {
        let Some(mut voice) = self.voices[slot].take() else {
            return;
        };
        self.states.record_end(voice.handle);
        if voice.is_sounding() {
            if let Some(tail) = (self.slots..self.voices.len()).find(|&t| self.voices[t].is_none())
            {
                voice.stop_now();
                self.voices[tail] = Some(voice);
                return;
            }
        }
        self.retire(voice);
    }

    /// Ends the voice in slot `index` (of the pool or the tails).
    fn finish(&mut self, index: usize) {
        if let Some(voice) = self.voices[index].take() {
            if index < self.slots {
                self.states.record_end(voice.handle);
            }
            self.retire(voice);
        }
    }

    /// Lets go of an ended voice's clip and reflection line (which sounds out what it holds).
    fn retire(&mut self, voice: Voice) {
        if let Some(slot) = self.reflections.get_mut(voice.reflection) {
            slot.release();
        }
        self.clips[voice.clip].voices -= 1;
        self.release_if_unused(voice.clip);
    }

    /// Hands a released clip back once no voice plays it.
    fn release_if_unused(&mut self, index: usize) {
        let slot = &mut self.clips[index];
        if slot.released && slot.voices == 0 {
            if let Some(data) = slot.data.take() {
                slot.released = false;
                self.emit(MixerEvent::ClipReleased {
                    clip: ClipId(index as u32),
                    data,
                });
            }
        }
    }

    fn emit(&mut self, event: MixerEvent) {
        if let Err(event) = self.events.push(event) {
            if self.pending_releases.len() < self.pending_releases.capacity() {
                self.pending_releases.push(event);
            }
        }
    }

    /// Sends the events the queue had no room for, oldest first; what still does not fit stays
    /// pending, in order (nothing is dropped, and nothing allocates: the vector keeps its
    /// capacity).
    fn flush_pending(&mut self) {
        while !self.pending_releases.is_empty() {
            let event = self.pending_releases.remove(0);
            if let Err(event) = self.events.push(event) {
                self.pending_releases.insert(0, event);
                return;
            }
        }
    }

    /// Chooses the voices mixed in the block ending before output frame `block_end`: the
    /// `max_audible` that rank highest by priority, then by audibility (gain, fade and buses).
    /// Voices that start later wait, virtual, and take no place.
    fn rank_voices(&mut self, block_end: u64) {
        for index in 0..self.buses.len() {
            let parent = if index == 0 {
                1.0
            } else {
                self.buses[self.buses[index].parent].level
            };
            let bus = &mut self.buses[index];
            bus.level = if bus.active {
                parent * bus.gain.peak() * bus.mute.peak()
            } else {
                0.0
            };
        }
        self.order.clear();
        for index in 0..self.slots {
            let Some(voice) = self.voices[index].as_mut() else {
                continue;
            };
            if voice.start_at >= block_end || voice.is_held() {
                // Not sounding in this block: it takes no place, and has nothing to fade.
                voice.set_audible(false, 0);
                continue;
            }
            let bus = self.buses.get(voice.bus).filter(|bus| bus.active);
            let bias = if voice.audible && !voice.fresh {
                AUDIBLE_BIAS
            } else {
                1.0
            };
            voice.rank_level = voice.gain.peak()
                * voice.envelope.peak()
                * bus.map_or(self.buses[0].level, |bus| bus.level)
                * bias;
            self.order.push(index as u32);
        }
        let audible = self.max_audible.min(self.order.len());
        if audible < self.order.len() {
            let voices = &self.voices;
            let rank = |&a: &u32, &b: &u32| -> Rank {
                let (va, vb) = (voices[a as usize].as_ref(), voices[b as usize].as_ref());
                let (Some(va), Some(vb)) = (va, vb) else {
                    return Rank::Equal;
                };
                vb.priority
                    .cmp(&va.priority)
                    .then(vb.rank_level.total_cmp(&va.rank_level))
                    .then(a.cmp(&b))
            };
            self.order.select_nth_unstable_by(audible, rank);
        }
        let min_ramp = self.min_ramp;
        for (place, &index) in self.order.iter().enumerate() {
            if let Some(voice) = self.voices[index as usize].as_mut() {
                voice.set_audible(place < audible, min_ramp);
            }
        }
    }

    /// Mixes every voice into its bus for `frames` frames, then the buses into the master.
    fn mix_block(&mut self, frames: usize, device_rate: f64) {
        let block_start = self.rendered;
        self.rank_voices(block_start + frames as u64);
        for bus in self.buses.iter_mut().filter(|bus| bus.active) {
            bus.mix[..2 * frames].fill(0.0);
            bus.sends[..frames].fill(0.0);
        }
        // Lines their voices let go of sound out what they hold.
        for slot in self.reflections.iter_mut().filter(|slot| !slot.in_use) {
            let bus = if self.buses.get(slot.bus).is_some_and(|b| b.active) {
                slot.bus
            } else {
                0
            };
            slot.sound_out(&mut self.buses[bus].mix[..2 * frames]);
        }
        let min_ramp = self.min_ramp;
        for index in 0..self.voices.len() {
            let Some(voice) = self.voices[index].as_mut() else {
                continue;
            };
            let Some(clip) = self.clips[voice.clip].data.as_deref() else {
                continue;
            };
            let bus = if self.buses.get(voice.bus).is_some_and(|b| b.active) {
                voice.bus
            } else {
                0
            };
            let rate = f64::from(clip.sample_rate()) / device_rate;
            let send_bus =
                Some(voice.send_bus).filter(|&b| self.buses.get(b).is_some_and(|b| b.active));
            let (mix, send) = mix_and_send(&mut self.buses, bus, send_bus, frames);
            let mut reflections = self.reflections.get_mut(voice.reflection);
            if let Some(slot) = reflections.as_deref_mut() {
                // Once let go, the line sounds out into the bus its voice last played on.
                slot.bus = bus;
            }
            if voice.render(
                clip,
                mix,
                send,
                reflections,
                block_start,
                rate,
                device_rate as f32,
                min_ramp,
            ) {
                self.finish(index);
            } else if index < self.slots {
                self.states
                    .publish(voice.handle, voice.cursor as u32, voice.started);
            }
        }
        // Each effect adds its output, from the sends, to its bus's mix.
        for bus in self.buses.iter_mut().filter(|bus| bus.active) {
            let Bus {
                effect, mix, sends, ..
            } = bus;
            if let Some(effect) = effect {
                for (out, &input) in mix[..2 * frames]
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .zip(&sends[..frames])
                {
                    let [left, right] = effect.process(input);
                    out[0] += left;
                    out[1] += right;
                }
            }
        }
        // Children were added after their parents: the last bus first, the master last.
        for index in (1..self.buses.len()).rev() {
            if !self.buses[index].active {
                continue;
            }
            let (head, tail) = self.buses.split_at_mut(index);
            let child = &mut tail[0];
            let parent = &mut head[child.parent];
            for (out, mixed) in parent.mix[..2 * frames]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(child.mix[..2 * frames].as_chunks::<2>().0)
            {
                let gain = child.gain.next() * child.mute.next();
                out[0] += mixed[0] * gain;
                out[1] += mixed[1] * gain;
            }
        }
        for bus in self.buses.iter_mut() {
            bus.fresh = false;
        }
    }
}

impl AudioCallback for AudioMixer {
    fn render(&mut self, output: &mut [f32], info: &AudioCallbackInfo) {
        let device_rate = f64::from(info.sample_rate.max(1));
        self.min_ramp = (device_rate * MIN_RAMP_SECONDS).ceil() as u32;
        self.device_rate = device_rate;
        self.flush_pending();
        while let Some(command) = self.commands.pop() {
            self.apply(command);
        }
        let channels = info.channels.count().max(1);
        for part in output.chunks_mut(BLOCK * channels) {
            let frames = part.len() / channels;
            self.mix_block(frames, device_rate);
            let master = &mut self.buses[0];
            for (out, mixed) in part
                .chunks_exact_mut(channels)
                .zip(master.mix.as_chunks::<2>().0)
            {
                let gain = master.gain.next() * master.mute.next();
                let (left, right) = (mixed[0] * gain, mixed[1] * gain);
                if channels == 1 {
                    out[0] = (left + right) * 0.5;
                } else {
                    out[0] = left;
                    out[1] = right;
                    out[2..].fill(0.0);
                }
            }
            self.rendered += frames as u64;
        }
        self.clock.store(self.rendered, Ordering::Release);
    }
}
