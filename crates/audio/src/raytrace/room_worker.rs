//! The acoustics worker of raytraced mode (FIX-7.5, ADR-051; findings AUD-08 and AUD-11).
//!
//! A thread traces the level's [`AcousticGeometry`] at about 20 Hz from the listener, with the
//! settings' ray count and bounce limit, and publishes the room's decay time per band from the
//! statistics of its recent traces, and each emitter's early reflections: the image-source
//! paths of the specular chains its rays found ([`super::image_source`]), kept from trace to
//! trace while they still exist. The game thread hands it the listener and up to
//! [`MAX_RECEIVERS`] emitters each tick through a triple buffer of fixed size, so neither side
//! waits and the game thread allocates nothing.

use super::acoustic_geometry::AcousticGeometry;
use super::diffraction::diffracted_sound;
use super::image_source::{ImagePath, SpecularChain};
use super::path_tracer::{PathTracer, TraceSettings, TraceStatistics};
use super::tracer::{AIR_ABSORPTION_COEFFS, SPEED_OF_SOUND};
use crate::mixer::{MAX_TAPS, REFLECTION_SECONDS};
use crate::synthesis::{triple_buffer_with, TripleBufferConsumer, TripleBufferProducer};
use korppi_math::Vec3;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// The emitters a trace collects reflections for.
pub const MAX_RECEIVERS: usize = 16;
/// The worker's period.
const PERIOD: Duration = Duration::from_millis(50);
/// Traces whose statistics the decay time averages.
const AVERAGED_TRACES: usize = 8;
/// Chains an emitter keeps from trace to trace, the strongest first.
const KEPT_CHAINS: usize = 16;

/// What the game thread tells the worker each tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoomInput {
    /// The listener's position.
    pub listener: Vec3,
    /// The emitters' positions; the first `receiver_count` count.
    pub receivers: [Vec3; MAX_RECEIVERS],
    /// Each receiver's key, which its reflections in [`RoomOutput`] carry and which keeps its
    /// chains from trace to trace.
    pub ids: [u64; MAX_RECEIVERS],
    /// How many of `receivers` count.
    pub receiver_count: usize,
    /// Rays per trace (`AudioSettings::ray_count`).
    pub rays: usize,
    /// Reflections per path at most (`AudioSettings::max_bounces`).
    pub max_bounces: usize,
}

impl Default for RoomInput {
    fn default() -> Self {
        Self {
            listener: Vec3::ZERO,
            receivers: [Vec3::ZERO; MAX_RECEIVERS],
            ids: [0; MAX_RECEIVERS],
            receiver_count: 0,
            rays: 128,
            max_bounces: 4,
        }
    }
}

/// One early reflection reaching the listener from an emitter.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EarlyPath {
    /// Seconds after the emitter's direct sound.
    pub delay: f32,
    /// Its length, in meters.
    pub length: f32,
    /// The unit direction it arrives from, seen from the listener.
    pub arrival: Vec3,
    /// The unit direction it leaves the emitter in.
    pub departure: Vec3,
    /// The share of a unit source's energy at `length` it keeps per band: its surfaces' specular
    /// reflectance, and the air's absorption over its extra length beyond the direct sound's.
    pub reflectance: [f32; 3],
}

/// An emitter's early reflections, the strongest first, and its diffracted and corner-reflected
/// amplitude per band when its direct path is blocked.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EmitterReflections {
    /// The emitter's key, from [`RoomInput::ids`].
    pub id: u64,
    /// The paths; the first `count` count.
    pub paths: [EarlyPath; MAX_TAPS],
    /// How many of `paths` count.
    pub count: usize,
    /// The diffracted and corner-reflected amplitude per band relative to the direct sound over
    /// the straight distance, when the direct line of sight is blocked.
    pub diffraction: [f32; 3],
}

/// What the worker publishes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RoomOutput {
    /// The room's decay time per band, once a trace has measured it.
    pub rt60: Option<[f32; 3]>,
    /// Traces completed.
    pub traces: u64,
    /// The early reflections of the last trace's emitters; the first `emitters` count.
    pub reflections: [EmitterReflections; MAX_RECEIVERS],
    /// How many of `reflections` count.
    pub emitters: usize,
}

impl RoomOutput {
    /// The early reflections of the emitter with key `id`, if the last trace had it.
    #[must_use]
    pub fn reflections_of(&self, id: u64) -> Option<&EmitterReflections> {
        self.reflections[..self.emitters]
            .iter()
            .find(|reflections| reflections.id == id)
    }
}

/// The chains and edge diffraction an emitter keeps from trace to trace.
#[derive(Debug, Clone, Copy, Default)]
struct KeptChains {
    id: u64,
    chains: [SpecularChain; KEPT_CHAINS],
    count: usize,
    listener: Vec3,
    emitter: Vec3,
    edge_diffraction: [f32; 3],
    has_edge_diffraction: bool,
}

/// How strong a path arrives: its amplitude for a unit source, in the middle band.
fn strength(path: &ImagePath) -> f32 {
    path.reflectance[1].max(0.0).sqrt() / path.length.max(1e-3)
}

/// The early reflections and diffraction of the emitter at `emitter`, keyed `id`: its kept
/// chains and the trace's, those that reach it within [`REFLECTION_SECONDS`] of its direct sound,
/// strongest first, plus its BTM edge diffraction combined with `bounce_energy`. Updates `kept`
/// to them.
#[allow(clippy::too_many_arguments)]
fn reflections_of(
    geometry: &AcousticGeometry,
    listener: Vec3,
    emitter: Vec3,
    id: u64,
    previous: Option<&KeptChains>,
    traced: &[SpecularChain],
    bounce_energy: [f32; 3],
    kept: &mut KeptChains,
) -> EmitterReflections {
    let direct = (emitter - listener).length();
    let mut found = [(SpecularChain::default(), ImagePath::default()); KEPT_CHAINS];
    let mut count = 0;
    let prev_chains = previous.map_or(&[][..], |previous| &previous.chains[..previous.count]);
    for chain in prev_chains.iter().chain(traced) {
        if found[..count].iter().any(|(other, _)| other.same(chain)) {
            continue;
        }
        let Some(path) = chain.image_path(geometry, listener, emitter) else {
            continue;
        };
        if (path.length - direct) / SPEED_OF_SOUND > REFLECTION_SECONDS {
            continue;
        }
        // Kept sorted, strongest first; a full list drops its weakest.
        let place = found[..count]
            .iter()
            .position(|(_, other)| strength(&path) > strength(other))
            .unwrap_or(count);
        if place == KEPT_CHAINS {
            continue;
        }
        count = (count + 1).min(KEPT_CHAINS);
        found.copy_within(place..count - 1, place + 1);
        found[place] = (*chain, path);
    }
    let edge_diffraction = match previous {
        Some(prev)
            if prev.has_edge_diffraction
                && (prev.listener - listener).length_squared() <= 1e-6
                && (prev.emitter - emitter).length_squared() <= 1e-6 =>
        {
            prev.edge_diffraction
        }
        _ => diffracted_sound(geometry, emitter, listener),
    };
    let d0 = direct.max(1e-3);
    let bounce_scale = 4.0 * std::f32::consts::PI * d0 * d0;
    let diffraction = std::array::from_fn(|band| {
        let edge = edge_diffraction[band];
        (edge * edge + bounce_scale * bounce_energy[band]).sqrt()
    });
    *kept = KeptChains {
        id,
        count,
        listener,
        emitter,
        edge_diffraction,
        has_edge_diffraction: true,
        ..KeptChains::default()
    };
    let mut reflections = EmitterReflections {
        id,
        count: count.min(MAX_TAPS),
        diffraction,
        ..EmitterReflections::default()
    };
    for (index, (chain, path)) in found[..count].iter().enumerate() {
        kept.chains[index] = *chain;
        if index < MAX_TAPS {
            let extra = (path.length - direct).max(0.0);
            reflections.paths[index] = EarlyPath {
                delay: extra / SPEED_OF_SOUND,
                length: path.length,
                arrival: path.arrival,
                departure: path.departure,
                reflectance: std::array::from_fn(|band| {
                    path.reflectance[band] * (-AIR_ABSORPTION_COEFFS[band] * extra).exp()
                }),
            };
        }
    }
    reflections
}

/// A running acoustics worker over one geometry; dropping it stops the thread.
pub struct RoomWorker {
    input: TripleBufferProducer<RoomInput>,
    output: TripleBufferConsumer<RoomOutput>,
    running: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    geometry: Arc<AcousticGeometry>,
}

impl RoomWorker {
    /// Starts a worker tracing `geometry`.
    #[must_use]
    pub fn spawn(geometry: Arc<AcousticGeometry>) -> Self {
        let (input, mut inputs) = triple_buffer_with(RoomInput::default());
        let (mut outputs, output) = triple_buffer_with(RoomOutput::default());
        let running = Arc::new(AtomicBool::new(true));
        let thread = {
            let (running, geometry) = (Arc::clone(&running), Arc::clone(&geometry));
            std::thread::Builder::new()
                .name("korppi-acoustics".into())
                .spawn(move || {
                    let mut tracer = PathTracer::new(0x5EED);
                    let mut recent = [TraceStatistics::default(); AVERAGED_TRACES];
                    let mut traces = 0u64;
                    let mut kept = [KeptChains::default(); MAX_RECEIVERS];
                    while running.load(Ordering::Acquire) {
                        let started = Instant::now();
                        let input = *inputs.read();
                        let settings = TraceSettings {
                            rays: input.rays,
                            max_bounces: input.max_bounces,
                            ..TraceSettings::default()
                        };
                        let emitters = input.receiver_count.min(MAX_RECEIVERS);
                        let receivers = &input.receivers[..emitters];
                        let responses =
                            tracer.trace(&geometry, input.listener, receivers, &settings);
                        let mut output = RoomOutput {
                            emitters,
                            ..RoomOutput::default()
                        };
                        let previous = kept;
                        for index in 0..emitters {
                            let id = input.ids[index];
                            output.reflections[index] = reflections_of(
                                &geometry,
                                input.listener,
                                receivers[index],
                                id,
                                previous.iter().find(|chains| {
                                    chains.id == id
                                        && (chains.count > 0 || chains.has_edge_diffraction)
                                }),
                                tracer.chains(),
                                responses[index].bounce_energy(),
                                &mut kept[index],
                            );
                        }
                        for chains in &mut kept[emitters..] {
                            chains.count = 0;
                            chains.has_edge_diffraction = false;
                        }
                        recent[traces as usize % AVERAGED_TRACES] = tracer.statistics();
                        traces += 1;
                        let mut statistics = TraceStatistics::default();
                        for trace in &recent {
                            statistics.merge(trace);
                        }
                        output.rt60 = statistics.rt60();
                        output.traces = traces;
                        outputs.write(output);
                        if let Some(rest) = PERIOD.checked_sub(started.elapsed()) {
                            std::thread::sleep(rest);
                        }
                    }
                })
                .expect("spawn the acoustics worker")
        };
        Self {
            input,
            output,
            running,
            thread: Some(thread),
            geometry,
        }
    }

    /// The geometry the worker traces.
    #[must_use]
    pub fn geometry(&self) -> &Arc<AcousticGeometry> {
        &self.geometry
    }

    /// Hands the worker this tick's listener, emitters and settings.
    pub fn update(&mut self, input: RoomInput) {
        self.input.write(input);
    }

    /// The worker's latest result.
    pub fn output(&mut self) -> RoomOutput {
        *self.output.read()
    }
}

impl Drop for RoomWorker {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
