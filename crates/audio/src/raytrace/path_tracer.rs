//! Monte Carlo acoustic path tracing (FIX-7.5, ADR-051; finding AUD-08).
//!
//! Rays leave the listener in uniformly random directions. Each segment loses energy to the air,
//! each hit absorbs the material's share per band, and the next direction is diffuse
//! (cosine-weighted about the normal) with the material's scattering coefficient as probability,
//! specular otherwise; Russian roulette ends weak paths without bias. A sphere around each
//! receiver (an emitter: the paths are reciprocal) collects the energy of every segment crossing
//! it, weighted by the chord's length over the sphere's volume. That estimator converges to the
//! intensity of a unit point source, `1 / (4π d²)` at distance `d`, so reflections and the direct
//! sound share one scale.
//!
//! Inputs: an [`AcousticGeometry`], the listener, the receivers and [`TraceSettings`]. Outputs: a
//! [`RoomResponse`] per receiver: an energy decay histogram per band (and from it the RT60) and the
//! early reflection paths, with their delay, energy, order and arrival direction at the listener.

use super::acoustic_geometry::AcousticGeometry;
use super::energy_decay::EnergyHistogram;
use super::geometry::AcousticRay;
use super::image_source::{ReflectingPlane, SpecularChain, MAX_IMAGE_ORDER};
use super::tracer::{AIR_ABSORPTION_COEFFS, SPEED_OF_SOUND};
use korppi_math::Vec3;
use std::f32::consts::{PI, TAU};

/// How far a bounce starts off its surface.
const SURFACE_OFFSET: f32 = 1e-4;
/// The energy below which a path plays Russian roulette, relative to its start.
const ROULETTE_LEVEL: f32 = 1e-3;
/// The specular chains a trace collects at most.
const MAX_CHAINS: usize = 64;

/// What a trace computes and how hard it tries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceSettings {
    /// Rays leaving the listener (`AudioSettings::ray_count`).
    pub rays: usize,
    /// Reflections a path makes at most (`AudioSettings::max_bounces`).
    pub max_bounces: usize,
    /// The radius of each receiver's sphere, in meters.
    pub receiver_radius: f32,
    /// The length of the response, in seconds.
    pub duration: f32,
    /// The width of a histogram bin, in seconds.
    pub bin: f32,
    /// Reflections are kept as paths up to this order...
    pub early_order: usize,
    /// ...and this delay, in seconds.
    pub early_time: f32,
}

impl Default for TraceSettings {
    fn default() -> Self {
        Self {
            rays: 128,
            max_bounces: 4,
            receiver_radius: 0.5,
            duration: 3.0,
            bin: 0.005,
            early_order: 3,
            early_time: 0.1,
        }
    }
}

/// A path from the listener that reached a receiver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionPath {
    /// When it arrives, in seconds after the sound leaves (at its closest approach).
    pub delay: f32,
    /// The intensity it deposits per band, on the scale of `1 / (4π d²)`.
    pub energy: [f32; 3],
    /// Where the sound arrives from at the listener: the path's first direction.
    pub direction: Vec3,
    /// Its number of reflections: 0 for the direct sound.
    pub order: usize,
}

/// What the paths reaching one receiver add up to.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomResponse {
    /// Intensity over time per band.
    pub histogram: EnergyHistogram,
    /// Rays traced.
    pub rays: usize,
    /// The most reflections any path made.
    pub deepest_bounce: usize,
    paths: Vec<ReflectionPath>,
    bounce_energy: [f32; 3],
}

impl RoomResponse {
    /// The decay time per band (T30 of the histogram's Schroeder curve, extrapolated to 60 dB).
    #[must_use]
    pub fn rt60(&self) -> [f32; 3] {
        self.histogram.extract_rt60()
    }

    /// The early paths: up to the settings' order and delay.
    #[must_use]
    pub fn paths(&self) -> &[ReflectionPath] {
        &self.paths
    }

    /// Energy per band gathered by next-event estimation shadow rays from bounce points to this
    /// receiver, on the intensity scale of `1 / (4π d²)`.
    #[must_use]
    pub fn bounce_energy(&self) -> [f32; 3] {
        self.bounce_energy
    }
}

/// What a trace saw of the room, enough for its decay time with few bounces: free paths between
/// reflections and the absorption at them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TraceStatistics {
    /// Free paths between two reflections.
    pub segments: usize,
    /// Their summed length, in meters.
    pub length: f64,
    /// Their summed squared length, in square meters.
    pub length_squared: f64,
    /// Reflections.
    pub reflections: usize,
    /// The absorption at the reflections, summed per band.
    pub absorption: [f64; 3],
}

impl TraceStatistics {
    /// Adds `other`'s counts, to average several traces.
    pub fn merge(&mut self, other: &Self) {
        self.segments += other.segments;
        self.length += other.length;
        self.length_squared += other.length_squared;
        self.reflections += other.reflections;
        for band in 0..3 {
            self.absorption[band] += other.absorption[band];
        }
    }

    /// The decay time per band of a diffuse room with these free paths and absorption: Eyring's
    /// formula with Kuttruff's correction for the free paths' spread γ² and with the air's
    /// absorption, `ln(10⁶) l / (c (−ln(1 − α)(1 + γ²/2 ln(1 − α)) + m l))` for a mean free path
    /// `l`. `None` before a free path between two reflections was traced.
    #[must_use]
    pub fn rt60(&self) -> Option<[f32; 3]> {
        if self.segments == 0 || self.reflections == 0 {
            return None;
        }
        let mean = self.length / self.segments as f64;
        let spread = (self.length_squared / self.segments as f64 / (mean * mean) - 1.0).max(0.0);
        Some(std::array::from_fn(|band| {
            let alpha = (self.absorption[band] / self.reflections as f64).clamp(1e-6, 0.999_999);
            let ln = (1.0 - alpha).ln();
            let per_reflection = (-ln * (1.0 + spread / 2.0 * ln)).max(1e-9);
            let rate = per_reflection + f64::from(AIR_ABSORPTION_COEFFS[band]) * mean;
            ((1e6f64).ln() * mean / (f64::from(SPEED_OF_SOUND) * rate)) as f32
        }))
    }
}

/// A seeded Monte Carlo path tracer (see the module documentation).
#[derive(Debug, Clone)]
pub struct PathTracer {
    state: u64,
    statistics: TraceStatistics,
    chains: Vec<SpecularChain>,
}

impl PathTracer {
    /// A tracer whose random sequence starts from `seed`: the same seed traces the same paths.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // splitmix64 spreads every seed over the state; xorshift needs it non-zero.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self {
            state: if z == 0 { 1 } else { z },
            statistics: TraceStatistics::default(),
            chains: Vec::with_capacity(MAX_CHAINS),
        }
    }

    /// The chains of planes the last trace's rays reflected off specularly, from the listener,
    /// up to the settings' early order (at most [`MAX_IMAGE_ORDER`]) and bounce limit: each
    /// once, the first 64 found. Their image sources give the early reflections
    /// ([`SpecularChain::image_path`]).
    #[must_use]
    pub fn chains(&self) -> &[SpecularChain] {
        &self.chains
    }

    /// What the last trace saw of the room.
    #[must_use]
    pub fn statistics(&self) -> TraceStatistics {
        self.statistics
    }

    /// A uniform number in [0, 1) (xorshift64*).
    fn uniform(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }

    /// A direction uniformly distributed over the sphere.
    fn sphere_direction(&mut self) -> Vec3 {
        let z = 1.0 - 2.0 * self.uniform();
        let phi = TAU * self.uniform();
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * phi.cos(), r * phi.sin(), z)
    }

    /// A direction about `normal` distributed as its cosine (Lambert's law).
    fn cosine_direction(&mut self, normal: Vec3) -> Vec3 {
        let tangent = if normal.x.abs() > 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        }
        .cross(normal)
        .normalize();
        let bitangent = normal.cross(tangent);
        let (u, phi) = (self.uniform(), TAU * self.uniform());
        let r = u.sqrt();
        (tangent * (r * phi.cos()) + bitangent * (r * phi.sin()) + normal * (1.0 - u).sqrt())
            .normalize()
    }

    /// Traces `settings.rays` paths from `listener` through `geometry` and returns, for each of
    /// `receivers`, what reached it.
    pub fn trace(
        &mut self,
        geometry: &AcousticGeometry,
        listener: Vec3,
        receivers: &[Vec3],
        settings: &TraceSettings,
    ) -> Vec<RoomResponse> {
        let bins = (settings.duration / settings.bin).ceil().max(1.0) as usize;
        let mut responses: Vec<RoomResponse> = receivers
            .iter()
            .map(|_| RoomResponse {
                histogram: EnergyHistogram::new(settings.bin, bins),
                rays: settings.rays,
                deepest_bounce: 0,
                paths: Vec::new(),
                bounce_energy: [0.0; 3],
            })
            .collect();
        if settings.rays == 0 {
            return responses;
        }
        let mut statistics = TraceStatistics::default();
        self.chains.clear();
        let image_order = settings
            .early_order
            .min(MAX_IMAGE_ORDER)
            .min(settings.max_bounces);
        let radius = settings.receiver_radius.max(1e-3);
        let sphere = 4.0 / 3.0 * PI * radius.powi(3);
        let start = 1.0 / settings.rays as f32;
        let reach = settings.duration * SPEED_OF_SOUND;
        let mut deepest = 0;

        for _ in 0..settings.rays {
            let first = self.sphere_direction();
            let (mut origin, mut direction) = (listener, first);
            let mut energy = [start; 3];
            let (mut length, mut order) = (0.0f32, 0usize);
            // The planes the path has reflected off, while every reflection was specular.
            let mut planes = [ReflectingPlane::default(); MAX_IMAGE_ORDER];
            let mut specular = true;
            loop {
                let ray = AcousticRay { origin, direction };
                let hit = geometry.raycast_closest(&ray, reach - length);
                let segment = hit.map_or(reach - length, |hit| hit.distance);
                for (receiver, response) in receivers.iter().zip(&mut responses) {
                    // The chord the segment cuts through the receiver's sphere.
                    let to = *receiver - origin;
                    let along = to.dot(direction);
                    let miss = to.length_squared() - along * along;
                    if miss >= radius * radius {
                        continue;
                    }
                    let half = (radius * radius - miss).sqrt();
                    let (enter, leave) = ((along - half).max(0.0), (along + half).min(segment));
                    if leave <= enter {
                        continue;
                    }
                    let closest = along.clamp(enter, leave);
                    let delay = (length + closest) / SPEED_OF_SOUND;
                    let weight = (leave - enter) / sphere;
                    let deposited = std::array::from_fn(|band| {
                        energy[band] * (-AIR_ABSORPTION_COEFFS[band] * closest).exp() * weight
                    });
                    response.histogram.add_energy(delay, deposited);
                    if order <= settings.early_order && delay <= settings.early_time {
                        response.paths.push(ReflectionPath {
                            delay,
                            energy: deposited,
                            direction: first,
                            order,
                        });
                    }
                }
                let Some(hit) = hit else {
                    break;
                };
                if order > 0 {
                    // A free path between two reflections (the first starts at the listener).
                    statistics.segments += 1;
                    statistics.length += f64::from(hit.distance);
                    statistics.length_squared += f64::from(hit.distance).powi(2);
                }
                statistics.reflections += 1;
                for band in 0..3 {
                    statistics.absorption[band] += f64::from(hit.material.absorption[band]);
                }
                if specular && order < image_order {
                    planes[order] = ReflectingPlane::through(hit.point, hit.normal);
                    let chain = SpecularChain::new(&planes[..=order]);
                    if self.chains.len() < MAX_CHAINS && !self.chains.iter().any(|c| c.same(&chain))
                    {
                        self.chains.push(chain);
                    }
                }
                length += hit.distance;
                if order == settings.max_bounces || length >= reach {
                    break;
                }
                for (band, value) in energy.iter_mut().enumerate() {
                    *value *= (-AIR_ABSORPTION_COEFFS[band] * hit.distance).exp()
                        * (1.0 - hit.material.absorption[band]).max(0.0);
                }
                order += 1;
                deepest = deepest.max(order);
                // Russian roulette: a weak path survives half the time, with twice the energy.
                if energy.iter().fold(0.0f32, |a, &b| a.max(b)) < start * ROULETTE_LEVEL {
                    if self.uniform() < 0.5 {
                        energy = energy.map(|value| value * 2.0);
                    } else {
                        break;
                    }
                }
                origin = hit.point + hit.normal * SURFACE_OFFSET;
                // Next-event estimation: a shadow ray from each bounce to each receiver gathers
                // energy reflected around corners.
                for (receiver, response) in receivers.iter().zip(&mut responses) {
                    let to_receiver = *receiver - origin;
                    let dist = to_receiver.length();
                    if dist <= SURFACE_OFFSET || length + dist > reach {
                        continue;
                    }
                    let dir = to_receiver / dist;
                    let cos = hit.normal.dot(dir);
                    if cos <= 0.0 {
                        continue;
                    }
                    let shadow = AcousticRay {
                        origin,
                        direction: dir,
                    };
                    if geometry
                        .raycast_closest(&shadow, dist - SURFACE_OFFSET)
                        .is_none()
                    {
                        let total_dist = (length + dist).max(1e-3);
                        let weight = cos / (4.0 * PI * total_dist * total_dist);
                        for band in 0..3 {
                            response.bounce_energy[band] +=
                                energy[band] * (-AIR_ABSORPTION_COEFFS[band] * dist).exp() * weight;
                        }
                    }
                }
                direction = if self.uniform() < hit.material.scattering {
                    specular = false;
                    self.cosine_direction(hit.normal)
                } else {
                    (direction - 2.0 * direction.dot(hit.normal) * hit.normal).normalize()
                };
            }
        }
        self.statistics = statistics;
        for response in &mut responses {
            response.deepest_bounce = deepest;
            response.paths.sort_by(|a, b| a.delay.total_cmp(&b.delay));
        }
        responses
    }
}
