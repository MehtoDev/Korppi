//! World configuration (Box2D `b2WorldDef`) and length units (Box2D `b2SetLengthUnitsPerMeter`).
//!
//! Box2D tunes its tolerances for objects from about 0.1 to 10 metres. A world measured in other
//! units, such as pixels, scales them with `length_units_per_meter`. Box2D's setting is
//! process-wide; here each world has its own, so worlds with different units can coexist.

use crate::distance::LINEAR_SLOP;
use crate::joint::Softness2d;
use crate::solver::RESTITUTION_ITERATIONS;

/// Most restitution passes a world runs (Box2D `B2_MAX_RESTITUTION_ITERATIONS`).
pub const MAX_RESTITUTION_ITERATIONS: usize = 63;
use serde::{Deserialize, Serialize};

/// The settings of a [`PhysicsWorld2d`](crate::world::PhysicsWorld2d), with Box2D's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorldConfig2d {
    /// World length units per metre (Box2D `b2SetLengthUnitsPerMeter`, default 1). It scales
    /// the linear slop, speculative distance, broadphase margin and sleep threshold; the
    /// speed-like defaults of [`Self::with_length_units`] scale with it too.
    pub length_units_per_meter: f32,
    /// Substeps per step (Box2D `subStepCount`, default 4).
    pub sub_steps: u32,
    /// Contact stiffness in Hertz (Box2D `contactHertz`, default 30).
    pub contact_hertz: f32,
    /// Contact damping ratio (Box2D `contactDampingRatio`, default 10).
    pub contact_damping_ratio: f32,
    /// Largest speed at which overlapping bodies are pushed apart (Box2D `contactSpeed`,
    /// default 3 m/s).
    pub contact_push_speed: f32,
    /// Approach speed below which contacts do not bounce (Box2D `restitutionThreshold`,
    /// default 1 m/s).
    pub restitution_threshold: f32,
    /// Passes of the restitution solver after the substeps (Box2D `restitutionIterations`,
    /// default 2, at most [`MAX_RESTITUTION_ITERATIONS`]); more can spin bouncing boxes less, 0
    /// turns restitution off.
    pub restitution_iterations: usize,
    /// Whether the restitution passes also solve the contacts without restitution, so that a
    /// bounce carries through resting contacts (Box2D `enableRestitutionPropagation`, off by
    /// default; costly).
    pub enable_restitution_propagation: bool,
    /// Approach speed above which contacts report hit events (Box2D `hitEventThreshold`,
    /// default 1 m/s).
    pub hit_event_threshold: f32,
    /// Speed limit of every body (Box2D `maximumLinearSpeed`, default 400 m/s).
    pub maximum_linear_speed: f32,
    /// Whether bodies may fall asleep (Box2D `enableSleep`).
    pub enable_sleep: bool,
    /// Whether fast bodies get continuous collision (Box2D `enableContinuous`).
    pub enable_continuous: bool,
    /// Whether contacts and joints start each step from the last step's impulses (Box2D
    /// `b2World_EnableWarmStarting`); turning it off is for debugging.
    pub enable_warm_starting: bool,
    /// How far two bodies may move and turn relative to each other, at their farthest points,
    /// before their contact is collided again instead of recycled (Box2D
    /// `contactRecycleDistance`, default ten linear slops, 0.05 m); 0 turns recycling off.
    pub contact_recycle_distance: f32,
}

impl Default for WorldConfig2d {
    fn default() -> Self {
        Self::with_length_units(1.0)
    }
}

impl WorldConfig2d {
    /// Box2D's defaults for a world measured in `units_per_meter` length units per metre: the
    /// speeds and thresholds are scaled like `b2DefaultWorldDef` after
    /// `b2SetLengthUnitsPerMeter`.
    #[must_use]
    pub const fn with_length_units(units_per_meter: f32) -> Self {
        Self {
            length_units_per_meter: units_per_meter,
            sub_steps: 4,
            contact_hertz: 30.0,
            contact_damping_ratio: 10.0,
            contact_push_speed: 3.0 * units_per_meter,
            restitution_threshold: 1.0 * units_per_meter,
            restitution_iterations: RESTITUTION_ITERATIONS,
            enable_restitution_propagation: false,
            hit_event_threshold: 1.0 * units_per_meter,
            maximum_linear_speed: 400.0 * units_per_meter,
            enable_sleep: true,
            enable_continuous: true,
            enable_warm_starting: true,
            contact_recycle_distance: 10.0 * LINEAR_SLOP * units_per_meter,
        }
    }

    /// Allowed overlap and collision tolerance (Box2D `B2_LINEAR_SLOP`).
    #[must_use]
    pub fn linear_slop(&self) -> f32 {
        LINEAR_SLOP * self.length_units_per_meter
    }

    /// Distance within which contact points are kept before shapes touch (Box2D
    /// `B2_SPECULATIVE_DISTANCE`).
    #[must_use]
    pub fn speculative_distance(&self) -> f32 {
        4.0 * self.linear_slop()
    }

    /// Largest distance a moving collider's broadphase box reaches beyond its shape and the
    /// speculative distance (Box2D `B2_MAX_AABB_MARGIN`); smaller shapes get an eighth of their
    /// size (`B2_AABB_MARGIN_FRACTION`), static ones the speculative distance.
    #[must_use]
    pub fn aabb_margin(&self) -> f32 {
        0.05 * self.length_units_per_meter
    }

    /// Speed below which a body counts as resting for sleep (Box2D `sleepThreshold`).
    #[must_use]
    pub fn sleep_threshold(&self) -> f32 {
        0.05 * self.length_units_per_meter
    }

    /// Contact softness for substep `h`: against moving bodies and, twice as stiff, against
    /// static ones; the stiffness is capped at an eighth of the substep rate (Box2D
    /// `b2PrepareContactsTask`).
    #[must_use]
    pub fn contact_softness(&self, h: f32) -> (Softness2d, Softness2d) {
        crate::solver::contact_softness(self.contact_hertz, self.contact_damping_ratio, h)
    }
}
