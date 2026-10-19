//! The 2D physics world: bodies, colliders and joints behind generational handles, the
//! broadphase, narrowphase and the Soft Step solver.
//!
//! Objects live in arrays indexed by slot ([`Handle::index`]); [`IdPool`]s hand out the
//! handles. A handle goes stale when its object is destroyed, and stays stale after the slot is
//! reused, as Box2D's `b2BodyId`, `b2ShapeId` and `b2JointId` do. A destroyed body's slot holds
//! an inert static body without a collider until it is reused, so the solver never sees it.

pub use crate::body::CONTINUOUS_SAFETY_FACTOR;
use crate::body::{BodyType, RigidBody2d, MAX_ROTATION};
use crate::bvh::DynamicTree2d;
use crate::chain::{Chain2d, ChainDef2d};
use crate::collider::{Collider2d, Filter2d};
pub use crate::config::WorldConfig2d;
use crate::constraint_graph::ConstraintGraph2d;
use crate::contact_manager::{Contact2d, ContactManager2d, NULL};
use crate::diagnostics::{Counters2d, Profile2d};
use crate::distance::{
    shape_distance, time_of_impact, CastOutput2d, RayCastInput2d, ShapeCastInput2d, ShapeProxy2d,
    SimplexCache2d, Sweep2d, ToiInput2d,
};
use crate::events::{
    BodyMoveEvent2d, ContactBeginEvent2d, ContactData2d, ContactEndEvent2d, ContactEvents2d,
    ContactHitEvent2d, JointEvent2d, SensorBeginEvent2d, SensorEndEvent2d, SensorEvents2d,
};
use crate::joint::{Joint2d, JointStep2d};
use crate::mover::PlaneResult2d;
use crate::narrowphase::{collide_shapes_cached, ContactManifold2d};
use crate::parallel::Shared;
use crate::shape::{Capsule2d, MassData2d, Shape2d};
use crate::snapshot::WorldSnapshot2d;
use crate::solver::{ContactMaterial2d, StepFrame2d};
use korppi_core::{parallel_for, Handle, IdPool, TaskExecutor};
use korppi_math::{Aabb2d, Ray2d, Rot2, Transform2d, Vec2};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

mod islands;
mod solve;

pub use islands::TIME_TO_SLEEP;
use islands::{
    BodyLinks2d, Island2d, JointLinks2d, SolverSet2d, SplitScratch2d, AWAKE_SET, DISABLED_SET,
    FIRST_SLEEPING_SET, STATIC_SET,
};
use solve::{SolveData, SolveParams, SolverScratch2d};

/// Handle of a body in a [`PhysicsWorld2d`] (Box2D `b2BodyId`).
pub type BodyId2d = Handle<RigidBody2d>;

/// Handle of a collider in a [`PhysicsWorld2d`] (Box2D `b2ShapeId`).
pub type ColliderId2d = Handle<Collider2d>;

/// Handle of a joint in a [`PhysicsWorld2d`] (Box2D `b2JointId`).
pub type JointId2d = Handle<JointInstance2d>;

/// Handle of a chain in a [`PhysicsWorld2d`] (Box2D `b2ChainId`).
pub type ChainId2d = Handle<Chain2d>;

/// Which colliders a query reports (Box2D `b2QueryFilter`): those whose category is in
/// `mask_bits` and whose mask includes `category_bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryFilter2d {
    /// The categories of the query.
    pub category_bits: u64,
    /// The categories the query reports.
    pub mask_bits: u64,
}

impl Default for QueryFilter2d {
    /// Category 1, reporting everything (Box2D `b2DefaultQueryFilter`).
    fn default() -> Self {
        Self::new(1, u64::MAX)
    }
}

impl QueryFilter2d {
    /// A query filter with these bits.
    #[must_use]
    pub const fn new(category_bits: u64, mask_bits: u64) -> Self {
        Self {
            category_bits,
            mask_bits,
        }
    }

    /// Whether the query reports a collider with `filter` (Box2D `b2ShouldQueryCollide`).
    #[must_use]
    pub const fn accepts(&self, filter: &Filter2d) -> bool {
        (filter.category_bits & self.mask_bits) != 0 && (filter.mask_bits & self.category_bits) != 0
    }
}

/// A hit of a ray or shape cast against the world (Box2D `b2RayResult` and the cast
/// callback's arguments).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit2d {
    /// The hit body.
    pub body_id: BodyId2d,
    /// The hit collider.
    pub collider_id: ColliderId2d,
    /// Hit point in world space.
    pub point: Vec2,
    /// Surface normal at the hit point in world space; zero for a cast that starts inside.
    pub normal: Vec2,
    /// Fraction of the cast's translation at the hit; zero for a cast that starts inside.
    pub fraction: f32,
}

/// An explosion for [`PhysicsWorld2d::explode`] (Box2D `b2ExplosionDef`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExplosionDef2d {
    /// The categories of colliders the explosion pushes.
    pub mask_bits: u64,
    /// Centre of the explosion in world space.
    pub position: Vec2,
    /// Radius of full strength.
    pub radius: f32,
    /// Distance beyond the radius over which the push fades to nothing.
    pub falloff: f32,
    /// Impulse per metre of the shape's perimeter facing the explosion, in N s / m; negative
    /// for an implosion.
    pub impulse_per_length: f32,
}

impl Default for ExplosionDef2d {
    /// Box2D `b2DefaultExplosionDef`: every category, no impulse.
    fn default() -> Self {
        Self {
            mask_bits: u64::MAX,
            position: Vec2::ZERO,
            radius: 0.0,
            falloff: 0.0,
            impulse_per_length: 0.0,
        }
    }
}

/// Instance of an active joint connection between two bodies in the world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointInstance2d {
    /// Body A.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_a: BodyId2d,
    /// Body B.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_b: BodyId2d,
    /// The joint constraint definition.
    pub joint: Joint2d,
    /// Whether the two bodies collide with each other (Box2D `collideConnected`, default
    /// `false`). Read every step, so changing it through [`PhysicsWorld2d::joint_mut`] takes
    /// effect on the next step.
    pub collide_connected: bool,
    /// Force at which the joint reports a [`JointEvent2d`], in newtons (Box2D
    /// `forceThreshold`, default `f32::MAX`: never).
    pub force_threshold: f32,
    /// Torque at which the joint reports a [`JointEvent2d`], in N·m (Box2D `torqueThreshold`,
    /// default `f32::MAX`: never).
    pub torque_threshold: f32,
}

impl JointInstance2d {
    /// Creates a joint between bodies `body_a` and `body_b` that do not collide with each other.
    #[must_use]
    pub const fn new(body_a: BodyId2d, body_b: BodyId2d, joint: Joint2d) -> Self {
        Self {
            body_a,
            body_b,
            joint,
            collide_connected: false,
            force_threshold: f32::MAX,
            torque_threshold: f32::MAX,
        }
    }

    /// Sets the force and torque at which the joint reports events. Zero reports every step
    /// the joint is solved.
    #[must_use]
    pub const fn with_event_thresholds(mut self, force: f32, torque: f32) -> Self {
        self.force_threshold = force;
        self.torque_threshold = torque;
        self
    }

    /// Sets whether the two bodies collide with each other.
    #[must_use]
    pub const fn with_collide_connected(mut self, collide_connected: bool) -> Self {
        self.collide_connected = collide_connected;
        self
    }

    /// The slots of the two bodies.
    const fn slots(&self) -> (u32, u32) {
        (self.body_a.index(), self.body_b.index())
    }
}

/// Main 2D physics simulation world adapting Box2D v3 into pure Rust.
#[derive(Debug, Serialize, Deserialize)]
pub struct PhysicsWorld2d {
    /// Bodies by slot.
    bodies: Vec<RigidBody2d>,
    #[serde(with = "crate::serde_ids::id_pool")]
    body_ids: IdPool<RigidBody2d>,
    /// Each body slot's first collider slot; a body's colliders form a list through
    /// `collider_next`, in creation order (Box2D `headShapeId`, `nextShapeId`).
    body_first_collider: Vec<Option<u32>>,
    /// Colliders by slot.
    colliders: Vec<Option<Collider2d>>,
    /// Each collider slot's body slot.
    collider_bodies: Vec<u32>,
    /// Each collider slot's next collider on the same body.
    collider_next: Vec<Option<u32>>,
    /// Each collider slot's broadphase proxy key (`proxy id * 3 + tree`), absent while its body
    /// is disabled. The trees' user data is the collider slot.
    collider_proxies: Vec<Option<u32>>,
    /// Each collider slot's box at its body's pose, grown by the speculative distance.
    collider_aabbs: Vec<Aabb2d>,
    /// Each collider slot's fat box, the box its proxy has in the tree.
    collider_fat_aabbs: Vec<Aabb2d>,
    /// Each collider slot's chain, if it is a segment of one.
    #[serde(with = "crate::serde_ids::option_handle_vec")]
    collider_chains: Vec<Option<ChainId2d>>,
    #[serde(with = "crate::serde_ids::id_pool")]
    collider_ids: IdPool<Collider2d>,
    /// Chains by slot.
    chains: Vec<Option<Chain2d>>,
    #[serde(with = "crate::serde_ids::id_pool")]
    chain_ids: IdPool<Chain2d>,
    /// One broadphase tree per body type (Box2D keeps static, kinematic and dynamic proxies
    /// apart), indexed by [`tree_of`]. Proxies created, grown or teleported since pairs were
    /// last found carry moved flags in their tree; only they look for new pairs.
    trees: [DynamicTree2d; 3],
    /// Joints by slot.
    joints: Vec<Option<JointInstance2d>>,
    #[serde(with = "crate::serde_ids::id_pool")]
    joint_ids: IdPool<JointInstance2d>,
    gravity: Vec2,
    config: WorldConfig2d,
    step_index: u64,
    contact_manager: ContactManager2d,
    /// Each body slot's solver set, island and sleep threshold.
    body_links: Vec<BodyLinks2d>,
    /// Each joint slot's body slots, fixed when it is created.
    joint_bodies: Vec<(u32, u32)>,
    /// Each joint slot's solver set and island.
    joint_links: Vec<JointLinks2d>,
    /// Solver sets by id: static, disabled, awake, then sleeping sets.
    sets: Vec<SolverSet2d>,
    /// Ids of freed sleeping sets, reused last freed first.
    free_sets: Vec<u32>,
    /// Islands by id.
    islands: Vec<Island2d>,
    /// Ids of freed islands, reused last freed first.
    free_islands: Vec<u32>,
    /// Number of live islands.
    island_count: usize,
    /// The island to split during the next step, the sleepiest of those that need it.
    island_to_split: u32,
    /// The sleepiest island that needs splitting found by the step's finalize, and its time.
    #[serde(skip)]
    split_candidate: (u32, f32),
    #[serde(skip)]
    split_scratch: SplitScratch2d,
    /// Per awake island (by index in the awake set), whether a body keeps it awake this step.
    #[serde(skip)]
    awake_islands: Vec<bool>,
    /// Bodies handed out by `body_mut` since the last step.
    touched_bodies: Vec<u32>,
    /// Contacts recycled by the last step.
    recycled_contacts: usize,
    /// The step's contacts that started or stopped touching or separated, by slot.
    #[serde(skip)]
    contact_changes: Vec<(u32, ContactChange)>,
    /// The awake contacts the step's narrowphase updates, and what it did to each.
    #[serde(skip)]
    collide_slots: Vec<u32>,
    #[serde(skip)]
    collide_outcomes: Vec<CollideOutcome>,
    constraint_graph: ConstraintGraph2d,
    /// The step's new collider pairs.
    #[serde(skip)]
    contact_pairs: Vec<(u32, u32)>,
    /// Per-worker buffers for parallel `moved_pairs`.
    #[serde(skip)]
    worker_contact_pairs: Vec<Vec<(u32, u32)>>,
    #[serde(skip)]
    candidates_scratch: Vec<u32>,
    /// The step's fast bullets, swept after every other body has moved.
    #[serde(skip)]
    fast_bullets: Vec<u32>,
    /// Sorted body pairs whose collisions a joint disables, rebuilt every step.
    #[serde(skip)]
    joint_filter_pairs: Vec<(u32, u32)>,
    /// Each awake body's pose at the start of the step and its motion during it, by slot, for
    /// continuous collision.
    #[serde(skip)]
    step_frames: Vec<StepFrame2d>,
    /// The solver's buffers.
    #[serde(skip)]
    solver: SolverScratch2d,
    /// Runs the step's parallel work; `None` steps on the calling thread.
    #[serde(skip)]
    executor: Option<Executor2d>,
    /// The last step's contact events.
    contact_events: ContactEvents2d,
    /// End events of contacts destroyed by calls since the last step, reported with the next.
    pending_end_events: Vec<ContactEndEvent2d>,
    /// The last step's body move events.
    body_events: Vec<BodyMoveEvent2d>,
    /// Each sensor collider slot's overlapping colliders after the last step, sorted.
    #[serde(with = "crate::serde_ids::nested_handle_vec")]
    sensor_overlaps: Vec<Vec<ColliderId2d>>,
    /// Live collider slots that have `is_sensor` set, cached so `overlap_sensors` skips scanning
    /// non-sensor colliders.
    #[serde(skip)]
    sensor_slots: Vec<u32>,
    /// Whether `sensor_slots` is up to date (`false` after deserializing or `collider_mut`).
    #[serde(skip)]
    sensors_cached: bool,
    /// The step's new overlaps of the sensor being processed.
    #[serde(skip)]
    sensor_scratch: Vec<ColliderId2d>,
    /// Sensors crossed by fast bodies during the step's continuous collision: (sensor collider
    /// slot, visitor).
    #[serde(skip)]
    sensor_hits: Vec<(u32, ColliderId2d)>,
    /// The last step's sensor events.
    sensor_events: SensorEvents2d,
    /// Sensor end events of calls since the last step, reported with the next.
    pending_sensor_end_events: Vec<SensorEndEvent2d>,
    /// The last step's joint events.
    joint_events: Vec<JointEvent2d>,
    /// Per joint slot, whether the joint reached a threshold during the step.
    #[serde(skip)]
    joint_thresholds_reached: Vec<bool>,
    /// Each joint slot's data for the step, prepared before the substeps.
    joint_steps: Vec<JointStep2d>,
    /// The last step's substep rate (Box2D `world->inv_h`), which turns joint impulses into
    /// forces.
    inv_h: f32,
    /// Called with each new manifold of colliders that enable pre-solve events.
    #[serde(skip)]
    pre_solve: Option<PreSolve2d>,
    /// Called with each continuous-collision impact of colliders that enable pre-solve events.
    #[serde(skip)]
    pre_continuous: Option<PreContinuous2d>,
    /// Decides the pairs of colliders that enable custom filtering.
    #[serde(skip)]
    custom_filter: Option<CustomFilter2d>,
    /// Where the last step spent its time.
    #[serde(skip)]
    profile: Profile2d,
    /// Mixes two colliders' friction; `None` takes the geometric mean.
    #[serde(skip)]
    friction_mixer: Option<MaterialMixer2d>,
    /// Mixes two colliders' restitution; `None` takes the larger.
    #[serde(skip)]
    restitution_mixer: Option<MaterialMixer2d>,
}

/// Mixes a property of two colliders from their values and user material ids (Box2D
/// `b2FrictionCallback`, `b2RestitutionCallback`).
pub type MaterialMixer2d = fn(f32, u64, f32, u64) -> f32;

/// A custom filter: whether two colliders may collide (Box2D `b2CustomFilterFcn`).
type CustomFilterFn = dyn FnMut(ColliderId2d, ColliderId2d) -> bool + Send;

/// The world's custom filter.
struct CustomFilter2d(Box<CustomFilterFn>);

/// The world's task executor.
#[derive(Clone)]
struct Executor2d(Arc<dyn TaskExecutor>);

impl std::fmt::Debug for Executor2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor2d")
            .field("worker_count", &self.0.worker_count())
            .finish()
    }
}

impl std::fmt::Debug for CustomFilter2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CustomFilter2d")
    }
}

/// Whether colliders `a` and `b` (slots and colliders) may collide: their filters, then the
/// custom filter if either enables custom filtering (Box2D `b2ShouldShapesCollide` and
/// `b2CustomFilterFcn`).
fn colliders_may_collide(
    custom_filter: &mut Option<CustomFilter2d>,
    collider_ids: &IdPool<Collider2d>,
    (slot_a, a): (u32, &Collider2d),
    (slot_b, b): (u32, &Collider2d),
) -> bool {
    if !a.filter.should_collide(&b.filter) {
        return false;
    }
    let (Some(CustomFilter2d(custom)), true) = (
        custom_filter.as_mut(),
        a.enable_custom_filtering || b.enable_custom_filtering,
    ) else {
        return true;
    };
    match (
        collider_ids.handle_at(slot_a),
        collider_ids.handle_at(slot_b),
    ) {
        (Some(id_a), Some(id_b)) => custom(id_a, id_b),
        _ => true,
    }
}

/// The world's pre-continuous callback.
struct PreContinuous2d(Box<PreContinuousFn>);

impl std::fmt::Debug for PreContinuous2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PreContinuous2d")
    }
}

/// A pre-solve callback: the two colliders and their new manifold (Box2D `b2PreSolveFcn`).
type PreSolveFn = dyn FnMut(ColliderId2d, ColliderId2d, &mut ContactManifold2d) + Send;

/// A pre-continuous callback: the obstacle's collider, the fast collider, the impact point and
/// normal; `false` ignores the impact (Box2D `b2PreContinuousFcn`).
type PreContinuousFn = dyn FnMut(ColliderId2d, ColliderId2d, Vec2, Vec2) -> bool + Send;

/// The world's pre-solve callback.
struct PreSolve2d(Box<PreSolveFn>);

impl std::fmt::Debug for PreSolve2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PreSolve2d")
    }
}

impl Default for PhysicsWorld2d {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld2d {
    /// Creates a new 2D physics simulation world with standard gravity (0, -9.81).
    #[must_use]
    pub fn new() -> Self {
        Self {
            bodies: Vec::new(),
            body_ids: IdPool::new(),
            body_first_collider: Vec::new(),
            colliders: Vec::new(),
            collider_bodies: Vec::new(),
            collider_next: Vec::new(),
            collider_proxies: Vec::new(),
            collider_aabbs: Vec::new(),
            collider_fat_aabbs: Vec::new(),
            collider_chains: Vec::new(),
            collider_ids: IdPool::new(),
            chains: Vec::new(),
            chain_ids: IdPool::new(),
            trees: Default::default(),
            joints: Vec::new(),
            joint_ids: IdPool::new(),
            gravity: Vec2::new(0.0, -9.81),
            config: WorldConfig2d::default(),
            step_index: 0,
            contact_manager: ContactManager2d::new(),
            body_links: Vec::new(),
            joint_bodies: Vec::new(),
            joint_links: Vec::new(),
            // The static, disabled and awake sets always exist.
            sets: vec![
                SolverSet2d::default(),
                SolverSet2d::default(),
                SolverSet2d::default(),
            ],
            free_sets: Vec::new(),
            islands: Vec::new(),
            free_islands: Vec::new(),
            island_count: 0,
            island_to_split: NULL,
            split_candidate: (NULL, 0.0),
            split_scratch: SplitScratch2d::default(),
            awake_islands: Vec::new(),
            touched_bodies: Vec::new(),
            recycled_contacts: 0,
            contact_changes: Vec::new(),
            collide_slots: Vec::new(),
            collide_outcomes: Vec::new(),
            constraint_graph: ConstraintGraph2d::default(),
            contact_pairs: Vec::new(),
            worker_contact_pairs: Vec::new(),
            candidates_scratch: Vec::new(),
            fast_bullets: Vec::new(),
            joint_filter_pairs: Vec::new(),
            step_frames: Vec::new(),
            solver: SolverScratch2d::default(),
            executor: None,
            contact_events: ContactEvents2d::default(),
            pending_end_events: Vec::new(),
            body_events: Vec::new(),
            sensor_overlaps: Vec::new(),
            sensor_slots: Vec::new(),
            sensors_cached: true,
            sensor_scratch: Vec::new(),
            sensor_hits: Vec::new(),
            sensor_events: SensorEvents2d::default(),
            pending_sensor_end_events: Vec::new(),
            joint_events: Vec::new(),
            joint_thresholds_reached: Vec::new(),
            joint_steps: Vec::new(),
            inv_h: 0.0,
            pre_solve: None,
            pre_continuous: None,
            custom_filter: None,
            profile: Profile2d::default(),
            friction_mixer: None,
            restitution_mixer: None,
        }
    }

    /// Sets the gravity acceleration vector in builder style.
    #[must_use]
    pub const fn with_gravity(mut self, gravity: Vec2) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the global gravity acceleration vector.
    pub fn set_gravity(&mut self, gravity: Vec2) {
        self.gravity = gravity;
    }

    /// Returns the current global gravity vector.
    #[must_use]
    pub const fn gravity(&self) -> Vec2 {
        self.gravity
    }

    /// Sets the world configuration in builder style.
    #[must_use]
    pub fn with_config(mut self, config: WorldConfig2d) -> Self {
        self.config = config;
        self
    }

    /// The world configuration.
    #[must_use]
    pub const fn config(&self) -> &WorldConfig2d {
        &self.config
    }

    /// The world configuration, mutably. Changes take effect on the next step; set the length
    /// units before creating bodies.
    pub fn config_mut(&mut self) -> &mut WorldConfig2d {
        &mut self.config
    }

    /// Makes room for `additional` more contacts (Box2D `b2WorldDef::capacity`). Contacts live
    /// while the fat boxes of two colliders overlap, so their number peaks when a scene's motions
    /// line up; reserving that peak keeps steps from allocating when it first happens.
    pub fn reserve_contacts(&mut self, additional: usize) {
        self.contact_manager.reserve(additional);
        self.contact_pairs.reserve(additional);
        for buf in &mut self.worker_contact_pairs {
            buf.reserve(additional);
        }
        let total = self.contact_manager.slot_count() + additional;
        let room = |len: usize| total.saturating_sub(len);
        self.constraint_graph.reserve_contacts(additional);
        self.contact_changes
            .reserve(room(self.contact_changes.len()));
        self.collide_slots.reserve(room(self.collide_slots.len()));
        self.collide_outcomes
            .reserve(room(self.collide_outcomes.len()));
        self.solver.reserve_contacts(total);
        for set in [AWAKE_SET, DISABLED_SET] {
            let contacts = &mut self.sets[set as usize].contacts;
            contacts.reserve(room(contacts.len()));
        }
    }

    /// Lets bodies fall asleep or not (Box2D `b2World_EnableSleeping`); turning sleep off wakes
    /// every sleeping body.
    pub fn enable_sleep(&mut self, enable: bool) {
        self.config.enable_sleep = enable;
        if !enable {
            self.wake_sleeping_bodies();
        }
    }

    /// Object counts (Box2D `b2World_GetCounters`).
    #[must_use]
    pub fn counters(&self) -> Counters2d {
        Counters2d {
            body_count: self.body_ids.len(),
            awake_body_count: self.sets[AWAKE_SET as usize].bodies.len(),
            island_count: self.island_count,
            recycled_contact_count: self.recycled_contacts,
            awake_contact_count: self.constraint_graph.contact_count()
                + self.sets[AWAKE_SET as usize].contacts.len(),
            collider_count: self.collider_ids.len(),
            contact_count: self.contact_manager.len(),
            joint_count: self.joint_ids.len(),
            tree_height: self.trees[DYNAMIC_TREE]
                .height()
                .max(self.trees[KINEMATIC_TREE].height()),
            static_tree_height: self.trees[STATIC_TREE].height(),
            color_count: self.constraint_graph.color_count(),
            color_counts: std::array::from_fn(|i| {
                if i < self.constraint_graph.color_count() {
                    self.constraint_graph.color(i).len()
                } else {
                    0
                }
            }),
        }
    }

    /// Where the last step spent its time (Box2D `b2World_GetProfile`).
    #[must_use]
    pub const fn profile(&self) -> &Profile2d {
        &self.profile
    }

    /// Wakes every sleeping set.
    fn wake_sleeping_bodies(&mut self) {
        for set in FIRST_SLEEPING_SET..self.sets.len() as u32 {
            if !self.sets[set as usize].bodies.is_empty() {
                self.wake_set(set);
            }
        }
    }

    /// Returns the current simulation step counter.
    #[must_use]
    pub const fn step_index(&self) -> u64 {
        self.step_index
    }

    /// The contact events of the last step: begin and hit events from the step, end events from
    /// the step and from the calls made before it (Box2D `b2World_GetContactEvents`). Only
    /// contacts with a collider that enables contact (or hit) events report them.
    #[must_use]
    pub const fn contact_events(&self) -> &ContactEvents2d {
        &self.contact_events
    }

    /// The sensor events of the last step: begin events from the step, end events from the step
    /// and from the calls made before it (Box2D `b2World_GetSensorEvents`).
    #[must_use]
    pub const fn sensor_events(&self) -> &SensorEvents2d {
        &self.sensor_events
    }

    /// The colliders overlapping sensor `id` after the last step, ordered by handle (Box2D
    /// `b2Shape_GetSensorData`); some may have been destroyed since. Empty for a stale handle
    /// or a collider that is not a sensor.
    pub fn sensor_overlaps(&self, id: ColliderId2d) -> impl Iterator<Item = ColliderId2d> + '_ {
        let overlaps = if self.collider_ids.contains(id) {
            self.sensor_overlaps[id.index() as usize].as_slice()
        } else {
            &[]
        };
        overlaps.iter().copied()
    }

    /// The joints whose force or torque reached their threshold during the last step, in slot
    /// order (Box2D `b2World_GetJointEvents`).
    #[must_use]
    pub fn joint_events(&self) -> &[JointEvent2d] {
        &self.joint_events
    }

    /// Sets the pre-solve callback (Box2D `b2World_SetPreSolveCallback`). Each step it receives
    /// the new manifold of every contact with a collider that enables pre-solve events, with the
    /// normal pointing from the first collider to the second, and may change it; leaving no
    /// points disables the contact for the step. It runs inside `step`, so it cannot change
    /// the world.
    pub fn set_pre_solve(
        &mut self,
        callback: impl FnMut(ColliderId2d, ColliderId2d, &mut ContactManifold2d) + Send + 'static,
    ) {
        self.pre_solve = Some(PreSolve2d(Box::new(callback)));
    }

    /// Removes the pre-solve callback.
    pub fn clear_pre_solve(&mut self) {
        self.pre_solve = None;
    }

    /// Sets the pre-continuous callback (Box2D `b2PreContinuousFcn`). When continuous collision
    /// would stop a fast body at an impact involving a collider that enables pre-solve events,
    /// it receives the obstacle's collider, the fast collider, the impact point and the normal
    /// from the obstacle to the fast collider; returning `false` ignores the impact, as for a
    /// one-way platform.
    pub fn set_pre_continuous(
        &mut self,
        callback: impl FnMut(ColliderId2d, ColliderId2d, Vec2, Vec2) -> bool + Send + 'static,
    ) {
        self.pre_continuous = Some(PreContinuous2d(Box::new(callback)));
    }

    /// Removes the pre-continuous callback.
    pub fn clear_pre_continuous(&mut self) {
        self.pre_continuous = None;
    }

    /// Sets the custom filter (Box2D `b2World_SetCustomFilterCallback`). For pairs where a
    /// collider enables custom filtering and the filters allow a collision, it decides whether
    /// the colliders collide: contacts, continuous collision and sensor overlaps all ask it. It
    /// runs inside `step` for every candidate pair, so it should be cheap and must not depend
    /// on being called once per pair.
    pub fn set_custom_filter(
        &mut self,
        callback: impl FnMut(ColliderId2d, ColliderId2d) -> bool + Send + 'static,
    ) {
        self.custom_filter = Some(CustomFilter2d(Box::new(callback)));
    }

    /// Removes the custom filter.
    pub fn clear_custom_filter(&mut self) {
        self.custom_filter = None;
    }

    /// Steps the world on `executor`'s workers, or on the calling thread with `None` (Box2D's
    /// `workerCount`, `enqueueTask` and `finishTask`). The solver's stages and their blocks
    /// spread over the workers; a step's result does not depend on the executor or its worker
    /// count. Like the callbacks, the executor is host wiring: an in-place restore keeps it,
    /// and snapshots do not hold it. `korppi_core::WorkerPool` is a ready executor.
    pub fn set_executor(&mut self, executor: Option<Arc<dyn TaskExecutor>>) {
        let workers = executor.as_ref().map_or(0, |e| e.worker_count());
        let min_cap = self.contact_pairs.capacity().max(64);
        if self.worker_contact_pairs.len() < workers {
            self.worker_contact_pairs
                .resize_with(workers, || Vec::with_capacity(min_cap));
        }
        for buf in &mut self.worker_contact_pairs {
            if buf.capacity() < min_cap {
                buf.reserve(min_cap - buf.len());
            }
        }
        self.executor = executor.map(Executor2d);
    }

    /// The world's executor, if it has one.
    #[must_use]
    pub fn executor(&self) -> Option<&Arc<dyn TaskExecutor>> {
        self.executor.as_ref().map(|e| &e.0)
    }

    /// Sets how contacts mix their colliders' friction (Box2D `b2WorldDef::frictionCallback`);
    /// `None` takes `sqrt(a * b)`. The mixer receives each friction and user material id.
    pub fn set_friction_mixer(&mut self, mixer: Option<MaterialMixer2d>) {
        self.friction_mixer = mixer;
    }

    /// Sets how contacts mix their colliders' restitution (Box2D
    /// `b2WorldDef::restitutionCallback`); `None` takes `max(a, b)`.
    pub fn set_restitution_mixer(&mut self, mixer: Option<MaterialMixer2d>) {
        self.restitution_mixer = mixer;
    }

    /// A move event for every body that was awake during the last step, in slot order (Box2D
    /// `b2World_GetBodyEvents`).
    #[must_use]
    pub fn body_events(&self) -> &[BodyMoveEvent2d] {
        &self.body_events
    }

    /// The touching contacts of body `id`, newest first (Box2D `b2Body_GetContactData`).
    pub fn body_contacts(&self, id: BodyId2d) -> impl Iterator<Item = ContactData2d> + '_ {
        let body = self.body_ids.contains(id).then_some(id.index());
        self.contact_data(body, |_| true)
    }

    /// The touching contacts of collider `id`, newest first (Box2D
    /// `b2Shape_GetContactData`).
    pub fn collider_contacts(&self, id: ColliderId2d) -> impl Iterator<Item = ContactData2d> + '_ {
        let slot = self.collider_ids.contains(id).then_some(id.index());
        let body = slot.map(|c| self.collider_bodies[c as usize]);
        self.contact_data(body, move |c| {
            Some(c.collider_a) == slot || Some(c.collider_b) == slot
        })
    }

    /// The touching contacts in body slot `body`'s list that `selected` picks.
    fn contact_data(
        &self,
        body: Option<u32>,
        selected: impl Fn(&Contact2d) -> bool + 'static,
    ) -> impl Iterator<Item = ContactData2d> + '_ {
        body.into_iter()
            .flat_map(|body| self.contact_manager.body_contacts(body))
            .map(|slot| self.contact_manager.at(slot))
            .filter(move |c| c.manifold.point_count > 0 && selected(c))
            .filter_map(|c| {
                Some(ContactData2d {
                    collider_a: self.collider_ids.handle_at(c.collider_a)?,
                    collider_b: self.collider_ids.handle_at(c.collider_b)?,
                    manifold: c.manifold,
                })
            })
    }

    /// Number of live bodies.
    #[must_use]
    pub const fn body_count(&self) -> usize {
        self.body_ids.len()
    }

    /// Adds a body to the world (Box2D `b2CreateBody`). A dynamic or kinematic body gets an
    /// island of its own; one created with `is_sleeping` (and sleep enabled) starts asleep in a
    /// sleeping set of its own, at rest.
    pub fn create_body(&mut self, mut body: RigidBody2d) -> BodyId2d {
        let id = self.body_ids.alloc();
        let slot = id.index();
        let asleep = body.is_sleeping && body.enable_sleep;
        body.is_sleeping = false;
        let set = if !body.enabled {
            DISABLED_SET
        } else if body.body_type == BodyType::Static {
            STATIC_SET
        } else if asleep {
            self.alloc_set()
        } else {
            AWAKE_SET
        };
        let links = BodyLinks2d::new(self.config.sleep_threshold(), body.body_type);
        if slot as usize == self.bodies.len() {
            self.bodies.push(body);
            self.body_first_collider.push(None);
            self.body_links.push(links);
        } else {
            self.bodies[slot as usize] = body;
            self.body_links[slot as usize] = links;
        }
        self.sync_pose(slot);
        self.add_body_to_set(slot, set);
        if set >= AWAKE_SET {
            self.create_island_for_body(set, slot);
        }
        if set >= FIRST_SLEEPING_SET {
            let body = &mut self.bodies[slot as usize];
            body.is_sleeping = true;
            body.linear_velocity = Vec2::ZERO;
            body.angular_velocity = 0.0;
        }
        id
    }

    /// Destroys a body with its collider, its contacts and its joints (Box2D `b2DestroyBody`).
    /// Bodies it touched or was jointed to wake up. Returns `false`, changing nothing, if `id`
    /// is stale.
    pub fn destroy_body(&mut self, id: BodyId2d) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        let slot = id.index();
        let attached: Vec<JointId2d> = self
            .joints()
            .filter(|(_, joint)| joint.body_a == id || joint.body_b == id)
            .map(|(joint_id, _)| joint_id)
            .collect();
        for joint_id in attached {
            self.destroy_joint(joint_id);
        }
        self.destroy_body_contacts(slot);
        while let Some(collider) = self.collider_of(id) {
            self.remove_collider(collider);
        }
        // Its chains go with it (Box2D `b2DestroyBody`).
        for chain_slot in 0..self.chains.len() {
            if self.chains[chain_slot]
                .as_ref()
                .is_some_and(|chain| chain.body == id)
            {
                self.chains[chain_slot] = None;
                if let Some(chain_id) = self.chain_ids.handle_at(chain_slot as u32) {
                    self.chain_ids.free(chain_id);
                }
            }
        }
        self.remove_body_from_island(slot);
        self.remove_body_from_set(slot);
        self.bodies[slot as usize] = RigidBody2d::new(BodyType::Static);
        self.body_links[slot as usize].body_type = BodyType::Static;
        self.body_links[slot as usize].touched = false;
        self.body_ids.free(id);
        true
    }

    /// The live bodies and their handles, in slot order.
    pub fn bodies(&self) -> impl Iterator<Item = (BodyId2d, &RigidBody2d)> + '_ {
        self.body_ids
            .iter()
            .map(|id| (id, &self.bodies[id.index() as usize]))
    }

    /// Returns the body `id`, or `None` if the handle is stale.
    #[must_use]
    pub fn body(&self, id: BodyId2d) -> Option<&RigidBody2d> {
        self.body_ids
            .contains(id)
            .then(|| &self.bodies[id.index() as usize])
    }

    /// Returns the body `id` mutably, waking it first as Box2D's setters do, or `None` if the
    /// handle is stale. The world keeps `is_sleeping`, `enabled` and `body_type` in step with
    /// its solver sets: change them with [`Self::set_awake`], [`Self::disable_body`],
    /// [`Self::enable_body`] and [`Self::set_body_type`], and teleport with
    /// [`Self::set_transform`].
    pub fn body_mut(&mut self, id: BodyId2d) -> Option<&mut RigidBody2d> {
        if !self.body_ids.contains(id) {
            return None;
        }
        self.wake_body(id.index());
        let slot = id.index();
        if !self.body_links[slot as usize].touched {
            self.body_links[slot as usize].touched = true;
            self.touched_bodies.push(slot);
        }
        Some(&mut self.bodies[slot as usize])
    }

    /// Wakes body `id` with the bodies it sleeps with, or puts its island to sleep at once,
    /// splitting it first if it needs it (Box2D `b2Body_SetAwake`). Returns `false` if `id` is
    /// stale.
    pub fn set_awake(&mut self, id: BodyId2d, awake: bool) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        self.set_awake_internal(id, awake)
    }

    fn set_awake_internal(&mut self, id: BodyId2d, awake: bool) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        let slot = id.index();
        let BodyLinks2d { set, island, .. } = self.body_links[slot as usize];
        if awake {
            self.wake_body(slot);
        } else if set == AWAKE_SET && island != NULL {
            if self.islands[island as usize].constraint_remove_count > 0 {
                self.split_island(island);
            }
            let island = self.body_links[slot as usize].island;
            self.try_sleep_island(island);
        }
        true
    }

    /// Wakes the bodies touching body `id` (Box2D `b2Body_WakeTouching`). Returns `false` if
    /// `id` is stale.
    pub fn wake_touching(&mut self, id: BodyId2d) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        let mut key = self.contact_manager.body_head(id.index());
        while key != NULL {
            let contact = self.contact_manager.at(key >> 1);
            let end = (key & 1) as usize;
            key = contact.edges[end].next;
            let other = contact.edges[end ^ 1].body;
            self.wake_body(other);
        }
        true
    }

    /// Sets the speed under which body `id` counts as resting: the speed of its fastest point,
    /// in length units per second (Box2D `b2Body_SetSleepThreshold`; 0.05 m/s by default,
    /// scaled by the world's length units). Returns `false` if `id` is stale.
    pub fn set_sleep_threshold(&mut self, id: BodyId2d, threshold: f32) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.body_links[id.index() as usize].sleep_threshold = threshold;
        true
    }

    /// The speed under which body `id` counts as resting, or `None` if the handle is stale.
    #[must_use]
    pub fn sleep_threshold(&self, id: BodyId2d) -> Option<f32> {
        self.body_ids
            .contains(id)
            .then(|| self.body_links[id.index() as usize].sleep_threshold)
    }

    /// Changes a body's type (Box2D `b2Body_SetType`): its contacts are destroyed, it and the
    /// bodies it touched or is jointed to wake up, it joins the static or awake set and gains
    /// or loses its island, its joints are relinked, and its mass is recomputed from its
    /// colliders (static and kinematic bodies are massless). A body that becomes or stops being
    /// static starts from rest; between dynamic and kinematic it keeps its velocity. A disabled
    /// body only changes type. Returns `false` if `id` is stale.
    pub fn set_body_type(&mut self, id: BodyId2d, body_type: BodyType) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        self.set_body_type_internal(id, body_type)
    }

    fn set_body_type_internal(&mut self, id: BodyId2d, body_type: BodyType) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        let slot = id.index();
        let old_type = self.body_links[slot as usize].body_type;
        if old_type == body_type {
            self.bodies[slot as usize].body_type = body_type;
            return true;
        }
        self.body_links[slot as usize].body_type = body_type;
        if self.body_links[slot as usize].set == DISABLED_SET {
            self.bodies[slot as usize].body_type = body_type;
            self.update_mass(slot);
            return true;
        }
        self.destroy_body_contacts(slot);
        self.wake_body(slot);
        // Park the joints in the static set, unlinked, waking their bodies.
        let attached = self.attached_joints(slot);
        for &joint in &attached {
            if self.joint_links[joint as usize].set == DISABLED_SET {
                continue;
            }
            let (a, b) = self.joint_bodies[joint as usize];
            self.wake_body(a);
            self.wake_body(b);
            self.unlink_joint(joint);
            self.transfer_joint(joint, STATIC_SET);
        }

        let body = &mut self.bodies[slot as usize];
        body.body_type = body_type;
        if old_type == BodyType::Static || body_type == BodyType::Static {
            body.linear_velocity = Vec2::ZERO;
            body.angular_velocity = 0.0;
        }
        let target = if body_type == BodyType::Static {
            STATIC_SET
        } else {
            AWAKE_SET
        };
        self.transfer_body(slot, target);
        if old_type == BodyType::Static {
            self.create_island_for_body(AWAKE_SET, slot);
        } else if body_type == BodyType::Static {
            self.remove_body_from_island(slot);
        }
        for &joint in &attached {
            if self.joint_links[joint as usize].set == DISABLED_SET {
                continue;
            }
            let (a, b) = self.joint_bodies[joint as usize];
            if self.bodies[a as usize].body_type == BodyType::Dynamic
                || self.bodies[b as usize].body_type == BodyType::Dynamic
            {
                self.transfer_joint(joint, AWAKE_SET);
            }
        }
        // The proxies move to the new type's tree and look for pairs again.
        self.destroy_body_proxies(slot);
        self.create_body_proxies(slot);
        for &joint in &attached {
            let (a, b) = self.joint_bodies[joint as usize];
            let other = if a == slot { b } else { a };
            if self.body_links[other as usize].set == DISABLED_SET
                || (self.bodies[slot as usize].body_type != BodyType::Dynamic
                    && self.bodies[other as usize].body_type != BodyType::Dynamic)
            {
                continue;
            }
            self.link_joint(joint);
        }
        self.update_mass(slot);
        true
    }

    /// Takes a body out of the simulation (Box2D `b2Body_Disable`): its contacts are destroyed,
    /// waking the bodies it touched, its broadphase proxies are removed, it leaves its island,
    /// and its joints wait in the disabled set. The body keeps its pose and is at rest until
    /// enabled. Returns `false` if `id` is stale.
    pub fn disable_body(&mut self, id: BodyId2d) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        let slot = id.index();
        if self.body_links[slot as usize].set == DISABLED_SET {
            return true;
        }
        self.destroy_body_contacts(slot);
        for joint in self.attached_joints(slot) {
            if self.joint_links[joint as usize].set == DISABLED_SET {
                continue;
            }
            self.unlink_joint(joint);
            self.transfer_joint(joint, DISABLED_SET);
        }
        self.destroy_body_proxies(slot);
        self.remove_body_from_island(slot);
        self.transfer_body(slot, DISABLED_SET);
        let body = &mut self.bodies[slot as usize];
        body.enabled = false;
        body.is_sleeping = false;
        body.linear_velocity = Vec2::ZERO;
        body.angular_velocity = 0.0;
        true
    }

    /// Returns a disabled body to the simulation (Box2D `b2Body_Enable`), awake and at rest, with
    /// its broadphase proxies, an island and its joints; a sleeping body jointed to it wakes.
    /// Returns `false` if `id` is stale.
    pub fn enable_body(&mut self, id: BodyId2d) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        let slot = id.index();
        if self.body_links[slot as usize].set != DISABLED_SET {
            return true;
        }
        let body = &mut self.bodies[slot as usize];
        body.enabled = true;
        body.sleep_time = 0.0;
        let set = if body.body_type == BodyType::Static {
            STATIC_SET
        } else {
            AWAKE_SET
        };
        self.transfer_body(slot, set);
        self.create_body_proxies(slot);
        if set != STATIC_SET {
            self.create_island_for_body(set, slot);
        }
        for joint in self.attached_joints(slot) {
            let (a, b) = self.joint_bodies[joint as usize];
            let (set_a, set_b) = (
                self.body_links[a as usize].set,
                self.body_links[b as usize].set,
            );
            if set_a == DISABLED_SET || set_b == DISABLED_SET {
                continue;
            }
            let joint_set = if self.bodies[a as usize].body_type != BodyType::Dynamic
                && self.bodies[b as usize].body_type != BodyType::Dynamic
            {
                STATIC_SET
            } else if set_a == STATIC_SET {
                set_b
            } else {
                set_a
            };
            self.transfer_joint(joint, joint_set);
            if joint_set != STATIC_SET {
                self.link_joint(joint);
            }
        }
        true
    }

    /// The joint slots attached to body slot `slot`, in slot order.
    fn attached_joints(&self, slot: u32) -> Vec<u32> {
        (0..self.joints.len() as u32)
            .filter(|&j| {
                self.joints[j as usize].is_some() && {
                    let (a, b) = self.joint_bodies[j as usize];
                    a == slot || b == slot
                }
            })
            .collect()
    }

    /// Teleports a body (Box2D `b2Body_SetTransform`), keeping its velocity and updating the
    /// broadphase. It does not wake the body. Returns `false` if `id` is stale.
    pub fn set_transform(&mut self, id: BodyId2d, position: Vec2, rotation: Rot2) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        let slot = id.index() as usize;
        let body = &mut self.bodies[slot];
        body.position = position;
        body.rotation = rotation;
        self.sync_pose(id.index());
        self.refresh_body_proxies(id.index(), ProxyUpdate::Reinsert);
        true
    }

    /// Records body slot `slot`'s pose as the world's and takes its centre of mass from it
    /// (Box2D `bodySim->center = b2TransformPoint(transform, localCenter)`).
    fn sync_pose(&mut self, slot: u32) {
        let body = &self.bodies[slot as usize];
        let links = &mut self.body_links[slot as usize];
        links.center = body.world_center();
        links.pose = body.transform();
    }

    /// Reconciles any bodies (and their colliders) handed out by [`Self::body_mut`] or
    /// [`Self::collider_mut`] since the last flush.
    fn flush_touched_bodies(&mut self) {
        if self.touched_bodies.is_empty() {
            return;
        }
        let mut touched = std::mem::take(&mut self.touched_bodies);
        for &slot in &touched {
            if self.body_ids.handle_at(slot).is_some() {
                self.body_touched(slot);
                if self.body_links[slot as usize].set != AWAKE_SET {
                    self.bodies[slot as usize].clear_forces();
                }
            }
        }
        touched.clear();
        self.touched_bodies = touched;
    }

    /// A body handed out by `body_mut` or `collider_mut` since the last flush: reconciles direct
    /// field mutations (`body_type`, `mass`, `rotational_inertia`, `motion_locks`, `is_sleeping`),
    /// updates its pose and broadphase proxies if moved, and invalidates contact anchors if its
    /// centre of mass moved.
    fn body_touched(&mut self, slot: u32) {
        self.body_links[slot as usize].touched = false;
        let new_type = self.bodies[slot as usize].body_type;
        let old_type = self.body_links[slot as usize].body_type;
        if new_type != old_type {
            let user_mass = self.bodies[slot as usize].mass_data();
            self.bodies[slot as usize].body_type = old_type;
            if let Some(id) = self.body_ids.handle_at(slot) {
                self.set_body_type_internal(id, new_type);
                if new_type == BodyType::Dynamic
                    && self.body_first_collider[slot as usize].is_none()
                {
                    self.bodies[slot as usize].apply_mass_data(user_mass);
                }
            }
        }
        {
            let body = &mut self.bodies[slot as usize];
            let mass_data = body.mass_data();
            body.apply_mass_data(mass_data);
            let locks = body.motion_locks;
            body.set_motion_locks(locks);
        }
        self.sync_pose(slot);
        self.refresh_body_proxies(slot, ProxyUpdate::Reinsert);
        self.mass_moved(slot);
        self.refresh_body_extents(slot);
        if self.bodies[slot as usize].is_sleeping
            && self.body_links[slot as usize].set == AWAKE_SET
            && self.config.enable_sleep
            && self.bodies[slot as usize].enable_sleep
        {
            if let Some(id) = self.body_ids.handle_at(slot) {
                self.set_awake_internal(id, false);
            }
        }
        self.bodies[slot as usize].is_sleeping =
            self.body_links[slot as usize].set >= FIRST_SLEEPING_SET;
    }

    /// Sets a body's velocities so that it reaches `target` after `time_step` (Box2D
    /// `b2Body_SetTargetTransform`), for moving kinematic bodies. A sleeping body is left alone
    /// unless `wake` is set and the motion is above the sleep threshold. Returns whether the
    /// velocities were set: never for a stale handle, a static or disabled body, or a
    /// non-positive `time_step`.
    pub fn set_target_transform(
        &mut self,
        id: BodyId2d,
        target: Transform2d,
        time_step: f32,
        wake: bool,
    ) -> bool {
        if !(time_step > 0.0 && time_step.is_finite()) {
            return false;
        }
        self.flush_touched_bodies();
        let Some(body) = self.body(id) else {
            return false;
        };
        if !body.enabled || body.body_type == BodyType::Static {
            return false;
        }
        if body.is_sleeping && !wake {
            return false;
        }
        let inv_time_step = 1.0 / time_step;
        let target_center = target.transform_point(body.local_center);
        let center = self.body_links[id.index() as usize].center;
        let linear_velocity = (target_center - center) * inv_time_step;
        let angular_velocity = body.rotation.angle_to(target.rotation) * inv_time_step;
        if body.is_sleeping {
            let max_extent = self.body_links[id.index() as usize].max_extent;
            let speed = linear_velocity.length() + angular_velocity.abs() * max_extent;
            if speed < self.body_links[id.index() as usize].sleep_threshold {
                return false;
            }
            self.wake_body(id.index());
        }
        let body = &mut self.bodies[id.index() as usize];
        body.linear_velocity = linear_velocity;
        body.angular_velocity = angular_velocity;
        true
    }

    /// Attaches a collider to a body. A body may hold several; they never collide with each
    /// other. A dynamic body's mass, centre of mass and rotational inertia are recomputed from
    /// all of its colliders' shapes and densities, replacing any mass set before (Box2D
    /// `b2UpdateBodyMassData`).
    ///
    /// # Panics
    ///
    /// Panics if `body_id` is stale.
    pub fn create_collider(&mut self, body_id: BodyId2d, collider: Collider2d) -> ColliderId2d {
        assert!(
            self.body_ids.contains(body_id),
            "stale body handle {body_id:?}"
        );
        self.flush_touched_bodies();
        let id = self.insert_collider(body_id.index(), collider, None);
        self.update_mass(body_id.index());
        id
    }

    /// Attaches a collider to body slot `body_slot`, as a segment of `chain` if given, without
    /// updating the body's mass (Box2D `b2CreateShapeInternal`).
    fn insert_collider(
        &mut self,
        body_slot: u32,
        collider: Collider2d,
        chain: Option<ChainId2d>,
    ) -> ColliderId2d {
        let is_sensor = collider.is_sensor;
        let id = self.collider_ids.alloc();
        let slot = id.index();
        if slot as usize == self.colliders.len() {
            self.colliders.push(Some(collider));
            self.collider_bodies.push(body_slot);
            self.collider_next.push(None);
            self.collider_proxies.push(None);
            self.collider_aabbs.push(Aabb2d::default());
            self.collider_fat_aabbs.push(Aabb2d::default());
            self.collider_chains.push(chain);
            self.sensor_overlaps.push(Vec::new());
        } else {
            let i = slot as usize;
            self.colliders[i] = Some(collider);
            self.collider_bodies[i] = body_slot;
            self.collider_next[i] = None;
            self.collider_proxies[i] = None;
            self.collider_chains[i] = chain;
        }
        if is_sensor && self.sensors_cached {
            self.sensor_slots.push(slot);
        }
        if self.bodies[body_slot as usize].enabled {
            self.create_proxy(slot);
        }
        // Append, keeping the body's colliders in creation order.
        match collider_slots(&self.body_first_collider, &self.collider_next, body_slot).last() {
            Some(last) => self.collider_next[last as usize] = Some(slot),
            None => self.body_first_collider[body_slot as usize] = Some(slot),
        }
        id
    }

    /// Attaches a chain to body `body_id`: one chain-segment collider per segment of `def`, with
    /// the definition's materials, filter and sensor events and without contact or hit events.
    /// The body's mass is unchanged, as chain segments have none (Box2D `b2CreateChain`).
    ///
    /// # Panics
    ///
    /// If `body_id` is stale, `def` has fewer than 2 points (3 for a loop), or its material
    /// count is neither 1 nor its segment count.
    pub fn create_chain(&mut self, body_id: BodyId2d, def: &ChainDef2d) -> ChainId2d {
        assert!(
            self.body_ids.contains(body_id),
            "stale body handle {body_id:?}"
        );
        let minimum = if def.is_loop { 3 } else { 2 };
        assert!(
            def.points.len() >= minimum,
            "a chain needs at least {minimum} points"
        );
        let count = def.segment_count();
        assert!(
            def.materials.len() == 1 || def.materials.len() == count,
            "a chain needs one material or one per segment"
        );
        self.flush_touched_bodies();
        let id = self.chain_ids.alloc();
        let mut segments = Vec::with_capacity(count);
        for (i, segment) in def.segments().enumerate() {
            let collider = Collider2d::new(Shape2d::ChainSegment(segment))
                .with_material(def.material(i))
                .with_filter(def.filter)
                .with_sensor_events(def.enable_sensor_events);
            segments.push(self.insert_collider(body_id.index(), collider, Some(id)));
        }
        self.refresh_body_extents(body_id.index());
        let chain = Chain2d {
            body: body_id,
            segments,
            is_loop: def.is_loop,
        };
        let slot = id.index() as usize;
        if slot == self.chains.len() {
            self.chains.push(Some(chain));
        } else {
            self.chains[slot] = Some(chain);
        }
        id
    }

    /// Destroys a chain and its segments; bodies touching them wake (Box2D `b2DestroyChain`).
    /// Returns `false`, changing nothing, if `id` is stale.
    pub fn destroy_chain(&mut self, id: ChainId2d) -> bool {
        if !self.chain_ids.contains(id) {
            return false;
        }
        self.flush_touched_bodies();
        let Some(chain) = self.chains[id.index() as usize].take() else {
            return false;
        };
        let body_slot = chain.body.index();
        for &segment in &chain.segments {
            self.destroy_collider_contacts(segment);
            self.remove_collider(segment);
        }
        self.refresh_body_extents(body_slot);
        self.chain_ids.free(id);
        true
    }

    /// Returns the chain `id`, or `None` if the handle is stale.
    #[must_use]
    pub fn chain(&self, id: ChainId2d) -> Option<&Chain2d> {
        if self.chain_ids.contains(id) {
            self.chains[id.index() as usize].as_ref()
        } else {
            None
        }
    }

    /// The chain collider `id` is a segment of (Box2D `b2Shape_GetParentChain`).
    #[must_use]
    pub fn collider_chain(&self, id: ColliderId2d) -> Option<ChainId2d> {
        if self.collider_ids.contains(id) {
            self.collider_chains[id.index() as usize]
        } else {
            None
        }
    }

    /// Destroys a collider and its contacts, waking the bodies involved, and recomputes its
    /// body's mass from the remaining colliders: a dynamic body left without one is massless
    /// (Box2D `b2DestroyShape` with `updateBodyMass`). Returns `false`, changing nothing, if
    /// `id` is stale or a segment of a chain, which goes with its chain only.
    pub fn destroy_collider(&mut self, id: ColliderId2d) -> bool {
        let Some(body_slot) = self.collider_body_slot(id) else {
            return false;
        };
        if self.collider_chains[id.index() as usize].is_some() {
            return false;
        }
        self.flush_touched_bodies();
        self.destroy_collider_contacts(id);
        self.wake_body(body_slot);
        self.remove_collider(id);
        self.update_mass(body_slot);
        true
    }

    /// Destroys the contacts of live collider `id`, waking the bodies of those that touched.
    fn destroy_collider_contacts(&mut self, id: ColliderId2d) {
        let slot = id.index();
        let mut key = self
            .contact_manager
            .body_head(self.collider_bodies[slot as usize]);
        while key != NULL {
            let contact = self.contact_manager.at(key >> 1);
            let next = contact.edges[(key & 1) as usize].next;
            if contact.collider_a == slot || contact.collider_b == slot {
                self.destroy_contact(key >> 1, false);
            }
            key = next;
        }
    }

    /// Returns the collider `id`, or `None` if the handle is stale.
    #[must_use]
    pub fn collider(&self, id: ColliderId2d) -> Option<&Collider2d> {
        if self.collider_ids.contains(id) {
            self.colliders[id.index() as usize].as_ref()
        } else {
            None
        }
    }

    /// Returns the collider `id` mutably, or `None` if the handle is stale.
    pub fn collider_mut(&mut self, id: ColliderId2d) -> Option<&mut Collider2d> {
        if self.collider_ids.contains(id) {
            self.sensors_cached = false;
            let body_slot = self.collider_bodies[id.index() as usize];
            if !self.body_links[body_slot as usize].touched {
                self.body_links[body_slot as usize].touched = true;
                self.touched_bodies.push(body_slot);
            }
            self.colliders[id.index() as usize].as_mut()
        } else {
            None
        }
    }

    /// The first collider attached to body `id`, if any.
    #[must_use]
    pub fn collider_of(&self, id: BodyId2d) -> Option<ColliderId2d> {
        self.colliders_of(id).next()
    }

    /// The colliders attached to body `id`, in creation order; none for a stale handle.
    pub fn colliders_of(&self, id: BodyId2d) -> impl Iterator<Item = ColliderId2d> + '_ {
        let first = if self.body_ids.contains(id) {
            self.body_first_collider[id.index() as usize]
        } else {
            None
        };
        ColliderSlots {
            next: &self.collider_next,
            current: first,
        }
        .filter_map(|slot| self.collider_ids.handle_at(slot))
    }

    /// Connects two bodies with a joint constraint. The bodies do not collide with each other;
    /// use [`Self::add_joint`] to choose otherwise.
    ///
    /// # Panics
    ///
    /// Panics if either body handle is stale or if `body_a == body_b`.
    pub fn create_joint(
        &mut self,
        body_a: BodyId2d,
        body_b: BodyId2d,
        joint: Joint2d,
    ) -> JointId2d {
        self.add_joint(JointInstance2d::new(body_a, body_b, joint))
    }

    /// Adds a joint. Unless it sets `collide_connected`, the contacts between its bodies are
    /// destroyed, as in Box2D's `b2CreateJoint`, waking the bodies.
    ///
    /// # Panics
    ///
    /// Panics if either body handle is stale or if `instance.body_a == instance.body_b`.
    pub fn add_joint(&mut self, instance: JointInstance2d) -> JointId2d {
        assert!(
            self.body_ids.contains(instance.body_a) && self.body_ids.contains(instance.body_b),
            "stale body handles: body_a={:?}, body_b={:?}",
            instance.body_a,
            instance.body_b
        );
        assert_ne!(
            instance.body_a, instance.body_b,
            "cannot create a joint between a body and itself: {:?}",
            instance.body_a
        );
        self.flush_touched_bodies();
        let mut instance = instance;
        instance.joint = instance.joint.sanitized(
            self.config.linear_slop(),
            self.config.length_units_per_meter,
        );
        let (a, b) = instance.slots();
        if !instance.collide_connected {
            let mut key = self.contact_manager.body_head(a);
            while key != NULL {
                let contact = self.contact_manager.at(key >> 1);
                let end = (key & 1) as usize;
                let (next, other) = (contact.edges[end].next, contact.edges[end ^ 1].body);
                if other == b {
                    self.destroy_contact(key >> 1, false);
                }
                key = next;
            }
        }
        let id = self.joint_ids.alloc();
        let slot = id.index();
        if slot as usize == self.joints.len() {
            self.joints.push(Some(instance));
            self.joint_bodies.push((a, b));
            self.joint_links.push(JointLinks2d::UNLINKED);
            self.joint_steps.push(JointStep2d::default());
        } else {
            self.joints[slot as usize] = Some(instance);
            self.joint_bodies[slot as usize] = (a, b);
            self.joint_links[slot as usize] = JointLinks2d::UNLINKED;
        }
        // The joint's set (Box2D `b2CreateJoint`): disabled with a disabled body, static between
        // non-dynamic bodies, awake (waking a sleeping body) with an awake body, else asleep.
        let (set_a, set_b) = (
            self.body_links[a as usize].set,
            self.body_links[b as usize].set,
        );
        let max_set = set_a.max(set_b);
        let set = if set_a == DISABLED_SET || set_b == DISABLED_SET {
            DISABLED_SET
        } else if self.bodies[a as usize].body_type != BodyType::Dynamic
            && self.bodies[b as usize].body_type != BodyType::Dynamic
        {
            STATIC_SET
        } else if set_a == AWAKE_SET || set_b == AWAKE_SET {
            if max_set >= FIRST_SLEEPING_SET {
                self.wake_set(max_set);
            }
            AWAKE_SET
        } else {
            if set_a != set_b && set_a >= FIRST_SLEEPING_SET && set_b >= FIRST_SLEEPING_SET {
                self.merge_sets(set_a, set_b);
            }
            self.body_links[a as usize]
                .set
                .max(self.body_links[b as usize].set)
        };
        self.add_joint_to_set(slot, set);
        if set > DISABLED_SET {
            self.link_joint(slot);
        }
        id
    }

    /// Destroys a joint and wakes its bodies (Box2D `b2DestroyJoint`), returning it. Other
    /// joints keep their handles. Returns `None` if `id` is stale.
    pub fn destroy_joint(&mut self, id: JointId2d) -> Option<JointInstance2d> {
        self.flush_touched_bodies();
        if !self.joint_ids.free(id) {
            return None;
        }
        let slot = id.index();
        let joint = self.joints[slot as usize].take()?;
        self.unlink_joint(slot);
        self.remove_joint_from_set(slot);
        self.joint_links[slot as usize] = JointLinks2d::UNLINKED;
        let (a, b) = self.joint_bodies[slot as usize];
        self.wake_body(a);
        self.wake_body(b);
        Some(joint)
    }

    /// Returns the joint `id`, or `None` if the handle is stale.
    #[must_use]
    pub fn joint(&self, id: JointId2d) -> Option<&JointInstance2d> {
        if self.joint_ids.contains(id) {
            self.joints[id.index() as usize].as_ref()
        } else {
            None
        }
    }

    /// Returns the joint `id` mutably, waking its bodies first as Box2D's joint setters do, or
    /// `None` if the handle is stale. Its bodies cannot change.
    pub fn joint_mut(&mut self, id: JointId2d) -> Option<&mut JointInstance2d> {
        if !self.joint_ids.contains(id) {
            return None;
        }
        let (a, b) = self.joint_bodies[id.index() as usize];
        self.wake_body(a);
        self.wake_body(b);
        self.joints[id.index() as usize].as_mut()
    }

    /// Joint `id` and its bodies, or `None` if the handle is stale.
    fn joint_and_bodies(&self, id: JointId2d) -> Option<(&Joint2d, &RigidBody2d, &RigidBody2d)> {
        let joint = self.joint(id)?;
        let (a, b) = self.joint_bodies[id.index() as usize];
        Some((
            &joint.joint,
            &self.bodies[a as usize],
            &self.bodies[b as usize],
        ))
    }

    /// The force joint `id` applied to body B during the last substep (Box2D
    /// `b2Joint_GetConstraintForce`), or `None` if the handle is stale.
    #[must_use]
    pub fn joint_constraint_force(&self, id: JointId2d) -> Option<Vec2> {
        let (joint, a, b) = self.joint_and_bodies(id)?;
        Some(joint.constraint_force(a, b, self.inv_h))
    }

    /// The torque joint `id` applied to body B during the last substep (Box2D
    /// `b2Joint_GetConstraintTorque`), or `None` if the handle is stale.
    #[must_use]
    pub fn joint_constraint_torque(&self, id: JointId2d) -> Option<f32> {
        let (joint, _, _) = self.joint_and_bodies(id)?;
        Some(joint.constraint_torque(self.inv_h))
    }

    /// How far joint `id`'s frames are from where it holds them (Box2D
    /// `b2Joint_GetLinearSeparation`), or `None` if the handle is stale.
    #[must_use]
    pub fn joint_linear_separation(&self, id: JointId2d) -> Option<f32> {
        let (joint, a, b) = self.joint_and_bodies(id)?;
        Some(joint.linear_separation(a, b))
    }

    /// How far joint `id`'s bodies are turned from where it holds them (Box2D
    /// `b2Joint_GetAngularSeparation`), or `None` if the handle is stale.
    #[must_use]
    pub fn joint_angular_separation(&self, id: JointId2d) -> Option<f32> {
        let (joint, a, b) = self.joint_and_bodies(id)?;
        Some(joint.angular_separation(a, b))
    }

    /// Wakes joint `id`'s bodies (Box2D `b2Joint_WakeBodies`). Returns `false` if the handle
    /// is stale.
    pub fn wake_joint_bodies(&mut self, id: JointId2d) -> bool {
        if !self.joint_ids.contains(id) {
            return false;
        }
        let (a, b) = self.joint_bodies[id.index() as usize];
        self.wake_body(a);
        self.wake_body(b);
        true
    }

    /// The live joints and their handles, in slot order.
    pub fn joints(&self) -> impl Iterator<Item = (JointId2d, &JointInstance2d)> + '_ {
        self.joint_ids
            .iter()
            .filter_map(|id| Some((id, self.joints[id.index() as usize].as_ref()?)))
    }

    /// Number of live joints.
    #[must_use]
    pub const fn joint_count(&self) -> usize {
        self.joint_ids.len()
    }

    /// The manifolds of the touching contacts, awake or asleep, with their bodies, in contact
    /// order. The list is built on each call, so that steps do not visit sleeping contacts.
    #[must_use]
    pub fn manifolds(&self) -> Vec<(BodyId2d, BodyId2d, ContactManifold2d)> {
        self.contact_manager
            .values()
            .filter(|c| c.is_touching)
            .filter_map(|c| {
                Some((
                    self.body_ids.handle_at(c.body_a)?,
                    self.body_ids.handle_at(c.body_b)?,
                    c.manifold,
                ))
            })
            .collect()
    }

    /// Returns a reference to the constraint graph partitioning active constraints into colors.
    /// It identifies bodies by slot ([`Handle::index`]).
    #[must_use]
    pub const fn constraint_graph(&self) -> &ConstraintGraph2d {
        &self.constraint_graph
    }

    /// Returns a reference to the contact manager maintaining persistent contacts. Contacts are
    /// keyed by body slot ([`Handle::index`]).
    #[must_use]
    pub const fn contact_manager(&self) -> &ContactManager2d {
        &self.contact_manager
    }

    /// Queries all broadphase trees for collider slots whose fat box overlaps `aabb`, evaluating
    /// un-flushed [`Self::touched_bodies`] at their current pose rather than their stale tree
    /// leaf.
    fn query_trees<F>(&self, aabb: &Aabb2d, mut callback: F)
    where
        F: FnMut(u32) -> bool,
    {
        let has_touched = !self.touched_bodies.is_empty();
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            let mut go_on = true;
            tree.query(aabb, |_, slot| {
                if has_touched {
                    let body_slot = self.collider_bodies[slot as usize];
                    if self.body_links[body_slot as usize].touched {
                        return true;
                    }
                }
                go_on = callback(slot);
                go_on
            });
            if !go_on {
                return;
            }
            if has_touched {
                for &body_slot in &self.touched_bodies {
                    if self.body_ids.handle_at(body_slot).is_none() {
                        continue;
                    }
                    let body = &self.bodies[body_slot as usize];
                    if !body.enabled || tree_of(body.body_type) != tree_idx {
                        continue;
                    }
                    let xf = body.transform();
                    for c in
                        collider_slots(&self.body_first_collider, &self.collider_next, body_slot)
                    {
                        let Some(collider) = self.colliders[c as usize].as_ref() else {
                            continue;
                        };
                        let fat = collider.shape.compute_aabb(&xf).fatten(
                            self.config.speculative_distance()
                                + self.proxy_margin(collider, body.body_type),
                        );
                        if fat.intersects(aabb) && !callback(c) {
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Ray-casts all broadphase trees, evaluating un-flushed [`Self::touched_bodies`] at their
    /// current pose rather than their stale tree leaf.
    fn ray_cast_trees<F>(&self, ray: &Ray2d, mut callback: F)
    where
        F: FnMut(u32, f32) -> f32,
    {
        let has_touched = !self.touched_bodies.is_empty();
        let mut fraction = 1.0f32;
        let p1 = ray.origin;
        let d = ray.direction;
        let r = d.normalize_or_zero();
        let v = Vec2::new(-r.y, r.x);
        let abs_v = v.abs();
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            tree.ray_cast(ray, fraction, |_, slot, max_fraction| {
                if has_touched {
                    let body_slot = self.collider_bodies[slot as usize];
                    if self.body_links[body_slot as usize].touched {
                        return max_fraction;
                    }
                }
                let value = callback(slot, max_fraction);
                if (0.0..=1.0).contains(&value) {
                    fraction = value;
                }
                value
            });
            if fraction == 0.0 {
                break;
            }
            if has_touched {
                for &body_slot in &self.touched_bodies {
                    if self.body_ids.handle_at(body_slot).is_none() {
                        continue;
                    }
                    let body = &self.bodies[body_slot as usize];
                    if !body.enabled || tree_of(body.body_type) != tree_idx {
                        continue;
                    }
                    let xf = body.transform();
                    for c in
                        collider_slots(&self.body_first_collider, &self.collider_next, body_slot)
                    {
                        let Some(collider) = self.colliders[c as usize].as_ref() else {
                            continue;
                        };
                        let fat = collider.shape.compute_aabb(&xf).fatten(
                            self.config.speculative_distance()
                                + self.proxy_margin(collider, body.body_type),
                        );
                        let p2 = p1 + d * fraction;
                        let segment = Aabb2d::new(p1.min(p2), p1.max(p2));
                        if !segment.intersects(&fat) {
                            continue;
                        }
                        let center = fat.center();
                        let half = fat.half_extents();
                        if abs_v.dot(half) < v.dot(p1 - center).abs() {
                            continue;
                        }
                        let value = callback(c, fraction);
                        if value == 0.0 {
                            return;
                        }
                        if (0.0..=1.0).contains(&value) && value <= fraction {
                            fraction = value;
                        }
                    }
                }
            }
        }
    }

    /// Box-casts all broadphase trees, evaluating un-flushed [`Self::touched_bodies`] at their
    /// current pose rather than their stale tree leaf.
    fn box_cast_trees<F>(&self, bounds: &Aabb2d, translation: Vec2, mut callback: F) -> f32
    where
        F: FnMut(u32, f32) -> f32,
    {
        let has_touched = !self.touched_bodies.is_empty();
        let mut fraction = 1.0f32;
        let p1 = bounds.center();
        let extension = bounds.half_extents();
        let v = Vec2::new(-translation.y, translation.x);
        let abs_v = v.abs();
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            tree.box_cast(bounds, translation, fraction, |_, slot, max_fraction| {
                if has_touched {
                    let body_slot = self.collider_bodies[slot as usize];
                    if self.body_links[body_slot as usize].touched {
                        return max_fraction;
                    }
                }
                let value = callback(slot, max_fraction);
                if (0.0..=1.0).contains(&value) {
                    fraction = value;
                }
                value
            });
            if fraction == 0.0 {
                break;
            }
            if has_touched {
                for &body_slot in &self.touched_bodies {
                    if self.body_ids.handle_at(body_slot).is_none() {
                        continue;
                    }
                    let body = &self.bodies[body_slot as usize];
                    if !body.enabled || tree_of(body.body_type) != tree_idx {
                        continue;
                    }
                    let xf = body.transform();
                    for c in
                        collider_slots(&self.body_first_collider, &self.collider_next, body_slot)
                    {
                        let Some(collider) = self.colliders[c as usize].as_ref() else {
                            continue;
                        };
                        let fat = collider.shape.compute_aabb(&xf).fatten(
                            self.config.speculative_distance()
                                + self.proxy_margin(collider, body.body_type),
                        );
                        let t = translation * fraction;
                        let swept = Aabb2d::new(
                            bounds.min.min(bounds.min + t),
                            bounds.max.max(bounds.max + t),
                        );
                        if !swept.intersects(&fat) {
                            continue;
                        }
                        let center = fat.center();
                        let half = fat.half_extents() + extension;
                        if abs_v.dot(half) < v.dot(p1 - center).abs() {
                            continue;
                        }
                        let value = callback(c, fraction);
                        if value == 0.0 {
                            return 0.0;
                        }
                        if (0.0..=1.0).contains(&value) && value < fraction {
                            fraction = value;
                        }
                    }
                }
            }
        }
        fraction
    }

    /// Casts a ray from `origin` along `translation` against every collider that `filter`
    /// accepts (Box2D `b2World_CastRay`), static colliders first, then kinematic and dynamic.
    /// `callback` receives each hit and returns -1 to skip it, 0 to stop, the hit's fraction
    /// to shorten the cast to it (finding the closest hit), or 1 to go on. A ray that starts
    /// inside a shape hits it at fraction 0.
    pub fn cast_ray<F>(
        &self,
        origin: Vec2,
        translation: Vec2,
        filter: QueryFilter2d,
        mut callback: F,
    ) where
        F: FnMut(&RayHit2d) -> f32,
    {
        let ray = Ray2d {
            origin,
            direction: translation,
        };
        self.ray_cast_trees(&ray, |slot, max_fraction| {
            let Some((collider_id, collider, body)) = self.query_collider(slot, filter) else {
                return max_fraction;
            };
            let xf = body.transform();
            let output = collider.shape.ray_cast(&RayCastInput2d {
                origin: xf.inv_transform_point(origin),
                translation: xf.rotation.inv_rotate(translation),
                max_fraction,
            });
            if !output.hit {
                return max_fraction;
            }
            let hit = self.world_hit(collider_id, slot, &xf, &output);
            callback(&hit)
        });
    }

    /// The closest hit of a ray from `origin` along `translation`, ignoring shapes the ray
    /// starts in (Box2D `b2World_CastRayClosest`).
    #[must_use]
    pub fn cast_ray_closest(
        &self,
        origin: Vec2,
        translation: Vec2,
        filter: QueryFilter2d,
    ) -> Option<RayHit2d> {
        let mut closest = None;
        self.cast_ray(origin, translation, filter, |hit| {
            if hit.fraction == 0.0 {
                return -1.0;
            }
            closest = Some(*hit);
            hit.fraction
        });
        closest
    }

    /// Casts the world-space shape `proxy` along `translation` against every collider that
    /// `filter` accepts (Box2D `b2World_CastShape`), calling `callback` as
    /// [`Self::cast_ray`] does. A cast that starts touching a shape hits it at fraction 0.
    pub fn cast_shape<F>(
        &self,
        proxy: &ShapeProxy2d,
        translation: Vec2,
        filter: QueryFilter2d,
        mut callback: F,
    ) where
        F: FnMut(&RayHit2d) -> f32,
    {
        let input = ShapeCastInput2d {
            proxy: *proxy,
            translation,
            max_fraction: 1.0,
            can_encroach: false,
        };
        let bounds = proxy_bounds(proxy);
        self.box_cast_trees(&bounds, translation, |slot, max_fraction| {
            let Some((collider_id, collider, body)) = self.query_collider(slot, filter) else {
                return max_fraction;
            };
            let xf = body.transform();
            let output = collider
                .shape
                .shape_cast(&local_cast(&input, &xf, max_fraction));
            if !output.hit {
                return max_fraction;
            }
            let hit = self.world_hit(collider_id, slot, &xf, &output);
            callback(&hit)
        });
    }

    /// Calls `callback` with every collider that `filter` accepts whose fat box overlaps `aabb`
    /// (Box2D `b2World_OverlapAABB`), until it returns `false`. The fat boxes are the
    /// broadphase's, so a collider can be reported a little before its shape reaches `aabb`.
    pub fn overlap_aabb<F>(&self, aabb: &Aabb2d, filter: QueryFilter2d, mut callback: F)
    where
        F: FnMut(ColliderId2d) -> bool,
    {
        self.query_trees(aabb, |slot| {
            if let Some((id, _, _)) = self.query_collider(slot, filter) {
                callback(id)
            } else {
                true
            }
        });
    }

    /// Calls `callback` with every collider that `filter` accepts whose shape overlaps the
    /// world-space shape `proxy` (Box2D `b2World_OverlapShape`), until it returns `false`.
    pub fn overlap_shape<F>(&self, proxy: &ShapeProxy2d, filter: QueryFilter2d, mut callback: F)
    where
        F: FnMut(ColliderId2d) -> bool,
    {
        let bounds = proxy_bounds(proxy);
        let tolerance = 0.1 * self.config.linear_slop();
        self.query_trees(&bounds, |slot| {
            let Some((id, collider, body)) = self.query_collider(slot, filter) else {
                return true;
            };
            let output = shape_distance(
                proxy,
                &ShapeProxy2d::of_shape(&collider.shape),
                &body.transform(),
                &mut SimplexCache2d::default(),
                true,
            );
            if output.distance <= tolerance {
                callback(id)
            } else {
                true
            }
        });
    }

    /// The fraction of `translation` a capsule `mover` (in world space) can move before it
    /// touches a collider that `filter` accepts (Box2D `b2World_CastMover`). Colliders it
    /// already touches are ignored, and a mover touching one may move a little closer.
    #[must_use]
    pub fn cast_mover(&self, mover: &Capsule2d, translation: Vec2, filter: QueryFilter2d) -> f32 {
        let input = ShapeCastInput2d {
            proxy: ShapeProxy2d::new(&[mover.point1, mover.point2], mover.radius),
            translation,
            max_fraction: 1.0,
            can_encroach: true,
        };
        let bounds = proxy_bounds(&input.proxy);
        self.box_cast_trees(&bounds, translation, |slot, max_fraction| {
            let Some((_, collider, body)) = self.query_collider(slot, filter) else {
                return max_fraction;
            };
            let output =
                collider
                    .shape
                    .shape_cast(&local_cast(&input, &body.transform(), max_fraction));
            if output.fraction == 0.0 {
                // Missed, or already touching.
                return max_fraction;
            }
            output.fraction
        })
    }

    /// Calls `callback` with the collision plane of every collider that `filter` accepts and
    /// that the capsule `mover` (in world space) touches (Box2D `b2World_CollideMover`), until
    /// it returns `false`. Plane normals point at the mover; offsets are how deep it is.
    pub fn collide_mover<F>(&self, mover: &Capsule2d, filter: QueryFilter2d, mut callback: F)
    where
        F: FnMut(ColliderId2d, &PlaneResult2d) -> bool,
    {
        let r = Vec2::splat(mover.radius);
        let bounds = Aabb2d::new(
            mover.point1.min(mover.point2) - r,
            mover.point1.max(mover.point2) + r,
        );
        self.query_trees(&bounds, |slot| {
            let Some((id, collider, body)) = self.query_collider(slot, filter) else {
                return true;
            };
            let xf = body.transform();
            let local = Capsule2d::new(
                xf.inv_transform_point(mover.point1),
                xf.inv_transform_point(mover.point2),
                mover.radius,
            );
            if let Some(mut result) = collider.shape.collide_mover(&local) {
                result.plane.normal = xf.rotation.rotate(result.plane.normal);
                let n = result.plane.normal;
                // Deep overlaps give no plane.
                if (1.0 - n.dot(n)).abs() < 100.0 * f32::EPSILON {
                    return callback(id, &result);
                }
            }
            true
        });
    }

    /// Pushes the dynamic bodies around an explosion (Box2D `b2World_Explode`): each collider
    /// within `radius + falloff` whose category the mask accepts gets an impulse of
    /// `impulse_per_length` times its perimeter facing the explosion, away from it at its
    /// closest point, fading over the falloff. Bodies it reaches wake.
    pub fn explode(&mut self, explosion: &ExplosionDef2d) {
        self.flush_touched_bodies();
        let extent = explosion.radius + explosion.falloff;
        let bounds = Aabb2d::new(
            explosion.position - Vec2::splat(extent),
            explosion.position + Vec2::splat(extent),
        );
        let mut hits = std::mem::take(&mut self.candidates_scratch);
        hits.clear();
        self.trees[DYNAMIC_TREE].query(&bounds, |_, slot| {
            hits.push(slot);
            true
        });
        for &slot in &hits {
            let Some(collider) = self.colliders[slot as usize] else {
                continue;
            };
            if collider.filter.category_bits & explosion.mask_bits == 0 {
                continue;
            }
            let body_slot = self.collider_bodies[slot as usize];
            let xf = self.bodies[body_slot as usize].transform();
            let local = xf.inv_transform_point(explosion.position);
            let output = shape_distance(
                &ShapeProxy2d::of_shape(&collider.shape),
                &ShapeProxy2d::new(&[local], 0.0),
                &Transform2d::IDENTITY,
                &mut SimplexCache2d::default(),
                true,
            );
            if output.distance > extent {
                continue;
            }
            self.wake_body(body_slot);
            if self.body_links[body_slot as usize].set != AWAKE_SET {
                continue;
            }
            let closest = if output.distance == 0.0 {
                collider.shape.centroid()
            } else {
                output.point_a
            };
            let mut direction = closest - local;
            direction = if direction.length_squared() > 100.0 * f32::EPSILON * f32::EPSILON {
                crate::math::normalize(direction)
            } else {
                Vec2::X
            };
            let perimeter = collider
                .shape
                .projected_perimeter(Vec2::new(-direction.y, direction.x));
            let scale = if output.distance > explosion.radius && explosion.falloff > 0.0 {
                crate::math::clamp((extent - output.distance) / explosion.falloff, 0.0, 1.0)
            } else {
                1.0
            };
            let magnitude = explosion.impulse_per_length * perimeter * scale;
            let impulse = xf.rotation.rotate(direction) * magnitude;
            let body = &mut self.bodies[body_slot as usize];
            body.linear_velocity += impulse * body.inv_mass;
            let r = xf.rotation.rotate(closest - body.local_center);
            body.angular_velocity += body.inv_inertia * crate::math::cross(r, impulse);
        }
        self.candidates_scratch = hits;
    }

    /// Applies the force of wind blowing at `wind` (in m/s) to collider `id` (Box2D
    /// `b2Shape_ApplyWind`): air drag on the projected area of a circle, capsule or polygon of
    /// unit depth moving relative to the air (`drag` scales the shape's own velocity), plus
    /// `lift` across its facing edges. A sleeping body is woken if `wake`, else left alone.
    /// Returns `false`, doing nothing, if `id` is stale.
    pub fn apply_wind(
        &mut self,
        id: ColliderId2d,
        wind: Vec2,
        drag: f32,
        lift: f32,
        wake: bool,
    ) -> bool {
        self.flush_touched_bodies();
        let Some(body_slot) = self.collider_body_slot(id) else {
            return false;
        };
        let Some(collider) = self.colliders[id.index() as usize] else {
            return false;
        };
        let shape = collider.shape;
        if matches!(shape, Shape2d::Segment(_) | Shape2d::ChainSegment(_))
            || self.bodies[body_slot as usize].body_type != BodyType::Dynamic
        {
            return true;
        }
        let set = self.body_links[body_slot as usize].set;
        if set >= FIRST_SLEEPING_SET && !wake {
            return true;
        }
        if set != AWAKE_SET {
            self.wake_body(body_slot);
        }
        let length_units = self.config.length_units_per_meter;
        let body = &mut self.bodies[body_slot as usize];
        let (force, torque) = wind_force(&shape, body, wind, drag, lift, length_units);
        body.force += force;
        body.torque += torque;
        true
    }

    /// The point of collider `id`'s shape closest to `target` in world space, `target` itself
    /// if inside (Box2D `b2Shape_GetClosestPoint`), or `None` if the handle is stale.
    #[must_use]
    pub fn collider_closest_point(&self, id: ColliderId2d, target: Vec2) -> Option<Vec2> {
        let (collider, body) = self.collider_and_body(id)?;
        let xf = body.transform();
        let output = shape_distance(
            &ShapeProxy2d::of_shape(&collider.shape),
            &ShapeProxy2d::new(&[Vec2::ZERO], 0.0),
            &xf.inv_mul(&Transform2d::from_translation(target)),
            &mut SimplexCache2d::default(),
            true,
        );
        Some(xf.transform_point(output.point_a))
    }

    /// Whether world point `point` is inside collider `id`'s shape (Box2D
    /// `b2Shape_TestPoint`), or `None` if the handle is stale.
    #[must_use]
    pub fn collider_test_point(&self, id: ColliderId2d, point: Vec2) -> Option<bool> {
        let (collider, body) = self.collider_and_body(id)?;
        Some(
            collider
                .shape
                .test_point(body.transform().inv_transform_point(point)),
        )
    }

    /// A ray cast from `origin` along `translation` against collider `id` alone, in world
    /// space (Box2D `b2Shape_RayCast`), or `None` if the handle is stale.
    #[must_use]
    pub fn collider_ray_cast(
        &self,
        id: ColliderId2d,
        origin: Vec2,
        translation: Vec2,
    ) -> Option<CastOutput2d> {
        let (collider, body) = self.collider_and_body(id)?;
        let xf = body.transform();
        let mut output = collider.shape.ray_cast(&RayCastInput2d {
            origin: xf.inv_transform_point(origin),
            translation: xf.rotation.inv_rotate(translation),
            max_fraction: 1.0,
        });
        output.point = xf.transform_point(output.point);
        output.normal = xf.rotation.rotate(output.normal);
        Some(output)
    }

    /// Live collider `id` and its body.
    fn collider_and_body(&self, id: ColliderId2d) -> Option<(&Collider2d, &RigidBody2d)> {
        let body_slot = self.collider_body_slot(id)?;
        Some((
            self.colliders[id.index() as usize].as_ref()?,
            &self.bodies[body_slot as usize],
        ))
    }

    /// The collider in slot `slot`, if `filter` accepts it (Box2D `b2ShouldQueryCollide`), with
    /// its handle and body.
    fn query_collider(
        &self,
        slot: u32,
        filter: QueryFilter2d,
    ) -> Option<(ColliderId2d, &Collider2d, &RigidBody2d)> {
        let (id, collider) = self.collider_at(slot)?;
        filter.accepts(&collider.filter).then(|| {
            (
                id,
                collider,
                &self.bodies[self.collider_bodies[slot as usize] as usize],
            )
        })
    }

    /// A cast output of collider slot `slot`, in the frame `xf`, as a world hit.
    fn world_hit(
        &self,
        collider_id: ColliderId2d,
        slot: u32,
        xf: &Transform2d,
        output: &CastOutput2d,
    ) -> RayHit2d {
        let body_slot = self.collider_bodies[slot as usize];
        RayHit2d {
            body_id: self
                .body_ids
                .handle_at(body_slot)
                .expect("a live collider's body is live"),
            collider_id,
            point: xf.transform_point(output.point),
            normal: xf.rotation.rotate(output.normal),
            fraction: output.fraction,
        }
    }

    /// Queries the physics world for all rigid bodies whose collision shapes intersect `aabb`,
    /// each body once.
    #[must_use]
    pub fn query_aabb(&self, aabb: &Aabb2d, filter: QueryFilter2d) -> Vec<BodyId2d> {
        let mut hits = Vec::new();
        self.query_trees(aabb, |collider_slot| {
            let slot = self.collider_bodies[collider_slot as usize];
            if let Some((_, collider)) = self
                .collider_at(collider_slot)
                .filter(|(_, col)| filter.accepts(&col.filter))
            {
                let tight_aabb = collider
                    .shape
                    .compute_aabb(&self.bodies[slot as usize].transform());
                if tight_aabb.intersects(aabb) {
                    if let Some(id) = self.body_ids.handle_at(slot) {
                        if !hits.contains(&id) {
                            hits.push(id);
                        }
                    }
                }
            }
            true
        });
        hits
    }

    /// Queries the physics world for the topmost rigid body containing `point`.
    #[must_use]
    pub fn query_point(&self, point: Vec2, filter: QueryFilter2d) -> Option<BodyId2d> {
        let point_aabb = Aabb2d::new(point - Vec2::splat(1e-4), point + Vec2::splat(1e-4));
        let mut hit_body = None;

        self.query_trees(&point_aabb, |collider_slot| {
            let slot = self.collider_bodies[collider_slot as usize];
            let body = &self.bodies[slot as usize];
            if let Some((_, collider)) = self
                .collider_at(collider_slot)
                .filter(|(_, col)| filter.accepts(&col.filter))
            {
                let local_point = body.rotation.inverse().rotate(point - body.position);
                if collider.shape.test_point(local_point) {
                    hit_body = self.body_ids.handle_at(slot);
                    return false;
                }
            }
            true
        });

        hit_body
    }

    /// The collider in collider slot `slot` and its handle.
    fn collider_at(&self, slot: u32) -> Option<(ColliderId2d, &Collider2d)> {
        let id = self.collider_ids.handle_at(slot)?;
        Some((id, self.colliders[slot as usize].as_ref()?))
    }

    /// The body slot that collider `id` is attached to, or `None` if the handle is stale.
    fn collider_body_slot(&self, id: ColliderId2d) -> Option<u32> {
        self.collider_ids
            .contains(id)
            .then(|| self.collider_bodies[id.index() as usize])
    }

    /// Detaches collider `id` from its body and removes its broadphase proxy, leaving the
    /// body's mass alone.
    fn remove_collider(&mut self, id: ColliderId2d) {
        let Some(body_slot) = self.collider_body_slot(id) else {
            return;
        };
        let slot = id.index();
        self.destroy_proxy(slot);
        // A destroyed sensor ends its overlaps (Box2D `b2DestroySensor`).
        for visitor in self.sensor_overlaps[slot as usize].drain(..) {
            self.pending_sensor_end_events.push(SensorEndEvent2d {
                sensor: id,
                visitor,
            });
        }
        // Unlink it from its body's list.
        let next = self.collider_next[slot as usize].take();
        if self.body_first_collider[body_slot as usize] == Some(slot) {
            self.body_first_collider[body_slot as usize] = next;
        } else if let Some(previous) =
            collider_slots(&self.body_first_collider, &self.collider_next, body_slot)
                .find(|&c| self.collider_next[c as usize] == Some(slot))
        {
            self.collider_next[previous as usize] = next;
        }
        self.colliders[slot as usize] = None;
        self.collider_chains[slot as usize] = None;
        if self.sensors_cached {
            self.sensor_slots.retain(|&s| s != slot);
        }
        self.collider_ids.free(id);
    }

    /// The margin of collider `collider`'s fat box on a body of type `body_type` (Box2D
    /// `b2UpdateShapeAABBs`): the speculative distance for static bodies, otherwise an eighth of
    /// the shape's size up to [`WorldConfig2d::aabb_margin`].
    fn proxy_margin(&self, collider: &Collider2d, body_type: BodyType) -> f32 {
        if body_type == BodyType::Static {
            self.config.speculative_distance()
        } else {
            self.config
                .aabb_margin()
                .min(AABB_MARGIN_FRACTION * collider.shape.margin_size())
        }
    }

    /// Creates collider slot `c`'s proxy in its body type's tree, at the body's pose, and marks
    /// it moved so that it looks for pairs (Box2D's default `invokeContactCreation`).
    fn create_proxy(&mut self, c: u32) {
        let body = &self.bodies[self.collider_bodies[c as usize] as usize];
        let Some(collider) = self.colliders[c as usize].as_ref() else {
            return;
        };
        let tree = tree_of(body.body_type);
        let aabb = collider
            .shape
            .compute_aabb(&body.transform())
            .fatten(self.config.speculative_distance());
        let fat = aabb.fatten(self.proxy_margin(collider, body.body_type));
        let id = self.trees[tree].insert_proxy(fat, c, true);
        self.collider_proxies[c as usize] = Some(id * TREE_COUNT + tree as u32);
        self.collider_aabbs[c as usize] = aabb;
        self.collider_fat_aabbs[c as usize] = fat;
    }

    /// Removes collider slot `c`'s proxy, if it has one.
    fn destroy_proxy(&mut self, c: u32) {
        if let Some(key) = self.collider_proxies[c as usize].take() {
            self.trees[(key % TREE_COUNT) as usize].destroy_proxy(key / TREE_COUNT);
        }
    }

    /// Recomputes collider slot `c`'s box at its body's pose; if it left the fat box, grows a
    /// new fat box around it and gives it to the proxy as `update` says, marking it moved.
    fn refresh_proxy(&mut self, c: u32, update: ProxyUpdate) {
        let Some(key) = self.collider_proxies[c as usize] else {
            return;
        };
        let body = &self.bodies[self.collider_bodies[c as usize] as usize];
        let Some(collider) = self.colliders[c as usize].as_ref() else {
            return;
        };
        let aabb = collider
            .shape
            .compute_aabb(&body.transform())
            .fatten(self.config.speculative_distance());
        let margin = self.proxy_margin(collider, body.body_type);
        self.collider_aabbs[c as usize] = aabb;
        if self.collider_fat_aabbs[c as usize].contains_aabb(&aabb) {
            return;
        }
        let fat = aabb.fatten(margin);
        self.collider_fat_aabbs[c as usize] = fat;
        let (tree, proxy) = (
            &mut self.trees[(key % TREE_COUNT) as usize],
            key / TREE_COUNT,
        );
        match update {
            ProxyUpdate::Reinsert => tree.reinsert_proxy(proxy, fat, true),
            ProxyUpdate::Mark => tree.mark_proxy_moved(proxy, fat),
            ProxyUpdate::Enlarge => tree.enlarge_proxy(proxy, fat),
        }
    }

    /// Creates the proxies of body slot `slot`'s colliders.
    fn create_body_proxies(&mut self, slot: u32) {
        let mut collider = self.body_first_collider[slot as usize];
        while let Some(c) = collider {
            self.create_proxy(c);
            collider = self.collider_next[c as usize];
        }
    }

    /// Removes the proxies of body slot `slot`'s colliders.
    fn destroy_body_proxies(&mut self, slot: u32) {
        let mut collider = self.body_first_collider[slot as usize];
        while let Some(c) = collider {
            self.destroy_proxy(c);
            collider = self.collider_next[c as usize];
        }
    }

    /// Refreshes the proxies of body slot `slot`'s colliders.
    fn refresh_body_proxies(&mut self, slot: u32, update: ProxyUpdate) {
        let mut collider = self.body_first_collider[slot as usize];
        while let Some(c) = collider {
            self.refresh_proxy(c, update);
            collider = self.collider_next[c as usize];
        }
    }

    /// Finds the new pairs of moved proxies and creates their contacts, not yet touching (Box2D
    /// `b2UpdateBroadPhasePairs`). The dynamic tree collides with itself, under every marked
    /// node, and with the static and kinematic trees, so a pair always has a dynamic body. Then
    /// the dynamic and kinematic trees rebuild their moved subtrees. Pairs pass the same filters
    /// as Box2D's: different bodies, no sensor, colliding shape types, the collision filters and
    /// custom filter, and no joint that disables it.
    fn update_pairs(&mut self) {
        let [static_tree, kinematic_tree, dynamic_tree] = &mut self.trees;
        if !(static_tree.has_moved()
            || kinematic_tree.needs_rebuild()
            || dynamic_tree.needs_rebuild())
        {
            return;
        }
        let pairs = &mut self.contact_pairs;
        pairs.clear();
        {
            let mut push = |a: u32, b: u32| pairs.push((a.min(b), a.max(b)));
            dynamic_tree.moved_pairs_with(static_tree, &mut push);
            dynamic_tree.moved_pairs_with(kinematic_tree, &mut push);
        }
        let self_pairs = dynamic_tree.sibling_pair_count().saturating_sub(1);
        match self.executor.as_ref().map(|e| e.0.as_ref()) {
            Some(executor) if executor.worker_count() > 1 && self_pairs >= 64 => {
                let workers = executor.worker_count();
                let min_cap = pairs.capacity().max(64);
                if self.worker_contact_pairs.len() < workers {
                    self.worker_contact_pairs
                        .resize_with(workers, || Vec::with_capacity(min_cap));
                }
                for buf in &mut self.worker_contact_pairs[..workers] {
                    buf.clear();
                    if buf.capacity() < min_cap {
                        buf.reserve(min_cap - buf.len());
                    }
                }
                let worker_pairs = Shared::new(&mut self.worker_contact_pairs);
                let dyn_tree: &DynamicTree2d = dynamic_tree;
                parallel_for(executor, self_pairs, 64, &|start, end, worker| {
                    // SAFETY: each worker receives its own index `worker < workers` and runs
                    // at most one block at a time.
                    let out = unsafe { worker_pairs.get_mut(worker) };
                    dyn_tree.moved_pairs_range(start + 1, end + 1, |a, b| {
                        out.push((a.min(b), a.max(b)));
                    });
                });
                let mut max_worker_len = min_cap;
                for buf in &mut self.worker_contact_pairs[..workers] {
                    max_worker_len = max_worker_len.max(buf.len());
                    pairs.extend_from_slice(buf);
                    buf.clear();
                }
                for buf in &mut self.worker_contact_pairs[..workers] {
                    if buf.capacity() < max_worker_len {
                        buf.reserve(max_worker_len - buf.len());
                    }
                }
            }
            _ => {
                dynamic_tree.moved_pairs(|a, b| pairs.push((a.min(b), a.max(b))));
            }
        }
        static_tree.clear_moved();
        // Box2D sorts the pairs so that contacts follow collider order.
        self.contact_pairs.sort_unstable();

        for i in 0..self.contact_pairs.len() {
            let key = self.contact_pairs[i];
            if self.contact_manager.contains_key(&key) {
                continue;
            }
            let (i_a, i_b) = (
                self.collider_bodies[key.0 as usize],
                self.collider_bodies[key.1 as usize],
            );
            let (Some(col_a), Some(col_b)) = (
                self.colliders[key.0 as usize].as_ref(),
                self.colliders[key.1 as usize].as_ref(),
            ) else {
                continue;
            };
            if i_a == i_b
                || col_a.is_sensor
                || col_b.is_sensor
                || !shapes_can_collide(&col_a.shape, &col_b.shape)
                || self
                    .joint_filter_pairs
                    .binary_search(&(i_a.min(i_b), i_a.max(i_b)))
                    .is_ok()
                || !colliders_may_collide(
                    &mut self.custom_filter,
                    &self.collider_ids,
                    (key.0, col_a),
                    (key.1, col_b),
                )
            {
                continue;
            }
            let material = mix_materials(col_a, col_b, self.friction_mixer, self.restitution_mixer);
            let mut contact =
                Contact2d::new(key, (i_a, i_b), ContactManifold2d::default(), material);
            contact.recycle = self.bodies[i_a as usize].enable_contact_recycling
                && self.bodies[i_b as usize].enable_contact_recycling;
            let slot = self.contact_manager.insert(key, contact);
            // Awake if either body is (Box2D `b2CreateContact`); else it waits, not updated.
            let set = if self.body_links[i_a as usize].set == AWAKE_SET
                || self.body_links[i_b as usize].set == AWAKE_SET
            {
                AWAKE_SET
            } else {
                DISABLED_SET
            };
            self.add_contact_to_set(slot, set, false);
        }
    }

    /// Updates every awake contact (Box2D `b2Collide`), then applies the changes in contact
    /// order (Box2D's contact state loop): a contact whose fat boxes no longer overlap is
    /// destroyed; one that starts touching reports a begin event, links its bodies' islands,
    /// waking a sleeping one, and joins the constraint graph; one that stops touching reports an
    /// end event, unlinks and leaves the graph.
    fn collide(&mut self) {
        self.contact_changes.clear();
        self.recycled_contacts = 0;
        // The awake contacts: the touching ones in the graph, the others in the awake set; a
        // contact is in one or the other.
        self.collide_slots.clear();
        self.collide_slots.extend(
            self.constraint_graph
                .contact_slots()
                .chain(self.sets[AWAKE_SET as usize].contacts.iter().copied()),
        );
        self.collide_outcomes.clear();
        self.collide_outcomes
            .resize(self.collide_slots.len(), CollideOutcome::Updated);
        {
            let Self {
                contact_manager,
                bodies,
                body_links,
                colliders,
                collider_ids,
                collider_fat_aabbs,
                pre_solve,
                friction_mixer,
                restitution_mixer,
                collide_slots,
                collide_outcomes,
                executor,
                config,
                ..
            } = self;
            let recycle_distance = config.contact_recycle_distance;
            let inputs = CollideInputs {
                bodies,
                body_links,
                colliders,
                collider_ids,
                collider_fat_aabbs,
                friction_mixer: *friction_mixer,
                restitution_mixer: *restitution_mixer,
                linear_slop: config.linear_slop(),
                recycle_distance,
                // A contact that does not touch must not miss the start of a touch.
                recycle_distance_apart: recycle_distance.min(config.speculative_distance()),
            };
            let executor = executor.as_ref().map(|e| e.0.as_ref());
            match (pre_solve.as_mut(), executor) {
                // Box2D `b2Collide`: a parallel loop over the contacts. The pre-solve callback
                // may change state, so with one the contacts are updated here, in order.
                (None, Some(executor)) if executor.worker_count() > 1 => {
                    let contacts = Shared::new(contact_manager.slots_mut());
                    let outcomes = Shared::new(collide_outcomes);
                    let slots = &collide_slots[..];
                    parallel_for(executor, slots.len(), 64, &|start, end, _| {
                        for (i, &slot) in slots.iter().enumerate().take(end).skip(start) {
                            // SAFETY: each awake contact is in one place, the graph or the
                            // awake set, so its slot appears once in `slots`, and each index
                            // is in one block of the loop.
                            let (contact, outcome) =
                                unsafe { (contacts.get_mut(slot as usize), outcomes.get_mut(i)) };
                            let contact = contact.as_mut().expect("live contact");
                            *outcome = update_contact(&inputs, contact, None);
                        }
                    });
                }
                (mut pre_solve, _) => {
                    for (i, &slot) in collide_slots.iter().enumerate() {
                        collide_outcomes[i] = update_contact(
                            &inputs,
                            contact_manager.at_mut(slot),
                            pre_solve.as_deref_mut(),
                        );
                    }
                }
            }
        }
        for (&slot, &outcome) in self.collide_slots.iter().zip(&self.collide_outcomes) {
            let change = match outcome {
                CollideOutcome::Updated => continue,
                CollideOutcome::Recycled => {
                    self.recycled_contacts += 1;
                    continue;
                }
                CollideOutcome::Disjoint => ContactChange::Disjoint,
                CollideOutcome::Started => ContactChange::Started,
                CollideOutcome::Stopped => ContactChange::Stopped,
            };
            self.contact_changes.push((slot, change));
        }
        self.contact_changes.sort_unstable_by_key(|&(slot, _)| slot);
        for i in 0..self.contact_changes.len() {
            let (slot, change) = self.contact_changes[i];
            match change {
                ContactChange::Disjoint => self.destroy_contact(slot, true),
                ContactChange::Started => {
                    let contact = self.contact_manager.at(slot);
                    if let Some(event) = touch_event(&self.colliders, &self.collider_ids, contact) {
                        self.contact_events.begin.push(ContactBeginEvent2d {
                            collider_a: event.collider_a,
                            collider_b: event.collider_b,
                        });
                    }
                    self.link_contact(slot);
                    self.remove_contact_from_set(slot);
                    self.add_contact_to_set(slot, AWAKE_SET, true);
                }
                ContactChange::Stopped => {
                    let contact = self.contact_manager.at(slot);
                    self.contact_events.end.extend(end_event(
                        &self.colliders,
                        &self.collider_ids,
                        contact,
                    ));
                    self.unlink_contact(slot);
                    self.remove_contact_from_set(slot);
                    self.add_contact_to_set(slot, AWAKE_SET, false);
                }
            }
        }
    }

    /// Destroys contact slot `slot` (Box2D `b2DestroyContact`). A touching contact reports an
    /// end event, with this step's events while stepping and with the next step's otherwise,
    /// and wakes both bodies.
    fn destroy_contact(&mut self, slot: u32, in_step: bool) {
        let contact = self.contact_manager.at(slot);
        let (touching, a, b) = (contact.is_touching, contact.body_a, contact.body_b);
        if touching {
            let event = end_event(&self.colliders, &self.collider_ids, contact);
            if in_step {
                self.contact_events.end.extend(event);
            } else {
                self.pending_end_events.extend(event);
            }
        }
        if self.contact_manager.at(slot).island != NULL {
            self.unlink_contact(slot);
        }
        self.remove_contact_from_set(slot);
        self.contact_manager.remove_slot(slot);
        if touching {
            self.wake_body(a);
            self.wake_body(b);
        }
    }

    /// Recomputes and caches the smallest and largest extents of body slot `slot`'s colliders
    /// about its centre of mass (Box2D `b2BodySim::minExtent`, `maxExtent`).
    fn refresh_body_extents(&mut self, slot: u32) {
        let (min_extent, max_extent) = extents_of(
            &self.colliders,
            &self.body_first_collider,
            &self.collider_next,
            &self.bodies[slot as usize],
            slot as usize,
        )
        .unwrap_or((f32::MAX, 0.0));
        let links = &mut self.body_links[slot as usize];
        links.min_extent = min_extent;
        links.max_extent = max_extent;
    }

    /// Destroys the contacts of body slot `slot` between steps, waking the bodies it touched.
    fn destroy_body_contacts(&mut self, slot: u32) {
        let mut key = self.contact_manager.body_head(slot);
        while key != NULL {
            let next = self.contact_manager.at(key >> 1).edges[(key & 1) as usize].next;
            self.destroy_contact(key >> 1, false);
            key = next;
        }
    }

    /// Recomputes body slot `slot`'s mass from its collider (Box2D `b2UpdateBodyMassData`).
    fn update_mass(&mut self, slot: u32) {
        // Sum the masses and first moments, then each shape's inertia about its centroid moved
        // to the centre of mass, in Box2D's order and rounding.
        let parts = || {
            collider_slots(&self.body_first_collider, &self.collider_next, slot).filter_map(|c| {
                let collider = self.colliders[c as usize].as_ref()?;
                Some(collider.shape.compute_mass_properties(collider.density))
            })
        };
        let (mut mass, mut moment) = (0.0, Vec2::ZERO);
        for part in parts() {
            mass += part.mass;
            moment += part.center * part.mass;
        }
        let mass_data = if mass > 0.0 {
            let center = moment * (1.0 / mass);
            let rotational_inertia = parts()
                .filter(|part| part.mass != 0.0)
                .map(|part| {
                    let offset = center - part.center;
                    part.rotational_inertia + part.mass * offset.dot(offset)
                })
                .fold(0.0, |sum, inertia| sum + inertia);
            MassData2d {
                mass,
                center,
                rotational_inertia,
            }
        } else {
            MassData2d::default()
        };
        // The centre moves with the mass; the velocity follows the body's material points.
        let old_center = self.body_links[slot as usize].center;
        let body = &mut self.bodies[slot as usize];
        body.apply_mass_data(mass_data);
        let center = body.world_center();
        let offset = center - old_center;
        body.linear_velocity += Vec2::new(
            -body.angular_velocity * offset.y,
            body.angular_velocity * offset.x,
        );
        self.body_links[slot as usize].center = center;
        self.mass_moved(slot);
        self.refresh_body_extents(slot);
    }

    /// Notes that body slot `slot`'s centre of mass may have moved: its contacts' anchors are
    /// relative to it, so they are collided again rather than recycled (Box2D
    /// `b2Body_SetMassData`).
    fn mass_moved(&mut self, slot: u32) {
        let body = &self.bodies[slot as usize];
        let local_center = body.local_center;
        let links = &mut self.body_links[slot as usize];
        links.touched = false;
        if links.local_center == local_center {
            return;
        }
        links.local_center = local_center;
        links.center = body.world_center();
        self.refresh_body_extents(slot);
        let mut key = self.contact_manager.body_head(slot);
        while key != NULL {
            let contact = self.contact_manager.at_mut(key >> 1);
            contact.relative_valid = false;
            key = contact.edges[(key & 1) as usize].next;
        }
    }

    /// Steps the physics simulation forward by `dt` seconds deterministically.
    pub fn step(&mut self, dt: f32) {
        // Clear the events first, so that none is stale after an early return.
        self.contact_events.start_step(&mut self.pending_end_events);
        self.sensor_events
            .start_step(&mut self.pending_sensor_end_events);
        self.body_events.clear();
        self.joint_events.clear();
        if !(dt > 0.0 && dt.is_finite()) {
            return;
        }
        // Box2D's substep rate: the substep count over the step, not one over the substep.
        let sub_steps = self.config.sub_steps.max(1);
        let inv_h = sub_steps as f32 * (1.0 / dt);
        self.inv_h = inv_h;
        let step_start = Instant::now();
        if !self.config.enable_sleep {
            self.wake_sleeping_bodies();
        }
        self.step_index += 1;

        let mut profile = Profile2d::default();
        let mut mark = step_start;

        // Reconcile any bodies handed out by `body_mut` or `collider_mut`.
        self.flush_touched_bodies();

        // Body pairs whose joints disable their collision (Box2D `b2ShouldBodiesCollide`).
        self.joint_filter_pairs.clear();
        self.joint_filter_pairs.extend(
            self.joints
                .iter()
                .flatten()
                .filter(|j| !j.collide_connected)
                .map(|j| {
                    let (a, b) = j.slots();
                    (a.min(b), a.max(b))
                }),
        );
        self.joint_filter_pairs.sort_unstable();
        self.joint_filter_pairs.dedup();

        // 1. New pairs of the proxies that moved since the last step.
        self.update_pairs();
        profile.broadphase = lap(&mut mark);

        // 2. Update the contacts: destroy the ones whose fat boxes separated, collide the rest.
        self.collide();
        profile.narrowphase = lap(&mut mark);

        // 4. Soft Step solve (Box2D v3 `b2SolverStep`): prepare the constraints, then per substep
        //    integrate velocities, warm start, solve with bias, integrate positions and relax;
        //    after the substeps apply restitution and store the impulses. The stages run on the
        //    executor's workers (see `solve`).
        if !self.sets[AWAKE_SET as usize].bodies.is_empty() {
            let h = dt / sub_steps as f32;
            let (contact_softness, static_softness) = self.config.contact_softness(h);
            self.joint_thresholds_reached.clear();
            self.joint_thresholds_reached
                .resize(self.joints.len(), false);
            let params = SolveParams {
                gravity: self.gravity,
                h,
                inv_h,
                sub_steps: sub_steps as usize,
                max_linear_speed: self.config.maximum_linear_speed,
                max_angular_speed: MAX_ROTATION / dt,
                contact_softness,
                static_softness,
                push_speed: self.config.contact_push_speed,
                restitution_threshold: self.config.restitution_threshold,
                restitution_iterations: self
                    .config
                    .restitution_iterations
                    .min(crate::config::MAX_RESTITUTION_ITERATIONS),
                restitution_propagation: self.config.enable_restitution_propagation,
                enable_warm_starting: self.config.enable_warm_starting,
                length_units: self.config.length_units_per_meter,
            };
            solve::solve(
                SolveData {
                    bodies: &self.bodies,
                    body_links: &self.body_links,
                    awake: &self.sets[AWAKE_SET as usize].bodies,
                    joint_bodies: &self.joint_bodies,
                    graph: &self.constraint_graph,
                    joints: &mut self.joints,
                    joint_steps: &mut self.joint_steps,
                    thresholds_reached: &mut self.joint_thresholds_reached,
                    contacts: self.contact_manager.slots_mut(),
                    trees: &mut self.trees[KINEMATIC_TREE..=DYNAMIC_TREE],
                    scratch: &mut self.solver,
                    executor: self.executor.as_ref().map(|e| e.0.as_ref()),
                },
                &params,
            );

            // Split the island picked last step (Box2D runs it beside the solver stages).
            if self.island_to_split != NULL {
                let island = self.island_to_split;
                self.split_island(island);
                self.island_to_split = NULL;
            }

            profile.solve = lap(&mut mark);
            self.finalize_bodies(dt);
            profile.continuous = lap(&mut mark);
            self.report_hits();
            for (slot, &reached) in self.joint_thresholds_reached.iter().enumerate() {
                if let (true, Some(joint)) = (reached, self.joint_ids.handle_at(slot as u32)) {
                    self.joint_events.push(JointEvent2d { joint });
                }
            }
        } else {
            self.trees[KINEMATIC_TREE].rebuild(false);
            self.trees[DYNAMIC_TREE].rebuild(false);
        }

        profile.solve += lap(&mut mark);

        // 6. Islands (Box2D's end of `b2SolverStep`): the sleepiest island that needs splitting
        //    is split during the next step, and the islands no body kept awake fall asleep.
        if self.config.enable_sleep && !self.awake_islands.is_empty() {
            self.island_to_split = self.split_candidate.0;
            for local in (0..self.awake_islands.len()).rev() {
                if !self.awake_islands[local] {
                    let island = self.sets[AWAKE_SET as usize].islands[local];
                    self.try_sleep_island(island);
                }
            }
        }
        self.awake_islands.clear();
        if self.config.enable_sleep || !self.fast_bullets.is_empty() {
            for event in &self.body_events {
                self.body_links[event.body.index() as usize].move_event = NULL;
            }
        }

        profile.islands = lap(&mut mark);
        self.overlap_sensors();
        profile.sensors = lap(&mut mark);

        // Forces and torques of awake bodies are cleared in `finalize_bodies`; clear all only when
        // no awake bodies stepped.
        if self.sets[AWAKE_SET as usize].bodies.is_empty() {
            for body in &mut self.bodies {
                body.clear_forces();
            }
        }
        #[cfg(debug_assertions)]
        if self.bodies.len() <= 2_048 {
            self.validate_sets();
        }
        profile.step = (Instant::now() - step_start).as_secs_f32() * 1000.0;
        self.profile = profile;
    }

    /// A snapshot of the whole simulation state (Box2D `b2World_GetSnapshot`): see
    /// [`WorldSnapshot2d`]. Take it between steps.
    #[must_use]
    pub fn create_snapshot(&self) -> WorldSnapshot2d {
        WorldSnapshot2d {
            world: self.clone_state(),
        }
    }

    /// Restores the world in place to `snapshot` (Box2D `b2World_Restore`): it continues bit
    /// for bit as it did from the snapshot. Handles held at the snapshot stay valid; objects
    /// created since are gone and their handles stale. The callbacks and material mixers stay.
    /// Reuses the world's allocations where they suffice.
    pub fn restore_snapshot(&mut self, snapshot: &WorldSnapshot2d) {
        // Every field is named, so that a new one cannot be forgotten.
        let Self {
            bodies,
            body_ids,
            body_first_collider,
            colliders,
            collider_bodies,
            collider_next,
            collider_proxies,
            collider_aabbs,
            collider_fat_aabbs,
            collider_chains,
            collider_ids,
            chains,
            chain_ids,
            trees,
            joints,
            joint_ids,
            gravity,
            config,
            step_index,
            contact_manager,
            body_links,
            joint_bodies,
            joint_links,
            sets,
            free_sets,
            islands,
            free_islands,
            island_count,
            island_to_split,
            split_candidate: _,
            split_scratch: _,
            awake_islands: _,
            touched_bodies,
            recycled_contacts,
            contact_changes: _,
            collide_slots: _,
            collide_outcomes: _,
            constraint_graph,
            contact_pairs: _,
            worker_contact_pairs: _,
            candidates_scratch: _,
            fast_bullets: _,
            joint_filter_pairs: _,
            step_frames: _,
            solver: _,
            executor: _,
            contact_events,
            pending_end_events,
            body_events,
            sensor_overlaps,
            sensor_slots: _,
            sensors_cached: _,
            sensor_scratch: _,
            sensor_hits: _,
            sensor_events,
            pending_sensor_end_events,
            joint_events,
            joint_thresholds_reached: _,
            joint_steps,
            inv_h,
            pre_solve: _,
            pre_continuous: _,
            custom_filter: _,
            profile: _,
            friction_mixer: _,
            restitution_mixer: _,
        } = &snapshot.world;
        self.bodies.clone_from(bodies);
        self.body_ids.clone_from(body_ids);
        self.body_first_collider.clone_from(body_first_collider);
        self.colliders.clone_from(colliders);
        self.collider_bodies.clone_from(collider_bodies);
        self.collider_next.clone_from(collider_next);
        self.collider_proxies.clone_from(collider_proxies);
        self.collider_aabbs.clone_from(collider_aabbs);
        self.collider_fat_aabbs.clone_from(collider_fat_aabbs);
        self.collider_chains.clone_from(collider_chains);
        self.collider_ids.clone_from(collider_ids);
        self.chains.clone_from(chains);
        self.chain_ids.clone_from(chain_ids);
        self.trees.clone_from(trees);
        self.joints.clone_from(joints);
        self.joint_ids.clone_from(joint_ids);
        self.gravity = *gravity;
        self.config = *config;
        self.step_index = *step_index;
        self.contact_manager.clone_from(contact_manager);
        self.body_links.clone_from(body_links);
        self.joint_bodies.clone_from(joint_bodies);
        self.joint_links.clone_from(joint_links);
        self.sets.clone_from(sets);
        self.free_sets.clone_from(free_sets);
        self.islands.clone_from(islands);
        self.free_islands.clone_from(free_islands);
        self.island_count = *island_count;
        self.island_to_split = *island_to_split;
        self.touched_bodies.clone_from(touched_bodies);
        self.recycled_contacts = *recycled_contacts;
        self.constraint_graph.clone_from(constraint_graph);
        self.contact_events.clone_from(contact_events);
        self.pending_end_events.clone_from(pending_end_events);
        self.body_events.clone_from(body_events);
        self.sensor_overlaps.clone_from(sensor_overlaps);
        self.rebuild_sensor_slots();
        self.sensor_events.clone_from(sensor_events);
        self.pending_sensor_end_events
            .clone_from(pending_sensor_end_events);
        self.joint_events.clone_from(joint_events);
        self.joint_steps.clone_from(joint_steps);
        self.inv_h = *inv_h;
    }

    /// A new world from `snapshot` (Box2D `b2CreateWorldFromSnapshot`), without callbacks or
    /// material mixers. Handles from the snapshot's world work in it.
    #[must_use]
    pub fn from_snapshot(snapshot: &WorldSnapshot2d) -> Self {
        snapshot.world.clone_state()
    }

    /// A deterministic hash of the simulation state (Box2D `b2World_GetStateHash`): body
    /// poses and velocities, contact and joint impulses, and the solver sets, colors and
    /// indices that order the solve. Two worlds with the same hash simulate the same; compare
    /// hashes to detect a desync in a rollback or replay. The value is Korppi's, not Box2D's.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        let mut hash = Fnv64::new();
        for slot in 0..self.bodies.len() as u32 {
            if self.body_ids.handle_at(slot).is_none() {
                continue;
            }
            let body = &self.bodies[slot as usize];
            let links = &self.body_links[slot as usize];
            hash.floats(&[
                body.position.x,
                body.position.y,
                body.rotation.c,
                body.rotation.s,
                body.linear_velocity.x,
                body.linear_velocity.y,
                body.angular_velocity,
            ]);
            hash.ints(&[links.set, links.local]);
        }
        for (_, contact) in self.contact_manager.iter() {
            hash.ints(&[
                contact.set,
                contact.color,
                contact.local,
                contact.manifold.point_count as u32,
            ]);
            for point in &contact.manifold.points[..contact.manifold.point_count] {
                hash.floats(&[
                    point.normal_impulse,
                    point.tangent_impulse,
                    point.total_normal_impulse,
                ]);
            }
        }
        for slot in 0..self.joints.len() {
            let Some(joint) = &self.joints[slot] else {
                continue;
            };
            let links = &self.joint_links[slot];
            hash.ints(&[links.set, links.color, links.local]);
            joint
                .joint
                .hash_impulses(&mut |value| hash.floats(&[value]));
        }
        for (slots, live) in [
            (self.body_ids.slot_count(), self.body_ids.len()),
            (self.collider_ids.slot_count(), self.collider_ids.len()),
            (self.chain_ids.slot_count(), self.chain_ids.len()),
            (
                self.contact_manager.slot_count(),
                self.contact_manager.len(),
            ),
            (self.joint_ids.slot_count(), self.joint_ids.len()),
            (self.islands.len(), self.island_count),
            (self.sets.len(), self.free_sets.len()),
        ] {
            hash.ints(&[slots as u32, live as u32]);
        }
        hash.0
    }

    /// A copy of the world's simulation state, without host wiring and per-step scratch.
    pub(crate) fn clone_state(&self) -> Self {
        let mut cloned = Self {
            bodies: self.bodies.clone(),
            body_ids: self.body_ids.clone(),
            body_first_collider: self.body_first_collider.clone(),
            colliders: self.colliders.clone(),
            collider_bodies: self.collider_bodies.clone(),
            collider_next: self.collider_next.clone(),
            collider_proxies: self.collider_proxies.clone(),
            collider_aabbs: self.collider_aabbs.clone(),
            collider_fat_aabbs: self.collider_fat_aabbs.clone(),
            collider_chains: self.collider_chains.clone(),
            collider_ids: self.collider_ids.clone(),
            chains: self.chains.clone(),
            chain_ids: self.chain_ids.clone(),
            trees: self.trees.clone(),
            joints: self.joints.clone(),
            joint_ids: self.joint_ids.clone(),
            gravity: self.gravity,
            config: self.config,
            step_index: self.step_index,
            contact_manager: self.contact_manager.clone(),
            body_links: self.body_links.clone(),
            joint_bodies: self.joint_bodies.clone(),
            joint_links: self.joint_links.clone(),
            sets: self.sets.clone(),
            free_sets: self.free_sets.clone(),
            islands: self.islands.clone(),
            free_islands: self.free_islands.clone(),
            island_count: self.island_count,
            island_to_split: self.island_to_split,
            split_candidate: Default::default(),
            split_scratch: Default::default(),
            awake_islands: Default::default(),
            touched_bodies: self.touched_bodies.clone(),
            recycled_contacts: self.recycled_contacts,
            contact_changes: Default::default(),
            collide_slots: Default::default(),
            collide_outcomes: Default::default(),
            constraint_graph: self.constraint_graph.clone(),
            contact_pairs: Default::default(),
            worker_contact_pairs: Default::default(),
            candidates_scratch: Default::default(),
            fast_bullets: Default::default(),
            joint_filter_pairs: Default::default(),
            step_frames: Default::default(),
            solver: Default::default(),
            executor: Default::default(),
            contact_events: self.contact_events.clone(),
            pending_end_events: self.pending_end_events.clone(),
            body_events: self.body_events.clone(),
            sensor_overlaps: self.sensor_overlaps.clone(),
            sensor_slots: Vec::new(),
            sensors_cached: false,
            sensor_scratch: Default::default(),
            sensor_hits: Default::default(),
            sensor_events: self.sensor_events.clone(),
            pending_sensor_end_events: self.pending_sensor_end_events.clone(),
            joint_events: self.joint_events.clone(),
            joint_thresholds_reached: Default::default(),
            joint_steps: self.joint_steps.clone(),
            inv_h: self.inv_h,
            pre_solve: Default::default(),
            pre_continuous: Default::default(),
            custom_filter: Default::default(),
            profile: Default::default(),
            friction_mixer: Default::default(),
            restitution_mixer: Default::default(),
        };
        cloned.flush_touched_bodies();
        cloned.rebuild_sensor_slots();
        cloned
    }

    /// Rebuilds `sensor_slots` from `colliders` after a snapshot restore or `collider_mut`.
    fn rebuild_sensor_slots(&mut self) {
        self.sensor_slots.clear();
        for (slot, col) in self.colliders.iter().enumerate() {
            if col.as_ref().is_some_and(|c| c.is_sensor) {
                self.sensor_slots.push(slot as u32);
            }
        }
        self.sensors_cached = true;
    }

    /// Recomputes every sensor's overlaps at the end of the step and reports the changes (Box2D
    /// `b2OverlapSensors`). A sensor overlaps a collider of another body that passes the filter,
    /// enables sensor events like the sensor, and lies within `10 * f32::EPSILON` of it, or that
    /// crossed it during continuous collision. A disabled sensor body overlaps nothing.
    fn overlap_sensors(&mut self) {
        if !self.sensors_cached {
            self.rebuild_sensor_slots();
        }
        if self.sensor_slots.is_empty() {
            self.sensor_hits.clear();
            return;
        }
        for s_idx in 0..self.sensor_slots.len() {
            let slot = self.sensor_slots[s_idx];
            let Some(sensor) = self.colliders[slot as usize].as_ref() else {
                continue;
            };
            let Some(sensor_id) = self.collider_ids.handle_at(slot) else {
                continue;
            };
            if !sensor.is_sensor {
                continue;
            }
            self.sensor_scratch.clear();
            self.sensor_scratch.extend(
                self.sensor_hits
                    .iter()
                    .filter(|&&(s, _)| s == slot)
                    .map(|&(_, visitor)| visitor),
            );
            let body_slot = self.collider_bodies[slot as usize];
            let body = &self.bodies[body_slot as usize];
            let proxy = ShapeProxy2d::of_shape(&sensor.shape);
            if body.enabled && sensor.enable_sensor_events {
                let xf = body.transform();
                self.candidates_scratch.clear();
                let candidates = &mut self.candidates_scratch;
                let sensor_box = sensor.shape.compute_aabb(&xf);
                for tree in &self.trees {
                    tree.query(&sensor_box, |_node, other| {
                        candidates.push(other);
                        true
                    });
                }
                for &other in &self.candidates_scratch {
                    let Some(visitor) = self.colliders[other as usize].as_ref() else {
                        continue;
                    };
                    let visitor_body = self.collider_bodies[other as usize];
                    if other == slot
                        || visitor_body == body_slot
                        || !visitor.enable_sensor_events
                        || !colliders_may_collide(
                            &mut self.custom_filter,
                            &self.collider_ids,
                            (slot, sensor),
                            (other, visitor),
                        )
                    {
                        continue;
                    }
                    let relative = xf.inv_mul(&self.bodies[visitor_body as usize].transform());
                    let output = shape_distance(
                        &proxy,
                        &ShapeProxy2d::of_shape(&visitor.shape),
                        &relative,
                        &mut SimplexCache2d::default(),
                        true,
                    );
                    let overlaps = output.distance < 10.0 * f32::EPSILON;
                    if overlaps {
                        self.sensor_scratch
                            .extend(self.collider_ids.handle_at(other));
                    }
                }
            }
            self.sensor_scratch.sort_unstable();
            self.sensor_scratch.dedup();

            // Both lists are sorted: walk them together for what ended and what began.
            let (old, new) = (&self.sensor_overlaps[slot as usize], &self.sensor_scratch);
            let (mut i, mut j) = (0, 0);
            while i < old.len() || j < new.len() {
                if j == new.len() || (i < old.len() && old[i] < new[j]) {
                    self.sensor_events.end.push(SensorEndEvent2d {
                        sensor: sensor_id,
                        visitor: old[i],
                    });
                    i += 1;
                } else if i == old.len() || new[j] < old[i] {
                    self.sensor_events.begin.push(SensorBeginEvent2d {
                        sensor: sensor_id,
                        visitor: new[j],
                    });
                    j += 1;
                } else {
                    i += 1;
                    j += 1;
                }
            }
            std::mem::swap(
                &mut self.sensor_scratch,
                &mut self.sensor_overlaps[slot as usize],
            );
        }
        self.sensor_hits.clear();
    }

    /// Reports the step's hits (Box2D's hit events in `b2SolverStep`): for each solved contact
    /// with a collider that enables hit events, the point that pushed and approached fastest,
    /// if faster than the threshold. The point is rebuilt from a body's centre and the point's
    /// anchor, from the static body if there is one.
    fn report_hits(&mut self) {
        let threshold = self.config.hit_event_threshold;
        for slot in self.constraint_graph.contact_slots() {
            let contact = self.contact_manager.at(slot);
            let manifold = &contact.manifold;
            let mut approach_speed = threshold;
            let mut best = None;
            for point in &manifold.points[..manifold.point_count] {
                // A speculative point that never pushed did not hit.
                if -point.normal_velocity > approach_speed && point.total_normal_impulse > 0.0 {
                    approach_speed = -point.normal_velocity;
                    best = Some(point);
                }
            }
            let Some(point) = best else {
                continue;
            };
            let hit_events = |c: u32| {
                self.colliders[c as usize]
                    .as_ref()
                    .is_some_and(|col| col.enable_hit_events)
            };
            if !hit_events(contact.collider_a) && !hit_events(contact.collider_b) {
                continue;
            }
            let (Some(collider_a), Some(collider_b)) = (
                self.collider_ids.handle_at(contact.collider_a),
                self.collider_ids.handle_at(contact.collider_b),
            ) else {
                continue;
            };
            let (body_a, body_b) = (
                &self.bodies[contact.body_a as usize],
                &self.bodies[contact.body_b as usize],
            );
            let (body, anchor) =
                if body_a.body_type != BodyType::Static && body_b.body_type == BodyType::Static {
                    (contact.body_b, point.anchor_b)
                } else {
                    (contact.body_a, point.anchor_a)
                };
            // The anchor is relative to the centre of mass.
            self.contact_events.hit.push(ContactHitEvent2d {
                collider_a,
                collider_b,
                point: self.body_links[body as usize].center + anchor,
                normal: manifold.normal,
                approach_speed,
            });
        }
    }

    /// Box2D `b2FinalizeBodies`, then its bullet pass: sleep timers, continuous collision, move
    /// events, and the fat boxes of the colliders that left theirs.
    ///
    /// A body rests while its sleep speed, the speed of its fastest point (or half its average
    /// over the step, if larger), stays under its threshold; its timer then grows, and an island
    /// stays awake while a body's timer is under [`TIME_TO_SLEEP`]. A sleepy body of an island
    /// that lost constraints makes the island a split candidate, the sleepiest winning.
    ///
    /// A dynamic body that is not resting is fast when it moved more than
    /// its `safety_factor` times its smallest extent this step; it is then swept from
    /// its pose at the start of the step and stopped at the first time of impact (Box2D
    /// `b2SolveContinuous`). The velocity is kept, apart from the gravity of the lost time: the
    /// next step's speculative contacts make the response. A fast body is swept against static
    /// bodies and its proxies are marked moved with the others; the trees are refit; then fast
    /// bullets are swept against every non-bullet body, in their final places, and grow their
    /// proxies in place.
    fn finalize_bodies(&mut self, dt: f32) {
        let inv_dt = 1.0 / dt;
        let enable_sleep = self.config.enable_sleep;
        let enable_continuous = self.config.enable_continuous;
        let speculative_distance = self.config.speculative_distance();
        self.fast_bullets.clear();
        self.awake_islands.clear();
        self.awake_islands
            .resize(self.sets[AWAKE_SET as usize].islands.len(), false);
        self.split_candidate = (NULL, 0.0);
        if self.step_frames.len() < self.bodies.len() {
            self.step_frames.resize(
                self.bodies.len(),
                StepFrame2d::of(&RigidBody2d::new(BodyType::Static)),
            );
        }
        let awake_len = self.sets[AWAKE_SET as usize].bodies.len();
        self.body_events.resize(
            awake_len,
            BodyMoveEvent2d {
                body: BodyId2d::from(NULL),
                transform: Transform2d::IDENTITY,
                fell_asleep: false,
            },
        );
        let any_serial_fast = std::sync::atomic::AtomicBool::new(false);
        {
            let aabb_margin = self.config.aabb_margin();
            let awake_bodies = &self.sets[AWAKE_SET as usize].bodies[..];
            let states = &self.solver.states[..];
            let body_ids = &self.body_ids;
            let body_first_collider = &self.body_first_collider[..];
            let collider_next = &self.collider_next[..];
            let colliders = &self.colliders[..];
            let collider_proxies = &self.collider_proxies[..];
            let trees = &self.trees;
            let bodies = Shared::new(&mut self.bodies);
            let body_links = Shared::new(&mut self.body_links);
            let step_frames = Shared::new(&mut self.step_frames);
            let body_events = Shared::new(&mut self.body_events);
            let collider_aabbs = Shared::new(&mut self.collider_aabbs);
            let collider_fat_aabbs = Shared::new(&mut self.collider_fat_aabbs);

            let update_range = |start: usize, end: usize| {
                let mut local_serial_fast = false;
                for i in start..end {
                    let slot = awake_bodies[i];
                    let index = slot as usize;
                    let state = states[i];
                    // SAFETY: each awake body appears once in `awake_bodies`, so `index` and
                    // `i` are unique to this iteration, and each collider `c` belongs to at
                    // most one body `slot`.
                    let (body, links, step_frame, event) = unsafe {
                        (
                            bodies.get_mut(index),
                            body_links.get_mut(index),
                            step_frames.get_mut(index),
                            body_events.get_mut(i),
                        )
                    };
                    let frame = StepFrame2d {
                        center: links.center,
                        rotation: body.rotation,
                        delta_position: state.delta_position,
                        delta_rotation: state.delta_rotation,
                    };
                    body.linear_velocity = state.linear_velocity;
                    body.angular_velocity = state.angular_velocity;
                    body.clear_forces();
                    body.rotation = (frame.delta_rotation * frame.rotation).normalize();
                    links.center = frame.center + frame.delta_position;
                    body.position = links.center - body.rotation.rotate(body.local_center);
                    let xf = body.transform();
                    links.pose = xf;
                    *step_frame = frame;
                    let (min_extent, max_extent) = (links.min_extent, links.max_extent);
                    let max_velocity =
                        body.linear_velocity.length() + body.angular_velocity.abs() * max_extent;
                    let max_delta_position =
                        frame.delta_position.length() + frame.delta_rotation.s.abs() * max_extent;
                    let sleep_velocity = max_velocity.max(0.5 * inv_dt * max_delta_position);
                    let mut fast = false;
                    if !enable_sleep || !body.enable_sleep || sleep_velocity > links.sleep_threshold
                    {
                        let max_motion = max_delta_position.max(max_velocity * dt);
                        fast = enable_continuous
                            && body.body_type == BodyType::Dynamic
                            && max_motion > body.safety_factor * min_extent;
                        body.sleep_time = 0.0;
                    } else {
                        body.sleep_time += dt;
                    }
                    links.is_fast = fast;
                    links.move_event = i as u32;
                    *event = BodyMoveEvent2d {
                        body: body_ids.handle_at(slot).expect("awake body is live"),
                        transform: xf,
                        fell_asleep: false,
                    };
                    let mut needs_serial_sweep = fast && body.is_bullet;
                    if fast && !body.is_bullet {
                        let xf0 = Transform2d {
                            translation: frame.center - frame.rotation.rotate(body.local_center),
                            rotation: frame.rotation,
                        };
                        let mut c_opt = body_first_collider[index];
                        while let Some(c) = c_opt {
                            let ci = c as usize;
                            if let Some(collider) = colliders[ci].as_ref() {
                                if !collider.is_sensor {
                                    let swept = collider
                                        .shape
                                        .compute_aabb(&xf0)
                                        .union(&collider.shape.compute_aabb(&xf));
                                    let mut hit_static = false;
                                    trees[STATIC_TREE].query(&swept, |_, _| {
                                        hit_static = true;
                                        false
                                    });
                                    if hit_static {
                                        needs_serial_sweep = true;
                                        break;
                                    }
                                }
                            }
                            c_opt = collider_next[ci];
                        }
                    }
                    local_serial_fast |= needs_serial_sweep;
                    if !needs_serial_sweep {
                        let body_type = body.body_type;
                        let mut c_opt = body_first_collider[index];
                        while let Some(c) = c_opt {
                            let ci = c as usize;
                            if let Some(collider) = colliders[ci].as_ref() {
                                let aabb = collider
                                    .shape
                                    .compute_aabb(&xf)
                                    .fatten(speculative_distance);
                                // SAFETY: collider `ci` belongs only to body `slot`, and its
                                // proxy is unique across all colliders.
                                unsafe {
                                    *collider_aabbs.get_mut(ci) = aabb;
                                    let fat_ref = collider_fat_aabbs.get_mut(ci);
                                    if !fat_ref.contains_aabb(&aabb) {
                                        let margin = if body_type == BodyType::Static {
                                            speculative_distance
                                        } else {
                                            aabb_margin.min(
                                                AABB_MARGIN_FRACTION * collider.shape.margin_size(),
                                            )
                                        };
                                        let fat = aabb.fatten(margin);
                                        *fat_ref = fat;
                                        if let Some(key) = collider_proxies[ci] {
                                            trees[(key % TREE_COUNT) as usize]
                                                .mark_proxy_moved_atomic(key / TREE_COUNT, fat);
                                        }
                                    }
                                }
                            }
                            c_opt = collider_next[ci];
                        }
                    } else {
                        // Mark in `step_frames` that this fast body still needs a serial sweep.
                        links.move_event |= 0x8000_0000;
                    }
                }
                if local_serial_fast {
                    any_serial_fast.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            };

            match self.executor.as_ref().map(|e| e.0.as_ref()) {
                Some(executor) if executor.worker_count() > 1 => {
                    parallel_for(executor, awake_len, 64, &|start, end, _| {
                        update_range(start, end);
                    });
                }
                _ => update_range(0, awake_len),
            }
        }

        let has_serial_fast = any_serial_fast.load(std::sync::atomic::Ordering::Relaxed);
        if has_serial_fast || enable_sleep {
            for i in 0..awake_len {
                let slot = self.sets[AWAKE_SET as usize].bodies[i];
                let index = slot as usize;
                let mut links = self.body_links[index];
                let needs_serial_sweep = (links.move_event & 0x8000_0000) != 0;
                if needs_serial_sweep {
                    links.move_event &= !0x8000_0000;
                    self.body_links[index].move_event = links.move_event;
                }
                let bullet = needs_serial_sweep && self.bodies[index].is_bullet;
                if needs_serial_sweep && !bullet {
                    self.sweep_fast_body(index, self.step_frames[index], links.min_extent, dt);
                    self.body_events[i].transform = self.bodies[index].transform();
                }

                // Any body can keep its island awake.
                if enable_sleep {
                    let sleep_time = self.bodies[index].sleep_time;
                    let island = &self.islands[links.island as usize];
                    if sleep_time < TIME_TO_SLEEP {
                        self.awake_islands[island.local as usize] = true;
                    } else if island.constraint_remove_count > 0 {
                        let (best, best_time) = self.split_candidate;
                        if sleep_time > best_time
                            || (sleep_time == best_time && (best == NULL || links.island > best))
                        {
                            self.split_candidate = (links.island, sleep_time);
                        }
                    }
                }

                if bullet {
                    self.fast_bullets.push(slot);
                } else if needs_serial_sweep {
                    self.refresh_body_proxies(slot, ProxyUpdate::Mark);
                }
            }
        }
        self.trees[KINEMATIC_TREE].refit();
        self.trees[DYNAMIC_TREE].refit();
        for i in 0..self.fast_bullets.len() {
            let index = self.fast_bullets[i] as usize;
            let min_extent = self.body_links[index].min_extent;
            self.sweep_fast_body(index, self.step_frames[index], min_extent, dt);
            self.refresh_body_proxies(index as u32, ProxyUpdate::Enlarge);
            let event = self.body_links[index].move_event;
            if event != NULL {
                self.body_events[event as usize].transform = self.bodies[index].transform();
            }
        }
    }

    /// Sweeps fast body `index` from `frame` to its current pose and stops it at the first
    /// impact of any of its colliders.
    fn sweep_fast_body(&mut self, index: usize, frame: StepFrame2d, min_extent: f32, dt: f32) {
        let linear_slop = self.config.linear_slop();
        let body = &self.bodies[index];
        // As Box2D v3.2, the sweep and the impacts are computed relative to the body's centre at
        // the start of the step, so that they keep their precision far from the world origin.
        let base = frame.center;
        let relative = |xf: Transform2d| Transform2d {
            translation: xf.translation - base,
            rotation: xf.rotation,
        };
        let sweep = Sweep2d {
            local_center: body.local_center,
            c1: frame.center - base,
            c2: self.body_links[index].center - base,
            q1: frame.rotation,
            q2: body.rotation,
        };
        let mut fraction = 1.0f32;
        // Sensors crossed before the first solid hit: (sensor slot, visitor, fraction).
        let mut sensor_hits = [None; MAX_CONTINUOUS_SENSOR_HITS];
        let mut sensor_hit_count = 0;
        for c in collider_slots(&self.body_first_collider, &self.collider_next, index as u32) {
            let Some(collider) = self.colliders[c as usize].as_ref() else {
                continue;
            };
            if collider.is_sensor {
                continue;
            }
            let proxy = ShapeProxy2d::of_shape(&collider.shape);
            let world_box = |time: f32| {
                let xf = sweep.transform(time);
                collider.shape.compute_aabb(&Transform2d {
                    translation: xf.translation + base,
                    rotation: xf.rotation,
                })
            };
            let swept_box = world_box(0.0).union(&world_box(1.0));
            // Fallback when the shapes already overlap: a small circle in the shape's core.
            let centroid = collider.shape.centroid();
            let core = ShapeProxy2d::new(
                &[centroid],
                CORE_FRACTION * collider.shape.extent(centroid).0,
            );
            // The shape's centroid where the sweep starts and ends.
            let centroid1 = sweep.transform(0.0).transform_point(centroid);
            let centroid2 = sweep.transform(1.0).transform_point(centroid);

            self.candidates_scratch.clear();
            let candidates = &mut self.candidates_scratch;
            // A fast body is swept against static bodies; a bullet against every body.
            let trees: &[DynamicTree2d] = if body.is_bullet {
                &self.trees
            } else {
                &self.trees[STATIC_TREE..=STATIC_TREE]
            };
            for tree in trees {
                tree.query(&swept_box, |_node, other| {
                    candidates.push(other);
                    true
                });
            }

            for &other_collider_slot in &self.candidates_scratch {
                let other_id = self.collider_bodies[other_collider_slot as usize];
                let other_index = other_id as usize;
                let other = &self.bodies[other_index];
                let Some(other_collider) = self.colliders[other_collider_slot as usize].as_ref()
                else {
                    continue;
                };
                let pair = ((index as u32).min(other_id), (index as u32).max(other_id));
                let allowed = if body.is_bullet {
                    !other.is_bullet
                } else {
                    other.body_type == BodyType::Static
                };
                // A sensor is crossed only if both colliders want sensor events.
                let is_sensor = other_collider.is_sensor;
                if other_index == index
                    || !allowed
                    || (is_sensor
                        && !(other_collider.enable_sensor_events && collider.enable_sensor_events))
                    || self.joint_filter_pairs.binary_search(&pair).is_ok()
                    || !colliders_may_collide(
                        &mut self.custom_filter,
                        &self.collider_ids,
                        (other_collider_slot, other_collider),
                        (c, collider),
                    )
                {
                    continue;
                }
                // A fast body that started behind a chain segment, or only skims along it,
                // ignores it (Box2D's "minimal clipping" in `b2ContinuousQueryCallback`).
                if let Shape2d::ChainSegment(chain_segment) = &other_collider.shape {
                    let xf = relative(other.transform());
                    let p1 = xf.transform_point(chain_segment.segment.point1);
                    let edge = xf.transform_point(chain_segment.segment.point2) - p1;
                    let length = edge.length();
                    if length > linear_slop {
                        let e = edge / length;
                        let cross = |v: Vec2| v.x * e.y - v.y * e.x;
                        let separation1 = cross(centroid1 - p1);
                        let separation2 = cross(centroid2 - p1);
                        let core = CORE_FRACTION * min_extent;
                        if separation1 < 0.0
                            || (separation1 - separation2 < core && separation2 > core)
                        {
                            continue;
                        }
                    }
                }
                let other_center = self.body_links[other_index].center - base;
                let other_sweep = Sweep2d {
                    local_center: other.local_center,
                    c1: other_center,
                    c2: other_center,
                    q1: other.rotation,
                    q2: other.rotation,
                };
                {
                    let other_proxy = ShapeProxy2d::of_shape(&other_collider.shape);
                    let mut input = ToiInput2d {
                        proxy_a: other_proxy,
                        proxy_b: proxy,
                        sweep_a: other_sweep,
                        sweep_b: sweep,
                        max_fraction: fraction,
                        linear_slop,
                    };
                    let mut output = time_of_impact(&input);
                    if is_sensor {
                        // Starting inside the sensor counts too (fraction 0).
                        if output.fraction <= fraction && sensor_hit_count < sensor_hits.len() {
                            sensor_hits[sensor_hit_count] = self
                                .collider_ids
                                .handle_at(c)
                                .map(|visitor| (other_collider_slot, visitor, output.fraction));
                            sensor_hit_count += 1;
                        }
                        continue;
                    }
                    if output.fraction == 0.0 {
                        input.proxy_b = core;
                        output = time_of_impact(&input);
                    }
                    if 0.0 < output.fraction && output.fraction < fraction {
                        let mut hit = true;
                        if let Some(PreContinuous2d(pre_continuous)) = &mut self.pre_continuous {
                            if collider.enable_pre_solve_events
                                || other_collider.enable_pre_solve_events
                            {
                                if let (Some(a), Some(b)) = (
                                    self.collider_ids.handle_at(other_collider_slot),
                                    self.collider_ids.handle_at(c),
                                ) {
                                    hit = pre_continuous(a, b, base + output.point, output.normal);
                                }
                            }
                        }
                        if hit {
                            fraction = output.fraction;
                        }
                    }
                }
            }
        }

        // Keep the sensors crossed before the body stopped.
        self.sensor_hits.extend(
            sensor_hits[..sensor_hit_count]
                .iter()
                .flatten()
                .filter(|&&(_, _, hit_fraction)| hit_fraction < fraction)
                .map(|&(sensor, visitor, _)| (sensor, visitor)),
        );

        if fraction < 1.0 {
            let rotation = Rot2::new(
                (1.0 - fraction) * sweep.q1.c + fraction * sweep.q2.c,
                (1.0 - fraction) * sweep.q1.s + fraction * sweep.q2.s,
            )
            .normalize();
            let center = sweep.c1 * (1.0 - fraction) + sweep.c2 * fraction;
            let gravity = self.gravity;
            let body = &mut self.bodies[index];
            body.rotation = rotation;
            body.position = base + (center - rotation.rotate(body.local_center));
            self.body_links[index].center = base + center;
            self.body_links[index].pose = body.transform();
            // The body lost the rest of the step, and with it that much gravity.
            let mut lost = gravity * (-(1.0 - fraction) * dt * body.gravity_scale);
            if body.motion_locks.linear_x {
                lost.x = 0.0;
            }
            if body.motion_locks.linear_y {
                lost.y = 0.0;
            }
            body.linear_velocity += lost;
        }
    }
}

/// What the narrowphase did to a contact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CollideOutcome {
    /// Its manifold was recomputed and it kept touching or not.
    Updated,
    /// Its manifold was recycled.
    Recycled,
    /// Its colliders' fat boxes separated, or a collider is gone: it is to be destroyed.
    Disjoint,
    /// Its manifold gained points.
    Started,
    /// Its manifold lost its points.
    Stopped,
}

/// What [`update_contact`] reads of the world.
struct CollideInputs<'a> {
    bodies: &'a [RigidBody2d],
    body_links: &'a [BodyLinks2d],
    colliders: &'a [Option<Collider2d>],
    collider_ids: &'a IdPool<Collider2d>,
    collider_fat_aabbs: &'a [Aabb2d],
    friction_mixer: Option<MaterialMixer2d>,
    restitution_mixer: Option<MaterialMixer2d>,
    linear_slop: f32,
    recycle_distance: f32,
    recycle_distance_apart: f32,
}

/// Updates one awake contact (Box2D `b2CollideTask`): recycles its manifold while its bodies
/// have barely moved relative to each other, else recomputes it, calling `pre_solve` on a new
/// manifold with points.
fn update_contact(
    inputs: &CollideInputs<'_>,
    contact: &mut Contact2d,
    pre_solve: Option<&mut PreSolve2d>,
) -> CollideOutcome {
    let CollideInputs {
        bodies,
        body_links,
        colliders,
        collider_ids,
        collider_fat_aabbs,
        friction_mixer,
        restitution_mixer,
        linear_slop,
        recycle_distance,
        recycle_distance_apart,
    } = *inputs;
    let key = (contact.collider_a, contact.collider_b);
    let (Some(col_a), Some(col_b)) = (
        colliders[key.0 as usize].as_ref(),
        colliders[key.1 as usize].as_ref(),
    ) else {
        return CollideOutcome::Disjoint;
    };
    if !collider_fat_aabbs[key.0 as usize].intersects(&collider_fat_aabbs[key.1 as usize]) {
        return CollideOutcome::Disjoint;
    }
    let (a, b) = (contact.body_a as usize, contact.body_b as usize);
    let (body_a, body_b) = (&bodies[a], &bodies[b]);
    let (xf_a, xf_b) = (body_a.transform(), body_b.transform());
    let was_touching = contact.is_touching;

    // Contact recycling (Box2D `b2CollideTask`): while the bodies have barely moved relative to
    // each other since the last real collide, keep the manifold and only update the
    // separations, as the substeps do.
    let fast = body_links[a].is_fast || body_links[b].is_fast;
    if !fast && recycle_distance > 0.0 && contact.relative_valid && contact.recycle {
        let pose = xf_a.inv_mul(&xf_b);
        let cached = contact.cached_relative_pose;
        let min_cos = relative_cos(xf_a.rotation, contact.cached_rotation_a)
            .min(relative_cos(xf_b.rotation, contact.cached_rotation_b));
        let max_extent = |body: &RigidBody2d, slot: usize| {
            if body.body_type == BodyType::Static {
                0.0
            } else {
                body_links[slot].max_extent
            }
        };
        let max_extent = max_extent(body_a, a).max(max_extent(body_b, b));
        let distance = (cached.translation - pose.translation).length();
        let turn = (pose.rotation.inverse() * cached.rotation).s;
        let tolerance = if was_touching {
            recycle_distance
        } else {
            recycle_distance_apart
        };
        if min_cos > RECYCLE_COS_ANGLE && distance + max_extent * turn.abs() < tolerance {
            let dq_a = xf_a.rotation * contact.cached_rotation_a.inverse();
            let dq_b = xf_b.rotation * contact.cached_rotation_b.inverse();
            let normal = contact.manifold.normal;
            let dc = body_links[b].center - body_links[a].center;
            let manifold = &mut contact.manifold;
            for point in &mut manifold.points[..manifold.point_count] {
                let r_a = dq_a.rotate(point.anchor_a);
                let r_b = dq_b.rotate(point.anchor_b);
                let dp = dc + (r_b - r_a);
                point.separation = point.base_separation + dp.dot(normal);
                point.normal_velocity = 0.0;
                point.persisted = true;
            }
            return CollideOutcome::Recycled;
        }
    }
    contact.cached_rotation_a = xf_a.rotation;
    contact.cached_rotation_b = xf_b.rotation;
    contact.cached_relative_pose = xf_a.inv_mul(&xf_b);
    contact.relative_valid = true;

    let mut manifold = collide_shapes_cached(
        &col_a.shape,
        &xf_a,
        &col_b.shape,
        &xf_b,
        linear_slop,
        &mut contact.cache,
    )
    .unwrap_or_default();
    if let (true, Some(PreSolve2d(pre_solve))) = (manifold.point_count > 0, pre_solve) {
        if col_a.enable_pre_solve_events || col_b.enable_pre_solve_events {
            if let (Some(a), Some(b)) =
                (collider_ids.handle_at(key.0), collider_ids.handle_at(key.1))
            {
                pre_solve(a, b, &mut manifold);
            }
        }
    }
    contact.material = mix_materials(col_a, col_b, friction_mixer, restitution_mixer);
    // Anchors relative to the centres of mass, as Box2D's `b2UpdateContact` keeps them.
    let offset_a = xf_a.rotation.rotate(body_a.local_center);
    let offset_b = xf_b.rotation.rotate(body_b.local_center);
    for point in &mut manifold.points[..manifold.point_count] {
        point.anchor_a -= offset_a;
        point.anchor_b -= offset_b;
        point.base_separation = point.separation;
    }
    contact.update(manifold);
    match (was_touching, contact.is_touching) {
        (false, true) => CollideOutcome::Started,
        (true, false) => CollideOutcome::Stopped,
        _ => CollideOutcome::Updated,
    }
}

/// The surface properties of a contact between colliders `a` and `b` (Box2D
/// `b2UpdateContact`): friction and restitution by the mixers or `sqrt(a * b)` and `max(a, b)`,
/// rolling resistance as the larger coefficient times the larger shape radius, and the sum of
/// the tangent speeds.
fn mix_materials(
    a: &Collider2d,
    b: &Collider2d,
    friction_mixer: Option<MaterialMixer2d>,
    restitution_mixer: Option<MaterialMixer2d>,
) -> ContactMaterial2d {
    let (id_a, id_b) = (a.user_material_id, b.user_material_id);
    let friction = friction_mixer.map_or_else(
        || (a.friction * b.friction).sqrt(),
        |mix| mix(a.friction, id_a, b.friction, id_b),
    );
    let restitution = restitution_mixer.map_or_else(
        || a.restitution.max(b.restitution),
        |mix| mix(a.restitution, id_a, b.restitution, id_b),
    );
    let rolling_resistance = if a.rolling_resistance > 0.0 || b.rolling_resistance > 0.0 {
        a.rolling_resistance.max(b.rolling_resistance) * a.shape.radius().max(b.shape.radius())
    } else {
        0.0
    };
    ContactMaterial2d {
        friction,
        restitution,
        rolling_resistance,
        tangent_speed: a.tangent_speed + b.tangent_speed,
    }
}

/// Milliseconds since `mark`, moving `mark` to now.
fn lap(mark: &mut Instant) -> f32 {
    let now = Instant::now();
    let ms = (now - *mark).as_secs_f32() * 1000.0;
    *mark = now;
    ms
}

/// The end event of `contact`, if one of its colliders enables contact events.
/// The colliders of `contact` for a begin event, if either enables contact events.
fn touch_event(
    colliders: &[Option<Collider2d>],
    collider_ids: &IdPool<Collider2d>,
    contact: &Contact2d,
) -> Option<ContactEndEvent2d> {
    end_event(colliders, collider_ids, contact)
}

fn end_event(
    colliders: &[Option<Collider2d>],
    collider_ids: &IdPool<Collider2d>,
    contact: &Contact2d,
) -> Option<ContactEndEvent2d> {
    let enabled = |c: u32| {
        colliders[c as usize]
            .as_ref()
            .is_some_and(|col| col.enable_contact_events)
    };
    if !enabled(contact.collider_a) && !enabled(contact.collider_b) {
        return None;
    }
    Some(ContactEndEvent2d {
        collider_a: collider_ids.handle_at(contact.collider_a)?,
        collider_b: collider_ids.handle_at(contact.collider_b)?,
    })
}

/// Broadphase tree of static bodies.
const STATIC_TREE: usize = 0;
/// Broadphase tree of kinematic bodies.
const KINEMATIC_TREE: usize = 1;
/// Broadphase tree of dynamic bodies.
const DYNAMIC_TREE: usize = 2;
/// Number of broadphase trees, and the stride of proxy keys.
const TREE_COUNT: u32 = 3;

/// How a refresh gives a proxy its new fat box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProxyUpdate {
    /// Remove and insert it again (Box2D `b2BroadPhase_MoveProxy`), for teleports.
    Reinsert,
    /// Set the leaf's box and mark its ancestors, refit later (Box2D
    /// `b2BroadPhase_MarkProxyMoved`), for bodies moved by the solver.
    Mark,
    /// Grow the leaf and its ancestors in place (Box2D `b2DynamicTree_EnlargeProxy`), for
    /// bullets after the refit.
    Enlarge,
}

/// Fraction of a shape's size its fat box reaches beyond it, up to the configured margin
/// (Box2D `B2_AABB_MARGIN_FRACTION`).
const AABB_MARGIN_FRACTION: f32 = 0.125;

/// The broadphase tree of bodies of type `body_type`.
const fn tree_of(body_type: BodyType) -> usize {
    match body_type {
        BodyType::Static => STATIC_TREE,
        BodyType::Kinematic => KINEMATIC_TREE,
        BodyType::Dynamic => DYNAMIC_TREE,
    }
}

/// Whether the shapes can make contacts (Box2D `b2CanCollide`): segments and chains do not
/// collide with each other.
const fn shapes_can_collide(a: &Shape2d, b: &Shape2d) -> bool {
    !matches!(
        (a, b),
        (
            Shape2d::Segment(_) | Shape2d::ChainSegment(_),
            Shape2d::Segment(_) | Shape2d::ChainSegment(_)
        )
    )
}

/// Most sensors one fast body can cross in a step (Box2D `B2_MAX_CONTINUOUS_SENSOR_HITS`).
const MAX_CONTINUOUS_SENSOR_HITS: usize = 8;

/// Radius of the fallback core circle, as a fraction of the smallest extent, used when a fast
/// shape already overlaps an obstacle (Box2D `B2_CORE_FRACTION`).
const CORE_FRACTION: f32 = 0.25;

/// 64-bit FNV-1a over 32-bit words (Box2D's snapshot hash).
struct Fnv64(u64);

impl Fnv64 {
    const fn new() -> Self {
        Self(14_695_981_039_346_656_037)
    }

    fn word(&mut self, word: u32) {
        self.0 = (self.0 ^ u64::from(word)).wrapping_mul(1_099_511_628_211);
    }

    fn floats(&mut self, values: &[f32]) {
        values.iter().for_each(|v| self.word(v.to_bits()));
    }

    fn ints(&mut self, values: &[u32]) {
        values.iter().for_each(|&v| self.word(v));
    }
}

/// The bounding box of `proxy`'s points grown by its radius (Box2D `b2MakeAABB`).
fn proxy_bounds(proxy: &ShapeProxy2d) -> Aabb2d {
    let points = &proxy.points[..proxy.count];
    let (lower, upper) = points[1..]
        .iter()
        .fold((points[0], points[0]), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    let r = Vec2::splat(proxy.radius);
    Aabb2d::new(lower - r, upper + r)
}

/// World-space shape cast `input` in the frame `xf`, up to `max_fraction`.
fn local_cast(input: &ShapeCastInput2d, xf: &Transform2d, max_fraction: f32) -> ShapeCastInput2d {
    let mut local = *input;
    for point in &mut local.proxy.points[..local.proxy.count] {
        *point = xf.inv_transform_point(*point);
    }
    local.translation = xf.rotation.inv_rotate(input.translation);
    local.max_fraction = max_fraction;
    local
}

/// The force and torque of wind `wind` on `shape` of `body` (Box2D `b2Shape_ApplyWind`):
/// drag `0.5 rho A v^2` on the projected area of unit depth, air density 1.225 kg/m^3.
fn wind_force(
    shape: &Shape2d,
    body: &RigidBody2d,
    wind: Vec2,
    drag: f32,
    lift: f32,
    length_units: f32,
) -> (Vec2, f32) {
    use crate::math::{
        cross, cross_sv, left_perp, length_and_normalize, lerp, normalize, right_perp,
    };
    let air_density = 1.225 / (length_units * length_units * length_units);
    let q = body.rotation;
    let lever = q.rotate(shape.centroid() - body.local_center);
    let shape_velocity = body.linear_velocity + cross_sv(body.angular_velocity, lever);
    let relative = wind - shape_velocity * drag;
    let (speed, direction) = length_and_normalize(relative);
    let lift_direction = |normal: Vec2| cross_sv(cross(normal, direction), direction);
    match shape {
        Shape2d::Circle(c) => {
            let area = 2.0 * c.radius;
            let force = direction * (0.5 * air_density * area * speed * speed);
            (force, cross(lever, force))
        }
        Shape2d::Capsule(c) => {
            let d = q.rotate(c.point2 - c.point1);
            let area = 2.0 * c.radius + cross(d, direction).abs();
            // The normal that faces the wind.
            let mut normal = left_perp(normalize(d));
            if normal.dot(direction) > 0.0 {
                normal = -normal;
            }
            let magnitude = 0.5 * air_density * area * speed * speed;
            let force = (direction + lift_direction(normal) * lift) * magnitude;
            let edge_lever = lever + normal * c.radius;
            (force, cross(edge_lever, force))
        }
        Shape2d::Polygon(p) => {
            // The rounding radius is left out.
            let vertices = p.vertices();
            let mut v1 = vertices[vertices.len() - 1];
            let (mut force, mut torque) = (Vec2::ZERO, 0.0);
            for &v2 in vertices {
                let edge_center = lerp(v1, v2, 0.5);
                let d = q.rotate(v2 - v1);
                v1 = v2;
                let area = cross(d, direction);
                if area <= 0.0 {
                    // Facing away.
                    continue;
                }
                let normal = right_perp(normalize(d));
                let magnitude = 0.5 * air_density * area * speed * speed;
                let f = (direction + lift_direction(normal) * lift) * magnitude;
                let edge_lever = q.rotate(edge_center - body.local_center);
                force += f;
                torque += cross(edge_lever, f);
            }
            (force, torque)
        }
        Shape2d::Segment(_) | Shape2d::ChainSegment(_) => (Vec2::ZERO, 0.0),
    }
}

/// The collider slots of one body, in creation order.
struct ColliderSlots<'a> {
    next: &'a [Option<u32>],
    current: Option<u32>,
}

impl Iterator for ColliderSlots<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let slot = self.current?;
        self.current = self.next[slot as usize];
        Some(slot)
    }
}

/// The collider slots of body slot `body`. Takes the world's fields rather than the world, so
/// that callers can iterate while they update other fields.
fn collider_slots<'a>(
    first: &[Option<u32>],
    next: &'a [Option<u32>],
    body: u32,
) -> ColliderSlots<'a> {
    ColliderSlots {
        next,
        current: first[body as usize],
    }
}

/// Box2D's `B2_CONTACT_RECYCLE_COS_ANGLE`: a contact is not recycled once a body turned more
/// than about 11.5° since its last real collide.
const RECYCLE_COS_ANGLE: f32 = 0.98;

/// The cosine of the angle between rotations `a` and `b` (Box2D `b2RelativeCos`).
fn relative_cos(a: Rot2, b: Rot2) -> f32 {
    a.c * b.c + a.s * b.s
}

/// The smallest and largest extents of body slot `slot`'s colliders about its centre of mass
/// (Box2D `minExtent`, `maxExtent`), or `None` without colliders.
fn extents_of(
    colliders: &[Option<Collider2d>],
    body_first_collider: &[Option<u32>],
    collider_next: &[Option<u32>],
    body: &RigidBody2d,
    slot: usize,
) -> Option<(f32, f32)> {
    collider_slots(body_first_collider, collider_next, slot as u32)
        .filter_map(|c| colliders[c as usize].as_ref())
        .map(|col| col.shape.extent(body.local_center))
        .reduce(|(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)))
}

/// How a contact changed during the collide pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContactChange {
    /// The fat boxes no longer overlap, or a collider is gone: the contact is destroyed.
    Disjoint,
    /// The manifold gained points.
    Started,
    /// The manifold lost its points.
    Stopped,
}
