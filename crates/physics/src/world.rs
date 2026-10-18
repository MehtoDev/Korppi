//! PhysicsWorld managing simulation ticks, broadphase trees, and constraint solving.

use crate::body::{BodyType, RigidBody};
use crate::bvh::DynamicTree;
use crate::collider::{Collider, QueryFilter3d};
use crate::contact::{Contact3d, ContactManager3d};
use crate::contact_solver::SPECULATIVE_DISTANCE;
use crate::distance::{
    shape_cast, shape_distance, CastOutput, ShapeProxy, SimplexCache, Sweep, OVERLAP_SLOP,
};
use crate::events::{
    BodyEvents3d, BodyMoveEvent3d, ContactBeginEvent3d, ContactEndEvent3d, ContactEvents3d,
    ContactHitEvent3d, JointEvent3d, JointEvents3d, SensorBeginEvent3d, SensorEndEvent3d,
    SensorEvents3d,
};
use crate::island::IslandManager3d;
use crate::joint::Joint;
use crate::math::{inv_mul_quat, inv_rotate, lerp, modified_cross, nlerp, Transform};
use crate::mesh_contact::{update_mesh_manifolds, MeshPair, MeshScratch};
use crate::mover::{collide_mover_shape, CapsuleMover, PlaneResult3d};
use crate::narrowphase::{
    collide_convex_shapes, collide_shapes_cached, shape_time_of_impact, shapes_overlap,
    triangles_of, CollisionBody, ContactManifold, Convex, MAX_MANIFOLD_POINTS,
};
use crate::shape::Shape;
use crate::snapshot::{BodySnapshot3d, WorldSnapshot3d};
use crate::solver::{SolverConfig3d, SolverScratch3d};
use korppi_core::{Handle, IdPool};
use korppi_math::{Aabb, Mat3, Quat, Ray, Vec3};
use serde::{Deserialize, Serialize};

/// A generational handle referencing a 3D rigid body in [`PhysicsWorld`].
pub type BodyId3d = Handle<RigidBody>;
/// Type alias for [`BodyId3d`].
pub type BodyId = BodyId3d;

/// A generational handle referencing a 3D collider in [`PhysicsWorld`].
pub type ColliderId3d = Handle<Collider>;
/// Type alias for [`ColliderId3d`].
pub type ColliderId = ColliderId3d;

/// A generational handle referencing a 3D joint in [`PhysicsWorld`].
pub type JointId3d = Handle<Joint>;
/// Type alias for [`JointId3d`].
pub type JointId = JointId3d;

/// An iterator over the live joints in the world and their handles.
pub struct Joints<'a> {
    pool: &'a IdPool<Joint>,
    joints: &'a [Option<Joint>],
    current: u32,
    remaining: usize,
}

impl<'a> Joints<'a> {
    /// Creates a new `Joints` iterator over the active joints in `pool`.
    #[must_use]
    pub fn new(pool: &'a IdPool<Joint>, joints: &'a [Option<Joint>]) -> Self {
        Self {
            pool,
            joints,
            current: 0,
            remaining: pool.len(),
        }
    }

    /// Returns an iterator over references to the active joints.
    pub fn iter(self) -> impl Iterator<Item = &'a Joint> + 'a {
        self.map(|(_, j)| j)
    }
}

impl<'a> Iterator for Joints<'a> {
    type Item = (JointId3d, &'a Joint);

    fn next(&mut self) -> Option<Self::Item> {
        while (self.current as usize) < self.joints.len() {
            let idx = self.current;
            self.current += 1;
            if let Some(id) = self.pool.handle_at(idx) {
                if let Some(Some(joint)) = self.joints.get(idx as usize) {
                    self.remaining = self.remaining.saturating_sub(1);
                    return Some((id, joint));
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for Joints<'a> {
    fn len(&self) -> usize {
        self.remaining
    }
}

/// Box3D's angular recycling threshold, `cos(angle / 2)^2` for a 10 degree turn
/// (`B3_CONTACT_RECYCLE_ANGULAR_DISTANCE`).
const CONTACT_RECYCLE_ANGULAR_DISTANCE: f32 = 0.992_403_9;

/// Broadphase tree index for static bodies (Box3D `b3_staticBody`).
pub const STATIC_TREE: usize = 0;
/// Broadphase tree index for kinematic bodies (Box3D `b3_kinematicBody`).
pub const KINEMATIC_TREE: usize = 1;
/// Broadphase tree index for dynamic bodies (Box3D `b3_dynamicBody`).
pub const DYNAMIC_TREE: usize = 2;
/// Number of broadphase dynamic trees (Box3D `b3_bodyTypeCount`).
pub const TREE_COUNT: usize = 3;

/// State transition recorded for an awake contact during `update_contacts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContactChange3d {
    Disjoint,
    Started,
    Stopped,
}

/// Outcome recorded per awake contact slot during parallel narrowphase (`update_contacts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CollideOutcome3d {
    Unchanged { touching: bool },
    Recycled { touching: bool },
    Disjoint,
    Started,
    Stopped,
    Deferred,
}

#[derive(Clone)]
pub(crate) struct Executor3d(pub(crate) std::sync::Arc<dyn korppi_core::TaskExecutor>);

impl std::fmt::Debug for Executor3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Executor3d")
            .field("worker_count", &self.0.worker_count())
            .finish()
    }
}

/// A collider in the world: its body, its definition and its broadphase proxy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColliderSlot {
    /// The body owning this collider.
    #[serde(with = "crate::serde_ids::handle")]
    pub body: BodyId3d,
    /// Collider definition including shape and physical properties.
    pub collider: Collider,
    /// Broadphase dynamic tree proxy identifier.
    #[serde(default)]
    pub proxy: u32,
    /// Speculative-expanded tight AABB in world space (Box3D `b3Shape::aabb`).
    #[serde(default)]
    pub aabb: Aabb,
    /// Broadphase fat AABB stored in the dynamic tree (Box3D `world->fatAABBs[shapeId]`).
    #[serde(default)]
    pub fat_aabb: Aabb,
    /// Whether this collider is currently queued in `move_buffer`.
    #[serde(default)]
    pub moved: bool,
}

impl ColliderSlot {
    /// The collider's shape pose on `body`.
    fn pose(&self, body: &RigidBody) -> (Vec3, Quat) {
        self.collider.world_pose(body.position, body.rotation)
    }

    /// The shape's bounds on `body`.
    fn bounds(&self, body: &RigidBody) -> Aabb {
        let (position, rotation) = self.pose(body);
        self.collider.shape.compute_aabb(position, rotation)
    }

    /// The collider as the narrowphase sees it on `body`.
    fn collision_body<'a>(&'a self, body: &RigidBody) -> CollisionBody<'a> {
        let (position, rotation) = self.pose(body);
        CollisionBody::new(self.body.index(), position, rotation, &self.collider)
    }
}

/// What a world holds and did in its last step (Box3D `b3Counters`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters3d {
    /// Bodies.
    pub body_count: usize,
    /// Colliders.
    pub collider_count: usize,
    /// Contacts, touching or not.
    pub contact_count: usize,
    /// Contacts with manifold points.
    pub touching_contact_count: usize,
    /// Contacts recycled in the last step.
    pub recycled_contact_count: usize,
    /// Joints.
    pub joint_count: usize,
}

/// Profiling data for physics simulation phases in milliseconds (Box3D `b3Profile`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Profile3d {
    /// Total step duration in milliseconds.
    pub step: f32,
    /// Broadphase pair search duration in milliseconds.
    pub broadphase: f32,
    /// Narrowphase collision detection duration in milliseconds.
    pub narrowphase: f32,
    /// Constraint solver execution duration in milliseconds.
    pub solve: f32,
    /// Continuous collision sweeping duration in milliseconds.
    pub continuous: f32,
}

/// Result of a 3D ray-cast query against the physics world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit3d {
    /// ID of the hit rigid body.
    pub body_id: BodyId3d,
    /// ID of the hit collider.
    pub collider_id: ColliderId3d,
    /// Hit position in world space.
    pub point: Vec3,
    /// Surface outward normal at hit point in world space.
    pub normal: Vec3,
    /// Distance parameter along the ray.
    pub fraction: f32,
    /// User material identifier at hit point (Box3D `b3RayResult::userMaterialId`).
    pub user_material_id: u64,
}

/// Result of a 3D shape-cast query against the physics world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeCastHit3d {
    /// ID of the hit rigid body.
    pub body_id: BodyId3d,
    /// ID of the hit collider.
    pub collider_id: ColliderId3d,
    /// Hit position in world space.
    pub point: Vec3,
    /// Surface outward normal at hit point in world space.
    pub normal: Vec3,
    /// Fraction along the sweep translation where collision occurred in `[0.0, 1.0]`.
    pub fraction: f32,
    /// GJK iterations performed during the shape cast.
    pub iterations: u32,
    /// User material identifier at hit point (Box3D `b3ShapeCastResult::userMaterialId`).
    pub user_material_id: u64,
}

/// Definition of an explosive impulse in world space (Box3D `b3ExplosionDef`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExplosionDef3d {
    /// Center of the explosion in world space.
    pub position: Vec3,
    /// Radius of the explosion sphere.
    pub radius: f32,
    /// Distance over which the explosion impulse falls off to zero beyond `radius`.
    pub falloff: f32,
    /// Impulse per unit projected area applied to exposed rigid bodies.
    pub impulse_per_area: f32,
}

/// Main 3D physics simulation world.
#[derive(Debug, Serialize, Deserialize)]
pub struct PhysicsWorld {
    pub(crate) bodies: Vec<RigidBody>,
    #[serde(with = "crate::serde_ids::id_pool")]
    pub(crate) body_ids: IdPool<RigidBody>,
    /// The colliders, by collider ID.
    #[serde(skip)]
    colliders: Vec<Option<ColliderSlot>>,
    #[serde(skip)]
    collider_ids: IdPool<Collider>,
    /// Each body's colliders, in the order they were created.
    #[serde(skip)]
    body_colliders: Vec<Vec<ColliderId3d>>,
    #[serde(skip)]
    trees: [DynamicTree; TREE_COUNT],
    /// Persistent constraint graph managing independent colors for parallel solving (Box3D `b3ConstraintGraph`).
    #[serde(skip)]
    pub constraint_graph: crate::constraint_graph::ConstraintGraph3d,
    gravity: Vec3,
    solver_config: SolverConfig3d,
    step_index: u64,
    /// The contacts between collider pairs the broadphase found, touching or not.
    #[serde(skip)]
    pub(crate) contacts: ContactManager3d,
    /// Lazily populated cache of touching contact manifolds for `PhysicsWorld::manifolds()`.
    #[serde(skip)]
    manifolds_cache: std::sync::OnceLock<Vec<ContactManifold>>,
    /// The contacts with manifolds this step.
    #[serde(skip)]
    touching_contacts: usize,
    /// The mesh contacts' working buffers.
    #[serde(skip)]
    mesh_scratch: MeshScratch,
    /// Contacts recycled this step.
    #[serde(skip)]
    recycled_contacts: usize,
    /// The bodies continuous collision found fast last step; their contacts are not recycled.
    #[serde(skip)]
    fast: Vec<bool>,
    #[serde(skip)]
    pub(crate) joints: Vec<Option<Joint>>,
    #[serde(skip)]
    pub(crate) joint_ids: IdPool<Joint>,
    /// Island graph manager tracking connected body islands and sleep states.
    #[serde(skip)]
    pub island_manager: IslandManager3d,
    /// Solver sets: 0 = static, 1 = disabled, 2 = awake, >= 3 = sleeping islands.
    #[serde(skip)]
    pub(crate) sets: Vec<crate::island::SolverSet3d>,
    #[serde(skip)]
    pub(crate) free_sets: Vec<u32>,
    /// Persistent connected islands of bodies, contacts, and joints.
    #[serde(skip)]
    pub(crate) islands: Vec<crate::island::Island3d>,
    #[serde(skip)]
    pub(crate) free_islands: Vec<u32>,
    #[serde(skip)]
    pub(crate) island_count: usize,
    #[serde(skip)]
    pub(crate) island_to_split: u32,
    #[serde(skip)]
    pub(crate) split_candidate: (u32, f32),
    #[serde(skip)]
    pub(crate) split_scratch: crate::island::SplitScratch3d,
    #[serde(skip)]
    pub(crate) awake_islands: Vec<bool>,
    #[serde(skip)]
    pub(crate) body_links: Vec<crate::island::BodyLinks3d>,
    #[serde(skip)]
    pub(crate) joint_links: Vec<crate::island::JointLinks3d>,
    #[serde(skip)]
    pub(crate) awake_contact_slots: Vec<u32>,
    /// Collider indices whose fat AABBs moved since the last broadphase pair search.
    #[serde(skip)]
    move_buffer: Vec<u32>,
    /// Body indices handed out via `body_mut` since the last step.
    #[serde(skip)]
    touched_bodies: Vec<u32>,
    /// Contact state transitions recorded during `update_contacts`.
    #[serde(skip)]
    contact_changes: Vec<(u32, ContactChange3d)>,
    #[serde(skip)]
    collide_outcomes: Vec<CollideOutcome3d>,
    #[serde(skip)]
    executor: Option<Executor3d>,
    #[serde(skip)]
    pub(crate) jointed_pairs_dirty: bool,
    #[serde(skip)]
    pub(crate) sensors_dirty: bool,
    #[serde(skip)]
    pairs_scratch: Vec<(u32, u32)>,
    /// The body pairs, ordered and sorted, joined by a joint that keeps them from colliding.
    #[serde(skip)]
    jointed_pairs: Vec<(u32, u32)>,
    #[serde(skip)]
    solver_scratch: SolverScratch3d,
    /// The inverse substep of the last step, for joint forces.
    #[serde(skip)]
    last_inv_h: f32,
    #[serde(skip)]
    sensor_count: usize,
    #[serde(skip)]
    hit_event_count: usize,
    #[serde(skip)]
    sensor_events: SensorEvents3d,
    /// The colliders that overlap each sensor collider, sorted, by collider ID.
    #[serde(skip)]
    sensor_overlaps: Vec<Vec<ColliderId3d>>,
    #[serde(skip)]
    sensor_scratch: Vec<ColliderId3d>,
    #[serde(skip)]
    sensor_candidates: Vec<u32>,
    /// Sensors a fast body swept through this step, as (sensor, visitor).
    #[serde(skip)]
    sensor_hits: Vec<(ColliderId3d, ColliderId3d)>,
    #[serde(skip)]
    continuous_candidates: Vec<u32>,
    #[serde(skip)]
    bullets: Vec<usize>,
    #[serde(skip)]
    compound_scratch: Vec<ContactManifold>,
    #[serde(skip)]
    contact_events: ContactEvents3d,
    #[serde(skip)]
    pub(crate) body_events: BodyEvents3d,
    #[serde(skip)]
    joint_events: JointEvents3d,
    #[serde(skip)]
    #[allow(dead_code)]
    was_awake: Vec<bool>,
    /// Callback combining surface friction for colliding bodies.
    #[serde(skip, default = "crate::material::default_friction_callback_fn")]
    pub friction_callback: crate::material::FrictionCallback3d,
    /// Callback combining surface restitution for colliding bodies.
    #[serde(skip, default = "crate::material::default_restitution_callback_fn")]
    pub restitution_callback: crate::material::RestitutionCallback3d,
    /// Profiling metrics recorded for each phase of the last simulation step.
    #[serde(skip)]
    pub profile: Profile3d,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    /// Creates a new physics world with standard downward gravity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bodies: Vec::with_capacity(64),
            body_ids: IdPool::new(),
            colliders: Vec::with_capacity(64),
            collider_ids: IdPool::new(),
            trees: [DynamicTree::new(), DynamicTree::new(), DynamicTree::new()],
            constraint_graph: crate::constraint_graph::ConstraintGraph3d::new(
                crate::constraint_graph::GRAPH_COLOR_COUNT,
            ),
            body_colliders: Vec::with_capacity(64),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            solver_config: SolverConfig3d::default(),
            step_index: 0,
            contacts: ContactManager3d::with_capacity(64),
            manifolds_cache: std::sync::OnceLock::new(),
            touching_contacts: 0,
            mesh_scratch: MeshScratch::default(),
            recycled_contacts: 0,
            fast: Vec::with_capacity(64),
            joints: Vec::with_capacity(64),
            joint_ids: IdPool::new(),
            island_manager: IslandManager3d::new(),
            sets: {
                let mut sets = Vec::with_capacity(32);
                sets.push(crate::island::SolverSet3d::with_capacity(
                    crate::island::STATIC_SET,
                    64,
                ));
                sets.push(crate::island::SolverSet3d::with_capacity(
                    crate::island::DISABLED_SET,
                    64,
                ));
                sets.push(crate::island::SolverSet3d::with_capacity(
                    crate::island::AWAKE_SET,
                    64,
                ));
                for id in crate::island::FIRST_SLEEPING_SET..19 {
                    sets.push(crate::island::SolverSet3d::with_capacity(id, 16));
                }
                sets
            },
            free_sets: (crate::island::FIRST_SLEEPING_SET..19).rev().collect(),
            islands: (0..32)
                .map(|_| crate::island::Island3d {
                    set: crate::contact::NULL,
                    local: crate::contact::NULL,
                    bodies: Vec::with_capacity(16),
                    contacts: Vec::with_capacity(16),
                    joints: Vec::with_capacity(16),
                    constraint_remove_count: 0,
                })
                .collect(),
            free_islands: (0..32).rev().collect(),
            island_count: 0,
            island_to_split: crate::contact::NULL,
            split_candidate: (crate::contact::NULL, 0.0),
            split_scratch: crate::island::SplitScratch3d::default(),
            awake_islands: Vec::with_capacity(64),
            body_links: Vec::with_capacity(64),
            joint_links: Vec::with_capacity(64),
            awake_contact_slots: Vec::with_capacity(64),
            move_buffer: Vec::with_capacity(64),
            touched_bodies: Vec::with_capacity(64),
            contact_changes: Vec::with_capacity(64),
            collide_outcomes: Vec::with_capacity(64),
            executor: None,
            jointed_pairs_dirty: true,
            sensors_dirty: true,
            pairs_scratch: Vec::with_capacity(128),
            jointed_pairs: Vec::new(),
            solver_scratch: SolverScratch3d::default(),
            last_inv_h: 0.0,
            sensor_count: 0,
            hit_event_count: 0,
            sensor_events: SensorEvents3d::default(),
            sensor_overlaps: Vec::with_capacity(64),
            sensor_scratch: Vec::new(),
            sensor_candidates: Vec::new(),
            sensor_hits: Vec::new(),
            continuous_candidates: Vec::new(),
            bullets: Vec::new(),
            compound_scratch: Vec::with_capacity(32),
            contact_events: ContactEvents3d::with_capacity(64),
            body_events: BodyEvents3d::with_capacity(64),
            joint_events: JointEvents3d::with_capacity(64),
            was_awake: Vec::with_capacity(64),
            friction_callback: crate::material::default_friction_callback,
            restitution_callback: crate::material::default_restitution_callback,
            profile: Profile3d::default(),
        }
    }

    /// Sets the task executor that runs narrowphase and solver stages across worker threads, or
    /// `None` to step on the calling thread (`b3WorldDef::workerCount` / `enqueueTask`).
    pub fn set_executor(
        &mut self,
        executor: Option<std::sync::Arc<dyn korppi_core::TaskExecutor>>,
    ) {
        self.executor = executor.map(Executor3d);
    }

    /// Returns the world's task executor, if one is attached.
    #[must_use]
    pub fn executor(&self) -> Option<&dyn korppi_core::TaskExecutor> {
        self.executor.as_ref().map(|e| e.0.as_ref())
    }

    /// Sets the global gravity acceleration vector.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
    }

    /// Returns the current gravity vector.
    #[must_use]
    pub const fn gravity(&self) -> Vec3 {
        self.gravity
    }

    /// Returns a reference to the active 3D soft-step solver configuration.
    #[must_use]
    pub const fn solver_config(&self) -> &SolverConfig3d {
        &self.solver_config
    }

    /// Returns a mutable reference to the 3D soft-step solver configuration.
    pub fn solver_config_mut(&mut self) -> &mut SolverConfig3d {
        &mut self.solver_config
    }

    /// Creates a new physics world with custom gravity.
    #[must_use]
    pub fn with_gravity(gravity: Vec3) -> Self {
        let mut world = Self::new();
        world.set_gravity(gravity);
        world
    }

    /// Sets the custom friction combining callback (Box3D `b3World_SetFrictionCallback`).
    pub fn set_friction_callback(&mut self, cb: crate::material::FrictionCallback3d) {
        self.friction_callback = cb;
    }

    /// Sets the custom restitution combining callback (Box3D `b3World_SetRestitutionCallback`).
    pub fn set_restitution_callback(&mut self, cb: crate::material::RestitutionCallback3d) {
        self.restitution_callback = cb;
    }

    /// Sets whether resting bodies are transitioned to sleep (Box3D `b3World_EnableSleeping`).
    pub fn enable_sleep(&mut self, enable: bool) {
        self.solver_config.enable_sleep = enable;
        if !enable {
            for set_id in crate::island::FIRST_SLEEPING_SET..self.sets.len() as u32 {
                if !self.sets[set_id as usize].bodies.is_empty() {
                    self.wake_set(set_id);
                }
            }
        }
    }

    /// Returns whether sleeping is enabled.
    #[must_use]
    pub const fn is_sleep_enabled(&self) -> bool {
        self.solver_config.enable_sleep
    }

    /// Sets whether impulse warm starting across time steps is enabled (Box3D `b3World_EnableWarmStarting`).
    pub fn enable_warm_starting(&mut self, enable: bool) {
        self.solver_config.enable_warm_starting = enable;
    }

    /// Returns whether impulse warm starting is enabled.
    #[must_use]
    pub const fn is_warm_starting_enabled(&self) -> bool {
        self.solver_config.enable_warm_starting
    }

    /// Returns the profiling execution times in milliseconds for the last step (Box3D `b3World_GetProfile`).
    #[must_use]
    pub const fn profile(&self) -> Profile3d {
        self.profile
    }

    /// Returns the base surface material of the specified collider (Box3D `b3Shape_GetSurfaceMaterial`).
    #[must_use]
    pub fn surface_material(
        &self,
        collider_id: ColliderId3d,
    ) -> Option<crate::material::SurfaceMaterial3d> {
        if !self.collider_ids.contains(collider_id) {
            return None;
        }
        let slot = self.colliders.get(collider_id.index() as usize)?.as_ref()?;
        Some(slot.collider.surface_material())
    }

    /// Sets the base surface material of the specified collider (Box3D `b3Shape_SetSurfaceMaterial`).
    pub fn set_surface_material(
        &mut self,
        collider_id: ColliderId3d,
        mat: crate::material::SurfaceMaterial3d,
    ) {
        if !self.collider_ids.contains(collider_id) {
            return;
        }
        if let Some(Some(slot)) = self.colliders.get_mut(collider_id.index() as usize) {
            slot.collider.set_surface_material(mat);
        }
    }

    /// Returns the number of mesh materials on the collider (Box3D `b3Shape_GetMeshMaterialCount`).
    #[must_use]
    pub fn mesh_material_count(&self, collider_id: ColliderId3d) -> usize {
        if !self.collider_ids.contains(collider_id) {
            return 0;
        }
        self.colliders
            .get(collider_id.index() as usize)
            .and_then(Option::as_ref)
            .map(|s| s.collider.mesh_material_count())
            .unwrap_or(0)
    }

    /// Returns the mesh surface material at `index` (Box3D `b3Shape_GetMeshSurfaceMaterial`).
    #[must_use]
    pub fn mesh_material(
        &self,
        collider_id: ColliderId3d,
        index: usize,
    ) -> Option<crate::material::SurfaceMaterial3d> {
        if !self.collider_ids.contains(collider_id) {
            return None;
        }
        let slot = self.colliders.get(collider_id.index() as usize)?.as_ref()?;
        slot.collider.mesh_material(index)
    }

    /// Sets the mesh surface material at `index` (Box3D `b3Shape_SetMeshMaterial`).
    pub fn set_mesh_material(
        &mut self,
        collider_id: ColliderId3d,
        index: usize,
        mat: crate::material::SurfaceMaterial3d,
    ) {
        if !self.collider_ids.contains(collider_id) {
            return;
        }
        if let Some(Some(slot)) = self.colliders.get_mut(collider_id.index() as usize) {
            slot.collider.set_mesh_material(index, mat);
        }
    }

    /// Sets the 3D soft-step solver configuration.
    pub fn set_solver_config(&mut self, config: SolverConfig3d) {
        self.solver_config = config;
    }

    /// Returns the current monotonic simulation step counter.
    #[must_use]
    pub const fn step_index(&self) -> u64 {
        self.step_index
    }

    /// Returns a reference to the island manager.
    #[must_use]
    pub const fn island_manager(&self) -> &IslandManager3d {
        &self.island_manager
    }

    /// Returns a mutable reference to the island manager.
    pub fn island_manager_mut(&mut self) -> &mut IslandManager3d {
        &mut self.island_manager
    }

    /// Returns a reference to the active constraint graph and color assignments.
    #[must_use]
    pub const fn constraint_graph(&self) -> &crate::constraint_graph::ConstraintGraph3d {
        &self.constraint_graph
    }

    /// Number of awake dynamic/kinematic bodies in `AWAKE_SET`.
    #[must_use]
    pub fn awake_body_count(&self) -> usize {
        self.sets
            .get(crate::island::AWAKE_SET as usize)
            .map_or(0, |s| s.bodies.len())
    }

    /// Returns a reference to the solver set at `set_id` (`0` = static, `1` = disabled, `2` = awake, `>= 3` = sleeping).
    #[must_use]
    pub fn solver_set(&self, set_id: u32) -> Option<&crate::island::SolverSet3d> {
        self.sets.get(set_id as usize)
    }

    /// Returns the persistent island ID for `body`, or `None` if static/disabled/stale.
    #[must_use]
    pub fn body_island(&self, body: impl Into<BodyId3d>) -> Option<u32> {
        let id: BodyId3d = body.into();
        if !self.body_ids.contains(id) {
            return None;
        }
        let island = self.body_links.get(id.index() as usize)?.island;
        (island != crate::contact::NULL).then_some(island)
    }

    /// Returns the solver set ID for `body` (0 = static, 1 = disabled, 2 = awake, >= 3 = sleeping island).
    #[must_use]
    pub fn body_solver_set(&self, body: impl Into<BodyId3d>) -> u32 {
        let id: BodyId3d = body.into();
        if !self.body_ids.contains(id) {
            return crate::island::DISABLED_SET;
        }
        let idx = id.index() as usize;
        if let Some(links) = self.body_links.get(idx) {
            if links.set != crate::contact::NULL {
                return links.set;
            }
        }
        crate::island::DISABLED_SET
    }

    /// Returns candidate collision pairs generated by the broadphase in the last step.
    #[must_use]
    pub fn broadphase_candidate_pairs(&self) -> &[(u32, u32)] {
        &self.pairs_scratch
    }

    /// Returns the number of broadphase proxies currently queued in the move buffer.
    #[must_use]
    pub fn broadphase_move_count(&self) -> usize {
        self.move_buffer.len()
    }

    /// Returns the tree height of the broadphase tree corresponding to `body_type`.
    #[must_use]
    pub fn broadphase_tree_height(&self, body_type: BodyType) -> usize {
        self.trees[body_type.tree_index()].height()
    }

    /// Wakes up the specified body and its connected island.
    pub fn wake_body(&mut self, id: impl Into<BodyId3d>) {
        let id: BodyId3d = id.into();
        if !self.body_ids.contains(id) {
            return;
        }
        self.wake_body_slot(id.index());
    }

    /// Transitions the specified body and its island to sleep.
    pub fn sleep_body(&mut self, id: impl Into<BodyId3d>) {
        let id: BodyId3d = id.into();
        if !self.body_ids.contains(id) {
            return;
        }
        let slot = id.index();
        if let Some(&links) = self.body_links.get(slot as usize) {
            if links.set == crate::island::AWAKE_SET && links.island != crate::contact::NULL {
                if self.islands[links.island as usize].constraint_remove_count > 0 {
                    self.split_island(links.island);
                }
                let island = self.body_links[slot as usize].island;
                self.try_sleep_island(island);
            }
        }
        if let Some(body) = self.bodies.get_mut(slot as usize) {
            body.put_to_sleep();
        }
    }

    /// Creates a new rigid body in the simulation world, returning its generational handle.
    pub fn create_body(&mut self, mut body: RigidBody) -> BodyId3d {
        body.recompute_mass_properties();
        let handle = self.body_ids.alloc();
        let slot = handle.index();
        let index = slot as usize;
        let asleep = body.is_sleeping && body.enable_sleep && self.solver_config.enable_sleep;
        body.is_sleeping = false;
        let set = match body.body_type {
            BodyType::Static => crate::island::STATIC_SET,
            BodyType::Kinematic | BodyType::Dynamic => {
                if asleep {
                    self.alloc_set()
                } else {
                    crate::island::AWAKE_SET
                }
            }
        };
        let links = crate::island::BodyLinks3d::default();
        if index < self.bodies.len() {
            self.bodies[index] = body;
            self.body_colliders[index].clear();
            self.body_links[index] = links;
        } else {
            self.bodies.push(body);
            self.body_colliders.push(Vec::new());
            self.body_links.push(links);
        }
        self.add_body_to_set(slot, set);
        if set >= crate::island::AWAKE_SET {
            self.create_island_for_body(set, slot);
        }
        if set >= crate::island::FIRST_SLEEPING_SET {
            let b = &mut self.bodies[index];
            b.is_sleeping = true;
            b.linear_velocity = Vec3::ZERO;
            b.angular_velocity = Vec3::ZERO;
        }
        self.island_manager
            .register_body(slot, self.bodies[index].is_sleeping);
        handle
    }

    /// Attaches a collider to body `body_id` and returns the collider's ID. A body takes any
    /// number of colliders; a dynamic body's mass, centre of mass and inertia become its
    /// colliders' (Box3D `b3CreateShape` and `b3UpdateBodyMassData`).
    ///
    /// # Panics
    /// If `body_id` does not exist.
    pub fn create_collider(
        &mut self,
        body_id: impl Into<BodyId3d>,
        collider: Collider,
    ) -> ColliderId3d {
        let body: BodyId3d = body_id.into();
        assert!(
            self.body_ids.contains(body),
            "body {} does not exist",
            body.index()
        );
        let body_idx = body.index() as usize;
        if matches!(collider.shape, Shape::Compound(_)) {
            assert!(
                self.bodies[body_idx].body_type == BodyType::Static,
                "compound shapes must be attached to static bodies (Box3D invariant)"
            );
            assert!(
                !collider.is_sensor,
                "compound shapes cannot be sensors (Box3D invariant)"
            );
        }
        let handle = self.collider_ids.alloc();
        let col_idx = handle.index() as usize;
        let b = &self.bodies[body_idx];
        let (position, rotation) = collider.world_pose(b.position, b.rotation);
        let tree_idx = b.body_type.tree_index();
        let aabb = collider
            .shape
            .compute_aabb(position, rotation)
            .expand(SPECULATIVE_DISTANCE);
        let margin = if b.body_type == BodyType::Static {
            SPECULATIVE_DISTANCE
        } else {
            collider.shape.aabb_margin()
        };
        let fat_aabb = aabb.expand(margin);
        let proxy_id = self.trees[tree_idx].create_fat_proxy(fat_aabb, handle.index());
        let proxy = (proxy_id << 2) | (tree_idx as u32);
        if collider.is_sensor {
            self.sensor_count += 1;
            self.sensors_dirty = true;
        }
        if collider.enable_hit_events {
            self.hit_event_count += 1;
        }
        let slot = ColliderSlot {
            body,
            collider,
            proxy,
            aabb,
            fat_aabb,
            moved: true,
        };
        self.move_buffer.push(handle.index());
        if col_idx < self.colliders.len() {
            self.colliders[col_idx] = Some(slot);
            self.sensor_overlaps[col_idx].clear();
        } else {
            self.colliders.push(Some(slot));
            self.sensor_overlaps.push(Vec::new());
        }
        self.body_colliders[body_idx].push(handle);
        self.update_body_mass(body_idx);
        handle
    }

    /// Sums body `index`'s colliders into its mass data (Box3D `b3UpdateBodyMassData`): each
    /// collider's mass and centre, then the inertia about the centre of mass by the parallel
    /// axis theorem, and the extents for continuous collision. A static or kinematic body keeps
    /// its centre at its origin. The centre of mass moving changes its velocity by the spin.
    fn update_body_mass(&mut self, index: usize) {
        let body = &mut self.bodies[index];
        let colliders: Vec<&Collider> = self.body_colliders[index]
            .iter()
            .filter_map(|&c| {
                if self.collider_ids.contains(c) {
                    self.colliders[c.index() as usize]
                        .as_ref()
                        .map(|s| &s.collider)
                } else {
                    None
                }
            })
            .collect();
        let old_center = body.center_of_mass();
        if body.body_type != BodyType::Dynamic {
            body.local_center = Vec3::ZERO;
        } else {
            let masses = || {
                colliders
                    .iter()
                    .copied()
                    .map(Collider::mass_data)
                    .filter(|m| m.mass > 0.0)
            };
            let mass: f32 = masses().map(|m| m.mass).sum();
            let center = if mass > 0.0 {
                masses().fold(Vec3::ZERO, |c, m| c + m.center * m.mass) * (1.0 / mass)
            } else {
                Vec3::ZERO
            };
            body.mass = mass;
            body.local_center = center;
            body.local_inertia = masses().fold(Mat3::ZERO, |inertia, m| {
                inertia + m.inertia + crate::math::steiner(m.mass, center - m.center)
            });
            body.recompute_mass_properties();
        }
        let origin = body.local_center;
        let (min_extent, max_extent) =
            colliders
                .iter()
                .fold((f32::MAX, Vec3::ZERO), |(lo, hi), c| {
                    let (min, max) = c.extents_about(origin);
                    (lo.min(min), hi.max(max))
                });
        body.min_extent = if min_extent == f32::MAX {
            0.0
        } else {
            min_extent
        };
        body.max_extent = max_extent;
        let new_center = body.center_of_mass();
        body.linear_velocity += body.angular_velocity.cross(new_center - old_center);
    }

    /// Collider `id`.
    #[must_use]
    pub fn collider(&self, id: impl Into<ColliderId3d>) -> Option<&Collider> {
        let id: ColliderId3d = id.into();
        if !self.collider_ids.contains(id) {
            return None;
        }
        self.colliders
            .get(id.index() as usize)?
            .as_ref()
            .map(|slot| &slot.collider)
    }

    /// The body collider `id` is on.
    #[must_use]
    pub fn collider_body(&self, id: impl Into<ColliderId3d>) -> Option<BodyId3d> {
        let id: ColliderId3d = id.into();
        if !self.collider_ids.contains(id) {
            return None;
        }
        self.colliders
            .get(id.index() as usize)?
            .as_ref()
            .map(|slot| slot.body)
    }

    /// The first collider of body `body`.
    #[must_use]
    pub fn collider_of(&self, body: impl Into<BodyId3d>) -> Option<ColliderId3d> {
        let body: BodyId3d = body.into();
        if !self.body_ids.contains(body) {
            return None;
        }
        self.body_colliders
            .get(body.index() as usize)
            .and_then(|list| list.first().copied())
    }

    /// The colliders of body `body`, in the order they were created.
    #[must_use]
    pub fn colliders_of(&self, body: impl Into<BodyId3d>) -> &[ColliderId3d] {
        let body: BodyId3d = body.into();
        if !self.body_ids.contains(body) {
            return &[];
        }
        self.body_colliders
            .get(body.index() as usize)
            .map_or(&[], Vec::as_slice)
    }

    /// Destroys contact slot `slot` (Box3D `b3DestroyContact`): unlinks it from its island,
    /// removes it from its solver set or constraint graph color, removes it from the contact
    /// pool, and wakes both bodies if it was touching outside of `step`.
    pub(crate) fn destroy_contact_slot(&mut self, slot: u32, in_step: bool) {
        let Some(contact) = self.contacts.get(slot) else {
            return;
        };
        let (touching, body_a, body_b, col_a, col_b, island) = (
            contact.is_touching,
            contact.body_a,
            contact.body_b,
            contact.collider_a,
            contact.collider_b,
            contact.island,
        );
        if touching {
            self.manifolds_cache.take();
        }
        if touching && in_step {
            let enable_events = self
                .colliders
                .get(col_a as usize)
                .and_then(Option::as_ref)
                .is_some_and(|s| s.collider.enable_contact_events)
                || self
                    .colliders
                    .get(col_b as usize)
                    .and_then(Option::as_ref)
                    .is_some_and(|s| s.collider.enable_contact_events);
            if enable_events {
                if let (Some(ca), Some(cb), Some(ba), Some(bb)) = (
                    self.collider_ids.handle_at(col_a),
                    self.collider_ids.handle_at(col_b),
                    self.body_ids.handle_at(body_a),
                    self.body_ids.handle_at(body_b),
                ) {
                    self.contact_events.end.push(ContactEndEvent3d {
                        collider_a: ca,
                        collider_b: cb,
                        body_a: ba,
                        body_b: bb,
                    });
                }
            }
        }
        if island != crate::contact::NULL {
            self.unlink_contact(slot);
        }
        self.remove_contact_from_set(slot);
        self.contacts.remove(slot);
        if touching && !in_step {
            self.wake_body_slot(body_a);
            self.wake_body_slot(body_b);
        }
    }

    /// Creates a joint in the simulation world, returning its ID. Unless the joint allows
    /// its bodies to collide, the contact between them goes at once (Box3D `b3CreateJoint`).
    pub fn create_joint(&mut self, joint: Joint) -> JointId3d {
        let (body_a, body_b) = (joint.body_a(), joint.body_b());
        if !joint.collide_connected() {
            self.jointed_pairs_dirty = true;
            // Every contact between the bodies' colliders.
            let key = ordered(body_a, body_b);
            let mut slot = self.contacts.body_head(body_a);
            while slot != crate::contact::NULL {
                let c = self.contacts.at(slot);
                let edge = usize::from(c.body_b == body_a);
                let next = c.edges[edge].next;
                if ordered(c.body_a, c.body_b) == key {
                    self.destroy_contact_slot(slot, false);
                }
                slot = next;
            }
            self.manifolds_cache.take();
        }
        let handle = self.joint_ids.alloc();
        let slot = handle.index();
        let idx = slot as usize;
        if idx < self.joints.len() {
            self.joints[idx] = Some(joint);
            self.joint_links[idx] = crate::island::JointLinks3d::default();
        } else {
            self.joints.push(Some(joint));
            self.joint_links
                .push(crate::island::JointLinks3d::default());
        }
        let set_a = self.body_links[body_a as usize].set;
        let set_b = self.body_links[body_b as usize].set;
        let set = if set_a == crate::island::DISABLED_SET || set_b == crate::island::DISABLED_SET {
            crate::island::DISABLED_SET
        } else if set_a == crate::island::STATIC_SET && set_b == crate::island::STATIC_SET {
            crate::island::STATIC_SET
        } else {
            crate::island::AWAKE_SET
        };
        if set == crate::island::AWAKE_SET {
            self.wake_body_slot(body_a);
            self.wake_body_slot(body_b);
        }
        self.add_joint_to_set(slot, set);
        if set != crate::island::DISABLED_SET && set != crate::island::STATIC_SET {
            self.link_joint(slot);
        }
        handle
    }

    /// Destroys a rigid body, its colliders and attached joints (Box3D `b3DestroyBody`).
    /// Touching bodies and joint partners wake up. Returns `false` if `id` is stale.
    pub fn destroy_body(&mut self, id: BodyId3d) -> bool {
        if !self.body_ids.contains(id) {
            return false;
        }
        let body_slot = id.index();
        let body_idx = body_slot as usize;

        // 1. Destroy attached joints
        let attached_joints: Vec<JointId3d> = self
            .joint_ids
            .iter()
            .filter(|&jid| {
                if let Some(Some(joint)) = self.joints.get(jid.index() as usize) {
                    joint.body_a() == body_slot || joint.body_b() == body_slot
                } else {
                    false
                }
            })
            .collect();
        for jid in attached_joints {
            self.destroy_joint(jid);
        }

        // 2. Destroy attached contacts
        let mut s = self.contacts.body_head(body_slot);
        while s != crate::contact::NULL {
            let c = self.contacts.at(s);
            let edge = usize::from(c.body_b == body_slot);
            let next = c.edges[edge].next;
            self.destroy_contact_slot(s, false);
            s = next;
        }

        // 3. Destroy attached colliders without updating body mass
        let colliders: Vec<ColliderId3d> = self
            .body_colliders
            .get(body_idx)
            .map(|list| list.as_slice())
            .unwrap_or(&[])
            .to_vec();
        for cid in colliders {
            self.destroy_collider_internal(cid, false);
        }
        self.body_colliders[body_idx].clear();

        // 4. Remove from island and solver set, make slot inert static body
        self.remove_body_from_island(body_slot);
        self.remove_body_from_set(body_slot);
        self.bodies[body_idx] = RigidBody::new(BodyType::Static, Vec3::ZERO);

        // 5. Free body ID
        self.body_ids.free(id);
        true
    }

    /// Detaches and destroys a collider, waking touching bodies and updating the owning body's mass.
    /// Returns `false` if `id` is stale.
    pub fn destroy_collider(&mut self, id: ColliderId3d) -> bool {
        self.destroy_collider_internal(id, true)
    }

    fn destroy_collider_internal(&mut self, id: ColliderId3d, update_mass: bool) -> bool {
        if !self.collider_ids.contains(id) {
            return false;
        }
        let slot_idx = id.index() as usize;
        let slot = self.colliders[slot_idx]
            .take()
            .expect("collider slot exists");
        let body_id = slot.body;
        let body_idx = body_id.index() as usize;

        // Remove broadphase proxy
        let tree_idx = (slot.proxy & 3) as usize;
        let proxy_id = slot.proxy >> 2;
        self.trees[tree_idx].destroy_proxy(proxy_id);
        if slot.collider.is_sensor {
            self.sensor_count = self.sensor_count.saturating_sub(1);
            self.sensors_dirty = true;
        }
        if slot.collider.enable_hit_events {
            self.hit_event_count = self.hit_event_count.saturating_sub(1);
        }

        // Destroy contacts on this collider and wake touching bodies
        let cid = id.index();
        if slot.moved {
            self.move_buffer.retain(|&c| c != cid);
        }
        let body_slot = body_id.index();
        let mut s = self.contacts.body_head(body_slot);
        while s != crate::contact::NULL {
            let contact = self.contacts.at(s);
            let edge = usize::from(contact.body_b == body_slot);
            let next = contact.edges[edge].next;
            if contact.collider_a == cid || contact.collider_b == cid {
                self.destroy_contact_slot(s, false);
            }
            s = next;
        }

        // Clean up sensor overlaps
        if slot_idx < self.sensor_overlaps.len() {
            self.sensor_overlaps[slot_idx].clear();
        }
        for overlaps in &mut self.sensor_overlaps {
            overlaps.retain(|&c| c != id);
        }

        // Remove from body colliders list
        if body_idx < self.body_colliders.len() {
            self.body_colliders[body_idx].retain(|&c| c != id);
        }

        // Free collider handle
        self.collider_ids.free(id);

        if update_mass && self.body_ids.contains(body_id) {
            self.wake_body(body_id);
            self.update_body_mass(body_idx);
        }

        true
    }

    /// Destroys a joint, waking both connected bodies (Box3D `b3DestroyJoint`).
    /// Returns the removed joint, or `None` if `id` is stale.
    pub fn destroy_joint(&mut self, id: JointId3d) -> Option<Joint> {
        if !self.joint_ids.contains(id) {
            return None;
        }
        let slot = id.index();
        let slot_idx = slot as usize;
        let (body_a, body_b, collide_connected) = {
            let joint = self.joints.get(slot_idx)?.as_ref()?;
            (joint.body_a(), joint.body_b(), joint.collide_connected())
        };
        if !collide_connected {
            self.jointed_pairs_dirty = true;
            self.queue_body_colliders_for_move(body_a);
            self.queue_body_colliders_for_move(body_b);
        }
        if self.joint_links[slot_idx].island != crate::contact::NULL {
            self.unlink_joint(slot);
        }
        self.remove_joint_from_set(slot);
        let joint = self.joints.get_mut(slot_idx)?.take()?;
        self.joint_ids.free(id);
        self.wake_body_slot(body_a);
        self.wake_body_slot(body_b);
        Some(joint)
    }

    fn queue_body_colliders_for_move(&mut self, body_slot: u32) {
        let Some(colliders) = self.body_colliders.get(body_slot as usize) else {
            return;
        };
        for &cid_handle in colliders {
            let cid = cid_handle.index();
            if let Some(Some(slot)) = self.colliders.get_mut(cid as usize) {
                if !slot.moved {
                    slot.moved = true;
                    self.move_buffer.push(cid);
                }
            }
        }
    }

    /// Returns a reference to the joint at `id`.
    #[must_use]
    pub fn joint(&self, id: impl Into<JointId3d>) -> Option<&Joint> {
        let id: JointId3d = id.into();
        if !self.joint_ids.contains(id) {
            return None;
        }
        self.joints.get(id.index() as usize)?.as_ref()
    }

    /// Returns a mutable reference to the joint at `id`.
    pub fn joint_mut(&mut self, id: impl Into<JointId3d>) -> Option<&mut Joint> {
        let id: JointId3d = id.into();
        if !self.joint_ids.contains(id) {
            return None;
        }
        let idx = id.index() as usize;
        if let Some(Some(j)) = self.joints.get(idx) {
            let (ba, bb) = (j.body_a(), j.body_b());
            self.wake_body_slot(ba);
            self.wake_body_slot(bb);
            self.queue_body_colliders_for_move(ba);
            self.queue_body_colliders_for_move(bb);
            self.jointed_pairs_dirty = true;
        }
        self.joints.get_mut(idx)?.as_mut()
    }

    /// The force joint `id` applied to its body B during the last substep, in newtons (Box3D
    /// `b3Joint_GetConstraintForce`).
    #[must_use]
    pub fn joint_constraint_force(&self, id: impl Into<JointId3d>) -> Option<Vec3> {
        self.joint(id).map(|j| j.constraint_force(self.last_inv_h))
    }

    /// The torque joint `id` applied to its body B during the last substep, in N·m (Box3D
    /// `b3Joint_GetConstraintTorque`).
    #[must_use]
    pub fn joint_constraint_torque(&self, id: impl Into<JointId3d>) -> Option<Vec3> {
        self.joint(id).map(|j| j.constraint_torque(self.last_inv_h))
    }

    /// How far joint `id`'s position constraint is from holding, in metres (Box3D
    /// `b3Joint_GetLinearSeparation`).
    #[must_use]
    pub fn joint_linear_separation(&self, id: impl Into<JointId3d>) -> Option<f32> {
        let joint = self.joint(id)?;
        Some(joint.linear_separation(self.body(joint.body_a())?, self.body(joint.body_b())?))
    }

    /// How far joint `id`'s rotation constraint is from holding, in radians (Box3D
    /// `b3Joint_GetAngularSeparation`).
    #[must_use]
    pub fn joint_angular_separation(&self, id: impl Into<JointId3d>) -> Option<f32> {
        let joint = self.joint(id)?;
        Some(joint.angular_separation(self.body(joint.body_a())?, self.body(joint.body_b())?))
    }

    /// Returns an iterator over all active joints in the world.
    #[must_use]
    pub fn joints(&self) -> Joints<'_> {
        Joints::new(&self.joint_ids, &self.joints)
    }

    /// Returns the number of active joints in the world.
    #[must_use]
    pub fn joint_count(&self) -> usize {
        self.joint_ids.len()
    }

    /// Returns the number of rigid bodies in the world.
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.body_ids.len()
    }

    /// Returns the number of active colliders in the world.
    #[must_use]
    pub fn collider_count(&self) -> usize {
        self.collider_ids.len()
    }

    /// Returns the live bodies and their handles, in slot order.
    pub fn bodies(&self) -> impl Iterator<Item = (BodyId3d, &RigidBody)> + '_ {
        self.body_ids
            .iter()
            .map(|id| (id, &self.bodies[id.index() as usize]))
    }

    /// Returns the live colliders and their handles, in slot order.
    pub fn colliders(&self) -> impl Iterator<Item = (ColliderId3d, &Collider)> + '_ {
        self.collider_ids.iter().filter_map(|id| {
            self.colliders[id.index() as usize]
                .as_ref()
                .map(|slot| (id, &slot.collider))
        })
    }

    /// Returns a reference to the rigid body at `id`.
    #[must_use]
    pub fn body(&self, id: impl Into<BodyId3d>) -> Option<&RigidBody> {
        let id: BodyId3d = id.into();
        if !self.body_ids.contains(id) {
            return None;
        }
        self.bodies.get(id.index() as usize)
    }

    /// Returns a mutable reference to the rigid body at `id`, waking its island first as Box3D
    /// setters do.
    pub fn body_mut(&mut self, id: impl Into<BodyId3d>) -> Option<&mut RigidBody> {
        let id: BodyId3d = id.into();
        if !self.body_ids.contains(id) {
            return None;
        }
        let slot = id.index();
        self.wake_body_slot(slot);
        if let Some(links) = self.body_links.get_mut(slot as usize) {
            if !links.touched {
                links.touched = true;
                self.touched_bodies.push(slot);
            }
        }
        self.bodies.get_mut(slot as usize)
    }

    /// Sets the position of body `id`, waking its island and updating its broadphase proxies if it
    /// left its fat AABB (Box3D `b3Body_SetTransform`).
    pub fn set_body_position(&mut self, id: impl Into<BodyId3d>, position: Vec3) {
        let id: BodyId3d = id.into();
        if !self.body_ids.contains(id) {
            return;
        }
        let slot = id.index();
        self.wake_body_slot(slot);
        self.bodies[slot as usize].position = position;
        self.refresh_body_proxies(slot);
    }

    /// Sets the rotation of body `id`, waking its island and updating its broadphase proxies if it
    /// left its fat AABB (Box3D `b3Body_SetTransform`).
    pub fn set_body_rotation(&mut self, id: impl Into<BodyId3d>, rotation: Quat) {
        let id: BodyId3d = id.into();
        if !self.body_ids.contains(id) {
            return;
        }
        let slot = id.index();
        self.wake_body_slot(slot);
        self.bodies[slot as usize].rotation = rotation;
        self.refresh_body_proxies(slot);
    }

    /// Returns the total number of contact points across all active manifolds.
    #[must_use]
    pub fn contact_count(&self) -> usize {
        self.contacts
            .iter()
            .filter(|c| c.is_touching)
            .flat_map(|c| &c.manifolds)
            .map(|m| m.points.len())
            .sum()
    }

    /// Returns a slice of active contact manifolds.
    #[must_use]
    pub fn manifolds(&self) -> &[ContactManifold] {
        self.manifolds_cache
            .get_or_init(|| {
                self.contacts
                    .iter()
                    .filter(|c| c.is_touching)
                    .flat_map(|c| c.manifolds.iter().copied())
                    .collect()
            })
            .as_slice()
    }

    /// The sensor events of the last step (Box3D `b3World_GetSensorEvents`).
    #[must_use]
    pub const fn sensor_events(&self) -> &SensorEvents3d {
        &self.sensor_events
    }

    /// The colliders that overlapped sensor collider `sensor` at the end of the last step,
    /// sorted (Box3D `b3Shape_GetSensorData`). Empty if it is not a sensor or does not exist.
    #[must_use]
    pub fn sensor_overlaps(&self, sensor: impl Into<ColliderId3d>) -> &[ColliderId3d] {
        let sensor: ColliderId3d = sensor.into();
        self.sensor_overlaps
            .get(sensor.index() as usize)
            .map_or(&[], Vec::as_slice)
    }

    /// The contact events of the last step (Box3D `b3World_GetContactEvents`).
    #[must_use]
    pub const fn contact_events(&self) -> &ContactEvents3d {
        &self.contact_events
    }

    /// The body movement and sleep events of the last step (Box3D `b3World_GetBodyEvents`).
    #[must_use]
    pub const fn body_events(&self) -> &BodyEvents3d {
        &self.body_events
    }

    /// The joint reaction events of the last step (Box3D `b3World_GetJointEvents`).
    #[must_use]
    pub const fn joint_events(&self) -> &JointEvents3d {
        &self.joint_events
    }

    /// Touching contacts touching rigid body `body_id` (Box3D `b3Body_GetContactData`).
    pub fn body_contacts(
        &self,
        body_id: impl Into<BodyId3d>,
    ) -> impl Iterator<Item = &Contact3d> + '_ {
        let bid = body_id.into().index();
        self.contacts
            .iter()
            .filter(move |c| (c.body_a == bid || c.body_b == bid) && c.is_touching)
    }

    /// Touching contacts touching collider `collider_id` (Box3D `b3Shape_GetContactData`).
    pub fn collider_contacts(
        &self,
        collider_id: impl Into<ColliderId3d>,
    ) -> impl Iterator<Item = &Contact3d> + '_ {
        let cid = collider_id.into().index();
        self.contacts
            .iter()
            .filter(move |c| (c.collider_a == cid || c.collider_b == cid) && c.is_touching)
    }

    /// Performs a ray-cast against all bodies in the physics world, returning the closest hit.
    #[must_use]
    pub fn ray_cast(&self, ray: &Ray, max_distance: f32) -> Option<RayHit3d> {
        self.ray_cast_filtered(ray, max_distance, &QueryFilter3d::DEFAULT)
    }

    /// Ray-casts against the colliders `filter` accepts (Box3D `b3QueryFilter`), returning the
    /// closest hit.
    #[must_use]
    pub fn ray_cast_filtered(
        &self,
        ray: &Ray,
        max_distance: f32,
        filter: &QueryFilter3d,
    ) -> Option<RayHit3d> {
        let mut closest_hit: Option<RayHit3d> = None;
        let mut current_max = max_distance;

        for tree in &self.trees {
            tree.ray_cast(ray, current_max, |collider_id, _| {
                let Some(Some(slot)) = self.colliders.get(collider_id as usize) else {
                    return current_max;
                };
                if !slot.collider.filter.should_query(filter) {
                    return current_max;
                }
                let (position, rotation) = slot.pose(&self.bodies[slot.body.index() as usize]);
                let local_ray = Ray::new(
                    rotation.conjugate() * (ray.origin - position),
                    rotation.conjugate() * ray.direction,
                );
                if let Some((dist, local_normal, tri_or_child)) = slot
                    .collider
                    .shape
                    .ray_cast_detailed(&local_ray, current_max)
                {
                    if dist < current_max {
                        current_max = dist;
                        let user_mat_id = if !slot.collider.materials.is_empty() {
                            let mat_idx = match &slot.collider.shape {
                                Shape::TriangleMesh(m) => m.material_index(tri_or_child) as usize,
                                Shape::HeightField(hf) => hf.material_index(tri_or_child) as usize,
                                _ => 0,
                            };
                            slot.collider
                                .materials
                                .get(mat_idx)
                                .map(|m| m.user_material_id)
                                .unwrap_or(slot.collider.user_material_id)
                        } else {
                            slot.collider.user_material_id
                        };
                        let cid = self.collider_ids.handle_at(collider_id).unwrap();
                        closest_hit = Some(RayHit3d {
                            body_id: slot.body,
                            collider_id: cid,
                            point: ray.point_at(dist),
                            normal: (rotation * local_normal).normalize(),
                            fraction: dist,
                            user_material_id: user_mat_id,
                        });
                    }
                }
                current_max
            });
        }

        closest_hit
    }

    /// Queries the physics world for all rigid bodies whose collision shapes intersect `aabb`,
    /// in ID order.
    #[must_use]
    pub fn query_aabb(&self, aabb: &Aabb) -> Vec<BodyId3d> {
        self.query_aabb_filtered(aabb, &QueryFilter3d::DEFAULT)
    }

    /// Queries for the bodies whose colliders `filter` accepts and whose bounds intersect
    /// `aabb`.
    #[must_use]
    pub fn query_aabb_filtered(&self, aabb: &Aabb, filter: &QueryFilter3d) -> Vec<BodyId3d> {
        let mut hits = Vec::new();
        for tree in &self.trees {
            tree.query(aabb, |collider_id| {
                if let Some(Some(slot)) = self.colliders.get(collider_id as usize) {
                    if slot.collider.filter.should_query(filter)
                        && slot
                            .bounds(&self.bodies[slot.body.index() as usize])
                            .intersects(aabb)
                    {
                        hits.push(slot.body);
                    }
                }
                true
            });
        }
        // A body with several colliders is listed once.
        hits.sort_unstable();
        hits.dedup();
        hits
    }

    /// Performs a ray cast returning the closest hit along `translation` (Box3D `b3World_CastRayClosest`).
    #[must_use]
    pub fn cast_ray_closest(
        &self,
        origin: Vec3,
        translation: Vec3,
        filter: &QueryFilter3d,
    ) -> Option<RayHit3d> {
        let mut closest: Option<RayHit3d> = None;
        self.cast_ray(origin, translation, filter, |hit| {
            closest = Some(*hit);
            hit.fraction
        });
        closest
    }

    /// Performs a ray cast against colliders matching `filter`, invoking `callback` on each hit (Box3D `b3World_CastRay`).
    ///
    /// The callback returns a fraction in `[0.0, 1.0]` to clip the ray to (1.0 to continue, 0.0 to stop).
    pub fn cast_ray(
        &self,
        origin: Vec3,
        translation: Vec3,
        filter: &QueryFilter3d,
        mut callback: impl FnMut(&RayHit3d) -> f32,
    ) {
        let length = translation.length();
        if length < 1e-6 {
            return;
        }

        let direction = translation / length;
        let ray = Ray::new(origin, direction);
        let mut max_fraction = 1.0f32;

        for tree in &self.trees {
            tree.ray_cast(&ray, length * max_fraction, |collider_idx, _| {
                let Some(Some(slot)) = self.colliders.get(collider_idx as usize) else {
                    return length * max_fraction;
                };
                if !slot.collider.filter.should_query(filter) {
                    return length * max_fraction;
                }

                let (position, rotation) = slot.pose(&self.bodies[slot.body.index() as usize]);
                let local_ray = Ray::new(
                    rotation.conjugate() * (ray.origin - position),
                    rotation.conjugate() * ray.direction,
                );

                if let Some((dist, local_normal, tri_or_child)) = slot
                    .collider
                    .shape
                    .ray_cast_detailed(&local_ray, length * max_fraction)
                {
                    let fraction = dist / length;
                    if fraction <= max_fraction {
                        let user_mat_id = if !slot.collider.materials.is_empty() {
                            let mat_idx = match &slot.collider.shape {
                                Shape::TriangleMesh(m) => m.material_index(tri_or_child) as usize,
                                Shape::HeightField(hf) => hf.material_index(tri_or_child) as usize,
                                _ => 0,
                            };
                            slot.collider
                                .materials
                                .get(mat_idx)
                                .map(|m| m.user_material_id)
                                .unwrap_or(slot.collider.user_material_id)
                        } else {
                            slot.collider.user_material_id
                        };
                        let cid = self.collider_ids.handle_at(collider_idx).unwrap();
                        let hit = RayHit3d {
                            body_id: slot.body,
                            collider_id: cid,
                            point: origin + translation * fraction,
                            normal: (rotation * local_normal).normalize(),
                            fraction,
                            user_material_id: user_mat_id,
                        };
                        let ret = callback(&hit);
                        if (0.0..=1.0).contains(&ret) {
                            max_fraction = ret;
                        }
                    }
                }

                length * max_fraction
            });
            if max_fraction == 0.0 {
                break;
            }
        }
    }

    /// Casts a convex shape proxy along `translation`, returning the closest hit (Box3D `b3World_CastShapeClosest`).
    #[must_use]
    pub fn cast_shape_closest(
        &self,
        proxy: &ShapeProxy<'_>,
        translation: Vec3,
        filter: &QueryFilter3d,
    ) -> Option<ShapeCastHit3d> {
        let mut closest: Option<ShapeCastHit3d> = None;
        self.cast_shape(proxy, translation, filter, |hit| {
            closest = Some(*hit);
            hit.fraction
        });
        closest
    }

    /// Casts a convex shape proxy along `translation`, invoking `callback` on each hit (Box3D `b3World_CastShape`).
    ///
    /// Broadphase leaves are visited via [`DynamicTree::box_cast`] (Box3D `b3DynamicTree_BoxCast`):
    /// leaf children are tested before descending into sibling internal subtrees, and the callback's
    /// return value clips `max_fraction` in `[0.0, 1.0]` (`0.0` stops immediately, `1.0` continues
    /// unclipped).
    pub fn cast_shape(
        &self,
        proxy: &ShapeProxy<'_>,
        translation: Vec3,
        filter: &QueryFilter3d,
        mut callback: impl FnMut(&ShapeCastHit3d) -> f32,
    ) {
        if proxy.points.is_empty() {
            return;
        }
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let r = Vec3::splat(proxy.radius);
        for &pt in proxy.points {
            min = min.min(pt - r);
            max = max.max(pt + r);
        }
        let origin_aabb = Aabb { min, max };
        let mut max_fraction = 1.0f32;

        for tree in &self.trees {
            tree.box_cast(
                &origin_aabb,
                translation,
                max_fraction,
                |collider_idx, current_max| {
                    let Some(Some(slot)) = self.colliders.get(collider_idx as usize) else {
                        return current_max;
                    };
                    if !slot.collider.filter.should_query(filter) {
                        return current_max;
                    }

                    let (position, rotation) = slot.pose(&self.bodies[slot.body.index() as usize]);
                    let shape_transform = Transform::new(position, rotation);

                    if let Some(cast_out) = shape_cast_shape(
                        &slot.collider.shape,
                        &shape_transform,
                        proxy,
                        translation,
                        current_max,
                    ) {
                        if cast_out.hit
                            && cast_out.fraction <= current_max
                            && cast_out.fraction > 0.0
                        {
                            let cid = self.collider_ids.handle_at(collider_idx).unwrap();
                            let hit_info = ShapeCastHit3d {
                                body_id: slot.body,
                                collider_id: cid,
                                point: cast_out.point,
                                normal: cast_out.normal,
                                fraction: cast_out.fraction,
                                iterations: cast_out.iterations,
                                user_material_id: slot.collider.user_material_id,
                            };
                            let ret = callback(&hit_info);
                            if (0.0..=1.0).contains(&ret) {
                                max_fraction = ret;
                            }
                            return ret;
                        }
                    }

                    current_max
                },
            );
            if max_fraction == 0.0 {
                break;
            }
        }
    }

    /// Queries the physics world for all colliders overlapping `aabb`, invoking `callback` for each (Box3D `b3World_OverlapAABB`).
    pub fn overlap_aabb(
        &self,
        aabb: &Aabb,
        filter: &QueryFilter3d,
        mut callback: impl FnMut(ColliderId3d) -> bool,
    ) {
        let mut stopped = false;
        for tree in &self.trees {
            tree.query(aabb, |collider_idx| {
                if let Some(Some(slot)) = self.colliders.get(collider_idx as usize) {
                    if slot.collider.filter.should_query(filter) {
                        let body = &self.bodies[slot.body.index() as usize];
                        if slot.bounds(body).intersects(aabb) {
                            if let Some(cid) = self.collider_ids.handle_at(collider_idx) {
                                if !callback(cid) {
                                    stopped = true;
                                    return false;
                                }
                            }
                        }
                    }
                }
                true
            });
            if stopped {
                break;
            }
        }
    }

    /// Queries the physics world for all colliders overlapping `proxy` placed at `transform` (Box3D `b3World_OverlapShape`).
    pub fn overlap_shape(
        &self,
        proxy: &ShapeProxy<'_>,
        transform: &Transform,
        filter: &QueryFilter3d,
        mut callback: impl FnMut(ColliderId3d) -> bool,
    ) {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for &pt in proxy.points {
            let world_pt = transform.apply(pt);
            min = min.min(world_pt - Vec3::splat(proxy.radius));
            max = max.max(world_pt + Vec3::splat(proxy.radius));
        }
        let proxy_aabb = Aabb { min, max };

        let mut stopped = false;
        for tree in &self.trees {
            tree.query(&proxy_aabb, |collider_idx| {
                if let Some(Some(slot)) = self.colliders.get(collider_idx as usize) {
                    if slot.collider.filter.should_query(filter) {
                        let body = &self.bodies[slot.body.index() as usize];
                        let (pos, rot) = slot.pose(body);
                        let shape_transform = Transform::new(pos, rot);
                        if shape_overlaps_proxy(
                            &slot.collider.shape,
                            &shape_transform,
                            proxy,
                            transform,
                        ) {
                            if let Some(cid) = self.collider_ids.handle_at(collider_idx) {
                                if !callback(cid) {
                                    stopped = true;
                                    return false;
                                }
                            }
                        }
                    }
                }
                true
            });
            if stopped {
                break;
            }
        }
    }

    /// Sweeps a capsule mover along `translation`, returning the fraction in `[0.0, 1.0]` at which it collides (Box3D `b3World_CastMover`).
    #[must_use]
    pub fn cast_mover(
        &self,
        mover: &CapsuleMover,
        translation: Vec3,
        filter: &QueryFilter3d,
    ) -> f32 {
        let proxy_points = [mover.center1, mover.center2];
        let proxy = ShapeProxy {
            points: &proxy_points,
            radius: mover.radius,
        };
        let mut min_fraction = 1.0f32;
        self.cast_shape(&proxy, translation, filter, |hit| {
            if hit.fraction < min_fraction {
                min_fraction = hit.fraction;
            }
            hit.fraction
        });
        min_fraction
    }

    /// Collides a capsule mover with all overlapping shapes in the world, generating collision planes (Box3D `b3World_CollideMover`).
    pub fn collide_mover(
        &self,
        mover: &CapsuleMover,
        filter: &QueryFilter3d,
        mut callback: impl FnMut(ColliderId3d, &PlaneResult3d) -> bool,
    ) {
        let mover_aabb = mover.compute_aabb();
        let mut stopped = false;
        for tree in &self.trees {
            tree.query(&mover_aabb, |collider_idx| {
                let Some(Some(slot)) = self.colliders.get(collider_idx as usize) else {
                    return true;
                };
                if !slot.collider.filter.should_query(filter) {
                    return true;
                }

                let (position, rotation) = slot.pose(&self.bodies[slot.body.index() as usize]);
                let shape_transform = Transform::new(position, rotation);
                let cid = self.collider_ids.handle_at(collider_idx).unwrap();

                let keep_going = collide_mover_shape(
                    &slot.collider.shape,
                    &shape_transform,
                    mover,
                    |plane_res| callback(cid, &plane_res),
                );
                if !keep_going {
                    stopped = true;
                    return false;
                }
                true
            });
            if stopped {
                break;
            }
        }
    }

    /// Applies an explosive radial impulse to dynamic bodies within blast radius (Box3D `b3World_Explode`).
    pub fn explode(&mut self, explosion: &ExplosionDef3d) {
        let max_range = explosion.radius + explosion.falloff;
        let blast_aabb = Aabb {
            min: explosion.position - Vec3::splat(max_range),
            max: explosion.position + Vec3::splat(max_range),
        };

        for body_idx in 0..self.bodies.len() {
            if self.body_ids.handle_at(body_idx as u32).is_none() {
                continue;
            }
            if self.bodies[body_idx].body_type != BodyType::Dynamic {
                continue;
            }

            let mut total_impulse = Vec3::ZERO;
            let mut total_torque_impulse = Vec3::ZERO;

            for i in 0..self.body_colliders[body_idx].len() {
                let cid = self.body_colliders[body_idx][i];
                let Some(Some(slot)) = self.colliders.get(cid.index() as usize) else {
                    continue;
                };

                let body = &self.bodies[body_idx];
                let (b_pos, b_rot) = slot.pose(body);
                let b_aabb = slot.collider.shape.compute_aabb(b_pos, b_rot);
                if !b_aabb.intersects(&blast_aabb) {
                    continue;
                }

                let inv_rot = b_rot.conjugate();
                let local_pos = inv_rot * (explosion.position - b_pos);

                let pt = [local_pos];
                let pt_proxy = ShapeProxy {
                    points: &pt,
                    radius: 0.0,
                };

                let corners;
                let centers;
                let zero = [Vec3::ZERO];
                let (shape_proxy_pts, shape_r): (&[Vec3], f32) = match &slot.collider.shape {
                    Shape::Sphere { radius } => (&zero, *radius),
                    Shape::Capsule {
                        radius,
                        half_height,
                    } => {
                        centers = [
                            Vec3::new(0.0, -half_height, 0.0),
                            Vec3::new(0.0, *half_height, 0.0),
                        ];
                        (&centers, *radius)
                    }
                    Shape::Box { half_extents } => {
                        let h = *half_extents;
                        corners = [
                            Vec3::new(-h.x, -h.y, -h.z),
                            Vec3::new(h.x, -h.y, -h.z),
                            Vec3::new(h.x, h.y, -h.z),
                            Vec3::new(-h.x, h.y, -h.z),
                            Vec3::new(-h.x, -h.y, h.z),
                            Vec3::new(h.x, -h.y, h.z),
                            Vec3::new(h.x, h.y, h.z),
                            Vec3::new(-h.x, h.y, h.z),
                        ];
                        (&corners, 0.0)
                    }
                    Shape::ConvexHull(hull) => (hull.points(), 0.0),
                    Shape::Triangle { a, b, c } => {
                        corners = [
                            *a,
                            *b,
                            *c,
                            Vec3::ZERO,
                            Vec3::ZERO,
                            Vec3::ZERO,
                            Vec3::ZERO,
                            Vec3::ZERO,
                        ];
                        (&corners[..3], 0.0)
                    }
                    _ => continue,
                };

                let shape_proxy = ShapeProxy {
                    points: shape_proxy_pts,
                    radius: shape_r,
                };

                let mut cache = SimplexCache::default();
                let dist_out = shape_distance(
                    &shape_proxy,
                    &pt_proxy,
                    &Transform::IDENTITY,
                    true,
                    &mut cache,
                );

                if dist_out.distance >= explosion.radius {
                    continue;
                }

                let closest_point = if dist_out.distance == 0.0 {
                    Vec3::ZERO
                } else {
                    dist_out.point_a
                };

                let mut direction = closest_point - local_pos;
                if direction.length_squared() > 1e-6 {
                    direction = direction.normalize();
                } else {
                    direction = Vec3::X;
                }

                let area = slot.collider.shape.projected_area(direction);
                let scale = if explosion.falloff > 0.0 {
                    (1.0 - dist_out.distance / explosion.radius).powf(explosion.falloff)
                } else {
                    1.0
                };

                let magnitude = explosion.impulse_per_area * area * scale;
                let world_impulse = (b_rot * direction) * magnitude;
                let world_point = b_pos + b_rot * closest_point;

                total_impulse += world_impulse;
                total_torque_impulse += (world_point - body.center_of_mass()).cross(world_impulse);
            }

            if total_impulse != Vec3::ZERO || total_torque_impulse != Vec3::ZERO {
                self.wake_body_slot(body_idx as u32);
                let body = &mut self.bodies[body_idx];
                body.linear_velocity += total_impulse * body.inv_mass;
                body.angular_velocity += body.world_inv_inertia() * total_torque_impulse;
            }
        }
    }

    /// Steps the physics simulation forward by `dt` seconds deterministically.
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        let step_start = std::time::Instant::now();
        self.step_index += 1;
        self.last_inv_h = self.solver_config.sub_steps.max(1) as f32 / dt;
        self.contact_events.clear();
        self.body_events.clear();
        self.joint_events.clear();
        let _ = self.manifolds_cache.take();

        if !self.solver_config.enable_sleep {
            for set_id in crate::island::FIRST_SLEEPING_SET..self.sets.len() as u32 {
                if !self.sets[set_id as usize].bodies.is_empty() {
                    self.wake_set(set_id);
                }
            }
        }

        // Process bodies mutated via `body_mut` since the last step.
        let mut t_idx = 0;
        while t_idx < self.touched_bodies.len() {
            let slot = self.touched_bodies[t_idx];
            t_idx += 1;
            let idx = slot as usize;
            if idx >= self.bodies.len() {
                continue;
            }
            self.body_links[idx].touched = false;
            if self.body_ids.handle_at(slot).is_none() {
                continue;
            }
            self.refresh_body_proxies(slot);
            if self.bodies[idx].is_sleeping && self.body_links[idx].set == crate::island::AWAKE_SET
            {
                if let Some(id) = self.body_ids.handle_at(slot) {
                    self.sleep_body(id);
                }
            }
        }
        self.touched_bodies.clear();

        let bp_start = std::time::Instant::now();
        // 1. Joints that keep their bodies from colliding
        if self.jointed_pairs_dirty {
            self.jointed_pairs.clear();
            self.jointed_pairs.extend(
                self.joints
                    .iter()
                    .flatten()
                    .filter(|joint| !joint.collide_connected())
                    .map(|joint| ordered(joint.body_a(), joint.body_b())),
            );
            self.jointed_pairs.sort_unstable();
            self.jointed_pairs.dedup();
            for idx in 0..self.jointed_pairs.len() {
                let (b_a, b_b) = self.jointed_pairs[idx];
                let mut s = self.contacts.body_head(b_a);
                while s != crate::contact::NULL {
                    let c = self.contacts.at(s);
                    let edge = usize::from(c.body_b == b_a);
                    let next = c.edges[edge].next;
                    if c.edges[edge ^ 1].body == b_b {
                        self.destroy_contact_slot(s, true);
                    }
                    s = next;
                }
            }
            self.jointed_pairs_dirty = false;
        }

        // 2. Broadphase pair discovery only for colliders in the move buffer (Box3D `b3FindPairsTask`).
        self.update_pairs();
        let broadphase_time = bp_start.elapsed().as_secs_f32() * 1000.0;

        // 3. Narrowphase over awake contacts (Box3D `b3Collide`).
        let np_start = std::time::Instant::now();
        self.update_contacts();
        let narrowphase_time = np_start.elapsed().as_secs_f32() * 1000.0;

        // 4. Soft-Step constraint solver in constraint-graph color order, then continuous collision
        self.sensor_hits.clear();
        let mut solve_time = 0.0;
        let mut continuous_time = 0.0;
        let had_awake_bodies = !self.sets[crate::island::AWAKE_SET as usize]
            .bodies
            .is_empty();

        if had_awake_bodies {
            let solve_start = std::time::Instant::now();
            crate::solver::solve_graph(
                &mut self.bodies,
                &self.sets[crate::island::AWAKE_SET as usize].bodies,
                &self.constraint_graph,
                &mut self.contacts,
                &mut self.joints,
                self.gravity,
                dt,
                &self.solver_config,
                &mut self.solver_scratch,
                self.executor.as_ref().map(|e| e.0.as_ref()),
            );
            solve_time = solve_start.elapsed().as_secs_f32() * 1000.0;

            // Split the island chosen in the previous step (Box3D `b3SplitIslandTask`).
            if self.island_to_split != crate::contact::NULL {
                let island = self.island_to_split;
                self.split_island(island);
                self.island_to_split = crate::contact::NULL;
            }

            if self.fast.len() < self.bodies.len() {
                self.fast.resize(self.bodies.len(), false);
            }
            for &slot in &self.sets[crate::island::AWAKE_SET as usize].bodies {
                self.fast[slot as usize] = false;
            }
            if self.solver_config.enable_continuous {
                let continuous_start = std::time::Instant::now();
                self.solve_continuous(dt);
                continuous_time = continuous_start.elapsed().as_secs_f32() * 1000.0;
            }

            // Finalize awake bodies: clear forces/torques, update sleep timers, emit move events,
            // refresh broadphase proxies, and pick split candidate.
            let enable_sleep = self.solver_config.enable_sleep;
            self.awake_islands.clear();
            self.awake_islands.resize(
                self.sets[crate::island::AWAKE_SET as usize].islands.len(),
                false,
            );
            self.split_candidate = (crate::contact::NULL, 0.0);

            let awake_len = self.sets[crate::island::AWAKE_SET as usize].bodies.len();
            for b_i in 0..awake_len {
                let slot = self.sets[crate::island::AWAKE_SET as usize].bodies[b_i];
                let idx = slot as usize;
                let body = &mut self.bodies[idx];
                body.force = Vec3::ZERO;
                body.torque = Vec3::ZERO;

                if enable_sleep && body.body_type == BodyType::Dynamic {
                    let lin_sq = body.linear_velocity.length_squared();
                    let ang_sq = body.angular_velocity.length_squared();
                    if !body.enable_sleep
                        || lin_sq > crate::island::SLEEP_LINEAR_TOLERANCE_SQ
                        || ang_sq > crate::island::SLEEP_ANGULAR_TOLERANCE_SQ
                    {
                        body.sleep_time = 0.0;
                    } else {
                        body.sleep_time += dt;
                    }
                    let sleep_time = body.sleep_time;
                    let island_id = self.body_links[idx].island;
                    if island_id != crate::contact::NULL {
                        let island = &self.islands[island_id as usize];
                        if sleep_time < crate::island::TIME_TO_SLEEP {
                            if (island.local as usize) < self.awake_islands.len() {
                                self.awake_islands[island.local as usize] = true;
                            }
                        } else if island.constraint_remove_count > 0 {
                            let (best, best_time) = self.split_candidate;
                            if sleep_time > best_time
                                || (sleep_time == best_time
                                    && (best == crate::contact::NULL || island_id > best))
                            {
                                self.split_candidate = (island_id, sleep_time);
                            }
                        }
                    }
                } else if body.body_type != BodyType::Static {
                    body.sleep_time = 0.0;
                    let island_id = self.body_links[idx].island;
                    if island_id != crate::contact::NULL {
                        let local = self.islands[island_id as usize].local as usize;
                        if local < self.awake_islands.len() {
                            self.awake_islands[local] = true;
                        }
                    }
                }

                if let Some(body_handle) = self.body_ids.handle_at(slot) {
                    let body = &self.bodies[idx];
                    self.body_links[idx].move_event = self.body_events.move_events.len() as u32;
                    self.body_events.move_events.push(BodyMoveEvent3d {
                        body: body_handle,
                        position: body.position,
                        rotation: body.rotation,
                        fell_asleep: false,
                    });
                } else {
                    self.body_links[idx].move_event = crate::contact::NULL;
                }
                self.refresh_body_proxies(slot);
            }

            // Contact hit events for awake touching contacts (Box3D `Hit Events`)
            if self.hit_event_count > 0 {
                let hit_threshold = self.solver_config.hit_event_threshold;
                for color_idx in 0..self.constraint_graph.color_count() {
                    let color = self.constraint_graph.color(color_idx);
                    for &contact_slot in &color.contact_slots {
                        let Some(contact) = self.contacts.get(contact_slot) else {
                            continue;
                        };
                        if !contact.is_touching {
                            continue;
                        }
                        let (Some(slot_a), Some(slot_b)) = (
                            self.colliders
                                .get(contact.collider_a as usize)
                                .and_then(Option::as_ref),
                            self.colliders
                                .get(contact.collider_b as usize)
                                .and_then(Option::as_ref),
                        ) else {
                            continue;
                        };
                        let enable_hit =
                            slot_a.collider.enable_hit_events || slot_b.collider.enable_hit_events;
                        if !enable_hit {
                            continue;
                        }
                        let (body_a, body_b) = (
                            &self.bodies[contact.body_a as usize],
                            &self.bodies[contact.body_b as usize],
                        );
                        let (center_a, center_b) =
                            (body_a.center_of_mass(), body_b.center_of_mass());
                        let mut best_speed = hit_threshold;
                        let mut hit_candidate = None;
                        for manifold in &contact.manifolds {
                            for point in &manifold.points {
                                let approach_speed = -point.relative_velocity;
                                if approach_speed > best_speed && point.total_normal_impulse > 0.0 {
                                    best_speed = approach_speed;
                                    let world_point = 0.5
                                        * (center_a + point.anchor_a + center_b + point.anchor_b);
                                    hit_candidate =
                                        Some((world_point, manifold.normal, approach_speed));
                                }
                            }
                        }
                        if let Some((point, normal, approach_speed)) = hit_candidate {
                            let (Some(cid_a), Some(cid_b), Some(bid_a), Some(bid_b)) = (
                                self.collider_ids.handle_at(contact.collider_a),
                                self.collider_ids.handle_at(contact.collider_b),
                                self.body_ids.handle_at(contact.body_a),
                                self.body_ids.handle_at(contact.body_b),
                            ) else {
                                continue;
                            };
                            self.contact_events.hit.push(ContactHitEvent3d {
                                collider_a: cid_a,
                                collider_b: cid_b,
                                body_a: bid_a,
                                body_b: bid_b,
                                point,
                                normal,
                                approach_speed,
                            });
                        }
                    }
                }
            }

            // Joint reaction events for awake joints (Box3D `b3JointEvents`)
            if !self.joint_ids.is_empty() {
                for color_idx in 0..self.constraint_graph.color_count() {
                    let color = self.constraint_graph.color(color_idx);
                    for &joint_slot in &color.joint_indices {
                        let Some(Some(joint)) = self.joints.get(joint_slot) else {
                            continue;
                        };
                        let base = joint.base();
                        if base.force_threshold < f32::MAX || base.torque_threshold < f32::MAX {
                            let linear_force = joint.constraint_force(self.last_inv_h).length();
                            let angular_torque = joint.constraint_torque(self.last_inv_h).length();
                            if linear_force >= base.force_threshold
                                || angular_torque >= base.torque_threshold
                            {
                                if let Some(joint_handle) =
                                    self.joint_ids.handle_at(joint_slot as u32)
                                {
                                    self.joint_events.events.push(JointEvent3d {
                                        joint: joint_handle,
                                        linear_force,
                                        angular_torque,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // 5. Transition resting islands to sleep (Box3D `b3TrySleepIsland`)
            if enable_sleep && !self.awake_islands.is_empty() {
                self.island_to_split = self.split_candidate.0;
                for local in (0..self.awake_islands.len()).rev() {
                    if !self.awake_islands[local] {
                        let island = self.sets[crate::island::AWAKE_SET as usize].islands[local];
                        self.try_sleep_island(island);
                    }
                }
            }
            self.awake_islands.clear();
        }

        // 6. Sensor overlaps at the bodies' new poses
        self.sensor_events.clear();
        if self.sensor_count > 0 && (had_awake_bodies || self.sensors_dirty) {
            self.overlap_sensors();
            self.sensors_dirty = false;
        }

        #[cfg(debug_assertions)]
        self.validate_sets();

        let step_time = step_start.elapsed().as_secs_f32() * 1000.0;
        self.profile = Profile3d {
            step: step_time,
            broadphase: broadphase_time,
            narrowphase: narrowphase_time,
            solve: solve_time,
            continuous: continuous_time,
        };
    }

    /// Broadphase pair discovery for colliders whose fat AABBs moved (Box3D `b3UpdateBroadPhasePairs`).
    fn update_pairs(&mut self) {
        self.pairs_scratch.clear();
        if self.move_buffer.is_empty() {
            return;
        }

        for m_idx in 0..self.move_buffer.len() {
            let cid = self.move_buffer[m_idx];
            let (query_aabb, body_slot, is_sensor) = {
                let Some(Some(slot)) = self.colliders.get_mut(cid as usize) else {
                    continue;
                };
                slot.moved = false;
                (slot.fat_aabb, slot.body.index(), slot.collider.is_sensor)
            };
            if is_sensor {
                continue;
            }
            let body_type = self.bodies[body_slot as usize].body_type;
            let target_trees: &[usize] = match body_type {
                BodyType::Dynamic => &[DYNAMIC_TREE, KINEMATIC_TREE, STATIC_TREE],
                BodyType::Kinematic | BodyType::Static => &[DYNAMIC_TREE],
            };
            let colliders = &self.colliders;
            let pairs = &mut self.pairs_scratch;
            for &t_idx in target_trees {
                self.trees[t_idx].query(&query_aabb, |other| {
                    if let Some(other_slot) = colliders.get(other as usize).and_then(Option::as_ref)
                    {
                        if other_slot.body.index() != body_slot && !other_slot.collider.is_sensor {
                            pairs.push(ordered(cid, other));
                        }
                    }
                    true
                });
            }
        }
        self.move_buffer.clear();

        if self.pairs_scratch.is_empty() {
            return;
        }
        self.pairs_scratch.sort_unstable();
        self.pairs_scratch.dedup();

        for pair_idx in 0..self.pairs_scratch.len() {
            let (c_a, c_b) = self.pairs_scratch[pair_idx];
            if self.contacts.slot_of(&(c_a, c_b)).is_some() {
                continue;
            }
            let (Some(slot_a), Some(slot_b)) = (
                self.colliders.get(c_a as usize).and_then(Option::as_ref),
                self.colliders.get(c_b as usize).and_then(Option::as_ref),
            ) else {
                continue;
            };
            let (col_a, col_b) = (&slot_a.collider, &slot_b.collider);
            let (b_a, b_b) = (slot_a.body.index(), slot_b.body.index());
            let (body_a, body_b) = (&self.bodies[b_a as usize], &self.bodies[b_b as usize]);
            if col_a.is_sensor
                || col_b.is_sensor
                || (col_a.shape.is_triangles() && col_b.shape.is_triangles())
                || !col_a.filter.should_collide(&col_b.filter)
                || (body_a.body_type != BodyType::Dynamic && body_b.body_type != BodyType::Dynamic)
                || self.jointed_pairs.binary_search(&ordered(b_a, b_b)).is_ok()
                || !slot_a.fat_aabb.intersects(&slot_b.fat_aabb)
            {
                continue;
            }
            let recycle = body_a.enable_contact_recycling && body_b.enable_contact_recycling;
            let new_slot = self
                .contacts
                .insert(Contact3d::new((c_a, c_b), (b_a, b_b), recycle));
            let set_a = self.body_links[b_a as usize].set;
            let set_b = self.body_links[b_b as usize].set;
            let set =
                if set_a == crate::island::DISABLED_SET || set_b == crate::island::DISABLED_SET {
                    crate::island::DISABLED_SET
                } else {
                    if set_a != crate::island::AWAKE_SET && set_b != crate::island::AWAKE_SET {
                        if set_a >= crate::island::FIRST_SLEEPING_SET {
                            self.wake_body_slot(b_a);
                        }
                        if set_b >= crate::island::FIRST_SLEEPING_SET {
                            self.wake_body_slot(b_b);
                        }
                    }
                    crate::island::AWAKE_SET
                };
            self.add_contact_to_set(new_slot, set, false);
        }
    }

    /// The collide stage (Box3D `b3Collide`): evaluates awake contacts, recycles or collides
    /// their manifolds while their fat AABBs overlap, and applies island/graph state changes
    /// in deterministic contact slot order.
    fn update_contacts(&mut self) {
        self.recycled_contacts = 0;
        self.touching_contacts = 0;

        self.awake_contact_slots.clear();
        self.awake_contact_slots
            .extend(self.constraint_graph.contact_slots());
        self.awake_contact_slots
            .extend_from_slice(&self.sets[crate::island::AWAKE_SET as usize].contacts);
        if self.awake_contact_slots.is_empty() {
            return;
        }

        self.contact_changes.clear();
        let count = self.awake_contact_slots.len();
        let parallel = self
            .executor
            .as_ref()
            .is_some_and(|e| e.0.worker_count() > 1);
        if parallel {
            self.collide_outcomes.clear();
            self.collide_outcomes
                .resize(count, CollideOutcome3d::Unchanged { touching: false });
            {
                let Self {
                    contacts,
                    colliders,
                    bodies,
                    fast,
                    solver_config,
                    friction_callback,
                    restitution_callback,
                    awake_contact_slots,
                    collide_outcomes,
                    executor,
                    ..
                } = self;
                let executor = executor.as_ref().expect("checked").0.as_ref();
                let contacts_shared = crate::parallel::Shared::new(contacts.slots_mut());
                let outcomes_shared = crate::parallel::Shared::new(collide_outcomes);
                let slots = &awake_contact_slots[..];
                let recycle_distance = solver_config.contact_recycle_distance;
                let f_cb = *friction_callback;
                let r_cb = *restitution_callback;

                korppi_core::parallel_for(executor, count, 64, &|start, end, _| {
                    for (i, &slot) in slots.iter().enumerate().take(end).skip(start) {
                        // SAFETY: each awake contact appears once in `slots` (either in the
                        // constraint graph or in `AWAKE_SET.contacts`), and each index `i` is
                        // in a single range `[start, end)`.
                        let (contact_slot, outcome) = unsafe {
                            (
                                contacts_shared.get_mut(slot as usize),
                                outcomes_shared.get_mut(i),
                            )
                        };
                        let Some(contact) = contact_slot.as_mut() else {
                            *outcome = CollideOutcome3d::Unchanged { touching: false };
                            continue;
                        };
                        *outcome = collide_contact_parallel(
                            contact,
                            colliders,
                            bodies,
                            fast,
                            recycle_distance,
                            f_cb,
                            r_cb,
                        );
                    }
                });
            }

            for i in 0..count {
                let slot = self.awake_contact_slots[i];
                match self.collide_outcomes[i] {
                    CollideOutcome3d::Disjoint => {
                        self.contact_changes.push((slot, ContactChange3d::Disjoint));
                    }
                    CollideOutcome3d::Recycled { touching } => {
                        self.recycled_contacts += 1;
                        if touching {
                            self.touching_contacts += 1;
                        }
                    }
                    CollideOutcome3d::Started => {
                        self.touching_contacts += 1;
                        self.contact_changes.push((slot, ContactChange3d::Started));
                    }
                    CollideOutcome3d::Stopped => {
                        self.contact_changes.push((slot, ContactChange3d::Stopped));
                    }
                    CollideOutcome3d::Unchanged { touching } => {
                        if touching {
                            self.touching_contacts += 1;
                        }
                    }
                    CollideOutcome3d::Deferred => {
                        let was_touching = self.contacts.get(slot).is_some_and(|c| c.was_touching);
                        let is_touching = self.update_contact(slot);
                        if is_touching {
                            self.touching_contacts += 1;
                        }
                        if is_touching && !was_touching {
                            self.contact_changes.push((slot, ContactChange3d::Started));
                        } else if !is_touching && was_touching {
                            self.contact_changes.push((slot, ContactChange3d::Stopped));
                        }
                        if let Some(contact) = self.contacts.get_mut(slot) {
                            contact.was_touching = is_touching;
                        }
                    }
                }
            }
        } else {
            for i in 0..count {
                let slot = self.awake_contact_slots[i];
                let (c_a, c_b, was_touching) = {
                    let Some(contact) = self.contacts.get(slot) else {
                        continue;
                    };
                    (contact.collider_a, contact.collider_b, contact.was_touching)
                };
                let (Some(slot_a), Some(slot_b)) = (
                    self.colliders.get(c_a as usize).and_then(Option::as_ref),
                    self.colliders.get(c_b as usize).and_then(Option::as_ref),
                ) else {
                    continue;
                };
                if !slot_a.fat_aabb.intersects(&slot_b.fat_aabb) {
                    self.contact_changes.push((slot, ContactChange3d::Disjoint));
                    continue;
                }

                let is_touching = self.update_contact(slot);
                if is_touching {
                    self.touching_contacts += 1;
                }
                if is_touching && !was_touching {
                    self.contact_changes.push((slot, ContactChange3d::Started));
                } else if !is_touching && was_touching {
                    self.contact_changes.push((slot, ContactChange3d::Stopped));
                }
                if let Some(contact) = self.contacts.get_mut(slot) {
                    contact.was_touching = is_touching;
                }
            }
        }

        if self.contact_changes.is_empty() {
            return;
        }
        self.contact_changes.sort_unstable_by_key(|&(slot, _)| slot);

        for idx in 0..self.contact_changes.len() {
            let (slot, change) = self.contact_changes[idx];
            match change {
                ContactChange3d::Disjoint => {
                    self.destroy_contact_slot(slot, true);
                }
                ContactChange3d::Started => {
                    let Some(contact) = self.contacts.get(slot) else {
                        continue;
                    };
                    let c_pair = (contact.collider_a, contact.collider_b);
                    let b_pair = (contact.body_a, contact.body_b);
                    let in_graph = contact.color != crate::contact::NULL;
                    let enable_events = self
                        .colliders
                        .get(c_pair.0 as usize)
                        .and_then(Option::as_ref)
                        .is_some_and(|s| s.collider.enable_contact_events)
                        || self
                            .colliders
                            .get(c_pair.1 as usize)
                            .and_then(Option::as_ref)
                            .is_some_and(|s| s.collider.enable_contact_events);

                    self.link_contact(slot);
                    if !in_graph {
                        self.remove_contact_from_set(slot);
                        self.add_contact_to_set(slot, crate::island::AWAKE_SET, true);
                    }
                    if enable_events {
                        if let (Some(ca), Some(cb), Some(ba), Some(bb)) = (
                            self.collider_ids.handle_at(c_pair.0),
                            self.collider_ids.handle_at(c_pair.1),
                            self.body_ids.handle_at(b_pair.0),
                            self.body_ids.handle_at(b_pair.1),
                        ) {
                            self.contact_events.begin.push(ContactBeginEvent3d {
                                collider_a: ca,
                                collider_b: cb,
                                body_a: ba,
                                body_b: bb,
                            });
                        }
                    }
                }
                ContactChange3d::Stopped => {
                    let Some(contact) = self.contacts.get(slot) else {
                        continue;
                    };
                    let c_pair = (contact.collider_a, contact.collider_b);
                    let b_pair = (contact.body_a, contact.body_b);
                    let in_graph = contact.color != crate::contact::NULL;
                    let in_island = contact.island != crate::contact::NULL;
                    let enable_events = self
                        .colliders
                        .get(c_pair.0 as usize)
                        .and_then(Option::as_ref)
                        .is_some_and(|s| s.collider.enable_contact_events)
                        || self
                            .colliders
                            .get(c_pair.1 as usize)
                            .and_then(Option::as_ref)
                            .is_some_and(|s| s.collider.enable_contact_events);

                    if in_island {
                        self.unlink_contact(slot);
                    }
                    if in_graph {
                        self.remove_contact_from_set(slot);
                        self.add_contact_to_set(slot, crate::island::AWAKE_SET, false);
                    }
                    if enable_events {
                        if let (Some(ca), Some(cb), Some(ba), Some(bb)) = (
                            self.collider_ids.handle_at(c_pair.0),
                            self.collider_ids.handle_at(c_pair.1),
                            self.body_ids.handle_at(b_pair.0),
                            self.body_ids.handle_at(b_pair.1),
                        ) {
                            self.contact_events.end.push(ContactEndEvent3d {
                                collider_a: ca,
                                collider_b: cb,
                                body_a: ba,
                                body_b: bb,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Updates the contact in `slot` (Box3D `b3CollideTask`): recycles its manifold when the
    /// bodies barely moved relative to each other since its last collide, else collides its
    /// colliders. Returns whether it touches.
    fn update_contact(&mut self, slot: u32) -> bool {
        let Some(contact) = self.contacts.get_mut(slot) else {
            return false;
        };
        let (Some(slot_a), Some(slot_b)) = (
            self.colliders
                .get(contact.collider_a as usize)
                .and_then(Option::as_ref),
            self.colliders
                .get(contact.collider_b as usize)
                .and_then(Option::as_ref),
        ) else {
            return false;
        };
        let (a, b) = (contact.body_a as usize, contact.body_b as usize);
        let (body_a, body_b) = (&self.bodies[a], &self.bodies[b]);
        let xf_a = Transform::new(body_a.position, body_a.rotation);
        let xf_b = Transform::new(body_b.position, body_b.rotation);

        let recycle_distance = self.solver_config.contact_recycle_distance;
        let is_fast = self.fast.get(a).copied().unwrap_or(false)
            || self.fast.get(b).copied().unwrap_or(false);
        let tolerance = if contact.is_touching {
            recycle_distance
        } else {
            recycle_distance.min(SPECULATIVE_DISTANCE)
        };
        if !is_fast && recycle_distance > 0.0 && contact.relative_valid && contact.recycle {
            let same_rot_a = body_a.rotation == contact.cached_rotation_a;
            let same_rot_b = body_b.rotation == contact.cached_rotation_b;
            let xf = xf_a.inv_mul(&xf_b);
            let distance_squared = (xf.p - contact.cached_relative_position).length_squared();
            if same_rot_a && same_rot_b {
                if distance_squared < tolerance * tolerance {
                    let dc = body_b.center_of_mass() - body_a.center_of_mass();
                    for manifold in &mut contact.manifolds {
                        let normal = manifold.normal;
                        for p in &mut manifold.points {
                            p.separation =
                                p.base_separation + (dc + (p.anchor_b - p.anchor_a)).dot(normal);
                        }
                    }
                    self.recycled_contacts += 1;
                    return contact.is_touching;
                }
            } else {
                // cos(angle / 2) between the rotations, squared: close to 1 for a small turn.
                let angle_a = body_a.rotation.dot(contact.cached_rotation_a);
                let angle_b = body_b.rotation.dot(contact.cached_rotation_b);
                let angular_distance = (angle_a * angle_a).min(angle_b * angle_b);
                if angular_distance > CONTACT_RECYCLE_ANGULAR_DISTANCE
                    && distance_squared < tolerance * tolerance
                {
                    let slack = tolerance - distance_squared.sqrt();
                    let extent = |body: &RigidBody| {
                        if body.body_type == BodyType::Static {
                            Vec3::ZERO
                        } else {
                            body.max_extent
                        }
                    };
                    let max_extent = extent(body_a).max(extent(body_b));
                    // Conservative advancement: how far the turn moves the farthest point.
                    let qr = inv_mul_quat(contact.cached_relative_rotation, xf.q);
                    let arc = modified_cross(Vec3::new(qr.x, qr.y, qr.z).abs(), max_extent);
                    if 4.0 * arc.length_squared() < slack * slack {
                        let rotation_a = Mat3::from_quat(
                            body_a.rotation * contact.cached_rotation_a.conjugate(),
                        );
                        let rotation_b = Mat3::from_quat(
                            body_b.rotation * contact.cached_rotation_b.conjugate(),
                        );
                        let dc = body_b.center_of_mass() - body_a.center_of_mass();
                        // Keep the anchors, update the separations as the substeps do.
                        for manifold in &mut contact.manifolds {
                            let normal = manifold.normal;
                            for p in &mut manifold.points {
                                let r_a = rotation_a * p.anchor_a;
                                let r_b = rotation_b * p.anchor_b;
                                p.separation = p.base_separation + (dc + (r_b - r_a)).dot(normal);
                            }
                        }
                        self.recycled_contacts += 1;
                        return contact.is_touching;
                    }
                }
            }
        }

        contact.cached_rotation_a = body_a.rotation;
        contact.cached_rotation_b = body_b.rotation;
        let relative = xf_a.inv_mul(&xf_b);
        contact.cached_relative_position = relative.p;
        contact.cached_relative_rotation = relative.q;
        contact.relative_valid = true;
        if !slot_a.aabb.intersects(&slot_b.aabb) {
            contact.separate();
            return false;
        }
        // Anchors are relative to the bodies' centres of mass.
        let (center_a, center_b) = (body_a.center_of_mass(), body_b.center_of_mass());
        let set_anchors = |manifold: &mut ContactManifold| {
            for p in &mut manifold.points {
                p.anchor_a = p.point - center_a;
                p.anchor_b = p.point - center_b;
            }
        };
        let flip = slot_b.collider.shape.is_triangles();
        if flip || slot_a.collider.shape.is_triangles() {
            // A mesh contact, the mesh as Box3D's shape A.
            let (mesh, convex, mesh_body, convex_body) = if flip {
                (slot_b, slot_a, body_b, body_a)
            } else {
                (slot_a, slot_b, body_a, body_b)
            };
            let (Some(triangles), Some(shape)) = (
                triangles_of(&mesh.collider.shape),
                Convex::of(&convex.collider.shape),
            ) else {
                contact.separate();
                return false;
            };
            let (mesh_position, mesh_rotation) = mesh.pose(mesh_body);
            let (convex_position, convex_rotation) = convex.pose(convex_body);
            let pair = MeshPair {
                triangles,
                xf_mesh: Transform::new(mesh_position, mesh_rotation),
                convex: &shape,
                xf_convex: Transform::new(convex_position, convex_rotation),
                convex_bounds: convex.bounds(convex_body),
                is_fast,
            };
            contact.is_touching = update_mesh_manifolds(
                &mut contact.manifolds,
                &mut contact.mesh_cache,
                &pair,
                (mesh.body.index(), convex.body.index()),
                flip,
                &mut self.mesh_scratch,
            );
            let radius_b = match &convex.collider.shape {
                Shape::Sphere { radius } | Shape::Capsule { radius, .. } => *radius,
                Shape::Box { half_extents } => half_extents.min_element(),
                Shape::ConvexHull(hull) => hull.inner_radius(),
                _ => 0.0,
            };
            let convex_mat = convex.collider.surface_material();
            let convex_vt = convex_rotation * convex_mat.tangent_velocity;

            for manifold in &mut contact.manifolds {
                let (f, r, rr, vt_mesh) =
                    if !mesh.collider.materials.is_empty() && !manifold.points.is_empty() {
                        let mut sum_f = 0.0;
                        let mut sum_r = 0.0;
                        let mut sum_rr = 0.0;
                        let mut sum_vt = Vec3::ZERO;
                        for p in &manifold.points {
                            let tri_idx = p.triangle_index as usize;
                            let mat_idx = match &mesh.collider.shape {
                                Shape::TriangleMesh(m) => m.material_index(tri_idx) as usize,
                                Shape::HeightField(hf) => hf.material_index(tri_idx) as usize,
                                _ => 0,
                            };
                            let m_mat = mesh
                                .collider
                                .materials
                                .get(mat_idx)
                                .copied()
                                .unwrap_or_else(|| mesh.collider.surface_material());
                            sum_f += (self.friction_callback)(
                                m_mat.friction,
                                m_mat.user_material_id,
                                convex_mat.friction,
                                convex_mat.user_material_id,
                            );
                            sum_r += (self.restitution_callback)(
                                m_mat.restitution,
                                m_mat.user_material_id,
                                convex_mat.restitution,
                                convex_mat.user_material_id,
                            );
                            sum_rr += m_mat.rolling_resistance.max(convex_mat.rolling_resistance)
                                * radius_b;
                            sum_vt += m_mat.tangent_velocity;
                        }
                        let inv_n = 1.0 / manifold.points.len() as f32;
                        (sum_f * inv_n, sum_r * inv_n, sum_rr * inv_n, sum_vt * inv_n)
                    } else {
                        let m_mat = mesh.collider.surface_material();
                        (
                            (self.friction_callback)(
                                m_mat.friction,
                                m_mat.user_material_id,
                                convex_mat.friction,
                                convex_mat.user_material_id,
                            ),
                            (self.restitution_callback)(
                                m_mat.restitution,
                                m_mat.user_material_id,
                                convex_mat.restitution,
                                convex_mat.user_material_id,
                            ),
                            m_mat.rolling_resistance.max(convex_mat.rolling_resistance) * radius_b,
                            m_mat.tangent_velocity,
                        )
                    };
                manifold.friction = f;
                manifold.restitution = r;
                manifold.rolling_resistance = rr;
                let vt_mesh_world = mesh_rotation * vt_mesh;
                manifold.tangent_velocity = if flip {
                    convex_vt - vt_mesh_world
                } else {
                    vt_mesh_world - convex_vt
                };
                set_anchors(manifold);
            }
            return contact.is_touching;
        }
        let is_compound_a = matches!(slot_a.collider.shape, Shape::Compound(_));
        let is_compound_b = matches!(slot_b.collider.shape, Shape::Compound(_));
        if is_compound_a || is_compound_b {
            let flip = is_compound_b;
            let (comp_slot, other_slot, comp_body, other_body) = if flip {
                (slot_b, slot_a, body_b, body_a)
            } else {
                (slot_a, slot_b, body_a, body_b)
            };
            let Shape::Compound(ref compound) = comp_slot.collider.shape else {
                unreachable!()
            };
            let (comp_pos, comp_rot) = comp_slot.pose(comp_body);
            let (other_pos, other_rot) = other_slot.pose(other_body);
            let other_bounds = other_slot.bounds(other_body).expand(SPECULATIVE_DISTANCE);

            // Transform other_bounds into compound local space
            let inv_rot = comp_rot.conjugate();
            let inv_mat = Mat3::from_quat(inv_rot);
            let local_center = inv_mat * (other_bounds.center() - comp_pos);
            let abs_mat = Mat3::from_cols(
                inv_mat.x_axis.abs(),
                inv_mat.y_axis.abs(),
                inv_mat.z_axis.abs(),
            );
            let local_half = abs_mat * other_bounds.half_extents();
            let local_min = local_center - local_half;
            let local_max = local_center + local_half;

            self.compound_scratch.clear();
            let mut cache = contact.cache;
            compound.query(local_min, local_max, |child_idx| {
                let child = compound.child(child_idx);
                let child_world_pos = comp_pos + comp_rot * child.position;
                let child_world_rot = comp_rot * child.rotation;
                let child_xf = Transform::new(child_world_pos, child_world_rot);
                let other_xf = Transform::new(other_pos, other_rot);

                let manifold_opt = if flip {
                    collide_convex_shapes(
                        &other_slot.collider.shape,
                        other_xf,
                        other_slot.body.index(),
                        &child.shape,
                        child_xf,
                        comp_slot.body.index(),
                        &mut cache,
                    )
                } else {
                    collide_convex_shapes(
                        &child.shape,
                        child_xf,
                        comp_slot.body.index(),
                        &other_slot.collider.shape,
                        other_xf,
                        other_slot.body.index(),
                        &mut cache,
                    )
                };

                if let Some(mut manifold) = manifold_opt {
                    let (slot_1, slot_2) = if flip {
                        (other_slot, comp_slot)
                    } else {
                        (comp_slot, other_slot)
                    };
                    let friction = (self.friction_callback)(
                        slot_1.collider.friction,
                        slot_1.collider.user_material_id,
                        slot_2.collider.friction,
                        slot_2.collider.user_material_id,
                    );
                    let restitution = (self.restitution_callback)(
                        slot_1.collider.restitution,
                        slot_1.collider.user_material_id,
                        slot_2.collider.restitution,
                        slot_2.collider.user_material_id,
                    );
                    let mut rolling_resistance = 0.0;
                    if comp_slot.collider.rolling_resistance > 0.0
                        || other_slot.collider.rolling_resistance > 0.0
                    {
                        let radius = child
                            .shape
                            .rolling_radius()
                            .max(other_slot.collider.shape.rolling_radius());
                        rolling_resistance = comp_slot
                            .collider
                            .rolling_resistance
                            .max(other_slot.collider.rolling_resistance)
                            * radius;
                    }
                    manifold.friction = friction;
                    manifold.restitution = restitution;
                    manifold.rolling_resistance = rolling_resistance;
                    let vt_comp = comp_rot * comp_slot.collider.tangent_velocity;
                    let vt_other = other_rot * other_slot.collider.tangent_velocity;
                    manifold.tangent_velocity = if flip {
                        vt_other - vt_comp
                    } else {
                        vt_comp - vt_other
                    };
                    for p in &mut manifold.points {
                        p.triangle_index = child_idx as u32;
                    }
                    set_anchors(&mut manifold);
                    self.compound_scratch.push(manifold);
                }
                true
            });
            contact.cache = cache;

            // Warm-start matching from previous step's manifolds
            for new_m in &mut self.compound_scratch {
                let child_idx = new_m
                    .points
                    .as_slice()
                    .first()
                    .map(|p| p.triangle_index)
                    .unwrap_or(u32::MAX);
                let matched_old = contact.manifolds.iter().find(|old_m| {
                    old_m
                        .points
                        .as_slice()
                        .iter()
                        .any(|p| p.triangle_index == child_idx)
                        || old_m.normal.dot(new_m.normal) > 0.995
                });
                if let Some(old_m) = matched_old {
                    new_m.friction_impulse = old_m.friction_impulse;
                    new_m.twist_impulse = old_m.twist_impulse;
                    new_m.rolling_impulse = old_m.rolling_impulse;
                    let mut claimed = [false; MAX_MANIFOLD_POINTS];
                    for point in &mut new_m.points {
                        let matched = old_m
                            .points
                            .iter()
                            .enumerate()
                            .find(|(j, p)| !claimed[*j] && p.feature_id == point.feature_id);
                        if let Some((j, p)) = matched {
                            point.normal_impulse = p.normal_impulse;
                            claimed[j] = true;
                        }
                    }
                }
            }

            contact.manifolds.clear();
            if self.compound_scratch.is_empty() {
                contact.separate();
            } else {
                contact.manifolds.extend_from_slice(&self.compound_scratch);
                contact.is_touching = true;
            }
            self.compound_scratch.clear();
            return contact.is_touching;
        }
        let collision_a = slot_a.collision_body(body_a);
        let collision_b = slot_b.collision_body(body_b);
        match collide_shapes_cached(collision_a, collision_b, &mut contact.cache) {
            Some(mut manifold) => {
                manifold.friction = (self.friction_callback)(
                    slot_a.collider.friction,
                    slot_a.collider.user_material_id,
                    slot_b.collider.friction,
                    slot_b.collider.user_material_id,
                );
                manifold.restitution = (self.restitution_callback)(
                    slot_a.collider.restitution,
                    slot_a.collider.user_material_id,
                    slot_b.collider.restitution,
                    slot_b.collider.user_material_id,
                );
                let (_, rot_a) = slot_a.pose(body_a);
                let (_, rot_b) = slot_b.pose(body_b);
                let vt_a = rot_a * slot_a.collider.tangent_velocity;
                let vt_b = rot_b * slot_b.collider.tangent_velocity;
                manifold.tangent_velocity = vt_a - vt_b;
                set_anchors(&mut manifold);
                contact.update(manifold);
            }
            None => contact.separate(),
        }
        contact.is_touching
    }

    /// The world's counts (Box3D `b3World_GetCounters`).
    #[must_use]
    pub fn counters(&self) -> Counters3d {
        Counters3d {
            body_count: self.body_ids.len(),
            collider_count: self.collider_ids.len(),
            contact_count: self.contacts.len(),
            touching_contact_count: self.touching_contacts,
            recycled_contact_count: self.recycled_contacts,
            joint_count: self.joint_ids.len(),
        }
    }

    /// The contacts between collider pairs that the broadphase found, touching or not, in a
    /// deterministic order.
    pub fn contacts(&self) -> impl Iterator<Item = &Contact3d> + '_ {
        self.contacts.iter()
    }

    /// Sweeps the fast bodies (Box3D's continuous collision in `b3FinalizeBodiesTask`): a
    /// dynamic body whose step moved a point of its shape more than its `safety_factor` times
    /// its smallest extent, by velocity or by the solver's motion. Bullets go after the other
    /// bodies have moved.
    fn solve_continuous(&mut self, dt: f32) {
        self.bullets.clear();
        for i in 0..self.sets[crate::island::AWAKE_SET as usize].bodies.len() {
            let slot = self.sets[crate::island::AWAKE_SET as usize].bodies[i];
            let index = slot as usize;
            let body = &self.bodies[index];
            if body.body_type != BodyType::Dynamic || body.is_sleeping {
                continue;
            }
            let (min_extent, max_extent) = (body.min_extent, body.max_extent);
            let motion = self.solver_scratch.motion(index);
            // The speed of the farthest point, and how far the step moved it.
            let local_omega = inv_rotate(body.rotation, body.angular_velocity);
            let dq = motion.delta_rotation;
            let local_turn = inv_rotate(body.rotation, Vec3::new(dq.x, dq.y, dq.z));
            let max_velocity = body.linear_velocity.length()
                + modified_cross(local_omega.abs(), max_extent).length();
            let max_delta_position = motion.delta_position.length()
                + 2.0 * modified_cross(local_turn.abs(), max_extent).length();
            if max_delta_position.max(max_velocity * dt) > body.safety_factor * min_extent {
                self.fast[index] = true;
                if body.is_bullet {
                    self.bullets.push(index);
                } else {
                    self.sweep_fast_body(index, dt);
                }
            }
        }
        for i in 0..self.bullets.len() {
            self.sweep_fast_body(self.bullets[i], dt);
        }
    }

    /// Box3D `b3SolveContinuous`: sweeps body `index` from its pose at the start of the step to
    /// its pose now against static shapes (a bullet also against kinematic and dynamic bodies
    /// that are not bullets), puts it back to the first time of impact with the gravity of the
    /// lost time taken off its velocity, and records the sensors it swept through before that.
    fn sweep_fast_body(&mut self, index: usize, dt: f32) {
        let (start_center, start_rotation) = self.solver_scratch.start_pose(index);
        let body = &self.bodies[index];
        let is_bullet = body.is_bullet;
        // The sweep is relative to the start, for precision far from the origin.
        let base = start_center;
        let sweep = Sweep {
            local_center: body.local_center,
            c1: Vec3::ZERO,
            c2: body.center_of_mass() - base,
            q1: start_rotation,
            q2: body.rotation,
        };
        let start_origin = start_center - start_rotation * body.local_center;

        let mut fraction = 1.0;
        let mut sensor_hits = [(0u32, 0u32, 0.0f32); 8];
        let mut sensor_hit_count = 0;
        for &fast_id in &self.body_colliders[index] {
            let Some(fast_slot) = self
                .colliders
                .get(fast_id.index() as usize)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let fast = &fast_slot.collider;
            // No continuous collision for sensors and meshes.
            if fast.is_sensor || !fast.shape.is_convex() {
                continue;
            }
            let shape_sweep = sweep.of_frame(&fast.local_frame);
            let (p1, q1) = fast.world_pose(start_origin, start_rotation);
            let (p2, q2) = fast.world_pose(body.position, body.rotation);
            let swept = fast
                .shape
                .compute_aabb(p1, q1)
                .expand(SPECULATIVE_DISTANCE)
                .union(&fast.shape.compute_aabb(p2, q2));
            self.continuous_candidates.clear();
            let candidates = &mut self.continuous_candidates;
            for tree in &self.trees {
                tree.query(&swept, |other| {
                    candidates.push(other);
                    true
                });
            }
            // Deterministic whatever the tree's shape.
            self.continuous_candidates.sort_unstable();
            self.continuous_candidates.dedup();

            for &other in &self.continuous_candidates {
                let Some(other_slot) = self.colliders.get(other as usize).and_then(Option::as_ref)
                else {
                    continue;
                };
                if other_slot.body.index() as usize == index {
                    continue;
                }
                let collider = &other_slot.collider;
                let other_body = &self.bodies[other_slot.body.index() as usize];
                let is_sensor = collider.is_sensor;
                if (is_sensor && !(collider.enable_sensor_events && fast.enable_sensor_events))
                    || !fast.filter.should_collide(&collider.filter)
                    || (!is_bullet && other_body.body_type != BodyType::Static)
                    || other_body.is_bullet
                    || self
                        .jointed_pairs
                        .binary_search(&ordered(index as u32, other_slot.body.index()))
                        .is_ok()
                {
                    continue;
                }
                let (position, rotation) = other_slot.pose(other_body);
                let other_sweep = Sweep::at_rest(&Transform::new(position - base, rotation));
                let hit = shape_time_of_impact(
                    &collider.shape,
                    &other_sweep,
                    &fast.shape,
                    &shape_sweep,
                    fraction,
                    is_sensor,
                );
                if is_sensor {
                    if hit <= fraction && sensor_hit_count < sensor_hits.len() {
                        sensor_hits[sensor_hit_count] = (other, fast_id.index(), hit);
                        sensor_hit_count += 1;
                    }
                } else if 0.0 < hit && hit < fraction {
                    fraction = hit;
                }
            }
        }

        if fraction < 1.0 {
            let rotation = nlerp(sweep.q1, sweep.q2, fraction);
            let center = base + lerp(sweep.c1, sweep.c2, fraction);
            let gravity = self.gravity;
            let body = &mut self.bodies[index];
            body.rotation = rotation;
            body.position = center - rotation * body.local_center;
            // The lost time's gravity; other forces are ignored, as in Box3D.
            let dv = gravity * (-(1.0 - fraction) * dt * body.gravity_scale);
            let locks = body.motion_locks;
            let dv = Vec3::new(
                if locks.linear_x { 0.0 } else { dv.x },
                if locks.linear_y { 0.0 } else { dv.y },
                if locks.linear_z { 0.0 } else { dv.z },
            );
            body.linear_velocity += dv;
        }
        for &(sensor, visitor, hit) in &sensor_hits[..sensor_hit_count] {
            // Not the sensors beyond the impact.
            if hit < fraction {
                if let (Some(sh), Some(vh)) = (
                    self.collider_ids.handle_at(sensor),
                    self.collider_ids.handle_at(visitor),
                ) {
                    self.sensor_hits.push((sh, vh));
                }
            }
        }
    }

    /// Updates the speculative AABBs of `body_slot`'s colliders, and moves their broadphase
    /// proxies (enlarging by `aabb_margin`) only when the speculative AABB exits the stored
    /// fat AABB (Box3D `b3FinalizePositionsTask`).
    fn refresh_body_proxies(&mut self, body_slot: u32) {
        let Some(body) = self.bodies.get(body_slot as usize) else {
            return;
        };
        let Some(colliders) = self.body_colliders.get(body_slot as usize) else {
            return;
        };
        let is_static = body.body_type == BodyType::Static;
        for &cid_handle in colliders {
            let cid = cid_handle.index();
            let Some(Some(slot)) = self.colliders.get_mut(cid as usize) else {
                continue;
            };
            let new_aabb = slot.bounds(body).expand(SPECULATIVE_DISTANCE);
            slot.aabb = new_aabb;
            if !slot.fat_aabb.contains(&new_aabb) {
                let margin = if is_static {
                    SPECULATIVE_DISTANCE
                } else {
                    slot.collider.shape.aabb_margin()
                };
                let new_fat_aabb = new_aabb.expand(margin);
                slot.fat_aabb = new_fat_aabb;
                let tree_idx = (slot.proxy & 3) as usize;
                let proxy_id = slot.proxy >> 2;
                self.trees[tree_idx].move_fat_proxy(proxy_id, new_fat_aabb);
                if !slot.moved {
                    slot.moved = true;
                    self.move_buffer.push(cid);
                }
            }
        }
    }

    /// Recomputes every sensor's overlaps and reports the changes (Box3D `b3OverlapSensors`).
    /// A sensor overlaps the convex colliders of other bodies when both enable sensor events,
    /// their filters agree and the shapes intersect.
    fn overlap_sensors(&mut self) {
        for sensor_index in 0..self.colliders.len() {
            let Some(sensor_slot) = self.colliders.get(sensor_index).and_then(Option::as_ref)
            else {
                continue;
            };
            let sensor = &sensor_slot.collider;
            if !sensor.is_sensor {
                continue;
            }
            let sensor_handle = self
                .collider_ids
                .handle_at(sensor_index as u32)
                .expect("live sensor collider");
            self.sensor_scratch.clear();
            if sensor.enable_sensor_events {
                let body = &self.bodies[sensor_slot.body.index() as usize];
                let bounds = sensor_slot.bounds(body).expand(SPECULATIVE_DISTANCE);
                self.sensor_candidates.clear();
                let candidates = &mut self.sensor_candidates;
                for tree in &self.trees {
                    tree.query(&bounds, |other| {
                        candidates.push(other);
                        true
                    });
                }
                self.sensor_candidates.sort_unstable();
                self.sensor_candidates.dedup();
                let sensor_body = sensor_slot.collision_body(body);
                for &other in &self.sensor_candidates {
                    let Some(visitor_slot) =
                        self.colliders.get(other as usize).and_then(Option::as_ref)
                    else {
                        continue;
                    };
                    let visitor = &visitor_slot.collider;
                    if visitor_slot.body == sensor_slot.body
                        || !visitor.enable_sensor_events
                        || !visitor.shape.is_convex()
                        || !sensor.filter.should_collide(&visitor.filter)
                    {
                        continue;
                    }
                    let visitor_body = visitor_slot
                        .collision_body(&self.bodies[visitor_slot.body.index() as usize]);
                    if shapes_overlap(sensor_body, visitor_body) {
                        let visitor_handle = self
                            .collider_ids
                            .handle_at(other)
                            .expect("live visitor collider");
                        self.sensor_scratch.push(visitor_handle);
                    }
                }
                // The bodies a sweep took through the sensor this step.
                self.sensor_scratch.extend(
                    self.sensor_hits
                        .iter()
                        .filter(|&&(s, _)| s == sensor_handle)
                        .map(|&(_, visitor)| visitor),
                );
                self.sensor_scratch.sort_unstable();
                self.sensor_scratch.dedup();
            }

            // Both lists are sorted: walk them together for what ended and what began.
            let (old, new) = (&self.sensor_overlaps[sensor_index], &self.sensor_scratch);
            let (mut i, mut j) = (0, 0);
            while i < old.len() || j < new.len() {
                if j == new.len() || (i < old.len() && old[i] < new[j]) {
                    self.sensor_events.end.push(SensorEndEvent3d {
                        sensor: sensor_handle,
                        visitor: old[i],
                    });
                    i += 1;
                } else if i == old.len() || new[j] < old[i] {
                    self.sensor_events.begin.push(SensorBeginEvent3d {
                        sensor: sensor_handle,
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
                &mut self.sensor_overlaps[sensor_index],
            );
        }
    }

    /// Captures a bitwise deterministic snapshot of the simulation world state.
    #[must_use]
    pub fn create_snapshot(&self) -> WorldSnapshot3d {
        let bodies = self
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| BodySnapshot3d::from_body(i as u32, b))
            .collect();
        WorldSnapshot3d {
            step_index: self.step_index,
            bodies,
            raw_bodies: self.bodies.clone(),
            body_ids: self.body_ids.clone(),
            colliders: self.colliders.clone(),
            collider_ids: self.collider_ids.clone(),
            body_colliders: self.body_colliders.clone(),
            trees: self.trees.clone(),
            move_buffer: self.move_buffer.clone(),
            touched_bodies: self.touched_bodies.clone(),
            manifolds: self.manifolds().to_vec(),
            contacts: self.contacts.clone(),
            fast: self.fast.clone(),
            joints: self.joints.clone(),
            joint_ids: self.joint_ids.clone(),
            sensor_overlaps: self.sensor_overlaps.clone(),
            island_manager: self.island_manager.clone(),
            sets: self.sets.clone(),
            free_sets: self.free_sets.clone(),
            islands: self.islands.clone(),
            free_islands: self.free_islands.clone(),
            island_count: self.island_count,
            island_to_split: self.island_to_split,
            body_links: self.body_links.clone(),
            joint_links: self.joint_links.clone(),
            constraint_graph: self.constraint_graph.clone(),
            gravity: self.gravity,
            solver_config: self.solver_config,
        }
    }

    /// Restores the simulation world to the exact state captured in `snapshot`.
    pub fn restore_snapshot(&mut self, snapshot: &WorldSnapshot3d) {
        self.step_index = snapshot.step_index;
        self.body_ids = snapshot.body_ids.clone();
        let _ = self.manifolds_cache.take();

        if !snapshot.raw_bodies.is_empty() {
            self.bodies = snapshot.raw_bodies.clone();
        } else if self.bodies.len() < snapshot.bodies.len() {
            self.bodies.resize(
                snapshot.bodies.len(),
                RigidBody::new(BodyType::Dynamic, Vec3::ZERO),
            );
        }

        for bs in &snapshot.bodies {
            let id = bs.id as usize;
            if let Some(b) = self.bodies.get_mut(id) {
                b.position = bs.position;
                b.rotation = bs.rotation;
                b.linear_velocity = bs.linear_velocity;
                b.angular_velocity = bs.angular_velocity;
                b.is_sleeping = bs.is_sleeping;
                b.sleep_time = bs.sleep_time;
                b.force = Vec3::ZERO;
                b.torque = Vec3::ZERO;
            }
        }

        self.colliders = snapshot.colliders.clone();
        self.collider_ids = snapshot.collider_ids.clone();
        self.body_colliders = snapshot.body_colliders.clone();

        self.trees = snapshot.trees.clone();
        self.move_buffer.clone_from(&snapshot.move_buffer);
        self.touched_bodies.clone_from(&snapshot.touched_bodies);
        self.sensor_count = 0;
        self.hit_event_count = 0;
        for slot in self.colliders.iter().flatten() {
            if slot.collider.is_sensor {
                self.sensor_count += 1;
            }
            if slot.collider.enable_hit_events {
                self.hit_event_count += 1;
            }
        }
        self.sensors_dirty = true;

        self.joints = snapshot.joints.clone();
        self.joint_ids = snapshot.joint_ids.clone();

        // Rebuild jointed pairs
        self.jointed_pairs.clear();
        self.jointed_pairs.extend(
            self.joints
                .iter()
                .flatten()
                .filter(|joint| !joint.collide_connected())
                .map(|joint| ordered(joint.body_a(), joint.body_b())),
        );
        self.jointed_pairs.sort_unstable();
        self.jointed_pairs.dedup();
        self.jointed_pairs_dirty = false;

        self.contacts = snapshot.contacts.clone();
        self.contacts.rebuild_index();
        self.fast.clone_from(&snapshot.fast);
        self.island_manager = snapshot.island_manager.clone();
        if !snapshot.sets.is_empty() {
            self.sets.clone_from(&snapshot.sets);
            self.free_sets.clone_from(&snapshot.free_sets);
            self.islands.clone_from(&snapshot.islands);
            self.free_islands.clone_from(&snapshot.free_islands);
            self.island_count = snapshot.island_count;
            self.island_to_split = snapshot.island_to_split;
            self.body_links.clone_from(&snapshot.body_links);
            self.joint_links.clone_from(&snapshot.joint_links);
            self.constraint_graph.clone_from(&snapshot.constraint_graph);
        }
        self.sensor_overlaps = snapshot.sensor_overlaps.clone();
        if self.sensor_overlaps.len() < self.colliders.len() {
            self.sensor_overlaps
                .resize_with(self.colliders.len(), Vec::new);
        }
        if self.body_colliders.len() < self.bodies.len() {
            self.body_colliders.resize_with(self.bodies.len(), Vec::new);
        }
        self.gravity = snapshot.gravity;
        self.solver_config = snapshot.solver_config;
    }

    /// Creates a new physics world initialized from a snapshot.
    #[must_use]
    pub fn from_snapshot(snapshot: &WorldSnapshot3d) -> Self {
        let mut world = Self::new();
        world.restore_snapshot(snapshot);
        world
    }

    /// Computes the 64-bit FNV-1a hash of the current simulation state.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        crate::snapshot::compute_state_hash(crate::snapshot::StateHashInput {
            body_count: self.bodies.len(),
            body_active: |slot: u32| self.body_ids.handle_at(slot).is_some(),
            body_data: |idx: usize| {
                let b = &self.bodies[idx];
                (
                    b.position,
                    b.rotation,
                    b.linear_velocity,
                    b.angular_velocity,
                    b.sleep_time,
                    b.is_sleeping,
                )
            },
            contacts: &self.contacts,
            joints: &self.joints,
            joint_ids: &self.joint_ids,
            body_ids: &self.body_ids,
            collider_ids: &self.collider_ids,
            sensor_overlaps: &self.sensor_overlaps,
        })
    }
}

/// A body pair with the smaller index first.
fn ordered(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

fn shape_cast_shape(
    shape: &Shape,
    shape_transform: &Transform,
    proxy: &ShapeProxy<'_>,
    translation: Vec3,
    max_fraction: f32,
) -> Option<CastOutput> {
    let inv_xf = shape_transform.inverse();
    let mut local_points = [Vec3::ZERO; 16];
    let count = proxy.points.len().min(16);
    for (dst, &src) in local_points[..count].iter_mut().zip(proxy.points) {
        *dst = inv_xf.apply(src);
    }
    let local_proxy = ShapeProxy {
        points: &local_points[..count],
        radius: proxy.radius,
    };
    let local_translation = inv_xf.q * translation;
    let can_encroach = proxy.radius > 0.0;

    let local_out = match shape {
        Shape::Sphere { radius } => {
            let zero = [Vec3::ZERO];
            let proxy_a = ShapeProxy {
                points: &zero,
                radius: *radius,
            };
            shape_cast(
                &proxy_a,
                &local_proxy,
                &Transform::IDENTITY,
                local_translation,
                max_fraction,
                can_encroach,
            )
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let centers = [
                Vec3::new(0.0, -half_height, 0.0),
                Vec3::new(0.0, *half_height, 0.0),
            ];
            let proxy_a = ShapeProxy {
                points: &centers,
                radius: *radius,
            };
            shape_cast(
                &proxy_a,
                &local_proxy,
                &Transform::IDENTITY,
                local_translation,
                max_fraction,
                can_encroach,
            )
        }
        Shape::Box { half_extents } => {
            let h = *half_extents;
            let corners = [
                Vec3::new(-h.x, -h.y, -h.z),
                Vec3::new(h.x, -h.y, -h.z),
                Vec3::new(h.x, h.y, -h.z),
                Vec3::new(-h.x, h.y, -h.z),
                Vec3::new(-h.x, -h.y, h.z),
                Vec3::new(h.x, -h.y, h.z),
                Vec3::new(h.x, h.y, h.z),
                Vec3::new(-h.x, h.y, h.z),
            ];
            let proxy_a = ShapeProxy {
                points: &corners,
                radius: 0.0,
            };
            shape_cast(
                &proxy_a,
                &local_proxy,
                &Transform::IDENTITY,
                local_translation,
                max_fraction,
                can_encroach,
            )
        }
        Shape::ConvexHull(hull) => {
            let proxy_a = ShapeProxy {
                points: hull.points(),
                radius: 0.0,
            };
            shape_cast(
                &proxy_a,
                &local_proxy,
                &Transform::IDENTITY,
                local_translation,
                max_fraction,
                can_encroach,
            )
        }
        Shape::Triangle { a, b, c } => {
            let tri_pts = [*a, *b, *c];
            let proxy_a = ShapeProxy {
                points: &tri_pts,
                radius: 0.0,
            };
            shape_cast(
                &proxy_a,
                &local_proxy,
                &Transform::IDENTITY,
                local_translation,
                max_fraction,
                can_encroach,
            )
        }
        Shape::TriangleMesh(mesh) => {
            let mut closest_out: Option<CastOutput> = None;
            let mut current_max = max_fraction;
            let vertices = mesh.vertices();
            for tri in mesh.triangles() {
                let tri_pts = [
                    vertices[tri.indices[0] as usize],
                    vertices[tri.indices[1] as usize],
                    vertices[tri.indices[2] as usize],
                ];
                let proxy_a = ShapeProxy {
                    points: &tri_pts,
                    radius: 0.0,
                };
                let out = shape_cast(
                    &proxy_a,
                    &local_proxy,
                    &Transform::IDENTITY,
                    local_translation,
                    current_max,
                    can_encroach,
                );
                if out.hit && out.fraction <= current_max {
                    current_max = out.fraction;
                    closest_out = Some(out);
                }
            }
            return closest_out.map(|out| CastOutput {
                point: shape_transform.apply(out.point),
                normal: shape_transform.q * out.normal,
                fraction: out.fraction,
                iterations: out.iterations,
                hit: true,
            });
        }
        Shape::Compound(comp) => {
            let mut closest_out: Option<CastOutput> = None;
            let mut current_max = max_fraction;
            for child in comp.children() {
                let child_xf = Transform::new(child.position, child.rotation);
                let child_world_xf = Transform::new(
                    shape_transform.apply(child_xf.p),
                    shape_transform.q * child_xf.q,
                );
                if let Some(out) = shape_cast_shape(
                    &child.shape,
                    &child_world_xf,
                    proxy,
                    translation,
                    current_max,
                ) {
                    if out.hit && out.fraction <= current_max {
                        current_max = out.fraction;
                        closest_out = Some(out);
                    }
                }
            }
            return closest_out;
        }
        Shape::HeightField(_) => return None,
    };

    if local_out.hit {
        Some(CastOutput {
            point: shape_transform.apply(local_out.point),
            normal: shape_transform.q * local_out.normal,
            fraction: local_out.fraction,
            iterations: local_out.iterations,
            hit: true,
        })
    } else {
        None
    }
}

fn shape_overlaps_proxy(
    shape: &Shape,
    shape_transform: &Transform,
    proxy: &ShapeProxy<'_>,
    proxy_transform: &Transform,
) -> bool {
    let rel_xf = shape_transform.inv_mul(proxy_transform);
    let mut cache = SimplexCache::default();
    match shape {
        Shape::Sphere { radius } => {
            let zero = [Vec3::ZERO];
            let shape_proxy = ShapeProxy {
                points: &zero,
                radius: *radius,
            };
            let out = shape_distance(&shape_proxy, proxy, &rel_xf, true, &mut cache);
            out.distance < OVERLAP_SLOP
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            let centers = [
                Vec3::new(0.0, -half_height, 0.0),
                Vec3::new(0.0, *half_height, 0.0),
            ];
            let shape_proxy = ShapeProxy {
                points: &centers,
                radius: *radius,
            };
            let out = shape_distance(&shape_proxy, proxy, &rel_xf, true, &mut cache);
            out.distance < OVERLAP_SLOP
        }
        Shape::Box { half_extents } => {
            let h = *half_extents;
            let corners = [
                Vec3::new(-h.x, -h.y, -h.z),
                Vec3::new(h.x, -h.y, -h.z),
                Vec3::new(h.x, h.y, -h.z),
                Vec3::new(-h.x, h.y, -h.z),
                Vec3::new(-h.x, -h.y, h.z),
                Vec3::new(h.x, -h.y, h.z),
                Vec3::new(h.x, h.y, h.z),
                Vec3::new(-h.x, h.y, h.z),
            ];
            let shape_proxy = ShapeProxy {
                points: &corners,
                radius: 0.0,
            };
            let out = shape_distance(&shape_proxy, proxy, &rel_xf, true, &mut cache);
            out.distance < OVERLAP_SLOP
        }
        Shape::ConvexHull(hull) => {
            let shape_proxy = ShapeProxy {
                points: hull.points(),
                radius: 0.0,
            };
            let out = shape_distance(&shape_proxy, proxy, &rel_xf, true, &mut cache);
            out.distance < OVERLAP_SLOP
        }
        Shape::Triangle { a, b, c } => {
            let pts = [*a, *b, *c];
            let shape_proxy = ShapeProxy {
                points: &pts,
                radius: 0.0,
            };
            let out = shape_distance(&shape_proxy, proxy, &rel_xf, true, &mut cache);
            out.distance < OVERLAP_SLOP
        }
        Shape::TriangleMesh(mesh) => {
            let vertices = mesh.vertices();
            for tri in mesh.triangles() {
                let pts = [
                    vertices[tri.indices[0] as usize],
                    vertices[tri.indices[1] as usize],
                    vertices[tri.indices[2] as usize],
                ];
                let shape_proxy = ShapeProxy {
                    points: &pts,
                    radius: 0.0,
                };
                let out = shape_distance(&shape_proxy, proxy, &rel_xf, true, &mut cache);
                if out.distance < OVERLAP_SLOP {
                    return true;
                }
            }
            false
        }
        Shape::Compound(comp) => {
            for child in comp.children() {
                let child_xf = Transform::new(child.position, child.rotation);
                let child_world_xf = Transform::new(
                    shape_transform.apply(child_xf.p),
                    shape_transform.q * child_xf.q,
                );
                if shape_overlaps_proxy(&child.shape, &child_world_xf, proxy, proxy_transform) {
                    return true;
                }
            }
            false
        }
        Shape::HeightField(_) => false,
    }
}

/// Evaluates a single awake contact on a worker thread during `update_contacts` (Box3D
/// `b3CollideTask`). Mesh and compound contacts that cannot be recycled and whose tight AABBs
/// overlap return `CollideOutcome3d::Deferred` so they can use the world's scratch buffers on
/// the main thread without per-worker heap allocations.
fn collide_contact_parallel(
    contact: &mut Contact3d,
    colliders: &[Option<ColliderSlot>],
    bodies: &[RigidBody],
    fast: &[bool],
    recycle_distance: f32,
    friction_callback: crate::material::FrictionCallback3d,
    restitution_callback: crate::material::RestitutionCallback3d,
) -> CollideOutcome3d {
    let (Some(slot_a), Some(slot_b)) = (
        colliders
            .get(contact.collider_a as usize)
            .and_then(Option::as_ref),
        colliders
            .get(contact.collider_b as usize)
            .and_then(Option::as_ref),
    ) else {
        return CollideOutcome3d::Unchanged { touching: false };
    };
    if !slot_a.fat_aabb.intersects(&slot_b.fat_aabb) {
        return CollideOutcome3d::Disjoint;
    }

    let was_touching = contact.was_touching;
    let (a, b) = (contact.body_a as usize, contact.body_b as usize);
    let (body_a, body_b) = (&bodies[a], &bodies[b]);
    let xf_a = Transform::new(body_a.position, body_a.rotation);
    let xf_b = Transform::new(body_b.position, body_b.rotation);

    let is_fast = fast.get(a).copied().unwrap_or(false) || fast.get(b).copied().unwrap_or(false);
    let tolerance = if contact.is_touching {
        recycle_distance
    } else {
        recycle_distance.min(SPECULATIVE_DISTANCE)
    };
    if !is_fast && recycle_distance > 0.0 && contact.relative_valid && contact.recycle {
        let same_rot_a = body_a.rotation == contact.cached_rotation_a;
        let same_rot_b = body_b.rotation == contact.cached_rotation_b;
        let xf = xf_a.inv_mul(&xf_b);
        let distance_squared = (xf.p - contact.cached_relative_position).length_squared();
        if same_rot_a && same_rot_b {
            if distance_squared < tolerance * tolerance {
                let dc = body_b.center_of_mass() - body_a.center_of_mass();
                for manifold in &mut contact.manifolds {
                    let normal = manifold.normal;
                    for p in &mut manifold.points {
                        p.separation =
                            p.base_separation + (dc + (p.anchor_b - p.anchor_a)).dot(normal);
                    }
                }
                contact.was_touching = contact.is_touching;
                return CollideOutcome3d::Recycled {
                    touching: contact.is_touching,
                };
            }
        } else {
            let angle_a = body_a.rotation.dot(contact.cached_rotation_a);
            let angle_b = body_b.rotation.dot(contact.cached_rotation_b);
            let angular_distance = (angle_a * angle_a).min(angle_b * angle_b);
            if angular_distance > CONTACT_RECYCLE_ANGULAR_DISTANCE
                && distance_squared < tolerance * tolerance
            {
                let slack = tolerance - distance_squared.sqrt();
                let extent = |body: &RigidBody| {
                    if body.body_type == BodyType::Static {
                        Vec3::ZERO
                    } else {
                        body.max_extent
                    }
                };
                let max_extent = extent(body_a).max(extent(body_b));
                let qr = inv_mul_quat(contact.cached_relative_rotation, xf.q);
                let arc = modified_cross(Vec3::new(qr.x, qr.y, qr.z).abs(), max_extent);
                if 4.0 * arc.length_squared() < slack * slack {
                    let rotation_a =
                        Mat3::from_quat(body_a.rotation * contact.cached_rotation_a.conjugate());
                    let rotation_b =
                        Mat3::from_quat(body_b.rotation * contact.cached_rotation_b.conjugate());
                    let dc = body_b.center_of_mass() - body_a.center_of_mass();
                    for manifold in &mut contact.manifolds {
                        let normal = manifold.normal;
                        for p in &mut manifold.points {
                            let r_a = rotation_a * p.anchor_a;
                            let r_b = rotation_b * p.anchor_b;
                            p.separation = p.base_separation + (dc + (r_b - r_a)).dot(normal);
                        }
                    }
                    contact.was_touching = contact.is_touching;
                    return CollideOutcome3d::Recycled {
                        touching: contact.is_touching,
                    };
                }
            }
        }
    }

    let tight_overlap = slot_a.aabb.intersects(&slot_b.aabb);
    if tight_overlap
        && (slot_a.collider.shape.is_triangles()
            || slot_b.collider.shape.is_triangles()
            || matches!(slot_a.collider.shape, Shape::Compound(_))
            || matches!(slot_b.collider.shape, Shape::Compound(_)))
    {
        return CollideOutcome3d::Deferred;
    }

    contact.cached_rotation_a = body_a.rotation;
    contact.cached_rotation_b = body_b.rotation;
    let relative = xf_a.inv_mul(&xf_b);
    contact.cached_relative_position = relative.p;
    contact.cached_relative_rotation = relative.q;
    contact.relative_valid = true;
    if !tight_overlap {
        contact.separate();
        contact.was_touching = false;
        return if was_touching {
            CollideOutcome3d::Stopped
        } else {
            CollideOutcome3d::Unchanged { touching: false }
        };
    }

    let (center_a, center_b) = (body_a.center_of_mass(), body_b.center_of_mass());
    let set_anchors = |manifold: &mut ContactManifold| {
        for p in &mut manifold.points {
            p.anchor_a = p.point - center_a;
            p.anchor_b = p.point - center_b;
        }
    };
    let collision_a = slot_a.collision_body(body_a);
    let collision_b = slot_b.collision_body(body_b);
    match collide_shapes_cached(collision_a, collision_b, &mut contact.cache) {
        Some(mut manifold) => {
            manifold.friction = (friction_callback)(
                slot_a.collider.friction,
                slot_a.collider.user_material_id,
                slot_b.collider.friction,
                slot_b.collider.user_material_id,
            );
            manifold.restitution = (restitution_callback)(
                slot_a.collider.restitution,
                slot_a.collider.user_material_id,
                slot_b.collider.restitution,
                slot_b.collider.user_material_id,
            );
            let (_, rot_a) = slot_a.pose(body_a);
            let (_, rot_b) = slot_b.pose(body_b);
            let vt_a = rot_a * slot_a.collider.tangent_velocity;
            let vt_b = rot_b * slot_b.collider.tangent_velocity;
            manifold.tangent_velocity = vt_a - vt_b;
            set_anchors(&mut manifold);
            contact.update(manifold);
        }
        None => contact.separate(),
    }
    let is_touching = contact.is_touching;
    contact.was_touching = is_touching;
    if is_touching && !was_touching {
        CollideOutcome3d::Started
    } else if !is_touching && was_touching {
        CollideOutcome3d::Stopped
    } else {
        CollideOutcome3d::Unchanged {
            touching: is_touching,
        }
    }
}
