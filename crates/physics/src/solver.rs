//! Box3D's Soft Step solver (`solver.c`): sub-stepped soft contacts and joints with a relax
//! pass, a restitution pass, and warm starting.
//!
//! A step prepares a contact constraint per touching manifold, then runs each substep's stages
//! in Box3D's order: integrate the moving bodies' velocities (gravity, forces, damping,
//! gyroscopic torque), warm start the joints and contacts, solve them with bias (contacts push
//! apart softly), integrate positions, and relax them without bias. After the substeps, the
//! restitution passes add bounce and the impulses are stored in the manifolds and joints for the
//! next step.
//!
//! Bodies move during the substeps: each keeps its pose at the start of the step and its motion
//! since (Box3D's body state deltas), and its pose is the start pose moved by that motion
//! (Box3D's finalize formula). Contacts measure their separation from the motion; joints read
//! the pose.

use crate::body::{BodyType, RigidBody, MAX_ROTATION};
use crate::contact_solver::{perp, BodyMotion3d, ContactConstraint3d, Softness3d};
use crate::joint::{Joint, Substep};
use crate::narrowphase::ContactManifold;
use korppi_math::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Box3D's contact tangents for a unit normal `n`: `t1 = perp(n)` and `t2 = t1 x n`.
#[must_use]
pub fn compute_orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let t1 = perp(n);
    (t1, t1.cross(n))
}

/// Configuration parameters for the 3D Soft-Step constraint solver adapting Erin Catto's Box3D.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig3d {
    /// Number of sub-steps per frame tick (default: 4).
    pub sub_steps: usize,
    /// Contact stiffness in Hertz (Box3D `contactHertz`, default 30). Clamped to an eighth of
    /// the substep rate; contacts with static bodies are twice as stiff with half the damping.
    pub hertz: f32,
    /// Contact damping ratio (Box3D `contactDampingRatio`, default 10).
    pub damping_ratio: f32,
    /// Largest speed at which overlapping bodies are pushed apart, in m/s (Box3D
    /// `contactSpeed`, default 3).
    pub contact_speed: f32,
    /// Default coefficient of friction (default: 0.5).
    pub friction: f32,
    /// Default coefficient of restitution (default: 0.0).
    pub restitution: f32,
    /// Largest linear speed of a body, in m/s (Box3D `maximumLinearSpeed`, default 400).
    #[serde(default = "default_max_linear_speed")]
    pub max_linear_speed: f32,
    /// Approach speed below which contacts do not bounce, in m/s (Box3D
    /// `restitutionThreshold`, default 1).
    #[serde(default = "default_restitution_threshold")]
    pub restitution_threshold: f32,
    /// Passes of the restitution solver after the substeps (Box3D `restitutionIterations`,
    /// default 2).
    #[serde(default = "default_restitution_iterations")]
    pub restitution_iterations: usize,
    /// Whether fast bodies are swept for continuous collision (Box3D `enableContinuous`,
    /// default true).
    #[serde(default = "default_enable_continuous")]
    pub enable_continuous: bool,
    /// How far, in metres, a touching contact's bodies may move relative to each other before
    /// its manifold is rebuilt instead of recycled (Box3D `contactRecycleDistance`, default 10
    /// linear slops; zero turns recycling off).
    #[serde(default = "default_contact_recycle_distance")]
    pub contact_recycle_distance: f32,
    /// Approach speed threshold in m/s for generating a contact hit event (Box3D
    /// `hitEventThreshold`, default 1.0).
    #[serde(default = "default_hit_event_threshold")]
    pub hit_event_threshold: f32,
    /// Whether bodies are allowed to sleep when resting (Box3D `enableSleep`, default true).
    #[serde(default = "default_enable_sleep")]
    pub enable_sleep: bool,
    /// Whether impulse warm-starting is enabled across time steps (Box3D `enableWarmStarting`, default true).
    #[serde(default = "default_enable_warm_starting")]
    pub enable_warm_starting: bool,
}

/// Box3D's default contact recycling distance, ten linear slops (`B3_CONTACT_RECYCLE_DISTANCE`).
pub const CONTACT_RECYCLE_DISTANCE: f32 = 10.0 * crate::contact_solver::LINEAR_SLOP;

fn default_contact_recycle_distance() -> f32 {
    CONTACT_RECYCLE_DISTANCE
}

/// Box3D's default approach speed threshold for contact hit events (1.0 m/s).
pub const DEFAULT_HIT_EVENT_THRESHOLD: f32 = 1.0;

fn default_hit_event_threshold() -> f32 {
    DEFAULT_HIT_EVENT_THRESHOLD
}

/// Box3D's default largest linear speed of a body, 400 m/s, faster than sound.
pub const MAX_LINEAR_SPEED: f32 = 400.0;

fn default_max_linear_speed() -> f32 {
    MAX_LINEAR_SPEED
}

fn default_restitution_threshold() -> f32 {
    1.0
}

fn default_restitution_iterations() -> usize {
    2
}

fn default_enable_continuous() -> bool {
    true
}

fn default_enable_sleep() -> bool {
    true
}

fn default_enable_warm_starting() -> bool {
    true
}

impl Default for SolverConfig3d {
    fn default() -> Self {
        Self {
            sub_steps: 4,
            hertz: 30.0,
            damping_ratio: 10.0,
            contact_speed: 3.0,
            friction: 0.5,
            restitution: 0.0,
            max_linear_speed: MAX_LINEAR_SPEED,
            restitution_threshold: default_restitution_threshold(),
            restitution_iterations: default_restitution_iterations(),
            enable_continuous: true,
            contact_recycle_distance: CONTACT_RECYCLE_DISTANCE,
            hit_event_threshold: DEFAULT_HIT_EVENT_THRESHOLD,
            enable_sleep: true,
            enable_warm_starting: true,
        }
    }
}

impl SolverConfig3d {
    /// Contact softness for substep `h`: between movable bodies, and against a static body at
    /// twice the stiffness and half the damping (Box3D `contactSoftness`, `staticSoftness`). The
    /// stiffness is capped at an eighth of the substep rate.
    #[must_use]
    pub fn contact_softness(&self, h: f32) -> (Softness3d, Softness3d) {
        let hertz = self.hertz.min(0.125 / h);
        (
            Softness3d::new(hertz, self.damping_ratio, h),
            Softness3d::new(2.0 * hertz, 0.5 * self.damping_ratio, h),
        )
    }
}

/// Sentinel sync bits telling worker threads to exit the solver stage loop.
const SENTINEL: u32 = u32::MAX;

/// Minimum items per parallel block (Box3D `minBodiesPerBlock`, `minContactsPerBlock`,
/// `minJointsPerBlock`).
const MIN_BODIES_PER_BLOCK: usize = 64;
const MIN_CONTACTS_PER_BLOCK: usize = 16;
const MIN_JOINTS_PER_BLOCK: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageKind {
    PrepareJoints,
    PrepareWideContacts,
    PrepareScalarContacts,
    IntegrateVelocities,
    WarmStart,
    Solve,
    IntegratePositions,
    Relax,
    FinalizePositions,
    Restitution,
    StoreWideImpulses,
    StoreScalarImpulses,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Items,
    GraphJoints,
    GraphWideContacts,
    GraphScalarContacts,
}

#[derive(Debug)]
struct SyncBlock {
    start: u32,
    count: u32,
    kind: BlockKind,
    sync: std::sync::atomic::AtomicU32,
}

#[derive(Debug)]
struct Stage {
    kind: StageKind,
    first_block: u32,
    block_count: u32,
    color: u32,
    completion: std::sync::atomic::AtomicU32,
}

#[derive(Debug, Clone, Copy, Default)]
struct ActiveColor3d {
    joint_start: u32,
    joint_count: u32,
    wide_start: u32,
    wide_count: u32,
    scalar_start: u32,
    scalar_count: u32,
}

/// Buffers the solver reuses from step to step, so that stepping does not allocate once they
/// have grown.
#[derive(Debug, Default)]
pub struct SolverScratch3d {
    start_positions: Vec<Vec3>,
    start_rotations: Vec<Quat>,
    inv_inertia_world: Vec<Mat3>,
    motions: Vec<BodyMotion3d>,
    contacts: Vec<ContactConstraint3d>,
    contact_manifolds: Vec<usize>,
    contact_refs: Vec<(u32, usize)>,
    scalar_contact_ranges: Vec<(u32, u32)>,
    wide_contacts: Vec<crate::solver_wide::WideContactConstraint3d>,
    wide_slots: Vec<[u32; crate::solver_wide::WIDE]>,
    overflow_contacts: Vec<ContactConstraint3d>,
    overflow_refs: Vec<(u32, usize)>,
    colored_joints: Vec<u32>,
    colors: Vec<ActiveColor3d>,
    blocks: Vec<SyncBlock>,
    stages: Vec<Stage>,
}

impl Clone for SolverScratch3d {
    fn clone(&self) -> Self {
        Self {
            start_positions: self.start_positions.clone(),
            start_rotations: self.start_rotations.clone(),
            inv_inertia_world: self.inv_inertia_world.clone(),
            motions: self.motions.clone(),
            ..Self::default()
        }
    }
}

impl SolverScratch3d {
    /// Body `index`'s centre of mass and rotation at the start of the last solved step.
    pub(crate) fn start_pose(&self, index: usize) -> (Vec3, Quat) {
        (self.start_positions[index], self.start_rotations[index])
    }

    /// Body `index`'s motion over the last solved step.
    pub(crate) fn motion(&self, index: usize) -> BodyMotion3d {
        self.motions[index]
    }
}

/// Steps bodies, contact manifolds and joints by `dt` under `gravity` with Box3D's Soft Step.
pub fn solve_contacts(
    bodies: &mut [RigidBody],
    manifolds: &mut [ContactManifold],
    joints: &mut [Option<Joint>],
    gravity: Vec3,
    dt: f32,
    config: &SolverConfig3d,
) {
    solve_contacts_with_buffers(
        bodies,
        manifolds,
        joints,
        gravity,
        dt,
        config,
        &mut SolverScratch3d::default(),
    );
}

/// Box3D `b3IntegrateRotation`: `q1` turned by the small rotation vector `delta_rotation`.
fn integrate_rotation(q1: Quat, delta_rotation: Vec3) -> Quat {
    let half = delta_rotation * 0.5;
    let qd = Quat::from_xyzw(half.x, half.y, half.z, 0.0) * q1;
    Quat::from_xyzw(q1.x + qd.x, q1.y + qd.y, q1.z + qd.z, q1.w + qd.w).normalize()
}

/// Whether a constraint between these bodies is solved: one of them moves and one is dynamic.
fn is_active(a: &RigidBody, b: &RigidBody) -> bool {
    (a.is_moving() || b.is_moving())
        && (a.body_type == BodyType::Dynamic || b.body_type == BodyType::Dynamic)
}

/// Whether `joint` joins two distinct bodies of `bodies` and is solved this step.
fn is_active_joint(bodies: &[RigidBody], joint: &Joint) -> bool {
    let (a, b) = (joint.body_a() as usize, joint.body_b() as usize);
    a != b && a < bodies.len() && b < bodies.len() && is_active(&bodies[a], &bodies[b])
}

/// [`solve_contacts`] with buffers reused from step to step, so that it does not allocate once
/// they have grown.
pub fn solve_contacts_with_buffers(
    bodies: &mut [RigidBody],
    manifolds: &mut [ContactManifold],
    joints: &mut [Option<Joint>],
    gravity: Vec3,
    dt: f32,
    config: &SolverConfig3d,
    scratch: &mut SolverScratch3d,
) {
    if bodies.is_empty() || dt <= 0.0 {
        return;
    }
    let sub_steps = config.sub_steps.max(1);
    let h = dt / sub_steps as f32;
    let inv_h = sub_steps as f32 / dt;
    let max_angular_speed = MAX_ROTATION / dt;
    let softness = config.contact_softness(h);

    let SolverScratch3d {
        start_positions,
        start_rotations,
        inv_inertia_world,
        motions,
        contacts,
        contact_manifolds,
        ..
    } = scratch;
    // Bodies move and turn about their centres of mass.
    start_positions.clear();
    start_positions.extend(bodies.iter().map(RigidBody::center_of_mass));
    start_rotations.clear();
    start_rotations.extend(bodies.iter().map(|b| b.rotation));
    inv_inertia_world.clear();
    inv_inertia_world.extend(bodies.iter().map(RigidBody::world_inv_inertia));
    motions.clear();
    motions.resize(bodies.len(), BodyMotion3d::NONE);

    // Prepare contacts (Box3D `b3PrepareContacts_Convex`)
    contacts.clear();
    contact_manifolds.clear();
    for (index, manifold) in manifolds.iter().enumerate() {
        let (a, b) = (manifold.body_a as usize, manifold.body_b as usize);
        if a == b || a >= bodies.len() || b >= bodies.len() || !is_active(&bodies[a], &bodies[b]) {
            continue;
        }
        contacts.push(ContactConstraint3d::prepare(
            manifold,
            bodies,
            inv_inertia_world,
            softness,
            config.enable_warm_starting,
        ));
        contact_manifolds.push(index);
    }

    // Prepare joints (Box3D `b3PrepareJointsTask`)
    for joint in joints.iter_mut().flatten() {
        if is_active_joint(bodies, joint) {
            joint.prepare(bodies, inv_inertia_world, h);
        }
    }
    let substep = Substep { h, inv_h };

    for _ in 0..sub_steps {
        // Integrate velocities (Box3D `b3IntegrateVelocitiesTask`)
        for (body, inv_inertia) in bodies.iter_mut().zip(inv_inertia_world.iter()) {
            if body.is_moving() {
                let q = body.rotation;
                body.integrate_velocities(gravity, inv_inertia, Quat::IDENTITY, q, h);
            }
        }

        // Warm start
        if config.enable_warm_starting {
            for joint in joints.iter().flatten() {
                if is_active_joint(bodies, joint) {
                    joint.warm_start_step(bodies, motions);
                }
            }
            for contact in contacts.iter() {
                contact.warm_start(bodies);
            }
        }

        // Solve with bias
        for joint in joints.iter_mut().flatten() {
            if is_active_joint(bodies, joint) {
                joint.solve_step(bodies, motions, substep, true);
            }
        }
        for contact in contacts.iter_mut() {
            contact.push(bodies, motions, inv_h, config.contact_speed);
        }

        // Integrate positions of every awake body after the motion locks and speed caps (Box3D
        // `b3IntegratePositionsTask`); the pose is the start pose moved by the motion.
        for (index, body) in bodies.iter_mut().enumerate() {
            if !body.is_moving() {
                continue;
            }
            let (mut v, mut w) = body
                .motion_locks
                .apply(body.linear_velocity, body.angular_velocity);
            if v.dot(v) > config.max_linear_speed * config.max_linear_speed {
                v *= config.max_linear_speed / v.length();
            }
            if w.dot(w) > max_angular_speed * max_angular_speed && !body.allow_fast_rotation {
                w *= max_angular_speed / w.length();
            }
            body.linear_velocity = v;
            body.angular_velocity = w;
            let motion = &mut motions[index];
            motion.delta_position += v * h;
            motion.delta_rotation = integrate_rotation(motion.delta_rotation, w * h);
            let center = start_positions[index] + motion.delta_position;
            body.rotation = (motion.delta_rotation * start_rotations[index]).normalize();
            body.position = center - body.rotation * body.local_center;
        }

        // Relax without bias
        for joint in joints.iter_mut().flatten() {
            if is_active_joint(bodies, joint) {
                joint.solve_step(bodies, motions, substep, false);
            }
        }
        for contact in contacts.iter_mut() {
            contact.relax(bodies, motions, inv_h);
        }
    }

    // Restitution (Box3D `b3ApplyRestitution_Convex`)
    if contacts.iter().any(ContactConstraint3d::has_restitution) {
        for _ in 0..config.restitution_iterations {
            for contact in contacts.iter_mut().filter(|c| c.has_restitution()) {
                contact.restitute(bodies, motions, inv_h, config.restitution_threshold);
            }
        }
    }

    // Store impulses
    for (contact, &index) in contacts.iter().zip(contact_manifolds.iter()) {
        contact.store(&mut manifolds[index]);
    }
}

fn block_dim(items: usize, min_size: usize, max_blocks: usize) -> (usize, usize) {
    if items == 0 {
        return (0, 0);
    }
    let size = if items <= min_size * max_blocks {
        min_size
    } else {
        items.div_ceil(max_blocks)
    };
    (size, items.div_ceil(size))
}

fn push_blocks(
    blocks: &mut Vec<SyncBlock>,
    items: usize,
    min_size: usize,
    max_blocks: usize,
    kind: BlockKind,
) -> (u32, u32) {
    let first = blocks.len() as u32;
    let (size, count) = block_dim(items, min_size, max_blocks);
    for i in 0..count {
        let start = i * size;
        blocks.push(SyncBlock {
            start: start as u32,
            count: (size.min(items - start)) as u32,
            kind,
            sync: std::sync::atomic::AtomicU32::new(0),
        });
    }
    (first, count as u32)
}

fn worker_start_index(worker: usize, block_count: usize, worker_count: usize) -> Option<usize> {
    if block_count <= worker_count {
        return (worker < block_count).then_some(worker);
    }
    let per_worker = block_count / worker_count;
    let remainder = block_count - per_worker * worker_count;
    Some(per_worker * worker + remainder.min(worker))
}

struct StageDone<'a> {
    completion: &'a std::sync::atomic::AtomicU32,
    count: u32,
}

impl Drop for StageDone<'_> {
    fn drop(&mut self) {
        if self.count > 0 {
            self.completion
                .fetch_add(self.count, std::sync::atomic::Ordering::Release);
        }
    }
}

struct Finish<'a>(&'a std::sync::atomic::AtomicU32);

impl Drop for Finish<'_> {
    fn drop(&mut self) {
        self.0.store(SENTINEL, std::sync::atomic::Ordering::Release);
    }
}

struct SolveContext3d<'a> {
    bodies: crate::parallel::Shared<'a, RigidBody>,
    motions: crate::parallel::Shared<'a, BodyMotion3d>,
    start_positions: &'a [Vec3],
    start_rotations: &'a [Quat],
    inv_inertia_world: &'a [Mat3],
    awake: &'a [u32],
    graph: &'a crate::constraint_graph::ConstraintGraph3d,
    colors: &'a [ActiveColor3d],
    colored_joints: &'a [u32],
    wide_slots: &'a [[u32; crate::solver_wide::WIDE]],
    wide_contacts: crate::parallel::Shared<'a, crate::solver_wide::WideContactConstraint3d>,
    scalar_contacts: crate::parallel::Shared<'a, ContactConstraint3d>,
    scalar_refs: &'a [(u32, usize)],
    scalar_contact_ranges: &'a [(u32, u32)],
    overflow_contacts: crate::parallel::Shared<'a, ContactConstraint3d>,
    overflow_refs: &'a [(u32, usize)],
    contact_slots: crate::parallel::Shared<'a, Option<crate::contact::Contact3d>>,
    joints: crate::parallel::Shared<'a, Option<Joint>>,
    stages: &'a [Stage],
    blocks: &'a [SyncBlock],
    gravity: Vec3,
    h: f32,
    inv_h: f32,
    sub_steps: usize,
    max_linear_speed: f32,
    max_angular_speed: f32,
    softness: (Softness3d, Softness3d),
    contact_speed: f32,
    restitution_threshold: f32,
    restitution_iterations: usize,
    enable_warm_starting: bool,
    any_restitution: std::sync::atomic::AtomicBool,
    sync_bits: std::sync::atomic::AtomicU32,
    main_claimed: std::sync::atomic::AtomicBool,
    next_worker: std::sync::atomic::AtomicU32,
    worker_count: usize,
}

impl SolveContext3d<'_> {
    fn task(&self) {
        use std::sync::atomic::Ordering;
        if self
            .main_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.orchestrate();
        } else {
            let worker = self.next_worker.fetch_add(1, Ordering::Relaxed) as usize;
            self.work(worker % self.worker_count);
        }
    }

    fn work(&self, worker: usize) {
        use std::sync::atomic::Ordering;
        let mut last = 0;
        loop {
            let mut bits = last;
            korppi_core::spin_until(|| {
                bits = self.sync_bits.load(Ordering::Acquire);
                bits != last
            });
            if bits == SENTINEL {
                return;
            }
            let stage = &self.stages[(bits & 0xFFFF) as usize];
            let sync = bits >> 16;
            self.execute_stage(stage, sync - 1, sync, worker);
            last = bits;
        }
    }

    fn execute_stage(&self, stage: &Stage, previous_sync: u32, sync: u32, worker: usize) {
        use std::sync::atomic::Ordering;
        let count = stage.block_count as usize;
        let Some(start) = worker_start_index(worker, count, self.worker_count) else {
            return;
        };
        let mut done = StageDone {
            completion: &stage.completion,
            count: 0,
        };
        let mut index = start;
        for _ in 0..count {
            let block = &self.blocks[stage.first_block as usize + index];
            if block.sync.load(Ordering::Relaxed) == previous_sync
                && block
                    .sync
                    .compare_exchange(previous_sync, sync, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                done.count += 1;
                self.execute_block(stage, block);
            }
            index = if index + 1 == count { 0 } else { index + 1 };
        }
    }

    fn main_stage(&self, index: usize, sync: u32) {
        use std::sync::atomic::Ordering;
        let stage = &self.stages[index];
        let count = stage.block_count;
        if count == 0 {
            return;
        }
        if count == 1 || self.worker_count == 1 {
            for block in &self.blocks[stage.first_block as usize..][..count as usize] {
                self.execute_block(stage, block);
            }
            return;
        }
        self.sync_bits
            .store(sync << 16 | index as u32, Ordering::Release);
        self.execute_stage(stage, sync - 1, sync, 0);
        while stage.completion.load(Ordering::Acquire) != count {
            std::hint::spin_loop();
        }
        stage.completion.store(0, Ordering::Relaxed);
    }

    fn orchestrate(&self) {
        use std::sync::atomic::Ordering;
        let _finish = Finish(&self.sync_bits);
        let colors = self.colors.len();
        let (mut body_sync, mut graph_sync) = (1, 1);

        // 0: PrepareJoints, 1: PrepareWideContacts, 2: PrepareScalarContacts
        self.main_stage(0, 1);
        self.main_stage(1, 1);
        self.main_stage(2, 1);

        let overflow = self.graph.color(self.graph.overflow_index());
        for &joint_idx in &overflow.joint_indices {
            // SAFETY: no block runs between stages, and overflow joints are in no block.
            unsafe { self.prepare_joint(joint_idx as u32) };
        }

        let mut stage = 3;
        for _ in 0..self.sub_steps {
            let mut index = stage;
            // IntegrateVelocities
            self.main_stage(index, body_sync);
            index += 1;
            body_sync += 1;

            // WarmStart
            if self.enable_warm_starting {
                for _ in 0..colors {
                    self.main_stage(index, graph_sync);
                    index += 1;
                }
                graph_sync += 1;
                self.overflow(StageKind::WarmStart);
            } else {
                index += colors;
            }

            // Solve with bias
            for _ in 0..colors {
                self.main_stage(index, graph_sync);
                index += 1;
            }
            graph_sync += 1;
            self.overflow(StageKind::Solve);

            // IntegratePositions
            self.main_stage(index, body_sync);
            index += 1;
            body_sync += 1;

            // Relax without bias
            for _ in 0..colors {
                self.main_stage(index, graph_sync);
                index += 1;
            }
            graph_sync += 1;
            self.overflow(StageKind::Relax);
        }

        stage += 1 + 2 * colors + 1 + colors;
        // FinalizePositions
        self.main_stage(stage, body_sync);
        stage += 1;

        // Restitution
        if self.any_restitution.load(Ordering::Relaxed) {
            for _ in 0..self.restitution_iterations {
                for color in 0..colors {
                    self.main_stage(stage + color, graph_sync);
                }
                graph_sync += 1;
                self.overflow(StageKind::Restitution);
            }
        }
        stage += colors;

        // StoreWideImpulses (second use of wide_blocks) and StoreScalarImpulses (second use of scalar_blocks)
        self.main_stage(stage, 2);
        self.main_stage(stage + 1, 2);
        self.overflow(StageKind::StoreWideImpulses);
    }

    fn execute_block(&self, stage: &Stage, block: &SyncBlock) {
        let range = block.start as usize..(block.start + block.count) as usize;
        let color = self.colors.get(stage.color as usize);
        match (stage.kind, block.kind) {
            (StageKind::PrepareJoints, _) => {
                for i in range {
                    // SAFETY: each colored joint is in one block of this stage, and bodies are read-only.
                    unsafe { self.prepare_joint(self.colored_joints[i]) };
                }
            }
            (StageKind::PrepareWideContacts, _) => {
                for i in range {
                    // SAFETY: each wide constraint is in one block of this stage, and bodies/contacts are read-only.
                    unsafe { self.prepare_wide(i) };
                }
            }
            (StageKind::PrepareScalarContacts, _) => {
                for i in range {
                    // SAFETY: each scalar constraint is in one block of this stage, and bodies/contacts are read-only.
                    unsafe { self.prepare_scalar(i) };
                }
            }
            (StageKind::IntegrateVelocities, _) => {
                for i in range {
                    // SAFETY: each awake body is in one block of this stage.
                    unsafe { self.integrate_velocity(i) };
                }
            }
            (StageKind::IntegratePositions, _) => {
                for i in range {
                    // SAFETY: each awake body is in one block of this stage.
                    unsafe { self.integrate_position(i) };
                }
            }
            (StageKind::FinalizePositions, _) => {
                for i in range {
                    // SAFETY: each awake body is in one block of this stage.
                    unsafe { self.finalize_position(i) };
                }
            }
            (StageKind::StoreWideImpulses, _) => {
                for i in range {
                    // SAFETY: each wide constraint is in one block of this stage, and contact slots are disjoint.
                    unsafe { self.store_wide(i) };
                }
            }
            (StageKind::StoreScalarImpulses, _) => {
                for i in range {
                    // SAFETY: each scalar contact range is in one block of this stage, and contact slots are disjoint.
                    unsafe { self.store_scalar_range(i) };
                }
            }
            (kind, BlockKind::GraphJoints) => {
                let color = color.expect("graph stage");
                for i in range {
                    let joint = self.colored_joints[color.joint_start as usize + i];
                    // SAFETY: within a color, no two constraints share a dynamic body.
                    unsafe { self.solve_joint(kind, joint) };
                }
            }
            (kind, BlockKind::GraphWideContacts) => {
                let color = color.expect("graph stage");
                for i in range {
                    // SAFETY: within a color, no two constraints share a dynamic body.
                    unsafe { self.solve_wide(kind, color.wide_start as usize + i) };
                }
            }
            (kind, BlockKind::GraphScalarContacts) => {
                let color = color.expect("graph stage");
                for i in range {
                    // SAFETY: within a color, no two constraints share a dynamic body, and all manifolds
                    // of a multi-manifold contact are in the same range `color.scalar_start + i`.
                    unsafe { self.solve_scalar_range(kind, color.scalar_start as usize + i) };
                }
            }
            (_, BlockKind::Items) => unreachable!("graph stages use graph blocks"),
        }
    }

    fn overflow(&self, kind: StageKind) {
        let color = self.graph.color(self.graph.overflow_index());
        if matches!(
            kind,
            StageKind::WarmStart | StageKind::Solve | StageKind::Relax
        ) {
            for &joint_idx in &color.joint_indices {
                // SAFETY: orchestrator runs overflow between stages while no worker block is active.
                unsafe { self.solve_joint(kind, joint_idx as u32) };
            }
        }
        for (i, &(slot, m_idx)) in self.overflow_refs.iter().enumerate() {
            // SAFETY: only orchestrator touches overflow constraints between stages.
            let constraint = unsafe { self.overflow_contacts.get_mut(i) };
            if kind == StageKind::StoreWideImpulses {
                // SAFETY: overflow contact slot is accessed exclusively by orchestrator.
                let contact_opt = unsafe { self.contact_slots.get_mut(slot as usize) };
                if let Some(manifold) = contact_opt
                    .as_mut()
                    .and_then(|c| c.manifolds.get_mut(m_idx))
                {
                    constraint.store(manifold);
                }
                continue;
            }
            // SAFETY: only orchestrator runs during overflow.
            unsafe {
                match kind {
                    StageKind::WarmStart => constraint.warm_start_shared(&self.bodies),
                    StageKind::Solve => constraint.push_shared(
                        &self.bodies,
                        &self.motions,
                        self.inv_h,
                        self.contact_speed,
                    ),
                    StageKind::Relax => {
                        constraint.relax_shared(&self.bodies, &self.motions, self.inv_h)
                    }
                    StageKind::Restitution => {
                        if constraint.has_restitution() {
                            constraint.restitute_shared(
                                &self.bodies,
                                &self.motions,
                                self.inv_h,
                                self.restitution_threshold,
                            );
                        }
                    }
                    _ => unreachable!("overflow has no {kind:?} stage"),
                }
            }
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.joints[joint_idx]` during the call, and `self.bodies` must
    /// be read-only.
    unsafe fn prepare_joint(&self, joint_idx: u32) {
        // SAFETY: forwarded from caller.
        let joint_opt = unsafe { self.joints.get_mut(joint_idx as usize) };
        if let Some(joint) = joint_opt.as_mut() {
            // SAFETY: bodies are read-only during PrepareJoints.
            let bodies_slice =
                unsafe { std::slice::from_raw_parts(self.bodies.get(0), self.bodies.len()) };
            if is_active_joint(bodies_slice, joint) {
                joint.prepare(bodies_slice, self.inv_inertia_world, self.h);
            }
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.wide_contacts[index]` during the call, and `self.bodies`,
    /// `self.motions`, and `self.contact_slots` must be read-only.
    unsafe fn prepare_wide(&self, index: usize) {
        // SAFETY: forwarded from caller.
        let (constraint, any_rest) = unsafe {
            crate::solver_wide::WideContactConstraint3d::prepare(
                self.wide_slots[index],
                &self.contact_slots,
                &self.bodies,
                &self.motions,
                self.inv_inertia_world,
                self.softness,
                self.enable_warm_starting,
            )
        };
        // SAFETY: forwarded from caller.
        unsafe { *self.wide_contacts.get_mut(index) = constraint };
        if any_rest {
            self.any_restitution
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// # Safety
    ///
    /// No other thread may access the scalar constraints in `self.scalar_contact_ranges[range_idx]`
    /// during the call, and `self.bodies` and `self.contact_slots` must be read-only.
    unsafe fn prepare_scalar(&self, range_idx: usize) {
        let (start, end) = self.scalar_contact_ranges[range_idx];
        // SAFETY: bodies are read-only during PrepareScalarContacts.
        let bodies_slice =
            unsafe { std::slice::from_raw_parts(self.bodies.get(0), self.bodies.len()) };
        let mut any_rest = false;
        for c_idx in start as usize..end as usize {
            let (slot, m_idx) = self.scalar_refs[c_idx];
            // SAFETY: contact_slots are read-only during PrepareScalarContacts.
            let contact = unsafe { self.contact_slots.get(slot as usize) }
                .as_ref()
                .expect("live contact");
            let manifold = &contact.manifolds[m_idx];
            let constraint = ContactConstraint3d::prepare(
                manifold,
                bodies_slice,
                self.inv_inertia_world,
                self.softness,
                self.enable_warm_starting,
            );
            any_rest |= constraint.has_restitution();
            // SAFETY: each c_idx belongs to exactly one range_idx.
            unsafe { *self.scalar_contacts.get_mut(c_idx) = constraint };
        }
        if any_rest {
            self.any_restitution
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.joints[joint_idx]` or its dynamic bodies during the call.
    unsafe fn solve_joint(&self, kind: StageKind, joint_idx: u32) {
        let substep = Substep {
            h: self.h,
            inv_h: self.inv_h,
        };
        match kind {
            StageKind::WarmStart => {
                // SAFETY: forwarded from caller.
                let joint_opt = unsafe { self.joints.get(joint_idx as usize) };
                if let Some(joint) = joint_opt.as_ref() {
                    // SAFETY: forwarded from caller.
                    unsafe { joint.warm_start_step_shared(&self.bodies, &self.motions) };
                }
            }
            StageKind::Solve => {
                // SAFETY: forwarded from caller.
                let joint_opt = unsafe { self.joints.get_mut(joint_idx as usize) };
                if let Some(joint) = joint_opt.as_mut() {
                    // SAFETY: forwarded from caller.
                    unsafe {
                        joint.solve_step_shared(&self.bodies, &self.motions, substep, true);
                    }
                }
            }
            StageKind::Relax => {
                // SAFETY: forwarded from caller.
                let joint_opt = unsafe { self.joints.get_mut(joint_idx as usize) };
                if let Some(joint) = joint_opt.as_mut() {
                    // SAFETY: forwarded from caller.
                    unsafe {
                        joint.solve_step_shared(&self.bodies, &self.motions, substep, false);
                    }
                }
            }
            StageKind::Restitution => {}
            _ => unreachable!("joints have no {kind:?} graph stage"),
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.wide_contacts[index]` or its dynamic bodies during the call.
    unsafe fn solve_wide(&self, kind: StageKind, index: usize) {
        // SAFETY: forwarded from caller.
        let constraint = unsafe { self.wide_contacts.get_mut(index) };
        // SAFETY: forwarded from caller.
        unsafe {
            match kind {
                StageKind::WarmStart => constraint.warm_start(&self.bodies, &self.motions),
                StageKind::Solve => {
                    constraint.push(&self.bodies, &self.motions, self.inv_h, self.contact_speed)
                }
                StageKind::Relax => constraint.relax(&self.bodies, &self.motions, self.inv_h),
                StageKind::Restitution => constraint.restitute(
                    &self.bodies,
                    &self.motions,
                    self.inv_h,
                    self.restitution_threshold,
                ),
                _ => unreachable!("wide contacts have no {kind:?} graph stage"),
            }
        }
    }

    /// # Safety
    ///
    /// No other thread may access the scalar constraints in `self.scalar_contact_ranges[range_idx]`
    /// or their dynamic bodies during the call.
    unsafe fn solve_scalar_range(&self, kind: StageKind, range_idx: usize) {
        let (start, end) = self.scalar_contact_ranges[range_idx];
        for c_idx in start as usize..end as usize {
            // SAFETY: forwarded from caller; all manifolds of a multi-manifold contact are in this range.
            let constraint = unsafe { self.scalar_contacts.get_mut(c_idx) };
            // SAFETY: forwarded from caller.
            unsafe {
                match kind {
                    StageKind::WarmStart => constraint.warm_start_shared(&self.bodies),
                    StageKind::Solve => constraint.push_shared(
                        &self.bodies,
                        &self.motions,
                        self.inv_h,
                        self.contact_speed,
                    ),
                    StageKind::Relax => {
                        constraint.relax_shared(&self.bodies, &self.motions, self.inv_h)
                    }
                    StageKind::Restitution => {
                        if constraint.has_restitution() {
                            constraint.restitute_shared(
                                &self.bodies,
                                &self.motions,
                                self.inv_h,
                                self.restitution_threshold,
                            );
                        }
                    }
                    _ => unreachable!("scalar contacts have no {kind:?} graph stage"),
                }
            }
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.wide_contacts[index]` or its contact slots during the call.
    unsafe fn store_wide(&self, index: usize) {
        // SAFETY: forwarded from caller.
        let constraint = unsafe { self.wide_contacts.get(index) };
        for (lane, &slot) in self.wide_slots[index].iter().enumerate() {
            if slot == crate::solver_wide::EMPTY_LANE {
                break;
            }
            // SAFETY: each contact slot is in at most one wide lane across the entire graph.
            let contact_opt = unsafe { self.contact_slots.get_mut(slot as usize) };
            if let Some(manifold) = contact_opt.as_mut().and_then(|c| c.manifolds.first_mut()) {
                constraint.store(lane, manifold);
            }
        }
    }

    /// # Safety
    ///
    /// No other thread may access the scalar constraints or contact slot of `range_idx` during the call.
    unsafe fn store_scalar_range(&self, range_idx: usize) {
        let (start, end) = self.scalar_contact_ranges[range_idx];
        for c_idx in start as usize..end as usize {
            let (slot, m_idx) = self.scalar_refs[c_idx];
            // SAFETY: forwarded from caller.
            let constraint = unsafe { self.scalar_contacts.get(c_idx) };
            // SAFETY: all manifolds of `slot` are in this same `range_idx`.
            let contact_opt = unsafe { self.contact_slots.get_mut(slot as usize) };
            if let Some(manifold) = contact_opt
                .as_mut()
                .and_then(|c| c.manifolds.get_mut(m_idx))
            {
                constraint.store(manifold);
            }
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.bodies[awake[index]]` during the call.
    unsafe fn integrate_velocity(&self, index: usize) {
        let i = self.awake[index] as usize;
        // SAFETY: each awake body slot appears once in `self.awake`.
        let body = unsafe { self.bodies.get_mut(i) };
        if body.is_moving() {
            // SAFETY: motions are read-only during IntegrateVelocities.
            let motion = unsafe { self.motions.read(i) };
            body.integrate_velocities(
                self.gravity,
                &self.inv_inertia_world[i],
                motion.delta_rotation,
                self.start_rotations[i],
                self.h,
            );
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.bodies[awake[index]]` or `self.motions[awake[index]]` during the call.
    unsafe fn integrate_position(&self, index: usize) {
        let i = self.awake[index] as usize;
        // SAFETY: each awake body slot appears once in `self.awake`.
        let body = unsafe { self.bodies.get_mut(i) };
        if !body.is_moving() {
            return;
        }
        let (mut v, mut w) = body
            .motion_locks
            .apply(body.linear_velocity, body.angular_velocity);
        if v.dot(v) > self.max_linear_speed * self.max_linear_speed {
            v *= self.max_linear_speed / v.length();
        }
        if w.dot(w) > self.max_angular_speed * self.max_angular_speed && !body.allow_fast_rotation {
            w *= self.max_angular_speed / w.length();
        }
        body.linear_velocity = v;
        body.angular_velocity = w;
        // SAFETY: each awake body slot appears once in `self.awake`.
        let motion = unsafe { self.motions.get_mut(i) };
        motion.delta_position += v * self.h;
        if w != Vec3::ZERO {
            motion.delta_rotation = integrate_rotation(motion.delta_rotation, w * self.h);
        }
    }

    /// # Safety
    ///
    /// No other thread may access `self.bodies[awake[index]]` during the call.
    unsafe fn finalize_position(&self, index: usize) {
        let i = self.awake[index] as usize;
        // SAFETY: each awake body slot appears once in `self.awake`.
        let body = unsafe { self.bodies.get_mut(i) };
        if !body.is_moving() {
            return;
        }
        // SAFETY: motions are read-only during FinalizePositions.
        let motion = unsafe { self.motions.read(i) };
        let center = self.start_positions[i] + motion.delta_position;
        body.rotation = (motion.delta_rotation * self.start_rotations[i]).normalize();
        body.position = center - body.rotation * body.local_center;
    }
}

/// Solves only the awake solver set's bodies and the constraint graph's colored contacts and
/// joints (Box3D `b3Solve`), in graph color order (`0..color_count`). Single-manifold convex
/// contacts in colors `0..OVERFLOW_INDEX` are packed four to a [`WideContactConstraint3d`] and
/// solved in [`korppi_math::FloatW`] SIMD lanes across `executor` workers with bit-identical
/// results on any worker count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_graph(
    bodies: &mut [RigidBody],
    awake: &[u32],
    graph: &crate::constraint_graph::ConstraintGraph3d,
    contact_manager: &mut crate::contact::ContactManager3d,
    joints: &mut [Option<Joint>],
    gravity: Vec3,
    dt: f32,
    config: &SolverConfig3d,
    scratch: &mut SolverScratch3d,
    executor: Option<&dyn korppi_core::TaskExecutor>,
) {
    if awake.is_empty() || dt <= 0.0 {
        return;
    }
    let worker_count = executor.map_or(1, |e| e.worker_count().max(1));
    let sub_steps = config.sub_steps.max(1);
    let h = dt / sub_steps as f32;
    let inv_h = sub_steps as f32 / dt;
    let max_angular_speed = MAX_ROTATION / dt;
    let softness = config.contact_softness(h);

    let SolverScratch3d {
        start_positions,
        start_rotations,
        inv_inertia_world,
        motions,
        contacts: scalar_contacts,
        contact_refs: scalar_refs,
        scalar_contact_ranges,
        wide_contacts,
        wide_slots,
        overflow_contacts,
        overflow_refs,
        colored_joints,
        colors,
        blocks,
        stages,
        ..
    } = scratch;

    if start_positions.len() < bodies.len() {
        start_positions.resize(bodies.len(), Vec3::ZERO);
        start_rotations.resize(bodies.len(), Quat::IDENTITY);
        inv_inertia_world.resize(bodies.len(), Mat3::ZERO);
        motions.resize(bodies.len(), BodyMotion3d::NONE);
    }
    for &slot in awake {
        let i = slot as usize;
        let body = &bodies[i];
        start_positions[i] = body.center_of_mass();
        start_rotations[i] = body.rotation;
        inv_inertia_world[i] = body.world_inv_inertia();
        motions[i] = BodyMotion3d::NONE;
    }

    colors.clear();
    colored_joints.clear();
    wide_slots.clear();
    scalar_refs.clear();
    scalar_contact_ranges.clear();
    overflow_contacts.clear();
    overflow_refs.clear();

    let overflow_index = graph.overflow_index();
    for color_idx in 0..overflow_index {
        let color = graph.color(color_idx);
        if color.joint_indices.is_empty() && color.contact_slots.is_empty() {
            continue;
        }
        let joint_start = colored_joints.len() as u32;
        for &joint_idx in &color.joint_indices {
            if let Some(joint) = joints.get(joint_idx).and_then(Option::as_ref) {
                let (a, b) = (joint.body_a() as usize, joint.body_b() as usize);
                if a < bodies.len() && !bodies[a].is_moving() {
                    inv_inertia_world[a] = Mat3::ZERO;
                    motions[a] = BodyMotion3d::NONE;
                }
                if b < bodies.len() && !bodies[b].is_moving() {
                    inv_inertia_world[b] = Mat3::ZERO;
                    motions[b] = BodyMotion3d::NONE;
                }
                if is_active_joint(bodies, joint) {
                    colored_joints.push(joint_idx as u32);
                }
            }
        }
        let joint_count = colored_joints.len() as u32 - joint_start;

        let wide_start = wide_slots.len() as u32;
        let scalar_start = scalar_contact_ranges.len() as u32;
        let mut current_wide = [crate::solver_wide::EMPTY_LANE; crate::solver_wide::WIDE];
        let mut current_lane = 0;

        for &slot in &color.contact_slots {
            let contact = contact_manager.at(slot);
            let (a, b) = (contact.body_a as usize, contact.body_b as usize);
            if a == b
                || a >= bodies.len()
                || b >= bodies.len()
                || !is_active(&bodies[a], &bodies[b])
                || contact.manifolds.is_empty()
            {
                continue;
            }
            if !bodies[a].is_moving() {
                inv_inertia_world[a] = Mat3::ZERO;
                motions[a] = BodyMotion3d::NONE;
            }
            if !bodies[b].is_moving() {
                inv_inertia_world[b] = Mat3::ZERO;
                motions[b] = BodyMotion3d::NONE;
            }
            if contact.manifolds.len() == 1 && !contact.manifolds[0].mesh {
                current_wide[current_lane] = slot;
                current_lane += 1;
                if current_lane == crate::solver_wide::WIDE {
                    wide_slots.push(current_wide);
                    current_wide = [crate::solver_wide::EMPTY_LANE; crate::solver_wide::WIDE];
                    current_lane = 0;
                }
            } else {
                let r_start = scalar_refs.len() as u32;
                for m_idx in 0..contact.manifolds.len() {
                    scalar_refs.push((slot, m_idx));
                }
                scalar_contact_ranges.push((r_start, scalar_refs.len() as u32));
            }
        }
        if current_lane > 0 {
            wide_slots.push(current_wide);
        }
        let wide_count = wide_slots.len() as u32 - wide_start;
        let scalar_count = scalar_contact_ranges.len() as u32 - scalar_start;
        if joint_count > 0 || wide_count > 0 || scalar_count > 0 {
            colors.push(ActiveColor3d {
                joint_start,
                joint_count,
                wide_start,
                wide_count,
                scalar_start,
                scalar_count,
            });
        }
    }

    wide_contacts.resize(
        wide_slots.len(),
        crate::solver_wide::WideContactConstraint3d::default(),
    );
    scalar_contacts.resize(scalar_refs.len(), ContactConstraint3d::default());

    // Prepare overflow contacts on the calling thread; overflow joints are prepared in orchestrate.
    let overflow = graph.color(overflow_index);
    let mut any_restitution = false;
    for &slot in &overflow.contact_slots {
        let contact = contact_manager.at(slot);
        let (a, b) = (contact.body_a as usize, contact.body_b as usize);
        if a == b || a >= bodies.len() || b >= bodies.len() || !is_active(&bodies[a], &bodies[b]) {
            continue;
        }
        if !bodies[a].is_moving() {
            inv_inertia_world[a] = Mat3::ZERO;
            motions[a] = BodyMotion3d::NONE;
        }
        if !bodies[b].is_moving() {
            inv_inertia_world[b] = Mat3::ZERO;
            motions[b] = BodyMotion3d::NONE;
        }
        for (m_idx, manifold) in contact.manifolds.iter().enumerate() {
            let c = ContactConstraint3d::prepare(
                manifold,
                bodies,
                inv_inertia_world,
                softness,
                config.enable_warm_starting,
            );
            any_restitution |= c.has_restitution();
            overflow_contacts.push(c);
            overflow_refs.push((slot, m_idx));
        }
    }
    for &joint_idx in &overflow.joint_indices {
        if let Some(joint) = joints.get(joint_idx).and_then(Option::as_ref) {
            let (a, b) = (joint.body_a() as usize, joint.body_b() as usize);
            if a < bodies.len() && !bodies[a].is_moving() {
                inv_inertia_world[a] = Mat3::ZERO;
                motions[a] = BodyMotion3d::NONE;
            }
            if b < bodies.len() && !bodies[b].is_moving() {
                inv_inertia_world[b] = Mat3::ZERO;
                motions[b] = BodyMotion3d::NONE;
            }
        }
    }

    let max_blocks = 4 * worker_count;
    blocks.clear();
    stages.clear();

    let body_blocks = push_blocks(
        blocks,
        awake.len(),
        MIN_BODIES_PER_BLOCK,
        max_blocks,
        BlockKind::Items,
    );
    let joint_blocks = push_blocks(
        blocks,
        colored_joints.len(),
        MIN_JOINTS_PER_BLOCK,
        max_blocks,
        BlockKind::Items,
    );
    let wide_blocks = push_blocks(
        blocks,
        wide_slots.len(),
        MIN_CONTACTS_PER_BLOCK,
        max_blocks,
        BlockKind::Items,
    );
    let scalar_blocks = push_blocks(
        blocks,
        scalar_contact_ranges.len(),
        MIN_CONTACTS_PER_BLOCK,
        max_blocks,
        BlockKind::Items,
    );

    let mut color_blocks = [(0u32, 0u32); crate::constraint_graph::GRAPH_COLOR_COUNT];
    for (i, color) in colors.iter().enumerate() {
        let (first, j_count) = push_blocks(
            blocks,
            color.joint_count as usize,
            MIN_JOINTS_PER_BLOCK,
            max_blocks,
            BlockKind::GraphJoints,
        );
        let (_, w_count) = push_blocks(
            blocks,
            color.wide_count as usize,
            MIN_CONTACTS_PER_BLOCK,
            max_blocks,
            BlockKind::GraphWideContacts,
        );
        let (_, s_count) = push_blocks(
            blocks,
            color.scalar_count as usize,
            MIN_CONTACTS_PER_BLOCK,
            max_blocks,
            BlockKind::GraphScalarContacts,
        );
        color_blocks[i] = (first, j_count + w_count + s_count);
    }

    let mut add_stage = |kind, (first_block, block_count): (u32, u32), color: usize| {
        stages.push(Stage {
            kind,
            first_block,
            block_count,
            color: color as u32,
            completion: std::sync::atomic::AtomicU32::new(0),
        });
    };

    add_stage(StageKind::PrepareJoints, joint_blocks, 0);
    add_stage(StageKind::PrepareWideContacts, wide_blocks, 0);
    add_stage(StageKind::PrepareScalarContacts, scalar_blocks, 0);
    add_stage(StageKind::IntegrateVelocities, body_blocks, 0);
    for kind in [StageKind::WarmStart, StageKind::Solve] {
        for (color, &cb) in color_blocks[..colors.len()].iter().enumerate() {
            add_stage(kind, cb, color);
        }
    }
    add_stage(StageKind::IntegratePositions, body_blocks, 0);
    for (color, &cb) in color_blocks[..colors.len()].iter().enumerate() {
        add_stage(StageKind::Relax, cb, color);
    }
    add_stage(StageKind::FinalizePositions, body_blocks, 0);
    for (color, &cb) in color_blocks[..colors.len()].iter().enumerate() {
        add_stage(StageKind::Restitution, cb, color);
    }
    add_stage(StageKind::StoreWideImpulses, wide_blocks, 0);
    add_stage(StageKind::StoreScalarImpulses, scalar_blocks, 0);

    let max_stage_blocks = stages
        .iter()
        .map(|s| s.block_count as usize)
        .max()
        .unwrap_or(0);
    let active_workers = worker_count.min(max_stage_blocks.max(1)).max(1);

    let context = SolveContext3d {
        bodies: crate::parallel::Shared::new(bodies),
        motions: crate::parallel::Shared::new(motions),
        start_positions,
        start_rotations,
        inv_inertia_world,
        awake,
        graph,
        colors,
        colored_joints,
        wide_slots,
        wide_contacts: crate::parallel::Shared::new(wide_contacts),
        scalar_contacts: crate::parallel::Shared::new(scalar_contacts),
        scalar_refs,
        scalar_contact_ranges,
        overflow_contacts: crate::parallel::Shared::new(overflow_contacts),
        overflow_refs,
        contact_slots: crate::parallel::Shared::new(contact_manager.slots_mut()),
        joints: crate::parallel::Shared::new(joints),
        stages,
        blocks,
        gravity,
        h,
        inv_h,
        sub_steps,
        max_linear_speed: config.max_linear_speed,
        max_angular_speed,
        softness,
        contact_speed: config.contact_speed,
        restitution_threshold: config.restitution_threshold,
        restitution_iterations: config.restitution_iterations,
        enable_warm_starting: config.enable_warm_starting,
        any_restitution: std::sync::atomic::AtomicBool::new(any_restitution),
        sync_bits: std::sync::atomic::AtomicU32::new(0),
        main_claimed: std::sync::atomic::AtomicBool::new(false),
        next_worker: std::sync::atomic::AtomicU32::new(1),
        worker_count: active_workers,
    };

    match executor {
        Some(executor) if active_workers > 1 => {
            executor.run(active_workers, &|_task, _| context.task());
            if !context
                .main_claimed
                .load(std::sync::atomic::Ordering::Acquire)
            {
                context.orchestrate();
            }
        }
        _ => context.orchestrate(),
    }
}
