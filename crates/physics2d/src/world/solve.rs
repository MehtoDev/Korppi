//! The constraint solve of a step, staged for parallel workers (Box2D `b2SolverStep`,
//! `b2SolverTask`).
//!
//! The solve gathers the awake bodies into a [`BodyState2d`] array, packs each graph color's
//! contacts four to a [`WideContactConstraint2d`] and the overflow's into scalar
//! [`ContactConstraint2d`]s, then runs Box2D's stages: prepare the joints and contacts, and per
//! substep integrate velocities, warm start, solve, integrate positions and relax; then
//! restitution and storing the impulses. A stage is a list of blocks of items (bodies, colored
//! joints, wide contacts). One thread, the orchestrator, walks the stages: it publishes each
//! stage through an atomic word of sync bits, claims blocks itself, and waits until every block
//! is done before the next. The other workers spin on the sync bits and claim blocks of the
//! published stage. A block is claimed by a compare-and-swap of its sync index, which grows by
//! one each time its group of blocks is used, so a late worker cannot claim a block twice.
//! The overflow constraints are solved by the orchestrator between stages, while no block is
//! running.
//!
//! A stage's blocks touch disjoint data: bodies by awake index, joints and wide contacts by
//! index, and within a graph color no two constraints share a dynamic body, the only bodies
//! whose states take velocities. So the result is the same whatever thread solves a block and
//! in whatever order, on any number of workers.

use super::islands::{BodyLinks2d, AWAKE_SET};
use super::JointInstance2d;
use crate::body::{BodyType, RigidBody2d};
use crate::body_state::BodyState2d;
use crate::bvh::DynamicTree2d;
use crate::constraint_graph::ConstraintGraph2d;
use crate::contact_manager::Contact2d;
use crate::joint::{JointSolveContext, JointStep2d, Softness2d};
use crate::parallel::Shared;
use crate::solver::ContactConstraint2d;
use crate::solver_wide::{PushParams, WideContactConstraint2d, WideLane, WIDE};
use korppi_core::{spin_until, TaskExecutor};
use korppi_math::Vec2;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// An empty lane of a wide constraint.
const EMPTY_LANE: u32 = u32::MAX;

/// The sync bits that tell the workers to leave.
const SENTINEL: u32 = u32::MAX;

/// Fewest items per block (Box2D's `minBodiesPerBlock`, `minContactsPerBlock`,
/// `minJointsPerBlock`).
const MIN_BODIES_PER_BLOCK: usize = 64;
const MIN_CONTACTS_PER_BLOCK: usize = 16;
const MIN_JOINTS_PER_BLOCK: usize = 8;

/// A solver stage (Box2D `b2SolverStageType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageKind {
    PrepareJoints,
    PrepareContacts,
    IntegrateVelocities,
    WarmStart,
    Solve,
    IntegratePositions,
    Relax,
    Restitution,
    StoreImpulses,
}

/// What a block's items are: a stage's own items, or a graph color's joints or wide contacts
/// (Box2D `b2SolverBlockType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Items,
    GraphJoints,
    GraphContacts,
}

/// A range of items that one worker processes at a time (Box2D `b2SyncBlock`).
#[derive(Debug)]
struct SyncBlock {
    start: u32,
    count: u32,
    kind: BlockKind,
    /// The sync index of the block's last use; claiming it for the next use swaps in the
    /// next index.
    sync: AtomicU32,
}

/// A stage and its blocks (Box2D `b2SolverStage`).
#[derive(Debug)]
struct Stage {
    kind: StageKind,
    first_block: u32,
    block_count: u32,
    /// The active color a graph stage solves.
    color: u32,
    /// Blocks done, counted by the workers that did them.
    completion: AtomicU32,
}

/// A graph color with constraints this step.
#[derive(Debug, Clone, Copy)]
struct ActiveColor {
    /// Its first joint in the step's colored joints, and their number.
    joint_start: u32,
    joint_count: u32,
    /// Its first wide constraint, and their number.
    wide_start: u32,
    wide_count: u32,
}

/// The solver's buffers, kept between steps so that steps do not allocate once they have grown
/// to the scene's size.
#[derive(Debug, Default)]
pub(super) struct SolverScratch2d {
    /// The awake bodies' states, by awake index.
    pub(super) states: Vec<BodyState2d>,
    /// The graph colors' contacts, four to a constraint, color by color.
    wide: Vec<WideContactConstraint2d>,
    /// Each wide constraint's contact slots, [`EMPTY_LANE`] for empty lanes.
    wide_slots: Vec<[u32; WIDE]>,
    /// The overflow's contacts, in the overflow color's order.
    overflow: Vec<ContactConstraint2d>,
    /// The joint slots of the active colors, color by color.
    colored_joints: Vec<u32>,
    colors: Vec<ActiveColor>,
    blocks: Vec<SyncBlock>,
    stages: Vec<Stage>,
}

impl Clone for SolverScratch2d {
    /// Scratch is not state: a clone starts empty.
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl SolverScratch2d {
    /// Makes room for `contacts` touching contacts.
    pub(super) fn reserve_contacts(&mut self, contacts: usize) {
        let wide = contacts.div_ceil(WIDE);
        self.wide.reserve(wide.saturating_sub(self.wide.len()));
        self.wide_slots
            .reserve(wide.saturating_sub(self.wide_slots.len()));
    }
}

/// The step's constants for the solve.
#[derive(Debug, Clone, Copy)]
pub(super) struct SolveParams {
    pub gravity: Vec2,
    /// Substep length and rate.
    pub h: f32,
    pub inv_h: f32,
    pub sub_steps: usize,
    pub max_linear_speed: f32,
    pub max_angular_speed: f32,
    pub contact_softness: Softness2d,
    pub static_softness: Softness2d,
    pub push_speed: f32,
    pub restitution_threshold: f32,
    pub restitution_iterations: usize,
    pub restitution_propagation: bool,
    pub enable_warm_starting: bool,
    pub length_units: f32,
}

/// The world's parts the solve reads and writes.
pub(super) struct SolveData<'a> {
    pub bodies: &'a [RigidBody2d],
    pub body_links: &'a [BodyLinks2d],
    pub awake: &'a [u32],
    pub joint_bodies: &'a [(u32, u32)],
    pub graph: &'a ConstraintGraph2d,
    pub joints: &'a mut [Option<JointInstance2d>],
    pub joint_steps: &'a mut [JointStep2d],
    pub thresholds_reached: &'a mut [bool],
    pub contacts: &'a mut [Option<Contact2d>],
    pub trees: &'a mut [DynamicTree2d],
    pub scratch: &'a mut SolverScratch2d,
    pub executor: Option<&'a dyn TaskExecutor>,
}

/// Box2D `b2ComputeBlockCount`: block size and count for `items` items.
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

/// Appends the blocks of `items` items (Box2D `b2InitBlocks`) and returns the first one's
/// index and their number.
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
            sync: AtomicU32::new(0),
        });
    }
    (first, count as u32)
}

/// Body `slot`'s awake index plus one, zero if it is not awake.
fn awake_index(links: &[BodyLinks2d], slot: u32) -> u32 {
    let links = &links[slot as usize];
    if links.set == AWAKE_SET {
        links.local + 1
    } else {
        0
    }
}

/// Solves the step's constraints (see the module documentation) and leaves the awake bodies'
/// new velocities and motion in `data.scratch.states`.
pub(super) fn solve(data: SolveData<'_>, params: &SolveParams) {
    let SolveData {
        bodies,
        body_links,
        awake,
        joint_bodies,
        graph,
        joints,
        joint_steps,
        thresholds_reached,
        contacts,
        trees,
        scratch,
        executor,
    } = data;
    let worker_count = executor.map_or(1, |e| e.worker_count().max(1));
    let warm_start_scale = if params.enable_warm_starting {
        1.0
    } else {
        0.0
    };

    // The awake bodies' states, by awake index.
    scratch.states.resize(awake.len(), BodyState2d::IDENTITY);
    match executor {
        Some(executor) if worker_count > 1 && awake.len() >= 256 => {
            let states_shared = Shared::new(&mut scratch.states);
            korppi_core::parallel_for(
                executor,
                awake.len(),
                MIN_BODIES_PER_BLOCK,
                &|start, end, _| {
                    for (local, &slot) in awake.iter().enumerate().take(end).skip(start) {
                        debug_assert_eq!(awake_index(body_links, slot) as usize, local + 1);
                        // SAFETY: each `local` in `0..awake.len()` belongs to exactly one block.
                        unsafe {
                            *states_shared.get_mut(local) =
                                BodyState2d::at_rest(&bodies[slot as usize]);
                        }
                    }
                },
            );
        }
        _ => {
            for (local, &slot) in awake.iter().enumerate() {
                debug_assert_eq!(awake_index(body_links, slot) as usize, local + 1);
                scratch.states[local] = BodyState2d::at_rest(&bodies[slot as usize]);
            }
        }
    }

    // The active colors, their joints and their contacts' lanes.
    scratch.colors.clear();
    scratch.colored_joints.clear();
    scratch.wide_slots.clear();
    let overflow_index = graph.overflow_index();
    for index in 0..overflow_index {
        let color = graph.color(index);
        if color.joint_indices.is_empty() && color.contact_slots.is_empty() {
            continue;
        }
        let active = ActiveColor {
            joint_start: scratch.colored_joints.len() as u32,
            joint_count: color.joint_indices.len() as u32,
            wide_start: scratch.wide_slots.len() as u32,
            wide_count: color.contact_slots.len().div_ceil(WIDE) as u32,
        };
        scratch
            .colored_joints
            .extend(color.joint_indices.iter().map(|&j| j as u32));
        for lanes in color.contact_slots.chunks(WIDE) {
            let mut slots = [EMPTY_LANE; WIDE];
            slots[..lanes.len()].copy_from_slice(lanes);
            scratch.wide_slots.push(slots);
        }
        scratch.colors.push(active);
    }
    scratch
        .wide
        .resize(scratch.wide_slots.len(), WideContactConstraint2d::default());

    // The overflow's contacts, prepared here, one at a time (Box2D
    // `b2PrepareContacts_Overflow`); the orchestrator prepares its joints.
    let overflow = graph.color(overflow_index);
    let mut any_restitution = false;
    scratch.overflow.clear();
    for &slot in &overflow.contact_slots {
        let contact = contacts[slot as usize].as_ref().expect("live contact");
        let (a, b) = (contact.body_a, contact.body_b);
        let constraint = ContactConstraint2d::prepare(
            (awake_index(body_links, a), awake_index(body_links, b)),
            (&bodies[a as usize], &bodies[b as usize]),
            &contact.manifold,
            contact.material,
            (params.contact_softness, params.static_softness),
            warm_start_scale,
        );
        any_restitution |= constraint.has_restitution();
        scratch.overflow.push(constraint);
    }

    // Blocks and stages (Box2D's stage setup in `b2SolverStep`), four blocks per worker.
    let max_blocks = 4 * worker_count;
    scratch.blocks.clear();
    scratch.stages.clear();
    let SolverScratch2d {
        states,
        wide,
        wide_slots,
        overflow: overflow_constraints,
        colored_joints,
        colors,
        blocks,
        stages,
    } = scratch;
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
    let contact_blocks = push_blocks(
        blocks,
        wide_slots.len(),
        MIN_CONTACTS_PER_BLOCK,
        max_blocks,
        BlockKind::Items,
    );
    // Each color's joint blocks, then its contact blocks, shared by all its stages.
    let mut color_blocks = [(0, 0); crate::constraint_graph::GRAPH_COLOR_COUNT];
    for (i, color) in colors.iter().enumerate() {
        let (first, joint_count) = push_blocks(
            blocks,
            color.joint_count as usize,
            MIN_JOINTS_PER_BLOCK,
            max_blocks,
            BlockKind::GraphJoints,
        );
        let (_, contact_count) = push_blocks(
            blocks,
            color.wide_count as usize,
            MIN_CONTACTS_PER_BLOCK,
            max_blocks,
            BlockKind::GraphContacts,
        );
        color_blocks[i] = (first, joint_count + contact_count);
    }
    let mut stage = |kind, (first_block, block_count): (u32, u32), color: usize| {
        stages.push(Stage {
            kind,
            first_block,
            block_count,
            color: color as u32,
            completion: AtomicU32::new(0),
        });
    };
    stage(StageKind::PrepareJoints, joint_blocks, 0);
    stage(StageKind::PrepareContacts, contact_blocks, 0);
    stage(StageKind::IntegrateVelocities, body_blocks, 0);
    for kind in [StageKind::WarmStart, StageKind::Solve] {
        for (color, &blocks) in color_blocks[..colors.len()].iter().enumerate() {
            stage(kind, blocks, color);
        }
    }
    stage(StageKind::IntegratePositions, body_blocks, 0);
    for kind in [StageKind::Relax, StageKind::Restitution] {
        for (color, &blocks) in color_blocks[..colors.len()].iter().enumerate() {
            stage(kind, blocks, color);
        }
    }
    stage(StageKind::StoreImpulses, contact_blocks, 0);

    let max_stage_blocks = stages
        .iter()
        .map(|s| s.block_count as usize)
        .max()
        .unwrap_or(0);
    let trees_need_rebuild = trees.iter().any(DynamicTree2d::needs_rebuild);
    if trees_need_rebuild {
        for tree in trees.iter_mut() {
            tree.reserve_rebuild_scratch();
        }
    }
    let max_tasks = max_stage_blocks.max(1) + usize::from(trees_need_rebuild);
    let active_workers = worker_count.min(max_tasks).max(1);
    let trees_len = trees.len();
    let context = SolveContext {
        bodies,
        body_links,
        awake,
        joint_bodies,
        graph,
        colors,
        colored_joints,
        wide_slots,
        stages,
        blocks,
        states: Shared::new(states),
        wide: Shared::new(wide),
        overflow: Shared::new(overflow_constraints),
        joints: Shared::new(joints),
        joint_steps: Shared::new(joint_steps),
        thresholds_reached: Shared::new(thresholds_reached),
        contacts: Shared::new(contacts),
        trees: Shared::new(trees),
        trees_len,
        params,
        warm_start_scale,
        any_restitution: AtomicBool::new(any_restitution),
        sync_bits: AtomicU32::new(0),
        main_claimed: AtomicBool::new(false),
        tree_claimed: AtomicBool::new(!trees_need_rebuild),
        next_worker: AtomicU32::new(1),
        worker_count: active_workers,
    };
    match executor {
        Some(executor) if active_workers > 1 => {
            // Every task races for the orchestrator's place; the others become workers. The
            // orchestrator can do all the work alone, so the solve finishes whatever the
            // executor runs, and if it runs nothing, this thread orchestrates.
            executor.run(active_workers, &|_task, _| context.task());
            if !context.main_claimed.load(Ordering::Acquire) {
                context.orchestrate();
            }
        }
        _ => context.orchestrate(),
    }
}

/// Everything the solve's threads share.
struct SolveContext<'a> {
    bodies: &'a [RigidBody2d],
    body_links: &'a [BodyLinks2d],
    awake: &'a [u32],
    joint_bodies: &'a [(u32, u32)],
    graph: &'a ConstraintGraph2d,
    colors: &'a [ActiveColor],
    colored_joints: &'a [u32],
    wide_slots: &'a [[u32; WIDE]],
    stages: &'a [Stage],
    blocks: &'a [SyncBlock],
    states: Shared<'a, BodyState2d>,
    wide: Shared<'a, WideContactConstraint2d>,
    overflow: Shared<'a, ContactConstraint2d>,
    joints: Shared<'a, Option<JointInstance2d>>,
    joint_steps: Shared<'a, JointStep2d>,
    thresholds_reached: Shared<'a, bool>,
    contacts: Shared<'a, Option<Contact2d>>,
    trees: Shared<'a, DynamicTree2d>,
    trees_len: usize,
    params: &'a SolveParams,
    warm_start_scale: f32,
    any_restitution: AtomicBool,
    /// `sync index << 16 | stage index` of the stage the workers may claim blocks of, or
    /// [`SENTINEL`] once the solve is done (Box2D `atomicSyncBits`).
    sync_bits: AtomicU32,
    /// Whether a task took the orchestrator's place (Box2D `mainClaimed`).
    main_claimed: AtomicBool,
    /// Whether a task took the async broadphase tree rebuild (`b2UpdateTreesTask`).
    tree_claimed: AtomicBool,
    /// Next helper worker index (`1..worker_count`).
    next_worker: AtomicU32,
    worker_count: usize,
}

/// Counts a worker's claimed blocks in a stage as done when dropped (even if executing one
/// panicked), so that the orchestrator does not wait forever and normal execution performs a
/// single atomic `fetch_add` per stage (Box2D `b2ExecuteStage`).
struct StageDone<'a> {
    completion: &'a AtomicU32,
    count: u32,
}

impl Drop for StageDone<'_> {
    fn drop(&mut self) {
        if self.count > 0 {
            self.completion.fetch_add(self.count, Ordering::Release);
        }
    }
}

/// Publishes the sentinel when the orchestrator leaves, normally or by a panic, so that the
/// workers leave too.
struct Finish<'a>(&'a AtomicU32);

impl Drop for Finish<'_> {
    fn drop(&mut self) {
        self.0.store(SENTINEL, Ordering::Release);
    }
}

/// Box2D `GetWorkerStartIndex`: where worker `worker` starts its sweep of `block_count` blocks,
/// staggered so that workers start on different blocks.
fn worker_start_index(worker: usize, block_count: usize, worker_count: usize) -> Option<usize> {
    if block_count <= worker_count {
        return (worker < block_count).then_some(worker);
    }
    let per_worker = block_count / worker_count;
    let remainder = block_count - per_worker * worker_count;
    Some(per_worker * worker + remainder.min(worker))
}

impl SolveContext<'_> {
    /// An executor task: the orchestrator if no task has become it, else a worker (one of which
    /// claims the concurrent broadphase tree rebuild before joining the stage loop, matching
    /// Box2D `b2UpdateTreesTask`).
    fn task(&self) {
        if self
            .main_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.orchestrate();
        } else {
            let worker = self.next_worker.fetch_add(1, Ordering::Relaxed) as usize;
            self.try_rebuild_trees();
            self.work(worker % self.worker_count);
        }
    }

    /// Rebuilds the dynamic and kinematic broadphase trees if no thread has claimed them yet
    /// (Box2D `b2UpdateTreesTask`).
    fn try_rebuild_trees(&self) {
        if self
            .tree_claimed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            for i in 0..self.trees_len {
                // SAFETY: `tree_claimed` was transitioned from `false` to `true` by this thread
                // alone, and nothing else in `SolveContext` accesses `trees`.
                unsafe { self.trees.get_mut(i).rebuild(false) };
            }
        }
    }

    /// A worker: claims blocks of each stage the orchestrator publishes, until the sentinel
    /// (Box2D's spinning part of `b2SolverTask`).
    fn work(&self, worker: usize) {
        let mut last = 0;
        loop {
            let mut bits = last;
            spin_until(|| {
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

    /// Claims and executes the unclaimed blocks of `stage`, starting at the worker's own
    /// (Box2D `b2ExecuteStage`).
    fn execute_stage(&self, stage: &Stage, previous_sync: u32, sync: u32, worker: usize) {
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

    /// Runs stage `index` with sync index `sync` and returns once all its blocks are done
    /// (Box2D `b2ExecuteMainStage`).
    fn main_stage(&self, index: usize, sync: u32) {
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
        // Wait for the blocks other workers claimed without yielding the orchestrator thread
        // (Box2D `b2ExecuteMainStage`).
        while stage.completion.load(Ordering::Acquire) != count {
            std::hint::spin_loop();
        }
        stage.completion.store(0, Ordering::Relaxed);
    }

    /// The orchestrator: every stage in order, with the overflow between them (Box2D's main
    /// part of `b2SolverTask`).
    fn orchestrate(&self) {
        {
            let _finish = Finish(&self.sync_bits);
            let colors = self.colors.len();
            let (mut body_sync, mut graph_sync) = (1, 1);
            // Prepare the colored joints and contacts, then the overflow's joints (its contacts
            // are prepared).
            self.main_stage(0, 1);
            self.main_stage(1, 1);
            let overflow = self.graph.color(self.graph.overflow_index());
            for &joint in &overflow.joint_indices {
                // SAFETY: no block runs between stages, and overflow joints are in no block.
                unsafe { self.prepare_joint(joint as u32) };
            }
            let mut stage = 2;
            for _ in 0..self.params.sub_steps {
                let mut index = stage;
                self.main_stage(index, body_sync);
                index += 1;
                body_sync += 1;
                for kind in [StageKind::WarmStart, StageKind::Solve] {
                    self.overflow(kind);
                    for _ in 0..colors {
                        self.main_stage(index, graph_sync);
                        index += 1;
                    }
                    graph_sync += 1;
                }
                self.main_stage(index, body_sync);
                index += 1;
                body_sync += 1;
                self.overflow(StageKind::Relax);
                for _ in 0..colors {
                    self.main_stage(index, graph_sync);
                    index += 1;
                }
                graph_sync += 1;
            }
            stage += 1 + 2 * colors + 1 + colors;
            if self.any_restitution.load(Ordering::Relaxed) {
                for _ in 0..self.params.restitution_iterations {
                    self.overflow(StageKind::Restitution);
                    for color in 0..colors {
                        self.main_stage(stage + color, graph_sync);
                    }
                    graph_sync += 1;
                }
            }
            stage += colors;
            self.overflow(StageKind::StoreImpulses);
            // The contact blocks' second use, after the prepare.
            self.main_stage(stage, 2);
        }
        self.try_rebuild_trees();
    }

    /// Executes one block of `stage` (Box2D `b2ExecuteBlock`).
    fn execute_block(&self, stage: &Stage, block: &SyncBlock) {
        let range = block.start as usize..(block.start + block.count) as usize;
        let color = self.colors.get(stage.color as usize);
        match (stage.kind, block.kind) {
            (StageKind::PrepareJoints, _) => {
                for i in range {
                    // SAFETY: each colored joint is in one block of this stage, and nothing
                    // else touches joints or joint steps during it.
                    unsafe { self.prepare_joint(self.colored_joints[i]) };
                }
            }
            (StageKind::PrepareContacts, _) => {
                for i in range {
                    // SAFETY: each wide constraint is in one block of this stage; contacts and
                    // states are only read during it.
                    unsafe { self.prepare_wide(i) };
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
            (StageKind::StoreImpulses, _) => {
                for i in range {
                    // SAFETY: each wide constraint, and so each colored contact, is in one block
                    // of this stage.
                    unsafe { self.store_wide(i) };
                }
            }
            (kind, BlockKind::GraphJoints) => {
                let color = color.expect("graph stage");
                for i in range {
                    let joint = self.colored_joints[color.joint_start as usize + i];
                    // SAFETY: the color's joints and contacts share no dynamic body, each joint
                    // is in one block of the stage, and no other color runs meanwhile.
                    unsafe { self.solve_joint(kind, joint) };
                }
            }
            (kind, BlockKind::GraphContacts) => {
                let color = color.expect("graph stage");
                for i in range {
                    // SAFETY: as for the joints.
                    unsafe { self.solve_wide(kind, color.wide_start as usize + i) };
                }
            }
            (_, BlockKind::Items) => unreachable!("graph stages have graph blocks"),
        }
    }

    /// The overflow's joints, then its contacts, one at a time (Box2D's `_Overflow`
    /// functions).
    fn overflow(&self, kind: StageKind) {
        let color = self.graph.color(self.graph.overflow_index());
        if matches!(
            kind,
            StageKind::WarmStart | StageKind::Solve | StageKind::Relax
        ) {
            for &joint in &color.joint_indices {
                // SAFETY: the orchestrator runs the overflow between stages, while no block
                // runs, and overflow joints are in no block.
                unsafe { self.solve_joint(kind, joint as u32) };
            }
        }
        for (i, &slot) in color.contact_slots.iter().enumerate() {
            // SAFETY: as above; only the orchestrator touches the overflow constraints.
            let constraint = unsafe { self.overflow.get_mut(i) };
            if kind == StageKind::StoreImpulses {
                // SAFETY: as above; the contact is the overflow's alone.
                let contact = unsafe { self.contacts.get_mut(slot as usize) };
                constraint.store(&mut contact.as_mut().expect("live contact").manifold);
                continue;
            }
            // SAFETY: as above.
            let (mut a, mut b) =
                unsafe { (self.load(constraint.index_a), self.load(constraint.index_b)) };
            let params = self.params;
            match kind {
                StageKind::WarmStart => constraint.warm_start(&mut a, &mut b),
                StageKind::Solve => {
                    constraint.push(&mut a, &mut b, params.inv_h, params.push_speed)
                }
                StageKind::Relax => constraint.relax(&mut a, &mut b, params.inv_h),
                StageKind::Restitution => constraint.apply_restitution(
                    &mut a,
                    &mut b,
                    params.inv_h,
                    (params.restitution_threshold, params.restitution_propagation),
                ),
                _ => unreachable!("the overflow has no {kind:?} stage"),
            }
            // SAFETY: as above.
            unsafe {
                self.store(constraint.index_a, &a);
                self.store(constraint.index_b, &b);
            }
        }
    }

    /// The state of the body at awake index `index - 1`, or the identity for zero.
    ///
    /// # Safety
    ///
    /// No other thread may write the state during the call.
    unsafe fn load(&self, index: u32) -> BodyState2d {
        if index == 0 {
            BodyState2d::IDENTITY
        } else {
            // SAFETY: forwarded from the caller.
            unsafe { self.states.read(index as usize - 1) }
        }
    }

    /// Writes `state` back to awake index `index - 1` if it is a dynamic body's.
    ///
    /// # Safety
    ///
    /// No other thread may access a dynamic body's state during the call.
    unsafe fn store(&self, index: u32, state: &BodyState2d) {
        if index != 0 && state.dynamic {
            // SAFETY: forwarded from the caller.
            unsafe { *self.states.get_mut(index as usize - 1) = *state };
        }
    }

    /// Prepares joint `joint` from its bodies' poses at the start of the step (Box2D
    /// `b2PrepareJointsTask`).
    ///
    /// # Safety
    ///
    /// No other thread may touch joint `joint` or its step during the call.
    unsafe fn prepare_joint(&self, joint: u32) {
        // SAFETY: forwarded from the caller.
        let (instance, slot) = unsafe {
            (
                self.joints.get_mut(joint as usize),
                self.joint_steps.get_mut(joint as usize),
            )
        };
        let Some(instance) = instance.as_mut() else {
            return;
        };
        if !self.params.enable_warm_starting {
            instance.joint.reset_impulses();
        }
        let (a, b) = self.joint_bodies[joint as usize];
        let mut step = instance.joint.prepare(
            (&self.bodies[a as usize], &self.bodies[b as usize]),
            (
                self.body_links[a as usize].center,
                self.body_links[b as usize].center,
            ),
            self.params.h,
            self.params.inv_h,
        );
        step.index_a = awake_index(self.body_links, a);
        step.index_b = awake_index(self.body_links, b);
        *slot = step;
    }

    /// Warm starts, solves or relaxes joint `joint` (Box2D `b2WarmStartJointsTask`,
    /// `b2SolveJointsTask`).
    ///
    /// # Safety
    ///
    /// No other thread may touch the joint, its step, its threshold flag or its dynamic
    /// bodies' states during the call.
    unsafe fn solve_joint(&self, kind: StageKind, joint: u32) {
        let j = joint as usize;
        // SAFETY: forwarded from the caller.
        let (instance, step) = unsafe { (self.joints.get_mut(j), self.joint_steps.get_mut(j)) };
        let Some(instance) = instance.as_mut() else {
            return;
        };
        // SAFETY: forwarded from the caller.
        let (mut a, mut b) = unsafe { (self.load(step.index_a), self.load(step.index_b)) };
        let context = |use_bias| JointSolveContext {
            h: self.params.h,
            inv_h: self.params.inv_h,
            use_bias,
            length_units: self.params.length_units,
        };
        match kind {
            StageKind::WarmStart => instance.joint.warm_start_step(step, &mut a, &mut b),
            StageKind::Solve => {
                instance
                    .joint
                    .solve_step(step, &mut a, &mut b, &context(true));
                // Box2D checks the thresholds in the biased solve stage.
                // SAFETY: forwarded from the caller.
                let reached = unsafe { self.thresholds_reached.get_mut(j) };
                if !*reached
                    && (instance.force_threshold < f32::MAX || instance.torque_threshold < f32::MAX)
                {
                    let (linear, angular) = instance.joint.reaction_impulses();
                    *reached = linear * self.params.inv_h >= instance.force_threshold
                        || angular * self.params.inv_h >= instance.torque_threshold;
                }
            }
            StageKind::Relax => instance
                .joint
                .solve_step(step, &mut a, &mut b, &context(false)),
            _ => {}
        }
        // SAFETY: forwarded from the caller.
        unsafe {
            self.store(step.index_a, &a);
            self.store(step.index_b, &b);
        }
    }

    /// # Safety
    ///
    /// No other thread may touch wide constraint `index` during the call, nor write its
    /// contacts or bodies' states.
    unsafe fn prepare_wide(&self, index: usize) {
        let lanes = self.wide_slots[index].map(|slot| {
            if slot == EMPTY_LANE {
                return None;
            }
            // SAFETY: forwarded from the caller.
            let contact = unsafe { self.contacts.get(slot as usize) }
                .as_ref()
                .expect("live contact");
            let (a, b) = (contact.body_a, contact.body_b);
            Some(WideLane {
                indices: (
                    awake_index(self.body_links, a),
                    awake_index(self.body_links, b),
                ),
                bodies: (&self.bodies[a as usize], &self.bodies[b as usize]),
                manifold: &contact.manifold,
                material: contact.material,
            })
        });
        // SAFETY: forwarded from the caller.
        let (constraint, any_restitution) =
            unsafe { WideContactConstraint2d::prepare(lanes, &self.states, self.warm_start_scale) };
        // SAFETY: forwarded from the caller.
        unsafe { *self.wide.get_mut(index) = constraint };
        if any_restitution {
            self.any_restitution.store(true, Ordering::Relaxed);
        }
    }

    /// # Safety
    ///
    /// No other thread may touch wide constraint `index` or its bodies' states during the call.
    unsafe fn solve_wide(&self, kind: StageKind, index: usize) {
        // SAFETY: forwarded from the caller.
        let constraint = unsafe { self.wide.get_mut(index) };
        let params = self.params;
        // SAFETY: forwarded from the caller.
        unsafe {
            match kind {
                StageKind::WarmStart => constraint.warm_start(&self.states),
                StageKind::Solve => constraint.push(
                    &self.states,
                    &PushParams {
                        contact_softness: params.contact_softness,
                        static_softness: params.static_softness,
                        inv_h: params.inv_h,
                        push_speed: params.push_speed,
                    },
                ),
                StageKind::Relax => constraint.relax(&self.states, params.inv_h),
                StageKind::Restitution => constraint.apply_restitution(
                    &self.states,
                    params.inv_h,
                    (params.restitution_threshold, params.restitution_propagation),
                ),
                _ => unreachable!("contacts have no {kind:?} graph stage"),
            }
        }
    }

    /// # Safety
    ///
    /// No other thread may touch wide constraint `index` or its contacts during the call.
    unsafe fn store_wide(&self, index: usize) {
        // SAFETY: forwarded from the caller.
        let constraint = unsafe { self.wide.get(index) };
        for (lane, &slot) in self.wide_slots[index].iter().enumerate() {
            if slot == EMPTY_LANE {
                break;
            }
            // SAFETY: forwarded from the caller.
            let contact = unsafe { self.contacts.get_mut(slot as usize) };
            constraint.store(lane, &mut contact.as_mut().expect("live contact").manifold);
        }
    }

    /// Box2D `b2IntegrateVelocitiesTask` for awake body `index`.
    ///
    /// # Safety
    ///
    /// No other thread may touch the body's state during the call.
    unsafe fn integrate_velocity(&self, index: usize) {
        let body = &self.bodies[self.awake[index] as usize];
        if body.body_type != BodyType::Dynamic {
            return;
        }
        // SAFETY: forwarded from the caller.
        let state = unsafe { self.states.get_mut(index) };
        let (v, w) = body.integrated_velocities(
            state.linear_velocity,
            state.angular_velocity,
            self.params.gravity,
            self.params.h,
        );
        state.linear_velocity = v;
        state.angular_velocity = w;
    }

    /// Box2D `b2IntegratePositionsTask` for awake body `index`: the motion locks and speed caps,
    /// then the motion over the substep.
    ///
    /// # Safety
    ///
    /// No other thread may touch the body's state during the call.
    unsafe fn integrate_position(&self, index: usize) {
        let body = &self.bodies[self.awake[index] as usize];
        // SAFETY: forwarded from the caller.
        let state = unsafe { self.states.get_mut(index) };
        let (v, w) = body.limited_velocities(
            state.linear_velocity,
            state.angular_velocity,
            self.params.max_linear_speed,
            self.params.max_angular_speed,
        );
        let h = self.params.h;
        state.linear_velocity = v;
        state.angular_velocity = w;
        state.delta_position += v * h;
        state.delta_rotation = state.delta_rotation.integrate(h * w);
    }
}
