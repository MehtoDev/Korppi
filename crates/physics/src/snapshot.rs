//! Bitwise deterministic 3D world snapshot serialization and replay.

use crate::body::RigidBody;
use crate::collider::Collider;
use crate::contact::ContactManager3d;
use crate::events::{
    BodyEvents3d, ContactEndEvent3d, ContactEvents3d, JointEvents3d, SensorEndEvent3d,
    SensorEvents3d,
};
use crate::joint::Joint;
use crate::solver::SolverConfig3d;
use crate::world::ColliderSlot;
use crate::ColliderId3d;
use korppi_core::IdPool;
use korppi_math::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Serde Fnv64 hasher for determinism and rollback hashing.
#[derive(Debug, Clone, Copy)]
pub struct Fnv64(pub u64);

impl Fnv64 {
    /// Creates a new Fnv64 hasher initialized with the FNV offset basis.
    #[must_use]
    pub const fn new() -> Self {
        Self(14_695_981_039_346_656_037)
    }

    /// Mixes a 32-bit word into the hash.
    pub fn word(&mut self, word: u32) {
        self.0 = (self.0 ^ u64::from(word)).wrapping_mul(1_099_511_628_211);
    }

    /// Mixes an array of 32-bit floats into the hash.
    pub fn floats(&mut self, values: &[f32]) {
        values.iter().for_each(|v| self.word(v.to_bits()));
    }

    /// Mixes an array of 32-bit unsigned integers into the hash.
    pub fn ints(&mut self, values: &[u32]) {
        values.iter().for_each(|&v| self.word(v));
    }
}

impl Default for Fnv64 {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialized state of a single 3D rigid body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BodySnapshot3d {
    /// Entity or body identifier.
    pub id: u32,
    /// World position translation.
    pub position: Vec3,
    /// Orientation quaternion.
    pub rotation: Quat,
    /// Linear velocity vector.
    pub linear_velocity: Vec3,
    /// Angular velocity vector.
    pub angular_velocity: Vec3,
    /// External force accumulator.
    #[serde(default)]
    pub force: Vec3,
    /// External torque accumulator.
    #[serde(default)]
    pub torque: Vec3,
    /// Whether the body is currently sleeping.
    pub is_sleeping: bool,
    /// Accumulated duration in seconds that the body has remained below velocity sleep thresholds.
    #[serde(default)]
    pub sleep_time: f32,
}

impl BodySnapshot3d {
    /// Creates a snapshot from an active rigid body.
    #[must_use]
    pub fn from_body(id: u32, body: &RigidBody) -> Self {
        Self {
            id,
            position: body.position,
            rotation: body.rotation,
            linear_velocity: body.linear_velocity,
            angular_velocity: body.angular_velocity,
            force: body.force,
            torque: body.torque,
            is_sleeping: body.is_sleeping,
            sleep_time: body.sleep_time,
        }
    }
}

fn default_gravity() -> Vec3 {
    Vec3::new(0.0, -9.81, 0.0)
}

const fn null_u32() -> u32 {
    crate::contact::NULL
}

impl PartialEq for crate::constraint_graph::GraphColor3d {
    fn eq(&self, other: &Self) -> bool {
        self.constraints == other.constraints
            && self.joint_indices == other.joint_indices
            && self.joint_pairs == other.joint_pairs
            && self.contact_keys == other.contact_keys
            && self.contact_slots == other.contact_slots
            && self.contact_pairs == other.contact_pairs
    }
}

impl PartialEq for crate::constraint_graph::ConstraintGraph3d {
    fn eq(&self, other: &Self) -> bool {
        self.color_count() == other.color_count()
            && self.overflow_index() == other.overflow_index()
            && self.colors().zip(other.colors()).all(|(a, b)| a == b)
    }
}

/// Bitwise snapshot of the entire simulated 3D world state at a fixed step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot3d {
    /// Monotonic tick step index.
    pub step_index: u64,
    /// Full rigid body states in slot order.
    #[serde(default)]
    pub bodies: Vec<RigidBody>,
    /// Live body handles.
    #[serde(default, with = "crate::serde_ids::id_pool")]
    pub body_ids: IdPool<RigidBody>,
    /// Colliders attached to bodies.
    #[serde(default)]
    pub colliders: Vec<Option<ColliderSlot>>,
    /// Live collider handles.
    #[serde(default, with = "crate::serde_ids::id_pool")]
    pub collider_ids: IdPool<Collider>,
    /// Each body's colliders.
    #[serde(default, with = "crate::serde_ids::nested_handle_vec")]
    pub body_colliders: Vec<Vec<ColliderId3d>>,
    /// The contacts, touching or not, with their manifolds, caches and recycling state.
    #[serde(default)]
    pub contacts: ContactManager3d,
    /// Contacts recycled in the last step.
    #[serde(default)]
    pub recycled_contacts: usize,
    /// Which bodies continuous collision found fast in the last step; their contacts are not
    /// recycled.
    #[serde(default)]
    pub fast: Vec<bool>,
    /// Snapshots of active joints with internal impulse accumulators.
    pub joints: Vec<Option<Joint>>,
    /// Live joint handles.
    #[serde(default, with = "crate::serde_ids::id_pool")]
    pub joint_ids: IdPool<Joint>,
    /// Inverse substep timestep (`1.0 / h`) from the last step, used for joint reaction queries.
    #[serde(default)]
    pub last_inv_h: f32,
    /// Overlapping colliders for sensors.
    #[serde(default, with = "crate::serde_ids::nested_handle_vec")]
    pub sensor_overlaps: Vec<Vec<ColliderId3d>>,
    /// Sensor events from the last step.
    #[serde(default)]
    pub sensor_events: SensorEvents3d,
    /// Pending sensor end events queued between steps.
    #[serde(default)]
    pub pending_sensor_end_events: Vec<SensorEndEvent3d>,
    /// Contact events from the last step.
    #[serde(default)]
    pub contact_events: ContactEvents3d,
    /// Pending contact end events queued between steps.
    #[serde(default)]
    pub pending_end_events: Vec<ContactEndEvent3d>,
    /// Body movement and sleep events from the last step.
    #[serde(default)]
    pub body_events: BodyEvents3d,
    /// Joint reaction events from the last step.
    #[serde(default)]
    pub joint_events: JointEvents3d,
    /// Persistent solver sets (`STATIC_SET`, `DISABLED_SET`, `AWAKE_SET`, and sleeping sets).
    #[serde(default)]
    pub sets: Vec<crate::island::SolverSet3d>,
    /// Free sleeping solver-set indices.
    #[serde(default)]
    pub free_sets: Vec<u32>,
    /// Persistent islands.
    #[serde(default)]
    pub islands: Vec<crate::island::Island3d>,
    /// Free island indices.
    #[serde(default)]
    pub free_islands: Vec<u32>,
    /// Live island count.
    #[serde(default)]
    pub island_count: usize,
    /// Island scheduled to be split on the next step, or `NULL`.
    #[serde(default = "null_u32")]
    pub island_to_split: u32,
    /// Per-body solver-set and island links.
    #[serde(default)]
    pub body_links: Vec<crate::island::BodyLinks3d>,
    /// Per-joint solver-set and island links.
    #[serde(default)]
    pub joint_links: Vec<crate::island::JointLinks3d>,
    /// Persistent constraint graph of awake contacts and joints.
    #[serde(default)]
    pub constraint_graph: crate::constraint_graph::ConstraintGraph3d,
    /// Broadphase dynamic trees (`STATIC_TREE`, `KINEMATIC_TREE`, `DYNAMIC_TREE`).
    #[serde(default)]
    pub trees: [crate::bvh::DynamicTree; 3],
    /// Broadphase move buffer of collider indices whose fat AABBs moved since the last pair search.
    #[serde(default)]
    pub move_buffer: Vec<u32>,
    /// Body indices touched via `body_mut` since the last step.
    #[serde(default)]
    pub touched_bodies: Vec<u32>,
    /// Gravity vector.
    #[serde(default = "default_gravity")]
    pub gravity: Vec3,
    /// Solver configuration parameters.
    #[serde(default)]
    pub solver_config: SolverConfig3d,
}

impl WorldSnapshot3d {
    /// Returns the 64-bit FNV-1a hash of the simulation state represented by this snapshot.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        compute_state_hash(StateHashInput {
            bodies: &self.bodies,
            body_ids: &self.body_ids,
            body_links: &self.body_links,
            contacts: &self.contacts,
            joints: &self.joints,
            joint_ids: &self.joint_ids,
            joint_links: &self.joint_links,
            collider_ids: &self.collider_ids,
            sensor_overlaps: &self.sensor_overlaps,
            sets: &self.sets,
            free_sets: &self.free_sets,
            islands: &self.islands,
            free_islands: &self.free_islands,
            island_count: self.island_count,
            island_to_split: self.island_to_split,
        })
    }
}

pub(crate) struct StateHashInput<'a> {
    pub bodies: &'a [RigidBody],
    pub body_ids: &'a IdPool<RigidBody>,
    pub body_links: &'a [crate::island::BodyLinks3d],
    pub contacts: &'a ContactManager3d,
    pub joints: &'a [Option<Joint>],
    pub joint_ids: &'a IdPool<Joint>,
    pub joint_links: &'a [crate::island::JointLinks3d],
    pub collider_ids: &'a IdPool<Collider>,
    pub sensor_overlaps: &'a [Vec<ColliderId3d>],
    pub sets: &'a [crate::island::SolverSet3d],
    pub free_sets: &'a [u32],
    pub islands: &'a [crate::island::Island3d],
    pub free_islands: &'a [u32],
    pub island_count: usize,
    pub island_to_split: u32,
}

pub(crate) fn compute_state_hash(input: StateHashInput<'_>) -> u64 {
    let mut hash = Fnv64::new();

    // 1. Bodies & body links
    let slot_count = input.body_ids.slot_count().max(input.bodies.len());
    for slot in 0..slot_count as u32 {
        let active = if input.body_ids.is_empty() {
            (slot as usize) < input.bodies.len()
        } else {
            input.body_ids.handle_at(slot).is_some()
        };
        if !active || slot as usize >= input.bodies.len() {
            continue;
        }
        let b = &input.bodies[slot as usize];
        hash.ints(&[
            slot,
            u32::from(b.is_sleeping),
            u32::from(b.enabled),
            b.body_type as u32,
        ]);
        hash.floats(&[
            b.position.x,
            b.position.y,
            b.position.z,
            b.rotation.x,
            b.rotation.y,
            b.rotation.z,
            b.rotation.w,
            b.linear_velocity.x,
            b.linear_velocity.y,
            b.linear_velocity.z,
            b.angular_velocity.x,
            b.angular_velocity.y,
            b.angular_velocity.z,
            b.force.x,
            b.force.y,
            b.force.z,
            b.torque.x,
            b.torque.y,
            b.torque.z,
            b.sleep_time,
        ]);
        if let Some(links) = input.body_links.get(slot as usize) {
            hash.ints(&[
                links.body_type as u32,
                links.set,
                links.local,
                links.island,
                links.island_index,
            ]);
        }
    }

    // 2. Contacts & contact graph/island links
    for contact in input.contacts.iter() {
        hash.ints(&[
            contact.collider_a,
            contact.collider_b,
            contact.body_a,
            contact.body_b,
            u32::from(contact.is_touching),
            contact.set,
            contact.color,
            contact.local,
            contact.island,
            contact.island_index,
            contact.manifolds.len() as u32,
        ]);
        for manifold in &contact.manifolds {
            hash.floats(&[
                manifold.normal.x,
                manifold.normal.y,
                manifold.normal.z,
                manifold.friction_impulse.x,
                manifold.friction_impulse.y,
                manifold.friction_impulse.z,
            ]);
            hash.ints(&[manifold.points.len() as u32]);
            for point in manifold.points.as_slice() {
                hash.floats(&[
                    point.point.x,
                    point.point.y,
                    point.point.z,
                    point.normal.x,
                    point.normal.y,
                    point.normal.z,
                    point.separation,
                    point.normal_impulse,
                    point.total_normal_impulse,
                ]);
            }
        }
    }

    // 3. Joints & joint links
    for slot in 0..input.joints.len() {
        if !input.joint_ids.is_empty() && input.joint_ids.handle_at(slot as u32).is_none() {
            continue;
        }
        if let Some(Some(joint)) = input.joints.get(slot) {
            hash.ints(&[slot as u32]);
            joint.hash_impulses(&mut |v| hash.floats(&[v]));
            if let Some(links) = input.joint_links.get(slot) {
                hash.ints(&[
                    links.set,
                    links.local,
                    links.color,
                    links.island,
                    links.island_index,
                ]);
            }
        }
    }

    // 4. Sensor overlaps
    for (slot, overlaps) in input.sensor_overlaps.iter().enumerate() {
        if overlaps.is_empty() {
            continue;
        }
        hash.ints(&[slot as u32, overlaps.len() as u32]);
        for cid in overlaps {
            hash.ints(&[cid.index()]);
        }
    }

    // 5. Solver sets & persistent islands
    for set in input.sets {
        if set.bodies.is_empty()
            && set.contacts.is_empty()
            && set.joints.is_empty()
            && set.islands.is_empty()
        {
            continue;
        }
        hash.ints(&[
            set.set_index,
            set.bodies.len() as u32,
            set.contacts.len() as u32,
            set.joints.len() as u32,
            set.islands.len() as u32,
        ]);
        hash.ints(&set.bodies);
        hash.ints(&set.contacts);
        hash.ints(&set.joints);
        hash.ints(&set.islands);
    }
    hash.ints(&[input.free_sets.len() as u32]);

    hash.ints(&[input.island_count as u32, input.island_to_split]);
    for (idx, island) in input.islands.iter().enumerate() {
        if island.set == crate::contact::NULL {
            continue;
        }
        hash.ints(&[
            idx as u32,
            island.set,
            island.local,
            island.constraint_remove_count,
            island.bodies.len() as u32,
            island.contacts.len() as u32,
            island.joints.len() as u32,
        ]);
        hash.ints(&island.bodies);
        for c in &island.contacts {
            hash.ints(&[c.id, c.body_a, c.body_b]);
        }
        for j in &island.joints {
            hash.ints(&[j.id, j.body_a, j.body_b]);
        }
    }
    hash.ints(&[input.free_islands.len() as u32]);

    // 6. Structure sizes and live counts
    for (slots, live) in [
        (input.body_ids.slot_count(), input.body_ids.len()),
        (input.collider_ids.slot_count(), input.collider_ids.len()),
        (input.joint_ids.slot_count(), input.joint_ids.len()),
        (input.contacts.slot_count(), input.contacts.len()),
    ] {
        hash.ints(&[slots as u32, live as u32]);
    }

    hash.0
}
