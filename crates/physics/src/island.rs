//! 3D solver sets and persistent islands: a port of Box3D's `solver_set.c` and `island.c`.
//!
//! Every body belongs to one solver set: static bodies to [`STATIC_SET`], disabled bodies to
//! [`DISABLED_SET`], awake dynamic and kinematic bodies to [`AWAKE_SET`], and each island that
//! fell asleep to a sleeping set (`>= FIRST_SLEEPING_SET`) of its own. The simulation step visits
//! [`AWAKE_SET`] only. Contacts and joints follow their bodies: an awake touching contact or awake
//! joint lives in a color of [` crate::constraint_graph::ConstraintGraph3d`], a non-touching
//! contact with an awake body lives in `sets[AWAKE_SET].contacts`, a non-touching contact between
//! resting bodies waits in `sets[DISABLED_SET].contacts`, and touching contacts and joints of a
//! sleeping island rest in that island's sleeping solver set.
//!
//! Islands persist across steps: a contact that begins touching or a joint that is created links
//! its bodies' islands, merging them (and waking any sleeping set involved). A contact that stops
//! touching or a joint that is destroyed increments `constraint_remove_count` on its island
//! without immediately traversing the graph. Only when an island with removed constraints has a
//! body ready to sleep is the island split via union-find (`split_island`), at most one island per
//! step, the sleepiest candidate first.

use crate::body::{BodyType, RigidBody};
use crate::contact::NULL;
use crate::world::PhysicsWorld;
use serde::{Deserialize, Serialize};

/// Minimum linear speed squared below which a body is considered resting for sleep (0.05 m/s).
pub const SLEEP_LINEAR_TOLERANCE_SQ: f32 = 0.05 * 0.05;
/// Minimum angular speed squared below which a body is considered resting for sleep (0.05 rad/s).
pub const SLEEP_ANGULAR_TOLERANCE_SQ: f32 = 0.05 * 0.05;
/// Duration in seconds of continuous rest before an island goes to sleep (Box3D `B3_TIME_TO_SLEEP`).
pub const TIME_TO_SLEEP: f32 = 0.5;

/// The set of static bodies (Box3D `b3_staticSet`).
pub const STATIC_SET: u32 = 0;
/// The set of disabled bodies and their joints, and of non-touching contacts between resting
/// bodies (Box3D `b3_disabledSet`).
pub const DISABLED_SET: u32 = 1;
/// The set of awake bodies (Box3D `b3_awakeSet`).
pub const AWAKE_SET: u32 = 2;
/// The first sleeping set (Box3D `b3_firstSleepingSet`).
pub const FIRST_SLEEPING_SET: u32 = 3;

const fn null_u32() -> u32 {
    NULL
}

/// Body links tracking solver set and persistent island membership (Box3D `b3Body`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyLinks3d {
    /// Solver set ID (0 = static, 1 = disabled, 2 = awake, >= 3 = sleeping).
    pub set: u32,
    /// Index in `sets[set].bodies`.
    #[serde(default = "null_u32")]
    pub local: u32,
    /// Persistent island ID, or `NULL` for static and disabled bodies.
    pub island: u32,
    /// Index in `islands[island].bodies`.
    #[serde(default = "null_u32")]
    pub island_index: u32,
    /// Index of this body's move event in the current step, or `NULL`.
    #[serde(default = "null_u32")]
    pub move_event: u32,
    /// Whether the body moved fast in the last step (disables contact recycling).
    #[serde(default)]
    pub is_fast: bool,
    /// Whether `body_mut` was called on this body since the last step.
    #[serde(default)]
    pub touched: bool,
}

impl Default for BodyLinks3d {
    fn default() -> Self {
        Self {
            set: NULL,
            local: NULL,
            island: NULL,
            island_index: NULL,
            move_event: NULL,
            is_fast: false,
            touched: false,
        }
    }
}

/// Solver set representing a group of bodies, contacts, joints and islands that sleep and wake
/// together (Box3D `b3SolverSet`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverSet3d {
    /// Solver set ID (0 = static, 1 = disabled, 2 = awake, >= 3 = sleeping).
    pub set_index: u32,
    /// Body slots in this solver set.
    pub bodies: Vec<u32>,
    /// Contact slots: in the awake and disabled sets the contacts that do not touch, in a
    /// sleeping set the touching ones.
    pub contacts: Vec<u32>,
    /// Joint slots; awake joints are in the constraint graph instead.
    pub joints: Vec<u32>,
    /// Island IDs in this solver set.
    pub islands: Vec<u32>,
}

impl SolverSet3d {
    /// Creates an empty solver set with the given `set_index`.
    #[must_use]
    pub fn new(set_index: u32) -> Self {
        Self {
            set_index,
            bodies: Vec::new(),
            contacts: Vec::new(),
            joints: Vec::new(),
            islands: Vec::new(),
        }
    }

    /// Creates an empty solver set with pre-allocated capacities.
    #[must_use]
    pub fn with_capacity(set_index: u32, capacity: usize) -> Self {
        Self {
            set_index,
            bodies: Vec::with_capacity(capacity),
            contacts: Vec::with_capacity(capacity),
            joints: Vec::with_capacity(capacity),
            islands: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.bodies.clear();
        self.contacts.clear();
        self.joints.clear();
        self.islands.clear();
    }
}

/// Where a joint is: its solver set (the constraint graph when awake) and persistent island.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JointLinks3d {
    /// Solver set ID (1 = disabled, 2 = awake, >= 3 = sleeping).
    pub set: u32,
    /// Index in the set's `joints`, or in its graph color's `joint_indices` when awake.
    pub local: u32,
    /// Constraint-graph color of an awake joint, else `NULL`.
    pub color: u32,
    /// Persistent island ID, or `NULL` when unlinked.
    pub island: u32,
    /// Index in `islands[island].joints`, or `NULL` when unlinked.
    pub island_index: u32,
}

impl Default for JointLinks3d {
    fn default() -> Self {
        Self::UNLINKED
    }
}

impl JointLinks3d {
    /// Initial unlinked state before the joint is added to a solver set or island.
    pub const UNLINKED: Self = Self {
        set: NULL,
        local: NULL,
        color: NULL,
        island: NULL,
        island_index: NULL,
    };
}

/// A constraint in a persistent island, with its bodies so splitting reads only the island.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IslandLink3d {
    /// Contact or joint slot index.
    pub id: u32,
    /// First endpoint body slot.
    pub body_a: u32,
    /// Second endpoint body slot.
    pub body_b: u32,
}

/// A persistent island of connected dynamic and kinematic bodies (Box3D `b3Island`).
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Island3d {
    /// Solver set ID owning this island (`AWAKE_SET` or `>= FIRST_SLEEPING_SET`).
    pub set: u32,
    /// Index of this island in `sets[set].islands`.
    pub local: u32,
    /// Body slots belonging to this island.
    pub bodies: Vec<u32>,
    /// Touching contacts linking bodies in this island.
    pub contacts: Vec<IslandLink3d>,
    /// Joints linking bodies in this island.
    pub joints: Vec<IslandLink3d>,
    /// Contacts and joints removed since the island was last whole; a split may be due.
    pub constraint_remove_count: u32,
}

/// Working arrays of an island split, kept on `PhysicsWorld` to avoid allocating in `step`.
#[derive(Debug, Default, Clone)]
pub(crate) struct SplitScratch3d {
    parents: Vec<u32>,
    ranks: Vec<u32>,
    contact_counts: Vec<u32>,
    joint_counts: Vec<u32>,
    root_map: Vec<u32>,
    component_bodies: Vec<u32>,
    component_contacts: Vec<u32>,
    component_joints: Vec<u32>,
    island_ids: Vec<u32>,
    bodies: Vec<u32>,
    contacts: Vec<IslandLink3d>,
    joints: Vec<IslandLink3d>,
}

/// Box3D `b3IslandFindParent`: the root of `node`, halving the path.
fn find_root(parents: &mut [u32], mut node: u32) -> u32 {
    while parents[node as usize] != node {
        let grand_parent = parents[parents[node as usize] as usize];
        parents[node as usize] = grand_parent;
        node = grand_parent;
    }
    node
}

/// Box3D `b3IslandUnion`: joins the components of `a` and `b` by rank, summing their constraint
/// counts.
fn union(scratch: &mut SplitScratch3d, a: u32, b: u32) {
    let root_a = find_root(&mut scratch.parents, a) as usize;
    let root_b = find_root(&mut scratch.parents, b) as usize;
    if root_a == root_b {
        return;
    }
    let (child, root) = if scratch.ranks[root_a] < scratch.ranks[root_b] {
        (root_a, root_b)
    } else {
        if scratch.ranks[root_a] == scratch.ranks[root_b] {
            scratch.ranks[root_a] += 1;
        }
        (root_b, root_a)
    };
    scratch.parents[child] = root as u32;
    scratch.contact_counts[root] += scratch.contact_counts[child];
    scratch.joint_counts[root] += scratch.joint_counts[child];
}

impl PhysicsWorld {
    /// Allocates a new or recycled sleeping solver set (Box3D `b3AllocId` on the set pool).
    pub(crate) fn alloc_set(&mut self) -> u32 {
        if let Some(id) = self.free_sets.pop() {
            self.sets[id as usize].clear();
            self.sets[id as usize].set_index = id;
            id
        } else {
            let id = self.sets.len() as u32;
            self.sets.push(SolverSet3d::new(id));
            id
        }
    }

    /// Adds body slot `slot` to solver set `set`.
    pub(crate) fn add_body_to_set(&mut self, slot: u32, set: u32) {
        let links = &mut self.body_links[slot as usize];
        links.set = set;
        links.local = self.sets[set as usize].bodies.len() as u32;
        self.sets[set as usize].bodies.push(slot);
    }

    /// Removes body slot `slot` from its current solver set.
    pub(crate) fn remove_body_from_set(&mut self, slot: u32) {
        let BodyLinks3d { set, local, .. } = self.body_links[slot as usize];
        if set == NULL {
            return;
        }
        let bodies = &mut self.sets[set as usize].bodies;
        bodies.swap_remove(local as usize);
        if let Some(&moved) = bodies.get(local as usize) {
            self.body_links[moved as usize].local = local;
        }
        let links = &mut self.body_links[slot as usize];
        links.set = NULL;
        links.local = NULL;
    }

    /// Moves body slot `slot` to solver set `set` (Box3D `b3TransferBody`).
    pub(crate) fn transfer_body(&mut self, slot: u32, set: u32) {
        if self.body_links[slot as usize].set == set {
            return;
        }
        self.remove_body_from_set(slot);
        self.add_body_to_set(slot, set);
    }

    /// Whether body slots `a` and `b` are dynamic.
    pub(crate) fn dynamics(&self, (a, b): (u32, u32)) -> (bool, bool) {
        let dynamic = |slot: u32| {
            self.bodies
                .get(slot as usize)
                .is_some_and(|b| b.body_type == BodyType::Dynamic)
        };
        (dynamic(a), dynamic(b))
    }

    /// Adds contact slot `slot` to set `set`, or, if it is awake and touching (`in_graph`), to a
    /// color of the constraint graph (Box3D `b3AddContactToGraph`).
    pub(crate) fn add_contact_to_set(&mut self, slot: u32, set: u32, in_graph: bool) {
        let contact = self.contacts.at(slot);
        let (key, bodies) = (
            (contact.collider_a, contact.collider_b),
            (contact.body_a, contact.body_b),
        );
        let (color, local) = if in_graph {
            let dynamics = self.dynamics(bodies);
            self.constraint_graph
                .add_contact(key, slot, bodies, dynamics)
        } else {
            let contacts = &mut self.sets[set as usize].contacts;
            contacts.push(slot);
            (NULL, contacts.len() as u32 - 1)
        };
        let contact = self.contacts.at_mut(slot);
        contact.set = set;
        contact.local = local;
        contact.color = color;
    }

    /// Removes contact slot `slot` from its solver set or constraint-graph color.
    pub(crate) fn remove_contact_from_set(&mut self, slot: u32) {
        let contact = self.contacts.at(slot);
        let (set, local, color) = (contact.set, contact.local, contact.color);
        if set == NULL {
            return;
        }
        let moved = if color == NULL {
            let contacts = &mut self.sets[set as usize].contacts;
            contacts.swap_remove(local as usize);
            contacts.get(local as usize).copied()
        } else {
            self.constraint_graph.remove_contact(color, local)
        };
        if let Some(moved) = moved {
            self.contacts.at_mut(moved).local = local;
        }
        let contact = self.contacts.at_mut(slot);
        contact.set = NULL;
        contact.local = NULL;
        contact.color = NULL;
    }

    /// Adds joint slot `slot` to solver set `set`; an awake joint goes into a color of the
    /// constraint graph (Box3D `b3CreateJointInGraph`).
    pub(crate) fn add_joint_to_set(&mut self, slot: u32, set: u32) {
        let (color, local) = if set == AWAKE_SET {
            let joint = self.joints[slot as usize].as_ref().expect("live joint");
            let bodies = (joint.body_a(), joint.body_b());
            let dynamics = self.dynamics(bodies);
            self.constraint_graph.add_joint(slot, bodies, dynamics)
        } else {
            let joints = &mut self.sets[set as usize].joints;
            joints.push(slot);
            (NULL, joints.len() as u32 - 1)
        };
        let links = &mut self.joint_links[slot as usize];
        links.set = set;
        links.local = local;
        links.color = color;
    }

    /// Removes joint slot `slot` from its solver set or constraint-graph color.
    pub(crate) fn remove_joint_from_set(&mut self, slot: u32) {
        let JointLinks3d {
            set, local, color, ..
        } = self.joint_links[slot as usize];
        if set == NULL {
            return;
        }
        let moved = if color == NULL {
            let joints = &mut self.sets[set as usize].joints;
            joints.swap_remove(local as usize);
            joints.get(local as usize).copied()
        } else {
            self.constraint_graph.remove_joint(color, local)
        };
        if let Some(moved) = moved {
            self.joint_links[moved as usize].local = local;
        }
        let links = &mut self.joint_links[slot as usize];
        links.set = NULL;
        links.local = NULL;
        links.color = NULL;
    }

    /// Moves joint slot `slot` to solver set `set` (Box3D `b3TransferJoint`).
    pub(crate) fn transfer_joint(&mut self, slot: u32, set: u32) {
        if self.joint_links[slot as usize].set == set {
            return;
        }
        self.remove_joint_from_set(slot);
        self.add_joint_to_set(slot, set);
    }

    /// Wakes body slot `slot`'s sleeping set if it is asleep (Box3D `b3WakeBody`). Returns
    /// whether it was asleep.
    pub(crate) fn wake_body_slot(&mut self, slot: u32) -> bool {
        let Some(&links) = self.body_links.get(slot as usize) else {
            return false;
        };
        if links.set != NULL && links.set >= FIRST_SLEEPING_SET {
            self.wake_set(links.set);
            return true;
        }
        if let Some(body) = self.bodies.get_mut(slot as usize) {
            if body.body_type != BodyType::Static {
                body.is_sleeping = false;
                body.sleep_time = 0.0;
            }
        }
        false
    }

    /// Wakes every body, contact, joint and island of sleeping set `set` and frees the set
    /// (Box3D `b3WakeSolverSet`).
    pub(crate) fn wake_set(&mut self, set: u32) {
        if set < FIRST_SLEEPING_SET || set as usize >= self.sets.len() {
            return;
        }
        let mut sleeping = std::mem::take(&mut self.sets[set as usize]);
        sleeping.set_index = set;
        for &slot in &sleeping.bodies {
            if let Some(body) = self.bodies.get_mut(slot as usize) {
                body.wake_up();
            }
            self.body_links[slot as usize].is_fast = false;
            self.add_body_to_set(slot, AWAKE_SET);

            let mut contact_slot = self.contacts.body_head(slot);
            while contact_slot != NULL {
                let contact = self.contacts.at(contact_slot);
                let end = usize::from(contact.body_b == slot);
                let next = contact.edges[end].next;
                if contact.set == DISABLED_SET {
                    self.remove_contact_from_set(contact_slot);
                    self.add_contact_to_set(contact_slot, AWAKE_SET, false);
                }
                contact_slot = next;
            }
        }
        for &contact_slot in &sleeping.contacts {
            let contact = self.contacts.at_mut(contact_slot);
            contact.set = NULL;
            contact.local = NULL;
            contact.color = NULL;
            self.add_contact_to_set(contact_slot, AWAKE_SET, true);
        }
        for &joint in &sleeping.joints {
            self.joint_links[joint as usize].set = NULL;
            self.joint_links[joint as usize].local = NULL;
            self.joint_links[joint as usize].color = NULL;
            self.add_joint_to_set(joint, AWAKE_SET);
        }
        for &island in &sleeping.islands {
            let awake = &mut self.sets[AWAKE_SET as usize].islands;
            self.islands[island as usize].set = AWAKE_SET;
            self.islands[island as usize].local = awake.len() as u32;
            awake.push(island);
        }
        sleeping.clear();
        self.sets[set as usize] = sleeping;
        self.free_sets.push(set);
    }

    /// Puts island `island` to sleep in a new sleeping set, unless it is waiting to be split
    /// (Box3D `b3TrySleepIsland`).
    pub(crate) fn try_sleep_island(&mut self, island: u32) {
        let (remove_count, body_count) = {
            let isl = &self.islands[island as usize];
            (isl.constraint_remove_count, isl.bodies.len())
        };
        if remove_count > 0 && body_count > 1 {
            return;
        }
        let set = self.alloc_set();
        let bodies = std::mem::take(&mut self.islands[island as usize].bodies);
        for &slot in &bodies {
            let move_event = self.body_links[slot as usize].move_event;
            if move_event != NULL {
                if let Some(ev) = self.body_events.move_events.get_mut(move_event as usize) {
                    ev.fell_asleep = true;
                }
                self.body_links[slot as usize].move_event = NULL;
            }
            self.body_links[slot as usize].is_fast = false;
            if let Some(body) = self.bodies.get_mut(slot as usize) {
                body.put_to_sleep();
            }
            self.transfer_body(slot, set);

            let mut contact_slot = self.contacts.body_head(slot);
            while contact_slot != NULL {
                let contact = self.contacts.at(contact_slot);
                let end = usize::from(contact.body_b == slot);
                let next = contact.edges[end].next;
                // Touching contacts move with the island below.
                if contact.color != NULL || contact.set == DISABLED_SET {
                    contact_slot = next;
                    continue;
                }
                let other = contact.edges[end ^ 1].body;
                if self.body_links[other as usize].set == AWAKE_SET {
                    contact_slot = next;
                    continue;
                }
                self.remove_contact_from_set(contact_slot);
                self.add_contact_to_set(contact_slot, DISABLED_SET, false);
                contact_slot = next;
            }
        }
        self.islands[island as usize].bodies = bodies;

        for i in 0..self.islands[island as usize].contacts.len() {
            let contact_slot = self.islands[island as usize].contacts[i].id;
            self.remove_contact_from_set(contact_slot);
            self.add_contact_to_set(contact_slot, set, false);
        }
        for i in 0..self.islands[island as usize].joints.len() {
            let joint = self.islands[island as usize].joints[i].id;
            self.transfer_joint(joint, set);
        }

        // Move the island itself from AWAKE_SET to `set`.
        let local = self.islands[island as usize].local;
        let awake = &mut self.sets[AWAKE_SET as usize].islands;
        awake.swap_remove(local as usize);
        if let Some(&moved) = awake.get(local as usize) {
            self.islands[moved as usize].local = local;
        }
        self.islands[island as usize].set = set;
        self.islands[island as usize].local = 0;
        self.sets[set as usize].islands.push(island);
        if self.island_to_split == island {
            self.island_to_split = NULL;
        }
    }

    /// Merges sleeping set `b` into sleeping set `a`, for a joint created between them (Box3D
    /// `b3MergeSolverSets`). The larger set keeps the bodies.
    pub(crate) fn merge_sets(&mut self, a: u32, b: u32) {
        let (keep, gone) =
            if self.sets[a as usize].bodies.len() >= self.sets[b as usize].bodies.len() {
                (a, b)
            } else {
                (b, a)
            };
        let mut merged = std::mem::take(&mut self.sets[gone as usize]);
        merged.set_index = gone;
        for &slot in &merged.bodies {
            self.add_body_to_set(slot, keep);
        }
        for &contact in &merged.contacts {
            self.add_contact_to_set(contact, keep, false);
        }
        for &joint in &merged.joints {
            self.add_joint_to_set(joint, keep);
        }
        for &island in &merged.islands {
            let islands = &mut self.sets[keep as usize].islands;
            self.islands[island as usize].set = keep;
            self.islands[island as usize].local = islands.len() as u32;
            islands.push(island);
        }
        merged.clear();
        self.sets[gone as usize] = merged;
        self.free_sets.push(gone);
    }

    /// Creates a new island in solver set `set` (Box3D `b3CreateIsland`).
    pub(crate) fn create_island(&mut self, set: u32) -> u32 {
        let id = if let Some(id) = self.free_islands.pop() {
            id
        } else {
            self.islands.push(Island3d::default());
            self.islands.len() as u32 - 1
        };
        let islands = &mut self.sets[set as usize].islands;
        let island = &mut self.islands[id as usize];
        island.set = set;
        island.local = islands.len() as u32;
        island.bodies.clear();
        island.contacts.clear();
        island.joints.clear();
        island.constraint_remove_count = 0;
        islands.push(id);
        self.island_count += 1;
        id
    }

    /// Frees island `id`, which must be empty (Box3D `b3DestroyIsland`).
    pub(crate) fn destroy_island(&mut self, id: u32) {
        if self.island_to_split == id {
            self.island_to_split = NULL;
        }
        let (set, local) = {
            let island = &self.islands[id as usize];
            (island.set, island.local)
        };
        if set != NULL {
            let islands = &mut self.sets[set as usize].islands;
            islands.swap_remove(local as usize);
            if let Some(&moved) = islands.get(local as usize) {
                self.islands[moved as usize].local = local;
            }
        }
        let island = &mut self.islands[id as usize];
        island.bodies.clear();
        island.contacts.clear();
        island.joints.clear();
        island.constraint_remove_count = 0;
        island.set = NULL;
        island.local = NULL;
        self.free_islands.push(id);
        self.island_count -= 1;
    }

    /// Gives body slot `slot` an island of its own in solver set `set` (Box3D
    /// `b3CreateIslandForBody`).
    pub(crate) fn create_island_for_body(&mut self, set: u32, slot: u32) {
        let island = self.create_island(set);
        self.islands[island as usize].bodies.push(slot);
        let links = &mut self.body_links[slot as usize];
        links.island = island;
        links.island_index = 0;
    }

    /// Takes body slot `slot` out of its island, destroying the island if it empties (Box3D
    /// `b3RemoveBodyFromIsland`).
    pub(crate) fn remove_body_from_island(&mut self, slot: u32) {
        let BodyLinks3d {
            island,
            island_index,
            ..
        } = self.body_links[slot as usize];
        if island == NULL {
            return;
        }
        let bodies = &mut self.islands[island as usize].bodies;
        bodies.swap_remove(island_index as usize);
        if let Some(&moved) = bodies.get(island_index as usize) {
            self.body_links[moved as usize].island_index = island_index;
        }
        if self.islands[island as usize].bodies.is_empty() {
            self.destroy_island(island);
        }
        let links = &mut self.body_links[slot as usize];
        links.island = NULL;
        links.island_index = NULL;
    }

    /// Merges islands `a` and `b`, either of which may be `NULL`, into the one with more bodies,
    /// and returns it (Box3D `b3MergeIslands`).
    pub(crate) fn merge_islands(&mut self, a: u32, b: u32) -> u32 {
        if a == b || b == NULL {
            return a;
        }
        if a == NULL {
            return b;
        }
        let (big, small) =
            if self.islands[a as usize].bodies.len() >= self.islands[b as usize].bodies.len() {
                (a, b)
            } else {
                (b, a)
            };
        let small_island = std::mem::take(&mut self.islands[small as usize]);
        for &slot in &small_island.bodies {
            let links = &mut self.body_links[slot as usize];
            links.island = big;
            links.island_index = self.islands[big as usize].bodies.len() as u32;
            self.islands[big as usize].bodies.push(slot);
        }
        for &link in &small_island.contacts {
            let contact = self.contacts.at_mut(link.id);
            contact.island = big;
            contact.island_index = self.islands[big as usize].contacts.len() as u32;
            self.islands[big as usize].contacts.push(link);
        }
        for &link in &small_island.joints {
            let links = &mut self.joint_links[link.id as usize];
            links.island = big;
            links.island_index = self.islands[big as usize].joints.len() as u32;
            self.islands[big as usize].joints.push(link);
        }
        self.islands[big as usize].constraint_remove_count += small_island.constraint_remove_count;
        let Island3d {
            set,
            local,
            mut bodies,
            mut contacts,
            mut joints,
            ..
        } = small_island;
        bodies.clear();
        contacts.clear();
        joints.clear();
        self.islands[small as usize] = Island3d {
            set,
            local,
            bodies,
            contacts,
            joints,
            constraint_remove_count: 0,
        };
        self.destroy_island(small);
        big
    }

    /// Links touching contact slot `slot` into its bodies' islands, waking a sleeping body that
    /// an awake one touches and merging the islands (Box3D `b3LinkContact`).
    pub(crate) fn link_contact(&mut self, slot: u32) {
        let contact = self.contacts.at(slot);
        let (a, b) = (contact.body_a, contact.body_b);
        let (set_a, set_b) = (
            self.body_links[a as usize].set,
            self.body_links[b as usize].set,
        );
        if set_a == AWAKE_SET && set_b >= FIRST_SLEEPING_SET {
            self.wake_set(set_b);
        } else if set_b == AWAKE_SET && set_a >= FIRST_SLEEPING_SET {
            self.wake_set(set_a);
        } else if set_a >= FIRST_SLEEPING_SET && set_b >= FIRST_SLEEPING_SET && set_a != set_b {
            self.merge_sets(set_a, set_b);
        }
        let island = self.merge_islands(
            self.body_links[a as usize].island,
            self.body_links[b as usize].island,
        );
        if island == NULL {
            return;
        }
        let contacts = &mut self.islands[island as usize].contacts;
        let contact = self.contacts.at_mut(slot);
        contact.island = island;
        contact.island_index = contacts.len() as u32;
        contacts.push(IslandLink3d {
            id: slot,
            body_a: a,
            body_b: b,
        });
    }

    /// Unlinks contact slot `slot` from its island, counting a removed constraint (Box3D
    /// `b3UnlinkContact`).
    pub(crate) fn unlink_contact(&mut self, slot: u32) {
        let contact = self.contacts.at(slot);
        let (island, index) = (contact.island, contact.island_index);
        if island == NULL {
            return;
        }
        let contacts = &mut self.islands[island as usize].contacts;
        contacts.swap_remove(index as usize);
        if let Some(moved) = contacts.get(index as usize) {
            self.contacts.at_mut(moved.id).island_index = index;
        }
        self.islands[island as usize].constraint_remove_count += 1;
        let contact = self.contacts.at_mut(slot);
        contact.island = NULL;
        contact.island_index = NULL;
    }

    /// Links joint slot `slot` into its bodies' islands, waking a sleeping body jointed to an
    /// awake one and merging the islands (Box3D `b3LinkJoint`).
    pub(crate) fn link_joint(&mut self, slot: u32) {
        let joint = self.joints[slot as usize].as_ref().expect("live joint");
        let (a, b) = (joint.body_a(), joint.body_b());
        let (set_a, set_b) = (
            self.body_links[a as usize].set,
            self.body_links[b as usize].set,
        );
        if set_a == AWAKE_SET && set_b >= FIRST_SLEEPING_SET {
            self.wake_set(set_b);
        } else if set_b == AWAKE_SET && set_a >= FIRST_SLEEPING_SET {
            self.wake_set(set_a);
        } else if set_a >= FIRST_SLEEPING_SET && set_b >= FIRST_SLEEPING_SET && set_a != set_b {
            self.merge_sets(set_a, set_b);
        }
        let island = self.merge_islands(
            self.body_links[a as usize].island,
            self.body_links[b as usize].island,
        );
        if island == NULL {
            return;
        }
        let joints = &mut self.islands[island as usize].joints;
        let links = &mut self.joint_links[slot as usize];
        links.island = island;
        links.island_index = joints.len() as u32;
        joints.push(IslandLink3d {
            id: slot,
            body_a: a,
            body_b: b,
        });
    }

    /// Unlinks joint slot `slot` from its island, if linked, counting a removed constraint
    /// (Box3D `b3UnlinkJoint`).
    pub(crate) fn unlink_joint(&mut self, slot: u32) {
        let JointLinks3d {
            island,
            island_index,
            ..
        } = self.joint_links[slot as usize];
        if island == NULL {
            return;
        }
        let joints = &mut self.islands[island as usize].joints;
        joints.swap_remove(island_index as usize);
        if let Some(moved) = joints.get(island_index as usize) {
            self.joint_links[moved.id as usize].island_index = island_index;
        }
        self.islands[island as usize].constraint_remove_count += 1;
        let links = &mut self.joint_links[slot as usize];
        links.island = NULL;
        links.island_index = NULL;
    }

    /// Splits awake island `base` into its connected parts (Box3D `b3SplitIsland`), by union-find
    /// over its contacts and joints; static bodies connect nothing.
    pub(crate) fn split_island(&mut self, base: u32) {
        let mut scratch = std::mem::take(&mut self.split_scratch);
        {
            let island = &mut self.islands[base as usize];
            std::mem::swap(&mut scratch.bodies, &mut island.bodies);
            std::mem::swap(&mut scratch.contacts, &mut island.contacts);
            std::mem::swap(&mut scratch.joints, &mut island.joints);
        }
        let body_count = scratch.bodies.len();
        if body_count == 0 {
            self.split_scratch = scratch;
            return;
        }
        for list in [
            &mut scratch.parents,
            &mut scratch.ranks,
            &mut scratch.contact_counts,
            &mut scratch.joint_counts,
        ] {
            list.clear();
            list.resize(body_count, 0);
        }
        for (i, parent) in scratch.parents.iter_mut().enumerate() {
            *parent = i as u32;
        }

        // Union over contacts then joints, counting each component's constraints.
        for which in 0..2 {
            let count = if which == 0 {
                scratch.contacts.len()
            } else {
                scratch.joints.len()
            };
            for i in 0..count {
                let link = if which == 0 {
                    scratch.contacts[i]
                } else {
                    scratch.joints[i]
                };
                let index_a = self.body_links[link.body_a as usize].island_index;
                let index_b = self.body_links[link.body_b as usize].island_index;
                let in_island = |index: u32, body: u32| {
                    index != NULL && self.body_links[body as usize].island == base
                };
                let (in_a, in_b) = (
                    in_island(index_a, link.body_a),
                    in_island(index_b, link.body_b),
                );
                let index = if in_a && in_b {
                    union(&mut scratch, index_a, index_b);
                    index_a
                } else if in_a {
                    index_a
                } else if in_b {
                    index_b
                } else {
                    continue;
                };
                let root = find_root(&mut scratch.parents, index) as usize;
                if which == 0 {
                    scratch.contact_counts[root] += 1;
                } else {
                    scratch.joint_counts[root] += 1;
                }
            }
        }

        // Flatten and count the components.
        let mut component_count = 0;
        for i in 0..body_count as u32 {
            let root = find_root(&mut scratch.parents, i);
            scratch.parents[i as usize] = root;
            if root == i {
                component_count += 1;
            }
        }
        if component_count == 1 {
            let island = &mut self.islands[base as usize];
            std::mem::swap(&mut scratch.bodies, &mut island.bodies);
            std::mem::swap(&mut scratch.contacts, &mut island.contacts);
            std::mem::swap(&mut scratch.joints, &mut island.joints);
            island.constraint_remove_count = 0;
            self.split_scratch = scratch;
            return;
        }

        scratch.root_map.clear();
        scratch.root_map.resize(body_count, NULL);
        scratch.component_bodies.clear();
        scratch.component_contacts.clear();
        scratch.component_joints.clear();
        for i in 0..body_count {
            let root = scratch.parents[i] as usize;
            if scratch.root_map[root] == NULL {
                scratch.root_map[root] = scratch.component_bodies.len() as u32;
                scratch.component_bodies.push(0);
                scratch
                    .component_contacts
                    .push(scratch.contact_counts[root]);
                scratch.component_joints.push(scratch.joint_counts[root]);
            }
            scratch.component_bodies[scratch.root_map[root] as usize] += 1;
        }

        scratch.island_ids.clear();
        for i in 0..scratch.component_bodies.len() {
            let id = self.create_island(AWAKE_SET);
            let island = &mut self.islands[id as usize];
            island.bodies.reserve(scratch.component_bodies[i] as usize);
            island
                .contacts
                .reserve(scratch.component_contacts[i] as usize);
            island.joints.reserve(scratch.component_joints[i] as usize);
            scratch.island_ids.push(id);
        }
        for i in 0..body_count {
            let slot = scratch.bodies[i];
            let root = scratch.parents[i] as usize;
            let island = scratch.island_ids[scratch.root_map[root] as usize];
            let links = &mut self.body_links[slot as usize];
            links.island = island;
            links.island_index = self.islands[island as usize].bodies.len() as u32;
            self.islands[island as usize].bodies.push(slot);
        }
        for i in 0..scratch.contacts.len() {
            let link = scratch.contacts[i];
            let island_a = self.body_links[link.body_a as usize].island;
            let island = if island_a == NULL {
                self.body_links[link.body_b as usize].island
            } else {
                island_a
            };
            let contact = self.contacts.at_mut(link.id);
            contact.island = island;
            contact.island_index = self.islands[island as usize].contacts.len() as u32;
            self.islands[island as usize].contacts.push(link);
        }
        for i in 0..scratch.joints.len() {
            let link = scratch.joints[i];
            let island_a = self.body_links[link.body_a as usize].island;
            let island = if island_a == NULL {
                self.body_links[link.body_b as usize].island
            } else {
                island_a
            };
            let links = &mut self.joint_links[link.id as usize];
            links.island = island;
            links.island_index = self.islands[island as usize].joints.len() as u32;
            self.islands[island as usize].joints.push(link);
        }
        self.destroy_island(base);
        scratch.bodies.clear();
        scratch.contacts.clear();
        scratch.joints.clear();
        self.split_scratch = scratch;
    }

    /// Validates that every body, contact, joint and island matches its solver-set and
    /// constraint-graph links (Box3D `b3ValidateSolverSets`).
    #[cfg(debug_assertions)]
    pub(crate) fn validate_sets(&self) {
        for (set_id, set) in self.sets.iter().enumerate() {
            for (local, &slot) in set.bodies.iter().enumerate() {
                let links = &self.body_links[slot as usize];
                assert_eq!(
                    (links.set, links.local),
                    (set_id as u32, local as u32),
                    "body {slot}"
                );
            }
            for (local, &slot) in set.contacts.iter().enumerate() {
                let c = self.contacts.at(slot);
                assert_eq!(
                    (c.set, c.local, c.color),
                    (set_id as u32, local as u32, NULL),
                    "contact {slot}"
                );
            }
            for (local, &slot) in set.joints.iter().enumerate() {
                let l = &self.joint_links[slot as usize];
                assert_eq!(
                    (l.set, l.local, l.color),
                    (set_id as u32, local as u32, NULL),
                    "joint {slot}"
                );
            }
            for (local, &island) in set.islands.iter().enumerate() {
                let i = &self.islands[island as usize];
                assert_eq!(
                    (i.set, i.local),
                    (set_id as u32, local as u32),
                    "island {island}"
                );
            }
        }
        for color in 0..self.constraint_graph.color_count() {
            let graph_color = self.constraint_graph.color(color);
            for (local, &slot) in graph_color.contact_slots.iter().enumerate() {
                let c = self.contacts.at(slot);
                assert_eq!(
                    (c.set, c.local, c.color),
                    (AWAKE_SET, local as u32, color as u32),
                    "graph contact {slot}"
                );
                assert!(c.is_touching, "graph contact {slot} must touch");
                assert!(c.island != NULL, "graph contact {slot} must be linked");
            }
            for (local, &slot) in graph_color.joint_indices.iter().enumerate() {
                let l = &self.joint_links[slot];
                assert_eq!(
                    (l.set, l.local, l.color),
                    (AWAKE_SET, local as u32, color as u32),
                    "graph joint {slot}"
                );
            }
        }
    }
}

/// Legacy graph-based island manager preserved for backward compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IslandManager3d {
    parent: Vec<u32>,
    sleeping: Vec<bool>,
    /// Solver set and island tracking for each body.
    pub body_links: Vec<BodyLinks3d>,
}

impl PartialEq for IslandManager3d {
    fn eq(&self, other: &Self) -> bool {
        self.parent == other.parent
            && self.sleeping == other.sleeping
            && self.body_links == other.body_links
    }
}

impl Eq for IslandManager3d {}

impl IslandManager3d {
    /// Creates an empty island manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a body into the island manager.
    pub fn register_body(&mut self, body_id: u32, sleeping: bool) {
        let idx = body_id as usize;
        if self.parent.len() <= idx {
            self.parent.resize(idx + 1, 0);
            self.sleeping.resize(idx + 1, false);
            self.body_links.resize(idx + 1, BodyLinks3d::default());
        }
        self.parent[idx] = body_id;
        self.sleeping[idx] = sleeping;
        self.body_links[idx] = BodyLinks3d {
            set: if sleeping {
                FIRST_SLEEPING_SET + body_id
            } else {
                AWAKE_SET
            },
            island: body_id,
            ..BodyLinks3d::default()
        };
    }

    /// Returns the solver set ID for `body_id`.
    #[must_use]
    pub fn body_solver_set(&self, body_id: u32) -> u32 {
        self.body_links
            .get(body_id as usize)
            .map_or(AWAKE_SET, |l| l.set)
    }

    /// Sets the solver set for `body_id`.
    pub fn set_body_set(&mut self, body_id: u32, set: u32) {
        let idx = body_id as usize;
        if self.body_links.len() <= idx {
            self.body_links.resize(idx + 1, BodyLinks3d::default());
        }
        self.body_links[idx].set = set;
    }

    /// Updates sleeping states for a slice of bodies (legacy helper).
    pub fn update_islands(
        &mut self,
        _bodies: &mut [RigidBody],
        _contacts: impl Iterator<Item = (u32, u32)>,
        _joints: impl Iterator<Item = (u32, u32)>,
        _dt: f32,
    ) {
    }
}
