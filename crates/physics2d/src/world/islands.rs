//! Solver sets and persistent islands: a port of Box2D v3.2's `solver_set.c` and `island.c`.
//!
//! Every body belongs to one solver set: static bodies to the static set, disabled bodies to
//! the disabled set, awake dynamic and kinematic bodies to the awake set, and each group of
//! islands that fell asleep together to a sleeping set of its own. The step visits the awake
//! set only. Contacts follow their bodies: a contact is awake while either body is awake (in
//! the constraint graph while touching), sleeps in its island's set while touching, and waits
//! in the disabled set while neither body is awake and it does not touch. Joints sleep with
//! their islands too.
//!
//! An island is a group of awake or sleeping dynamic and kinematic bodies connected by touching
//! contacts and joints. Islands persist: a contact that starts touching links its bodies'
//! islands, merging them (and waking a sleeping one), and a joint does the same when created.
//! A contact that stops touching, or a joint that is destroyed, only counts a removed
//! constraint. Only when a body of such an island would sleep is the island split, one island
//! per step, the sleepiest first. An island falls asleep, all at once, when every body in it has
//! been slow for [`TIME_TO_SLEEP`].

use super::{PhysicsWorld2d, NULL};
use crate::body::BodyType;
use serde::{Deserialize, Serialize};

/// The set of static bodies (Box2D `b2_staticSet`).
pub(super) const STATIC_SET: u32 = 0;
/// The set of disabled bodies and their joints, and of contacts between resting bodies (Box2D
/// `b2_disabledSet`).
pub(super) const DISABLED_SET: u32 = 1;
/// The set of awake bodies (Box2D `b2_awakeSet`).
pub(super) const AWAKE_SET: u32 = 2;
/// The first sleeping set (Box2D `b2_firstSleepingSet`).
pub(super) const FIRST_SLEEPING_SET: u32 = 3;

/// How long every body of an island must have been slow before the island sleeps, in seconds
/// (Box2D `B2_TIME_TO_SLEEP`).
pub const TIME_TO_SLEEP: f32 = 0.5;

/// A group of bodies, contacts and joints that sleep and wake together (Box2D `b2SolverSet`).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(super) struct SolverSet2d {
    /// Body slots.
    pub(super) bodies: Vec<u32>,
    /// Contact slots: in the awake and disabled sets the contacts that do not touch, in a
    /// sleeping set the touching ones.
    pub(super) contacts: Vec<u32>,
    /// Joint slots; awake joints are in the constraint graph instead.
    pub(super) joints: Vec<u32>,
    /// Island ids.
    pub(super) islands: Vec<u32>,
}

impl SolverSet2d {
    fn clear(&mut self) {
        self.bodies.clear();
        self.contacts.clear();
        self.joints.clear();
        self.islands.clear();
    }
}

/// Where a body is: its solver set and island, with its sleep threshold.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct BodyLinks2d {
    pub(super) set: u32,
    pub(super) local: u32,
    pub(super) island: u32,
    pub(super) island_index: u32,
    /// Speed of the body's fastest point under which it counts as resting, in length units per
    /// second (Box2D `sleepThreshold`).
    pub(super) sleep_threshold: f32,
    /// The body's move event in the step being taken, or [`NULL`].
    pub(super) move_event: u32,
    /// Whether the body was fast in the last step (Box2D `b2_isFast`); its contacts are not
    /// recycled.
    pub(super) is_fast: bool,
    /// The centre of mass its contacts' anchors are relative to.
    pub(super) local_center: korppi_math::Vec2,
    /// The world centre of mass (Box2D `b2BodySim::center`), kept apart from the origin: a step
    /// moves it by the body's motion and rebuilds the origin from it, so that rounding does not
    /// drift it.
    pub(super) center: korppi_math::Vec2,
    /// The pose the world last gave the body; a different one at the start of a step was set
    /// through `body_mut` and counts as a teleport.
    pub(super) pose: korppi_math::Transform2d,
    /// Cached smallest and largest extents of the body's colliders about `local_center` (Box2D
    /// `b2BodySim::minExtent`, `maxExtent`).
    pub(super) min_extent: f32,
    pub(super) max_extent: f32,
    /// Whether `body_mut` handed the body out since the last step, so its mass may have moved.
    pub(super) touched: bool,
}

impl BodyLinks2d {
    pub(super) const fn new(sleep_threshold: f32) -> Self {
        Self {
            set: NULL,
            local: NULL,
            island: NULL,
            island_index: NULL,
            sleep_threshold,
            move_event: NULL,
            is_fast: false,
            local_center: korppi_math::Vec2::ZERO,
            center: korppi_math::Vec2::ZERO,
            pose: korppi_math::Transform2d {
                translation: korppi_math::Vec2::ZERO,
                rotation: korppi_math::Rot2::IDENTITY,
            },
            min_extent: f32::MAX,
            max_extent: 0.0,
            touched: false,
        }
    }
}

/// Where a joint is: its solver set (the constraint graph when awake) and island.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct JointLinks2d {
    pub(super) set: u32,
    /// Index in the set's joints, or in its graph color's when awake.
    pub(super) local: u32,
    /// The constraint graph color of an awake joint, else [`NULL`].
    pub(super) color: u32,
    pub(super) island: u32,
    pub(super) island_index: u32,
}

impl JointLinks2d {
    pub(super) const UNLINKED: Self = Self {
        set: NULL,
        local: NULL,
        color: NULL,
        island: NULL,
        island_index: NULL,
    };
}

/// A constraint in an island, with its bodies so that splitting reads only the island.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct IslandLink2d {
    pub(super) id: u32,
    pub(super) body_a: u32,
    pub(super) body_b: u32,
}

/// A persistent island (Box2D `b2Island`).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(super) struct Island2d {
    pub(super) set: u32,
    pub(super) local: u32,
    pub(super) bodies: Vec<u32>,
    pub(super) contacts: Vec<IslandLink2d>,
    pub(super) joints: Vec<IslandLink2d>,
    /// Contacts and joints removed since the island was last whole; a split may be due.
    pub(super) constraint_remove_count: u32,
}

/// Working arrays of an island split, kept to avoid allocating.
#[derive(Debug, Default)]
pub(super) struct SplitScratch2d {
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
    contacts: Vec<IslandLink2d>,
    joints: Vec<IslandLink2d>,
}

/// Box2D `b2IslandFindParent`: the root of `node`, halving the path.
fn find_root(parents: &mut [u32], mut node: u32) -> u32 {
    while parents[node as usize] != node {
        let grand_parent = parents[parents[node as usize] as usize];
        parents[node as usize] = grand_parent;
        node = grand_parent;
    }
    node
}

/// Box2D `b2IslandUnion`: joins the components of `a` and `b` by rank, summing their constraint
/// counts.
fn union(scratch: &mut SplitScratch2d, a: u32, b: u32) {
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

impl PhysicsWorld2d {
    /// A new, empty sleeping set (Box2D `b2AllocId` on the set pool).
    pub(super) fn alloc_set(&mut self) -> u32 {
        if let Some(id) = self.free_sets.pop() {
            self.sets[id as usize].clear();
            id
        } else {
            self.sets.push(SolverSet2d::default());
            self.sets.len() as u32 - 1
        }
    }

    /// Adds body slot `slot` to set `set`.
    pub(super) fn add_body_to_set(&mut self, slot: u32, set: u32) {
        let links = &mut self.body_links[slot as usize];
        links.set = set;
        links.local = self.sets[set as usize].bodies.len() as u32;
        self.sets[set as usize].bodies.push(slot);
    }

    /// Removes body slot `slot` from its set.
    pub(super) fn remove_body_from_set(&mut self, slot: u32) {
        let BodyLinks2d { set, local, .. } = self.body_links[slot as usize];
        let bodies = &mut self.sets[set as usize].bodies;
        bodies.swap_remove(local as usize);
        if let Some(&moved) = bodies.get(local as usize) {
            self.body_links[moved as usize].local = local;
        }
        let links = &mut self.body_links[slot as usize];
        links.set = NULL;
        links.local = NULL;
    }

    /// Moves body slot `slot` to set `set` (Box2D `b2TransferBody`).
    pub(super) fn transfer_body(&mut self, slot: u32, set: u32) {
        if self.body_links[slot as usize].set == set {
            return;
        }
        self.remove_body_from_set(slot);
        self.add_body_to_set(slot, set);
    }

    /// Whether body slots `a` and `b` are dynamic.
    fn dynamics(&self, (a, b): (u32, u32)) -> (bool, bool) {
        let dynamic = |slot: u32| self.bodies[slot as usize].body_type == BodyType::Dynamic;
        (dynamic(a), dynamic(b))
    }

    /// Adds contact slot `slot` to set `set`, or, if it is awake and touching, to a color of the
    /// constraint graph (Box2D `b2AddContactToGraph`).
    pub(super) fn add_contact_to_set(&mut self, slot: u32, set: u32, in_graph: bool) {
        let contact = self.contact_manager.at(slot);
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
        let contact = self.contact_manager.at_mut(slot);
        contact.set = set;
        contact.local = local;
        contact.color = color;
    }

    /// Removes contact slot `slot` from its set or graph color.
    pub(super) fn remove_contact_from_set(&mut self, slot: u32) {
        let contact = self.contact_manager.at(slot);
        let (set, local, color) = (contact.set, contact.local, contact.color);
        let moved = if color == NULL {
            let contacts = &mut self.sets[set as usize].contacts;
            contacts.swap_remove(local as usize);
            contacts.get(local as usize).copied()
        } else {
            self.constraint_graph.remove_contact(color, local)
        };
        if let Some(moved) = moved {
            self.contact_manager.at_mut(moved).local = local;
        }
        let contact = self.contact_manager.at_mut(slot);
        contact.set = NULL;
        contact.local = NULL;
        contact.color = NULL;
    }

    /// Adds joint slot `slot` to set `set`; an awake joint goes in a color of the constraint
    /// graph (Box2D `b2CreateJointInGraph`).
    pub(super) fn add_joint_to_set(&mut self, slot: u32, set: u32) {
        let (color, local) = if set == AWAKE_SET {
            let bodies = self.joint_bodies[slot as usize];
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

    /// Removes joint slot `slot` from its set or graph color.
    pub(super) fn remove_joint_from_set(&mut self, slot: u32) {
        let JointLinks2d {
            set, local, color, ..
        } = self.joint_links[slot as usize];
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

    /// Moves joint slot `slot` to set `set` (Box2D `b2TransferJoint`).
    pub(super) fn transfer_joint(&mut self, slot: u32, set: u32) {
        if self.joint_links[slot as usize].set == set {
            return;
        }
        self.remove_joint_from_set(slot);
        self.add_joint_to_set(slot, set);
    }

    /// Wakes body slot `slot`'s sleeping set, if it is asleep (Box2D `b2WakeBody`). Returns
    /// whether it was.
    pub(super) fn wake_body(&mut self, slot: u32) -> bool {
        let set = self.body_links[slot as usize].set;
        if set != NULL && set >= FIRST_SLEEPING_SET {
            self.wake_set(set);
            return true;
        }
        false
    }

    /// Wakes every body, contact, joint and island of sleeping set `set` and frees the set
    /// (Box2D `b2WakeSolverSet`). The bodies start from rest with their sleep timers reset; the
    /// contacts of a woken body that waited in the disabled set become awake again.
    pub(super) fn wake_set(&mut self, set: u32) {
        let mut sleeping = std::mem::take(&mut self.sets[set as usize]);
        for &slot in &sleeping.bodies {
            let body = &mut self.bodies[slot as usize];
            body.is_sleeping = false;
            body.sleep_time = 0.0;
            self.body_links[slot as usize].is_fast = false;
            body.linear_velocity = korppi_math::Vec2::ZERO;
            body.angular_velocity = 0.0;
            self.add_body_to_set(slot, AWAKE_SET);
            let mut key = self.contact_manager.body_head(slot);
            while key != NULL {
                let contact_slot = key >> 1;
                let contact = self.contact_manager.at(contact_slot);
                key = contact.edges[(key & 1) as usize].next;
                if contact.set == DISABLED_SET {
                    self.remove_contact_from_set(contact_slot);
                    self.add_contact_to_set(contact_slot, AWAKE_SET, false);
                }
            }
        }
        for &contact_slot in &sleeping.contacts {
            let contact = self.contact_manager.at_mut(contact_slot);
            contact.set = NULL;
            self.add_contact_to_set(contact_slot, AWAKE_SET, true);
        }
        for &joint in &sleeping.joints {
            self.joint_links[joint as usize].set = NULL;
            self.add_joint_to_set(joint, AWAKE_SET);
        }
        for &island in &sleeping.islands {
            let awake = &mut self.sets[AWAKE_SET as usize].islands;
            self.islands[island as usize].set = AWAKE_SET;
            self.islands[island as usize].local = awake.len() as u32;
            awake.push(island);
        }
        // Keep the emptied arrays for the next set that sleeps.
        sleeping.clear();
        self.sets[set as usize] = sleeping;
        self.free_sets.push(set);
    }

    /// Puts island `island` to sleep in a new sleeping set, unless it is waiting to be split
    /// (Box2D `b2TrySleepIsland`). Its bodies stop and report falling asleep; their contacts
    /// that do not touch wait in the disabled set unless the other body is awake.
    pub(super) fn try_sleep_island(&mut self, island: u32) {
        let (remove_count, body_count) = {
            let island = &self.islands[island as usize];
            (island.constraint_remove_count, island.bodies.len())
        };
        if remove_count > 0 && body_count > 1 {
            return;
        }
        let set = self.alloc_set();
        let bodies = std::mem::take(&mut self.islands[island as usize].bodies);
        for &slot in &bodies {
            let move_event = self.body_links[slot as usize].move_event;
            if move_event != NULL {
                self.body_events[move_event as usize].fell_asleep = true;
                self.body_links[slot as usize].move_event = NULL;
            }
            self.body_links[slot as usize].is_fast = false;
            let body = &mut self.bodies[slot as usize];
            body.is_sleeping = true;
            body.linear_velocity = korppi_math::Vec2::ZERO;
            body.angular_velocity = 0.0;
            self.transfer_body(slot, set);

            let mut key = self.contact_manager.body_head(slot);
            while key != NULL {
                let contact_slot = key >> 1;
                let end = (key & 1) as usize;
                let contact = self.contact_manager.at(contact_slot);
                key = contact.edges[end].next;
                // Touching contacts move with the island below.
                if contact.color != NULL {
                    continue;
                }
                // The other body may still be awake; it takes the contact along if it sleeps.
                let other = contact.edges[end ^ 1].body;
                if self.body_links[other as usize].set == AWAKE_SET {
                    continue;
                }
                self.remove_contact_from_set(contact_slot);
                self.add_contact_to_set(contact_slot, DISABLED_SET, false);
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

        // Move the island itself.
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

    /// Merges sleeping set `b` into sleeping set `a`, for a joint created between them (Box2D
    /// `b2MergeSolverSets`). The larger set keeps the bodies.
    pub(super) fn merge_sets(&mut self, a: u32, b: u32) {
        let (keep, gone) =
            if self.sets[a as usize].bodies.len() >= self.sets[b as usize].bodies.len() {
                (a, b)
            } else {
                (b, a)
            };
        let mut merged = std::mem::take(&mut self.sets[gone as usize]);
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

    /// A new island in set `set` (Box2D `b2CreateIsland`).
    pub(super) fn create_island(&mut self, set: u32) -> u32 {
        let id = if let Some(id) = self.free_islands.pop() {
            id
        } else {
            self.islands.push(Island2d::default());
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

    /// Frees island `id`, which must be empty (Box2D `b2DestroyIsland`).
    pub(super) fn destroy_island(&mut self, id: u32) {
        if self.island_to_split == id {
            self.island_to_split = NULL;
        }
        let (set, local) = {
            let island = &self.islands[id as usize];
            (island.set, island.local)
        };
        let islands = &mut self.sets[set as usize].islands;
        islands.swap_remove(local as usize);
        if let Some(&moved) = islands.get(local as usize) {
            self.islands[moved as usize].local = local;
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

    /// Gives body slot `slot` an island of its own in set `set` (Box2D
    /// `b2CreateIslandForBody`).
    pub(super) fn create_island_for_body(&mut self, set: u32, slot: u32) {
        let island = self.create_island(set);
        self.islands[island as usize].bodies.push(slot);
        let links = &mut self.body_links[slot as usize];
        links.island = island;
        links.island_index = 0;
    }

    /// Takes body slot `slot` out of its island, destroying the island if it empties (Box2D
    /// `b2RemoveBodyFromIsland`).
    pub(super) fn remove_body_from_island(&mut self, slot: u32) {
        let BodyLinks2d {
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

    /// Merges islands `a` and `b`, either of which may be [`NULL`], into the one with more
    /// bodies, and returns it (Box2D `b2MergeIslands`).
    fn merge_islands(&mut self, a: u32, b: u32) -> u32 {
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
            let contact = self.contact_manager.at_mut(link.id);
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
        // Hand the emptied arrays back for reuse, then free the island.
        let Island2d {
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
        self.islands[small as usize] = Island2d {
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
    /// an awake one touches and merging the islands (Box2D `b2LinkContact`).
    pub(super) fn link_contact(&mut self, slot: u32) {
        let contact = self.contact_manager.at(slot);
        let (a, b) = (contact.body_a, contact.body_b);
        let (set_a, set_b) = (
            self.body_links[a as usize].set,
            self.body_links[b as usize].set,
        );
        if set_a == AWAKE_SET && set_b >= FIRST_SLEEPING_SET {
            self.wake_set(set_b);
        }
        if set_b == AWAKE_SET && set_a >= FIRST_SLEEPING_SET {
            self.wake_set(set_a);
        }
        let island = self.merge_islands(
            self.body_links[a as usize].island,
            self.body_links[b as usize].island,
        );
        let contacts = &mut self.islands[island as usize].contacts;
        let contact = self.contact_manager.at_mut(slot);
        contact.island = island;
        contact.island_index = contacts.len() as u32;
        contacts.push(IslandLink2d {
            id: slot,
            body_a: a,
            body_b: b,
        });
    }

    /// Unlinks contact slot `slot` from its island, counting a removed constraint (Box2D
    /// `b2UnlinkContact`).
    pub(super) fn unlink_contact(&mut self, slot: u32) {
        let contact = self.contact_manager.at(slot);
        let (island, index) = (contact.island, contact.island_index);
        let contacts = &mut self.islands[island as usize].contacts;
        contacts.swap_remove(index as usize);
        if let Some(moved) = contacts.get(index as usize) {
            self.contact_manager.at_mut(moved.id).island_index = index;
        }
        self.islands[island as usize].constraint_remove_count += 1;
        let contact = self.contact_manager.at_mut(slot);
        contact.island = NULL;
        contact.island_index = NULL;
    }

    /// Links joint slot `slot` into its bodies' islands, waking a sleeping body jointed to an
    /// awake one and merging the islands (Box2D `b2LinkJoint`).
    pub(super) fn link_joint(&mut self, slot: u32) {
        let (a, b) = self.joint_bodies[slot as usize];
        let (set_a, set_b) = (
            self.body_links[a as usize].set,
            self.body_links[b as usize].set,
        );
        if set_a == AWAKE_SET && set_b >= FIRST_SLEEPING_SET {
            self.wake_set(set_b);
        } else if set_b == AWAKE_SET && set_a >= FIRST_SLEEPING_SET {
            self.wake_set(set_a);
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
        joints.push(IslandLink2d {
            id: slot,
            body_a: a,
            body_b: b,
        });
    }

    /// Unlinks joint slot `slot` from its island, if linked, counting a removed constraint
    /// (Box2D `b2UnlinkJoint`).
    pub(super) fn unlink_joint(&mut self, slot: u32) {
        let JointLinks2d {
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

    /// Splits awake island `base` into its connected parts (Box2D `b2SplitIsland`), by
    /// union-find over its contacts and joints; static bodies connect nothing. A contact or
    /// joint goes to the island of its first non-static body.
    pub(super) fn split_island(&mut self, base: u32) {
        let mut scratch = std::mem::take(&mut self.split_scratch);
        {
            let island = &mut self.islands[base as usize];
            std::mem::swap(&mut scratch.bodies, &mut island.bodies);
            std::mem::swap(&mut scratch.contacts, &mut island.contacts);
            std::mem::swap(&mut scratch.joints, &mut island.joints);
        }
        let body_count = scratch.bodies.len();
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
                } else {
                    index_b
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
            // Still whole: give the arrays back.
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
            let contact = self.contact_manager.at_mut(link.id);
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
}

impl PhysicsWorld2d {
    /// Checks that every body, contact, joint and island is where its links say, once (Box2D
    /// `b2ValidateSolverSets`). Debug builds run it after each step.
    #[cfg(debug_assertions)]
    pub(super) fn validate_sets(&self) {
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
                let c = self.contact_manager.at(slot);
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
                let c = self.contact_manager.at(slot);
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
        let mut contacts = 0;
        for c in self.contact_manager.values() {
            contacts += 1;
            assert!(c.set != NULL);
            if c.island != NULL {
                let link = self.islands[c.island as usize].contacts[c.island_index as usize];
                assert_eq!((link.body_a, link.body_b), (c.body_a, c.body_b));
            }
        }
        let listed: usize = self.sets.iter().map(|s| s.contacts.len()).sum::<usize>()
            + self.constraint_graph.contact_count();
        assert_eq!(listed, contacts, "every contact is listed once");
        for (slot, links) in self.body_links.iter().enumerate() {
            if links.island != NULL {
                assert_eq!(
                    self.islands[links.island as usize].bodies[links.island_index as usize],
                    slot as u32
                );
            }
        }
    }
}
