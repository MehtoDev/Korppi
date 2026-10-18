//! Persistent contact manager retaining contact manifolds and warm-start impulses across frames.

use crate::distance::SimplexCache2d;
use crate::narrowphase::ContactManifold2d;
use crate::solver::ContactMaterial2d;
use korppi_math::{Rot2, Transform2d};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A missing contact, set or island.
pub(crate) const NULL: u32 = u32::MAX;

/// One end of a contact in its body's list of contacts (Box2D `b2ContactEdge`). Keys are
/// `contact slot << 1 | end`, end 0 being body A.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContactEdge2d {
    pub(crate) body: u32,
    pub(crate) prev: u32,
    pub(crate) next: u32,
}

/// Persistent contact between two colliders on different bodies (Box2D keys contacts by shape
/// pair). The manifold normal points from collider A to collider B.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contact2d {
    /// Slot of collider A (`collider_a < collider_b`).
    pub collider_a: u32,
    /// Slot of collider B.
    pub collider_b: u32,
    /// Slot of collider A's body.
    pub body_a: u32,
    /// Slot of collider B's body.
    pub body_b: u32,
    /// Latest contact manifold with normal and impulses.
    pub manifold: ContactManifold2d,
    /// The surface properties mixed from the two colliders.
    pub material: ContactMaterial2d,
    /// Whether the manifold has points. Box2D keeps a contact while the colliders' fat boxes
    /// overlap, touching or not.
    pub is_touching: bool,
    /// The contact's ends in its bodies' contact lists.
    pub(crate) edges: [ContactEdge2d; 2],
    /// The solver set holding the contact: the awake set while either body is awake, a
    /// sleeping set while touching in a sleeping island, the disabled set while both bodies
    /// rest without touching.
    pub(crate) set: u32,
    /// Index in the set's contacts, or in its graph color's.
    pub(crate) local: u32,
    /// The constraint graph color of a touching awake contact, else [`NULL`].
    pub(crate) color: u32,
    /// The island the contact links while touching, and its index in the island's contacts.
    pub(crate) island: u32,
    pub(crate) island_index: u32,
    /// Whether both bodies allow recycling (Box2D `b2_contactRecycleFlag`).
    pub(crate) recycle: bool,
    /// Whether the cached poses below are from the last real collide (Box2D
    /// `b2_simRelativeTransformValid`).
    pub(crate) relative_valid: bool,
    /// The bodies' rotations and B's pose relative to A at the last real collide.
    pub(crate) cached_rotation_a: Rot2,
    pub(crate) cached_rotation_b: Rot2,
    pub(crate) cached_relative_pose: Transform2d,
    /// The simplex of the last distance query between the colliders, which chain segments
    /// warm start the next from (Box2D `b2ContactSim::cache`).
    pub(crate) cache: SimplexCache2d,
}

impl Contact2d {
    /// Creates a contact between colliders `colliders` on bodies `bodies`, with a manifold whose
    /// normal points from the first collider to the second.
    #[must_use]
    pub const fn new(
        colliders: (u32, u32),
        bodies: (u32, u32),
        manifold: ContactManifold2d,
        material: ContactMaterial2d,
    ) -> Self {
        Self {
            collider_a: colliders.0,
            collider_b: colliders.1,
            body_a: bodies.0,
            body_b: bodies.1,
            manifold,
            material,
            is_touching: manifold.point_count > 0,
            edges: [
                ContactEdge2d {
                    body: bodies.0,
                    prev: NULL,
                    next: NULL,
                },
                ContactEdge2d {
                    body: bodies.1,
                    prev: NULL,
                    next: NULL,
                },
            ],
            set: NULL,
            local: NULL,
            color: NULL,
            island: NULL,
            island_index: NULL,
            recycle: false,
            relative_valid: false,
            cached_rotation_a: Rot2::IDENTITY,
            cached_rotation_b: Rot2::IDENTITY,
            cached_relative_pose: Transform2d::IDENTITY,
            cache: SimplexCache2d::EMPTY,
        }
    }

    /// Updates this contact with a newly generated manifold, transferring cached impulses
    /// for matching geometric feature IDs to preserve warm-starting across frames.
    pub fn update(&mut self, mut new_manifold: ContactManifold2d) {
        let old_manifold = self.manifold;

        for new_pt in &mut new_manifold.points[..new_manifold.point_count] {
            // Only a point made by the same pair of features carries its impulses over, as in
            // Box2D's `b2UpdateContact`.
            let old_pt = old_manifold.points[..old_manifold.point_count]
                .iter()
                .find(|old_pt| old_pt.id == new_pt.id);
            new_pt.persisted = old_pt.is_some();
            if let Some(old_pt) = old_pt {
                new_pt.normal_impulse = old_pt.normal_impulse;
                new_pt.tangent_impulse = old_pt.tangent_impulse;
            }
        }

        new_manifold.rolling_impulse = old_manifold.rolling_impulse;
        self.is_touching = new_manifold.point_count > 0;
        self.manifold = new_manifold;
    }
}

/// The world's contacts, keyed by collider pair `(collider_a, collider_b)` with
/// `collider_a < collider_b`.
///
/// Contacts live in a pool of slots reused last-freed first, with a hash index from collider
/// pair to slot (Box2D keeps contacts in arrays and a pair set). Each body's contacts form a
/// list through the contacts' edges, newest first (Box2D `headContactKey`). Once the pool and
/// index have grown to the scene's size, contacts can begin and end without allocating.
/// Iteration is in slot order, which depends only on the order contacts were created and
/// destroyed, so it is deterministic.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ContactManager2d {
    slots: Vec<Option<Contact2d>>,
    free: Vec<u32>,
    #[serde(with = "crate::serde_ids::pair_map")]
    index: HashMap<(u32, u32), u32>,
    /// Each body slot's first contact key.
    body_heads: Vec<u32>,
}

impl ContactManager2d {
    /// Creates an empty contact manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Makes room for `additional` more contacts, so that up to that many can come and go
    /// without allocating. The index keeps twice the room, since a hash table that churns needs
    /// space to clean up in place.
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
        self.free.reserve(additional);
        self.index.reserve(2 * additional);
    }

    /// Number of contacts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether there are no contacts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// The slot of the contact between the colliders of `key`.
    #[must_use]
    pub fn slot_of(&self, key: &(u32, u32)) -> Option<u32> {
        self.index.get(key).copied()
    }

    /// Number of slots, live or free.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// The contact between the colliders of `key`.
    #[must_use]
    pub fn get(&self, key: &(u32, u32)) -> Option<&Contact2d> {
        self.slots[*self.index.get(key)? as usize].as_ref()
    }

    /// The contact between the colliders of `key`, mutably.
    pub fn get_mut(&mut self, key: &(u32, u32)) -> Option<&mut Contact2d> {
        self.slots[*self.index.get(key)? as usize].as_mut()
    }

    /// Whether the colliders of `key` have a contact.
    #[must_use]
    pub fn contains_key(&self, key: &(u32, u32)) -> bool {
        self.index.contains_key(key)
    }

    /// Adds `contact` under `key`, which must be free, at the head of its bodies' contact lists,
    /// and returns its slot.
    pub(crate) fn insert(&mut self, key: (u32, u32), mut contact: Contact2d) -> u32 {
        debug_assert!(!self.index.contains_key(&key));
        let slot = self.free.pop().unwrap_or(self.slots.len() as u32);
        for end in 0..2 {
            let body = contact.edges[end].body as usize;
            if self.body_heads.len() <= body {
                self.body_heads.resize(body + 1, NULL);
            }
            let head = self.body_heads[body];
            contact.edges[end].prev = NULL;
            contact.edges[end].next = head;
            if head != NULL {
                self.edge_mut(head).prev = (slot << 1) | end as u32;
            }
            self.body_heads[body] = (slot << 1) | end as u32;
        }
        if slot as usize == self.slots.len() {
            self.slots.push(Some(contact));
        } else {
            self.slots[slot as usize] = Some(contact);
        }
        self.index.insert(key, slot);
        slot
    }

    /// Removes and returns the contact in `slot`, unlinking it from its bodies' lists and
    /// freeing the slot.
    pub(crate) fn remove_slot(&mut self, slot: u32) -> Option<Contact2d> {
        let contact = self.slots[slot as usize].take()?;
        self.index.remove(&(contact.collider_a, contact.collider_b));
        for (end, edge) in contact.edges.iter().enumerate() {
            if edge.prev == NULL {
                self.body_heads[edge.body as usize] = edge.next;
            } else {
                self.edge_mut(edge.prev).next = edge.next;
            }
            if edge.next != NULL {
                self.edge_mut(edge.next).prev = edge.prev;
            }
            debug_assert!(self.body_heads[edge.body as usize] != (slot << 1) | end as u32);
        }
        self.free.push(slot);
        Some(contact)
    }

    /// The first contact key of body slot `body`'s list.
    pub(crate) fn body_head(&self, body: u32) -> u32 {
        self.body_heads.get(body as usize).copied().unwrap_or(NULL)
    }

    /// The contact in `slot`.
    pub(crate) fn at(&self, slot: u32) -> &Contact2d {
        self.slots[slot as usize]
            .as_ref()
            .expect("live contact slot")
    }

    /// Every slot, mutably, for the solver's parallel stages.
    pub(crate) fn slots_mut(&mut self) -> &mut [Option<Contact2d>] {
        &mut self.slots
    }

    /// The contact in `slot`, mutably.
    pub(crate) fn at_mut(&mut self, slot: u32) -> &mut Contact2d {
        self.slots[slot as usize]
            .as_mut()
            .expect("live contact slot")
    }

    /// The edge of contact key `key`.
    fn edge_mut(&mut self, key: u32) -> &mut ContactEdge2d {
        &mut self.at_mut(key >> 1).edges[(key & 1) as usize]
    }

    /// The contact slots of body slot `body`, newest first.
    pub(crate) fn body_contacts(&self, body: u32) -> impl Iterator<Item = u32> + '_ {
        let mut key = self.body_head(body);
        std::iter::from_fn(move || {
            if key == NULL {
                return None;
            }
            let slot = key >> 1;
            key = self.at(slot).edges[(key & 1) as usize].next;
            Some(slot)
        })
    }

    /// The contacts and their keys, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = ((u32, u32), &Contact2d)> + '_ {
        self.slots
            .iter()
            .flatten()
            .map(|c| ((c.collider_a, c.collider_b), c))
    }

    /// The contacts, in slot order.
    pub fn values(&self) -> impl Iterator<Item = &Contact2d> + '_ {
        self.slots.iter().flatten()
    }

    /// The contacts, mutably, in slot order.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Contact2d> + '_ {
        self.slots.iter_mut().flatten()
    }
}
