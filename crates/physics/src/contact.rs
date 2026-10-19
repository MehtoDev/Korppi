//! Persistent contacts between collider pairs (Box3D `b3Contact`, `contact.c`).
//!
//! A [`Contact3d`] lives while the broadphase pairs its colliders, touching or not, and keeps
//! its manifolds (one between convex shapes, one per cluster of triangle normals against a
//! mesh), the narrowphase's caches and what contact recycling needs between steps. The
//! [`ContactManager3d`] keeps contacts in slots with a free list and finds a pair's slot by a
//! hash map (Box3D's contact array and pair set), so a step looks contacts up instead of
//! searching them; it keeps the buffers of ended contacts for new ones, so contacts begin and
//! end without allocating once the scene has grown. Iteration is in slot order, which depends
//! only on the order contacts were created and destroyed, so it is deterministic.

use crate::mesh_contact::MeshContactCache;
use crate::narrowphase::{ContactCache, ContactManifold};
use korppi_math::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sentinel for "no index" in contact, island, and solver-set link fields.
pub(crate) const NULL: u32 = u32::MAX;

const fn null_u32() -> u32 {
    NULL
}

/// One end of a contact in a body's doubly linked list of contacts (Box3D `b3ContactEdge`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContactEdge3d {
    pub(crate) body: u32,
    pub(crate) prev: u32,
    pub(crate) next: u32,
}

impl Default for ContactEdge3d {
    fn default() -> Self {
        Self {
            body: NULL,
            prev: NULL,
            next: NULL,
        }
    }
}

/// A persistent contact between the colliders of two bodies (Box3D `b3Contact`). Its manifold
/// normal points from collider A to collider B, `collider_a < collider_b`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contact3d {
    /// Collider A (a body's index while bodies have one collider).
    pub collider_a: u32,
    /// Collider B.
    pub collider_b: u32,
    /// Collider A's body.
    pub body_a: u32,
    /// Collider B's body.
    pub body_b: u32,
    /// Links into `body_a` (`edges[0]`) and `body_b`'s (`edges[1]`) contact lists.
    #[serde(default)]
    pub(crate) edges: [ContactEdge3d; 2],
    /// Solver set (`STATIC_SET` when not touching, `AWAKE_SET` or a sleeping set when touching),
    /// or `NULL` before the contact has been placed in a set.
    #[serde(default = "null_u32")]
    pub(crate) set: u32,
    /// Index in its solver set's `contacts` (or in `color`'s `contact_slots` when in the
    /// constraint graph).
    #[serde(default = "null_u32")]
    pub(crate) local: u32,
    /// Constraint-graph color when `set == AWAKE_SET` and `is_touching`, else `NULL`.
    #[serde(default = "null_u32")]
    pub(crate) color: u32,
    /// Persistent island this touching contact belongs to, or `NULL`.
    #[serde(default = "null_u32")]
    pub(crate) island: u32,
    /// Index in `island`'s `contacts` list, or `NULL`.
    #[serde(default = "null_u32")]
    pub(crate) island_index: u32,
    /// The latest manifolds, none while the colliders are apart.
    pub(crate) manifolds: Vec<ContactManifold>,
    /// Whether the contact has manifolds.
    pub is_touching: bool,
    /// Whether the contact had manifolds on the previous step (for begin/end events).
    pub(crate) was_touching: bool,
    /// The narrowphase's memory of a convex pair.
    pub(crate) cache: ContactCache,
    /// The triangles near the shape, and their caches, of a mesh pair.
    pub(crate) mesh_cache: MeshContactCache,
    /// Whether both bodies allow recycling (Box3D `b3_contactRecycleFlag`).
    pub(crate) recycle: bool,
    /// Whether the poses below are from a real collide (Box3D `b3_relativeTransformValid`).
    pub(crate) relative_valid: bool,
    /// The bodies' rotations, and B's pose relative to A, at the last real collide.
    pub(crate) cached_rotation_a: Quat,
    pub(crate) cached_rotation_b: Quat,
    pub(crate) cached_relative_position: Vec3,
    pub(crate) cached_relative_rotation: Quat,
    /// The step in which the broadphase last paired the colliders.
    pub(crate) seen: u64,
}

impl Contact3d {
    /// A new contact between `colliders` on `bodies`, not touching yet.
    pub(crate) fn new(colliders: (u32, u32), bodies: (u32, u32), recycle: bool) -> Self {
        Self {
            collider_a: colliders.0,
            collider_b: colliders.1,
            body_a: bodies.0,
            body_b: bodies.1,
            edges: [
                ContactEdge3d {
                    body: bodies.0,
                    prev: NULL,
                    next: NULL,
                },
                ContactEdge3d {
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
            manifolds: Vec::new(),
            is_touching: false,
            was_touching: false,
            cache: ContactCache::default(),
            mesh_cache: MeshContactCache::default(),
            recycle,
            relative_valid: false,
            cached_rotation_a: Quat::IDENTITY,
            cached_rotation_b: Quat::IDENTITY,
            cached_relative_position: Vec3::ZERO,
            cached_relative_rotation: Quat::IDENTITY,
            seen: 0,
        }
    }

    /// The latest manifolds: none while the colliders are apart, one between convex shapes,
    /// one per cluster of triangle normals against a mesh. Normals point from collider A's
    /// body to collider B's.
    #[must_use]
    pub fn manifolds(&self) -> &[ContactManifold] {
        &self.manifolds
    }

    /// Takes the convex `manifold` (of these bodies, in this order) as the contact's manifold:
    /// each point takes the normal impulse of the old point with the same feature id, each old
    /// point at most once, and the manifold keeps the friction, twist and rolling impulses
    /// while the contact keeps touching (Box3D `b3ComputeConvexManifold`).
    pub(crate) fn update(&mut self, mut manifold: ContactManifold) {
        if let Some(old) = self.manifolds.first() {
            manifold.friction_impulse = old.friction_impulse;
            manifold.twist_impulse = old.twist_impulse;
            manifold.rolling_impulse = old.rolling_impulse;
            let mut claimed = [false; crate::narrowphase::MAX_MANIFOLD_POINTS];
            for point in &mut manifold.points {
                let matched = old
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
        self.manifolds.clear();
        if !manifold.points.is_empty() {
            self.manifolds.push(manifold);
        }
        self.is_touching = !self.manifolds.is_empty();
    }

    /// Ends touching: the manifolds and their impulses go (Box3D frees them).
    pub(crate) fn separate(&mut self) {
        self.manifolds.clear();
        self.is_touching = false;
    }
}

/// The contacts of a world (Box3D's contact array and pair set): slots reused through a free
/// list, a map from collider pair to slot, and per-body doubly linked contact lists. Once it
/// has grown to the scene's size, contacts begin and end without allocating.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ContactManager3d {
    slots: Vec<Option<Contact3d>>,
    free: Vec<u32>,
    /// Head contact slot for each body (`NULL` when the body has no contacts).
    #[serde(default)]
    body_heads: Vec<u32>,
    /// Collider pair to slot, rebuilt from the slots after deserializing.
    #[serde(skip)]
    index: HashMap<(u32, u32), u32>,
    /// The emptied buffers of ended contacts, for new ones (Box3D's manifold pool).
    #[serde(skip)]
    pool: Vec<(Vec<ContactManifold>, MeshContactCache)>,
}

impl PartialEq for ContactManager3d {
    /// The same contacts in the same slots.
    fn eq(&self, other: &Self) -> bool {
        self.slots == other.slots && self.free == other.free && self.body_heads == other.body_heads
    }
}

impl ContactManager3d {
    /// Creates an empty contact manager with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free: Vec::with_capacity(capacity),
            body_heads: Vec::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            pool: (0..capacity)
                .map(|_| (Vec::with_capacity(4), MeshContactCache::default()))
                .collect(),
        }
    }

    /// The number of contacts, touching or not.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Whether there are no contacts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of slots, live or free.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// The slot of the contact between the colliders of `key`.
    #[must_use]
    pub fn slot_of(&self, key: &(u32, u32)) -> Option<u32> {
        self.index.get(key).copied()
    }

    /// The contact in `slot`, if it is live.
    #[must_use]
    pub fn get(&self, slot: u32) -> Option<&Contact3d> {
        self.slots.get(slot as usize)?.as_ref()
    }

    /// The contact in `slot`, mutably.
    pub(crate) fn get_mut(&mut self, slot: u32) -> Option<&mut Contact3d> {
        self.slots.get_mut(slot as usize)?.as_mut()
    }

    /// The contact in `slot` (panics if the slot is free).
    #[must_use]
    pub(crate) fn at(&self, slot: u32) -> &Contact3d {
        self.slots[slot as usize]
            .as_ref()
            .expect("live contact slot")
    }

    /// The contact in `slot`, mutably (panics if the slot is free).
    pub(crate) fn at_mut(&mut self, slot: u32) -> &mut Contact3d {
        self.slots[slot as usize]
            .as_mut()
            .expect("live contact slot")
    }

    /// Mutable slice of all contact slots, for disjoint parallel access by slot index.
    pub(crate) fn slots_mut(&mut self) -> &mut [Option<Contact3d>] {
        &mut self.slots
    }

    /// Head of `body`'s contact list (`NULL` when the body has no contacts).
    #[must_use]
    pub(crate) fn body_head(&self, body: u32) -> u32 {
        self.body_heads.get(body as usize).copied().unwrap_or(NULL)
    }

    /// Iterates over the contact slots attached to `body` in $O(\text{degree})$ without allocating.
    pub(crate) fn body_contact_slots(&self, body: u32) -> impl Iterator<Item = u32> + '_ {
        let mut slot = self.body_head(body);
        std::iter::from_fn(move || {
            if slot == NULL {
                return None;
            }
            let current = slot;
            let c = self.at(current);
            let edge = usize::from(c.body_b == body);
            slot = c.edges[edge].next;
            Some(current)
        })
    }

    /// Contact slots attached to `body`, walked in $O(\text{degree})$ via the body's contact list.
    #[must_use]
    pub fn body_contacts(&self, body: u32) -> Vec<u32> {
        self.body_contact_slots(body).collect()
    }

    /// The live contacts, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = &Contact3d> + '_ {
        self.slots.iter().flatten()
    }

    /// Adds `contact` under its collider pair, which must be free, and returns its slot.
    pub(crate) fn insert(&mut self, mut contact: Contact3d) -> u32 {
        let key = (contact.collider_a, contact.collider_b);
        let (a, b) = (contact.body_a, contact.body_b);
        if let Some((manifolds, mesh_cache)) = self.pool.pop() {
            contact.manifolds = manifolds;
            contact.mesh_cache = mesh_cache;
        } else {
            contact.manifolds.reserve(4);
        }
        debug_assert!(!self.index.contains_key(&key));
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slots[slot as usize] = Some(contact);
                slot
            }
            None => {
                self.slots.push(Some(contact));
                self.slots.len() as u32 - 1
            }
        };
        self.index.insert(key, slot);
        self.link_body(slot, 0, a);
        self.link_body(slot, 1, b);
        slot
    }

    /// Removes the contact in `slot`, freeing the slot and keeping its buffers; false when the
    /// slot is free.
    pub(crate) fn remove(&mut self, slot: u32) -> bool {
        let Some(c) = self.get(slot) else {
            return false;
        };
        let (a, b) = (c.body_a, c.body_b);
        self.unlink_body(slot, 0, a);
        self.unlink_body(slot, 1, b);
        let mut contact = self.slots[slot as usize]
            .take()
            .expect("checked live above");
        self.index.remove(&(contact.collider_a, contact.collider_b));
        self.free.push(slot);
        contact.manifolds.clear();
        contact.mesh_cache.clear();
        self.pool.push((contact.manifolds, contact.mesh_cache));
        true
    }

    fn link_body(&mut self, slot: u32, edge: usize, body: u32) {
        if self.body_heads.len() <= body as usize {
            self.body_heads.resize(body as usize + 1, NULL);
        }
        let head = self.body_heads[body as usize];
        let c = self.at_mut(slot);
        c.edges[edge] = ContactEdge3d {
            body,
            prev: NULL,
            next: head,
        };
        if head != NULL {
            let next = self.at_mut(head);
            let next_edge = usize::from(next.body_b == body);
            next.edges[next_edge].prev = slot;
        }
        self.body_heads[body as usize] = slot;
    }

    fn unlink_body(&mut self, slot: u32, edge: usize, body: u32) {
        let link = self.at(slot).edges[edge];
        if link.prev != NULL {
            let prev = self.at_mut(link.prev);
            let prev_edge = usize::from(prev.body_b == body);
            prev.edges[prev_edge].next = link.next;
        } else if let Some(head) = self.body_heads.get_mut(body as usize) {
            if *head == slot {
                *head = link.next;
            }
        }
        if link.next != NULL {
            let next = self.at_mut(link.next);
            let next_edge = usize::from(next.body_b == body);
            next.edges[next_edge].prev = link.prev;
        }
    }

    /// Rebuilds the pair map from the slots, after deserializing.
    pub(crate) fn rebuild_index(&mut self) {
        self.index.clear();
        for (slot, contact) in self.slots.iter().enumerate() {
            if let Some(c) = contact {
                self.index.insert((c.collider_a, c.collider_b), slot as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrowphase::{ContactPoint, ContactPoints};

    fn manifold(points: &[(u32, f32)]) -> ContactManifold {
        let mut list = ContactPoints::new();
        for &(feature_id, normal_impulse) in points {
            list.push(ContactPoint {
                feature_id,
                normal_impulse,
                ..ContactPoint::default()
            });
        }
        ContactManifold {
            body_a: 1,
            body_b: 2,
            points: list,
            friction_impulse: Vec3::X,
            ..ContactManifold::default()
        }
    }

    #[test]
    fn points_warm_start_by_feature() {
        let mut contact = Contact3d::new((1, 2), (1, 2), true);
        contact.update(manifold(&[(7, 1.0), (9, 2.0)]));
        assert!(contact.is_touching);
        // Point 9 keeps its impulse, the new point 5 starts cold, and each old point warm
        // starts one new point.
        contact.update(manifold(&[(9, 0.0), (5, 0.0), (9, 0.0)]));
        let impulses: Vec<f32> = contact.manifolds[0]
            .points
            .iter()
            .map(|p| p.normal_impulse)
            .collect();
        assert_eq!(impulses, [2.0, 0.0, 0.0]);
        contact.separate();
        assert!(!contact.is_touching && contact.manifolds().is_empty());
    }

    #[test]
    fn slots_are_reused() {
        let mut contacts = ContactManager3d::default();
        let a = contacts.insert(Contact3d::new((1, 2), (1, 2), true));
        let b = contacts.insert(Contact3d::new((1, 3), (1, 3), true));
        assert_eq!(contacts.slot_of(&(1, 3)), Some(b));
        contacts.remove(a);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts.slot_of(&(1, 2)), None);
        let c = contacts.insert(Contact3d::new((2, 3), (2, 3), true));
        assert_eq!(c, a, "the freed slot is reused");
        contacts.index.clear();
        contacts.rebuild_index();
        assert_eq!(contacts.slot_of(&(2, 3)), Some(c));
    }
}
