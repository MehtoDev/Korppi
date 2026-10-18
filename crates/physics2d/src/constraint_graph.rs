//! Persistent constraint graph coloring: a port of Box2D v3.2's `constraint_graph.c`.
//!
//! A color holds constraints that share no dynamic body, so the constraints of one color can be
//! solved in any order, or in parallel. A constraint gets its color when it joins the graph (a
//! contact when it starts touching or wakes, a joint when it is created awake or wakes) and keeps
//! it until it leaves. Constraints between two dynamic bodies take the first free color among
//! the low [`dynamic_color_count`](ConstraintGraph2d::dynamic_color_count) ones. Constraints with
//! a static or kinematic body take the last free color below the overflow, so they are solved
//! after the dynamic ones and have the last word, which keeps bodies from being pushed through
//! the ground. Constraints that fit nowhere go to the overflow color, which the solver visits
//! first in every stage.

use serde::{Deserialize, Serialize};

/// Number of graph colors, the overflow color included (Box2D `B2_GRAPH_COLOR_COUNT`).
pub const GRAPH_COLOR_COUNT: usize = 24;

/// A set of constraints whose dynamic bodies are all different.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphColor2d {
    /// The dynamic body slots used by the color's constraints, as a bit set.
    bodies: Vec<u64>,
    /// Body pairs of the constraints added with [`ConstraintGraph2d::add_constraint`].
    pub constraints: Vec<(u32, u32)>,
    /// Joint slots in the color.
    pub joint_indices: Vec<usize>,
    /// The body pair of each joint.
    pub(crate) joint_pairs: Vec<(u32, u32)>,
    /// Contact keys (collider pairs) in the color.
    pub contact_keys: Vec<(u32, u32)>,
    /// The contact slot of each contact key.
    pub(crate) contact_slots: Vec<u32>,
    /// The body pair of each contact.
    pub(crate) contact_pairs: Vec<(u32, u32)>,
}

impl GraphColor2d {
    /// The body pairs of the color's constraints, joints and contacts.
    pub fn pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.constraints
            .iter()
            .chain(&self.joint_pairs)
            .chain(&self.contact_pairs)
            .copied()
    }

    fn has(&self, body: u32) -> bool {
        self.bodies
            .get(body as usize / 64)
            .is_some_and(|word| word & (1 << (body % 64)) != 0)
    }

    fn set(&mut self, body: u32) {
        let word = body as usize / 64;
        if self.bodies.len() <= word {
            self.bodies.resize(word + 1, 0);
        }
        self.bodies[word] |= 1 << (body % 64);
    }

    fn clear(&mut self, body: u32) {
        if let Some(word) = self.bodies.get_mut(body as usize / 64) {
            *word &= !(1 << (body % 64));
        }
    }

    /// Number of contacts and joints in the color (Box2D `colorCounts`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.contact_slots.len() + self.joint_indices.len() + self.constraints.len()
    }

    /// Whether the color holds no constraint.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The constraint graph: colors of independent contacts and joints, the last one the overflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintGraph2d {
    colors: Vec<GraphColor2d>,
    overflow_index: usize,
}

impl Default for ConstraintGraph2d {
    fn default() -> Self {
        Self::new(GRAPH_COLOR_COUNT)
    }
}

impl ConstraintGraph2d {
    /// Creates a constraint graph with `color_count` colors, the last being the overflow.
    ///
    /// # Panics
    ///
    /// Panics if `color_count` is under 2.
    #[must_use]
    pub fn new(color_count: usize) -> Self {
        assert!(color_count >= 2, "Must have at least 2 graph colors");
        Self {
            colors: vec![GraphColor2d::default(); color_count],
            overflow_index: color_count - 1,
        }
    }

    /// Number of colors open to constraints between two dynamic bodies (Box2D
    /// `B2_DYNAMIC_COLOR_COUNT`): all but the overflow and three more kept for constraints with
    /// a static body.
    #[must_use]
    pub fn dynamic_color_count(&self) -> usize {
        self.overflow_index.saturating_sub(3).max(1)
    }

    /// The color for a constraint between bodies `body_a` and `body_b`, marking them in it
    /// (Box2D `b2AssignJointColor`): two dynamic bodies take the first color free of both among
    /// the dynamic colors; one dynamic body takes the last color below the overflow free of it,
    /// down to color 1; otherwise the overflow.
    fn assign_color(
        &mut self,
        body_a: u32,
        body_b: u32,
        dynamic_a: bool,
        dynamic_b: bool,
    ) -> usize {
        if dynamic_a && dynamic_b {
            for i in 0..self.dynamic_color_count() {
                let color = &mut self.colors[i];
                if color.has(body_a) || color.has(body_b) {
                    continue;
                }
                color.set(body_a);
                color.set(body_b);
                return i;
            }
        } else if dynamic_a || dynamic_b {
            let body = if dynamic_a { body_a } else { body_b };
            for i in (1..self.overflow_index).rev() {
                let color = &mut self.colors[i];
                if color.has(body) {
                    continue;
                }
                color.set(body);
                return i;
            }
        }
        self.overflow_index
    }

    /// Unmarks bodies `body_a` and `body_b` in color `color`.
    fn release(&mut self, color: usize, body_a: u32, body_b: u32) {
        if color != self.overflow_index {
            self.colors[color].clear(body_a);
            self.colors[color].clear(body_b);
        }
    }

    /// Adds touching contact slot `slot` (collider pair `key`) between bodies `body_a` and
    /// `body_b` and returns its color and index in it (Box2D `b2AddContactToGraph`).
    pub(crate) fn add_contact(
        &mut self,
        key: (u32, u32),
        slot: u32,
        (body_a, body_b): (u32, u32),
        (dynamic_a, dynamic_b): (bool, bool),
    ) -> (u32, u32) {
        let index = self.assign_color(body_a, body_b, dynamic_a, dynamic_b);
        let color = &mut self.colors[index];
        color.contact_keys.push(key);
        color.contact_slots.push(slot);
        color.contact_pairs.push((body_a, body_b));
        (index as u32, color.contact_slots.len() as u32 - 1)
    }

    /// Removes the contact at `local` in color `color`, returning the contact slot moved into
    /// its place, if any (Box2D `b2RemoveContactFromGraph`).
    pub(crate) fn remove_contact(&mut self, color: u32, local: u32) -> Option<u32> {
        let (index, local) = (color as usize, local as usize);
        let (body_a, body_b) = self.colors[index].contact_pairs[local];
        self.release(index, body_a, body_b);
        let color = &mut self.colors[index];
        color.contact_keys.swap_remove(local);
        color.contact_slots.swap_remove(local);
        color.contact_pairs.swap_remove(local);
        color.contact_slots.get(local).copied()
    }

    /// Adds joint slot `slot` between bodies `body_a` and `body_b` and returns its color and
    /// index in it (Box2D `b2CreateJointInGraph`).
    pub(crate) fn add_joint(
        &mut self,
        slot: u32,
        (body_a, body_b): (u32, u32),
        (dynamic_a, dynamic_b): (bool, bool),
    ) -> (u32, u32) {
        let index = self.assign_color(body_a, body_b, dynamic_a, dynamic_b);
        let color = &mut self.colors[index];
        color.joint_indices.push(slot as usize);
        color.joint_pairs.push((body_a, body_b));
        (index as u32, color.joint_indices.len() as u32 - 1)
    }

    /// Removes the joint at `local` in color `color`, returning the joint slot moved into its
    /// place, if any (Box2D `b2RemoveJointFromGraph`).
    pub(crate) fn remove_joint(&mut self, color: u32, local: u32) -> Option<u32> {
        let (index, local) = (color as usize, local as usize);
        let (body_a, body_b) = self.colors[index].joint_pairs[local];
        self.release(index, body_a, body_b);
        let color = &mut self.colors[index];
        color.joint_indices.swap_remove(local);
        color.joint_pairs.swap_remove(local);
        color.joint_indices.get(local).map(|&joint| joint as u32)
    }

    /// Adds a constraint between two dynamic bodies, returning its color.
    pub fn add_constraint(&mut self, body_a: u32, body_b: u32) -> usize {
        let index = self.assign_color(body_a, body_b, true, true);
        self.colors[index].constraints.push((body_a, body_b));
        index
    }

    /// The color at `index`.
    #[must_use]
    pub fn color(&self, index: usize) -> &GraphColor2d {
        &self.colors[index]
    }

    /// The colors, the overflow last.
    pub fn colors(&self) -> impl Iterator<Item = &GraphColor2d> + '_ {
        self.colors.iter()
    }

    /// Number of colors, the overflow included.
    #[must_use]
    pub fn color_count(&self) -> usize {
        self.colors.len()
    }

    /// Index of the overflow color.
    #[must_use]
    pub const fn overflow_index(&self) -> usize {
        self.overflow_index
    }

    /// Number of contacts in the graph.
    #[must_use]
    pub fn contact_count(&self) -> usize {
        self.colors.iter().map(|c| c.contact_slots.len()).sum()
    }

    /// The contact slots of every color, the overflow last.
    pub(crate) fn contact_slots(&self) -> impl Iterator<Item = u32> + '_ {
        self.colors
            .iter()
            .flat_map(|c| c.contact_slots.iter().copied())
    }

    /// Makes room for `additional` more contacts in each color.
    pub(crate) fn reserve_contacts(&mut self, additional: usize) {
        for color in &mut self.colors {
            color.contact_keys.reserve(additional);
            color.contact_slots.reserve(additional);
            color.contact_pairs.reserve(additional);
        }
    }
}
