//! Persistent constraint graph coloring: a port of Box3D's `constraint_graph.c`.
//!
//! A color holds constraints (contacts and joints) that share no dynamic bodies, allowing them to
//! be solved in parallel or in any order without memory conflicts or race conditions.
//!
//! Dynamic-dynamic constraints take the first available color among the low 20 dynamic colors (0..20).
//! Constraints with static bodies take the higher colors (20..23), guaranteeing they are solved after
//! dynamic constraints so static boundaries have the final word. Overflow constraints (> 24 degree)
//! are placed in the overflow color (23).

use serde::{Deserialize, Serialize};

/// Number of graph colors, including the overflow color (Box3D `B3_GRAPH_COLOR_COUNT`).
pub const GRAPH_COLOR_COUNT: usize = 24;

/// Index of the overflow color (Box3D `B3_OVERFLOW_INDEX`).
pub const OVERFLOW_INDEX: usize = GRAPH_COLOR_COUNT - 1;

/// Number of colors reserved for dynamic-dynamic constraints (Box3D `B3_DYNAMIC_COLOR_COUNT`).
pub const DYNAMIC_COLOR_COUNT: usize = GRAPH_COLOR_COUNT - 4;

/// A set of constraints whose dynamic bodies are all distinct.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphColor3d {
    /// Dynamic body slots marked in this color as a bit set.
    bodies: Vec<u64>,
    /// Body pairs of constraints added with `add_constraint`.
    pub constraints: Vec<(u32, u32)>,
    /// Joint slots assigned to this color.
    pub joint_indices: Vec<usize>,
    /// Body pairs for each joint in this color.
    pub joint_pairs: Vec<(u32, u32)>,
    /// Contact keys (collider pairs) in this color.
    pub contact_keys: Vec<(u32, u32)>,
    /// Contact slot index for each contact key.
    pub contact_slots: Vec<u32>,
    /// Body pairs for each contact in this color.
    pub contact_pairs: Vec<(u32, u32)>,
}

impl GraphColor3d {
    /// Iterates over all body pairs in this color (constraints, joints, and contacts).
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

    /// Number of constraints (contacts, joints, user constraints) in this color.
    #[must_use]
    pub fn len(&self) -> usize {
        self.contact_slots.len() + self.joint_indices.len() + self.constraints.len()
    }

    /// Whether this color has zero constraints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all constraints and dynamic body bitset markings in this color.
    pub fn reset(&mut self) {
        self.bodies.fill(0);
        self.constraints.clear();
        self.joint_indices.clear();
        self.joint_pairs.clear();
        self.contact_keys.clear();
        self.contact_slots.clear();
        self.contact_pairs.clear();
    }

    fn with_capacity(contact_cap: usize, joint_cap: usize) -> Self {
        Self {
            bodies: Vec::with_capacity(8),
            constraints: Vec::with_capacity(joint_cap),
            joint_indices: Vec::with_capacity(joint_cap),
            joint_pairs: Vec::with_capacity(joint_cap),
            contact_keys: Vec::with_capacity(contact_cap),
            contact_slots: Vec::with_capacity(contact_cap),
            contact_pairs: Vec::with_capacity(contact_cap),
        }
    }
}

/// Persistent constraint graph managing independent colors for parallel solving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintGraph3d {
    colors: Vec<GraphColor3d>,
    overflow_index: usize,
}

impl Default for ConstraintGraph3d {
    fn default() -> Self {
        Self::new(GRAPH_COLOR_COUNT)
    }
}

impl ConstraintGraph3d {
    /// Creates a new constraint graph with the specified number of colors.
    #[must_use]
    pub fn new(color_count: usize) -> Self {
        assert!(color_count >= 2, "Must have at least 2 graph colors");
        Self {
            colors: (0..color_count)
                .map(|_| GraphColor3d::with_capacity(256, 32))
                .collect(),
            overflow_index: color_count - 1,
        }
    }

    /// Number of colors available for constraints between two dynamic bodies (Box3D `B3_DYNAMIC_COLOR_COUNT`).
    #[must_use]
    pub fn dynamic_color_count(&self) -> usize {
        self.overflow_index.saturating_sub(3).max(1)
    }

    /// Number of colors in the graph, including overflow.
    #[must_use]
    pub fn color_count(&self) -> usize {
        self.colors.len()
    }

    /// Index of the overflow color.
    #[must_use]
    pub const fn overflow_index(&self) -> usize {
        self.overflow_index
    }

    /// Number of constraints in the color at `index`.
    #[must_use]
    pub fn color_len(&self, index: usize) -> usize {
        self.colors.get(index).map_or(0, GraphColor3d::len)
    }

    /// Returns a list of all body pairs in the color at `index`.
    #[must_use]
    pub fn color_pairs(&self, index: usize) -> Vec<(u32, u32)> {
        self.colors
            .get(index)
            .map(|c| c.pairs().collect())
            .unwrap_or_default()
    }

    /// The color at `index`.
    #[must_use]
    pub fn color(&self, index: usize) -> &GraphColor3d {
        &self.colors[index]
    }

    /// Iterates over all colors in order (0..color_count).
    pub fn colors(&self) -> impl Iterator<Item = &GraphColor3d> + '_ {
        self.colors.iter()
    }

    /// Total number of awake touching contacts across all colors in the constraint graph.
    #[must_use]
    pub fn contact_count(&self) -> usize {
        self.colors.iter().map(|c| c.contact_slots.len()).sum()
    }

    /// Total number of awake joints across all colors in the constraint graph.
    #[must_use]
    pub fn joint_count(&self) -> usize {
        self.colors.iter().map(|c| c.joint_indices.len()).sum()
    }

    /// Contact slots across all colors in the constraint graph.
    pub fn contact_slots(&self) -> impl Iterator<Item = u32> + '_ {
        self.colors
            .iter()
            .flat_map(|c| c.contact_slots.iter().copied())
    }

    /// Clears all colors in the graph without reallocating buffers.
    pub fn clear(&mut self) {
        for color in &mut self.colors {
            color.reset();
        }
    }

    /// Assigns a color for a constraint between bodies `body_a` and `body_b` (Box3D `b3AssignJointColor`).
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
            for i in (self.dynamic_color_count()..self.overflow_index).rev() {
                let color = &mut self.colors[i];
                if color.has(body) {
                    continue;
                }
                color.set(body);
                return i;
            }
            // Fall back to scanning dynamic colors if static range is full
            for i in (1..self.dynamic_color_count()).rev() {
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

    /// Releases body markings from `color`.
    fn release(&mut self, color: usize, body_a: u32, body_b: u32) {
        if color != self.overflow_index {
            self.colors[color].clear(body_a);
            self.colors[color].clear(body_b);
        }
    }

    /// Adds touching contact slot `slot` between bodies `body_a` and `body_b` (Box3D `b3AddContactToGraph`).
    pub fn add_contact(
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

    /// Removes contact from color at `local`, returning the slot moved into its place (Box3D `b3RemoveContactFromGraph`).
    pub fn remove_contact(&mut self, color: u32, local: u32) -> Option<u32> {
        let (index, local) = (color as usize, local as usize);
        let (body_a, body_b) = self.colors[index].contact_pairs[local];
        self.release(index, body_a, body_b);
        let color = &mut self.colors[index];
        color.contact_keys.swap_remove(local);
        color.contact_slots.swap_remove(local);
        color.contact_pairs.swap_remove(local);
        color.contact_slots.get(local).copied()
    }

    /// Adds joint slot `slot` between bodies `body_a` and `body_b` (Box3D `b3CreateJointInGraph`).
    pub fn add_joint(
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

    /// Removes joint from color at `local`, returning the joint slot moved into its place (Box3D `b3RemoveJointFromGraph`).
    pub fn remove_joint(&mut self, color: u32, local: u32) -> Option<u32> {
        let (index, local) = (color as usize, local as usize);
        let (body_a, body_b) = self.colors[index].joint_pairs[local];
        self.release(index, body_a, body_b);
        let color = &mut self.colors[index];
        color.joint_indices.swap_remove(local);
        color.joint_pairs.swap_remove(local);
        color.joint_indices.get(local).map(|&j| j as u32)
    }

    /// Adds a constraint between two dynamic bodies directly.
    pub fn add_constraint(&mut self, body_a: u32, body_b: u32) -> usize {
        let index = self.assign_color(body_a, body_b, true, true);
        self.colors[index].constraints.push((body_a, body_b));
        index
    }
}
