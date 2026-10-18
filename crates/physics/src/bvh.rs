//! Dynamic Bounding Volume Hierarchy (BVH) adapting Erin Catto's Box3D dynamic tree.

use korppi_math::{Aabb, Ray, Vec3};
use serde::{Deserialize, Serialize};

const NULL_NODE: u32 = u32::MAX;
const AABB_EXTENSION: f32 = 0.1;
const AABB_MULTIPLIER: f32 = 2.0;

/// A node in the dynamic bounding volume hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "SavedTreeNode", into = "SavedTreeNode")]
pub struct TreeNode {
    /// Enlarged AABB for broadphase temporal coherence.
    pub aabb: Aabb,
    /// User payload (e.g. body or collider ID).
    pub user_data: u32,
    /// Parent node index in the pool.
    pub parent: u32,
    /// First child node index (or next free node index if unused).
    pub child1: u32,
    /// Second child node index.
    pub child2: u32,
    /// Leaf = 0, free = -1, internal node > 0.
    pub height: i32,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct SavedTreeNode {
    aabb: Option<Aabb>,
    user_data: u32,
    parent: u32,
    child1: u32,
    child2: u32,
    height: i32,
}

impl From<TreeNode> for SavedTreeNode {
    fn from(node: TreeNode) -> Self {
        let finite = node.aabb.min.is_finite() && node.aabb.max.is_finite();
        Self {
            aabb: finite.then_some(node.aabb),
            user_data: node.user_data,
            parent: node.parent,
            child1: node.child1,
            child2: node.child2,
            height: node.height,
        }
    }
}

impl From<SavedTreeNode> for TreeNode {
    fn from(node: SavedTreeNode) -> Self {
        Self {
            aabb: node.aabb.unwrap_or_default(),
            user_data: node.user_data,
            parent: node.parent,
            child1: node.child1,
            child2: node.child2,
            height: node.height,
        }
    }
}

impl TreeNode {
    /// Returns `true` if this node is a leaf (holds user payload).
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.child1 == NULL_NODE
    }
}

/// Dynamic AABB Tree with incremental SAH rebalancing, ray casts, and spatial queries.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct DynamicTree {
    root: u32,
    nodes: Vec<TreeNode>,
    free_head: u32,
    node_count: usize,
}

const TREE_STACK_SIZE: usize = 512;

impl DynamicTree {
    /// Returns the height of the tree.
    #[must_use]
    pub fn height(&self) -> usize {
        if self.root == NULL_NODE {
            0
        } else {
            self.nodes[self.root as usize].height as usize
        }
    }

    /// Creates an empty dynamic tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: NULL_NODE,
            nodes: Vec::with_capacity(64),
            free_head: NULL_NODE,
            node_count: 0,
        }
    }

    /// Returns the total number of active nodes (leaves + internal nodes).
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns the fat AABB stored at `proxy_id`.
    #[must_use]
    pub fn aabb(&self, proxy_id: u32) -> Aabb {
        self.nodes[proxy_id as usize].aabb
    }

    fn allocate_node(&mut self) -> u32 {
        if self.free_head != NULL_NODE {
            let node_id = self.free_head;
            self.free_head = self.nodes[node_id as usize].child1;
            self.node_count += 1;
            node_id
        } else {
            let node_id = self.nodes.len() as u32;
            self.nodes.push(TreeNode {
                aabb: Aabb::default(),
                user_data: 0,
                parent: NULL_NODE,
                child1: NULL_NODE,
                child2: NULL_NODE,
                height: 0,
            });
            self.node_count += 1;
            node_id
        }
    }

    fn free_node(&mut self, node_id: u32) {
        let node = &mut self.nodes[node_id as usize];
        node.parent = NULL_NODE;
        node.child1 = self.free_head;
        node.height = -1;
        self.free_head = node_id;
        self.node_count -= 1;
    }

    /// Creates a leaf proxy node for an AABB and associates it with `user_data`.
    pub fn create_proxy(&mut self, aabb: Aabb, user_data: u32) -> u32 {
        self.create_fat_proxy(aabb.expand(AABB_EXTENSION), user_data)
    }

    /// Creates a leaf proxy node for a pre-expanded `fat_aabb` and associates it with `user_data`.
    pub fn create_fat_proxy(&mut self, fat_aabb: Aabb, user_data: u32) -> u32 {
        let proxy_id = self.allocate_node();

        let node = &mut self.nodes[proxy_id as usize];
        node.aabb = fat_aabb;
        node.user_data = user_data;
        node.height = 0;
        node.parent = NULL_NODE;
        node.child1 = NULL_NODE;
        node.child2 = NULL_NODE;

        self.insert_leaf(proxy_id);
        proxy_id
    }

    /// Destroys a proxy leaf node.
    pub fn destroy_proxy(&mut self, proxy_id: u32) {
        assert!(proxy_id < self.nodes.len() as u32);
        assert!(self.nodes[proxy_id as usize].is_leaf());

        self.remove_leaf(proxy_id);
        self.free_node(proxy_id);
    }

    /// Replaces the proxy's fat AABB and re-inserts the leaf in the tree.
    pub fn move_fat_proxy(&mut self, proxy_id: u32, fat_aabb: Aabb) {
        assert!(proxy_id < self.nodes.len() as u32);
        assert!(self.nodes[proxy_id as usize].is_leaf());

        self.remove_leaf(proxy_id);
        self.nodes[proxy_id as usize].aabb = fat_aabb;
        self.insert_leaf(proxy_id);
    }

    /// Updates the proxy AABB. Returns `true` if the node was re-inserted.
    pub fn move_proxy(&mut self, proxy_id: u32, aabb: Aabb, displacement: Vec3) -> bool {
        assert!(proxy_id < self.nodes.len() as u32);
        assert!(self.nodes[proxy_id as usize].is_leaf());

        let node_aabb = self.nodes[proxy_id as usize].aabb;
        if node_aabb.contains(&aabb) {
            return false;
        }

        let fat_aabb = aabb.fatten(displacement * AABB_MULTIPLIER, AABB_EXTENSION);
        self.move_fat_proxy(proxy_id, fat_aabb);
        true
    }

    fn insert_leaf(&mut self, leaf: u32) {
        if self.root == NULL_NODE {
            self.root = leaf;
            self.nodes[leaf as usize].parent = NULL_NODE;
            return;
        }

        // 1. Find the best sibling using Surface Area Heuristic (SAH)
        let leaf_aabb = self.nodes[leaf as usize].aabb;
        let mut index = self.root;

        while !self.nodes[index as usize].is_leaf() {
            let child1 = self.nodes[index as usize].child1;
            let child2 = self.nodes[index as usize].child2;

            let area = self.nodes[index as usize].aabb.surface_area();
            let combined_area = self.nodes[index as usize]
                .aabb
                .union(&leaf_aabb)
                .surface_area();

            // Cost of creating a new parent for this node and the new leaf
            let cost = 2.0 * combined_area;
            let inheritance_cost = 2.0 * (combined_area - area);

            // Cost of descending into child1
            let cost1 = if self.nodes[child1 as usize].is_leaf() {
                self.nodes[child1 as usize]
                    .aabb
                    .union(&leaf_aabb)
                    .surface_area()
                    + inheritance_cost
            } else {
                let old_area = self.nodes[child1 as usize].aabb.surface_area();
                let new_area = self.nodes[child1 as usize]
                    .aabb
                    .union(&leaf_aabb)
                    .surface_area();
                (new_area - old_area) + inheritance_cost
            };

            // Cost of descending into child2
            let cost2 = if self.nodes[child2 as usize].is_leaf() {
                self.nodes[child2 as usize]
                    .aabb
                    .union(&leaf_aabb)
                    .surface_area()
                    + inheritance_cost
            } else {
                let old_area = self.nodes[child2 as usize].aabb.surface_area();
                let new_area = self.nodes[child2 as usize]
                    .aabb
                    .union(&leaf_aabb)
                    .surface_area();
                (new_area - old_area) + inheritance_cost
            };

            // Descend according to the minimum cost
            if cost < cost1 && cost < cost2 {
                break;
            }

            index = if cost1 < cost2 { child1 } else { child2 };
        }

        let sibling = index;

        // 2. Create a new parent
        let old_parent = self.nodes[sibling as usize].parent;
        let new_parent = self.allocate_node();

        self.nodes[new_parent as usize].parent = old_parent;
        self.nodes[new_parent as usize].aabb = leaf_aabb.union(&self.nodes[sibling as usize].aabb);
        self.nodes[new_parent as usize].height = self.nodes[sibling as usize].height + 1;

        if old_parent != NULL_NODE {
            if self.nodes[old_parent as usize].child1 == sibling {
                self.nodes[old_parent as usize].child1 = new_parent;
            } else {
                self.nodes[old_parent as usize].child2 = new_parent;
            }
        } else {
            self.root = new_parent;
        }

        self.nodes[new_parent as usize].child1 = sibling;
        self.nodes[new_parent as usize].child2 = leaf;
        self.nodes[sibling as usize].parent = new_parent;
        self.nodes[leaf as usize].parent = new_parent;

        // 3. Walk back up the tree refitting AABBs and heights
        let mut walk = self.nodes[leaf as usize].parent;
        while walk != NULL_NODE {
            walk = self.balance(walk);

            let c1 = self.nodes[walk as usize].child1;
            let c2 = self.nodes[walk as usize].child2;

            self.nodes[walk as usize].height = 1 + self.nodes[c1 as usize]
                .height
                .max(self.nodes[c2 as usize].height);
            self.nodes[walk as usize].aabb = self.nodes[c1 as usize]
                .aabb
                .union(&self.nodes[c2 as usize].aabb);

            walk = self.nodes[walk as usize].parent;
        }
    }

    fn remove_leaf(&mut self, leaf: u32) {
        if leaf == self.root {
            self.root = NULL_NODE;
            return;
        }

        let parent = self.nodes[leaf as usize].parent;
        let grand_parent = self.nodes[parent as usize].parent;
        let sibling = if self.nodes[parent as usize].child1 == leaf {
            self.nodes[parent as usize].child2
        } else {
            self.nodes[parent as usize].child1
        };

        if grand_parent != NULL_NODE {
            if self.nodes[grand_parent as usize].child1 == parent {
                self.nodes[grand_parent as usize].child1 = sibling;
            } else {
                self.nodes[grand_parent as usize].child2 = sibling;
            }
            self.nodes[sibling as usize].parent = grand_parent;
            self.free_node(parent);

            let mut walk = grand_parent;
            while walk != NULL_NODE {
                walk = self.balance(walk);
                let c1 = self.nodes[walk as usize].child1;
                let c2 = self.nodes[walk as usize].child2;

                self.nodes[walk as usize].aabb = self.nodes[c1 as usize]
                    .aabb
                    .union(&self.nodes[c2 as usize].aabb);
                self.nodes[walk as usize].height = 1 + self.nodes[c1 as usize]
                    .height
                    .max(self.nodes[c2 as usize].height);

                walk = self.nodes[walk as usize].parent;
            }
        } else {
            self.root = sibling;
            self.nodes[sibling as usize].parent = NULL_NODE;
            self.free_node(parent);
        }
    }

    fn balance(&mut self, i_a: u32) -> u32 {
        let a = &self.nodes[i_a as usize];
        if a.is_leaf() || a.height < 2 {
            return i_a;
        }

        let i_b = a.child1;
        let i_c = a.child2;
        let balance = self.nodes[i_c as usize].height - self.nodes[i_b as usize].height;

        // Rotate C up
        if balance > 1 {
            let i_f = self.nodes[i_c as usize].child1;
            let i_g = self.nodes[i_c as usize].child2;

            self.nodes[i_c as usize].child1 = i_a;
            self.nodes[i_c as usize].parent = self.nodes[i_a as usize].parent;
            self.nodes[i_a as usize].parent = i_c;

            if self.nodes[i_c as usize].parent != NULL_NODE {
                let p = self.nodes[i_c as usize].parent as usize;
                if self.nodes[p].child1 == i_a {
                    self.nodes[p].child1 = i_c;
                } else {
                    self.nodes[p].child2 = i_c;
                }
            } else {
                self.root = i_c;
            }

            if self.nodes[i_f as usize].height > self.nodes[i_g as usize].height {
                self.nodes[i_c as usize].child2 = i_f;
                self.nodes[i_a as usize].child2 = i_g;
                self.nodes[i_g as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = self.nodes[i_b as usize]
                    .aabb
                    .union(&self.nodes[i_g as usize].aabb);
                self.nodes[i_c as usize].aabb = self.nodes[i_a as usize]
                    .aabb
                    .union(&self.nodes[i_f as usize].aabb);

                self.nodes[i_a as usize].height = 1 + self.nodes[i_b as usize]
                    .height
                    .max(self.nodes[i_g as usize].height);
                self.nodes[i_c as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_f as usize].height);
            } else {
                self.nodes[i_c as usize].child2 = i_g;
                self.nodes[i_a as usize].child2 = i_f;
                self.nodes[i_f as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = self.nodes[i_b as usize]
                    .aabb
                    .union(&self.nodes[i_f as usize].aabb);
                self.nodes[i_c as usize].aabb = self.nodes[i_a as usize]
                    .aabb
                    .union(&self.nodes[i_g as usize].aabb);

                self.nodes[i_a as usize].height = 1 + self.nodes[i_b as usize]
                    .height
                    .max(self.nodes[i_f as usize].height);
                self.nodes[i_c as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_g as usize].height);
            }

            return i_c;
        }

        // Rotate B up
        if balance < -1 {
            let i_d = self.nodes[i_b as usize].child1;
            let i_e = self.nodes[i_b as usize].child2;

            self.nodes[i_b as usize].child1 = i_a;
            self.nodes[i_b as usize].parent = self.nodes[i_a as usize].parent;
            self.nodes[i_a as usize].parent = i_b;

            if self.nodes[i_b as usize].parent != NULL_NODE {
                let p = self.nodes[i_b as usize].parent as usize;
                if self.nodes[p].child1 == i_a {
                    self.nodes[p].child1 = i_b;
                } else {
                    self.nodes[p].child2 = i_b;
                }
            } else {
                self.root = i_b;
            }

            if self.nodes[i_d as usize].height > self.nodes[i_e as usize].height {
                self.nodes[i_b as usize].child2 = i_d;
                self.nodes[i_a as usize].child1 = i_e;
                self.nodes[i_e as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = self.nodes[i_c as usize]
                    .aabb
                    .union(&self.nodes[i_e as usize].aabb);
                self.nodes[i_b as usize].aabb = self.nodes[i_a as usize]
                    .aabb
                    .union(&self.nodes[i_d as usize].aabb);

                self.nodes[i_a as usize].height = 1 + self.nodes[i_c as usize]
                    .height
                    .max(self.nodes[i_e as usize].height);
                self.nodes[i_b as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_d as usize].height);
            } else {
                self.nodes[i_b as usize].child2 = i_e;
                self.nodes[i_a as usize].child1 = i_d;
                self.nodes[i_d as usize].parent = i_a;
                self.nodes[i_a as usize].aabb = self.nodes[i_c as usize]
                    .aabb
                    .union(&self.nodes[i_d as usize].aabb);
                self.nodes[i_b as usize].aabb = self.nodes[i_a as usize]
                    .aabb
                    .union(&self.nodes[i_e as usize].aabb);

                self.nodes[i_a as usize].height = 1 + self.nodes[i_c as usize]
                    .height
                    .max(self.nodes[i_d as usize].height);
                self.nodes[i_b as usize].height = 1 + self.nodes[i_a as usize]
                    .height
                    .max(self.nodes[i_e as usize].height);
            }

            return i_b;
        }

        i_a
    }

    /// Spatial volume query finding all leaves intersecting `query_aabb`.
    pub fn query<F: FnMut(u32) -> bool>(&self, query_aabb: &Aabb, mut callback: F) {
        if self.root == NULL_NODE {
            return;
        }

        let mut stack = [0u32; TREE_STACK_SIZE];
        let mut stack_top = 0;
        stack[stack_top] = self.root;
        stack_top += 1;

        while stack_top > 0 {
            stack_top -= 1;
            let node_id = stack[stack_top];
            let node = &self.nodes[node_id as usize];
            if node.aabb.intersects(query_aabb) {
                if node.is_leaf() {
                    if !callback(node.user_data) {
                        return;
                    }
                } else {
                    assert!(
                        stack_top + 2 <= stack.len(),
                        "DynamicTree::query stack overflow"
                    );
                    stack[stack_top] = node.child1;
                    stack_top += 1;
                    stack[stack_top] = node.child2;
                    stack_top += 1;
                }
            }
        }
    }

    /// Casts a ray across the dynamic tree.
    pub fn ray_cast<F: FnMut(u32, f32) -> f32>(
        &self,
        ray: &Ray,
        mut max_fraction: f32,
        mut callback: F,
    ) {
        if self.root == NULL_NODE {
            return;
        }

        let mut stack = [0u32; TREE_STACK_SIZE];
        let mut stack_top = 0;
        stack[stack_top] = self.root;
        stack_top += 1;

        while stack_top > 0 {
            stack_top -= 1;
            let node_id = stack[stack_top];
            let node = &self.nodes[node_id as usize];
            if let Some(t) = node.aabb.intersect_ray(ray, 0.0, max_fraction) {
                if t <= max_fraction {
                    if node.is_leaf() {
                        let new_max = callback(node.user_data, t);
                        if new_max < max_fraction {
                            max_fraction = new_max;
                            if max_fraction == 0.0 {
                                break;
                            }
                        }
                    } else {
                        assert!(
                            stack_top + 2 <= stack.len(),
                            "DynamicTree::ray_cast stack overflow"
                        );
                        let c1 = &self.nodes[node.child1 as usize];
                        let c2 = &self.nodes[node.child2 as usize];
                        let t1 = c1.aabb.intersect_ray(ray, 0.0, max_fraction);
                        let t2 = c2.aabb.intersect_ray(ray, 0.0, max_fraction);
                        match (t1, t2) {
                            (Some(t1_val), Some(t2_val)) => {
                                if t1_val < t2_val {
                                    stack[stack_top] = node.child2;
                                    stack_top += 1;
                                    stack[stack_top] = node.child1;
                                    stack_top += 1;
                                } else {
                                    stack[stack_top] = node.child1;
                                    stack_top += 1;
                                    stack[stack_top] = node.child2;
                                    stack_top += 1;
                                }
                            }
                            (Some(_), None) => {
                                stack[stack_top] = node.child1;
                                stack_top += 1;
                            }
                            (None, Some(_)) => {
                                stack[stack_top] = node.child2;
                                stack_top += 1;
                            }
                            (None, None) => {}
                        }
                    }
                }
            }
        }
    }

    /// Casts an axis-aligned box `origin_aabb` along `translation` up to `max_fraction`,
    /// visiting overlapping leaf proxies in Box3D `b3DynamicTree_BoxCast` order (testing both
    /// children of each internal node, calling leaf callbacks before pushing internal children,
    /// and ordering two internal children so the one closer to `origin_aabb`'s center is
    /// searched first).
    pub fn box_cast<F>(
        &self,
        origin_aabb: &Aabb,
        translation: Vec3,
        mut max_fraction: f32,
        mut callback: F,
    ) where
        F: FnMut(u32, f32) -> f32,
    {
        if self.root == NULL_NODE {
            return;
        }

        let p1 = origin_aabb.center();
        let extension = origin_aabb.half_extents();
        let d = translation;
        let abs_d = d.abs();
        let total = |fraction: f32| {
            let t = translation * fraction;
            Aabb {
                min: origin_aabb.min.min(origin_aabb.min + t),
                max: origin_aabb.max.max(origin_aabb.max + t),
            }
        };
        let mut swept = total(max_fraction);

        let root_node = &self.nodes[self.root as usize];
        if root_node.is_leaf() {
            if swept.intersects(&root_node.aabb)
                && test_bounds_ray_overlap(
                    root_node.aabb.min - extension,
                    root_node.aabb.max + extension,
                    p1,
                    d,
                    abs_d,
                )
            {
                let _ = callback(root_node.user_data, max_fraction);
            }
            return;
        }

        let mut stack = [0u32; TREE_STACK_SIZE];
        let mut stack_top = 0;
        stack[stack_top] = self.root;
        stack_top += 1;

        while stack_top > 0 {
            stack_top -= 1;
            let parent_id = stack[stack_top];
            let parent = &self.nodes[parent_id as usize];
            let children = [parent.child1, parent.child2];
            let mut hits = [NULL_NODE; 2];
            let mut hit_count = 0;

            for &child_id in &children {
                let node = &self.nodes[child_id as usize];
                if !swept.intersects(&node.aabb) {
                    continue;
                }
                if !test_bounds_ray_overlap(
                    node.aabb.min - extension,
                    node.aabb.max + extension,
                    p1,
                    d,
                    abs_d,
                ) {
                    continue;
                }
                hits[hit_count] = child_id;
                hit_count += 1;
            }

            if hit_count == 2 {
                let n0 = &self.nodes[hits[0] as usize];
                let n1 = &self.nodes[hits[1] as usize];
                if !n0.is_leaf() && !n1.is_leaf() {
                    let d1 = (n0.aabb.center() - p1).length_squared();
                    let d2 = (n1.aabb.center() - p1).length_squared();
                    if d1 < d2 {
                        hits.swap(0, 1);
                    }
                }
            }

            for &child_id in &hits[..hit_count] {
                let node = &self.nodes[child_id as usize];
                if node.is_leaf() {
                    let value = callback(node.user_data, max_fraction);
                    if value == 0.0 {
                        return;
                    }
                    if value > 0.0 && value < max_fraction {
                        max_fraction = value;
                        swept = total(max_fraction);
                    }
                } else {
                    assert!(
                        stack_top < stack.len(),
                        "DynamicTree::box_cast stack overflow"
                    );
                    stack[stack_top] = child_id;
                    stack_top += 1;
                }
            }
        }
    }
}

/// Box3D `b3TestBoundsRayOverlap`: separating-axis test between the line of ray `(ray_start, ray_delta)`
/// and the axis-aligned box `[node_min, node_max]`.
#[inline]
fn test_bounds_ray_overlap(
    node_min: Vec3,
    node_max: Vec3,
    ray_start: Vec3,
    ray_delta: Vec3,
    abs_ray_delta: Vec3,
) -> bool {
    let node_center = 0.5 * (node_min + node_max);
    let node_extent = node_max - node_center;
    let rel_start = ray_start - node_center;
    let cross_abs = ray_delta.cross(rel_start).abs();
    let mod_cross = Vec3::new(
        abs_ray_delta.y * node_extent.z + abs_ray_delta.z * node_extent.y,
        abs_ray_delta.z * node_extent.x + abs_ray_delta.x * node_extent.z,
        abs_ray_delta.x * node_extent.y + abs_ray_delta.y * node_extent.x,
    );
    cross_abs.x <= mod_cross.x && cross_abs.y <= mod_cross.y && cross_abs.z <= mod_cross.z
}
