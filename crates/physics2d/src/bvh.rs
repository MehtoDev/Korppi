//! Dynamic AABB tree: a port of Box2D v3.2's `dynamic_tree.c`.
//!
//! Nodes come in sibling pairs at even indices, so a node's children are `pair` and `pair + 1`.
//! The root is node 0 and node 1 is always empty, which lets every traversal start from a pair.
//! Parents live in a separate array, and proxies (the stable ids of the leaves) in another.
//!
//! - **Insertion** picks the sibling by branch and bound on the surface area heuristic
//!   (`b2FindBestSibling`), then walks up refitting the ancestors and, for new proxies, rotating
//!   nodes where that shrinks them (`b2RotateNodes`).
//! - **Moved flags** live in the nodes. A proxy created or moved with the flag set marks itself
//!   and its ancestors, so pair finding only descends where something moved
//!   ([`moved_pairs`](DynamicTree2d::moved_pairs), [`moved_pairs_with`](DynamicTree2d::moved_pairs_with)).
//! - **Moving proxies** in a world grow their leaf in place:
//!   [`mark_proxy_moved`](DynamicTree2d::mark_proxy_moved) followed by one
//!   [`refit`](DynamicTree2d::refit) of the whole tree, or
//!   [`enlarge_proxy`](DynamicTree2d::enlarge_proxy), which grows the ancestors at once.
//! - **Rebuilding** ([`rebuild`](DynamicTree2d::rebuild)) sorts the moved subtrees again by
//!   median split, keeps the subtrees that did not move, and lays the whole tree out depth
//!   first, which the refit's single backward sweep needs.
//!
//! The world gives the tree the fat boxes it computes. For standalone use the tree can fatten
//! boxes itself: [`create_proxy`](DynamicTree2d::create_proxy) and
//! [`move_proxy`](DynamicTree2d::move_proxy) reach [`set_margin`](DynamicTree2d::set_margin)
//! (0.1 m by default) beyond the boxes they are given. Queries and ray casts traverse a stack on
//! the call stack, so they never allocate.

use korppi_math::{Aabb2d, Ray2d, Vec2};
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;

/// A missing node, parent or proxy.
const NULL_INDEX: u32 = u32::MAX;
/// The root node; node 1 beside it is always empty.
const ROOT: u32 = 0;
/// The node or something below it moved since pairs were last found.
const MOVED: u32 = 1 << 30;
/// The node is a leaf (or empty).
const LEAF: u32 = 1 << 31;
/// The low bits hold a leaf's proxy id or an internal node's first child.
const INDEX_MASK: u32 = !(MOVED | LEAF);
/// The flags of an empty node: a leaf with no proxy, so traversals skip it.
const EMPTY: u32 = INDEX_MASK | LEAF;

/// Default proxy margin of [`DynamicTree2d::create_proxy`], in metres.
const DEFAULT_AABB_MARGIN: f32 = 0.1;
/// How far [`DynamicTree2d::move_proxy`] stretches a box along the displacement, in steps.
const AABB_MULTIPLIER: f32 = 2.0;

/// Most entries a tree traversal keeps pending (Box2D `B2_TREE_STACK_SIZE`). A balanced tree
/// needs about its height, so this covers any tree that fits in memory.
const TREE_STACK_SIZE: usize = 512;

/// A node of the tree: a leaf holding a proxy, or an internal node over a pair of children.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(from = "SavedNode", into = "SavedNode")]
pub struct TreeNode2d {
    /// The box: a leaf's proxy box, or the union of the children's boxes.
    pub aabb: Aabb2d,
    /// Leaf and moved flags, and the proxy id of a leaf or the first child of an internal node.
    flag_index: u32,
    /// The user data of a leaf, or the height of an internal node.
    data: u32,
}

/// A tree node as saved: an empty box (infinite, as Box2D's empty node) as `None`, since
/// formats like JSON have no infinity.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct SavedNode {
    aabb: Option<Aabb2d>,
    flag_index: u32,
    data: u32,
}

impl From<TreeNode2d> for SavedNode {
    fn from(node: TreeNode2d) -> Self {
        let finite = node.aabb.min.is_finite() && node.aabb.max.is_finite();
        Self {
            aabb: finite.then_some(node.aabb),
            flag_index: node.flag_index,
            data: node.data,
        }
    }
}

impl From<SavedNode> for TreeNode2d {
    fn from(node: SavedNode) -> Self {
        Self {
            aabb: node.aabb.unwrap_or(Aabb2d::empty()),
            flag_index: node.flag_index,
            data: node.data,
        }
    }
}

impl TreeNode2d {
    const EMPTY: Self = Self {
        aabb: Aabb2d::empty(),
        flag_index: EMPTY,
        data: 0,
    };

    /// Whether this node is a leaf.
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.flag_index & LEAF != 0
    }

    /// Whether this node or something below it moved since its tree last found pairs.
    #[must_use]
    pub const fn is_moved(&self) -> bool {
        self.flag_index & MOVED != 0
    }

    /// The proxy id of a leaf, or the first of an internal node's two children.
    const fn index(&self) -> u32 {
        self.flag_index & INDEX_MASK
    }

    /// The height of the subtree under this node: 0 for a leaf.
    const fn height(&self) -> u32 {
        if self.is_leaf() {
            0
        } else {
            self.data
        }
    }

    /// The internal node over the children at `pair` (Box2D `b2MakeInternalNode`).
    fn internal(nodes: &[Self], pair: u32) -> Self {
        let (c1, c2) = (&nodes[pair as usize], &nodes[pair as usize + 1]);
        Self {
            aabb: c1.aabb.union(&c2.aabb),
            flag_index: pair | ((c1.flag_index | c2.flag_index) & MOVED),
            data: 1 + c1.height().max(c2.height()),
        }
    }

    const fn leaf(aabb: Aabb2d, proxy: u32, user_data: u32, moved: bool) -> Self {
        Self {
            aabb,
            flag_index: proxy | LEAF | if moved { MOVED } else { 0 },
            data: user_data,
        }
    }

    const fn set_child(&mut self, pair: u32) {
        self.flag_index = (self.flag_index & !INDEX_MASK) | pair;
    }
}

/// A leaf's stable id: where it is and what it holds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct TreeProxy {
    user_data: u32,
    /// The leaf node, or [`NULL_INDEX`] for a free proxy.
    node: u32,
    /// The next free proxy.
    next: u32,
}

/// The pending entries of a traversal, on the call stack so that queries do not allocate.
struct TreeStack<T: Copy> {
    items: [MaybeUninit<T>; TREE_STACK_SIZE],
    len: usize,
}

impl<T: Copy> TreeStack<T> {
    const fn new() -> Self {
        Self {
            items: [const { MaybeUninit::uninit() }; TREE_STACK_SIZE],
            len: 0,
        }
    }

    fn push(&mut self, item: T) {
        assert!(
            self.len < TREE_STACK_SIZE,
            "tree traversal deeper than {TREE_STACK_SIZE}"
        );
        self.items[self.len].write(item);
        self.len += 1;
    }

    fn pop(&mut self) -> Option<T> {
        self.len = self.len.checked_sub(1)?;
        // SAFETY: `0..=self.len` was written by `push` before `self.len` was incremented.
        Some(unsafe { self.items[self.len].assume_init() })
    }

    fn top_mut(&mut self) -> &mut T {
        debug_assert!(self.len > 0);
        // SAFETY: `self.len > 0`, and `0..self.len` were initialized by `push`.
        unsafe { self.items[self.len - 1].assume_init_mut() }
    }
}

/// Whether two boxes overlap, touching included (Box2D `b2AABB_Overlaps`).
fn overlaps(a: &Aabb2d, b: &Aabb2d) -> bool {
    a.min.x <= b.max.x && a.min.y <= b.max.y && b.min.x <= a.max.x && b.min.y <= a.max.y
}

/// Box2D's `b2EnlargeAABB`: grows `a` to contain `b`, returning whether it grew.
fn enlarge(a: &mut Aabb2d, b: &Aabb2d) -> bool {
    let grown = Aabb2d {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
    };
    let changed = grown != *a;
    *a = grown;
    changed
}

/// A node of the rebuild: where it is written, the pair its children go in, and its leaves.
#[derive(Debug, Clone, Copy)]
struct RebuildItem {
    node: u32,
    pair: u32,
    child_count: i32,
    start: usize,
    split: usize,
    end: usize,
}

/// A 2D dynamic AABB tree with Box2D v3.2's insertion, rotations, moved flags and rebuild.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicTree2d {
    nodes: Vec<TreeNode2d>,
    /// Parent of each node; free pairs chain through their first node's entry.
    parents: Vec<u32>,
    proxies: Vec<TreeProxy>,
    proxy_free: u32,
    proxy_count: usize,
    pair_free: u32,
    /// Whether every child sits after its parent, which the refit sweep needs.
    dfs_ordered: bool,
    /// The rebuild's output array, swapped with `nodes` when done.
    #[serde(skip)]
    swap_nodes: Vec<TreeNode2d>,
    #[serde(skip)]
    leaf_indices: Vec<usize>,
    #[serde(skip)]
    leaf_nodes: Vec<TreeNode2d>,
    #[serde(skip)]
    leaf_centers: Vec<Vec2>,
    /// How far [`create_proxy`](Self::create_proxy) and [`move_proxy`](Self::move_proxy) boxes
    /// reach beyond the boxes they are given.
    margin: f32,
}

impl Default for DynamicTree2d {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicTree2d {
    /// Creates an empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: vec![TreeNode2d::EMPTY, TreeNode2d::EMPTY],
            parents: vec![NULL_INDEX, NULL_INDEX],
            proxies: Vec::new(),
            proxy_free: NULL_INDEX,
            proxy_count: 0,
            pair_free: NULL_INDEX,
            dfs_ordered: true,
            swap_nodes: Vec::new(),
            leaf_indices: Vec::new(),
            leaf_nodes: Vec::new(),
            leaf_centers: Vec::new(),
            margin: DEFAULT_AABB_MARGIN,
        }
    }

    /// Sets how far [`create_proxy`](Self::create_proxy) and [`move_proxy`](Self::move_proxy)
    /// boxes reach beyond the boxes they are given, for proxies created or moved from now on.
    pub const fn set_margin(&mut self, margin: f32) {
        self.margin = margin;
    }

    /// Height of the tree: 0 for a single leaf, 0 when empty (Box2D `b2DynamicTree_GetHeight`).
    #[must_use]
    pub fn height(&self) -> i32 {
        self.nodes[ROOT as usize].height() as i32
    }

    /// Number of proxies.
    #[must_use]
    pub const fn proxy_count(&self) -> usize {
        self.proxy_count
    }

    /// Number of nodes in use, leaves and internal nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        (2 * self.proxy_count).saturating_sub(1)
    }

    /// The perimeters of the internal nodes below the root over the root's: what the surface
    /// area heuristic minimizes, as a measure of tree quality (Box2D
    /// `b2DynamicTree_GetAreaRatio`).
    #[must_use]
    pub fn area_ratio(&self) -> f32 {
        if self.proxy_count == 0 {
            return 0.0;
        }
        let root_area = self.nodes[ROOT as usize].aabb.perimeter();
        if root_area <= 0.0 {
            return 0.0;
        }
        // Free and empty nodes are tagged as leaves.
        let internal_area: f32 = self.nodes[2..]
            .iter()
            .filter(|node| !node.is_leaf())
            .map(|node| node.aabb.perimeter())
            .sum();
        internal_area / root_area
    }

    /// The box around every proxy.
    #[must_use]
    pub fn root_bounds(&self) -> Aabb2d {
        if self.proxy_count == 0 {
            return Aabb2d::new(Vec2::ZERO, Vec2::ZERO);
        }
        self.nodes[ROOT as usize].aabb
    }

    /// Whether a proxy moved since the moved flags were last cleared.
    #[must_use]
    pub fn has_moved(&self) -> bool {
        self.nodes[ROOT as usize].is_moved()
    }

    /// Whether [`rebuild`](Self::rebuild) has work: a proxy moved, or insertions left the nodes
    /// out of depth-first order.
    #[must_use]
    pub fn needs_rebuild(&self) -> bool {
        self.has_moved() || !self.dfs_ordered
    }

    /// Creates a proxy for `aabb` enlarged by the margin, holding `user_data`. Returns its id.
    pub fn create_proxy(&mut self, aabb: Aabb2d, user_data: u32) -> u32 {
        self.insert_proxy(aabb.fatten(self.margin), user_data, false)
    }

    /// Creates a proxy for exactly `aabb`, holding `user_data`, and returns its id (Box2D
    /// `b2CreateTreeProxyInternal`). A `moved` proxy marks itself and its ancestors.
    pub fn insert_proxy(&mut self, aabb: Aabb2d, user_data: u32, moved: bool) -> u32 {
        let proxy = if self.proxy_free == NULL_INDEX {
            self.proxies.push(TreeProxy {
                user_data,
                node: NULL_INDEX,
                next: NULL_INDEX,
            });
            self.proxies.len() as u32 - 1
        } else {
            let proxy = self.proxy_free;
            self.proxy_free = self.proxies[proxy as usize].next;
            self.proxies[proxy as usize] = TreeProxy {
                user_data,
                node: NULL_INDEX,
                next: NULL_INDEX,
            };
            proxy
        };
        self.proxy_count += 1;
        self.insert_leaf(aabb, proxy, moved, true);
        proxy
    }

    /// Destroys proxy `proxy`.
    pub fn destroy_proxy(&mut self, proxy: u32) {
        assert!(
            self.proxies[proxy as usize].node != NULL_INDEX,
            "proxy {proxy} is not in the tree"
        );
        self.remove_leaf(proxy);
        let entry = &mut self.proxies[proxy as usize];
        entry.node = NULL_INDEX;
        entry.next = self.proxy_free;
        self.proxy_free = proxy;
        self.proxy_count -= 1;
    }

    /// Moves proxy `proxy` to `aabb` unless its box already contains it; otherwise its new box
    /// reaches the margin beyond `aabb`, stretched along `displacement`. Returns whether the
    /// proxy was inserted again.
    pub fn move_proxy(&mut self, proxy: u32, aabb: Aabb2d, displacement: Vec2) -> bool {
        if self.proxy_aabb(proxy).contains_aabb(&aabb) {
            return false;
        }
        let mut fat = aabb.fatten(self.margin);
        if displacement.length_squared() > 0.0 {
            fat = fat.enlarge_by_displacement(displacement * AABB_MULTIPLIER);
        }
        self.reinsert_proxy(proxy, fat, false);
        true
    }

    /// Removes proxy `proxy` and inserts it again with box `aabb`, without rotations (Box2D
    /// `b2DynamicTree_MoveProxyInternal`).
    pub fn reinsert_proxy(&mut self, proxy: u32, aabb: Aabb2d, moved: bool) {
        self.remove_leaf(proxy);
        self.insert_leaf(aabb, proxy, moved, false);
    }

    /// Grows proxy `proxy`'s box to `aabb` in place, grows its ancestors to contain it, and
    /// marks them all moved (Box2D `b2DynamicTree_EnlargeProxy`). `aabb` must not fit in the
    /// proxy's box.
    pub fn enlarge_proxy(&mut self, proxy: u32, aabb: Aabb2d) {
        let mut index = self.proxies[proxy as usize].node;
        let leaf = &mut self.nodes[index as usize];
        debug_assert!(!leaf.aabb.contains_aabb(&aabb));
        leaf.aabb = aabb;
        leaf.flag_index |= MOVED;

        index = self.parents[index as usize];
        while index != NULL_INDEX {
            let node = &mut self.nodes[index as usize];
            let changed = enlarge(&mut node.aabb, &aabb);
            node.flag_index |= MOVED;
            index = self.parents[index as usize];
            if !changed {
                break;
            }
        }
        // Mark the rest of the way up, stopping at an ancestor marked before.
        while index != NULL_INDEX {
            let node = &mut self.nodes[index as usize];
            if node.is_moved() {
                break;
            }
            node.flag_index |= MOVED;
            index = self.parents[index as usize];
        }
    }

    /// Gives proxy `proxy` box `aabb` and marks it and its ancestors moved, leaving the
    /// ancestors' boxes to the next [`refit`](Self::refit) (Box2D
    /// `b2DynamicTree_MarkProxyMoved`). `aabb` must not fit in the proxy's box.
    pub fn mark_proxy_moved(&mut self, proxy: u32, aabb: Aabb2d) {
        let mut index = self.proxies[proxy as usize].node;
        let leaf = &mut self.nodes[index as usize];
        debug_assert!(!leaf.aabb.contains_aabb(&aabb));
        leaf.aabb = aabb;
        leaf.flag_index |= MOVED;
        index = self.parents[index as usize];
        while index != NULL_INDEX {
            let node = &mut self.nodes[index as usize];
            if node.is_moved() {
                break;
            }
            node.flag_index |= MOVED;
            index = self.parents[index as usize];
        }
    }

    /// Gives proxy `proxy` box `aabb` and atomically marks it and its ancestors moved (Box2D
    /// `b2DynamicTree_MarkProxyMoved`), safe to call concurrently for distinct proxies while no
    /// other method accesses `self`.
    ///
    /// # Safety
    /// Each concurrent call must pass a distinct `proxy`, and nothing else may read or write
    /// `self.nodes` until all concurrent calls have finished.
    pub(crate) unsafe fn mark_proxy_moved_atomic(&self, proxy: u32, aabb: Aabb2d) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let nodes = self.nodes.as_ptr() as *mut TreeNode2d;
        let mut index = self.proxies[proxy as usize].node as usize;
        // SAFETY: `proxy` is unique to this caller, so its leaf node `index` is not accessed
        // by any other thread's `aabb` write, and all `flag_index` accesses during the parallel
        // pass go through `AtomicU32`.
        unsafe {
            let leaf = nodes.add(index);
            debug_assert!(!(*leaf).aabb.contains_aabb(&aabb));
            (*leaf).aabb = aabb;
            let leaf_flags = &*(&raw mut (*leaf).flag_index).cast::<AtomicU32>();
            leaf_flags.fetch_or(MOVED, Ordering::Relaxed);
            index = self.parents[index] as usize;
            while index != NULL_INDEX as usize {
                let node = nodes.add(index);
                let flags = &*(&raw mut (*node).flag_index).cast::<AtomicU32>();
                if flags.load(Ordering::Relaxed) & MOVED != 0 {
                    break;
                }
                if flags.fetch_or(MOVED, Ordering::Relaxed) & MOVED != 0 {
                    break;
                }
                index = self.parents[index] as usize;
            }
        }
    }

    /// Grows the moved internal nodes to contain their children (Box2D `b2DynamicTree_Refit`):
    /// one sweep from the last pair to the first when the nodes are in depth-first order.
    pub fn refit(&mut self) {
        if !self.has_moved() {
            return;
        }
        if !self.dfs_ordered {
            self.refit_subtree(ROOT);
            return;
        }
        let nodes = &mut self.nodes;
        for pair in (0..nodes.len()).step_by(2).rev() {
            for i in pair..pair + 2 {
                let node = nodes[i];
                if node.flag_index & (LEAF | MOVED) == MOVED {
                    let children = node.index() as usize;
                    nodes[i].aabb = nodes[children].aabb.union(&nodes[children + 1].aabb);
                }
            }
        }
    }

    /// Clears every moved flag (Box2D `b2DynamicTree_ClearMoved`).
    pub fn clear_moved(&mut self) {
        let root = &mut self.nodes[ROOT as usize];
        if !root.is_moved() {
            return;
        }
        root.flag_index &= !MOVED;
        if root.is_leaf() {
            return;
        }
        let mut stack = TreeStack::new();
        stack.push(root.index());
        while let Some(pair) = stack.pop() {
            for i in pair..pair + 2 {
                let node = &mut self.nodes[i as usize];
                if node.is_moved() {
                    node.flag_index &= !MOVED;
                    if !node.is_leaf() {
                        stack.push(node.index());
                    }
                }
            }
        }
    }

    /// Pre-reserves the scratch vectors used by [`Self::rebuild`] on the calling thread so that a
    /// concurrent rebuild on a worker thread never allocates.
    pub(crate) fn reserve_rebuild_scratch(&mut self) {
        let leaves = self.proxy_count;
        if self.leaf_indices.capacity() < leaves {
            self.leaf_indices.reserve(leaves - self.leaf_indices.len());
        }
        if self.leaf_centers.capacity() < leaves {
            self.leaf_centers.reserve(leaves - self.leaf_centers.len());
        }
        if self.leaf_nodes.capacity() < leaves {
            self.leaf_nodes.reserve(leaves - self.leaf_nodes.len());
        }
        let nodes = self.nodes.len();
        if self.swap_nodes.capacity() < nodes {
            self.swap_nodes.reserve(nodes - self.swap_nodes.len());
        }
        if self.parents.len() < nodes {
            self.parents.resize(nodes, NULL_INDEX);
        }
    }

    /// Rebuilds the moved subtrees, or the whole tree if `full_build`, by median split, keeping
    /// the subtrees that did not move, and lays the tree out depth first. Clears the moved flags
    /// and returns how many leaves and kept subtrees were sorted (Box2D
    /// `b2DynamicTree_Rebuild`).
    pub fn rebuild(&mut self, full_build: bool) -> usize {
        if self.proxy_count == 0 {
            return 0;
        }
        let root = self.nodes[ROOT as usize];
        if !full_build && !root.is_moved() && self.dfs_ordered {
            return 0;
        }

        // Gather the build leaves: a marked internal node is descended and left behind;
        // everything else is a build leaf, kept subtrees included.
        self.leaf_indices.clear();
        self.leaf_nodes.clear();
        self.leaf_centers.clear();
        let mut stack = TreeStack::new();
        if !root.is_leaf() && (full_build || root.is_moved()) {
            stack.push(root.index());
        } else {
            self.push_build_leaf(root);
        }
        while let Some(pair) = stack.pop() {
            for i in pair..pair + 2 {
                let node = self.nodes[i as usize];
                if !node.is_leaf() && (full_build || node.is_moved()) {
                    stack.push(node.index());
                } else {
                    self.push_build_leaf(node);
                }
            }
        }

        let leaf_count = self.leaf_nodes.len();
        self.build_tree(leaf_count);
        std::mem::swap(&mut self.nodes, &mut self.swap_nodes);
        self.pair_free = NULL_INDEX;
        self.dfs_ordered = true;
        leaf_count
    }

    /// The box of proxy `proxy`.
    #[must_use]
    pub fn proxy_aabb(&self, proxy: u32) -> Aabb2d {
        self.nodes[self.proxies[proxy as usize].node as usize].aabb
    }

    /// The user data of proxy `proxy`.
    #[must_use]
    pub fn user_data(&self, proxy: u32) -> u32 {
        self.proxies[proxy as usize].user_data
    }

    /// Calls `callback(proxy, user_data)` for each proxy whose box overlaps `aabb`, in Box2D's
    /// order, until it returns `false` (Box2D `b2DynamicTree_Query`).
    pub fn query<F>(&self, aabb: &Aabb2d, mut callback: F)
    where
        F: FnMut(u32, u32) -> bool,
    {
        if self.proxy_count == 0 {
            return;
        }
        let mut stack = TreeStack::new();
        stack.push(self.root_pair());
        while let Some(pair) = stack.pop() {
            for node in &self.nodes[pair as usize..pair as usize + 2] {
                if !overlaps(aabb, &node.aabb) {
                    continue;
                }
                if node.is_leaf() {
                    if !callback(node.index(), node.data) {
                        return;
                    }
                } else {
                    stack.push(node.index());
                }
            }
        }
    }

    /// Casts `ray` up to `max_fraction` along it, calling `callback(proxy, user_data,
    /// max_fraction)` for each proxy whose box the remaining ray crosses, nearer subtrees first
    /// (Box2D `b2DynamicTree_CastRay`). The callback returns 0 to stop, a value in
    /// (0, `max_fraction`] to clip the ray there, or anything else to go on unchanged.
    pub fn ray_cast<F>(&self, ray: &Ray2d, max_fraction: f32, mut callback: F)
    where
        F: FnMut(u32, u32, f32) -> f32,
    {
        if self.proxy_count == 0 {
            return;
        }
        let p1 = ray.origin;
        let d = ray.direction;
        let r = d.normalize_or_zero();
        // Perpendicular to the ray: the separating axis of the segment (Gino, p80).
        let v = Vec2::new(-r.y, r.x);
        let abs_v = v.abs();
        let mut max_fraction = max_fraction;
        let mut p2 = p1 + d * max_fraction;
        let mut segment = Aabb2d {
            min: p1.min(p2),
            max: p1.max(p2),
        };

        let mut stack = TreeStack::new();
        stack.push(self.root_pair());
        while let Some(pair) = stack.pop() {
            let mut hits = [ROOT; 2];
            let mut hit_count = 0;
            for i in pair..pair + 2 {
                let node = &self.nodes[i as usize];
                if !overlaps(&segment, &node.aabb) {
                    continue;
                }
                let c = node.aabb.center();
                let h = node.aabb.half_extents();
                if abs_v.dot(h) < v.dot(p1 - c).abs() {
                    continue;
                }
                hits[hit_count] = i;
                hit_count += 1;
            }
            if hit_count == 2 {
                let (a, b) = (&self.nodes[hits[0] as usize], &self.nodes[hits[1] as usize]);
                // Push the nearer internal node last, so that it is searched first.
                if !a.is_leaf()
                    && !b.is_leaf()
                    && a.aabb.center().distance_squared(p1) < b.aabb.center().distance_squared(p1)
                {
                    hits.swap(0, 1);
                }
            }
            for &i in &hits[..hit_count] {
                let node = &self.nodes[i as usize];
                if !node.is_leaf() {
                    stack.push(node.index());
                    continue;
                }
                let value = callback(node.index(), node.data, max_fraction);
                if value == 0.0 {
                    return;
                }
                if 0.0 < value && value <= max_fraction {
                    max_fraction = value;
                    p2 = p1 + d * max_fraction;
                    segment = Aabb2d {
                        min: p1.min(p2),
                        max: p1.max(p2),
                    };
                }
            }
        }
    }

    /// Casts `aabb` along `translation` up to `max_fraction` of it (Box2D
    /// `b2DynamicTree_CastBox`), calling `callback(proxy, user_data, max_fraction)` for each
    /// proxy the swept box may touch, nearest subtrees first. The callback returns 0 to stop,
    /// a fraction in (0, `max_fraction`) to shorten the cast, or anything else to go on.
    pub fn box_cast<F>(&self, aabb: &Aabb2d, translation: Vec2, max_fraction: f32, mut callback: F)
    where
        F: FnMut(u32, u32, f32) -> f32,
    {
        if self.proxy_count == 0 {
            return;
        }
        let p1 = aabb.center();
        let extension = aabb.half_extents();
        // Perpendicular to the cast: the separating axis of the segment (Gino, p80).
        let v = Vec2::new(-translation.y, translation.x);
        let abs_v = v.abs();
        let mut max_fraction = max_fraction;
        let total = |fraction: f32| {
            let t = translation * fraction;
            Aabb2d {
                min: aabb.min.min(aabb.min + t),
                max: aabb.max.max(aabb.max + t),
            }
        };
        let mut swept = total(max_fraction);

        let mut stack = TreeStack::new();
        stack.push(self.root_pair());
        while let Some(pair) = stack.pop() {
            let mut hits = [ROOT; 2];
            let mut hit_count = 0;
            for i in pair..pair + 2 {
                let node = &self.nodes[i as usize];
                if !overlaps(&swept, &node.aabb) {
                    continue;
                }
                let c = node.aabb.center();
                let h = node.aabb.half_extents() + extension;
                if abs_v.dot(h) < v.dot(p1 - c).abs() {
                    continue;
                }
                hits[hit_count] = i;
                hit_count += 1;
            }
            if hit_count == 2 {
                let (a, b) = (&self.nodes[hits[0] as usize], &self.nodes[hits[1] as usize]);
                // Push the nearer internal node last, so that it is searched first.
                if !a.is_leaf()
                    && !b.is_leaf()
                    && a.aabb.center().distance_squared(p1) < b.aabb.center().distance_squared(p1)
                {
                    hits.swap(0, 1);
                }
            }
            for &i in &hits[..hit_count] {
                let node = &self.nodes[i as usize];
                if !node.is_leaf() {
                    stack.push(node.index());
                    continue;
                }
                let value = callback(node.index(), node.data, max_fraction);
                if value == 0.0 {
                    return;
                }
                if 0.0 < value && value < max_fraction {
                    max_fraction = value;
                    swept = total(max_fraction);
                }
            }
        }
    }

    /// Calls `pair(a, b)` with the user data of each pair of proxies whose boxes overlap and of
    /// which at least one moved, once per pair (Box2D's self pairs in
    /// `b2UpdateBroadPhasePairs`). Every marked internal node collides its two subtrees.
    pub fn moved_pairs<F: FnMut(u32, u32)>(&self, pair: F) {
        self.moved_pairs_range(1, self.sibling_pair_count(), pair);
    }

    /// Number of sibling pairs in `nodes` (`k in 1..self.sibling_pair_count()`).
    #[must_use]
    pub(crate) fn sibling_pair_count(&self) -> usize {
        self.nodes.len() / 2
    }

    /// Calls `pair(a, b)` for sibling pairs `first = 2 * k` with `k in start..end` (Box2D
    /// `b2SelfPairsTask`).
    pub(crate) fn moved_pairs_range<F: FnMut(u32, u32)>(
        &self,
        start: usize,
        end: usize,
        mut pair: F,
    ) {
        for k in start..end {
            let first = 2 * k;
            let (a, b) = (self.nodes[first], self.nodes[first + 1]);
            if (a.flag_index | b.flag_index) & MOVED != 0 {
                collide_subtrees(&self.nodes, &self.nodes, &a, &b, &mut pair);
            }
        }
    }

    /// Calls `pair(a, b)` with the user data of each proxy `a` of this tree and `b` of `other`
    /// whose boxes overlap and of which at least one moved (Box2D's cross pairs in
    /// `b2UpdateBroadPhasePairs`).
    pub fn moved_pairs_with<F: FnMut(u32, u32)>(&self, other: &Self, mut pair: F) {
        let (a, b) = (self.nodes[ROOT as usize], other.nodes[ROOT as usize]);
        collide_subtrees(&self.nodes, &other.nodes, &a, &b, &mut pair);
    }

    /// Calls `pair(smaller, larger)` with the proxy ids of every pair of proxies whose boxes
    /// overlap, once per pair, by smaller proxy id.
    pub fn overlapping_pairs<F: FnMut(u32, u32)>(&self, mut pair: F) {
        for (proxy, entry) in self.proxies.iter().enumerate() {
            if entry.node == NULL_INDEX {
                continue;
            }
            let proxy = proxy as u32;
            self.query(&self.proxy_aabb(proxy), |other, _| {
                if proxy < other {
                    pair(proxy, other);
                }
                true
            });
        }
    }

    /// The pair a traversal starts from: the root's children, or the root beside the empty
    /// node when the root is a leaf.
    fn root_pair(&self) -> u32 {
        let root = &self.nodes[ROOT as usize];
        if root.is_leaf() {
            ROOT
        } else {
            root.index()
        }
    }

    /// A free pair of sibling nodes: the last freed, or two new ones.
    fn allocate_pair(&mut self) -> u32 {
        if self.pair_free != NULL_INDEX {
            let pair = self.pair_free;
            self.pair_free = self.parents[pair as usize];
            return pair;
        }
        let pair = self.nodes.len() as u32;
        self.nodes.extend([TreeNode2d::EMPTY; 2]);
        // After a rebuild the parents can outnumber the nodes.
        if self.parents.len() < self.nodes.len() {
            self.parents.resize(self.nodes.len(), NULL_INDEX);
        }
        pair
    }

    fn free_pair(&mut self, pair: u32) {
        let pair = pair as usize;
        self.nodes[pair] = TreeNode2d::EMPTY;
        self.nodes[pair + 1] = TreeNode2d::EMPTY;
        self.parents[pair] = self.pair_free;
        self.parents[pair + 1] = NULL_INDEX;
        self.pair_free = pair as u32;
    }

    /// Points what hangs below node `index` back at it: a leaf's proxy, or the children's
    /// parent (Box2D `b2LinkChildren`).
    fn link_children(&mut self, index: u32) {
        let node = self.nodes[index as usize];
        if node.is_leaf() {
            self.proxies[node.index() as usize].node = index;
        } else {
            let pair = node.index() as usize;
            self.parents[pair] = index;
            self.parents[pair + 1] = index;
        }
    }

    /// Whether node `index`'s children sit after it, as the refit sweep needs.
    fn is_ordered(&self, index: u32) -> bool {
        let node = &self.nodes[index as usize];
        node.is_leaf() || index < node.index()
    }

    /// The sibling for a new leaf with box `box_d` (Box2D `b2FindBestSibling`): the node whose
    /// union with the leaf, plus the growth of its ancestors, costs the least perimeter. One
    /// greedy path is followed, pruned by the lower bound of each subtree's cost.
    fn find_best_sibling(&self, box_d: &Aabb2d) -> u32 {
        let center_d = box_d.center();
        let area_d = box_d.perimeter();
        let nodes = &self.nodes;
        let mut index = ROOT;
        if nodes[index as usize].is_leaf() {
            return index;
        }
        let root_box = nodes[index as usize].aabb;
        let mut area_base = root_box.perimeter();
        let mut direct_cost = root_box.union(box_d).perimeter();
        let mut inherited_cost = 0.0;
        let mut best_sibling = index;
        let mut best_cost = direct_cost;

        loop {
            let child1 = nodes[index as usize].index();
            let child2 = child1 + 1;

            // Cost of a new parent for this node and the leaf.
            let cost = direct_cost + inherited_cost;
            if cost < best_cost {
                best_sibling = index;
                best_cost = cost;
            }
            // Growth the children's ancestors see.
            inherited_cost += direct_cost - area_base;

            let (node1, node2) = (&nodes[child1 as usize], &nodes[child2 as usize]);
            let (leaf1, leaf2) = (node1.is_leaf(), node2.is_leaf());

            let mut lower_cost1 = f32::MAX;
            let direct_cost1 = node1.aabb.union(box_d).perimeter();
            let mut area1 = 0.0;
            if leaf1 {
                let cost1 = direct_cost1 + inherited_cost;
                if cost1 < best_cost {
                    best_sibling = child1;
                    best_cost = cost1;
                }
            } else {
                area1 = node1.aabb.perimeter();
                // Child 1 as the sibling, or a descendant at the lower bound of its cost.
                lower_cost1 = inherited_cost + direct_cost1 + (area_d - area1).min(0.0);
            }

            let mut lower_cost2 = f32::MAX;
            let direct_cost2 = node2.aabb.union(box_d).perimeter();
            let mut area2 = 0.0;
            if leaf2 {
                let cost2 = direct_cost2 + inherited_cost;
                if cost2 < best_cost {
                    best_sibling = child2;
                    best_cost = cost2;
                }
            } else {
                area2 = node2.aabb.perimeter();
                lower_cost2 = inherited_cost + direct_cost2 + (area_d - area2).min(0.0);
            }

            if leaf1 && leaf2 {
                break;
            }
            // Can the cost still go down?
            if best_cost <= lower_cost1 && best_cost <= lower_cost2 {
                break;
            }
            if lower_cost1 == lower_cost2 && !leaf1 {
                // No clear choice, as when both children contain the leaf: take the nearer.
                lower_cost1 = (node1.aabb.center() - center_d).length_squared();
                lower_cost2 = (node2.aabb.center() - center_d).length_squared();
            }
            if lower_cost1 < lower_cost2 && !leaf1 {
                index = child1;
                area_base = area1;
                direct_cost = direct_cost1;
            } else {
                index = child2;
                area_base = area2;
                direct_cost = direct_cost2;
            }
        }
        best_sibling
    }

    /// Swaps child `down` of a node with grandchild `up` under the other child, and refits that
    /// other child (Box2D `b2SwapNodes`).
    fn swap_with_grandchild(&mut self, down: u32, up: u32) {
        self.nodes.swap(down as usize, up as usize);
        self.link_children(down);
        self.link_children(up);
        if !self.is_ordered(down) || !self.is_ordered(up) {
            self.dfs_ordered = false;
        }
        // The sibling of the node that went down holds a different subtree now.
        let c = down ^ 1;
        self.nodes[c as usize] = TreeNode2d::internal(&self.nodes, self.nodes[c as usize].index());
    }

    /// Swaps a child of node `a` with a grandchild under its other child when that shrinks the
    /// other child most (Box2D `b2RotateNodes`). With `a` over `b` and `c`, `b` over `d` and
    /// `e`, and `c` over `f` and `g`: `b` may trade places with `f` or `g`, or `c` with `d` or
    /// `e`.
    fn rotate_nodes(&mut self, a: u32) {
        let nodes = &self.nodes;
        let b = nodes[a as usize].index();
        let c = b + 1;
        let (node_b, node_c) = (&nodes[b as usize], &nodes[c as usize]);
        if node_b.is_leaf() && node_c.is_leaf() {
            return;
        }
        let mut best = None;
        let mut best_delta = 0.0;
        if !node_c.is_leaf() {
            let f = node_c.index();
            let g = f + 1;
            let area_c = node_c.aabb.perimeter();
            // b <-> f leaves c over (b, g).
            let delta_bf = node_b.aabb.union(&nodes[g as usize].aabb).perimeter() - area_c;
            if delta_bf < best_delta {
                best = Some((b, f));
                best_delta = delta_bf;
            }
            // b <-> g leaves c over (f, b).
            let delta_bg = node_b.aabb.union(&nodes[f as usize].aabb).perimeter() - area_c;
            if delta_bg < best_delta {
                best = Some((b, g));
                best_delta = delta_bg;
            }
        }
        if !node_b.is_leaf() {
            let d = node_b.index();
            let e = d + 1;
            let area_b = node_b.aabb.perimeter();
            // c <-> d leaves b over (c, e).
            let delta_cd = node_c.aabb.union(&nodes[e as usize].aabb).perimeter() - area_b;
            if delta_cd < best_delta {
                best = Some((c, d));
                best_delta = delta_cd;
            }
            // c <-> e leaves b over (d, c).
            let delta_ce = node_c.aabb.union(&nodes[d as usize].aabb).perimeter() - area_b;
            if delta_ce < best_delta {
                best = Some((c, e));
            }
        }
        if let Some((down, up)) = best {
            self.swap_with_grandchild(down, up);
        }
    }

    /// Inserts proxy `proxy`'s leaf with box `aabb` (Box2D `b2InsertLeaf`). The sibling's slot
    /// becomes the new parent; the sibling moves down into a new pair beside the leaf.
    fn insert_leaf(&mut self, aabb: Aabb2d, proxy: u32, moved: bool, rotate: bool) {
        let leaf = TreeNode2d::leaf(aabb, proxy, self.proxies[proxy as usize].user_data, moved);
        if self.nodes[ROOT as usize].flag_index == EMPTY {
            self.nodes[ROOT as usize] = leaf;
            self.proxies[proxy as usize].node = ROOT;
            return;
        }

        let sibling = self.find_best_sibling(&aabb);
        let pair = self.allocate_pair();
        self.nodes[pair as usize] = self.nodes[sibling as usize];
        self.nodes[pair as usize + 1] = leaf;
        self.parents[pair as usize] = sibling;
        self.parents[pair as usize + 1] = sibling;
        self.link_children(pair);
        self.proxies[proxy as usize].node = pair + 1;
        self.nodes[sibling as usize] = TreeNode2d::internal(&self.nodes, pair);
        if !self.is_ordered(sibling) || !self.is_ordered(pair) {
            self.dfs_ordered = false;
        }

        // Walk back up refitting the ancestors, the root included.
        let mut index = sibling;
        while index != NULL_INDEX {
            if rotate {
                self.rotate_nodes(index);
            }
            self.nodes[index as usize] =
                TreeNode2d::internal(&self.nodes, self.nodes[index as usize].index());
            index = self.parents[index as usize];
        }
    }

    /// Removes proxy `proxy`'s leaf (Box2D `b2RemoveLeaf`): its sibling takes the parent's slot
    /// and the pair is freed.
    fn remove_leaf(&mut self, proxy: u32) {
        let leaf = self.proxies[proxy as usize].node;
        if leaf == ROOT {
            self.nodes[ROOT as usize] = TreeNode2d::EMPTY;
            return;
        }
        let parent = self.parents[leaf as usize];
        self.nodes[parent as usize] = self.nodes[(leaf ^ 1) as usize];
        self.link_children(parent);
        self.free_pair(leaf & !1);

        let mut index = self.parents[parent as usize];
        while index != NULL_INDEX {
            self.nodes[index as usize] =
                TreeNode2d::internal(&self.nodes, self.nodes[index as usize].index());
            index = self.parents[index as usize];
        }
    }

    /// Recomputes the moved internal boxes under `root` when the nodes are not in depth-first
    /// order (Box2D `b2RefitSubtree`). A negative entry is a deferred union, pushed before the
    /// children so that both are done when it runs.
    fn refit_subtree(&mut self, root: u32) {
        let node = self.nodes[root as usize];
        if node.is_leaf() || !node.is_moved() {
            return;
        }
        let mut stack = TreeStack::new();
        stack.push(!i64::from(root));
        stack.push(i64::from(node.index()));
        stack.push(i64::from(node.index()) + 1);
        while let Some(item) = stack.pop() {
            if item < 0 {
                let index = (!item) as usize;
                let pair = self.nodes[index].index() as usize;
                self.nodes[index].aabb = self.nodes[pair].aabb.union(&self.nodes[pair + 1].aabb);
                continue;
            }
            let node = self.nodes[item as usize];
            if node.is_leaf() || !node.is_moved() {
                continue;
            }
            let pair = i64::from(node.index());
            stack.push(!item);
            stack.push(pair);
            stack.push(pair + 1);
        }
    }

    fn push_build_leaf(&mut self, mut node: TreeNode2d) {
        node.flag_index &= !MOVED;
        self.leaf_indices.push(self.leaf_nodes.len());
        self.leaf_centers.push(node.aabb.center());
        self.leaf_nodes.push(node);
    }

    /// Two new sibling nodes at the end of the rebuild's array, under `parent`.
    fn bump_pair(&mut self, parent: u32) -> u32 {
        let pair = self.swap_nodes.len() as u32;
        self.swap_nodes.extend([TreeNode2d::EMPTY; 2]);
        if self.parents.len() < self.swap_nodes.len() {
            self.parents.resize(self.swap_nodes.len(), NULL_INDEX);
        }
        self.parents[pair as usize] = parent;
        self.parents[pair as usize + 1] = parent;
        pair
    }

    /// Builds the new tree in `swap_nodes` over the gathered build leaves (Box2D
    /// `b2BuildTree`): each node splits its leaves at the median of their centers along the
    /// longer axis.
    fn build_tree(&mut self, leaf_count: usize) {
        self.swap_nodes.clear();
        self.swap_nodes.reserve(self.nodes.len());
        self.swap_nodes.extend([TreeNode2d::EMPTY; 2]);
        if self.parents.len() < self.nodes.len() {
            self.parents.resize(self.nodes.len(), NULL_INDEX);
        }
        self.parents[ROOT as usize] = NULL_INDEX;
        self.parents[ROOT as usize + 1] = NULL_INDEX;
        if leaf_count == 1 {
            let leaf = self.leaf_nodes[self.leaf_indices[0]];
            self.place_leaf(leaf, ROOT);
            return;
        }

        let mut stack = TreeStack::new();
        let pair = self.bump_pair(ROOT);
        let split = partition_mid(&mut self.leaf_indices, &mut self.leaf_centers);
        stack.push(RebuildItem {
            node: ROOT,
            pair,
            child_count: -1,
            start: 0,
            split,
            end: leaf_count,
        });
        loop {
            let top = stack.len - 1;
            let item = stack.top_mut();
            item.child_count += 1;
            let item = *item;
            if item.child_count == 2 {
                // Both children written: the parent can be finished.
                self.swap_nodes[item.node as usize] =
                    TreeNode2d::internal(&self.swap_nodes, item.pair);
                if top == 0 {
                    break;
                }
                stack.len -= 1;
                continue;
            }
            let slot = item.child_count as u32;
            let (start, end) = if slot == 0 {
                (item.start, item.split)
            } else {
                (item.split, item.end)
            };
            let node = item.pair + slot;
            if end - start == 1 {
                let leaf = self.leaf_nodes[self.leaf_indices[start]];
                self.place_leaf(leaf, node);
                continue;
            }
            let pair = self.bump_pair(node);
            let split = start
                + partition_mid(
                    &mut self.leaf_indices[start..end],
                    &mut self.leaf_centers[start..end],
                );
            stack.push(RebuildItem {
                node,
                pair,
                child_count: -1,
                start,
                split,
                end,
            });
        }
    }

    /// Writes build leaf `leaf` at `index` of the new array: a proxy, or a kept subtree copied
    /// whole (Box2D `b2PlaceLeaf`).
    fn place_leaf(&mut self, leaf: TreeNode2d, index: u32) {
        if leaf.is_leaf() {
            self.swap_nodes[index as usize] = leaf;
            self.proxies[leaf.index() as usize].node = index;
        } else {
            self.copy_subtree(leaf, index);
        }
    }

    /// Copies the kept subtree under `node` from the old array into the new one depth first,
    /// with `node` at `index` (Box2D `b2CopySubtree`).
    fn copy_subtree(&mut self, mut node: TreeNode2d, index: u32) {
        let mut stack = TreeStack::new();
        let mut old_pair = node.index();
        let mut new_pair = self.bump_pair(index);
        node.set_child(new_pair);
        self.swap_nodes[index as usize] = node;
        loop {
            let (left, right) = (
                self.nodes[old_pair as usize],
                self.nodes[old_pair as usize + 1],
            );
            self.swap_nodes[new_pair as usize] = left;
            self.swap_nodes[new_pair as usize + 1] = right;
            if right.is_leaf() {
                self.proxies[right.index() as usize].node = new_pair + 1;
            } else {
                stack.push((right.index(), new_pair + 1));
            }
            if !left.is_leaf() {
                // Descend into the left subtree.
                let left_index = new_pair;
                old_pair = left.index();
                new_pair = self.bump_pair(left_index);
                self.swap_nodes[left_index as usize].set_child(new_pair);
                continue;
            }
            self.proxies[left.index() as usize].node = new_pair;
            // Descend a right subtree pushed before.
            let Some((old, new_index)) = stack.pop() else {
                break;
            };
            old_pair = old;
            new_pair = self.bump_pair(new_index);
            self.swap_nodes[new_index as usize].set_child(new_pair);
        }
    }
}

/// Splits `indices` and `centers` along the longer axis of the centers' bounds at its middle,
/// by Hoare partition, and returns the size of the first part (Box2D `b2PartitionMid`).
fn partition_mid(indices: &mut [usize], centers: &mut [Vec2]) -> usize {
    let count = indices.len();
    if count <= 2 {
        return count / 2;
    }
    let mut lower = centers[0];
    let mut upper = centers[0];
    for &c in &centers[1..] {
        lower = lower.min(c);
        upper = upper.max(c);
    }
    let d = upper - lower;
    let c = Vec2::new(0.5 * (lower.x + upper.x), 0.5 * (lower.y + upper.y));

    let (mut i1, mut i2) = (0, count);
    if d.x > d.y {
        let pivot = c.x;
        while i1 < i2 {
            while i1 < i2 && centers[i1].x < pivot {
                i1 += 1;
            }
            while i1 < i2 && centers[i2 - 1].x >= pivot {
                i2 -= 1;
            }
            if i1 < i2 {
                indices.swap(i1, i2 - 1);
                centers.swap(i1, i2 - 1);
                i1 += 1;
                i2 -= 1;
            }
        }
    } else {
        let pivot = c.y;
        while i1 < i2 {
            while i1 < i2 && centers[i1].y < pivot {
                i1 += 1;
            }
            while i1 < i2 && centers[i2 - 1].y >= pivot {
                i2 -= 1;
            }
            if i1 < i2 {
                indices.swap(i1, i2 - 1);
                centers.swap(i1, i2 - 1);
                i1 += 1;
                i2 -= 1;
            }
        }
    }
    if i1 > 0 && i1 < count {
        i1
    } else {
        count / 2
    }
}

/// Collides subtree `a` of `nodes_a` with subtree `b` of `nodes_b`, calling `pair` with the user
/// data of every two overlapping leaves of which at least one moved (Box2D
/// `b2CollideCrossPairs`). Subtrees of one tree must be disjoint, so no pair comes twice.
fn collide_subtrees<F: FnMut(u32, u32)>(
    nodes_a: &[TreeNode2d],
    nodes_b: &[TreeNode2d],
    a: &TreeNode2d,
    b: &TreeNode2d,
    pair: &mut F,
) {
    if !(a.is_moved() || b.is_moved()) || !overlaps(&a.aabb, &b.aabb) {
        return;
    }
    match (a.is_leaf(), b.is_leaf()) {
        (true, true) => {
            pair(a.data, b.data);
            return;
        }
        (true, false) => {
            collide_leaf(a, nodes_b, b.index(), pair);
            return;
        }
        (false, true) => {
            collide_leaf(b, nodes_a, a.index(), pair);
            return;
        }
        (false, false) => {}
    }
    let mut stack = TreeStack::new();
    stack.push((a.index(), b.index()));
    while let Some((pair_a, pair_b)) = stack.pop() {
        for lane in 0..4 {
            let node_a = &nodes_a[pair_a as usize + (lane >> 1)];
            let node_b = &nodes_b[pair_b as usize + (lane & 1)];
            if (node_a.is_moved() || node_b.is_moved()) && overlaps(&node_a.aabb, &node_b.aabb) {
                expand_pair(nodes_a, nodes_b, node_a, node_b, &mut stack, pair);
            }
        }
    }
}

/// Two overlapping nodes, at least one moved: a pair of leaves, a leaf against a subtree, or
/// two subtrees whose children are collided next (Box2D `b2ExpandPair`).
fn expand_pair<F: FnMut(u32, u32)>(
    nodes_a: &[TreeNode2d],
    nodes_b: &[TreeNode2d],
    a: &TreeNode2d,
    b: &TreeNode2d,
    stack: &mut TreeStack<(u32, u32)>,
    pair: &mut F,
) {
    match (a.is_leaf(), b.is_leaf()) {
        (true, true) => pair(a.data, b.data),
        (true, false) => collide_leaf(a, nodes_b, b.index(), pair),
        (false, true) => collide_leaf(b, nodes_a, a.index(), pair),
        (false, false) => stack.push((a.index(), b.index())),
    }
}

/// Collides leaf `leaf` with the subtree under the children at `first` of `nodes` (Box2D
/// `b2CollideProxyAndSubtree`): a leaf that did not move only meets moved nodes.
fn collide_leaf<F: FnMut(u32, u32)>(
    leaf: &TreeNode2d,
    nodes: &[TreeNode2d],
    first: u32,
    pair: &mut F,
) {
    let mut stack = TreeStack::new();
    stack.push(first);
    while let Some(first) = stack.pop() {
        for node in &nodes[first as usize..first as usize + 2] {
            if !(leaf.is_moved() || node.is_moved()) || !overlaps(&leaf.aabb, &node.aabb) {
                continue;
            }
            if node.is_leaf() {
                pair(leaf.data, node.data);
            } else {
                stack.push(node.index());
            }
        }
    }
}
