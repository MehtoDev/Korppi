//! Triangle meshes with Box3D's bounding volume hierarchy (`mesh.c`, `b3MeshData`).
//!
//! [`Mesh3d::new`] drops degenerate triangles, builds a BVH of axis-aligned boxes over the rest
//! (binned SAH splits, or median splits for grid-like meshes), stores the triangles in the
//! tree's depth-first order, so every leaf holds a run of them, and flags each edge two
//! triangles share as concave, convex or flat for the mesh contacts. Triangles are one-sided:
//! they face where their vertices turn counter-clockwise. Queries walk the tree with a stack of
//! at most [`MESH_STACK_SIZE`] nodes and never allocate.

use crate::contact_solver::LINEAR_SLOP;
use crate::math::modified_cross;
use korppi_math::Vec3;
use serde::{Deserialize, Serialize};

/// Box3D `B3_BIN_COUNT`: the buckets of a binned SAH split.
const BIN_COUNT: usize = 8;

/// Box3D `B3_DESIRED_TRIANGLES_PER_LEAF`.
const DESIRED_TRIANGLES_PER_LEAF: usize = 4;

/// Box3D `B3_MAXIMUM_TRIANGLES_PER_LEAF`: a node this full is split even without a good split.
const MAXIMUM_TRIANGLES_PER_LEAF: usize = 8;

/// Box3D `B3_MESH_STACK_SIZE`: the deepest a query walks; a deeper tree is rejected.
pub const MESH_STACK_SIZE: usize = 256;

/// Box3D `B3_HUGE` (float positions): bounds beyond it are rejected.
const HUGE: f32 = 1.0e5;

/// The node kind of a leaf; inner nodes store their split axis, 0 to 2.
const LEAF: u8 = 3;

/// Edge flags of a triangle (Box3D `b3MeshEdgeFlags`): bit `e` marks edge `e` concave, bit
/// `e + 4` marks it convex ("inverse concave"); both mark it flat.
pub(crate) const CONCAVE_EDGE1: u8 = 0x01;
pub(crate) const CONCAVE_EDGE2: u8 = 0x02;
pub(crate) const CONCAVE_EDGE3: u8 = 0x04;
pub(crate) const INVERSE_CONCAVE_EDGE1: u8 = 0x10;
pub(crate) const INVERSE_CONCAVE_EDGE2: u8 = 0x20;
pub(crate) const INVERSE_CONCAVE_EDGE3: u8 = 0x40;
pub(crate) const FLAT_EDGE1: u8 = CONCAVE_EDGE1 | INVERSE_CONCAVE_EDGE1;
pub(crate) const FLAT_EDGE2: u8 = CONCAVE_EDGE2 | INVERSE_CONCAVE_EDGE2;
pub(crate) const FLAT_EDGE3: u8 = CONCAVE_EDGE3 | INVERSE_CONCAVE_EDGE3;
pub(crate) const ALL_FLAT_EDGES: u8 = FLAT_EDGE1 | FLAT_EDGE2 | FLAT_EDGE3;

/// How [`Mesh3d::with_options`] builds a mesh (Box3D `b3MeshDef`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshOptions {
    /// Split nodes at the median of their triangles' centres instead of by SAH: faster to
    /// build, as good for grid-like meshes (`useMedianSplit`).
    pub median_split: bool,
    /// Flag the edges two triangles share, so that contacts do not catch on internal edges
    /// (`identifyEdges`).
    pub identify_edges: bool,
    /// The indices wind clockwise seen from the front (`clockWiseWinding`).
    pub clockwise: bool,
    /// Optional per-triangle material indices (Box3D `b3MeshDef::materialIndices`).
    pub material_indices: Option<Vec<u8>>,
}

impl Default for MeshOptions {
    /// SAH splits, identified edges, counter-clockwise triangles, no material indices.
    fn default() -> Self {
        Self {
            median_split: false,
            identify_edges: true,
            clockwise: false,
            material_indices: None,
        }
    }
}

/// What a triangle's edge borders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshEdge {
    /// No other triangle, or edges were not identified.
    Open,
    /// A triangle bending up in front of this one, as in a valley.
    Concave,
    /// A triangle bending away behind this one, as on a ridge.
    Convex,
    /// A triangle within 5 degrees of this one's plane.
    Flat,
}

/// A triangle of a mesh (Box3D `b3Triangle`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshTriangle {
    /// The corners, counter-clockwise seen from the front.
    pub vertices: [Vec3; 3],
    /// The corners' vertex indices.
    pub indices: [u32; 3],
    /// The edge flags: bit `e` concave, bit `e + 4` convex, both flat; edge `e` runs from
    /// corner `e` to the next.
    pub flags: u8,
}

impl MeshTriangle {
    /// What edge `edge` (0 to 2, from corner `edge` to the next) borders.
    #[must_use]
    pub fn edge(&self, edge: usize) -> MeshEdge {
        let concave = self.flags & (CONCAVE_EDGE1 << edge) != 0;
        let convex = self.flags & (INVERSE_CONCAVE_EDGE1 << edge) != 0;
        match (concave, convex) {
            (true, true) => MeshEdge::Flat,
            (true, false) => MeshEdge::Concave,
            (false, true) => MeshEdge::Convex,
            (false, false) => MeshEdge::Open,
        }
    }
}

/// Where a ray enters a mesh's front (Box3D `b3CastOutput`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshRayHit {
    /// The fraction of the translation at the hit.
    pub fraction: f32,
    /// The hit point.
    pub point: Vec3,
    /// The triangle's normal.
    pub normal: Vec3,
    /// The triangle, an index for [`Mesh3d::triangle`].
    pub triangle: usize,
}

/// A BVH node (Box3D `b3MeshNode`): an inner node's right child is `value` nodes after it, its
/// left child right after it; a leaf holds `value` triangles from `triangle_offset`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct MeshNode {
    lower: Vec3,
    upper: Vec3,
    kind: u8,
    value: u32,
    triangle_offset: u32,
}

impl MeshNode {
    fn is_leaf(&self) -> bool {
        self.kind == LEAF
    }
}

/// An axis-aligned box (Box3D `b3AABB`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Bounds {
    lower: Vec3,
    upper: Vec3,
}

impl Bounds {
    /// Box3D `B3_BOUNDS3_EMPTY`.
    const EMPTY: Self = Self {
        lower: Vec3::splat(f32::MAX),
        upper: Vec3::splat(-f32::MAX),
    };

    fn union(self, other: Self) -> Self {
        Self {
            lower: self.lower.min(other.lower),
            upper: self.upper.max(other.upper),
        }
    }

    fn add_point(self, point: Vec3) -> Self {
        Self {
            lower: self.lower.min(point),
            upper: self.upper.max(point),
        }
    }

    /// Box3D `b3AABB_Area`: the surface area.
    fn area(self) -> f32 {
        let d = self.upper - self.lower;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Box3D `b3IsSaneAABB`.
    fn is_sane(self) -> bool {
        self.lower.is_finite()
            && self.upper.is_finite()
            && self.upper.cmpge(self.lower).all()
            && self.lower.cmpge(Vec3::splat(-HUGE)).all()
            && self.upper.cmple(Vec3::splat(HUGE)).all()
    }
}

/// A triangle while the tree is built (Box3D `b3Primitive`).
#[derive(Debug, Clone, Copy)]
struct Primitive {
    bounds: Bounds,
    center: Vec3,
    triangle: usize,
}

/// Box3D `b3Split`.
struct Split {
    left: Bounds,
    right: Bounds,
    axis: Option<u8>,
    index: usize,
}

/// Box3D `b3MajorAxis`.
fn major_axis(v: Vec3) -> usize {
    if v.x < v.y {
        if v.y < v.z {
            2
        } else {
            1
        }
    } else if v.x < v.z {
        2
    } else {
        0
    }
}

fn bounds_of(primitives: &[Primitive]) -> Bounds {
    primitives
        .iter()
        .fold(Bounds::EMPTY, |b, p| b.union(p.bounds))
}

/// Box3D `b3SplitBinnedSah`: the cheapest split between eight buckets of centres on any axis
/// the centres span, partitioning `primitives` in place; no axis if no split has both sides.
fn split_binned_sah(primitives: &mut [Primitive]) -> Split {
    let count = primitives.len();
    let centers = primitives[1..].iter().fold(
        Bounds {
            lower: primitives[0].center,
            upper: primitives[0].center,
        },
        |b, p| b.add_point(p.center),
    );
    let mut split = Split {
        left: Bounds::EMPTY,
        right: Bounds::EMPTY,
        axis: None,
        index: 0,
    };
    let mut best_bucket = None;
    let mut best_cost = f32::MAX;
    let bucket_of = |axis: usize, center: Vec3| {
        let factor =
            BIN_COUNT as f32 * (1.0 - f32::EPSILON) / (centers.upper[axis] - centers.lower[axis]);
        (factor * (center[axis] - centers.lower[axis])) as usize
    };
    for axis in 0..3 {
        if 0.5 * (centers.upper[axis] - centers.lower[axis]) < LINEAR_SLOP {
            continue;
        }
        let mut buckets = [(0usize, Bounds::EMPTY); BIN_COUNT];
        for p in primitives.iter() {
            let bucket = &mut buckets[bucket_of(axis, p.center)];
            bucket.0 += 1;
            bucket.1 = bucket.1.union(p.bounds);
        }
        for i in 0..BIN_COUNT - 1 {
            let (left_count, left) = buckets[..=i]
                .iter()
                .fold((0, Bounds::EMPTY), |(n, b), &(c, bb)| (n + c, b.union(bb)));
            let (right_count, right) = buckets[i + 1..]
                .iter()
                .fold((0, Bounds::EMPTY), |(n, b), &(c, bb)| (n + c, b.union(bb)));
            debug_assert_eq!(left_count + right_count, count);
            if left_count > 0 && right_count > 0 {
                let cost = left_count as f32 * left.area() + right_count as f32 * right.area();
                if cost < best_cost {
                    best_bucket = Some(i);
                    best_cost = cost;
                    split = Split {
                        left,
                        right,
                        axis: Some(axis as u8),
                        index: left_count,
                    };
                }
            }
        }
    }
    if let (Some(bucket), Some(axis)) = (best_bucket, split.axis) {
        let mut split_index = 0;
        for i in 0..count {
            if bucket_of(axis as usize, primitives[i].center) <= bucket {
                primitives.swap(i, split_index);
                split_index += 1;
            }
        }
        debug_assert_eq!(split_index, split.index);
    }
    split
}

/// Box3D `b3SplitHalf`: halves `primitives` as they lie, on the major axis of their bounds.
fn split_half(primitives: &[Primitive]) -> Split {
    let index = primitives.len() / 2;
    let left = bounds_of(&primitives[..index]);
    let right = bounds_of(&primitives[index..]);
    let all = left.union(right);
    Split {
        left,
        right,
        axis: Some(major_axis((all.upper - all.lower) * 0.5) as u8),
        index,
    }
}

/// Box3D `b3SplitMedian`: a Hoare partition at the middle of the centres' longest axis, or the
/// halves when one side would be nearly empty.
fn split_median(primitives: &mut [Primitive]) -> Split {
    let count = primitives.len();
    debug_assert!(count > 2);
    let (lower, upper) = primitives[1..].iter().fold(
        (primitives[0].center, primitives[0].center),
        |(lo, hi), p| (lo.min(p.center), hi.max(p.center)),
    );
    let d = upper - lower;
    let c = (lower + upper) * 0.5;
    let axis = if d.x >= d.y && d.x >= d.z {
        0
    } else if d.y >= d.z {
        1
    } else {
        2
    };
    let pivot = c[axis];
    let (mut i1, mut i2) = (0, count);
    while i1 < i2 {
        while i1 < i2 && primitives[i1].center[axis] < pivot {
            i1 += 1;
        }
        while i1 < i2 && primitives[i2 - 1].center[axis] >= pivot {
            i2 -= 1;
        }
        if i1 < i2 {
            primitives.swap(i1, i2 - 1);
            i1 += 1;
            i2 -= 1;
        }
    }
    if i1 == 0 || i1 == count - 1 {
        // Failed to split.
        i1 = count / 2;
    }
    Split {
        left: bounds_of(&primitives[..i1]),
        right: bounds_of(&primitives[i1..]),
        axis: Some(axis as u8),
        index: i1,
    }
}

fn leaf(bounds: Bounds, count: usize, offset: usize) -> MeshNode {
    MeshNode {
        lower: bounds.lower,
        upper: bounds.upper,
        kind: LEAF,
        value: count as u32,
        triangle_offset: offset as u32,
    }
}

/// Box3D `b3BuildRecursive`: the subtree of `primitives`, which start at `offset` of all the
/// mesh's, in pre-order; returns its root and height.
fn build_recursive(
    nodes: &mut Vec<MeshNode>,
    primitives: &mut [Primitive],
    offset: usize,
    median_split: bool,
) -> (usize, usize) {
    let count = primitives.len();
    if count <= DESIRED_TRIANGLES_PER_LEAF {
        nodes.push(leaf(bounds_of(primitives), count, offset));
        return (nodes.len() - 1, 1);
    }
    let mut split = if median_split {
        split_median(primitives)
    } else {
        split_binned_sah(primitives)
    };
    if split.axis.is_none() {
        if count > MAXIMUM_TRIANGLES_PER_LEAF {
            // Re-split, less well.
            split = split_half(primitives);
        } else {
            // Only a few triangles left: a leaf. Box3D counts no height here.
            nodes.push(leaf(bounds_of(primitives), count, offset));
            return (nodes.len() - 1, 0);
        }
    }
    let index = nodes.len();
    nodes.push(leaf(Bounds::EMPTY, 0, 0));
    let (left, right) = primitives.split_at_mut(split.index);
    let (_, left_height) = build_recursive(nodes, left, offset, median_split);
    let (right_index, right_height) =
        build_recursive(nodes, right, offset + split.index, median_split);
    let bounds = split.left.union(split.right);
    nodes[index] = MeshNode {
        lower: bounds.lower,
        upper: bounds.upper,
        kind: split.axis.unwrap_or(0),
        value: (right_index - index) as u32,
        triangle_offset: 0,
    };
    (index, left_height.max(right_height) + 1)
}

/// Box3D `b3TestBoundsOverlap`.
fn bounds_overlap(lower1: Vec3, upper1: Vec3, lower2: Vec3, upper2: Vec3) -> bool {
    (lower2 - upper1)
        .max(lower1 - upper2)
        .cmple(Vec3::ZERO)
        .all()
}

/// Box3D `b3TestBoundsRayOverlap`: no edge axis separates the box and the ray's line.
fn bounds_ray_overlap(lower: Vec3, upper: Vec3, start: Vec3, delta: Vec3) -> bool {
    let center = (lower + upper) * 0.5;
    let extent = upper - center;
    let start = start - center;
    let separation = delta.cross(start).abs() - modified_cross(delta.abs(), extent);
    separation.cmple(Vec3::ZERO).all()
}

/// Box3D `b3TestBoundsTriangleOverlap`: the separating axis test of a box (centre, half
/// extents) and a triangle.
pub(crate) fn bounds_triangle_overlap(center: Vec3, extent: Vec3, v: [Vec3; 3]) -> bool {
    let [v1, v2, v3] = v.map(|p| p - center);
    let min = v1.min(v2.min(v3));
    let max = v1.max(v2.max(v3));
    if (min - extent).max(-(max + extent)).cmpgt(Vec3::ZERO).any() {
        return false;
    }
    let (e1, e2, e3) = (v2 - v1, v3 - v2, v1 - v3);
    let normal = e1.cross(e2);
    if normal.dot(v1).abs() - normal.abs().dot(extent) > 0.0 {
        return false;
    }
    for (edge, sum, other) in [(e1, v1 + v3, e3), (e2, v1 + v2, e1), (e3, v2 + v3, e2)] {
        let separation = edge.cross(sum).abs()
            - edge.cross(other).abs()
            - modified_cross(edge.abs(), extent) * 2.0;
        if separation.cmpgt(Vec3::ZERO).any() {
            return false;
        }
    }
    true
}

/// Box3D `b3IntersectRayTriangle`: the fraction of `delta` at which the ray crosses the front
/// of the triangle, or 1 when it misses or meets the back.
pub(crate) fn intersect_ray_triangle(
    start: Vec3,
    delta: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
) -> f32 {
    let (e1, e2, e3) = (v3 - v2, v1 - v3, v2 - v1);
    let (m1, m2, m3) = ((v2 + v3) * 0.5, (v3 + v1) * 0.5, (v1 + v2) * 0.5);
    let volumes = [
        e1.cross(m1 - start),
        e2.cross(m2 - start),
        e3.cross(m3 - start),
    ]
    .map(|n| n.x * delta.x + n.y * delta.y + n.z * delta.z);
    if volumes.iter().any(|&v| v < 0.0) {
        return 1.0;
    }
    let normal = (v2 - v1).cross(v3 - v1);
    let denominator = normal.dot(delta);
    if denominator >= 0.0 {
        return 1.0;
    }
    let lambda = normal.dot(v1 - start) / denominator;
    if lambda <= 0.0 {
        return 1.0;
    }
    lambda.min(1.0)
}

/// A static triangle mesh (Box3D `b3MeshData`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mesh3d {
    vertices: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    flags: Vec<u8>,
    nodes: Vec<MeshNode>,
    lower: Vec3,
    upper: Vec3,
    surface_area: f32,
    tree_height: usize,
    degenerate_count: usize,
    material_indices: Option<Vec<u8>>,
}

impl Mesh3d {
    /// The mesh of `vertices` and the counter-clockwise triangles `indices`, with SAH splits and
    /// identified edges; `None` when no triangle has area, an index is out of range or the
    /// tree is too deep (Box3D `b3CreateMesh`).
    #[must_use]
    pub fn new(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<Self> {
        Self::with_options(vertices, indices, MeshOptions::default())
    }

    /// The mesh of `vertices` and `indices` built as `options` say.
    #[must_use]
    pub fn with_options(
        vertices: &[Vec3],
        indices: &[[u32; 3]],
        options: MeshOptions,
    ) -> Option<Self> {
        if vertices.len() < 3
            || indices.is_empty()
            || indices
                .iter()
                .flatten()
                .any(|&i| i as usize >= vertices.len())
        {
            return None;
        }
        let min_area = 0.01 * LINEAR_SLOP * LINEAR_SLOP;
        let mut primitives = Vec::with_capacity(indices.len());
        let (mut degenerate_count, mut surface_area) = (0, 0.0);
        let mut mesh_bounds = Bounds::EMPTY;
        for (triangle, &[i1, i2, i3]) in indices.iter().enumerate() {
            let [v1, v2, v3] = [i1, i2, i3].map(|i| vertices[i as usize]);
            let area = 0.5 * (v2 - v1).cross(v3 - v1).length();
            if area < min_area {
                // A triangle that repeats a vertex is dropped without counting.
                if i1 != i2 && i1 != i3 && i2 != i3 {
                    degenerate_count += 1;
                }
                continue;
            }
            surface_area += area;
            let bounds = Bounds {
                lower: v1.min(v2.min(v3)),
                upper: v1.max(v2.max(v3)),
            };
            primitives.push(Primitive {
                bounds,
                center: (bounds.lower + bounds.upper) * 0.5,
                triangle,
            });
            mesh_bounds = mesh_bounds.union(bounds);
        }
        if !mesh_bounds.is_sane() {
            return None;
        }

        let mut nodes = Vec::with_capacity(2 * primitives.len());
        let (_, tree_height) =
            build_recursive(&mut nodes, &mut primitives, 0, options.median_split);
        let material_indices = options.material_indices.as_ref().map(|opt_mats| {
            primitives
                .iter()
                .map(|p| opt_mats.get(p.triangle).copied().unwrap_or(0))
                .collect()
        });
        let mut mesh = Self {
            vertices: vertices.to_vec(),
            triangles: primitives
                .iter()
                .map(|p| {
                    let [i1, i2, i3] = indices[p.triangle];
                    if options.clockwise {
                        [i1, i3, i2]
                    } else {
                        [i1, i2, i3]
                    }
                })
                .collect(),
            flags: vec![0; primitives.len()],
            nodes,
            lower: mesh_bounds.lower,
            upper: mesh_bounds.upper,
            surface_area,
            tree_height,
            degenerate_count,
            material_indices,
        };
        if !mesh.sort_triangles() {
            return None;
        }
        if options.identify_edges {
            mesh.identify_edges();
        }
        Some(mesh)
    }

    /// Box3D `b3SortMeshTriangles`: the triangles in the tree's depth-first order, each leaf
    /// pointing at its run. False when the tree is deeper than a query can walk.
    fn sort_triangles(&mut self) -> bool {
        let mut sorted = Vec::with_capacity(self.triangles.len());
        let mut sorted_mats = self
            .material_indices
            .as_ref()
            .map(|_| Vec::with_capacity(self.triangles.len()));
        let mut stack = Vec::with_capacity(MESH_STACK_SIZE);
        stack.push(0usize);
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if node.is_leaf() {
                let start = node.triangle_offset as usize;
                let count = node.value as usize;
                sorted.extend_from_slice(&self.triangles[start..start + count]);
                if let (Some(ref mut sm), Some(ref mats)) =
                    (&mut sorted_mats, &self.material_indices)
                {
                    sm.extend_from_slice(&mats[start..start + count]);
                }
                self.nodes[index].triangle_offset = (sorted.len() - count) as u32;
            } else {
                if stack.len() >= MESH_STACK_SIZE - 2 {
                    return false;
                }
                stack.push(index + node.value as usize);
                stack.push(index + 1);
            }
        }
        self.triangles = sorted;
        if let Some(sm) = sorted_mats {
            self.material_indices = Some(sm);
        }
        true
    }

    /// Returns the optional per-triangle material indices (Box3D `b3GetMeshMaterialIndices`).
    #[must_use]
    pub fn material_indices(&self) -> Option<&[u8]> {
        self.material_indices.as_deref()
    }

    /// Returns the material index for the given triangle index (defaults to 0).
    #[must_use]
    pub fn material_index(&self, triangle_idx: usize) -> u8 {
        self.material_indices
            .as_ref()
            .and_then(|m| m.get(triangle_idx).copied())
            .unwrap_or(0)
    }

    /// Box3D `b3IdentifyEdges`: each edge two triangles share is concave where the second
    /// triangle's far corner lies in front of the first, convex where it lies behind, and
    /// flat (both) where their normals are within 5 degrees.
    fn identify_edges(&mut self) {
        struct Edge {
            triangle1: usize,
            triangle2: usize,
            edge1: usize,
            edge2: usize,
            count: u32,
        }
        let vertices = &self.vertices;
        let mut edges = Vec::with_capacity(3 * self.triangles.len());
        let mut normals = Vec::with_capacity(self.triangles.len());
        for (t, &[i1, i2, i3]) in self.triangles.iter().enumerate() {
            for e in 0..3 {
                edges.push(Edge {
                    triangle1: t,
                    triangle2: usize::MAX,
                    edge1: e,
                    edge2: usize::MAX,
                    count: 1,
                });
            }
            let [v1, v2, v3] = [i1, i2, i3].map(|i| vertices[i as usize]);
            normals.push(crate::contact_solver::normalize((v2 - v1).cross(v3 - v1)));
        }
        let key = |triangle: [u32; 3], e: usize| {
            let (a, b) = (triangle[e], triangle[(e + 1) % 3]);
            (u64::from(a.min(b)) << 32) | u64::from(a.max(b))
        };
        let mut map = std::collections::HashMap::with_capacity(edges.len());
        for i in 0..edges.len() {
            let k = key(self.triangles[edges[i].triangle1], edges[i].edge1);
            match map.get(&k) {
                None => {
                    map.insert(k, i);
                }
                Some(&base) => {
                    let (triangle, edge) = (edges[i].triangle1, edges[i].edge1);
                    let first = &mut edges[base];
                    if first.count == 1 {
                        first.triangle2 = triangle;
                        first.edge2 = edge;
                    }
                    first.count += 1;
                }
            }
        }
        let cos_5_degrees = 0.9962;
        for edge in edges.iter().filter(|e| e.count == 2) {
            let [j1, j2, j3] = self.triangles[edge.triangle2];
            let opposite = [j3, j1, j2][edge.edge2];
            let [v1, v2, v3] = self.triangles[edge.triangle1].map(|i| vertices[i as usize]);
            // Box3D `b3SignedVolume`.
            let signed_volume = (v2 - v1)
                .cross(v3 - v1)
                .dot(vertices[opposite as usize] - v1);
            let cos_angle = normals[edge.triangle1].dot(normals[edge.triangle2]);
            let flat = cos_angle > cos_5_degrees;
            if signed_volume > 0.0 || flat {
                self.flags[edge.triangle1] |= CONCAVE_EDGE1 << edge.edge1;
                self.flags[edge.triangle2] |= CONCAVE_EDGE1 << edge.edge2;
            }
            if signed_volume < 0.0 || flat {
                self.flags[edge.triangle1] |= INVERSE_CONCAVE_EDGE1 << edge.edge1;
                self.flags[edge.triangle2] |= INVERSE_CONCAVE_EDGE1 << edge.edge2;
            }
        }
    }

    /// The number of triangles kept.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// The number of BVH nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The BVH's height, as Box3D counts it.
    #[must_use]
    pub fn tree_height(&self) -> usize {
        self.tree_height
    }

    /// The triangles dropped for having no area, other than those repeating a vertex.
    #[must_use]
    pub fn degenerate_count(&self) -> usize {
        self.degenerate_count
    }

    /// The area of the triangles, one-sided.
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        self.surface_area
    }

    /// The local bounds.
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.lower, self.upper)
    }

    /// The vertices.
    #[must_use]
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Triangle `index`, in the tree's order (Box3D `b3GetMeshTriangle`).
    #[must_use]
    pub fn triangle(&self, index: usize) -> MeshTriangle {
        let indices = self.triangles[index];
        MeshTriangle {
            vertices: indices.map(|i| self.vertices[i as usize]),
            indices,
            flags: self.flags[index],
        }
    }

    /// The triangles, each a [`MeshTriangle`], in the tree's order.
    pub fn triangles(&self) -> impl Iterator<Item = MeshTriangle> + '_ {
        (0..self.triangles.len()).map(|i| self.triangle(i))
    }

    /// Calls `visit` with each triangle whose box and triangle overlap the box `lower` to
    /// `upper`, in the tree's order, until it returns false (Box3D `b3QueryMesh`).
    pub fn query(&self, lower: Vec3, upper: Vec3, mut visit: impl FnMut(usize) -> bool) {
        let center = (lower + upper) * 0.5;
        let extent = upper - center;
        let mut stack = [0u32; MESH_STACK_SIZE];
        let mut count = 0;
        let mut index = 0usize;
        loop {
            let node = &self.nodes[index];
            if bounds_overlap(node.lower, node.upper, lower, upper) {
                if !node.is_leaf() {
                    stack[count] = index as u32 + node.value;
                    count += 1;
                    index += 1;
                    continue;
                }
                let start = node.triangle_offset as usize;
                for t in start..start + node.value as usize {
                    let v = self.triangles[t].map(|i| self.vertices[i as usize]);
                    if bounds_triangle_overlap(center, extent, v) && !visit(t) {
                        return;
                    }
                }
            }
            if count == 0 {
                return;
            }
            count -= 1;
            index = stack[count] as usize;
        }
    }

    /// Where the ray from `origin` along `translation` first enters a triangle's front, within
    /// `max_fraction` of the translation (Box3D `b3RayCastMesh`). The tree is walked front to
    /// back and the ray shortened at each hit.
    #[must_use]
    pub fn ray_cast(
        &self,
        origin: Vec3,
        translation: Vec3,
        max_fraction: f32,
    ) -> Option<MeshRayHit> {
        let mut best: Option<MeshRayHit> = None;
        let mut fraction = max_fraction;
        let bounds = |fraction: f32| {
            let end = origin + translation * fraction;
            (origin.min(end), origin.max(end))
        };
        let (mut ray_lower, mut ray_upper) = bounds(fraction);
        let mut stack = [0u32; MESH_STACK_SIZE];
        let mut count = 0;
        let mut index = 0usize;
        loop {
            let node = &self.nodes[index];
            if bounds_overlap(node.lower, node.upper, ray_lower, ray_upper)
                && bounds_ray_overlap(node.lower, node.upper, origin, translation)
            {
                if !node.is_leaf() {
                    // Front child first.
                    let (near, far) = if translation[node.kind as usize] > 0.0 {
                        (index + 1, index + node.value as usize)
                    } else {
                        (index + node.value as usize, index + 1)
                    };
                    stack[count] = far as u32;
                    count += 1;
                    index = near;
                    continue;
                }
                let start = node.triangle_offset as usize;
                for t in start..start + node.value as usize {
                    let [v1, v2, v3] = self.triangles[t].map(|i| self.vertices[i as usize]);
                    let alpha = intersect_ray_triangle(origin, translation, v1, v2, v3);
                    if alpha < fraction {
                        fraction = alpha;
                        best = Some(MeshRayHit {
                            fraction: alpha,
                            point: origin + translation * alpha,
                            normal: crate::contact_solver::normalize((v2 - v1).cross(v3 - v1)),
                            triangle: t,
                        });
                        (ray_lower, ray_upper) = bounds(fraction);
                    }
                }
            }
            if count == 0 {
                return best;
            }
            count -= 1;
            index = stack[count] as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_strip(n: u32) -> Mesh3d {
        let mut vertices = Vec::new();
        for i in 0..=n {
            vertices.push(Vec3::new(i as f32, 0.0, 0.0));
            vertices.push(Vec3::new(i as f32, 0.0, 1.0));
        }
        let mut indices = Vec::new();
        for i in 0..n {
            let (a, b, c, d) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
            indices.push([a, b, c]);
            indices.push([c, b, d]);
        }
        Mesh3d::new(&vertices, &indices).unwrap()
    }

    #[test]
    fn leaves_cover_their_triangles() {
        let mesh = quad_strip(50);
        let mut seen = vec![false; mesh.triangle_count()];
        for node in mesh.nodes.iter().filter(|n| n.is_leaf()) {
            for t in node.triangle_offset..node.triangle_offset + node.value {
                assert!(!seen[t as usize], "each triangle in one leaf");
                seen[t as usize] = true;
                for v in mesh.triangle(t as usize).vertices {
                    assert!(v.cmpge(node.lower).all() && v.cmple(node.upper).all());
                }
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn queries_find_what_brute_force_finds() {
        let mesh = quad_strip(50);
        for (lower, upper) in [
            (Vec3::new(3.2, -0.1, 0.2), Vec3::new(7.9, 0.1, 0.4)),
            (Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.5, 1.0, 0.1)),
            (Vec3::new(10.0, 0.5, 0.0), Vec3::new(20.0, 1.0, 1.0)),
        ] {
            let mut found = Vec::new();
            mesh.query(lower, upper, |t| {
                found.push(t);
                true
            });
            let center = (lower + upper) * 0.5;
            let expected: Vec<usize> = (0..mesh.triangle_count())
                .filter(|&t| {
                    bounds_triangle_overlap(center, upper - center, mesh.triangle(t).vertices)
                })
                .collect();
            assert_eq!(found, expected);
        }
    }
}
