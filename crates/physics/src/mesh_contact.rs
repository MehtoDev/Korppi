//! Box3D's mesh contacts (`mesh_contact.c`, `b3ComputeMeshManifolds`).
//!
//! A mesh contact keeps the triangles near the convex shape, found by a BVH query over the
//! shape's bounds widened by a margin, each with its own narrowphase cache, and requeries only
//! when the shape leaves the widened bounds. Each step the shape collides with each triangle;
//! face contacts are accepted and mark their edges and corners as found. Edge and corner
//! contacts are tentative: for spheres, nearest first, one is accepted only when it adds its
//! feature to the found set, and for capsules and hulls one is dropped when it touches a flat
//! edge a face contact already found, so nothing catches on internal edges. The accepted
//! triangle contacts are clustered by normal (cosine above 0.996) into manifolds of at most four
//! points, separated by a rest offset of one linear slop, and each manifold takes the friction
//! impulses of the old manifold with the closest normal and each point the normal impulse of
//! the old point of the same feature and triangle.

use crate::contact_solver::{perp, LINEAR_SLOP, SPECULATIVE_DISTANCE};
use crate::manifold::{FeaturePair, LocalPoint, SatCache, AXIS_EDGE_PAIR};
use crate::math::Transform;
use crate::mesh::{
    bounds_triangle_overlap, Mesh3d, MeshTriangle, ALL_FLAT_EDGES, FLAT_EDGE1, FLAT_EDGE2,
    FLAT_EDGE3,
};
use crate::narrowphase::{
    ContactCache, ContactManifold, ContactPoint, ContactPoints, Convex, MAX_MANIFOLD_POINTS,
};
use crate::triangle_manifold::{
    collide_triangle_and_capsule, collide_triangle_and_hull, collide_triangle_and_sphere,
    TriangleFeature,
};
use korppi_math::{Aabb, Mat3, Vec3};
use serde::{Deserialize, Serialize};

/// Box3D `B3_MESH_REST_OFFSET`: the gap a mesh contact keeps, which pushes shapes apart after
/// a time of impact.
pub(crate) const MESH_REST_OFFSET: f32 = LINEAR_SLOP;

/// Box3D `B3_MAX_MESH_CONTACT_TRIANGLES`: the most triangles a mesh contact considers.
const MAX_MESH_CONTACT_TRIANGLES: usize = 256;

/// Box3D `B3_MAX_POINTS_PER_TRIANGLE`: the point room per triangle.
const MAX_POINTS_PER_TRIANGLE: usize = 32;

/// Box3D `B3_MAX_AABB_MARGIN`.
const MAX_AABB_MARGIN: f32 = 0.05;

/// Box3D `B3_MAX_EDGE_COUNT` and `B3_MAX_VERTEX_COUNT`: the found features a contact tracks;
/// past them a tentative contact is accepted, which may give a ghost collision.
const MAX_FOUND: usize = 64;

/// Clusters join triangle contacts whose normals are closer than this cosine.
const CLUSTER_THRESHOLD: f32 = 0.996;

/// A new manifold takes the impulses of an old one whose normal is closer than this cosine.
const NORMAL_MATCH_TOLERANCE: f32 = 0.995;

/// A triangle near the shape and the narrowphase cache of its pair (Box3D `b3TriangleCache`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct TriangleCache {
    triangle: u32,
    cache: ContactCache,
}

/// What a mesh contact keeps between steps (Box3D `b3MeshContact`): the widened bounds of the
/// last query, in world space, and the triangles it found, in increasing order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshContactCache {
    query_lower: Vec3,
    query_upper: Vec3,
    triangles: Vec<TriangleCache>,
}

impl MeshContactCache {
    /// Forgets the triangles, keeping the buffer.
    pub(crate) fn clear(&mut self) {
        self.query_lower = Vec3::ZERO;
        self.query_upper = Vec3::ZERO;
        self.triangles.clear();
    }
}

use crate::height_field::HeightField3d;

/// The triangles a mesh contact collides with: a mesh, a height field, or a single triangle.
pub(crate) enum Triangles<'a> {
    Mesh(&'a Mesh3d),
    HeightField(&'a HeightField3d),
    Single([Vec3; 3]),
}

impl Triangles<'_> {
    /// Writes the triangles overlapping the local box into `out`, in increasing order, and
    /// returns how many (Box3D `b3QueryMeshTriangles`).
    fn query(
        &self,
        lower: Vec3,
        upper: Vec3,
        out: &mut [u32; MAX_MESH_CONTACT_TRIANGLES],
    ) -> usize {
        match self {
            Self::Mesh(mesh) => {
                let mut count = 0;
                mesh.query(lower, upper, |triangle| {
                    out[count] = triangle as u32;
                    count += 1;
                    count < out.len()
                });
                count
            }
            Self::HeightField(hf) => {
                let mut count = 0;
                hf.query(lower, upper, |triangle| {
                    out[count] = triangle as u32;
                    count += 1;
                    count < out.len()
                });
                count
            }
            Self::Single(vertices) => {
                let center = (lower + upper) * 0.5;
                usize::from(bounds_triangle_overlap(center, upper - center, *vertices))
            }
        }
    }

    fn get(&self, index: u32) -> MeshTriangle {
        match self {
            Self::Mesh(mesh) => mesh.triangle(index as usize),
            Self::HeightField(hf) => hf.triangle(index as usize),
            Self::Single(vertices) => MeshTriangle {
                vertices: *vertices,
                indices: [0, 1, 2],
                flags: 0,
            },
        }
    }
}

/// A triangle's contact in the shape's frame (Box3D's `b3LocalManifold` of a triangle).
#[derive(Debug, Clone, Copy)]
struct TriangleManifold {
    normal: Vec3,
    start: usize,
    count: usize,
    feature: TriangleFeature,
    squared_distance: f32,
    flags: u8,
    triangle: u32,
    triangle_normal: Vec3,
    indices: [u32; 3],
}

/// Box3D `b3Cluster`.
#[derive(Debug, Clone, Copy)]
struct Cluster {
    manifold_normal: Vec3,
    triangle_normal: Vec3,
    start: usize,
    capacity: usize,
    count: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct ClusterPoint {
    triangle: u32,
    point: Vec3,
    separation: f32,
    pair: FeaturePair,
}

/// Box3D `b3Point2D`: a point projected onto a cluster's plane.
#[derive(Debug, Clone, Copy, Default)]
struct Point2 {
    x: f32,
    y: f32,
    separation: f32,
    original: usize,
}

/// The buffers a world reuses for its mesh contacts, so that they do not allocate once grown.
#[derive(Debug, Default)]
pub(crate) struct MeshScratch {
    caches: Vec<TriangleCache>,
    points: Vec<LocalPoint>,
    manifolds: Vec<TriangleManifold>,
    accepted: Vec<usize>,
    tentative: Vec<usize>,
    tentative_order: Vec<(f32, usize)>,
    clusters: Vec<Cluster>,
    memberships: Vec<usize>,
    cluster_points: Vec<ClusterPoint>,
    cull: Vec<Point2>,
    old: Vec<ContactManifold>,
    consumed: Vec<bool>,
}

/// The feature keys a contact has met (Box3D `b3FoundEdges`, `b3FoundVertices`).
struct Found {
    keys: [u64; MAX_FOUND],
    count: usize,
}

impl Found {
    const fn new() -> Self {
        Self {
            keys: [0; MAX_FOUND],
            count: 0,
        }
    }

    /// Adds `key`; false when it was there. A full set accepts every key.
    fn add(&mut self, key: u64) -> bool {
        if self.find(key) {
            return false;
        }
        if self.count < MAX_FOUND {
            self.keys[self.count] = key;
            self.count += 1;
        }
        true
    }

    fn find(&self, key: u64) -> bool {
        self.keys[..self.count].contains(&key)
    }
}

fn edge_key(vertex1: u32, vertex2: u32) -> u64 {
    (u64::from(vertex1.min(vertex2)) << 32) | u64::from(vertex1.max(vertex2))
}

/// Box3D `b3AABB_Contains`.
fn contains(lower: Vec3, upper: Vec3, bounds: &Aabb) -> bool {
    lower.cmple(bounds.min).all() && bounds.max.cmple(upper).all()
}

/// Box3D `b3RefreshCache`: requeries the triangles near the shape's `bounds` when it left the
/// widened bounds of the last query, keeping the caches of the triangles still found.
fn refresh_cache(
    cache: &mut MeshContactCache,
    triangles: &Triangles<'_>,
    xf_mesh: &Transform,
    bounds: &Aabb,
    scratch: &mut Vec<TriangleCache>,
) {
    if contains(cache.query_lower, cache.query_upper, bounds) {
        return;
    }
    let radius = Vec3::splat(MAX_AABB_MARGIN + SPECULATIVE_DISTANCE);
    cache.query_lower = bounds.min - radius;
    cache.query_upper = bounds.max + radius;
    // The query box in the mesh's frame (Box3D `b3AABB_Transform` by the inverse transform).
    let inverse = xf_mesh.inverse();
    let center = inverse.apply((cache.query_lower + cache.query_upper) * 0.5);
    let r = Mat3::from_quat(inverse.q);
    let half = (cache.query_upper - cache.query_lower) * 0.5;
    let extent = (r.x_axis * half.x).abs() + (r.y_axis * half.y).abs() + (r.z_axis * half.z).abs();
    let mut found = [0u32; MAX_MESH_CONTACT_TRIANGLES];
    let count = triangles.query(center - extent, center + extent, &mut found);

    scratch.clear();
    let mut old = 0;
    for &triangle in &found[..count] {
        while old < cache.triangles.len() && cache.triangles[old].triangle < triangle {
            old += 1;
        }
        let kept = cache
            .triangles
            .get(old)
            .filter(|t| t.triangle == triangle)
            .map_or_else(ContactCache::default, |t| t.cache);
        scratch.push(TriangleCache {
            triangle,
            cache: kept,
        });
    }
    cache.triangles.clear();
    cache.triangles.extend_from_slice(scratch);
}

/// Box3D `b3IsBetterCullCandidate`: a larger score, or a deeper separation within the score
/// tolerance.
fn is_better(score: f32, separation: f32, best_score: f32, best_separation: f32, tol: f32) -> bool {
    if score > best_score + tol {
        return true;
    }
    if score < best_score - tol {
        return false;
    }
    separation < best_separation - LINEAR_SLOP
}

/// Box3D `b3CullPoints`: at most four of the projected points, the two farthest apart, the
/// one making the largest triangle with them and the one adding the most area, ties broken by
/// depth; they are written to the start of `points`, and their count returned.
fn cull_points(points: &mut [Point2]) -> usize {
    let count = points.len();
    if count <= 1 {
        return count;
    }
    let tol = 0.25 * LINEAR_SLOP;
    let tol_sqr = tol * tol;
    let mut finals = [Point2::default(); 4];
    let mut count1 = count;

    // The two points farthest apart.
    let (mut best_score, mut best_separation) = (0.0, f32::MAX);
    let (mut best1, mut best2) = (None, None);
    for i in 0..count1 {
        for j in i + 1..count1 {
            let (dx, dy) = (points[j].x - points[i].x, points[j].y - points[i].y);
            let score = dx * dx + dy * dy;
            let separation = points[i].separation + points[j].separation;
            if is_better(score, separation, best_score, best_separation, tol_sqr) {
                best1 = Some(i);
                best2 = Some(j);
                best_score = score;
                best_separation = separation;
            }
        }
    }
    let (Some(best1), Some(best2)) = (best1, best2) else {
        return deepest_only(points);
    };
    if best_score < tol_sqr {
        return deepest_only(points);
    }
    finals[0] = points[best1];
    finals[1] = points[best2];
    points[best2] = points[count1 - 1];
    points[best1] = points[count1 - 2];
    count1 -= 2;
    if count1 == 0 {
        points[..2].copy_from_slice(&finals[..2]);
        return 2;
    }

    // The point making the largest triangle.
    let a = (finals[0].x, finals[0].y);
    let mut b = (finals[1].x, finals[1].y);
    let mut ba = (b.0 - a.0, b.1 - a.1);
    let cross = |u: (f32, f32), v: (f32, f32)| u.0 * v.1 - u.1 * v.0;
    let (mut best_score, mut best_separation) = (0.0, f32::MAX);
    let mut best = None;
    let mut best_signed_area = 0.0;
    for (i, p) in points[..count1].iter().enumerate() {
        let signed_area = cross(ba, (p.x - a.0, p.y - a.1));
        let score = signed_area.abs();
        if is_better(score, p.separation, best_score, best_separation, tol_sqr) {
            best_signed_area = signed_area;
            best_score = score;
            best_separation = p.separation;
            best = Some(i);
        }
    }
    let Some(best3) = best else {
        // Collinear.
        points[..2].copy_from_slice(&finals[..2]);
        return 2;
    };
    finals[2] = points[best3];
    if count1 == 1 {
        points[..3].copy_from_slice(&finals[..3]);
        return 3;
    }
    points[best3] = points[count1 - 1];
    count1 -= 1;

    // The point adding the most area outside the triangle, counter-clockwise.
    let mut c = (finals[2].x, finals[2].y);
    if best_signed_area < 0.0 {
        std::mem::swap(&mut b, &mut c);
        ba = (b.0 - a.0, b.1 - a.1);
    }
    let cb = (c.0 - b.0, c.1 - b.1);
    let ac = (a.0 - c.0, a.1 - c.1);
    let (mut best_score, mut best_separation) = (0.0, f32::MAX);
    let mut best = None;
    for (i, p) in points[..count1].iter().enumerate() {
        let u1 = cross((p.x - a.0, p.y - a.1), ba);
        let u2 = cross((p.x - b.0, p.y - b.1), cb);
        let u3 = cross((p.x - c.0, p.y - c.1), ac);
        let score = u1.max(u2.max(u3));
        if is_better(score, p.separation, best_score, best_separation, tol_sqr) {
            best_score = score;
            best_separation = p.separation;
            best = Some(i);
        }
    }
    let Some(best4) = best else {
        points[..3].copy_from_slice(&finals[..3]);
        return 3;
    };
    finals[3] = points[best4];
    points[..4].copy_from_slice(&finals);
    4
}

/// The deepest point alone, first.
fn deepest_only(points: &mut [Point2]) -> usize {
    let mut deepest = 0;
    for i in 1..points.len() {
        if points[i].separation < points[deepest].separation {
            deepest = i;
        }
    }
    points[0] = points[deepest];
    1
}

/// Box3D `b3ReduceCluster`: a cluster's points reduced to at most four, projected onto the
/// plane of its triangle normal.
fn reduce_cluster(points: &mut [ClusterPoint], normal: Vec3, cull: &mut Vec<Point2>) -> usize {
    if points.len() <= 1 {
        return points.len();
    }
    let u = perp(normal);
    let v = normal.cross(u);
    let origin = points[0].point;
    cull.clear();
    cull.extend(points.iter().enumerate().map(|(i, p)| {
        let d = p.point - origin;
        Point2 {
            x: d.dot(u),
            y: d.dot(v),
            separation: p.separation,
            original: i,
        }
    }));
    let count = cull_points(cull);
    let mut finals = [ClusterPoint::default(); MAX_MANIFOLD_POINTS];
    for (out, p) in finals.iter_mut().zip(&cull[..count]) {
        *out = points[p.original];
    }
    points[..count].copy_from_slice(&finals[..count]);
    count
}

/// Box3D's `QSORT` (Alexey Tourbin's quicksort after Sedgewick, insertion sort below 16), so
/// that equal keys end in Box3D's order.
fn box3d_qsort<T: Copy>(a: &mut [T], less: impl Fn(&T, &T) -> bool) {
    const THRESH: usize = 16;
    let n = a.len();
    if n <= 1 {
        return;
    }
    let lt = |a: &[T], i: usize, j: usize| less(&a[i], &a[j]);
    let mut stack = [(0usize, 0usize); 48];
    let mut sp = 0;
    let (mut l, mut r) = (0usize, n - 1);
    loop {
        if r - l + 1 >= THRESH {
            // Q_PARTITION.
            let m = l + ((r - l) >> 1);
            let (a1, a2, a3) = (l + 1, m, r);
            if lt(a, a2, a1) {
                if lt(a, a3, a2) {
                    a.swap(a1, a3);
                } else {
                    a.swap(a1, a2);
                    if lt(a, a3, a2) {
                        a.swap(a2, a3);
                    }
                }
            } else if lt(a, a3, a2) {
                a.swap(a2, a3);
                if lt(a, a2, a1) {
                    a.swap(a1, a2);
                }
            }
            a.swap(l, m);
            let (mut i, mut j) = (l + 1, r);
            loop {
                i += 1;
                while lt(a, i, l) {
                    i += 1;
                }
                j -= 1;
                while lt(a, l, j) {
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                a.swap(i, j);
            }
            let i = j + 1;
            a.swap(l, j);
            let j = j - 1;
            // Q_SUBFILES, the larger first.
            let (l1, r1, l2, r2) = if j - l >= r - i {
                (l, j, i, r)
            } else {
                (i, r, l, j)
            };
            if l2 == r2 {
                (l, r) = (l1, r1);
            } else {
                stack[sp] = (l1, r1);
                sp += 1;
                (l, r) = (l2, r2);
            }
        } else {
            for i in l + 1..=r {
                let mut j = i;
                while j > l && lt(a, j, j - 1) {
                    a.swap(j, j - 1);
                    j -= 1;
                }
            }
            if sp == 0 {
                return;
            }
            sp -= 1;
            (l, r) = stack[sp];
        }
    }
}

/// A mesh contact's inputs: the triangles and the mesh's pose, the convex shape, its pose and
/// world bounds, and whether the convex body is fast.
pub(crate) struct MeshPair<'a> {
    pub triangles: Triangles<'a>,
    pub xf_mesh: Transform,
    pub convex: &'a Convex<'a>,
    pub xf_convex: Transform,
    pub convex_bounds: Aabb,
    pub is_fast: bool,
}

/// Box3D `b3ComputeMeshManifolds`: replaces `manifolds` with the contact's new manifolds
/// between bodies `(mesh_body, convex_body)`, normals from the mesh to the shape; when `flip`,
/// the manifolds are stored from the shape to the mesh instead. Returns whether any touches.
/// The manifolds' materials are left for the caller.
pub(crate) fn update_mesh_manifolds(
    manifolds: &mut Vec<ContactManifold>,
    cache: &mut MeshContactCache,
    pair: &MeshPair<'_>,
    (mesh_body, convex_body): (u32, u32),
    flip: bool,
    scratch: &mut MeshScratch,
) -> bool {
    refresh_cache(
        cache,
        &pair.triangles,
        &pair.xf_mesh,
        &pair.convex_bounds,
        &mut scratch.caches,
    );
    let triangle_count = cache.triangles.len();

    // The mesh in the shape's frame.
    let a_to_b = pair.xf_convex.inv_mul(&pair.xf_mesh);
    let relative = Mat3::from_quat(a_to_b.q);
    let capacity = MAX_POINTS_PER_TRIANGLE * triangle_count;
    scratch.points.clear();
    scratch.points.resize(capacity, LocalPoint::default());
    scratch.manifolds.clear();
    scratch.accepted.clear();
    scratch.tentative.clear();
    scratch.tentative_order.clear();
    let mut found_edges = Found::new();
    let mut found_vertices = Found::new();
    let mut total = 0;
    let found_triangle = |edges: &mut Found, vertices: &mut Found, [i1, i2, i3]: [u32; 3]| {
        edges.add(edge_key(i1, i2));
        edges.add(edge_key(i2, i3));
        edges.add(edge_key(i3, i1));
        vertices.add(u64::from(i1));
        vertices.add(u64::from(i2));
        vertices.add(u64::from(i3));
    };

    for index in 0..triangle_count {
        if total + 3 >= capacity {
            break;
        }
        let entry = &mut cache.triangles[index];
        let triangle = pair.triangles.get(entry.triangle);
        let v = triangle.vertices.map(|p| relative * p + a_to_b.p);
        let points = &mut scratch.points[total..capacity];
        let contact = match pair.convex {
            Convex::Sphere(sphere) => collide_triangle_and_sphere(points, &v, sphere),
            Convex::Capsule(capsule) => {
                collide_triangle_and_capsule(points, &v, capsule, &mut entry.cache.simplex)
            }
            hull => {
                // A cached edge axis is dangerous at speed: the hull can turn about the edge
                // and tunnel.
                if pair.is_fast && entry.cache.sat.kind == AXIS_EDGE_PAIR {
                    entry.cache.sat = SatCache::default();
                }
                collide_triangle_and_hull(points, &v, &hull.view(), &mut entry.cache.sat)
            }
        };
        if contact.count == 0 {
            continue;
        }
        debug_assert_ne!(contact.feature, TriangleFeature::None);
        let manifold = TriangleManifold {
            normal: contact.normal,
            start: total,
            count: contact.count,
            feature: contact.feature,
            squared_distance: contact.squared_distance,
            flags: triangle.flags,
            triangle: entry.triangle,
            triangle_normal: crate::contact_solver::normalize((v[1] - v[0]).cross(v[2] - v[0])),
            indices: triangle.indices,
        };
        total += contact.count;
        let m = scratch.manifolds.len();
        scratch.manifolds.push(manifold);
        let accept = match contact.feature {
            TriangleFeature::TriangleFace => true,
            TriangleFeature::HullFace => {
                manifold.triangle_normal.dot(manifold.normal) > 0.5 || {
                    // Deep overlap.
                    let points = &scratch.points[manifold.start..manifold.start + manifold.count];
                    points.iter().map(|p| p.separation).fold(f32::MAX, f32::min)
                        < -2.0 * LINEAR_SLOP
                }
            }
            _ => false,
        };
        if accept {
            found_triangle(&mut found_edges, &mut found_vertices, manifold.indices);
            scratch.accepted.push(m);
        } else {
            scratch
                .tentative_order
                .push((manifold.squared_distance, scratch.tentative.len()));
            scratch.tentative.push(m);
        }
    }

    if matches!(pair.convex, Convex::Sphere(_)) {
        // The nearest triangles first; a tentative contact counts only if it adds its feature.
        box3d_qsort(&mut scratch.tentative_order, |a, b| a.0 < b.0);
        for &(_, t) in &scratch.tentative_order {
            let m = scratch.manifolds[scratch.tentative[t]];
            let [i1, i2, i3] = m.indices;
            let edge1 = found_edges.add(edge_key(i1, i2));
            let edge2 = found_edges.add(edge_key(i2, i3));
            let edge3 = found_edges.add(edge_key(i3, i1));
            let vertex1 = found_vertices.add(u64::from(i1));
            let vertex2 = found_vertices.add(u64::from(i2));
            let vertex3 = found_vertices.add(u64::from(i3));
            let collide = match m.feature {
                TriangleFeature::Edge1 => edge1,
                TriangleFeature::Edge2 => edge2,
                TriangleFeature::Edge3 => edge3,
                TriangleFeature::Vertex1 => vertex1,
                TriangleFeature::Vertex2 => vertex2,
                TriangleFeature::Vertex3 => vertex3,
                _ => false,
            };
            if collide {
                scratch.accepted.push(scratch.tentative[t]);
            }
        }
    } else {
        // Only flat edges are ignored: a hull could tunnel at a concave edge's time of impact.
        for &t in &scratch.tentative {
            let m = &scratch.manifolds[t];
            let [i1, i2, i3] = m.indices;
            let flags = m.flags;
            if flags & ALL_FLAT_EDGES == ALL_FLAT_EDGES
                || (flags & FLAT_EDGE1 == FLAT_EDGE1 && found_edges.find(edge_key(i1, i2)))
                || (flags & FLAT_EDGE2 == FLAT_EDGE2 && found_edges.find(edge_key(i2, i3)))
                || (flags & FLAT_EDGE3 == FLAT_EDGE3 && found_edges.find(edge_key(i3, i1)))
            {
                continue;
            }
            scratch.accepted.push(t);
        }
    }

    if scratch.accepted.is_empty() {
        manifolds.clear();
        return false;
    }

    // Clusters by manifold and triangle normal; the first close cluster is taken.
    scratch.clusters.clear();
    scratch.memberships.clear();
    let mut cluster_point_count = 0;
    for &a in &scratch.accepted {
        let m = &scratch.manifolds[a];
        cluster_point_count += m.count;
        let cluster = scratch.clusters.iter().position(|c| {
            c.manifold_normal.dot(m.normal) > CLUSTER_THRESHOLD
                && c.triangle_normal.dot(m.triangle_normal) > CLUSTER_THRESHOLD
        });
        match cluster {
            Some(c) => {
                scratch.memberships.push(c);
                scratch.clusters[c].capacity += m.count;
            }
            None => {
                scratch.memberships.push(scratch.clusters.len());
                scratch.clusters.push(Cluster {
                    manifold_normal: m.normal,
                    triangle_normal: m.triangle_normal,
                    start: 0,
                    capacity: m.count,
                    count: 0,
                });
            }
        }
    }
    if cluster_point_count == 0 {
        return false;
    }
    let mut offset = 0;
    for c in &mut scratch.clusters {
        c.start = offset;
        offset += c.capacity;
    }
    scratch.cluster_points.clear();
    scratch
        .cluster_points
        .resize(cluster_point_count, ClusterPoint::default());
    for (&a, &c) in scratch.accepted.iter().zip(&scratch.memberships) {
        let m = scratch.manifolds[a];
        let cluster = &mut scratch.clusters[c];
        for p in &scratch.points[m.start..m.start + m.count] {
            scratch.cluster_points[cluster.start + cluster.count] = ClusterPoint {
                triangle: m.triangle,
                point: p.point,
                separation: p.separation,
                pair: p.pair,
            };
            cluster.count += 1;
        }
    }
    for c in &mut scratch.clusters {
        let points = &mut scratch.cluster_points[c.start..c.start + c.count];
        c.count = reduce_cluster(points, c.triangle_normal, &mut scratch.cull);
    }

    // The old manifolds, normals from the mesh to the shape.
    scratch.old.clear();
    scratch.old.extend_from_slice(manifolds);
    if flip {
        for m in &mut scratch.old {
            m.flip();
        }
    }
    scratch.consumed.clear();
    scratch.consumed.resize(scratch.old.len(), false);
    manifolds.clear();

    let matrix_b = Mat3::from_quat(pair.xf_convex.q);
    for c in &scratch.clusters {
        let normal = matrix_b * c.manifold_normal;
        let mut manifold = ContactManifold {
            body_a: mesh_body,
            body_b: convex_body,
            normal,
            mesh: true,
            ..ContactManifold::default()
        };
        let mut best_dot = NORMAL_MATCH_TOLERANCE;
        let mut best = None;
        for (j, old) in scratch.old.iter().enumerate() {
            if scratch.consumed[j] {
                continue;
            }
            let dot = old.normal.dot(normal);
            if dot > best_dot {
                best = Some(j);
                best_dot = dot;
            }
        }
        if let Some(j) = best {
            let old = &scratch.old[j];
            manifold.friction_impulse = old.friction_impulse;
            manifold.rolling_impulse = old.rolling_impulse;
            manifold.twist_impulse = old.twist_impulse;
            scratch.consumed[j] = true;
        }
        let mut points = ContactPoints::new();
        for source in &scratch.cluster_points[c.start..c.start + c.count] {
            let separation = source.separation - MESH_REST_OFFSET;
            let feature_id = source.pair.id();
            let mut normal_impulse = 0.0;
            if let Some(j) = best {
                // Each old point is claimed once.
                if let Some(old) = scratch.old[j]
                    .points
                    .as_mut_slice()
                    .iter_mut()
                    .find(|p| p.feature_id == feature_id && p.triangle_index == source.triangle)
                {
                    normal_impulse = old.normal_impulse;
                    old.triangle_index = u32::MAX;
                }
            }
            points.push(ContactPoint {
                point: pair.xf_convex.p + matrix_b * source.point,
                normal,
                separation,
                normal_impulse,
                feature_id,
                triangle_index: source.triangle,
                base_separation: separation,
                ..ContactPoint::default()
            });
        }
        manifold.points = points;
        if flip {
            manifold.flip();
        }
        manifolds.push(manifold);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qsort_sorts_like_box3d() {
        // Long enough to partition; equal keys keep a deterministic order.
        let mut values: Vec<(u32, usize)> =
            (0..40).map(|i| ((i * 7919) % 13, i as usize)).collect();
        box3d_qsort(&mut values, |a, b| a.0 < b.0);
        assert!(values.windows(2).all(|w| w[0].0 <= w[1].0));
        let mut short = [(3, 0), (1, 1), (3, 2), (1, 3)];
        box3d_qsort(&mut short, |a, b| a.0 < b.0);
        // Insertion sort: stable.
        assert_eq!(short, [(1, 1), (1, 3), (3, 0), (3, 2)]);
    }

    #[test]
    fn culling_keeps_a_quad() {
        let mut points: Vec<Point2> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| Point2 {
                x,
                y,
                separation: 0.0,
                original: i,
            })
            .collect();
        let count = cull_points(&mut points);
        assert_eq!(count, 4);
        let mut kept: Vec<usize> = points[..4].iter().map(|p| p.original).collect();
        kept.sort_unstable();
        assert_eq!(kept, [0, 1, 2, 3], "the centre adds no area");
    }
}
