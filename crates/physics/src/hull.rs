//! Convex hulls with Box3D's half-edge topology (`b3HullData`, `hull.c`).
//!
//! A [`Hull3d`] is built from a point cloud by Box3D's quickhull (`b3CreateHull`), with its
//! exact volume, centroid, inertia tensor and inner radius. A [`HullView`] is what the manifold
//! builders read: the points, face planes, half-edges, one half-edge per face and per vertex,
//! and the local bounds.
//! Boxes are hulls built from Box3D's box template ([`BoxHull`], `b3MakeBoxHull`), with the same
//! vertex, edge and face numbering, so that ties between features resolve as in Box3D. Hulls
//! built from point clouds come with FIX-3.5 (P3D-11).

use korppi_math::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

/// The most vertices a hull has (`B3_MAX_HULL_VERTICES`).
pub const MAX_HULL_VERTICES: usize = 128;

/// The most faces a hull has (`B3_MAX_HULL_FACES`).
pub const MAX_HULL_FACES: usize = 128;

/// The most edges a hull has (`B3_MAX_HULL_EDGES`), each two half-edges.
pub const MAX_HULL_EDGES: usize = 128;

/// A plane `dot(normal, x) = offset` (Box3D `b3Plane`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

impl Plane {
    /// Box3D `b3MakePlaneFromPoints`: the plane through three points, facing where they turn
    /// counter-clockwise.
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let normal = crate::contact_solver::normalize((b - a).cross(c - a));
        Self {
            normal,
            offset: normal.dot(a),
        }
    }

    /// Box3D `b3MakePlaneFromNormalAndPoint`.
    pub fn from_normal_and_point(normal: Vec3, point: Vec3) -> Self {
        Self {
            normal,
            offset: normal.dot(point),
        }
    }

    /// Box3D `b3PlaneSeparation`: the signed distance of `point` from the plane.
    pub fn separation(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.offset
    }
}

/// A half-edge (Box3D `b3HullHalfEdge`): the next half-edge counter-clockwise around its face,
/// its twin, its origin vertex and the face on its left.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct HalfEdge {
    pub next: u8,
    pub twin: u8,
    pub origin: u8,
    pub face: u8,
}

const fn edge(next: u8, twin: u8, origin: u8, face: u8) -> HalfEdge {
    HalfEdge {
        next,
        twin,
        origin,
        face,
    }
}

/// The topology of Box3D's box template (`s_boxHull` in `hull.c`).
const BOX_EDGES: [HalfEdge; 24] = [
    edge(2, 1, 2, 0),
    edge(17, 0, 1, 5),
    edge(4, 3, 1, 0),
    edge(20, 2, 5, 3),
    edge(6, 5, 5, 0),
    edge(23, 4, 6, 4),
    edge(0, 7, 6, 0),
    edge(18, 6, 2, 2),
    edge(10, 9, 0, 1),
    edge(21, 8, 3, 5),
    edge(12, 11, 3, 1),
    edge(16, 10, 7, 2),
    edge(14, 13, 7, 1),
    edge(19, 12, 4, 4),
    edge(8, 15, 4, 1),
    edge(22, 14, 0, 3),
    edge(7, 17, 3, 2),
    edge(9, 16, 2, 5),
    edge(11, 19, 6, 2),
    edge(5, 18, 7, 4),
    edge(15, 21, 1, 3),
    edge(1, 20, 0, 5),
    edge(3, 23, 4, 3),
    edge(13, 22, 5, 4),
];
const BOX_FACE_EDGES: [u8; 6] = [0, 8, 16, 20, 19, 21];
const BOX_VERTEX_EDGES: [u8; 8] = [8, 1, 0, 9, 13, 3, 5, 11];

/// What the manifold builders read of a hull.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HullView<'a> {
    pub points: &'a [Vec3],
    pub planes: &'a [Plane],
    pub edges: &'a [HalfEdge],
    /// A half-edge of each face.
    pub face_edges: &'a [u8],
    /// A half-edge leaving each vertex.
    pub vertex_edges: &'a [u8],
    /// The centre and half extents of the local bounds.
    pub aabb_center: Vec3,
    pub aabb_extents: Vec3,
    /// The centroid, which a mesh triangle must have in front to collide.
    pub center: Vec3,
}

impl HullView<'_> {
    /// Box3D `b3FindHullSupportVertex`: the vertex farthest along `direction`, the first of
    /// equals.
    pub fn support_vertex(&self, direction: Vec3) -> usize {
        let mut best = 0;
        let mut best_dot = -f32::MAX;
        for (index, point) in self.points.iter().enumerate() {
            let dot = direction.dot(*point);
            if dot > best_dot {
                best = index;
                best_dot = dot;
            }
        }
        best
    }

    /// Box3D `b3FindHullSupportFace`: the face whose normal is most along `direction`.
    pub fn support_face(&self, direction: Vec3) -> usize {
        let mut best = 0;
        let mut best_dot = -f32::MAX;
        for (index, plane) in self.planes.iter().enumerate() {
            let dot = plane.normal.dot(direction);
            if dot > best_dot {
                best = index;
                best_dot = dot;
            }
        }
        best
    }
}

/// A convex hull (Box3D `b3HullData`): points, face planes and half-edges numbered as Box3D's
/// `b3CreateHull` numbers them, with the hull's volume, centroid, inertia tensor at unit density
/// about the centroid, and the radius of the largest sphere about the centroid inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hull3d {
    points: Vec<Vec3>,
    planes: Vec<Plane>,
    edges: Vec<HalfEdge>,
    face_edges: Vec<u8>,
    vertex_edges: Vec<u8>,
    lower: Vec3,
    upper: Vec3,
    center: Vec3,
    volume: f32,
    central_inertia: Mat3,
    surface_area: f32,
    inner_radius: f32,
}

impl Hull3d {
    /// The hull of `points` with at most [`MAX_HULL_VERTICES`] vertices, or `None` when the
    /// points span no volume (Box3D `b3CreateHull`).
    #[must_use]
    pub fn new(points: &[Vec3]) -> Option<Self> {
        Self::with_max_vertices(points, MAX_HULL_VERTICES)
    }

    /// A rectangular box hull matching Box3D's `b3MakeBoxHull`.
    #[must_use]
    pub fn box_hull(hx: f32, hy: f32, hz: f32) -> Self {
        let points = [
            Vec3::new(-hx, -hy, -hz),
            Vec3::new(hx, -hy, -hz),
            Vec3::new(hx, hy, -hz),
            Vec3::new(-hx, hy, -hz),
            Vec3::new(-hx, -hy, hz),
            Vec3::new(hx, -hy, hz),
            Vec3::new(hx, hy, hz),
            Vec3::new(-hx, hy, hz),
        ];
        Self::new(&points).expect("box hull has volume")
    }

    /// The hull of `points` with at most `max_vertices` vertices (4 to 128): quickhull stops
    /// adding the farthest outside points once it has that many.
    #[must_use]
    pub fn with_max_vertices(points: &[Vec3], max_vertices: usize) -> Option<Self> {
        let built = crate::quickhull::build(points, max_vertices)?;
        let (lower, upper) = built
            .points
            .iter()
            .skip(1)
            .fold((built.points[0], built.points[0]), |(lo, hi), &p| {
                (lo.min(p), hi.max(p))
            });
        let mut hull = Self {
            points: built.points,
            planes: built.planes,
            edges: built.edges,
            face_edges: built.face_edges,
            vertex_edges: built.vertex_edges,
            lower,
            upper,
            center: Vec3::ZERO,
            volume: 0.0,
            central_inertia: Mat3::ZERO,
            surface_area: 0.0,
            inner_radius: 0.0,
        };
        hull.update_bulk_properties().then_some(hull)
    }

    /// Box3D `b3UpdateHullBulkProperties` (Kallay's moments of a solid from its faces' fans):
    /// volume, centroid, inertia about it, surface area and inner radius. False for a hull
    /// without volume.
    fn update_bulk_properties(&mut self) -> bool {
        let origin = self.points[0];
        let (mut area, mut volume, mut center) = (0.0, 0.0, Vec3::ZERO);
        let (mut xx, mut xy, mut yy, mut xz, mut zz, mut yz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for &first in &self.face_edges {
            let edge1 = self.edges[first as usize];
            let mut edge2 = self.edges[edge1.next as usize];
            let mut edge3_index = edge2.next;
            let v1 = self.points[edge1.origin as usize] - origin;
            loop {
                let edge3 = self.edges[edge3_index as usize];
                let v2 = self.points[edge2.origin as usize] - origin;
                let v3 = self.points[edge3.origin as usize] - origin;
                area += (v2 - v1).cross(v3 - v1).length();
                let det = v1.dot(v2.cross(v3));
                volume += det;
                let v4 = v1 + (v2 + v3);
                center += v4 * det;
                xx += det * (v1.x * v1.x + v2.x * v2.x + v3.x * v3.x + v4.x * v4.x);
                yy += det * (v1.y * v1.y + v2.y * v2.y + v3.y * v3.y + v4.y * v4.y);
                zz += det * (v1.z * v1.z + v2.z * v2.z + v3.z * v3.z + v4.z * v4.z);
                xy += det * (v1.x * v1.y + v2.x * v2.y + v3.x * v3.y + v4.x * v4.y);
                xz += det * (v1.x * v1.z + v2.x * v2.z + v3.x * v3.z + v4.x * v4.z);
                yz += det * (v1.y * v1.z + v2.y * v2.z + v3.y * v3.z + v4.y * v4.z);
                edge2 = edge3;
                edge3_index = edge3.next;
                if edge3_index == first {
                    break;
                }
            }
        }
        let local_center = if volume > 0.0 {
            center * (0.25 / volume)
        } else {
            Vec3::ZERO
        };
        let center = local_center + origin;
        let radius = self
            .planes
            .iter()
            .fold(f32::MAX, |r, plane| r.min(-plane.separation(center)));
        let inertia = Mat3::from_cols(
            Vec3::new(yy + zz, -xy, -xz),
            Vec3::new(-xy, xx + zz, -yz),
            Vec3::new(-xz, -yz, xx + yy),
        );
        let mass = volume / 6.0;
        self.center = center;
        self.central_inertia = inertia * (1.0 / 120.0) - crate::math::steiner(mass, local_center);
        self.volume = mass;
        self.surface_area = 0.5 * area;
        self.inner_radius = radius;
        mass > 0.0 && volume > 0.0 && area > 0.0 && radius > 0.0
    }

    /// The vertices, in Box3D's order.
    #[must_use]
    pub fn points(&self) -> &[Vec3] {
        &self.points
    }

    /// The number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.points.len()
    }

    /// The number of faces, convex polygons.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.planes.len()
    }

    /// The number of half-edges, twice the edges.
    #[must_use]
    pub fn half_edge_count(&self) -> usize {
        self.edges.len()
    }

    /// The edges as pairs of vertex indices, each once.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.edges
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[edge, twin]| (edge.origin as usize, twin.origin as usize))
    }

    /// The volume, m³.
    #[must_use]
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// The centroid.
    #[must_use]
    pub fn center(&self) -> Vec3 {
        self.center
    }

    /// The inertia tensor about the centroid at a density of 1 kg/m³.
    #[must_use]
    pub fn central_inertia(&self) -> Mat3 {
        self.central_inertia
    }

    /// The surface area, m².
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        self.surface_area
    }

    /// The radius of the largest sphere about the centroid inside the hull.
    #[must_use]
    pub fn inner_radius(&self) -> f32 {
        self.inner_radius
    }

    /// The local bounds.
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.lower, self.upper)
    }

    /// Box3D `b3RayCastHull`: the fraction of `translation` from `origin` at which the ray
    /// enters the hull, and the face normal there; zero with a zero normal when it starts
    /// inside.
    #[must_use]
    pub fn ray_cast(
        &self,
        origin: Vec3,
        translation: Vec3,
        max_fraction: f32,
    ) -> Option<(f32, Vec3)> {
        let (mut lower, mut upper) = (0.0f32, max_fraction);
        let mut best_face = None;
        for (index, plane) in self.planes.iter().enumerate() {
            let distance = plane.offset - plane.normal.dot(origin);
            let denominator = plane.normal.dot(translation);
            if denominator == 0.0 {
                if distance < 0.0 {
                    return None;
                }
            } else {
                let fraction = distance / denominator;
                if denominator < 0.0 {
                    if fraction > lower {
                        best_face = Some(index);
                        lower = fraction;
                    }
                } else if fraction < upper {
                    upper = fraction;
                }
                if upper < lower {
                    return None;
                }
            }
        }
        Some(best_face.map_or((0.0, Vec3::ZERO), |face| (lower, self.planes[face].normal)))
    }

    pub(crate) fn view(&self) -> HullView<'_> {
        HullView {
            points: &self.points,
            planes: &self.planes,
            edges: &self.edges,
            face_edges: &self.face_edges,
            vertex_edges: &self.vertex_edges,
            aabb_center: (self.lower + self.upper) * 0.5,
            aabb_extents: (self.upper - self.lower) * 0.5,
            center: self.center,
        }
    }
}

/// A box as a hull (Box3D `b3BoxHull` from `b3MakeBoxHull`), centred on the body origin.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoxHull {
    points: [Vec3; 8],
    planes: [Plane; 6],
    half_extents: Vec3,
}

impl BoxHull {
    /// The box of `half_extents`, each at least a fifth of the linear slop as in Box3D.
    pub fn new(half_extents: Vec3) -> Self {
        let min = 0.2 * crate::contact_solver::LINEAR_SLOP;
        let h = half_extents.max(Vec3::splat(min));
        let (lower, upper) = (-h, h);
        let planes = [
            Plane::from_normal_and_point(-Vec3::X, lower),
            Plane::from_normal_and_point(Vec3::X, upper),
            Plane::from_normal_and_point(-Vec3::Y, lower),
            Plane::from_normal_and_point(Vec3::Y, upper),
            Plane::from_normal_and_point(-Vec3::Z, lower),
            Plane::from_normal_and_point(Vec3::Z, upper),
        ];
        let points = [
            Vec3::new(h.x, h.y, h.z),
            Vec3::new(-h.x, h.y, h.z),
            Vec3::new(-h.x, -h.y, h.z),
            Vec3::new(h.x, -h.y, h.z),
            Vec3::new(h.x, h.y, -h.z),
            Vec3::new(-h.x, h.y, -h.z),
            Vec3::new(-h.x, -h.y, -h.z),
            Vec3::new(h.x, -h.y, -h.z),
        ];
        Self {
            points,
            planes,
            half_extents: h,
        }
    }

    /// The corners, in Box3D's order.
    pub fn points(&self) -> &[Vec3; 8] {
        &self.points
    }

    pub fn view(&self) -> HullView<'_> {
        HullView {
            points: &self.points,
            planes: &self.planes,
            edges: &BOX_EDGES,
            face_edges: &BOX_FACE_EDGES,
            vertex_edges: &BOX_VERTEX_EDGES,
            aabb_center: Vec3::ZERO,
            aabb_extents: self.half_extents,
            center: Vec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_hulls_are_closed_and_convex() {
        // A squashed, wavy sphere of 30 points with an interior point and duplicates.
        let mut points: Vec<Vec3> = (0..30)
            .map(|i| {
                let phi = (1.0 - 2.0 * (i as f32 + 0.5) / 30.0).acos();
                let theta = 2.399_963 * i as f32;
                let r = 0.5 + 0.1 * (5.0 * theta).sin();
                Vec3::new(
                    r * phi.sin() * theta.cos(),
                    0.6 * r * phi.cos(),
                    r * phi.sin() * theta.sin(),
                )
            })
            .collect();
        points.push(Vec3::ZERO);
        points.push(points[3]);
        let hull = Hull3d::new(&points).expect("a hull");
        let view = hull.view();
        let (v, e, f) = (
            hull.vertex_count(),
            hull.half_edge_count() / 2,
            hull.face_count(),
        );
        assert_eq!(v + f, e + 2, "Euler's formula");
        for (i, edge) in view.edges.iter().enumerate() {
            let twin = view.edges[edge.twin as usize];
            assert_eq!(twin.twin as usize, i);
            assert_eq!(view.edges[edge.next as usize].origin, twin.origin);
            let plane = view.planes[edge.face as usize];
            assert!(plane.separation(view.points[edge.origin as usize]).abs() < 1e-4);
        }
        for (face, &edge) in view.face_edges.iter().enumerate() {
            assert_eq!(view.edges[edge as usize].face as usize, face);
        }
        for (vertex, &edge) in view.vertex_edges.iter().enumerate() {
            assert_eq!(view.edges[edge as usize].origin as usize, vertex);
        }
        // Every input point is inside every face plane.
        for p in &points {
            for plane in view.planes {
                assert!(plane.separation(*p) < 1e-4);
            }
        }
        assert!(hull.inner_radius() > 0.0 && hull.volume() > 0.0);
        assert!(
            Hull3d::with_max_vertices(&points, 6)
                .unwrap()
                .vertex_count()
                <= 6
        );
    }

    #[test]
    fn box_topology_is_consistent() {
        let hull = BoxHull::new(Vec3::new(1.0, 2.0, 3.0));
        let view = hull.view();
        for (i, e) in view.edges.iter().enumerate() {
            let twin = view.edges[e.twin as usize];
            assert_eq!(twin.twin as usize, i, "twins pair up");
            // An edge runs from its origin to its twin's origin, on the face to its left: both
            // ends lie on that face's plane.
            let plane = view.planes[e.face as usize];
            for v in [e.origin, twin.origin] {
                assert!(plane.separation(view.points[v as usize]).abs() < 1e-6);
            }
            // The next edge starts where this one ends.
            assert_eq!(view.edges[e.next as usize].origin, twin.origin);
        }
        for (face, &e) in view.face_edges.iter().enumerate() {
            assert_eq!(view.edges[e as usize].face as usize, face);
        }
        for (vertex, &e) in view.vertex_edges.iter().enumerate() {
            assert_eq!(view.edges[e as usize].origin as usize, vertex);
        }
        assert_eq!(view.support_vertex(Vec3::new(-1.0, 1.0, 1.0)), 1);
        assert_eq!(view.support_face(Vec3::new(0.1, -1.0, 0.0)), 2);
    }
}
