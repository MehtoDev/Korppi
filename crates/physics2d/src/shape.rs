//! 2D collision primitives, support mapping functions, and mass properties.

use crate::hull::Hull2d;
use crate::math::{cross, cross_vs, lerp, normalize, right_perp};
use korppi_math::{Aabb2d, Rot2, Transform2d, Vec2};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Computed mass and rotational inertia for a 2D shape.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MassData2d {
    /// Total mass in kilograms.
    pub mass: f32,
    /// Center of mass in local shape space.
    pub center: Vec2,
    /// Rotational inertia (moment of inertia) around the z-axis through the center of mass.
    pub rotational_inertia: f32,
}

impl Default for MassData2d {
    fn default() -> Self {
        Self {
            mass: 0.0,
            center: Vec2::ZERO,
            rotational_inertia: 0.0,
        }
    }
}

/// A 2D circle shape defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle2d {
    /// Local center offset.
    pub center: Vec2,
    /// Circle radius.
    pub radius: f32,
}

impl Circle2d {
    /// Creates a circle with the specified center and radius.
    #[must_use]
    pub const fn new(center: Vec2, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Computes the support point in the given direction in local space.
    #[must_use]
    pub fn support(&self, direction: Vec2) -> Vec2 {
        let len = direction.length();
        if len > 1e-6 {
            self.center + (direction / len) * self.radius
        } else {
            self.center
        }
    }

    /// Computes the world AABB under a transform.
    #[must_use]
    pub fn compute_aabb(&self, transform: &Transform2d) -> Aabb2d {
        let center = transform.transform_point(self.center);
        let r = Vec2::splat(self.radius);
        Aabb2d::new(center - r, center + r)
    }

    /// Computes mass properties for a given material density (Box2D `b2ComputeCircleMass`).
    #[must_use]
    pub fn compute_mass_properties(&self, density: f32) -> MassData2d {
        let rr = self.radius * self.radius;
        let mass = density * PI * rr;
        MassData2d {
            mass,
            center: self.center,
            rotational_inertia: 0.5 * mass * rr,
        }
    }
}

/// A 2D capsule (lozenge) defined by two end-points and a radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Capsule2d {
    /// First interior vertex.
    pub point1: Vec2,
    /// Second interior vertex.
    pub point2: Vec2,
    /// Rounding radius.
    pub radius: f32,
}

impl Capsule2d {
    /// Creates a new capsule.
    #[must_use]
    pub const fn new(point1: Vec2, point2: Vec2, radius: f32) -> Self {
        Self {
            point1,
            point2,
            radius,
        }
    }

    /// Computes the support point in the given direction in local space.
    #[must_use]
    pub fn support(&self, direction: Vec2) -> Vec2 {
        let len = direction.length();
        let unit_dir = if len > 1e-6 {
            direction / len
        } else {
            Vec2::ZERO
        };
        let p = if self.point1.dot(direction) > self.point2.dot(direction) {
            self.point1
        } else {
            self.point2
        };
        p + unit_dir * self.radius
    }

    /// Computes the world AABB under a transform.
    #[must_use]
    pub fn compute_aabb(&self, transform: &Transform2d) -> Aabb2d {
        let p1 = transform.transform_point(self.point1);
        let p2 = transform.transform_point(self.point2);
        let min = p1.min(p2) - Vec2::splat(self.radius);
        let max = p1.max(p2) + Vec2::splat(self.radius);
        Aabb2d::new(min, max)
    }

    /// Computes mass properties for a given material density (Box2D `b2ComputeCapsuleMass`).
    #[must_use]
    pub fn compute_mass_properties(&self, density: f32) -> MassData2d {
        let rr = self.radius * self.radius;
        let length = (self.point2 - self.point1).length();
        let circle_mass = density * PI * rr;
        let box_mass = density * (2.0 * self.radius) * length;

        // The two half circles add up to a full circle; each sits half a length from the
        // centre, with its own centroid a further 4r / 3pi out (parallel-axis theorem twice).
        let half_circle_centroid = 4.0 * self.radius / (3.0 * PI);
        let h = 0.5 * length;
        let circle_inertia = circle_mass * (0.5 * rr + h * h + 2.0 * h * half_circle_centroid);
        let box_inertia = box_mass * (4.0 * rr + length * length) / 12.0;

        MassData2d {
            mass: circle_mass + box_mass,
            center: (self.point1 + self.point2) * 0.5,
            rotational_inertia: circle_inertia + box_inertia,
        }
    }
}

/// Most vertices a polygon has (Box2D `B2_MAX_POLYGON_VERTICES`).
pub const MAX_POLYGON_VERTICES: usize = 8;

/// A convex polygon of up to [`MAX_POLYGON_VERTICES`] vertices (Box2D `b2Polygon`): vertices
/// counter-clockwise, the outward normal of each edge, the centroid and a rounding radius.
///
/// Polygons are made from a [`Hull2d`], so their vertices are always a valid hull:
/// [`Self::from_vertices`] computes one from any points, [`Self::from_hull`] and
/// [`Self::from_offset_hull`] take one, and the box makers build boxes directly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PolygonData", into = "PolygonData")]
pub struct Polygon2d {
    pub(crate) vertices: [Vec2; MAX_POLYGON_VERTICES],
    pub(crate) normals: [Vec2; MAX_POLYGON_VERTICES],
    pub(crate) centroid: Vec2,
    pub(crate) count: usize,
    /// Rounding radius: the polygon's surface lies this far outside its vertices and edges.
    pub radius: f32,
}

/// A polygon's serialized form: deserializing checks that the vertices form a hull.
#[derive(Serialize, Deserialize)]
#[serde(rename = "Polygon2d")]
struct PolygonData {
    vertices: Vec<Vec2>,
    normals: Vec<Vec2>,
    centroid: Vec2,
    radius: f32,
}

impl From<Polygon2d> for PolygonData {
    fn from(polygon: Polygon2d) -> Self {
        Self {
            vertices: polygon.vertices().to_vec(),
            normals: polygon.normals().to_vec(),
            centroid: polygon.centroid,
            radius: polygon.radius,
        }
    }
}

impl TryFrom<PolygonData> for Polygon2d {
    type Error = &'static str;

    fn try_from(data: PolygonData) -> Result<Self, Self::Error> {
        if !Hull2d::validate(&data.vertices) || data.normals.len() != data.vertices.len() {
            return Err("polygon vertices are not a convex hull with one normal per edge");
        }
        let count = data.vertices.len();
        let mut polygon = Self {
            vertices: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            normals: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            centroid: data.centroid,
            count,
            radius: data.radius,
        };
        polygon.vertices[..count].copy_from_slice(&data.vertices);
        polygon.normals[..count].copy_from_slice(&data.normals);
        Ok(polygon)
    }
}

impl Polygon2d {
    /// The convex hull of `points` as a polygon (Box2D `b2ComputeHull` then `b2MakePolygon`),
    /// or `None` if the hull fails (see [`Hull2d::compute`]).
    #[must_use]
    pub fn from_vertices(points: &[Vec2], radius: f32) -> Option<Self> {
        Hull2d::compute(points).map(|hull| Self::from_hull(&hull, radius))
    }

    /// A polygon of `hull`'s points (Box2D `b2MakePolygon`).
    #[must_use]
    pub fn from_hull(hull: &Hull2d, radius: f32) -> Self {
        Self::from_points(hull.points(), &Transform2d::IDENTITY, radius)
    }

    /// A polygon of `hull`'s points moved by `position` and turned by `rotation` (Box2D
    /// `b2MakeOffsetRoundedPolygon`).
    #[must_use]
    pub fn from_offset_hull(hull: &Hull2d, position: Vec2, rotation: Rot2, radius: f32) -> Self {
        Self::from_points(
            hull.points(),
            &Transform2d::from_translation_rotation(position, rotation),
            radius,
        )
    }

    /// The polygon of hull points `points` under `xf`, with normals and centroid computed.
    fn from_points(points: &[Vec2], xf: &Transform2d, radius: f32) -> Self {
        let count = points.len();
        let mut polygon = Self {
            vertices: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            normals: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            centroid: Vec2::ZERO,
            count,
            radius,
        };
        for (vertex, &point) in polygon.vertices.iter_mut().zip(points) {
            *vertex = xf.transform_point(point);
        }
        for i in 0..count {
            let edge = polygon.vertices[polygon.next(i)] - polygon.vertices[i];
            polygon.normals[i] = normalize(cross_vs(edge, 1.0));
        }
        polygon.centroid = polygon_centroid(polygon.vertices());
        polygon
    }

    /// A box centred on the origin with the given half-extents (Box2D `b2MakeBox`).
    #[must_use]
    pub fn from_box(half_width: f32, half_height: f32) -> Self {
        Self::offset_box(half_width, half_height, Vec2::ZERO, Rot2::IDENTITY)
    }

    /// A square centred on the origin (Box2D `b2MakeSquare`).
    #[must_use]
    pub fn square(half_width: f32) -> Self {
        Self::from_box(half_width, half_width)
    }

    /// A box with rounded corners (Box2D `b2MakeRoundedBox`).
    #[must_use]
    pub fn rounded_box(half_width: f32, half_height: f32, radius: f32) -> Self {
        Self::offset_rounded_box(half_width, half_height, Vec2::ZERO, Rot2::IDENTITY, radius)
    }

    /// A box centred on `center` and turned by `rotation` (Box2D `b2MakeOffsetBox`); its
    /// centroid is `center`.
    #[must_use]
    pub fn offset_box(half_width: f32, half_height: f32, center: Vec2, rotation: Rot2) -> Self {
        Self::offset_rounded_box(half_width, half_height, center, rotation, 0.0)
    }

    /// An offset box with rounded corners (Box2D `b2MakeOffsetRoundedBox`).
    #[must_use]
    pub fn offset_rounded_box(
        half_width: f32,
        half_height: f32,
        center: Vec2,
        rotation: Rot2,
        radius: f32,
    ) -> Self {
        let (hw, hh) = (half_width.abs(), half_height.abs());
        let xf = Transform2d::from_translation_rotation(center, rotation);
        let corners = [
            Vec2::new(-hw, -hh),
            Vec2::new(hw, -hh),
            Vec2::new(hw, hh),
            Vec2::new(-hw, hh),
        ];
        let normals = [
            Vec2::new(0.0, -1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, 0.0),
        ];
        let mut polygon = Self {
            vertices: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            normals: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            centroid: center,
            count: 4,
            radius,
        };
        for i in 0..4 {
            polygon.vertices[i] = xf.transform_point(corners[i]);
            polygon.normals[i] = rotation.rotate(normals[i]);
        }
        polygon
    }

    /// A capsule or segment as a two-vertex rounded polygon (Box2D `b2MakeCapsule`), as the
    /// manifolds collide them with polygons.
    pub(crate) fn capsule(p1: Vec2, p2: Vec2, radius: f32) -> Self {
        let normal = right_perp(normalize(p2 - p1));
        let mut polygon = Self {
            vertices: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            normals: [Vec2::ZERO; MAX_POLYGON_VERTICES],
            centroid: lerp(p1, p2, 0.5),
            count: 2,
            radius,
        };
        polygon.vertices[..2].copy_from_slice(&[p1, p2]);
        polygon.normals[..2].copy_from_slice(&[normal, -normal]);
        polygon
    }

    /// The polygon moved by `xf` (Box2D `b2TransformPolygon`).
    #[must_use]
    pub fn transformed(&self, xf: &Transform2d) -> Self {
        let mut polygon = *self;
        for i in 0..self.count {
            polygon.vertices[i] = xf.transform_point(self.vertices[i]);
            polygon.normals[i] = xf.rotation.rotate(self.normals[i]);
        }
        polygon.centroid = xf.transform_point(self.centroid);
        polygon
    }

    /// The vertices, counter-clockwise.
    #[must_use]
    pub fn vertices(&self) -> &[Vec2] {
        &self.vertices[..self.count]
    }

    /// The outward unit normal of each edge, edge `i` running from vertex `i` to the next.
    #[must_use]
    pub fn normals(&self) -> &[Vec2] {
        &self.normals[..self.count]
    }

    /// Number of vertices.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Area centroid of the vertices, ignoring the rounding radius (Box2D `b2Polygon::centroid`).
    #[must_use]
    pub const fn centroid(&self) -> Vec2 {
        self.centroid
    }

    /// The vertex after vertex `i`.
    pub(crate) const fn next(&self, i: usize) -> usize {
        if i + 1 < self.count {
            i + 1
        } else {
            0
        }
    }

    /// Computes the support point in the given direction in local space.
    #[must_use]
    pub fn support(&self, direction: Vec2) -> Vec2 {
        let mut best_index = 0;
        let mut max_dot = self.vertices[0].dot(direction);

        for (i, &vertex) in self.vertices().iter().enumerate().skip(1) {
            let dot = vertex.dot(direction);
            if dot > max_dot {
                max_dot = dot;
                best_index = i;
            }
        }

        let p = self.vertices[best_index];
        if self.radius > 0.0 {
            let len = direction.length();
            if len > 1e-6 {
                p + (direction / len) * self.radius
            } else {
                p
            }
        } else {
            p
        }
    }

    /// Computes the world AABB under a transform.
    #[must_use]
    pub fn compute_aabb(&self, transform: &Transform2d) -> Aabb2d {
        let mut aabb = Aabb2d::empty();
        for &vertex in self.vertices() {
            let p = transform.transform_point(vertex);
            aabb = aabb.union(&Aabb2d::from_point(p));
        }
        if self.radius > 0.0 {
            aabb = aabb.fatten(self.radius);
        }
        aabb
    }

    /// Computes mass properties for a given material density using polygon triangulation
    /// (Box2D `b2ComputePolygonMass`).
    ///
    /// A rounded polygon is approximated, as in Box2D, by pushing each vertex out along its
    /// corner's bisector by `1.412 * radius`.
    #[must_use]
    pub fn compute_mass_properties(&self, density: f32) -> MassData2d {
        let count = self.count;
        let vertex = |i: usize| {
            if self.radius > 0.0 {
                let previous = self.normals[if i == 0 { count - 1 } else { i - 1 }];
                let bisector = (previous + self.normals[i]).normalize();
                self.vertices[i] + bisector * (1.412 * self.radius)
            } else {
                self.vertices[i]
            }
        };

        // Triangles fan out from the first vertex to reduce round-off.
        let origin = vertex(0);
        let inv3 = 1.0 / 3.0;
        let mut center = Vec2::ZERO;
        let mut area = 0.0f32;
        let mut inertia = 0.0f32;
        for i in 1..count - 1 {
            let e1 = vertex(i) - origin;
            let e2 = vertex(i + 1) - origin;
            let d = e1.x * e2.y - e1.y * e2.x;
            let triangle_area = 0.5 * d;
            area += triangle_area;
            center += (e1 + e2) * (triangle_area * inv3);

            let intx2 = e1.x * e1.x + e2.x * e1.x + e2.x * e2.x;
            let inty2 = e1.y * e1.y + e2.y * e1.y + e2.y * e2.y;
            inertia += (0.25 * inv3 * d) * (intx2 + inty2);
        }

        if area <= f32::EPSILON {
            return MassData2d::default();
        }
        center /= area;
        let mass = density * area;
        MassData2d {
            mass,
            center: origin + center,
            // Inertia about the first vertex, shifted to the centre of mass.
            rotational_inertia: (density * inertia - mass * center.dot(center)).max(0.0),
        }
    }
}

/// A 2D line segment primitive defined by two endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Segment2d {
    /// First endpoint.
    pub point1: Vec2,
    /// Second endpoint.
    pub point2: Vec2,
}

impl Segment2d {
    /// Creates a line segment between two points.
    #[must_use]
    pub const fn new(point1: Vec2, point2: Vec2) -> Self {
        Self { point1, point2 }
    }

    /// Computes the support point in the given direction.
    #[must_use]
    pub fn support(&self, direction: Vec2) -> Vec2 {
        if self.point1.dot(direction) > self.point2.dot(direction) {
            self.point1
        } else {
            self.point2
        }
    }

    /// Computes the world AABB under a transform.
    #[must_use]
    pub fn compute_aabb(&self, transform: &Transform2d) -> Aabb2d {
        let p1 = transform.transform_point(self.point1);
        let p2 = transform.transform_point(self.point2);
        Aabb2d::new(p1.min(p2), p1.max(p2))
    }
}

/// A one-sided segment of a chain (Box2D `b2ChainSegment`): it collides on its right, the
/// side of the right perpendicular of `point1 -> point2`, and the neighbouring vertices
/// `ghost1` and `ghost2` keep bodies from catching on its ends. A counter-clockwise loop
/// therefore collides outside, a clockwise one inside; a chain of ground runs right to left.
/// Chain segments have no mass and do not collide with segments or other chain segments.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChainSegment2d {
    /// The vertex before `segment.point1`.
    pub ghost1: Vec2,
    /// The segment itself.
    pub segment: Segment2d,
    /// The vertex after `segment.point2`.
    pub ghost2: Vec2,
}

impl ChainSegment2d {
    /// The segment from `point1` to `point2`, between ghost vertices `ghost1` and `ghost2`.
    #[must_use]
    pub const fn new(ghost1: Vec2, point1: Vec2, point2: Vec2, ghost2: Vec2) -> Self {
        Self {
            ghost1,
            segment: Segment2d::new(point1, point2),
            ghost2,
        }
    }
}

/// Unified 2D collision geometry enum.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Shape2d {
    /// Circle primitive.
    Circle(Circle2d),
    /// Capsule primitive.
    Capsule(Capsule2d),
    /// Convex polygon primitive.
    Polygon(Polygon2d),
    /// Line segment primitive.
    Segment(Segment2d),
    /// One-sided segment of a chain, with ghost vertices.
    ChainSegment(ChainSegment2d),
}

impl Shape2d {
    /// The rounding radius of the shape (Box2D `b2GetShapeRadius`): a circle's or capsule's
    /// radius, a polygon's rounding radius, and zero for segments and chain segments.
    #[must_use]
    pub const fn radius(&self) -> f32 {
        match self {
            Self::Circle(c) => c.radius,
            Self::Capsule(c) => c.radius,
            Self::Polygon(p) => p.radius,
            Self::Segment(_) | Self::ChainSegment(_) => 0.0,
        }
    }

    /// The size Box2D scales a shape's broadphase margin by (`b2ComputeShapeMargin`): a circle's
    /// radius, half a capsule's length plus its radius, a polygon's largest vertex distance from
    /// its centroid, half a segment's length.
    #[must_use]
    pub fn margin_size(&self) -> f32 {
        match self {
            Self::Circle(c) => c.radius,
            Self::Capsule(c) => 0.5 * (c.point2 - c.point1).length() + c.radius,
            Self::Polygon(p) => {
                let centroid = p.centroid();
                p.vertices()
                    .iter()
                    .map(|&v| (v - centroid).length_squared())
                    .fold(0.0, f32::max)
                    .sqrt()
            }
            Self::Segment(s) | Self::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                0.5 * (s.point2 - s.point1).length()
            }
        }
    }

    /// Computes the world AABB of this shape under a transform.
    #[must_use]
    pub fn compute_aabb(&self, transform: &Transform2d) -> Aabb2d {
        match self {
            Self::Circle(c) => c.compute_aabb(transform),
            Self::Capsule(cap) => cap.compute_aabb(transform),
            Self::Polygon(p) => p.compute_aabb(transform),
            Self::Segment(s) => s.compute_aabb(transform),
            Self::ChainSegment(c) => c.segment.compute_aabb(transform),
        }
    }

    /// Centre of the shape in its local frame (Box2D `b2GetShapeCentroid`): a circle's centre,
    /// a capsule's or segment's midpoint, a polygon's area centroid.
    #[must_use]
    pub fn centroid(&self) -> Vec2 {
        match self {
            Self::Circle(c) => c.center,
            Self::Capsule(c) => (c.point1 + c.point2) * 0.5,
            Self::Polygon(p) => p.centroid(),
            Self::Segment(s) | Self::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                (s.point1 + s.point2) * 0.5
            }
        }
    }

    /// The smallest and largest extent of the shape (Box2D `b2ComputeShapeExtent`): the
    /// smallest is how far the shape reaches from its core in every direction (zero for
    /// segments), the largest is how far it reaches from `local_center`. Continuous
    /// collision treats a body as fast when it moves more than half its smallest extent.
    #[must_use]
    pub fn extent(&self, local_center: Vec2) -> (f32, f32) {
        let farthest = |points: &mut dyn Iterator<Item = Vec2>| {
            points
                .map(|p| (p - local_center).length_squared())
                .fold(0.0f32, f32::max)
                .sqrt()
        };
        match self {
            Self::Circle(c) => (c.radius, (c.center - local_center).length() + c.radius),
            Self::Capsule(c) => (
                c.radius,
                farthest(&mut [c.point1, c.point2].into_iter()) + c.radius,
            ),
            Self::Polygon(p) => {
                let centroid = p.centroid();
                let min_extent = p
                    .vertices()
                    .iter()
                    .zip(p.normals())
                    .map(|(&v, &n)| n.dot(v - centroid))
                    .fold(f32::INFINITY, f32::min);
                (
                    min_extent + p.radius,
                    farthest(&mut p.vertices().iter().copied()) + p.radius,
                )
            }
            Self::Segment(s) | Self::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                (0.0, farthest(&mut [s.point1, s.point2].into_iter()))
            }
        }
    }

    /// Computes mass properties for this shape given a material density.
    #[must_use]
    pub fn compute_mass_properties(&self, density: f32) -> MassData2d {
        match self {
            Self::Circle(c) => c.compute_mass_properties(density),
            Self::Capsule(cap) => cap.compute_mass_properties(density),
            Self::Polygon(p) => p.compute_mass_properties(density),
            Self::Segment(_) | Self::ChainSegment(_) => MassData2d::default(),
        }
    }

    /// Returns the maximum distance from the local origin to any surface point on this shape.
    #[must_use]
    pub fn max_radius(&self) -> f32 {
        match self {
            Self::Circle(c) => c.center.length() + c.radius,
            Self::Capsule(cap) => {
                let r1 = cap.point1.length() + cap.radius;
                let r2 = cap.point2.length() + cap.radius;
                r1.max(r2)
            }
            Self::Polygon(p) => {
                let mut max_r = 0.0f32;
                for &v in p.vertices() {
                    max_r = max_r.max(v.length());
                }
                max_r + p.radius
            }
            Self::Segment(s) | Self::ChainSegment(ChainSegment2d { segment: s, .. }) => {
                s.point1.length().max(s.point2.length())
            }
        }
    }
}

/// Area centroid of a convex polygon's vertices (Box2D `b2ComputePolygonCentroid`), from a
/// triangle fan about the first vertex to reduce round-off.
fn polygon_centroid(vertices: &[Vec2]) -> Vec2 {
    let origin = vertices[0];
    let inv3 = 1.0 / 3.0;
    let mut center = Vec2::ZERO;
    let mut area = 0.0f32;
    for i in 1..vertices.len() - 1 {
        let e1 = vertices[i] - origin;
        let e2 = vertices[i + 1] - origin;
        let a = 0.5 * cross(e1, e2);
        let weight = a * inv3;
        let sum = e1 + e2;
        center = Vec2::new(center.x + weight * sum.x, center.y + weight * sum.y);
        area += a;
    }
    let inv_area = 1.0 / area;
    origin + Vec2::new(center.x * inv_area, center.y * inv_area)
}
