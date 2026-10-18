//! Narrowphase: contact manifolds of shape pairs, ported from Box3D.
//!
//! Spheres, capsules, boxes and convex hulls collide with Box3D's manifold builders
//! ([`crate::manifold`]): points midway between the surfaces, kept up to the speculative
//! distance apart (Box3D adds no margin between spheres, or between a capsule and a sphere),
//! with the contact's GJK simplex and separating axis cached between steps in a
//! [`ContactCache`]. Meshes and triangles collide with convex shapes through Box3D's mesh
//! contacts ([`crate::mesh_contact`]), one-sided, in one manifold per cluster of normals.
//! Sensors test overlaps with the GJK distance.

use crate::collider::Collider;
use crate::contact_solver::normalize;
use crate::distance::{
    shape_distance, time_of_impact, ShapeProxy, SimplexCache, Sweep, OVERLAP_SLOP,
};
use crate::hull::{BoxHull, Hull3d, HullView};
use crate::manifold::{
    collide_capsule_and_sphere, collide_capsules, collide_hull_and_capsule,
    collide_hull_and_sphere, collide_hulls, collide_spheres, Capsule, LocalManifold, SatCache,
    Sphere,
};
use crate::math::Transform;
use crate::mesh_contact::{
    update_mesh_manifolds, MeshContactCache, MeshPair, MeshScratch, Triangles,
};
use crate::shape::Shape;
use korppi_math::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// A single point of physical contact between two colliding bodies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactPoint {
    /// World-space contact position.
    pub point: Vec3,
    /// Normal vector pointing from body A to body B.
    pub normal: Vec3,
    /// Penetration distance (negative indicates penetration depth).
    pub separation: f32,
    /// Accumulated normal impulse for warm starting.
    pub normal_impulse: f32,
    /// Normal impulse applied over the last step's relax and restitution passes; zero for a
    /// speculative point that did not touch.
    #[serde(default)]
    pub total_normal_impulse: f32,
    /// The features of the two shapes that made the point (Box3D's `featureId`), or a mesh
    /// triangle's index; warm starting matches points by it.
    #[serde(default)]
    pub feature_id: u32,
    /// The point relative to body A's centre of mass when the manifold was built, in world
    /// orientation (Box3D `anchorA`).
    #[serde(default)]
    pub anchor_a: Vec3,
    /// The point relative to body B's centre of mass (Box3D `anchorB`).
    #[serde(default)]
    pub anchor_b: Vec3,
    /// The separation when the manifold was built; a recycled contact moves it with the bodies
    /// (Box3D `baseSeparation`).
    #[serde(default)]
    pub base_separation: f32,
    /// The mesh triangle that made the point; warm starting matches mesh points by it and
    /// their feature (Box3D `triangleIndex`).
    #[serde(default)]
    pub triangle_index: u32,
    /// The relative normal approach velocity before the step (Box3D `normalVelocity`).
    #[serde(default)]
    pub relative_velocity: f32,
}
impl Default for ContactPoint {
    fn default() -> Self {
        Self {
            point: Vec3::ZERO,
            normal: Vec3::ZERO,
            separation: 0.0,
            normal_impulse: 0.0,
            total_normal_impulse: 0.0,
            feature_id: 0,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            base_separation: 0.0,
            triangle_index: 0,
            relative_velocity: 0.0,
        }
    }
}

/// Maximum number of contact points supported per contact manifold (typically 4 for planar contact patches).
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// Inline fixed-capacity buffer of contact points avoiding runtime heap allocations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactPoints {
    points: [ContactPoint; MAX_MANIFOLD_POINTS],
    count: usize,
}

impl Default for ContactPoints {
    fn default() -> Self {
        Self::new()
    }
}

impl ContactPoints {
    /// Creates an empty collection of contact points.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            points: [ContactPoint {
                point: Vec3::ZERO,
                normal: Vec3::ZERO,
                separation: 0.0,
                normal_impulse: 0.0,
                total_normal_impulse: 0.0,
                feature_id: 0,
                anchor_a: Vec3::ZERO,
                anchor_b: Vec3::ZERO,
                base_separation: 0.0,
                triangle_index: 0,
                relative_velocity: 0.0,
            }; MAX_MANIFOLD_POINTS],
            count: 0,
        }
    }

    /// Creates a contact points collection containing a single point.
    #[must_use]
    pub const fn from_single(point: ContactPoint) -> Self {
        let mut points = [ContactPoint {
            point: Vec3::ZERO,
            normal: Vec3::ZERO,
            separation: 0.0,
            normal_impulse: 0.0,
            total_normal_impulse: 0.0,
            feature_id: 0,
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            base_separation: 0.0,
            triangle_index: 0,
            relative_velocity: 0.0,
        }; MAX_MANIFOLD_POINTS];
        points[0] = point;
        Self { points, count: 1 }
    }

    /// Pushes a contact point if capacity remains.
    pub fn push(&mut self, point: ContactPoint) {
        if self.count < MAX_MANIFOLD_POINTS {
            self.points[self.count] = point;
            self.count += 1;
        }
    }

    /// Clears all contact points.
    pub fn clear(&mut self) {
        self.count = 0;
    }

    /// Truncates the buffer to `new_len`.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.count {
            self.count = new_len;
        }
    }

    /// Returns the number of contact points.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns whether the buffer contains no contact points.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns a slice of the active contact points.
    #[must_use]
    pub fn as_slice(&self) -> &[ContactPoint] {
        &self.points[..self.count]
    }

    /// Returns a mutable slice of the active contact points.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [ContactPoint] {
        &mut self.points[..self.count]
    }
}

impl std::ops::Deref for ContactPoints {
    type Target = [ContactPoint];

    fn deref(&self) -> &Self::Target {
        &self.points[..self.count]
    }
}

impl std::ops::DerefMut for ContactPoints {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.points[..self.count]
    }
}

impl FromIterator<ContactPoint> for ContactPoints {
    fn from_iter<T: IntoIterator<Item = ContactPoint>>(iter: T) -> Self {
        let mut cp = Self::new();
        for p in iter {
            cp.push(p);
        }
        cp
    }
}

impl From<Vec<ContactPoint>> for ContactPoints {
    fn from(vec: Vec<ContactPoint>) -> Self {
        vec.into_iter().collect()
    }
}

impl<'a> IntoIterator for &'a ContactPoints {
    type Item = &'a ContactPoint;
    type IntoIter = std::slice::Iter<'a, ContactPoint>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut ContactPoints {
    type Item = &'a mut ContactPoint;
    type IntoIter = std::slice::IterMut<'a, ContactPoint>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

/// An owning iterator over [`ContactPoints`].
#[derive(Debug, Clone, Copy)]
pub struct ContactPointsIter {
    points: ContactPoints,
    index: usize,
}

impl Iterator for ContactPointsIter {
    type Item = ContactPoint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.points.count {
            let item = self.points.points[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.points.count - self.index;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for ContactPointsIter {}

impl IntoIterator for ContactPoints {
    type Item = ContactPoint;
    type IntoIter = ContactPointsIter;

    fn into_iter(self) -> Self::IntoIter {
        ContactPointsIter {
            points: self,
            index: 0,
        }
    }
}

/// Contact manifold describing contact points and physical interaction between two colliding bodies.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactManifold {
    /// First body index.
    pub body_a: u32,
    /// Second body index.
    pub body_b: u32,
    /// Generated contact points.
    pub points: ContactPoints,
    /// Combined friction coefficient: `sqrt(f1 * f2)`.
    pub friction: f32,
    /// Combined restitution coefficient: `max(r1, r2)`.
    pub restitution: f32,
    /// Manifold normal from body A to body B: the axis of friction, twist and rolling
    /// resistance.
    #[serde(default)]
    pub normal: Vec3,
    /// Combined rolling resistance, `max(r1, r2)` times the larger rounded radius (Box3D's
    /// contact `rollingResistance`).
    #[serde(default)]
    pub rolling_resistance: f32,
    /// Accumulated friction impulse as a world vector, for warm starting in a new tangent basis.
    #[serde(default)]
    pub friction_impulse: Vec3,
    /// Accumulated twist friction impulse about the normal.
    #[serde(default)]
    pub twist_impulse: f32,
    /// Accumulated rolling resistance impulse.
    #[serde(default)]
    pub rolling_impulse: Vec3,
    /// Whether the manifold is one of a mesh contact's, which the solver relaxes in Box3D's
    /// mesh order (twist friction before rolling resistance).
    #[serde(default)]
    pub mesh: bool,
    /// Relative surface tangent velocity (Box3D `b3Contact::tangentVelocity`).
    #[serde(default)]
    pub tangent_velocity: Vec3,
}

impl Default for ContactManifold {
    fn default() -> Self {
        Self {
            body_a: 0,
            body_b: 0,
            points: ContactPoints::new(),
            friction: 0.0,
            restitution: 0.0,
            normal: Vec3::ZERO,
            rolling_resistance: 0.0,
            friction_impulse: Vec3::ZERO,
            twist_impulse: 0.0,
            rolling_impulse: Vec3::ZERO,
            mesh: false,
            tangent_velocity: Vec3::ZERO,
        }
    }
}

impl ContactManifold {
    /// Swaps bodies A and B: the anchors swap, the normals and the accumulated vector
    /// impulses change sign, and relative surface tangent velocity is negated.
    pub fn flip(&mut self) {
        std::mem::swap(&mut self.body_a, &mut self.body_b);
        for p in &mut self.points {
            p.normal = -p.normal;
            std::mem::swap(&mut p.anchor_a, &mut p.anchor_b);
        }
        self.normal = -self.normal;
        self.friction_impulse = -self.friction_impulse;
        self.twist_impulse = -self.twist_impulse;
        self.rolling_impulse = -self.rolling_impulse;
        self.tangent_velocity = -self.tangent_velocity;
    }
}

/// Collision body descriptor combining body identifier, spatial transform, and attached collider.
#[derive(Debug, Clone, Copy)]
pub struct CollisionBody<'a> {
    /// Identifier of the rigid body.
    pub body_id: u32,
    /// World position of the rigid body.
    pub position: Vec3,
    /// Orientation quaternion of the rigid body.
    pub rotation: Quat,
    /// Attached collider.
    pub collider: &'a Collider,
}

impl<'a> CollisionBody<'a> {
    /// Creates a new collision body descriptor.
    #[must_use]
    pub const fn new(body_id: u32, position: Vec3, rotation: Quat, collider: &'a Collider) -> Self {
        Self {
            body_id,
            position,
            rotation,
            collider,
        }
    }
}

/// What a contact keeps between steps for its narrowphase (Box3D `b3ContactCache`): the GJK
/// simplex of hull–round pairs and the separating axis of hull pairs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ContactCache {
    /// The last GJK simplex.
    pub simplex: SimplexCache,
    /// The last separating or touching axis.
    pub sat: SatCache,
}

/// A convex shape as Box3D's manifold builders see it, ranked as Box3D orders a contact's
/// shapes: a hull (or box) collides as shape A against anything, a capsule against spheres.
pub(crate) enum Convex<'a> {
    Sphere(Sphere),
    Capsule(Capsule),
    Box(BoxHull),
    Hull(&'a Hull3d),
}

impl<'a> Convex<'a> {
    pub(crate) fn of(shape: &'a Shape) -> Option<Self> {
        match *shape {
            Shape::Sphere { radius } => Some(Self::Sphere(Sphere {
                center: Vec3::ZERO,
                radius,
            })),
            Shape::Capsule {
                radius,
                half_height,
            } => Some(Self::Capsule(Capsule {
                center1: Vec3::new(0.0, -half_height, 0.0),
                center2: Vec3::new(0.0, half_height, 0.0),
                radius,
            })),
            Shape::Box { half_extents } => Some(Self::Box(BoxHull::new(half_extents))),
            Shape::ConvexHull(ref hull) => Some(Self::Hull(hull)),
            _ => None,
        }
    }

    const fn rank(&self) -> u8 {
        match self {
            Self::Sphere(_) => 0,
            Self::Capsule(_) => 1,
            Self::Box(_) | Self::Hull(_) => 2,
        }
    }

    /// The hull of a box or hull.
    pub(crate) fn view(&self) -> HullView<'_> {
        match self {
            Self::Box(hull) => hull.view(),
            Self::Hull(hull) => hull.view(),
            _ => unreachable!("only hulls have a hull view"),
        }
    }
}

/// The points GJK sees of a convex shape (Box3D `b3MakeShapeProxy`): a sphere's centre, a
/// capsule's segment, a box's corners in Box3D's order, a hull's or triangle's vertices.
pub(crate) struct ProxyPoints<'a> {
    inline: [Vec3; 8],
    len: usize,
    borrowed: Option<&'a [Vec3]>,
    radius: f32,
}

impl<'a> ProxyPoints<'a> {
    pub(crate) fn of(shape: &'a Shape) -> Self {
        let mut inline = [Vec3::ZERO; 8];
        let (len, radius) = match shape {
            Shape::Sphere { radius } => (1, *radius),
            Shape::Capsule {
                radius,
                half_height,
            } => {
                inline[0] = Vec3::new(0.0, -half_height, 0.0);
                inline[1] = Vec3::new(0.0, *half_height, 0.0);
                (2, *radius)
            }
            Shape::Box { half_extents } => {
                inline = *BoxHull::new(*half_extents).points();
                (8, 0.0)
            }
            Shape::Triangle { a, b, c } => {
                inline[..3].copy_from_slice(&[*a, *b, *c]);
                (3, 0.0)
            }
            Shape::ConvexHull(hull) => {
                return Self {
                    inline,
                    len: 0,
                    borrowed: Some(hull.points()),
                    radius: 0.0,
                }
            }
            Shape::TriangleMesh(_) | Shape::HeightField(_) | Shape::Compound(_) => (1, 0.0),
        };
        Self {
            inline,
            len,
            borrowed: None,
            radius,
        }
    }

    pub(crate) fn proxy(&self) -> ShapeProxy<'_> {
        ShapeProxy {
            points: self.borrowed.unwrap_or(&self.inline[..self.len]),
            radius: self.radius,
        }
    }
}

fn transform_of(body: &CollisionBody<'_>) -> Transform {
    Transform::new(body.position, body.rotation)
}

/// Whether two shapes overlap (Box3D's `b3Overlap*` sensor tests): their GJK distance with
/// radii is below [`OVERLAP_SLOP`]. `body_b` must be convex; `body_a` may be a triangle mesh,
/// which overlaps when one of the triangles its BVH finds does.
#[must_use]
pub fn shapes_overlap(body_a: CollisionBody<'_>, body_b: CollisionBody<'_>) -> bool {
    debug_assert!(body_b.collider.shape.is_convex());
    let points_b = ProxyPoints::of(&body_b.collider.shape);
    let proxy_b = points_b.proxy();
    let xf = transform_of(&body_a).inv_mul(&transform_of(&body_b));
    let overlaps = |proxy_a: &ShapeProxy<'_>| {
        shape_distance(proxy_a, &proxy_b, &xf, true, &mut SimplexCache::default()).distance
            < OVERLAP_SLOP
    };
    match triangles_of(&body_a.collider.shape) {
        Some(triangles) => triangles_overlap(
            &triangles,
            &transform_of(&body_a),
            &proxy_b,
            &transform_of(&body_b),
        ),
        None => overlaps(&ProxyPoints::of(&body_a.collider.shape).proxy()),
    }
}

/// Collides two shapes into a contact manifold with points up to the speculative distance
/// apart, with the mixed materials (Box3D's default mixing: `sqrt` of the frictions, the
/// larger restitution, the larger rolling resistance times the larger rounded radius).
#[must_use]
pub fn collide_shapes(
    body_a: CollisionBody<'_>,
    body_b: CollisionBody<'_>,
) -> Option<ContactManifold> {
    collide_shapes_cached(body_a, body_b, &mut ContactCache::default())
}

/// [`collide_shapes`] starting from and updating a contact's `cache`.
pub fn collide_shapes_cached(
    body_a: CollisionBody<'_>,
    body_b: CollisionBody<'_>,
    cache: &mut ContactCache,
) -> Option<ContactManifold> {
    let (col_a, col_b) = (body_a.collider, body_b.collider);
    let mut manifold = collide_pair(body_a, body_b, cache)?;
    if manifold.mesh {
        let (mesh, convex) = if col_a.shape.is_triangles() {
            (col_a, col_b)
        } else {
            (col_b, col_a)
        };
        (
            manifold.friction,
            manifold.restitution,
            manifold.rolling_resistance,
        ) = mesh_materials(mesh, convex);
        return Some(manifold);
    }
    manifold.friction = (col_a.friction * col_b.friction).sqrt();
    manifold.restitution = col_a.restitution.max(col_b.restitution);
    if col_a.rolling_resistance > 0.0 || col_b.rolling_resistance > 0.0 {
        let radius = col_a
            .shape
            .rolling_radius()
            .max(col_b.shape.rolling_radius());
        manifold.rolling_resistance =
            col_a.rolling_resistance.max(col_b.rolling_resistance) * radius;
    }
    Some(manifold)
}

/// A manifold of shape A's frame `xf` in world space, between body `body_a` there and body
/// `body_b` at `position_b`.
fn world_manifold(
    local: &LocalManifold,
    xf: &Transform,
    (body_a, body_b): (u32, u32),
    position_b: Vec3,
) -> ContactManifold {
    let rotation = Mat3::from_quat(xf.q);
    let normal = rotation * local.normal;
    let mut points = ContactPoints::new();
    for p in local.points() {
        let anchor_a = rotation * p.point;
        let point = xf.p + anchor_a;
        points.push(ContactPoint {
            point,
            normal,
            separation: p.separation,
            feature_id: p.pair.id(),
            anchor_a,
            anchor_b: point - position_b,
            base_separation: p.separation,
            ..ContactPoint::default()
        });
    }
    ContactManifold {
        body_a,
        body_b,
        points,
        normal,
        ..ContactManifold::default()
    }
}

/// The manifold of a convex shape pair, before the materials are mixed; a pair with a mesh or
/// triangle gives its first mesh manifold.
/// Collides two convex shapes into a contact manifold before materials are mixed.
pub(crate) fn collide_convex_shapes(
    shape_a: &Shape,
    xf_a: Transform,
    body_a_id: u32,
    shape_b: &Shape,
    xf_b: Transform,
    body_b_id: u32,
    cache: &mut ContactCache,
) -> Option<ContactManifold> {
    let (Some(convex_a), Some(convex_b)) = (Convex::of(shape_a), Convex::of(shape_b)) else {
        return None;
    };
    // Box3D's shape A is the higher ranked one.
    let flip = convex_b.rank() > convex_a.rank();
    let (first, second, xf_first, xf_second, id_first, id_second) = if flip {
        (convex_b, convex_a, xf_b, xf_a, body_b_id, body_a_id)
    } else {
        (convex_a, convex_b, xf_a, xf_b, body_a_id, body_b_id)
    };
    let xf = xf_first.inv_mul(&xf_second);
    let local = match (&first, &second) {
        (Convex::Sphere(a), Convex::Sphere(b)) => collide_spheres(a, b, &xf),
        (Convex::Capsule(a), Convex::Sphere(b)) => collide_capsule_and_sphere(a, b, &xf),
        (Convex::Capsule(a), Convex::Capsule(b)) => collide_capsules(a, b, &xf),
        (Convex::Sphere(_) | Convex::Capsule(_), _) => {
            unreachable!("shape A is ranked at least as high as shape B")
        }
        (hull, Convex::Sphere(b)) => {
            collide_hull_and_sphere(&hull.view(), b, &xf, &mut cache.simplex)
        }
        (hull, Convex::Capsule(b)) => {
            collide_hull_and_capsule(&hull.view(), b, &xf, &mut cache.simplex)
        }
        (a, b) => collide_hulls(&a.view(), &b.view(), &xf, &mut cache.sat),
    };
    if local.count == 0 {
        return None;
    }
    let mut manifold = world_manifold(&local, &xf_first, (id_first, id_second), xf_second.p);
    if flip {
        manifold.flip();
    }
    Some(manifold)
}

fn collide_pair(
    body_a: CollisionBody<'_>,
    body_b: CollisionBody<'_>,
    cache: &mut ContactCache,
) -> Option<ContactManifold> {
    if Convex::of(&body_a.collider.shape).is_some() && Convex::of(&body_b.collider.shape).is_some()
    {
        collide_convex_shapes(
            &body_a.collider.shape,
            transform_of(&body_a),
            body_a.body_id,
            &body_b.collider.shape,
            transform_of(&body_b),
            body_b.body_id,
            cache,
        )
    } else {
        collide_with_triangles(body_a, body_b)
    }
}

/// The triangles of a mesh or triangle shape.
pub(crate) fn triangles_of(shape: &Shape) -> Option<Triangles<'_>> {
    match shape {
        Shape::TriangleMesh(mesh) => Some(Triangles::Mesh(mesh)),
        Shape::HeightField(hf) => Some(Triangles::HeightField(hf)),
        Shape::Triangle { a, b, c } => Some(Triangles::Single([*a, *b, *c])),
        _ => None,
    }
}

/// Box3D's mesh contact materials: the mixed friction and restitution, and the convex shape's
/// rolling resistance times its radius (a hull's inner radius).
pub(crate) fn mesh_materials(mesh: &Collider, convex: &Collider) -> (f32, f32, f32) {
    let radius = match &convex.shape {
        Shape::Sphere { radius } | Shape::Capsule { radius, .. } => *radius,
        Shape::Box { half_extents } => half_extents.min_element(),
        Shape::ConvexHull(hull) => hull.inner_radius(),
        _ => 0.0,
    };
    (
        (mesh.friction * convex.friction).sqrt(),
        mesh.restitution.max(convex.restitution),
        convex.rolling_resistance * radius,
    )
}

/// A mesh or triangle against a convex shape, without a contact's memory: the first manifold.
/// Meshes and triangles do not collide with each other, as in Box3D.
fn collide_with_triangles(
    body_a: CollisionBody<'_>,
    body_b: CollisionBody<'_>,
) -> Option<ContactManifold> {
    let (mesh_body, convex_body, flip) = match (
        triangles_of(&body_a.collider.shape),
        triangles_of(&body_b.collider.shape),
    ) {
        (Some(_), None) => (body_a, body_b, false),
        (None, Some(_)) => (body_b, body_a, true),
        _ => return None,
    };
    let triangles = triangles_of(&mesh_body.collider.shape)?;
    let convex = Convex::of(&convex_body.collider.shape)?;
    let pair = MeshPair {
        triangles,
        xf_mesh: transform_of(&mesh_body),
        convex: &convex,
        xf_convex: transform_of(&convex_body),
        convex_bounds: convex_body
            .collider
            .shape
            .compute_aabb(convex_body.position, convex_body.rotation),
        is_fast: false,
    };
    let mut manifolds = Vec::new();
    if !update_mesh_manifolds(
        &mut manifolds,
        &mut MeshContactCache::default(),
        &pair,
        (mesh_body.body_id, convex_body.body_id),
        flip,
        &mut MeshScratch::default(),
    ) {
        return None;
    }
    let mut manifold = manifolds[0];
    for p in &mut manifold.points {
        p.anchor_a = p.point - body_a.position;
        p.anchor_b = p.point - body_b.position;
    }
    Some(manifold)
}

/// Box3D `b3OverlapMesh`: whether a triangle of `triangles` (in the frame `xf_mesh`) is within
/// a tenth of a linear slop of `proxy` (in the frame `xf_proxy`).
fn triangles_overlap(
    triangles: &Triangles<'_>,
    xf_mesh: &Transform,
    proxy: &ShapeProxy<'_>,
    xf_proxy: &Transform,
) -> bool {
    let xf = xf_mesh.inv_mul(xf_proxy);
    let (lower, upper) = proxy.points.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(-f32::MAX)),
        |(lo, hi), &p| {
            let p = xf.apply(p);
            (lo.min(p), hi.max(p))
        },
    );
    let radius = Vec3::splat(proxy.radius);
    let (lower, upper) = (lower - radius, upper + radius);
    let overlaps = |corners: [Vec3; 3]| {
        let triangle = ShapeProxy {
            points: &corners,
            radius: 0.0,
        };
        shape_distance(&triangle, proxy, &xf, true, &mut SimplexCache::default()).distance
            < 0.1 * crate::contact_solver::LINEAR_SLOP
    };
    match triangles {
        Triangles::Mesh(mesh) => {
            let mut found = false;
            mesh.query(lower, upper, |t| {
                found = overlaps(mesh.triangle(t).vertices);
                !found
            });
            found
        }
        Triangles::HeightField(hf) => {
            let mut found = false;
            hf.query(lower, upper, |t| {
                found = overlaps(hf.triangle(t).vertices);
                !found
            });
            found
        }
        Triangles::Single(corners) => overlaps(*corners),
    }
}

/// Box3D `b3ShapeTimeOfImpact`: the fraction of the sweeps at which convex `shape_b` first
/// comes within a linear slop of `shape_a`, below `max_fraction`, or `max_fraction` when it
/// does not; zero when they start overlapped. A mesh or triangle `shape_a`, assumed static, is
/// swept triangle by triangle over those its BVH finds in the swept bounds, from the front
/// only, with Box3D's early out for shapes finishing in front and its fallback: a triangle the
/// shape starts overlapping is swept against a small sphere at the shape's centroid, of half
/// its smallest extent.
pub(crate) fn shape_time_of_impact(
    shape_a: &Shape,
    sweep_a: &Sweep,
    shape_b: &Shape,
    sweep_b: &Sweep,
    max_fraction: f32,
    is_sensor: bool,
) -> f32 {
    let points_b = ProxyPoints::of(shape_b);
    let proxy_b = points_b.proxy();
    let Some(triangles) = triangles_of(shape_a) else {
        let points_a = ProxyPoints::of(shape_a);
        return time_of_impact(&points_a.proxy(), sweep_a, &proxy_b, sweep_b, max_fraction)
            .fraction;
    };

    let xf_a = sweep_a.transform(0.0);
    let centroid = match shape_b {
        Shape::ConvexHull(hull) => hull.center(),
        _ => Vec3::ZERO,
    };
    let xf_b1 = sweep_b.transform(0.0);
    let centroid1 = xf_a.apply_inverse(xf_b1.apply(centroid));
    let centroid2 = xf_a.apply_inverse(sweep_b.transform(1.0).apply(centroid));
    let fallback_radius = (0.5 * shape_b.extents().0).max(crate::contact_solver::LINEAR_SLOP);
    // The swept bounds in the mesh's frame.
    let xf_b2 = sweep_b.transform(max_fraction);
    let bounds1 = shape_b.compute_aabb(xf_b1.p, xf_b1.q);
    let bounds2 = shape_b.compute_aabb(xf_b2.p, xf_b2.q);
    let (lower, upper) = (bounds1.min.min(bounds2.min), bounds1.max.max(bounds2.max));
    let inverse = xf_a.inverse();
    let center = inverse.apply((lower + upper) * 0.5);
    let r = Mat3::from_quat(inverse.q);
    let half = (upper - lower) * 0.5;
    let extent = (r.x_axis * half.x).abs() + (r.y_axis * half.y).abs() + (r.z_axis * half.z).abs();

    let mut best = max_fraction;
    let mut visit = |[a, b, c]: [Vec3; 3]| {
        let n = normalize((b - a).cross(c - a));
        let (offset1, offset2) = (n.dot(centroid1 - a), n.dot(centroid2 - a));
        if offset1 < 0.0 {
            // Started behind.
            return;
        }
        if !is_sensor && offset1 - offset2 < fallback_radius && offset2 > fallback_radius {
            // Finished in front.
            return;
        }
        let corners = [a, b, c];
        let proxy_a = ShapeProxy {
            points: &corners,
            radius: 0.0,
        };
        let fraction = time_of_impact(&proxy_a, sweep_a, &proxy_b, sweep_b, best).fraction;
        if 0.0 < fraction && fraction < best {
            best = fraction;
        } else if fraction == 0.0 {
            let center = [centroid];
            let fallback = ShapeProxy {
                points: &center,
                radius: fallback_radius + crate::contact_solver::LINEAR_SLOP,
            };
            let fraction = time_of_impact(&proxy_a, sweep_a, &fallback, sweep_b, best).fraction;
            if 0.0 < fraction && fraction < best {
                best = fraction;
            }
        }
    };
    match triangles {
        Triangles::Mesh(mesh) => mesh.query(center - extent, center + extent, |t| {
            visit(mesh.triangle(t).vertices);
            true
        }),
        Triangles::HeightField(hf) => hf.query(center - extent, center + extent, |t| {
            visit(hf.triangle(t).vertices);
            true
        }),
        Triangles::Single(corners) => visit(corners),
    }
    best
}
