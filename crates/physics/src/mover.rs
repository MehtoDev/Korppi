//! Character mover collision and plane solver (Box3D `mover.c`).

use crate::distance::{
    closest_point_on_segment, segment_distance, shape_distance, ShapeProxy, SimplexCache,
};
use crate::math::Transform;
use crate::shape::Shape;
use korppi_math::{Aabb, Vec3};
use serde::{Deserialize, Serialize};

/// A 3D directed plane defined by normal and offset (Box3D `b3Plane`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Plane3d {
    /// Outward unit normal of the plane.
    pub normal: Vec3,
    /// Distance from the origin along `normal` or signed penetration offset.
    pub offset: f32,
}

impl Plane3d {
    /// Constructs a plane from normal and offset.
    pub const fn new(normal: Vec3, offset: f32) -> Self {
        Self { normal, offset }
    }

    /// Signed separation of a point from the plane (Box3D `b3PlaneSeparation`).
    ///
    /// Positive when in front of the plane, negative when behind.
    #[inline]
    pub fn separation(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.offset
    }
}

/// A plane used in character mover velocity resolution and clamping (Box3D `b3CollisionPlane`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CollisionPlane3d {
    /// The geometric plane.
    pub plane: Plane3d,
    /// Maximum allowed push magnitude for this plane.
    pub push_limit: f32,
    /// Accumulated push applied by the plane solver.
    pub push: f32,
    /// Whether this plane clips the character mover's velocity.
    pub clip_velocity: bool,
}

impl CollisionPlane3d {
    /// Constructs a collision plane with initial push of 0.
    pub const fn new(plane: Plane3d, push_limit: f32, clip_velocity: bool) -> Self {
        Self {
            plane,
            push_limit,
            push: 0.0,
            clip_velocity,
        }
    }
}

/// Result of a mover collision against a shape (Box3D `b3PlaneResult`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlaneResult3d {
    /// The resulting collision plane.
    pub plane: Plane3d,
    /// Contact point on the surface of the colliding shape.
    pub point: Vec3,
    /// Triangle index if the shape is a mesh or height field, otherwise 0.
    pub triangle_index: u32,
    /// Material index on the hit shape.
    pub material_index: u32,
}

/// Result of solving collision planes against a target motion delta (Box3D `b3PlaneSolverResult`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneSolverResult3d {
    /// Resolved displacement vector avoiding plane penetrations.
    pub delta: Vec3,
    /// Iteration count performed by the solver.
    pub iteration_count: u32,
}

/// A capsule mover used for character movement queries (Box3D `b3Capsule`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapsuleMover {
    /// Start point of the central segment.
    pub center1: Vec3,
    /// End point of the central segment.
    pub center2: Vec3,
    /// Capsule radius.
    pub radius: f32,
}

impl CapsuleMover {
    /// Constructs a capsule mover.
    pub const fn new(center1: Vec3, center2: Vec3, radius: f32) -> Self {
        Self {
            center1,
            center2,
            radius,
        }
    }

    /// Computes the axis-aligned bounding box enclosing this capsule.
    #[must_use]
    pub fn compute_aabb(&self) -> Aabb {
        let r = Vec3::splat(self.radius);
        Aabb {
            min: self.center1.min(self.center2) - r,
            max: self.center1.max(self.center2) + r,
        }
    }
}

/// Computes an arbitrary unit vector perpendicular to `a` (Box3D `b3Perp`).
#[must_use]
pub fn perp(a: Vec3) -> Vec3 {
    let p = if a.x < -0.5 || a.x > 0.5 {
        Vec3::new(a.y, -a.x, 0.0)
    } else {
        Vec3::new(0.0, a.z, -a.y)
    };
    p.normalize()
}

/// Solves character displacement against collision planes (Box3D `b3SolvePlanes`).
pub fn solve_planes(target_delta: Vec3, planes: &mut [CollisionPlane3d]) -> PlaneSolverResult3d {
    for plane in planes.iter_mut() {
        plane.push = 0.0;
    }

    let mut delta = target_delta;
    let tolerance = crate::contact_solver::LINEAR_SLOP;

    let mut iteration = 0;
    while iteration < 20 {
        let mut total_push = 0.0f32;
        for plane in planes.iter_mut() {
            // Add slop to prevent jitter (Box3D b3SolvePlanes)
            let separation = plane.plane.separation(delta) + crate::contact_solver::LINEAR_SLOP;
            let push = -separation;

            let accumulated_push = plane.push;
            plane.push = (plane.push + push).clamp(0.0, plane.push_limit);
            let applied_push = plane.push - accumulated_push;
            delta += plane.plane.normal * applied_push;

            total_push += applied_push.abs();
        }

        if total_push < tolerance {
            break;
        }
        iteration += 1;
    }

    PlaneSolverResult3d {
        delta,
        iteration_count: iteration,
    }
}

/// Clips a velocity vector against active pushing planes (Box3D `b3ClipVector`).
#[must_use]
pub fn clip_vector(vector: Vec3, planes: &[CollisionPlane3d]) -> Vec3 {
    let mut v = vector;
    for plane in planes {
        if plane.push == 0.0 || !plane.clip_velocity {
            continue;
        }
        let d = v.dot(plane.plane.normal).min(0.0);
        v -= plane.plane.normal * d;
    }
    v
}

/// Collides a capsule mover with a sphere in local coordinates (Box3D `b3CollideMoverAndSphere`).
#[must_use]
pub fn collide_mover_and_sphere(
    sphere_radius: f32,
    sphere_center: Vec3,
    mover: &CapsuleMover,
) -> Option<PlaneResult3d> {
    let total_radius = mover.radius + sphere_radius;
    let closest = closest_point_on_segment(mover.center1, mover.center2, sphere_center);
    let diff = closest - sphere_center;
    let len = diff.length();

    if len > total_radius {
        return None;
    }

    let linear_slop = crate::contact_solver::LINEAR_SLOP;
    let (normal, distance) = if len < linear_slop {
        let mover_axis = mover.center2 - mover.center1;
        let mover_len = mover_axis.length();
        let n = if mover_len > linear_slop {
            perp(mover_axis / mover_len)
        } else {
            Vec3::Y
        };
        (n, 0.0)
    } else {
        (diff / len, len)
    };

    Some(PlaneResult3d {
        plane: Plane3d::new(normal, total_radius - distance),
        point: sphere_center,
        triangle_index: 0,
        material_index: 0,
    })
}

/// Collides a capsule mover with another capsule in local coordinates (Box3D `b3CollideMoverAndCapsule`).
#[must_use]
pub fn collide_mover_and_capsule(
    cap_center1: Vec3,
    cap_center2: Vec3,
    cap_radius: f32,
    mover: &CapsuleMover,
) -> Option<PlaneResult3d> {
    let total_radius = mover.radius + cap_radius;
    let approach = segment_distance(cap_center1, cap_center2, mover.center1, mover.center2);
    let diff = approach.point2 - approach.point1;
    let len = diff.length();

    if len > total_radius {
        return None;
    }

    let linear_slop = crate::contact_solver::LINEAR_SLOP;
    let (normal, distance) = if len < linear_slop {
        let mover_axis = mover.center2 - mover.center1;
        let mover_len = mover_axis.length();
        let n = if mover_len > linear_slop {
            perp(mover_axis / mover_len)
        } else {
            Vec3::Y
        };
        (n, 0.0)
    } else {
        (diff / len, len)
    };

    Some(PlaneResult3d {
        plane: Plane3d::new(normal, total_radius - distance),
        point: approach.point1,
        triangle_index: 0,
        material_index: 0,
    })
}

/// Collides a capsule mover with a convex hull in local coordinates (Box3D `b3CollideMoverAndHull`).
#[must_use]
pub fn collide_mover_and_hull(hull_points: &[Vec3], mover: &CapsuleMover) -> Option<PlaneResult3d> {
    let proxy_a = ShapeProxy {
        points: hull_points,
        radius: 0.0,
    };
    let mover_points = [mover.center1, mover.center2];
    let proxy_b = ShapeProxy {
        points: &mover_points,
        radius: mover.radius,
    };

    let mut cache = SimplexCache::default();
    let distance_output =
        shape_distance(&proxy_a, &proxy_b, &Transform::IDENTITY, false, &mut cache);

    if distance_output.distance == 0.0 {
        return None;
    }

    if distance_output.distance <= mover.radius {
        Some(PlaneResult3d {
            plane: Plane3d::new(
                distance_output.normal,
                mover.radius - distance_output.distance,
            ),
            point: distance_output.point_a,
            triangle_index: 0,
            material_index: 0,
        })
    } else {
        None
    }
}

/// Collides a shape placed at `transform` against `mover` in world space (Box3D `b3CollideMover`).
pub fn collide_mover_shape(
    shape: &Shape,
    transform: &Transform,
    mover: &CapsuleMover,
    mut callback: impl FnMut(PlaneResult3d) -> bool,
) -> bool {
    collide_mover_shape_dyn(shape, transform, mover, &mut callback)
}

fn collide_mover_shape_dyn(
    shape: &Shape,
    transform: &Transform,
    mover: &CapsuleMover,
    callback: &mut dyn FnMut(PlaneResult3d) -> bool,
) -> bool {
    let inv_xf = transform.inverse();
    let local_mover = CapsuleMover {
        center1: inv_xf.apply(mover.center1),
        center2: inv_xf.apply(mover.center2),
        radius: mover.radius,
    };

    let to_world = |local_res: PlaneResult3d| -> PlaneResult3d {
        let world_normal = transform.q * local_res.plane.normal;
        let world_point = transform.apply(local_res.point);
        PlaneResult3d {
            plane: Plane3d::new(world_normal, local_res.plane.offset),
            point: world_point,
            triangle_index: local_res.triangle_index,
            material_index: local_res.material_index,
        }
    };

    match shape {
        Shape::Sphere { radius } => {
            if let Some(res) = collide_mover_and_sphere(*radius, Vec3::ZERO, &local_mover) {
                if !callback(to_world(res)) {
                    return false;
                }
            }
        }
        Shape::Capsule {
            radius,
            half_height,
        } => {
            if let Some(res) = collide_mover_and_capsule(
                Vec3::new(0.0, -half_height, 0.0),
                Vec3::new(0.0, *half_height, 0.0),
                *radius,
                &local_mover,
            ) {
                if !callback(to_world(res)) {
                    return false;
                }
            }
        }
        Shape::Box { half_extents } => {
            let h = *half_extents;
            let corners = [
                Vec3::new(-h.x, -h.y, -h.z),
                Vec3::new(h.x, -h.y, -h.z),
                Vec3::new(h.x, h.y, -h.z),
                Vec3::new(-h.x, h.y, -h.z),
                Vec3::new(-h.x, -h.y, h.z),
                Vec3::new(h.x, -h.y, h.z),
                Vec3::new(h.x, h.y, h.z),
                Vec3::new(-h.x, h.y, h.z),
            ];
            if let Some(res) = collide_mover_and_hull(&corners, &local_mover) {
                if !callback(to_world(res)) {
                    return false;
                }
            }
        }
        Shape::ConvexHull(hull) => {
            if let Some(res) = collide_mover_and_hull(hull.points(), &local_mover) {
                if !callback(to_world(res)) {
                    return false;
                }
            }
        }
        Shape::Triangle { a, b, c } => {
            let center = (local_mover.center1 + local_mover.center2) * 0.5;
            let signed_vol = (*b - *a).cross(*c - *a).dot(center - *a);
            if signed_vol >= 0.0 {
                let tri_points = [*a, *b, *c];
                let proxy_a = ShapeProxy {
                    points: &tri_points,
                    radius: 0.0,
                };
                let mover_points = [local_mover.center1, local_mover.center2];
                let proxy_b = ShapeProxy {
                    points: &mover_points,
                    radius: local_mover.radius,
                };
                let mut cache = SimplexCache::default();
                let dist_out =
                    shape_distance(&proxy_a, &proxy_b, &Transform::IDENTITY, false, &mut cache);
                if dist_out.distance > 0.0 && dist_out.distance <= local_mover.radius {
                    let res = PlaneResult3d {
                        plane: Plane3d::new(
                            dist_out.normal,
                            local_mover.radius - dist_out.distance,
                        ),
                        point: dist_out.point_a,
                        triangle_index: 0,
                        material_index: 0,
                    };
                    if !callback(to_world(res)) {
                        return false;
                    }
                }
            }
        }
        Shape::TriangleMesh(mesh) => {
            let mover_aabb = local_mover.compute_aabb();
            let center = (local_mover.center1 + local_mover.center2) * 0.5;
            let vertices = mesh.vertices();
            let mover_points = [local_mover.center1, local_mover.center2];
            let proxy_b = ShapeProxy {
                points: &mover_points,
                radius: local_mover.radius,
            };

            for (tri_idx, tri) in mesh.triangles().enumerate() {
                let v1 = vertices[tri.indices[0] as usize];
                let v2 = vertices[tri.indices[1] as usize];
                let v3 = vertices[tri.indices[2] as usize];
                let tri_min = v1.min(v2).min(v3);
                let tri_max = v1.max(v2).max(v3);
                let tri_aabb = Aabb {
                    min: tri_min,
                    max: tri_max,
                };
                if !tri_aabb.intersects(&mover_aabb) {
                    continue;
                }

                let signed_vol = (v2 - v1).cross(v3 - v1).dot(center - v1);
                if signed_vol < 0.0 {
                    continue;
                }

                let tri_points = [v1, v2, v3];
                let proxy_a = ShapeProxy {
                    points: &tri_points,
                    radius: 0.0,
                };
                let mut cache = SimplexCache::default();
                let dist_out =
                    shape_distance(&proxy_a, &proxy_b, &Transform::IDENTITY, false, &mut cache);
                if dist_out.distance > 0.0 && dist_out.distance <= local_mover.radius {
                    let res = PlaneResult3d {
                        plane: Plane3d::new(
                            dist_out.normal,
                            local_mover.radius - dist_out.distance,
                        ),
                        point: dist_out.point_a,
                        triangle_index: tri_idx as u32,
                        material_index: 0,
                    };
                    if !callback(to_world(res)) {
                        return false;
                    }
                }
            }
        }
        Shape::Compound(comp) => {
            for child in comp.children() {
                let child_xf = Transform::new(child.position, child.rotation);
                let child_world_xf =
                    Transform::new(transform.apply(child_xf.p), transform.q * child_xf.q);
                if !collide_mover_shape_dyn(&child.shape, &child_world_xf, mover, callback) {
                    return false;
                }
            }
        }
        Shape::HeightField(_) => {}
    }

    true
}
