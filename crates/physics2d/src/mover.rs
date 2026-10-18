//! Character mover helpers (Box2D v3.2 `mover.c`): collision planes from
//! [`PhysicsWorld2d::collide_mover`](crate::world::PhysicsWorld2d::collide_mover), a solver that
//! finds the translation the planes allow, and velocity clipping against them.
//!
//! A kinematic character typically casts its capsule along its intended motion
//! ([`PhysicsWorld2d::cast_mover`](crate::world::PhysicsWorld2d::cast_mover)), moves by the
//! allowed fraction, gathers the planes of what it touches, solves them for the final
//! translation and clips its velocity so it slides along walls and floors.

use crate::distance::LINEAR_SLOP;
use crate::math::clamp;
use korppi_math::Vec2;

/// A plane: the points `p` with `normal · p = offset` (Box2D `b2Plane`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Plane2d {
    /// Unit normal.
    pub normal: Vec2,
    /// Distance of the plane from the origin along the normal.
    pub offset: f32,
}

impl Plane2d {
    /// The plane of `normal` and `offset`.
    #[must_use]
    pub const fn new(normal: Vec2, offset: f32) -> Self {
        Self { normal, offset }
    }

    /// How far `point` is in front of the plane (Box2D `b2PlaneSeparation`).
    #[must_use]
    pub fn separation(&self, point: Vec2) -> f32 {
        self.normal.dot(point) - self.offset
    }
}

/// A mover touching a shape (Box2D `b2PlaneResult`): the plane that pushes the mover out, its
/// offset being how deep the mover is, and the touching point on the shape.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlaneResult2d {
    /// The collision plane, normal pointing at the mover.
    pub plane: Plane2d,
    /// The closest point on the shape.
    pub point: Vec2,
}

/// A plane for [`solve_planes`] (Box2D `b2CollisionPlane`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionPlane2d {
    /// The plane.
    pub plane: Plane2d,
    /// Largest push the plane gives, `f32::MAX` for a rigid plane, less for a soft one.
    pub push_limit: f32,
    /// The push [`solve_planes`] found.
    pub push: f32,
    /// Whether [`clip_vector`] clips against the plane; usually off for soft planes.
    pub clip_velocity: bool,
}

impl CollisionPlane2d {
    /// A plane that pushes at most `push_limit`.
    #[must_use]
    pub const fn new(plane: Plane2d, push_limit: f32, clip_velocity: bool) -> Self {
        Self {
            plane,
            push_limit,
            push: 0.0,
            clip_velocity,
        }
    }
}

/// The result of [`solve_planes`] (Box2D `b2PlaneSolverResult`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlaneSolverResult2d {
    /// The translation the planes allow.
    pub delta: Vec2,
    /// Iterations the solver took.
    pub iteration_count: u32,
}

/// The translation closest to `target_delta` that the planes allow, from the position the
/// planes were gathered at (Box2D `b2SolvePlanes`): each plane pushes the translation out of
/// itself, up to its push limit, until the pushes settle within a linear slop. Sets each
/// plane's `push`.
pub fn solve_planes(target_delta: Vec2, planes: &mut [CollisionPlane2d]) -> PlaneSolverResult2d {
    for plane in planes.iter_mut() {
        plane.push = 0.0;
    }
    let mut delta = target_delta;
    let mut iteration = 0;
    while iteration < 20 {
        let mut total_push = 0.0f32;
        for plane in planes.iter_mut() {
            // A slop of separation keeps the mover from jittering.
            let push = -(plane.plane.separation(delta) + LINEAR_SLOP);
            let accumulated = plane.push;
            plane.push = clamp(plane.push + push, 0.0, plane.push_limit);
            let push = plane.push - accumulated;
            delta = Vec2::new(
                delta.x + push * plane.plane.normal.x,
                delta.y + push * plane.plane.normal.y,
            );
            total_push += push.abs();
        }
        if total_push < LINEAR_SLOP {
            break;
        }
        iteration += 1;
    }
    PlaneSolverResult2d {
        delta,
        iteration_count: iteration,
    }
}

/// `vector` without its components into the planes that pushed and clip velocity (Box2D
/// `b2ClipVector`), so a mover slides along them.
#[must_use]
pub fn clip_vector(vector: Vec2, planes: &[CollisionPlane2d]) -> Vec2 {
    let mut v = vector;
    for plane in planes {
        if plane.push == 0.0 || !plane.clip_velocity {
            continue;
        }
        let into = vector_dot_min(v, plane.plane.normal);
        v = Vec2::new(
            v.x - into * plane.plane.normal.x,
            v.y - into * plane.plane.normal.y,
        );
    }
    v
}

/// `min(0, v · n)`, the component of `v` into a plane of normal `n`.
fn vector_dot_min(v: Vec2, n: Vec2) -> f32 {
    v.dot(n).min(0.0)
}
