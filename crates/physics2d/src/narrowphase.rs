//! 2D narrowphase: world-space contact manifolds of shape pairs, and time of impact.

use crate::distance::{
    time_of_impact, ShapeProxy2d, SimplexCache2d, Sweep2d, ToiInput2d, ToiState2d, LINEAR_SLOP,
};
use crate::manifold;
use crate::shape::Shape2d;
use korppi_math::{Rot2, Transform2d, Vec2};
use serde::{Deserialize, Serialize};

/// A single contact point in a 2D collision manifold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ManifoldPoint2d {
    /// World-space contact position.
    pub point: Vec2,
    /// Contact position relative to body A, in world orientation: to its origin in a manifold
    /// from [`collide_shapes`], to its centre of mass in a world's contacts (as Box2D's
    /// `b2UpdateContact` shifts them). The solver uses the anchors rather than `point`, so
    /// contacts keep their precision far from the origin.
    pub anchor_a: Vec2,
    /// Contact position relative to body B, in world orientation, like `anchor_a`.
    pub anchor_b: Vec2,
    /// Separation distance along the contact normal (negative for overlap/penetration).
    pub separation: f32,
    /// The separation when the manifold was last computed (Box2D `baseSeparation`); a recycled
    /// contact updates `separation` from it and the bodies' motion since.
    pub base_separation: f32,
    /// Accumulated normal impulse for solver warm starting.
    pub normal_impulse: f32,
    /// Accumulated tangent impulse for solver warm starting.
    pub tangent_impulse: f32,
    /// Normal impulse applied over the whole last step, warm start and restitution included
    /// (Box2D `totalNormalImpulse`); positive when the point actually pushed.
    pub total_normal_impulse: f32,
    /// Relative normal velocity before the last step's solve, negative when approaching (Box2D
    /// `normalVelocity`).
    pub normal_velocity: f32,
    /// Geometric feature identifier for contact persistence and warm-starting matching.
    pub id: u16,
    /// Indicates whether this contact point was matched with a point from the prior step.
    pub persisted: bool,
}

impl Default for ManifoldPoint2d {
    fn default() -> Self {
        Self {
            point: Vec2::ZERO,
            anchor_a: Vec2::ZERO,
            anchor_b: Vec2::ZERO,
            separation: 0.0,
            base_separation: 0.0,
            normal_impulse: 0.0,
            tangent_impulse: 0.0,
            total_normal_impulse: 0.0,
            normal_velocity: 0.0,
            id: 0,
            persisted: false,
        }
    }
}

/// A 2D contact manifold containing up to 2 contact points and a contact normal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactManifold2d {
    /// Contact points.
    pub points: [ManifoldPoint2d; 2],
    /// Number of valid contact points (1 or 2).
    pub point_count: usize,
    /// Contact normal pointing from Shape A to Shape B.
    pub normal: Vec2,
    /// Accumulated rolling resistance impulse, for warm starting (Box2D `rollingImpulse`).
    pub rolling_impulse: f32,
}

impl Default for ContactManifold2d {
    fn default() -> Self {
        Self {
            points: [ManifoldPoint2d::default(), ManifoldPoint2d::default()],
            point_count: 0,
            normal: Vec2::Y,
            rolling_impulse: 0.0,
        }
    }
}

/// The contact manifold of two shapes in world space, or `None` when they are further apart
/// than the speculative distance. The normal points from shape A to shape B.
///
/// Pairs use the Box2D ports in [`crate::manifold`]: pairs Box2D only has one way round are
/// swapped and the normal flipped back. Segments and chain segments do not collide with each
/// other, as in Box2D.
#[must_use]
pub fn collide_shapes(
    shape_a: &Shape2d,
    xf_a: &Transform2d,
    shape_b: &Shape2d,
    xf_b: &Transform2d,
) -> Option<ContactManifold2d> {
    collide_shapes_with_slop(shape_a, xf_a, shape_b, xf_b, LINEAR_SLOP)
}

/// [`collide_shapes`] for a world whose linear slop is `linear_slop`
/// ([`WorldConfig2d::linear_slop`](crate::config::WorldConfig2d::linear_slop)).
#[must_use]
pub fn collide_shapes_with_slop(
    shape_a: &Shape2d,
    xf_a: &Transform2d,
    shape_b: &Shape2d,
    xf_b: &Transform2d,
    linear_slop: f32,
) -> Option<ContactManifold2d> {
    collide_shapes_cached(
        shape_a,
        xf_a,
        shape_b,
        xf_b,
        linear_slop,
        &mut SimplexCache2d::default(),
    )
}

/// [`collide_shapes_with_slop`] with a contact's simplex cache, which chain segments warm start
/// their distance query from (Box2D `b2ContactSim::cache`).
#[must_use]
pub fn collide_shapes_cached(
    shape_a: &Shape2d,
    xf_a: &Transform2d,
    shape_b: &Shape2d,
    xf_b: &Transform2d,
    linear_slop: f32,
    cache: &mut SimplexCache2d,
) -> Option<ContactManifold2d> {
    collide_convex(shape_a, xf_a, shape_b, xf_b, linear_slop, cache)
}

/// Collides two convex shapes and converts the manifold to world space.
fn collide_convex(
    shape_a: &Shape2d,
    xf_a: &Transform2d,
    shape_b: &Shape2d,
    xf_b: &Transform2d,
    linear_slop: f32,
    cache: &mut SimplexCache2d,
) -> Option<ContactManifold2d> {
    use Shape2d::{Capsule, ChainSegment, Circle, Polygon, Segment};
    // Box2D's contact registry has each pair one way round; collide the other order swapped.
    let flipped = matches!(
        (shape_a, shape_b),
        (
            Circle(_),
            Capsule(_) | Polygon(_) | Segment(_) | ChainSegment(_)
        ) | (Capsule(_), Polygon(_) | Segment(_) | ChainSegment(_))
            | (Polygon(_), Segment(_) | ChainSegment(_))
    );
    let (first, xf_first, second, xf_second) = if flipped {
        (shape_b, xf_b, shape_a, xf_a)
    } else {
        (shape_a, xf_a, shape_b, xf_b)
    };
    let xf = xf_first.inv_mul(xf_second);
    let local = match (first, second) {
        (Circle(a), Circle(b)) => manifold::collide_circles(a, b, &xf, linear_slop),
        (Capsule(a), Circle(b)) => manifold::collide_capsule_and_circle(a, b, &xf, linear_slop),
        (Capsule(a), Capsule(b)) => manifold::collide_capsules(a, b, &xf, linear_slop),
        (Polygon(a), Circle(b)) => manifold::collide_polygon_and_circle(a, b, &xf, linear_slop),
        (Polygon(a), Capsule(b)) => manifold::collide_polygon_and_capsule(a, b, &xf, linear_slop),
        (Polygon(a), Polygon(b)) => manifold::collide_polygons(a, b, &xf, linear_slop),
        (Segment(a), Circle(b)) => manifold::collide_segment_and_circle(a, b, &xf, linear_slop),
        (Segment(a), Capsule(b)) => manifold::collide_segment_and_capsule(a, b, &xf, linear_slop),
        (Segment(a), Polygon(b)) => manifold::collide_segment_and_polygon(a, b, &xf, linear_slop),
        (ChainSegment(a), Circle(b)) => {
            manifold::collide_chain_segment_and_circle(a, b, &xf, linear_slop)
        }
        (ChainSegment(a), Capsule(b)) => {
            manifold::collide_chain_segment_and_capsule(a, b, &xf, cache, linear_slop)
        }
        (ChainSegment(a), Polygon(b)) => {
            manifold::collide_chain_segment_and_polygon(a, b, &xf, cache, linear_slop)
        }
        // Segments and chain segments do not collide with each other.
        _ => return None,
    };
    if local.point_count == 0 {
        return None;
    }
    // As Box2D's `b2UpdateContact`: anchors relative to each body's origin, never through
    // world coordinates.
    let origin_delta = xf_first.translation - xf_second.translation;
    let sign = if flipped { -1.0 } else { 1.0 };
    let mut world = ContactManifold2d {
        normal: xf_first.rotation.rotate(local.normal) * sign,
        point_count: local.point_count,
        ..ContactManifold2d::default()
    };
    for (point, local_point) in world
        .points
        .iter_mut()
        .zip(&local.points[..local.point_count])
    {
        let anchor_first = xf_first.rotation.rotate(local_point.point);
        let anchor_second = anchor_first + origin_delta;
        let (anchor_a, anchor_b) = if flipped {
            (anchor_second, anchor_first)
        } else {
            (anchor_first, anchor_second)
        };
        *point = ManifoldPoint2d {
            point: xf_first.translation + anchor_first,
            anchor_a,
            anchor_b,
            separation: local_point.separation,
            id: local_point.id,
            ..ManifoldPoint2d::default()
        };
    }
    Some(world)
}

/// Time of impact of two shapes moving over one interval, as in Box2D's continuous collision:
/// a convenience over [`time_of_impact`] for bodies whose origins move linearly and that turn
/// about their origins.
#[derive(Debug, Clone, Copy)]
pub struct TimeOfImpact2d;

impl TimeOfImpact2d {
    /// The fraction of the interval at which shape A, moving by `velocity_a`, first touches
    /// shape B, moving by `velocity_b`; `None` when they stay apart. Velocities are the
    /// displacements over the whole interval.
    #[must_use]
    pub fn calculate(
        shape_a: &Shape2d,
        xf_a: &Transform2d,
        velocity_a: Vec2,
        shape_b: &Shape2d,
        xf_b: &Transform2d,
        velocity_b: Vec2,
    ) -> Option<f32> {
        Self::calculate_full(
            shape_a, xf_a, velocity_a, 0.0, shape_b, xf_b, velocity_b, 0.0,
        )
        .map(|(t, _)| t)
    }

    /// Like [`Self::calculate`], with each shape also turning by `delta_angle_*` radians about
    /// its origin over the interval. Returns the fraction and the world normal from A toward B
    /// at the impact (zero if the shapes overlap at the start, which reports fraction 0).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_full(
        shape_a: &Shape2d,
        xf_a: &Transform2d,
        velocity_a: Vec2,
        delta_angle_a: f32,
        shape_b: &Shape2d,
        xf_b: &Transform2d,
        velocity_b: Vec2,
        delta_angle_b: f32,
    ) -> Option<(f32, Vec2)> {
        let sweep = |xf: &Transform2d, velocity: Vec2, delta_angle: f32| Sweep2d {
            local_center: Vec2::ZERO,
            c1: xf.translation,
            c2: xf.translation + velocity,
            q1: xf.rotation,
            q2: xf.rotation * Rot2::from_angle(delta_angle),
        };
        let output = time_of_impact(&ToiInput2d {
            proxy_a: ShapeProxy2d::of_shape(shape_a),
            proxy_b: ShapeProxy2d::of_shape(shape_b),
            sweep_a: sweep(xf_a, velocity_a, delta_angle_a),
            sweep_b: sweep(xf_b, velocity_b, delta_angle_b),
            max_fraction: 1.0,
            linear_slop: LINEAR_SLOP,
        });
        (output.fraction < 1.0 || output.state == ToiState2d::Overlapped)
            .then_some((output.fraction, output.normal))
    }
}
