//! Contacts begin within Box3D's speculative distance, so approaching shapes stop at the
//! surface (P3D-06), and boxes and capsules get Box3D's manifolds (P3D-23).
//!
//! Box3D's manifold builders keep points up to `B3_SPECULATIVE_DISTANCE` (4 linear slops,
//! 2 cm) apart, and the solver lets a speculative point close only its gap in a substep, so a
//! shape moving at 2 m/s towards the ground from 1 cm stops at the surface in that step
//! (`convex_manifold.c`, `contact_solver.c`). Boxes collide as hulls with SAT and clipping, a
//! capsule lying on a face or along another capsule rests on two points. Before FIX-3.3 a
//! manifold existed only once the shapes overlapped, so impacts sank up to a step's travel,
//! lying capsules rested on one point and crossed box edges could get a contact point at the
//! midpoint of the two centres.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_3_speculative.c`):
//! 60 Hz, 4 substeps, no sleep, friction 0.6, a static ground box with its top at y = 0, 1 kg
//! shapes (1 m boxes, 0.5 m balls, capsules of radius 0.25 with centres 1 m apart).

use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, ContactManifold, PhysicsWorld, RigidBody, Shape};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

const DT: f32 = 1.0 / 60.0;
const CAPSULE_RADIUS: f32 = 0.25;
const CAPSULE_HALF_HEIGHT: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
enum Kind {
    Box,
    Ball,
    LyingCapsule,
    StandingCapsule,
}

impl Kind {
    fn shape(self) -> Shape {
        match self {
            Self::Box => Shape::Box {
                half_extents: Vec3::splat(0.5),
            },
            Self::Ball => Shape::Sphere { radius: 0.5 },
            Self::LyingCapsule | Self::StandingCapsule => Shape::Capsule {
                radius: CAPSULE_RADIUS,
                half_height: CAPSULE_HALF_HEIGHT,
            },
        }
    }

    /// The height of the body origin above the shape's lowest point.
    fn bottom_offset(self) -> f32 {
        match self {
            Self::Box | Self::Ball => 0.5,
            Self::LyingCapsule => CAPSULE_RADIUS,
            Self::StandingCapsule => CAPSULE_HALF_HEIGHT + CAPSULE_RADIUS,
        }
    }

    fn rotation(self) -> Quat {
        match self {
            Self::LyingCapsule => Quat::from_rotation_z(FRAC_PI_2),
            _ => Quat::IDENTITY,
        }
    }
}

fn add(world: &mut PhysicsWorld, body_type: BodyType, kind: Kind, position: Vec3) -> u32 {
    let body = RigidBody::new(body_type, position)
        .with_rotation(kind.rotation())
        .with_enable_sleep(false);
    let id = world.create_body(body);
    world.create_collider(
        id,
        Collider::new(kind.shape())
            .with_friction(0.6)
            .with_mass(1.0),
    );
    id.into()
}

fn world_with_ground(gravity: f32) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, gravity, 0.0));
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.6)
        .with_mass(1.0),
    );
    world
}

/// The manifold of body `id`, if it has one.
fn manifold_of(
    world: &PhysicsWorld,
    id: impl Into<korppi_physics::BodyId3d>,
) -> Option<&ContactManifold> {
    let id = id.into().index();
    world
        .manifolds()
        .iter()
        .find(|m| m.body_a == id || m.body_b == id)
}

fn point_count(world: &PhysicsWorld, id: impl Into<korppi_physics::BodyId3d>) -> usize {
    manifold_of(world, id).map_or(0, |m| m.points.len())
}

#[test]
fn approaching_shapes_stop_at_the_surface() {
    // Box3D: from 1 cm above the ground at 2 m/s (3.3 cm a step) every shape stops at the
    // surface. Deepest bottoms -0.00007 (box, ball, standing capsule) and -0.00012 (lying
    // capsule); after the first step 0.00021, -0.00002, 0.00023 and -0.00002; at rest on 4, 1,
    // 2 and 2 points.
    for (kind, resting_points) in [
        (Kind::Box, 4),
        (Kind::Ball, 1),
        (Kind::LyingCapsule, 2),
        (Kind::StandingCapsule, 2),
    ] {
        let mut world = world_with_ground(-9.81);
        let offset = kind.bottom_offset();
        let id = add(
            &mut world,
            BodyType::Dynamic,
            kind,
            Vec3::new(0.0, offset + 0.01, 0.0),
        );
        world.body_mut(id).unwrap().linear_velocity = Vec3::new(0.0, -2.0, 0.0);
        let bottom = |world: &PhysicsWorld| world.body(id).unwrap().position.y - offset;
        let mut lowest = f32::MAX;
        let mut first = 0.0;
        for step in 0..60 {
            world.step(DT);
            lowest = lowest.min(bottom(&world));
            if step == 0 {
                first = bottom(&world);
            }
        }
        assert!(
            lowest >= -0.006,
            "{kind:?}: should stop at the surface, sank to {lowest} (Box3D -0.0001)"
        );
        assert!(
            (-0.006..=0.011).contains(&first),
            "{kind:?}: the first step should end at the surface, the bottom is at {first}"
        );
        let resting = bottom(&world);
        assert!(
            (-0.006..=0.001).contains(&resting),
            "{kind:?}: should rest on the ground, its bottom is at {resting}"
        );
        assert_eq!(
            point_count(&world, id),
            resting_points,
            "{kind:?}: Box3D rests on {resting_points} points"
        );
    }
}

#[test]
fn contacts_begin_within_the_speculative_distance() {
    // Box3D, gravity off, the shape at rest with its bottom 1 cm above the ground: after a step
    // the box has a manifold of 4 points, the ball 1 and the capsules 2 (a standing capsule's
    // segment is clipped to the face, so its top end is a point too), the closest 1 cm apart,
    // and nothing moves. At 3 cm, beyond the 2 cm speculative distance, there is no manifold.
    for (kind, gap, expected) in [
        (Kind::Box, 0.01, 4),
        (Kind::Box, 0.03, 0),
        (Kind::Ball, 0.01, 1),
        (Kind::Ball, 0.03, 0),
        (Kind::LyingCapsule, 0.01, 2),
        (Kind::LyingCapsule, 0.03, 0),
        (Kind::StandingCapsule, 0.01, 2),
    ] {
        let mut world = world_with_ground(0.0);
        let id = add(
            &mut world,
            BodyType::Dynamic,
            kind,
            Vec3::new(0.0, kind.bottom_offset() + gap, 0.0),
        );
        world.step(DT);
        assert_eq!(
            point_count(&world, id),
            expected,
            "{kind:?} {gap} m above the ground: Box3D has {expected} points"
        );
        if expected > 0 {
            let manifold = manifold_of(&world, id).unwrap();
            let closest = manifold
                .points
                .iter()
                .map(|p| p.separation)
                .fold(f32::MAX, f32::min);
            assert!(
                (closest - gap).abs() <= 0.001,
                "{kind:?}: the closest point should be {gap} m apart, it is {closest}"
            );
        }
        let body = world.body(id).unwrap();
        assert!(
            body.linear_velocity.length() <= 1e-6 && body.angular_velocity.length() <= 1e-6,
            "{kind:?}: a speculative contact must not push, the body moves at {:?}",
            body.linear_velocity
        );
    }
}

#[test]
fn crossed_box_edges_touch_at_the_crossing() {
    // Box3D: a box turned 45° about z, 5 mm above a static box turned 45° about x and offset
    // (0.3, 0.2) in x and z, touches it at one point on the crossing of the two edges,
    // (0.30158, 0.70961, 0.00158), 5 mm apart along y.
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);
    let lower = world.create_body(
        RigidBody::new(BodyType::Static, Vec3::ZERO)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_4)),
    );
    world.create_collider(lower, Collider::new(Kind::Box.shape()).with_mass(1.0));
    let half_diagonal = 0.5 * 2.0f32.sqrt();
    let upper = world.create_body(
        RigidBody::new(
            BodyType::Dynamic,
            Vec3::new(0.3, 2.0 * half_diagonal + 0.005, 0.2),
        )
        .with_rotation(Quat::from_rotation_z(FRAC_PI_4))
        .with_enable_sleep(false),
    );
    world.create_collider(upper, Collider::new(Kind::Box.shape()).with_mass(1.0));
    world.step(DT);
    let manifold =
        manifold_of(&world, upper).expect("the edges are within the speculative distance");
    assert_eq!(manifold.points.len(), 1, "Box3D: one edge-edge point");
    let p = manifold.points[0];
    assert!(
        (0.29..=0.31).contains(&p.point.x)
            && (half_diagonal..=half_diagonal + 0.005).contains(&p.point.y)
            && p.point.z.abs() <= 0.01,
        "the point should be on the crossing of the edges, (0.3016, 0.7096, 0.0016), it is at {:?}",
        p.point
    );
    assert!((0.004..=0.006).contains(&p.separation), "{}", p.separation);
    assert!(manifold.normal.y.abs() >= 0.999, "{:?}", manifold.normal);
}

#[test]
fn lying_capsule_rests_on_two_points() {
    // Box3D: a capsule lying on the ground for 2 s rests on two points without turning
    // (spin 0) or moving (0.00002 m), at a height of 0.24988.
    let mut world = world_with_ground(-9.81);
    let id = add(
        &mut world,
        BodyType::Dynamic,
        Kind::LyingCapsule,
        Vec3::new(0.0, CAPSULE_RADIUS, 0.0),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    assert_eq!(point_count(&world, id), 2);
    let body = world.body(id).unwrap();
    assert!(
        body.angular_velocity.length() <= 1e-3,
        "{:?}",
        body.angular_velocity
    );
    let drift = Vec3::new(body.position.x, 0.0, body.position.z).length();
    assert!(
        drift <= 1e-3,
        "the capsule should not roll, it moved {drift} m"
    );
    assert!(
        (CAPSULE_RADIUS - 0.006..=CAPSULE_RADIUS + 0.001).contains(&body.position.y),
        "{}",
        body.position.y
    );
}

#[test]
fn parallel_capsules_rest_on_two_points() {
    // Box3D gives parallel capsules two points once both ends touch; within the speculative
    // distance one point at an end stops that end, so a capsule dropped from 1 cm onto a
    // parallel static one tips, sinks 2.1 cm and slides 1.4 cm along its axis before it settles
    // on two points (from step 7), at -0.00009, without spin.
    let mut world = PhysicsWorld::new();
    add(
        &mut world,
        BodyType::Static,
        Kind::LyingCapsule,
        Vec3::new(0.0, -CAPSULE_RADIUS, 0.0),
    );
    let id = add(
        &mut world,
        BodyType::Dynamic,
        Kind::LyingCapsule,
        Vec3::new(0.0, CAPSULE_RADIUS + 0.01, 0.0),
    );
    let mut lowest = f32::MAX;
    for step in 0..60 {
        world.step(DT);
        lowest = lowest.min(world.body(id).unwrap().position.y - CAPSULE_RADIUS);
        if step == 6 {
            assert_eq!(point_count(&world, id), 2, "two points from step 7");
        }
    }
    assert!(
        lowest >= -0.03,
        "Box3D sinks to -0.0213, Korppi to {lowest}"
    );
    assert_eq!(point_count(&world, id), 2, "two points after 1 s");
    let body = world.body(id).unwrap();
    assert!(
        (-0.006..=0.001).contains(&(body.position.y - CAPSULE_RADIUS)),
        "{}",
        body.position.y
    );
    assert!(
        body.angular_velocity.length() <= 1e-3,
        "{:?}",
        body.angular_velocity
    );
    let drift = Vec3::new(body.position.x, 0.0, body.position.z).length();
    assert!(drift <= 0.02, "Box3D slides 0.0136 m, Korppi {drift}");
}
