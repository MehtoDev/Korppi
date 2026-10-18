//! Joints report when their force or torque crosses a threshold, and a pre-solve callback can
//! disable contacts (P2D-11).
//!
//! Before FIX-2.4 neither existed. As in Box2D v3.2, a joint's force and torque thresholds
//! default to `f32::MAX`; a joint reports an event in a step if, during the biased solve stage
//! of any substep, its reaction impulse over the substep reaches a threshold. That stage runs
//! before the relax pass restores the full impulse, so it sees less than the joint's full force.
//! The pre-solve callback sees each new manifold of colliders that enable pre-solve events and
//! may empty it. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_4_joint_events.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d, WeldJoint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{JointInstance2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn sleepless(kind: BodyType, position: Vec2) -> RigidBody2d {
    let mut body = RigidBody2d::new(kind).with_position(position);
    body.enable_sleep = false;
    body
}

/// A bob released 1 m to the side of a static pivot. Returns the number of joint events in
/// `steps` steps and the step of the first.
fn pendulum_events(force_threshold: f32, steps: usize) -> (usize, Option<usize>) {
    let mut world = PhysicsWorld2d::new();
    let anchor = world.create_body(sleepless(BodyType::Static, Vec2::new(0.0, 5.0)));
    let bob = world.create_body(sleepless(BodyType::Dynamic, Vec2::new(1.0, 5.0)));
    world.create_collider(
        bob,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
    );
    let pin = RevoluteJoint2d::new(Vec2::ZERO, Vec2::new(-1.0, 0.0), 0.0, 0.0);
    let joint = world.add_joint(
        JointInstance2d::new(anchor, bob, Joint2d::Revolute(pin))
            .with_event_thresholds(force_threshold, f32::MAX),
    );
    let (mut count, mut first) = (0, None);
    for step in 1..=steps {
        world.step(DT);
        let events = world.joint_events();
        assert!(events.iter().all(|e| e.joint == joint));
        if !events.is_empty() {
            first.get_or_insert(step);
        }
        count += events.len();
    }
    (count, first)
}

#[test]
fn a_swinging_pendulum_reports_force_events() {
    // The 0.19635 kg bob weighs 1.9262 N. Box2D v3.2: 61 events above its weight (the first at
    // step 21), 42 above 1.5 times it (from step 27), none above twice it.
    let weight = 0.25 * 0.25 * std::f32::consts::PI * 9.81;
    assert_eq!(pendulum_events(weight, 120), (61, Some(21)));
    assert_eq!(pendulum_events(1.5 * weight, 120), (42, Some(27)));
    assert_eq!(pendulum_events(2.0 * weight, 120), (0, None));
    // A zero threshold reports every step; the default never does.
    assert_eq!(pendulum_events(0.0, 30), (30, Some(1)));
    assert_eq!(pendulum_events(f32::MAX, 120), (0, None));
}

fn beam_events(torque_threshold: f32) -> usize {
    let mut world = PhysicsWorld2d::new();
    let anchor = world.create_body(sleepless(BodyType::Static, Vec2::new(0.0, 5.0)));
    let beam = world.create_body(sleepless(BodyType::Dynamic, Vec2::new(1.0, 5.0)));
    world.create_collider(
        beam,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.0, 0.1))),
    );
    let weld = WeldJoint2d::new(Vec2::ZERO, Vec2::new(-1.0, 0.0), 0.0, 0.0);
    world.add_joint(
        JointInstance2d::new(anchor, beam, Joint2d::Weld(weld))
            .with_event_thresholds(f32::MAX, torque_threshold),
    );
    (0..60)
        .map(|_| {
            world.step(DT);
            world.joint_events().len()
        })
        .sum()
}

#[test]
fn a_welded_beam_reports_torque_events() {
    // The 0.4 kg beam needs 3.924 N m to hold. Box2D v3.2: an event every step below that; above
    // it, two from the first steps' sag.
    assert_eq!(beam_events(3.5), 60);
    assert_eq!(beam_events(4.5), 2);
}

#[test]
fn pre_solve_can_disable_contacts() {
    // A box with pre-solve events falls through the ground while the callback empties its
    // manifolds; the other box lands. Box2D v3.2: y = -3.9254 and 0.49991 after 1 s.
    let mut world = PhysicsWorld2d::new();
    world.set_pre_solve(|_a, _b, manifold| manifold.point_count = 0);
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))),
    );
    let unit = || Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)));
    let ghost = world.create_body(sleepless(BodyType::Dynamic, Vec2::new(-2.0, 1.0)));
    world.create_collider(ghost, unit().with_pre_solve_events(true));
    let solid = world.create_body(sleepless(BodyType::Dynamic, Vec2::new(2.0, 1.0)));
    world.create_collider(solid, unit());
    for _ in 0..60 {
        world.step(DT);
    }
    let ghost_y = world.body(ghost).unwrap().position.y;
    let solid_y = world.body(solid).unwrap().position.y;
    assert!(
        (ghost_y + 3.9254).abs() < 0.01,
        "the ghost is at y = {ghost_y:.4}"
    );
    assert!(
        (solid_y - 0.49991).abs() < 0.002,
        "the solid box is at y = {solid_y:.4}"
    );

    // Without the callback the ghost collides again (Box2D v3.2: y = 0.49991 after 1 s).
    world.clear_pre_solve();
    world.set_transform(ghost, Vec2::new(-2.0, 1.0), korppi_math::Rot2::IDENTITY);
    world.body_mut(ghost).unwrap().linear_velocity = Vec2::ZERO;
    for _ in 0..60 {
        world.step(DT);
    }
    let ghost_y = world.body(ghost).unwrap().position.y;
    assert!(
        (ghost_y - 0.49991).abs() < 0.002,
        "the ghost is at y = {ghost_y:.4}"
    );
}

#[test]
fn pre_continuous_can_ignore_impacts() {
    // A one-way platform: two balls (r 0.1) fall at 50 m/s onto a 0.1 m platform. The one with
    // pre-solve events has its impacts and contacts ignored and passes through; the other lands.
    // Box2D v3.2 after 12 steps: y = -5.0000 and 0.14804.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    world.set_pre_solve(|_a, _b, manifold| manifold.point_count = 0);
    world.set_pre_continuous(|_a, _b, _point, _normal| false);
    let platform = world.create_body(RigidBody2d::new(BodyType::Static));
    world.create_collider(
        platform,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(5.0, 0.05))),
    );
    let ball = |world: &mut PhysicsWorld2d, x: f32, pre_solve: bool| {
        let id = world.create_body(
            RigidBody2d::new(BodyType::Dynamic)
                .with_position(Vec2::new(x, 5.0))
                .with_linear_velocity(Vec2::new(0.0, -50.0)),
        );
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.1)))
                .with_pre_solve_events(pre_solve),
        );
        id
    };
    let ghost = ball(&mut world, -1.0, true);
    let solid = ball(&mut world, 1.0, false);
    for _ in 0..12 {
        world.step(DT);
    }
    let ghost_y = world.body(ghost).unwrap().position.y;
    let solid_y = world.body(solid).unwrap().position.y;
    assert!(
        (ghost_y + 5.0).abs() < 0.01,
        "the ghost is at y = {ghost_y:.4}"
    );
    assert!(
        (solid_y - 0.14804).abs() < 0.002,
        "the solid ball is at y = {solid_y:.4}"
    );
}
