//! Integration tests for 2D joint constraints (Distance, Revolute, Weld, and the motor joint
//! dragging a body, which replaced the mouse joint as in Box2D v3.2).

use korppi_math::{Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::joint::{
    DistanceJoint2d, Joint2d, MotorJoint2d, RevoluteJoint2d, WeldJoint2d,
};

#[test]
fn distance_joint_enforces_length_between_bodies() {
    let mut body_a = RigidBody2d::new(BodyType::Static);
    body_a.position = Vec2::new(0.0, 10.0);

    let mut body_b = RigidBody2d::new(BodyType::Dynamic);
    body_b.set_mass(1.0, 0.5);
    body_b.position = Vec2::new(0.0, 7.0); // 3m below

    let mut joint = DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 3.0, 60.0, 1.0);

    // If body B tries to fall under gravity
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        body_b.integrate_velocities(Vec2::new(0.0, -9.81), dt);
        joint.solve(&mut body_a, &mut body_b, dt);
        body_b.integrate_positions(dt);
    }

    // Distance between A and B must remain strictly close to 3.0m
    let dist = (body_b.position - body_a.position).length();
    assert!(
        (dist - 3.0).abs() < 0.05,
        "Distance {:.3} should stay close to 3.0m",
        dist
    );
}

#[test]
fn revolute_joint_pins_pendulum_to_anchor() {
    let mut anchor_body = RigidBody2d::new(BodyType::Static);
    anchor_body.position = Vec2::new(0.0, 10.0);

    let mut bob = RigidBody2d::new(BodyType::Dynamic);
    bob.set_mass(1.0, 0.5);
    // Placed horizontally to the right of the anchor: length 2.0
    bob.position = Vec2::new(2.0, 10.0);

    let mut pin = RevoluteJoint2d::new(Vec2::ZERO, Vec2::new(-2.0, 0.0), 60.0, 1.0);

    // Simulate swinging
    let dt = 1.0 / 60.0;
    for _ in 0..30 {
        bob.integrate_velocities(Vec2::new(0.0, -9.81), dt);
        bob.integrate_positions(dt);

        pin.solve(&mut anchor_body, &mut bob, dt);
    }

    // Distance from anchor must remain strictly close to 2.0m
    let dist = (bob.position - anchor_body.position).length();
    assert!(
        (dist - 2.0).abs() < 0.05,
        "Pendulum radius {:.3} should stay close to 2.0m",
        dist
    );
}

#[test]
fn weld_joint_locks_relative_angle_and_translation() {
    let mut body_a = RigidBody2d::new(BodyType::Static);
    let mut body_b = RigidBody2d::new(BodyType::Dynamic);
    body_b.set_mass(1.0, 0.5);
    body_b.position = Vec2::new(1.0, 0.0);

    let mut weld = WeldJoint2d::new(Vec2::new(1.0, 0.0), Vec2::ZERO, 60.0, 1.0);

    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        body_b.integrate_velocities(Vec2::new(0.0, -9.81), dt);
        body_b.integrate_positions(dt);

        weld.solve(&mut body_a, &mut body_b, dt);
    }

    assert!((body_b.position - Vec2::new(1.0, 0.0)).length() < 0.05);
    assert!(body_b.rotation.angle().abs() < 0.05);
}

#[test]
fn motor_joint_drags_body_to_target() {
    // Box2D v3.2 removed the mouse joint; its samples drag with a motor joint: a 7.5 Hz,
    // critically damped spring with 1000 N toward the target, and an angular velocity row with
    // 0.25 N·m as angular friction (owner decision; reference `fix_2_6_motor_joint_tests.c`).
    // Box2D has the body at (4.98179, 4.98179) after 0.25 s and on target after 1 s.
    let mut world = korppi_physics2d::PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let ground = world.create_body(RigidBody2d::new(BodyType::Static));
    let body = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_mass(1.0, 1.0));
    let drag = MotorJoint2d::default()
        .with_local_frames(
            Transform2d::from_translation(Vec2::new(5.0, 5.0)),
            Transform2d::IDENTITY,
        )
        .with_linear_spring(7.5, 1.0, 1000.0)
        .with_angular_velocity(0.0, 0.25);
    world.create_joint(ground, body, Joint2d::Motor(drag));

    for step in 1..=60 {
        world.step(1.0 / 60.0);
        let position = world.body(body).unwrap().position;
        let expected = match step {
            15 => Vec2::new(4.981_79, 4.981_79),
            60 => Vec2::new(5.0, 5.0),
            _ => continue,
        };
        assert!(
            (position - expected).length() < 1e-4,
            "step {step}: body at {position:?}, Box2D {expected:?}"
        );
    }
}

#[test]
fn joint_enum_dispatch_works() {
    let mut a = RigidBody2d::new(BodyType::Static);
    let mut b = RigidBody2d::new(BodyType::Dynamic);
    b.set_mass(1.0, 0.5);
    b.position = Vec2::new(0.0, 2.0);

    let mut joint = Joint2d::Distance(DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 2.0, 60.0, 1.0));
    joint.solve(&mut a, &mut b, 0.016);
    assert!((b.position - Vec2::new(0.0, 2.0)).length() < 0.1);
}

#[test]
fn test_world_joint_query_and_removal() {
    use korppi_physics2d::PhysicsWorld2d;

    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -9.81));
    let b1 = world.create_body(RigidBody2d::new(BodyType::Static));
    let b2 = world.create_body(RigidBody2d::new(BodyType::Dynamic));

    let j = DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 2.0, 60.0, 1.0);
    let idx = world.create_joint(b1, b2, Joint2d::Distance(j));

    assert_eq!(world.joint_count(), 1);
    assert_eq!(world.joint_count(), 1);
    assert!(world.joint(idx).is_some());

    let removed = world.destroy_joint(idx);
    assert!(removed.is_some());
    assert_eq!(world.joint_count(), 0);
    assert!(world.joint(idx).is_none());
}
