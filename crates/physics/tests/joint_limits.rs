//! Revolute and prismatic joints keep their angle and translation within their limits (P3D-03).
//!
//! Box3D solves each limit as a one-sided row with an accumulated, clamped impulse: speculative
//! while the joint is inside its range, so it stops exactly at the limit, and pushing back once
//! it is past it (`revolute_joint.c`, `prismatic_joint.c`). Before FIX-3.1 the limits were
//! stored but never read: a hinge limited to ±0.1 rad and spun at 5 rad/s turned to 2.5 rad in
//! 0.5 s (audit probe Q3).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_1_joints.c`):
//! 60 Hz, 4 substeps, 1 kg and 1 kg·m² bodies without shapes, no damping and no sleep. A
//! revolute hinge turns about local z and its angle is zero when both bodies have the same
//! rotation; a prismatic slider moves along its axis from where its anchors coincide.

use korppi_math::Vec3;
use korppi_physics::joint::{Joint, PrismaticJoint, RevoluteJoint};
use korppi_physics::{BodyId3d, BodyType, PhysicsWorld, RigidBody};

const DT: f32 = 1.0 / 60.0;

fn body(body_type: BodyType, position: Vec3) -> RigidBody {
    let mut body = RigidBody::new(body_type, position).with_enable_sleep(false);
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    body
}

fn world_with_gravity(gravity: Vec3) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(gravity);
    world
}

/// The twist of body B relative to body A about z (Box3D `b3RevoluteJoint_GetAngle` with
/// identity joint frames).
fn hinge_angle(world: &PhysicsWorld, a: impl Into<BodyId3d>, b: impl Into<BodyId3d>) -> f32 {
    let qa = world.body(a).unwrap().rotation;
    let mut qb = world.body(b).unwrap().rotation;
    if qa.dot(qb) < 0.0 {
        qb = -qb;
    }
    let relative = qa.conjugate() * qb;
    if relative.w < 0.0 {
        2.0 * (-relative.z).atan2(-relative.w)
    } else {
        2.0 * relative.z.atan2(relative.w)
    }
}

/// A body hinged at its centre to a static base and spun at `spin` rad/s about the hinge for
/// 0.5 s without gravity. Returns the largest and the final angle.
fn spinning_hinge(spin: f32, limits: Option<(f32, f32)>) -> (f32, f32) {
    let mut world = world_with_gravity(Vec3::ZERO);
    let base = world.create_body(body(BodyType::Static, Vec3::ZERO));
    let mut wheel = body(BodyType::Dynamic, Vec3::ZERO);
    wheel.angular_velocity = Vec3::new(0.0, 0.0, spin);
    let wheel = world.create_body(wheel);
    let mut hinge = RevoluteJoint::new(base, wheel, Vec3::ZERO, Vec3::ZERO, Vec3::Z);
    if let Some((lower, upper)) = limits {
        hinge = hinge.with_limits(lower, upper);
    }
    world.create_joint(Joint::Revolute(hinge));

    let mut largest: f32 = 0.0;
    for _ in 0..30 {
        world.step(DT);
        let angle = hinge_angle(&world, base, wheel);
        if angle.abs() > largest.abs() {
            largest = angle;
        }
    }
    (largest, hinge_angle(&world, base, wheel))
}

#[test]
fn revolute_upper_limit_stops_a_spinning_body() {
    let (largest, last) = spinning_hinge(5.0, Some((-0.1, 0.1)));

    // Box3D: the hinge stops at 0.10000 rad and stays there. Probe Q3 saw 2.5 rad.
    assert!(
        (0.0..=0.105).contains(&largest),
        "a hinge limited to 0.1 rad turned to {largest} rad"
    );
    assert!(
        (last - 0.1).abs() < 0.01,
        "the hinge should rest on its upper limit, it is at {last} rad"
    );
}

#[test]
fn revolute_lower_limit_stops_a_spinning_body() {
    let (largest, last) = spinning_hinge(-5.0, Some((-0.1, 0.1)));

    // Box3D: the hinge stops at -0.10000 rad.
    assert!(
        (-0.105..=0.0).contains(&largest),
        "a hinge limited to -0.1 rad turned to {largest} rad"
    );
    assert!(
        (last + 0.1).abs() < 0.01,
        "the hinge should rest on its lower limit, it is at {last} rad"
    );
}

#[test]
fn revolute_without_limit_spins_freely() {
    // Box3D: 2.49989 rad after 0.5 s at 5 rad/s.
    let (_, last) = spinning_hinge(5.0, None);
    assert!(
        (2.49..=2.51).contains(&last),
        "an unlimited hinge should turn 2.5 rad, turned {last} rad"
    );
}

#[test]
fn revolute_limits_are_ordered_and_kept_within_a_turn() {
    // As Box3D does, the limits are ordered and clamped to ±0.99 π.
    let hinge = RevoluteJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Z).with_limits(4.0, -7.0);
    let bound = 0.99 * std::f32::consts::PI;
    assert_eq!((hinge.lower_angle, hinge.upper_angle), (-bound, bound));
    assert!(hinge.enable_limit);
}

#[test]
fn revolute_limit_holds_a_falling_arm() {
    let mut world = world_with_gravity(Vec3::new(0.0, -9.81, 0.0));
    let pivot = world.create_body(body(BodyType::Static, Vec3::new(0.0, 10.0, 0.0)));
    let arm = world.create_body(body(BodyType::Dynamic, Vec3::new(1.0, 10.0, 0.0)));
    world.create_joint(Joint::Revolute(
        RevoluteJoint::new(pivot, arm, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), Vec3::Z)
            .with_limits(-0.25, 0.25),
    ));

    let mut lowest: f32 = 0.0;
    for _ in 0..120 {
        world.step(DT);
        lowest = lowest.min(hinge_angle(&world, pivot, arm));
    }

    // Box3D: the arm swings down onto its lower limit and stays at -0.25007 rad, its pin 0.13 mm
    // from the pivot.
    assert!(
        lowest >= -0.27,
        "the arm swung past its -0.25 rad limit to {lowest} rad"
    );
    let last = hinge_angle(&world, pivot, arm);
    assert!(
        (-0.26..=-0.24).contains(&last),
        "the arm should hang on its lower limit, it is at {last} rad"
    );
    let arm_body = world.body(arm).unwrap();
    let pin = arm_body.position + arm_body.rotation * Vec3::new(-1.0, 0.0, 0.0);
    let drift = (pin - Vec3::new(0.0, 10.0, 0.0)).length();
    assert!(drift <= 0.01, "the hinge pin drifted {drift} m");
}

/// A body on a static rail along `axis`, launched at `speed` along it, with limits ±0.5 m.
/// Returns the largest and the final translation.
fn slider(gravity: Vec3, axis: Vec3, speed: f32, seconds: f32) -> (f32, f32) {
    let mut world = world_with_gravity(gravity);
    let rail = world.create_body(body(BodyType::Static, Vec3::ZERO));
    let mut carriage = body(BodyType::Dynamic, Vec3::ZERO);
    carriage.linear_velocity = axis * speed;
    let carriage = world.create_body(carriage);
    world.create_joint(Joint::Prismatic(
        PrismaticJoint::new(rail, carriage, Vec3::ZERO, Vec3::ZERO, axis).with_limits(-0.5, 0.5),
    ));

    let translation = |world: &PhysicsWorld| world.body(carriage).unwrap().position.dot(axis);
    let mut largest: f32 = 0.0;
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
        let t = translation(&world);
        if t.abs() > largest.abs() {
            largest = t;
        }
    }
    (largest, translation(&world))
}

#[test]
fn prismatic_upper_limit_stops_a_slider() {
    let (largest, last) = slider(Vec3::ZERO, Vec3::X, 3.0, 1.0);

    // Box3D: the slider stops at 0.50000 m. Before FIX-3.1 prismatic joints had no limits.
    assert!(
        (0.0..=0.51).contains(&largest),
        "a slider limited to 0.5 m travelled {largest} m"
    );
    assert!(
        (last - 0.5).abs() < 0.01,
        "the slider should rest on its upper limit, it is at {last} m"
    );
}

#[test]
fn prismatic_lower_limit_stops_a_slider() {
    let (largest, last) = slider(Vec3::ZERO, Vec3::X, -3.0, 1.0);

    // Box3D: the slider stops at -0.50000 m.
    assert!(
        (-0.51..=0.0).contains(&largest),
        "a slider limited to -0.5 m travelled {largest} m"
    );
    assert!(
        (last + 0.5).abs() < 0.01,
        "the slider should rest on its lower limit, it is at {last} m"
    );
}

#[test]
fn prismatic_limit_holds_a_vertical_slider() {
    let (largest, last) = slider(Vec3::new(0.0, -9.81, 0.0), Vec3::Y, 0.0, 2.0);

    // Box3D: the slider falls onto its lower limit and rests at -0.50007 m.
    assert!(
        largest >= -0.52,
        "the slider fell past its -0.5 m limit to {largest} m"
    );
    assert!(
        (-0.52..=-0.49).contains(&last),
        "the slider should rest on its lower limit, it is at {last} m"
    );
}

#[test]
fn prismatic_limits_are_ordered() {
    let slider = PrismaticJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::X).with_limits(2.0, -1.0);
    assert_eq!(
        (slider.lower_translation, slider.upper_translation),
        (-1.0, 2.0)
    );
    assert!(slider.enable_limit);
}
