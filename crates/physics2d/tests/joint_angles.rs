//! Joint angles stay continuous when a body's rotation crosses ±π (P2D-03).
//!
//! Each body's `angle()` is wrapped to (-π, π] on its own, so the naive difference
//! `angle_b - angle_a` jumps by 2π whenever one body crosses ±π before the other. Box2D measures
//! the angle of the relative rotation `qA⁻¹·qB` instead, and unwinds it against targets. Before
//! FIX-2.1 a ±0.5 rad revolute limit read 2.27 rad after the wrap and spun the body at 333 rad/s
//! (audit probe P4), and tumbling ragdolls exploded.

use korppi_math::{Rot2, Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::joint::{
    Joint2d, MotorJoint2d, PrismaticJoint2d, RevoluteJoint2d, WeldJoint2d,
};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};
use std::f32::consts::PI;

const DT: f32 = 1.0 / 60.0;

/// Angle of `b` relative to `a`, in (-π, π], computed independently of the crate.
fn relative_angle(a: Rot2, b: Rot2) -> f32 {
    (a.c * b.s - a.s * b.c).atan2(a.c * b.c + a.s * b.s)
}

/// `angle` wrapped into (-π, π].
fn wrap(angle: f32) -> f32 {
    relative_angle(Rot2::IDENTITY, Rot2::from_angle(angle))
}

fn body(world: &PhysicsWorld2d, id: BodyId2d) -> &RigidBody2d {
    world.body(id).unwrap()
}

#[test]
fn revolute_limit_holds_when_both_bodies_cross_pi() {
    // Probe P4: both bodies start at 3.0 rad; B turns at 1 rad/s into a ±0.5 rad limit.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let a =
        world.create_body(RigidBody2d::new(BodyType::Static).with_rotation(Rot2::from_angle(3.0)));
    let b = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_rotation(Rot2::from_angle(3.0))
            .with_angular_velocity(1.0)
            .with_mass(1.0, 1.0),
    );
    world.create_joint(
        a,
        b,
        Joint2d::Revolute(
            RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0).with_limits(-0.5, 0.5),
        ),
    );

    let (mut max_angle, mut max_speed) = (0.0f32, 0.0f32);
    for _ in 0..120 {
        world.step(DT);
        max_angle = max_angle.max(relative_angle(
            body(&world, a).rotation,
            body(&world, b).rotation,
        ));
        max_speed = max_speed.max(body(&world, b).angular_velocity.abs());
    }
    assert!(
        max_angle > 0.45,
        "B never reached the limit ({max_angle:.3} rad)"
    );
    assert!(
        max_angle < 0.52,
        "limit exceeded: {max_angle:.3} rad (probe P4: 2.27)"
    );
    assert!(
        max_speed <= 1.0 + 1e-3,
        "B spun up to {max_speed:.2} rad/s (probe P4: 333)"
    );
}

#[test]
fn revolute_angle_is_continuous_across_pi() {
    let a = RigidBody2d::new(BodyType::Static).with_rotation(Rot2::from_angle(3.0));
    let b = RigidBody2d::new(BodyType::Dynamic).with_rotation(Rot2::from_angle(-3.0));
    let joint = RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0);
    // The naive difference of the wrapped angles is -6. Box2D v3.2 reports 0.2862 instead of the
    // exact 2*pi - 6 = 0.2832, because it builds rotations with approximate sin/cos.
    let expected = 2.0 * PI - 6.0;
    let angle = joint.angle(&a, &b);
    assert!(
        (angle - expected).abs() < 0.01,
        "joint angle {angle} instead of {expected}"
    );

    let angle = joint.with_reference_angle(0.5).angle(&a, &b);
    assert!(
        (angle - (expected - 0.5)).abs() < 0.01,
        "joint angle with a reference {angle} instead of {}",
        expected - 0.5
    );
}

#[test]
fn welded_pair_tumbling_past_pi_stays_welded() {
    // Two bodies welded at the origin, B turned 1 rad against A, spinning together at 3 rad/s
    // about the weld for 5 s: each body crosses ±π several times, never at the same moment.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let spin = 3.0;
    let a = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(-0.5, 0.0))
            .with_linear_velocity(Vec2::new(0.0, -0.5 * spin))
            .with_angular_velocity(spin)
            .with_mass(1.0, 1.0),
    );
    let b = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.5, 0.0))
            .with_rotation(Rot2::from_angle(1.0))
            .with_linear_velocity(Vec2::new(0.0, 0.5 * spin))
            .with_angular_velocity(spin)
            .with_mass(1.0, 1.0),
    );
    let weld = WeldJoint2d::new(
        Vec2::new(0.5, 0.0),
        Rot2::from_angle(1.0).inv_rotate(Vec2::new(-0.5, 0.0)),
        0.0,
        0.0,
    )
    .with_reference_angle(1.0);
    let local_b = weld.local_frame_b.translation;
    world.create_joint(a, b, Joint2d::Weld(weld));

    let (mut angle_error, mut gap, mut speed_error) = (0.0f32, 0.0f32, 0.0f32);
    for _ in 0..300 {
        world.step(DT);
        let (ba, bb) = (body(&world, a), body(&world, b));
        angle_error = angle_error.max(wrap(relative_angle(ba.rotation, bb.rotation) - 1.0).abs());
        let anchor_a = ba.position + ba.rotation.rotate(Vec2::new(0.5, 0.0));
        let anchor_b = bb.position + bb.rotation.rotate(local_b);
        gap = gap.max((anchor_b - anchor_a).length());
        speed_error = speed_error.max((bb.angular_velocity - spin).abs());
    }
    assert!(
        angle_error < 0.01,
        "weld angle drifted {angle_error:.4} rad"
    );
    assert!(gap < 0.01, "weld anchors separated {gap:.4} m");
    assert!(speed_error < 0.1, "spin changed by {speed_error:.3} rad/s");
}

#[test]
fn prismatic_angle_lock_holds_past_pi() {
    // A slider turned 0.5 rad against a kinematic rail that spins at 2 rad/s for 4 s.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    // Sleep off, as in the Box2D reference: a shapeless body has no extent, so Box2D lets it
    // sleep however fast it spins.
    world.enable_sleep(false);
    let rail = world.create_body(RigidBody2d::new(BodyType::Kinematic).with_angular_velocity(2.0));
    let slider = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_rotation(Rot2::from_angle(0.5))
            .with_angular_velocity(2.0)
            .with_mass(1.0, 1.0),
    );
    world.create_joint(
        rail,
        slider,
        Joint2d::Prismatic(
            PrismaticJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::X, 0.0, 0.0)
                .with_reference_angle(0.5),
        ),
    );

    let (mut angle_error, mut speed_error) = (0.0f32, 0.0f32);
    for _ in 0..240 {
        world.step(DT);
        let (r, s) = (body(&world, rail), body(&world, slider));
        angle_error = angle_error.max(wrap(relative_angle(r.rotation, s.rotation) - 0.5).abs());
        speed_error = speed_error.max((s.angular_velocity - 2.0).abs());
    }
    assert!(
        angle_error < 0.01,
        "slider angle drifted {angle_error:.4} rad"
    );
    assert!(
        speed_error < 0.1,
        "slider spin changed by {speed_error:.3} rad/s"
    );
}

/// Turns B from -3.0 rad toward a joint target of +3.0 rad, 0.28 rad away across ±π, and returns
/// (final error to the target, total angle B travelled).
fn turn_across_pi(joint: Joint2d) -> (f32, f32) {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let a = world.create_body(RigidBody2d::new(BodyType::Static));
    let b = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_rotation(Rot2::from_angle(-3.0))
            .with_mass(1.0, 1.0),
    );
    world.create_joint(a, b, joint);
    let mut travelled = 0.0f32;
    let mut previous = body(&world, b).rotation;
    for _ in 0..180 {
        world.step(DT);
        let current = body(&world, b).rotation;
        travelled += relative_angle(previous, current).abs();
        previous = current;
    }
    (wrap(previous.angle() - 3.0).abs(), travelled)
}

#[test]
fn revolute_spring_turns_the_short_way_across_pi() {
    let (error, travelled) = turn_across_pi(Joint2d::Revolute(
        RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0)
            .with_spring(2.0, 1.0)
            .with_target_angle(3.0),
    ));
    assert!(
        error < 0.02,
        "spring stopped {error:.4} rad from its target"
    );
    assert!(
        travelled < 0.6,
        "spring turned {travelled:.3} rad, the long way round"
    );
}

#[test]
fn motor_joint_turns_the_short_way_across_pi() {
    // Since FIX-2.6 Box2D v3.2's motor joint, whose angular spring turns frame B toward frame A
    // (owner decision; reference `fix_2_6_motor_joint_tests.c`): Box2D ends on target, having
    // turned 0.28318 rad.
    let target = Transform2d::from_translation_rotation(Vec2::ZERO, Rot2::from_angle(3.0));
    let (error, travelled) = turn_across_pi(Joint2d::Motor(
        MotorJoint2d::default()
            .with_local_frames(target, Transform2d::IDENTITY)
            .with_linear_spring(2.0, 1.0, 1000.0)
            .with_angular_spring(2.0, 1.0, 1000.0),
    ));
    assert!(
        error < 0.02,
        "motor joint stopped {error:.4} rad from its offset"
    );
    assert!(
        travelled < 0.6,
        "motor joint turned {travelled:.3} rad, the long way round"
    );
}
