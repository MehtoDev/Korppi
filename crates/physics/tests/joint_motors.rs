//! Joint motors never exceed their maximum torque or force (P3D-04).
//!
//! Box3D accumulates a motor's impulse and clamps the accumulated value, so over any period a
//! motor delivers at most its maximum torque or force times that period. Before FIX-3.1 each
//! iteration clamped only its own impulse and added it to an accumulator that was never
//! clamped and was re-applied every step: a 0.1 N·m motor spun a 0.1 kg·m² wheel to 30.5 rad/s
//! in 1 s instead of 1 rad/s (audit probe Q4).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_1_joints.c`):
//! 60 Hz, 4 substeps, 1 kg bodies without shapes, no damping and no sleep.

use korppi_math::{Mat3, Vec3};
use korppi_physics::joint::{Joint, PrismaticJoint, RevoluteJoint};
use korppi_physics::{BodyType, PhysicsWorld, RigidBody};

const DT: f32 = 1.0 / 60.0;

fn body(body_type: BodyType, position: Vec3, inertia: f32) -> RigidBody {
    let mut body = RigidBody::new(body_type, position).with_enable_sleep(false);
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    body.local_inertia = Mat3::from_diagonal(Vec3::splat(inertia));
    body
}

fn run(world: &mut PhysicsWorld, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

/// A wheel of inertia `inertia` hinged at its centre to a static base and driven by a motor,
/// without gravity. Returns its spin after 1 s and after 2 s.
fn motor_wheel(inertia: f32, max_torque: f32, speed: f32) -> (f32, f32) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);
    let base = world.create_body(body(BodyType::Static, Vec3::ZERO, 1.0));
    let wheel = world.create_body(body(BodyType::Dynamic, Vec3::ZERO, inertia));
    world.create_joint(Joint::Revolute(
        RevoluteJoint::new(base, wheel, Vec3::ZERO, Vec3::ZERO, Vec3::Z)
            .with_motor(speed, max_torque),
    ));
    run(&mut world, 1.0);
    let after_one = world.body(wheel).unwrap().angular_velocity.z;
    run(&mut world, 1.0);
    (after_one, world.body(wheel).unwrap().angular_velocity.z)
}

#[test]
fn revolute_motor_torque_is_limited() {
    // T = 0.1 N·m on I = 0.1 kg·m² gives 1 rad/s per second. Box3D: 1.00000 rad/s after 1 s
    // and 2.00001 rad/s after 2 s. Probe Q4 saw 30.5 rad/s after 1 s.
    let (after_one, after_two) = motor_wheel(0.1, 0.1, 1000.0);
    assert!(
        (0.99..=1.01).contains(&after_one),
        "a 0.1 N·m motor should spin the wheel to 1 rad/s in 1 s, reached {after_one}"
    );
    assert!(
        (1.98..=2.02).contains(&after_two),
        "the motor's torque should stay limited, the wheel reached {after_two} rad/s in 2 s"
    );
}

#[test]
fn revolute_motor_reaches_and_holds_its_speed() {
    // Box3D: 2.00000 rad/s after 1 s and after 2 s.
    let (after_one, after_two) = motor_wheel(1.0, 10.0, 2.0);
    for (when, spin) in [("1 s", after_one), ("2 s", after_two)] {
        assert!(
            (1.99..=2.01).contains(&spin),
            "a motor strong enough should hold 2 rad/s, the wheel spins at {spin} after {when}"
        );
    }
}

#[test]
fn prismatic_motor_force_is_limited() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);
    let rail = world.create_body(body(BodyType::Static, Vec3::ZERO, 1.0));
    let carriage = world.create_body(body(BodyType::Dynamic, Vec3::ZERO, 1.0));
    world.create_joint(Joint::Prismatic(
        PrismaticJoint::new(rail, carriage, Vec3::ZERO, Vec3::ZERO, Vec3::X).with_motor(100.0, 1.0),
    ));
    run(&mut world, 1.0);

    // A 1 N motor on 1 kg gives 1 m/s per second. Box3D: 1.00000 m/s.
    let speed = world.body(carriage).unwrap().linear_velocity.x;
    assert!(
        (0.99..=1.01).contains(&speed),
        "a 1 N motor should drive the 1 kg slider to 1 m/s in 1 s, reached {speed}"
    );
}

/// A 1 kg arm 1 m from its hinge, 9.81 N·m of gravity torque, held by a motor at speed 0.
/// Returns how far the arm's centre dropped in 1 s.
fn motor_brake(max_torque: f32) -> f32 {
    let mut world = PhysicsWorld::new();
    let pivot = world.create_body(body(BodyType::Static, Vec3::new(0.0, 10.0, 0.0), 1.0));
    let arm = world.create_body(body(BodyType::Dynamic, Vec3::new(1.0, 10.0, 0.0), 1.0));
    world.create_joint(Joint::Revolute(
        RevoluteJoint::new(pivot, arm, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), Vec3::Z)
            .with_motor(0.0, max_torque),
    ));
    run(&mut world, 1.0);
    10.0 - world.body(arm).unwrap().position.y
}

#[test]
fn strong_motor_brake_holds_an_arm() {
    // Box3D: the arm's centre drops 0.00021 m.
    let drop = motor_brake(12.0);
    assert!(
        drop <= 0.05,
        "a 12 N·m brake should hold the 9.81 N·m arm, it dropped {drop} m"
    );
}

#[test]
fn weak_motor_brake_slips() {
    // Box3D: the arm's centre drops 0.89138 m.
    let drop = motor_brake(5.0);
    assert!(
        (0.79..=0.99).contains(&drop),
        "a 5 N·m brake should let the 9.81 N·m arm slip about 0.89 m, it dropped {drop} m"
    );
}
