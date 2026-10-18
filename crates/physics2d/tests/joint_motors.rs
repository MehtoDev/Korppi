//! Joint motors never exceed their maximum torque or force (P2D-06).
//!
//! Box2D accumulates a motor's impulse over the substep and clamps the accumulated value to
//! `h * maxMotorTorque`. Before FIX-2.1 each solver iteration clamped only its own delta and
//! overwrote the accumulator, so a 1 N·m motor spun a 1 kg·m² wheel to 2.25 rad/s in 1 s
//! instead of 1 rad/s (audit probe P3).

use korppi_math::{Rot2, Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::joint::{
    Joint2d, MotorJoint2d, PrismaticJoint2d, RevoluteJoint2d, WheelJoint2d,
};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

/// A static base and a 1 kg, 1 kg·m² body at `position`, without gravity.
fn base_and_body(position: Vec2) -> (PhysicsWorld2d, BodyId2d, BodyId2d) {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    // Sleep off, as in the Box2D reference: a shapeless body has no extent, so Box2D lets it
    // sleep however fast it spins.
    world.enable_sleep(false);
    let base = world.create_body(RigidBody2d::new(BodyType::Static));
    let body = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(position)
            .with_mass(1.0, 1.0),
    );
    (world, base, body)
}

fn run(world: &mut PhysicsWorld2d, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

/// Asserts that a saturated drive delivered its full limit over the run, and no more.
fn assert_saturated(what: &str, reached: f32, bound: f32) {
    assert!(
        reached <= bound * 1.001,
        "{what}: reached {reached:.4}, but its limit allows only {bound:.4} (probe P3: 2.25x)"
    );
    assert!(
        reached >= bound * 0.98,
        "{what}: reached only {reached:.4} of the {bound:.4} its limit allows"
    );
}

#[test]
fn revolute_motor_torque_is_capped() {
    // Probe P3: 1 N·m on 1 kg·m² for 1 s gives at most 1 rad/s.
    let (mut world, base, wheel) = base_and_body(Vec2::ZERO);
    world.create_joint(
        base,
        wheel,
        Joint2d::Revolute(
            RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0).with_motor(1000.0, 1.0),
        ),
    );
    run(&mut world, 1.0);
    let speed = world.body(wheel).unwrap().angular_velocity;
    assert_saturated("revolute motor", speed, 1.0);
}

#[test]
fn prismatic_motor_force_is_capped() {
    // 2 N on 1 kg for 1 s gives at most 2 m/s.
    let (mut world, base, slider) = base_and_body(Vec2::ZERO);
    world.create_joint(
        base,
        slider,
        Joint2d::Prismatic(
            PrismaticJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::X, 0.0, 0.0)
                .with_motor(1000.0, 2.0),
        ),
    );
    run(&mut world, 1.0);
    let speed = world.body(slider).unwrap().linear_velocity.x;
    assert_saturated("prismatic motor", speed, 2.0);
}

#[test]
fn wheel_motor_torque_is_capped() {
    let (mut world, chassis, wheel) = base_and_body(Vec2::ZERO);
    world.create_joint(
        chassis,
        wheel,
        Joint2d::Wheel(
            WheelJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::Y, 0.0, 0.0).with_motor(1000.0, 1.0),
        ),
    );
    run(&mut world, 1.0);
    let speed = world.body(wheel).unwrap().angular_velocity;
    assert_saturated("wheel motor", speed, 1.0);
}

#[test]
fn motor_joint_force_and_torque_are_capped() {
    // Targets far out of reach: 100 m away and 3 rad around, with 2 N and 1 N·m. Since
    // FIX-2.6 this is Box2D v3.2's motor joint, whose springs pull toward frame A (owner
    // decision; reference `fix_2_6_motor_joint_tests.c`): Box2D reaches exactly 2 m/s and
    // 1 rad/s.
    let (mut world, base, body) = base_and_body(Vec2::ZERO);
    let target =
        Transform2d::from_translation_rotation(Vec2::new(100.0, 0.0), Rot2::from_angle(3.0));
    world.create_joint(
        base,
        body,
        Joint2d::Motor(
            MotorJoint2d::default()
                .with_local_frames(target, Transform2d::IDENTITY)
                .with_linear_spring(4.0, 0.7, 2.0)
                .with_angular_spring(4.0, 0.7, 1.0),
        ),
    );
    run(&mut world, 1.0);
    let body = world.body(body).unwrap();
    assert_saturated("motor joint force", body.linear_velocity.length(), 2.0);
    assert_saturated("motor joint torque", body.angular_velocity, 1.0);
}

#[test]
fn motor_reaches_and_holds_its_speed_when_strong_enough() {
    let (mut world, base, wheel) = base_and_body(Vec2::ZERO);
    world.create_joint(
        base,
        wheel,
        Joint2d::Revolute(
            RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0).with_motor(5.0, 50.0),
        ),
    );
    run(&mut world, 0.5);
    for _ in 0..30 {
        world.step(DT);
        let speed = world.body(wheel).unwrap().angular_velocity;
        assert!(
            (speed - 5.0).abs() < 1e-3,
            "motor runs at {speed:.4} instead of 5 rad/s"
        );
    }
}

#[test]
fn motor_brake_slips_when_the_load_exceeds_its_torque() {
    // A 1 kg arm held horizontally 1 m from its pivot by a motor braking at speed 0. Gravity
    // pulls with 9.81 N·m about the arm's centre: a 12 N·m brake holds it, a 5 N·m brake slips
    // (the old 2.25x over-strength made 5 N·m behave like 11 N·m and hold).
    let drop = |max_torque: f32| {
        let mut world = PhysicsWorld2d::new();
        let pivot = world.create_body(RigidBody2d::new(BodyType::Static));
        let arm = world.create_body(
            RigidBody2d::new(BodyType::Dynamic)
                .with_position(Vec2::new(1.0, 0.0))
                .with_mass(1.0, 1.0),
        );
        world.create_joint(
            pivot,
            arm,
            Joint2d::Revolute(
                RevoluteJoint2d::new(Vec2::ZERO, Vec2::new(-1.0, 0.0), 0.0, 0.0)
                    .with_motor(0.0, max_torque),
            ),
        );
        run(&mut world, 1.0);
        -world.body(arm).unwrap().position.y
    };
    let held = drop(12.0);
    assert!(held < 0.01, "a 12 N·m brake let the arm drop {held:.4} m");
    let slipped = drop(5.0);
    assert!(
        slipped > 0.3,
        "a 5 N·m brake held the arm (dropped {slipped:.4} m)"
    );
}
