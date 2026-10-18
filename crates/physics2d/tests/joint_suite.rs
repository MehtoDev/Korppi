//! Comprehensive integration tests for 2D joint suite:
//! - RevoluteJoint2d with angular limits and motor drive
//! - PrismaticJoint2d with slider axis, limits, and linear motor
//! - WheelJoint2d with suspension spring and wheel motor
//! - MotorJoint2d springs tracking a target frame

use korppi_math::{Rot2, Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::joint::{
    DistanceJoint2d, Joint2d, MotorJoint2d, MoverJoint2d, PogoJoint2d, PrismaticJoint2d,
    RevoluteJoint2d, WeldJoint2d, WheelJoint2d,
};
use korppi_physics2d::world::PhysicsWorld2d;
use std::f32::consts::PI;

#[test]
fn revolute_joint_angular_limits_prevent_over_rotation() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);

    let anchor = world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::ZERO));

    let mut arm = RigidBody2d::new(BodyType::Dynamic)
        .with_position(Vec2::new(2.0, 0.0))
        .with_mass(1.0, 1.0);
    // Apply a strong initial angular velocity trying to spin counter-clockwise
    arm.angular_velocity = 10.0;
    let arm_id = world.create_body(arm);

    // Limit angle between -PI/4 and +PI/4 (~ -0.785 to +0.785 rad)
    let rev = RevoluteJoint2d::new(Vec2::ZERO, Vec2::new(-2.0, 0.0), 60.0, 1.0)
        .with_limits(-PI / 4.0, PI / 4.0);
    world.create_joint(anchor, arm_id, Joint2d::Revolute(rev));

    // Simulate for 1 second
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }

    let arm_body = world.body(arm_id).unwrap();
    let angle = arm_body.rotation.angle();
    // Angle must be clamped at or very near upper limit PI/4 (0.785 rad)
    assert!(
        angle <= PI / 4.0 + 0.05,
        "Arm angle {:.3} rad exceeded upper limit {:.3} rad",
        angle,
        PI / 4.0
    );
    assert!(
        angle >= -PI / 4.0 - 0.05,
        "Arm angle {:.3} rad below lower limit",
        angle
    );
}

#[test]
fn revolute_joint_motor_drives_angular_velocity() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    // Sleep off, as in the Box2D reference: a shapeless body has no extent, so Box2D lets it
    // sleep however fast it spins.
    world.enable_sleep(false);

    let anchor = world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::ZERO));

    let disk = RigidBody2d::new(BodyType::Dynamic)
        .with_position(Vec2::ZERO)
        .with_mass(1.0, 2.0);
    let disk_id = world.create_body(disk);

    // Motor set to rotate at 5.0 rad/s with max torque 50.0 N*m
    let rev = RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 60.0, 1.0).with_motor(5.0, 50.0);
    world.create_joint(anchor, disk_id, Joint2d::Revolute(rev));

    // Simulate for 0.5 seconds
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    let disk_body = world.body(disk_id).unwrap();
    assert!(
        (disk_body.angular_velocity - 5.0).abs() < 0.15,
        "Disk angular velocity {:.3} should match motor speed 5.0 rad/s",
        disk_body.angular_velocity
    );
}

#[test]
fn prismatic_joint_locks_lateral_motion_and_limits_translation() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);

    let track = world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::ZERO));

    // Slider moves along X axis (1, 0)
    let slider = RigidBody2d::new(BodyType::Dynamic)
        .with_position(Vec2::new(1.0, 0.0))
        .with_mass(2.0, 1.0);
    let slider_id = world.create_body(slider);

    // Give slider a strong velocity pushing it along +X and try to knock it sideways along +Y
    world.body_mut(slider_id).unwrap().linear_velocity = Vec2::new(20.0, 10.0);
    world.body_mut(slider_id).unwrap().angular_velocity = 5.0;

    let prismatic = PrismaticJoint2d::new(
        Vec2::ZERO,
        Vec2::ZERO,
        Vec2::new(1.0, 0.0), // axis along X
        60.0,
        1.0,
    )
    .with_limits(0.0, 4.0); // max translation 4.0m
    world.create_joint(track, slider_id, Joint2d::Prismatic(prismatic));

    // Step simulation
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }

    let slider_body = world.body(slider_id).unwrap();
    // Lateral position Y must remain 0
    assert!(
        slider_body.position.y.abs() < 0.05,
        "Slider lateral position Y={:.4} must stay ~0",
        slider_body.position.y
    );
    // Rotation must remain 0
    assert!(
        slider_body.rotation.angle().abs() < 0.05,
        "Slider angle {:.4} must stay ~0",
        slider_body.rotation.angle()
    );
    // Translation X must not exceed upper limit 4.0m
    assert!(
        slider_body.position.x <= 4.05,
        "Slider translation X={:.3} should be clamped by limit 4.0m",
        slider_body.position.x
    );
    assert!(
        slider_body.position.x >= 3.9,
        "Slider should have pushed up against limit 4.0m, got {:.3}",
        slider_body.position.x
    );
}

#[test]
fn prismatic_joint_motor_drives_slider_along_axis() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);

    let rail = world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::ZERO));
    let cart = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::ZERO)
            .with_mass(2.0, 1.0),
    );

    let prismatic = PrismaticJoint2d::new(
        Vec2::ZERO,
        Vec2::ZERO,
        Vec2::new(0.0, 1.0), // Vertical Y axis
        60.0,
        1.0,
    )
    .with_motor(3.0, 100.0); // Drive up at 3.0 m/s with 100 N max force

    world.create_joint(rail, cart, Joint2d::Prismatic(prismatic));

    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    let cart_body = world.body(cart).unwrap();
    assert!(
        (cart_body.linear_velocity.y - 3.0).abs() < 0.15,
        "Cart vertical velocity {:.3} should approach motor speed 3.0 m/s",
        cart_body.linear_velocity.y
    );
    assert!(
        cart_body.linear_velocity.x.abs() < 0.05,
        "Lateral velocity should remain ~0"
    );
}

#[test]
fn wheel_joint_suspension_and_motor() {
    let mut world = PhysicsWorld2d::new();
    // Sleep off, as in the Box2D reference: a shapeless body has no extent, so Box2D lets it
    // sleep however fast it spins.
    world.enable_sleep(false);

    // Chassis fixed or falling
    let chassis =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 5.0)));

    let wheel = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 4.0)) // 1m below chassis
            .with_mass(1.0, 0.5),
    );

    // Vertical suspension axis (0, 1), spring at 10 Hz, damp 0.7, motor at 10 rad/s
    let wheel_joint = WheelJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::new(0.0, 1.0), 10.0, 0.7)
        .with_motor(10.0, 20.0);

    world.create_joint(chassis, wheel, Joint2d::Wheel(wheel_joint));

    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }

    let wheel_body = world.body(wheel).unwrap();
    // Wheel should maintain lateral constraint X ~ 0
    assert!(
        wheel_body.position.x.abs() < 0.05,
        "Wheel X={:.3} should stay centered on suspension axis",
        wheel_body.position.x
    );
    // Wheel motor should spin wheel towards 10.0 rad/s
    assert!(
        (wheel_body.angular_velocity - 10.0).abs() < 0.5,
        "Wheel angular velocity {:.3} should approach motor speed 10.0 rad/s",
        wheel_body.angular_velocity
    );
}

#[test]
fn motor_joint_tracks_target_relative_offsets() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);

    let base = world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::ZERO));

    // Follower initially at (0, 0)
    let follower = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::ZERO)
            .with_mass(1.0, 1.0),
    );

    // Motor joint pulling toward relative target (3.0, 2.0) and angle PI/2 with 4 Hz springs
    // limited to 100 N and 50 N·m. Since FIX-2.6 this is Box2D v3.2's motor joint (owner
    // decision; reference `fix_2_6_motor_joint_tests.c`): Box2D is 33 µm off after 1 s and on
    // target after 3 s, where v3.1.1's correction-factor joint was still 0.58 m off at 1.5 s.
    let target =
        Transform2d::from_translation_rotation(Vec2::new(3.0, 2.0), Rot2::from_angle(PI / 2.0));
    let motor_joint = MotorJoint2d::default()
        .with_local_frames(target, Transform2d::IDENTITY)
        .with_linear_spring(4.0, 0.7, 100.0)
        .with_angular_spring(4.0, 0.7, 50.0);
    world.create_joint(base, follower, Joint2d::Motor(motor_joint));

    for _ in 0..180 {
        world.step(1.0 / 60.0);
    }

    let follower_body = world.body(follower).unwrap();
    let pos_err = (follower_body.position - Vec2::new(3.0, 2.0)).length();
    let rot_err = (follower_body.rotation.angle() - PI / 2.0).abs();

    assert!(
        pos_err < 1e-3,
        "Follower position {:?} should track target (3.0, 2.0), err={:.4}",
        follower_body.position,
        pos_err
    );
    assert!(
        rot_err < 1e-3,
        "Follower angle {:.3} should track target PI/2 ({:.3}), err={:.4}",
        follower_body.rotation.angle(),
        PI / 2.0,
        rot_err
    );
}

#[test]
fn joint_enum_full_variant_coverage() {
    let dist = DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 1.0, 30.0, 1.0);
    let rev = RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 30.0, 1.0);
    let pris = PrismaticJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::new(1.0, 0.0), 30.0, 1.0);
    let wheel = WheelJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::new(0.0, 1.0), 30.0, 1.0);
    let motor = MotorJoint2d::default()
        .with_linear_spring(30.0, 1.0, 10.0)
        .with_angular_spring(30.0, 1.0, 10.0);
    let weld = WeldJoint2d::new(Vec2::ZERO, Vec2::ZERO, 30.0, 1.0);
    let mover = MoverJoint2d::new(Vec2::X, Vec2::new(10.0, 10.0));
    let pogo = PogoJoint2d::new(Vec2::Y, 1.0, 30.0, 1.0);

    let mut joints = vec![
        Joint2d::Distance(dist),
        Joint2d::Revolute(rev),
        Joint2d::Prismatic(pris),
        Joint2d::Wheel(wheel),
        Joint2d::Motor(motor),
        Joint2d::Weld(weld),
        Joint2d::Mover(mover),
        Joint2d::Pogo(pogo),
        Joint2d::Filter,
    ];

    let mut b1 = RigidBody2d::new(BodyType::Dynamic).with_mass(1.0, 1.0);
    let mut b2 = RigidBody2d::new(BodyType::Dynamic).with_mass(1.0, 1.0);

    for j in &mut joints {
        j.solve(&mut b1, &mut b2, 1.0 / 60.0);
    }
}

#[test]
fn test_2d_joint_warm_start_restores_constraint_velocity_at_substep_zero() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -9.81));
    let anchor = world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::ZERO));
    let body = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, -2.0))
            .with_mass(10.0, 10.0),
    );

    let dist = DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 2.0, 60.0, 1.0);
    let joint_id = world.create_joint(anchor, body, Joint2d::Distance(dist));

    // Step several times to establish steady tension
    for _ in 0..30 {
        world.step(1.0 / 60.0);
    }

    // Inspect the world's joint instance: its accumulated impulse must be positive
    let joint_inst = world.joint(joint_id).unwrap();
    if let Joint2d::Distance(d) = &joint_inst.joint {
        assert!(
            d.impulse.abs() > 0.1,
            "Distance joint must accumulate impulse under gravity"
        );
    } else {
        panic!("Expected Distance joint");
    }

    // At step boundary, warm start applies the impulse at step_i == 0.
    // Verify Joint2d::warm_start alters velocity directly.
    let mut dyn_body = world.body(body).unwrap().clone();
    let mut static_body = world.body(anchor).unwrap().clone();
    let v_before = dyn_body.linear_velocity;
    joint_inst.joint.warm_start(&mut static_body, &mut dyn_body);
    // Warm start must push the body upward against gravity
    assert!(
        dyn_body.linear_velocity.y > v_before.y,
        "Warm start must apply upward impulse to counteract gravity (before: {}, after: {})",
        v_before.y,
        dyn_body.linear_velocity.y
    );
}
