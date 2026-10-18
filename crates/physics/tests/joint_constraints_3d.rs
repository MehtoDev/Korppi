//! Integration test suite for 3D physics joint constraints:
//! Spherical (ball-and-socket), Revolute (hinge), Prismatic (slider), Distance, and Weld joints.

use korppi_math::{Quat, Vec3};
use korppi_physics::joint::{
    DistanceJoint, Joint, PrismaticJoint, RevoluteJoint, SphericalJoint, WeldJoint,
};
use korppi_physics::{BodyType, PhysicsWorld, RigidBody};

#[test]
fn spherical_joint_maintains_anchor_position_under_gravity() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Static anchor body
    let body_a = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, 5.0, 0.0),
        ..Default::default()
    });

    // Dynamic pendulum bob
    let body_b = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(2.0, 5.0, 0.0),
        mass: 2.0,
        ..Default::default()
    });

    // Spherical joint connecting anchor (local origin) to bob (local (-2, 0, 0))
    world.create_joint(Joint::Spherical(SphericalJoint::new(
        body_a,
        body_b,
        Vec3::ZERO,
        Vec3::new(-2.0, 0.0, 0.0),
    )));

    let dt = 1.0 / 60.0;
    for _ in 0..180 {
        world.step(dt);
    }

    let b_pos = world.body(body_b).unwrap().position;
    let dist_to_pivot = (b_pos - Vec3::new(0.0, 5.0, 0.0)).length();

    // The length of the pendulum arm must remain rigidly 2.0 meters
    assert!(
        (dist_to_pivot - 2.0).abs() < 0.05,
        "Spherical joint anchor drifted! Expected distance 2.0m, got {:.3}m",
        dist_to_pivot
    );
}

#[test]
fn revolute_joint_constrains_rotation_to_single_axis() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    let body_a = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::ZERO,
        ..Default::default()
    });

    let body_b = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(1.0, 0.0, 0.0),
        mass: 1.0,
        ..Default::default()
    });

    // Hinge joint aligned with Z axis
    world.create_joint(Joint::Revolute(RevoluteJoint::new(
        body_a,
        body_b,
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    )));

    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world
            .body_mut(body_b)
            .unwrap()
            .apply_torque(Vec3::new(50.0, 50.0, 20.0));
        world.step(dt);
    }

    let b = world.body(body_b).unwrap();
    // Angular velocity around X and Y must be constrained to ~0; only Z should spin freely
    assert!(
        b.angular_velocity.x.abs() < 0.1,
        "Revolute joint leaked angular X velocity: {:.3}",
        b.angular_velocity.x
    );
    assert!(
        b.angular_velocity.y.abs() < 0.1,
        "Revolute joint leaked angular Y velocity: {:.3}",
        b.angular_velocity.y
    );
    assert!(
        b.angular_velocity.z.abs() > 1.0,
        "Revolute joint should allow free rotation around Z, got {:.3}",
        b.angular_velocity.z
    );
}

#[test]
fn revolute_joint_motor_drives_angular_velocity() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    let body_a = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::ZERO,
        ..Default::default()
    });

    let body_b = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(1.0, 0.0, 0.0),
        mass: 1.0,
        ..Default::default()
    });

    let mut rev = RevoluteJoint::new(
        body_a,
        body_b,
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    rev.enable_motor = true;
    rev.motor_speed = 5.0; // 5 rad/s
    rev.max_motor_torque = 50.0;

    world.create_joint(Joint::Revolute(rev));

    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    let b = world.body(body_b).unwrap();
    assert!(
        (b.angular_velocity.z - 5.0).abs() < 0.2,
        "Revolute motor should reach target speed 5.0 rad/s, got {:.3}",
        b.angular_velocity.z
    );
}

#[test]
fn prismatic_joint_constrains_translation_to_slider_axis() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let body_a = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::ZERO,
        ..Default::default()
    });

    let body_b = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(1.0, 0.0, 0.0),
        mass: 1.0,
        ..Default::default()
    });

    // Slider along X axis: gravity pulls in -Y, but Y and Z should stay locked at 0
    world.create_joint(Joint::Prismatic(PrismaticJoint::new(
        body_a,
        body_b,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
    )));

    // Push along X
    world
        .body_mut(body_b)
        .unwrap()
        .apply_force(Vec3::new(20.0, 0.0, 0.0));

    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    let b = world.body(body_b).unwrap();
    assert!(
        b.position.y.abs() < 0.05,
        "Prismatic joint sagged under gravity! Y position was {:.3}",
        b.position.y
    );
    assert!(
        b.position.z.abs() < 0.05,
        "Prismatic joint drifted on Z axis! Z position was {:.3}",
        b.position.z
    );
    assert!(
        b.position.x > 1.2,
        "Prismatic joint should slide along X, pos.x was {:.3}",
        b.position.x
    );
}

#[test]
fn distance_joint_preserves_target_distance() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let body_a = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, 10.0, 0.0),
        ..Default::default()
    });

    let body_b = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(0.0, 6.0, 0.0),
        mass: 3.0,
        ..Default::default()
    });

    // Target distance = 4.0 meters
    world.create_joint(Joint::Distance(DistanceJoint::new(
        body_a,
        body_b,
        Vec3::ZERO,
        Vec3::ZERO,
        4.0,
    )));

    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step(dt);
    }

    let b = world.body(body_b).unwrap();
    let current_dist = (b.position - Vec3::new(0.0, 10.0, 0.0)).length();
    assert!(
        (current_dist - 4.0).abs() < 0.05,
        "Distance joint violated constraint! Expected 4.0m, got {:.3}m",
        current_dist
    );
}

#[test]
fn weld_joint_locks_relative_transform_rigidly() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    let body_a = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        ..Default::default()
    });

    let body_b = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(0.0, 2.0, 0.0),
        rotation: Quat::IDENTITY,
        mass: 1.0,
        ..Default::default()
    });

    world.create_joint(Joint::Weld(WeldJoint::new(
        body_a,
        body_b,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    )));

    // Apply high force and torque trying to displace and rotate body B
    world
        .body_mut(body_b)
        .unwrap()
        .apply_force(Vec3::new(100.0, -100.0, 50.0));
    world
        .body_mut(body_b)
        .unwrap()
        .apply_torque(Vec3::new(40.0, 20.0, -30.0));

    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    let b = world.body(body_b).unwrap();
    // B must remain welded at (0, 2, 0) with identity orientation
    let pos_err = (b.position - Vec3::new(0.0, 2.0, 0.0)).length();
    let rot_err =
        (b.rotation.w - 1.0).abs() + b.rotation.x.abs() + b.rotation.y.abs() + b.rotation.z.abs();

    assert!(
        pos_err < 0.05,
        "Weld joint linear displacement exceeded tolerance: {:.3}",
        pos_err
    );
    assert!(
        rot_err < 0.05,
        "Weld joint angular deviation exceeded tolerance: {:.3}",
        rot_err
    );
}

#[test]
fn test_3d_joint_warm_start_restores_constraint_velocity_prior_to_solve() {
    let mut body_a = RigidBody {
        body_type: BodyType::Dynamic,
        mass: 2.0,
        inv_mass: 0.5,
        ..Default::default()
    };
    let mut body_b = RigidBody {
        body_type: BodyType::Dynamic,
        mass: 2.0,
        inv_mass: 0.5,
        ..Default::default()
    };

    // 1. SphericalJoint
    let mut spherical = SphericalJoint::new(0, 1, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0));
    spherical.linear_impulse = Vec3::new(10.0, -5.0, 2.0);
    spherical.warm_start(&mut body_a, &mut body_b);
    assert_eq!(body_a.linear_velocity, Vec3::new(-5.0, 2.5, -1.0));
    assert_eq!(body_b.linear_velocity, Vec3::new(5.0, -2.5, 1.0));

    // Reset velocities
    body_a.linear_velocity = Vec3::ZERO;
    body_b.linear_velocity = Vec3::ZERO;

    // 2. DistanceJoint
    let mut dist_joint = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 5.0);
    body_a.position = Vec3::ZERO;
    body_b.position = Vec3::new(5.0, 0.0, 0.0);
    dist_joint.impulse = 4.0;
    dist_joint.warm_start(&mut body_a, &mut body_b);
    assert_eq!(body_a.linear_velocity, Vec3::new(-2.0, 0.0, 0.0));
    assert_eq!(body_b.linear_velocity, Vec3::new(2.0, 0.0, 0.0));

    // Reset velocities
    body_a.linear_velocity = Vec3::ZERO;
    body_b.linear_velocity = Vec3::ZERO;

    // 3. WeldJoint
    let mut weld = WeldJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO);
    weld.linear_impulse = Vec3::new(2.0, 4.0, 6.0);
    weld.warm_start(&mut body_a, &mut body_b);
    assert_eq!(body_a.linear_velocity, Vec3::new(-1.0, -2.0, -3.0));
    assert_eq!(body_b.linear_velocity, Vec3::new(1.0, 2.0, 3.0));

    // 4. Joint enum dispatch
    body_a.linear_velocity = Vec3::ZERO;
    body_b.linear_velocity = Vec3::ZERO;
    let joint_enum = Joint::Weld(weld);
    joint_enum.warm_start(&mut body_a, &mut body_b);
    assert_eq!(body_a.linear_velocity, Vec3::new(-1.0, -2.0, -3.0));
    assert_eq!(body_b.linear_velocity, Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_world_joints_querying() {
    let mut world = PhysicsWorld::new();
    let b1 = world.create_body(RigidBody::default());
    let b2 = world.create_body(RigidBody::default());
    let j = Joint::Weld(WeldJoint::new(b1, b2, Vec3::ZERO, Vec3::ZERO));
    world.create_joint(j);

    assert_eq!(world.joint_count(), 1);
    assert_eq!(world.joints().len(), 1);
}
