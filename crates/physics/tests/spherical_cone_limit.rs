//! Integration tests for 3D SphericalJoint cone limit constraint.

use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::joint::{Joint, SphericalJoint};
use korppi_physics::PhysicsWorld;
use std::f32::consts::PI;

#[test]
fn spherical_joint_cone_limit_prevents_excessive_swing() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO); // isolate joint kinematics from gravity

    // Body A is static anchor
    let body_a = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));

    // Body B is dynamic body attached at anchor
    let mut body_b_def = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.0, 2.0));
    body_b_def.mass = 1.0;
    // Apply initial torque trying to swing B around X axis
    body_b_def.angular_velocity = Vec3::new(5.0, 0.0, 0.0);
    let body_b = world.create_body(body_b_def);

    // 30 degree cone limit (PI / 6.0 radians)
    let max_cone = PI / 6.0;
    let joint = SphericalJoint::new(
        body_a,
        body_b,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -2.0),
    )
    .with_cone_limit(max_cone);

    world.create_joint(Joint::Spherical(joint));

    // Step simulation
    let mut limit_pushed = false;
    let mut swing = 0.0;
    for _ in 0..100 {
        world.step(1.0 / 60.0);

        if let Some(Joint::Spherical(sj)) = world.joint(0) {
            swing =
                sj.current_swing_angle(world.body(body_a).unwrap(), world.body(body_b).unwrap());
            // Swing angle must never exceed the cone limit by more than a tiny constraint slop
            assert!(
                swing <= max_cone + 0.05,
                "Swing angle {} exceeded cone limit {}",
                swing,
                max_cone
            );
            limit_pushed |= sj.swing_impulse > 0.0;
        }
    }

    // Box3D runs this scene so: the body reaches the cone at step ~31 (largest swing 0.5246 rad),
    // the cone limit stops it and it rebounds at -0.175 rad/s, swinging back to 0.3268 rad by
    // step 100 with no cone torque left. So the limit must have pushed during the run, and the
    // body ends back inside the cone.
    assert!(
        limit_pushed,
        "Cone limit should have generated stopping impulse"
    );
    assert!(
        swing < max_cone,
        "the body should rebound into the cone, its swing is {swing}"
    );
}

#[test]
fn spherical_joint_free_swing_generates_zero_impulse_below_limit() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    let body_a = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));

    let mut body_b_def = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.0, 2.0));
    body_b_def.mass = 1.0;
    // Gentle rotation well below limit
    body_b_def.angular_velocity = Vec3::new(0.1, 0.0, 0.0);
    let body_b = world.create_body(body_b_def);

    // Large cone limit of 60 degrees (PI / 3.0)
    let max_cone = PI / 3.0;
    let joint = SphericalJoint::new(
        body_a,
        body_b,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -2.0),
    )
    .with_cone_limit(max_cone);

    world.create_joint(Joint::Spherical(joint));

    // Step 5 times: angle reaches ~0.008 rad << PI/3
    for _ in 0..5 {
        world.step(1.0 / 60.0);
    }

    if let Some(Joint::Spherical(sj)) = world.joint(0) {
        let swing =
            sj.current_swing_angle(world.body(body_a).unwrap(), world.body(body_b).unwrap());
        assert!(swing < max_cone);
        assert_eq!(
            sj.swing_impulse, 0.0,
            "Impulse must be zero when inside cone"
        );
    }
}
