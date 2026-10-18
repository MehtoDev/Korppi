//! Integration tests for bitwise deterministic 3D world snapshot serialization and rollback.

use korppi_math::{Quat, Vec3};
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::shape::Shape;
use korppi_physics::snapshot::{BodySnapshot3d, WorldSnapshot3d};
use korppi_physics::PhysicsWorld;

#[test]
fn body_snapshot_3d_captures_and_serializes_state() {
    let mut body = RigidBody::new(BodyType::Dynamic, Vec3::new(1.0, 2.0, 3.0));
    body.rotation = Quat::from_axis_angle(Vec3::Y, 0.5);
    body.linear_velocity = Vec3::new(4.0, 5.0, 6.0);
    body.angular_velocity = Vec3::new(0.1, 0.2, 0.3);
    body.is_sleeping = true;

    let snapshot = BodySnapshot3d::from_body(42, &body);
    assert_eq!(snapshot.id, 42);
    assert_eq!(snapshot.position, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(snapshot.rotation, body.rotation);
    assert_eq!(snapshot.linear_velocity, Vec3::new(4.0, 5.0, 6.0));
    assert_eq!(snapshot.angular_velocity, Vec3::new(0.1, 0.2, 0.3));
    assert!(snapshot.is_sleeping);

    // Test serde round-trip
    let json = serde_json::to_string(&snapshot).expect("Failed to serialize BodySnapshot3d");
    let deserialized: BodySnapshot3d =
        serde_json::from_str(&json).expect("Failed to deserialize BodySnapshot3d");
    assert_eq!(snapshot, deserialized);
}

#[test]
fn world_snapshot_3d_rollback_produces_bit_exact_simulation() {
    let mut world = PhysicsWorld::new();

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        }),
    );

    let mut b1_def = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 3.0, 0.0));
    b1_def.mass = 2.0;
    b1_def.inv_mass = 0.5;
    let b1 = world.create_body(b1_def);
    world.create_collider(
        b1,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        }),
    );

    let mut b2_def = RigidBody::new(BodyType::Dynamic, Vec3::new(0.1, 5.0, 0.0));
    b2_def.mass = 1.0;
    b2_def.inv_mass = 1.0;
    let b2 = world.create_body(b2_def);
    world.create_collider(b2, Collider::new(Shape::Sphere { radius: 0.5 }));

    let dt = 1.0 / 60.0;

    // Step 20 ticks
    for _ in 0..20 {
        world.step(dt);
    }
    assert_eq!(world.step_index(), 20);

    // Capture checkpoint snapshot
    let snapshot: WorldSnapshot3d = world.create_snapshot();
    assert_eq!(snapshot.step_index, 20);
    assert_eq!(snapshot.bodies.len(), 3);

    // Record baseline: step 40 more ticks (to tick 60)
    for _ in 0..40 {
        world.step(dt);
    }
    assert_eq!(world.step_index(), 60);

    let b1_pos_baseline = world.body(b1).unwrap().position;
    let b1_rot_baseline = world.body(b1).unwrap().rotation;
    let b1_vel_baseline = world.body(b1).unwrap().linear_velocity;
    let b1_ang_baseline = world.body(b1).unwrap().angular_velocity;

    let b2_pos_baseline = world.body(b2).unwrap().position;
    let b2_rot_baseline = world.body(b2).unwrap().rotation;
    let b2_vel_baseline = world.body(b2).unwrap().linear_velocity;
    let b2_ang_baseline = world.body(b2).unwrap().angular_velocity;

    // Rollback to checkpoint at tick 20
    world.restore_snapshot(&snapshot);
    assert_eq!(world.step_index(), 20);

    let b1_pos_rewound = world.body(b1).unwrap().position;
    assert_eq!(
        b1_pos_rewound,
        snapshot.bodies[b1.index() as usize].position
    );

    // Re-simulate 40 ticks forward (to tick 60)
    for _ in 0..40 {
        world.step(dt);
    }
    assert_eq!(world.step_index(), 60);

    // Verify bitwise determinism
    let b1_pos_resim = world.body(b1).unwrap().position;
    let b1_rot_resim = world.body(b1).unwrap().rotation;
    let b1_vel_resim = world.body(b1).unwrap().linear_velocity;
    let b1_ang_resim = world.body(b1).unwrap().angular_velocity;

    let b2_pos_resim = world.body(b2).unwrap().position;
    let b2_rot_resim = world.body(b2).unwrap().rotation;
    let b2_vel_resim = world.body(b2).unwrap().linear_velocity;
    let b2_ang_resim = world.body(b2).unwrap().angular_velocity;

    fn vec3_bits(v: Vec3) -> [u32; 3] {
        [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
    }
    fn quat_bits(q: Quat) -> [u32; 4] {
        [q.x.to_bits(), q.y.to_bits(), q.z.to_bits(), q.w.to_bits()]
    }

    assert_eq!(vec3_bits(b1_pos_baseline), vec3_bits(b1_pos_resim));
    assert_eq!(quat_bits(b1_rot_baseline), quat_bits(b1_rot_resim));
    assert_eq!(vec3_bits(b1_vel_baseline), vec3_bits(b1_vel_resim));
    assert_eq!(vec3_bits(b1_ang_baseline), vec3_bits(b1_ang_resim));

    assert_eq!(vec3_bits(b2_pos_baseline), vec3_bits(b2_pos_resim));
    assert_eq!(quat_bits(b2_rot_baseline), quat_bits(b2_rot_resim));
    assert_eq!(vec3_bits(b2_vel_baseline), vec3_bits(b2_vel_resim));
    assert_eq!(vec3_bits(b2_ang_baseline), vec3_bits(b2_ang_resim));
}

use korppi_physics::joint::{Joint, RevoluteJoint};
use korppi_physics::BodyId3d;

fn test_scene() -> (PhysicsWorld, Vec<BodyId3d>, korppi_physics::JointId3d) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(30.0, 0.5, 30.0),
        })
        .with_friction(0.6),
    );

    let mut bodies = Vec::new();
    // 1. Stack of boxes
    for i in 0..4 {
        let b = world.create_body(RigidBody::new(
            BodyType::Dynamic,
            Vec3::new(0.0, 0.5 + i as f32, 0.0),
        ));
        world.create_collider(
            b,
            Collider::new(Shape::Box {
                half_extents: Vec3::splat(0.5),
            })
            .with_friction(0.6)
            .with_mass(1.0),
        );
        bodies.push(b);
    }

    // 2. Resting box far away that will fall asleep
    let sleeper = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(10.0, 0.5, 0.0)));
    world.create_collider(
        sleeper,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.6)
        .with_mass(1.0),
    );
    bodies.push(sleeper);

    // 3. Pendulum with revolute joint
    let pivot = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(-5.0, 10.0, 0.0)));
    let bob = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(-5.0, 8.0, 0.0)));
    world.create_collider(
        bob,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(2.0),
    );
    bodies.push(bob);

    let hinge = RevoluteJoint::new(pivot, bob, Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0), Vec3::Z)
        .with_spring(10.0, 1.0)
        .with_target_angle(0.5);
    let joint_id = world.create_joint(Joint::Revolute(hinge));

    // 4. Sensor with visitor
    let sensor = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(5.0, 1.0, 0.0)));
    world.create_collider(
        sensor,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        })
        .with_sensor(true)
        .with_sensor_events(true),
    );
    let visitor = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(5.0, 1.0, 0.0)));
    world.create_collider(
        visitor,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_sensor_events(true),
    );
    bodies.push(visitor);

    (world, bodies, joint_id)
}

fn state_bits(world: &PhysicsWorld, bodies: &[BodyId3d]) -> Vec<u32> {
    bodies
        .iter()
        .flat_map(|&id| {
            let b = world.body(id).unwrap();
            [
                b.position.x,
                b.position.y,
                b.position.z,
                b.rotation.x,
                b.rotation.y,
                b.rotation.z,
                b.rotation.w,
                b.linear_velocity.x,
                b.linear_velocity.y,
                b.linear_velocity.z,
                b.angular_velocity.x,
                b.angular_velocity.y,
                b.angular_velocity.z,
            ]
            .map(f32::to_bits)
        })
        .collect()
}

#[test]
fn a_restored_world_replays_bit_for_bit_with_sleeping_bodies_and_joints() {
    let dt = 1.0 / 60.0;
    let (mut world, bodies, _joint) = test_scene();

    // Step 20 ticks
    for _ in 0..20 {
        world.step(dt);
    }
    let hash_20 = world.state_hash();
    let snapshot = world.create_snapshot();
    assert_eq!(snapshot.state_hash(), hash_20);

    // Step 60 more ticks to tick 80
    for _ in 0..60 {
        world.step(dt);
    }
    let first = state_bits(&world, &bodies);
    let hash_80 = world.state_hash();
    assert_ne!(hash_20, hash_80);
    // The sleeper box at index 4 is resting on the ground
    assert!(
        world.body(bodies[4]).unwrap().is_sleeping,
        "the resting box sleeps"
    );

    // Rollback to tick 20
    world.restore_snapshot(&snapshot);
    assert_eq!(world.state_hash(), hash_20, "restored to step 20");

    // Resimulate 60 ticks
    for _ in 0..60 {
        world.step(dt);
    }
    assert_eq!(state_bits(&world, &bodies), first, "replay diverged");
    assert_eq!(world.state_hash(), hash_80);
}

#[test]
fn a_serialized_snapshot_builds_a_world_that_replays_bit_for_bit() {
    let dt = 1.0 / 60.0;
    let (mut world, bodies, _joint) = test_scene();

    for _ in 0..20 {
        world.step(dt);
    }
    let json = serde_json::to_string(&world.create_snapshot()).expect("serialize snapshot");

    for _ in 0..60 {
        world.step(dt);
    }
    let original_bits = state_bits(&world, &bodies);
    let original_hash = world.state_hash();

    let snapshot: WorldSnapshot3d = serde_json::from_str(&json).expect("deserialize snapshot");
    let mut copy = PhysicsWorld::from_snapshot(&snapshot);
    assert_eq!(copy.state_hash(), snapshot.state_hash());

    for _ in 0..60 {
        copy.step(dt);
    }
    assert_eq!(state_bits(&copy, &bodies), original_bits);
    assert_eq!(copy.state_hash(), original_hash);
}

#[test]
fn restoring_keeps_held_handles_and_invalidates_later_created_handles() {
    let dt = 1.0 / 60.0;
    let (mut world, bodies, joint) = test_scene();

    for _ in 0..10 {
        world.step(dt);
    }
    let snapshot = world.create_snapshot();

    // Destroy a body and a joint after snapshot
    assert!(world.destroy_body(bodies[0]));
    assert!(world.destroy_joint(joint).is_some());
    let late_body = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 50.0, 0.0)));
    let late_joint = world.create_joint(Joint::Spherical(
        korppi_physics::joint::SphericalJoint::new(bodies[1], bodies[2], Vec3::ZERO, Vec3::ZERO),
    ));

    // Restore
    world.restore_snapshot(&snapshot);

    // Old handles must be alive again
    assert!(
        world.body(bodies[0]).is_some(),
        "destroyed after snapshot, restored"
    );
    assert!(world.joint(joint).is_some(), "destroyed joint restored");

    // Later created handles must be stale
    assert!(
        world.body(late_body).is_none(),
        "created after snapshot must be stale"
    );
    assert!(
        world.joint(late_joint).is_none(),
        "created after snapshot must be stale"
    );
}

#[test]
fn the_state_hash_tells_diverging_worlds_apart() {
    let dt = 1.0 / 60.0;
    let (mut a, bodies, _) = test_scene();
    let (mut b, _, _) = test_scene();

    assert_eq!(
        a.state_hash(),
        b.state_hash(),
        "identical worlds hash alike"
    );

    for _ in 0..10 {
        a.step(dt);
        b.step(dt);
    }
    assert_eq!(a.state_hash(), b.state_hash());

    // Smallest change to linear velocity changes the hash
    let body = b.body_mut(bodies[0]).unwrap();
    body.linear_velocity.x = f32::from_bits(body.linear_velocity.x.to_bits() + 1);
    assert_ne!(a.state_hash(), b.state_hash());
}
