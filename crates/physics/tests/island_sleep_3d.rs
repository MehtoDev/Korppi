//! Integration tests for 3D Island graph management, sleeping state transitions, and awake propagation.

use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::shape::Shape;
use korppi_physics::world::PhysicsWorld;
use std::time::Instant;

#[test]
fn test_body_settles_and_falls_asleep() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Static floor at y = 0.0 (top surface at y = 0.5)
    let floor_body = RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 0.0));
    let floor_id = world.create_body(floor_body);
    world.create_collider(
        floor_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    // Dynamic box resting right on top at y = 1.0 (bottom surface at y = 0.5)
    let box_body = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0));
    let box_id = world.create_body(box_body);
    world.create_collider(
        box_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    assert!(
        !world.body(box_id).unwrap().is_sleeping,
        "Body must start awake"
    );

    // Step for 1 second (60 ticks): box should settle in ~0.2s and sleep after 0.5s of rest
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    assert!(
        world.body(box_id).unwrap().is_sleeping,
        "Body should be sleeping after resting for > 0.5s"
    );

    // Applying an impulse to the body must wake it up
    world
        .body_mut(box_id)
        .unwrap()
        .apply_linear_impulse(Vec3::new(5.0, 0.0, 0.0));
    assert!(
        !world.body(box_id).unwrap().is_sleeping,
        "Body must wake up immediately after receiving impulse"
    );
}

#[test]
fn test_connected_stack_sleeps_and_wakes_together() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let floor_id = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 0.0)));
    world.create_collider(
        floor_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    // Stack of 3 boxes at y = 1.0, y = 2.0, y = 3.0
    let b0 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0)));
    world.create_collider(
        b0,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 2.0, 0.0)));
    world.create_collider(
        b1,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 3.0, 0.0)));
    world.create_collider(
        b2,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    let dt = 1.0 / 60.0;
    for _ in 0..80 {
        world.step(dt);
    }

    assert!(world.body(b0).unwrap().is_sleeping, "b0 should sleep");
    assert!(world.body(b1).unwrap().is_sleeping, "b1 should sleep");
    assert!(world.body(b2).unwrap().is_sleeping, "b2 should sleep");

    // Waking b0 must propagate through contact graph to wake b1 and b2
    world
        .body_mut(b0)
        .unwrap()
        .apply_linear_impulse(Vec3::new(2.0, 0.0, 0.0));
    world.step(dt);

    assert!(!world.body(b0).unwrap().is_sleeping, "b0 must be awake");
    assert!(
        !world.body(b1).unwrap().is_sleeping,
        "b1 must be awake via island propagation"
    );
    assert!(
        !world.body(b2).unwrap().is_sleeping,
        "b2 must be awake via island propagation"
    );
}

#[test]
fn test_disjoint_islands_remain_independent() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let floor_id = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 0.0)));
    world.create_collider(
        floor_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    let b_left = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(-5.0, 1.0, 0.0)));
    world.create_collider(
        b_left,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    let b_right = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(5.0, 1.0, 0.0)));
    world.create_collider(
        b_right,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    let dt = 1.0 / 60.0;
    for _ in 0..70 {
        world.step(dt);
    }

    assert!(world.body(b_left).unwrap().is_sleeping);
    assert!(world.body(b_right).unwrap().is_sleeping);

    // Wake left body
    world
        .body_mut(b_left)
        .unwrap()
        .apply_linear_impulse(Vec3::new(0.0, 5.0, 0.0));
    world.step(dt);

    assert!(
        !world.body(b_left).unwrap().is_sleeping,
        "Left body must be awake"
    );
    assert!(
        world.body(b_right).unwrap().is_sleeping,
        "Right body must remain sleeping"
    );
}

#[test]
fn test_sleeping_skips_simulation_work() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let floor_id = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 0.0)));
    world.create_collider(
        floor_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        })
        .with_friction(0.5)
        .with_restitution(0.0),
    );

    // Create 15 resting boxes
    for i in 0..15 {
        let id = world.create_body(RigidBody::new(
            BodyType::Dynamic,
            Vec3::new((i as f32) * 2.0 - 15.0, 1.0, 0.0),
        ));
        world.create_collider(
            id,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            })
            .with_friction(0.5)
            .with_restitution(0.0),
        );
    }

    let dt = 1.0 / 60.0;
    for _ in 0..80 {
        world.step(dt);
    }

    for i in 1..=15 {
        assert!(
            world.body(i).unwrap().is_sleeping,
            "Body {i} should be sleeping"
        );
    }

    let start = Instant::now();
    for _ in 0..100 {
        world.step(dt);
    }
    let duration_sleeping = start.elapsed();

    println!("100 steps while sleeping took: {:?}", duration_sleeping);
    assert!(
        duration_sleeping.as_millis() < 50,
        "Sleeping simulation should skip solver and be near-instant"
    );
}
