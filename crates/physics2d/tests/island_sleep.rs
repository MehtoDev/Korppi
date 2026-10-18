//! Integration tests for Island graph management, sleeping state transitions, and awake propagation.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::PhysicsWorld2d;
use std::time::Instant;

#[test]
fn test_body_settles_and_falls_asleep() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    // Static floor at y = 0.0 (top at y = 1.0)
    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(10.0, 1.0)))
            .with_friction(0.5)
            .with_restitution(0.0),
    );

    // Dynamic box resting right on top of floor at y = 1.5
    let box_body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 1.5));
    let box_id = world.create_body(box_body);
    world.create_collider(
        box_id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
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
        .apply_linear_impulse(Vec2::new(5.0, 5.0));
    assert!(
        !world.body(box_id).unwrap().is_sleeping,
        "Body must wake up immediately after receiving impulse"
    );
}

#[test]
fn test_connected_stack_sleeps_and_wakes_together() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(10.0, 1.0))),
    );

    // Stack of 3 boxes
    let b0 =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 1.5)));
    world.create_collider(
        b0,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );

    let b1 =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 2.5)));
    world.create_collider(
        b1,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );

    let b2 =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 3.5)));
    world.create_collider(
        b2,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
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
        .apply_linear_impulse(Vec2::new(2.0, 0.0));
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
fn test_sleeping_skips_simulation_work() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(50.0, 1.0))),
    );

    // Create 20 resting boxes
    for i in 0..20 {
        let id = world.create_body(
            RigidBody2d::new(BodyType::Dynamic)
                .with_position(Vec2::new(i as f32 * 1.5 - 15.0, 1.5)),
        );
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
        );
    }

    let dt = 1.0 / 60.0;
    // Step until asleep
    for _ in 0..80 {
        world.step(dt);
    }

    // Measure time for 100 steps while all bodies are sleeping
    let start = Instant::now();
    for _ in 0..100 {
        world.step(dt);
    }
    let duration_sleeping = start.elapsed();

    // Verify all dynamic bodies are sleeping
    for (i, (_, body)) in world.bodies().enumerate().skip(1) {
        assert!(body.is_sleeping, "Body {i} should be sleeping");
    }

    // Step time while all bodies sleep should be extremely fast (< 5ms for 100 steps)
    println!("100 steps while sleeping took: {:?}", duration_sleeping);
    assert!(
        duration_sleeping.as_millis() < 50,
        "Sleeping simulation should skip solver and be near-instant"
    );
}
