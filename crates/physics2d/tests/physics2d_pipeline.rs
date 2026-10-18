//! Integration tests for 2D physics simulation pipeline, ECS integration, and determinism.

use korppi_ecs::{Schedule, Stage, World};
use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{DistanceJoint2d, Joint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::PhysicsWorld2d;

#[test]
fn circle_settles_on_ground_box_in_physics_world() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    // Static ground box at y = 0.0 with half-height 1.0 (top surface at y = 1.0)
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 1.0)))
            .with_friction(0.4)
            .with_restitution(0.0),
    );

    // Dynamic circle dropping from y = 4.0 with radius 0.5
    let mut ball_body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 4.0));
    ball_body.set_mass(1.0, 0.25);
    let ball = world.create_body(ball_body);
    world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5)))
            .with_friction(0.4)
            .with_restitution(0.0),
    );

    // Step physics for 120 ticks (2 seconds at 60Hz)
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step(dt);
    }

    // Top of ground is y = 1.0, circle radius is 0.5 -> resting position y ~ 1.5
    let final_pos = world.body(ball).unwrap().position;
    assert!(
        (final_pos.y - 1.5).abs() < 0.05,
        "Expected resting y ~ 1.5, got {:.3}",
        final_pos.y
    );
    assert!(
        world.body(ball).unwrap().linear_velocity.length() < 0.1,
        "Expected ball velocity to settle near zero, got {:.3}",
        world.body(ball).unwrap().linear_velocity.length()
    );
}

#[test]
fn distance_joint_maintains_length_in_physics_world() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    // Static ceiling anchor at (0, 10)
    let anchor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 10.0)));

    // Dynamic bob at (0, 6)
    let mut bob_body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 6.0));
    bob_body.set_mass(2.0, 1.0);
    let bob = world.create_body(bob_body);

    // Distance joint with length 4.0
    world.create_joint(
        anchor,
        bob,
        Joint2d::Distance(DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 4.0, 60.0, 1.0)),
    );

    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    let pos_a = world.body(anchor).unwrap().position;
    let pos_b = world.body(bob).unwrap().position;
    let dist = (pos_b - pos_a).length();
    assert!(
        (dist - 4.0).abs() < 0.05,
        "Distance joint expected length 4.0, got {:.3}",
        dist
    );
}

#[test]
fn ecs_schedule_fixed_update_integrates_physics2d() {
    let mut ecs_world = World::new();
    let mut schedule = Schedule::new();

    // Setup ECS entities
    let ground_body = RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0));
    let ground_col = Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 1.0)));
    let _ground_entity = ecs_world.spawn((ground_body, ground_col));

    let mut drop_body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 5.0));
    drop_body.set_mass(1.0, 0.25);
    let drop_col = Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5)));
    let _drop_entity = ecs_world.spawn((drop_body, drop_col));

    // Register a physics stepping system in FixedUpdate
    let mut phys_world = PhysicsWorld2d::new();
    phys_world.set_gravity(Vec2::new(0.0, -9.81));

    schedule.add_system(Stage::FixedUpdate, move |world: &mut World| {
        // Query dynamic bodies and step physics
        let dt = 1.0 / 60.0;
        let mut query = world.query_mut::<&mut RigidBody2d>();
        for body in &mut query {
            if body.body_type == BodyType::Dynamic {
                body.integrate_velocities(Vec2::new(0.0, -9.81), dt);
                body.integrate_positions(dt);
            }
        }
    });

    for _ in 0..10 {
        schedule.run_stage(Stage::FixedUpdate, &mut ecs_world);
    }

    let query = ecs_world.query_mut::<&RigidBody2d>();
    let bodies: Vec<&RigidBody2d> = query.collect();
    assert_eq!(bodies.len(), 2);

    let updated_drop = bodies
        .into_iter()
        .find(|b| b.body_type == BodyType::Dynamic)
        .unwrap();
    assert!(
        updated_drop.position.y < 5.0,
        "Dynamic body should fall under gravity in ECS FixedUpdate stage"
    );
}

#[test]
fn physics2d_simulation_is_strictly_deterministic_across_runs() {
    let run_simulation = || -> Vec<(Vec2, f32, Vec2)> {
        let mut world = PhysicsWorld2d::new();
        world.set_gravity(Vec2::new(0.0, -9.81));

        // Ground
        let ground = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
        world.create_collider(
            ground,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(50.0, 1.0)))
                .with_friction(0.4)
                .with_restitution(0.1),
        );

        // Stack of 3 boxes
        let mut body_ids = Vec::new();
        for i in 0..3 {
            let mut box_body = RigidBody2d::new(BodyType::Dynamic)
                .with_position(Vec2::new(0.0, 2.0 + (i as f32) * 1.5));
            box_body.set_mass(2.0, 0.8);
            let b_id = world.create_body(box_body);
            world.create_collider(
                b_id,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
                    .with_friction(0.4)
                    .with_restitution(0.1),
            );
            body_ids.push(b_id);
        }

        // Simulate 120 steps
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step(dt);
        }

        body_ids
            .into_iter()
            .map(|id| {
                let b = world.body(id).unwrap();
                (b.position, b.rotation.angle(), b.linear_velocity)
            })
            .collect()
    };

    let run1 = run_simulation();
    let run2 = run_simulation();

    assert_eq!(run1.len(), 3);
    assert_eq!(
        run1, run2,
        "Physics 2D simulation must be bitwise identical across multiple runs"
    );
}

#[test]
fn world_snapshot_roundtrip_restores_exact_state() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 1.0))),
    );

    let mut body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 5.0));
    body.set_mass(1.0, 0.5);
    let ball = world.create_body(body);
    world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );

    // Step 30 ticks
    let dt = 1.0 / 60.0;
    for _ in 0..30 {
        world.step(dt);
    }

    // Take snapshot
    let snapshot = world.create_snapshot();

    // Step 30 more ticks
    for _ in 0..30 {
        world.step(dt);
    }
    let state_at_60 = (
        world.body(ball).unwrap().position,
        world.body(ball).unwrap().linear_velocity,
    );

    // Restore snapshot back to step 30
    world.restore_snapshot(&snapshot);

    // Step 30 ticks again
    for _ in 0..30 {
        world.step(dt);
    }
    let replayed_state_at_60 = (
        world.body(ball).unwrap().position,
        world.body(ball).unwrap().linear_velocity,
    );

    assert_eq!(
        state_at_60, replayed_state_at_60,
        "Simulation state after restoring snapshot must match original run exactly"
    );
}
