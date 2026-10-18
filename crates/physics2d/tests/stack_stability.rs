//! Integration test for 2D vertical box stacking stability and warm-started persistent contacts.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::PhysicsWorld2d;

#[test]
fn test_ten_box_stack_stability_under_soft_step_solver() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    // Ground: Static box at y = 0.0 with half-height 1.0 (top surface at y = 1.0)
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 1.0)))
            .with_friction(0.6)
            .with_restitution(0.0),
    );

    // 10 vertical boxes: each 1.0 x 1.0 (half-extents 0.5, 0.5)
    // Box 0 rests at y = 1.5, Box 1 at y = 2.5, ..., Box 9 at y = 10.5
    let box_count = 10;
    let mut box_ids = Vec::new();

    for i in 0..box_count {
        let y = 1.5 + i as f32 * 1.0;
        let mut body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, y));
        // Mass = 1.0, inertia for 1x1 box = m * (w^2 + h^2) / 12 = 1.0 * (1 + 1) / 12 = 1/6
        body.set_mass(1.0, 1.0 / 6.0);
        let id = world.create_body(body);
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
                .with_friction(0.6)
                .with_restitution(0.0),
        );
        box_ids.push(id);
    }

    let dt = 1.0 / 60.0;
    // Simulate for 1,000 steps (~16.6 seconds of physical time)
    for _ in 0..1000 {
        world.step(dt);
    }

    // Check stack stability
    for (i, &id) in box_ids.iter().enumerate() {
        let b = world.body(id).unwrap();
        let expected_y = 1.5 + i as f32 * 1.0;

        assert!(
            b.position.x.abs() < 0.15,
            "Box {} drifted horizontally: x = {} (expected |x| < 0.15)",
            i,
            b.position.x
        );
        assert!(
            (b.position.y - expected_y).abs() < 0.2,
            "Box {} vertical drift: y = {} (expected ~{})",
            i,
            b.position.y,
            expected_y
        );
        assert!(
            b.linear_velocity.length() < 0.2,
            "Box {} excessive velocity: {:?} (expected near rest)",
            i,
            b.linear_velocity
        );
        assert!(
            b.angular_velocity.abs() < 0.2,
            "Box {} excessive rotation speed: {} rad/s",
            i,
            b.angular_velocity
        );
    }
}
