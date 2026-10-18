//! Integration tests for Continuous Collision Detection (CCD) preventing tunneling.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::narrowphase::TimeOfImpact2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::PhysicsWorld2d;

#[test]
fn test_conservative_advancement_finds_exact_toi() {
    let circle = Circle2d::new(Vec2::ZERO, 0.5);
    let box_wall = Polygon2d::from_box(0.1, 5.0);

    // Bullet circle starting at x = -5.0, moving right at 50 m/s
    // Thin wall at x = 0.0 (half-extents: hx = 0.1, so left edge is at x = -0.1)
    // Circle radius is 0.5, so circle touches wall when center is at x = -0.1 - 0.5 = -0.6
    // Distance to cover = -0.6 - (-5.0) = 4.4 meters.
    // Velocity = 50 m/s. Expected TOI fraction in 1 second dt (velocity = 50) = 4.4 / 50.0 = 0.088
    let xf_circle = korppi_math::Transform2d::from_translation(Vec2::new(-5.0, 0.0));
    let xf_wall = korppi_math::Transform2d::from_translation(Vec2::new(0.0, 0.0));

    let toi = TimeOfImpact2d::calculate(
        &Shape2d::Circle(circle),
        &xf_circle,
        Vec2::new(50.0, 0.0),
        &Shape2d::Polygon(box_wall),
        &xf_wall,
        Vec2::ZERO,
    );

    assert!(toi.is_some(), "Conservative advancement must detect impact");
    let t = toi.unwrap();
    assert!(
        (t - 0.088).abs() < 0.01,
        "Expected TOI fraction ~0.088, got {}",
        t
    );
}

#[test]
fn test_fast_bullet_does_not_tunnel_through_thin_wall() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::ZERO);

    // Thin static wall at x = 0.0, width = 0.2 (hx = 0.1), height = 10.0
    let wall =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        wall,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.1, 5.0)))
            .with_friction(0.0)
            .with_restitution(0.0),
    );

    // Fast bullet moving at 120 m/s towards the wall from x = -2.0
    // In one tick (dt = 1/60s), discrete motion would travel 2.0 meters,
    // placing it at x = 0.0 (inside or past the wall), and in 2 ticks at x = +2.0 (tunneled completely!)
    let bullet = RigidBody2d::new(BodyType::Dynamic)
        .with_position(Vec2::new(-2.0, 0.0))
        .with_linear_velocity(Vec2::new(120.0, 0.0))
        .with_bullet(true);

    let bullet_id = world.create_body(bullet);
    world.create_collider(
        bullet_id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.2)))
            .with_friction(0.0)
            .with_restitution(0.0),
    );

    // Step world forward
    let dt = 1.0 / 60.0;
    for _ in 0..5 {
        world.step(dt);
    }

    let final_pos = world.body(bullet_id).unwrap().position;
    // Wall left edge is at x = -0.1. Circle radius is 0.2.
    // Circle should stop in front of the wall (x <= -0.1) and NEVER cross to positive x!
    assert!(
        final_pos.x <= 0.0,
        "Bullet tunneled through thin wall! final_pos.x = {}",
        final_pos.x
    );
}

#[test]
fn test_rotating_bullet_detects_angular_impact() {
    let rod = Polygon2d::from_box(0.8, 0.05);
    let obstacle = Polygon2d::from_box(0.1, 0.2);

    let xf_rod = korppi_math::Transform2d::from_translation(Vec2::new(-2.0, 0.0));
    let xf_obstacle = korppi_math::Transform2d::from_translation(Vec2::new(0.0, 0.5));

    // Rod moves horizontally towards obstacle: disp = (2.0, 0.0).
    // Without rotation (delta_angle = 0.0), y-extent is [-0.05, 0.05], obstacle is at [0.3, 0.7].
    // Distance in y is > 0.25, so with delta_angle = 0.0 it must NOT collide:
    let toi_no_rot = TimeOfImpact2d::calculate_full(
        &Shape2d::Polygon(rod),
        &xf_rod,
        Vec2::new(2.0, 0.0),
        0.0,
        &Shape2d::Polygon(obstacle),
        &xf_obstacle,
        Vec2::ZERO,
        0.0,
    );
    assert!(
        toi_no_rot.is_none(),
        "Non-rotating rod should miss the obstacle above it"
    );

    // With rotation of 90 degrees (FRAC_PI_2), the rod swings vertically into [0.3, 0.7]!
    let toi_rot = TimeOfImpact2d::calculate_full(
        &Shape2d::Polygon(rod),
        &xf_rod,
        Vec2::new(2.0, 0.0),
        std::f32::consts::FRAC_PI_2,
        &Shape2d::Polygon(obstacle),
        &xf_obstacle,
        Vec2::ZERO,
        0.0,
    );
    assert!(toi_rot.is_some(), "Rotating rod must collide with obstacle");
    let (t, _normal) = toi_rot.unwrap();
    assert!(
        t > 0.0 && t <= 1.0,
        "Impact fraction must be within [0, 1], got {}",
        t
    );
}

#[test]
fn test_fast_spinning_bullet_does_not_tunnel_past_obstacle() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::ZERO);

    // Obstacle at x = 0.0, y = 0.5, half-extents (0.1, 0.2)
    let obstacle =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.5)));
    world.create_collider(
        obstacle,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.1, 0.2))),
    );

    // Bullet starting at x = -2.0, y = 0.0, horizontal rod (0.8, 0.05)
    // Moving at 60 m/s horizontally. In dt = 1/30, disp.x = 2.0.
    // Angular velocity omega = 15 * PI rad/s (spinning rapidly).
    let bullet = RigidBody2d::new(BodyType::Dynamic)
        .with_position(Vec2::new(-2.0, 0.0))
        .with_linear_velocity(Vec2::new(60.0, 0.0))
        .with_angular_velocity(15.0 * std::f32::consts::PI)
        .with_bullet(true);

    let bullet_id = world.create_body(bullet);
    world.create_collider(
        bullet_id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.8, 0.05))),
    );

    let dt = 1.0 / 30.0;
    for _ in 0..5 {
        world.step(dt);
    }

    let final_pos = world.body(bullet_id).unwrap().position;
    // Without angular CCD, it would have sailed right through to x > 5.0.
    // With angular CCD, it is stopped at or before the obstacle, whose far face is at x = 0.1.
    // Box2D v3.2 deflects the rod downward off the obstacle and ends at x = -0.61.
    assert!(
        final_pos.x <= 0.1,
        "Spinning bullet should be stopped before or at obstacle, but got x = {}",
        final_pos.x
    );
}
