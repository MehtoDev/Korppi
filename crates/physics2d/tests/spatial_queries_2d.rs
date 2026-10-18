//! Integration tests for 2D spatial queries (ray cast, AABB query, point query) on PhysicsWorld2d.

use korppi_math::{Aabb2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::{PhysicsWorld2d, QueryFilter2d, RayHit2d};

#[test]
fn ray_cast_hits_2d_circle_with_accurate_normal_and_fraction() {
    let mut world = PhysicsWorld2d::new();

    let circle_body =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(10.0, 0.0)));
    world.create_collider(
        circle_body,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 2.0))),
    );

    let hit: RayHit2d = world
        .cast_ray_closest(Vec2::ZERO, Vec2::X * 20.0, QueryFilter2d::default())
        .expect("Ray should hit circle");

    assert_eq!(hit.body_id, circle_body);
    // Center at (10, 0), radius 2 -> hit at x = 8.0, 8 m of the ray's 20 m
    assert!((hit.fraction * 20.0 - 8.0).abs() < 1e-4);
    assert!((hit.point - Vec2::new(8.0, 0.0)).length() < 1e-4);
    assert!((hit.normal - Vec2::new(-1.0, 0.0)).length() < 1e-4);
}

#[test]
fn ray_cast_hits_2d_box_polygon() {
    let mut world = PhysicsWorld2d::new();

    let box_body =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 5.0)));
    world.create_collider(
        box_body,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.0, 1.0))),
    );

    // Ray shooting along +Y
    let hit: RayHit2d = world
        .cast_ray_closest(Vec2::ZERO, Vec2::Y * 10.0, QueryFilter2d::default())
        .expect("Ray should hit box");

    assert_eq!(hit.body_id, box_body);
    // Center at (0, 5), half-extents 1.0 -> bottom face at y = 4.0, 4 m of the ray's 10 m
    assert!((hit.fraction * 10.0 - 4.0).abs() < 1e-4);
    assert!((hit.point - Vec2::new(0.0, 4.0)).length() < 1e-4);
    assert!((hit.normal - Vec2::new(0.0, -1.0)).length() < 1e-4);
}

#[test]
fn ray_cast_2d_closest_hit_selection() {
    let mut world = PhysicsWorld2d::new();

    let b1 =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(4.0, 0.0)));
    world.create_collider(
        b1,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0))),
    );

    let b2 =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(10.0, 0.0)));
    world.create_collider(
        b2,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0))),
    );

    let hit = world
        .cast_ray_closest(Vec2::ZERO, Vec2::X * 20.0, QueryFilter2d::default())
        .expect("Ray should hit closest body");

    assert_eq!(hit.body_id, b1);
    assert!((hit.fraction * 20.0 - 3.0).abs() < 1e-4);
}

#[test]
fn query_point_finds_enclosing_body() {
    let mut world = PhysicsWorld2d::new();

    let b =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(5.0, 5.0)));
    world.create_collider(
        b,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 2.0))),
    );

    // Point inside circle (at 5.5, 5.0 -> dist 0.5 < 2.0)
    assert_eq!(
        world.query_point(Vec2::new(5.5, 5.0), QueryFilter2d::default()),
        Some(b)
    );

    // Point outside circle (at 10.0, 10.0)
    assert_eq!(
        world.query_point(Vec2::new(10.0, 10.0), QueryFilter2d::default()),
        None
    );
}

#[test]
fn query_aabb_returns_overlapping_2d_bodies() {
    let mut world = PhysicsWorld2d::new();

    let b1 =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        b1,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0))),
    );

    let b2 =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(2.0, 0.0)));
    world.create_collider(
        b2,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.0, 1.0))),
    );

    let b3 =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(50.0, 0.0)));
    world.create_collider(
        b3,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0))),
    );

    let query = Aabb2d::new(Vec2::splat(-2.0), Vec2::splat(3.0));
    let hits = world.query_aabb(&query, QueryFilter2d::default());

    assert_eq!(hits.len(), 2);
    assert!(hits.contains(&b1));
    assert!(hits.contains(&b2));
    assert!(!hits.contains(&b3));
}
