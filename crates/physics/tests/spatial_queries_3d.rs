//! Integration tests for 3D spatial queries (ray cast and AABB overlap) on PhysicsWorld.

use korppi_math::{Aabb, Ray, Vec3};
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::shape::Shape;
use korppi_physics::{PhysicsWorld, RayHit3d};

#[test]
fn ray_cast_hits_sphere_with_accurate_normal_and_fraction() {
    let mut world = PhysicsWorld::new();

    let sphere_body =
        world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 10.0)));
    world.create_collider(sphere_body, Collider::new(Shape::Sphere { radius: 1.0 }));

    // Ray directed from origin along +Z
    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let hit: RayHit3d = world.ray_cast(&ray, 20.0).expect("Ray should hit sphere");

    assert_eq!(hit.body_id, sphere_body);
    // Sphere center is at (0, 0, 10), radius is 1.0 -> hit at (0, 0, 9.0)
    assert!((hit.fraction - 9.0).abs() < 1e-4);
    assert!((hit.point - Vec3::new(0.0, 0.0, 9.0)).length() < 1e-4);
    // Outward normal facing ray origin -> (0, 0, -1)
    assert!((hit.normal - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4);
}

#[test]
fn ray_cast_hits_box_front_face() {
    let mut world = PhysicsWorld::new();

    let box_body = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(5.0, 0.0, 0.0)));
    world.create_collider(
        box_body,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    // Ray directed along +X
    let ray = Ray::new(Vec3::ZERO, Vec3::X);
    let hit = world.ray_cast(&ray, 10.0).expect("Ray should hit box");

    assert_eq!(hit.body_id, box_body);
    // Box center at (5, 0, 0), half extents 1.0 -> front face at X = 4.0
    assert!((hit.fraction - 4.0).abs() < 1e-4);
    assert!((hit.point - Vec3::new(4.0, 0.0, 0.0)).length() < 1e-4);
    assert!((hit.normal - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-4);
}

#[test]
fn ray_cast_selects_closest_body_among_occluders() {
    let mut world = PhysicsWorld::new();

    // Body 1 at Z = 5.0
    let b1 = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 5.0)));
    world.create_collider(b1, Collider::new(Shape::Sphere { radius: 1.0 }));

    // Body 2 at Z = 12.0
    let b2 = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 12.0)));
    world.create_collider(b2, Collider::new(Shape::Sphere { radius: 1.0 }));

    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let hit = world
        .ray_cast(&ray, 20.0)
        .expect("Ray should hit closest body");

    assert_eq!(hit.body_id, b1);
    assert!((hit.fraction - 4.0).abs() < 1e-4);
}

#[test]
fn ray_cast_returns_none_when_missing_all_bodies() {
    let mut world = PhysicsWorld::new();

    let b = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 10.0, 0.0)));
    world.create_collider(b, Collider::new(Shape::Sphere { radius: 1.0 }));

    // Ray shooting along +Z
    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    assert!(world.ray_cast(&ray, 50.0).is_none());
}

#[test]
fn query_aabb_returns_all_overlapping_bodies() {
    let mut world = PhysicsWorld::new();

    // Body 1 at (0, 0, 0)
    let b1 = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(b1, Collider::new(Shape::Sphere { radius: 1.0 }));

    // Body 2 at (2, 0, 0)
    let b2 = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(2.0, 0.0, 0.0)));
    world.create_collider(
        b2,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    // Body 3 far away at (50, 0, 0)
    let b3 = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(50.0, 0.0, 0.0)));
    world.create_collider(b3, Collider::new(Shape::Sphere { radius: 1.0 }));

    // Query volume covering [-2, -2, -2] to [3, 3, 3] -> should hit b1 and b2, not b3
    let query = Aabb {
        min: Vec3::splat(-2.0),
        max: Vec3::splat(3.0),
    };

    let hits = world.query_aabb(&query);
    assert_eq!(hits.len(), 2);
    assert!(hits.contains(&b1));
    assert!(hits.contains(&b2));
    assert!(!hits.contains(&b3));
}
