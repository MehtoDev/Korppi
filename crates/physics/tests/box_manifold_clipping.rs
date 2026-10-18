//! Integration tests for Sutherland-Hodgman contact clipping on 3D box manifolds.
//! Verifies that face-face contacts produce multiple contact points (up to 4)
//! and that vertical stacks of boxes remain stable without toppling.

use korppi_math::{Quat, Vec3};
use korppi_physics::narrowphase::{collide_shapes, CollisionBody};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

#[test]
fn face_face_contact_generates_four_contact_points() {
    let collider_a = Collider {
        shape: Shape::Box {
            half_extents: Vec3::splat(1.0),
        },
        friction: 0.4,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let collider_b = Collider {
        shape: Shape::Box {
            half_extents: Vec3::splat(1.0),
        },
        friction: 0.4,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    let body_a = CollisionBody::new(1, Vec3::ZERO, Quat::IDENTITY, &collider_a);
    let body_b = CollisionBody::new(2, Vec3::new(0.0, 1.95, 0.0), Quat::IDENTITY, &collider_b);

    let manifold =
        collide_shapes(body_a, body_b).expect("Box-box face contact must generate manifold");

    // A face-face contact patch between two boxes must produce 4 corner contact points
    assert_eq!(
        manifold.points.len(),
        4,
        "Face-face box contact must generate 4 contact points for angular stability, got {}",
        manifold.points.len()
    );

    // Verify contact points form a polygon of non-zero area (not all at same position)
    let p0 = manifold.points[0].point;
    let mut max_dist = 0.0f32;
    for pt in &manifold.points {
        let dist = (pt.point - p0).length();
        if dist > max_dist {
            max_dist = dist;
        }
        // Normal must point along Y from A to B
        assert!(
            pt.normal.y > 0.9,
            "Contact normal must point upwards along +Y, got {:?}",
            pt.normal
        );
        // Separation must be negative (penetrating)
        assert!(
            pt.separation <= 0.0,
            "Contact separation must be negative or zero, got {}",
            pt.separation
        );
    }
    assert!(
        max_dist > 0.5,
        "Contact points must span a 2D patch, max distance was {:.3}",
        max_dist
    );
}

#[test]
fn ten_box_vertical_stack_remains_stable_without_toppling() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Ground box
    let ground = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, -1.0, 0.0),
        rotation: Quat::IDENTITY,
        ..Default::default()
    });
    world.create_collider(
        ground,
        Collider {
            shape: Shape::Box {
                half_extents: Vec3::new(20.0, 1.0, 20.0),
            },
            friction: 0.6,
            restitution: 0.0,
            is_sensor: false,
            ..Default::default()
        },
    );

    // Stack of 10 boxes of half-extents (0.5, 0.5, 0.5) (1m cubed)
    let box_half = 0.5;
    let mut stack_bodies = Vec::new();
    for i in 0..10 {
        let y = box_half + (i as f32) * (box_half * 2.0);
        let body = world.create_body(RigidBody {
            body_type: BodyType::Dynamic,
            position: Vec3::new(0.0, y, 0.0),
            rotation: Quat::IDENTITY,
            mass: 1.0,
            ..Default::default()
        });
        world.create_collider(
            body,
            Collider {
                shape: Shape::Box {
                    half_extents: Vec3::splat(box_half),
                },
                friction: 0.6,
                restitution: 0.0,
                is_sensor: false,
                ..Default::default()
            },
        );
        stack_bodies.push(body);
    }

    // Step physics for 200 ticks
    let dt = 1.0 / 60.0;
    for _ in 0..200 {
        world.step(dt);
    }

    // Verify stack did not topple: lateral displacement (X, Z) of top box must be small
    let top_body = world.body(*stack_bodies.last().unwrap()).unwrap();
    let lateral_offset = (top_body.position.x * top_body.position.x
        + top_body.position.z * top_body.position.z)
        .sqrt();

    assert!(
        lateral_offset < 0.25,
        "10-box vertical stack toppled! Top box lateral offset was {:.3}m",
        lateral_offset
    );
}
