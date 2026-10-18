//! A second collider joins its body without leaking a proxy (P3D-13).
//!
//! Before FIX-3.1 a second `create_collider` on a body replaced its collider and left the first
//! broadphase proxy in the tree, never moved again: queries found the body twice and the stale
//! proxy kept producing pairs. FIX-3.1 rejected a second collider; since FIX-3.5 a body holds
//! any number, as a Box3D body holds any number of shapes, each with its own proxy that follows
//! the body, and the body's mass is theirs.

use std::panic::{catch_unwind, AssertUnwindSafe};

use korppi_math::{Aabb, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

fn around(center: Vec3, half: f32) -> Aabb {
    Aabb {
        min: center - Vec3::splat(half),
        max: center + Vec3::splat(half),
    }
}

#[test]
fn second_collider_joins_its_body() {
    let mut world = PhysicsWorld::new();
    let ball = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 5.0, 0.0)));
    let first = Collider::new(Shape::Sphere { radius: 0.5 });
    let second = Collider::new(Shape::Box {
        half_extents: Vec3::splat(0.25),
    });
    let first_id = world.create_collider(ball, first.clone());
    let sphere_mass = world.body(ball).unwrap().mass;
    let second_id = world.create_collider(ball, second.clone());

    assert_ne!(first_id, second_id);
    assert_eq!(world.colliders_of(ball), [first_id, second_id]);
    assert_eq!(
        world.collider(first_id),
        Some(&first),
        "the first collider stays"
    );
    assert_eq!(world.collider(second_id), Some(&second));
    assert_eq!(world.collider_body(second_id), Some(ball));
    // Box3D sums the shapes: the 0.5 m ball and the 0.5 m box at 1000 kg/m³.
    let mass = world.body(ball).unwrap().mass;
    assert!((mass - (sphere_mass + 125.0)).abs() < 1e-3, "{mass}");
    assert_eq!(
        world.query_aabb(&around(Vec3::new(0.0, 5.0, 0.0), 1.0)),
        [ball],
        "the body must be found once"
    );
}

#[test]
fn every_proxy_follows_its_body() {
    let mut world = PhysicsWorld::new();
    let ball = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 5.0, 0.0)));
    world.create_collider(ball, Collider::new(Shape::Sphere { radius: 0.5 }));
    world.create_collider(ball, Collider::new(Shape::Sphere { radius: 0.4 }));

    for _ in 0..60 {
        world.step(DT);
    }

    // After a 1 s fall the ball is about 4.9 m lower: found there once, and not at the start.
    // A box over both places finds it once; a stale proxy left at the start would find it there.
    let fallen = world.body(ball).unwrap().position;
    assert!(
        fallen.y < 1.0,
        "the ball should have fallen, it is at {fallen:?}"
    );
    assert_eq!(world.query_aabb(&around(fallen, 0.6)), [ball]);
    let both = Aabb {
        min: fallen - Vec3::splat(0.6),
        max: Vec3::new(0.6, 5.6, 0.6),
    };
    assert_eq!(world.query_aabb(&both), [ball]);
    assert!(world
        .query_aabb(&around(Vec3::new(0.0, 5.0, 0.0), 0.6))
        .is_empty());
}

#[test]
fn a_collider_needs_its_body() {
    let mut world = PhysicsWorld::new();
    let missing = catch_unwind(AssertUnwindSafe(|| {
        world.create_collider(7, Collider::new(Shape::Sphere { radius: 1.0 }))
    }));
    assert!(
        missing.is_err(),
        "a collider on a missing body must be rejected"
    );
    assert_eq!(
        world.counters().collider_count,
        0,
        "the world is left unchanged"
    );
}
