//! A second collider joins its body without leaking a proxy (P2D-08).
//!
//! Before FIX-2.1 a second `create_collider` replaced the collider but left the first
//! broadphase proxy in the tree, never moved again: queries returned the body twice and the
//! stale proxy kept producing pairs (audit probe P5). FIX-2.1 rejected a second collider; since
//! FIX-2.4 a body holds several, each with its own proxy that follows the body.

use korppi_math::{Aabb2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, QueryFilter2d};

fn around(center: Vec2) -> Aabb2d {
    Aabb2d::new(center - Vec2::splat(1.0), center + Vec2::splat(1.0))
}

#[test]
fn a_second_collider_joins_the_body_and_moves_with_it() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let body = world.create_body(RigidBody2d::new(BodyType::Dynamic));
    let circle = world.create_collider(
        body,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );
    let square = world.create_collider(
        body,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    assert_ne!(circle, square);
    assert_eq!(
        world.colliders_of(body).collect::<Vec<_>>(),
        vec![circle, square]
    );
    assert_eq!(
        world.query_aabb(&around(Vec2::ZERO), QueryFilter2d::default()),
        vec![body],
        "the body must be found once (probe P5: twice)"
    );

    // Both proxies follow the body; nothing stays behind.
    world.body_mut(body).unwrap().position = Vec2::new(100.0, 0.0);
    world.step(1.0 / 60.0);
    assert_eq!(
        world.query_aabb(&around(Vec2::new(100.0, 0.0)), QueryFilter2d::default()),
        vec![body]
    );
    assert!(world
        .query_aabb(&around(Vec2::ZERO), QueryFilter2d::default())
        .is_empty());
}

#[test]
fn every_body_can_still_have_its_own_collider() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let bodies: Vec<BodyId2d> = (0..3)
        .map(|i| {
            let body = world.create_body(
                RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(5.0 * i as f32, 0.0)),
            );
            world.create_collider(
                body,
                Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
            );
            body
        })
        .collect();
    for (i, &body) in bodies.iter().enumerate() {
        assert_eq!(
            world.query_aabb(
                &around(Vec2::new(5.0 * i as f32, 0.0)),
                QueryFilter2d::default()
            ),
            vec![body]
        );
    }
}
