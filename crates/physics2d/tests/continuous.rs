//! Fast bodies do not tunnel, and continuous collision keeps their velocity (P2D-07).
//!
//! Before FIX-2.3 only bullets had continuous collision, run every substep against every body,
//! and it zeroed the normal velocity at impact: a 50 m/s ball passed through a 0.1 m wall (audit
//! probe P10) and an elastic bullet hitting a wall stopped dead (probe P8). As in Box2D v3.2,
//! a body that moves more than half its smallest extent in a step is swept against static
//! bodies (bullets also against other bodies) and stopped at the time of impact; the next
//! step's contacts make the response. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_3_continuous.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn add_static_box(world: &mut PhysicsWorld2d, position: Vec2, hx: f32, hy: f32) {
    let id = world.create_body(RigidBody2d::new(BodyType::Static).with_position(position));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))),
    );
}

fn add_ball(
    world: &mut PhysicsWorld2d,
    velocity: Vec2,
    restitution: f32,
    bullet: bool,
) -> BodyId2d {
    let id = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_linear_velocity(velocity)
            .with_bullet(bullet),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.1)))
            .with_restitution(restitution),
    );
    id
}

#[test]
fn fast_ball_stops_at_a_thin_wall() {
    // Probe P10: a non-bullet ball (r 0.1) at 50 m/s toward a 0.1 m static wall at x = 5.
    // Box2D v3.2: never past x = 4.8585, resting against the wall at x = 4.85.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    add_static_box(&mut world, Vec2::new(5.0, 0.0), 0.05, 2.0);
    let ball = add_ball(&mut world, Vec2::new(50.0, 0.0), 0.0, false);
    let mut farthest = f32::NEG_INFINITY;
    for _ in 0..60 {
        world.step(DT);
        farthest = farthest.max(world.body(ball).unwrap().position.x);
    }
    let body = world.body(ball).unwrap();
    assert!(
        farthest < 4.86,
        "the ball reached x = {farthest:.4} (probe P10: 6.33)"
    );
    assert!(
        (body.position.x - 4.85).abs() < 0.01,
        "the ball ended at x = {:.4}",
        body.position.x
    );
    assert!(body.linear_velocity.x.abs() < 0.01);
}

#[test]
fn elastic_bullet_bounces_off_a_thin_wall() {
    // Probe P8: an elastic bullet at 120 m/s. Box2D v3.2: bounces back at -120 m/s.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    add_static_box(&mut world, Vec2::new(5.0, 0.0), 0.05, 2.0);
    let bullet = add_ball(&mut world, Vec2::new(120.0, 0.0), 1.0, true);
    let mut farthest = f32::NEG_INFINITY;
    for _ in 0..30 {
        world.step(DT);
        farthest = farthest.max(world.body(bullet).unwrap().position.x);
    }
    let vx = world.body(bullet).unwrap().linear_velocity.x;
    assert!(farthest < 4.87, "the bullet reached x = {farthest:.4}");
    assert!(
        (vx + 120.0).abs() < 1.0,
        "the bullet left at {vx:.3} m/s (probe P8: 0)"
    );
}

#[test]
fn bullet_carries_a_thin_dynamic_plank() {
    // A bullet at 200 m/s into a free 0.1 m plank: it stays behind the plank and both move on at
    // the shared momentum velocity. Box2D v3.2: 27.151 m/s, bullet at x = 17.515, plank 17.665.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let plank =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(5.0, 0.0)));
    world.create_collider(
        plank,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.05, 1.0))),
    );
    let bullet = add_ball(&mut world, Vec2::new(200.0, 0.0), 0.0, true);
    for _ in 0..30 {
        world.step(DT);
    }
    let (b, p) = (world.body(bullet).unwrap(), world.body(plank).unwrap());
    assert!(
        b.position.x < p.position.x,
        "the bullet passed the plank: bullet x = {:.3}, plank x = {:.3}",
        b.position.x,
        p.position.x
    );
    assert!(
        (p.linear_velocity.x - 27.151).abs() < 0.5,
        "the plank moves at {:.3} m/s",
        p.linear_velocity.x
    );
    assert!(
        (b.linear_velocity.x - 27.151).abs() < 0.5,
        "the bullet moves at {:.3} m/s",
        b.linear_velocity.x
    );
}

#[test]
fn fast_falling_box_lands_on_a_thin_floor() {
    // A box thrown down at 200 m/s onto a 0.1 m floor. Box2D v3.2: never below y = 0.5499.
    let mut world = PhysicsWorld2d::new();
    add_static_box(&mut world, Vec2::ZERO, 5.0, 0.05);
    let id = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 10.0))
            .with_linear_velocity(Vec2::new(0.0, -200.0)),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    let mut lowest = f32::INFINITY;
    for _ in 0..60 {
        world.step(DT);
        lowest = lowest.min(world.body(id).unwrap().position.y);
    }
    let rest = world.body(id).unwrap().position.y;
    assert!(lowest > 0.54, "the box sank to y = {lowest:.4}");
    assert!(
        (rest - 0.5499).abs() < 0.002,
        "the box rests at y = {rest:.4}"
    );
}

/// A ball of radius 0.25 at 30 m/s (0.5 m per step) toward a 2 cm static wall at x = 3, with
/// continuous safety factor `factor`: its x after 0.5 s.
fn ball_toward_wall(factor: Option<f32>) -> f32 {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let wall =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(3.0, 0.0)));
    world.create_collider(
        wall,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.01, 2.0))),
    );
    let mut body = RigidBody2d::new(BodyType::Dynamic).with_linear_velocity(Vec2::new(30.0, 0.0));
    if let Some(factor) = factor {
        body = body.with_safety_factor(factor);
    }
    let ball = world.create_body(body);
    world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
    );
    for _ in 0..30 {
        world.step(DT);
    }
    world.body(ball).unwrap().position.x
}

#[test]
fn a_body_sets_its_continuous_safety_factor() {
    // Box2D v3.2's `b2BodyDef::safetyFactor` (`fix_2_review.c`): a body is fast when it moves
    // more than the factor times its smallest extent in a step. At the default 0.5 the ball is
    // fast and stops at the wall; at 4 it is not, and passes through, as in Box2D.
    assert_eq!(RigidBody2d::new(BodyType::Dynamic).safety_factor, 0.5);
    assert_eq!(ball_toward_wall(None), 2.740_007_6);
    assert_eq!(ball_toward_wall(Some(4.0)), 15.0);
}
