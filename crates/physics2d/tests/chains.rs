//! Chains are Box2D v3.2's one-sided chain segments with ghost vertices (P2D-12).
//!
//! A chain is a polyline the world splits into chain segments, one collider each. A segment
//! collides on its right, the side of the right perpendicular of `point1 -> point2`, and its
//! neighbours' vertices (the ghosts) keep bodies sliding across its ends from catching on them.
//! Before FIX-2.6 a chain was one two-sided shape that collided as plain segments, so a box
//! caught on interior vertices and bodies hit chains from behind. Expected values are Box2D's,
//! from `docs/audits/probes/box2d_reference/fix_2_6_chains.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::chain::ChainDef2d;
use korppi_physics2d::collider::{Collider2d, SurfaceMaterial2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, ChainId2d, PhysicsWorld2d, QueryFilter2d, WorldConfig2d};

const DT: f32 = 1.0 / 60.0;

fn world(gravity: f32, sleep: bool) -> PhysicsWorld2d {
    PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, gravity))
        .with_config(WorldConfig2d {
            enable_sleep: sleep,
            ..WorldConfig2d::default()
        })
}

fn material(friction: f32) -> SurfaceMaterial2d {
    SurfaceMaterial2d {
        friction,
        ..SurfaceMaterial2d::default()
    }
}

/// A flat ground of 40 half-metre segments, right to left from x = 10 to x = -10, so that it
/// collides on top.
fn flat_ground(world: &mut PhysicsWorld2d, friction: f32) -> ChainId2d {
    let points = (0..41)
        .map(|i| Vec2::new(10.0 - 0.5 * i as f32, 0.0))
        .collect();
    let def = ChainDef2d::open(points, Vec2::new(10.5, 0.0), Vec2::new(-10.5, 0.0))
        .with_material(material(friction));
    let ground = world.create_body(RigidBody2d::new(BodyType::Static));
    world.create_chain(ground, &def)
}

fn add_box(world: &mut PhysicsWorld2d, x: f32, y: f32, friction: f32) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, y)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))).with_friction(friction),
    );
    id
}

fn add_ball(world: &mut PhysicsWorld2d, x: f32, y: f32, radius: f32) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, y)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, radius))).with_friction(0.6),
    );
    id
}

fn set_velocity(world: &mut PhysicsWorld2d, id: BodyId2d, velocity: Vec2) {
    world.body_mut(id).unwrap().linear_velocity = velocity;
}

fn assert_near(actual: f32, expected: f32, tolerance: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual}, Box2D {expected}"
    );
}

#[test]
fn a_box_slides_over_interior_vertices_without_catching() {
    // Box2D v3.2: a frictionless box crosses 32 vertices at 8 m/s in 2 s and ends at x = 8,
    // y = 0.49997, still at 8 m/s, never faster than 0.00016 m/s vertically or 0.00032 rad/s.
    let mut world = world(-10.0, false);
    flat_ground(&mut world, 0.0);
    let slider = add_box(&mut world, -8.0, 0.5, 0.0);
    set_velocity(&mut world, slider, Vec2::new(8.0, 0.0));
    let (mut max_vy, mut max_w) = (0.0f32, 0.0f32);
    for _ in 0..120 {
        world.step(DT);
        let body = world.body(slider).unwrap();
        max_vy = max_vy.max(body.linear_velocity.y.abs());
        max_w = max_w.max(body.angular_velocity.abs());
    }
    let body = world.body(slider).unwrap();
    assert_near(body.position.x, 8.0, 1e-3, "x");
    assert_near(body.position.y, 0.499_97, 1e-4, "y");
    assert_near(body.linear_velocity.x, 8.0, 1e-3, "vx");
    assert!(max_vy < 1e-3, "the box bumped: max |vy| {max_vy}");
    assert!(max_w < 1e-3, "the box tipped: max |w| {max_w}");
}

#[test]
fn chains_collide_on_their_right_side_only() {
    // Box2D v3.2: a ball dropped from above comes to rest at y = 0.24987; one thrown up at
    // 9 m/s from below passes through and rises to y = 2.0313.
    let mut world = world(-10.0, false);
    flat_ground(&mut world, 0.6);
    let above = add_ball(&mut world, -3.0, 2.0, 0.25);
    let below = add_ball(&mut world, 3.0, -2.0, 0.25);
    set_velocity(&mut world, below, Vec2::new(0.0, 9.0));
    let mut highest = f32::MIN;
    for _ in 0..60 {
        world.step(DT);
        highest = highest.max(world.body(below).unwrap().position.y);
    }
    assert_near(
        world.body(above).unwrap().position.y,
        0.249_87,
        1e-4,
        "above y",
    );
    assert_near(highest, 2.0313, 1e-3, "below rose to");
}

#[test]
fn a_box_across_a_vertex_touches_both_segments() {
    // Box2D v3.2: a box on the vertex at x = 0 rests level at y = 0.49998 with four contacts of
    // two points each: the two segments under it and, within the speculative distance, the two
    // beyond its corners.
    let mut world = world(-10.0, false);
    flat_ground(&mut world, 0.6);
    let block = add_box(&mut world, 0.0, 0.5, 0.6);
    for _ in 0..60 {
        world.step(DT);
    }
    let body = world.body(block).unwrap();
    assert_near(body.position.y, 0.499_98, 1e-4, "y");
    assert_near(body.rotation.angle(), 0.0, 1e-5, "angle");
    let contacts: Vec<_> = world.body_contacts(block).collect();
    assert_eq!(contacts.len(), 4);
    let points: usize = contacts.iter().map(|c| c.manifold.point_count).sum();
    assert_eq!(points, 8);
}

#[test]
fn a_loop_collides_on_the_side_its_winding_gives() {
    // Box2D v3.2: inside a clockwise square loop the normals point in and hold a ball at
    // y = -1.75007; inside a counter-clockwise one they point out and it falls to y = -11.28126.
    for (clockwise, expected, tolerance) in [(true, -1.750_07, 1e-4), (false, -11.281_26, 1e-3)] {
        let mut world = world(-10.0, false);
        let mut points = vec![
            Vec2::new(-2.0, 2.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(2.0, -2.0),
            Vec2::new(-2.0, -2.0),
        ];
        if !clockwise {
            points.reverse();
        }
        let ground = world.create_body(RigidBody2d::new(BodyType::Static));
        let chain = world.create_chain(ground, &ChainDef2d::looped(points));
        assert_eq!(world.chain(chain).unwrap().segment_count(), 4);
        let ball = add_ball(&mut world, 0.5, 0.0, 0.25);
        for _ in 0..90 {
            world.step(DT);
        }
        let y = world.body(ball).unwrap().position.y;
        assert_near(
            y,
            expected,
            tolerance,
            &format!("clockwise {clockwise}: ball y"),
        );
    }
}

#[test]
fn each_segment_can_have_its_own_material() {
    // Two segments, right to left: ice (friction 0) on the right, rough (friction 1) on the
    // left. Box2D v3.2: a box sliding left from x = 6 at 6 m/s stops on the rough one at
    // x = -1.3863.
    let mut world = world(-10.0, false);
    let points = vec![
        Vec2::new(10.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(-10.0, 0.0),
    ];
    let def = ChainDef2d::open(points, Vec2::new(11.0, 0.0), Vec2::new(-11.0, 0.0))
        .with_segment_materials(vec![material(0.0), material(1.0)]);
    let ground = world.create_body(RigidBody2d::new(BodyType::Static));
    let chain = world.create_chain(ground, &def);
    let segments = world.chain(chain).unwrap().segments().to_vec();
    assert_eq!(world.collider(segments[0]).unwrap().friction, 0.0);
    assert_eq!(world.collider(segments[1]).unwrap().friction, 1.0);
    let block = add_box(&mut world, 6.0, 0.5, 1.0);
    set_velocity(&mut world, block, Vec2::new(-6.0, 0.0));
    for _ in 0..180 {
        world.step(DT);
    }
    let body = world.body(block).unwrap();
    assert_near(body.position.x, -1.3863, 1e-3, "x");
    assert_near(body.linear_velocity.x, 0.0, 1e-4, "vx");
}

#[test]
fn rays_hit_chains_from_the_front_only() {
    // Box2D v3.2: a ray cast down from y = 4 over 8 m hits halfway; one cast up from y = -4
    // misses.
    let mut world = world(-10.0, false);
    flat_ground(&mut world, 0.6);
    let hit = world
        .cast_ray_closest(
            Vec2::new(0.25, 4.0),
            Vec2::new(0.0, -8.0),
            QueryFilter2d::default(),
        )
        .expect("the ray from above hits the chain");
    assert_near(hit.fraction, 0.5, 1e-5, "fraction");
    assert_near(hit.normal.y, 1.0, 1e-6, "normal y");
    let up = world.cast_ray_closest(
        Vec2::new(0.25, -4.0),
        Vec2::new(0.0, 8.0),
        QueryFilter2d::default(),
    );
    assert!(up.is_none());
}

#[test]
fn fast_bodies_hit_chains_from_the_front_only() {
    // Box2D v3.2, without gravity: a ball at 50 m/s from above stops on the chain at y = 0.1;
    // one at 50 m/s from below passes through to y = 20.
    let mut world = world(0.0, false);
    flat_ground(&mut world, 0.6);
    let down = add_ball(&mut world, -3.0, 5.0, 0.1);
    let up = add_ball(&mut world, 3.0, -5.0, 0.1);
    set_velocity(&mut world, down, Vec2::new(0.0, -50.0));
    set_velocity(&mut world, up, Vec2::new(0.0, 50.0));
    for _ in 0..30 {
        world.step(DT);
    }
    assert_near(world.body(down).unwrap().position.y, 0.1, 1e-3, "down y");
    assert_near(world.body(up).unwrap().position.y, 20.0, 1e-3, "up y");
}

#[test]
fn destroying_a_chain_wakes_the_bodies_on_it() {
    // Box2D v3.2, sleep on: a box falls asleep on the 40-segment chain within a second;
    // destroying the chain wakes it, and 30 steps later it has fallen to y = -0.7604. A segment
    // cannot be destroyed on its own.
    let mut world = world(-10.0, true);
    let chain = flat_ground(&mut world, 0.6);
    let block = add_box(&mut world, 0.25, 0.5, 0.6);
    for _ in 0..60 {
        world.step(DT);
    }
    let segments = world.chain(chain).unwrap().segments().to_vec();
    assert_eq!(segments.len(), 40);
    assert_eq!(world.collider_chain(segments[7]), Some(chain));
    assert!(world.body(block).unwrap().is_sleeping);
    assert!(!world.destroy_collider(segments[7]));

    assert!(world.destroy_chain(chain));
    assert!(!world.body(block).unwrap().is_sleeping);
    assert!(world.chain(chain).is_none());
    assert!(segments.iter().all(|&s| world.collider(s).is_none()));
    assert!(!world.destroy_chain(chain));
    for _ in 0..30 {
        world.step(DT);
    }
    assert_near(world.body(block).unwrap().position.y, -0.7604, 1e-3, "y");
}
