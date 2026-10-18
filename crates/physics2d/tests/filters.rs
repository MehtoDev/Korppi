//! Collision filters: 64 category bits, group index, a custom filter callback, and filtered
//! queries (P2D-24).
//!
//! Before FIX-2.4 filters were 32 bits with no group index, there was no custom filter, and
//! queries ignored filters. As in Box2D v3.2 (`b2Filter`, `b2ShouldShapesCollide`,
//! `b2QueryFilter`), two colliders in the same non-zero group always collide if it is positive
//! and never if it is negative; otherwise each one's mask must include the other's category.
//! Expected values are Box2D's, from `docs/audits/probes/box2d_reference/fix_2_4_filters.c`.

use korppi_math::{Aabb2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::{Collider2d, Filter2d};
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, QueryFilter2d};

const DT: f32 = 1.0 / 60.0;

fn run(world: &mut PhysicsWorld2d, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

fn add_box(
    world: &mut PhysicsWorld2d,
    kind: BodyType,
    position: Vec2,
    half: Vec2,
    collider: Collider2d,
) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(kind).with_position(position));
    let shape = Shape2d::Polygon(Polygon2d::from_box(half.x, half.y));
    world.create_collider(id, Collider2d { shape, ..collider });
    id
}

fn with_filter(filter: Filter2d) -> Collider2d {
    Collider2d::default().with_filter(filter)
}

/// Two unit boxes overlapping by half; the speed of the second after 0.5 s.
fn push_apart_speed(a: Filter2d, b: Filter2d) -> f32 {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    add_box(
        &mut world,
        BodyType::Dynamic,
        Vec2::ZERO,
        Vec2::splat(0.5),
        with_filter(a),
    );
    let second = add_box(
        &mut world,
        BodyType::Dynamic,
        Vec2::new(0.5, 0.0),
        Vec2::splat(0.5),
        with_filter(b),
    );
    run(&mut world, 0.5);
    world.body(second).unwrap().linear_velocity.length()
}

#[test]
fn groups_override_the_bits() {
    // Box2D v3.2: colliding boxes separate at 0.1732 m/s after 0.5 s; filtered ones stay put.
    let all = u64::MAX;
    let collide = |speed: f32| (speed - 0.1732).abs() < 0.005;
    assert!(collide(push_apart_speed(
        Filter2d::new(1, all),
        Filter2d::new(1, all)
    )));
    assert_eq!(
        push_apart_speed(
            Filter2d::new(1, all).with_group(-3),
            Filter2d::new(1, all).with_group(-3)
        ),
        0.0
    );
    assert!(collide(push_apart_speed(
        Filter2d::new(1, 0).with_group(3),
        Filter2d::new(1, 0).with_group(3)
    )));
    assert_eq!(
        push_apart_speed(Filter2d::new(1, 0), Filter2d::new(1, 0)),
        0.0
    );
    assert!(collide(push_apart_speed(
        Filter2d::new(1, all).with_group(-3),
        Filter2d::new(1, all).with_group(-4)
    )));
}

/// A box dropped from y = 2 onto the ground; its y after 1 s.
fn drop(ground: Filter2d, falling: Collider2d, world: &mut PhysicsWorld2d) -> BodyId2d {
    add_box(
        world,
        BodyType::Static,
        Vec2::new(0.0, -0.5),
        Vec2::new(20.0, 0.5),
        with_filter(ground),
    );
    add_box(
        world,
        BodyType::Dynamic,
        Vec2::new(0.0, 2.0),
        Vec2::splat(0.5),
        falling,
    )
}

#[test]
fn categories_use_all_64_bits() {
    // Box2D v3.2: the box in category bit 40 lands (y = 0.49991) when the ground's mask has
    // that bit, and falls to y = -2.9254 when it does not.
    let bit40 = 1u64 << 40;
    let mut world = PhysicsWorld2d::new();
    let lands = drop(
        Filter2d::new(1, bit40),
        with_filter(Filter2d::new(bit40, u64::MAX)),
        &mut world,
    );
    run(&mut world, 1.0);
    assert!((world.body(lands).unwrap().position.y - 0.49991).abs() < 0.002);

    let mut world = PhysicsWorld2d::new();
    let falls = drop(
        Filter2d::new(1, !bit40),
        with_filter(Filter2d::new(bit40, u64::MAX)),
        &mut world,
    );
    run(&mut world, 1.0);
    assert!((world.body(falls).unwrap().position.y + 2.9254).abs() < 0.01);
}

#[test]
fn the_custom_filter_applies_to_colliders_that_enable_it() {
    // Box2D v3.2: with a filter that rejects every pair, the box that enables custom filtering
    // falls through (y = -2.9254) and the other lands.
    let mut world = PhysicsWorld2d::new();
    world.set_custom_filter(|_a, _b| false);
    add_box(
        &mut world,
        BodyType::Static,
        Vec2::new(0.0, -0.5),
        Vec2::new(20.0, 0.5),
        Collider2d::default(),
    );
    let custom = add_box(
        &mut world,
        BodyType::Dynamic,
        Vec2::new(-2.0, 2.0),
        Vec2::splat(0.5),
        Collider2d::default().with_custom_filtering(true),
    );
    let plain = add_box(
        &mut world,
        BodyType::Dynamic,
        Vec2::new(2.0, 2.0),
        Vec2::splat(0.5),
        Collider2d::default(),
    );
    run(&mut world, 1.0);
    assert!((world.body(custom).unwrap().position.y + 2.9254).abs() < 0.01);
    assert!((world.body(plain).unwrap().position.y - 0.49991).abs() < 0.002);
}

#[test]
fn queries_respect_their_filter() {
    // A shape is reported when its category is in the query's mask and the query's category in
    // its mask (Box2D b2ShouldQueryCollide).
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let first = add_box(
        &mut world,
        BodyType::Static,
        Vec2::ZERO,
        Vec2::splat(0.5),
        with_filter(Filter2d::new(1, u64::MAX)),
    );
    let second = add_box(
        &mut world,
        BodyType::Static,
        Vec2::new(3.0, 0.0),
        Vec2::splat(0.5),
        with_filter(Filter2d::new(2, u64::MAX)),
    );
    let area = Aabb2d::new(Vec2::new(-1.0, -1.0), Vec2::new(4.0, 1.0));
    let mut all = world.query_aabb(&area, QueryFilter2d::default());
    all.sort();
    assert_eq!(all, vec![first, second]);
    assert_eq!(
        world.query_aabb(&area, QueryFilter2d::new(1, 2)),
        vec![second]
    );
    let mut category4 = world.query_aabb(&area, QueryFilter2d::new(4, u64::MAX));
    category4.sort();
    assert_eq!(category4, vec![first, second]);

    // Box2D v3.2: a ray with mask 2 skips the first box and hits the second at x = 2.5.
    let hit = world
        .cast_ray_closest(
            Vec2::new(-2.0, 0.0),
            Vec2::X * 12.0,
            QueryFilter2d::new(1, 2),
        )
        .expect("the ray hits the second box");
    assert_eq!(hit.body_id, second);
    assert!((hit.point.x - 2.5).abs() < 1e-4);
    assert_eq!(
        world.query_point(Vec2::ZERO, QueryFilter2d::new(1, 2)),
        None
    );
    assert_eq!(
        world.query_point(Vec2::ZERO, QueryFilter2d::default()),
        Some(first)
    );
}
