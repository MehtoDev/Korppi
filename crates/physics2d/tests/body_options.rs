//! A body's type can change, it can be disabled and enabled, teleported, and a kinematic body can
//! be driven to a target pose (P2D-21 remainder).
//!
//! Before FIX-2.4 none of these existed: changing `body_type` by hand left the old mass and
//! contacts, and moving `position` by hand left the broadphase behind. Expected values are
//! Box2D v3.2's (`b2Body_SetType`, `b2Body_Disable`, `b2Body_Enable`, `b2Body_SetTransform`,
//! `b2Body_SetTargetTransform`), from `docs/audits/probes/box2d_reference/fix_2_4_body_options.c`.

use korppi_math::{Aabb2d, Rot2, Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, QueryFilter2d};

const DT: f32 = 1.0 / 60.0;

fn run(world: &mut PhysicsWorld2d, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

fn unit_box(world: &mut PhysicsWorld2d, kind: BodyType, position: Vec2) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(kind).with_position(position));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    id
}

fn add_ground(world: &mut PhysicsWorld2d) -> BodyId2d {
    let id =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))),
    );
    id
}

fn y(world: &PhysicsWorld2d, id: BodyId2d) -> f32 {
    world.body(id).unwrap().position.y
}

#[test]
fn changing_the_type_follows_box2d() {
    let mut world = PhysicsWorld2d::new();
    let id = unit_box(&mut world, BodyType::Dynamic, Vec2::new(0.0, 10.0));
    run(&mut world, 0.5);

    // Static: massless and still. Box2D v3.2 drops the velocity.
    assert!(world.set_body_type(id, BodyType::Static));
    let frozen = y(&world, id);
    let body = world.body(id).unwrap();
    assert_eq!((body.mass, body.linear_velocity), (0.0, Vec2::ZERO));
    run(&mut world, 0.5);
    assert_eq!(y(&world, id), frozen, "a static body must not move");

    // Dynamic again: mass from the collider, falling from rest (Box2D: 1.2365 m in 0.5 s).
    assert!(world.set_body_type(id, BodyType::Dynamic));
    let body = world.body(id).unwrap();
    assert_eq!((body.mass, body.linear_velocity), (1.0, Vec2::ZERO));
    run(&mut world, 0.5);
    let fall = frozen - y(&world, id);
    assert!(
        (fall - 1.2365).abs() < 0.01,
        "the box fell {fall:.4} m in 0.5 s"
    );

    // Kinematic: keeps its velocity and ignores gravity (Box2D: 2.0000 m in 1 s).
    let mover = unit_box(&mut world, BodyType::Dynamic, Vec2::new(10.0, 10.0));
    world.body_mut(mover).unwrap().linear_velocity = Vec2::new(2.0, 0.0);
    run(&mut world, 0.5);
    let v0 = world.body(mover).unwrap().linear_velocity;
    let p0 = world.body(mover).unwrap().position;
    assert!(world.set_body_type(mover, BodyType::Kinematic));
    assert_eq!(world.body(mover).unwrap().linear_velocity, v0);
    run(&mut world, 1.0);
    let travel = world.body(mover).unwrap().position - p0;
    assert!(
        (travel - v0).length() < 0.001,
        "the kinematic body moved {travel:?}, expected {v0:?}"
    );
    assert!(!world.set_body_type(
        BodyId2d::new(99, std::num::NonZeroU32::MIN),
        BodyType::Static
    ));
}

#[test]
fn turning_the_ground_dynamic_wakes_the_box_on_it() {
    // Box2D v3.2: both fall; after 1 s the floor is at y = -4.9255 and the box at -3.9255.
    let mut world = PhysicsWorld2d::new();
    let floor = unit_box(&mut world, BodyType::Static, Vec2::ZERO);
    let crate_box = unit_box(&mut world, BodyType::Dynamic, Vec2::new(0.0, 1.0));
    run(&mut world, 3.0);
    assert!(world.body(crate_box).unwrap().is_sleeping);

    world.set_body_type(floor, BodyType::Dynamic);
    assert!(
        !world.body(crate_box).unwrap().is_sleeping,
        "losing its contact wakes the box"
    );
    run(&mut world, 1.0);
    assert!(
        (y(&world, floor) + 4.9255).abs() < 0.02,
        "the floor is at y = {:.4}",
        y(&world, floor)
    );
    assert!(
        (y(&world, crate_box) + 3.9255).abs() < 0.02,
        "the box is at y = {:.4}",
        y(&world, crate_box)
    );
}

#[test]
fn a_disabled_body_stays_put_and_leaves_the_broadphase() {
    let mut world = PhysicsWorld2d::new();
    let floor = add_ground(&mut world);
    let crate_box = unit_box(&mut world, BodyType::Dynamic, Vec2::new(0.0, 0.5));
    let faller = unit_box(&mut world, BodyType::Dynamic, Vec2::new(5.0, 10.0));
    run(&mut world, 0.5);

    assert!(world.disable_body(faller));
    assert!(!world.body(faller).unwrap().is_enabled());
    let held = y(&world, faller);
    let around = Aabb2d::new(Vec2::new(4.0, held - 1.0), Vec2::new(6.0, held + 1.0));
    assert!(
        world
            .query_aabb(&around, QueryFilter2d::default())
            .is_empty(),
        "a disabled body must leave the broadphase"
    );
    run(&mut world, 1.0);
    assert_eq!(y(&world, faller), held, "a disabled body must not move");

    // Enabled again, it falls from rest (Box2D v3.2: 1.2365 m in 0.5 s).
    assert!(world.enable_body(faller));
    assert_eq!(world.body(faller).unwrap().linear_velocity, Vec2::ZERO);
    assert_eq!(
        world.query_aabb(&around, QueryFilter2d::default()),
        vec![faller]
    );
    run(&mut world, 0.5);
    let fall = held - y(&world, faller);
    assert!(
        (fall - 1.2365).abs() < 0.01,
        "the body fell {fall:.4} m in 0.5 s"
    );

    // Disabling the ground wakes the box asleep on it, which falls through (Box2D: y = -4.4255).
    run(&mut world, 3.0);
    assert!(world.body(crate_box).unwrap().is_sleeping);
    world.disable_body(floor);
    assert!(!world.body(crate_box).unwrap().is_sleeping);
    run(&mut world, 1.0);
    assert!(
        (y(&world, crate_box) + 4.4255).abs() < 0.01,
        "the box is at y = {:.4}",
        y(&world, crate_box)
    );
}

#[test]
fn a_body_created_disabled_joins_when_enabled() {
    let mut world = PhysicsWorld2d::new();
    let id = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 10.0))
            .with_enabled(false),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    assert!(world
        .query_aabb(
            &Aabb2d::new(Vec2::splat(-5.0), Vec2::splat(15.0)),
            QueryFilter2d::default()
        )
        .is_empty());
    run(&mut world, 0.5);
    assert_eq!(y(&world, id), 10.0);
    world.enable_body(id);
    run(&mut world, 0.5);
    assert!((10.0 - y(&world, id) - 1.2365).abs() < 0.01);
}

#[test]
fn a_joint_to_a_disabled_body_does_nothing() {
    // Box2D v3.2 with sleep off: once its anchor is disabled, the bob drops 4.9254 m in 1 s.
    // With sleep on the bob is asleep by then, and disabling the anchor does not wake it.
    for sleep in [false, true] {
        let mut world = PhysicsWorld2d::new();
        let anchor = world
            .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 5.0)));
        let mut bob_body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 4.0));
        bob_body.enable_sleep = sleep;
        let bob = world.create_body(bob_body);
        world.create_collider(
            bob,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
        );
        let joint = RevoluteJoint2d::new(Vec2::ZERO, Vec2::new(0.0, 1.0), 0.0, 0.0);
        world.create_joint(anchor, bob, Joint2d::Revolute(joint));
        run(&mut world, 1.0);
        let hanging = y(&world, bob);
        world.disable_body(anchor);
        assert_eq!(world.body(bob).unwrap().is_sleeping, sleep);
        run(&mut world, 1.0);
        let drop = hanging - y(&world, bob);
        let expected = if sleep { 0.0 } else { 4.9254 };
        assert!(
            (drop - expected).abs() < 0.01,
            "sleep {sleep}: the bob dropped {drop:.4} m, Box2D {expected}"
        );
        assert_eq!(world.joint_count(), 1, "disabling keeps the joint");
    }
}

#[test]
fn teleporting_keeps_velocity_and_sleep() {
    let mut world = PhysicsWorld2d::new();
    add_ground(&mut world);
    let crate_box = unit_box(&mut world, BodyType::Dynamic, Vec2::new(0.0, 0.5));
    let faller = unit_box(&mut world, BodyType::Dynamic, Vec2::new(5.0, 10.0));
    run(&mut world, 0.5);

    // Box2D v3.2: the velocity survives the teleport (-4.905 m/s); 0.25 s later y = 10.4621.
    let v = world.body(faller).unwrap().linear_velocity;
    assert!(world.set_transform(faller, Vec2::new(-5.0, 12.0), Rot2::from_angle(0.5)));
    let body = world.body(faller).unwrap();
    assert_eq!(body.position, Vec2::new(-5.0, 12.0));
    assert_eq!(body.linear_velocity, v);
    let near = Aabb2d::new(Vec2::new(-6.0, 11.0), Vec2::new(-4.0, 13.0));
    assert_eq!(
        world.query_aabb(&near, QueryFilter2d::default()),
        vec![faller],
        "the broadphase must follow the teleport"
    );
    run(&mut world, 0.25);
    assert!(
        (y(&world, faller) - 10.4621).abs() < 0.01,
        "y = {:.4}",
        y(&world, faller)
    );

    // Teleporting does not wake a sleeping body: it stays where it was put.
    run(&mut world, 3.0);
    assert!(world.body(crate_box).unwrap().is_sleeping);
    world.set_transform(crate_box, Vec2::new(2.0, 5.0), Rot2::IDENTITY);
    run(&mut world, 1.0);
    assert!(world.body(crate_box).unwrap().is_sleeping);
    assert_eq!(world.body(crate_box).unwrap().position, Vec2::new(2.0, 5.0));
}

#[test]
fn a_kinematic_body_reaches_its_target_in_one_step() {
    // Box2D v3.2: (1, 0.5) at 0.3 rad after one step, then onward at the same velocity.
    let mut world = PhysicsWorld2d::new();
    let platform = unit_box(&mut world, BodyType::Kinematic, Vec2::ZERO);
    let target = Transform2d::from_translation_rotation(Vec2::new(1.0, 0.5), Rot2::from_angle(0.3));
    assert!(world.set_target_transform(platform, target, DT, true));
    world.step(DT);
    let body = world.body(platform).unwrap();
    assert!(
        (body.position - Vec2::new(1.0, 0.5)).length() < 0.001,
        "at {:?}",
        body.position
    );
    assert!(
        (body.rotation.angle() - 0.3).abs() < 0.002,
        "at {:.5} rad",
        body.rotation.angle()
    );
    world.step(DT);
    let x = world.body(platform).unwrap().position.x;
    assert!((x - 2.0).abs() < 0.001, "x = {x:.4} after two steps");

    // A static body has no target.
    let wall = unit_box(&mut world, BodyType::Static, Vec2::new(5.0, 0.0));
    assert!(!world.set_target_transform(wall, target, DT, true));
}

/// Two resting boxes; the second is moved onto the first, overlapping it by 0.1 m, by `teleport`.
/// The begin-touch events of the next step and the moved box's height after it.
fn teleport_onto_box(teleport: impl FnOnce(&mut PhysicsWorld2d, BodyId2d)) -> (usize, u32) {
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(korppi_physics2d::world::WorldConfig2d {
            enable_sleep: false,
            ..Default::default()
        });
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(0.6),
    );
    let mut boxes = [0.0, 5.0].map(|x| {
        let id =
            world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, 0.5)));
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
                .with_friction(0.6)
                .with_contact_events(true),
        );
        id
    });
    for _ in 0..60 {
        world.step(1.0 / 60.0);
    }
    let moved = boxes[1];
    teleport(&mut world, moved);
    world.step(1.0 / 60.0);
    boxes.swap(0, 1);
    (
        world.contact_events().begin.len(),
        world.body(moved).unwrap().position.y.to_bits(),
    )
}

#[test]
fn a_teleport_pairs_in_the_same_step_also_through_body_mut() {
    // Box2D's `b2Body_SetTransform` moves the proxy at once, so the next step pairs the boxes and
    // they begin touching (`fix_2_review.c`). Moving a body through `body_mut` does the same: the
    // step finds the new pose and moves the proxy before looking for pairs.
    let expected = (1, 0x3fb4_54b5);
    assert_eq!(
        teleport_onto_box(|world, id| {
            world.set_transform(id, Vec2::new(0.0, 1.4), Rot2::IDENTITY);
        }),
        expected,
        "set_transform"
    );
    assert_eq!(
        teleport_onto_box(|world, id| {
            world.body_mut(id).unwrap().position = Vec2::new(0.0, 1.4);
        }),
        expected,
        "body_mut"
    );
}
