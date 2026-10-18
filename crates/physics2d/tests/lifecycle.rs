//! Bodies, colliders and joints can be destroyed, and their handles go stale (P2D-10).
//!
//! Before FIX-2.4 nothing could be destroyed except joints, and removing a joint shifted the
//! index of every later joint. Bodies, colliders and joints now have generational handles, as
//! Box2D v3.2's `b2BodyId`, `b2ShapeId` and `b2JointId` do: a destroyed object's handle is
//! rejected even after its slot is reused. Destroying a body destroys its collider, contacts and
//! joints, and wakes the bodies it touched. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_4_lifecycle.c`.

use korppi_math::{Aabb2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, JointId2d, PhysicsWorld2d, QueryFilter2d};

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
    hx: f32,
    hy: f32,
) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(kind).with_position(position));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))),
    );
    id
}

fn add_ball(world: &mut PhysicsWorld2d, position: Vec2) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(position));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
    );
    id
}

/// Pins `bob` to `anchor` at the world point `pivot`.
fn pin(world: &mut PhysicsWorld2d, anchor: BodyId2d, bob: BodyId2d, pivot: Vec2) -> JointId2d {
    let local = |id: BodyId2d| pivot - world.body(id).unwrap().position;
    let joint = RevoluteJoint2d::new(local(anchor), local(bob), 0.0, 0.0);
    world.create_joint(anchor, bob, Joint2d::Revolute(joint))
}

#[test]
fn destroying_the_ground_drops_the_box_asleep_on_it() {
    // Box2D v3.2: the box sleeps after 3 s, wakes when the ground is destroyed and is at
    // y = -4.4255 one second later.
    let mut world = PhysicsWorld2d::new();
    let ground = add_box(&mut world, BodyType::Static, Vec2::new(0.0, -0.5), 5.0, 0.5);
    let crate_box = add_box(&mut world, BodyType::Dynamic, Vec2::new(0.0, 0.5), 0.5, 0.5);
    run(&mut world, 3.0);
    assert!(
        world.body(crate_box).unwrap().is_sleeping,
        "the box should be asleep"
    );

    assert!(world.destroy_body(ground));
    assert!(
        world.body(ground).is_none(),
        "a destroyed body's handle must be stale"
    );
    assert!(!world.destroy_body(ground), "a body is destroyed only once");
    assert!(
        !world.body(crate_box).unwrap().is_sleeping,
        "losing its contact wakes the box"
    );
    let leftovers = world.query_aabb(
        &Aabb2d::new(Vec2::new(-5.0, -1.0), Vec2::new(5.0, -0.1)),
        QueryFilter2d::default(),
    );
    assert!(
        leftovers.is_empty(),
        "the ground is still in the broadphase: {leftovers:?}"
    );

    run(&mut world, 1.0);
    let y = world.body(crate_box).unwrap().position.y;
    assert!(
        (y + 4.4255).abs() < 0.01,
        "the box is at y = {y:.4} after 1 s"
    );
    assert_eq!(world.body_count(), 1);
}

#[test]
fn destroying_the_anchor_destroys_its_joint_and_frees_the_bob() {
    // Box2D v3.2: the joint goes with the anchor and the bob drops 4.9254 m in 1 s.
    let mut world = PhysicsWorld2d::new();
    let anchor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 5.0)));
    let bob = add_ball(&mut world, Vec2::new(0.0, 4.0));
    let joint = pin(&mut world, anchor, bob, Vec2::new(0.0, 5.0));
    run(&mut world, 1.0);
    let hanging = world.body(bob).unwrap().position.y;

    world.destroy_body(anchor);
    assert!(
        world.joint(joint).is_none(),
        "the joint must go with its body"
    );
    assert_eq!(world.joint_count(), 0);
    run(&mut world, 1.0);
    let drop = hanging - world.body(bob).unwrap().position.y;
    assert!(
        (drop - 4.9254).abs() < 0.01,
        "the bob dropped {drop:.4} m in 1 s"
    );
}

#[test]
fn destroying_one_joint_keeps_the_handles_of_the_others() {
    // Three pendulums; the first joint is destroyed after 1 s. Box2D v3.2: its bob is at
    // y = -0.9255 one second later, and the other two still hang at y = 4.
    let mut world = PhysicsWorld2d::new();
    let anchor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 5.0)));
    let bobs: Vec<BodyId2d> = (0..3)
        .map(|i| add_ball(&mut world, Vec2::new(2.0 * i as f32, 4.0)))
        .collect();
    let joints: Vec<JointId2d> = bobs
        .iter()
        .enumerate()
        .map(|(i, &bob)| pin(&mut world, anchor, bob, Vec2::new(2.0 * i as f32, 5.0)))
        .collect();
    run(&mut world, 1.0);

    let removed = world.destroy_joint(joints[0]).expect("the joint exists");
    assert_eq!(removed.body_b, bobs[0]);
    assert!(
        world.destroy_joint(joints[0]).is_none(),
        "a joint is destroyed only once"
    );
    for i in 1..3 {
        let joint = world
            .joint(joints[i])
            .expect("the other joints keep their handles");
        assert_eq!(
            joint.body_b, bobs[i],
            "joint {i} now points at another body"
        );
    }

    // A new joint may reuse the freed slot, but never the stale handle.
    let replacement = pin(&mut world, anchor, bobs[1], Vec2::new(2.0, 5.0));
    assert_ne!(replacement, joints[0]);
    assert!(world.joint(joints[0]).is_none());
    world.destroy_joint(replacement);

    run(&mut world, 1.0);
    let y0 = world.body(bobs[0]).unwrap().position.y;
    assert!(
        (y0 + 0.9255).abs() < 0.01,
        "the freed bob is at y = {y0:.4}"
    );
    for &bob in &bobs[1..] {
        let p = world.body(bob).unwrap().position;
        let rest = Vec2::new(p.x.round(), 4.0);
        assert!((p - rest).length() < 0.01, "a pinned bob moved to {p:?}");
    }
}

#[test]
fn a_stale_body_handle_is_rejected_after_its_slot_is_reused() {
    let mut world = PhysicsWorld2d::new();
    let first = add_box(&mut world, BodyType::Dynamic, Vec2::ZERO, 0.5, 0.5);
    let collider = world.collider_of(first).expect("the box has a collider");
    assert!(world.destroy_body(first));
    assert!(
        world.collider(collider).is_none(),
        "the collider goes with its body"
    );

    let second = add_box(&mut world, BodyType::Dynamic, Vec2::new(3.0, 0.0), 0.5, 0.5);
    assert_ne!(first, second);
    assert!(world.body(first).is_none());
    assert!(world.body_mut(first).is_none());
    assert!(
        !world.destroy_body(first),
        "a stale handle must not destroy the new body"
    );
    assert_eq!(world.body(second).unwrap().position, Vec2::new(3.0, 0.0));
    assert_eq!(world.body_count(), 1);
    assert_eq!(
        world.bodies().map(|(id, _)| id).collect::<Vec<_>>(),
        vec![second]
    );
}

#[test]
fn destroying_the_collider_leaves_a_massless_body() {
    // Box2D v3.2: a falling box that loses its only shape after 0.5 s has no mass, so it keeps
    // its velocity (-4.905 m/s) and gravity no longer acts on it.
    let mut world = PhysicsWorld2d::new();
    let id = add_box(
        &mut world,
        BodyType::Dynamic,
        Vec2::new(0.0, 10.0),
        0.5,
        0.5,
    );
    run(&mut world, 0.5);
    let collider = world.collider_of(id).unwrap();
    assert!(world.destroy_collider(collider));
    assert!(world.collider(collider).is_none());
    assert!(world.collider_of(id).is_none());
    assert!(
        !world.destroy_collider(collider),
        "a collider is destroyed only once"
    );

    let body = world.body(id).expect("the body stays");
    assert_eq!(body.mass, 0.0);
    let v0 = body.linear_velocity.y;
    assert!((v0 + 4.905).abs() < 0.001, "the box fell at {v0:.4} m/s");
    run(&mut world, 1.0);
    let v1 = world.body(id).unwrap().linear_velocity.y;
    assert!(
        (v1 - v0).abs() < 1e-5,
        "the massless body's velocity changed by {:.5}",
        v1 - v0
    );
    assert!(world
        .query_aabb(
            &Aabb2d::new(Vec2::splat(-100.0), Vec2::splat(100.0)),
            QueryFilter2d::default()
        )
        .is_empty());
}
