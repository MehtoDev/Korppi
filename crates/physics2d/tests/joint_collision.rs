//! Bodies joined by a joint do not collide with each other unless the joint allows it (P2D-09).
//!
//! Box2D's `b2JointDef::collideConnected` defaults to `false`: contacts between the two bodies of
//! a joint are destroyed when the joint is created and are not created while it exists. Before
//! FIX-2.1 jointed bodies always collided, so overlapping ragdoll limbs fought their joints
//! (audit probe P6).

use korppi_math::{Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{Joint2d, PrismaticJoint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, JointInstance2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn unit_box() -> Collider2d {
    Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
}

/// Two dynamic unit boxes overlapping by 0.1 m, in a world without gravity.
fn overlapping_boxes() -> (PhysicsWorld2d, BodyId2d, BodyId2d) {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let a = world.create_body(RigidBody2d::new(BodyType::Dynamic));
    let b =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.9, 0.0)));
    world.create_collider(a, unit_box());
    world.create_collider(b, unit_box());
    (world, a, b)
}

/// The revolute joint of probe P6, pinning the boxes at the middle of their overlap.
fn pin() -> Joint2d {
    Joint2d::Revolute(RevoluteJoint2d::new(
        Vec2::new(0.45, 0.0),
        Vec2::new(-0.45, 0.0),
        0.0,
        0.0,
    ))
}

fn touching(world: &PhysicsWorld2d, a: BodyId2d, b: BodyId2d) -> bool {
    world
        .manifolds()
        .iter()
        .any(|&(x, y, ref m)| (x, y) == (a.min(b), a.max(b)) && m.point_count > 0)
}

fn relative_angle(world: &PhysicsWorld2d, a: BodyId2d, b: BodyId2d) -> f32 {
    let qa: Rot2 = world.body(a).unwrap().rotation;
    let qb: Rot2 = world.body(b).unwrap().rotation;
    (qa.c * qb.s - qa.s * qb.c).atan2(qa.c * qb.c + qa.s * qb.s)
}

#[test]
fn jointed_bodies_do_not_collide_by_default() {
    let (mut world, a, b) = overlapping_boxes();
    world.create_joint(a, b, pin());

    let mut twist = 0.0f32;
    for _ in 0..60 {
        world.step(DT);
        assert!(!touching(&world, a, b), "jointed boxes produced a contact");
        twist = twist.max(relative_angle(&world, a, b).abs());
    }
    assert!(
        twist < 1e-3,
        "jointed boxes twisted {twist:.4} rad (probe P6: 0.024)"
    );
    let drift = (world.body(b).unwrap().position - Vec2::new(0.9, 0.0)).length();
    assert!(drift < 1e-3, "jointed boxes pushed each other {drift:.4} m");
}

#[test]
fn collide_connected_lets_jointed_bodies_collide() {
    let (mut world, a, b) = overlapping_boxes();
    let instance = JointInstance2d::new(a, b, pin()).with_collide_connected(true);
    assert!(instance.collide_connected);
    world.add_joint(instance);
    world.step(DT);
    assert!(
        touching(&world, a, b),
        "collide_connected joint suppressed the contact"
    );
}

#[test]
fn creating_a_joint_removes_the_contact_between_its_bodies() {
    let (mut world, a, b) = overlapping_boxes();
    world.step(DT);
    assert!(
        touching(&world, a, b),
        "overlapping boxes must touch before the joint exists"
    );

    world.create_joint(a, b, pin());
    let bodies = [a.index(), b.index()];
    assert!(
        !world
            .contact_manager()
            .values()
            .any(|c| bodies.contains(&c.body_a) && bodies.contains(&c.body_b)),
        "the contact must be destroyed when the joint is created"
    );
    world.step(DT);
    assert!(!touching(&world, a, b));
}

#[test]
fn removing_the_joint_restores_collision() {
    // A box on a vertical rail that belongs to a floor: the rail lets it fall through the floor,
    // removing the rail lets it land. Box2D creates contacts only for bodies that move, so two
    // resting bodies stay apart after their joint is removed until one moves; this scene keeps
    // the box moving. Box2D v3.2: y = -17.0 on the rail; without it the box never sinks below
    // y = 0.99996 and rests there. How deep the box sinks on impact is the contact solver's
    // accuracy (P2D-05), not this joint's; here it only has to be stopped by the floor.
    let drop = |keep_rail: bool| {
        let mut world = PhysicsWorld2d::new();
        let floor = world.create_body(RigidBody2d::new(BodyType::Static));
        world.create_collider(
            floor,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(5.0, 0.5))),
        );
        let box_body = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 3.0)));
        world.create_collider(box_body, unit_box());
        let rail = world.create_joint(
            floor,
            box_body,
            Joint2d::Prismatic(PrismaticJoint2d::new(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::Y,
                0.0,
                0.0,
            )),
        );
        world.step(DT);
        if !keep_rail {
            world.destroy_joint(rail);
        }
        let mut lowest = f32::INFINITY;
        for _ in 0..120 {
            world.step(DT);
            lowest = lowest.min(world.body(box_body).unwrap().position.y);
        }
        (lowest, world.body(box_body).unwrap().position.y)
    };
    let (_, on_rail) = drop(true);
    assert!(
        on_rail < -1.0,
        "the rail must let the box pass the floor (y = {on_rail:.3})"
    );
    let (lowest, landed) = drop(false);
    assert!(
        lowest > 0.5,
        "without the rail the floor must stop the box, its centre reached y = {lowest:.4}"
    );
    assert!(
        (0.9..=1.1).contains(&landed),
        "without the rail the box must stay on the floor at y = 1.0, it is at {landed:.4}"
    );
}

#[test]
fn jointed_bodies_still_collide_with_other_bodies() {
    let (mut world, a, b) = overlapping_boxes();
    world.create_joint(a, b, pin());
    let c =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 0.9)));
    world.create_collider(c, unit_box());
    world.step(DT);
    assert!(
        touching(&world, a, c),
        "a third body must still collide with a jointed body"
    );
}

#[test]
fn a_bullet_passes_through_the_body_it_is_jointed_to() {
    // A bullet sliding on a prismatic rail that belongs to a static wall it would otherwise hit.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let wall =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(5.0, 0.0)));
    world.create_collider(
        wall,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.05, 2.0))),
    );
    let bullet = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_bullet(true)
            .with_linear_velocity(Vec2::new(120.0, 0.0)),
    );
    world.create_collider(
        bullet,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.1))),
    );
    world.create_joint(
        wall,
        bullet,
        Joint2d::Prismatic(PrismaticJoint2d::new(
            Vec2::ZERO,
            Vec2::ZERO,
            Vec2::X,
            0.0,
            0.0,
        )),
    );
    for _ in 0..10 {
        world.step(DT);
    }
    let body = world.body(bullet).unwrap();
    assert!(
        body.position.x > 15.0,
        "the bullet stopped at x = {:.3} against the body it is jointed to",
        body.position.x
    );
    assert!((body.linear_velocity.x - 120.0).abs() < 1e-3);
}
