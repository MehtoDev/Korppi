//! Box3D events and contact queries (P3D-15).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_6_events.c`):
//! 60 Hz, 4 substeps, static ground, dynamic shapes.

use korppi_math::Vec3;
use korppi_physics::{
    BodyType, Collider, ContactHitEvent3d, Joint, PhysicsWorld, RigidBody, Shape, SphericalJoint,
};

const DT: f32 = 1.0 / 60.0;

#[test]
fn contact_begin_and_end_events() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    let ground_col = world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_friction(0.5),
    );

    let ball = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0)));
    let ball_col = world.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_restitution(0.8)
            .with_friction(0.5)
            .with_contact_events(true),
    );

    let mut total_begins = 0;
    let mut total_ends = 0;

    for _ in 0..90 {
        world.step(DT);
        let events = world.contact_events();
        for begin in &events.begin {
            assert!(
                (begin.collider_a == ground_col && begin.collider_b == ball_col)
                    || (begin.collider_a == ball_col && begin.collider_b == ground_col),
                "colliders match"
            );
            assert!(
                (begin.body_a == ground && begin.body_b == ball)
                    || (begin.body_a == ball && begin.body_b == ground),
                "bodies match"
            );
        }
        total_begins += events.begin.len();
        total_ends += events.end.len();
    }

    assert!(
        (1..=5).contains(&total_begins),
        "total begins: {total_begins}"
    );
    assert!((1..=5).contains(&total_ends), "total ends: {total_ends}");
}

#[test]
fn contact_hit_events() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    let ground_col = world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_friction(0.5),
    );

    let ball = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0))
            .with_linear_velocity(Vec3::new(0.0, -20.0, 0.0)),
    );
    let ball_col = world.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_friction(0.5)
            .with_hit_events(true),
    );

    let mut hit_count = 0;
    let mut hit_event: Option<ContactHitEvent3d> = None;

    for _ in 0..20 {
        world.step(DT);
        let events = world.contact_events();
        if !events.hit.is_empty() && hit_count == 0 {
            hit_count += events.hit.len();
            hit_event = Some(events.hit[0]);
        }
    }

    assert_eq!(hit_count, 1, "exactly 1 hit event triggered");
    let hit = hit_event.expect("hit captured");
    assert!(
        (hit.approach_speed - 20.0).abs() <= 1.0,
        "speed: {}",
        hit.approach_speed
    );
    assert!(
        (hit.normal.y - 1.0).abs() <= 0.05,
        "normal: {:?}",
        hit.normal
    );
    assert!((hit.point.y - 0.0).abs() <= 0.1, "point: {:?}", hit.point);
    assert_eq!(hit.collider_a, ground_col);
    assert_eq!(hit.collider_b, ball_col);
    assert_eq!(hit.body_a, ground);
    assert_eq!(hit.body_b, ball);
}

#[test]
fn body_move_events() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let ball = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 10.0, 0.0)));
    world.create_collider(ball, Collider::new(Shape::Sphere { radius: 0.5 }));

    world.step(DT);
    let events = world.body_events();

    assert_eq!(events.move_events.len(), 1);
    let mv = events.move_events[0];
    assert_eq!(mv.body, ball);
    assert!(
        mv.position.y > 9.99 && mv.position.y < 9.999,
        "y: {}",
        mv.position.y
    );
    assert!(!mv.fell_asleep, "fell asleep false");
}

#[test]
fn joint_events() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let b1 = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 5.0, 0.0)));
    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 4.0, 0.0)));
    // Heavy 100 kg body
    world.create_collider(
        b2,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_density(100_000.0),
    );

    let joint_id = world.create_joint(
        Joint::Spherical(SphericalJoint::new(
            b1,
            b2,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        ))
        .with_force_threshold(5.0),
    );

    world.step(DT);
    let events = world.joint_events();

    assert_eq!(events.events.len(), 1);
    assert_eq!(events.events[0].joint, joint_id);
    assert!(events.events[0].linear_force >= 5.0);
}

#[test]
fn contact_queries() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    let ground_col = world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_friction(0.5),
    );

    let box_body = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    let box_col = world.create_collider(
        box_body,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.5),
    );

    for _ in 0..60 {
        world.step(DT);
    }

    // Query contacts of box body
    let body_contacts: Vec<_> = world.body_contacts(box_body).collect();
    assert_eq!(body_contacts.len(), 1);
    let c = body_contacts[0];
    assert!(c.is_touching);
    assert_eq!(c.manifolds().len(), 1);
    assert_eq!(c.manifolds()[0].points.len(), 4);

    // Query contacts of box collider
    let col_contacts: Vec<_> = world.collider_contacts(box_col).collect();
    assert_eq!(col_contacts.len(), 1);

    // Query contacts of ground collider
    let ground_contacts: Vec<_> = world.collider_contacts(ground_col).collect();
    assert_eq!(ground_contacts.len(), 1);
}
