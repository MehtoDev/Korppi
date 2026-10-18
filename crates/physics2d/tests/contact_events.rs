//! Contact begin, end and hit events, body move events and contact data (P2D-11).
//!
//! Before FIX-2.4 the world reported nothing but its manifold list. As in Box2D v3.2, colliders
//! opt in to contact and hit events (both off by default); begin and hit events describe the
//! last step, and end events also include contacts destroyed by calls since the step before it.
//! Every awake body reports a move event per step, flagged when it falls asleep. Expected values
//! are Box2D's, from `docs/audits/probes/box2d_reference/fix_2_4_contact_events.c`.

use korppi_math::{Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, ColliderId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn add_ground(world: &mut PhysicsWorld2d) -> (BodyId2d, ColliderId2d) {
    let id =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    let collider = world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))),
    );
    (id, collider)
}

fn add_box(
    world: &mut PhysicsWorld2d,
    position: Vec2,
    collider: Collider2d,
) -> (BodyId2d, ColliderId2d) {
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(position));
    let collider_id = world.create_collider(id, collider);
    (id, collider_id)
}

fn unit_box() -> Collider2d {
    Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
}

#[test]
fn begin_and_end_events_follow_box2d() {
    let mut world = PhysicsWorld2d::new();
    let (floor, floor_collider) = add_ground(&mut world);
    let (crate_box, box_collider) = add_box(
        &mut world,
        Vec2::new(0.0, 2.0),
        unit_box().with_contact_events(true),
    );

    // Box2D v3.2: one begin event, at step 34, and no end event while it rests.
    let (mut begin_step, mut begins, mut ends) = (None, 0, 0);
    for step in 1..=45 {
        world.step(DT);
        let events = world.contact_events();
        if let Some(event) = events.begin.first() {
            begin_step.get_or_insert(step);
            let pair = [event.collider_a, event.collider_b];
            assert!(
                pair.contains(&floor_collider) && pair.contains(&box_collider),
                "{event:?}"
            );
        }
        begins += events.begin.len();
        ends += events.end.len();
    }
    assert_eq!(begin_step, Some(34), "the box must land at step 34");
    assert_eq!((begins, ends), (1, 0));

    // Lifted clear while awake: one end event on the next step.
    assert!(!world.body(crate_box).unwrap().is_sleeping);
    world.set_transform(crate_box, Vec2::new(0.0, 3.0), Rot2::IDENTITY);
    world.step(DT);
    assert_eq!(
        (
            world.contact_events().begin.len(),
            world.contact_events().end.len()
        ),
        (0, 1)
    );

    // It lands and falls asleep; destroying the ground ends the contact, reported with the next
    // step.
    for _ in 0..120 {
        world.step(DT);
    }
    assert!(world.body(crate_box).unwrap().is_sleeping);
    world.destroy_body(floor);
    assert!(
        world.contact_events().end.is_empty(),
        "end events belong to the next step"
    );
    world.step(DT);
    let end = world.contact_events().end.clone();
    assert_eq!(end.len(), 1);
    assert!([end[0].collider_a, end[0].collider_b].contains(&floor_collider));
}

#[test]
fn colliders_report_no_events_unless_enabled() {
    let mut world = PhysicsWorld2d::new();
    add_ground(&mut world);
    add_box(&mut world, Vec2::new(0.0, 2.0), unit_box());
    for _ in 0..120 {
        world.step(DT);
        let events = world.contact_events();
        assert!(events.begin.is_empty() && events.end.is_empty() && events.hit.is_empty());
    }
}

#[test]
fn a_landing_box_reports_one_hit() {
    // Box2D v3.2: one hit, at step 39, approaching at 6.2130 m/s, at a corner midway between the
    // surfaces (y = 0.0098), with the normal pointing from the ground (A) to the box.
    let mut world = PhysicsWorld2d::new();
    let (_, floor_collider) = add_ground(&mut world);
    add_box(
        &mut world,
        Vec2::new(0.0, 2.5),
        unit_box().with_hit_events(true),
    );
    let mut hits = Vec::new();
    for step in 1..=120 {
        world.step(DT);
        hits.extend(world.contact_events().hit.iter().map(|hit| (step, *hit)));
    }
    assert_eq!(hits.len(), 1, "{hits:?}");
    let (step, hit) = hits[0];
    assert_eq!(step, 39);
    assert!(
        (hit.approach_speed - 6.2130).abs() < 0.01,
        "approach {:.4}",
        hit.approach_speed
    );
    assert!(
        (hit.point.y - 0.0098).abs() < 0.005 && (hit.point.x.abs() - 0.5).abs() < 0.001,
        "point {:?}",
        hit.point
    );
    let up = if hit.collider_a == floor_collider {
        1.0
    } else {
        -1.0
    };
    assert!(
        (hit.normal - Vec2::new(0.0, up)).length() < 1e-5,
        "normal {:?}",
        hit.normal
    );
}

#[test]
fn awake_bodies_report_moves_until_they_fall_asleep() {
    // Box2D v3.2: the resting box and the kinematic body both move in the first step; the box
    // falls asleep at step 30 and reports no move after.
    let mut world = PhysicsWorld2d::new();
    add_ground(&mut world);
    let (crate_box, _) = add_box(&mut world, Vec2::new(0.0, 0.5), unit_box());
    let kinematic = world.create_body(
        RigidBody2d::new(BodyType::Kinematic)
            .with_position(Vec2::new(5.0, 5.0))
            .with_linear_velocity(Vec2::new(1.0, 0.0)),
    );
    world.step(DT);
    let moved: Vec<BodyId2d> = world.body_events().iter().map(|e| e.body).collect();
    assert_eq!(moved, vec![crate_box, kinematic]);
    let event = world.body_events()[1];
    assert_eq!(
        event.transform.translation,
        world.body(kinematic).unwrap().position
    );

    let (mut asleep_step, mut moves_after) = (None, 0);
    for step in 2..=120 {
        world.step(DT);
        for event in world.body_events().iter().filter(|e| e.body == crate_box) {
            if asleep_step.is_some() {
                moves_after += 1;
            }
            if event.fell_asleep {
                asleep_step = Some(step);
            }
        }
    }
    assert_eq!(asleep_step, Some(30));
    assert_eq!(moves_after, 0);
}

#[test]
fn contact_data_lists_touching_contacts() {
    // A box resting on the ground touches it with one two-point manifold (Box2D
    // b2Body_GetContactData, b2Shape_GetContactData).
    let mut world = PhysicsWorld2d::new();
    let (floor, floor_collider) = add_ground(&mut world);
    let (crate_box, box_collider) = add_box(&mut world, Vec2::new(0.0, 0.5), unit_box());
    world.step(DT);
    let contacts: Vec<_> = world.body_contacts(crate_box).collect();
    assert_eq!(contacts.len(), 1);
    let pair = [contacts[0].collider_a, contacts[0].collider_b];
    assert!(pair.contains(&floor_collider) && pair.contains(&box_collider));
    assert_eq!(contacts[0].manifold.point_count, 2);
    assert_eq!(world.collider_contacts(floor_collider).count(), 1);
    assert_eq!(world.body_contacts(floor).count(), 1);
}
