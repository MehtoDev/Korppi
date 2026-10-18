//! Sensors detect overlapping colliders and report begin and end events (P2D-11).
//!
//! Before FIX-2.4 a sensor collider simply never collided: it was undetectable, so triggers,
//! pickups and damage zones were impossible. As in Box2D v3.2, a sensor computes its overlaps at
//! the end of each step with colliders that, like it, enable sensor events; a fast body that
//! crosses a sensor within one step is caught by its continuous sweep. End events double-buffer
//! like contact end events. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_4_sensors.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, ColliderId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn add_sensor(
    world: &mut PhysicsWorld2d,
    kind: BodyType,
    position: Vec2,
    hx: f32,
    hy: f32,
) -> (BodyId2d, ColliderId2d) {
    let body = world.create_body(RigidBody2d::new(kind).with_position(position));
    let collider = world.create_collider(
        body,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy)))
            .with_sensor(true)
            .with_sensor_events(true),
    );
    (body, collider)
}

fn add_ball(
    world: &mut PhysicsWorld2d,
    position: Vec2,
    velocity: Vec2,
    radius: f32,
    events: bool,
) -> (BodyId2d, ColliderId2d) {
    let body = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(position)
            .with_linear_velocity(velocity),
    );
    let collider = world.create_collider(
        body,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, radius)))
            .with_sensor_events(events),
    );
    (body, collider)
}

/// Steps `steps` times, returning the first begin step, the first end step and the event counts.
fn tally(world: &mut PhysicsWorld2d, steps: usize) -> (Option<usize>, Option<usize>, usize, usize) {
    let (mut first_begin, mut first_end, mut begins, mut ends) = (None, None, 0, 0);
    for step in 1..=steps {
        world.step(DT);
        let events = world.sensor_events();
        if !events.begin.is_empty() {
            first_begin.get_or_insert(step);
        }
        if !events.end.is_empty() {
            first_end.get_or_insert(step);
        }
        begins += events.begin.len();
        ends += events.end.len();
    }
    (first_begin, first_end, begins, ends)
}

#[test]
fn a_ball_falls_through_a_sensor() {
    // Box2D v3.2: the ball enters at step 41, leaves at step 54 and falls on through.
    let mut world = PhysicsWorld2d::new();
    let (_, sensor) = add_sensor(&mut world, BodyType::Static, Vec2::ZERO, 2.0, 0.5);
    let (ball, visitor) = add_ball(&mut world, Vec2::new(0.0, 3.0), Vec2::ZERO, 0.25, true);
    let (mut first_begin, mut first_end, mut begins, mut ends) = (None, None, 0, 0);
    for step in 1..=90 {
        world.step(DT);
        let events = world.sensor_events();
        for event in &events.begin {
            assert_eq!((event.sensor, event.visitor), (sensor, visitor));
            first_begin.get_or_insert(step);
        }
        for event in &events.end {
            assert_eq!((event.sensor, event.visitor), (sensor, visitor));
            first_end.get_or_insert(step);
        }
        begins += events.begin.len();
        ends += events.end.len();
        let overlaps: Vec<ColliderId2d> = world.sensor_overlaps(sensor).collect();
        let inside = first_begin.is_some() && first_end.is_none();
        assert_eq!(
            overlaps,
            if inside { vec![visitor] } else { vec![] },
            "step {step}"
        );
    }
    assert_eq!(
        (first_begin, first_end, begins, ends),
        (Some(41), Some(54), 1, 1)
    );
    assert!(
        world.body(ball).unwrap().position.y < -3.0,
        "a sensor must not stop the ball"
    );
}

#[test]
fn no_events_unless_both_colliders_enable_them() {
    let mut world = PhysicsWorld2d::new();
    add_sensor(&mut world, BodyType::Static, Vec2::ZERO, 2.0, 0.5);
    add_ball(&mut world, Vec2::new(0.0, 3.0), Vec2::ZERO, 0.25, false);
    assert_eq!(tally(&mut world, 90), (None, None, 0, 0));
}

#[test]
fn a_fast_ball_crossing_a_thin_sensor_is_caught() {
    // A 100 m/s ball and a 0.1 m sensor. Box2D v3.2: begins at step 3; step 4's sweep starts
    // inside the sensor, which counts as a hit, so the overlap ends at step 5.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    add_sensor(&mut world, BodyType::Static, Vec2::new(5.0, 0.0), 0.05, 2.0);
    add_ball(&mut world, Vec2::ZERO, Vec2::new(100.0, 0.0), 0.1, true);
    assert_eq!(tally(&mut world, 10), (Some(3), Some(5), 1, 1));
}

#[test]
fn a_fast_ball_that_skips_over_a_sensor_is_caught() {
    // The same ball starting 0.8 m further back is outside the sensor at every step's end (x =
    // 4.2 after step 3, 5.867 after step 4), so only the continuous sweep sees it pass. Box2D
    // v3.2: begin at step 4, end at step 5.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    add_sensor(&mut world, BodyType::Static, Vec2::new(5.0, 0.0), 0.05, 2.0);
    add_ball(
        &mut world,
        Vec2::new(-0.8, 0.0),
        Vec2::new(100.0, 0.0),
        0.1,
        true,
    );
    assert_eq!(tally(&mut world, 10), (Some(4), Some(5), 1, 1));
}

#[test]
fn destroying_the_visitor_or_the_sensor_ends_the_overlap() {
    // Box2D v3.2: either way the end event comes with the next step.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let (sensor_body, sensor) = add_sensor(&mut world, BodyType::Static, Vec2::ZERO, 2.0, 2.0);
    let (ball, _) = add_ball(&mut world, Vec2::ZERO, Vec2::ZERO, 0.25, true);
    world.step(DT);
    assert_eq!(world.sensor_events().begin.len(), 1);
    world.destroy_body(ball);
    assert!(world.sensor_events().end.is_empty());
    world.step(DT);
    assert_eq!(world.sensor_events().end.len(), 1);

    let (_, visitor) = add_ball(&mut world, Vec2::ZERO, Vec2::ZERO, 0.25, true);
    world.step(DT);
    assert_eq!(world.sensor_events().begin.len(), 1);
    world.destroy_collider(sensor);
    assert!(world.sensor_events().end.is_empty());
    world.step(DT);
    let end = world.sensor_events().end.clone();
    assert_eq!(end.len(), 1);
    assert_eq!((end[0].sensor, end[0].visitor), (sensor, visitor));
    assert!(world.body(sensor_body).is_some());
}

#[test]
fn sensors_detect_each_other() {
    // Box2D v3.2: two overlapping sensors that enable sensor events each report the other.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let (_, a) = add_sensor(&mut world, BodyType::Static, Vec2::ZERO, 1.0, 1.0);
    let (_, b) = add_sensor(&mut world, BodyType::Dynamic, Vec2::new(0.5, 0.0), 1.0, 1.0);
    world.step(DT);
    let mut pairs: Vec<_> = world
        .sensor_events()
        .begin
        .iter()
        .map(|e| (e.sensor, e.visitor))
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![(a, b), (b, a)]);
}

#[test]
fn a_moving_sensor_finds_static_colliders() {
    // A kinematic sensor at 6 m/s sweeps over three static posts. Box2D v3.2: three begins and
    // three ends, the first entering at step 14 and leaving at step 27.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let (sensor_body, _) = add_sensor(&mut world, BodyType::Kinematic, Vec2::ZERO, 0.5, 0.5);
    world.body_mut(sensor_body).unwrap().linear_velocity = Vec2::new(6.0, 0.0);
    for i in 0..3 {
        let post = world.create_body(
            RigidBody2d::new(BodyType::Static).with_position(Vec2::new(2.0 + 2.0 * i as f32, 0.0)),
        );
        world.create_collider(
            post,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.1, 0.1)))
                .with_sensor_events(true),
        );
    }
    assert_eq!(tally(&mut world, 90), (Some(14), Some(27), 3, 3));
}
