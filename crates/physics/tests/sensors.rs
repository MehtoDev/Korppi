//! Sensors make no contacts and report the colliders that overlap them (P3D-02).
//!
//! A Box3D sensor never produces a collision response; at the end of each step it recomputes
//! which shapes overlap it and reports the changes as begin and end events (`sensor.c`). Only
//! convex shapes of other bodies take part, and only if both the sensor and the visitor
//! enable sensor events. Before FIX-3.1 `Collider::is_sensor` was never read: a sphere
//! dropped onto a sensor box came to rest on it (audit probe Q2).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_1_bodies.c`),
//! which runs each scene at 60 Hz with 4 substeps, 1 kg shapes, friction 0.6 and no sleep.

use korppi_math::Vec3;
use korppi_physics::{
    BodyId3d, BodyType, Collider, ColliderId3d, PhysicsWorld, RigidBody, SensorBeginEvent3d,
    SensorEndEvent3d, Shape,
};

const DT: f32 = 1.0 / 60.0;

fn body(body_type: BodyType, position: Vec3) -> RigidBody {
    let mut body = RigidBody::new(body_type, position).with_enable_sleep(false);
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    body
}

fn add(
    world: &mut PhysicsWorld,
    body_type: BodyType,
    position: Vec3,
    collider: Collider,
) -> (BodyId3d, ColliderId3d) {
    let id = world.create_body(body(body_type, position));
    let collider_id = world.create_collider(id, collider.with_friction(0.6).with_mass(1.0));
    (id, collider_id)
}

fn cuboid(half_extents: Vec3) -> Collider {
    Collider::new(Shape::Box { half_extents })
}

fn ball(radius: f32) -> Collider {
    Collider::new(Shape::Sphere { radius })
}

/// The ground, a static sensor box 1 m above it spanning y = 1.5 to 2.5, and a 0.5 m ball
/// dropped from y = 4. Returns the world, the sensor and the ball.
fn trigger_scene(
    sensor_events: bool,
    ball_events: bool,
) -> (
    PhysicsWorld,
    (BodyId3d, ColliderId3d),
    (BodyId3d, ColliderId3d),
) {
    let mut world = PhysicsWorld::new();
    add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, -0.5, 0.0),
        cuboid(Vec3::new(5.0, 0.5, 5.0)),
    );
    let sensor = add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, 2.0, 0.0),
        cuboid(Vec3::new(1.0, 0.5, 1.0))
            .with_sensor(true)
            .with_sensor_events(sensor_events),
    );
    let ball = add(
        &mut world,
        BodyType::Dynamic,
        Vec3::new(0.0, 4.0, 0.0),
        ball(0.5).with_sensor_events(ball_events),
    );
    (world, sensor, ball)
}

/// Sensor events with the step they were reported in, numbered from 1.
type Recorded = (
    Vec<(usize, SensorBeginEvent3d)>,
    Vec<(usize, SensorEndEvent3d)>,
);

/// Steps `steps` times and returns each step's sensor events.
fn record(world: &mut PhysicsWorld, steps: usize) -> Recorded {
    let (mut begins, mut ends) = (Vec::new(), Vec::new());
    for step in 1..=steps {
        world.step(DT);
        let events = world.sensor_events();
        begins.extend(events.begin.iter().map(|&e| (step, e)));
        ends.extend(events.end.iter().map(|&e| (step, e)));
    }
    (begins, ends)
}

#[test]
fn sphere_falls_through_a_sensor_and_is_reported() {
    let (mut world, (_sensor_body, sensor), (ball_body, ball)) = trigger_scene(true, true);

    let mut overlapping_steps = 0;
    let (mut begins, mut ends) = (Vec::new(), Vec::new());
    for step in 1..=120 {
        world.step(DT);
        let events = world.sensor_events();
        begins.extend(events.begin.iter().map(|&e| (step, e)));
        ends.extend(events.end.iter().map(|&e| (step, e)));
        if world.sensor_overlaps(sensor) == [ball] {
            overlapping_steps += 1;
        }
    }

    // Box3D: the ball passes the sensor and rests on the ground at y = 0.49993; probe Q2 saw it
    // rest on the sensor at y = 1.5.
    let y = world.body(ball_body).unwrap().position.y;
    assert!(
        (0.49..=0.51).contains(&y),
        "the ball should fall through the sensor onto the ground, it is at y = {y}"
    );
    // Box3D reports one begin event at step 27, when the ball's bottom passes the sensor's top
    // at y = 2.5, and one end event at step 47, when its top passes the bottom at y = 1.5.
    let expected = SensorBeginEvent3d {
        sensor,
        visitor: ball,
    };
    assert_eq!(begins.len(), 1, "one begin event, got {begins:?}");
    assert_eq!(begins[0].1, expected);
    assert!(
        (26..=28).contains(&begins[0].0),
        "the overlap begins at step 27 in Box3D, here at step {}",
        begins[0].0
    );
    assert_eq!(ends.len(), 1, "one end event, got {ends:?}");
    assert_eq!(
        ends[0].1,
        SensorEndEvent3d {
            sensor,
            visitor: ball,
        }
    );
    assert!(
        (46..=48).contains(&ends[0].0),
        "the overlap ends at step 47 in Box3D, here at step {}",
        ends[0].0
    );
    assert_eq!(
        overlapping_steps,
        ends[0].0 - begins[0].0,
        "the sensor lists the ball exactly while it overlaps"
    );
    assert!(world.sensor_overlaps(sensor).is_empty());
}

#[test]
fn sensor_events_need_both_colliders_to_enable_them() {
    // Box3D: one begin and one end with both enabled, nothing if either one is off.
    for (sensor_events, ball_events, expected) in
        [(true, true, 1), (true, false, 0), (false, true, 0)]
    {
        let (mut world, _, (ball_body, _)) = trigger_scene(sensor_events, ball_events);
        let (begins, ends) = record(&mut world, 60);
        assert_eq!(
            (begins.len(), ends.len()),
            (expected, expected),
            "sensor events {sensor_events}, ball events {ball_events}: got {begins:?} {ends:?}"
        );
        // Either way the sensor does not stop the ball.
        assert!(world.body(ball_body).unwrap().position.y < 1.5);
    }
}

#[test]
fn moving_sensor_reports_each_visitor_as_it_passes() {
    let mut world = PhysicsWorld::new();
    let mut sweeper = body(BodyType::Kinematic, Vec3::new(-3.0, 0.5, 0.0));
    sweeper.linear_velocity = Vec3::new(2.0, 0.0, 0.0);
    let sensor_body = world.create_body(sweeper);
    let sensor = world.create_collider(
        sensor_body,
        ball(0.5)
            .with_sensor(true)
            .with_sensor_events(true)
            .with_mass(1.0),
    );
    let visitor = |world: &mut PhysicsWorld, body_type, x| {
        add(
            world,
            body_type,
            Vec3::new(x, 0.25, 0.0),
            cuboid(Vec3::splat(0.25)).with_sensor_events(true),
        )
    };
    let first = visitor(&mut world, BodyType::Static, -1.0).1;
    let second = visitor(&mut world, BodyType::Static, 1.0).1;
    let third = visitor(&mut world, BodyType::Kinematic, 3.0).1;

    let (begins, ends) = record(&mut world, 240);

    // Box3D, as the sensor's centre passes within 0.75 m of each box: begins at steps 38, 98
    // and 158, ends at steps 83, 143 and 203.
    let begin = |step, visitor| (step, SensorBeginEvent3d { sensor, visitor });
    let end = |step, visitor| (step, SensorEndEvent3d { sensor, visitor });
    assert_eq!(
        begins,
        [begin(38, first), begin(98, second), begin(158, third)]
    );
    assert_eq!(ends, [end(83, first), end(143, second), end(203, third)]);
}

#[test]
fn sensor_on_a_dynamic_body_falls_through_the_ground() {
    let mut world = PhysicsWorld::new();
    let (_ground_body, ground) = add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.0, -0.5, 0.0),
        cuboid(Vec3::new(5.0, 0.5, 5.0)),
    );
    let (trigger_body, _trigger_col) = add(
        &mut world,
        BodyType::Dynamic,
        Vec3::new(0.0, 1.0, 0.0),
        cuboid(Vec3::splat(0.5)).with_sensor(true),
    );

    for _ in 0..60 {
        world.step(DT);
        assert!(
            world.manifolds().is_empty(),
            "a sensor makes no contact manifolds"
        );
    }

    // Box3D: y = -3.92544 after 1 s; a sensor still gives its body mass.
    let fallen = world.body(trigger_body).unwrap();
    assert!(
        fallen.position.y < -2.0,
        "nothing should stop a sensor body, it is at y = {}",
        fallen.position.y
    );
    assert!((fallen.mass - 1.0).abs() < 1e-6);
    assert!(world.sensor_overlaps(ground).is_empty(), "not a sensor");
    assert!(world.sensor_overlaps(1000).is_empty(), "no such body");
}

#[test]
fn overlapping_sensors_detect_each_other() {
    let mut world = PhysicsWorld::new();
    let sensor = || cuboid(Vec3::ONE).with_sensor(true).with_sensor_events(true);
    let (_a_body, a) = add(&mut world, BodyType::Static, Vec3::ZERO, sensor());
    let (_b_body, b) = add(
        &mut world,
        BodyType::Static,
        Vec3::new(0.5, 0.0, 0.0),
        sensor(),
    );

    let (begins, ends) = record(&mut world, 3);

    // Box3D: two begin events in the first step, one per sensor.
    assert_eq!(
        begins,
        [
            (
                1,
                SensorBeginEvent3d {
                    sensor: a,
                    visitor: b
                }
            ),
            (
                1,
                SensorBeginEvent3d {
                    sensor: b,
                    visitor: a
                }
            ),
        ]
    );
    assert!(ends.is_empty());
    assert_eq!(world.sensor_overlaps(a), [b]);
    assert_eq!(world.sensor_overlaps(b), [a]);
}

#[test]
fn sensor_overlaps_within_the_overlap_slop() {
    // Box3D tests sensor overlaps with GJK distance against `B3_OVERLAP_SLOP`, a tenth of the
    // linear slop (0.5 mm): a ball 0.4 mm beyond a sensor box's face overlaps it, one 0.6 mm
    // beyond does not (`fix_3_3_speculative.c`).
    for (gap, overlaps) in [(0.0004, true), (0.0006, false)] {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::ZERO);
        let (_sensor_body, sensor) = add(
            &mut world,
            BodyType::Static,
            Vec3::ZERO,
            cuboid(Vec3::splat(0.5))
                .with_sensor(true)
                .with_sensor_events(true),
        );
        let (_ball_body, ball) = add(
            &mut world,
            BodyType::Dynamic,
            Vec3::new(1.0 + gap, 0.0, 0.0),
            ball(0.5).with_sensor_events(true),
        );
        world.step(DT);
        assert_eq!(
            world.sensor_overlaps(sensor) == [ball],
            overlaps,
            "a ball {gap} m from the sensor: Box3D says overlapping = {overlaps}"
        );
    }
}
