//! World configuration and length units (P2D-26).
//!
//! Before FIX-2.4 the world's tolerances and limits were scattered constants, a world measured
//! in pixels could not rescale them, and sleep, continuous collision and warm starting could
//! not be switched off. As in Box2D v3.2 (`b2WorldDef`, `b2SetLengthUnitsPerMeter`), a
//! `WorldConfig2d` holds them. Box2D's length units are a process-wide setting; here they belong
//! to the world, so worlds with different units (and parallel tests) do not interfere. Expected
//! values are Box2D's, from `docs/audits/probes/box2d_reference/fix_2_4_world_config.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, WorldConfig2d};

const DT: f32 = 1.0 / 60.0;

fn run(world: &mut PhysicsWorld2d, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

fn add_ground(world: &mut PhysicsWorld2d, scale: f32) {
    let id = world.create_body(
        RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5 * scale)),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(
            20.0 * scale,
            0.5 * scale,
        )))
        .with_friction(0.6)
        .with_density(1.0 / (scale * scale)),
    );
}

fn add_box(world: &mut PhysicsWorld2d, position: Vec2, scale: f32) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(position));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(
            0.5 * scale,
            0.5 * scale,
        )))
        .with_friction(0.6)
        .with_density(1.0 / (scale * scale)),
    );
    id
}

/// A world `scale` units per metre of the scene, configured for `units` per metre.
fn scaled_world(scale: f32, units: f32) -> PhysicsWorld2d {
    let config = WorldConfig2d {
        enable_sleep: false,
        ..WorldConfig2d::with_length_units(units)
    };
    PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -9.81 * scale))
        .with_config(config)
}

/// The top of a five-box stack after 3 s, in metres of the scene.
fn stack_top(scale: f32, units: f32) -> f32 {
    let mut world = scaled_world(scale, units);
    add_ground(&mut world, scale);
    let mut top = None;
    for i in 0..5 {
        top = Some(add_box(
            &mut world,
            Vec2::new(0.0, (0.5 + i as f32) * scale),
            scale,
        ));
    }
    run(&mut world, 3.0);
    world.body(top.unwrap()).unwrap().position.y / scale
}

/// The speed of a box falling from 20 m after 1.5 s, in metres per second of the scene.
fn fall_speed(scale: f32, units: f32) -> f32 {
    let mut world = scaled_world(scale, units);
    let id = add_box(&mut world, Vec2::new(0.0, 20.0 * scale), scale);
    run(&mut world, 1.5);
    -world.body(id).unwrap().linear_velocity.y / scale
}

#[test]
fn a_world_in_pixels_behaves_like_one_in_metres() {
    // Box2D v3.2: the stack top rests at 4.4927 m either way; a box falls at 14.715 m/s after
    // 1.5 s in metres and in pixels with 100 units per metre, but only 4 m/s in pixels without
    // them, where the 400 m/s speed limit becomes 400 px/s.
    let metres = stack_top(1.0, 1.0);
    assert!((metres - 4.4927).abs() < 0.002, "stack top {metres:.5}");
    assert!((stack_top(100.0, 100.0) - metres).abs() < 1e-3);
    let fall = fall_speed(1.0, 1.0);
    assert!((fall - 14.715).abs() < 0.01, "fall speed {fall:.4}");
    assert!((fall_speed(100.0, 100.0) - fall).abs() < 1e-2);
    let capped = fall_speed(100.0, 1.0);
    assert!(
        (capped - 4.0).abs() < 1e-3,
        "unscaled pixels fall at {capped:.4} m/s"
    );
    let pixels = WorldConfig2d::with_length_units(100.0);
    assert!((pixels.linear_slop() - 0.5).abs() < 1e-6);
    assert!((pixels.maximum_linear_speed - 40_000.0).abs() < 1e-2);
}

#[test]
fn sleep_can_be_switched_off_and_on() {
    // Box2D v3.2: awake after 3 s with sleep off, asleep a second after switching it on, and
    // woken when switching it off again.
    let config = WorldConfig2d {
        enable_sleep: false,
        ..WorldConfig2d::default()
    };
    let mut world = PhysicsWorld2d::new().with_config(config);
    add_ground(&mut world, 1.0);
    let id = add_box(&mut world, Vec2::new(0.0, 0.5), 1.0);
    run(&mut world, 3.0);
    assert!(!world.body(id).unwrap().is_sleeping);
    world.enable_sleep(true);
    run(&mut world, 1.0);
    assert!(world.body(id).unwrap().is_sleeping);
    world.enable_sleep(false);
    assert!(!world.body(id).unwrap().is_sleeping);
}

fn fast_ball_x(continuous: bool) -> f32 {
    let config = WorldConfig2d {
        enable_continuous: continuous,
        ..WorldConfig2d::default()
    };
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::ZERO)
        .with_config(config);
    let wall =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(5.0, 0.0)));
    world.create_collider(
        wall,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.05, 2.0))),
    );
    let ball = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_linear_velocity(Vec2::new(50.0, 0.0)),
    );
    world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.1))),
    );
    run(&mut world, 1.0);
    world.body(ball).unwrap().position.x
}

#[test]
fn continuous_collision_can_be_switched_off() {
    // Box2D v3.2: the 50 m/s ball stops at the wall (x = 4.85) with continuous collision and
    // tunnels to x = 50 without.
    assert!((fast_ball_x(true) - 4.85).abs() < 0.01);
    assert!((fast_ball_x(false) - 50.0).abs() < 0.01);
}

fn ten_box_stack_top(warm_starting: bool) -> f32 {
    let config = WorldConfig2d {
        enable_sleep: false,
        enable_warm_starting: warm_starting,
        ..WorldConfig2d::default()
    };
    let mut world = PhysicsWorld2d::new().with_config(config);
    add_ground(&mut world, 1.0);
    let mut top = None;
    for i in 0..10 {
        top = Some(add_box(&mut world, Vec2::new(0.0, 0.5 + i as f32), 1.0));
    }
    run(&mut world, 2.0);
    world.body(top.unwrap()).unwrap().position.y
}

#[test]
fn warm_starting_can_be_switched_off() {
    // Box2D v3.2: the ten-box stack's top is at 9.4681 with warm starting and sinks to 9.3584
    // without it.
    let warm = ten_box_stack_top(true);
    let cold = ten_box_stack_top(false);
    assert!((warm - 9.4681).abs() < 0.005, "warm {warm:.4}");
    assert!((cold - 9.3584).abs() < 0.005, "cold {cold:.4}");
}

#[test]
fn limits_and_tuning_follow_box2d() {
    // Maximum linear speed 10 m/s: a falling box is capped at it.
    let config = WorldConfig2d {
        maximum_linear_speed: 10.0,
        ..WorldConfig2d::default()
    };
    let mut world = PhysicsWorld2d::new().with_config(config);
    let id = add_box(&mut world, Vec2::new(0.0, 100.0), 1.0);
    run(&mut world, 2.0);
    assert!((world.body(id).unwrap().linear_velocity.y + 10.0).abs() < 1e-3);

    // Contact push speed. Box2D v3.2: overlapping boxes separate at 0.17239 m/s after one step
    // at push speed 3, and 0.01437 m/s at 0.25.
    for (push_speed, expected) in [(3.0, 0.17239), (0.25, 0.01437)] {
        let config = WorldConfig2d {
            contact_push_speed: push_speed,
            ..WorldConfig2d::default()
        };
        let mut world = PhysicsWorld2d::new()
            .with_gravity(Vec2::ZERO)
            .with_config(config);
        add_box(&mut world, Vec2::ZERO, 1.0);
        let other = add_box(&mut world, Vec2::new(0.5, 0.0), 1.0);
        world.step(DT);
        let speed = world.body(other).unwrap().linear_velocity.length();
        assert!(
            (speed - expected).abs() < 0.001,
            "push speed {push_speed}: {speed:.5}"
        );
    }

    // Restitution threshold. Box2D v3.2: an elastic ball hitting at about 2 m/s bounces to
    // 0.6917 with the threshold at 1 m/s and not at all at 3 m/s.
    for (threshold, expected) in [(1.0, 0.69169), (3.0, 0.0)] {
        let config = WorldConfig2d {
            enable_sleep: false,
            restitution_threshold: threshold,
            ..WorldConfig2d::default()
        };
        let mut world = PhysicsWorld2d::new().with_config(config);
        add_ground(&mut world, 1.0);
        let ball = world
            .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 0.7)));
        world.create_collider(
            ball,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))).with_restitution(1.0),
        );
        let (mut apex, mut bounced) = (0.0_f32, false);
        for _ in 0..90 {
            world.step(DT);
            let body = world.body(ball).unwrap();
            bounced |= body.linear_velocity.y > 0.1;
            if bounced {
                apex = apex.max(body.position.y);
            }
        }
        assert!(
            (apex - expected).abs() < 0.005,
            "threshold {threshold}: apex {apex:.4}"
        );
    }

    // Contact stiffness. Box2D v3.2: a five-box stack's top at 4.4927 with 30 Hz contacts,
    // 4.3084 with 5 Hz.
    for (hertz, expected) in [(30.0, 4.49267), (5.0, 4.30839)] {
        let config = WorldConfig2d {
            enable_sleep: false,
            contact_hertz: hertz,
            ..WorldConfig2d::default()
        };
        let mut world = PhysicsWorld2d::new().with_config(config);
        add_ground(&mut world, 1.0);
        let mut top = None;
        for i in 0..5 {
            top = Some(add_box(&mut world, Vec2::new(0.0, 0.5 + i as f32), 1.0));
        }
        run(&mut world, 2.0);
        let y = world.body(top.unwrap()).unwrap().position.y;
        assert!((y - expected).abs() < 0.005, "{hertz} Hz: top {y:.4}");
    }
}

#[test]
fn counters_and_profile_describe_the_world() {
    // Box2D b2World_GetCounters and b2World_GetProfile: a ground, a box resting on it and a
    // falling one.
    let mut world = PhysicsWorld2d::new();
    add_ground(&mut world, 1.0);
    add_box(&mut world, Vec2::new(0.0, 0.5), 1.0);
    add_box(&mut world, Vec2::new(5.0, 10.0), 1.0);
    world.step(DT);
    let counters = world.counters();
    assert_eq!(
        (
            counters.body_count,
            counters.awake_body_count,
            counters.collider_count
        ),
        (3, 2, 3)
    );
    assert_eq!((counters.contact_count, counters.joint_count), (1, 0));
    assert!(counters.tree_height >= 1);
    assert!(counters.color_count >= 1);
    let profile = *world.profile();
    assert!(profile.step > 0.0);
    let parts = profile.broadphase
        + profile.narrowphase
        + profile.solve
        + profile.continuous
        + profile.islands
        + profile.sensors;
    assert!(parts <= profile.step * 1.001 + 1e-3, "{profile:?}");
}
