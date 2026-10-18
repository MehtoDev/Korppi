//! Contacts follow Box2D v3's Soft Step solver (P2D-05) and restitution pass (P2D-02).
//!
//! Before FIX-2.2 contacts were warm-started once per step, solved twice with bias and never
//! relaxed, so push-out velocity stayed in bodies: a ball dropped 2 m bounced to 0.26 m whatever
//! its restitution (audit probe P7), a dropped box sank 5.8 cm, and resting stacks kept moving.
//! Expected values are Box2D v3.2's in the same scenes, from
//! `docs/audits/probes/box2d_reference/fix_2_2_soft_step.c`.

use korppi_math::{Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

/// A static floor whose top face is at y = 0.
fn add_floor(world: &mut PhysicsWorld2d, friction: f32) {
    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(friction),
    );
}

fn add_box(world: &mut PhysicsWorld2d, position: Vec2, angle: f32, friction: f32) -> BodyId2d {
    let id = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(position)
            .with_rotation(Rot2::from_angle(angle)),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))).with_friction(friction),
    );
    id
}

fn add_ball(world: &mut PhysicsWorld2d, height: f32, restitution: f32) -> BodyId2d {
    let id = world.create_body(
        RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, height + 0.25)),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25)))
            .with_friction(0.3)
            .with_restitution(restitution),
    );
    id
}

/// Probe P7: a ball of radius 0.25 dropped 2 m; height of its lowest point at the first apex.
fn bounce_apex(restitution: f32) -> f32 {
    let mut world = PhysicsWorld2d::new();
    add_floor(&mut world, 0.3);
    let ball = add_ball(&mut world, 2.0, restitution);
    let (mut bounced, mut apex) = (false, 0.0f32);
    for _ in 0..240 {
        world.step(DT);
        let body = world.body(ball).unwrap();
        if body.linear_velocity.y > 0.0 {
            bounced = true;
        }
        if bounced {
            apex = apex.max(body.position.y - 0.25);
            if body.linear_velocity.y < 0.0 {
                break;
            }
        }
    }
    apex
}

#[test]
fn restitution_sets_the_bounce_height() {
    // Box2D v3.2: 0, 0.485 and 1.954 m.
    let dead = bounce_apex(0.0);
    assert!(
        dead < 0.01,
        "an inelastic ball bounced to {dead:.4} m (probe P7: 0.257)"
    );
    let half = bounce_apex(0.5);
    assert!(
        (half - 0.485).abs() < 0.03,
        "e = 0.5 bounced to {half:.4} m, Box2D 0.485 (probe P7: 0.317)"
    );
    let elastic = bounce_apex(1.0);
    assert!(
        (elastic - 1.954).abs() < 0.05,
        "e = 1 bounced to {elastic:.4} m, Box2D 1.954 (probe P7: 0.257)"
    );
}

#[test]
fn slow_impacts_do_not_bounce() {
    // Below the 1 m/s restitution threshold an elastic ball dropped 2 cm stays down (Box2D: 0).
    let mut world = PhysicsWorld2d::new();
    add_floor(&mut world, 0.3);
    let ball = add_ball(&mut world, 0.02, 1.0);
    let (mut landed, mut highest) = (false, 0.0f32);
    for _ in 0..60 {
        world.step(DT);
        let height = world.body(ball).unwrap().position.y - 0.25;
        landed = landed || height < 0.005;
        if landed {
            highest = highest.max(height);
        }
    }
    assert!(landed, "the ball never landed");
    assert!(highest < 0.005, "the ball bounced back to {highest:.4} m");
}

#[test]
fn dropped_box_lands_without_sinking_or_bouncing() {
    // Box2D v3.2: never below y = 0.49991, rests there, rises at most 0.075 m/s after landing.
    let mut world = PhysicsWorld2d::new();
    add_floor(&mut world, 0.3);
    let id = add_box(&mut world, Vec2::new(0.0, 2.5), 0.0, 0.3);
    let (mut lowest, mut rise, mut landed) = (f32::INFINITY, 0.0f32, false);
    for _ in 0..120 {
        world.step(DT);
        let body = world.body(id).unwrap();
        lowest = lowest.min(body.position.y);
        landed = landed || body.position.y < 0.51;
        if landed {
            rise = rise.max(body.linear_velocity.y);
        }
    }
    let rest = world.body(id).unwrap().position.y;
    assert!(lowest > 0.495, "the box sank to y = {lowest:.4}");
    assert!(
        (rest - 0.4999).abs() < 0.002,
        "the box rests at y = {rest:.4}"
    );
    assert!(rise < 0.15, "the box bounced up at {rise:.4} m/s");
}

#[test]
fn ten_box_stack_comes_to_rest_and_sleeps() {
    // Box2D v3.2: asleep at step 43, columns within 6 mm of x = 0, the top sinks 3.1 cm.
    let mut world = PhysicsWorld2d::new();
    add_floor(&mut world, 0.6);
    let boxes: Vec<BodyId2d> = (0..10)
        .map(|i| add_box(&mut world, Vec2::new(0.0, 0.5 + i as f32), 0.0, 0.6))
        .collect();
    let mut asleep_at = None;
    for step in 1..=300 {
        world.step(DT);
        if asleep_at.is_none() && world.body(boxes[9]).unwrap().is_sleeping {
            asleep_at = Some(step);
        }
    }
    let asleep_at = asleep_at.expect("the stack never fell asleep");
    assert!(
        asleep_at <= 90,
        "the stack fell asleep only at step {asleep_at}"
    );
    for (i, &id) in boxes.iter().enumerate() {
        let p = world.body(id).unwrap().position;
        assert!(p.x.abs() < 0.02, "box {i} drifted to x = {:.4}", p.x);
        let sink = 0.5 + i as f32 - p.y;
        assert!(sink.abs() < 0.05, "box {i} sank {sink:.4} m");
    }
}

#[test]
fn pyramid_comes_to_rest_and_sleeps() {
    // 55 boxes in ten rows. Box2D v3.2: asleep at step 41, top at (0.0008, 9.4925).
    let mut world = PhysicsWorld2d::new();
    add_floor(&mut world, 0.6);
    let mut top = None;
    for row in 0..10 {
        let n = 10 - row;
        for i in 0..n {
            let x = (i as f32 - 0.5 * (n - 1) as f32) * 1.05;
            top = Some(add_box(
                &mut world,
                Vec2::new(x, 0.5 + row as f32),
                0.0,
                0.6,
            ));
        }
    }
    let top = top.expect("the pyramid has boxes");
    let mut asleep_at = None;
    for step in 1..=600 {
        world.step(DT);
        if asleep_at.is_none() && world.body(top).unwrap().is_sleeping {
            asleep_at = Some(step);
        }
    }
    let asleep_at = asleep_at.expect("the pyramid never fell asleep");
    assert!(
        asleep_at <= 90,
        "the pyramid fell asleep only at step {asleep_at}"
    );
    let p = world.body(top).unwrap().position;
    assert!(p.x.abs() < 0.01, "the top box drifted to x = {:.4}", p.x);
    assert!(p.y > 9.45, "the pyramid sank: top at y = {:.4}", p.y);
}

fn slope_slide(friction: f32) -> f32 {
    let mut world = PhysicsWorld2d::new();
    let angle = 20.0f32.to_radians();
    let slope = world
        .create_body(RigidBody2d::new(BodyType::Static).with_rotation(Rot2::from_angle(angle)));
    world.create_collider(
        slope,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(10.0, 0.5))).with_friction(friction),
    );
    let start = Rot2::from_angle(angle).rotate(Vec2::new(0.0, 1.0));
    let id = add_box(&mut world, start, angle, friction);
    for _ in 0..60 {
        world.step(DT);
    }
    (world.body(id).unwrap().position - start).length()
}

#[test]
fn friction_holds_or_lets_go_on_a_slope() {
    // A 20 degree slope. Box2D v3.2: mu = 0.6 holds (0.16 mm in 1 s); mu = 0.2 slides 0.767 m,
    // close to g (sin - mu cos) t^2 / 2 = 0.755 m.
    let held = slope_slide(0.6);
    assert!(held < 0.005, "mu = 0.6 slid {held:.4} m");
    let slid = slope_slide(0.2);
    assert!((slid - 0.767).abs() < 0.05, "mu = 0.2 slid {slid:.4} m");
}

#[test]
fn resting_box_barely_overlaps_the_floor() {
    // Box2D v3.2 over the fifth second: deepest separation -0.00009 m, still, at y = 0.49991.
    let mut world = PhysicsWorld2d::new();
    add_floor(&mut world, 0.3);
    let mut body = RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 0.5));
    body.enable_sleep = false;
    let id = world.create_body(body);
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    let (mut deepest, mut fastest) = (0.0f32, 0.0f32);
    for step in 0..300 {
        world.step(DT);
        if step >= 240 {
            for (_, _, manifold) in world.manifolds() {
                for point in &manifold.points[..manifold.point_count] {
                    deepest = deepest.min(point.separation);
                }
            }
            fastest = fastest.max(world.body(id).unwrap().linear_velocity.length());
        }
    }
    assert!(
        deepest > -0.005,
        "the box overlaps the floor by {:.5} m",
        -deepest
    );
    assert!(fastest < 0.001, "the resting box moves at {fastest:.5} m/s");
    let y = world.body(id).unwrap().position.y;
    assert!((y - 0.4999).abs() < 0.002, "the box rests at y = {y:.5}");
}

/// A box with restitution 0.8, tilted 0.3 rad and dropped from 2 m onto a floor with friction
/// 0.6, after 1 s with `iterations` restitution iterations: its height and spin, as bits.
fn tilted_bounce(iterations: usize) -> [u32; 2] {
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(korppi_physics2d::world::WorldConfig2d {
            enable_sleep: false,
            restitution_iterations: iterations,
            ..Default::default()
        });
    add_floor(&mut world, 0.6);
    let id = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 2.0))
            .with_rotation(Rot2::from_angle(0.3)),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
            .with_friction(0.6)
            .with_restitution(0.8),
    );
    for _ in 0..60 {
        world.step(DT);
    }
    let body = world.body(id).unwrap();
    [body.position.y.to_bits(), body.angular_velocity.to_bits()]
}

#[test]
fn restitution_iterations_are_a_world_setting() {
    // Box2D v3.2's `restitutionIterations` (`fix_2_review.c`): without iterations the box does
    // not bounce; one bounces it; two (the default) and more give the same, to the bit.
    assert_eq!(tilted_bounce(0), [0x3f00_bd2e, 0x3cde_ff70], "0 iterations");
    assert_eq!(tilted_bounce(1), [0x3f5c_b88e, 0xbf11_19ac], "1 iteration");
    assert_eq!(tilted_bounce(2), [0x3f5c_b888, 0xbf11_1950], "2 iterations");
    assert_eq!(tilted_bounce(8), [0x3f5c_b888, 0xbf11_1950], "8 iterations");
}

/// A ball with restitution 0.9 dropped from 3 m onto two stacked boxes without restitution,
/// after 1 s: the ball's height and vertical speed and the top box's, as bits.
fn ball_on_stack(propagate: bool) -> [u32; 3] {
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(korppi_physics2d::world::WorldConfig2d {
            enable_sleep: false,
            enable_restitution_propagation: propagate,
            ..Default::default()
        });
    add_floor(&mut world, 0.6);
    let mut top = None;
    for i in 0..2 {
        top = Some(add_box(
            &mut world,
            Vec2::new(0.0, 0.5 + i as f32),
            0.0,
            0.6,
        ));
    }
    let ball =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.1, 3.0)));
    world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25)))
            .with_friction(0.6)
            .with_restitution(0.9),
    );
    for _ in 0..60 {
        world.step(DT);
    }
    let (ball, top) = (world.body(ball).unwrap(), world.body(top.unwrap()).unwrap());
    [
        ball.position.y.to_bits(),
        ball.linear_velocity.y.to_bits(),
        top.linear_velocity.y.to_bits(),
    ]
}

#[test]
fn restitution_propagation_lets_the_bounce_pass_through_resting_contacts() {
    // Box2D v3.2's `enableRestitutionPropagation` (`fix_2_review.c`): off by default, the
    // restitution pass skips contacts without restitution; on, it solves them too.
    assert_eq!(
        ball_on_stack(false),
        [0x400f_db3f, 0xc03e_db7f, 0xb882_9900]
    );
    assert_eq!(ball_on_stack(true), [0x4015_15a3, 0xc036_24d5, 0xb893_d100]);
}
