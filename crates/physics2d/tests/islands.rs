//! Islands persist and bodies sleep like Box2D v3.2's (P2D-16).
//!
//! Before FIX-2.5 the world rebuilt every island by breadth-first search each step, slept a body
//! when its linear and angular speeds were both under 0.05, woke a body one step after something
//! started touching it, left kinematic bodies out of islands and visited sleeping bodies in every
//! loop. As in Box2D, islands persist: touching and joints merge them, and an island that lost
//! constraints is split only when one of its bodies would sleep, one island per step. A body
//! rests while the speed of its fastest point stays under its own threshold; an island sleeps
//! when all its bodies rested for [`TIME_TO_SLEEP`]. Sleeping islands live in solver sets the
//! step does not visit, and a contact that starts touching wakes them before the solve.
//! Expected values are Box2D's, from `docs/audits/probes/box2d_reference/fix_2_5_islands.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn world(gravity: f32) -> PhysicsWorld2d {
    PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, gravity))
}

fn add_box(
    world: &mut PhysicsWorld2d,
    kind: BodyType,
    x: f32,
    y: f32,
    hx: f32,
    hy: f32,
) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(kind).with_position(Vec2::new(x, y)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))).with_friction(0.6),
    );
    id
}

fn awake(world: &PhysicsWorld2d, id: BodyId2d) -> bool {
    !world.body(id).unwrap().is_sleeping
}

/// The steps at which each body's awake state changed, as (body index, awake, step).
fn track(
    world: &mut PhysicsWorld2d,
    bodies: &[BodyId2d],
    steps: std::ops::RangeInclusive<u32>,
    mut before: impl FnMut(&mut PhysicsWorld2d, u32),
) -> Vec<(usize, bool, u32)> {
    let mut last: Vec<bool> = bodies.iter().map(|&b| awake(world, b)).collect();
    let mut changes = Vec::new();
    for step in steps {
        before(world, step);
        world.step(DT);
        for (k, &body) in bodies.iter().enumerate() {
            if awake(world, body) != last[k] {
                last[k] = !last[k];
                changes.push((k, last[k], step));
            }
        }
    }
    changes
}

#[test]
fn islands_merge_when_touching_or_jointed_and_persist() {
    // Three stacks of three boxes and a two-link pendulum. Each body starts in its own island;
    // the joint between the links merges theirs (the one to the static pivot does not); touching
    // merges each stack. The stacks sleep at step 30; the awake contacts are then none. At step
    // 100 the middle box of the first stack is destroyed, waking the other two, which stay in
    // one island while the top one falls; they sleep again by step 200.
    let mut world = world(-10.0);
    add_box(&mut world, BodyType::Static, 0.0, -0.5, 20.0, 0.5);
    let stacks: Vec<[BodyId2d; 3]> = (0..3)
        .map(|s| {
            [0, 1, 2].map(|i| {
                add_box(
                    &mut world,
                    BodyType::Dynamic,
                    -5.0 + 5.0 * s as f32,
                    0.5 + i as f32,
                    0.5,
                    0.5,
                )
            })
        })
        .collect();
    let pivot =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(10.0, 8.0)));
    let link1 = add_box(&mut world, BodyType::Dynamic, 11.0, 8.0, 0.5, 0.1);
    let link2 = add_box(&mut world, BodyType::Dynamic, 12.0, 8.0, 0.5, 0.1);
    let hinge = |a: Vec2, b: Vec2| Joint2d::Revolute(RevoluteJoint2d::new(a, b, 0.0, 0.0));
    world.create_joint(pivot, link1, hinge(Vec2::ZERO, Vec2::new(-1.0, 0.0)));
    assert_eq!(world.counters().island_count, 11);
    world.create_joint(
        link1,
        link2,
        hinge(Vec2::new(0.5, 0.0), Vec2::new(-0.5, 0.0)),
    );
    assert_eq!(
        (
            world.counters().island_count,
            world.counters().awake_body_count
        ),
        (10, 11)
    );

    // Box2D: (step, islands, awake bodies, awake contacts, first stack awake, top box y).
    let box2d = [
        (1, 4, 11, 9, true, 2.4992),
        (2, 4, 11, 9, true, 2.4992),
        (10, 4, 11, 9, true, 2.4984),
        (30, 4, 2, 0, false, 2.4978),
        (31, 4, 2, 0, false, 2.4978),
        (40, 4, 2, 0, false, 2.4978),
        (99, 4, 2, 0, false, 2.4978),
        (100, 4, 4, 1, true, 2.4960),
        (101, 4, 4, 1, true, 2.4915),
        (110, 4, 4, 1, true, 2.3259),
        (140, 4, 4, 2, true, 1.4996),
        (170, 4, 4, 2, true, 1.4991),
        (200, 4, 2, 0, false, 1.4991),
        (300, 4, 2, 0, false, 1.4991),
    ];
    let mut sample = 0;
    for step in 1..=300 {
        if step == 100 {
            world.destroy_body(stacks[0][1]);
        }
        world.step(DT);
        if step == box2d[sample].0 {
            let (_, islands, bodies, contacts, stack_awake, top_y) = box2d[sample];
            let c = world.counters();
            assert_eq!(
                (c.island_count, c.awake_body_count, c.awake_contact_count),
                (islands, bodies, contacts),
                "step {step}: (islands, awake bodies, awake contacts)"
            );
            assert_eq!(
                (awake(&world, stacks[0][0]), awake(&world, stacks[0][2])),
                (stack_awake, stack_awake),
                "step {step}: first stack awake"
            );
            let y = world.body(stacks[0][2]).unwrap().position.y;
            assert!(
                (y - top_y).abs() < 2e-3,
                "step {step}: top box y {y}, Box2D {top_y}"
            );
            sample += 1;
            if sample == box2d.len() {
                break;
            }
        }
    }
}

#[test]
fn a_body_rests_while_its_fastest_point_is_slow() {
    // No gravity. A plank and a square spin at 0.03 rad/s: the plank's tip moves at 0.06 m/s,
    // above the 0.05 m/s threshold, so it never sleeps; the square's corner moves at 0.021 m/s.
    // A box drifting at 0.3 m/s with a threshold of 0.5 m/s, and one creeping at 0.04 m/s, rest
    // too. Box2D v3.2 puts the three to sleep at step 30, where they stop.
    let mut world = world(0.0);
    let plank = add_box(&mut world, BodyType::Dynamic, 0.0, 0.0, 2.0, 0.1);
    let square = add_box(&mut world, BodyType::Dynamic, 0.0, 5.0, 0.5, 0.5);
    let drifter = add_box(&mut world, BodyType::Dynamic, 0.0, 10.0, 0.5, 0.5);
    let creeper = add_box(&mut world, BodyType::Dynamic, 0.0, 15.0, 0.5, 0.5);
    world.body_mut(plank).unwrap().angular_velocity = 0.03;
    world.body_mut(square).unwrap().angular_velocity = 0.03;
    world.body_mut(drifter).unwrap().linear_velocity = Vec2::new(0.3, 0.0);
    assert!(world.set_sleep_threshold(drifter, 0.5));
    world.body_mut(creeper).unwrap().linear_velocity = Vec2::new(0.04, 0.0);

    let changes = track(
        &mut world,
        &[plank, square, drifter, creeper],
        1..=300,
        |_, _| {},
    );
    assert_eq!(
        changes,
        vec![(1, false, 30), (2, false, 30), (3, false, 30)]
    );
    let plank_angle = world.body(plank).unwrap().rotation.angle();
    assert!(
        (plank_angle - 0.15).abs() < 1e-4,
        "plank angle {plank_angle}"
    );
    let drifter_body = world.body(drifter).unwrap();
    assert!((drifter_body.position.x - 0.15).abs() < 1e-4);
    assert_eq!(drifter_body.linear_velocity, Vec2::ZERO);
    assert!((world.body(creeper).unwrap().position.x - 0.02).abs() < 1e-4);
}

#[test]
fn a_touch_wakes_a_sleeping_body_before_the_solve() {
    // A box falls asleep on the ground at step 30. A ball dropped at step 60 starts touching it
    // at step 83; Box2D v3.2 wakes the box in that step, before the solve, so the box is pushed
    // in the same step (the push itself depends on the solver order, see `graph_coloring.rs`).
    let mut world = world(-10.0);
    add_box(&mut world, BodyType::Static, 0.0, -0.5, 20.0, 0.5);
    let target = add_box(&mut world, BodyType::Dynamic, 0.0, 0.5, 0.5, 0.5);
    let mut touch_step = None;
    let mut pushed = false;
    let changes = track(&mut world, &[target], 1..=100, |world, step| {
        if step == 60 {
            let id = world.create_body(
                RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 2.0)),
            );
            world.create_collider(
                id,
                Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25)))
                    .with_friction(0.6)
                    .with_contact_events(true),
            );
        }
        // The events and velocity of the step just taken.
        if step > 60 && touch_step.is_none() && !world.contact_events().begin.is_empty() {
            touch_step = Some(step - 1);
            pushed = world.body(target).unwrap().linear_velocity.y < 0.0;
        }
    });
    assert_eq!(touch_step, Some(83));
    assert_eq!(changes, vec![(0, false, 30), (0, true, 83)]);
    assert!(
        pushed,
        "the box must take part in the solve of the step it wakes in"
    );
}

#[test]
fn kinematic_bodies_belong_to_islands() {
    // A box rides a kinematic platform moving at 0.5 m/s; their island stays awake until the
    // platform stops at step 120, then sleeps at step 153 with them both. A lone kinematic body
    // creeping at 0.01 m/s, under the threshold, sleeps at step 30 and stops there. Box2D v3.2.
    let mut world = world(-10.0);
    let platform = add_box(&mut world, BodyType::Kinematic, 0.0, 0.0, 2.0, 0.1);
    world.body_mut(platform).unwrap().linear_velocity = Vec2::new(0.5, 0.0);
    let rider = add_box(&mut world, BodyType::Dynamic, 0.0, 0.6, 0.5, 0.5);
    let creeper = add_box(&mut world, BodyType::Kinematic, 0.0, 10.0, 0.5, 0.5);
    world.body_mut(creeper).unwrap().linear_velocity = Vec2::new(0.01, 0.0);

    let mut islands = Vec::new();
    let changes = track(
        &mut world,
        &[platform, rider, creeper],
        1..=240,
        |world, step| {
            if step == 120 {
                world.body_mut(platform).unwrap().linear_velocity = Vec2::ZERO;
            }
            if step == 3 || step == 120 {
                islands.push(world.counters().island_count);
            }
        },
    );
    assert_eq!(islands, vec![2, 2]);
    assert_eq!(
        changes,
        vec![(2, false, 30), (0, false, 153), (1, false, 153)]
    );
    let x = |id| world.body(id).unwrap().position.x;
    assert!((x(platform) - 0.991_67).abs() < 1e-4);
    assert!((x(rider) - 0.991_66).abs() < 1e-3);
    assert!((x(creeper) - 0.005).abs() < 1e-5);
}

#[test]
fn an_island_splits_when_its_sleepiest_body_would_sleep() {
    // Two boxes side by side share an island and sleep at step 30. At step 60 the right one is
    // sent sliding away: both wake. Their island splits only once the left box has rested
    // 0.5 s, at step 90, a step later than it could have slept; the right one sleeps at step
    // 118. Box2D v3.2.
    let mut world = world(-10.0);
    add_box(&mut world, BodyType::Static, 0.0, -0.5, 20.0, 0.5);
    let left = add_box(&mut world, BodyType::Dynamic, -0.5, 0.5, 0.5, 0.5);
    let right = add_box(&mut world, BodyType::Dynamic, 0.5, 0.5, 0.5, 0.5);
    let mut islands = Vec::new();
    let changes = track(&mut world, &[left, right], 1..=240, |world, step| {
        if step == 60 {
            world.body_mut(right).unwrap().linear_velocity = Vec2::new(3.0, 0.0);
            assert!(awake(world, left) && awake(world, right));
        }
        if matches!(step, 2 | 90 | 91) {
            islands.push(world.counters().island_count);
        }
    });
    assert_eq!(islands, vec![1, 1, 2]);
    assert_eq!(
        changes,
        vec![
            (0, false, 30),
            (1, false, 30),
            (0, true, 60),
            (1, true, 60),
            (0, false, 90),
            (1, false, 118)
        ]
    );
    assert!((world.body(left).unwrap().position.x + 0.499_99).abs() < 1e-3);
    assert!((world.body(right).unwrap().position.x - 1.2436).abs() < 1e-2);
}

#[test]
fn sleep_can_be_requested_and_woken_through_the_api() {
    // Box2D v3.2: set_awake(false) puts a falling box to sleep in the air at once, at rest;
    // set_awake(true) wakes it, still at rest. Waking the bodies touching the ground wakes a
    // sleeping stack, and a body that may not sleep keeps its island awake.
    let mut world = world(-10.0);
    let ground = add_box(&mut world, BodyType::Static, 0.0, -0.5, 20.0, 0.5);
    let faller = add_box(&mut world, BodyType::Dynamic, -5.0, 5.0, 0.5, 0.5);
    let bottom = add_box(&mut world, BodyType::Dynamic, 5.0, 0.5, 0.5, 0.5);
    let top = add_box(&mut world, BodyType::Dynamic, 5.0, 1.5, 0.5, 0.5);
    for _ in 0..10 {
        world.step(DT);
    }
    assert!(world.set_awake(faller, false));
    let body = world.body(faller).unwrap();
    assert!(body.is_sleeping);
    assert!((body.position.y - 4.857_64).abs() < 1e-4);
    assert_eq!(body.linear_velocity, Vec2::ZERO);
    for _ in 0..10 {
        world.step(DT);
    }
    assert!((world.body(faller).unwrap().position.y - 4.857_64).abs() < 1e-4);
    world.set_awake(faller, true);
    assert!(awake(&world, faller));
    assert_eq!(world.body(faller).unwrap().linear_velocity, Vec2::ZERO);
    world.step(DT);
    let body = world.body(faller).unwrap();
    assert!((body.position.y - 4.855_90).abs() < 1e-4);
    assert!((body.linear_velocity.y + 0.166_67).abs() < 1e-4);

    for _ in 0..100 {
        world.step(DT);
    }
    assert!(!awake(&world, bottom) && !awake(&world, top));
    world.wake_touching(ground);
    assert!(awake(&world, bottom) && awake(&world, top) && awake(&world, faller));
    world.body_mut(top).unwrap().enable_sleep = false;
    for _ in 0..100 {
        world.step(DT);
    }
    assert!(awake(&world, bottom) && awake(&world, top));
}
