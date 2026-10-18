//! The world steps in parallel, with results independent of the worker count (P2D-17).
//!
//! As Box2D v3.2, a world given a task executor spreads its step over the executor's workers:
//! the narrowphase runs as a parallel loop, and the solver runs its stages as blocks that
//! workers claim, graph color by graph color, with the colored contacts solved four at a time
//! by the SIMD contact solver. Constraints of one color share no dynamic
//! body, and everything workers produce is merged in a fixed order, so a world steps bit for bit
//! the same on any number of workers, and as Box2D. Before FIX-2.7 the colors were solved one
//! after the other on the calling thread. Box2D's pyramid value is from
//! `docs/audits/probes/box2d_reference/fix_2_7_parallel.c`.

use korppi_core::{TaskExecutor, WorkerPool};
use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::chain::ChainDef2d;
use korppi_physics2d::collider::{Collider2d, Filter2d};
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d, WeldJoint2d};
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, WorldConfig2d};
use std::sync::{Arc, Mutex, PoisonError};

const DT: f32 = 1.0 / 60.0;
const WORKER_COUNTS: [usize; 5] = [1, 2, 3, 4, 8];

/// Worker-pool tests take turns so multi-worker timing assertions do not contend for cores with
/// concurrent 8-worker determinism runs.
static SERIAL: Mutex<()> = Mutex::new(());

fn add_box(world: &mut PhysicsWorld2d, x: f32, y: f32) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, y)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))).with_friction(0.6),
    );
    id
}

/// A ten-row pyramid of unit boxes on a ground box, sleep off, as the probe builds it.
fn pyramid() -> (PhysicsWorld2d, BodyId2d) {
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(WorldConfig2d {
            enable_sleep: false,
            ..WorldConfig2d::default()
        });
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(0.6),
    );
    let mut top = None;
    for row in 0..10 {
        for i in 0..10 - row {
            top = Some(add_box(
                &mut world,
                -4.5 + 0.5 * row as f32 + i as f32,
                0.5 + row as f32,
            ));
        }
    }
    (world, top.unwrap())
}

/// A busy scene with sleep on: stacks and a pyramid, balls and capsules raining down, fast
/// bullets, hinged and welded chains with a motor, a kinematic paddle, sensors and a chain
/// ground.
fn busy() -> PhysicsWorld2d {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -10.0));
    let ground = world.create_body(RigidBody2d::new(BodyType::Static));
    let points = (0..41).map(|i| Vec2::new(20.0 - i as f32, 0.0)).collect();
    world.create_chain(
        ground,
        &ChainDef2d::open(points, Vec2::new(21.0, 0.0), Vec2::new(-21.0, 0.0)),
    );
    for column in 0..4 {
        for row in 0..8 {
            add_box(&mut world, -15.0 + 3.0 * column as f32, 0.5 + row as f32);
        }
    }
    for row in 0..6 {
        for i in 0..6 - row {
            add_box(
                &mut world,
                2.0 + 0.5 * row as f32 + i as f32,
                0.5 + row as f32,
            );
        }
    }
    for i in 0..30 {
        let body = world.create_body(
            RigidBody2d::new(BodyType::Dynamic)
                .with_position(Vec2::new(-10.0 + 0.7 * i as f32, 12.0 + (i % 5) as f32))
                .with_angular_velocity(0.3 * (i % 7) as f32 - 1.0)
                .with_bullet(i % 10 == 0)
                .with_linear_velocity(Vec2::new(0.0, if i % 10 == 0 { -60.0 } else { 0.0 })),
        );
        let shape = if i % 2 == 0 {
            Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.3))
        } else {
            Shape2d::Capsule(Capsule2d::new(
                Vec2::new(-0.3, 0.0),
                Vec2::new(0.3, 0.0),
                0.2,
            ))
        };
        world.create_collider(body, Collider2d::new(shape).with_restitution(0.3));
    }
    let anchor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(12.0, 10.0)));
    let mut previous = anchor;
    for i in 0..12 {
        let link = world.create_body(
            RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(12.5 + i as f32, 10.0)),
        );
        world.create_collider(
            link,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.1))).with_filter(
                Filter2d {
                    group_index: -1,
                    ..Filter2d::DEFAULT
                },
            ),
        );
        let anchor_a = if i == 0 {
            Vec2::ZERO
        } else {
            Vec2::new(0.5, 0.0)
        };
        let joint = if i % 3 == 2 {
            Joint2d::Weld(WeldJoint2d::new(anchor_a, Vec2::new(-0.5, 0.0), 0.0, 0.0))
        } else {
            let hinge = RevoluteJoint2d::new(anchor_a, Vec2::new(-0.5, 0.0), 0.0, 0.0);
            Joint2d::Revolute(if i == 0 {
                hinge.with_motor(1.0, 50.0)
            } else {
                hinge
            })
        };
        world.create_joint(previous, link, joint);
        previous = link;
    }
    let paddle = world.create_body(
        RigidBody2d::new(BodyType::Kinematic)
            .with_position(Vec2::new(-5.0, 4.0))
            .with_angular_velocity(1.5),
    );
    world.create_collider(
        paddle,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(2.0, 0.1))),
    );
    let sensor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 3.0)));
    world.create_collider(
        sensor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(4.0, 1.0)))
            .with_sensor(true)
            .with_sensor_events(true),
    );
    world
}

fn executor(workers: usize) -> Arc<dyn TaskExecutor> {
    Arc::new(WorkerPool::new(workers))
}

#[test]
fn a_pyramid_matches_box2d_bit_for_bit_on_any_number_of_workers() {
    let _turn = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    // Box2D v3.2 (SSE2) after 2 s: the top box at x = -0.00127752, y = 9.49234581, exactly.
    for workers in WORKER_COUNTS {
        let (mut world, top) = pyramid();
        world.set_executor(Some(executor(workers)));
        for _ in 0..120 {
            world.step(DT);
        }
        let body = world.body(top).unwrap();
        let bits = [
            body.position.x,
            body.position.y,
            body.rotation.c,
            body.rotation.s,
        ]
        .map(f32::to_bits);
        assert_eq!(
            bits,
            [0xbaa7_727b, 0x4117_e0a6, 0x3f80_0000, 0xb8f1_9b42],
            "{workers} workers"
        );
    }
}

#[test]
fn a_busy_world_steps_the_same_on_any_number_of_workers() {
    let _turn = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    // The serial world's state hash after every step, then the same on each worker count.
    let mut serial = busy();
    let hashes: Vec<u64> = (0..180)
        .map(|_| {
            serial.step(DT);
            serial.state_hash()
        })
        .collect();
    assert!(
        serial.counters().awake_body_count < serial.counters().body_count,
        "some bodies fell asleep, so sleep is exercised"
    );
    for workers in WORKER_COUNTS {
        let mut world = busy();
        world.set_executor(Some(executor(workers)));
        for (step, &hash) in hashes.iter().enumerate() {
            world.step(DT);
            assert_eq!(world.state_hash(), hash, "{workers} workers, step {step}");
        }
    }
}

#[test]
fn an_executor_is_host_wiring() {
    let _turn = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    // Like the callbacks, the executor survives an in-place restore and is not in snapshots.
    let (mut world, top) = pyramid();
    world.set_executor(Some(executor(4)));
    world.step(DT);
    let snapshot = world.create_snapshot();
    world.restore_snapshot(&snapshot);
    assert_eq!(world.executor().map(|e| e.worker_count()), Some(4));
    assert!(PhysicsWorld2d::from_snapshot(&snapshot)
        .executor()
        .is_none());
    world.set_executor(None);
    world.step(DT);
    assert!(world.body(top).is_some());
}

fn reaudit_circles(n: usize) -> PhysicsWorld2d {
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(WorldConfig2d {
            enable_sleep: false,
            ..WorldConfig2d::default()
        });
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(5000.0, 0.5))).with_friction(0.6),
    );
    for i in 0..n {
        let body = world.create_body(
            RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(i as f32 * 1.5, 0.5)),
        );
        world.create_collider(
            body,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))).with_friction(0.6),
        );
    }
    world
}

#[test]
fn four_workers_speed_up_the_reaudit_circle_scene_without_changing_its_hash() {
    let _turn = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    // P2D-28 (Probe P12, 8,000 and 16,000 circles after 120 warmup steps):
    // 4 workers must produce the exact same state hash as 1 worker and speed up both scenes by
    // at least 1.35x (before FIX-8.5, 4 workers achieved only 1.04x on 8,000 and 0.91x on 16,000).
    let warmup = 120;
    let measured = 60;

    let run_scene = |n: usize, workers: usize| -> (u64, f64) {
        let mut world = reaudit_circles(n);
        world.set_executor(Some(executor(workers)));
        for _ in 0..warmup {
            world.step(DT);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..measured {
            world.step(DT);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / measured as f64;
        (world.state_hash(), ms)
    };

    for n in [8000, 16000] {
        let (hash_1, ms_1) = run_scene(n, 1);
        let (hash_4, ms_4) = run_scene(n, 4);
        assert_eq!(
            hash_4, hash_1,
            "4 workers must produce the exact same state hash as 1 worker on P12 (N={n})"
        );
        let speedup = ms_1 / ms_4;
        assert!(
            speedup >= 1.35,
            "4 workers must speed up P12 (N={n} circles) by at least 1.35x (1 worker: {ms_1:.3} ms/step, 4 workers: {ms_4:.3} ms/step, speedup: {speedup:.2}x)"
        );
    }
}
