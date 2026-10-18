//! Steady-state steps of the 2D world do not allocate (P2D-14, rule 4).
//!
//! Before FIX-2.5 the broadphase allocated a stack on every tree query, and contacts lived in a
//! `BTreeMap` that allocated a node whenever two colliders started touching. After a warm-up that
//! lets every buffer reach the size the scene needs, stepping must not touch the heap, even while
//! contacts begin and end, fast bodies are swept, sensors gain and lose visitors and every event
//! kind is reported. Box2D gets the same from fixed-size polygons, stack-allocated tree stacks and
//! pooled contact arrays.

use korppi_core::{TaskExecutor, WorkerPool};
use korppi_math::{Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, JointInstance2d, PhysicsWorld2d};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

struct CountingAllocator;

thread_local! {
    /// Whether this thread's allocations count.
    static TRACKING: Cell<bool> = const { Cell::new(false) };
}

/// Allocations of the tracking threads.
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Tests that count allocations take turns, since the count is shared.
static COUNTING: Mutex<()> = Mutex::new(());

fn count() {
    let _ = TRACKING.try_with(|tracking| {
        if tracking.get() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    });
}

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract. Counting only touches a const-initialized `Cell` thread-local and an
// atomic, which never allocate, so it cannot recurse into the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged; the caller upholds `dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocations made while running `f`, on this thread and on the threads that opted in with
/// [`track_pool`].
fn allocations_during(f: impl FnOnce()) -> usize {
    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    TRACKING.with(|t| t.set(true));
    f();
    TRACKING.with(|t| t.set(false));
    ALLOC_COUNT.load(Ordering::Relaxed)
}

/// Makes every thread of `pool` count its allocations from now on: one task per worker, each
/// waiting until all have started, puts a task on every thread.
fn track_pool(pool: &WorkerPool) {
    let workers = pool.worker_count();
    let started = AtomicUsize::new(0);
    pool.run(workers, &|_, _| {
        TRACKING.with(|t| t.set(true));
        started.fetch_add(1, Ordering::AcqRel);
        while started.load(Ordering::Acquire) < workers {
            std::hint::spin_loop();
        }
    });
    // The calling thread ran one of the tasks; it counts only inside `allocations_during`.
    TRACKING.with(|t| t.set(false));
}

fn add_static_box(world: &mut PhysicsWorld2d, position: Vec2, hx: f32, hy: f32) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(BodyType::Static).with_position(position));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy)))
            .with_contact_events(true)
            .with_sensor_events(true),
    );
    id
}

/// A scene that keeps changing: contacts begin and end, a bullet is swept every step, a sensor
/// gains and loses a visitor, a joint reports force events, and every body stays awake.
fn busy_world() -> PhysicsWorld2d {
    let mut world = PhysicsWorld2d::new();
    add_static_box(&mut world, Vec2::new(0.0, -0.5), 30.0, 0.5);

    // An elastic ball bouncing forever: its contact begins and ends again and again.
    let ball =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(-8.0, 3.0)));
    world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.4)))
            .with_restitution(1.0)
            .with_contact_events(true)
            .with_hit_events(true),
    );

    // A stack, and a capsule resting on a box.
    for i in 0..4 {
        let id = world.create_body(
            RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(-3.0, 0.5 + i as f32)),
        );
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
        );
    }
    let rod =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(-3.0, 4.75)));
    world.create_collider(
        rod,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(-0.4, 0.0),
            Vec2::new(0.4, 0.0),
            0.25,
        ))),
    );

    // A pendulum swinging through a sensor, with a joint that reports force events.
    let pivot =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(4.0, 6.0)));
    let bob =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(7.0, 6.0)));
    world.create_collider(
        bob,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.3))).with_sensor_events(true),
    );
    world.add_joint(
        JointInstance2d::new(
            pivot,
            bob,
            Joint2d::Revolute(RevoluteJoint2d::new(
                Vec2::ZERO,
                Vec2::new(-3.0, 0.0),
                0.0,
                0.0,
            )),
        )
        .with_event_thresholds(1.0, f32::MAX),
    );
    let sensor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(4.0, 3.0)));
    world.create_collider(
        sensor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)))
            .with_sensor(true)
            .with_sensor_events(true),
    );

    // A bullet bouncing between two thin walls, swept every step.
    add_static_box(&mut world, Vec2::new(12.0, 5.0), 0.05, 5.0);
    add_static_box(&mut world, Vec2::new(20.0, 5.0), 0.05, 5.0);
    let mut bullet = RigidBody2d::new(BodyType::Dynamic)
        .with_position(Vec2::new(16.0, 5.0))
        .with_linear_velocity(Vec2::new(150.0, 0.0))
        .with_rotation(Rot2::IDENTITY)
        .with_bullet(true);
    bullet.gravity_scale = 0.0;
    let bullet = world.create_body(bullet);
    world.create_collider(
        bullet,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.1))).with_restitution(1.0),
    );
    world
}

#[test]
fn steady_state_steps_do_not_allocate() {
    let mut world = busy_world();
    let dt = 1.0 / 60.0;
    // Warm up for 10 s so every buffer reaches the size the scene needs.
    for _ in 0..600 {
        world.step(dt);
    }
    let allocations = allocations_during(|| {
        for _ in 0..600 {
            world.step(dt);
        }
    });
    assert_eq!(
        allocations, 0,
        "600 steady-state steps allocated {allocations} times"
    );
}

#[test]
fn parallel_steps_do_not_allocate_on_any_thread() {
    // The same scene on four workers (FIX-2.7): the solver's stages, blocks and wide
    // constraints live in buffers the world keeps, and the workers allocate nothing either.
    let pool = Arc::new(WorkerPool::new(4));
    track_pool(&pool);
    // The count sees the workers' allocations.
    let started = AtomicUsize::new(0);
    let seen = allocations_during(|| {
        pool.run(4, &|_, _| {
            started.fetch_add(1, Ordering::AcqRel);
            while started.load(Ordering::Acquire) < 4 {
                std::hint::spin_loop();
            }
            std::hint::black_box(vec![0u8; 64]);
        });
    });
    assert_eq!(seen, 4, "the four workers' allocations");
    let mut world = busy_world();
    world.set_executor(Some(pool));
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step(dt);
    }
    let allocations = allocations_during(|| {
        for _ in 0..600 {
            world.step(dt);
        }
    });
    assert_eq!(
        allocations, 0,
        "600 parallel steps allocated {allocations} times"
    );
}

#[test]
fn the_scene_keeps_changing() {
    // Guard the scene itself: contacts must keep beginning and ending, the sensor must keep
    // gaining and losing its visitor, and the bullet must keep bouncing.
    let mut world = busy_world();
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step(dt);
    }
    let (mut begins, mut ends, mut sensor_events, mut hits, mut joint_events) = (0, 0, 0, 0, 0);
    for _ in 0..600 {
        world.step(dt);
        let events = world.contact_events();
        begins += events.begin.len();
        ends += events.end.len();
        hits += events.hit.len();
        sensor_events += world.sensor_events().begin.len() + world.sensor_events().end.len();
        joint_events += world.joint_events().len();
    }
    assert!(
        begins > 5 && ends > 5,
        "contacts {begins} begun, {ends} ended"
    );
    assert!(hits > 5, "{hits} hits");
    assert!(sensor_events > 2, "{sensor_events} sensor events");
    assert!(joint_events > 10, "{joint_events} joint events");
    let bullet_x = world
        .bodies()
        .map(|(_, b)| b)
        .find(|b| b.is_bullet)
        .unwrap()
        .position
        .x;
    assert!(
        (12.0..20.0).contains(&bullet_x),
        "the bullet escaped to x = {bullet_x}"
    );
}

#[test]
fn many_contacts_beginning_and_ending_do_not_allocate() {
    // Forty elastic balls dropped from different heights bounce out of phase, so contacts begin
    // and end every step in a store far larger than one node of any tree. Each ball has at most
    // one contact, with the ground; how many exist at once peaks rarely, when the bounces line
    // up, so the world reserves room for all of them, as Box2D's capacity hints do.
    let mut world = PhysicsWorld2d::new();
    world.reserve_contacts(40);
    add_static_box(&mut world, Vec2::new(0.0, -0.5), 60.0, 0.5);
    for i in 0..40 {
        let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(
            -50.0 + 2.5 * i as f32,
            1.0 + 0.37 * (i % 9) as f32,
        )));
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.4)))
                .with_restitution(1.0)
                .with_contact_events(true),
        );
    }
    let dt = 1.0 / 60.0;
    for _ in 0..600 {
        world.step(dt);
    }
    let mut churn = 0;
    let allocations = allocations_during(|| {
        for _ in 0..600 {
            world.step(dt);
            churn += world.contact_events().begin.len();
        }
    });
    assert!(churn > 200, "only {churn} contacts began");
    assert_eq!(
        allocations, 0,
        "600 steps of contact churn allocated {allocations} times"
    );
}

#[test]
fn ray_casts_do_not_allocate() {
    // Gameplay casts rays every frame; a capsule used to build a heap polygon for each cast.
    use korppi_physics2d::world::QueryFilter2d;
    let world = busy_world();
    let mut hit = None;
    let allocations = allocations_during(|| {
        hit = world.cast_ray_closest(
            Vec2::new(-3.0, 10.0),
            Vec2::new(0.0, -20.0),
            QueryFilter2d::default(),
        );
    });
    assert!(hit.is_some(), "the ray must hit the capsule on the stack");
    assert_eq!(allocations, 0, "a ray cast allocated {allocations} times");
}
