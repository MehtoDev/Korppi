//! Integration tests verifying zero runtime heap allocations during steady-state simulation steps.

use korppi_core::{TaskExecutor, WorkerPool};
use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::joint::{DistanceJoint, Joint};
use korppi_physics::shape::Shape;
use korppi_physics::world::PhysicsWorld;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

struct CountingAllocator;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
}

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static COUNTING: Mutex<()> = Mutex::new(());

fn count_alloc() {
    let _ = TRACKING.try_with(|tracking| {
        if tracking.get() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    });
}

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract. The counter only touches a const-initialized `Cell` thread-local and
// an atomic, which never allocate, so counting cannot recurse into the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_alloc();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_alloc();
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

fn set_tracking(enable: bool) {
    TRACKING.with(|t| t.set(enable));
}

fn reset_alloc_count() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

fn get_alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

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
    TRACKING.with(|t| t.set(false));
}

#[test]
fn test_freefall_step_zero_allocations() {
    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Dynamic body in free-fall (no contacts)
    let body_id = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 50.0, 0.0)));
    world.create_collider(
        body_id,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_friction(0.5)
            .with_restitution(0.0),
    );

    let dt = 1.0 / 60.0;
    // Warm-up step to prime scratchpad buffers
    world.step(dt);

    // Track allocations over 60 simulation steps
    reset_alloc_count();
    set_tracking(true);

    for _ in 0..60 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Free-falling simulation step loop must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn test_joint_system_step_zero_allocations() {
    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // 2 dynamic bodies connected by a distance joint, no contact manifolds
    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 10.0, 0.0)));
    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 8.0, 0.0)));
    world.create_joint(Joint::Distance(DistanceJoint::new(
        b1,
        b2,
        Vec3::ZERO,
        Vec3::ZERO,
        2.0,
    )));

    let dt = 1.0 / 60.0;
    // Warm-up step
    world.step(dt);

    reset_alloc_count();
    set_tracking(true);

    for _ in 0..60 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Joint constraint simulation step loop must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn test_sleeping_stack_step_zero_allocations() {
    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let floor_id = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 0.0)));
    world.create_collider(
        floor_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        }),
    );

    let box_id = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0)));
    world.create_collider(
        box_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        }),
    );

    let dt = 1.0 / 60.0;
    // Step until the box settles and falls asleep
    for _ in 0..60 {
        world.step(dt);
    }
    assert!(
        world.body(box_id).unwrap().is_sleeping,
        "Box must be sleeping"
    );

    // Track allocations while resting/sleeping
    reset_alloc_count();
    set_tracking(true);

    for _ in 0..60 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Sleeping simulation step loop must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn test_gjk_epa_convex_collision_zero_allocations() {
    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let floor_id = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 0.0, 0.0)));
    world.create_collider(
        floor_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(5.0, 0.5, 5.0),
        }),
    );

    let hull_verts = vec![
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
    ];
    let hull_id = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.8, 0.0)));
    world.create_collider(
        hull_id,
        Collider::new(Shape::convex_hull(&hull_verts).unwrap()),
    );

    let dt = 1.0 / 60.0;
    // Step once to establish initial contacts
    world.step(dt);

    reset_alloc_count();
    set_tracking(true);

    for _ in 0..10 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Active GJK/EPA convex collision step loop must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn test_compound_collision_step_zero_allocations() {
    use korppi_math::Quat;
    use korppi_physics::{Compound3d, CompoundChild};

    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let b1 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        },
        Vec3::new(-1.05, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let b2 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        },
        Vec3::new(1.05, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let compound = Compound3d::new(vec![b1, b2]).unwrap();

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::Compound(compound)).with_friction(0.6),
    );

    let plank = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.69, 0.0)).with_enable_sleep(false),
    );
    world.create_collider(
        plank,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(1.2, 0.2, 0.5),
        })
        .with_friction(0.6),
    );

    let dt = 1.0 / 60.0;
    // Step to establish initial contact manifolds and prime scratchpad buffers
    for _ in 0..5 {
        world.step(dt);
    }

    reset_alloc_count();
    set_tracking(true);

    for _ in 0..10 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Compound collision step loop must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn test_events_step_zero_allocations() {
    use korppi_physics::{Joint, SphericalJoint};

    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_friction(0.5)
        .with_contact_events(true)
        .with_hit_events(true),
    );

    let ball = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0))
            .with_linear_velocity(Vec3::new(0.0, -5.0, 0.0)),
    );
    world.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_restitution(0.5)
            .with_contact_events(true)
            .with_hit_events(true),
    );

    let anchor = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(2.0, 5.0, 0.0)));
    let suspended = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 4.0, 0.0)));
    world.create_collider(suspended, Collider::new(Shape::Sphere { radius: 0.5 }));
    world.create_joint(
        Joint::Spherical(SphericalJoint::new(
            anchor,
            suspended,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        ))
        .with_force_threshold(0.0),
    );

    let dt = 1.0 / 60.0;
    // Step to establish initial contact, hit event, and prime scratchpad buffers
    for _ in 0..15 {
        world.step(dt);
    }

    reset_alloc_count();
    set_tracking(true);

    for _ in 0..20 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Simulation step with active contact, hit, body and joint events must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn test_graph_coloring_and_broadphase_step_zero_allocations() {
    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.7),
    );

    let mut bodies = Vec::new();
    for i in 0..5 {
        let b = world.create_body(
            RigidBody::new(BodyType::Dynamic, Vec3::new(i as f32 * 1.5, 0.5, 0.0))
                .with_enable_sleep(false),
        );
        world.create_collider(
            b,
            Collider::new(Shape::Box {
                half_extents: Vec3::splat(0.5),
            })
            .with_friction(0.7),
        );
        bodies.push(b);
    }

    world.create_joint(Joint::Distance(DistanceJoint::new(
        bodies[0],
        bodies[1],
        Vec3::ZERO,
        Vec3::ZERO,
        1.5,
    )));

    let dt = 1.0 / 60.0;
    // Warm-up to prime 3-tree broadphase and constraint graph colors
    for _ in 0..10 {
        world.step(dt);
    }

    reset_alloc_count();
    set_tracking(true);

    for _ in 0..20 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Broadphase pair tracking and graph coloring step loop must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn test_mesh_collision_step_zero_allocations() {
    use korppi_physics::Mesh3d;

    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let vertices = vec![
        Vec3::new(-10.0, 0.0, -10.0),
        Vec3::new(10.0, 0.0, -10.0),
        Vec3::new(10.0, 0.0, 10.0),
        Vec3::new(-10.0, 0.0, 10.0),
    ];
    let indices = vec![[0, 2, 1], [0, 3, 2]];
    let mesh = Mesh3d::new(&vertices, &indices).unwrap();

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::TriangleMesh(mesh)).with_friction(0.5),
    );

    let box_b = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)).with_enable_sleep(false),
    );
    world.create_collider(
        box_b,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.5),
    );

    let dt = 1.0 / 60.0;
    // Warm-up steps to prime mesh scratch and manifolds
    for _ in 0..10 {
        world.step(dt);
    }

    reset_alloc_count();
    set_tracking(true);

    for _ in 0..20 {
        world.step(dt);
    }

    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "Mesh contact manifold generation and step loop must perform 0 heap allocations, got {allocs}"
    );
}

#[test]
fn parallel_steps_do_not_allocate_on_any_thread() {
    let _turn = COUNTING.lock().unwrap_or_else(PoisonError::into_inner);
    let pool = Arc::new(WorkerPool::new(4));
    track_pool(&pool);

    // Verify that allocations on worker threads are counted.
    reset_alloc_count();
    set_tracking(true);
    let started = AtomicUsize::new(0);
    pool.run(4, &|_, _| {
        started.fetch_add(1, Ordering::AcqRel);
        while started.load(Ordering::Acquire) < 4 {
            std::hint::spin_loop();
        }
        std::hint::black_box(vec![0u8; 64]);
    });
    set_tracking(false);
    assert_eq!(get_alloc_count(), 4, "the four workers' allocations");

    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
    world.enable_sleep(false);
    let executor: Arc<dyn TaskExecutor> = pool;
    world.set_executor(Some(executor));

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.6),
    );
    let mut prev = None;
    for row in 0..5 {
        for i in 0..5 - row {
            let b = world.create_body(RigidBody::new(
                BodyType::Dynamic,
                Vec3::new(-2.0 + 0.5 * row as f32 + i as f32, 0.5 + row as f32, 0.0),
            ));
            world.create_collider(
                b,
                Collider::new(Shape::Box {
                    half_extents: Vec3::splat(0.5),
                })
                .with_friction(0.6)
                .with_restitution(0.2)
                .with_contact_events(true)
                .with_hit_events(true),
            );
            if row == 0 && i == 1 {
                if let Some(p) = prev {
                    world.create_joint(Joint::Distance(DistanceJoint::new(
                        p,
                        b,
                        Vec3::ZERO,
                        Vec3::ZERO,
                        1.0,
                    )));
                }
            }
            prev = Some(b);
        }
    }

    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step(dt);
    }

    reset_alloc_count();
    set_tracking(true);
    for _ in 0..120 {
        world.step(dt);
    }
    set_tracking(false);
    let allocs = get_alloc_count();
    assert_eq!(
        allocs, 0,
        "120 parallel steps allocated {allocs} times on main or worker threads"
    );
}
