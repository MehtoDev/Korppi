//! Zero-allocation regression test for simulation ticks (rule 4, ECS-06).
//!
//! This binary installs a counting global allocator that only counts allocations made by the
//! measuring thread while counting is on. After warm-up passes that let every archetype, edge,
//! pair ID and buffer reach its steady-state size, the operations a gameplay tick performs must not
//! allocate: query passes, cached query passes, `Schedule::run`, spawning, despawning, adding and
//! removing components, and parent/child churn with cascading despawns.

use korppi_ecs::{ChildOf, Component, Entity, Schedule, Stage, World};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

/// Forwards to [`System`] and counts allocations made on threads that enabled counting.
struct CountingAllocator;

fn record_allocation() {
    // `try_with` because allocations can happen while thread-locals are being torn down.
    let counting = COUNTING.try_with(Cell::get).unwrap_or(false);
    if counting {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract. The counter only touches const-initialized `Cell` thread-locals, which
// never allocate, so counting cannot recurse into the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: forwarded unchanged; the caller upholds `alloc_zeroed`'s contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
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

/// Number of heap allocations `f` makes on this thread.
fn allocations_in(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(Cell::get)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position(f32);
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Velocity(f32);
impl Component for Velocity {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Health(f32);
impl Component for Health {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Effect;
impl Component for Effect {}

fn movement(world: &mut World) {
    for (p, v) in world.query_mut::<(&mut Position, &Velocity)>() {
        p.0 += v.0;
    }
}

/// A parent with children in two archetypes, as an enemy with weapons and effects would be.
fn spawn_family(world: &mut World) -> Entity {
    let parent = world.spawn((Position(0.0), Velocity(1.0)));
    for i in 0..8 {
        let child = if i % 2 == 0 {
            world.spawn((Position(i as f32),))
        } else {
            world.spawn((Position(i as f32), Effect))
        };
        world.add_pair::<ChildOf>(child, parent);
    }
    parent
}

#[test]
fn simulation_tick_operations_do_not_allocate_in_steady_state() {
    const N: usize = 1_000;
    let mut world = World::new();
    let movers: Vec<Entity> = (0..N)
        .map(|i| world.spawn((Position(i as f32), Velocity(1.0))))
        .collect();
    let mut spawned = Vec::with_capacity(N);
    let mut schedule = Schedule::new();
    schedule.add_system(Stage::Update, movement);
    let mut cached = world.query_state::<(&mut Position, &Velocity)>();

    // Warm-up: every archetype, edge, pair ID and buffer reaches its steady-state size.
    for _ in 0..3 {
        movement(&mut world);
        schedule.run(&mut world);
        cached.for_each(&mut world, |(p, v)| p.0 += v.0);
        for i in 0..N {
            spawned.push(world.spawn((Position(i as f32), Velocity(0.0))));
        }
        for e in spawned.drain(..) {
            world.despawn(e);
        }
        for &e in &movers {
            world.add_component(e, Health(1.0));
        }
        for &e in &movers {
            world.remove_component::<Health>(e);
        }
        for _ in 0..4 {
            let parent = spawn_family(&mut world);
            world.despawn(parent);
        }
    }

    assert_eq!(allocations_in(|| movement(&mut world)), 0, "query pass");
    assert_eq!(
        allocations_in(|| schedule.run(&mut world)),
        0,
        "Schedule::run"
    );
    assert_eq!(
        allocations_in(|| cached.for_each(&mut world, |(p, v)| p.0 += v.0)),
        0,
        "cached query pass"
    );
    assert_eq!(
        allocations_in(|| {
            for (p, v) in cached.iter_mut(&mut world) {
                p.0 += v.0;
            }
        }),
        0,
        "cached query iterator"
    );
    assert_eq!(
        allocations_in(|| {
            for i in 0..N {
                spawned.push(world.spawn((Position(i as f32), Velocity(0.0))));
            }
        }),
        0,
        "spawn"
    );
    assert_eq!(
        allocations_in(|| {
            for e in spawned.drain(..) {
                world.despawn(e);
            }
        }),
        0,
        "despawn"
    );
    assert_eq!(
        allocations_in(|| {
            for &e in &movers {
                world.add_component(e, Health(1.0));
            }
        }),
        0,
        "add_component"
    );
    assert_eq!(
        allocations_in(|| {
            for &e in &movers {
                world.remove_component::<Health>(e);
            }
        }),
        0,
        "remove_component"
    );
    assert_eq!(
        allocations_in(|| {
            for _ in 0..4 {
                let parent = spawn_family(&mut world);
                world.despawn(parent);
            }
        }),
        0,
        "parent/child churn with cascading despawn"
    );
    assert_eq!(world.entity_count(), N);
}

struct Gravity(f32);

#[derive(Debug, Clone, Copy, PartialEq)]
struct Spark(u32);
impl Component for Spark {}

#[test]
fn systems_with_parameters_and_commands_do_not_allocate_in_steady_state() {
    use korppi_ecs::{Commands, Query, Res};

    let mut world = World::new();
    world.insert_resource(Gravity(-1.0));
    for i in 0..1_000 {
        world.spawn((Position(i as f32), Velocity(0.0)));
    }
    let mut schedule = Schedule::new();
    schedule.add(
        Stage::Update,
        |mut moving: Query<(&mut Velocity, &mut Position)>, gravity: Res<Gravity>| {
            moving.for_each(|(v, p)| {
                v.0 += gravity.0;
                p.0 += v.0;
            });
        },
    );
    // Every run despawns last run's sparks and spawns new ones, through commands.
    schedule.add(
        Stage::Update,
        |mut sparks: Query<(Entity, &Spark)>, mut commands: Commands| {
            for (spark, _) in sparks.iter_mut() {
                commands.despawn(spark);
            }
            for i in 0..16 {
                let spark = commands.spawn((Spark(i),));
                commands.insert(spark, Position(0.0));
            }
        },
    );
    for _ in 0..5 {
        schedule.run(&mut world);
    }

    assert_eq!(allocations_in(|| schedule.run(&mut world)), 0);
    assert_eq!(world.entity_count(), 1_016);
}

#[test]
fn observed_churn_does_not_allocate_in_steady_state() {
    use korppi_ecs::{OnAdd, OnRemove};

    struct Live(i64);

    let mut world = World::new();
    world.insert_resource(Live(0));
    world.observe::<OnAdd<Spark>>(|world, _| world.resource_mut::<Live>().unwrap().0 += 1);
    world.observe::<OnRemove<Spark>>(|world, _| world.resource_mut::<Live>().unwrap().0 -= 1);
    let mut sparks = Vec::with_capacity(64);
    let churn = |world: &mut World, sparks: &mut Vec<Entity>| {
        for i in 0..64 {
            sparks.push(world.spawn((Spark(i), Position(0.0))));
        }
        for spark in sparks.drain(..) {
            world.despawn(spark);
        }
    };
    for _ in 0..3 {
        churn(&mut world, &mut sparks);
    }

    assert_eq!(allocations_in(|| churn(&mut world, &mut sparks)), 0);
    assert_eq!(world.resource::<Live>().unwrap().0, 0);
}

#[test]
fn instantiating_prefabs_does_not_allocate_in_steady_state() {
    use korppi_ecs::{ChildOf, Prefab};

    let mut world = World::new();
    let enemy = world.spawn((Health(10.0), Velocity(1.0), Prefab));
    let weapon = world.spawn((Spark(1), Prefab));
    world.add_pair::<ChildOf>(weapon, enemy);
    let mut enemies = Vec::with_capacity(32);
    let churn = |world: &mut World, enemies: &mut Vec<Entity>| {
        for _ in 0..32 {
            enemies.push(world.instantiate(enemy));
        }
        for (v, _) in world.query_mut::<(&Velocity, &Health)>() {
            assert_eq!(v.0, 1.0);
        }
        for e in enemies.drain(..) {
            world.despawn(e);
        }
    };
    for _ in 0..3 {
        churn(&mut world, &mut enemies);
    }

    assert_eq!(allocations_in(|| churn(&mut world, &mut enemies)), 0);
    assert_eq!(world.entity_count(), 2);
}
