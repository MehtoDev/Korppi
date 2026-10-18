//! Integration tests for archetype-based Entity Component System (ECS).

use korppi_ecs::{Component, Schedule, Stage, With, Without, World};
use korppi_math::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position(Vec3);
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Velocity(Vec3);
impl Component for Velocity {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Mass(f32);
impl Component for Mass {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Frozen;
impl Component for Frozen {}

#[test]
fn entity_lifecycle_and_generational_safety() {
    let mut world = World::new();
    assert_eq!(world.entity_count(), 0);

    let e1 = world.spawn((Position(Vec3::new(1.0, 2.0, 3.0)), Velocity(Vec3::X)));
    assert!(world.is_alive(e1));
    assert_eq!(world.entity_count(), 1);

    // Retrieve components
    assert_eq!(
        world.get::<Position>(e1),
        Some(&Position(Vec3::new(1.0, 2.0, 3.0)))
    );
    assert_eq!(world.get::<Velocity>(e1), Some(&Velocity(Vec3::X)));
    assert_eq!(world.get::<Mass>(e1), None);

    // Despawn
    assert!(world.despawn(e1));
    assert!(!world.is_alive(e1));
    assert_eq!(world.entity_count(), 0);
    assert_eq!(world.get::<Position>(e1), None);

    // Re-spawning should reuse index with a new generation
    let e2 = world.spawn((Position(Vec3::ZERO),));
    assert_eq!(e2.index(), e1.index());
    assert_ne!(e2.generation(), e1.generation());

    // Stale entity e1 cannot access e2's data
    assert_eq!(world.get::<Position>(e1), None);
    assert_eq!(world.get::<Position>(e2), Some(&Position(Vec3::ZERO)));
}

#[test]
fn archetype_transition_on_add_and_remove_component() {
    let mut world = World::new();

    // Spawn with only Position
    let e = world.spawn((Position(Vec3::new(5.0, 0.0, 0.0)),));
    assert!(world.has::<Position>(e));
    assert!(!world.has::<Velocity>(e));

    // Add Velocity: moves entity across archetype edge
    world.add_component(e, Velocity(Vec3::new(0.0, 10.0, 0.0)));
    assert!(world.has::<Position>(e));
    assert!(world.has::<Velocity>(e));
    assert_eq!(
        world.get::<Position>(e),
        Some(&Position(Vec3::new(5.0, 0.0, 0.0)))
    );
    assert_eq!(
        world.get::<Velocity>(e),
        Some(&Velocity(Vec3::new(0.0, 10.0, 0.0)))
    );

    // Remove Velocity: moves entity back across archetype edge
    let removed = world.remove_component::<Velocity>(e);
    assert_eq!(removed, Some(Velocity(Vec3::new(0.0, 10.0, 0.0))));
    assert!(world.has::<Position>(e));
    assert!(!world.has::<Velocity>(e));
}

#[test]
fn query_iteration_updates_matching_archetypes() {
    let mut world = World::new();

    // Archetype 1: Position + Velocity
    let e1 = world.spawn((Position(Vec3::new(0.0, 0.0, 0.0)), Velocity(Vec3::X)));
    let e2 = world.spawn((Position(Vec3::new(10.0, 0.0, 0.0)), Velocity(Vec3::Y)));

    // Archetype 2: Position only (should not match query)
    let e3 = world.spawn((Position(Vec3::new(100.0, 0.0, 0.0)),));

    // Archetype 3: Position + Velocity + Mass (should match query)
    let e4 = world.spawn((
        Position(Vec3::new(50.0, 0.0, 0.0)),
        Velocity(Vec3::Z),
        Mass(10.0),
    ));

    // Execute query simulation step
    for (pos, vel) in world.query_mut::<(&mut Position, &Velocity)>() {
        pos.0 += vel.0;
    }

    assert_eq!(world.get::<Position>(e1), Some(&Position(Vec3::X)));
    assert_eq!(
        world.get::<Position>(e2),
        Some(&Position(Vec3::new(10.0, 1.0, 0.0)))
    );
    assert_eq!(
        world.get::<Position>(e3),
        Some(&Position(Vec3::new(100.0, 0.0, 0.0)))
    ); // Unmodified
    assert_eq!(
        world.get::<Position>(e4),
        Some(&Position(Vec3::new(50.0, 0.0, 1.0)))
    );
}

#[test]
fn query_filters_with_and_without_isolate_entities() {
    let mut world = World::new();

    // Normal moving entity
    let e_active = world.spawn((Position(Vec3::ZERO), Velocity(Vec3::X)));

    // Frozen moving entity
    let e_frozen = world.spawn((Position(Vec3::ZERO), Velocity(Vec3::X), Frozen));

    // Query for entities with Velocity and WITHOUT Frozen
    for (pos, vel, _) in world.query_mut::<(&mut Position, &Velocity, Without<Frozen>)>() {
        pos.0 += vel.0;
    }

    assert_eq!(world.get::<Position>(e_active), Some(&Position(Vec3::X)));
    assert_eq!(world.get::<Position>(e_frozen), Some(&Position(Vec3::ZERO))); // Frozen untouched

    // Query for entities with Velocity and WITH Frozen
    let mut frozen_count = 0;
    for _ in world.query_mut::<(&Velocity, With<Frozen>)>() {
        frozen_count += 1;
    }
    assert_eq!(frozen_count, 1);
}

#[test]
fn schedule_runs_systems_in_defined_stage_order() {
    let mut world = World::new();
    let mut schedule = Schedule::new();

    let e = world.spawn((Position(Vec3::ZERO), Velocity(Vec3::new(1.0, 0.0, 0.0))));

    // Add movement system in FixedUpdate
    schedule.add_system(Stage::FixedUpdate, |world: &mut World| {
        for (pos, vel) in world.query_mut::<(&mut Position, &Velocity)>() {
            pos.0 += vel.0;
        }
    });

    // Run stage twice
    schedule.run_stage(Stage::FixedUpdate, &mut world);
    schedule.run_stage(Stage::FixedUpdate, &mut world);

    assert_eq!(
        world.get::<Position>(e),
        Some(&Position(Vec3::new(2.0, 0.0, 0.0)))
    );
}

#[test]
fn query_for_each_updates_components() {
    let mut world = World::new();
    let e1 = world.spawn((Position(Vec3::ZERO), Velocity(Vec3::X)));
    let e2 = world.spawn((Position(Vec3::Y), Velocity(Vec3::Y)));

    world
        .query_mut::<(&mut Position, &Velocity)>()
        .for_each(|(pos, vel)| {
            pos.0 += vel.0;
        });

    assert_eq!(world.get::<Position>(e1), Some(&Position(Vec3::X)));
    assert_eq!(
        world.get::<Position>(e2),
        Some(&Position(Vec3::new(0.0, 2.0, 0.0)))
    );
}

#[test]
fn query_state_caches_and_incrementally_matches_archetypes() {
    let mut world = World::new();
    let mut qstate = world.query_state::<(&mut Position, &Velocity)>();

    // Initial state matching: 0 archetypes
    assert_eq!(qstate.matched_archetypes().len(), 0);

    // Spawn entity -> creates Archetype 1
    world.spawn((Position(Vec3::ZERO), Velocity(Vec3::X)));
    qstate.update_archetypes(world.archetypes());
    assert_eq!(qstate.matched_archetypes().len(), 1);

    // Spawn non-matching entity -> creates Archetype 2
    world.spawn((Position(Vec3::ZERO), Mass(5.0)));
    qstate.update_archetypes(world.archetypes());
    assert_eq!(qstate.matched_archetypes().len(), 1);

    // Spawn matching entity in new archetype -> creates Archetype 3
    world.spawn((Position(Vec3::ZERO), Velocity(Vec3::Y), Mass(10.0)));
    qstate.update_archetypes(world.archetypes());
    assert_eq!(qstate.matched_archetypes().len(), 2);
}

#[test]
fn sorted_column_storage_and_typed_slices() {
    let mut world = World::new();
    let _e1 = world.spawn((Position(Vec3::X), Velocity(Vec3::Y), Mass(1.0)));
    let _e2 = world.spawn((Position(Vec3::Y), Velocity(Vec3::Z), Mass(2.0)));

    let arch = &world.archetypes()[1];
    let pos_id = world.components().id_of::<Position>().unwrap();
    let col_idx = arch.column_index(pos_id).unwrap();

    // SAFETY: pos_id corresponds to Position component type.
    let slice = unsafe { arch.column_slice::<Position>(col_idx) };
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0], Position(Vec3::X));
    assert_eq!(slice[1], Position(Vec3::Y));
}

#[test]
fn cached_query_follows_archetype_creation_and_slot_reuse() {
    use korppi_ecs::ChildOf;

    let mut world = World::new();
    let mut moving = world.query_state::<(&mut Position, &Velocity)>();

    // Archetypes created after the query state are matched on its next run.
    let a = world.spawn((Position(Vec3::ZERO), Velocity(Vec3::X)));
    moving.for_each(&mut world, |(p, v)| p.0 += v.0);
    let b = world.spawn((Position(Vec3::ZERO), Velocity(Vec3::Y), Mass(1.0)));
    let still = world.spawn((Position(Vec3::ZERO), Mass(1.0)));
    for (p, v) in moving.iter_mut(&mut world) {
        p.0 += v.0;
    }
    assert_eq!(
        world.get::<Position>(a),
        Some(&Position(Vec3::new(2.0, 0.0, 0.0)))
    );
    assert_eq!(world.get::<Position>(b), Some(&Position(Vec3::Y)));
    assert_eq!(world.get::<Position>(still), Some(&Position(Vec3::ZERO)));

    // A child's archetype (Position, Velocity, ChildOf parent) matches until the parent's despawn
    // frees it; the next new archetype, which does not match, reuses the freed slot.
    let parent = world.spawn((Mass(0.0),));
    let child = world.spawn((Position(Vec3::ZERO), Velocity(Vec3::Z)));
    world.add_pair::<ChildOf>(child, parent);
    moving.for_each(&mut world, |(p, v)| p.0 += v.0);
    assert_eq!(world.get::<Position>(child), Some(&Position(Vec3::Z)));
    let slots = world.archetype_count();
    world.despawn(parent);
    let reuser = world.spawn((Position(Vec3::ZERO), Mass(2.0), Frozen));
    assert_eq!(
        world.archetype_count(),
        slots,
        "the freed archetype slot is reused"
    );

    moving.for_each(&mut world, |(p, v)| p.0 += v.0);
    assert_eq!(world.get::<Position>(reuser), Some(&Position(Vec3::ZERO)));
    assert_eq!(
        world.get::<Position>(a),
        Some(&Position(Vec3::new(4.0, 0.0, 0.0)))
    );
    assert_eq!(
        world.get::<Position>(b),
        Some(&Position(Vec3::new(0.0, 3.0, 0.0)))
    );
    assert_eq!(moving.iter_mut(&mut world).count(), 2);
}

#[test]
#[should_panic(expected = "QueryState")]
fn cached_query_rejects_a_world_it_was_not_created_from() {
    let mut first = World::new();
    let mut second = World::new();
    let mut positions = first.query_state::<&Position>();
    second.spawn((Position(Vec3::ZERO),));
    positions.for_each(&mut second, |_| {});
}

#[test]
fn query_chunks_are_typed_slices_of_each_archetype() {
    use korppi_ecs::Entity;

    let mut world = World::new();
    let small: Vec<Entity> = (0..3)
        .map(|i| world.spawn((Position(Vec3::splat(i as f32)), Velocity(Vec3::X))))
        .collect();
    let heavy: Vec<Entity> = (0..2)
        .map(|_| world.spawn((Position(Vec3::ZERO), Velocity(Vec3::Y), Mass(1.0))))
        .collect();
    world.spawn((Position(Vec3::ZERO),));

    let mut chunks: Vec<Vec<Entity>> = Vec::new();
    world
        .query_mut::<(Entity, &mut Position, &Velocity)>()
        .for_each_chunk(|(entities, positions, velocities)| {
            assert_eq!(entities.len(), positions.len());
            assert_eq!(positions.len(), velocities.len());
            for (p, v) in positions.iter_mut().zip(velocities) {
                p.0 += v.0;
            }
            chunks.push(entities.to_vec());
        });
    assert_eq!(chunks, vec![small.clone(), heavy.clone()]);
    assert_eq!(
        world.get::<Position>(small[2]),
        Some(&Position(Vec3::new(3.0, 2.0, 2.0)))
    );
    assert_eq!(world.get::<Position>(heavy[0]), Some(&Position(Vec3::Y)));

    let mut cached = world.query_state::<(&mut Position, &Velocity)>();
    let mut rows = 0;
    cached.for_each_chunk(&mut world, |(positions, velocities)| {
        rows += positions.len();
        for (p, v) in positions.iter_mut().zip(velocities) {
            p.0 -= v.0;
        }
    });
    assert_eq!(rows, 5);
    assert_eq!(
        world.get::<Position>(small[2]),
        Some(&Position(Vec3::splat(2.0)))
    );
}
