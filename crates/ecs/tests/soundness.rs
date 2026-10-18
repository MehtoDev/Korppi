//! Regression tests for ECS soundness findings (FIX-5.1).

use korppi_ecs::component::ComponentRegistry;
use korppi_ecs::{Component, Entity, Relation, World};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Velocity {
    x: f32,
    y: f32,
}
impl Component for Velocity {}

/// Zero-sized tag component (ECS-04).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Frozen;
impl Component for Frozen {}

/// Non-zero sized relation (ECS-12).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Weight(f32);
impl Component for Weight {}
impl Relation for Weight {}

#[test]
#[should_panic(expected = "Conflicting access")]
fn query_mut_rejects_duplicate_mutable_terms_ecs_01() {
    let mut world = World::new();
    world.spawn((Position { x: 1.0, y: 0.0 },));
    // Querying the same component mutably twice is undefined behavior and must panic on construction.
    let _ = world.query_mut::<(&mut Position, &mut Position)>();
}

#[test]
#[should_panic(expected = "Conflicting access")]
fn query_mut_rejects_mutable_and_immutable_aliasing_ecs_01() {
    let mut world = World::new();
    world.spawn((Position { x: 1.0, y: 0.0 },));
    // Querying the same component immutably and mutably at the same time is aliasing UB and must panic.
    let _ = world.query_mut::<(&Position, &mut Position)>();
}

#[test]
fn query_mut_distinct_mutable_terms_succeeds_ecs_01() {
    let mut world = World::new();
    world.spawn((Position { x: 1.0, y: 0.0 }, Velocity { x: 2.0, y: 0.0 }));
    for (pos, vel) in world.query_mut::<(&mut Position, &mut Velocity)>() {
        pos.x += vel.x;
    }
    let pos = world.query_mut::<&Position>().next().unwrap();
    assert_eq!(pos.x, 3.0);
}

#[test]
fn get_by_id_type_checked_rejects_mismatched_type_ecs_02() {
    let mut world = World::new();
    let e = world.spawn((Position { x: 1.0, y: 2.0 },));

    let mut forged = ComponentRegistry::new();
    let pos_id = forged.register::<Position>();

    // Reading Position as u32 must not return the raw float bits; it must be type-checked and return None.
    let read_u32 = world.get_by_id::<u32>(e, pos_id);
    assert_eq!(read_u32, None);

    // Reading as the correct type succeeds.
    let read_pos = world.get_by_id::<Position>(e, pos_id);
    assert_eq!(read_pos, Some(&Position { x: 1.0, y: 2.0 }));
}

#[test]
#[should_panic(expected = "Duplicate component type")]
fn duplicate_bundle_types_rejected_on_registration_ecs_03() {
    let mut world = World::new();
    // Spawning a bundle with duplicate component types must panic on registration.
    world.spawn((Position { x: 1.0, y: 0.0 }, Position { x: 2.0, y: 0.0 }));
}

#[test]
fn zero_sized_components_get_query_and_removal_succeed_ecs_04() {
    let mut world = World::new();
    let e1 = world.spawn((Position { x: 1.0, y: 0.0 }, Frozen));
    let e2 = world.spawn((Position { x: 2.0, y: 0.0 }, Frozen));

    // 1. Has check
    assert!(world.has::<Frozen>(e1));
    assert!(world.has::<Frozen>(e2));

    // 2. Direct get of zero-sized component
    assert!(world.get::<Frozen>(e1).is_some());
    assert_eq!(world.get::<Frozen>(e1), Some(&Frozen));

    // 3. Query with &Frozen
    let count = world.query_mut::<(&Frozen, &Position)>().count();
    assert_eq!(count, 2);

    // 4. Safe removal of zero-sized component without panicking or corrupting table
    let removed = world.remove_component::<Frozen>(e1);
    assert_eq!(removed, Some(Frozen));
    assert!(!world.has::<Frozen>(e1));
    assert!(world.has::<Frozen>(e2));

    // 5. Verify world integrity: entities and remaining components are consistent
    let remaining: Vec<(Entity, f32)> = world
        .query_mut::<(Entity, &Position)>()
        .map(|(e, p)| (e, p.x))
        .collect();
    assert_eq!(remaining.len(), 2);
    assert!(remaining.contains(&(e1, 1.0)));
    assert!(remaining.contains(&(e2, 2.0)));
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn non_zero_sized_add_pair_fails_before_mutating_index_ecs_12() {
    let mut world = World::new();
    let a = world.spawn((Position { x: 0.0, y: 0.0 },));
    let t = world.spawn((Position { x: 0.0, y: 0.0 },));

    // try_add_pair with a data relation must fail before inserting into relationship index
    let result = world.try_add_pair::<Weight>(a, t);
    assert!(result.is_err());

    // Index must NOT record the pair
    assert!(!world.has_pair::<Weight>(a, t));
    assert_eq!(world.get_pair::<Weight>(a, t), None);
}

/// Heap-owning component, so a read of a dropped value is visible (and reported by Miri).
#[derive(Debug, Clone, PartialEq)]
struct Name(String);
impl Component for Name {}

#[test]
fn pair_ids_are_rejected_by_the_by_id_methods_ecs_02() {
    use korppi_ecs::ChildOf;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let mut world = World::new();
    let parent = world.spawn((Position { x: 0.0, y: 0.0 },));
    let child = world.spawn((Position { x: 1.0, y: 0.0 }, Name("child".into())));
    world.add_pair::<ChildOf>(child, parent);
    let bystander = world.spawn((Position { x: 2.0, y: 0.0 }, Name("bystander".into())));

    let childof = world.components().id_of::<ChildOf>().unwrap();
    let pair = world.components().pair_to_id(childof, parent).unwrap();

    // A pair added behind the relationship index's back would put `bystander` in the pair's
    // archetype without the index knowing; the target's despawn then frees that archetype under a
    // live entity. Pairs go through `add_pair`/`set_pair`/`remove_pair` only.
    let added = catch_unwind(AssertUnwindSafe(|| {
        world.add_component_by_id(bystander, pair, ChildOf);
    }));
    assert!(added.is_err(), "add_component_by_id must reject a pair id");
    assert_eq!(world.remove_component_by_id::<ChildOf>(child, pair), None);
    assert!(world.has_pair::<ChildOf>(child, parent));

    // The rejected calls changed nothing: the cascade removes exactly parent and child.
    world.despawn(parent);
    assert!(!world.is_alive(child));
    assert!(world.is_alive(bystander));
    assert_eq!(world.entity_count(), 1);
    assert_eq!(world.entities().collect::<Vec<_>>(), vec![bystander]);
    assert_eq!(
        world.get::<Name>(bystander),
        Some(&Name("bystander".into()))
    );
}

/// A component whose `Drop` panics.
struct PanicsOnDrop;
impl Component for PanicsOnDrop {}
impl Drop for PanicsOnDrop {
    fn drop(&mut self) {
        panic!("PanicsOnDrop dropped");
    }
}

/// Unwinding out of a component's `Drop` in the middle of a despawn would leave the archetype's
/// columns and entity list out of step (a later query reads past a column). The world aborts
/// instead, which this test observes from a child process running the scenario.
#[test]
#[cfg_attr(miri, ignore = "Miri cannot spawn processes")]
fn panicking_drop_during_a_structural_change_aborts_ecs_12() {
    const CHILD: &str = "KORPPI_ECS_PANICKING_DROP_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let mut world = World::new();
        world.spawn((Position { x: 1.0, y: 0.0 },));
        let doomed = world.spawn((Position { x: 2.0, y: 0.0 }, PanicsOnDrop));
        world.spawn((Position { x: 3.0, y: 0.0 }, PanicsOnDrop));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| world.despawn(doomed)));
        // Reached only if the panic unwound out of the world: the world is still in use.
        let _ = world.query_mut::<&Position>().count();
        return;
    }

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "panicking_drop_during_a_structural_change_aborts_ecs_12",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .expect("failed to re-run the test binary");
    assert!(
        !output.status.success(),
        "the world kept running after a component's Drop panicked mid-despawn"
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(output.status.signal(), Some(6), "expected SIGABRT");
    }
}
