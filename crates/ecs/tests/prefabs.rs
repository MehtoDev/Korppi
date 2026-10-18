//! Prefabs (ADR-048): instances read their prefab's components through `IsA` until they
//! override them; queries skip prefabs unless they ask for them.

use korppi_ecs::{ChildOf, Component, Entity, Has, IsA, Prefab, RelationshipError, With, World};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Health(f32);
impl Component for Health {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Speed(f32);
impl Component for Speed {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Weapon(u32);
impl Component for Weapon {}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

#[test]
fn instances_read_their_prefabs_components_until_they_override_them() {
    let mut world = World::new();
    let goblin = world.spawn((Health(100.0), Speed(2.0), Prefab));
    let a = world.instantiate(goblin);
    let b = world.instantiate(goblin);
    assert!(world.has_pair::<IsA>(a, goblin));
    assert_eq!(world.get::<Health>(a), Some(&Health(100.0)));
    assert!(world.has::<Speed>(b));
    assert!(!world.has::<Prefab>(a), "the Prefab tag is not inherited");

    // An instance overrides a component by adding its own value.
    world.add_component(a, Health(50.0));
    assert_eq!(world.get::<Health>(a), Some(&Health(50.0)));
    assert_eq!(world.get::<Health>(b), Some(&Health(100.0)));

    // `get_mut` reaches owned values only, so an instance never writes its prefab.
    assert!(world.get_mut::<Speed>(a).is_none());
    world.get_mut::<Health>(a).unwrap().0 = 40.0;

    // Instances that do not override see the prefab's changes.
    world.get_mut::<Health>(goblin).unwrap().0 = 120.0;
    assert_eq!(world.get::<Health>(b), Some(&Health(120.0)));
    assert_eq!(world.get::<Health>(a), Some(&Health(40.0)));
}

#[test]
fn queries_see_inherited_components_and_skip_prefabs() {
    let mut world = World::new();
    let goblin = world.spawn((Health(100.0), Speed(2.0), Prefab));
    let a = world.instantiate(goblin);
    let b = world.instantiate(goblin);
    world.add_component(a, Health(50.0));
    let plain = world.spawn((Health(1.0),));

    let reads: Vec<(Entity, f32)> = world
        .query_mut::<(Entity, &Health)>()
        .map(|(e, h)| (e, h.0))
        .collect();
    let mut reads = reads;
    reads.sort_by_key(|&(e, _)| e);
    assert_eq!(reads, vec![(a, 50.0), (b, 100.0), (plain, 1.0)]);

    // A mutable term matches owned components only.
    let writers = sorted(
        world
            .query_mut::<(Entity, &mut Health)>()
            .map(|(e, _)| e)
            .collect(),
    );
    assert_eq!(writers, sorted(vec![a, plain]));

    let with_speed = sorted(
        world
            .query_mut::<(Entity, With<Speed>)>()
            .map(|(e, ())| e)
            .collect(),
    );
    assert_eq!(with_speed, sorted(vec![a, b]));
    let has_speed = sorted(world.query_mut::<(Entity, Has<Speed>)>().collect());
    assert_eq!(
        has_speed,
        sorted(vec![(a, true), (b, true), (plain, false)])
    );

    // A query that names `Prefab` sees prefabs.
    let prefabs: Vec<Entity> = world
        .query_mut::<(Entity, With<Prefab>)>()
        .map(|(e, ())| e)
        .collect();
    assert_eq!(prefabs, vec![goblin]);
    let all = world.query_mut::<(Entity, &Health, Has<Prefab>)>().count();
    assert_eq!(all, 4);
}

#[test]
fn cached_queries_follow_changes_to_the_prefab() {
    let mut world = World::new();
    let goblin = world.spawn((Health(100.0), Speed(2.0), Prefab));
    let a = world.instantiate(goblin);
    let b = world.instantiate(goblin);
    let mut speeds = world.query_state::<(Entity, &Speed)>();
    assert_eq!(speeds.iter_mut(&mut world).count(), 2);

    // The prefab loses Speed: so do its instances.
    world.remove_component::<Speed>(goblin);
    assert_eq!(speeds.iter_mut(&mut world).count(), 0);
    assert!(!world.has::<Speed>(a));

    world.add_component(goblin, Speed(3.0));
    let mut total = 0.0;
    speeds.for_each(&mut world, |(_, s)| total += s.0);
    assert_eq!(total, 6.0);
    assert_eq!(world.get::<Speed>(b), Some(&Speed(3.0)));
}

#[test]
fn instantiate_creates_instances_of_the_prefabs_children() {
    let mut world = World::new();
    let goblin = world.spawn((Health(100.0), Prefab));
    let dagger = world.spawn((Weapon(7), Prefab));
    let shield = world.spawn((Weapon(3), Prefab));
    world.add_pair::<ChildOf>(dagger, goblin);
    world.add_pair::<ChildOf>(shield, goblin);

    let a = world.instantiate(goblin);
    let children = world.children(a).to_vec();
    assert_eq!(children.len(), 2);
    assert!(world.has_pair::<IsA>(children[0], dagger));
    assert!(world.has_pair::<IsA>(children[1], shield));
    assert_eq!(world.get::<Weapon>(children[0]), Some(&Weapon(7)));
    assert_eq!(world.parent_of(children[1]), Some(a));

    // Despawning the instance despawns its children's instances, not the prefabs.
    world.despawn(a);
    assert!(children.iter().all(|&c| !world.is_alive(c)));
    assert!(world.is_alive(goblin) && world.is_alive(dagger) && world.is_alive(shield));
    assert_eq!(world.children(goblin), [dagger, shield]);
}

#[test]
fn nested_prefabs_inherit_transitively_and_isa_cycles_are_rejected() {
    let mut world = World::new();
    let base = world.spawn((Health(10.0), Speed(1.0), Prefab));
    let elite = world.spawn((Prefab,));
    world.add_pair::<IsA>(elite, base);
    world.add_component(elite, Health(20.0));

    let e = world.instantiate(elite);
    assert_eq!(world.get::<Health>(e), Some(&Health(20.0)));
    assert_eq!(world.get::<Speed>(e), Some(&Speed(1.0)));
    assert_eq!(
        world
            .query_mut::<(Entity, &Speed)>()
            .map(|(x, s)| (x, s.0))
            .collect::<Vec<_>>(),
        vec![(e, 1.0)]
    );

    assert_eq!(
        world.try_add_pair::<IsA>(base, e),
        Err(RelationshipError::CycleDetected)
    );
}

#[test]
fn a_despawned_prefab_leaves_instances_with_their_own_components_only() {
    let mut world = World::new();
    let goblin = world.spawn((Health(100.0), Speed(2.0), Prefab));
    let a = world.instantiate(goblin);
    let b = world.instantiate(goblin);
    world.add_component(a, Health(50.0));

    world.despawn(goblin);
    assert!(world.is_alive(a) && world.is_alive(b));
    assert_eq!(world.get::<Health>(a), Some(&Health(50.0)));
    assert_eq!(world.get::<Health>(b), None);
    assert_eq!(world.query_mut::<&Speed>().count(), 0);
}

#[test]
#[should_panic(expected = "inherited")]
fn chunk_iteration_refuses_inherited_components() {
    let mut world = World::new();
    let goblin = world.spawn((Health(100.0), Prefab));
    world.instantiate(goblin);
    world
        .query_mut::<&Health>()
        .for_each_chunk(|healths| assert!(!healths.is_empty()));
}
