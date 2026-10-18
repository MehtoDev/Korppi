//! Unit and integration test suite for ECS relationship pairs, hierarchies, and cascade cleanup.

use korppi_ecs::prelude::*;
use korppi_ecs::relation::{
    ChildOf, CleanupPolicy, IsA, Relation, RelationshipError, WithPair, WithRelation,
    WithoutRelation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Likes;
impl Component for Likes {}
impl Relation for Likes {
    const CLEANUP: CleanupPolicy = CleanupPolicy::Remove;
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Affinity(pub f32);
impl Component for Affinity {}
impl Relation for Affinity {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Platoon;
impl Component for Platoon {}
impl Relation for Platoon {
    const EXCLUSIVE: bool = true;
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}
impl Component for Position {}

#[test]
fn test_basic_pair_tag_addition_and_query() {
    let mut world = World::new();

    let alice = world.spawn_empty();
    let bob = world.spawn_empty();
    let charlie = world.spawn_empty();

    world.add_pair::<Likes>(alice, bob);
    world.add_pair::<Likes>(alice, charlie);

    assert!(world.has_pair::<Likes>(alice, bob));
    assert!(world.has_pair::<Likes>(alice, charlie));
    assert!(!world.has_pair::<Likes>(bob, alice));
    assert!(!world.has_pair::<Likes>(charlie, alice));

    assert!(world.has_relation::<Likes>(alice));
    assert!(!world.has_relation::<Likes>(bob));

    let targets = world.targets_of::<Likes>(alice);
    assert_eq!(targets.len(), 2);
    assert!(targets.contains(&bob));
    assert!(targets.contains(&charlie));

    let sources = world.sources_of::<Likes>(bob);
    assert_eq!(sources, vec![alice]);

    // Remove pair
    assert!(world.remove_pair::<Likes>(alice, bob));
    assert!(!world.has_pair::<Likes>(alice, bob));
    assert!(world.has_pair::<Likes>(alice, charlie));
    assert_eq!(world.targets_of::<Likes>(alice), vec![charlie]);
}

#[test]
fn test_pair_with_data_payload() {
    let mut world = World::new();

    let alice = world.spawn_empty();
    let bob = world.spawn_empty();

    world.set_pair(alice, bob, Affinity(0.85));

    assert!(world.has_pair::<Affinity>(alice, bob));
    assert_eq!(
        world.get_pair::<Affinity>(alice, bob),
        Some(&Affinity(0.85))
    );

    // Mutate pair data
    if let Some(affinity) = world.get_pair_mut::<Affinity>(alice, bob) {
        affinity.0 = 0.95;
    }
    assert_eq!(
        world.get_pair::<Affinity>(alice, bob),
        Some(&Affinity(0.95))
    );

    // Remove pair with payload
    let removed = world.remove_pair_data::<Affinity>(alice, bob);
    assert_eq!(removed, Some(Affinity(0.95)));
    assert!(!world.has_pair::<Affinity>(alice, bob));
}

#[test]
fn test_exclusive_relation_replacement() {
    let mut world = World::new();

    let unit = world.spawn_empty();
    let platoon1 = world.spawn_empty();
    let platoon2 = world.spawn_empty();

    world.add_pair::<Platoon>(unit, platoon1);
    assert!(world.has_pair::<Platoon>(unit, platoon1));
    assert_eq!(world.target_of::<Platoon>(unit), Some(platoon1));

    // Adding to platoon2 should atomically remove platoon1 due to exclusive relation
    world.add_pair::<Platoon>(unit, platoon2);
    assert!(!world.has_pair::<Platoon>(unit, platoon1));
    assert!(world.has_pair::<Platoon>(unit, platoon2));
    assert_eq!(world.target_of::<Platoon>(unit), Some(platoon2));
    assert_eq!(world.targets_of::<Platoon>(unit), vec![platoon2]);
}

#[test]
fn test_built_in_child_of_hierarchy_and_bidirectional_indexing() {
    let mut world = World::new();

    let sun = world.spawn_empty();
    let earth = world.spawn_empty();
    let moon = world.spawn_empty();

    world.add_pair::<ChildOf>(earth, sun);
    world.add_pair::<ChildOf>(moon, earth);

    assert_eq!(world.parent_of(moon), Some(earth));
    assert_eq!(world.parent_of(earth), Some(sun));
    assert_eq!(world.parent_of(sun), None);

    let earth_children = world.children_of(earth);
    assert_eq!(earth_children, vec![moon]);

    let sun_children = world.children_of(sun);
    assert_eq!(sun_children, vec![earth]);

    let moon_ancestors: Vec<Entity> = world.ancestors(moon).collect();
    assert_eq!(moon_ancestors, vec![earth, sun]);

    let sun_descendants: Vec<Entity> = world.descendants(sun).collect();
    assert_eq!(sun_descendants, vec![earth, moon]);
}

#[test]
fn test_cycle_prevention() {
    let mut world = World::new();

    let root = world.spawn_empty();
    let middle = world.spawn_empty();
    let leaf = world.spawn_empty();

    world.add_pair::<ChildOf>(middle, root);
    world.add_pair::<ChildOf>(leaf, middle);

    // Attempting to make root a child of leaf must fail with cycle error
    let res = world.try_add_pair::<ChildOf>(root, leaf);
    assert_eq!(res, Err(RelationshipError::CycleDetected));

    // Also attempting self-parenting must fail
    let self_res = world.try_add_pair::<ChildOf>(root, root);
    assert_eq!(self_res, Err(RelationshipError::CycleDetected));

    // Hierarchy remains intact
    assert_eq!(world.parent_of(root), None);
    assert_eq!(world.parent_of(middle), Some(root));
    assert_eq!(world.parent_of(leaf), Some(middle));
}

#[test]
fn test_cascade_delete_policy() {
    let mut world = World::new();

    let parent = world.spawn_empty();
    let child1 = world.spawn_empty();
    let child2 = world.spawn_empty();
    let grandchild = world.spawn_empty();

    world.add_pair::<ChildOf>(child1, parent);
    world.add_pair::<ChildOf>(child2, parent);
    world.add_pair::<ChildOf>(grandchild, child1);

    assert!(world.is_alive(parent));
    assert!(world.is_alive(child1));
    assert!(world.is_alive(child2));
    assert!(world.is_alive(grandchild));

    // Despawn parent -> ChildOf has CleanupPolicy::Delete, so entire subtree is cascade despawned
    assert!(world.despawn(parent));

    assert!(!world.is_alive(parent));
    assert!(!world.is_alive(child1));
    assert!(!world.is_alive(child2));
    assert!(!world.is_alive(grandchild));
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn test_remove_pair_cleanup_policy() {
    let mut world = World::new();

    let alice = world.spawn_empty();
    let bob = world.spawn_empty();

    world.add_pair::<Likes>(alice, bob);
    assert!(world.has_pair::<Likes>(alice, bob));

    // Likes has CleanupPolicy::Remove, so despawning bob leaves alice alive but removes the pair
    assert!(world.despawn(bob));

    assert!(world.is_alive(alice));
    assert!(!world.is_alive(bob));
    assert!(!world.has_pair::<Likes>(alice, bob));
    assert!(!world.has_relation::<Likes>(alice));
}

#[test]
fn test_isa_prefab_relationship() {
    let mut world = World::new();

    let warrior_prefab = world.spawn_empty();
    let unit1 = world.spawn_empty();
    let unit2 = world.spawn_empty();

    world.add_pair::<IsA>(unit1, warrior_prefab);
    world.add_pair::<IsA>(unit2, warrior_prefab);

    assert!(world.has_pair::<IsA>(unit1, warrior_prefab));
    assert!(world.has_pair::<IsA>(unit2, warrior_prefab));
    assert_eq!(world.targets_of::<IsA>(unit1), vec![warrior_prefab]);

    let instances = world.sources_of::<IsA>(warrior_prefab);
    assert_eq!(instances.len(), 2);
    assert!(instances.contains(&unit1));
    assert!(instances.contains(&unit2));

    // Despawn prefab -> CleanupPolicy::Remove removes pair from instances without killing them
    assert!(world.despawn(warrior_prefab));
    assert!(world.is_alive(unit1));
    assert!(world.is_alive(unit2));
    assert!(!world.has_pair::<IsA>(unit1, warrior_prefab));
    assert!(!world.has_relation::<IsA>(unit1));
}

#[test]
fn test_world_query_with_pairs_and_relations() {
    let mut world = World::new();

    let root_a = world.spawn_empty();
    let root_b = world.spawn_empty();

    let c1 = world.spawn((Position { x: 1.0, y: 1.0 },));
    let c2 = world.spawn((Position { x: 2.0, y: 2.0 },));
    let c3 = world.spawn((Position { x: 3.0, y: 3.0 },));
    let standalone = world.spawn((Position { x: 4.0, y: 4.0 },));

    world.add_pair::<ChildOf>(c1, root_a);
    world.add_pair::<ChildOf>(c2, root_a);
    world.add_pair::<ChildOf>(c3, root_b);

    // Query all entities with ChildOf relation (wildcard)
    let with_childof: Vec<Entity> = world
        .query_mut::<(Entity, WithRelation<ChildOf>)>()
        .map(|(e, _)| e)
        .collect();
    assert_eq!(with_childof.len(), 3);
    assert!(with_childof.contains(&c1));
    assert!(with_childof.contains(&c2));
    assert!(with_childof.contains(&c3));
    assert!(!with_childof.contains(&standalone));

    // Query entities that are children specifically of root_a
    let all_children: Vec<Entity> = world
        .query_mut::<(Entity, WithPair<ChildOf>)>()
        .map(|(e, _)| e)
        .collect();
    let children_of_a: Vec<Entity> = all_children
        .into_iter()
        .filter(|&e| world.has_pair::<ChildOf>(e, root_a))
        .collect();
    assert_eq!(children_of_a.len(), 2);
    assert!(children_of_a.contains(&c1));
    assert!(children_of_a.contains(&c2));

    // Query entities without ChildOf relation
    let without_childof: Vec<Entity> = world
        .query_mut::<(Entity, WithoutRelation<ChildOf>)>()
        .map(|(e, _)| e)
        .collect();
    assert!(without_childof.contains(&standalone));
    assert!(without_childof.contains(&root_a));
    assert!(without_childof.contains(&root_b));
    assert!(!without_childof.contains(&c1));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MonitoredBy;
impl Component for MonitoredBy {}
impl Relation for MonitoredBy {
    const CLEANUP: CleanupPolicy = CleanupPolicy::Clear;
}

#[test]
fn test_cleanup_policy_clear_removes_pair_from_archetype() {
    let mut world = World::new();
    let source = world.spawn((Position { x: 1.0, y: 1.0 },));
    let target = world.spawn((Position { x: 2.0, y: 2.0 },));

    world.add_pair::<MonitoredBy>(source, target);
    assert!(world.has_pair::<MonitoredBy>(source, target));
    assert_eq!(
        world
            .query_mut::<(Entity, WithRelation<MonitoredBy>)>()
            .count(),
        1
    );

    // Despawning target with CleanupPolicy::Clear must remove pair from source and its archetype table
    world.despawn(target);
    assert!(!world.has_pair::<MonitoredBy>(source, target));
    assert_eq!(
        world
            .query_mut::<(Entity, WithRelation<MonitoredBy>)>()
            .count(),
        0
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OwnedBy;
impl Component for OwnedBy {}
impl Relation for OwnedBy {
    const CLEANUP: CleanupPolicy = CleanupPolicy::Delete;
}

#[test]
fn test_deterministic_cascade_despawn_order() {
    let mut expected_first_reused_index = None;

    for _ in 0..32 {
        let mut world = World::new();
        let target = world.spawn((Position { x: 0.0, y: 0.0 },));
        let c1 = world.spawn((Position { x: 1.0, y: 0.0 },));
        let c2 = world.spawn((Position { x: 2.0, y: 0.0 },));

        world.add_pair::<ChildOf>(c1, target);
        world.add_pair::<OwnedBy>(c2, target);

        world.despawn(target);

        let next = world.spawn((Position { x: 3.0, y: 0.0 },));
        let reused_index = next.index();

        if let Some(expected) = expected_first_reused_index {
            assert_eq!(
                reused_index, expected,
                "Despawn order must be strictly deterministic across independent worlds"
            );
        } else {
            expected_first_reused_index = Some(reused_index);
        }
    }
}

#[test]
fn test_relationship_target_lifecycle_recycles_pair_ids_and_empty_archetypes() {
    let mut world = World::new();
    let initial_entity = world.spawn((Position { x: 1.0, y: 2.0 },));
    let initial_archetype_count = world.archetype_count();

    // Repeatedly spawn parent and child, add relation, and despawn parent
    for _ in 0..100 {
        let parent = world.spawn((Position { x: 0.0, y: 0.0 },));
        let child = world.spawn((Position { x: 0.0, y: 0.0 },));
        world.add_pair::<ChildOf>(child, parent);
        world.despawn(parent);
    }

    assert_eq!(world.entity_count(), 1);
    assert!(world.is_alive(initial_entity));
    // Empty archetypes and pair IDs from dead targets must have been cleaned up / recycled
    // so archetype count does not grow unboundedly with every target lifetime
    assert!(
        world.archetype_count() <= initial_archetype_count + 2,
        "Archetype count must not leak across target lifetimes, got {}",
        world.archetype_count()
    );
}

#[test]
fn test_large_hierarchy_despawn_integrity() {
    let mut w = World::new();
    let (parents, n) = if cfg!(miri) {
        (50, 1_000)
    } else {
        (1_000, 100_000)
    };
    let t_build_start = std::time::Instant::now();
    let ps: Vec<_> = (0..parents)
        .map(|i| {
            w.spawn((Position {
                x: i as f32,
                y: 0.0,
            },))
        })
        .collect();
    for i in 0..n {
        let c = w.spawn((Position {
            x: i as f32,
            y: 0.0,
        },));
        w.add_pair::<ChildOf>(c, ps[i % parents]);
    }
    let t_build = t_build_start.elapsed();

    let t_despawn_start = std::time::Instant::now();
    for &p in &ps {
        assert!(w.despawn(p));
    }
    let t_despawn = t_despawn_start.elapsed();

    eprintln!(
        "Build: {:?} ({:.1} ns/child), Despawn: {:?} ({:.1} ns/entity)",
        t_build,
        t_build.as_nanos() as f64 / n as f64,
        t_despawn,
        t_despawn.as_nanos() as f64 / (n + parents) as f64,
    );
    assert_eq!(w.entity_count(), 0);
}

// --- Cleanup of dead targets without limits (verification of the FIX-5.3 first pass) ---------

macro_rules! data_components {
    ($($name:ident),*) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq)]
            struct $name(u32);
            impl Component for $name {}
        )*
    };
}
data_components!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

macro_rules! remove_relations {
    ($($name:ident),*) => {
        $(
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            struct $name;
            impl Component for $name {}
            impl Relation for $name {
                const CLEANUP: CleanupPolicy = CleanupPolicy::Remove;
            }
        )*
    };
}
remove_relations!(Watches0, Watches1, Watches2, Watches3, Watches4, Watches5);

fn pos(x: f32) -> Position {
    Position { x, y: 0.0 }
}

/// A parent whose six children live in six archetypes: (Position, T0) ... (Position, T5).
fn parent_with_six_child_archetypes(world: &mut World) -> (Entity, Vec<Entity>) {
    let parent = world.spawn((pos(0.0),));
    let children = vec![
        world.spawn((pos(1.0), T0(0))),
        world.spawn((pos(1.0), T1(1))),
        world.spawn((pos(1.0), T2(2))),
        world.spawn((pos(1.0), T3(3))),
        world.spawn((pos(1.0), T4(4))),
        world.spawn((pos(1.0), T5(5))),
    ];
    for &child in &children {
        world.add_pair::<ChildOf>(child, parent);
    }
    (parent, children)
}

/// The entities a query `(Entity, Q)` yields, in iteration order.
fn sources_matching<Q: WorldQuery>(world: &mut World) -> Vec<Entity> {
    world.query_mut::<(Entity, Q)>().map(|(e, _)| e).collect()
}

#[test]
fn a_reused_pair_id_never_lands_in_an_archetype_of_the_dead_pair() {
    let mut world = World::new();
    let (parent, children) = parent_with_six_child_archetypes(&mut world);
    world.despawn(parent);
    assert!(children.iter().all(|&c| !world.is_alive(c)));
    assert_eq!(world.entity_count(), 0);

    // The dead parent's pair ID is free; the next new pair takes it.
    let prefab = world.spawn((pos(9.0),));
    let instance = world.spawn((pos(2.0), T5(50)));
    world.add_pair::<IsA>(instance, prefab);

    assert!(world.has_pair::<IsA>(instance, prefab));
    assert_eq!(
        sources_matching::<WithRelation<IsA>>(&mut world),
        vec![instance]
    );
    assert_eq!(
        sources_matching::<WithRelation<ChildOf>>(&mut world),
        Vec::<Entity>::new()
    );
    assert_eq!(world.get::<T5>(instance), Some(&T5(50)));
    assert_eq!(world.get::<Position>(instance), Some(&pos(2.0)));
}

#[test]
fn a_data_pair_reusing_a_freed_pair_id_keeps_its_value() {
    let mut world = World::new();
    let (parent, _) = parent_with_six_child_archetypes(&mut world);
    world.despawn(parent);

    // A pair with data takes the ID a zero-sized `ChildOf` pair had; its column must hold the data.
    let owner = world.spawn((pos(9.0),));
    let holder = world.spawn((pos(2.0), T5(50)));
    world.set_pair(holder, owner, Affinity(0.75));

    assert_eq!(
        world.get_pair::<Affinity>(holder, owner),
        Some(&Affinity(0.75))
    );
    assert_eq!(
        sources_matching::<WithRelation<Affinity>>(&mut world),
        vec![holder]
    );
    assert_eq!(
        sources_matching::<WithRelation<ChildOf>>(&mut world),
        Vec::<Entity>::new()
    );
    assert_eq!(world.get::<T5>(holder), Some(&T5(50)));
}

#[test]
fn every_remove_relation_to_a_dead_target_is_removed_from_its_sources() {
    let mut world = World::new();
    let target = world.spawn((pos(0.0),));
    let s0 = world.spawn((pos(1.0),));
    let s1 = world.spawn((pos(1.0),));
    let s2 = world.spawn((pos(1.0),));
    let s3 = world.spawn((pos(1.0),));
    let s4 = world.spawn((pos(1.0),));
    let s5 = world.spawn((pos(1.0),));
    let all = world.spawn((pos(2.0),));
    world.add_pair::<Watches0>(s0, target);
    world.add_pair::<Watches1>(s1, target);
    world.add_pair::<Watches2>(s2, target);
    world.add_pair::<Watches3>(s3, target);
    world.add_pair::<Watches4>(s4, target);
    world.add_pair::<Watches5>(s5, target);
    world.add_pair::<Watches0>(all, target);
    world.add_pair::<Watches1>(all, target);
    world.add_pair::<Watches2>(all, target);
    world.add_pair::<Watches3>(all, target);
    world.add_pair::<Watches4>(all, target);
    world.add_pair::<Watches5>(all, target);

    world.despawn(target);

    let sources = [s0, s1, s2, s3, s4, s5, all];
    assert!(sources.iter().all(|&s| world.is_alive(s)));
    assert_eq!(world.entity_count(), sources.len());
    assert!(!world.has_relation::<Watches0>(s0) && !world.has_relation::<Watches0>(all));
    assert!(!world.has_relation::<Watches5>(s5) && !world.has_relation::<Watches5>(all));
    assert_eq!(
        sources_matching::<WithRelation<Watches0>>(&mut world),
        Vec::<Entity>::new()
    );
    assert_eq!(
        sources_matching::<WithRelation<Watches4>>(&mut world),
        Vec::<Entity>::new()
    );
    assert_eq!(
        sources_matching::<WithRelation<Watches5>>(&mut world),
        Vec::<Entity>::new()
    );
    // Every source is back in the plain (Position) archetype.
    let mut plain = sources_matching::<&Position>(&mut world);
    plain.sort();
    let mut expected = sources.to_vec();
    expected.sort();
    assert_eq!(plain, expected);
    let position_only = world.components().id_of::<Position>().unwrap();
    for &s in &sources {
        let archetype = world
            .archetypes()
            .iter()
            .find(|a| a.entities().contains(&s))
            .unwrap();
        assert_eq!(archetype.components(), &[position_only]);
    }
}

/// An entity with T0..T11, built one component at a time.
fn wide(world: &mut World) -> Entity {
    let e = world.spawn((T0(0), T1(1), T2(2), T3(3), T4(4), T5(5)));
    world.add_component(e, T6(6));
    world.add_component(e, T7(7));
    world.add_component(e, T8(8));
    world.add_component(e, T9(9));
    world.add_component(e, T10(10));
    world.add_component(e, T11(11));
    e
}

/// Removes and re-adds each of T0..T11 on `e`, calling `check` while each one is missing.
fn cycle_components(world: &mut World, e: Entity, mut check: impl FnMut(&mut World)) {
    macro_rules! cycle {
        ($($t:ident($v:expr)),*) => {
            $(
                assert_eq!(world.remove_component::<$t>(e), Some($t($v)));
                check(world);
                world.add_component(e, $t($v));
            )*
        };
    }
    cycle!(
        T0(0),
        T1(1),
        T2(2),
        T3(3),
        T4(4),
        T5(5),
        T6(6),
        T7(7),
        T8(8),
        T9(9),
        T10(10),
        T11(11)
    );
}

#[test]
fn freeing_archetypes_with_many_edges_leaves_no_edge_into_a_freed_slot() {
    let mut world = World::new();
    let target = world.spawn((pos(0.0),));
    let source = wide(&mut world);
    world.add_pair::<Likes>(source, target);
    // 13 archetypes hold the pair: (T0..T11, Likes) and each one missing a component, linked by
    // 25 edges.
    cycle_components(&mut world, source, |_| {});

    world.despawn(target);
    assert!(!world.has_pair::<Likes>(source, target));
    assert_eq!(world.get::<T11>(source), Some(&T11(11)));

    // The next pair reuses the freed pair ID. Every archetype it reaches must be a new one that
    // records `IsA`, never a leftover of the dead `Likes` pair.
    let prefab = world.spawn((pos(1.0),));
    let instance = wide(&mut world);
    world.add_pair::<IsA>(instance, prefab);
    cycle_components(&mut world, instance, |world| {
        assert_eq!(sources_matching::<WithRelation<IsA>>(world), vec![instance]);
        assert_eq!(
            sources_matching::<WithRelation<Likes>>(world),
            Vec::<Entity>::new()
        );
    });

    assert!(world.has_pair::<IsA>(instance, prefab));
    for e in [source, instance] {
        assert_eq!(world.get::<T0>(e), Some(&T0(0)));
        assert_eq!(world.get::<T11>(e), Some(&T11(11)));
    }
}

#[test]
fn repeated_parent_lifetimes_with_many_child_archetypes_do_not_grow_the_world() {
    let mut world = World::new();
    let (parent, _) = parent_with_six_child_archetypes(&mut world);
    world.despawn(parent);
    let after_first = world.archetype_count();

    for _ in 0..50 {
        let (parent, children) = parent_with_six_child_archetypes(&mut world);
        let prefab = world.spawn((pos(0.0),));
        world.add_pair::<IsA>(children[5], prefab);
        world.despawn(prefab);
        world.despawn(parent);
    }

    assert_eq!(world.entity_count(), 0);
    assert!(
        world.archetype_count() <= after_first + 2,
        "archetypes grew from {after_first} to {}",
        world.archetype_count()
    );
}

#[test]
fn a_freed_archetype_is_forgotten_by_every_pair_it_held() {
    let mut world = World::new();
    let parent = world.spawn((pos(0.0),));
    let target = world.spawn((pos(0.0),));
    // The child's archetype holds a pair to each of them.
    let child = world.spawn((pos(1.0),));
    world.add_pair::<ChildOf>(child, parent);
    world.add_pair::<Likes>(child, target);
    let fan = world.spawn((pos(2.0), T6(6)));
    world.add_pair::<Likes>(fan, target);

    // The parent's despawn frees the child's archetype; a new archetype of live entities takes
    // the slot. The target's despawn must leave that archetype alone.
    world.despawn(parent);
    let reuser = world.spawn((pos(3.0), T7(7)));
    world.despawn(target);

    assert!(world.is_alive(reuser) && world.is_alive(fan));
    assert_eq!(world.get::<T7>(reuser), Some(&T7(7)));
    assert_eq!(world.get::<Position>(reuser), Some(&pos(3.0)));
    assert!(!world.has_relation::<Likes>(fan));
    assert_eq!(world.get::<T6>(fan), Some(&T6(6)));
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn children_keep_the_order_they_were_attached_in() {
    let mut world = World::new();
    let parent = world.spawn((pos(0.0),));
    let other = world.spawn((pos(0.0),));
    let c: Vec<Entity> = (0..5).map(|i| world.spawn((pos(i as f32),))).collect();
    // Attach in an order unlike the entities' index order.
    for i in [3, 0, 4, 1, 2] {
        world.add_pair::<ChildOf>(c[i], parent);
    }
    assert_eq!(world.children(parent), [c[3], c[0], c[4], c[1], c[2]]);
    assert_eq!(
        world.children_of(parent),
        vec![c[3], c[0], c[4], c[1], c[2]]
    );

    // Detaching, re-parenting and despawning keep the remaining children in order.
    world.remove_pair::<ChildOf>(c[0], parent);
    assert_eq!(world.children(parent), [c[3], c[4], c[1], c[2]]);
    world.add_pair::<ChildOf>(c[4], other);
    assert_eq!(world.children(parent), [c[3], c[1], c[2]]);
    assert_eq!(world.children(other), [c[4]]);
    world.despawn(c[1]);
    assert_eq!(world.children(parent), [c[3], c[2]]);
    // Attaching a child to its own parent again changes nothing.
    world.add_pair::<ChildOf>(c[3], parent);
    assert_eq!(world.children(parent), [c[3], c[2]]);
    assert!(world.children(c[0]).is_empty());
}
