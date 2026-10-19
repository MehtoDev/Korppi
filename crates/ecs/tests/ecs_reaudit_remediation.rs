//! Regression tests for `FIX-13.2` ECS re-audit findings (`ECS-17` through `ECS-21`).

use korppi_ecs::{
    Added, Changed, ChildOf, CleanupPolicy, Commands, Component, Entity, OnAdd, OnRemove, OnSet,
    Prefab, Relation, Schedule, Stage, World,
};
use std::alloc::Layout;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Pos(f32);
impl Component for Pos {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Selected;
impl Component for Selected {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Likes;
impl Component for Likes {}
impl Relation for Likes {
    const EXCLUSIVE: bool = false;
    const CLEANUP: CleanupPolicy = CleanupPolicy::Remove;
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Weight(u32);
impl Component for Weight {}
impl Relation for Weight {
    const EXCLUSIVE: bool = true;
    const CLEANUP: CleanupPolicy = CleanupPolicy::Remove;
}

// =========================================================================
// ECS-17 (Probe E13): Generational handle validation across relationship API
// =========================================================================

#[test]
fn ecs_17_stale_entity_handles_cannot_read_or_mutate_recycled_entity_relationships() {
    let mut world = World::new();

    // 1. Stale source handle with recycled slot index.
    let e1_gen1 = world.spawn((Pos(1.0),));
    let target = world.spawn((Pos(2.0),));
    world.despawn(e1_gen1);

    let e1_gen2 = world.spawn((Pos(3.0),));
    assert_eq!(e1_gen1.index(), e1_gen2.index());
    assert_ne!(e1_gen1.generation(), e1_gen2.generation());

    world.add_pair::<ChildOf>(e1_gen2, target);
    world.add_pair::<Likes>(e1_gen2, target);
    world.set_pair::<Weight>(e1_gen2, target, Weight(42));

    // Stale source handle must not read e1_gen2's relationships.
    assert!(!world.has_pair::<ChildOf>(e1_gen1, target));
    assert!(!world.has_relation::<ChildOf>(e1_gen1));
    assert_eq!(world.target_of::<ChildOf>(e1_gen1), None);
    assert!(world.targets_of::<Likes>(e1_gen1).is_empty());
    assert_eq!(world.parent_of(e1_gen1), None);
    assert_eq!(world.ancestors(e1_gen1).count(), 0);
    assert_eq!(world.get_pair::<Weight>(e1_gen1, target), None);
    assert_eq!(world.get_pair_mut::<Weight>(e1_gen1, target), None);

    // Stale source handle must not mutate or remove e1_gen2's relationships.
    assert!(!world.remove_pair::<Likes>(e1_gen1, target));
    assert!(world.has_pair::<Likes>(e1_gen2, target));
    assert_eq!(world.remove_pair_data::<Weight>(e1_gen1, target), None);
    assert_eq!(world.get_pair::<Weight>(e1_gen2, target), Some(&Weight(42)));

    // 2. Stale target handle with recycled slot index.
    let p_gen1 = world.spawn((Pos(10.0),));
    world.despawn(p_gen1);
    let p_gen2 = world.spawn((Pos(20.0),));
    assert_eq!(p_gen1.index(), p_gen2.index());
    assert_ne!(p_gen1.generation(), p_gen2.generation());

    let child = world.spawn((Pos(30.0),));
    world.add_pair::<ChildOf>(child, p_gen2);
    world.set_pair::<Weight>(child, p_gen2, Weight(99));

    assert!(world.sources_of::<ChildOf>(p_gen1).is_empty());
    assert!(world.children(p_gen1).is_empty());
    assert!(world.children_of(p_gen1).is_empty());
    assert_eq!(world.descendants(p_gen1).count(), 0);
    assert!(!world.has_pair::<ChildOf>(child, p_gen1));
    assert_eq!(world.get_pair::<Weight>(child, p_gen1), None);
    assert_eq!(world.get_pair_mut::<Weight>(child, p_gen1), None);
    assert!(!world.remove_pair::<ChildOf>(child, p_gen1));
    assert_eq!(world.remove_pair_data::<Weight>(child, p_gen1), None);

    // Live handles still see the intact relationships.
    assert_eq!(world.parent_of(child), Some(p_gen2));
    assert_eq!(world.children(p_gen2), &[child]);
    assert_eq!(world.get_pair::<Weight>(child, p_gen2), Some(&Weight(99)));
}

// =========================================================================
// ECS-18 (Probe E14): Per-row change detection without false bulk_changed
// =========================================================================

#[test]
fn ecs_18_filtered_and_short_circuited_mutable_queries_only_mark_yielded_rows_changed() {
    let mut world = World::new();
    let mut pos_changed = world.query_state::<(Entity, Changed<Pos>)>();
    let mut mutate_added_selected = world.query_state::<(&mut Pos, Added<Selected>)>();

    let e1 = world.spawn((Pos(1.0), Selected));
    let e2 = world.spawn((Pos(2.0), Selected));

    // Baseline: clear initial Added/Changed ticks for both queries.
    assert_eq!(mutate_added_selected.iter_mut(&mut world).count(), 2);
    assert_eq!(pos_changed.iter_mut(&mut world).count(), 2);
    assert_eq!(pos_changed.iter_mut(&mut world).count(), 0);

    // Re-add `Selected` only on `e1` so `e1` and `e2` stay in the same archetype `(Pos, Selected)`
    // while only `e1` has `Added<Selected>`.
    world.remove_component::<Selected>(e1);
    world.add_component(e1, Selected);

    let mut mutated = 0;
    mutate_added_selected.for_each(&mut world, |(p, ())| {
        p.0 += 10.0;
        mutated += 1;
    });
    assert_eq!(mutated, 1);

    // Case A: Only `e1` was yielded and mutated; `e2` must NOT be reported as `Changed<Pos>`.
    let changed: Vec<Entity> = pos_changed.iter_mut(&mut world).map(|(e, ())| e).collect();
    assert_eq!(changed, vec![e1]);

    // Case B: Short-circuited `.next()` on a mutable query marks only the yielded row.
    if let Some((_e, p)) = world.query_mut::<(Entity, &mut Pos)>().next() {
        p.0 += 100.0;
    }
    let changed_after_next: Vec<Entity> =
        pos_changed.iter_mut(&mut world).map(|(e, ())| e).collect();
    assert_eq!(changed_after_next.len(), 1);

    // Case C: Tuple ordering `(Changed<Pos>, &mut Pos)` vs `(&mut Pos, Changed<Pos>)` must both
    // evaluate filters before stamping the current run's change tick.
    let mut filter_first = world.query_state::<(Entity, Changed<Pos>, &mut Pos)>();
    assert_eq!(filter_first.iter_mut(&mut world).count(), 2);
    assert_eq!(filter_first.iter_mut(&mut world).count(), 0);
    world.get_mut::<Pos>(e2).unwrap().0 = 77.0;
    let seen: Vec<Entity> = filter_first
        .iter_mut(&mut world)
        .map(|(e, (), p)| {
            p.0 += 1.0;
            e
        })
        .collect();
    assert_eq!(seen, vec![e2]);
    assert_eq!(filter_first.iter_mut(&mut world).count(), 0);
}

// =========================================================================
// ECS-19 (Probe E15): CommandQueue unwind safety and unplaced entity cleanup
// =========================================================================

struct DropCounter(Arc<AtomicUsize>);
impl Component for DropCounter {}
impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone, Copy)]
struct Boom;
impl Component for Boom {}

#[test]
fn ecs_19_command_queue_unwind_drops_remaining_commands_and_reclaims_unplaced_entities() {
    let mut world = World::new();
    world.observe::<OnAdd<Boom>>(|_w, _e| {
        panic!("observer panic during CommandQueue::apply");
    });

    let drops = Arc::new(AtomicUsize::new(0));
    let recorded = Arc::new(Mutex::new(Vec::<Entity>::new()));
    let run_count = Arc::new(AtomicUsize::new(0));

    let drops_sys = Arc::clone(&drops);
    let recorded_sys = Arc::clone(&recorded);
    let run_count_sys = Arc::clone(&run_count);

    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, move |mut cmds: Commands| {
        let step = run_count_sys.fetch_add(1, Ordering::SeqCst);
        if step == 0 {
            let e1 = cmds.spawn((Pos(1.0), Boom));
            let e2 = cmds.spawn((Pos(2.0), DropCounter(Arc::clone(&drops_sys))));
            recorded_sys.lock().unwrap().extend([e1, e2]);
        } else if step == 2 {
            let e_aborted = cmds.spawn((Pos(99.0), DropCounter(Arc::clone(&drops_sys))));
            recorded_sys.lock().unwrap().push(e_aborted);
            panic!("system panic before CommandQueue::apply");
        }
    });

    // Case A: Panic inside `CommandQueue::apply` (during `e1`'s `OnAdd<Boom>` observer).
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        schedule.run(&mut world);
    }));
    assert!(res.is_err());

    let (e1, e2) = {
        let g = recorded.lock().unwrap();
        (g[0], g[1])
    };

    // 1. Unexecuted command payload `DropCounter` must have been dropped during unwind.
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    // 2. `e1` was placed before its observer panicked; `e2` was never placed and must be reclaimed.
    assert!(world.is_alive(e1));
    assert!(!world.is_alive(e2));
    assert_eq!(world.entity_count(), 1);
    assert_eq!(world.entities().count(), 1);

    // 3. Re-running `schedule` must not re-run the old queued commands.
    schedule.run(&mut world);
    assert_eq!(world.entity_count(), 1);

    // 4. Next entity spawn reuses `e2`'s freed slot index with an incremented generation.
    let e3 = world.spawn((Pos(3.0),));
    assert_eq!(e3.index(), e2.index());
    assert!(e3.generation() > e2.generation());

    // Case B: Panic inside the system body before `CommandQueue::apply` is reached.
    let res2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        schedule.run(&mut world);
    }));
    assert!(res2.is_err());

    let e_aborted = recorded.lock().unwrap()[2];
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert!(!world.is_alive(e_aborted));
    assert_eq!(world.entity_count(), 2);

    // Subsequent normal run does not execute the aborted command, and reuses `e_aborted`'s slot.
    schedule.run(&mut world);
    assert_eq!(world.entity_count(), 2);
    let e4 = world.spawn((Pos(4.0),));
    assert_eq!(e4.index(), e_aborted.index());
    assert!(e4.generation() > e_aborted.generation());
}

// =========================================================================
// ECS-20 (Probe E16): Dynamic byte observers, ID prefab inheritance, pair observers
// =========================================================================

#[test]
fn ecs_20_dynamic_bytes_id_prefab_inheritance_and_pair_observers_are_consistent() {
    let mut world = World::new();

    // 1. Prefab inheritance through ID-based and dynamic-byte APIs.
    let dyn_id = world.register_dynamic_component("Mana", Layout::new::<u32>());
    let prefab = world.spawn((Pos(100.0), Prefab));
    let health_id = world.components().id_of::<Pos>().unwrap();
    world.insert_bytes(prefab, dyn_id, &50u32.to_ne_bytes());

    let inst = world.instantiate(prefab);
    assert!(world.has::<Pos>(inst));
    assert!(world.has_component_id(inst, health_id));
    assert_eq!(world.get_by_id::<Pos>(inst, health_id), Some(&Pos(100.0)));
    assert!(world.has_component_id(inst, dyn_id));
    assert_eq!(
        world.get_bytes(inst, dyn_id),
        Some(&50u32.to_ne_bytes()[..])
    );
    assert!(world.entities_with(health_id).any(|e| e == inst));
    assert!(world.entities_with(dyn_id).any(|e| e == inst));

    // 2. Relationship pairs trigger OnAdd / OnSet / OnRemove observers.
    let child_adds = Arc::new(AtomicUsize::new(0));
    let child_removes = Arc::new(AtomicUsize::new(0));
    let weight_sets = Arc::new(AtomicUsize::new(0));

    {
        let child_adds = Arc::clone(&child_adds);
        world.observe::<OnAdd<ChildOf>>(move |_w, _e| {
            child_adds.fetch_add(1, Ordering::SeqCst);
        });
    }
    {
        let child_removes = Arc::clone(&child_removes);
        world.observe::<OnRemove<ChildOf>>(move |_w, _e| {
            child_removes.fetch_add(1, Ordering::SeqCst);
        });
    }
    {
        let weight_sets = Arc::clone(&weight_sets);
        world.observe::<OnSet<Weight>>(move |_w, _e| {
            weight_sets.fetch_add(1, Ordering::SeqCst);
        });
    }

    let parent = world.spawn((Pos(0.0),));
    let child = world.spawn((Pos(1.0),));
    world.add_pair::<ChildOf>(child, parent);
    assert_eq!(child_adds.load(Ordering::SeqCst), 1);

    world.set_pair::<Weight>(child, parent, Weight(10));
    world.set_pair::<Weight>(child, parent, Weight(20));
    assert_eq!(weight_sets.load(Ordering::SeqCst), 2);

    world.remove_pair::<ChildOf>(child, parent);
    assert_eq!(child_removes.load(Ordering::SeqCst), 1);
}

// =========================================================================
// ECS-21 (Probe E17): Column::swap_remove_and_take ZST swapped flag & descendants
// =========================================================================

#[test]
fn ecs_21_zst_swap_remove_reports_swapped_and_descendants_traverse_accurately() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Tag;
    impl Component for Tag {}

    // Test through World component removal on ZST columns with multiple entities:
    let mut world = World::new();
    let e0 = world.spawn((Tag, Pos(1.0)));
    let e1 = world.spawn((Tag, Pos(2.0)));
    let e2 = world.spawn((Tag, Pos(3.0)));

    assert_eq!(world.remove_component::<Tag>(e0), Some(Tag));
    assert!(!world.has::<Tag>(e0));
    assert!(world.has::<Tag>(e1));
    assert!(world.has::<Tag>(e2));
    assert_eq!(world.get::<Pos>(e0), Some(&Pos(1.0)));
    assert_eq!(world.get::<Pos>(e1), Some(&Pos(2.0)));
    assert_eq!(world.get::<Pos>(e2), Some(&Pos(3.0)));

    // Hierarchy descendants order check:
    let root = world.spawn((Pos(0.0),));
    let c1 = world.spawn((Pos(1.0),));
    let c2 = world.spawn((Pos(2.0),));
    let g1 = world.spawn((Pos(3.0),));
    world.add_pair::<ChildOf>(c1, root);
    world.add_pair::<ChildOf>(c2, root);
    world.add_pair::<ChildOf>(g1, c1);

    let desc: Vec<Entity> = world.descendants(root).collect();
    assert_eq!(desc, vec![c1, g1, c2]);
}
