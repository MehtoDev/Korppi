//! Query terms beyond plain component references (ADR-048): optional components, `Has<T>`,
//! the target of an exclusive relation, change detection, and bundles of twelve components.

use korppi_ecs::{Added, Changed, ChildOf, Component, Entity, Has, Target, World};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position(f32);
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Velocity(f32);
impl Component for Velocity {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Frozen;
impl Component for Frozen {}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

#[test]
fn optional_terms_yield_the_component_where_it_exists() {
    let mut world = World::new();
    let moving = world.spawn((Position(0.0), Velocity(2.0)));
    let still = world.spawn((Position(5.0),));

    let mut seen: Vec<(Entity, Option<f32>)> = world
        .query_mut::<(Entity, &Position, Option<&Velocity>)>()
        .map(|(e, _, v)| (e, v.map(|v| v.0)))
        .collect();
    seen.sort_by_key(|&(e, _)| e);
    assert_eq!(seen, vec![(moving, Some(2.0)), (still, None)]);

    for (p, v) in world.query_mut::<(&mut Position, Option<&mut Velocity>)>() {
        if let Some(v) = v {
            p.0 += v.0;
            v.0 = 0.0;
        } else {
            p.0 = -1.0;
        }
    }
    assert_eq!(world.get::<Position>(moving), Some(&Position(2.0)));
    assert_eq!(world.get::<Velocity>(moving), Some(&Velocity(0.0)));
    assert_eq!(world.get::<Position>(still), Some(&Position(-1.0)));
}

#[test]
fn has_terms_report_presence_without_borrowing() {
    let mut world = World::new();
    let frozen = world.spawn((Position(0.0), Frozen));
    let free = world.spawn((Position(1.0),));

    let flags = sorted(world.query_mut::<(Entity, Has<Frozen>)>().collect());
    assert_eq!(flags, vec![(frozen, true), (free, false)]);

    // `Has<T>` does not borrow `T`, so it combines with `&mut T`.
    let mut count = 0;
    for (_, has) in world.query_mut::<(&mut Frozen, Has<Frozen>)>() {
        assert!(has);
        count += 1;
    }
    assert_eq!(count, 1);
}

#[test]
fn target_terms_yield_each_entitys_parent() {
    let mut world = World::new();
    let a = world.spawn((Position(0.0),));
    let b = world.spawn((Position(0.0),));
    let a1 = world.spawn((Position(1.0),));
    let a2 = world.spawn((Position(2.0), Frozen));
    let b1 = world.spawn((Position(3.0),));
    world.add_pair::<ChildOf>(a1, a);
    world.add_pair::<ChildOf>(a2, a);
    world.add_pair::<ChildOf>(b1, b);

    let parents = sorted(world.query_mut::<(Entity, Target<ChildOf>)>().collect());
    assert_eq!(parents, sorted(vec![(a1, a), (a2, a), (b1, b)]));

    // Re-parenting moves the child to an archetype whose target is the new parent.
    world.add_pair::<ChildOf>(a2, b);
    let a2_parent = world
        .query_mut::<(Entity, Target<ChildOf>)>()
        .find(|&(child, _)| child == a2)
        .map(|(_, parent)| parent);
    assert_eq!(a2_parent, Some(b));
}

#[test]
fn added_and_changed_report_what_happened_since_the_previous_run() {
    let mut world = World::new();
    let mut added = world.query_state::<(Entity, Added<Position>)>();
    let mut changed = world.query_state::<(Entity, Changed<Position>)>();
    let run = |state: &mut korppi_ecs::QueryState<(Entity, Added<Position>)>, world: &mut World| {
        sorted(state.iter_mut(world).map(|(e, _)| e).collect::<Vec<_>>())
    };
    let run_changed = |state: &mut korppi_ecs::QueryState<(Entity, Changed<Position>)>,
                       world: &mut World| {
        sorted(state.iter_mut(world).map(|(e, _)| e).collect::<Vec<_>>())
    };

    let e1 = world.spawn((Position(0.0),));
    let e2 = world.spawn((Position(0.0), Velocity(1.0)));
    // A new component counts as added and as changed.
    assert_eq!(run(&mut added, &mut world), sorted(vec![e1, e2]));
    assert_eq!(run_changed(&mut changed, &mut world), sorted(vec![e1, e2]));
    // Nothing happened since.
    assert_eq!(run(&mut added, &mut world), Vec::<Entity>::new());
    assert_eq!(run_changed(&mut changed, &mut world), Vec::<Entity>::new());

    // `get_mut` changes one entity's component.
    world.get_mut::<Position>(e2).unwrap().0 = 4.0;
    assert_eq!(run_changed(&mut changed, &mut world), vec![e2]);
    assert_eq!(run(&mut added, &mut world), Vec::<Entity>::new());

    // A mutable query marks every row it visits.
    for p in world.query_mut::<&mut Position>() {
        p.0 += 1.0;
    }
    assert_eq!(run_changed(&mut changed, &mut world), sorted(vec![e1, e2]));

    // Moving to another archetype keeps the ticks; adding the component anew does not.
    world.add_component(e1, Frozen);
    assert_eq!(run(&mut added, &mut world), Vec::<Entity>::new());
    world.remove_component::<Position>(e1);
    world.add_component(e1, Position(9.0));
    assert_eq!(run(&mut added, &mut world), vec![e1]);
    // Overwriting with `add_component` counts as a change.
    world.add_component(e2, Position(0.5));
    assert_eq!(run_changed(&mut changed, &mut world), sorted(vec![e1, e2]));
}

#[test]
fn change_detection_survives_the_change_tick_wrapping_around() {
    let mut world = World::new();
    let mut changed = world.query_state::<(Entity, Changed<Position>)>();
    let e = world.spawn((Position(0.0),));
    assert_eq!(changed.iter_mut(&mut world).count(), 1);

    // Far more runs than the tick counter holds, with the component untouched throughout.
    for _ in 0..5 {
        world.advance_change_tick(u32::MAX / 4);
        assert_eq!(changed.iter_mut(&mut world).count(), 0);
    }
    world.get_mut::<Position>(e).unwrap().0 = 1.0;
    assert_eq!(
        changed
            .iter_mut(&mut world)
            .map(|(e, _)| e)
            .collect::<Vec<_>>(),
        vec![e]
    );
}

#[test]
fn bundles_hold_up_to_twelve_components() {
    macro_rules! components {
        ($($name:ident),*) => {
            $(
                #[derive(Debug, Clone, Copy, PartialEq)]
                struct $name(u8);
                impl Component for $name {}
            )*
        };
    }
    components!(C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11);

    let mut world = World::new();
    let e = world.spawn((
        C0(0),
        C1(1),
        C2(2),
        C3(3),
        C4(4),
        C5(5),
        C6(6),
        C7(7),
        C8(8),
        C9(9),
        C10(10),
        C11(11),
    ));
    assert_eq!(world.get::<C0>(e), Some(&C0(0)));
    assert_eq!(world.get::<C11>(e), Some(&C11(11)));
    assert_eq!(
        world.archetypes()[world.archetype_count() - 1]
            .components()
            .len(),
        12
    );
}

#[test]
fn a_query_that_mutates_and_filters_on_changes_does_not_see_its_own_writes() {
    let mut world = World::new();
    let mut react = world.query_state::<(&mut Position, Changed<Position>)>();
    let a = world.spawn((Position(1.0),));
    let b = world.spawn((Position(2.0),));

    // First run: both are new; the query rewrites them.
    let mut visited = 0;
    react.for_each(&mut world, |(p, ())| {
        p.0 *= 10.0;
        visited += 1;
    });
    assert_eq!(visited, 2);

    // Its own writes are not changes to react to on its next run; another one's are.
    assert_eq!(react.iter_mut(&mut world).count(), 0);
    world.get_mut::<Position>(b).unwrap().0 = 0.0;
    let mut seen = Vec::new();
    react.for_each(&mut world, |(p, ())| seen.push(p.0));
    assert_eq!(seen, vec![0.0]);
    assert_eq!(world.get::<Position>(a), Some(&Position(10.0)));
}

#[test]
fn rows_that_arrive_after_a_mutable_pass_are_not_reported_as_changed_by_it() {
    let mut world = World::new();
    let mut changed = world.query_state::<(Entity, Changed<Position>)>();
    let resident = world.spawn((Position(0.0), Velocity(1.0)));
    let newcomer = world.spawn((Position(0.0),));
    assert_eq!(changed.iter_mut(&mut world).count(), 2);

    // A mutable pass over (Position, Velocity), then `newcomer` moves into that archetype with
    // its untouched Position.
    for (p, _) in world.query_mut::<(&mut Position, &Velocity)>() {
        p.0 += 1.0;
    }
    world.add_component(newcomer, Velocity(0.0));
    let seen: Vec<Entity> = changed.iter_mut(&mut world).map(|(e, _)| e).collect();
    assert_eq!(seen, vec![resident]);
}
