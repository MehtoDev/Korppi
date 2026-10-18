//! Observers (ADR-048): code that runs when a component is added, set or removed, after the
//! change that triggers it.

use korppi_ecs::{
    ChildOf, Commands, Component, Entity, OnAdd, OnRemove, OnSet, Query, Schedule, Stage, World,
};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Body(f32);
impl Component for Body {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position(f32);
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Enemy;
impl Component for Enemy {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Health(i32);
impl Component for Health {}

/// What the observers saw, in order.
#[derive(Default)]
struct Events(Vec<(&'static str, Entity, Option<f32>)>);

fn record(world: &mut World, event: &'static str, entity: Entity) {
    let body = world.get::<Body>(entity).map(|b| b.0);
    world
        .resource_mut::<Events>()
        .unwrap()
        .0
        .push((event, entity, body));
}

fn observed_world() -> World {
    let mut world = World::new();
    world.insert_resource(Events::default());
    world.observe::<OnAdd<Body>>(|world, entity| record(world, "add", entity));
    world.observe::<OnSet<Body>>(|world, entity| record(world, "set", entity));
    world.observe::<OnRemove<Body>>(|world, entity| record(world, "remove", entity));
    world
}

fn events(world: &mut World) -> Vec<(&'static str, Entity, Option<f32>)> {
    std::mem::take(&mut world.resource_mut::<Events>().unwrap().0)
}

#[test]
fn observers_see_adds_sets_and_removes_after_the_change() {
    let mut world = observed_world();

    let a = world.spawn((Body(1.0), Position(0.0)));
    // After the change: the observer already reads the new value.
    assert_eq!(
        events(&mut world),
        vec![("add", a, Some(1.0)), ("set", a, Some(1.0))]
    );

    let b = world.spawn((Position(0.0),));
    assert_eq!(events(&mut world), vec![]);
    world.add_component(b, Body(2.0));
    assert_eq!(
        events(&mut world),
        vec![("add", b, Some(2.0)), ("set", b, Some(2.0))]
    );
    world.add_component(b, Body(3.0));
    assert_eq!(events(&mut world), vec![("set", b, Some(3.0))]);
    // Another component's changes do not trigger them.
    world.add_component(b, Health(1));
    assert_eq!(events(&mut world), vec![]);

    assert_eq!(world.remove_component::<Body>(b), Some(Body(3.0)));
    assert_eq!(events(&mut world), vec![("remove", b, None)]);
    world.despawn(a);
    assert_eq!(events(&mut world), vec![("remove", a, None)]);
    assert!(!world.is_alive(a));
}

#[test]
fn despawn_cascades_report_every_removed_component() {
    let mut world = observed_world();
    let parent = world.spawn((Body(1.0),));
    let child = world.spawn((Body(2.0),));
    let grandchild = world.spawn((Position(0.0), Body(3.0)));
    let unobserved = world.spawn((Position(0.0),));
    world.add_pair::<ChildOf>(child, parent);
    world.add_pair::<ChildOf>(grandchild, child);
    world.add_pair::<ChildOf>(unobserved, child);
    events(&mut world);

    world.despawn(parent);
    assert_eq!(
        events(&mut world),
        vec![
            ("remove", parent, None),
            ("remove", child, None),
            ("remove", grandchild, None),
        ]
    );
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn observers_can_change_the_world_and_trigger_each_other() {
    let mut world = World::new();
    // Every new enemy gets health; full health triggers a second observer.
    world.observe::<OnAdd<Enemy>>(|world, entity| world.add_component(entity, Health(100)));
    world.observe::<OnSet<Health>>(|world, entity| {
        if world.get::<Health>(entity) == Some(&Health(100)) {
            world.add_component(entity, Position(9.0));
        }
    });

    let e = world.spawn((Enemy,));
    assert_eq!(world.get::<Health>(e), Some(&Health(100)));
    assert_eq!(world.get::<Position>(e), Some(&Position(9.0)));

    // An observer that despawns what it observes does not disturb the change that triggered it.
    world.observe::<OnAdd<Position>>(|world, entity| {
        world.despawn(entity);
    });
    let doomed = world.spawn((Position(0.0),));
    assert!(!world.is_alive(doomed));
    assert!(world.is_alive(e));
}

#[test]
fn commands_trigger_observers_when_they_apply() {
    let mut world = observed_world();
    let mut schedule = Schedule::new();
    schedule.add(
        Stage::Update,
        |mut bodies: Query<(Entity, &Body)>, mut commands: Commands| {
            for (entity, _) in bodies.iter_mut() {
                commands.despawn(entity);
            }
            commands.spawn((Body(5.0),));
        },
    );
    let old = world.spawn((Body(1.0),));
    events(&mut world);
    schedule.run(&mut world);
    let seen = events(&mut world);
    assert_eq!(seen[0], ("remove", old, None));
    assert_eq!(seen[1].0, "add");
    assert_eq!(seen[2].0, "set");
    assert_eq!(seen[1].2, Some(5.0));
}

#[test]
fn a_removed_observer_stops_running() {
    let mut world = World::new();
    world.insert_resource(Events::default());
    let id = world.observe::<OnAdd<Body>>(|world, entity| record(world, "add", entity));
    let a = world.spawn((Body(1.0),));
    assert!(world.remove_observer(id));
    assert!(!world.remove_observer(id));
    world.spawn((Body(2.0),));
    assert_eq!(events(&mut world), vec![("add", a, Some(1.0))]);
}
