//! Systems with typed parameters, commands and the schedule (ADR-048): what gameplay code writes.

use korppi_ecs::{
    Changed, ChildOf, Commands, Component, Entity, Local, Query, Res, ResMut, Schedule, Stage,
    With, World,
};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position(f32);
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Velocity(f32);
impl Component for Velocity {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Health(i32);
impl Component for Health {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Corpse;
impl Component for Corpse {}

struct Time {
    dt: f32,
}

#[derive(Default)]
struct Log(Vec<&'static str>);

fn movement(mut moving: Query<(&mut Position, &Velocity)>, time: Res<Time>) {
    moving.for_each(|(p, v)| p.0 += v.0 * time.dt);
}

#[test]
fn a_system_reads_resources_and_updates_queried_components() {
    let mut world = World::new();
    world.insert_resource(Time { dt: 0.5 });
    let a = world.spawn((Position(0.0), Velocity(2.0)));
    let still = world.spawn((Position(7.0),));

    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, movement);
    schedule.run(&mut world);
    schedule.run(&mut world);

    assert_eq!(world.get::<Position>(a), Some(&Position(2.0)));
    assert_eq!(world.get::<Position>(still), Some(&Position(7.0)));
    world.resource_mut::<Time>().unwrap().dt = 1.0;
    schedule.run(&mut world);
    assert_eq!(world.get::<Position>(a), Some(&Position(4.0)));
    assert_eq!(world.remove_resource::<Time>().map(|t| t.dt), Some(1.0));
    assert!(world.resource::<Time>().is_none());
}

fn reap(mut living: Query<(Entity, &Health, &Position)>, mut commands: Commands) {
    for (entity, health, position) in living.iter_mut() {
        if health.0 <= 0 {
            // Structural changes inside the loop are recorded, then applied after the system.
            commands.despawn(entity);
            let corpse = commands.spawn((Corpse, *position));
            commands.insert(corpse, Health(0));
        }
    }
}

fn count_corpses(corpses: Query<&Corpse>, mut log: ResMut<Log>) {
    let mut corpses = corpses;
    if corpses.iter_mut().count() > 0 {
        log.0.push("corpses");
    }
}

#[test]
fn commands_recorded_while_iterating_apply_after_the_system() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let alive = world.spawn((Health(10), Position(1.0)));
    let dead = world.spawn((Health(0), Position(2.0)));
    let also_dead = world.spawn((Health(-3), Position(3.0)));

    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, reap);
    schedule.add(Stage::Update, count_corpses);
    schedule.run(&mut world);

    assert!(world.is_alive(alive));
    assert!(!world.is_alive(dead) && !world.is_alive(also_dead));
    let mut corpse_positions: Vec<f32> = world
        .query_mut::<(&Position, &Health, With<Corpse>)>()
        .map(|(p, h, ())| {
            assert_eq!(h.0, 0);
            p.0
        })
        .collect();
    corpse_positions.sort_by(f32::total_cmp);
    assert_eq!(corpse_positions, vec![2.0, 3.0]);
    // The second system already saw the corpses: commands apply after each system.
    assert_eq!(world.resource::<Log>().unwrap().0, vec!["corpses"]);
    assert_eq!(world.entity_count(), 3);
}

#[test]
fn commands_cover_components_pairs_and_custom_closures() {
    let mut world = World::new();
    let parent = world.spawn((Position(0.0),));
    let child = world.spawn((Position(1.0), Velocity(1.0)));

    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, move |mut commands: Commands| {
        commands.add_pair::<ChildOf>(child, parent);
        commands.remove::<Velocity>(child);
        commands.add(|world: &mut World| world.insert_resource(Log(vec!["custom"])));
    });
    schedule.run(&mut world);

    assert_eq!(world.parent_of(child), Some(parent));
    assert!(!world.has::<Velocity>(child));
    assert_eq!(world.resource::<Log>().unwrap().0, vec!["custom"]);
}

fn step(name: &'static str) -> impl FnMut(ResMut<Log>) + Send + 'static {
    move |mut log: ResMut<Log>| log.0.push(name)
}

#[test]
fn the_schedule_orders_named_systems_and_honours_run_conditions() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule
        .add(Stage::Update, step("render"))
        .name("render")
        .after("physics");
    schedule
        .add(Stage::Update, step("input"))
        .name("input")
        .before("physics");
    schedule.add(Stage::Update, step("physics")).name("physics");
    schedule
        .add(Stage::Update, step("paused-only"))
        .name("paused-only")
        .run_if(|world: &World| world.resource::<Time>().is_some_and(|time| time.dt == 0.0));
    schedule.add(Stage::PreUpdate, step("pre")).name("pre");
    schedule.add_system(Stage::PostUpdate, |world: &mut World| {
        world.resource_mut::<Log>().unwrap().0.push("exclusive");
    });

    schedule.run(&mut world);
    assert_eq!(
        world.resource::<Log>().unwrap().0,
        vec!["pre", "input", "physics", "render", "exclusive"]
    );

    world.resource_mut::<Log>().unwrap().0.clear();
    world.insert_resource(Time { dt: 0.0 });
    assert!(schedule.set_enabled("physics", false));
    assert!(schedule.remove("render"));
    assert!(!schedule.remove("render"));
    schedule.run(&mut world);
    assert_eq!(
        world.resource::<Log>().unwrap().0,
        vec!["pre", "input", "paused-only", "exclusive"]
    );
}

#[test]
#[should_panic(expected = "cycle")]
fn an_ordering_cycle_is_rejected() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, step("a")).name("a").after("b");
    schedule.add(Stage::Update, step("b")).name("b").after("a");
    schedule.run(&mut world);
}

#[test]
#[should_panic(expected = "Conflicting access")]
fn parameters_with_conflicting_access_are_rejected() {
    fn aliasing(_writer: Query<&mut Position>, _reader: Query<&Position>) {}
    let mut world = World::new();
    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, aliasing);
    schedule.run(&mut world);
}

#[test]
#[should_panic(expected = "Conflicting access")]
fn a_resource_borrowed_twice_mutably_is_rejected() {
    fn aliasing(_a: ResMut<Log>, _b: Res<Log>) {}
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, aliasing);
    schedule.run(&mut world);
}

#[test]
#[should_panic(expected = "Log")]
fn a_missing_resource_is_reported_by_name() {
    let mut world = World::new();
    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, step("x"));
    schedule.run(&mut world);
}

#[test]
fn changed_filters_compare_with_the_systems_previous_run() {
    #[derive(Default)]
    struct Seen(Vec<Entity>);

    let mut world = World::new();
    world.insert_resource(Seen::default());
    let a = world.spawn((Position(0.0),));
    let b = world.spawn((Position(0.0),));
    let mut schedule = Schedule::new();
    schedule.add(
        Stage::Update,
        |mut changed: Query<(Entity, Changed<Position>)>, mut seen: ResMut<Seen>| {
            seen.0.clear();
            changed.for_each(|(e, ())| seen.0.push(e));
            seen.0.sort();
        },
    );

    schedule.run(&mut world);
    assert_eq!(world.resource::<Seen>().unwrap().0, vec![a, b]);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Seen>().unwrap().0, Vec::<Entity>::new());
    world.get_mut::<Position>(b).unwrap().0 = 1.0;
    schedule.run(&mut world);
    assert_eq!(world.resource::<Seen>().unwrap().0, vec![b]);
}

#[test]
fn local_state_belongs_to_the_system_and_persists() {
    let mut world = World::new();
    world.insert_resource(Log::default());
    let mut schedule = Schedule::new();
    schedule.add(
        Stage::Update,
        |mut runs: Local<u32>, mut log: ResMut<Log>| {
            *runs += 1;
            if *runs == 3 {
                log.0.push("third");
            }
        },
    );
    for _ in 0..4 {
        schedule.run(&mut world);
    }
    assert_eq!(world.resource::<Log>().unwrap().0, vec!["third"]);
}

#[test]
fn two_queries_of_one_system_iterate_together() {
    fn drift(mut movers: Query<(Entity, &mut Position)>, mut pushers: Query<(Entity, &Velocity)>) {
        for (mover, position) in movers.iter_mut() {
            for (pusher, velocity) in pushers.iter_mut() {
                if mover != pusher {
                    position.0 += velocity.0;
                }
            }
        }
    }
    let mut world = World::new();
    let a = world.spawn((Position(0.0), Velocity(1.0)));
    let b = world.spawn((Position(0.0), Velocity(10.0)));
    let c = world.spawn((Position(0.0),));
    let mut schedule = Schedule::new();
    schedule.add(Stage::Update, drift);
    schedule.run(&mut world);
    assert_eq!(world.get::<Position>(a), Some(&Position(10.0)));
    assert_eq!(world.get::<Position>(b), Some(&Position(1.0)));
    assert_eq!(world.get::<Position>(c), Some(&Position(11.0)));
}

#[test]
fn get_mut_reaches_one_entity_and_marks_only_its_row() {
    #[derive(Default)]
    struct Seen(Vec<Entity>);
    struct Victim(Entity);

    let mut world = World::new();
    world.insert_resource(Seen::default());
    let a = world.spawn((Position(0.0),));
    let b = world.spawn((Position(0.0),));
    world.insert_resource(Victim(b));
    let mut schedule = Schedule::new();
    schedule
        .add(
            Stage::Update,
            |mut positions: Query<&mut Position>, victim: Res<Victim>| {
                if let Some(p) = positions.get_mut(victim.0) {
                    p.0 -= 1.0;
                }
            },
        )
        .name("hit");
    schedule
        .add(
            Stage::Update,
            |mut changed: Query<(Entity, Changed<Position>)>, mut seen: ResMut<Seen>| {
                seen.0.clear();
                changed.for_each(|(e, ())| seen.0.push(e));
                seen.0.sort();
            },
        )
        .after("hit");

    schedule.run(&mut world);
    assert_eq!(world.resource::<Seen>().unwrap().0, vec![a, b]);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Seen>().unwrap().0, vec![b]);
    assert_eq!(world.get::<Position>(b), Some(&Position(-2.0)));
    assert_eq!(world.get::<Position>(a), Some(&Position(0.0)));
    world.despawn(b);
    schedule.run(&mut world);
    assert_eq!(world.resource::<Seen>().unwrap().0, Vec::<Entity>::new());
}
