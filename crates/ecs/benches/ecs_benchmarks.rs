//! Criterion benchmarks of the ECS's hot paths, for tracking changes between commits.
//!
//! The comparison with Flecs, which the FIX-5 exit criterion uses, is the audit harness in
//! `docs/audits/probes/ecs_flecs_bench`; these benchmarks have no reference engine.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use korppi_ecs::{ChildOf, Component, Entity, Query, Res, Schedule, Stage, World};
use korppi_math::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position(Vec3);
impl Component for Position {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Velocity(Vec3);
impl Component for Velocity {}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Health(f32);
impl Component for Health {}

struct Dt(f32);

const N: usize = 100_000;

fn moving_world() -> World {
    let mut world = World::new();
    for i in 0..N {
        world.spawn((
            Position(Vec3::new(i as f32, 0.0, 0.0)),
            Velocity(Vec3::new(1.0, 0.0, 0.0)),
        ));
    }
    world
}

fn queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_100k");
    let mut world = moving_world();
    group.bench_function("for_loop", |b| {
        b.iter(|| {
            for (pos, vel) in world.query_mut::<(&mut Position, &Velocity)>() {
                pos.0 += black_box(vel.0);
            }
        });
    });
    group.bench_function("for_each", |b| {
        b.iter(|| {
            world
                .query_mut::<(&mut Position, &Velocity)>()
                .for_each(|(pos, vel)| pos.0 += black_box(vel.0));
        });
    });
    let mut cached = world.query_state::<(&mut Position, &Velocity)>();
    group.bench_function("cached_for_each", |b| {
        b.iter(|| cached.for_each(&mut world, |(pos, vel)| pos.0 += black_box(vel.0)));
    });
    let mut schedule = Schedule::new();
    world.insert_resource(Dt(0.016));
    schedule.add(
        Stage::Update,
        |mut moving: Query<(&mut Position, &Velocity)>, dt: Res<Dt>| {
            moving.for_each(|(pos, vel)| pos.0 += vel.0 * dt.0);
        },
    );
    group.bench_function("parameter_system", |b| b.iter(|| schedule.run(&mut world)));
    group.finish();
}

fn structural_changes(c: &mut Criterion) {
    let mut group = c.benchmark_group("structural_10k");
    group.bench_function("spawn_despawn", |b| {
        let mut world = moving_world();
        let mut spawned = Vec::with_capacity(10_000);
        b.iter(|| {
            for i in 0..10_000 {
                spawned.push(world.spawn((Position(Vec3::splat(i as f32)), Velocity(Vec3::ZERO))));
            }
            for e in spawned.drain(..) {
                world.despawn(e);
            }
        });
    });
    group.bench_function("add_remove", |b| {
        let mut world = moving_world();
        let entities: Vec<Entity> = world.entities().take(10_000).collect();
        b.iter(|| {
            for &e in &entities {
                world.add_component(e, Health(1.0));
            }
            for &e in &entities {
                world.remove_component::<Health>(e);
            }
        });
    });
    group.bench_function("hierarchy_despawn", |b| {
        b.iter_batched(
            || {
                let mut world = World::new();
                let parents: Vec<Entity> = (0..100)
                    .map(|i| world.spawn((Position(Vec3::splat(i as f32)),)))
                    .collect();
                for i in 0..10_000 {
                    let child = world.spawn((Position(Vec3::splat(i as f32)),));
                    world.add_pair::<ChildOf>(child, parents[i % parents.len()]);
                }
                (world, parents)
            },
            |(mut world, parents)| {
                for parent in parents {
                    world.despawn(parent);
                }
                world
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(benches, queries, structural_changes);
criterion_main!(benches);
