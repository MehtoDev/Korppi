//! Benchmarks for scene hierarchy traversal, transform propagation, and spawning.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use korppi_ecs::World;
use korppi_math::Vec3;
use korppi_scene::hierarchy::{attach_child, get_descendants};
use korppi_scene::name::Name;
use korppi_scene::registry::SceneRegistry;
use korppi_scene::scene::Scene;
use korppi_scene::transform::{propagate_transforms, LocalTransform};

fn bench_transform_propagation_10000(c: &mut Criterion) {
    let mut world = World::new();

    // Build 100 hierarchy trees, each with 100 descendant nodes = 10,000 entities
    let num_trees = 100;
    let depth_per_tree = 100;
    let mut roots = Vec::new();

    for t in 0..num_trees {
        let root = world.spawn((
            Name::new(format!("Root_{t}")),
            LocalTransform::from_translation(Vec3::new(t as f32, 0.0, 0.0)),
        ));
        roots.push(root);

        let mut current_parent = root;
        for d in 1..depth_per_tree {
            let child = world.spawn((
                Name::new(format!("Node_{t}_{d}")),
                LocalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            ));
            attach_child(&mut world, current_parent, child).unwrap();
            current_parent = child;
        }
    }

    // Initial propagation
    propagate_transforms(&mut world);

    c.bench_function("transform_propagation_10000_entities", |b| {
        let mut angle: f32 = 0.0;
        b.iter(|| {
            angle += 0.01;
            // Mutate roots to trigger propagation
            for &root in &roots {
                if let Some(trans) = world.get_mut::<LocalTransform>(root) {
                    trans.translation.x += angle.sin() * 0.01;
                }
            }
            propagate_transforms(black_box(&mut world));
        });
    });
}

fn bench_hierarchy_traversal_descendants_10000(c: &mut Criterion) {
    let mut world = World::new();

    let root = world.spawn((
        Name::new("MassiveRoot"),
        LocalTransform::from_translation(Vec3::ZERO),
    ));

    let mut current = root;
    for i in 1..10000 {
        let child = world.spawn((
            Name::new(format!("Node_{i}")),
            LocalTransform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        ));
        attach_child(&mut world, current, child).unwrap();
        current = child;
    }

    c.bench_function("hierarchy_get_descendants_10000", |b| {
        b.iter(|| {
            let descendants = get_descendants(black_box(&world), black_box(root));
            black_box(descendants.len())
        });
    });
}

fn bench_scene_spawning_1000(c: &mut Criterion) {
    let mut template_world = World::new();
    let registry = SceneRegistry::default();

    let mut roots = Vec::new();
    for i in 0..100 {
        let parent = template_world.spawn((
            Name::new(format!("Parent_{i}")),
            LocalTransform::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
        ));
        roots.push(parent);

        for j in 0..9 {
            let child = template_world.spawn((
                Name::new(format!("Child_{i}_{j}")),
                LocalTransform::from_translation(Vec3::new(0.0, j as f32, 0.0)),
            ));
            attach_child(&mut template_world, parent, child).unwrap();
            roots.push(child);
        }
    }

    let scene = Scene::from_world(&template_world, &roots, &registry).unwrap();

    c.bench_function("scene_spawn_and_remap_1000_entities", |b| {
        b.iter(|| {
            let mut target_world = World::new();
            let spawned = scene
                .spawn(black_box(&mut target_world), black_box(&registry))
                .unwrap();
            black_box(spawned.len())
        });
    });
}

criterion_group!(
    benches,
    bench_transform_propagation_10000,
    bench_hierarchy_traversal_descendants_10000,
    bench_scene_spawning_1000
);
criterion_main!(benches);
