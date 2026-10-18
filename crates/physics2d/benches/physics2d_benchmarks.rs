//! Benchmarks for 2D dynamic tree queries and physics simulation stepping.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use korppi_math::{Aabb2d, Ray2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::bvh::DynamicTree2d;
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::PhysicsWorld2d;

fn bench_dynamic_tree_2d_query(c: &mut Criterion) {
    let mut tree = DynamicTree2d::new();
    for i in 0..1000 {
        let x = (i % 30) as f32 * 2.0;
        let y = (i / 30) as f32 * 2.0;
        let aabb = Aabb2d::new(Vec2::new(x, y), Vec2::new(x + 1.0, y + 1.0));
        tree.create_proxy(aabb, i as u32);
    }

    let query_box = Aabb2d::new(Vec2::new(10.0, 10.0), Vec2::new(20.0, 20.0));

    c.bench_function("dynamic_tree_2d_query_1000", |b| {
        b.iter(|| {
            let mut count = 0;
            tree.query(black_box(&query_box), |_node, _user_data| {
                count += 1;
                true
            });
            count
        });
    });
}

fn bench_dynamic_tree_2d_raycast(c: &mut Criterion) {
    let mut tree = DynamicTree2d::new();
    for i in 0..1000 {
        let x = (i % 30) as f32 * 2.0;
        let y = (i / 30) as f32 * 2.0;
        let aabb = Aabb2d::new(Vec2::new(x, y), Vec2::new(x + 1.0, y + 1.0));
        tree.create_proxy(aabb, i as u32);
    }

    let ray = Ray2d::new(Vec2::new(-5.0, 15.0), Vec2::X);

    c.bench_function("dynamic_tree_2d_raycast_1000", |b| {
        b.iter(|| {
            let mut hit_count = 0;
            tree.ray_cast(black_box(&ray), 100.0, |_node_id, _user_data, t| {
                hit_count += 1;
                t
            });
            hit_count
        });
    });
}

fn bench_physics2d_stack_step(c: &mut Criterion) {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    // Ground
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(50.0, 1.0))),
    );

    // Stack of 20 boxes
    for i in 0..20 {
        let mut box_body = RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 2.0 + (i as f32) * 1.2));
        box_body.set_mass(1.0, 0.5);
        let b = world.create_body(box_body);
        world.create_collider(
            b,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
        );
    }

    c.bench_function("physics2d_stack_step_20", |b| {
        b.iter(|| {
            world.step(black_box(1.0 / 60.0));
        });
    });
}

criterion_group!(
    benches,
    bench_dynamic_tree_2d_query,
    bench_dynamic_tree_2d_raycast,
    bench_physics2d_stack_step
);
criterion_main!(benches);
