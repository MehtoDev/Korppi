//! Benchmarks for SIMD mathematics and spatial query operations.
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use korppi_math::{Aabb, Mat4, Ray, Transform, Vec3};

fn bench_mat4_multiplication(c: &mut Criterion) {
    let m1 = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
    let m2 = Mat4::from_scale(Vec3::splat(2.5));

    c.bench_function("mat4_mul", |b| {
        b.iter(|| black_box(m1) * black_box(m2));
    });
}

fn bench_transform_combine(c: &mut Criterion) {
    let t1 = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
    let t2 = Transform::from_scale(Vec3::splat(1.5));

    c.bench_function("transform_combine", |b| {
        b.iter(|| black_box(t1).combine(black_box(&t2)));
    });
}

fn bench_aabb_raycast(c: &mut Criterion) {
    let aabb = Aabb::from_min_max(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, 3.0));
    let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::Z);

    c.bench_function("aabb_raycast", |b| {
        b.iter(|| black_box(&aabb).intersect_ray(black_box(&ray), 0.0, 10.0));
    });
}

criterion_group!(
    benches,
    bench_mat4_multiplication,
    bench_transform_combine,
    bench_aabb_raycast
);
criterion_main!(benches);
