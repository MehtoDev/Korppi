//! Integration tests for 2D and 3D Debug Drawing utilities in korppi_render.

use korppi_math::{Quat, Vec2, Vec3};
use korppi_render::render2d::{DebugDraw2d, QuadBatch2d};
use korppi_render::DebugDraw3d;

#[test]
fn debug_draw_2d_compiles_primitives_to_quad_batch() {
    let mut batch = QuadBatch2d::new(1024);
    let mut debug = DebugDraw2d::new(&mut batch);

    // 1. Line
    debug.draw_line(Vec2::ZERO, Vec2::new(10.0, 10.0), 0.1, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(debug.batch.quad_count(), 1);

    // 2. Circle wireframe (16 segments = 16 line quads)
    debug.draw_circle(Vec2::new(5.0, 5.0), 2.0, 16, 0.05, [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(debug.batch.quad_count(), 1 + 16);

    // 3. Capsule wireframe (2 lines + 2 * 8 arc segments = 18 quads)
    debug.draw_capsule(
        Vec2::new(0.0, 0.0),
        Vec2::new(0.0, 4.0),
        1.0,
        16,
        0.05,
        [0.0, 0.0, 1.0, 1.0],
    );
    assert!(debug.batch.quad_count() > 17);

    // 4. Polygon (triangle = 3 edge quads)
    let poly = [
        Vec2::new(0.0, 0.0),
        Vec2::new(2.0, 0.0),
        Vec2::new(1.0, 2.0),
    ];
    let prev = debug.batch.quad_count();
    debug.draw_polygon(&poly, 0.05, [1.0, 1.0, 0.0, 1.0]);
    assert_eq!(debug.batch.quad_count(), prev + 3);

    // 5. Contact point cross (2 quads)
    let prev = debug.batch.quad_count();
    debug.draw_point(Vec2::new(1.0, 1.0), 0.2, [1.0, 0.0, 1.0, 1.0]);
    assert_eq!(debug.batch.quad_count(), prev + 1);

    // 6. Normal arrow (1 line quad)
    let prev = debug.batch.quad_count();
    debug.draw_normal(
        Vec2::new(1.0, 1.0),
        Vec2::Y,
        1.0,
        0.05,
        [0.0, 1.0, 1.0, 1.0],
    );
    assert_eq!(debug.batch.quad_count(), prev + 1);
}

#[test]
fn debug_draw_3d_compiles_wireframes_to_line_buffers() {
    let mut debug = DebugDraw3d::new();

    // 1. Line
    debug.draw_line(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(debug.line_count(), 1);
    assert_eq!(debug.vertices().len(), 2);
    assert_eq!(debug.indices().len(), 2);

    // 2. Wireframe box (12 edges = 12 lines)
    debug.draw_box(
        Vec3::ZERO,
        Vec3::splat(1.0),
        Quat::IDENTITY,
        [0.0, 1.0, 0.0, 1.0],
    );
    assert_eq!(debug.line_count(), 1 + 12);

    // 3. Wireframe sphere (3 great circles, 16 segments each = 48 lines)
    debug.draw_sphere(Vec3::ZERO, 1.0, 16, [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(debug.line_count(), 1 + 12 + 48);

    // 4. Contact point cross (3 orthogonal axes = 3 lines)
    let prev = debug.line_count();
    debug.draw_cross(Vec3::new(1.0, 2.0, 3.0), 0.2, [1.0, 1.0, 0.0, 1.0]);
    assert_eq!(debug.line_count(), prev + 3);

    // 5. Normal vector (1 line)
    let prev = debug.line_count();
    debug.draw_normal(Vec3::ZERO, Vec3::Y, 2.0, [1.0, 0.0, 1.0, 1.0]);
    assert_eq!(debug.line_count(), prev + 1);
}
