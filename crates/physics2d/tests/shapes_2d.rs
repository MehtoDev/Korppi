//! Integration tests for 2D collision shapes and mass properties.

use korppi_math::{Rot2, Transform2d, Vec2};
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Segment2d, Shape2d};
use std::f32::consts::{FRAC_PI_2, PI};

#[test]
fn circle_computes_bounds_mass_and_support() {
    let circle = Circle2d::new(Vec2::ZERO, 2.0);
    assert_eq!(circle.radius, 2.0);

    // Support point in direction (1, 0)
    let sup_x = circle.support(Vec2::new(1.0, 0.0));
    assert!((sup_x - Vec2::new(2.0, 0.0)).length() < 1e-5);

    // Support point in direction (0, 1)
    let sup_y = circle.support(Vec2::new(0.0, 1.0));
    assert!((sup_y - Vec2::new(0.0, 2.0)).length() < 1e-5);

    // AABB at identity
    let aabb_ident = circle.compute_aabb(&Transform2d::IDENTITY);
    assert_eq!(aabb_ident.min, Vec2::new(-2.0, -2.0));
    assert_eq!(aabb_ident.max, Vec2::new(2.0, 2.0));

    // AABB under translation
    let xf = Transform2d::from_translation(Vec2::new(10.0, 20.0));
    let aabb_xf = circle.compute_aabb(&xf);
    assert_eq!(aabb_xf.min, Vec2::new(8.0, 18.0));
    assert_eq!(aabb_xf.max, Vec2::new(12.0, 22.0));

    // Mass properties: density = 1.0
    // Area = pi * r^2 = 4 * pi
    // Inertia = 0.5 * m * r^2 = 0.5 * (4 * pi) * 4 = 8 * pi
    let mass_data = circle.compute_mass_properties(1.0);
    assert!((mass_data.mass - 4.0 * PI).abs() < 1e-4);
    assert!((mass_data.rotational_inertia - 8.0 * PI).abs() < 1e-4);
    assert_eq!(mass_data.center, Vec2::ZERO);
}

#[test]
fn polygon_box_computes_bounds_mass_and_support() {
    // 2x4 box centered at origin: half_extents = (1.0, 2.0)
    let poly = Polygon2d::from_box(1.0, 2.0);
    assert_eq!(poly.vertices().len(), 4);

    // Support in (1, 1) direction should yield corner (1, 2)
    let sup = poly.support(Vec2::new(1.0, 1.0));
    assert_eq!(sup, Vec2::new(1.0, 2.0));

    // Support in (-1, -1) direction should yield corner (-1, -2)
    let sup_neg = poly.support(Vec2::new(-1.0, -1.0));
    assert_eq!(sup_neg, Vec2::new(-1.0, -2.0));

    // AABB under 90 degree rotation
    let rot90 = Transform2d::from_rotation(Rot2::from_angle(FRAC_PI_2));
    let aabb_rot = poly.compute_aabb(&rot90);
    // (1, 2) rotated 90 deg -> (-2, 1); width becomes 4 ([-2, 2]), height becomes 2 ([-1, 1])
    assert!((aabb_rot.min - Vec2::new(-2.0, -1.0)).length() < 1e-4);
    assert!((aabb_rot.max - Vec2::new(2.0, 1.0)).length() < 1e-4);

    // Mass properties: density = 1.0
    // Area = 2 * 4 = 8.0
    // Inertia = (1/12) * m * (w^2 + h^2) = (1/12) * 8 * (4 + 16) = 160 / 12 = 13.3333
    let mass_data = poly.compute_mass_properties(1.0);
    assert!((mass_data.mass - 8.0).abs() < 1e-4);
    assert!((mass_data.rotational_inertia - (160.0 / 12.0)).abs() < 1e-4);
    assert_eq!(mass_data.center, Vec2::ZERO);
}

#[test]
fn capsule_computes_bounds_and_support() {
    let capsule = Capsule2d::new(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0), 0.5);

    // Support in +X direction: p2 (1, 0) + radius * (1, 0) = (1.5, 0)
    let sup_x = capsule.support(Vec2::new(1.0, 0.0));
    assert!((sup_x - Vec2::new(1.5, 0.0)).length() < 1e-5);

    // Support in -X direction: p1 (-1, 0) + radius * (-1, 0) = (-1.5, 0)
    let sup_neg_x = capsule.support(Vec2::new(-1.0, 0.0));
    assert!((sup_neg_x - Vec2::new(-1.5, 0.0)).length() < 1e-5);

    // AABB at identity
    let aabb = capsule.compute_aabb(&Transform2d::IDENTITY);
    assert_eq!(aabb.min, Vec2::new(-1.5, -0.5));
    assert_eq!(aabb.max, Vec2::new(1.5, 0.5));
}

#[test]
fn shape_enum_computes_aabb_and_mass() {
    let circle_shape = Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0));
    let box_shape = Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5));
    let seg_shape = Shape2d::Segment(Segment2d::new(Vec2::new(-1.0, 0.0), Vec2::new(1.0, 0.0)));

    let aabb_c = circle_shape.compute_aabb(&Transform2d::IDENTITY);
    assert_eq!(aabb_c.min, Vec2::new(-1.0, -1.0));

    let mass_b = box_shape.compute_mass_properties(1.0);
    assert!((mass_b.mass - 1.0).abs() < 1e-4);

    let aabb_s = seg_shape.compute_aabb(&Transform2d::IDENTITY);
    assert_eq!(aabb_s.min, Vec2::new(-1.0, 0.0));
    assert_eq!(aabb_s.max, Vec2::new(1.0, 0.0));
}
