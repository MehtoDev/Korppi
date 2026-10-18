//! Integration tests verifying narrowphase collision detection for all 2D shape pairs.

use korppi_math::{Transform2d, Vec2};
use korppi_physics2d::narrowphase::collide_shapes;
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Segment2d, Shape2d};

#[test]
fn test_capsule_and_circle_collision() {
    let cap = Shape2d::Capsule(Capsule2d::new(
        Vec2::new(-1.0, 0.0),
        Vec2::new(1.0, 0.0),
        0.5,
    ));
    let xf_cap = Transform2d::IDENTITY;

    // Circle above the capsule interior
    let circle = Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5));
    let xf_circle = Transform2d::from_translation(Vec2::new(0.0, 0.8));

    let manifold = collide_shapes(&cap, &xf_cap, &circle, &xf_circle)
        .expect("Capsule and circle should collide");
    assert_eq!(manifold.point_count, 1);
    assert!(
        manifold.points[0].separation < 0.0,
        "Should have negative separation (overlap)"
    );
    assert!(
        manifold.normal.y > 0.9,
        "Normal should point upward from capsule to circle"
    );

    // Inverted pair
    let inv_manifold =
        collide_shapes(&circle, &xf_circle, &cap, &xf_cap).expect("Inverted pair should collide");
    assert_eq!(inv_manifold.point_count, 1);
    assert!(
        inv_manifold.normal.y < -0.9,
        "Inverted normal should point downward"
    );
}

#[test]
fn test_capsule_and_capsule_collision() {
    let cap1 = Shape2d::Capsule(Capsule2d::new(
        Vec2::new(-1.0, 0.0),
        Vec2::new(1.0, 0.0),
        0.5,
    ));
    let xf1 = Transform2d::IDENTITY;

    // Parallel capsule above cap1
    let cap2 = Shape2d::Capsule(Capsule2d::new(
        Vec2::new(-1.0, 0.0),
        Vec2::new(1.0, 0.0),
        0.5,
    ));
    let xf2 = Transform2d::from_translation(Vec2::new(0.0, 0.8));

    let manifold =
        collide_shapes(&cap1, &xf1, &cap2, &xf2).expect("Parallel capsules should collide");
    assert!(manifold.point_count >= 1);
    assert!(manifold.points[0].separation < 0.0);
    assert!(manifold.normal.y > 0.9);
}

#[test]
fn test_polygon_and_capsule_collision() {
    let poly = Shape2d::Polygon(Polygon2d::from_box(2.0, 0.5));
    let xf_poly = Transform2d::IDENTITY;

    let cap = Shape2d::Capsule(Capsule2d::new(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.0),
        0.3,
    ));
    let xf_cap = Transform2d::from_translation(Vec2::new(0.0, 0.7));

    let manifold =
        collide_shapes(&poly, &xf_poly, &cap, &xf_cap).expect("Polygon and capsule should collide");
    assert!(manifold.point_count >= 1);
    assert!(manifold.points[0].separation < 0.0);
    assert!(manifold.normal.y > 0.9);

    // Inverted pair
    let inv = collide_shapes(&cap, &xf_cap, &poly, &xf_poly).expect("Inverted pair should collide");
    assert!(inv.point_count >= 1);
    assert!(inv.normal.y < -0.9);
}

#[test]
fn test_segment_and_circle_collision() {
    let seg = Shape2d::Segment(Segment2d::new(Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0)));
    let xf_seg = Transform2d::IDENTITY;

    let circle = Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5));
    let xf_circle = Transform2d::from_translation(Vec2::new(0.0, 0.3));

    let manifold = collide_shapes(&seg, &xf_seg, &circle, &xf_circle)
        .expect("Segment and circle should collide");
    assert_eq!(manifold.point_count, 1);
    assert!(manifold.points[0].separation < 0.0);
    assert!(manifold.normal.y > 0.9);
}

#[test]
fn test_segment_and_capsule_collision() {
    let seg = Shape2d::Segment(Segment2d::new(Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0)));
    let xf_seg = Transform2d::IDENTITY;

    let cap = Shape2d::Capsule(Capsule2d::new(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.0),
        0.5,
    ));
    let xf_cap = Transform2d::from_translation(Vec2::new(0.0, 0.3));

    let manifold =
        collide_shapes(&seg, &xf_seg, &cap, &xf_cap).expect("Segment and capsule should collide");
    assert!(manifold.point_count >= 1);
    assert!(manifold.points[0].separation < 0.0);
    assert!(manifold.normal.y > 0.9);
}

#[test]
fn test_segment_and_polygon_collision() {
    let seg = Shape2d::Segment(Segment2d::new(Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0)));
    let xf_seg = Transform2d::IDENTITY;

    let poly = Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5));
    let xf_poly = Transform2d::from_translation(Vec2::new(0.0, 0.4));

    let manifold =
        collide_shapes(&seg, &xf_seg, &poly, &xf_poly).expect("Segment and polygon should collide");
    assert!(manifold.point_count >= 1);
    assert!(manifold.points[0].separation < 0.0);
    assert!(manifold.normal.y > 0.9);
}
