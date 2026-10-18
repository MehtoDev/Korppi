//! Integration tests for 2D narrowphase collision detection and manifold generation.

use korppi_math::{Transform2d, Vec2};
use korppi_physics2d::narrowphase::{collide_shapes, TimeOfImpact2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};

#[test]
fn circle_to_circle_collision_generates_contact_manifold() {
    let circle_a = Circle2d::new(Vec2::ZERO, 1.0);
    let circle_b = Circle2d::new(Vec2::ZERO, 1.0);

    let xf_a = Transform2d::from_translation(Vec2::new(0.0, 0.0));
    // Placed 1.5 units away along X -> overlap of 0.5 units (1.0 + 1.0 - 1.5 = 0.5)
    let xf_b = Transform2d::from_translation(Vec2::new(1.5, 0.0));

    let shape_a = Shape2d::Circle(circle_a);
    let shape_b = Shape2d::Circle(circle_b);

    let manifold = collide_shapes(&shape_a, &xf_a, &shape_b, &xf_b);
    assert!(manifold.is_some());
    let m = manifold.unwrap();

    assert_eq!(m.point_count, 1);
    // Normal should point from A to B: (+1, 0)
    assert!((m.normal - Vec2::new(1.0, 0.0)).length() < 1e-4);
    // Separation is negative for penetration (-0.5)
    assert!((m.points[0].separation - (-0.5)).abs() < 1e-4);
    // Contact point at x ~ 1.0
    assert!((m.points[0].point.x - 1.0).abs() < 0.3);

    // Separated circles (3.0 units away)
    let xf_far = Transform2d::from_translation(Vec2::new(3.0, 0.0));
    assert!(collide_shapes(&shape_a, &xf_a, &shape_b, &xf_far).is_none());
}

#[test]
fn box_to_box_collision_generates_two_contact_points() {
    // Box A: ground box 10x2 (half_extents 5, 1) at y = 0
    let box_a = Shape2d::Polygon(Polygon2d::from_box(5.0, 1.0));
    let xf_a = Transform2d::from_translation(Vec2::new(0.0, 0.0));

    // Box B: falling box 2x2 (half_extents 1, 1) at y = 1.9 (penetrating top by 0.1)
    let box_b = Shape2d::Polygon(Polygon2d::from_box(1.0, 1.0));
    let xf_b = Transform2d::from_translation(Vec2::new(0.0, 1.9));

    let manifold = collide_shapes(&box_a, &xf_a, &box_b, &xf_b);
    assert!(manifold.is_some());
    let m = manifold.unwrap();

    // Box resting on flat box should generate a 2-point manifold
    assert_eq!(m.point_count, 2);
    // Normal pointing from A to B (+Y)
    assert!((m.normal - Vec2::new(0.0, 1.0)).length() < 1e-4);
    // Separation should be approx -0.1
    assert!((m.points[0].separation - (-0.1)).abs() < 1e-3);
    assert!((m.points[1].separation - (-0.1)).abs() < 1e-3);
}

#[test]
fn polygon_to_circle_collision_generates_manifold() {
    // Ground box
    let box_shape = Shape2d::Polygon(Polygon2d::from_box(5.0, 1.0));
    let xf_box = Transform2d::from_translation(Vec2::new(0.0, 0.0));

    // Circle radius 1.0 at y = 1.8 -> penetrates top surface (y = 1.0) by 0.2
    let circle_shape = Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0));
    let xf_circle = Transform2d::from_translation(Vec2::new(0.0, 1.8));

    let manifold = collide_shapes(&box_shape, &xf_box, &circle_shape, &xf_circle);
    assert!(manifold.is_some());
    let m = manifold.unwrap();

    assert_eq!(m.point_count, 1);
    assert!((m.normal - Vec2::new(0.0, 1.0)).length() < 1e-4);
    assert!((m.points[0].separation - (-0.2)).abs() < 1e-3);
}

#[test]
fn time_of_impact_prevents_tunneling() {
    // High-speed circle moving towards static box
    let circle = Circle2d::new(Vec2::ZERO, 0.5);
    let xf_circle_start = Transform2d::from_translation(Vec2::new(0.0, 10.0));
    // Fast velocity downwards: 20 units in one step
    let velocity_circle = Vec2::new(0.0, -20.0);

    let ground_box = Polygon2d::from_box(5.0, 1.0);
    let xf_box = Transform2d::from_translation(Vec2::new(0.0, 0.0));

    let toi = TimeOfImpact2d::calculate(
        &Shape2d::Circle(circle),
        &xf_circle_start,
        velocity_circle,
        &Shape2d::Polygon(ground_box),
        &xf_box,
        Vec2::ZERO,
    );

    assert!(toi.is_some());
    let t = toi.unwrap();
    // Distance to cover before touching top surface (y = 1.0):
    // From center y = 10.0, bottom of circle is y = 9.5. Top of box is y = 1.0.
    // Distance = 8.5. Velocity = 20.0. Time fraction = 8.5 / 20.0 = 0.425.
    assert!((t - 0.425).abs() < 0.05);
}
