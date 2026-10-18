//! Contact manifolds match Box2D v3.2's (P2D-18).
//!
//! Before FIX-2.3 the narrowphase clipped separated polygons by face even when their closest
//! features were two corners, turned capsules into 1e-4 thick boxes, met crossing or parallel
//! capsules with a closest-point heuristic, and put polygon–circle contacts on the circle's
//! surface. Manifolds are now computed in shape A's frame, as in Box2D, which must not cost
//! precision far from the origin.
//!
//! The expected values are Box2D's for the same pairs and scenes, from
//! `docs/audits/probes/box2d_reference/fix_2_3_manifolds.c`. Shape A sits at the origin, so
//! the manifolds read the same in A's frame and in the world.

use korppi_math::{Rot2, Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::narrowphase::{collide_shapes, ContactManifold2d};
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Segment2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};
use std::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

fn at(x: f32, y: f32, angle: f32) -> Transform2d {
    Transform2d::from_translation_rotation(Vec2::new(x, y), Rot2::from_angle(angle))
}

fn floor_box() -> Shape2d {
    Shape2d::Polygon(Polygon2d::from_box(2.0, 0.5))
}

fn unit_box() -> Shape2d {
    Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))
}

fn rod() -> Shape2d {
    Shape2d::Capsule(Capsule2d::new(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.0),
        0.25,
    ))
}

fn circle(radius: f32) -> Shape2d {
    Shape2d::Circle(Circle2d::new(Vec2::ZERO, radius))
}

/// Asserts the manifold of `a` at the origin and `b` at `xf_b` against Box2D's: the normal and
/// each point's position, separation and feature id. `tolerance` is in meters.
fn assert_manifold(
    name: &str,
    a: &Shape2d,
    b: &Shape2d,
    xf_b: Transform2d,
    normal: (f32, f32),
    points: &[(f32, f32, f32, u16)],
    tolerance: f32,
) {
    let manifold: ContactManifold2d = collide_shapes(a, &Transform2d::IDENTITY, b, &xf_b)
        .unwrap_or_else(|| panic!("{name}: no manifold, Box2D has {} points", points.len()));
    assert_eq!(manifold.point_count, points.len(), "{name}: point count");
    let close = |value: f32, expected: f32| (value - expected).abs() <= tolerance;
    assert!(
        close(manifold.normal.x, normal.0) && close(manifold.normal.y, normal.1),
        "{name}: normal {:?}, Box2D {normal:?}",
        manifold.normal
    );
    for (i, &(x, y, separation, id)) in points.iter().enumerate() {
        let p = manifold.points[i];
        assert!(
            close(p.point.x, x) && close(p.point.y, y) && close(p.separation, separation),
            "{name}: point {i} at {:?} separated {:.6}, Box2D ({x}, {y}) separated {separation}",
            p.point,
            p.separation
        );
        assert_eq!(p.id, id, "{name}: point {i} feature id");
    }
}

#[test]
fn polygon_manifolds_match_box2d() {
    assert_manifold(
        "box on box",
        &floor_box(),
        &unit_box(),
        at(0.3, 0.995, 0.0),
        (0.0, 1.0),
        &[(0.8, 0.4975, -0.005, 513), (-0.2, 0.4975, -0.005, 768)],
        1e-5,
    );
    // Separated corner to corner: the closest features are two vertices, so the normal is the
    // diagonal, not a face normal.
    assert_manifold(
        "separated corners",
        &unit_box(),
        &unit_box(),
        at(1.01, 1.01, 0.0),
        (FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        &[(0.505, 0.505, 0.014_142, 512)],
        1e-5,
    );
    let rounded = Shape2d::Polygon(Polygon2d::rounded_box(0.4, 0.4, 0.1));
    assert_manifold(
        "box and rounded box",
        &floor_box(),
        &rounded,
        at(-0.3, 0.995, 0.0),
        (0.0, 1.0),
        &[(0.1, 0.4975, -0.005, 513), (-0.7, 0.4975, -0.005, 768)],
        1e-5,
    );
    assert!(
        collide_shapes(
            &floor_box(),
            &Transform2d::IDENTITY,
            &unit_box(),
            &at(0.2, 1.2, 0.4)
        )
        .is_none(),
        "a tilted box clear of the speculative distance must give no manifold"
    );
}

#[test]
fn capsule_manifolds_match_box2d() {
    // A capsule lying on a box touches along its length: two points.
    assert_manifold(
        "capsule on box",
        &floor_box(),
        &rod(),
        at(0.0, 0.745, 0.0),
        (0.0, 1.0),
        &[(0.5, 0.4975, -0.005, 513), (-0.5, 0.4975, -0.005, 768)],
        1e-5,
    );
    // Box2D builds rotations with approximate trigonometry, hence the looser tolerance.
    assert_manifold(
        "tilted capsule on box",
        &floor_box(),
        &rod(),
        at(0.0, 0.9, 0.3),
        (0.0, 1.0),
        &[
            (0.477_448, 0.649_235, 0.298_469, 513),
            (-0.477_448, 0.500_765, 0.001_531, 768),
        ],
        2e-3,
    );
    assert_manifold(
        "parallel capsules",
        &rod(),
        &rod(),
        at(0.2, 0.495, 0.0),
        (0.0, 1.0),
        &[(-0.3, 0.2475, -0.005, 0), (0.5, 0.2475, -0.005, 1)],
        1e-5,
    );
    assert_manifold(
        "crossed capsules",
        &rod(),
        &rod(),
        at(0.0, 0.49, FRAC_PI_2),
        (0.0, 1.0),
        &[(0.0, -0.005, -0.51, 0), (0.0, 0.495, 0.49, 1)],
        1e-5,
    );
    assert_manifold(
        "capsule and circle",
        &rod(),
        &circle(0.25),
        at(0.7, 0.1, 0.0),
        (0.894_427, 0.447_214),
        &[(0.6, 0.05, -0.276_393, 0)],
        1e-5,
    );
}

#[test]
fn circle_and_segment_manifolds_match_box2d() {
    // The contact point lies midway between the surfaces.
    assert_manifold(
        "circle on box",
        &floor_box(),
        &circle(0.5),
        at(0.3, 0.99, 0.0),
        (0.0, 1.0),
        &[(0.3, 0.495, -0.01, 0)],
        1e-5,
    );
    assert_manifold(
        "circle at box corner",
        &floor_box(),
        &circle(0.25),
        at(2.1, 0.6, 0.0),
        (FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        &[(1.961_612, 0.461_612, -0.108_579, 0)],
        1e-5,
    );
    assert_manifold(
        "circles",
        &circle(0.5),
        &circle(0.25),
        at(0.6, 0.3, 0.0),
        (0.894_427, 0.447_214),
        &[(0.411_803, 0.205_902, -0.079_18, 0)],
        1e-5,
    );
    let line = Shape2d::Segment(Segment2d::new(Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0)));
    assert_manifold(
        "segment and box",
        &line,
        &unit_box(),
        at(0.0, 0.499, 0.0),
        (0.0, 1.0),
        &[(0.5, -0.0005, -0.001, 257), (-0.5, -0.0005, -0.001, 0)],
        1e-5,
    );
    assert_manifold(
        "segment and capsule",
        &line,
        &rod(),
        at(0.1, 0.249, 0.0),
        (0.0, 1.0),
        &[(-0.4, -0.0005, -0.001, 0), (0.6, -0.0005, -0.001, 1)],
        1e-5,
    );
}

#[test]
fn swapped_pairs_point_their_normal_from_a_to_b() {
    // Box2D only has circle-after-polygon; with the circle first, the normal must still point
    // from the circle to the box.
    let manifold = collide_shapes(
        &circle(0.5),
        &at(0.3, 0.99, 0.0),
        &floor_box(),
        &Transform2d::IDENTITY,
    )
    .expect("circle and box overlap");
    assert!(
        (manifold.normal - Vec2::new(0.0, -1.0)).length() < 1e-5,
        "{:?}",
        manifold.normal
    );
    assert!((manifold.points[0].point - Vec2::new(0.3, 0.495)).length() < 1e-5);
}

#[test]
fn capsules_rest_flat_on_the_ground_and_on_each_other() {
    // Box2D v3.2 after 2 s: the dropped capsule lies at y = 0.24992, level; three stacked
    // capsules stay level within 0.001 rad at y = 0.2498, 0.7486 and 1.2479.
    let mut world = PhysicsWorld2d::new();
    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(0.6),
    );
    let mut add_capsule = |position: Vec2| {
        let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(position));
        world.create_collider(id, Collider2d::new(rod()).with_friction(0.6));
        id
    };
    let stack: Vec<BodyId2d> = (0..3)
        .map(|i| add_capsule(Vec2::new(5.0 + 0.1 * i as f32, 0.25 + 0.5 * i as f32)))
        .collect();
    let dropped = add_capsule(Vec2::new(0.0, 1.0));
    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }
    let body = world.body(dropped).unwrap();
    assert!(
        (body.position.y - 0.2499).abs() < 0.002,
        "the dropped capsule rests at y = {:.4}",
        body.position.y
    );
    assert!(
        body.rotation.angle().abs() < 0.01,
        "the dropped capsule tilted to {:.4}",
        body.rotation.angle()
    );
    for (i, &id) in stack.iter().enumerate() {
        let body = world.body(id).unwrap();
        let expected = 0.2498 + 0.4991 * i as f32;
        assert!(
            (body.position.y - expected).abs() < 0.005,
            "capsule {i} rests at y = {:.4}",
            body.position.y
        );
        assert!(
            body.rotation.angle().abs() < 0.01,
            "capsule {i} tilted to {:.4}",
            body.rotation.angle()
        );
        assert!(
            (body.position.x - (5.0 + 0.1 * i as f32)).abs() < 0.01,
            "capsule {i} slid to x = {:.4}",
            body.position.x
        );
    }
}

#[test]
fn a_box_held_on_a_ramp_far_from_the_origin_stays_put() {
    // A 0.6 m box held by friction (0.6) on a 20° ramp, built at x = 0, 10 km, 30 km and 100 km.
    // Box2D v3.2: slides under 0.0004 m, tilts at most 0.0014 rad and comes to rest at every
    // distance. With B's pose in A's frame built from rotated world positions instead of their
    // difference (`Transform2d::inv_mul`), the box at 100 km still moved at 0.022 m/s after 4 s.
    let slope = 20.0_f32.to_radians();
    for x0 in [0.0, 1.0e4, 3.0e4, 1.0e5] {
        let mut world = PhysicsWorld2d::new();
        let ramp = world.create_body(
            RigidBody2d::new(BodyType::Static)
                .with_position(Vec2::new(x0, 0.0))
                .with_rotation(Rot2::from_angle(slope)),
        );
        world.create_collider(
            ramp,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(10.0, 0.5))).with_friction(0.6),
        );
        let offset = Rot2::from_angle(slope).rotate(Vec2::new(0.0, 0.8));
        let id = world.create_body(
            RigidBody2d::new(BodyType::Dynamic)
                .with_position(Vec2::new(x0 + offset.x, offset.y))
                .with_rotation(Rot2::from_angle(slope)),
        );
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.3, 0.3))).with_friction(0.6),
        );
        let start = world.body(id).unwrap().position;
        let mut max_tilt = 0.0_f32;
        for _ in 0..240 {
            world.step(1.0 / 60.0);
            max_tilt = max_tilt.max((world.body(id).unwrap().rotation.angle() - slope).abs());
        }
        let body = world.body(id).unwrap();
        let slid = (body.position - start).length();
        assert!(slid < 0.002, "at x = {x0}: the box slid {slid:.4} m");
        assert!(
            max_tilt < 0.003,
            "at x = {x0}: the box tilted {max_tilt:.4} rad"
        );
        let speed = body.linear_velocity.length();
        assert!(
            speed < 0.001,
            "at x = {x0}: the box moves at {speed:.4} m/s"
        );
    }
}
