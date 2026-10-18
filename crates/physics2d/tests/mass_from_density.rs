//! A body's mass, centre of mass and rotational inertia come from its collider (P2D-04, P2D-20).
//!
//! Before FIX-2.2 every dynamic body was 1 kg with I = 1 kg·m² and its centre of mass at its
//! origin, whatever its shape. As in Box2D v3.2, a collider's density now gives the body its mass
//! properties, and the body moves about that centre of mass. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_2_mass.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};
use std::f32::consts::FRAC_PI_2;

const DT: f32 = 1.0 / 60.0;

/// A dynamic body at the origin carrying `collider`.
fn body_with(world: &mut PhysicsWorld2d, collider: Collider2d) -> BodyId2d {
    let body = world.create_body(RigidBody2d::new(BodyType::Dynamic));
    world.create_collider(body, collider);
    body
}

fn assert_close(what: &str, value: f32, expected: f32, tolerance: f32) {
    assert!(
        (value - expected).abs() <= tolerance,
        "{what}: {value:.6}, Box2D v3.2 gives {expected:.6}"
    );
}

#[test]
fn collider_density_sets_mass_centre_and_inertia() {
    let polygon = |points: &[(f32, f32)], radius| {
        let points: Vec<Vec2> = points.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
        Shape2d::Polygon(Polygon2d::from_vertices(&points, radius).unwrap())
    };
    // (name, shape, density, Box2D mass, centre, inertia about the centre)
    let cases = [
        (
            "circle",
            Shape2d::Circle(Circle2d::new(Vec2::new(0.3, -0.2), 0.5)),
            2.0,
            FRAC_PI_2, // 2 * pi * 0.5^2; Box2D: 1.570796
            (0.3, -0.2),
            0.196_350,
        ),
        (
            "capsule",
            Shape2d::Capsule(Capsule2d::new(
                Vec2::new(-0.4, 0.1),
                Vec2::new(0.6, 0.3),
                0.25,
            )),
            2.0,
            1.412_503,
            (0.1, 0.2),
            0.266_494,
        ),
        (
            "box",
            Shape2d::Polygon(Polygon2d::from_box(0.6, 0.4)),
            2.0,
            1.92,
            (0.0, 0.0),
            0.332_800,
        ),
        (
            "offset quad",
            polygon(&[(0.5, 0.5), (1.5, 0.5), (1.5, 1.2), (0.5, 1.0)], 0.0),
            2.0,
            1.2,
            (1.027_778, 0.802_778),
            0.137_065,
        ),
        (
            "rounded box",
            polygon(&[(-0.5, -0.3), (0.5, -0.3), (0.5, 0.3), (-0.5, 0.3)], 0.1),
            2.0,
            1.918_748,
            (0.0, 0.0),
            0.332_383,
        ),
        (
            "hexagon",
            polygon(
                &[
                    (-0.5, -0.3),
                    (0.5, -0.3),
                    (0.6, 0.1),
                    (0.3, 0.4),
                    (-0.3, 0.4),
                    (-0.6, 0.1),
                ],
                0.0,
            ),
            1.0,
            0.71,
            (0.0, 0.030_516),
            0.090_705,
        ),
    ];
    for (name, shape, density, mass, (cx, cy), inertia) in cases {
        let mut world = PhysicsWorld2d::new();
        let id = body_with(&mut world, Collider2d::new(shape).with_density(density));
        let body = world.body(id).unwrap();
        assert_close(&format!("{name} mass"), body.mass, mass, 1e-4 * mass);
        assert_close(&format!("{name} centre x"), body.local_center.x, cx, 1e-5);
        assert_close(&format!("{name} centre y"), body.local_center.y, cy, 1e-5);
        assert_close(
            &format!("{name} inertia"),
            body.rotational_inertia,
            inertia,
            1e-4 * inertia,
        );
        assert_close(
            &format!("{name} inverse mass"),
            body.inv_mass,
            1.0 / mass,
            1e-3,
        );
    }
}

#[test]
fn static_and_kinematic_bodies_have_no_mass() {
    for body_type in [BodyType::Static, BodyType::Kinematic] {
        let mut world = PhysicsWorld2d::new();
        let id = world.create_body(RigidBody2d::new(body_type));
        world.create_collider(
            id,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.6, 0.4))).with_density(2.0),
        );
        let body = world.body(id).unwrap();
        assert_eq!(
            (
                body.mass,
                body.inv_mass,
                body.rotational_inertia,
                body.inv_inertia
            ),
            (0.0, 0.0, 0.0, 0.0),
            "{body_type:?}"
        );
    }
}

#[test]
fn bodies_without_mass_giving_colliders_are_massless() {
    // Box2D v3.2: no collider, or a zero-density one, leaves a dynamic body massless; gravity does
    // not act on it and it keeps its velocity: x = 1, y = 0 after 1 s.
    for zero_density in [false, true] {
        let mut world = PhysicsWorld2d::new();
        let id = world.create_body(
            RigidBody2d::new(BodyType::Dynamic).with_linear_velocity(Vec2::new(1.0, 0.0)),
        );
        if zero_density {
            world.create_collider(
                id,
                Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))).with_density(0.0),
            );
        }
        for _ in 0..60 {
            world.step(DT);
        }
        let body = world.body(id).unwrap();
        assert_eq!((body.mass, body.rotational_inertia), (0.0, 0.0));
        assert_close("x after 1 s", body.position.x, 1.0, 1e-4);
        assert_close("y after 1 s", body.position.y, 0.0, 1e-4);
    }
}

#[test]
fn body_spins_about_its_centre_of_mass() {
    // A circle centred 1 m from its body's origin, spinning at 2 rad/s without gravity: the centre
    // of mass stays put and the origin circles it (Box2D: origin at (1.416105, -0.909317)).
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let id = body_with(
        &mut world,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::new(1.0, 0.0), 0.5))),
    );
    world.body_mut(id).unwrap().angular_velocity = 2.0;
    for _ in 0..60 {
        world.step(DT);
    }
    let body = world.body(id).unwrap();
    let centre = body.world_center();
    assert!(
        (centre - Vec2::new(1.0, 0.0)).length() < 1e-4,
        "the centre of mass moved to {centre:?}"
    );
    assert_close("origin x", body.position.x, 1.416_105, 1e-3);
    assert_close("origin y", body.position.y, -0.909_317, 1e-3);
}

#[test]
fn impulses_act_about_the_centre_of_mass() {
    // The same 0.785 kg circle, centred at (1, 0). An impulse of (2, 0) through the centre gives
    // 2.546479 m/s and no spin; through (1, 0.5) it also spins the body at -10.185916 rad/s.
    for (point, spin) in [
        (Vec2::new(1.0, 0.0), 0.0),
        (Vec2::new(1.0, 0.5), -10.185_916),
    ] {
        let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
        let id = body_with(
            &mut world,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::new(1.0, 0.0), 0.5))),
        );
        let body = world.body_mut(id).unwrap();
        body.apply_impulse_at_point(Vec2::new(2.0, 0.0), point);
        assert_close("linear velocity x", body.linear_velocity.x, 2.546_479, 1e-4);
        assert_close("linear velocity y", body.linear_velocity.y, 0.0, 1e-6);
        assert_close("angular velocity", body.angular_velocity, spin, 1e-3);
    }
}

#[test]
fn box_pendulum_swings_with_its_shape_inertia() {
    // A 1 x 0.5 m box pinned at its top-left corner falls from horizontal. Its swing depends on
    // the box's own inertia (1/12 m (w^2 + h^2) about its centre); Box2D's angles by time:
    let expected = [
        (15, -0.360_781),
        (30, -1.233_791),
        (45, -2.015_454),
        (60, -2.189_875),
    ];
    let mut world = PhysicsWorld2d::new();
    let pivot =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 10.0)));
    let id =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.5, 9.75)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.25))),
    );
    world.create_joint(
        pivot,
        id,
        Joint2d::Revolute(RevoluteJoint2d::new(
            Vec2::ZERO,
            Vec2::new(-0.5, 0.25),
            0.0,
            0.0,
        )),
    );
    let mut step = 0;
    for (at, angle) in expected {
        while step < at {
            world.step(DT);
            step += 1;
        }
        let measured = world.body(id).unwrap().rotation.angle();
        assert_close(&format!("angle after {at} steps"), measured, angle, 0.02);
    }
}

#[test]
fn mass_set_after_the_collider_wins() {
    let mut world = PhysicsWorld2d::new();
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_mass(5.0, 2.0));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    let body = world.body(id).unwrap();
    assert_close("mass after the collider", body.mass, 1.0, 1e-6);
    assert_close(
        "inertia after the collider",
        body.rotational_inertia,
        1.0 / 6.0,
        1e-6,
    );

    world.body_mut(id).unwrap().set_mass(5.0, 2.0);
    let body = world.body(id).unwrap();
    assert_eq!((body.mass, body.rotational_inertia), (5.0, 2.0));
}
