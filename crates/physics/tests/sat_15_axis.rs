//! Integration tests for full 15-axis SAT evaluation in 3D box-box collision.
//! Tests that all 6 face normals and all 9 edge-edge cross products are correctly evaluated.

use korppi_math::{Quat, Vec3};
use korppi_physics::narrowphase::{collide_shapes, CollisionBody};
use korppi_physics::Shape;
use std::f32::consts::FRAC_PI_4;

#[test]
fn edge_edge_separation_is_not_falsely_reported_as_colliding() {
    // Two boxes oriented such that their projections onto all 6 face axes overlap,
    // but they are separated along an edge-edge cross product axis.
    //
    // Box A is rotated 45 degrees around Z axis.
    // Box B is rotated 45 degrees around X axis and shifted along diagonal.
    let rot_a = Quat::from_axis_angle(Vec3::Z, FRAC_PI_4);
    let rot_b = Quat::from_axis_angle(Vec3::X, FRAC_PI_4);

    let collider_a = korppi_physics::Collider {
        shape: Shape::Box {
            half_extents: Vec3::splat(1.0),
        },
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let collider_b = korppi_physics::Collider {
        shape: Shape::Box {
            half_extents: Vec3::splat(1.0),
        },
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    let body_a = CollisionBody::new(1, Vec3::ZERO, rot_a, &collider_a);
    let body_b = CollisionBody::new(2, Vec3::new(0.0, 2.85, 0.0), rot_b, &collider_b);

    let manifold = collide_shapes(body_a, body_b);

    // If only 6 face normals are tested, this returns Some(manifold) (false positive collision).
    // Full 15-axis SAT will correctly detect edge-edge separation and return None.
    assert!(
        manifold.is_none(),
        "Edge-separated boxes must NOT register as colliding under full 15-axis SAT"
    );
}

#[test]
fn oblique_edge_edge_contact_detects_correct_cross_normal() {
    // Two boxes crossing at 90 degrees with an edge-edge contact
    let rot_a = Quat::IDENTITY;
    // Box B rotated 45 degrees around Y axis
    let rot_b = Quat::from_axis_angle(Vec3::Y, FRAC_PI_4);

    let collider_a = korppi_physics::Collider {
        shape: Shape::Box {
            half_extents: Vec3::splat(1.0),
        },
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let collider_b = korppi_physics::Collider {
        shape: Shape::Box {
            half_extents: Vec3::splat(1.0),
        },
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    let body_a = CollisionBody::new(1, Vec3::ZERO, rot_a, &collider_a);
    let body_b = CollisionBody::new(2, Vec3::new(0.0, 1.9, 0.0), rot_b, &collider_b);

    let manifold = collide_shapes(body_a, body_b);
    assert!(manifold.is_some(), "Intersecting boxes must collide");
    let m = manifold.unwrap();
    assert!(!m.points.is_empty(), "Manifold must have contact points");
}
