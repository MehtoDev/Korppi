//! Integration tests for GJK / EPA narrowphase collision detection on general convex shapes:
//! Capsule-Sphere, Capsule-Capsule, Capsule-Box, and ConvexHull.

use korppi_math::{Quat, Vec3};
use korppi_physics::narrowphase::{collide_shapes, CollisionBody};
use korppi_physics::{Collider, Shape};

#[test]
fn capsule_sphere_collision_detection() {
    let cap_collider = Collider {
        shape: Shape::Capsule {
            radius: 0.5,
            half_height: 1.0,
        },
        friction: 0.4,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let sphere_collider = Collider {
        shape: Shape::Sphere { radius: 0.5 },
        friction: 0.4,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    let body_cap = CollisionBody::new(1, Vec3::ZERO, Quat::IDENTITY, &cap_collider);
    // Sphere at (0.8, 0.5, 0.0) -> distance to spine is 0.8, radius sum is 0.5 + 0.5 = 1.0 -> penetrates by 0.2
    let body_sph = CollisionBody::new(
        2,
        Vec3::new(0.8, 0.5, 0.0),
        Quat::IDENTITY,
        &sphere_collider,
    );

    let manifold = collide_shapes(body_cap, body_sph).expect("Capsule and sphere must collide");

    assert!(!manifold.points.is_empty(), "Must have contact points");
    let pt = &manifold.points[0];
    assert!(
        pt.normal.x > 0.9,
        "Normal from cap to sphere must point along +X, got {:?}",
        pt.normal
    );
    assert!(
        pt.separation < 0.0,
        "Separation must be negative for penetrating shapes, got {}",
        pt.separation
    );
    assert!(
        (pt.separation - (-0.2)).abs() < 0.05,
        "Separation should be ~ -0.2m, got {}",
        pt.separation
    );
}

#[test]
fn capsule_capsule_perpendicular_collision_detection() {
    let cap1 = Collider {
        shape: Shape::Capsule {
            radius: 0.4,
            half_height: 1.0,
        },
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let cap2 = Collider {
        shape: Shape::Capsule {
            radius: 0.4,
            half_height: 1.0,
        },
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    // Capsule 1 along Y at origin
    let body1 = CollisionBody::new(1, Vec3::ZERO, Quat::IDENTITY, &cap1);
    // Capsule 2 rotated 90 deg around Z (along X axis), positioned at (0.0, 0.0, 0.6)
    let rot_z90 = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
    let body2 = CollisionBody::new(2, Vec3::new(0.0, 0.0, 0.6), rot_z90, &cap2);

    // Spine distance is 0.6, radius sum is 0.8 -> penetration 0.2
    let manifold = collide_shapes(body1, body2).expect("Crossing capsules must collide");

    assert!(!manifold.points.is_empty());
    let pt = &manifold.points[0];
    assert!(
        pt.normal.z > 0.9,
        "Normal must point along +Z, got {:?}",
        pt.normal
    );
    assert!(pt.separation < 0.0);
    assert!(
        (pt.separation - (-0.2)).abs() < 0.05,
        "Separation should be ~ -0.2m, got {}",
        pt.separation
    );
}

#[test]
fn capsule_box_collision_detection() {
    let cap = Collider {
        shape: Shape::Capsule {
            radius: 0.3,
            half_height: 0.8,
        },
        friction: 0.5,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let box_col = Collider {
        shape: Shape::Box {
            half_extents: Vec3::new(2.0, 0.5, 2.0),
        },
        friction: 0.5,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    // Box top surface at y = 0.5.
    let body_box = CollisionBody::new(1, Vec3::ZERO, Quat::IDENTITY, &box_col);
    // Capsule centered at y = 1.5, bottom spherical cap extends down to y = 1.5 - 0.8 - 0.3 = 0.4 (penetrates box by 0.1)
    let body_cap = CollisionBody::new(2, Vec3::new(0.0, 1.5, 0.0), Quat::IDENTITY, &cap);

    let manifold = collide_shapes(body_box, body_cap).expect("Capsule resting on box must collide");

    assert!(!manifold.points.is_empty());
    let pt = &manifold.points[0];
    assert!(
        pt.normal.y > 0.9,
        "Normal from box to capsule must point along +Y, got {:?}",
        pt.normal
    );
    assert!(
        pt.separation <= 0.0,
        "Separation must be penetrating, got {}",
        pt.separation
    );
}

#[test]
fn convex_hull_box_collision_detection() {
    // Octahedron convex hull
    let octahedron_vertices = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    let hull = Collider {
        shape: Shape::convex_hull(&octahedron_vertices).unwrap(),
        friction: 0.4,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let box_col = Collider {
        shape: Shape::Box {
            half_extents: Vec3::splat(1.0),
        },
        friction: 0.4,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    let body_box = CollisionBody::new(1, Vec3::ZERO, Quat::IDENTITY, &box_col);
    // Octahedron positioned at (1.8, 0.0, 0.0) -> left tip at x = 0.8 penetrates box right face at x = 1.0 by 0.2
    let body_hull = CollisionBody::new(2, Vec3::new(1.8, 0.0, 0.0), Quat::IDENTITY, &hull);

    let manifold = collide_shapes(body_box, body_hull).expect("Convex hull and box must collide");

    assert!(!manifold.points.is_empty());
    let pt = &manifold.points[0];
    assert!(
        pt.normal.x > 0.9,
        "Normal from box to hull must point along +X, got {:?}",
        pt.normal
    );
    assert!(
        pt.separation < 0.0,
        "Separation must be negative, got {}",
        pt.separation
    );
}

#[test]
fn separated_capsule_and_hull_report_no_collision() {
    let cap = Collider {
        shape: Shape::Capsule {
            radius: 0.5,
            half_height: 1.0,
        },
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };
    let hull = Collider {
        // A hull needs volume: three points and a fourth off their plane.
        shape: Shape::convex_hull(&[
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, -0.5),
        ])
        .unwrap(),
        friction: 0.3,
        restitution: 0.0,
        is_sensor: false,
        ..Default::default()
    };

    let body_cap = CollisionBody::new(1, Vec3::ZERO, Quat::IDENTITY, &cap);
    let body_hull = CollisionBody::new(2, Vec3::new(5.0, 5.0, 5.0), Quat::IDENTITY, &hull);

    assert!(
        collide_shapes(body_cap, body_hull).is_none(),
        "Separated shapes must not collide"
    );
}
