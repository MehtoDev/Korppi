//! Integration tests for concave triangle mesh leaf collisions and dedicated capsule-box narrowphase.

use korppi_math::{Quat, Vec3};
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::narrowphase::{collide_shapes, CollisionBody};
use korppi_physics::shape::Shape;
use korppi_physics::world::PhysicsWorld;

#[test]
fn test_capsule_box_horizontal_face_contact_generates_two_points() {
    let box_collider = Collider::new(Shape::Box {
        half_extents: Vec3::new(2.0, 0.5, 2.0),
    });
    let box_body = CollisionBody::new(0, Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, &box_collider);

    let capsule_collider = Collider::new(Shape::Capsule {
        radius: 0.5,
        half_height: 1.0,
    });
    // Rotate capsule 90 degrees around Z so its spine axis aligns with the X axis
    let cap_rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    // Position capsule so its bottom penetrates 0.1m into the box top face (box top is at y = 0.5)
    let capsule_body = CollisionBody::new(1, Vec3::new(0.0, 0.9, 0.0), cap_rot, &capsule_collider);

    let manifold =
        collide_shapes(capsule_body, box_body).expect("Capsule and box must be colliding");

    assert_eq!(manifold.body_a, 1);
    assert_eq!(manifold.body_b, 0);
    // Normal points from capsule (body_a) to box (body_b) -> -Y
    assert!(
        manifold.points[0].normal.y < -0.9,
        "Normal should point downward towards the box, got {:?}",
        manifold.points[0].normal
    );
    assert!(
        (manifold.points[0].separation - (-0.1)).abs() < 0.05,
        "Separation depth should be ~ -0.1, got {}",
        manifold.points[0].separation
    );
    // Crucial: horizontal capsule on a flat box face must generate two contact points along its spine
    assert_eq!(
        manifold.points.len(),
        2,
        "Horizontal capsule on flat box face must produce 2 contact points"
    );
}

#[test]
fn test_capsule_box_vertical_end_contact() {
    let box_collider = Collider::new(Shape::Box {
        half_extents: Vec3::new(2.0, 0.5, 2.0),
    });
    let box_body = CollisionBody::new(0, Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, &box_collider);

    let capsule_collider = Collider::new(Shape::Capsule {
        radius: 0.5,
        half_height: 1.0,
    });
    // Vertical capsule at y = 1.9 -> bottom hemisphere center at y = 0.9, lowest point at y = 0.4
    // Top face of box at y = 0.5 -> penetration depth = 0.1
    let capsule_body = CollisionBody::new(
        1,
        Vec3::new(0.0, 1.9, 0.0),
        Quat::IDENTITY,
        &capsule_collider,
    );

    let manifold =
        collide_shapes(capsule_body, box_body).expect("Capsule and box must be colliding");

    assert_eq!(manifold.body_a, 1);
    assert_eq!(manifold.body_b, 0);
    assert!(
        manifold.points[0].normal.y < -0.9,
        "Normal should point downward towards the box"
    );
    assert!(
        (manifold.points[0].separation - (-0.1)).abs() < 0.05,
        "Separation should be ~ -0.1"
    );
    // Box3D's `b3CollideHullAndCapsule` clips the whole capsule segment to the face when the
    // closest points line up with its normal, and keeps both ends if one is within the
    // speculative distance: the bottom end touches at -0.1 and the top end is a second point
    // 1.9 m above the face, which never pushes (`fix_3_3_speculative.c` rests a standing capsule
    // on two points). So the end makes exactly one point within the speculative distance.
    let touching: Vec<f32> = manifold
        .points
        .iter()
        .map(|p| p.separation)
        .filter(|&s| s <= korppi_physics::SPECULATIVE_DISTANCE)
        .collect();
    assert_eq!(
        touching.len(),
        1,
        "Vertical capsule end on flat box face should produce 1 contact point, got {touching:?}"
    );
    assert!(
        manifold
            .points
            .iter()
            .all(|p| p.separation <= korppi_physics::SPECULATIVE_DISTANCE || p.separation >= 1.8),
        "any other point is the far end of the capsule (Box3D: 1.9 m away), {:?}",
        manifold.points
    );
}

#[test]
fn test_concave_mesh_valley_allows_entry_and_contacts_floor() {
    // Construct a concave trough (V-shaped valley):
    // Left slope: from (-2, 2) down to (0, 0)
    // Right slope: from (0, 0) up to (2, 2)
    let vertices = vec![
        Vec3::new(-2.0, 2.0, -2.0), // 0: left-top-back
        Vec3::new(-2.0, 2.0, 2.0),  // 1: left-top-front
        Vec3::new(0.0, 0.0, -2.0),  // 2: bottom-back
        Vec3::new(0.0, 0.0, 2.0),   // 3: bottom-front
        Vec3::new(2.0, 2.0, -2.0),  // 4: right-top-back
        Vec3::new(2.0, 2.0, 2.0),   // 5: right-top-front
    ];

    // Counter-clockwise seen from inside the valley, as Box3D's one-sided triangles need.
    let indices = vec![
        // Left slope (two triangles with upward-inward normals)
        [0, 1, 2],
        [1, 3, 2],
        // Right slope (two triangles with upward-inward normals)
        [2, 3, 4],
        [3, 5, 4],
    ];

    let mesh_collider =
        Collider::new(Shape::triangle_mesh(&vertices, &indices).expect("mesh builds"));
    let mesh_body = CollisionBody::new(0, Vec3::new(0.0, 0.0, 0.0), Quat::IDENTITY, &mesh_collider);

    // Sphere positioned in the upper open region of the valley: (0, 1.5, 0) with radius 0.4
    // In a convex hull representation, the top is sealed between (-2, 2) and (2, 2),
    // so a sphere at (0, 1.5, 0) would falsely collide with the ceiling.
    // In concave leaf collision, this position is completely empty space!
    let sphere_collider = Collider::new(Shape::Sphere { radius: 0.4 });
    let sphere_upper = CollisionBody::new(
        1,
        Vec3::new(0.0, 1.5, 0.0),
        Quat::IDENTITY,
        &sphere_collider,
    );

    let upper_manifold = collide_shapes(sphere_upper, mesh_body);
    assert!(
        upper_manifold.is_none(),
        "Sphere inside upper open concave valley must NOT collide with phantom ceiling"
    );

    // Now lower the sphere towards the bottom floor of the trough: (0, 0.4, 0)
    // Distance from (0, 0.4, 0) to slopes:
    // Left plane equation: x + y = 0 -> distance to (0, 0.4) is 0.4 / sqrt(2) ~ 0.283 < 0.4 (penetrating!)
    let sphere_lower = CollisionBody::new(
        1,
        Vec3::new(0.0, 0.4, 0.0),
        Quat::IDENTITY,
        &sphere_collider,
    );

    let lower_manifold = collide_shapes(sphere_lower, mesh_body);
    assert!(
        lower_manifold.is_some(),
        "Sphere at bottom of concave valley must collide with actual valley leaf triangles"
    );
}

#[test]
fn test_concave_mesh_simulation_step_settles_inside_valley() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Create static concave valley
    let vertices = vec![
        Vec3::new(-2.0, 2.0, -2.0),
        Vec3::new(-2.0, 2.0, 2.0),
        Vec3::new(0.0, 0.0, -2.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(2.0, 2.0, -2.0),
        Vec3::new(2.0, 2.0, 2.0),
    ];
    // Counter-clockwise seen from inside the valley.
    let indices = vec![[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]];

    let terrain_id = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        terrain_id,
        Collider::new(Shape::triangle_mesh(&vertices, &indices).expect("mesh builds")),
    );

    // Drop dynamic sphere from y = 1.8 down into the valley
    let sphere_id = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.8, 0.0)));
    world.create_collider(sphere_id, Collider::new(Shape::Sphere { radius: 0.3 }));

    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step(dt);
    }

    let sphere_body = world.body(sphere_id).unwrap();
    // Sphere should have fallen into the valley and settled near the bottom (y between 0.2 and 0.5)
    assert!(
        sphere_body.position.y < 1.0 && sphere_body.position.y > 0.1,
        "Sphere must settle near bottom of valley, actual y: {}",
        sphere_body.position.y
    );
}
