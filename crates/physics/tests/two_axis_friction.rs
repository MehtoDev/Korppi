//! Integration tests for two-axis orthonormal friction and Coulomb cone clamping in 3D physics.
//! Verifies isotropic deceleration along arbitrary sliding headings and torsional damping on multi-point contact manifolds.

use korppi_math::{Quat, Vec3};
use korppi_physics::solver::compute_orthonormal_basis;
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

#[test]
fn orthonormal_basis_satisfies_invariants() {
    let normals = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.577, 0.577, 0.577).normalize(),
        Vec3::new(-0.3, 0.8, -0.5).normalize(),
    ];

    for n in normals {
        let (t1, t2) = compute_orthonormal_basis(n);
        // Orthogonality to normal
        assert!(
            n.dot(t1).abs() < 1e-6,
            "t1 not perpendicular to n: dot={}",
            n.dot(t1)
        );
        assert!(
            n.dot(t2).abs() < 1e-6,
            "t2 not perpendicular to n: dot={}",
            n.dot(t2)
        );
        // Orthogonality between tangents
        assert!(
            t1.dot(t2).abs() < 1e-6,
            "t1 not perpendicular to t2: dot={}",
            t1.dot(t2)
        );
        // Unit lengths
        assert!((t1.length() - 1.0).abs() < 1e-6, "t1 not unit length");
        assert!((t2.length() - 1.0).abs() < 1e-6, "t2 not unit length");
    }
}

#[test]
fn diagonal_sliding_friction_decelerates_isotropically() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Ground box
    let ground = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, -1.0, 0.0),
        rotation: Quat::IDENTITY,
        ..Default::default()
    });
    world.create_collider(
        ground,
        Collider {
            shape: Shape::Box {
                half_extents: Vec3::new(50.0, 1.0, 50.0),
            },
            friction: 0.5,
            restitution: 0.0,
            is_sensor: false,
            ..Default::default()
        },
    );

    // Sliding box with initial velocity at 45 degrees in X-Z plane
    let slider = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(0.0, 0.5, 0.0),
        rotation: Quat::IDENTITY,
        linear_velocity: Vec3::new(5.0, 0.0, 5.0),
        mass: 2.0,
        ..Default::default()
    });
    world.create_collider(
        slider,
        Collider {
            shape: Shape::Box {
                half_extents: Vec3::splat(0.5),
            },
            friction: 0.5,
            restitution: 0.0,
            is_sensor: false,
            ..Default::default()
        },
    );

    let dt = 1.0 / 60.0;
    // Step until slider decelerates
    for _ in 0..120 {
        world.step(dt);
    }

    let b = world.body(slider).unwrap();
    // After 2 seconds with mu = 0.5, slider should come to rest
    assert!(
        b.linear_velocity.length() < 0.1,
        "Slider should have come to rest, velocity was {:?}",
        b.linear_velocity
    );

    // Lateral displacements in X and Z should be nearly identical (isotropic friction cone)
    let ratio = b.position.x / b.position.z;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "Friction deceleration must be isotropic along diagonal: pos=({:.3}, {:.3}), ratio={:.3}",
        b.position.x,
        b.position.z,
        ratio
    );
}

#[test]
fn multi_point_contact_damps_torsional_spin() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Ground
    let ground = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, -1.0, 0.0),
        rotation: Quat::IDENTITY,
        ..Default::default()
    });
    world.create_collider(
        ground,
        Collider {
            shape: Shape::Box {
                half_extents: Vec3::new(20.0, 1.0, 20.0),
            },
            friction: 0.6,
            restitution: 0.0,
            is_sensor: false,
            ..Default::default()
        },
    );

    // Box resting on ground spinning around Y
    let spinning_box = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(0.0, 0.5, 0.0),
        rotation: Quat::IDENTITY,
        angular_velocity: Vec3::new(0.0, 10.0, 0.0),
        mass: 1.0,
        ..Default::default()
    });
    world.create_collider(
        spinning_box,
        Collider {
            shape: Shape::Box {
                half_extents: Vec3::splat(0.5),
            },
            friction: 0.6,
            restitution: 0.0,
            is_sensor: false,
            ..Default::default()
        },
    );

    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step(dt);
    }

    let b = world.body(spinning_box).unwrap();
    // Angular velocity must be substantially damped by the 4-point contact patch friction torques
    assert!(
        b.angular_velocity.y.abs() < 1.0,
        "Spinning box torsional angular velocity should be damped, got {:.3}",
        b.angular_velocity.y
    );
}
