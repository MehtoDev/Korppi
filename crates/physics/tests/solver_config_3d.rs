//! Integration tests for 3D Soft-Step configurable constraint solver (Box3D).

use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::shape::Shape;
use korppi_physics::solver::SolverConfig3d;
use korppi_physics::PhysicsWorld;

#[test]
fn solver_config_3d_defaults_match_box3d_specification() {
    let config = SolverConfig3d::default();
    assert_eq!(config.sub_steps, 4);
    assert!((config.hertz - 30.0).abs() < 1e-4);
    assert!((config.damping_ratio - 10.0).abs() < 1e-4);
    assert!((config.contact_speed - 3.0).abs() < 1e-4);
    assert!((config.friction - 0.5).abs() < 1e-4);
    assert!((config.restitution - 0.0).abs() < 1e-4);
}

#[test]
fn physics_world_exposes_configurable_solver() {
    let mut world = PhysicsWorld::new();
    assert_eq!(world.solver_config().sub_steps, 4);

    let custom = SolverConfig3d {
        sub_steps: 8,
        hertz: 60.0,
        damping_ratio: 5.0,
        contact_speed: 4.0,
        friction: 0.8,
        restitution: 0.5,
        ..Default::default()
    };
    world.set_solver_config(custom);
    assert_eq!(world.solver_config().sub_steps, 8);
    assert!((world.solver_config().hertz - 60.0).abs() < 1e-4);

    world.solver_config_mut().hertz = 45.0;
    assert!((world.solver_config().hertz - 45.0).abs() < 1e-4);
}

#[test]
fn soft_step_solver_tunes_spring_compliance_dynamically() {
    let mut world_soft = PhysicsWorld::new();
    world_soft.set_solver_config(SolverConfig3d {
        sub_steps: 4,
        hertz: 5.0, // Very soft spring
        damping_ratio: 1.0,
        contact_speed: 3.0,
        friction: 0.5,
        restitution: 0.0,
        ..Default::default()
    });

    let ground = world_soft.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world_soft.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        }),
    );

    let mut body_def = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 2.0, 0.0));
    body_def.mass = 10.0;
    body_def.inv_mass = 0.1;
    let b_soft = world_soft.create_body(body_def);
    world_soft.create_collider(
        b_soft,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    // Step a few ticks
    for _ in 0..10 {
        world_soft.step(1.0 / 60.0);
    }

    let pos_soft = world_soft.body(b_soft).unwrap().position.y;

    let mut world_stiff = PhysicsWorld::new();
    world_stiff.set_solver_config(SolverConfig3d {
        sub_steps: 4,
        hertz: 60.0, // Stiff spring
        damping_ratio: 10.0,
        contact_speed: 3.0,
        friction: 0.5,
        restitution: 0.0,
        ..Default::default()
    });

    let ground_stiff = world_stiff.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world_stiff.create_collider(
        ground_stiff,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        }),
    );

    let mut body_def_stiff = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 2.0, 0.0));
    body_def_stiff.mass = 10.0;
    body_def_stiff.inv_mass = 0.1;
    let b_stiff = world_stiff.create_body(body_def_stiff);
    world_stiff.create_collider(
        b_stiff,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    for _ in 0..10 {
        world_stiff.step(1.0 / 60.0);
    }

    let pos_stiff = world_stiff.body(b_stiff).unwrap().position.y;

    // Both should be valid numbers (finite) and settle stably
    assert!(pos_soft.is_finite());
    assert!(pos_stiff.is_finite());
    // Soft world should penetrate slightly deeper under heavy mass before spring pushes back
    assert!(pos_soft <= pos_stiff + 0.05);
}
