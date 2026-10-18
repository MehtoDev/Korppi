//! Primary test for 3D physics materials, conveyors, callbacks, and configuration (P3D-25).

use korppi_math::Vec3;
use korppi_physics::mesh::{Mesh3d, MeshOptions};
use korppi_physics::shape::Shape;
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, SurfaceMaterial3d};

#[test]
fn test_conveyor_belt_accelerates_box_to_target_speed() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
    world.enable_sleep(false);

    // Static floor with conveyor tangent velocity (3.0, 0, 0)
    let floor_body = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    let floor_collider = world.create_collider(
        floor_body,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.8)
        .with_tangent_velocity(Vec3::new(3.0, 0.0, 0.0)),
    );
    assert_eq!(
        world
            .surface_material(floor_collider)
            .unwrap()
            .tangent_velocity,
        Vec3::new(3.0, 0.0, 0.0)
    );

    // Dynamic box placed at (0, 0.5, 0) with zero initial velocity
    let box_body = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    world.create_collider(
        box_body,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.8)
        .with_density(1000.0),
    );

    // Step 60 ticks (1 second) at 60 Hz
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    let b = world.body(box_body).unwrap();
    // Box should accelerate up to the conveyor belt speed of 3.0 m/s along x
    assert!(
        (b.linear_velocity.x - 3.0).abs() <= 0.05,
        "box vx {} should reach conveyor speed 3.0 m/s",
        b.linear_velocity.x
    );
    assert!(
        b.linear_velocity.y.abs() <= 0.05,
        "box vy {} should remain near 0",
        b.linear_velocity.y
    );
    assert!(
        b.linear_velocity.z.abs() <= 0.05,
        "box vz {} should remain near 0",
        b.linear_velocity.z
    );
    // In 1 second accelerating from 0 to 3 m/s, position x should be ~2.43 m (Box3D reference: 2.43311 m)
    assert!(
        (b.position.x - 2.433).abs() <= 0.2,
        "box pos x {} should match Box3D reference ~2.43 m",
        b.position.x
    );
}

#[test]
fn test_conveyor_belt_moving_with_box_has_zero_relative_friction() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
    world.enable_sleep(false);

    // Static floor with conveyor tangent velocity (3.0, 0, 0)
    let floor_body = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        floor_body,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.8)
        .with_tangent_velocity(Vec3::new(3.0, 0.0, 0.0)),
    );

    // Dynamic box already moving at (3.0, 0, 0)
    let mut box_rb = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0));
    box_rb.linear_velocity = Vec3::new(3.0, 0.0, 0.0);
    let box_body = world.create_body(box_rb);
    world.create_collider(
        box_body,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.8)
        .with_density(1000.0),
    );

    // Step 60 ticks
    let dt = 1.0 / 60.0;
    for _ in 0..60 {
        world.step(dt);
    }

    let b = world.body(box_body).unwrap();
    // Because relative slip is zero, box should remain moving at 3.0 m/s without decelerating
    assert!(
        (b.linear_velocity.x - 3.0).abs() <= 0.02,
        "box vx {} should remain 3.0 m/s with zero relative slip",
        b.linear_velocity.x
    );
    assert!(
        (b.position.x - 3.0).abs() <= 0.1,
        "box pos x {} should be ~3.0 m after 1 s at 3 m/s",
        b.position.x
    );
}

#[test]
fn test_custom_friction_and_restitution_callbacks() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
    world.enable_sleep(false);

    // Custom mixing callbacks
    world.set_friction_callback(|fa, _, fb, _| fa + fb);
    world.set_restitution_callback(|ra, _, rb, _| ra.min(rb));

    // Floor with restitution 0.9
    let floor_body = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        floor_body,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_restitution(0.9)
        .with_user_material_id(1),
    );

    // Sphere dropped from y = 5.0 with restitution 0.5
    let sphere_body =
        world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 5.0, 0.0)));
    world.create_collider(
        sphere_body,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_restitution(0.5)
            .with_user_material_id(2)
            .with_density(1000.0),
    );

    // Rebound test: min(0.9, 0.5) = 0.5 gives rebound height ~1.57 m (Box3D probe: 1.57459 m).
    // If it used max(0.9, 0.5) = 0.9, rebound height would be > 4.0 m!
    let dt = 1.0 / 60.0;
    let mut max_y = 0.0f32;
    let mut hit_ground = false;
    for _ in 0..120 {
        world.step(dt);
        let pos = world.body(sphere_body).unwrap().position;
        if pos.y < 0.6 {
            hit_ground = true;
        }
        if hit_ground && pos.y > max_y {
            max_y = pos.y;
        }
    }

    assert!(
        (1.4..=1.8).contains(&max_y),
        "rebound height {max_y} should match min restitution ~1.57 m (Box3D reference)"
    );
}

#[test]
fn test_per_triangle_mesh_materials_and_shape_material_api() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));
    world.enable_sleep(false);

    // A two-triangle flat quad floor: triangle 0 (ice, friction 0.0), triangle 1 (rubber, friction 1.0)
    let vertices = vec![
        Vec3::new(-10.0, 0.0, -10.0),
        Vec3::new(10.0, 0.0, -10.0),
        Vec3::new(10.0, 0.0, 10.0),
        Vec3::new(-10.0, 0.0, 10.0),
    ];
    let indices = vec![
        [0, 2, 1], // triangle 0 (upward facing normal)
        [0, 3, 2], // triangle 1 (upward facing normal)
    ];
    let material_indices = vec![0u8, 1u8];
    let mesh = Mesh3d::with_options(
        &vertices,
        &indices,
        MeshOptions {
            material_indices: Some(material_indices),
            ..Default::default()
        },
    )
    .expect("valid mesh");

    let mesh_body = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    let mat_ice = SurfaceMaterial3d {
        friction: 0.0,
        user_material_id: 101,
        ..Default::default()
    };

    let mat_rubber = SurfaceMaterial3d {
        friction: 1.0,
        user_material_id: 102,
        ..Default::default()
    };

    let mesh_collider = world.create_collider(
        mesh_body,
        Collider::new(Shape::TriangleMesh(mesh)).with_mesh_materials(vec![mat_ice, mat_rubber]),
    );

    // Verify shape material API
    assert_eq!(world.mesh_material_count(mesh_collider), 2);
    assert_eq!(
        world
            .mesh_material(mesh_collider, 0)
            .unwrap()
            .user_material_id,
        101
    );
    assert_eq!(
        world
            .mesh_material(mesh_collider, 1)
            .unwrap()
            .user_material_id,
        102
    );

    // Mutate material 0 at runtime
    let mut updated_mat = mat_ice;
    updated_mat.user_material_id = 999;
    world.set_mesh_material(mesh_collider, 0, updated_mat);
    assert_eq!(
        world
            .mesh_material(mesh_collider, 0)
            .unwrap()
            .user_material_id,
        999
    );

    // Raycast closest on triangle 0 should return user_material_id 999
    let hit = world.cast_ray_closest(
        Vec3::new(5.0, 5.0, -5.0),
        Vec3::new(0.0, -10.0, 0.0),
        &Default::default(),
    );
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().user_material_id, 999);
}

#[test]
fn test_world_configuration_toggles_and_profiler() {
    let mut world = PhysicsWorld::with_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Test sleep toggle
    assert!(world.is_sleep_enabled());
    world.enable_sleep(false);
    assert!(!world.is_sleep_enabled());

    // Test warm starting toggle
    assert!(world.is_warm_starting_enabled());
    world.enable_warm_starting(false);
    assert!(!world.is_warm_starting_enabled());

    // Create a body and step
    let body = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 2.0, 0.0)));
    world.create_collider(body, Collider::new(Shape::Sphere { radius: 0.5 }));

    world.step(1.0 / 60.0);

    // Profile check
    let profile = world.profile();
    assert!(profile.step >= 0.0, "step time must be non-negative");
    assert!(profile.solve >= 0.0, "solve time must be non-negative");
}
