//! Integration tests for korppi_physics pipeline, BVH, collision detection, and determinism.

use korppi_math::{Aabb, Quat, Ray, Vec3};
use korppi_physics::{BodyType, Collider, DynamicTree, PhysicsWorld, RigidBody, Shape};

#[test]
fn dynamic_bvh_tree_handles_inserts_raycasts_and_queries() {
    let mut tree = DynamicTree::new();

    // Insert 3 leaf nodes with user data (e.g. entity id or index)
    let aabb1 = Aabb::from_min_max(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
    let aabb2 = Aabb::from_min_max(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0));
    let aabb3 = Aabb::from_min_max(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 6.0, 1.0));

    let leaf1 = tree.create_proxy(aabb1, 101);
    let leaf2 = tree.create_proxy(aabb2, 102);
    let _leaf3 = tree.create_proxy(aabb3, 103);

    assert_eq!(tree.node_count(), 5); // 3 leaves + 2 internal nodes

    // Raycast hitting leaf 1
    let ray = Ray::new(Vec3::new(0.5, 0.5, -5.0), Vec3::Z);
    let mut hit_user_data = None;
    tree.ray_cast(&ray, 10.0, |user_data, t| {
        hit_user_data = Some(user_data);
        t // Clip ray to closest hit
    });
    assert_eq!(hit_user_data, Some(101));

    // Overlap query around leaf 2
    let query_box = Aabb::from_min_max(Vec3::new(4.5, -0.5, -0.5), Vec3::new(6.5, 1.5, 1.5));
    let mut overlaps = Vec::new();
    tree.query(&query_box, |user_data| {
        overlaps.push(user_data);
        true
    });
    assert_eq!(overlaps, vec![102]);

    // Move proxy 1
    let new_aabb1 = Aabb::from_min_max(Vec3::new(10.0, 0.0, 0.0), Vec3::new(11.0, 1.0, 1.0));
    tree.move_proxy(leaf1, new_aabb1, Vec3::ZERO);

    // Destroy proxy 2
    tree.destroy_proxy(leaf2);
    assert_eq!(tree.node_count(), 3); // 2 leaves + 1 internal
}

#[test]
fn sphere_ground_collision_resolves_contact_and_settles() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Static ground box at y = -1.0 with thickness 2.0 (top surface at y = 0.0)
    let ground_body = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, -1.0, 0.0),
        rotation: Quat::IDENTITY,
        ..Default::default()
    });
    world.create_collider(
        ground_body,
        Collider {
            shape: Shape::Box {
                half_extents: Vec3::new(10.0, 1.0, 10.0),
            },
            friction: 0.3,
            restitution: 0.0, // Non-bouncing to settle quickly
            is_sensor: false,
            ..Default::default()
        },
    );

    // Dynamic sphere dropping from y = 5.0 with radius 1.0
    let sphere_body = world.create_body(RigidBody {
        body_type: BodyType::Dynamic,
        position: Vec3::new(0.0, 5.0, 0.0),
        rotation: Quat::IDENTITY,
        mass: 1.0,
        ..Default::default()
    });
    world.create_collider(
        sphere_body,
        Collider {
            shape: Shape::Sphere { radius: 1.0 },
            friction: 0.3,
            restitution: 0.0,
            is_sensor: false,
            ..Default::default()
        },
    );

    // Step physics for 120 ticks
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step(dt);
    }

    // Sphere has radius 1.0, top of ground is y = 0.0 -> resting position y ~ 1.0
    let final_pos = world.body(sphere_body).unwrap().position;
    assert!(
        (final_pos.y - 1.0).abs() < 0.05,
        "Expected resting y ~ 1.0, got {}",
        final_pos.y
    );
    assert!(world.body(sphere_body).unwrap().linear_velocity.length() < 0.1);
}

#[test]
fn physics_simulation_is_strictly_deterministic_across_runs() {
    let run_simulation = || -> Vec<Vec3> {
        let mut world = PhysicsWorld::new();
        world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

        let ground = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            ..Default::default()
        });
        world.create_collider(
            ground,
            Collider {
                shape: Shape::Box {
                    half_extents: Vec3::new(50.0, 1.0, 50.0),
                },
                friction: 0.4,
                restitution: 0.2,
                is_sensor: false,
                ..Default::default()
            },
        );

        // Spawn a small stack of 3 boxes
        let mut bodies = Vec::new();
        for i in 0..3 {
            let body = world.create_body(RigidBody {
                body_type: BodyType::Dynamic,
                position: Vec3::new(0.0, 2.0 + (i as f32) * 2.1, 0.0),
                rotation: Quat::IDENTITY,
                mass: 2.0,
                ..Default::default()
            });
            world.create_collider(
                body,
                Collider {
                    shape: Shape::Box {
                        half_extents: Vec3::splat(1.0),
                    },
                    friction: 0.4,
                    restitution: 0.2,
                    is_sensor: false,
                    ..Default::default()
                },
            );
            bodies.push(body);
        }

        // Simulate 120 ticks
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            world.step(dt);
        }

        bodies
            .into_iter()
            .map(|b| world.body(b).unwrap().position)
            .collect()
    };

    let run1 = run_simulation();
    let run2 = run_simulation();

    assert_eq!(run1.len(), 3);
    assert_eq!(
        run1, run2,
        "Physics simulation must be bitwise identical across runs"
    );
}
