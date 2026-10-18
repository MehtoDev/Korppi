//! Primary test for 3D constraint graph coloring, persistent islands, and solver sets (P3D-18).

use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::joint::{DistanceJoint, Joint};
use korppi_physics::shape::Shape;
use korppi_physics::PhysicsWorld;

#[test]
fn test_constraint_graph_color_independence_and_limits() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    // Create a central body b0 and 25 dynamic bodies surrounding it touching b0.
    // Body b0 has degree 25, which exceeds the 23 non-overflow colors (0..23).
    // Therefore, at least one constraint must be assigned to the overflow color (23).
    let b0 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::ZERO));
    world.create_collider(b0, Collider::new(Shape::Sphere { radius: 10.0 }));

    let mut others = Vec::new();
    for i in 0..25 {
        let angle = (i as f32) * (std::f32::consts::TAU / 25.0);
        let pos = Vec3::new(angle.cos() * 10.5, 0.0, angle.sin() * 10.5);
        let b = world.create_body(RigidBody::new(BodyType::Dynamic, pos));
        world.create_collider(b, Collider::new(Shape::Sphere { radius: 1.0 }));
        others.push(b);
    }

    world.step(1.0 / 60.0);

    // Verify graph coloring on the world
    let graph = world.constraint_graph();
    assert_eq!(graph.color_count(), 24);

    // Every non-overflow color must have complete body independence (no body appears twice)
    for color_idx in 0..graph.color_count() - 1 {
        let pairs = graph.color_pairs(color_idx);
        let mut seen_bodies = std::collections::HashSet::new();
        for (ba, bb) in pairs {
            assert!(
                seen_bodies.insert(ba),
                "Color {color_idx} contains duplicate dynamic body {ba}"
            );
            assert!(
                seen_bodies.insert(bb),
                "Color {color_idx} contains duplicate dynamic body {bb}"
            );
        }
    }

    // Overflow color must contain the overflowing constraints
    let overflow_idx = graph.overflow_index();
    assert_eq!(overflow_idx, 23);
    assert!(
        graph.color_len(overflow_idx) > 0,
        "Degree 25 node must place constraints into overflow color"
    );
}

#[test]
fn test_constraint_graph_priority_dynamic_vs_static() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Ground is static
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        }),
    );

    // 2 Dynamic bodies: b1 resting on ground, b2 on b1
    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    world.create_collider(
        b1,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        }),
    );

    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.5, 0.0)));
    world.create_collider(
        b2,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        }),
    );

    world.step(1.0 / 60.0);

    let graph = world.constraint_graph();
    let dyn_color_count = graph.dynamic_color_count();
    assert_eq!(dyn_color_count, 20);

    // The contact between b1 and b2 (dynamic-dynamic) should be in colors 0..20
    // The contact between b1 and ground (dynamic-static) should be in colors 20..23 (higher priority)
    let b1_idx = b1.index();
    let b2_idx = b2.index();
    let g_idx = ground.index();

    let mut found_dyn_dyn = false;
    let mut found_dyn_static = false;

    for color_idx in 0..graph.color_count() {
        for (ba, bb) in graph.color_pairs(color_idx) {
            let pair = (ba.min(bb), ba.max(bb));
            if pair == (b1_idx.min(b2_idx), b1_idx.max(b2_idx)) {
                assert!(
                    color_idx < dyn_color_count,
                    "Dynamic-dynamic constraint should be in dynamic colors 0..{dyn_color_count}, got {color_idx}"
                );
                found_dyn_dyn = true;
            }
            if pair == (b1_idx.min(g_idx), b1_idx.max(g_idx)) {
                assert!(
                    color_idx >= dyn_color_count || color_idx == graph.overflow_index(),
                    "Dynamic-static constraint should be in static priority colors {dyn_color_count}..23, got {color_idx}"
                );
                found_dyn_static = true;
            }
        }
    }

    assert!(found_dyn_dyn);
    assert!(found_dyn_static);
}

#[test]
fn test_solver_sets_sleeping_and_island_propagation() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));
    world.enable_sleep(true);

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.8),
    );

    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    world.create_collider(
        b1,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.8),
    );

    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.5, 0.0)));
    world.create_collider(
        b2,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        })
        .with_friction(0.8),
    );

    // Step 2 seconds (120 ticks) until settled and sleeping
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        world.step(dt);
    }

    assert!(world.body(b1).unwrap().is_sleeping);
    assert!(world.body(b2).unwrap().is_sleeping);

    // While sleeping, both bodies belong to a sleeping solver set (>= 3)
    assert!(world.body_solver_set(b1) >= 3);
    assert_eq!(world.body_solver_set(b1), world.body_solver_set(b2));

    // Applying impulse to b2 must wake b2 AND propagate to b1 via the island
    world
        .body_mut(b2)
        .unwrap()
        .apply_linear_impulse(Vec3::new(5.0, 0.0, 0.0));
    world.step(dt);

    assert!(!world.body(b2).unwrap().is_sleeping);
    assert!(!world.body(b1).unwrap().is_sleeping);

    // Awake bodies are in the awake solver set (set 2)
    assert_eq!(world.body_solver_set(b1), 2);
    assert_eq!(world.body_solver_set(b2), 2);
}

#[test]
fn test_joint_coloring_and_solver_sets() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.0, 0.0)));
    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 0.0, 0.0)));
    let b3 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(4.0, 0.0, 0.0)));

    let _j1 = world.create_joint(Joint::Distance(DistanceJoint::new(
        b1,
        b2,
        Vec3::ZERO,
        Vec3::ZERO,
        2.0,
    )));
    let _j2 = world.create_joint(Joint::Distance(DistanceJoint::new(
        b2,
        b3,
        Vec3::ZERO,
        Vec3::ZERO,
        2.0,
    )));

    world.step(1.0 / 60.0);

    let graph = world.constraint_graph();
    // j1 touches (b1, b2), j2 touches (b2, b3). Since b2 is shared, they cannot have the same color!
    let mut j1_color = None;
    let mut j2_color = None;

    for color_idx in 0..graph.color_count() {
        for (ba, bb) in graph.color_pairs(color_idx) {
            let pair = (ba.min(bb), ba.max(bb));
            if pair == (b1.index().min(b2.index()), b1.index().max(b2.index())) {
                j1_color = Some(color_idx);
            }
            if pair == (b2.index().min(b3.index()), b2.index().max(b3.index())) {
                j2_color = Some(color_idx);
            }
        }
    }

    assert!(j1_color.is_some());
    assert!(j2_color.is_some());
    assert_ne!(
        j1_color.unwrap(),
        j2_color.unwrap(),
        "Constraints sharing dynamic body b2 must not have the same color"
    );
}
