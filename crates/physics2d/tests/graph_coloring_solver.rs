//! Integration tests for Constraint Graph Coloring in the 2D solver.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::joint::{DistanceJoint2d, Joint2d};
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::PhysicsWorld2d;
use std::collections::HashSet;

#[test]
fn test_solver_executes_over_graph_color_batches() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    // Create a horizontal grid of 4 connected boxes:
    // Box 0 <-> Box 1 <-> Box 2 <-> Box 3
    // Resting on a static floor
    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 1.0))),
    );

    let mut box_ids = Vec::new();
    for i in 0..4 {
        let b = world.create_body(
            RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(i as f32 * 1.0, 1.5)),
        );
        world.create_collider(
            b,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.45, 0.45))),
        );
        box_ids.push(b);
    }

    // Connect with distance joints: (0, 1), (1, 2), (2, 3)
    for i in 0..3 {
        let joint = DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 1.0, 0.0, 0.0);
        world.create_joint(box_ids[i], box_ids[i + 1], Joint2d::Distance(joint));
    }

    // Step once
    world.step(1.0 / 60.0);

    // Verify constraint graph is populated and used
    let graph = world.constraint_graph();
    assert!(graph.color_count() >= 2);

    let mut total_color_constraints = 0;
    for i in 0..graph.color_count() {
        let color = graph.color(i);
        total_color_constraints += color.joint_indices.len() + color.contact_keys.len();

        // In primary non-overflow colors, dynamic bodies must be strictly disjoint!
        if i < graph.overflow_index() {
            let mut dynamic_seen = HashSet::new();
            for &j_idx in &color.joint_indices {
                // Verify no dynamic body appears multiple times in this color
                let (b_a, b_b) = (box_ids[j_idx], box_ids[j_idx + 1]);
                assert!(
                    dynamic_seen.insert(b_a),
                    "Dynamic body {b_a:?} appears multiple times in color {i}"
                );
                assert!(
                    dynamic_seen.insert(b_b),
                    "Dynamic body {b_b:?} appears multiple times in color {i}"
                );
            }
        }
    }

    assert!(
        total_color_constraints > 0,
        "Constraint graph must partition active constraints into colors"
    );
}

#[test]
fn test_graph_coloring_preserves_simulation_stability() {
    let mut world = PhysicsWorld2d::new();
    world.set_gravity(Vec2::new(0.0, -9.81));

    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, 0.0)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(10.0, 1.0))),
    );

    // Stack 4 boxes
    let mut boxes = Vec::new();
    for i in 0..4 {
        let b = world.create_body(
            RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 1.5 + i as f32 * 1.0)),
        );
        world.create_collider(
            b,
            Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
        );
        boxes.push(b);
    }

    // Step for 120 frames (2 physical seconds)
    for _ in 0..120 {
        world.step(1.0 / 60.0);
    }

    // Stack should remain vertically aligned (|x| < 0.05)
    for (i, &b) in boxes.iter().enumerate() {
        let pos = world.body(b).unwrap().position;
        assert!(
            pos.x.abs() < 0.05,
            "Box {i} drifted excessively in X: {}",
            pos.x
        );
        assert!(pos.y > 1.4, "Box {i} fell below expected height: {}", pos.y);
    }
}
