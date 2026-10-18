//! Integration tests for advanced solver features: Constraint Graph Coloring, Island Sleep, and Snapshots.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::constraint_graph::ConstraintGraph2d;
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::snapshot::WorldSnapshot2d;
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

#[test]
fn constraint_graph_colors_independent_batches_without_body_collisions() {
    let mut graph = ConstraintGraph2d::new(8);

    // Constraint 0: Body 0 <-> Body 1
    let c0 = graph.add_constraint(0, 1);
    // Constraint 1: Body 2 <-> Body 3 (disjoint from 0 & 1 -> should be same color as c0)
    let c1 = graph.add_constraint(2, 3);
    // Constraint 2: Body 1 <-> Body 2 (conflicts with Body 1 and Body 2 -> must be different color)
    let c2 = graph.add_constraint(1, 2);

    assert_eq!(c0, 0, "First constraint should be in Color 0");
    assert_eq!(c1, 0, "Disjoint constraint should share Color 0");
    assert_ne!(c2, 0, "Conflicting constraint must be in a different color");

    // Verify invariants: for every color batch, no body appears twice
    for (color_idx, color) in graph.colors().enumerate() {
        let mut seen = std::collections::HashSet::new();
        for (b_a, b_b) in color.pairs() {
            assert!(
                seen.insert(b_a),
                "Body {} appeared multiple times in Color {}",
                b_a,
                color_idx
            );
            assert!(
                seen.insert(b_b),
                "Body {} appeared multiple times in Color {}",
                b_b,
                color_idx
            );
        }
    }
}

/// A shapeless 1 kg, 1 kg·m² body at the origin.
fn hinge_body(world: &mut PhysicsWorld2d) -> BodyId2d {
    world.create_body(RigidBody2d::new(BodyType::Dynamic).with_mass(1.0, 1.0))
}

fn hinge(world: &mut PhysicsWorld2d, a: BodyId2d, b: BodyId2d) {
    world.create_joint(
        a,
        b,
        Joint2d::Revolute(RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0)),
    );
}

#[test]
fn jointed_bodies_sleep_and_wake_together() {
    // Two hinged bodies share an island (Box2D v3.2: 2 islands, then 1), so putting one to
    // sleep puts both to sleep and waking one wakes both. Values from fix_2_5_islands.c.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let (a, b) = (hinge_body(&mut world), hinge_body(&mut world));
    assert_eq!(world.counters().island_count, 2);
    hinge(&mut world, a, b);
    assert_eq!(world.counters().island_count, 1);
    let asleep = |world: &PhysicsWorld2d, id| world.body(id).unwrap().is_sleeping;

    world.set_awake(a, false);
    assert!(asleep(&world, a));
    assert!(asleep(&world, b), "jointed bodies sleep together");
    world.set_awake(a, true);
    assert!(!asleep(&world, a));
    assert!(!asleep(&world, b), "jointed bodies wake together");
}

#[test]
fn world_snapshot_roundtrip_preserves_bitwise_determinism() {
    // Since FIX-2.6 a snapshot holds the whole world (owner decision): a snapshot taken from a
    // stepped world survives JSON unchanged, and the world built from it keeps simulating bit
    // for bit as the original, with equal state hashes.
    let mut world = PhysicsWorld2d::new();
    let mut body = RigidBody2d::new(BodyType::Dynamic);
    body.position = Vec2::new(12.345, -67.890);
    body.linear_velocity = Vec2::new(1.23, -4.56);
    body.angular_velocity = 0.789;
    let id = world.create_body(body.with_mass(1.0, 1.0));
    for _ in 0..42 {
        world.step(1.0 / 60.0);
    }

    let snapshot = world.create_snapshot();
    let serialized = serde_json::to_string(&snapshot).expect("serialization succeeds");
    let deserialized: WorldSnapshot2d =
        serde_json::from_str(&serialized).expect("deserialization succeeds");
    assert_eq!(deserialized.step_index(), 42);
    assert_eq!(deserialized.state_hash(), snapshot.state_hash());

    let mut copy = PhysicsWorld2d::from_snapshot(&deserialized);
    assert_eq!(copy.body(id), world.body(id));
    for _ in 0..30 {
        world.step(1.0 / 60.0);
        copy.step(1.0 / 60.0);
    }
    assert_eq!(copy.body(id), world.body(id));
    assert_eq!(copy.state_hash(), world.state_hash());
}

#[test]
fn separate_chains_form_separate_islands() {
    // 256 bodies hinged every other body: the even and the odd ones form two chains, so two
    // islands (Box2D v3.2). Putting body 0 to sleep sleeps the even chain only; waking body 254
    // wakes it again. Values from fix_2_5_islands.c.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let n = 256;
    let bodies: Vec<BodyId2d> = (0..n).map(|_| hinge_body(&mut world)).collect();
    for i in 0..n - 2 {
        hinge(&mut world, bodies[i], bodies[i + 2]);
    }
    let counters = world.counters();
    assert_eq!((counters.island_count, counters.awake_body_count), (2, 256));
    let asleep = |world: &PhysicsWorld2d, i: usize| world.body(bodies[i]).unwrap().is_sleeping;

    world.set_awake(bodies[0], false);
    assert!(asleep(&world, n - 2));
    assert!(!asleep(&world, n - 1));
    let counters = world.counters();
    assert_eq!((counters.island_count, counters.awake_body_count), (2, 128));

    world.set_awake(bodies[n - 2], true);
    assert!(!asleep(&world, 0));
    assert_eq!(world.counters().awake_body_count, 256);
}
