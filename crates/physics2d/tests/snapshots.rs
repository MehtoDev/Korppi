//! Snapshots hold the whole world, as Box2D v3.2's (P2D-13).
//!
//! A snapshot (Box2D `b2World_GetSnapshot`) captures everything the simulation continues from:
//! bodies, colliders, chains and joints with their accumulated impulses, the broadphase trees,
//! contacts with their manifolds and warm-starting impulses, islands, solver sets and sleep
//! timers. Restoring it in place (`b2World_Restore`) keeps the world's callbacks and the handles
//! held at the snapshot; a new world can be built from it (`b2CreateWorldFromSnapshot`), also
//! after a serde round trip. The state hash (`b2World_GetStateHash`) tells worlds that will
//! simulate differently apart. Before FIX-2.6 a snapshot held only body poses and velocities, so
//! a restored world diverged by up to 0.3 mm (audit probe S1). Box2D's behaviour, replaying bit
//! for bit with equal hashes, is checked in `docs/audits/probes/box2d_reference/fix_2_6_snapshots.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::chain::ChainDef2d;
use korppi_physics2d::collider::{Collider2d, Filter2d, SurfaceMaterial2d};
use korppi_physics2d::joint::{Joint2d, RevoluteJoint2d};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::snapshot::WorldSnapshot2d;
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const DT: f32 = 1.0 / 60.0;

/// The probe's scene, with sleep on: a chain ground, a stack of five boxes, a ball dropped on
/// it, a box that falls asleep, and a motorised chain of five links. Returns its dynamic bodies,
/// the sleeper sixth.
fn scene() -> (PhysicsWorld2d, Vec<BodyId2d>) {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -10.0));
    let ground = world.create_body(RigidBody2d::new(BodyType::Static));
    let points = [10.0, 5.0, 0.0, -5.0, -10.0]
        .map(|x| Vec2::new(x, 0.0))
        .to_vec();
    world.create_chain(
        ground,
        &ChainDef2d::open(points, Vec2::new(11.0, 0.0), Vec2::new(-11.0, 0.0)).with_material(
            SurfaceMaterial2d {
                friction: 0.6,
                ..SurfaceMaterial2d::default()
            },
        ),
    );
    let mut bodies = Vec::new();
    let mut dynamic = |world: &mut PhysicsWorld2d, x: f32, y: f32, shape: Shape2d, group| {
        let id =
            world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, y)));
        world.create_collider(
            id,
            Collider2d::new(shape)
                .with_friction(0.6)
                .with_filter(Filter2d {
                    group_index: group,
                    ..Filter2d::DEFAULT
                }),
        );
        bodies.push(id);
        id
    };
    let unit = Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5));
    for i in 0..5 {
        dynamic(&mut world, -4.0, 0.5 + i as f32, unit, 0);
    }
    dynamic(
        &mut world,
        -4.2,
        8.0,
        Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.3)),
        0,
    );
    dynamic(&mut world, 8.0, 0.5, unit, 0);
    let mut previous =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(2.0, 6.0)));
    for i in 0..5 {
        let link = dynamic(
            &mut world,
            2.4 + 0.8 * i as f32,
            6.0,
            Shape2d::Polygon(Polygon2d::from_box(0.4, 0.1)),
            -1,
        );
        let anchor_a = if i == 0 {
            Vec2::ZERO
        } else {
            Vec2::new(0.4, 0.0)
        };
        let mut hinge = RevoluteJoint2d::new(anchor_a, Vec2::new(-0.4, 0.0), 0.0, 0.0);
        if i == 0 {
            hinge = hinge.with_motor(1.0, 20.0);
        }
        world.create_joint(previous, link, Joint2d::Revolute(hinge));
        previous = link;
    }
    (world, bodies)
}

fn run(world: &mut PhysicsWorld2d, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

/// The exact bits of every body's pose and velocities.
fn state_bits(world: &PhysicsWorld2d, bodies: &[BodyId2d]) -> Vec<u32> {
    bodies
        .iter()
        .flat_map(|&id| {
            let b = world.body(id).unwrap();
            [
                b.position.x,
                b.position.y,
                b.rotation.c,
                b.rotation.s,
                b.linear_velocity.x,
                b.linear_velocity.y,
                b.angular_velocity,
            ]
            .map(f32::to_bits)
        })
        .collect()
}

#[test]
fn a_restored_world_replays_bit_for_bit() {
    // Audit probe S1 on Box2D v3.2's terms: snapshot at step 20, run to 80, restore, run to 80
    // again. Box2D replays bit for bit, its hash back to step 20's on restore and equal to
    // step 80's after the replay; the far box is asleep by then.
    let (mut world, bodies) = scene();
    run(&mut world, 20);
    let hash_20 = world.state_hash();
    let snapshot = world.create_snapshot();
    run(&mut world, 60);
    let first = state_bits(&world, &bodies);
    let hash_80 = world.state_hash();
    assert!(
        world.body(bodies[6]).unwrap().is_sleeping,
        "the far box sleeps"
    );
    assert_ne!(hash_20, hash_80);

    world.restore_snapshot(&snapshot);
    assert_eq!(world.state_hash(), hash_20, "restored to step 20");
    run(&mut world, 60);
    assert_eq!(state_bits(&world, &bodies), first, "the replay diverged");
    assert_eq!(world.state_hash(), hash_80);
}

#[test]
fn a_serialized_snapshot_builds_a_world_that_replays_bit_for_bit() {
    // Box2D v3.2 `b2CreateWorldFromSnapshot`: a world built from the image replays as the
    // original. Here the snapshot also goes through JSON first.
    let (mut world, bodies) = scene();
    run(&mut world, 20);
    let json = serde_json::to_string(&world.create_snapshot()).unwrap();
    run(&mut world, 60);
    let snapshot: WorldSnapshot2d = serde_json::from_str(&json).unwrap();
    let mut copy = PhysicsWorld2d::from_snapshot(&snapshot);
    run(&mut copy, 60);
    assert_eq!(state_bits(&copy, &bodies), state_bits(&world, &bodies));
    assert_eq!(copy.state_hash(), world.state_hash());
}

#[test]
fn restoring_keeps_held_handles_and_callbacks() {
    let (mut world, bodies) = scene();
    // The ball reports its new manifolds to the pre-solve callback.
    let ball = world.collider_of(bodies[5]).unwrap();
    world.collider_mut(ball).unwrap().enable_pre_solve_events = true;
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    world.set_pre_solve(move |_, _, _| {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    run(&mut world, 10);
    let snapshot = world.create_snapshot();
    let (joint, _) = world.joints().next().unwrap();

    // Destroy a box and create a body after the snapshot.
    assert!(world.destroy_body(bodies[0]));
    let late = world.create_body(RigidBody2d::new(BodyType::Dynamic));
    world.restore_snapshot(&snapshot);
    assert!(
        world.body(bodies[0]).is_some(),
        "destroyed after the snapshot, back"
    );
    assert!(
        world.body(late).is_none(),
        "created after the snapshot, gone"
    );
    assert!(
        world.joint(joint).is_some(),
        "held joint handles stay valid"
    );

    // The pre-solve callback is host wiring: an in-place restore keeps it, a world built from
    // the snapshot starts without it. The ball reaches the stack within the next second.
    let before = calls.load(Ordering::Relaxed);
    run(&mut world, 60);
    assert!(
        calls.load(Ordering::Relaxed) > before,
        "the callback still runs"
    );
    let mut copy = PhysicsWorld2d::from_snapshot(&snapshot);
    let before = calls.load(Ordering::Relaxed);
    run(&mut copy, 60);
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[test]
fn the_state_hash_tells_diverging_worlds_apart() {
    let (mut a, bodies) = scene();
    let (mut b, _) = scene();
    assert_eq!(
        a.state_hash(),
        b.state_hash(),
        "identical worlds hash alike"
    );
    run(&mut a, 10);
    run(&mut b, 10);
    assert_eq!(a.state_hash(), b.state_hash());
    // The smallest change to a velocity changes the hash.
    let ball = b.body_mut(bodies[5]).unwrap();
    ball.linear_velocity.x = f32::from_bits(ball.linear_velocity.x.to_bits() + 1);
    assert_ne!(a.state_hash(), b.state_hash());
}
