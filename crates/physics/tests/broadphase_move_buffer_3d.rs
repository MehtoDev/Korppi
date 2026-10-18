//! Primary tests for the 3D broadphase move buffer and persistent pair set (FIX-8.2 / P3D-27,
//! Probe Q13).

use korppi_math::Vec3;
use korppi_physics::{BodyId3d, BodyType, Collider, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

fn spheres_scene(n: usize, sleep: bool) -> (PhysicsWorld, Vec<BodyId3d>) {
    let mut world = PhysicsWorld::new();
    world.enable_sleep(sleep);
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(200.0, 0.5, 200.0),
        })
        .with_friction(0.6),
    );
    let side = (n as f32).sqrt().ceil() as usize;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let pos = Vec3::new((i % side) as f32 * 1.5, 0.49, (i / side) as f32 * 1.5);
        let id = world.create_body(RigidBody::new(BodyType::Dynamic, pos).with_enable_sleep(sleep));
        world.create_collider(
            id,
            Collider::new(Shape::Sphere { radius: 0.5 }).with_friction(0.6),
        );
        ids.push(id);
    }
    (world, ids)
}

#[test]
fn resting_awake_spheres_clear_the_move_buffer_and_keep_persistent_pairs() {
    let n = 1000;
    let (mut world, _ids) = spheres_scene(n, false);

    // Newly created colliders are queued in the broadphase move buffer before the first step.
    assert!(
        world.broadphase_move_count() >= n,
        "newly created dynamic colliders must be queued in the move buffer"
    );

    // Step 5 frames so all spheres settle inside their fat AABBs with sleep disabled.
    for _ in 0..5 {
        world.step(DT);
    }

    assert_eq!(
        world.counters().contact_count,
        n,
        "all 1,000 sphere-ground contacts must persist in the contact manager"
    );
    assert_eq!(
        world.counters().touching_contact_count,
        n,
        "all 1,000 sphere-ground contacts must remain touching"
    );
    assert_eq!(
        world.broadphase_move_count(),
        0,
        "resting awake bodies whose bounds stay inside their fat AABBs must not enter the move buffer"
    );

    // Step 30 more frames with sleep off (Probe Q13): broadphase does 0 tree queries.
    for _ in 0..30 {
        world.step(DT);
        assert_eq!(world.broadphase_move_count(), 0);
    }

    assert_eq!(world.counters().contact_count, n);
    assert_eq!(world.counters().touching_contact_count, n);
    assert!(
        world.profile().broadphase < 0.05,
        "broadphase with an empty move buffer on 1,000 awake resting spheres must take < 0.05 ms, took {:.4} ms",
        world.profile().broadphase
    );
}

#[test]
fn moving_one_body_outside_its_fat_aabb_queries_only_that_proxy_and_updates_pairs() {
    let n = 250;
    let (mut world, ids) = spheres_scene(n, false);
    for _ in 0..5 {
        world.step(DT);
    }
    assert_eq!(world.broadphase_move_count(), 0);
    assert_eq!(world.counters().contact_count, n);

    // Move sphere 0 on top of sphere 1 so its fat AABB moves and overlaps sphere 1.
    let target_pos = world.body(ids[1]).unwrap().position + Vec3::new(0.0, 0.9, 0.0);
    world.set_body_position(ids[0], target_pos);
    assert_eq!(
        world.broadphase_move_count(),
        1,
        "moving one single-collider body outside its fat AABB must queue exactly 1 proxy in the move buffer"
    );

    world.step(DT);
    // Sphere 0 now has a touching contact with sphere 1.
    let mut contacts_on_0 = world.body_contacts(ids[0]);
    let target_slot = ids[1].index();
    assert!(
        contacts_on_0.any(|c| c.body_a == target_slot || c.body_b == target_slot),
        "moved sphere must form a touching contact with sphere 1"
    );
}
