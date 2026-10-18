//! Primary tests for 3D solver sets, persistent islands with split-on-sleep, and solving by
//! constraint graph colour (FIX-8.1 / P3D-26, Probes Q9 and Q10).

use std::time::Instant;

use korppi_math::Vec3;
use korppi_physics::{
    BodyId3d, BodyType, Collider, DistanceJoint, Joint, PhysicsWorld, RigidBody, Shape, AWAKE_SET,
    FIRST_SLEEPING_SET, STATIC_SET,
};

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
fn sleeping_world_moves_bodies_and_contacts_into_sleeping_sets_and_steps_for_free() {
    let n = 1000;
    let (mut world, ids) = spheres_scene(n, true);

    // Step 2 seconds (120 frames) so all spheres settle and transition to sleeping solver sets.
    for _ in 0..120 {
        world.step(DT);
    }

    assert_eq!(
        world.awake_body_count(),
        0,
        "all dynamic bodies must have left the awake solver set"
    );
    assert_eq!(
        world.solver_set(AWAKE_SET).unwrap().bodies.len(),
        0,
        "awake set must hold 0 bodies when the world is asleep"
    );
    assert_eq!(
        world.solver_set(AWAKE_SET).unwrap().contacts.len(),
        0,
        "awake set must hold 0 non-touching contacts when the world is asleep"
    );
    assert_eq!(
        world.constraint_graph().contact_count(),
        0,
        "touching contacts of sleeping islands must leave the active constraint graph"
    );

    for &id in &ids {
        let set_id = world.body_solver_set(id);
        assert!(
            set_id >= FIRST_SLEEPING_SET,
            "sleeping sphere {id:?} must live in a sleeping solver set (>= {FIRST_SLEEPING_SET}), got {set_id}"
        );
        let set = world
            .solver_set(set_id)
            .expect("sleeping solver set must exist");
        assert_eq!(set.bodies, vec![id.index()]);
        assert_eq!(
            set.contacts.len(),
            1,
            "sleeping set must own the sphere's touching floor contact"
        );
    }

    // Probe Q9: stepping a sleeping world must do no per-body or per-contact work.
    let steps = 50;
    let start = Instant::now();
    for _ in 0..steps {
        world.step(DT);
    }
    let ms_per_step = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    assert_eq!(
        world.body_events().move_events.len(),
        0,
        "a sleeping step must emit 0 body move events"
    );
    assert!(
        ms_per_step < 0.05,
        "a sleeping 1,000-body world must step in < 0.05 ms/step (Box3D: 0.000 ms), took {ms_per_step:.4} ms/step"
    );
}

#[test]
fn waking_one_body_in_a_sleeping_world_wakes_and_costs_only_its_island() {
    let n = 1000;
    let (mut world, ids) = spheres_scene(n, true);
    for _ in 0..120 {
        world.step(DT);
    }
    assert_eq!(world.awake_body_count(), 0);

    // Push a single sphere via body_mut (as in Probe Q10).
    let target = ids[n / 2];
    if let Some(b) = world.body_mut(target) {
        b.wake_up();
        b.apply_linear_impulse(Vec3::new(500.0, 0.0, 0.0));
    }

    let start = Instant::now();
    world.step(DT);
    let wake_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Only the pushed sphere (and any sphere it immediately collided with, none on step 1) is awake.
    assert_eq!(
        world.awake_body_count(),
        1,
        "only the woken island must enter the awake solver set"
    );
    assert_eq!(world.body_solver_set(target), AWAKE_SET);
    assert_eq!(
        world.body_events().move_events.len(),
        1,
        "only the 1 awake body may emit a move event"
    );
    assert_eq!(world.body_events().move_events[0].body, target);

    // The other 999 spheres remain in their sleeping solver sets.
    let still_asleep = ids
        .iter()
        .filter(|&&id| id != target && world.body_solver_set(id) >= FIRST_SLEEPING_SET)
        .count();
    assert_eq!(still_asleep, n - 1);

    assert!(
        wake_ms < 1.0,
        "waking 1 island in 1,000 sleeping spheres must cost only that island (< 1.0 ms), took {wake_ms:.3} ms"
    );
}

#[test]
fn persistent_islands_merge_when_linked_and_split_before_sleeping() {
    let mut world = PhysicsWorld::new();
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        })
        .with_friction(0.8),
    );
    assert_eq!(world.body_solver_set(ground), STATIC_SET);

    // Two dynamic boxes Resting on the floor, linked by a distance joint so they merge into one island.
    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(-2.0, 0.5, 0.0)));
    world.create_collider(
        b1,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.8),
    );
    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 0.5, 0.0)));
    world.create_collider(
        b2,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.8),
    );

    let joint = world.create_joint(Joint::Distance(DistanceJoint::new(
        b1,
        b2,
        Vec3::ZERO,
        Vec3::ZERO,
        4.0,
    )));

    // Both bodies must be merged into the same persistent island immediately.
    assert_eq!(
        world.body_island(b1),
        world.body_island(b2),
        "joint must merge b1 and b2 into one persistent island"
    );

    world.step(DT);
    assert_eq!(world.body_island(b1), world.body_island(b2));

    // Destroy the joint so the island has a removed constraint, then keep b2 moving while b1 rests.
    world.destroy_joint(joint);
    for _ in 0..90 {
        world.body_mut(b2).unwrap().linear_velocity = Vec3::new(1.0, 0.0, 0.0);
        world.step(DT);
    }

    // b1 wanted to sleep, which triggered `split_island` on the formerly merged island:
    // b1 is now asleep in its own sleeping solver set while b2 remains awake!
    assert_ne!(
        world.body_island(b1),
        world.body_island(b2),
        "island must split so resting b1 separates from moving b2"
    );
    assert!(
        world.body(b1).unwrap().is_sleeping,
        "resting b1 must fall asleep after splitting from moving b2"
    );
    assert!(world.body_solver_set(b1) >= FIRST_SLEEPING_SET);
    assert!(!world.body(b2).unwrap().is_sleeping);
    assert_eq!(world.body_solver_set(b2), AWAKE_SET);
}
