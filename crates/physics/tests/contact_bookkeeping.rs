//! Contacts persist between steps, keyed by their body pair, and warm start by feature and by
//! Box3D's contact recycling (P3D-07, P3D-10).
//!
//! Box3D keeps a contact for each pair the broadphase finds, in arrays with a pair set, so a
//! step looks a contact up instead of searching (`contact.c`, `physics_world.c`). A new
//! manifold point takes the impulse of the old point with the same feature id. A contact whose
//! bodies barely moved relative to each other since its last collide is recycled: it keeps its
//! manifold and only updates the separations from the bodies' motion (`b3CollideTask`, on by
//! default within 5 cm and 10 degrees). Before FIX-3.4 every candidate pair searched the last
//! step's manifolds and its points took the impulse of the nearest old point within 0.2 m, so
//! a step's cost grew with the square of the contacts (audit probe Scale: 91 ms for 8,000
//! spheres), and nothing was recycled.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_4_contacts.c`):
//! 60 Hz, 4 substeps, no sleep, friction 0.6, 1 kg boxes of 1 m on a static ground box with its
//! top at y = 0.

use korppi_math::Vec3;
use korppi_physics::{BodyId3d, BodyType, Collider, PhysicsWorld, RigidBody, Shape};
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

fn world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.6)
        .with_mass(1.0),
    );
    world
}

fn add_box(world: &mut PhysicsWorld, position: Vec3) -> BodyId3d {
    let id =
        world.create_body(RigidBody::new(BodyType::Dynamic, position).with_enable_sleep(false));
    world.create_collider(
        id,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.6)
        .with_mass(1.0),
    );
    id
}

#[test]
fn recycled_contacts_follow_box3d() {
    // Box3D: a box 0.2 m deep in the ground is pushed out; from step 5 its contact is mostly
    // recycled. Height, vertical speed and drift per step, and the recycled contacts.
    #[rustfmt::skip]
    let box3d: [(f32, f32, f32, f32, usize); 12] = [
        (0.34849, 0.19471, -0.002010, 0.002453, 0),
        (0.39742, 0.48254, -0.001401, 0.002450, 0),
        (0.43977, 0.31904, -0.000438, 0.002088, 0),
        (0.46493, 0.15554, 0.000526, 0.001727, 0),
        (0.47952, 0.02279, 0.001548, 0.001310, 1),
        (0.48802, 0.01762, 0.002660, 0.000412, 1),
        (0.49300, 0.00569, 0.003241, 0.000017, 0),
        (0.49590, 0.00295, 0.003477, -0.000191, 1),
        (0.49759, 0.00154, 0.003643, -0.000322, 1),
        (0.49857, 0.00080, 0.003722, -0.000386, 1),
        (0.49914, 0.00042, 0.003769, -0.000425, 1),
        (0.49947, 0.00022, 0.003794, -0.000446, 1),
    ];
    let mut w = world();
    let id = add_box(&mut w, Vec3::new(0.0, 0.3, 0.0));
    for (step, &(y, vy, x, z, recycled)) in box3d.iter().enumerate() {
        w.step(DT);
        let b = w.body(id).unwrap();
        let (p, v) = (b.position, b.linear_velocity);
        assert!(
            (p.y - y).abs() <= 2e-5 && (v.y - vy).abs() <= 2e-5,
            "step {}: Box3D is at y {y} moving at {vy}, Korppi at {} moving at {}",
            step + 1,
            p.y,
            v.y
        );
        assert!(
            (p.x - x).abs() <= 2e-5 && (p.z - z).abs() <= 2e-5,
            "step {}: Box3D drifted to ({x}, {z}), Korppi to ({}, {})",
            step + 1,
            p.x,
            p.z
        );
        assert_eq!(
            w.counters().recycled_contact_count,
            recycled,
            "step {}: Box3D recycled {recycled} contacts",
            step + 1
        );
    }
}

#[test]
fn dead_box_drop_rebounds_as_in_box3d() {
    // Box3D: a box dropped 2 m with no restitution rebounds at 0.07482 m/s at most and rests at
    // 0.49993. Before FIX-3.4 Korppi rebounded at 0.08946.
    let mut w = world();
    let id = add_box(&mut w, Vec3::new(0.0, 2.5, 0.0));
    let (mut impact, mut rebound, mut bounced) = (0.0f32, 0.0f32, false);
    for _ in 0..180 {
        w.step(DT);
        let vy = w.body(id).unwrap().linear_velocity.y;
        if !bounced {
            impact = impact.max(-vy);
            bounced = vy >= 0.0 && impact > 0.1;
        }
        if bounced {
            rebound = rebound.max(vy);
        }
    }
    assert!(
        (0.07..=0.08).contains(&rebound),
        "Box3D rebounds at 0.07482 m/s, Korppi at {rebound}"
    );
    let y = w.body(id).unwrap().position.y;
    assert!((0.499..=0.5).contains(&y), "{y}");
}

#[test]
fn resting_stack_recycles_its_contacts() {
    // Box3D: five boxes stacked on the ground for 1 s have 5 contacts, all recycled in the
    // last step.
    let mut w = world();
    for i in 0..5 {
        add_box(&mut w, Vec3::new(0.0, 0.5 + i as f32, 0.0));
    }
    for _ in 0..60 {
        w.step(DT);
    }
    let counters = w.counters();
    assert_eq!(counters.contact_count, 5);
    assert_eq!(counters.recycled_contact_count, 5);
}

/// The time of a step of `n` spheres resting on the ground in a grid, the best of three runs
/// of ten steps.
fn resting_spheres_step_time(n: usize) -> f64 {
    let mut w = PhysicsWorld::new();
    let ground = w.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    w.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(200.0, 0.5, 200.0),
        })
        .with_mass(1.0),
    );
    let side = (n as f32).sqrt().ceil() as usize;
    for i in 0..n {
        let position = Vec3::new((i % side) as f32 * 1.5, 0.49, (i / side) as f32 * 1.5);
        let id =
            w.create_body(RigidBody::new(BodyType::Dynamic, position).with_enable_sleep(false));
        w.create_collider(
            id,
            Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
        );
    }
    for _ in 0..5 {
        w.step(DT);
    }
    (0..3)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..10 {
                w.step(DT);
            }
            start.elapsed().as_secs_f64() / 10.0
        })
        .fold(f64::MAX, f64::min)
}

#[test]
fn contact_bookkeeping_scales_linearly() {
    // Audit probe Scale: with the quadratic matching, four times the resting spheres cost
    // 9 to 13 times as much per step. Box3D's pair set makes it linear: allow 7 times, well
    // below the quadratic growth, for timing noise and the broadphase's log factor.
    let small = resting_spheres_step_time(1000);
    let large = resting_spheres_step_time(4000);
    let ratio = large / small;
    assert!(
        ratio < 7.0,
        "4,000 resting spheres should cost about 4 times 1,000, they cost {ratio:.1} times \
         ({:.2} ms against {:.2} ms per step)",
        large * 1e3,
        small * 1e3
    );
}
