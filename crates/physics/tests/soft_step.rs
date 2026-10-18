//! Contacts follow Box3D's Soft Step: soft push-out, a relax pass, and Poisson restitution
//! (P3D-05, P3D-08).
//!
//! Each Box3D substep integrates velocities, warm starts, pushes overlapping bodies apart with a
//! soft bias capped at the contact speed, integrates positions and relaxes the contacts without
//! bias, which takes the push-out velocity back out; after the substeps two restitution passes
//! bounce the points that approached faster than 1 m/s (`solver.c`, `contact_solver.c`). Before
//! FIX-3.2 Korppi warm started once per step, had no relax pass and added restitution on top of
//! the solved velocity, so contacts gained energy: an e = 0 sphere dropped 2 m rebounded at 30%
//! of its impact speed and e = 0.5 at 81% (audit probe Q5).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_2_soft_step.c`):
//! 60 Hz, 4 substeps, 1 kg shapes, friction 0.6, no sleep, a static ground with its top at
//! y = 0. With Box3D's speculative contacts (FIX-3.3), which stop an impact at the surface, and
//! its feature-id warm starting and contact recycling (FIX-3.4), the drops and the push-out
//! match Box3D to its printed digits (`contact_bookkeeping.rs` follows the push-out step by
//! step); the bounds here are looser.

use korppi_math::Vec3;
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

fn body(body_type: BodyType, position: Vec3) -> RigidBody {
    RigidBody::new(body_type, position).with_enable_sleep(false)
}

fn ground(world: &mut PhysicsWorld) {
    let id = world.create_body(body(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        id,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.6)
        .with_mass(1.0),
    );
}

fn unit_box() -> Shape {
    Shape::Box {
        half_extents: Vec3::splat(0.5),
    }
}

fn ball() -> Shape {
    Shape::Sphere { radius: 0.5 }
}

/// What a dropped shape did: its fastest approach, fastest rebound after the first contact, the
/// apex after the rebound, its lowest centre and where it ended.
#[derive(Debug)]
struct Drop {
    impact: f32,
    rebound: f32,
    apex: f32,
    lowest: f32,
    last: f32,
}

/// Drops `shape` with restitution `e` so that its bottom falls `height` onto the ground, and
/// follows it for 3 s.
fn drop(shape: Shape, e: f32, height: f32) -> Drop {
    let mut world = PhysicsWorld::new();
    ground(&mut world);
    let id = world.create_body(body(BodyType::Dynamic, Vec3::new(0.0, 0.5 + height, 0.0)));
    world.create_collider(
        id,
        Collider::new(shape)
            .with_friction(0.6)
            .with_restitution(e)
            .with_mass(1.0),
    );
    let mut d = Drop {
        impact: 0.0,
        rebound: 0.0,
        apex: f32::MIN,
        lowest: f32::MAX,
        last: 0.0,
    };
    let mut bounced = false;
    for _ in 0..180 {
        world.step(DT);
        let b = world.body(id).unwrap();
        let (vy, y) = (b.linear_velocity.y, b.position.y);
        d.lowest = d.lowest.min(y);
        if !bounced {
            d.impact = d.impact.max(-vy);
            bounced = vy >= 0.0 && d.impact > 0.1;
        }
        if bounced {
            d.rebound = d.rebound.max(vy);
            if vy > 0.0 || d.rebound > 0.0 {
                d.apex = d.apex.max(y);
            }
        }
    }
    d.last = world.body(id).unwrap().position.y;
    d
}

#[test]
fn inelastic_drops_land_dead() {
    // Box3D, from a 2 m drop at 6.21300 m/s: the sphere does not rebound at all, the box at
    // 0.07482 m/s (ratio 0.012, apex 1.8 mm above rest). Probe Q5 saw ratios of 0.30 and 0.32.
    for (what, shape, bound) in [("sphere", ball(), 0.01), ("box", unit_box(), 0.03)] {
        let d = drop(shape, 0.0, 2.0);
        // Box3D hits at 6.21300 m/s, and so does Korppi with speculative contacts.
        assert!(
            (6.1..=6.5).contains(&d.impact),
            "{what}: should hit the ground at about 6.2 m/s, hit at {}",
            d.impact
        );
        let ratio = d.rebound / d.impact;
        assert!(
            ratio <= bound,
            "{what} with e = 0 rebounded at {ratio:.3} of its impact speed ({d:?})"
        );
        assert!(
            (0.49..=0.505).contains(&d.last),
            "{what} should come to rest on the ground, it is at {}",
            d.last
        );
    }
}

#[test]
fn half_elastic_drops_rebound_at_half_speed() {
    // Box3D: rebound ratios 0.50000 (sphere) and 0.49438 (box), apexes 0.98537 and 0.97438.
    // Probe Q5 saw 0.81 and 0.83.
    for (what, shape) in [("sphere", ball()), ("box", unit_box())] {
        let d = drop(shape, 0.5, 2.0);
        let ratio = d.rebound / d.impact;
        assert!(
            (0.45..=0.52).contains(&ratio),
            "{what} with e = 0.5 rebounded at {ratio:.3} of its impact speed ({d:?})"
        );
        // A quarter of the drop's energy: the centre rises about 0.49 m above rest.
        assert!(
            (0.9..=1.03).contains(&d.apex),
            "{what} with e = 0.5 should rise to about 0.98 m, rose to {}",
            d.apex
        );
    }
}

#[test]
fn elastic_drop_rebounds_at_its_impact_speed() {
    // Box3D: e = 1 rebounds at the impact speed (ratio 1.00000) to an apex of 2.45448.
    let d = drop(ball(), 1.0, 2.0);
    let ratio = d.rebound / d.impact;
    assert!(
        (0.97..=1.001).contains(&ratio),
        "an elastic ball should rebound at its impact speed, rebounded at {ratio:.3} ({d:?})"
    );
    assert!(
        d.apex >= 2.35,
        "an elastic ball dropped 2 m should rise back near its 2.5 m start, rose to {}",
        d.apex
    );
}

#[test]
fn elastic_drop_gains_no_energy() {
    // Box3D: the apex, 2.45448, stays below the 2.5 m start.
    let d = drop(ball(), 1.0, 2.0);
    assert!(
        d.apex <= 2.5,
        "an elastic ball dropped from 2.5 m rose to {}, above its start",
        d.apex
    );
}

#[test]
fn slow_impacts_do_not_bounce() {
    // Box3D: a 4 cm drop hits at 0.81750 m/s, below the 1 m/s restitution threshold, and an
    // elastic ball does not rebound.
    let d = drop(ball(), 1.0, 0.04);
    assert!(d.impact < 1.0, "{d:?}");
    assert!(
        d.rebound <= 0.02,
        "an impact below the restitution threshold should not bounce ({d:?})"
    );
}

#[test]
fn overlap_is_pushed_out_gently() {
    // Box3D: a box 0.2 m deep in the ground ends its first step at y = 0.34849 moving up at
    // 0.19471 m/s, the soft push capped at 3 m/s and its velocity relaxed away each substep; it
    // is pushed out at most at 0.48254 m/s and comes to rest at 0.49993 without overshooting.
    let mut world = PhysicsWorld::new();
    ground(&mut world);
    let id = world.create_body(body(BodyType::Dynamic, Vec3::new(0.0, 0.3, 0.0)));
    world.create_collider(
        id,
        Collider::new(unit_box()).with_friction(0.6).with_mass(1.0),
    );
    world.step(DT);
    let first = world.body(id).unwrap();
    assert!(
        (first.position.y - 0.348_49).abs() < 2e-5
            && (first.linear_velocity.y - 0.194_71).abs() < 2e-5,
        "the first push-out step should match Box3D (0.34849, 0.19471): y {}, vy {}",
        first.position.y,
        first.linear_velocity.y
    );
    let (mut fastest, mut highest) = (0.0f32, f32::MIN);
    for _ in 1..120 {
        world.step(DT);
        let b = world.body(id).unwrap();
        fastest = fastest.max(b.linear_velocity.y);
        highest = highest.max(b.position.y);
    }
    assert!(
        fastest <= 0.6,
        "the push-out should be soft, at most about 0.48 m/s, it reached {fastest}"
    );
    assert!(
        highest <= 0.505,
        "the box should not fly off the ground, its centre rose to {highest}"
    );
    let last = world.body(id).unwrap().position.y;
    assert!(
        (0.49..=0.505).contains(&last),
        "the box should rest at 0.5, it is at {last}"
    );
}

#[test]
fn stack_comes_to_rest() {
    // Box3D: five boxes stacked on the ground move at most 0.00032 m/s after 1 s, drift
    // 0.24 mm and sink 5.9 mm in total.
    let mut world = PhysicsWorld::new();
    ground(&mut world);
    let boxes: Vec<_> = (0..5)
        .map(|i| {
            let id =
                world.create_body(body(BodyType::Dynamic, Vec3::new(0.0, 0.5 + i as f32, 0.0)));
            world.create_collider(
                id,
                Collider::new(unit_box()).with_friction(0.6).with_mass(1.0),
            );
            id
        })
        .collect();
    let mut late_speed = 0.0f32;
    for step in 0..180 {
        world.step(DT);
        if step >= 60 {
            for &id in &boxes {
                late_speed = late_speed.max(world.body(id).unwrap().linear_velocity.length());
            }
        }
    }
    assert!(
        late_speed <= 0.01,
        "the stack should come to rest, its boxes still move at {late_speed} m/s"
    );
    for (i, &id) in boxes.iter().enumerate() {
        let p = world.body(id).unwrap().position;
        let drift = Vec3::new(p.x, 0.0, p.z).length();
        let sink = 0.5 + i as f32 - p.y;
        assert!(drift <= 0.01, "box {i} drifted {drift} m sideways");
        assert!(sink <= 0.03, "box {i} sank {sink} m");
    }
}
