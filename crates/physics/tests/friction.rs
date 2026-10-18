//! Friction acts at the contact's friction centre, with twist friction about the normal and
//! rolling resistance, as in Box3D (P3D-09).
//!
//! Box3D solves one two-axis friction constraint per manifold at a weighted friction centre,
//! keeps its impulse as a world vector so it warm starts in any tangent basis, resists spin
//! about the normal with twist friction limited by each point's lever arm, and resists the
//! bodies' relative rotation with rolling resistance (`contact_solver.c`). Before FIX-3.2
//! Korppi solved friction per point on a basis that flipped with the normal, had no twist
//! friction and no rolling resistance, so a rolling ball never slowed on flat ground.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_2_soft_step.c`):
//! 60 Hz, 4 substeps, 1 kg shapes (1 m boxes, 0.5 m balls), no sleep, a static ground with its
//! top at y = 0; both colliders of a contact have the friction a test names.

use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

fn body(body_type: BodyType, position: Vec3) -> RigidBody {
    RigidBody::new(body_type, position).with_enable_sleep(false)
}

fn unit_box() -> Shape {
    Shape::Box {
        half_extents: Vec3::splat(0.5),
    }
}

fn ball() -> Shape {
    Shape::Sphere { radius: 0.5 }
}

/// A world with a 40 m ground box whose top is at y = 0.
fn world_with_ground(friction: f32, rolling_resistance: f32) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    let ground = world.create_body(body(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(friction)
        .with_rolling_resistance(rolling_resistance)
        .with_mass(1.0),
    );
    world
}

/// Adds a dynamic body resting on the ground and lets it settle for 0.1 s.
fn settled(world: &mut PhysicsWorld, collider: Collider) -> u32 {
    let id = world.create_body(body(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    world.create_collider(id, collider.with_mass(1.0));
    run(world, 6);
    id.into()
}

fn run(world: &mut PhysicsWorld, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

#[test]
fn sliding_box_stops_where_coulomb_friction_says_in_any_direction() {
    // Box3D: at 5 m/s with friction 0.5 the box slides 2.53839 m (v^2 / 2 mu g = 2.548) and
    // stops at step 65, without turning, along x and diagonally alike.
    for velocity in [
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0).normalize() * 5.0,
    ] {
        let mut world = world_with_ground(0.5, 0.0);
        let id = settled(&mut world, Collider::new(unit_box()).with_friction(0.5));
        let start = world.body(id).unwrap().position;
        world.body_mut(id).unwrap().linear_velocity = velocity;
        let mut stop = None;
        for step in 1..=180 {
            world.step(DT);
            if stop.is_none() && world.body(id).unwrap().linear_velocity.length() < 1e-3 {
                stop = Some(step);
            }
        }
        let b = world.body(id).unwrap();
        let slid = Vec3::new(b.position.x - start.x, 0.0, b.position.z - start.z);
        assert!(
            (2.45..=2.6).contains(&slid.length()),
            "sliding at {velocity:?}: the box should stop after about 2.54 m, slid {}",
            slid.length()
        );
        assert!(
            slid.normalize().dot(velocity.normalize()) > 0.999,
            "the box should slide straight, it went {slid:?}"
        );
        let stop = stop.expect("the box should stop");
        assert!(
            (60..=70).contains(&stop),
            "stopped at step {stop}, Box3D at 65"
        );
        let turn = 2.0 * b.rotation.w.abs().min(1.0).acos();
        assert!(turn < 0.01, "the box should not turn, it turned {turn} rad");
    }
}

/// A box resting on a 30° slope for 1.1 s; how far it moved in the last second and its speed.
fn box_on_slope(friction: f32) -> (f32, f32) {
    let mut world = PhysicsWorld::new();
    let tilt = Quat::from_rotation_z(std::f32::consts::FRAC_PI_6);
    let slope = world.create_body(body(BodyType::Static, Vec3::ZERO).with_rotation(tilt));
    world.create_collider(
        slope,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_friction(friction)
        .with_mass(1.0),
    );
    let id = world.create_body(body(BodyType::Dynamic, tilt * Vec3::Y).with_rotation(tilt));
    world.create_collider(
        id,
        Collider::new(unit_box())
            .with_friction(friction)
            .with_mass(1.0),
    );
    run(&mut world, 6);
    let start = world.body(id).unwrap().position;
    run(&mut world, 60);
    let b = world.body(id).unwrap();
    ((b.position - start).length(), b.linear_velocity.length())
}

#[test]
fn friction_holds_a_box_on_a_slope_steeper_ones_let_it_slide() {
    // Box3D on a 30° slope: friction 0.7 (above tan 30° = 0.577) holds the box still; 0.3 lets
    // it slide 1.44091 m in a second, reaching 2.63008 m/s.
    let (moved, speed) = box_on_slope(0.7);
    assert!(
        moved <= 0.01 && speed <= 0.01,
        "friction 0.7 should hold the box on a 30° slope, it moved {moved} m at {speed} m/s"
    );
    let (moved, speed) = box_on_slope(0.3);
    assert!(
        (1.35..=1.5).contains(&moved),
        "friction 0.3 should let the box slide about 1.44 m in 1 s, it slid {moved} m"
    );
    assert!((2.5..=2.75).contains(&speed), "{speed}");
}

#[test]
fn twist_friction_stops_a_spinning_box() {
    // Box3D: a box spun at 5 rad/s about the vertical stops at step 15, in place.
    let mut world = world_with_ground(0.5, 0.0);
    let id = settled(&mut world, Collider::new(unit_box()).with_friction(0.5));
    world.body_mut(id).unwrap().angular_velocity = Vec3::new(0.0, 5.0, 0.0);
    let mut stop = None;
    for step in 1..=180 {
        world.step(DT);
        if stop.is_none() && world.body(id).unwrap().angular_velocity.y.abs() < 1e-3 {
            stop = Some(step);
        }
    }
    let stop = stop.expect("twist friction should stop the spin");
    assert!(
        (12..=18).contains(&stop),
        "stopped at step {stop}, Box3D at 15"
    );
    let p = world.body(id).unwrap().position;
    assert!(Vec3::new(p.x, 0.0, p.z).length() < 0.01, "{p:?}");
}

/// A ball resting on the ground launched sliding at 2 m/s without spin; its speed and spin
/// after 1 s and its speed and distance after 4 s.
fn rolling_ball(rolling_resistance: f32) -> (f32, f32, f32, f32) {
    let mut world = world_with_ground(0.5, rolling_resistance);
    let id = settled(
        &mut world,
        Collider::new(ball())
            .with_friction(0.5)
            .with_rolling_resistance(rolling_resistance),
    );
    world.body_mut(id).unwrap().linear_velocity = Vec3::new(2.0, 0.0, 0.0);
    run(&mut world, 60);
    let b = world.body(id).unwrap();
    let (speed_1, spin_1) = (b.linear_velocity.x, -b.angular_velocity.z);
    run(&mut world, 180);
    let b = world.body(id).unwrap();
    (speed_1, spin_1, b.linear_velocity.x, b.position.x)
}

#[test]
fn sliding_ball_starts_rolling_and_keeps_rolling() {
    // Box3D: friction brings the ball to rolling at 5/7 of its speed, 1.42838 m/s with
    // 2.85768 rad/s of spin, which it keeps (1.42839 m/s after 4 s, 5.74605 m travelled).
    let (speed_1, spin_1, speed_4, distance) = rolling_ball(0.0);
    assert!(
        (1.41..=1.45).contains(&speed_1),
        "the ball should roll at 1.43 m/s, rolls at {speed_1}"
    );
    assert!(
        (spin_1 - 2.0 * speed_1).abs() < 0.02,
        "rolling without slip: {spin_1} rad/s"
    );
    assert!(
        (speed_4 - speed_1).abs() < 0.01,
        "a rolling ball keeps its speed: {speed_4}"
    );
    assert!((5.6..=5.9).contains(&distance), "{distance}");
}

#[test]
fn rolling_resistance_stops_a_rolling_ball() {
    // Box3D with rolling resistance 0.2 (0.1 m of lever with the 0.5 m radius): 0.03259 m/s after
    // 1 s, stopped after 4 s having rolled 0.77598 m. Before FIX-3.2 nothing slowed it.
    let (speed_1, _, speed_4, distance) = rolling_ball(0.2);
    assert!(
        speed_1 <= 0.1,
        "rolling resistance should nearly stop the ball in 1 s, {speed_1}"
    );
    assert!(
        speed_4.abs() <= 1e-3,
        "the ball should stop, it rolls at {speed_4}"
    );
    assert!((0.7..=0.85).contains(&distance), "{distance}");
}

#[test]
fn only_rolling_resistance_stops_a_ball_spinning_in_place() {
    // Box3D: a single contact point has no lever arm for twist friction, so a ball spun at
    // 5 rad/s about the vertical keeps spinning (5.00000 rad/s after 2 s); rolling resistance
    // stops it (0).
    for (rolling_resistance, expected) in [(0.0, 5.0), (0.2, 0.0)] {
        let mut world = world_with_ground(0.5, rolling_resistance);
        let id = settled(
            &mut world,
            Collider::new(ball())
                .with_friction(0.5)
                .with_rolling_resistance(rolling_resistance),
        );
        world.body_mut(id).unwrap().angular_velocity = Vec3::new(0.0, 5.0, 0.0);
        run(&mut world, 120);
        let spin = world.body(id).unwrap().angular_velocity.y;
        assert!(
            (spin - expected).abs() <= 0.01,
            "rolling resistance {rolling_resistance}: spin {spin}, Box3D {expected}"
        );
    }
}
