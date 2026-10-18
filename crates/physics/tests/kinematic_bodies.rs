//! Kinematic bodies move with their velocity and push dynamic bodies, but nothing moves them
//! (P3D-01).
//!
//! Box3D integrates the position of every awake body, kinematic ones included, with gravity
//! scaled to zero and infinite mass. Before FIX-3.1 only dynamic bodies were integrated: a
//! kinematic box at 1 m/s stayed at the origin (audit probe Q1) while its velocity still
//! dragged the bodies resting on it like a conveyor belt.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_1_bodies.c`),
//! which runs each scene at 60 Hz with 4 substeps, 1 kg shapes, friction 0.6, no damping and
//! no sleep.

use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

fn run(world: &mut PhysicsWorld, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

/// A body without damping or sleep, as in the reference.
fn body(body_type: BodyType, position: Vec3) -> RigidBody {
    let mut body = RigidBody::new(body_type, position).with_enable_sleep(false);
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    body
}

fn add_box(world: &mut PhysicsWorld, body: RigidBody, half_extents: Vec3) -> u32 {
    let id = world.create_body(body);
    world.create_collider(
        id,
        Collider::new(Shape::Box { half_extents })
            .with_friction(0.6)
            .with_mass(1.0),
    );
    id.into()
}

fn position(world: &PhysicsWorld, id: impl Into<korppi_physics::BodyId3d>) -> Vec3 {
    world.body(id).unwrap().position
}

#[test]
fn kinematic_body_moves_with_its_velocity() {
    let mut world = PhysicsWorld::new();
    let mut platform = body(BodyType::Kinematic, Vec3::ZERO);
    platform.linear_velocity = Vec3::X;
    platform.angular_velocity = Vec3::Y;
    let platform = add_box(&mut world, platform, Vec3::splat(0.5));

    run(&mut world, 1.0);

    // Box3D: x = 1.00000, rotated by 1.00000 rad about y, velocity unchanged. Probe Q1 saw
    // the body still at the origin.
    let moved = world.body(platform).unwrap();
    assert!(
        (moved.position - Vec3::X).length() < 1e-4,
        "a kinematic body at 1 m/s should travel 1 m in 1 s, reached {:?}",
        moved.position
    );
    assert_eq!(
        moved.linear_velocity,
        Vec3::X,
        "nothing changes its velocity"
    );
    assert_eq!(moved.angular_velocity, Vec3::Y);
    let (axis, angle) = moved.rotation.to_axis_angle();
    assert!(
        (angle - 1.0).abs() < 1e-3 && axis.y > 0.9999,
        "a kinematic body spinning at 1 rad/s should turn 1 rad, turned {angle} about {axis:?}"
    );
}

#[test]
fn kinematic_body_is_not_moved_by_gravity_forces_or_contacts() {
    let mut world = PhysicsWorld::new();
    add_box(
        &mut world,
        body(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)),
        Vec3::new(5.0, 0.5, 5.0),
    );
    let wall_start = Vec3::new(0.0, 1.0, 0.0);
    let wall = add_box(
        &mut world,
        body(BodyType::Kinematic, wall_start),
        Vec3::new(0.5, 1.0, 2.0),
    );
    world
        .body_mut(wall)
        .unwrap()
        .apply_force(Vec3::new(1000.0, 1000.0, 0.0));
    // A box slid into the wall at 10 m/s and a ball dropped on top of it.
    let mut thrown = body(BodyType::Dynamic, Vec3::new(-2.0, 0.25, 0.0));
    thrown.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
    let thrown = add_box(&mut world, thrown, Vec3::splat(0.25));
    let ball = world.create_body(body(BodyType::Dynamic, Vec3::new(0.0, 3.0, 0.0)));
    world.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.25 })
            .with_friction(0.6)
            .with_mass(1.0),
    );

    let mut drift: f32 = 0.0;
    for _ in 0..120 {
        world.step(DT);
        drift = drift.max((position(&world, wall) - wall_start).length());
    }

    // Box3D: the wall never moves, the box stops against it at x = -0.74988 and the ball
    // rests on it at y = 2.24972.
    assert_eq!(
        drift, 0.0,
        "gravity, a force or a contact moved the kinematic wall"
    );
    assert_eq!(world.body(wall).unwrap().linear_velocity, Vec3::ZERO);
    let box_x = position(&world, thrown).x;
    assert!(
        (-0.76..=-0.74).contains(&box_x),
        "the box should stop against the wall's face at x = -0.75, ended at {box_x}"
    );
    let ball_y = position(&world, ball).y;
    assert!(
        (2.24..=2.26).contains(&ball_y),
        "the ball should rest on the wall's top at y = 2.25, ended at {ball_y}"
    );
}

/// A 0.5 m box resting on a 4 m kinematic platform that starts moving at `velocity` after
/// 0.5 s. Returns the platform and box positions after `seconds`, the box velocity, and the
/// lowest and highest box height over the platform.
fn platform_scene(velocity: Vec3, seconds: f32) -> (Vec3, Vec3, Vec3, f32, f32) {
    let mut world = PhysicsWorld::new();
    let platform = add_box(
        &mut world,
        body(BodyType::Kinematic, Vec3::ZERO),
        Vec3::new(2.0, 0.25, 2.0),
    );
    let cargo = add_box(
        &mut world,
        body(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)),
        Vec3::splat(0.25),
    );
    run(&mut world, 0.5);
    world.body_mut(platform).unwrap().linear_velocity = velocity;
    let (mut lowest, mut highest) = (f32::MAX, f32::MIN);
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
        let height = position(&world, cargo).y - position(&world, platform).y;
        lowest = lowest.min(height);
        highest = highest.max(height);
    }
    (
        position(&world, platform),
        position(&world, cargo),
        world.body(cargo).unwrap().linear_velocity,
        lowest,
        highest,
    )
}

#[test]
fn kinematic_platform_carries_a_resting_box() {
    let (platform, cargo, velocity, lowest, highest) = platform_scene(Vec3::X, 2.0);

    // Box3D: the platform reaches x = 2.00000; friction brings the box to 1 m/s while it lags
    // 0.083 m behind, and it rides 0.4997 m above the platform's centre.
    assert!(
        (platform - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4,
        "the platform should travel 2 m, reached {platform:?}"
    );
    let lag = platform.x - cargo.x;
    assert!(
        (0.07..=0.1).contains(&lag),
        "friction should bring the box up to speed within 0.07-0.1 m of slip (Box3D 0.083), \
         it is {lag} m behind"
    );
    assert!(
        (velocity - Vec3::X).length() < 0.01,
        "the box should move with the platform, moves at {velocity:?}"
    );
    assert!(
        lowest >= 0.49 && highest <= 0.51,
        "the box should ride on the platform, its height over it ranged {lowest}-{highest}"
    );
}

#[test]
fn kinematic_elevator_lifts_a_resting_box() {
    let (platform, cargo, velocity, lowest, highest) = platform_scene(Vec3::Y, 1.0);

    // Box3D: the platform rises to y = 1.00000 and carries the box to y = 1.49972.
    assert!(
        (platform - Vec3::Y).length() < 1e-4,
        "the elevator should rise 1 m, reached {platform:?}"
    );
    assert!(
        (velocity - Vec3::Y).length() < 0.01,
        "the box should rise with the elevator, moves at {velocity:?}"
    );
    assert!(
        lowest >= 0.49 && highest <= 0.51,
        "the box should stay on the elevator, its height over it ranged {lowest}-{highest}"
    );
    assert!(
        cargo.y > 1.48,
        "the box should be lifted, it is at {cargo:?}"
    );
}

#[test]
fn kinematic_body_moves_without_any_dynamic_body_in_the_world() {
    // Nothing needs solving here, but kinematic bodies still move.
    let mut world = PhysicsWorld::new();
    let mut spinner = body(BodyType::Kinematic, Vec3::ZERO).with_rotation(Quat::IDENTITY);
    spinner.linear_velocity = Vec3::new(0.0, 0.0, -2.0);
    let spinner = world.create_body(spinner);

    run(&mut world, 0.5);

    let moved = position(&world, spinner);
    assert!(
        (moved - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-4,
        "a lone kinematic body at 2 m/s should travel 1 m in 0.5 s, reached {moved:?}"
    );
}
