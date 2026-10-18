//! Bodies integrate as in Box2D v3.2, and forces and torques can be applied (P2D-21).
//!
//! Before FIX-2.2 damping was explicit (`v * (1 - c h)`), speeds were unbounded, and the only way
//! to push a body was an impulse. Each scene is a 1 x 1 m box of density 1 (1 kg,
//! I = 1/6 kg·m²), spaced 10 m from the others, without gravity unless stated. Expected values
//! are Box2D's, from `docs/audits/probes/box2d_reference/fix_2_2_integration.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, MotionLocks2d, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};
use std::f32::consts::PI;

const DT: f32 = 1.0 / 60.0;

/// Adds `body` as a unit box at x = 10 * (number of bodies so far).
fn unit_box(world: &mut PhysicsWorld2d, body: RigidBody2d) -> BodyId2d {
    let x = 10.0 * world.body_count() as f32;
    let id = world.create_body(body.with_position(Vec2::new(x, 0.0)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    id
}

fn moving(v: Vec2, w: f32) -> RigidBody2d {
    RigidBody2d::new(BodyType::Dynamic)
        .with_linear_velocity(v)
        .with_angular_velocity(w)
}

fn run(world: &mut PhysicsWorld2d, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

fn assert_close(what: &str, value: f32, expected: f32, tolerance: f32) {
    assert!(
        (value - expected).abs() <= tolerance,
        "{what}: {value:.6}, Box2D v3.2 gives {expected:.6}"
    );
}

#[test]
fn damping_follows_the_pade_approximation() {
    // v / (1 + c h) every substep: 10 m/s and 10 rad/s damped at 2 for 1 s become 1.364609.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let mut body = moving(Vec2::new(10.0, 0.0), 10.0);
    body.linear_damping = 2.0;
    body.angular_damping = 2.0;
    let id = unit_box(&mut world, body);
    run(&mut world, 60);
    let body = world.body(id).unwrap();
    assert_close("linear speed", body.linear_velocity.x, 1.364_609, 1e-4);
    assert_close("angular speed", body.angular_velocity, 1.364_609, 1e-4);
}

#[test]
fn speeds_are_capped() {
    // 400 m/s, and a quarter turn per step unless fast rotation is allowed.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let fast = unit_box(&mut world, moving(Vec2::new(500.0, 0.0), 0.0));
    let spinning = unit_box(&mut world, moving(Vec2::ZERO, 100.0));
    let allowed = unit_box(
        &mut world,
        moving(Vec2::ZERO, 100.0).with_fast_rotation(true),
    );
    run(&mut world, 1);
    assert_close(
        "capped speed",
        world.body(fast).unwrap().linear_velocity.x,
        400.0,
        1e-3,
    );
    assert_close(
        "capped spin",
        world.body(spinning).unwrap().angular_velocity,
        0.25 * PI / DT,
        1e-3,
    );
    assert_close(
        "spin with fast rotation allowed",
        world.body(allowed).unwrap().angular_velocity,
        100.0,
        1e-3,
    );
}

#[test]
fn forces_and_torques_act_for_one_step() {
    // 10 N for one step: 0.166667 m/s; the same force 0.5 m off centre also spins the box at
    // 0.5 rad/s; 3 N·m gives 0.3 rad/s. Forces are cleared after the step.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let pushed = unit_box(&mut world, moving(Vec2::ZERO, 0.0));
    let levered = unit_box(&mut world, moving(Vec2::ZERO, 0.0));
    let twisted = unit_box(&mut world, moving(Vec2::ZERO, 0.0));
    world
        .body_mut(pushed)
        .unwrap()
        .apply_force_to_center(Vec2::new(10.0, 0.0));
    let lever_point = world.body(levered).unwrap().world_center() + Vec2::new(0.5, 0.0);
    world
        .body_mut(levered)
        .unwrap()
        .apply_force(Vec2::new(0.0, 10.0), lever_point);
    world.body_mut(twisted).unwrap().apply_torque(3.0);

    run(&mut world, 1);
    assert_close(
        "pushed",
        world.body(pushed).unwrap().linear_velocity.x,
        0.166_667,
        1e-5,
    );
    let lever = world.body(levered).unwrap();
    assert_close("levered linear", lever.linear_velocity.y, 0.166_667, 1e-5);
    assert_close("levered angular", lever.angular_velocity, 0.5, 1e-5);
    assert_close(
        "twisted",
        world.body(twisted).unwrap().angular_velocity,
        0.3,
        1e-5,
    );

    run(&mut world, 1);
    assert_close(
        "pushed after the force is cleared",
        world.body(pushed).unwrap().linear_velocity.x,
        0.166_667,
        1e-5,
    );
    assert_close(
        "twisted after the torque is cleared",
        world.body(twisted).unwrap().angular_velocity,
        0.3,
        1e-5,
    );
}

#[test]
fn gravity_scale_scales_gravity() {
    let mut world = PhysicsWorld2d::new();
    let mut body = moving(Vec2::ZERO, 0.0);
    body.gravity_scale = 0.5;
    let id = unit_box(&mut world, body);
    run(&mut world, 60);
    let body = world.body(id).unwrap();
    assert_close("fall speed", body.linear_velocity.y, -4.905_006, 1e-4);
    assert_close("fall distance", body.position.y, -2.462_718, 1e-3);
}

#[test]
fn motion_locks_hold_their_axes() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let sliding = unit_box(
        &mut world,
        moving(Vec2::new(3.0, 4.0), 0.0).with_motion_locks(MotionLocks2d {
            linear_x: true,
            ..MotionLocks2d::default()
        }),
    );
    let upright = unit_box(
        &mut world,
        moving(Vec2::ZERO, 0.0).with_motion_locks(MotionLocks2d {
            angular_z: true,
            ..MotionLocks2d::default()
        }),
    );
    let point = world.body(upright).unwrap().world_center() + Vec2::new(0.5, 0.0);
    world
        .body_mut(upright)
        .unwrap()
        .apply_impulse_at_point(Vec2::new(0.0, 1.0), point);
    run(&mut world, 60);

    let body = world.body(sliding).unwrap();
    assert_close("locked x", body.position.x, 0.0, 1e-6);
    assert_close("free y", body.position.y, 4.0, 1e-3);
    assert_close("locked x speed", body.linear_velocity.x, 0.0, 0.0);
    let body = world.body(upright).unwrap();
    assert_close("locked angle", body.rotation.angle(), 0.0, 0.0);
    assert_close("locked spin", body.angular_velocity, 0.0, 0.0);
    assert_close(
        "rise from the off-centre impulse",
        body.position.y,
        1.0,
        1e-3,
    );
}
