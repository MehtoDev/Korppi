//! Surface materials: rolling resistance, tangent speed and mixing callbacks (P2D-25).
//!
//! Before FIX-2.4 a collider had only friction and restitution, mixed one fixed way. As in
//! Box2D v3.2 (`b2SurfaceMaterial`), a collider also has a rolling resistance (scaled by the
//! larger shape radius, it resists relative spin), a tangent speed (conveyor belts) and a user
//! material id that the world's friction and restitution mixers receive. Expected values are
//! Box2D's, from `docs/audits/probes/box2d_reference/fix_2_4_materials.c`.

use korppi_math::Vec2;
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn run(world: &mut PhysicsWorld2d, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

fn add_ground(world: &mut PhysicsWorld2d, collider: Collider2d) {
    let id =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        id,
        Collider2d {
            shape: Shape2d::Polygon(Polygon2d::from_box(50.0, 0.5)),
            ..collider
        },
    );
}

fn add_body(world: &mut PhysicsWorld2d, body: RigidBody2d, collider: Collider2d) -> BodyId2d {
    let id = world.create_body(body);
    world.create_collider(id, collider);
    id
}

fn ball(friction: f32) -> Collider2d {
    Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))).with_friction(friction)
}

fn unit_box(friction: f32) -> Collider2d {
    Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))).with_friction(friction)
}

/// A ball rolling at 4 m/s; its x velocity and x after 2 s.
fn rolling_ball(rolling_resistance: f32) -> (f32, f32) {
    let mut world = PhysicsWorld2d::new();
    add_ground(&mut world, Collider2d::default().with_friction(0.6));
    let id = add_body(
        &mut world,
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 0.5))
            .with_linear_velocity(Vec2::new(4.0, 0.0))
            .with_angular_velocity(-8.0),
        ball(0.6).with_rolling_resistance(rolling_resistance),
    );
    run(&mut world, 2.0);
    let body = world.body(id).unwrap();
    (body.linear_velocity.x, body.position.x)
}

#[test]
fn rolling_resistance_slows_a_rolling_ball() {
    // Box2D v3.2 after 2 s: 3.9969 m/s at x = 7.9939 without resistance, 2.6946 m/s at
    // x = 6.6879 with 0.1.
    let (v, x) = rolling_ball(0.0);
    assert!(
        (v - 3.9969).abs() < 0.01 && (x - 7.9939).abs() < 0.02,
        "free ball: {v:.4} m/s at {x:.4}"
    );
    let (v, x) = rolling_ball(0.1);
    assert!(
        (v - 2.6946).abs() < 0.01 && (x - 6.6879).abs() < 0.02,
        "resisted ball: {v:.4} m/s at {x:.4}"
    );
}

#[test]
fn a_conveyor_carries_a_box() {
    // Ground with tangent speed 2 m/s. Box2D v3.2: the box rides at 2 m/s, at x = 3.6083 after
    // 2 s.
    let mut world = PhysicsWorld2d::new();
    add_ground(
        &mut world,
        Collider2d::default()
            .with_friction(0.6)
            .with_tangent_speed(2.0),
    );
    let id = add_body(
        &mut world,
        RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 0.6)),
        unit_box(0.6),
    );
    run(&mut world, 2.0);
    let body = world.body(id).unwrap();
    assert!(
        (body.linear_velocity.x - 2.0).abs() < 0.01,
        "vx {:.4}",
        body.linear_velocity.x
    );
    assert!(
        (body.position.x - 3.6083).abs() < 0.02,
        "x {:.4}",
        body.position.x
    );
}

fn min_friction(a: f32, _material_a: u64, b: f32, _material_b: u64) -> f32 {
    a.min(b)
}

/// A box (friction 0.1) sliding at 5 m/s on ground (friction 0.9); how far it slid in 6 s.
fn slide(min_mixing: bool) -> f32 {
    let mut world = PhysicsWorld2d::new();
    if min_mixing {
        world.set_friction_mixer(Some(min_friction));
    }
    add_ground(&mut world, Collider2d::default().with_friction(0.9));
    let id = add_body(
        &mut world,
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 0.5))
            .with_linear_velocity(Vec2::new(5.0, 0.0)),
        unit_box(0.1),
    );
    run(&mut world, 6.0);
    world.body(id).unwrap().position.x
}

#[test]
fn the_friction_mixer_replaces_the_geometric_mean() {
    // Box2D v3.2: sqrt(0.1 * 0.9) = 0.3 stops the box at 4.2369 m; min(0.1, 0.9) at 12.7318 m.
    let x = slide(false);
    assert!((x - 4.2369).abs() < 0.02, "slid {x:.4} m");
    let x = slide(true);
    assert!((x - 12.7318).abs() < 0.02, "slid {x:.4} m with min mixing");
}

fn min_restitution(a: f32, material_a: u64, b: f32, material_b: u64) -> f32 {
    // Material 7 never bounces.
    if material_a == 7 || material_b == 7 {
        0.0
    } else {
        a.min(b)
    }
}

/// A ball (restitution 1) dropped from y = 3 onto ground (restitution 0.5); its highest y
/// after the first bounce.
fn bounce(mixer: bool, ground_material: u64) -> f32 {
    let mut world = PhysicsWorld2d::new();
    if mixer {
        world.set_restitution_mixer(Some(min_restitution));
    }
    add_ground(
        &mut world,
        Collider2d::default()
            .with_friction(0.6)
            .with_restitution(0.5)
            .with_user_material_id(ground_material),
    );
    let id = add_body(
        &mut world,
        RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 3.0)),
        ball(0.6).with_restitution(1.0),
    );
    let (mut apex, mut bounced) = (0.0_f32, false);
    for _ in 0..180 {
        world.step(DT);
        let body = world.body(id).unwrap();
        bounced |= body.linear_velocity.y > 0.0;
        if bounced {
            apex = apex.max(body.position.y);
        }
    }
    apex
}

#[test]
fn the_restitution_mixer_sees_material_ids() {
    // Box2D v3.2 apexes: 2.9786 with max mixing, 1.0963 with min, and no bounce (0.4999, at
    // rest) when the ground is material 7.
    let apex = bounce(false, 0);
    assert!((apex - 2.9786).abs() < 0.01, "apex {apex:.4}");
    let apex = bounce(true, 0);
    assert!(
        (apex - 1.0963).abs() < 0.01,
        "apex {apex:.4} with min mixing"
    );
    let apex = bounce(true, 7);
    assert!(apex < 0.6, "material 7 bounced to {apex:.4}");
}
