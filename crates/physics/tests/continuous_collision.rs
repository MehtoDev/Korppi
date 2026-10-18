//! Fast bodies do not tunnel: Box3D's continuous collision (P3D-06).
//!
//! After the solve Box3D checks every dynamic body: one that moved more than its safety factor
//! (0.5) times its smallest extent in a step is swept from its pose at the start of the step to
//! its new pose against static shapes, and a bullet also against kinematic and dynamic bodies;
//! at the first time of impact it is put back to that pose, keeping its velocity (minus the
//! gravity of the lost time), and the next step's speculative contact stops it
//! (`solver.c` `b3SolveContinuous`, `distance.c` `b3TimeOfImpact`). A sensor the sweep passes
//! through sees the body that step. Before FIX-3.3 nothing did this: a 0.1 m ball at 50 m/s
//! passed straight through a 0.1 m wall (audit probe Q6).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_3_continuous.c`):
//! 60 Hz, 4 substeps, no sleep, 1 kg shapes, friction 0.6, walls 4 m square.

use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape, SolverConfig3d};

const DT: f32 = 1.0 / 60.0;

fn world(gravity: f32, continuous: bool) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, gravity, 0.0));
    world.set_solver_config(SolverConfig3d {
        enable_continuous: continuous,
        ..SolverConfig3d::default()
    });
    world
}

fn add(world: &mut PhysicsWorld, body: RigidBody, collider: Collider) -> u32 {
    let id = world.create_body(body.with_enable_sleep(false));
    world.create_collider(id, collider.with_friction(0.6).with_mass(1.0));
    id.into()
}

fn wall(world: &mut PhysicsWorld, half_thickness: f32) {
    add(
        world,
        RigidBody::new(BodyType::Static, Vec3::ZERO),
        Collider::new(Shape::Box {
            half_extents: Vec3::new(half_thickness, 2.0, 2.0),
        }),
    );
}

fn fast(position: Vec3, velocity: Vec3) -> RigidBody {
    let mut body = RigidBody::new(BodyType::Dynamic, position);
    body.linear_velocity = velocity;
    body
}

fn ball(radius: f32) -> Collider {
    Collider::new(Shape::Sphere { radius })
}

/// Steps `steps` times; the largest x body `id` reached.
fn run(world: &mut PhysicsWorld, id: u32, steps: usize) -> f32 {
    let mut farthest = f32::MIN;
    for _ in 0..steps {
        world.step(DT);
        farthest = farthest.max(world.body(id).unwrap().position.x);
    }
    farthest
}

#[test]
fn fast_ball_stops_at_a_thin_wall() {
    // Box3D: a 0.1 m ball at 50 m/s (0.83 m a step) towards a 0.1 m wall stops a slop short of
    // it, closes in to x = -0.13993 and rests against it at -0.14963. Without continuous
    // collision it passes through, to x = 6.33333.
    let mut w = world(0.0, true);
    wall(&mut w, 0.05);
    let id = add(
        &mut w,
        fast(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)),
        ball(0.1),
    );
    let farthest = run(&mut w, id, 10);
    let b = w.body(id).unwrap();
    assert!(
        (-0.16..=-0.13).contains(&farthest),
        "the ball should stop at the wall's face (x = -0.15), it reached {farthest}"
    );
    assert!(
        (-0.16..=-0.145).contains(&b.position.x),
        "the ball should rest against the wall, it is at {}",
        b.position.x
    );
    assert!(
        b.linear_velocity.length() <= 0.01,
        "{:?}",
        b.linear_velocity
    );

    let mut w = world(0.0, false);
    wall(&mut w, 0.05);
    let id = add(
        &mut w,
        fast(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)),
        ball(0.1),
    );
    run(&mut w, id, 10);
    let x = w.body(id).unwrap().position.x;
    assert!(
        x >= 1.0,
        "with continuous collision off the ball tunnels as in Box3D (6.33), it is at {x}"
    );
}

#[test]
fn fast_box_stops_at_a_thin_wall() {
    // Box3D: a 0.2 m box at 100 m/s (1.67 m a step) towards a 2 cm wall stops a slop short of it
    // (x = -0.115 aligned, -0.160 turned) and comes back off it (final x -0.347 and -0.255; the
    // rebound's direction and spin depend on the manifold's point order).
    let turned = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0).normalize(), 0.6);
    for (what, rotation, lowest, highest) in [
        ("aligned", Quat::IDENTITY, -0.12, -0.11),
        ("turned", turned, -0.2, -0.13),
    ] {
        let mut w = world(0.0, true);
        wall(&mut w, 0.01);
        let id = add(
            &mut w,
            fast(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0)).with_rotation(rotation),
            Collider::new(Shape::Box {
                half_extents: Vec3::splat(0.1),
            }),
        );
        let farthest = run(&mut w, id, 10);
        assert!(
            (lowest..=highest).contains(&farthest),
            "{what}: the box should stop at the wall, it reached x = {farthest}"
        );
        let x = w.body(id).unwrap().position.x;
        assert!(
            (-1.0..=highest).contains(&x),
            "{what}: the box should stay on its side of the wall, it is at x = {x}"
        );
    }
}

#[test]
fn fast_ball_stops_on_a_mesh() {
    // Box3D: a 0.1 m ball fired down at 60 m/s onto a two-triangle mesh ground stops on it
    // (lowest y 0.09047) and rests at 0.10493 (Box3D's mesh contacts add a 5 mm rest offset,
    // FIX-3.5).
    let mut w = world(-9.81, true);
    add(
        &mut w,
        RigidBody::new(BodyType::Static, Vec3::ZERO),
        Collider::new(
            Shape::triangle_mesh(
                &[
                    Vec3::new(-20.0, 0.0, -20.0),
                    Vec3::new(-20.0, 0.0, 20.0),
                    Vec3::new(20.0, 0.0, 20.0),
                    Vec3::new(20.0, 0.0, -20.0),
                ],
                &[[0, 1, 2], [0, 2, 3]],
            )
            .expect("triangle mesh builds"),
        ),
    );
    let id = add(
        &mut w,
        fast(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -60.0, 0.0)),
        ball(0.1),
    );
    let mut lowest = f32::MAX;
    for _ in 0..60 {
        w.step(DT);
        lowest = lowest.min(w.body(id).unwrap().position.y);
    }
    assert!(
        lowest >= 0.09,
        "the ball should not pass the mesh, it fell to y = {lowest}"
    );
    let y = w.body(id).unwrap().position.y;
    assert!(
        (0.09..=0.11).contains(&y),
        "the ball should rest on the mesh, y = {y}"
    );
}

#[test]
fn only_bullets_hit_thin_dynamic_bodies() {
    // Box3D: a 5 cm ball at 100 m/s towards a free 4 cm plank 2 m away, gravity off. As a bullet
    // it hits the plank: both move on at 50 m/s, the ball 0.068 m behind the plank's centre. A
    // fast body that is not a bullet sweeps only against static shapes, so it passes through
    // (x = 14.67) and the plank stays still.
    for bullet in [true, false] {
        let mut w = world(0.0, true);
        let plank = add(
            &mut w,
            RigidBody::new(BodyType::Dynamic, Vec3::ZERO),
            Collider::new(Shape::Box {
                half_extents: Vec3::new(0.02, 1.0, 1.0),
            }),
        );
        let id = add(
            &mut w,
            fast(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0)).with_bullet(bullet),
            ball(0.05),
        );
        run(&mut w, id, 10);
        let (ball_body, plank_body) = (w.body(id).unwrap(), w.body(plank).unwrap());
        if bullet {
            assert!(
                ball_body.position.x - plank_body.position.x <= -0.06,
                "the bullet should stay behind the plank: ball {}, plank {}",
                ball_body.position.x,
                plank_body.position.x
            );
            assert!(
                (40.0..=60.0).contains(&plank_body.linear_velocity.x),
                "the bullet should push the plank to about 50 m/s, {}",
                plank_body.linear_velocity.x
            );
            assert!(
                (40.0..=60.0).contains(&ball_body.linear_velocity.x),
                "{}",
                ball_body.linear_velocity.x
            );
        } else {
            assert!(
                ball_body.position.x >= 10.0,
                "a fast ball that is not a bullet passes dynamic bodies, it is at {}",
                ball_body.position.x
            );
            assert!(plank_body.linear_velocity.length() <= 1e-3);
        }
    }
}

#[test]
fn fast_ball_is_seen_by_a_thin_sensor() {
    // Box3D: a 0.1 m ball at 50 m/s through a 0.1 m static sensor wall, both with sensor events,
    // is inside it only during step 3; the sensor reports it then and lets it go in step 4, one
    // begin and one end event. The ball passes on (x = 6.33).
    let mut w = world(0.0, true);
    let sensor = add(
        &mut w,
        RigidBody::new(BodyType::Static, Vec3::ZERO),
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.05, 2.0, 2.0),
        })
        .with_sensor(true)
        .with_sensor_events(true),
    );
    let id = add(
        &mut w,
        fast(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(50.0, 0.0, 0.0)),
        ball(0.1).with_sensor_events(true),
    );
    let sensor_col = w.collider_of(sensor).unwrap();
    let ball_col = w.collider_of(id).unwrap();
    let mut begins = Vec::new();
    let mut ends = Vec::new();
    for step in 1..=10 {
        w.step(DT);
        let events = w.sensor_events();
        begins.extend(events.begin.iter().map(|e| (step, e.sensor, e.visitor)));
        ends.extend(events.end.iter().map(|e| (step, e.sensor, e.visitor)));
    }
    assert_eq!(
        begins,
        [(3, sensor_col, ball_col)],
        "Box3D reports the ball in step 3"
    );
    assert_eq!(
        ends,
        [(4, sensor_col, ball_col)],
        "and lets it go in step 4"
    );
    assert!(w.body(id).unwrap().position.x >= 6.0);
}
