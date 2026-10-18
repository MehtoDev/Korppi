//! Bodies take any number of colliders, each placed by a local frame, and get their mass,
//! centre of mass and inertia tensor from the colliders' densities, as in Box3D (P3D-13).
//!
//! A Box3D body sums its shapes' mass data (`b3UpdateBodyMassData`): each shape's mass from its
//! density (1000 kg/m³ by default) and volume, its centre and its inertia tensor about it, moved
//! to the body's centre of mass by the parallel axis theorem. The body's origin is where its
//! shapes and joint frames are placed; it moves, turns and is pushed about its centre of mass.
//! Before FIX-3.5 a body had one collider centred on its origin, a mass set on the body (1 kg by
//! default) and a diagonal inertia estimated from it.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_5_bodies.c`):
//! 60 Hz, 4 substeps, no sleep, friction 0.6, a static ground box with its top at y = 0.

use korppi_math::{Mat3, Quat, Vec3};
use korppi_physics::{
    BodyId3d, BodyType, Collider, Frame3d, Joint, PhysicsWorld, RevoluteJoint, RigidBody, Shape,
};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_6};

const DT: f32 = 1.0 / 60.0;

fn body(world: &mut PhysicsWorld, body_type: BodyType, position: Vec3) -> BodyId3d {
    world.create_body(RigidBody::new(body_type, position).with_enable_sleep(false))
}

fn cuboid(half_extents: Vec3) -> Collider {
    Collider::new(Shape::Box { half_extents }).with_friction(0.6)
}

fn capsule(radius: f32, half_height: f32) -> Collider {
    Collider::new(Shape::Capsule {
        radius,
        half_height,
    })
    .with_friction(0.6)
}

fn at(position: Vec3) -> Frame3d {
    Frame3d::at(position)
}

fn ground(world: &mut PhysicsWorld) {
    let ground = body(world, BodyType::Static, Vec3::new(0.0, -0.5, 0.0));
    world.create_collider(ground, cuboid(Vec3::new(20.0, 0.5, 20.0)));
}

fn close(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1.0)
}

/// Checks body `id`'s mass, local centre of mass and inertia tensor against Box3D's.
fn assert_mass(
    world: &PhysicsWorld,
    id: impl Into<BodyId3d>,
    what: &str,
    mass: f32,
    center: Vec3,
    inertia: [[f32; 3]; 3],
) {
    let b = world.body(id).unwrap();
    assert!(
        close(b.mass, mass, 1e-5),
        "{what}: Box3D's mass is {mass}, Korppi's {}",
        b.mass
    );
    assert!(
        (b.local_center - center).length() <= 1e-5,
        "{what}: Box3D's centre of mass is {center:?}, Korppi's {:?}",
        b.local_center
    );
    let expected = Mat3::from_cols_array_2d(&inertia).transpose();
    for (row, column) in (0..3).flat_map(|r| (0..3).map(move |c| (r, c))) {
        let (k, e) = (b.local_inertia.col(column)[row], expected.col(column)[row]);
        assert!(
            (k - e).abs() <= 1e-4 * mass.max(1.0),
            "{what}: Box3D's inertia is {expected:?}, Korppi's {:?}",
            b.local_inertia
        );
    }
}

#[test]
fn shapes_give_their_bodies_box3d_mass() {
    let mut w = PhysicsWorld::new();
    // Box3D, default density 1000 kg/m³: a 1 m box weighs 1000 kg, a 0.5 m ball 523.59882.
    let id = body(&mut w, BodyType::Dynamic, Vec3::ZERO);
    w.create_collider(id, cuboid(Vec3::splat(0.5)));
    let i = 166.666_67;
    assert_mass(
        &w,
        id,
        "box",
        1000.0,
        Vec3::ZERO,
        [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
    );

    let id = body(&mut w, BodyType::Dynamic, Vec3::ZERO);
    w.create_collider(id, Collider::new(Shape::Sphere { radius: 0.5 }));
    let i = 52.359_88;
    assert_mass(
        &w,
        id,
        "ball",
        523.598_8,
        Vec3::ZERO,
        [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
    );

    // Box3D's exact capsule inertia: 43.56506 across, 7.77217 along its axis.
    let id = body(&mut w, BodyType::Dynamic, Vec3::ZERO);
    w.create_collider(id, capsule(0.25, 0.5));
    let (across, along) = (43.565_06, 7.772_17);
    assert_mass(
        &w,
        id,
        "capsule",
        261.799_4,
        Vec3::ZERO,
        [[across, 0.0, 0.0], [0.0, along, 0.0], [0.0, 0.0, across]],
    );

    // The same capsule lying along x, 0.3 above the origin.
    let id = body(&mut w, BodyType::Dynamic, Vec3::ZERO);
    w.create_collider(
        id,
        capsule(0.25, 0.5).with_local_frame(Frame3d::new(
            Vec3::new(0.0, 0.3, 0.0),
            Quat::from_rotation_z(FRAC_PI_2),
        )),
    );
    assert_mass(
        &w,
        id,
        "lying capsule",
        261.799_4,
        Vec3::new(0.0, 0.3, 0.0),
        [[along, 0.0, 0.0], [0.0, across, 0.0], [0.0, 0.0, across]],
    );

    // Three boxes: a 1 m box at the origin, a 1 x 0.5 x 0.5 one beside it and a 0.5 m one at
    // (0, 0, 1) turned 30° about z, density 500: 1312.5 kg with products of inertia.
    let id = body(&mut w, BodyType::Dynamic, Vec3::ZERO);
    w.create_collider(id, cuboid(Vec3::splat(0.5)));
    w.create_collider(
        id,
        cuboid(Vec3::new(0.5, 0.25, 0.25)).with_local_frame(at(Vec3::new(1.0, -0.25, 0.0))),
    );
    w.create_collider(
        id,
        cuboid(Vec3::splat(0.25))
            .with_local_frame(Frame3d::new(
                Vec3::new(0.0, 0.0, 1.0),
                Quat::from_rotation_z(FRAC_PI_6),
            ))
            .with_density(500.0),
    );
    assert_mass(
        &w,
        id,
        "three boxes",
        1312.5,
        Vec3::new(0.190_48, -0.047_62, 0.047_62),
        [
            [251.860_12, 50.595_24, 11.904_76],
            [50.595_24, 457.217_3, -2.976_19],
            [11.904_76, -2.976_19, 410.342_3],
        ],
    );
}

#[test]
fn offset_mass_turns_about_its_centre() {
    // Box3D: a hammer, a steel box head (7800 kg/m³) on a wooden capsule handle (700 kg/m³),
    // weighs 19.98669 kg with its centre 0.32782 above the origin. Thrown at (3, 6, 0) m/s
    // spinning at 10 rad/s, after 1 s its centre is at (3, 3.40238, 0), its origin at
    // (2.82206, 3.67771, 0), and it still spins at 10 rad/s about z.
    let mut w = PhysicsWorld::new();
    let id = body(&mut w, BodyType::Dynamic, Vec3::new(0.0, 2.0, 0.0));
    w.create_collider(id, capsule(0.03, 0.3).with_density(700.0));
    w.create_collider(
        id,
        cuboid(Vec3::new(0.12, 0.05, 0.05))
            .with_local_frame(at(Vec3::new(0.0, 0.35, 0.0)))
            .with_density(7800.0),
    );
    assert_mass(
        &w,
        id,
        "hammer",
        19.986_69,
        Vec3::new(0.0, 0.327_82, 0.0),
        [
            [0.220_12, 0.0, 0.0],
            [0.0, 0.106_02, 0.0],
            [0.0, 0.0, 0.294_37],
        ],
    );
    let b = w.body_mut(id).unwrap();
    b.linear_velocity = Vec3::new(3.0, 6.0, 0.0);
    b.angular_velocity = Vec3::new(0.0, 0.0, 10.0);
    for _ in 0..60 {
        w.step(DT);
    }
    let b = w.body(id).unwrap();
    let center = b.center_of_mass();
    assert!(
        (center - Vec3::new(3.0, 3.402_38, 0.0)).length() <= 1e-3,
        "the centre should fly to Box3D's (3, 3.40238, 0), it is at {center:?}"
    );
    assert!(
        (b.position - Vec3::new(2.822_06, 3.677_71, 0.0)).length() <= 1e-3,
        "the origin should turn about the centre to Box3D's (2.82206, 3.67771, 0), it is at {:?}",
        b.position
    );
    assert!((b.angular_velocity - Vec3::new(0.0, 0.0, 10.0)).length() <= 1e-3);
}

#[test]
fn compound_body_rests_on_its_shapes() {
    // Box3D: a 1 m box with a 1 x 0.5 x 0.5 box beside it, their bottoms level, dropped 0.5 m,
    // rests flat at 0.49996 on two contacts after 2 s.
    let mut w = PhysicsWorld::new();
    ground(&mut w);
    let id = body(&mut w, BodyType::Dynamic, Vec3::new(0.0, 1.0, 0.0));
    w.create_collider(id, cuboid(Vec3::splat(0.5)));
    w.create_collider(
        id,
        cuboid(Vec3::new(0.5, 0.25, 0.25)).with_local_frame(at(Vec3::new(1.0, -0.25, 0.0))),
    );
    for _ in 0..120 {
        w.step(DT);
    }
    let b = w.body(id).unwrap();
    assert!((0.494..=0.501).contains(&b.position.y), "{}", b.position.y);
    let tilt = 2.0 * Vec3::new(b.rotation.x, 0.0, b.rotation.z).length();
    assert!(
        tilt <= 1e-3,
        "the body should rest flat, it tilts {tilt} rad"
    );
    assert_eq!(
        w.counters().contact_count,
        2,
        "one contact per box with the ground"
    );
}

#[test]
fn joints_hold_bodies_at_their_origins() {
    // Box3D: a 0.5 m box placed 1 m beside its body's origin, hinged at that origin about z to a
    // static body, swings from horizontal to -1.13435 rad in 0.5 s, turning at -4.12260 rad/s;
    // the origin stays at the hinge (within 0.6 mm) and the centre is at (0.42299, 2.09317).
    let mut w = PhysicsWorld::new();
    let anchor = body(&mut w, BodyType::Static, Vec3::new(0.0, 3.0, 0.0));
    let id = body(&mut w, BodyType::Dynamic, Vec3::new(0.0, 3.0, 0.0));
    w.create_collider(id, cuboid(Vec3::splat(0.25)).with_local_frame(at(Vec3::X)));
    w.create_joint(Joint::Revolute(RevoluteJoint::new(
        anchor,
        id,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Z,
    )));
    for _ in 0..30 {
        w.step(DT);
    }
    let b = w.body(id).unwrap();
    let angle = 2.0 * b.rotation.z.atan2(b.rotation.w);
    assert!(
        (angle - -1.134_35).abs() <= 2e-3,
        "Box3D swings to -1.13435 rad, Korppi to {angle}"
    );
    assert!(
        (b.angular_velocity.z - -4.122_60).abs() <= 5e-3,
        "{}",
        b.angular_velocity.z
    );
    assert!(
        (b.position - Vec3::new(0.0, 3.0, 0.0)).length() <= 2e-3,
        "the origin should stay at the hinge, it is at {:?}",
        b.position
    );
    let center = b.center_of_mass();
    assert!(
        (center - Vec3::new(0.422_99, 2.093_17, 0.0)).length() <= 3e-3,
        "{center:?}"
    );
}

#[test]
fn forces_turn_bodies_about_their_centre() {
    // Box3D: 100 N along y on a 1 kg body (a 1 m box of density 1 placed 1 m beside the origin)
    // for one step: at the centre of mass it moves at 1.66667 m/s without turning, at the origin
    // it also turns at -10 rad/s about z.
    for (at_origin, spin) in [(false, 0.0), (true, -10.0)] {
        let mut w = PhysicsWorld::new();
        w.set_gravity(Vec3::ZERO);
        let id = body(&mut w, BodyType::Dynamic, Vec3::ZERO);
        w.create_collider(
            id,
            cuboid(Vec3::splat(0.5))
                .with_local_frame(at(Vec3::X))
                .with_density(1.0),
        );
        let b = w.body_mut(id).unwrap();
        let point = if at_origin {
            Vec3::ZERO
        } else {
            b.center_of_mass()
        };
        b.apply_force_at_point(Vec3::new(0.0, 100.0, 0.0), point);
        w.step(DT);
        let b = w.body(id).unwrap();
        assert!((b.linear_velocity - Vec3::new(0.0, 1.666_67, 0.0)).length() <= 1e-4);
        assert!(
            (b.angular_velocity - Vec3::new(0.0, 0.0, spin)).length() <= 1e-3,
            "pushed at the {}: Box3D spins at {spin}, Korppi at {:?}",
            if at_origin { "origin" } else { "centre" },
            b.angular_velocity
        );
    }
}

#[test]
fn a_body_carries_a_sensor_and_a_solid_shape() {
    // Box3D: a static body with a 1 m box and a 0.5 m sensor ball 1.5 m above it; a 0.1 m ball
    // dropped from y = 3 enters the sensor in step 26, leaves it in step 41 and lands on the box
    // (y = 0.59993).
    let mut w = PhysicsWorld::new();
    let base = body(&mut w, BodyType::Static, Vec3::ZERO);
    w.create_collider(base, cuboid(Vec3::splat(0.5)));
    let sensor = w.create_collider(
        base,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_local_frame(at(Vec3::new(0.0, 1.5, 0.0)))
            .with_sensor(true)
            .with_sensor_events(true),
    );
    let ball = body(&mut w, BodyType::Dynamic, Vec3::new(0.0, 3.0, 0.0));
    let ball_collider = w.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.1 })
            .with_friction(0.6)
            .with_sensor_events(true),
    );
    assert_eq!(w.collider_body(sensor), Some(base));
    let (mut begin, mut end) = (None, None);
    for step in 1..=120 {
        w.step(DT);
        let events = w.sensor_events();
        if begin.is_none() && !events.begin.is_empty() {
            assert_eq!(
                (events.begin[0].sensor, events.begin[0].visitor),
                (sensor, ball_collider)
            );
            begin = Some(step);
        }
        if end.is_none() && !events.end.is_empty() {
            end = Some(step);
        }
    }
    assert_eq!(
        (begin, end),
        (Some(26), Some(41)),
        "Box3D: in at step 26, out at 41"
    );
    let y = w.body(ball).unwrap().position.y;
    assert!(
        (0.594..=0.601).contains(&y),
        "the ball should land on the box, y = {y}"
    );
}
