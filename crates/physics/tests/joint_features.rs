//! Joints follow Box3D: soft constraints on local frames, with springs, limits, motors and
//! Box3D's joint types (P3D-24, P3D-08).
//!
//! Box3D joints are defined by a local frame on each body (a revolute hinge turns about frame
//! z, a prismatic slider and a wheel's suspension move along frame A's x axis, a wheel spins
//! about frame B's z axis), their rigid rows are soft constraints (60 Hz, damping ratio 2) with
//! position correction only in the biased pass, and every joint type has its springs, limits
//! and motors (`joint.c`, `*_joint.c`). Before FIX-3.2 Korppi's joints used Baumgarte
//! correction, corrected the angular rows of revolute and prismatic joints at velocity level
//! only, welded bodies back to equal rotations, and had no springs, twist limits, spherical
//! motor, distance limits or motor, nor wheel, motor, parallel or filter joints.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_2_joints.c`):
//! 60 Hz, 4 substeps, 1 kg and 1 kg·m² bodies without shapes, no damping, no sleep. Box3D's
//! scenes build rotations with its approximate sine and cosine, so where a body starts turned
//! the bounds allow for its start being up to 0.5% off Korppi's exact one.

use korppi_math::{Quat, Vec3};
use korppi_physics::joint::{
    DistanceJoint, FilterJoint, Frame3d, Joint, MotorJoint, ParallelJoint, PrismaticJoint,
    RevoluteJoint, SphericalJoint, WeldJoint, WheelJoint,
};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

fn world(gravity: f32) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, gravity, 0.0));
    world
}

fn fixed(world: &mut PhysicsWorld, position: Vec3) -> u32 {
    world
        .create_body(RigidBody::new(BodyType::Static, position))
        .into()
}

fn free(world: &mut PhysicsWorld, position: Vec3, rotation: Quat) -> u32 {
    world
        .create_body(
            RigidBody::new(BodyType::Dynamic, position)
                .with_rotation(rotation)
                .with_enable_sleep(false),
        )
        .into()
}

fn run(world: &mut PhysicsWorld, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

fn pose(world: &PhysicsWorld, id: u32) -> (Vec3, Quat) {
    let b = world.body(id).unwrap();
    (b.position, b.rotation)
}

/// The angle of body B's rotation relative to body A's (Box3D `b3GetQuatAngle`).
fn relative_angle(world: &PhysicsWorld, a: u32, b: u32) -> f32 {
    let q = pose(world, a).1.conjugate() * pose(world, b).1;
    2.0 * Vec3::new(q.x, q.y, q.z).length().atan2(q.w.abs())
}

fn world_point(world: &PhysicsWorld, id: u32, local: Vec3) -> Vec3 {
    let (p, q) = pose(world, id);
    p + q * local
}

#[test]
fn rigid_joints_hold_under_load() {
    // Revolute pendulum, Box3D: the pin drifts at most 0.40 mm while the bob drops 1.00039 m,
    // never leaving its swing plane, with a linear separation of 0.12 mm at the end.
    let mut w = world(-9.81);
    let pivot = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let bob = free(&mut w, Vec3::new(1.0, 10.0, 0.0), Quat::IDENTITY);
    let hinge = w.create_joint(Joint::Revolute(RevoluteJoint::new(
        pivot,
        bob,
        Vec3::ZERO,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::Z,
    )));
    let (mut drift, mut drop, mut off_plane) = (0.0f32, 0.0f32, 0.0f32);
    for _ in 0..120 {
        w.step(DT);
        let pin = world_point(&w, bob, Vec3::new(-1.0, 0.0, 0.0));
        drift = drift.max((pin - Vec3::new(0.0, 10.0, 0.0)).length());
        drop = drop.max(10.0 - pose(&w, bob).0.y);
        off_plane = off_plane.max(pose(&w, bob).0.z.abs());
    }
    assert!(drift <= 0.002, "the pendulum's pin drifted {drift} m");
    assert!(
        (0.99..=1.01).contains(&drop),
        "the bob should swing down 1 m, dropped {drop}"
    );
    assert!(
        off_plane <= 1e-4,
        "the bob left its swing plane by {off_plane} m"
    );
    let separation = w.joint_linear_separation(hinge).unwrap();
    assert!(separation <= 0.002, "linear separation {separation}");

    // Spherical chain of five 1 m links swinging from horizontal, Box3D: anchors at most
    // 1.31 mm apart over 3 s.
    let mut w = world(-9.81);
    let mut previous = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let mut previous_anchor = Vec3::ZERO;
    let mut links = Vec::new();
    for i in 0..5 {
        let link = free(&mut w, Vec3::new(0.5 + i as f32, 10.0, 0.0), Quat::IDENTITY);
        w.create_joint(Joint::Spherical(SphericalJoint::new(
            previous,
            link,
            previous_anchor,
            Vec3::new(-0.5, 0.0, 0.0),
        )));
        links.push(link);
        previous = link;
        previous_anchor = Vec3::new(0.5, 0.0, 0.0);
    }
    let mut gap = 0.0f32;
    for _ in 0..180 {
        w.step(DT);
        let mut end = Vec3::new(0.0, 10.0, 0.0);
        for &link in &links {
            gap = gap.max((world_point(&w, link, Vec3::new(-0.5, 0.0, 0.0)) - end).length());
            end = world_point(&w, link, Vec3::new(0.5, 0.0, 0.0));
        }
    }
    assert!(gap <= 0.01, "the chain's anchors came {gap} m apart");

    // Weld cantilever, Box3D: sags 0.10 mm and tilts 0.04 mrad.
    let mut w = world(-9.81);
    let wall = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let beam = free(&mut w, Vec3::new(1.0, 10.0, 0.0), Quat::IDENTITY);
    w.create_joint(Joint::Weld(WeldJoint::new(
        wall,
        beam,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
    )));
    let (mut sag, mut tilt) = (0.0f32, 0.0f32);
    for _ in 0..120 {
        w.step(DT);
        sag = sag.max((pose(&w, beam).0 - Vec3::new(1.0, 10.0, 0.0)).length());
        tilt = tilt.max(relative_angle(&w, wall, beam));
    }
    assert!(
        sag <= 0.002 && tilt <= 0.002,
        "the weld sagged {sag} m and tilted {tilt} rad"
    );
}

#[test]
fn hanging_bob_constraint_force_carries_its_weight() {
    // Box3D: a 1 kg bob hanging still pulls its pivot with 9.81000 N straight down.
    let mut w = world(-9.81);
    let pivot = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let bob = free(&mut w, Vec3::new(0.0, 9.0, 0.0), Quat::IDENTITY);
    let hinge = w.create_joint(Joint::Revolute(RevoluteJoint::new(
        pivot,
        bob,
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::Z,
    )));
    run(&mut w, 60);
    let force = w.joint_constraint_force(hinge).unwrap();
    assert!((9.7..=9.9).contains(&force.y), "constraint force {force:?}");
    assert!(
        force.x.abs() + force.z.abs() <= 0.01,
        "constraint force {force:?}"
    );
    assert!(w.joint_constraint_force(1000).is_none());
}

#[test]
fn joints_keep_their_axes_and_rest_orientation() {
    // Weld at a turned rest pose, Box3D: the 0.7 rad turn changes by 0.00000.
    let mut w = world(0.0);
    let a = fixed(&mut w, Vec3::ZERO);
    let turn = Quat::from_rotation_y(0.7);
    let b = free(&mut w, Vec3::X, turn);
    w.create_joint(Joint::Weld(
        WeldJoint::new(a, b, Vec3::X, Vec3::ZERO)
            .with_local_frames(Frame3d::new(Vec3::X, turn), Frame3d::IDENTITY),
    ));
    run(&mut w, 60);
    let angle = relative_angle(&w, a, b);
    assert!(
        (angle - 0.7).abs() <= 1e-3,
        "the weld should keep the 0.7 rad turn, {angle}"
    );

    // A slider pushed and spun off its axis, Box3D: off its axis at most 0.62 mm, turned at
    // most 1.1 mrad, back to 0 at the end. Before FIX-3.2 its angular rows had no position
    // correction.
    let mut w = world(-9.81);
    let rail = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let slider = free(&mut w, Vec3::new(1.0, 10.0, 0.0), Quat::IDENTITY);
    w.body_mut(slider).unwrap().angular_velocity = Vec3::new(5.0, 3.0, 2.0);
    w.body_mut(slider).unwrap().linear_velocity = Vec3::new(1.0, 2.0, 3.0);
    w.create_joint(Joint::Prismatic(PrismaticJoint::new(
        rail,
        slider,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::X,
    )));
    let (mut off_axis, mut turned) = (0.0f32, 0.0f32);
    for _ in 0..120 {
        w.step(DT);
        let p = pose(&w, slider).0;
        off_axis = off_axis.max(((p.y - 10.0).powi(2) + p.z * p.z).sqrt());
        turned = turned.max(relative_angle(&w, rail, slider));
    }
    assert!(
        off_axis <= 0.005,
        "the slider left its axis by {off_axis} m"
    );
    assert!(turned <= 0.005, "the slider turned {turned} rad");
    assert!(relative_angle(&w, rail, slider) <= 1e-3);

    // A hinge spun across its axis, Box3D: tilts at most 0.69 mrad, back to 0, and keeps its
    // 5 rad/s about the hinge.
    let mut w = world(0.0);
    let base = fixed(&mut w, Vec3::ZERO);
    let wheel = free(&mut w, Vec3::ZERO, Quat::IDENTITY);
    w.body_mut(wheel).unwrap().angular_velocity = Vec3::new(4.0, 0.0, 5.0);
    w.create_joint(Joint::Revolute(RevoluteJoint::new(
        base,
        wheel,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::Z,
    )));
    let tilt = |w: &PhysicsWorld| (pose(w, wheel).1 * Vec3::Z).z.min(1.0).acos();
    let mut worst = 0.0f32;
    for _ in 0..120 {
        w.step(DT);
        worst = worst.max(tilt(&w));
    }
    assert!(worst <= 0.005, "the hinge axis tilted {worst} rad");
    assert!(
        tilt(&w) <= 1e-3,
        "the hinge axis stayed tilted {}",
        tilt(&w)
    );
    let spin = w.body(wheel).unwrap().angular_velocity.z;
    assert!(
        (spin - 5.0).abs() <= 0.01,
        "the hinge spin changed to {spin}"
    );
}

#[test]
fn joint_springs_act_as_springs() {
    // Revolute spring of 1 Hz, undamped, released 0.5 rad from its target. Box3D (from 0.5018
    // rad): -0.46352 rad after half a period and 0.42692 after a period.
    let mut w = world(0.0);
    let base = fixed(&mut w, Vec3::ZERO);
    let wheel = free(&mut w, Vec3::ZERO, Quat::from_rotation_z(0.5));
    let spring =
        RevoluteJoint::new(base, wheel, Vec3::ZERO, Vec3::ZERO, Vec3::Z).with_spring(1.0, 0.0);
    let hinge = w.create_joint(Joint::Revolute(spring));
    let angle = |w: &PhysicsWorld| match w.joint(hinge).unwrap() {
        Joint::Revolute(j) => j.angle(w.body(base).unwrap(), w.body(wheel).unwrap()),
        _ => unreachable!(),
    };
    run(&mut w, 30);
    assert!(
        (-0.5..=-0.42).contains(&angle(&w)),
        "after half a period {}",
        angle(&w)
    );
    run(&mut w, 30);
    assert!(
        (0.38..=0.47).contains(&angle(&w)),
        "after a period {}",
        angle(&w)
    );

    // Vertical 1 Hz prismatic spring with damping 1, Box3D: rests at -0.24849 m, g / (2 pi)^2.
    let mut w = world(-9.81);
    let rail = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let slider = free(&mut w, Vec3::new(0.0, 10.0, 0.0), Quat::IDENTITY);
    w.create_joint(Joint::Prismatic(
        PrismaticJoint::new(rail, slider, Vec3::ZERO, Vec3::ZERO, Vec3::Y).with_spring(1.0, 1.0),
    ));
    run(&mut w, 240);
    let sag = 10.0 - pose(&w, slider).0.y;
    assert!(
        (sag - 0.248_49).abs() <= 0.005,
        "a 1 Hz spring should sag 0.248 m, sagged {sag}"
    );

    // Weld with a 2 Hz angular spring, Box3D: the beam bends 0.03105 rad.
    let mut w = world(-9.81);
    let wall = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let beam = free(&mut w, Vec3::new(1.0, 10.0, 0.0), Quat::IDENTITY);
    w.create_joint(Joint::Weld(
        WeldJoint::new(
            wall,
            beam,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
        )
        .with_angular_spring(2.0, 1.0),
    ));
    run(&mut w, 180);
    let bend = relative_angle(&w, wall, beam);
    assert!(
        (bend - 0.031_05).abs() <= 0.003,
        "the soft weld should bend 0.031 rad, {bend}"
    );

    // Spherical spring of 2 Hz, damping 0.5, from 0.6 rad. Box3D: 0.05884 rad after 0.25 s, 0
    // after 2 s.
    let mut w = world(0.0);
    let base = fixed(&mut w, Vec3::ZERO);
    let tilted = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.6);
    let body = free(&mut w, Vec3::ZERO, tilted);
    w.create_joint(Joint::Spherical(
        SphericalJoint::new(base, body, Vec3::ZERO, Vec3::ZERO).with_spring(
            2.0,
            0.5,
            Quat::IDENTITY,
        ),
    ));
    run(&mut w, 15);
    let early = relative_angle(&w, base, body);
    assert!((early - 0.058_84).abs() <= 0.02, "after 0.25 s {early}");
    run(&mut w, 105);
    assert!(relative_angle(&w, base, body) <= 1e-3);
}

#[test]
fn distance_joint_ropes_and_motors() {
    // A rope: a spring without stiffness limited to 2 m. Box3D: stretches at most to 2.00013 m
    // and hangs at 2.00006.
    let mut w = world(-9.81);
    let anchor = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let weight = free(&mut w, Vec3::new(0.5, 10.0, 0.0), Quat::IDENTITY);
    w.create_joint(Joint::Distance(
        DistanceJoint::new(anchor, weight, Vec3::ZERO, Vec3::ZERO, 2.0)
            .with_spring(0.0, 0.0)
            .with_length_range(0.0, 2.0),
    ));
    let mut longest = 0.0f32;
    for _ in 0..180 {
        w.step(DT);
        longest = longest.max((pose(&w, weight).0 - Vec3::new(0.0, 10.0, 0.0)).length());
    }
    assert!(longest <= 2.01, "the rope stretched to {longest} m");
    let length = (pose(&w, weight).0 - Vec3::new(0.0, 10.0, 0.0)).length();
    assert!(
        (length - 2.0).abs() <= 0.01,
        "the weight should hang at 2 m, {length}"
    );

    // A winch: a 20 N distance motor reeling in at 1 m/s. Box3D: 2.04684 m after 1 s, rising
    // at 1.00000 m/s.
    let mut w = world(-9.81);
    let anchor = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let weight = free(&mut w, Vec3::new(0.0, 7.0, 0.0), Quat::IDENTITY);
    w.create_joint(Joint::Distance(
        DistanceJoint::new(anchor, weight, Vec3::ZERO, Vec3::ZERO, 3.0)
            .with_spring(0.0, 0.0)
            .with_length_range(0.5, 3.0)
            .with_motor(-1.0, 20.0),
    ));
    run(&mut w, 60);
    let length = 10.0 - pose(&w, weight).0.y;
    assert!(
        (length - 2.046_84).abs() <= 0.02,
        "after 1 s of winding the rope is {length} m"
    );
    let climb = w.body(weight).unwrap().linear_velocity.y;
    assert!(
        (climb - 1.0).abs() <= 0.01,
        "the winch should lift at 1 m/s, {climb}"
    );
}

#[test]
fn spherical_twist_limit_and_motor() {
    // Twist limits of ±0.3 rad on a body spun at 5 rad/s about the twist axis. Box3D: stops at
    // 0.30000 rad. Before FIX-3.2 spherical joints had only a cone limit.
    let mut w = world(0.0);
    let base = fixed(&mut w, Vec3::ZERO);
    let body = free(&mut w, Vec3::ZERO, Quat::IDENTITY);
    w.body_mut(body).unwrap().angular_velocity = Vec3::new(0.0, 0.0, 5.0);
    w.create_joint(Joint::Spherical(
        SphericalJoint::new(base, body, Vec3::ZERO, Vec3::ZERO).with_twist_limits(-0.3, 0.3),
    ));
    let mut largest = 0.0f32;
    for _ in 0..60 {
        w.step(DT);
        largest = largest.max(relative_angle(&w, base, body));
    }
    assert!(
        largest <= 0.305,
        "the twist limit let the body turn {largest} rad"
    );
    assert!((relative_angle(&w, base, body) - 0.3).abs() <= 0.005);

    // A 0.5 N·m spherical motor towards 10 rad/s about (1, 2, 2)/3. Box3D: 0.50000 rad/s after
    // 1 s, along the target.
    let mut w = world(0.0);
    let base = fixed(&mut w, Vec3::ZERO);
    let body = free(&mut w, Vec3::ZERO, Quat::IDENTITY);
    let target = Vec3::new(1.0, 2.0, 2.0) * (10.0 / 3.0);
    w.create_joint(Joint::Spherical(
        SphericalJoint::new(base, body, Vec3::ZERO, Vec3::ZERO).with_motor(target, 0.5),
    ));
    run(&mut w, 60);
    let spin = w.body(body).unwrap().angular_velocity;
    assert!(
        (spin.length() - 0.5).abs() <= 0.005,
        "the motor's torque should be capped: {spin:?}"
    );
    assert!(
        spin.normalize().dot(target.normalize()) >= 0.999,
        "{spin:?}"
    );
}

#[test]
fn wheel_joint_suspension_spin_and_steering() {
    // A wheel on a 1 Hz vertical suspension (frame A's x axis turned onto -y), spun by a 2 N·m
    // motor to 10 rad/s. Box3D: sags 0.24849 m, does not drift, spins at 10.00000 rad/s.
    let mut w = world(-9.81);
    let chassis = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let wheel = free(&mut w, Vec3::new(0.0, 9.0, 0.0), Quat::IDENTITY);
    let down = Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2);
    w.create_joint(Joint::Wheel(
        WheelJoint::new(
            chassis,
            wheel,
            Frame3d::new(Vec3::new(0.0, -1.0, 0.0), down),
            Frame3d::new(Vec3::ZERO, down),
        )
        .with_suspension(1.0, 1.0)
        .with_spin_motor(10.0, 2.0),
    ));
    run(&mut w, 240);
    let (p, q) = pose(&w, wheel);
    assert!(
        (9.0 - p.y - 0.248_49).abs() <= 0.005,
        "suspension sag {}",
        9.0 - p.y
    );
    assert!(p.x.abs() + p.z.abs() <= 1e-3, "the wheel drifted to {p:?}");
    let spin = w
        .body(wheel)
        .unwrap()
        .angular_velocity
        .dot(q * down * Vec3::Z);
    assert!((spin - 10.0).abs() <= 0.01, "spin {spin}");

    // Steering by a 4 Hz spring to 0.4 rad about the suspension axis. Box3D: 0.40000 rad.
    let mut w = world(0.0);
    let chassis = fixed(&mut w, Vec3::ZERO);
    let wheel = free(&mut w, Vec3::ZERO, Quat::IDENTITY);
    w.create_joint(Joint::Wheel(
        WheelJoint::new(chassis, wheel, Frame3d::IDENTITY, Frame3d::IDENTITY)
            .without_suspension_spring()
            .with_steering(4.0, 1.0, 100.0, 0.4),
    ));
    run(&mut w, 120);
    let spin_axis = pose(&w, wheel).1 * Vec3::Z;
    let steer = (-spin_axis.y).atan2(spin_axis.z);
    assert!((steer - 0.4).abs() <= 0.005, "steering angle {steer}");
}

#[test]
fn motor_joint_drives_and_holds() {
    // Velocity drive capped at 1 N and 1 N·m towards 2 m/s and 3 rad/s. Box3D: 1.00000 after
    // 1 s, 2.00000 and 3.00000 after 3 s.
    let mut w = world(0.0);
    let base = fixed(&mut w, Vec3::ZERO);
    let body = free(&mut w, Vec3::ZERO, Quat::IDENTITY);
    w.create_joint(Joint::Motor(MotorJoint::new(base, body).with_velocity(
        Vec3::new(2.0, 0.0, 0.0),
        1.0,
        Vec3::new(0.0, 3.0, 0.0),
        1.0,
    )));
    run(&mut w, 60);
    let b = w.body(body).unwrap();
    assert!(
        (b.linear_velocity.x - 1.0).abs() <= 0.01,
        "{:?}",
        b.linear_velocity
    );
    assert!(
        (b.angular_velocity.y - 1.0).abs() <= 0.01,
        "{:?}",
        b.angular_velocity
    );
    run(&mut w, 120);
    let b = w.body(body).unwrap();
    assert!(
        (b.linear_velocity.x - 2.0).abs() <= 0.01,
        "{:?}",
        b.linear_velocity
    );
    assert!(
        (b.angular_velocity.y - 3.0).abs() <= 0.01,
        "{:?}",
        b.angular_velocity
    );

    // Position springs of 2 Hz hold a body against gravity. Box3D: it sags 0.06212 m.
    let mut w = world(-9.81);
    let base = fixed(&mut w, Vec3::new(0.0, 10.0, 0.0));
    let body = free(&mut w, Vec3::new(0.0, 10.0, 0.0), Quat::IDENTITY);
    w.create_joint(Joint::Motor(
        MotorJoint::new(base, body)
            .with_linear_spring(2.0, 1.0, 100.0)
            .with_angular_spring(2.0, 1.0, 100.0),
    ));
    run(&mut w, 180);
    let sag = 10.0 - pose(&w, body).0.y;
    assert!((sag - 0.062_12).abs() <= 0.003, "sag {sag}");
}

#[test]
fn parallel_joint_keeps_a_body_upright() {
    // A 2 Hz parallel joint, damping 1, on a body tilted 0.5 rad and spun at 2 rad/s about its
    // z axis. Box3D: upright after 3 s, still spinning at 1.87603 rad/s about the upright.
    let mut w = world(0.0);
    let base = fixed(&mut w, Vec3::ZERO);
    let body = free(&mut w, Vec3::ZERO, Quat::from_rotation_x(0.5));
    w.body_mut(body).unwrap().angular_velocity = Vec3::new(0.0, 0.0, 2.0);
    w.create_joint(Joint::Parallel(
        ParallelJoint::new(base, body).with_spring(2.0, 1.0),
    ));
    run(&mut w, 180);
    let axis = pose(&w, body).1 * Vec3::Z;
    assert!(axis.z.min(1.0).acos() <= 0.01, "tilt {}", axis.z.acos());
    let spin = w.body(body).unwrap().angular_velocity.dot(axis);
    assert!(
        (spin - 1.876).abs() <= 0.05,
        "spin about the upright {spin}"
    );
}

#[test]
fn filter_joint_disables_collision_between_its_bodies() {
    // Box3D: a box joined to an upper ground by a filter joint falls through it onto the lower
    // ground (0.49993).
    let mut w = world(-9.81);
    let ground = |w: &mut PhysicsWorld, y: f32| {
        let id = fixed(w, Vec3::new(0.0, y, 0.0));
        w.create_collider(
            id,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(5.0, 0.5, 5.0),
            }),
        );
        id
    };
    let upper = ground(&mut w, 2.0);
    ground(&mut w, -0.5);
    let dropped = free(&mut w, Vec3::new(0.0, 4.0, 0.0), Quat::IDENTITY);
    w.create_collider(
        dropped,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        }),
    );
    w.create_joint(Joint::Filter(FilterJoint::new(upper, dropped)));
    run(&mut w, 120);
    let y = pose(&w, dropped).0.y;
    assert!(
        (0.49..=0.51).contains(&y),
        "the box should land on the lower ground, it is at {y}"
    );
}
