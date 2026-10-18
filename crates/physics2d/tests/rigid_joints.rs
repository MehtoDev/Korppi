//! Joints without a spring are rigid; springs stay optional (P2D-01).
//!
//! Box2D keeps the softness of a joint's rigid rows (`constraintHertz`, 60 Hz, damping ratio 2)
//! separate from the joint's optional user spring. Before FIX-2.1 a single `hertz` did both, so a
//! joint created with `hertz = 0` applied no impulse at all: a pinned pendulum fell freely
//! (audit probes P1 and P2).

use korppi_math::{Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::joint::{
    DistanceJoint2d, Joint2d, PrismaticJoint2d, RevoluteJoint2d, WeldJoint2d, WheelJoint2d,
};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn dynamic_at(position: Vec2) -> RigidBody2d {
    RigidBody2d::new(BodyType::Dynamic)
        .with_position(position)
        .with_mass(1.0, 1.0)
}

fn static_at(position: Vec2) -> RigidBody2d {
    RigidBody2d::new(BodyType::Static).with_position(position)
}

/// World-space point of a body-local anchor.
fn anchor(world: &PhysicsWorld2d, body: BodyId2d, local: Vec2) -> Vec2 {
    let b = world.body(body).unwrap();
    b.position + b.rotation.rotate(local)
}

/// Steps `seconds` of simulation, returning the largest value `measure` took after any step.
fn run_max(
    world: &mut PhysicsWorld2d,
    seconds: f32,
    measure: impl Fn(&PhysicsWorld2d) -> f32,
) -> f32 {
    let mut max = 0.0f32;
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
        max = max.max(measure(world));
    }
    max
}

/// A pendulum bob 1 m to the side of a static pivot, pinned with the given revolute joint.
fn pendulum(joint: RevoluteJoint2d) -> (PhysicsWorld2d, BodyId2d) {
    let mut world = PhysicsWorld2d::new();
    let pivot = world.create_body(static_at(Vec2::new(0.0, 10.0)));
    let bob = world.create_body(dynamic_at(Vec2::new(1.0, 10.0)));
    world.create_joint(pivot, bob, Joint2d::Revolute(joint));
    (world, bob)
}

fn pivot_drift(world: &PhysicsWorld2d, bob: BodyId2d) -> f32 {
    (anchor(world, bob, Vec2::new(-1.0, 0.0)) - Vec2::new(0.0, 10.0)).length()
}

#[test]
fn revolute_without_spring_keeps_its_pivot() {
    let (mut world, bob) = pendulum(RevoluteJoint2d::new(
        Vec2::ZERO,
        Vec2::new(-1.0, 0.0),
        0.0,
        0.0,
    ));
    let drift = run_max(&mut world, 2.0, |w| pivot_drift(w, bob));
    let swing = run_max(&mut world, 2.0, |w| 10.0 - w.body(bob).unwrap().position.y);
    assert!(
        drift < 0.01,
        "pivot drifted {drift:.4} m (probe P1: free fall)"
    );
    assert!(
        swing > 0.5,
        "the pendulum never swung (largest drop {swing:.3} m)"
    );
}

#[test]
fn weld_without_spring_holds_position_and_angle() {
    // A 1 m beam welded at its left end to a static wall; gravity bends it about the weld.
    let mut world = PhysicsWorld2d::new();
    let wall = world.create_body(static_at(Vec2::new(0.0, 10.0)));
    let beam = world.create_body(dynamic_at(Vec2::new(1.0, 10.0)));
    world.create_joint(
        wall,
        beam,
        Joint2d::Weld(WeldJoint2d::new(
            Vec2::new(0.5, 0.0),
            Vec2::new(-0.5, 0.0),
            0.0,
            0.0,
        )),
    );
    let sag = run_max(&mut world, 2.0, |w| {
        (w.body(beam).unwrap().position - Vec2::new(1.0, 10.0)).length()
    });
    let tilt = run_max(&mut world, 1.0, |w| {
        w.body(beam).unwrap().rotation.angle().abs()
    });
    assert!(
        sag < 0.01,
        "welded beam moved {sag:.4} m (probe P2: free fall)"
    );
    assert!(tilt < 0.01, "welded beam rotated {tilt:.4} rad");
}

#[test]
fn prismatic_without_spring_holds_its_axis_and_angle() {
    // Horizontal slider: gravity pulls across the axis, and the slider starts spinning.
    let mut world = PhysicsWorld2d::new();
    let rail = world.create_body(static_at(Vec2::new(0.0, 10.0)));
    let slider = world.create_body(dynamic_at(Vec2::new(1.0, 10.0)).with_angular_velocity(5.0));
    world.create_joint(
        rail,
        slider,
        Joint2d::Prismatic(PrismaticJoint2d::new(
            Vec2::ZERO,
            Vec2::ZERO,
            Vec2::X,
            0.0,
            0.0,
        )),
    );
    let off_axis = run_max(&mut world, 2.0, |w| {
        (w.body(slider).unwrap().position.y - 10.0).abs()
    });
    let tilt = run_max(&mut world, 1.0, |w| {
        w.body(slider).unwrap().rotation.angle().abs()
    });
    assert!(off_axis < 0.01, "slider left its axis by {off_axis:.4} m");
    assert!(tilt < 0.02, "slider rotated {tilt:.4} rad");
}

#[test]
fn distance_without_spring_keeps_its_length() {
    // A 3 m rigid rod swinging under gravity.
    let mut world = PhysicsWorld2d::new();
    let top = world.create_body(static_at(Vec2::new(0.0, 10.0)));
    let weight = world
        .create_body(dynamic_at(Vec2::new(0.0, 7.0)).with_linear_velocity(Vec2::new(3.0, 0.0)));
    world.create_joint(
        top,
        weight,
        Joint2d::Distance(DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 3.0, 0.0, 0.0)),
    );
    let error = run_max(&mut world, 2.0, |w| {
        ((w.body(weight).unwrap().position - Vec2::new(0.0, 10.0)).length() - 3.0).abs()
    });
    assert!(error < 0.01, "rod length changed by {error:.4} m");
}

#[test]
fn wheel_without_spring_stays_on_its_axis() {
    // Horizontal suspension axis: with no spring the wheel slides freely along it, but gravity
    // must not pull it off the axis.
    let mut world = PhysicsWorld2d::new();
    let chassis = world.create_body(static_at(Vec2::new(0.0, 10.0)));
    let wheel = world.create_body(dynamic_at(Vec2::new(1.0, 10.0)));
    world.create_joint(
        chassis,
        wheel,
        Joint2d::Wheel(WheelJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::X, 0.0, 0.0)),
    );
    let off_axis = run_max(&mut world, 2.0, |w| {
        (w.body(wheel).unwrap().position.y - 10.0).abs()
    });
    assert!(off_axis < 0.01, "wheel left its axis by {off_axis:.4} m");
}

#[test]
fn soft_distance_and_weld_joints_still_act_as_springs() {
    // 2 Hz spring under gravity: static stretch g / (2*pi*2)^2 = 0.062 m, overshooting to ~0.12 m.
    let mut world = PhysicsWorld2d::new();
    let top = world.create_body(static_at(Vec2::new(0.0, 10.0)));
    let weight = world.create_body(dynamic_at(Vec2::new(0.0, 7.0)));
    world.create_joint(
        top,
        weight,
        Joint2d::Distance(DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 3.0, 2.0, 0.1)),
    );
    let stretch = run_max(&mut world, 2.0, |w| {
        (w.body(weight).unwrap().position - Vec2::new(0.0, 10.0)).length() - 3.0
    });
    assert!(
        (0.05..0.5).contains(&stretch),
        "2 Hz distance spring stretched {stretch:.4} m"
    );

    // A 1 Hz weld lets the beam sag far more than a rigid weld would.
    let mut world = PhysicsWorld2d::new();
    let wall = world.create_body(static_at(Vec2::new(0.0, 10.0)));
    let beam = world.create_body(dynamic_at(Vec2::new(1.0, 10.0)));
    world.create_joint(
        wall,
        beam,
        Joint2d::Weld(WeldJoint2d::new(
            Vec2::new(0.5, 0.0),
            Vec2::new(-0.5, 0.0),
            1.0,
            0.5,
        )),
    );
    let sag = run_max(&mut world, 2.0, |w| {
        (w.body(beam).unwrap().position - Vec2::new(1.0, 10.0)).length()
    });
    assert!(sag > 0.05, "1 Hz weld sagged only {sag:.4} m");
}

#[test]
fn revolute_spring_is_off_unless_enabled_and_pulls_toward_its_target() {
    // A free hinge keeps whatever angle it is given, whatever its spring settings.
    let hinge = |joint: RevoluteJoint2d| {
        let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
        // Sleep off, as in the Box2D reference: a shapeless body has no extent, so Box2D lets it
        // sleep however fast it spins.
        world.enable_sleep(false);
        let base = world.create_body(static_at(Vec2::ZERO));
        let arm = world.create_body(dynamic_at(Vec2::ZERO).with_rotation(Rot2::from_angle(1.0)));
        world.create_joint(base, arm, Joint2d::Revolute(joint));
        for _ in 0..180 {
            world.step(DT);
        }
        let arm = world.body(arm).unwrap();
        (arm.rotation.angle(), arm.position.length())
    };

    let (angle, drift) = hinge(RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 2.0, 1.0));
    assert!(
        (angle - 1.0).abs() < 1e-3,
        "a hinge without enable_spring moved from 1.0 to {angle:.4} rad"
    );
    assert!(drift < 0.01);

    let (angle, drift) = hinge(
        RevoluteJoint2d::new(Vec2::ZERO, Vec2::ZERO, 0.0, 0.0)
            .with_spring(2.0, 1.0)
            .with_target_angle(-0.5),
    );
    assert!(
        (angle + 0.5).abs() < 0.02,
        "a 2 Hz spring left the hinge at {angle:.4} rad instead of its -0.5 rad target"
    );
    assert!(drift < 0.01);
}

#[test]
fn prismatic_spring_pulls_toward_its_target_translation() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let rail = world.create_body(static_at(Vec2::ZERO));
    let slider = world.create_body(dynamic_at(Vec2::new(1.0, 0.0)));
    world.create_joint(
        rail,
        slider,
        Joint2d::Prismatic(
            PrismaticJoint2d::new(Vec2::ZERO, Vec2::ZERO, Vec2::X, 0.0, 0.0)
                .with_spring(2.0, 1.0)
                .with_target_translation(0.25),
        ),
    );
    for _ in 0..180 {
        world.step(DT);
    }
    let x = world.body(slider).unwrap().position.x;
    assert!(
        (x - 0.25).abs() < 0.01,
        "a 2 Hz slider spring left the slider at x = {x:.4} instead of 0.25"
    );
}

#[test]
fn constraint_softness_is_configurable_per_joint() {
    // A weight hanging from a pin softened to 1 Hz sags visibly, a pin at the default 60 Hz does
    // not. The softness only slows position correction, it is not a spring: the biased solve uses
    // it and the relax pass then holds the pin rigidly, so the weight sags 0.068 m, not a 1 Hz
    // spring's g / (2*pi)^2 = 0.25 m. Box2D v3.2: 0.06755 m (0.07 mm at 60 Hz).
    let sag = |joint: RevoluteJoint2d| {
        let mut world = PhysicsWorld2d::new();
        let pivot = world.create_body(static_at(Vec2::new(0.0, 10.0)));
        // Sleep off, as in the reference scene: the weight would fall asleep part-way down.
        let mut weight = dynamic_at(Vec2::new(0.0, 9.0));
        weight.enable_sleep = false;
        let weight = world.create_body(weight);
        world.create_joint(pivot, weight, Joint2d::Revolute(joint));
        run_max(&mut world, 2.0, |w| {
            (anchor(w, weight, Vec2::new(0.0, 1.0)) - Vec2::new(0.0, 10.0)).length()
        })
    };
    let pin = RevoluteJoint2d::new(Vec2::ZERO, Vec2::new(0.0, 1.0), 0.0, 0.0);

    let soft = sag(pin.with_constraint_softness(1.0, 0.7));
    assert!(
        (soft - 0.0676).abs() < 0.005,
        "a 1 Hz pin should sag 0.068 m, as in Box2D, it sagged {soft:.4} m"
    );
    let stiff = sag(pin);
    assert!(stiff < 0.001, "the default pin sagged {stiff:.5} m");
}
