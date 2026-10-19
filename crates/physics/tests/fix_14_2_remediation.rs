//! Regression tests for Milestone FIX-14.2 (`crates/physics`) remediating
//! audit findings P3D-30 and P3D-31 from `docs/audits/2026-10-16-physics3d-reaudit.md`.

use korppi_math::{Quat, Vec3};
use korppi_physics::{
    BodyType, Collider, DistanceJoint, Joint, PhysicsWorld, RevoluteJoint, RigidBody, Shape,
};

const DT: f32 = 1.0 / 60.0;

/// P3D-30 (Part 1, Probe Q14): When `enable_warm_starting(false)` is set:
/// 1. Step 1 from a cold start must match `enable_warm_starting(true)` from a cold start, because
///    `enable_warm_starting(false)` only zeroes inter-step impulses at prepare time while
///    substeps `1..sub_steps` still warm-start within the step.
/// 2. Resetting body poses and velocities before Step 2 with `enable_warm_starting(false)` must
///    produce the exact same joint impulse and body velocities as Step 1 (proving that
///    `PrepareJoints` zeroes cached joint impulses when `enable_warm_starting == false`).
#[test]
fn p3d_30_warm_starting_disabled_resets_joint_impulses_and_preserves_substep_warm_start() {
    let run_cold_step = |warm_start: bool| {
        let mut w = PhysicsWorld::new();
        w.set_gravity(Vec3::ZERO);
        w.enable_warm_starting(warm_start);

        let a = w.create_body(
            RigidBody::new(BodyType::Dynamic, Vec3::new(-1.0, 0.0, 0.0))
                .with_linear_velocity(Vec3::new(-2.0, 0.0, 0.0)),
        );
        let b = w.create_body(
            RigidBody::new(BodyType::Dynamic, Vec3::new(1.0, 0.0, 0.0))
                .with_linear_velocity(Vec3::new(2.0, 0.0, 0.0)),
        );
        w.create_collider(a, Collider::new(Shape::Sphere { radius: 0.5 }));
        w.create_collider(b, Collider::new(Shape::Sphere { radius: 0.5 }));
        let jid = w.create_joint(Joint::Distance(
            DistanceJoint::new(a, b, Vec3::ZERO, Vec3::ZERO, 2.0).with_spring(10.0, 0.5),
        ));
        let rev = w.create_joint(Joint::Revolute(
            RevoluteJoint::new(
                a,
                b,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::Z,
            )
            .with_motor(5.0, 100.0),
        ));

        w.step(DT);
        let imp_dist = match w.joint(jid).unwrap() {
            Joint::Distance(dj) => dj.impulse,
            _ => unreachable!(),
        };
        let imp_rev = match w.joint(rev).unwrap() {
            Joint::Revolute(rj) => rj.motor_impulse,
            _ => unreachable!(),
        };
        let vel_a = w.body(a).unwrap().linear_velocity;
        let vel_b = w.body(b).unwrap().linear_velocity;
        (w, a, b, jid, rev, imp_dist, imp_rev, vel_a, vel_b)
    };

    let (_, _, _, _, _, imp_dist_true, imp_rev_true, vel_a_true, vel_b_true) = run_cold_step(true);
    let (mut w_false, a, b, jid, rev, imp_dist_step1, imp_rev_step1, vel_a_step1, vel_b_step1) =
        run_cold_step(false);

    // Cold step 1 with warm starting off must equal cold step 1 with warm starting on,
    // because substeps 1..sub_steps still warm-start within the step.
    assert!(
        (imp_dist_step1 - imp_dist_true).abs() < 1e-6,
        "step 1 distance impulse with warm_start=false ({imp_dist_step1}) must match cold step with warm_start=true ({imp_dist_true})"
    );
    assert!(
        (imp_rev_step1 - imp_rev_true).abs() < 1e-6,
        "step 1 revolute motor impulse with warm_start=false ({imp_rev_step1}) must match cold step with warm_start=true ({imp_rev_true})"
    );
    assert!((vel_a_step1 - vel_a_true).length() < 1e-6);
    assert!((vel_b_step1 - vel_b_true).length() < 1e-6);

    // Reset bodies `a` and `b` to their initial state and step again with `enable_warm_starting(false)`.
    // Because `PrepareJoints` zeroes joint impulses when `enable_warm_starting == false`,
    // step 2 must produce the exact same impulses and velocities as step 1.
    {
        let body_a = w_false.body_mut(a).unwrap();
        body_a.position = Vec3::new(-1.0, 0.0, 0.0);
        body_a.rotation = Quat::IDENTITY;
        body_a.linear_velocity = Vec3::new(-2.0, 0.0, 0.0);
        body_a.angular_velocity = Vec3::ZERO;
    }
    {
        let body_b = w_false.body_mut(b).unwrap();
        body_b.position = Vec3::new(1.0, 0.0, 0.0);
        body_b.rotation = Quat::IDENTITY;
        body_b.linear_velocity = Vec3::new(2.0, 0.0, 0.0);
        body_b.angular_velocity = Vec3::ZERO;
    }
    w_false.step(DT);

    let imp_dist_step2 = match w_false.joint(jid).unwrap() {
        Joint::Distance(dj) => dj.impulse,
        _ => unreachable!(),
    };
    let imp_rev_step2 = match w_false.joint(rev).unwrap() {
        Joint::Revolute(rj) => rj.motor_impulse,
        _ => unreachable!(),
    };
    assert!(
        (imp_dist_step2 - imp_dist_step1).abs() < 1e-6,
        "distance joint impulse in step 2 ({imp_dist_step2}) must equal step 1 ({imp_dist_step1}) when warm starting is disabled"
    );
    assert!(
        (imp_rev_step2 - imp_rev_step1).abs() < 1e-6,
        "revolute motor impulse in step 2 ({imp_rev_step2}) must equal step 1 ({imp_rev_step1}) when warm starting is disabled"
    );
}

/// P3D-30 (Part 2): Overflow constraints (`OVERFLOW_INDEX = 23`) must be solved *before*
/// graph colors `0..colors` so that static-dynamic contacts in the high priority colors have
/// the final word on body velocities during `Solve` and `Relax`.
#[test]
fn p3d_30_overflow_solved_before_colored_static_contacts() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO);
    w.solver_config_mut().sub_steps = 1;

    // Static ground at y = -0.5 (top face at y = 0.0)
    let ground = w.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    w.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_restitution(0.0),
    );

    // Central dynamic hub resting on the static ground at y = 0.5 (radius 0.5 -> touching y = 0.0)
    let hub = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    w.create_collider(
        hub,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_restitution(0.0),
    );

    // Surround `hub` with 22 small dynamic spheres all overlapping `hub` and pushing it downward
    // at -10 m/s. Since `hub` is shared by all 22 dynamic-dynamic contacts, colors 0..20 fill up
    // and the remaining dynamic-dynamic contacts spill into `OVERFLOW_INDEX` (23), while `hub`'s
    // contact with `ground` is in static-dynamic color 22 (`OVERFLOW_INDEX - 1`).
    for i in 0..22 {
        let angle = (i as f32) * std::f32::consts::TAU / 22.0;
        let offset = Vec3::new(angle.cos() * 0.01, 1.249, angle.sin() * 0.01);
        let presser = w.create_body(
            RigidBody::new(BodyType::Dynamic, offset)
                .with_linear_velocity(Vec3::new(0.0, -10.0, 0.0)),
        );
        w.create_collider(
            presser,
            Collider::new(Shape::Sphere { radius: 0.25 })
                .with_restitution(0.0)
                .with_filter(korppi_physics::Filter3d {
                    group_index: -1,
                    ..Default::default()
                }),
        );
    }

    w.step(DT);
    let overflow_contacts = w.constraint_graph().color(23).contact_slots.len();
    let static_priority_contacts = w.constraint_graph().color(22).contact_slots.len();
    assert!(
        overflow_contacts > 0,
        "expected overflow color 23 to have contacts, got {overflow_contacts}"
    );
    assert!(
        static_priority_contacts > 0,
        "expected static-dynamic priority color 22 to have the ground contact, got {static_priority_contacts}"
    );

    // Because overflow (23) is solved BEFORE color 22 (ground), the ground's relax pass runs last
    // and clamps the hub's downward normal velocity against the static floor to >= 0.0.
    let hub_vel = w.body(hub).unwrap().linear_velocity;
    assert!(
        hub_vel.y >= -1e-5,
        "hub resting on static ground must not retain downward velocity from overflow contacts after relax pass, got vy = {}",
        hub_vel.y
    );
}

/// P3D-31 (Probe Q15): 3D sleep evaluation must account for `max_extent * local_omega`,
/// Soft Step `delta_position`/`delta_rotation`, and per-body `sleep_threshold`.
#[test]
fn p3d_31_sleep_accounts_for_extents_delta_position_and_per_body_threshold() {
    // 1. Long rotating bar (half_extent x = 20 m) rotating at omega = 0.03 rad/s:
    // Angular speed 0.03 rad/s is below 0.05, but tip speed is 20 * 0.03 = 0.60 m/s (12x 0.05 m/s).
    // It must NOT fall asleep.
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO);
    let bar = w.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::ZERO)
            .with_angular_velocity(Vec3::new(0.0, 0.03, 0.0)),
    );
    w.create_collider(
        bar,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 0.5),
        }),
    );
    for _ in 0..60 {
        w.step(DT);
    }
    let b = w.body(bar).unwrap();
    assert!(
        !b.is_sleeping,
        "40 m rotating bar with tip speed 0.60 m/s must not fall asleep"
    );
    assert_eq!(b.sleep_time, 0.0);

    // 2. Per-body `sleep_threshold`: raising the bar's sleep threshold above 0.60 m/s allows it to sleep.
    assert_eq!(w.sleep_threshold(bar), Some(0.05));
    assert!(w.set_sleep_threshold(bar, 1.0));
    assert_eq!(w.sleep_threshold(bar), Some(1.0));
    for _ in 0..40 {
        w.step(DT);
    }
    assert!(
        w.body(bar).unwrap().is_sleeping,
        "bar must sleep once its per-body sleep_threshold is raised to 1.0 m/s"
    );

    // 3. Position-correction `delta_position`: a body with zero initial velocity that is pushed by
    // overlapping static geometry must observe `0.5 * inv_dt * max_delta_position`
    // in its sleep velocity and reset `sleep_time` to 0.0 during the correction step.
    let mut w2 = PhysicsWorld::new();
    w2.set_gravity(Vec3::ZERO);
    let floor = w2.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    w2.create_collider(
        floor,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(5.0, 0.5, 5.0),
        }),
    );
    // Deeply overlapping sphere at y = 0.1 (radius 0.5 -> penetration 0.4 m) with heavy linear damping
    // so its end-of-step velocity is damped below 0.05 m/s while Soft Step position push (`delta_position`)
    // moves it upward at ~3 m/s (`contact_speed`).
    let mut body_def = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.1, 0.0));
    body_def.linear_damping = 1000.0;
    let overlapping = w2.create_body(body_def);
    w2.create_collider(overlapping, Collider::new(Shape::Sphere { radius: 0.5 }));
    w2.step(DT);
    let b_ov = w2.body(overlapping).unwrap();
    assert!(
        b_ov.linear_velocity.length() < 0.05,
        "heavy damping keeps end-of-step linear_velocity ({}) below 0.05 m/s",
        b_ov.linear_velocity.length()
    );
    assert_eq!(
        b_ov.sleep_time, 0.0,
        "position correction delta_position must prevent sleep_time from advancing while overlapping"
    );
}
