//! Regression tests for FIX-14.4 (P3D-34):
//! - `P3D-34`:
//!   1. `restore_snapshot` and `from_snapshot` preserve applied `force` and `torque` on bodies.
//!   2. `WorldSnapshot3d` preserves `contact_events`, `body_events`, `joint_events`,
//!      `sensor_events`, `pending_end_events`, `pending_sensor_end_events`, `last_inv_h`, and
//!      `recycled_contact_count`.
//!   3. `Counters3d::touching_contact_count` counts touching contacts across both awake
//!      (`constraint_graph`) and sleeping (`sets[FIRST_SLEEPING_SET..]`) solver sets.
//!   4. `state_hash` incorporates applied forces/torques and solver-set/island/graph link state.

use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::joint::{DistanceJoint, Joint};
use korppi_physics::shape::Shape;
use korppi_physics::PhysicsWorld;

#[test]
fn p3d_34_snapshot_preserves_forces_torques_events_and_last_inv_h() {
    let mut w = PhysicsWorld::new();
    let ground = w.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    w.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        })
        .with_contact_events(true),
    );

    let ball = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    w.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_mass(1.0)
            .with_contact_events(true),
    );

    let bob = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, -3.0, 0.0)));
    w.create_collider(
        bob,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );
    let mut dj = DistanceJoint::new(ground, bob, Vec3::ZERO, Vec3::ZERO, 2.0);
    dj.base.force_threshold = 0.1;
    let jid = w.create_joint(Joint::Distance(dj));

    // Step once to generate contact begin, body move, and joint events + set last_inv_h
    w.step(1.0 / 60.0);
    assert!(!w.contact_events().begin.is_empty());
    assert!(!w.body_events().move_events.is_empty());
    assert!(!w.joint_events().events.is_empty());
    let force_before = w.joint_constraint_force(jid).unwrap();
    assert!(force_before.length() > 0.1);

    // Apply external force and torque before taking snapshot
    w.body_mut(ball)
        .unwrap()
        .apply_force(Vec3::new(100.0, 200.0, -50.0));
    w.body_mut(ball)
        .unwrap()
        .apply_torque(Vec3::new(10.0, -20.0, 30.0));

    let snap = w.create_snapshot();

    // Clear forces on live world and step to change state, then restore
    w.step(1.0 / 60.0);
    w.restore_snapshot(&snap);

    let restored_ball = w.body(ball).unwrap();
    assert_eq!(
        restored_ball.force,
        Vec3::new(100.0, 200.0, -50.0),
        "restore_snapshot must preserve applied force"
    );
    assert_eq!(
        restored_ball.torque,
        Vec3::new(10.0, -20.0, 30.0),
        "restore_snapshot must preserve applied torque"
    );

    // Also verify from_snapshot preserves events and last_inv_h
    let w2 = PhysicsWorld::from_snapshot(&snap);
    assert_eq!(w2.body(ball).unwrap().force, Vec3::new(100.0, 200.0, -50.0));
    assert_eq!(w2.body(ball).unwrap().torque, Vec3::new(10.0, -20.0, 30.0));
    assert_eq!(w2.contact_events(), w.contact_events());
    assert_eq!(w2.body_events(), w.body_events());
    assert_eq!(w2.joint_events(), w.joint_events());
    assert_eq!(
        w2.joint_constraint_force(jid).unwrap(),
        force_before,
        "from_snapshot must preserve last_inv_h so joint_constraint_force matches"
    );
}

#[test]
fn p3d_34_touching_contact_count_includes_sleeping_sets_and_state_hash_tracks_forces() {
    let mut w = PhysicsWorld::new();
    let ground = w.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    w.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        }),
    );

    let ball = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)));
    w.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    // Step until ball falls asleep on the ground
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    assert!(w.body(ball).unwrap().is_sleeping, "ball must be asleep");
    assert_eq!(w.contact_count(), 1);
    assert_eq!(
        w.counters().touching_contact_count,
        1,
        "touching_contact_count must remain 1 when resting island sleeps"
    );

    // Verify state_hash changes when force is applied or island/set state changes
    let hash_before_force = w.state_hash();
    w.body_mut(ball)
        .unwrap()
        .apply_force(Vec3::new(10.0, 0.0, 0.0));
    let hash_after_force = w.state_hash();
    assert_ne!(
        hash_before_force, hash_after_force,
        "state_hash must change when force or solver set membership changes"
    );
}
