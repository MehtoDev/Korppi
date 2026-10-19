//! Regression tests for Milestone FIX-14.3 (3D Explosions, Inter-Step Events & Contact Queries):
//! - `P3D-32`: `PhysicsWorld::explode` queries `DYNAMIC_TREE`, uses unattenuated core `[0, radius]`
//!   and linear falloff `(radius, radius + falloff]`, and filters by `ExplosionDef3d::mask_bits`.
//! - `P3D-33`: Inter-step `destroy_body`, `destroy_collider`, and `create_joint` buffer
//!   `ContactEndEvent3d` and `SensorEndEvent3d` for the next step, and `sensor_overlaps`,
//!   `body_contacts`, and `collider_contacts` reject stale generational handles and walk per-body
//!   contact edge lists in $O(\text{degree})$.

use korppi_math::Vec3;
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::Collider;
use korppi_physics::joint::{DistanceJoint, Joint};
use korppi_physics::shape::Shape;
use korppi_physics::world::ExplosionDef3d;
use korppi_physics::PhysicsWorld;

const DT: f32 = 1.0 / 60.0;

#[test]
fn p3d_32_explosion_unattenuated_core_and_linear_falloff() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO);

    // Sphere 1 at x = 1.5 (surface at d = 1.0, inside core radius = 5.0 -> scale = 1.0)
    let b_core_near = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(1.5, 0.0, 0.0)));
    w.create_collider(
        b_core_near,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    // Sphere 2 at x = 3.0 (surface at d = 2.5, inside core radius = 5.0 -> scale = 1.0)
    let b_core_far = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(3.0, 0.0, 0.0)));
    w.create_collider(
        b_core_far,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    // Sphere 3 at x = 6.0 (surface at d = 5.5, in falloff (5.0, 10.0] -> scale = (10 - 5.5) / 5 = 0.9)
    let b_falloff = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(6.0, 0.0, 0.0)));
    w.create_collider(
        b_falloff,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    // Sphere 4 at x = 12.0 (surface at d = 11.5 > radius + falloff = 10.0 -> scale = 0.0)
    let b_outside = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(12.0, 0.0, 0.0)));
    w.create_collider(
        b_outside,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    // Sphere 5 at x = 2.0 with category_bits = 2, masked out by mask_bits = 1
    let b_masked = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 0.0, 0.0)));
    w.create_collider(
        b_masked,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_mass(1.0)
            .with_filter(korppi_physics::collider::Filter3d::new(2, u64::MAX)),
    );

    w.explode(&ExplosionDef3d {
        mask_bits: 1,
        position: Vec3::ZERO,
        radius: 5.0,
        falloff: 5.0,
        impulse_per_area: 100.0,
    });

    let v_core_near = w.body(b_core_near).unwrap().linear_velocity.x;
    let v_core_far = w.body(b_core_far).unwrap().linear_velocity.x;
    let v_falloff = w.body(b_falloff).unwrap().linear_velocity.x;
    let v_outside = w.body(b_outside).unwrap().linear_velocity;
    let v_masked = w.body(b_masked).unwrap().linear_velocity;

    // Both core spheres (d <= radius) must receive identical unattenuated impulse (scale = 1.0)
    assert!(
        (v_core_near - v_core_far).abs() < 1e-4,
        "core spheres inside radius must both get scale = 1.0: near={v_core_near}, far={v_core_far}"
    );
    // Sphere in falloff zone at d = 5.5 must get 0.9 of the core velocity
    assert!(
        v_falloff > 0.0 && (v_falloff - 0.9 * v_core_near).abs() < 1e-3,
        "falloff sphere at d=5.5 (radius=5, falloff=5) must get 0.9 * core velocity: got {v_falloff}, core={v_core_near}"
    );
    assert_eq!(v_outside, Vec3::ZERO);
    assert_eq!(
        v_masked,
        Vec3::ZERO,
        "collider filtered out by ExplosionDef3d::mask_bits must receive zero impulse"
    );
}

#[test]
fn p3d_33_inter_step_destroy_emits_contact_and_sensor_end_events() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO);

    let a = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::ZERO));
    let b = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.5, 0.0, 0.0)));
    let s_body = w.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));

    w.create_collider(
        a,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_contact_events(true),
    );
    w.create_collider(
        b,
        Collider::new(Shape::Sphere { radius: 0.5 })
            .with_contact_events(true)
            .with_sensor_events(true),
    );
    w.create_collider(
        s_body,
        Collider::new(Shape::Sphere { radius: 2.0 })
            .with_sensor(true)
            .with_sensor_events(true),
    );

    w.step(DT);
    assert_eq!(w.contact_events().begin.len(), 1);
    assert_eq!(w.sensor_events().begin.len(), 1);

    // Destroy `b` between steps while touching `a` and overlapping `s_body`
    assert!(w.destroy_body(b));
    w.step(DT);

    assert_eq!(
        w.contact_events().end.len(),
        1,
        "destroying touching body between steps must emit ContactEndEvent3d on next step"
    );
    assert_eq!(
        w.sensor_events().end.len(),
        1,
        "destroying overlapping visitor body between steps must emit SensorEndEvent3d on next step"
    );

    // Also verify joint creation with collide_connected = false emits ContactEndEvent3d on next step
    let c = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.5, 0.0, 0.0)));
    w.create_collider(
        c,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_contact_events(true),
    );
    w.step(DT);
    assert_eq!(w.contact_events().begin.len(), 1);

    w.create_joint(Joint::Distance(DistanceJoint::new(
        a,
        c,
        Vec3::ZERO,
        Vec3::ZERO,
        0.5,
    )));
    w.step(DT);
    assert_eq!(
        w.contact_events().end.len(),
        1,
        "creating non-colliding joint between touching bodies must emit ContactEndEvent3d on next step"
    );
}

#[test]
fn p3d_33_stale_handles_in_sensor_overlaps_body_contacts_and_collider_contacts() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO);

    let a = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::ZERO));
    let b = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.5, 0.0, 0.0)));
    let s_body = w.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));

    let col_a = w.create_collider(
        a,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_sensor_events(true),
    );
    let _col_b = w.create_collider(b, Collider::new(Shape::Sphere { radius: 0.5 }));
    let col_s = w.create_collider(
        s_body,
        Collider::new(Shape::Sphere { radius: 2.0 })
            .with_sensor(true)
            .with_sensor_events(true),
    );

    w.step(DT);
    assert_eq!(w.body_contacts(a).count(), 1);
    assert_eq!(w.collider_contacts(col_a).count(), 1);
    assert_eq!(w.sensor_overlaps(col_s).len(), 1);

    // Destroy `a` and `col_s`, then allocate a new body and colliders reusing those slots
    assert!(w.destroy_body(a));
    assert!(w.destroy_collider(col_s));

    let a_new = w.create_body(RigidBody::new(BodyType::Dynamic, Vec3::ZERO));
    let col_a_new = w.create_collider(
        a_new,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_sensor_events(true),
    );
    let col_s_new = w.create_collider(
        s_body,
        Collider::new(Shape::Sphere { radius: 2.0 })
            .with_sensor(true)
            .with_sensor_events(true),
    );
    assert_eq!(a_new.index(), a.index());
    assert_ne!(a_new, a);

    w.step(DT);
    assert_eq!(w.body_contacts(a_new).count(), 1);
    assert_eq!(w.collider_contacts(col_a_new).count(), 1);
    assert_eq!(w.sensor_overlaps(col_s_new).len(), 1);

    // Stale handles must return empty results, never the new occupant's contacts/overlaps
    assert_eq!(
        w.body_contacts(a).count(),
        0,
        "stale BodyId3d must return 0 contacts"
    );
    assert_eq!(
        w.collider_contacts(col_a).count(),
        0,
        "stale ColliderId3d must return 0 contacts"
    );
    assert!(
        w.sensor_overlaps(col_s).is_empty(),
        "stale sensor ColliderId3d must return empty slice"
    );
}
