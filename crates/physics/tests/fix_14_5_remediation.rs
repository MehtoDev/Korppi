//! Regression tests for FIX-14.5 (`P3D-35`, probe `Q19`):
//! 1. Non-finite or non-positive `dt` (`NaN`, `INFINITY`, `NEG_INFINITY`, `-1.0`, `0.0`) is
//!    rejected as a no-op in `PhysicsWorld::step` and `solve_contacts_with_buffers`.
//! 2. `PhysicsWorld::create_joint` rejects self-joints (`body_a == body_b`).
//! 3. `PhysicsWorld::disable_body`, `enable_body`, `is_body_enabled`, `set_body_type`, and
//!    `set_target_transform` manage solver sets, islands, broadphase proxies, contacts, and joints.
//! 4. Direct mutations via `PhysicsWorld::body_mut` (`position`, `rotation`, `body_type`,
//!    `enabled`, `is_sleeping`, `mass`) are reconciled before `create_joint`, `create_snapshot`,
//!    `state_hash`, and spatial queries (`ray_cast`, `cast_ray`, `query_aabb`, `overlap_aabb`,
//!    `overlap_shape`, `cast_shape`, `collide_mover`) even before the next `step()`.

use korppi_math::{Aabb, Quat, Ray, Vec3};
use korppi_physics::distance::ShapeProxy;
use korppi_physics::mover::CapsuleMover;
use korppi_physics::{
    solve_contacts_with_buffers, BodyType, Collider, DistanceJoint, Joint, PhysicsWorld,
    QueryFilter3d, RigidBody, Shape, SolverConfig3d, SolverScratch3d, Transform3d, AWAKE_SET,
    DISABLED_SET, STATIC_SET,
};

#[test]
fn non_finite_and_non_positive_dt_are_rejected() {
    let mut world = PhysicsWorld::new();
    let b = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 5.0, 0.0))
            .with_linear_velocity(Vec3::new(1.0, 2.0, 3.0)),
    );
    world.create_collider(b, Collider::new(Shape::Sphere { radius: 0.5 }));
    let initial_hash = world.state_hash();

    for bad_dt in [0.0, -1.0 / 60.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        world.step(bad_dt);
        assert_eq!(world.step_index(), 0, "step_index advanced on dt={bad_dt}");
        let body = world.body(b).unwrap();
        assert!(
            body.position.x.is_finite()
                && body.position.y.is_finite()
                && body.position.z.is_finite(),
            "non-finite position after dt={bad_dt}: {:?}",
            body.position
        );
        assert_eq!(world.state_hash(), initial_hash);
    }

    // Also verify standalone solve_contacts_with_buffers
    let mut bodies = vec![RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 5.0, 0.0))];
    let mut manifolds = Vec::new();
    let mut joints = Vec::new();
    let config = SolverConfig3d::default();
    let mut scratch = SolverScratch3d::default();
    for bad_dt in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        solve_contacts_with_buffers(
            &mut bodies,
            &mut manifolds,
            &mut joints,
            Vec3::new(0.0, -9.81, 0.0),
            bad_dt,
            &config,
            &mut scratch,
        );
        assert_eq!(bodies[0].position, Vec3::new(0.0, 5.0, 0.0));
        assert!(bodies[0].linear_velocity.x.is_finite());
    }
}

#[test]
#[should_panic(expected = "create_joint requires two distinct bodies")]
fn create_joint_rejects_self_joint() {
    let mut world = PhysicsWorld::new();
    let b = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::ZERO));
    world.create_joint(Joint::Distance(DistanceJoint::new(
        b.index(),
        b.index(),
        Vec3::ZERO,
        Vec3::Y,
        1.0,
    )));
}

#[test]
fn disable_enable_set_body_type_and_set_target_transform_lifecycle() {
    let mut world = PhysicsWorld::new();

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        }),
    );

    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.9, 0.0)));
    world.create_collider(
        b1,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_contact_events(true),
    );
    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 2.5, 0.0)));
    world.create_collider(b2, Collider::new(Shape::Sphere { radius: 0.5 }));

    let j = world.create_joint(Joint::Distance(DistanceJoint::new(
        b1.index(),
        b2.index(),
        Vec3::ZERO,
        Vec3::ZERO,
        1.6,
    )));

    world.step(1.0 / 60.0);
    assert!(world.is_body_enabled(b1));
    assert!(world.contact_count() > 0);

    // Disable b1: its contacts are destroyed, its proxy is removed from broadphase, and its joint
    // moves to DISABLED_SET.
    let pos_when_disabled = world.body(b1).unwrap().position;
    assert!(world.disable_body(b1));
    assert!(!world.is_body_enabled(b1));
    assert_eq!(world.body_solver_set(b1), DISABLED_SET);
    assert_eq!(world.contact_count(), 0);
    let ray = Ray::new(Vec3::new(-5.0, pos_when_disabled.y, 0.0), Vec3::X);
    assert!(
        world.ray_cast(&ray, 10.0).is_none(),
        "disabled body should not be hit by ray_cast"
    );

    world.step(1.0 / 60.0);
    assert_eq!(world.body(b1).unwrap().position, pos_when_disabled);

    // Re-enable b1: proxy and joint are restored, and stepping re-establishes contacts.
    assert!(world.enable_body(b1));
    assert!(world.is_body_enabled(b1));
    assert_eq!(world.body_solver_set(b1), AWAKE_SET);
    assert!(
        world.ray_cast(&ray, 10.0).is_some(),
        "re-enabled body should be hit by ray_cast"
    );
    let _ = j;

    // Switch b2 from Dynamic -> Kinematic and drive it with set_target_transform.
    assert!(world.set_body_type(b2, BodyType::Kinematic));
    assert_eq!(world.body(b2).unwrap().body_type, BodyType::Kinematic);
    assert_eq!(world.body(b2).unwrap().inv_mass, 0.0);

    let target_pos = Vec3::new(3.0, 4.0, -2.0);
    let target_rot = Quat::from_axis_angle(Vec3::Y, 0.5);
    let dt = 1.0 / 60.0;
    assert!(world.set_target_transform(b2, Transform3d::new(target_pos, target_rot), dt, true));
    world.step(dt);
    let b2_after = world.body(b2).unwrap();
    assert!(
        (b2_after.position - target_pos).length() < 1e-4,
        "kinematic body did not reach target position: got {:?}, expected {:?}",
        b2_after.position,
        target_pos
    );
    assert!(
        b2_after.rotation.dot(target_rot).abs() > 0.999,
        "kinematic body did not reach target rotation: got {:?}, expected {:?}",
        b2_after.rotation,
        target_rot
    );

    // Switch b2 to Static: velocity zeroes and set becomes STATIC_SET.
    assert!(world.set_body_type(b2, BodyType::Static));
    assert_eq!(world.body_solver_set(b2), STATIC_SET);
    assert_eq!(world.body(b2).unwrap().linear_velocity, Vec3::ZERO);
}

#[test]
fn body_mut_mutations_are_visible_to_queries_joints_and_snapshots_before_step() {
    let mut world = PhysicsWorld::new();
    let b = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::ZERO));
    let cid = world.create_collider(b, Collider::new(Shape::Sphere { radius: 0.5 }));

    // Mutate position via body_mut without calling set_body_position or step().
    let new_pos = Vec3::new(20.0, 0.0, 0.0);
    world.body_mut(b).unwrap().position = new_pos;

    // 1. Ray cast at old origin must miss; ray cast at new_pos must hit.
    let ray_old = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
    assert!(world.ray_cast(&ray_old, 10.0).is_none());
    let ray_new = Ray::new(Vec3::new(20.0, 5.0, 0.0), Vec3::NEG_Y);
    let hit = world
        .ray_cast(&ray_new, 10.0)
        .expect("ray_cast at new_pos should hit");
    assert_eq!(hit.body_id, b);
    assert_eq!(hit.collider_id, cid);
    assert!(world
        .cast_ray_closest(
            Vec3::new(20.0, 5.0, 0.0),
            Vec3::new(0.0, -10.0, 0.0),
            &QueryFilter3d::DEFAULT
        )
        .is_some());

    // 2. query_aabb and overlap_aabb at old vs new bounds.
    let old_box = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
    let new_box = Aabb::new(Vec3::new(19.0, -1.0, -1.0), Vec3::new(21.0, 1.0, 1.0));
    assert!(world.query_aabb(&old_box).is_empty());
    assert_eq!(world.query_aabb(&new_box), vec![b]);

    let mut overlapped = Vec::new();
    world.overlap_aabb(&new_box, &QueryFilter3d::DEFAULT, |c| {
        overlapped.push(c);
        true
    });
    assert_eq!(overlapped, vec![cid]);

    // 3. overlap_shape, cast_shape_closest, and collide_mover at new_pos.
    let pt = [Vec3::ZERO];
    let probe = ShapeProxy {
        points: &pt,
        radius: 0.25,
    };
    let mut shape_overlapped = Vec::new();
    world.overlap_shape(
        &probe,
        &Transform3d::new(new_pos, Quat::IDENTITY),
        &QueryFilter3d::DEFAULT,
        |c| {
            shape_overlapped.push(c);
            true
        },
    );
    assert_eq!(shape_overlapped, vec![cid]);

    let cast_origin = [Vec3::new(20.0, 3.0, 0.0)];
    let cast_proxy = ShapeProxy {
        points: &cast_origin,
        radius: 0.2,
    };
    assert!(world
        .cast_shape_closest(
            &cast_proxy,
            Vec3::new(0.0, -5.0, 0.0),
            &QueryFilter3d::DEFAULT
        )
        .is_some());

    let mover = CapsuleMover {
        center1: Vec3::new(20.0, 0.2, 0.0),
        center2: Vec3::new(20.0, 1.0, 0.0),
        radius: 0.4,
    };
    let mut planes = 0;
    world.collide_mover(&mover, &QueryFilter3d::DEFAULT, |_, _| {
        planes += 1;
        true
    });
    assert!(planes > 0);

    // 4. Mutate body_type via body_mut and verify create_snapshot / create_joint flush it.
    let b_static = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(22.0, 0.0, 0.0)));
    world.body_mut(b_static).unwrap().body_type = BodyType::Static;
    let snap = world.create_snapshot();
    assert_eq!(snap.body_links[b_static.index() as usize].set, STATIC_SET);

    let b_other = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(24.0, 0.0, 0.0)));
    let j = world.create_joint(Joint::Distance(DistanceJoint::new(
        b_static.index(),
        b_other.index(),
        Vec3::ZERO,
        Vec3::ZERO,
        2.0,
    )));
    // Both endpoints are static after flushing b_static, so the joint belongs in STATIC_SET.
    let snap2 = world.create_snapshot();
    assert_eq!(snap2.joint_links[j.index() as usize].set, STATIC_SET);
}
