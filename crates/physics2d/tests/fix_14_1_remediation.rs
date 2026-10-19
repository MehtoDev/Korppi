//! Regression tests for `FIX-14.1` (`P2D-29` and `P2D-30`), ported from probes `P16` and `P17` in
//! `docs/audits/probes/physics2d_reaudit/src/main.rs`.

use korppi_math::{Aabb2d, Rot2, Transform2d, Vec2};
use korppi_physics2d::{
    BodyType, Capsule2d, Circle2d, Collider2d, DistanceJoint2d, ExplosionDef2d, Joint2d,
    JointInstance2d, MotionLocks2d, PhysicsWorld2d, QueryFilter2d, RigidBody2d, Shape2d,
    ShapeProxy2d,
};

const DT: f32 = 1.0 / 60.0;

#[test]
fn p2d_29_non_finite_dt_and_target_transform_time_step_are_ignored() {
    let mut world = PhysicsWorld2d::new();
    let body = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(1.0, 2.0))
            .with_linear_velocity(Vec2::new(3.0, -1.0)),
    );
    world.create_collider(
        body,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );

    // Non-finite dt values (NaN, +INFINITY, -INFINITY) must be no-ops just like dt <= 0.0.
    for bad_dt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -DT] {
        world.step(bad_dt);
        let state = world.body(body).unwrap();
        assert_eq!(state.position, Vec2::new(1.0, 2.0));
        assert_eq!(state.linear_velocity, Vec2::new(3.0, -1.0));
        assert!(state.position.x.is_finite() && state.position.y.is_finite());
        assert!(state.rotation.c.is_finite() && state.rotation.s.is_finite());
    }

    // Subsequent valid step must integrate normally without NaN poisoning.
    world.step(DT);
    let state = world.body(body).unwrap();
    assert!(state.position.x.is_finite() && state.position.y.is_finite());
    assert!(state.position.x > 1.0);

    // set_target_transform must also reject non-finite time_step.
    let kinematic = world.create_body(
        RigidBody2d::new(BodyType::Kinematic)
            .with_position(Vec2::ZERO)
            .with_linear_velocity(Vec2::new(1.0, 2.0)),
    );
    let target = Transform2d::from_translation_rotation(Vec2::new(5.0, 5.0), Rot2::from_angle(0.5));
    for bad_dt in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -DT] {
        assert!(!world.set_target_transform(kinematic, target, bad_dt, true));
        let k_state = world.body(kinematic).unwrap();
        assert_eq!(k_state.linear_velocity, Vec2::new(1.0, 2.0));
        assert_eq!(k_state.angular_velocity, 0.0);
    }
}

#[test]
#[should_panic(expected = "cannot create a joint between a body and itself")]
fn p2d_29_self_joint_panics() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let body = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::ZERO));
    world.create_collider(
        body,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );
    world.add_joint(JointInstance2d::new(
        body,
        body,
        Joint2d::Distance(DistanceJoint2d::new(
            Vec2::new(-1.0, 0.0),
            Vec2::new(1.0, 0.0),
            4.0,
            0.0,
            0.0,
        )),
    ));
}

#[test]
fn p2d_30_body_mut_teleport_updates_spatial_queries_snapshots_and_target_transform_before_step() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::ZERO));
    let collider_id = world.create_collider(
        id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );

    // Mutate position directly via body_mut without calling set_transform or step.
    world.body_mut(id).unwrap().position = Vec2::new(10.0, 0.0);

    // 1. Raycasts must miss the old origin and hit the new position immediately.
    assert!(world
        .cast_ray_closest(
            Vec2::new(0.0, 5.0),
            Vec2::new(0.0, -10.0),
            QueryFilter2d::default(),
        )
        .is_none());
    let hit = world
        .cast_ray_closest(
            Vec2::new(10.0, 5.0),
            Vec2::new(0.0, -10.0),
            QueryFilter2d::default(),
        )
        .expect("raycast at new position must hit");
    assert_eq!(hit.body_id, id);
    assert_eq!(hit.collider_id, collider_id);

    // 2. Point and AABB queries must find the body at (10, 0) and not at (0, 0).
    assert_eq!(
        world.query_point(Vec2::ZERO, QueryFilter2d::default()),
        None
    );
    assert_eq!(
        world.query_point(Vec2::new(10.0, 0.0), QueryFilter2d::default()),
        Some(id)
    );
    assert!(world
        .query_aabb(
            &Aabb2d::new(Vec2::splat(-1.0), Vec2::splat(1.0)),
            QueryFilter2d::default(),
        )
        .is_empty());
    assert_eq!(
        world.query_aabb(
            &Aabb2d::new(Vec2::new(9.0, -1.0), Vec2::new(11.0, 1.0)),
            QueryFilter2d::default(),
        ),
        vec![id]
    );

    // 3. overlap_aabb, overlap_shape, cast_shape, cast_mover, collide_mover must see (10, 0).
    let mut aabb_hits = Vec::new();
    world.overlap_aabb(
        &Aabb2d::new(Vec2::splat(-0.2), Vec2::splat(0.2)),
        QueryFilter2d::default(),
        |cid| {
            aabb_hits.push(cid);
            true
        },
    );
    assert!(aabb_hits.is_empty());
    world.overlap_aabb(
        &Aabb2d::new(Vec2::new(9.8, -0.2), Vec2::new(10.2, 0.2)),
        QueryFilter2d::default(),
        |cid| {
            aabb_hits.push(cid);
            true
        },
    );
    assert_eq!(aabb_hits, vec![collider_id]);

    let mut shape_hits = Vec::new();
    let probe_shape = ShapeProxy2d::new(&[Vec2::new(10.0, 0.0)], 0.25);
    world.overlap_shape(&probe_shape, QueryFilter2d::default(), |cid| {
        shape_hits.push(cid);
        true
    });
    assert_eq!(shape_hits, vec![collider_id]);

    let mover = Capsule2d::new(Vec2::new(10.0, 2.0), Vec2::new(10.0, 2.5), 0.25);
    let fraction = world.cast_mover(&mover, Vec2::new(0.0, -4.0), QueryFilter2d::default());
    assert!(fraction < 1.0);

    // 4. Snapshot taken before step must have flushed broadphase proxies as well.
    let snapshot = world.create_snapshot();
    let restored = PhysicsWorld2d::from_snapshot(&snapshot);
    assert_eq!(
        restored.query_point(Vec2::new(10.0, 0.0), QueryFilter2d::default()),
        Some(id)
    );

    // 5. set_target_transform after body_mut teleport must compute velocity from (10, 0), not (0, 0).
    assert!(world.set_target_transform(
        id,
        Transform2d::from_translation(Vec2::new(12.0, 0.0)),
        1.0,
        true,
    ));
    let v = world.body(id).unwrap().linear_velocity;
    assert!((v - Vec2::new(2.0, 0.0)).length() < 1e-5, "got v={v:?}");

    // 6. explode after body_mut teleport must push the body at its new location.
    world.body_mut(id).unwrap().position = Vec2::new(30.0, 0.0);
    world.body_mut(id).unwrap().linear_velocity = Vec2::ZERO;
    world.explode(&ExplosionDef2d {
        mask_bits: u64::MAX,
        position: Vec2::new(29.0, 0.0),
        radius: 2.0,
        falloff: 1.0,
        impulse_per_length: 10.0,
    });
    assert!(world.body(id).unwrap().linear_velocity.x > 0.0);
}

#[test]
fn p2d_30_body_touched_reconciles_direct_mass_locks_and_type_mutations() {
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -10.0));
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::ZERO));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );

    // Direct mutation of mass and rotational_inertia via body_mut must reconcile inv_mass and inv_inertia.
    {
        let body = world.body_mut(id).unwrap();
        body.mass = 100.0;
        body.rotational_inertia = 50.0;
        body.linear_velocity = Vec2::new(5.0, 7.0);
        body.angular_velocity = 3.0;
        body.motion_locks = MotionLocks2d {
            linear_x: true,
            linear_y: false,
            angular_z: true,
        };
    }
    world.step(DT);
    let body = world.body(id).unwrap();
    assert!((body.inv_mass - 0.01).abs() < 1e-6);
    assert_eq!(body.inv_inertia, 0.0);
    assert_eq!(body.linear_velocity.x, 0.0);
    assert_eq!(body.angular_velocity, 0.0);

    // Negative mass and inertia via body_mut must clamp to 0.0.
    {
        let body = world.body_mut(id).unwrap();
        body.mass = -25.0;
        body.rotational_inertia = -10.0;
        body.inv_mass = -1.0;
        body.inv_inertia = -1.0;
    }
    world.step(DT);
    let body = world.body(id).unwrap();
    assert_eq!(body.mass, 0.0);
    assert_eq!(body.inv_mass, 0.0);
    assert_eq!(body.rotational_inertia, 0.0);
    assert_eq!(body.inv_inertia, 0.0);

    // Direct mutation of body_type to Static via body_mut must reconcile solver set and zero velocity/mass.
    let pos_before = world.body(id).unwrap().position;
    {
        let body = world.body_mut(id).unwrap();
        body.body_type = BodyType::Static;
        body.linear_velocity = Vec2::new(10.0, 10.0);
    }
    world.step(DT);
    let body = world.body(id).unwrap();
    assert_eq!(body.body_type, BodyType::Static);
    assert_eq!(body.linear_velocity, Vec2::ZERO);
    assert_eq!(body.position, pos_before);
}
