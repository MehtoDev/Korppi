//! Regression tests for Milestone FIX-15.2 (`ENG-11`, `ENG-12`).

use korppi_ecs::{Changed, Entity, Query, Schedule, Stage};
use korppi_engine::app::{EngineApp, EntityJoint2d, EntityJoint3d, Joint3dConfig};
use korppi_math::{Quat, Vec2, Vec3};
use korppi_physics::{BodyType, Collider, DistanceJoint, RevoluteJoint, RigidBody, Shape};
use korppi_physics2d::{
    BodyType as BodyType2d, Circle2d, Collider2d, DistanceJoint2d, Joint2d, RigidBody2d, Shape2d,
};
use korppi_scene::{attach_child, GlobalTransform, LocalTransform};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const DT: Duration = Duration::from_nanos(16_666_667);

/// `ENG-11`: Child physics bodies in a scene hierarchy must have their world-space physics poses
/// converted into parent-local space before updating `LocalTransform`, and unmoved physics bodies
/// must not trigger `Changed<RigidBody>`, `Changed<LocalTransform>`, or `Changed<GlobalTransform>`
/// on every steady-state tick.
#[test]
fn test_eng_11_child_physics_hierarchy_and_change_detection() {
    let mut app = EngineApp::new().expect("engine creation");
    app.physics_3d.set_gravity(Vec3::ZERO);
    app.physics_2d.set_gravity(Vec2::ZERO);

    // 1. 3D parent at (10, 0, 0) rotated 90 degrees around Z, child at world (10, 5, 0)
    let parent_rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let parent = app.world.spawn((
        LocalTransform {
            translation: Vec3::new(10.0, 0.0, 0.0),
            rotation: parent_rot,
            scale: Vec3::ONE,
        },
        GlobalTransform::default(),
    ));
    let child = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(10.0, 5.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.5 }),
        LocalTransform::default(),
        GlobalTransform::default(),
    ));
    attach_child(&mut app.world, parent, child).expect("attach_child 3d");

    // 2. 2D parent at (20, 3, 0), child2d at world (24, 9)
    let parent_2d = app.world.spawn((
        LocalTransform::from_translation(Vec3::new(20.0, 3.0, 0.0)),
        GlobalTransform::default(),
    ));
    let child_2d = app.world.spawn((
        RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(24.0, 9.0)),
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
        LocalTransform::default(),
        GlobalTransform::default(),
    ));
    attach_child(&mut app.world, parent_2d, child_2d).expect("attach_child 2d");

    // 3. Static 3D and 2D bodies for change-detection noise verification
    let static_3d = app.world.spawn((
        RigidBody::new(BodyType::Static, Vec3::new(-5.0, 0.0, 0.0)),
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
        LocalTransform::from_translation(Vec3::new(-5.0, 0.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(-5.0, 0.0, 0.0)),
    ));
    let static_2d = app.world.spawn((
        RigidBody2d::new(BodyType2d::Static).with_position(Vec2::new(-10.0, 0.0)),
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0))),
        LocalTransform::from_translation(Vec3::new(-10.0, 0.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(-10.0, 0.0, 0.0)),
    ));

    let mut change_sched = Schedule::new();
    let changed_rb3d = Arc::new(AtomicUsize::new(0));
    let changed_local = Arc::new(AtomicUsize::new(0));
    let changed_global = Arc::new(AtomicUsize::new(0));
    let changed_rb2d = Arc::new(AtomicUsize::new(0));

    let c_rb3d = Arc::clone(&changed_rb3d);
    change_sched.add(
        Stage::Update,
        move |mut q: Query<(Entity, Changed<RigidBody>)>| {
            for (entity, _) in q.iter_mut() {
                if entity == static_3d {
                    c_rb3d.fetch_add(1, Ordering::SeqCst);
                }
            }
        },
    );
    let c_local = Arc::clone(&changed_local);
    change_sched.add(
        Stage::Update,
        move |mut q: Query<(Entity, Changed<LocalTransform>)>| {
            for (entity, _) in q.iter_mut() {
                if entity == static_3d || entity == static_2d {
                    c_local.fetch_add(1, Ordering::SeqCst);
                }
            }
        },
    );
    let c_global = Arc::clone(&changed_global);
    change_sched.add(
        Stage::Update,
        move |mut q: Query<(Entity, Changed<GlobalTransform>)>| {
            for (entity, _) in q.iter_mut() {
                if entity == static_3d || entity == static_2d {
                    c_global.fetch_add(1, Ordering::SeqCst);
                }
            }
        },
    );
    let c_rb2d = Arc::clone(&changed_rb2d);
    change_sched.add(
        Stage::Update,
        move |mut q: Query<(Entity, Changed<RigidBody2d>)>| {
            for (entity, _) in q.iter_mut() {
                if entity == static_2d {
                    c_rb2d.fetch_add(1, Ordering::SeqCst);
                }
            }
        },
    );

    // Initial tick to register bodies and consume creation change-ticks
    app.update(DT);
    change_sched.run_stage(Stage::Update, &mut app.world);
    changed_rb3d.store(0, Ordering::SeqCst);
    changed_local.store(0, Ordering::SeqCst);
    changed_global.store(0, Ordering::SeqCst);
    changed_rb2d.store(0, Ordering::SeqCst);

    // Verify 3D child GlobalTransform and LocalTransform
    let child_global = app
        .world
        .get::<GlobalTransform>(child)
        .unwrap()
        .translation();
    let child_local = app.world.get::<LocalTransform>(child).unwrap().translation;
    assert!(
        (child_global - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-4,
        "3D child GlobalTransform must match world physics position (10, 5, 0), got {child_global:?}"
    );
    // Under a +90deg Z rotation at (10, 0, 0), world offset (0, 5, 0) corresponds to local (5, 0, 0)
    assert!(
        (child_local - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4,
        "3D child LocalTransform must be in parent-local space (5, 0, 0), got {child_local:?}"
    );

    // Verify 2D child GlobalTransform and LocalTransform
    let child_2d_global = app
        .world
        .get::<GlobalTransform>(child_2d)
        .unwrap()
        .translation();
    let child_2d_local = app
        .world
        .get::<LocalTransform>(child_2d)
        .unwrap()
        .translation;
    assert!(
        (child_2d_global - Vec3::new(24.0, 9.0, 0.0)).length() < 1e-4,
        "2D child GlobalTransform must match world physics position (24, 9, 0), got {child_2d_global:?}"
    );
    assert!(
        (child_2d_local - Vec3::new(4.0, 6.0, 0.0)).length() < 1e-4,
        "2D child LocalTransform must be in parent-local space (4, 6, 0), got {child_2d_local:?}"
    );

    // Run 5 steady-state ticks where static_3d and static_2d do not move
    for _ in 0..5 {
        app.update(DT);
        change_sched.run_stage(Stage::Update, &mut app.world);
    }

    assert_eq!(
        changed_rb3d.load(Ordering::SeqCst),
        0,
        "unmoved Static 3D body must not trigger Changed<RigidBody>"
    );
    assert_eq!(
        changed_rb2d.load(Ordering::SeqCst),
        0,
        "unmoved Static 2D body must not trigger Changed<RigidBody2d>"
    );
    assert_eq!(
        changed_local.load(Ordering::SeqCst),
        0,
        "unmoved Static bodies must not trigger Changed<LocalTransform>"
    );
    assert_eq!(
        changed_global.load(Ordering::SeqCst),
        0,
        "unmoved Static bodies must not trigger Changed<GlobalTransform>"
    );
}

/// `ENG-12`: ECS mutations to `EntityJoint3d` and `EntityJoint2d` must be synchronized to the
/// physics worlds on subsequent ticks, and `Joint3dConfig` must expose full 3D joint parameters
/// (limits, motors, springs, and all `Joint3d` variants).
#[test]
fn test_eng_12_joint_mutation_sync_and_full_3d_joint_config() {
    let mut app = EngineApp::new().expect("engine creation");

    let a3 = app
        .world
        .spawn((RigidBody::new(BodyType::Static, Vec3::new(0.0, 10.0, 0.0)),));
    let b3 = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 7.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.25 }),
    ));
    let j3 = app.world.spawn((EntityJoint3d {
        entity_a: a3,
        entity_b: b3,
        config: Joint3dConfig::Distance {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            distance: 3.0,
        },
    },));

    let a2 = app
        .world
        .spawn((RigidBody2d::new(BodyType2d::Static).with_position(Vec2::new(0.0, 10.0)),));
    let b2 = app.world.spawn((
        RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(0.0, 7.0)),
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
    ));
    let j2 = app.world.spawn((EntityJoint2d {
        entity_a: a2,
        entity_b: b2,
        joint: Joint2d::Distance(DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 3.0, 0.0, 0.0)),
    },));

    for _ in 0..10 {
        app.update(DT);
    }

    // Mutate both distance joints in ECS from 3.0m to 1.0m
    app.world.get_mut::<EntityJoint3d>(j3).unwrap().config = Joint3dConfig::Distance {
        local_anchor_a: Vec3::ZERO,
        local_anchor_b: Vec3::ZERO,
        distance: 1.0,
    };
    app.world.get_mut::<EntityJoint2d>(j2).unwrap().joint =
        Joint2d::Distance(DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 1.0, 0.0, 0.0));

    for _ in 0..60 {
        app.update(DT);
    }

    let dist_3d =
        (app.world.get::<RigidBody>(b3).unwrap().position - Vec3::new(0.0, 10.0, 0.0)).length();
    let dist_2d =
        (app.world.get::<RigidBody2d>(b2).unwrap().position - Vec2::new(0.0, 10.0)).length();
    assert!(
        (dist_3d - 1.0).abs() < 0.15,
        "mutated EntityJoint3d distance 3.0 -> 1.0 must pull body to ~1.0m, got {dist_3d}"
    );
    assert!(
        (dist_2d - 1.0).abs() < 0.15,
        "mutated EntityJoint2d distance 3.0 -> 1.0 must pull body to ~1.0m, got {dist_2d}"
    );

    // Also verify full 3D joint configuration with motor & limits via Joint3dConfig::from
    let hinge_base = app
        .world
        .spawn((RigidBody::new(BodyType::Static, Vec3::new(5.0, 5.0, 0.0)),));
    let hinge_arm = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(6.0, 5.0, 0.0)),
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.25),
        }),
    ));
    let full_hinge = RevoluteJoint::new(0, 0, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), Vec3::Z)
        .with_motor(6.0, 50.0)
        .with_limits(-1.0, 1.0);
    let hinge_ent = app.world.spawn((EntityJoint3d {
        entity_a: hinge_base,
        entity_b: hinge_arm,
        config: Joint3dConfig::from(full_hinge),
    },));

    for _ in 0..30 {
        app.update(DT);
    }

    let arm_vel = app
        .world
        .get::<RigidBody>(hinge_arm)
        .unwrap()
        .angular_velocity;
    let arm_pos = app.world.get::<RigidBody>(hinge_arm).unwrap().position;
    assert!(
        arm_pos.y > 5.1 || arm_vel.length() > 0.01,
        "full 3D RevoluteJoint motor and limits must drive the connected body, got pos={arm_pos:?}, vel={arm_vel:?}"
    );

    // Mutate the full joint to a spring DistanceJoint via Joint3dConfig::from
    let spring_dist = DistanceJoint::new(0, 0, Vec3::ZERO, Vec3::ZERO, 2.5).with_spring(8.0, 0.7);
    app.world
        .get_mut::<EntityJoint3d>(hinge_ent)
        .unwrap()
        .config = Joint3dConfig::from(spring_dist);

    for _ in 0..60 {
        app.update(DT);
    }
    let new_sep = (app.world.get::<RigidBody>(hinge_arm).unwrap().position
        - Vec3::new(5.0, 5.0, 0.0))
    .length();
    assert!(
        (new_sep - 2.5).abs() < 0.35,
        "mutating EntityJoint3d to a full spring DistanceJoint(2.5m) must settle near 2.5m, got {new_sep}"
    );
}
