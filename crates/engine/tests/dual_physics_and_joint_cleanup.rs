//! Integration tests for `FIX-10.5` (`ENG-05`, `ENG-07`):
//! - `ENG-05`: Disambiguating entities that carry both 3D `RigidBody` and 2D `RigidBody2d` so
//!   2D physics never clobbers 3D transforms.
//! - `ENG-07`: Cleaning up orphaned `EntityJoint3d` and `EntityJoint2d` components and standalone
//!   joint entities when a referenced body is despawned.

use korppi_engine::app::{EntityJoint2d, EntityJoint3d, Joint3dConfig};
use korppi_engine::EngineApp;
use korppi_math::{Transform, Vec2, Vec3};
use korppi_physics::{BodyType, Collider, RigidBody, Shape};
use korppi_physics2d::{
    body::BodyType as BodyType2d,
    collider::Collider2d,
    joint::{Joint2d, RevoluteJoint2d},
    shape::{Circle2d, Shape2d},
    RigidBody2d,
};
use korppi_scene::{GlobalTransform, LocalTransform};
use std::time::Duration;

const DT: Duration = Duration::from_nanos(1_000_000_000 / 60);

#[test]
fn dual_2d_and_3d_rigid_bodies_on_same_entity_disambiguates_in_favor_of_3d() {
    let mut app = EngineApp::new().expect("EngineApp creation should succeed");

    let mut body_3d = RigidBody::new(BodyType::Kinematic, Vec3::new(10.0, 20.0, 30.0));
    body_3d.linear_velocity = Vec3::new(6.0, 12.0, 18.0);

    let mut body_2d =
        RigidBody2d::new(BodyType2d::Kinematic).with_position(Vec2::new(-50.0, -60.0));
    body_2d.linear_velocity = Vec2::new(-10.0, -20.0);

    let entity = app.world.spawn((
        body_3d,
        Collider::new(Shape::Sphere { radius: 0.5 }),
        body_2d,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
        LocalTransform::from_translation(Vec3::new(10.0, 20.0, 30.0)),
        GlobalTransform::from_transform(Transform::from_translation(Vec3::new(10.0, 20.0, 30.0))),
    ));

    for _ in 0..10 {
        app.update(DT);
    }

    // 3D physics owns the dual-component entity; 2D physics must not register it or clobber its transform.
    assert_eq!(app.physics_3d.body_count(), 1);
    assert_eq!(app.physics_2d.body_count(), 0);

    let local = app
        .world
        .get::<LocalTransform>(entity)
        .expect("LocalTransform");
    let global = app
        .world
        .get::<GlobalTransform>(entity)
        .expect("GlobalTransform");
    assert!(
        local.translation.x > 10.5 && local.translation.y > 21.0 && local.translation.z > 31.5,
        "LocalTransform should follow 3D kinematic motion, got {:?}",
        local.translation
    );
    assert_eq!(global.translation(), local.translation);

    // Removing the 3D RigidBody allows the entity's 2D RigidBody2d to take over cleanly.
    app.world.remove_component::<RigidBody>(entity);
    app.update(DT);
    assert_eq!(app.physics_3d.body_count(), 0);
    assert_eq!(app.physics_2d.body_count(), 1);

    // Re-attaching a 3D RigidBody removes the 2D body and restores 3D precedence.
    app.world.add_component(
        entity,
        RigidBody::new(BodyType::Static, Vec3::new(1.0, 2.0, 3.0)),
    );
    app.update(DT);
    assert_eq!(app.physics_3d.body_count(), 1);
    assert_eq!(app.physics_2d.body_count(), 0);
    let global_after = app
        .world
        .get::<GlobalTransform>(entity)
        .expect("GlobalTransform");
    assert_eq!(global_after.translation(), Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn despawning_referenced_body_cleans_up_orphaned_3d_joint_entities_and_components() {
    let mut app = EngineApp::new().expect("EngineApp creation should succeed");

    // 1. Standalone 3D joint entity
    let body_a = app
        .world
        .spawn((RigidBody::new(BodyType::Static, Vec3::new(0.0, 5.0, 0.0)),));
    let body_b = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 4.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.25 }),
    ));
    let joint_entity = app.world.spawn((EntityJoint3d {
        entity_a: body_a,
        entity_b: body_b,
        config: Joint3dConfig::Spherical {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
        },
    },));

    app.update(DT);
    assert_eq!(app.physics_3d.joint_count(), 1);
    assert!(app.world.is_alive(joint_entity));

    // Despawn body_a: physics joint is destroyed AND standalone joint_entity is despawned.
    assert!(app.world.despawn(body_a));
    app.update(DT);
    assert_eq!(app.physics_3d.joint_count(), 0);
    assert!(
        !app.world.is_alive(joint_entity),
        "standalone orphaned 3D joint entity must be despawned when a connected body is despawned"
    );

    // 2. Embedded EntityJoint3d component on a body entity
    let anchor = app
        .world
        .spawn((RigidBody::new(BodyType::Static, Vec3::new(2.0, 5.0, 0.0)),));
    let bob = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 4.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.25 }),
        EntityJoint3d {
            entity_a: anchor,
            entity_b: body_b,
            config: Joint3dConfig::Spherical {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
            },
        },
    ));

    app.update(DT);
    assert_eq!(app.physics_3d.joint_count(), 1);
    assert!(app.world.get::<EntityJoint3d>(bob).is_some());

    // Despawn anchor: bob remains alive with its RigidBody, but its orphaned EntityJoint3d is removed.
    assert!(app.world.despawn(anchor));
    app.update(DT);
    assert_eq!(app.physics_3d.joint_count(), 0);
    assert!(
        app.world.is_alive(bob),
        "body entity carrying an embedded joint must stay alive"
    );
    assert!(app.world.get::<RigidBody>(bob).is_some());
    assert!(
        app.world.get::<EntityJoint3d>(bob).is_none(),
        "orphaned EntityJoint3d component must be removed from surviving entity"
    );
}

#[test]
fn despawning_referenced_body_cleans_up_orphaned_2d_joint_entities_and_components() {
    let mut app = EngineApp::new().expect("EngineApp creation should succeed");

    // 1. Standalone 2D joint entity
    let body_a = app
        .world
        .spawn((RigidBody2d::new(BodyType2d::Static).with_position(Vec2::new(0.0, 5.0)),));
    let body_b = app.world.spawn((
        RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(0.0, 4.0)),
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
    ));
    let joint_entity = app.world.spawn((EntityJoint2d {
        entity_a: body_a,
        entity_b: body_b,
        joint: Joint2d::Revolute(RevoluteJoint2d::new(
            Vec2::ZERO,
            Vec2::new(0.0, 1.0),
            0.0,
            0.0,
        )),
    },));

    app.update(DT);
    assert_eq!(app.physics_2d.joint_count(), 1);
    assert!(app.world.is_alive(joint_entity));

    assert!(app.world.despawn(body_a));
    app.update(DT);
    assert_eq!(app.physics_2d.joint_count(), 0);
    assert!(
        !app.world.is_alive(joint_entity),
        "standalone orphaned 2D joint entity must be despawned when a connected body is despawned"
    );

    // 2. Embedded EntityJoint2d component on a body entity
    let anchor = app
        .world
        .spawn((RigidBody2d::new(BodyType2d::Static).with_position(Vec2::new(2.0, 5.0)),));
    let bob = app.world.spawn((
        RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(2.0, 4.0)),
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.25))),
        EntityJoint2d {
            entity_a: anchor,
            entity_b: body_b,
            joint: Joint2d::Revolute(RevoluteJoint2d::new(
                Vec2::ZERO,
                Vec2::new(0.0, 1.0),
                0.0,
                0.0,
            )),
        },
    ));

    app.update(DT);
    assert_eq!(app.physics_2d.joint_count(), 1);
    assert!(app.world.get::<EntityJoint2d>(bob).is_some());

    assert!(app.world.despawn(anchor));
    app.update(DT);
    assert_eq!(app.physics_2d.joint_count(), 0);
    assert!(
        app.world.is_alive(bob),
        "body entity carrying an embedded joint must stay alive"
    );
    assert!(app.world.get::<RigidBody2d>(bob).is_some());
    assert!(
        app.world.get::<EntityJoint2d>(bob).is_none(),
        "orphaned EntityJoint2d component must be removed from surviving entity"
    );
}
