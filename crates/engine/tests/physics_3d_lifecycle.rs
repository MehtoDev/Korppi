//! Despawning an entity removes its 3D body and joints from the physics world (P3D-16).
//!
//! Before FIX-3.6 the engine created 3D bodies for new entities but never removed them: a
//! despawned ground kept holding up the boxes on it, and a despawned joint entity kept its joint.

use korppi_engine::app::{EntityJoint3d, Joint3dConfig};
use korppi_engine::EngineApp;
use korppi_math::Vec3;
use korppi_physics::{
    body::{BodyType, RigidBody},
    collider::Collider,
    shape::Shape,
};
use std::time::Duration;

const DT: Duration = Duration::from_nanos(1_000_000_000 / 60);

fn run(app: &mut EngineApp, seconds: f32) {
    for _ in 0..(seconds * 60.0).round() as usize {
        app.update(DT);
    }
}

fn spawn_box(
    app: &mut EngineApp,
    kind: BodyType,
    position: Vec3,
    hx: f32,
    hy: f32,
    hz: f32,
) -> korppi_ecs::Entity {
    app.world.spawn((
        RigidBody::new(kind, position),
        Collider::new(Shape::Box {
            half_extents: Vec3::new(hx, hy, hz),
        }),
    ))
}

#[test]
fn despawning_the_ground_entity_drops_the_box_on_it() {
    let mut app = EngineApp::new().expect("engine");
    let ground = spawn_box(
        &mut app,
        BodyType::Static,
        Vec3::new(0.0, -0.5, 0.0),
        5.0,
        0.5,
        5.0,
    );
    let crate_box = spawn_box(
        &mut app,
        BodyType::Dynamic,
        Vec3::new(0.0, 0.5, 0.0),
        0.5,
        0.5,
        0.5,
    );
    run(&mut app, 3.0);
    assert_eq!(app.physics_3d.body_count(), 2);
    let resting = app.world.get::<RigidBody>(crate_box).unwrap().position.y;
    assert!(
        (resting - 0.5).abs() < 0.01,
        "the box rests at y = {resting:.4}"
    );

    assert!(app.world.despawn(ground));
    run(&mut app, 1.0);
    assert_eq!(
        app.physics_3d.body_count(),
        1,
        "the ground's body must be destroyed"
    );
    let y = app.world.get::<RigidBody>(crate_box).unwrap().position.y;
    assert!(
        y < -4.0,
        "the box must fall once its ground is gone; it is at y = {y:.3}"
    );
}

#[test]
fn despawning_a_joint_entity_destroys_the_joint() {
    let mut app = EngineApp::new().expect("engine");
    let anchor = app
        .world
        .spawn((RigidBody::new(BodyType::Static, Vec3::new(0.0, 5.0, 0.0)),));
    let bob = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 4.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.25 }),
    ));
    let joint = app.world.spawn((EntityJoint3d {
        entity_a: anchor,
        entity_b: bob,
        config: Joint3dConfig::Spherical {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
        },
    },));
    run(&mut app, 1.0);
    assert_eq!(app.physics_3d.joint_count(), 1);
    let hanging = app.world.get::<RigidBody>(bob).unwrap().position.y;
    assert!(
        (hanging - 4.0).abs() < 0.01,
        "the bob hangs at y = {hanging:.4}"
    );

    assert!(app.world.despawn(joint));
    run(&mut app, 1.0);
    assert_eq!(
        app.physics_3d.joint_count(),
        0,
        "the joint must be destroyed"
    );
    let y = app.world.get::<RigidBody>(bob).unwrap().position.y;
    assert!(
        y < 0.0,
        "the bob must fall once its joint is gone; it is at y = {y:.3}"
    );
}

#[test]
fn despawning_a_jointed_body_forgets_its_joint() {
    // Destroying a body destroys its joints; the engine must not keep a handle to them, and a
    // joint entity whose body is gone creates no new joint.
    let mut app = EngineApp::new().expect("engine");
    let anchor = app
        .world
        .spawn((RigidBody::new(BodyType::Static, Vec3::new(0.0, 5.0, 0.0)),));
    let bob = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 4.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.25 }),
    ));
    app.world.spawn((EntityJoint3d {
        entity_a: anchor,
        entity_b: bob,
        config: Joint3dConfig::Spherical {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
        },
    },));
    run(&mut app, 0.5);
    assert!(app.world.despawn(anchor));
    run(&mut app, 0.5);
    assert_eq!(app.physics_3d.joint_count(), 0);
    assert_eq!(app.physics_3d.body_count(), 1);
    assert!(app.world.get::<RigidBody>(bob).unwrap().position.y < 3.0);
}
