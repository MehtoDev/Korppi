//! Bidirectional synchronization between ECS components and the 2D/3D physics worlds
//! (`ENG-01`, `ENG-06`).

use korppi_ecs::Stage;
use korppi_engine::EngineApp;
use korppi_math::{Quat, Rot2, Transform, Vec2, Vec3};
use korppi_physics::{
    body::{BodyType as BodyType3d, RigidBody},
    collider::Collider,
    shape::Shape,
};
use korppi_physics2d::{
    body::BodyType as BodyType2d,
    collider::Collider2d,
    shape::{Circle2d, Polygon2d, Shape2d},
    RigidBody2d,
};
use korppi_scene::{GlobalTransform, LocalTransform};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const DT: Duration = Duration::from_nanos(1_000_000_000 / 60);

fn run(app: &mut EngineApp, ticks: usize) {
    for _ in 0..ticks {
        app.update(DT);
    }
}

#[test]
fn ecs_mutations_to_3d_rigid_body_propagate_to_physics() {
    let mut app = EngineApp::new().expect("engine");
    app.physics_3d.set_gravity(Vec3::ZERO);

    let entity = app.world.spawn((
        RigidBody::new(BodyType3d::Dynamic, Vec3::ZERO),
        Collider::new(Shape::Sphere { radius: 0.5 }),
    ));

    // Initial tick registers the body in physics_3d; let it settle/sleep.
    run(&mut app, 60);

    // 1. Mutate linear_velocity in ECS after registration.
    app.world
        .get_mut::<RigidBody>(entity)
        .unwrap()
        .linear_velocity = Vec3::new(60.0, 0.0, 0.0);
    run(&mut app, 1);

    let rb = app.world.get::<RigidBody>(entity).unwrap();
    assert!(
        (rb.linear_velocity.x - 60.0).abs() < 1e-3,
        "ECS linear_velocity mutation was clobbered: got {:?}",
        rb.linear_velocity
    );
    assert!(
        rb.position.x > 0.9,
        "body should have integrated x > 0.9m in 1/60s at 60 m/s, got {}",
        rb.position.x
    );

    // 2. Teleport position and rotation in ECS and stop velocity.
    let target_pos = Vec3::new(-10.0, 25.0, 4.0);
    let target_rot = Quat::from_axis_angle(Vec3::Y, 1.2);
    {
        let rb = app.world.get_mut::<RigidBody>(entity).unwrap();
        rb.position = target_pos;
        rb.rotation = target_rot;
        rb.linear_velocity = Vec3::ZERO;
    }
    run(&mut app, 1);

    let rb = app.world.get::<RigidBody>(entity).unwrap();
    assert!(
        (rb.position - target_pos).length() < 1e-4,
        "ECS position teleport was ignored: got {:?}",
        rb.position
    );
    assert!(
        rb.rotation.dot(target_rot).abs() > 0.9999,
        "ECS rotation teleport was ignored: got {:?}",
        rb.rotation
    );

    // 3. Apply a one-step external force in ECS and verify it accelerates the body once
    // and is cleared after the step.
    let mass = app.world.get::<RigidBody>(entity).unwrap().mass;
    assert!(mass > 0.0);
    app.world
        .get_mut::<RigidBody>(entity)
        .unwrap()
        .apply_force(Vec3::new(0.0, mass * 60.0, 0.0));
    run(&mut app, 1);

    let rb = app.world.get::<RigidBody>(entity).unwrap();
    assert!(
        (rb.linear_velocity.y - 1.0).abs() < 1e-2,
        "applied force should produce ~1.0 m/s delta-v over 1/60s, got {}",
        rb.linear_velocity.y
    );
    assert_eq!(
        rb.force,
        Vec3::ZERO,
        "accumulated force must be cleared in ECS after the physics step"
    );

    // Next step without new force should keep constant velocity (~1.0 m/s), not accelerate again.
    run(&mut app, 1);
    let rb = app.world.get::<RigidBody>(entity).unwrap();
    assert!(
        (rb.linear_velocity.y - 1.0).abs() < 1e-2,
        "force was re-applied on subsequent tick: vel.y = {}",
        rb.linear_velocity.y
    );

    // 4. Mutate body_type from Static to Dynamic on an existing body.
    app.physics_3d.set_gravity(Vec3::new(0.0, -9.81, 0.0));
    let platform = app.world.spawn((
        RigidBody::new(BodyType3d::Static, Vec3::new(0.0, 10.0, 0.0)),
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        }),
    ));
    run(&mut app, 10);
    assert!((app.world.get::<RigidBody>(platform).unwrap().position.y - 10.0).abs() < 1e-5);

    app.world.get_mut::<RigidBody>(platform).unwrap().body_type = BodyType3d::Dynamic;
    run(&mut app, 30);
    let platform_y = app.world.get::<RigidBody>(platform).unwrap().position.y;
    assert!(
        platform_y < 9.0,
        "body converted from Static to Dynamic should fall under gravity, got y = {platform_y}"
    );
}

#[test]
fn late_collider_attachment_mutation_and_removal_3d() {
    let mut app = EngineApp::new().expect("engine");

    let ground = app.world.spawn((
        RigidBody::new(BodyType3d::Static, Vec3::new(0.0, -0.5, 0.0)),
        Collider::new(Shape::Box {
            half_extents: Vec3::new(5.0, 0.5, 5.0),
        }),
    ));

    // Spawn bare dynamic body without a Collider first.
    let ball = app.world.spawn((RigidBody::new(
        BodyType3d::Dynamic,
        Vec3::new(0.0, 2.0, 0.0),
    ),));
    run(&mut app, 1);
    assert_eq!(app.physics_3d.collider_count(), 1);

    // Attach Collider after the body is already registered in physics_3d.
    app.world
        .add_component(ball, Collider::new(Shape::Sphere { radius: 0.5 }));
    run(&mut app, 120);

    assert_eq!(
        app.physics_3d.collider_count(),
        2,
        "late Collider component must be registered in physics_3d"
    );
    let resting_y = app.world.get::<RigidBody>(ball).unwrap().position.y;
    assert!(
        (resting_y - 0.5).abs() < 0.02,
        "ball with late-attached radius 0.5 collider should rest at y ~ 0.5, got {resting_y}"
    );

    // Mutate the Collider shape radius from 0.5 to 1.25 and teleport above ground.
    app.world.get_mut::<Collider>(ball).unwrap().shape = Shape::Sphere { radius: 1.25 };
    app.world.get_mut::<RigidBody>(ball).unwrap().position = Vec3::new(0.0, 3.0, 0.0);
    run(&mut app, 120);

    let resized_y = app.world.get::<RigidBody>(ball).unwrap().position.y;
    assert!(
        (resized_y - 1.25).abs() < 0.03,
        "ball with mutated radius 1.25 collider should rest at y ~ 1.25, got {resized_y}"
    );

    // Remove the ground's Collider component while keeping its RigidBody alive.
    let removed = app.world.remove_component::<Collider>(ground);
    assert!(removed.is_some());
    run(&mut app, 60);

    assert_eq!(
        app.physics_3d.collider_count(),
        1,
        "removing Collider component from ECS must destroy the physics collider"
    );
    let fallen_y = app.world.get::<RigidBody>(ball).unwrap().position.y;
    assert!(
        fallen_y < -2.0,
        "ball should fall after ground collider is removed, got y = {fallen_y}"
    );
}

#[test]
fn ecs_mutations_to_2d_rigid_body_and_collider_propagate_to_physics() {
    let mut app = EngineApp::new().expect("engine");

    let ground = app.world.spawn((
        RigidBody2d::new(BodyType2d::Static).with_position(Vec2::new(0.0, -0.5)),
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(5.0, 0.5))),
    ));

    // Spawn a bare dynamic 2D body (massless in Box2D v3 until a collider is attached).
    let body = app
        .world
        .spawn((RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(0.0, 2.0)),));
    run(&mut app, 5);
    assert!(
        (app.world.get::<RigidBody2d>(body).unwrap().position.y - 2.0).abs() < 1e-4,
        "bare 2D dynamic body without collider is massless and stays at y = 2.0"
    );

    // Late-attach a Collider2d component; it must give the body mass and make it fall onto the ground.
    app.world.add_component(
        body,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
    );
    run(&mut app, 120);

    let resting_y = app.world.get::<RigidBody2d>(body).unwrap().position.y;
    assert!(
        (resting_y - 0.5).abs() < 0.02,
        "2D body with late-attached circle collider should rest at y ~ 0.5, got {resting_y}"
    );

    // Mutate linear_velocity and position/rotation in ECS after body is asleep on the ground.
    {
        let rb = app.world.get_mut::<RigidBody2d>(body).unwrap();
        rb.position = Vec2::new(-3.0, 0.5);
        rb.rotation = Rot2::from_angle(0.75);
        rb.linear_velocity = Vec2::new(30.0, 0.0);
    }
    run(&mut app, 1);

    let rb = app.world.get::<RigidBody2d>(body).unwrap();
    assert!(
        rb.linear_velocity.x > 25.0,
        "2D ECS linear_velocity mutation was clobbered: got {:?}",
        rb.linear_velocity
    );
    assert!(
        rb.position.x > -2.6,
        "2D ECS position teleport + velocity was ignored: got {:?}",
        rb.position
    );

    // Remove the ground's Collider2d component and verify the body falls through.
    assert!(app.world.remove_component::<Collider2d>(ground).is_some());
    run(&mut app, 60);

    let fallen_y = app.world.get::<RigidBody2d>(body).unwrap().position.y;
    assert!(
        fallen_y < -2.0,
        "2D body should fall after ground Collider2d is removed, got y = {fallen_y}"
    );
}

#[test]
fn physics_step_preserves_transform_scale_during_fixed_update_and_without_local_transform() {
    let mut app = EngineApp::new().expect("engine");

    let scale_3d_local = Vec3::new(2.0, 3.0, 4.0);
    let scale_3d_global_only = Vec3::new(5.0, 6.0, 7.0);
    let scale_2d_local = Vec3::new(1.5, 2.5, 3.5);
    let scale_2d_global_only = Vec3::new(4.5, 5.5, 6.5);

    let e3d_with_local = app.world.spawn((
        RigidBody::new(BodyType3d::Dynamic, Vec3::new(0.0, 5.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.5 }),
        LocalTransform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: scale_3d_local,
        },
        GlobalTransform::from_transform(Transform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: scale_3d_local,
        }),
    ));

    let e3d_global_only = app.world.spawn((
        RigidBody::new(BodyType3d::Dynamic, Vec3::new(2.0, 5.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.5 }),
        GlobalTransform::from_transform(Transform {
            translation: Vec3::new(2.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: scale_3d_global_only,
        }),
    ));

    let e2d_with_local = app.world.spawn((
        RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(0.0, 5.0)),
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
        LocalTransform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: scale_2d_local,
        },
        GlobalTransform::from_transform(Transform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: scale_2d_local,
        }),
    ));

    let e2d_global_only = app.world.spawn((
        RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(2.0, 5.0)),
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
        GlobalTransform::from_transform(Transform {
            translation: Vec3::new(2.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: scale_2d_global_only,
        }),
    ));

    // Observe scales during Stage::Update (immediately after FixedUpdate physics sync,
    // before PostUpdate's propagate_transforms runs).
    let observed_in_update = Arc::new(Mutex::new(None::<[Vec3; 4]>));
    let observed_clone = Arc::clone(&observed_in_update);

    app.schedule.add_system(Stage::Update, move |world| {
        let s1 = world
            .get::<GlobalTransform>(e3d_with_local)
            .unwrap()
            .scale();
        let s2 = world
            .get::<GlobalTransform>(e3d_global_only)
            .unwrap()
            .scale();
        let s3 = world
            .get::<GlobalTransform>(e2d_with_local)
            .unwrap()
            .scale();
        let s4 = world
            .get::<GlobalTransform>(e2d_global_only)
            .unwrap()
            .scale();
        *observed_clone.lock().unwrap() = Some([s1, s2, s3, s4]);
    });

    run(&mut app, 2);

    let [s1, s2, s3, s4] = observed_in_update
        .lock()
        .unwrap()
        .expect("Update stage ran");
    assert!(
        (s1 - scale_3d_local).length() < 1e-4,
        "3D GlobalTransform.scale with LocalTransform clobbered before PostUpdate: got {s1:?}"
    );
    assert!(
        (s2 - scale_3d_global_only).length() < 1e-4,
        "3D GlobalTransform.scale without LocalTransform clobbered: got {s2:?}"
    );
    assert!(
        (s3 - scale_2d_local).length() < 1e-4,
        "2D GlobalTransform.scale with LocalTransform clobbered before PostUpdate: got {s3:?}"
    );
    assert!(
        (s4 - scale_2d_global_only).length() < 1e-4,
        "2D GlobalTransform.scale without LocalTransform clobbered: got {s4:?}"
    );
}
