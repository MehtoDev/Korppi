//! Integration tests for Korppi Engine Coordinator (`korppi_engine`).

use korppi_audio::AudioListener;
use korppi_ecs::Stage;
use korppi_engine::{EngineApp, GraphicsBackend};
use korppi_math::{Transform, Vec2, Vec3};
use korppi_physics::{BodyType, Collider, RigidBody, Shape};
use korppi_physics2d::{
    body::BodyType as BodyType2d,
    collider::Collider2d,
    shape::{Circle2d, Shape2d},
    RigidBody2d,
};
use korppi_scene::{attach_child, GlobalTransform, LocalTransform};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn test_engine_app_creation_and_defaults() {
    let app = EngineApp::new().expect("Failed to initialize EngineApp");
    assert_eq!(app.config.tick_rate, 60);
    assert_eq!(app.config.backend, GraphicsBackend::Headless);
    assert!(app.is_running);
    assert_eq!(app.metrics.total_fixed_ticks, 0);
    assert_eq!(app.metrics.total_frames, 0);
}

#[test]
fn test_deterministic_fixed_timestep_simulation() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Counter tracking system executions per stage
    let pre_count = Arc::new(AtomicU32::new(0));
    let fixed_count = Arc::new(AtomicU32::new(0));
    let update_count = Arc::new(AtomicU32::new(0));
    let post_count = Arc::new(AtomicU32::new(0));
    let render_count = Arc::new(AtomicU32::new(0));

    {
        let c = pre_count.clone();
        app.schedule.add_system(Stage::PreUpdate, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let c = fixed_count.clone();
        app.schedule.add_system(Stage::FixedUpdate, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let c = update_count.clone();
        app.schedule.add_system(Stage::Update, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let c = post_count.clone();
        app.schedule.add_system(Stage::PostUpdate, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        let c = render_count.clone();
        app.schedule.add_system(Stage::Render, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });
    }

    // Step by 1/60th second (exactly 1 fixed update)
    let dt_60 = Duration::from_nanos(1_000_000_000 / 60);
    app.update(dt_60);

    assert_eq!(app.metrics.total_frames, 1);
    assert_eq!(app.metrics.total_fixed_ticks, 1);
    assert_eq!(pre_count.load(Ordering::SeqCst), 1);
    assert_eq!(fixed_count.load(Ordering::SeqCst), 1);
    assert_eq!(update_count.load(Ordering::SeqCst), 1);
    assert_eq!(post_count.load(Ordering::SeqCst), 1);
    assert_eq!(render_count.load(Ordering::SeqCst), 1);

    // Step by 3/60th second (3 fixed updates)
    app.update(dt_60 * 3);
    assert_eq!(app.metrics.total_frames, 2);
    assert_eq!(app.metrics.total_fixed_ticks, 4);
    assert_eq!(fixed_count.load(Ordering::SeqCst), 4);
}

#[test]
fn test_dual_physics_and_transform_synchronization() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Spawn 3D dynamic body falling under gravity
    let body_3d = RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 10.0, 0.0));
    let col_3d = Collider::new(Shape::Sphere { radius: 1.0 });

    let e_3d = app.world.spawn((
        body_3d,
        col_3d,
        LocalTransform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        GlobalTransform::from_transform(Transform::from_translation(Vec3::new(0.0, 10.0, 0.0))),
    ));

    // Spawn 2D dynamic body falling under 2D gravity
    let mut body_2d = RigidBody2d::new(BodyType2d::Dynamic);
    body_2d.position = Vec2::new(0.0, 15.0);
    let col_2d = Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0)));

    let e_2d = app.world.spawn((
        body_2d,
        col_2d,
        LocalTransform::from_translation(Vec3::new(0.0, 15.0, 0.0)),
        GlobalTransform::from_transform(Transform::from_translation(Vec3::new(0.0, 15.0, 0.0))),
    ));

    // Step 10 ticks (10 / 60 seconds)
    let dt_60 = Duration::from_nanos(1_000_000_000 / 60);
    for _ in 0..10 {
        app.update(dt_60);
    }

    // Verify 3D body position fell downwards
    let b3d = app.world.get::<RigidBody>(e_3d).unwrap();
    let gt3d = app.world.get::<GlobalTransform>(e_3d).unwrap();
    assert!(
        b3d.position.y < 10.0,
        "3D body should have fallen: {}",
        b3d.position.y
    );
    assert!(
        gt3d.translation().y < 10.0,
        "3D GlobalTransform should reflect fall: {}",
        gt3d.translation().y
    );

    // Verify 2D body position fell downwards
    let b2d = app.world.get::<RigidBody2d>(e_2d).unwrap();
    let gt2d = app.world.get::<GlobalTransform>(e_2d).unwrap();
    assert!(
        b2d.position.y < 15.0,
        "2D body should have fallen: {}",
        b2d.position.y
    );
    assert!(
        gt2d.translation().y < 15.0,
        "2D GlobalTransform should reflect fall: {}",
        gt2d.translation().y
    );
}

#[test]
fn test_scene_hierarchy_transform_propagation() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    let parent = app.world.spawn((
        LocalTransform::from_translation(Vec3::new(5.0, 0.0, 0.0)),
        GlobalTransform::default(),
    ));

    let child = app.world.spawn((
        LocalTransform::from_translation(Vec3::new(0.0, 3.0, 0.0)),
        GlobalTransform::default(),
    ));

    attach_child(&mut app.world, parent, child).expect("Failed to attach child");

    // Run engine update
    app.update(Duration::from_millis(16));

    let child_gt = app.world.get::<GlobalTransform>(child).unwrap();
    assert_eq!(child_gt.translation(), Vec3::new(5.0, 3.0, 0.0));
}

#[test]
fn test_audio_and_input_integration() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Add audio listener
    app.world.spawn((
        AudioListener::default(),
        GlobalTransform::from_transform(Transform::from_translation(Vec3::ZERO)),
    ));

    // Advance frame
    app.update(Duration::from_millis(16));

    // Verify audio coordinator was updated without errors
    assert_eq!(app.audio.mode(), korppi_audio::SpatializerMode::Standard);
}

#[test]
fn test_runtime_backend_toggle() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");
    assert_eq!(app.config.backend, GraphicsBackend::Headless);
    assert!(app.graphics_device.is_none());

    // Toggle to OpenGl (Mesa EGL headless)
    let res = app.set_graphics_backend(GraphicsBackend::OpenGl);
    assert!(res.is_ok(), "OpenGl headless initialization failed");
    assert!(app.graphics_device.is_some());
    assert_eq!(app.config.backend, GraphicsBackend::OpenGl);

    // Toggle back to Headless
    let res = app.set_graphics_backend(GraphicsBackend::Headless);
    assert!(res.is_ok());
    assert!(app.graphics_device.is_none());
}

#[test]
fn test_frame_budget_stress_test() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Spawn 100 physics bodies
    for i in 0..100 {
        let x = (i % 10) as f32 * 2.0;
        let y = (i / 10) as f32 * 2.0 + 5.0;
        let body = RigidBody::new(BodyType::Dynamic, Vec3::new(x, y, 0.0));
        let col = Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        });
        app.world.spawn((
            body,
            col,
            LocalTransform::from_translation(Vec3::new(x, y, 0.0)),
            GlobalTransform::from_transform(Transform::from_translation(Vec3::new(x, y, 0.0))),
        ));
    }

    let dt = Duration::from_nanos(1_000_000_000 / 60);
    for _ in 0..30 {
        app.update(dt);
    }

    assert_eq!(app.metrics.total_fixed_ticks, 30);
    assert_eq!(app.metrics.total_frames, 30);
    assert!(
        !app.metrics.is_budget_exceeded(100.0),
        "Frame took too long: {} ms",
        app.metrics.frame_time_ms
    );
}
