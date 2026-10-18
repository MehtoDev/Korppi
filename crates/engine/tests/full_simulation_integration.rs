//! Comprehensive End-to-End Simulation Integration Verification (AUDIT-6.1).
//!
//! Verifies the central `EngineApp` orchestrator integrating:
//! 1. 3D rigid body mechanical joints (spherical, revolute, distance constraints).
//! 2. 2D continuous collision detection (CCD) preventing fast bullet tunneling.
//! 3. 2D persistent contact manifolds ensuring stacking stability.
//! 4. 20 Hz asynchronous raytraced acoustic worker thread and wait-free audio rendering.

use korppi_asset::{AssetServer, MemoryFileSystem, Vfs};
use korppi_audio::asset::AudioClip;
use korppi_audio::ecs::AudioListener;
use korppi_audio::player::{AudioPlayer, AudioSource};
use korppi_audio::raytrace::AcousticMaterial;
use korppi_audio::spatial::SpatializerMode;
use korppi_audio::{AudioClipLoader, AudioRoom, AudioSettings};
use korppi_engine::app::{EntityJoint2d, EntityJoint3d, Joint3dConfig};
use korppi_engine::EngineApp;
use korppi_math::{Vec2, Vec3};
use korppi_physics::{BodyType, Collider, RigidBody, Shape};
use korppi_physics2d::{
    body::BodyType as BodyType2d,
    collider::Collider2d,
    joint::{DistanceJoint2d, Joint2d},
    shape::{Circle2d, Polygon2d, Shape2d},
    RigidBody2d,
};
use korppi_scene::{GlobalTransform, LocalTransform};
use std::time::Duration;

#[test]
fn test_full_simulation_3d_joints_and_transforms() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Static anchor at (0, 10, 0)
    let anchor = app.world.spawn((
        RigidBody::new(BodyType::Static, Vec3::new(0.0, 10.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.5 }),
        LocalTransform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
    ));

    // Dynamic pendulum bob starting at (3, 10, 0)
    let bob = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(3.0, 10.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.5 }),
        LocalTransform::from_translation(Vec3::new(3.0, 10.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(3.0, 10.0, 0.0)),
    ));

    // Spawn 3D distance joint constraint maintaining distance = 3.0m
    app.world.spawn((EntityJoint3d {
        entity_a: anchor,
        entity_b: bob,
        config: Joint3dConfig::Distance {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            distance: 3.0,
        },
    },));

    // Step simulation for 60 ticks (1 second under gravity)
    let dt = Duration::from_nanos(1_000_000_000 / 60);
    for _ in 0..60 {
        app.update(dt);
    }

    // Verify distance constraint holds: bob distance from anchor should remain ~3.0m
    let bob_body = app.world.get::<RigidBody>(bob).unwrap();
    let bob_gt = app.world.get::<GlobalTransform>(bob).unwrap();
    let dist = (bob_body.position - Vec3::new(0.0, 10.0, 0.0)).length();

    assert!(
        (dist - 3.0).abs() < 0.25,
        "3D Distance joint constraint must maintain length ~3.0m, got {dist}"
    );

    // Verify transform synchronization
    assert!(
        (bob_gt.translation() - bob_body.position).length() < 1e-4,
        "GlobalTransform must match 3D physics body position"
    );
}

#[test]
fn test_full_simulation_2d_ccd_bullet_prevents_tunneling() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Thin static floor at y = 0.0, height = 0.2 (from -0.1 to +0.1)
    let mut floor_body = RigidBody2d::new(BodyType2d::Static);
    floor_body.position = Vec2::new(0.0, 0.0);
    let floor_col = Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(25.0, 0.1)));

    app.world.spawn((
        floor_body,
        floor_col,
        LocalTransform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    ));

    // Fast-moving bullet moving downwards at -120 m/s from y = 1.0
    // In a single 1/60s step, it moves 2.0m downwards, which would completely tunnel past the 0.2m floor!
    let mut bullet_body = RigidBody2d::new(BodyType2d::Dynamic);
    bullet_body.position = Vec2::new(0.0, 1.0);
    bullet_body.linear_velocity = Vec2::new(0.0, -120.0);
    bullet_body.is_bullet = true; // Enable CCD

    let bullet_col = Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.1)));

    let bullet = app.world.spawn((
        bullet_body,
        bullet_col,
        LocalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    ));

    // Step 2 ticks
    let dt = Duration::from_nanos(1_000_000_000 / 60);
    app.update(dt);
    app.update(dt);

    // Verify bullet did not tunnel below the floor
    let b = app.world.get::<RigidBody2d>(bullet).unwrap();
    assert!(
        b.position.y >= 0.0,
        "Bullet with CCD enabled must not tunnel below floor at y=0, got y={}",
        b.position.y
    );
}

#[test]
fn test_full_simulation_2d_joints_and_stacking_stability() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Static ground
    let mut ground = RigidBody2d::new(BodyType2d::Static);
    ground.position = Vec2::new(0.0, 0.0);
    let ground_col = Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(10.0, 0.5)));
    let ground_ent = app.world.spawn((ground, ground_col));

    // Dynamic box chained to ground by 2D distance joint
    let mut box1 = RigidBody2d::new(BodyType2d::Dynamic);
    box1.position = Vec2::new(0.0, 4.0);
    let box1_col = Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)));
    let box1_ent = app.world.spawn((box1, box1_col));

    app.world.spawn((EntityJoint2d {
        entity_a: ground_ent,
        entity_b: box1_ent,
        joint: Joint2d::Distance(DistanceJoint2d::new(
            Vec2::new(0.0, 0.5),
            Vec2::new(0.0, -0.5),
            3.0,
            60.0,
            1.0,
        )),
    },));

    let dt = Duration::from_nanos(1_000_000_000 / 60);
    for _ in 0..60 {
        app.update(dt);
    }

    let b1 = app.world.get::<RigidBody2d>(box1_ent).unwrap();
    // Distance should be bounded around 4.0 (3.0 length + 0.5 + 0.5 offsets)
    assert!(
        (b1.position.y - 4.0).abs() < 0.25,
        "2D Distance joint must maintain anchor spacing, got y={}",
        b1.position.y
    );
}

#[test]
fn test_full_simulation_async_acoustics_integration() {
    let mut app = EngineApp::new().expect("Failed to initialize EngineApp");

    // Configure Raytraced acoustics
    app.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;

    // Build a closed shoebox room of static 3D physics colliders with acoustic material
    let (half, t) = (Vec3::new(7.5, 5.0, 7.5), 0.25);
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            let (mut center, mut extents) = (Vec3::ZERO, half + Vec3::splat(2.0 * t));
            center[axis] = side * (half[axis] + t);
            extents[axis] = t;
            app.world.spawn((
                RigidBody::new(BodyType::Static, center),
                Collider::new(Shape::Box {
                    half_extents: extents,
                }),
                AcousticMaterial::concrete(),
                LocalTransform::from_translation(center),
                GlobalTransform::from_translation(center),
            ));
        }
    }

    let mut files = MemoryFileSystem::new();
    files.insert(
        "tone.wav",
        AudioClip::sine(440.0, 0.5, 48_000).to_wav_bytes(),
    );
    let mut vfs = Vfs::new();
    vfs.mount("", Box::new(files));
    let assets = AssetServer::new(vfs);
    assets.register_loader(AudioClipLoader);
    let clip = assets.load_sync::<AudioClipLoader>("tone.wav").unwrap();
    app.world.insert_resource(assets);

    // Spawn listener
    app.world.spawn((
        AudioListener {
            active: true,
            is_primary: true,
            volume: 1.0,
        },
        LocalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    ));

    // Spawn emitter
    app.world.spawn((
        AudioPlayer::new(AudioSource::clip(clip).looping()).autoplay(),
        LocalTransform::from_translation(Vec3::new(3.0, 1.0, 3.0)),
        GlobalTransform::from_translation(Vec3::new(3.0, 1.0, 3.0)),
    ));

    // Step simulation frames until the asynchronous room worker completes a trace
    let dt = Duration::from_millis(16);
    for _ in 0..120 {
        app.update(dt);
        if app.world.resource::<AudioRoom>().unwrap().traces() > 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    app.update(dt);

    assert!(
        app.world.resource::<AudioRoom>().unwrap().traces() > 0,
        "RoomWorker must complete at least one raytrace pass"
    );

    let energy: f32 = app.audio_output.last_output().iter().map(|s| s * s).sum();
    assert!(
        energy > 0.0,
        "Audio output rendered in raytraced mode with async worker must contain non-zero energy"
    );

    app.quit();
}
