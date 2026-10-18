//! Comprehensive Standalone Release Binary Export Verification (AUDIT-6.2).
//!
//! Validates that a project can be bundled into an optimized standalone distribution package
//! (`.pak` archive + manifest) and headlessly instantiated as an `EngineApp` runtime
//! with all remediated subsystems active:
//! 1. 3D physics with mechanical joints and transform synchronization.
//! 2. 2D physics with continuous collision detection (CCD) and persistent manifolds.
//! 3. 20 Hz asynchronous raytraced acoustics and audio rendering.

use korppi_asset::AssetServer;
use korppi_audio::asset::AudioClip;
use korppi_audio::ecs::AudioListener;
use korppi_audio::player::{AudioPlayer, AudioSource};
use korppi_audio::raytrace::AcousticMaterial;
use korppi_audio::spatial::SpatializerMode;
use korppi_audio::{AudioClipLoader, AudioRoom, AudioSettings};
use korppi_engine::app::{EntityJoint2d, EntityJoint3d, Joint3dConfig};
use korppi_engine::GraphicsBackend;
use korppi_math::{Vec2, Vec3};
use korppi_physics::{BodyType, Collider, RigidBody, Shape};
use korppi_physics2d::{
    body::BodyType as BodyType2d,
    collider::Collider2d,
    joint::{DistanceJoint2d, Joint2d},
    shape::{Circle2d, Polygon2d, Shape2d},
    RigidBody2d,
};
use korppi_project::export::{ExportConfig, ProjectExporter};
use korppi_project::manifest::{
    EntryPointsConfig, MetadataConfig, ProjectManifest, SubsystemsConfig,
};
use korppi_project::Project;
use korppi_scene::{GlobalTransform, LocalTransform};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

fn create_temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = format!(
        "korppi_export_test_{}_{}",
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    path.push(unique);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn test_standalone_export_bundles_and_simulates_all_remediated_subsystems() {
    let temp = create_temp_dir("standalone_all_subsystems");

    // 1. Create a project layout
    let project_dir = temp.join("project_src");
    let export_dir = temp.join("dist");
    fs::create_dir_all(&project_dir).unwrap();
    fs::create_dir_all(&export_dir).unwrap();

    let mut project = Project::create_empty(&project_dir, "AuditDungeon", "AuditTeam")
        .expect("Failed to create project");

    project.manifest = ProjectManifest {
        metadata: MetadataConfig {
            name: "AuditDungeon".to_string(),
            version: "1.0.0".to_string(),
            uuid: "00000000-0000-0000-0000-000000000001".to_string(),
            author: "AuditTeam".to_string(),
            engine_version: "0.1.0".to_string(),
            window_title: "Audit Dungeon Standalone".to_string(),
            default_resolution: [1280, 720],
        },
        entry_points: EntryPointsConfig {
            start_scene: "scenes/main.scene.json".to_string(),
            main_script: None,
            splash_screen: None,
        },
        subsystems: SubsystemsConfig {
            default_backend: GraphicsBackend::OpenGl,
            physics_gravity_3d: [0.0, -9.81, 0.0],
            physics_gravity_2d: [0.0, -9.81],
            fixed_tick_rate: 60,
            max_sub_steps: 4,
            audio_master_volume: 0.9,
            vfs_mounts: Vec::new(),
        },
    };

    // Write a mock start scene into scenes/ and an audio clip into assets/
    fs::write(
        project_dir.join("scenes").join("main.scene.json"),
        b"{\"name\": \"main_dungeon\", \"entities\": []}",
    )
    .unwrap();
    fs::write(
        project_dir.join("assets").join("tone.wav"),
        AudioClip::sine(440.0, 0.5, 48_000).to_wav_bytes(),
    )
    .unwrap();

    // 2. Export standalone release package
    let export_config = ExportConfig {
        output_dir: export_dir.clone(),
        target_name: "AuditDungeon".to_string(),
        embed_archive: false,
    };

    let export_output = ProjectExporter::export(&project, &export_config)
        .expect("ProjectExporter::export failed unexpectedly");

    assert!(export_output.package_path.exists());
    assert!(export_output.manifest_path.exists());

    // 3. Automated headless verification check
    ProjectExporter::verify_exported_bundle(&export_output).expect("verify_exported_bundle failed");

    // 4. Load runtime EngineApp directly from exported release distribution
    let mut app =
        ProjectExporter::load_exported_app(&export_output).expect("load_exported_app failed");

    // Verify configuration mapped accurately
    assert_eq!(app.config.app_name, "AuditDungeon");
    assert_eq!(app.config.tick_rate, 60);
    assert_eq!(app.physics_3d.gravity(), Vec3::new(0.0, -9.81, 0.0));
    assert_eq!(app.physics_2d.gravity(), Vec2::new(0.0, -9.81));

    // 5. Test 3D Physics & Mechanical Joints in the exported runtime
    let anchor_3d = app.world.spawn((
        RigidBody::new(BodyType::Static, Vec3::new(0.0, 5.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.2 }),
        LocalTransform::from_translation(Vec3::new(0.0, 5.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(0.0, 5.0, 0.0)),
    ));
    let bob_3d = app.world.spawn((
        RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 5.0, 0.0)),
        Collider::new(Shape::Sphere { radius: 0.2 }),
        LocalTransform::from_translation(Vec3::new(2.0, 5.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(2.0, 5.0, 0.0)),
    ));
    app.world.spawn((EntityJoint3d {
        entity_a: anchor_3d,
        entity_b: bob_3d,
        config: Joint3dConfig::Distance {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            distance: 2.0,
        },
    },));

    // 6. Test 2D Physics & CCD Bullet Tunneling Prevention
    let mut floor_2d = RigidBody2d::new(BodyType2d::Static);
    floor_2d.position = Vec2::new(0.0, 0.0);
    let floor_2d_col = Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.1)));
    app.world.spawn((floor_2d, floor_2d_col));

    let mut bullet_2d = RigidBody2d::new(BodyType2d::Dynamic);
    bullet_2d.position = Vec2::new(0.0, 1.0);
    bullet_2d.linear_velocity = Vec2::new(0.0, -100.0);
    bullet_2d.is_bullet = true; // CCD enabled
    let bullet_2d_col = Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.05)));
    let bullet_ent = app.world.spawn((bullet_2d, bullet_2d_col));

    // 7. Test 2D Joints
    let mut ground_anchor_2d = RigidBody2d::new(BodyType2d::Static);
    ground_anchor_2d.position = Vec2::new(5.0, 5.0);
    let g_col = Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)));
    let g_ent = app.world.spawn((ground_anchor_2d, g_col));

    let mut chained_box = RigidBody2d::new(BodyType2d::Dynamic);
    chained_box.position = Vec2::new(5.0, 3.0);
    let c_col = Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5)));
    let c_ent = app.world.spawn((chained_box, c_col));

    app.world.spawn((EntityJoint2d {
        entity_a: g_ent,
        entity_b: c_ent,
        joint: Joint2d::Distance(DistanceJoint2d::new(Vec2::ZERO, Vec2::ZERO, 2.0, 60.0, 1.0)),
    },));

    // 8. Test Asynchronous Raytraced Acoustics in the exported runtime
    app.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;

    // Closed room of static 3D physics colliders around the scene
    let (half, t) = (Vec3::new(10.0, 8.0, 10.0), 0.25);
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

    let clip = app
        .world
        .resource::<AssetServer>()
        .unwrap()
        .load_sync::<AudioClipLoader>("assets/tone.wav")
        .unwrap();

    app.world.spawn((
        AudioListener {
            active: true,
            is_primary: true,
            volume: 1.0,
        },
        LocalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        GlobalTransform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    ));
    app.world.spawn((
        AudioPlayer::new(AudioSource::clip(clip).looping()).autoplay(),
        LocalTransform::from_translation(Vec3::new(2.0, 1.0, 2.0)),
        GlobalTransform::from_translation(Vec3::new(2.0, 1.0, 2.0)),
    ));

    // 9. Step simulation for 60 ticks (1 second)
    let dt = Duration::from_nanos(1_000_000_000 / 60);
    for _ in 0..60 {
        app.update(dt);
    }
    for _ in 0..60 {
        if app.world.resource::<AudioRoom>().unwrap().traces() > 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
        app.update(dt);
    }

    // 10. Verify physics invariants
    // 3D Distance constraint
    let bob_body = app.world.get::<RigidBody>(bob_3d).unwrap();
    let dist_3d = (bob_body.position - Vec3::new(0.0, 5.0, 0.0)).length();
    assert!(
        (dist_3d - 2.0).abs() < 0.25,
        "3D distance joint constraint violated: expected ~2.0m, got {dist_3d}"
    );

    // 2D CCD Bullet constraint: did not tunnel below y=0
    let bullet_after = app.world.get::<RigidBody2d>(bullet_ent).unwrap();
    assert!(
        bullet_after.position.y >= 0.0,
        "2D CCD bullet tunneled past floor: y = {}",
        bullet_after.position.y
    );

    // 2D Distance joint constraint: chained box distance from anchor
    let chained_after = app.world.get::<RigidBody2d>(c_ent).unwrap();
    let dist_2d = (chained_after.position - Vec2::new(5.0, 5.0)).length();
    assert!(
        (dist_2d - 2.0).abs() < 0.25,
        "2D distance joint constraint violated: expected ~2.0m, got {dist_2d}"
    );

    // 11. Verify acoustics output rendered through the mixer
    assert!(
        app.world.resource::<AudioRoom>().unwrap().traces() > 0,
        "Exported application must complete at least one raytrace pass"
    );
    let energy: f32 = app.audio_output.last_output().iter().map(|s| s * s).sum();
    assert!(
        energy > 0.0,
        "Exported application must render acoustic energy"
    );

    // 12. Clean up
    app.quit();

    let _ = fs::remove_dir_all(temp);
}
