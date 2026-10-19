//! Primary regression tests for Milestone FIX-15.4 (`crates/project`):
//! - `PRJ-03`: `ProjectExporter::load_exported_app` deserializes and spawns `start_scene` into
//!   `app.world`, registers all standard asset loaders (`TextureLoader`, `MeshLoader`,
//!   `ShaderLoader`, `AudioClipLoader`, `OggVorbisLoader`, `WavAudioLoader`), mounts custom
//!   `vfs_mounts`, and `verify_exported_bundle` rejects corrupt `start_scene` data.
//! - `PRJ-04`: `Project::validate` rejects path traversal (`..`) and absolute paths across
//!   `entry_points` and `vfs_mounts`, `fixed_tick_rate == 0`, zero `default_resolution`,
//!   non-finite gravity or negative/non-finite `audio_master_volume`, and nil UUIDs;
//!   `ProjectManifest::new` generates a deterministic non-nil UUID.

use korppi_asset::{AssetServer, MeshLoader, ShaderLoader, TextureLoader};
use korppi_audio::asset::AudioClip;
use korppi_audio::{AudioClipLoader, OggVorbisLoader, WavAudioLoader};
use korppi_project::export::{ExportConfig, ExportOutput, ProjectExporter};
use korppi_project::manifest::{ProjectManifest, VfsMountConfig};
use korppi_project::Project;
use korppi_scene::{GlobalTransform, LocalTransform, Name, Scene, SceneRegistry};
use std::fs;
use std::path::PathBuf;

fn create_temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = format!(
        "korppi_fix_15_4_{}_{}",
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
fn test_prj_03_load_exported_app_spawns_start_scene_registers_loaders_and_mounts_vfs() {
    let dir = create_temp_dir("prj_03");
    let mut project = Project::create_empty(&dir, "SceneLoadApp", "Auditor").expect("create_empty");

    // 1. Starter scene created by Project::create_empty must be valid Scene JSON
    let starter_json =
        fs::read_to_string(dir.join(&project.manifest.entry_points.start_scene)).unwrap();
    Scene::from_json(&starter_json).expect("starter scene must parse via Scene::from_json");

    // 2. Author a Scene with a named entity and transforms in scenes/main.scene.json
    let mut src_app = korppi_engine::EngineApp::new().expect("src_app");
    let registry = SceneRegistry::default();
    let hero = src_app.world.spawn((
        Name::new("HeroFromStartScene"),
        LocalTransform::default(),
        GlobalTransform::default(),
    ));
    let scene = Scene::from_world(&src_app.world, &[hero], &registry).expect("scene from_world");
    let valid_json = scene.to_json_pretty().expect("scene to_json_pretty");
    fs::write(
        dir.join(&project.manifest.entry_points.start_scene),
        valid_json,
    )
    .unwrap();

    // 3. Write a 1x1 PNG texture into assets/icon.png, an OBJ mesh into assets/quad.obj,
    // a WAV audio file into assets/beep.wav, and a custom VFS mount asset into addons/extra/mod_icon.png
    let png_bytes: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    fs::write(dir.join("assets").join("icon.png"), png_bytes).unwrap();
    let obj_bytes = b"v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n";
    fs::write(dir.join("assets").join("tri.obj"), obj_bytes).unwrap();
    let glsl_bytes = b"#version 450\nvoid main() {}\n";
    fs::write(dir.join("assets").join("test.vert"), glsl_bytes).unwrap();
    let wav_bytes = AudioClip::sine(440.0, 0.05, 48_000).to_wav_bytes();
    fs::write(dir.join("assets").join("beep.wav"), &wav_bytes).unwrap();

    fs::create_dir_all(dir.join("addons").join("extra")).unwrap();
    fs::write(
        dir.join("addons").join("extra").join("mod_icon.png"),
        png_bytes,
    )
    .unwrap();
    project.manifest.subsystems.vfs_mounts.push(VfsMountConfig {
        virtual_mount: "mods".to_string(),
        physical_path: "addons/extra".to_string(),
    });

    let cfg = ExportConfig {
        output_dir: dir.join("dist"),
        target_name: "SceneLoadApp".to_string(),
        embed_archive: false,
    };
    let output = ProjectExporter::export(&project, &cfg).expect("export");
    ProjectExporter::verify_exported_bundle(&output).expect("verify_exported_bundle");

    let app = ProjectExporter::load_exported_app(&output).expect("load_exported_app");
    assert_eq!(
        app.world.entity_count(),
        1,
        "load_exported_app must instantiate entities from start_scene into app.world"
    );
    let spawned_entity = app.world.entities().next().unwrap();
    assert_eq!(
        app.world.get::<Name>(spawned_entity).unwrap().as_str(),
        "HeroFromStartScene"
    );

    // 4. Verify all standard asset loaders and custom vfs_mounts work in the exported app
    let assets = app.world.resource::<AssetServer>().expect("AssetServer");
    assert!(
        assets.load_sync::<TextureLoader>("assets/icon.png").is_ok(),
        "TextureLoader must be registered on exported AssetServer"
    );
    assert!(
        assets.load_sync::<MeshLoader>("assets/tri.obj").is_ok(),
        "MeshLoader must be registered on exported AssetServer"
    );
    assert!(
        assets.load_sync::<ShaderLoader>("assets/test.vert").is_ok(),
        "ShaderLoader must be registered on exported AssetServer"
    );
    assert!(
        assets
            .load_sync::<AudioClipLoader>("assets/beep.wav")
            .is_ok(),
        "AudioClipLoader must be registered on exported AssetServer"
    );
    assert!(
        assets
            .load_sync::<WavAudioLoader>("assets/beep.wav")
            .is_ok(),
        "WavAudioLoader must be registered on exported AssetServer"
    );
    // OggVorbisLoader must also be registered (not MissingLoader; loading a missing file returns VfsError, not MissingLoader)
    let ogg_err = assets
        .load_sync::<OggVorbisLoader>("assets/nonexistent.ogg")
        .unwrap_err();
    assert!(
        !matches!(ogg_err, korppi_asset::AssetError::MissingLoader(_)),
        "OggVorbisLoader must be registered on exported AssetServer, got {ogg_err:?}"
    );

    // Custom vfs_mount ("mods" -> "addons/extra") must resolve in exported app
    assert!(
        assets
            .load_sync::<TextureLoader>("mods/mod_icon.png")
            .is_ok(),
        "Custom vfs_mount 'mods/mod_icon.png' must resolve in exported AssetServer"
    );

    // 5. Corrupt non-JSON start_scene must fail load_exported_app and verify_exported_bundle
    fs::write(
        dir.join(&project.manifest.entry_points.start_scene),
        b"NOT VALID JSON {{{{",
    )
    .unwrap();
    let corrupt_out = ProjectExporter::export(&project, &cfg).expect("export corrupt scene");
    assert!(
        ProjectExporter::load_exported_app(&corrupt_out).is_err(),
        "load_exported_app must return Err when start_scene contains invalid JSON"
    );
    assert!(
        ProjectExporter::verify_exported_bundle(&corrupt_out).is_err(),
        "verify_exported_bundle must return Err when start_scene contains invalid JSON"
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn test_prj_04_project_validation_rejects_traversal_invalid_subsystems_and_nil_uuid() {
    let dir = create_temp_dir("prj_04");
    let outside_scene = dir.parent().unwrap().join(format!(
        "korppi_prj04_outside_{}.scene.json",
        std::process::id()
    ));
    fs::write(&outside_scene, b"{\"entities\": []}").unwrap();

    let project = Project::create_empty(&dir, "ValAudit", "Auditor").expect("create_empty");

    // 1. ProjectManifest::new must generate a non-nil UUID
    assert_ne!(
        project.manifest.metadata.uuid, "00000000-0000-0000-0000-000000000000",
        "ProjectManifest::new must generate a non-nil UUID"
    );
    assert!(!project.manifest.metadata.uuid.trim().is_empty());
    let m2 = ProjectManifest::new("AnotherGame", "Auditor");
    assert_ne!(
        project.manifest.metadata.uuid, m2.metadata.uuid,
        "Distinct project names must produce distinct deterministic UUIDs"
    );
    assert!(project.validate().is_ok());

    // 2. Reject parent traversal `..` in start_scene
    {
        let mut p = project.clone();
        p.manifest.entry_points.start_scene = format!(
            "../{}",
            outside_scene.file_name().unwrap().to_string_lossy()
        );
        assert!(
            p.validate().is_err(),
            "Project::validate must reject '..' traversal in start_scene"
        );
        assert!(
            ProjectExporter::export(
                &p,
                &ExportConfig {
                    output_dir: dir.join("dist_trav"),
                    target_name: "ValAudit".to_string(),
                    embed_archive: false,
                }
            )
            .is_err(),
            "ProjectExporter::export must reject '..' traversal in start_scene"
        );
    }

    // 3. Reject parent traversal or absolute paths in vfs_mounts, main_script, splash_screen
    {
        let mut p = project.clone();
        p.manifest.subsystems.vfs_mounts.push(VfsMountConfig {
            virtual_mount: "mods".to_string(),
            physical_path: "../outside".to_string(),
        });
        assert!(p.validate().is_err());
    }
    {
        let mut p = project.clone();
        p.manifest.entry_points.main_script = Some("/etc/passwd".to_string());
        assert!(p.validate().is_err());
    }

    // 4. Reject zero default_resolution
    {
        let mut p = project.clone();
        p.manifest.metadata.default_resolution = [0, 720];
        assert!(p.validate().is_err());
    }

    // 5. Reject zero fixed_tick_rate (and never panic in load_exported_app / verify_exported_bundle)
    {
        let mut p = project.clone();
        p.manifest.subsystems.fixed_tick_rate = 0;
        assert!(p.validate().is_err());

        // Even if an ExportOutput is manually constructed with fixed_tick_rate = 0, verify_exported_bundle must return Err, not panic
        let valid_out = ProjectExporter::export(
            &project,
            &ExportConfig {
                output_dir: dir.join("dist_valid"),
                target_name: "ValAudit".to_string(),
                embed_archive: false,
            },
        )
        .expect("valid export");
        let mut forged_out: ExportOutput = valid_out;
        forged_out.manifest.subsystems.fixed_tick_rate = 0;
        let verify_res = ProjectExporter::verify_exported_bundle(&forged_out);
        assert!(
            verify_res.is_err(),
            "verify_exported_bundle with fixed_tick_rate=0 must return Err without panicking"
        );
    }

    // 6. Reject non-finite gravity and non-finite / negative audio_master_volume
    {
        let mut p = project.clone();
        p.manifest.subsystems.physics_gravity_3d = [f32::NAN, -9.81, 0.0];
        assert!(p.validate().is_err());
    }
    {
        let mut p = project.clone();
        p.manifest.subsystems.physics_gravity_2d = [0.0, f32::INFINITY];
        assert!(p.validate().is_err());
    }
    {
        let mut p = project.clone();
        p.manifest.subsystems.audio_master_volume = f32::NAN;
        assert!(p.validate().is_err());
    }
    {
        let mut p = project.clone();
        p.manifest.subsystems.audio_master_volume = -0.5;
        assert!(p.validate().is_err());
    }

    // 7. Reject nil UUID
    {
        let mut p = project.clone();
        p.manifest.metadata.uuid = "00000000-0000-0000-0000-000000000000".to_string();
        assert!(p.validate().is_err());
    }

    let _ = fs::remove_file(outside_scene);
    let _ = fs::remove_dir_all(dir);
}
