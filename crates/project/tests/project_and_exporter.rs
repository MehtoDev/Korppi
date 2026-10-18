//! Integration tests for project manifest, layout, asset packing, and standalone exporter.

use korppi_asset::Vfs;
use korppi_engine::GraphicsBackend;
use korppi_project::export::{AssetPacker, ExportConfig, ProjectExporter};
use korppi_project::manifest::{
    EntryPointsConfig, MetadataConfig, ProjectManifest, SubsystemsConfig, VfsMountConfig,
};
use korppi_project::Project;
use std::fs;
use std::path::PathBuf;

fn create_temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = format!(
        "korppi_test_{}_{}",
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
fn test_project_manifest_serialization_roundtrip() {
    let manifest = ProjectManifest {
        metadata: MetadataConfig {
            name: "Echoes of the Crypt".to_string(),
            version: "0.1.0".to_string(),
            uuid: "12345678-1234-5678-1234-567812345678".to_string(),
            author: "Korppi Team".to_string(),
            engine_version: "0.1.0".to_string(),
            window_title: "Echoes of the Crypt - Raytraced Dungeon".to_string(),
            default_resolution: [1920, 1080],
        },
        entry_points: EntryPointsConfig {
            start_scene: "scenes/dungeon_entrance.scene.json".to_string(),
            main_script: Some("scripts/main.lua".to_string()),
            splash_screen: Some("assets/textures/splash.png".to_string()),
        },
        subsystems: SubsystemsConfig {
            default_backend: GraphicsBackend::OpenGl,
            physics_gravity_3d: [0.0, -9.81, 0.0],
            physics_gravity_2d: [0.0, -9.81],
            fixed_tick_rate: 60,
            max_sub_steps: 8,
            audio_master_volume: 0.8,
            vfs_mounts: vec![VfsMountConfig {
                virtual_mount: "mods/".to_string(),
                physical_path: "addons/extra/".to_string(),
            }],
        },
    };

    let toml_str = manifest.to_toml().expect("Failed to serialize to TOML");
    let parsed = ProjectManifest::from_toml(&toml_str).expect("Failed to deserialize from TOML");

    assert_eq!(manifest, parsed);
}

#[test]
fn test_canonical_project_creation_and_directory_layout() {
    let temp = create_temp_dir("layout");
    let project =
        Project::create_empty(&temp, "Super Monkey Ball", "MonkeyDev").expect("Failed to create");

    assert_eq!(project.manifest.metadata.name, "Super Monkey Ball");
    assert_eq!(project.manifest.metadata.author, "MonkeyDev");

    // Check canonical directories exist
    assert!(temp.join("assets").is_dir());
    assert!(temp.join("scenes").is_dir());
    assert!(temp.join("config").is_dir());
    assert!(temp.join("build").is_dir());
    assert!(temp.join("korppi.project.toml").is_file());

    // Clean up
    let _ = fs::remove_dir_all(temp);
}

#[test]
fn test_project_loading_and_validation() {
    let temp = create_temp_dir("load_val");
    let _ = Project::create_empty(&temp, "Tetris", "ArcadeDev").expect("Failed to create");

    // Load from disk
    let loaded = Project::load(&temp).expect("Failed to load project");
    assert_eq!(loaded.manifest.metadata.name, "Tetris");

    // Validation should pass initially (or create starter start_scene)
    assert!(loaded.validate().is_ok());

    // Clean up
    let _ = fs::remove_dir_all(temp);
}

#[test]
fn test_vfs_mount_binding() {
    let temp = create_temp_dir("vfs_mount");
    let project = Project::create_empty(&temp, "VfsTest", "Dev").expect("Failed to create");

    // Write dummy scene and asset
    fs::write(
        temp.join("scenes").join("level_01.scene.json"),
        b"{\"entities\": []}",
    )
    .unwrap();
    fs::write(
        temp.join("assets").join("sprite.png"),
        b"PNG_MOCK_DATA_BYTES",
    )
    .unwrap();

    let mut vfs = Vfs::new();
    project.bind_vfs(&mut vfs).expect("Failed to bind VFS");

    assert!(vfs.exists("scenes/level_01.scene.json"));
    assert!(vfs.exists("assets/sprite.png"));

    let sprite_bytes = vfs.read("assets/sprite.png").unwrap();
    assert_eq!(sprite_bytes, b"PNG_MOCK_DATA_BYTES");

    // Clean up
    let _ = fs::remove_dir_all(temp);
}

#[test]
fn test_asset_packing_into_archive() {
    let temp = create_temp_dir("pack");
    let project = Project::create_empty(&temp, "PackGame", "Dev").expect("Failed to create");

    fs::write(temp.join("assets").join("hero.png"), b"HERO_DATA").unwrap();
    fs::write(
        temp.join("scenes").join("start.json"),
        b"{\"name\": \"start\"}",
    )
    .unwrap();

    let packer = AssetPacker::new();
    let packed = packer.pack(&project).expect("Failed to pack assets");

    assert!(packed.data.len() > 8);
    assert_eq!(&packed.data[0..4], b"EARC"); // Magic archive header
    assert!(packed.entries.contains(&"assets/hero.png".to_string()));
    assert!(packed.entries.contains(&"scenes/start.json".to_string()));

    // Clean up
    let _ = fs::remove_dir_all(temp);
}

#[test]
fn test_standalone_exporter_and_headless_verification() {
    let temp = create_temp_dir("export");
    let project = Project::create_empty(&temp, "ExportShowcase", "Dev").expect("Failed to create");

    // Put a mock scene file
    fs::write(
        temp.join("scenes").join("main.scene.json"),
        b"{\"entities\": []}",
    )
    .unwrap();

    let config = ExportConfig {
        output_dir: temp.join("build").join("export"),
        target_name: "ExportShowcase".to_string(),
        embed_archive: false,
    };

    let export_output =
        ProjectExporter::export(&project, &config).expect("Export failed unexpectedly");

    assert!(export_output.package_path.exists());
    assert!(export_output.manifest_path.exists());

    // Headless verification test
    let verify_result = ProjectExporter::verify_exported_bundle(&export_output);
    assert!(
        verify_result.is_ok(),
        "Verification failed: {:?}",
        verify_result.err()
    );

    // Clean up
    let _ = fs::remove_dir_all(temp);
}
