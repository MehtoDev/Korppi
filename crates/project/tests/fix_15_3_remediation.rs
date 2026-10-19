//! Primary regression tests for Milestone FIX-15.3 (`crates/project`):
//! - `PRJ-01`: `AssetPacker::pack` deterministic lexicographical `.pak` ordering, inclusion of
//!   custom `vfs_mounts` and `entry_points` (`start_scene`, `main_script`, `splash_screen`),
//!   and rejection/exclusion of symlinks or paths escaping `project.root_dir`.
//! - `PRJ-02`: `ProjectExporter::export` standalone executable emission and embedded `.pak`
//!   archive bundling when `ExportConfig::embed_archive` is `true`, plus roundtrip loading via
//!   `ProjectExporter::load_exported_app` and `ProjectExporter::verify_exported_bundle`.

use korppi_asset::{ArchiveFileSystem, FileSystem};
use korppi_project::export::{AssetPacker, ExportConfig, ProjectExporter};
use korppi_project::manifest::VfsMountConfig;
use korppi_project::Project;
use std::fs;
use std::path::PathBuf;

fn create_temp_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let unique = format!(
        "korppi_fix_15_3_{}_{}",
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
fn test_prj_01_asset_packer_deterministic_order_mounts_entry_points_and_symlink_rejection() {
    let dir = create_temp_dir("prj_01");
    let mut project = Project::create_empty(&dir, "PackAudit", "Auditor").expect("create_empty");

    // 1. Write assets in non-lexicographical order
    fs::write(dir.join("assets").join("z_last.txt"), b"z_payload").unwrap();
    fs::write(dir.join("assets").join("a_first.txt"), b"a_payload").unwrap();
    fs::write(dir.join("assets").join("m_mid.txt"), b"m_payload").unwrap();

    // 2. Configure custom vfs_mount and entry points outside assets/scenes/config
    fs::create_dir_all(dir.join("addons").join("extra")).unwrap();
    fs::write(
        dir.join("addons").join("extra").join("mod.txt"),
        b"mod_data",
    )
    .unwrap();
    fs::create_dir_all(dir.join("scripts")).unwrap();
    fs::write(dir.join("scripts").join("main.lua"), b"-- main script").unwrap();
    fs::create_dir_all(dir.join("branding")).unwrap();
    fs::write(dir.join("branding").join("splash.png"), b"SPLASH_PNG").unwrap();

    project.manifest.entry_points.main_script = Some("scripts/main.lua".to_string());
    project.manifest.entry_points.splash_screen = Some("branding/splash.png".to_string());
    project.manifest.subsystems.vfs_mounts.push(VfsMountConfig {
        virtual_mount: "mods".to_string(),
        physical_path: "addons/extra".to_string(),
    });

    // 3. Create a symlink inside assets/ pointing to an external file outside project.root_dir
    let outside_file = dir.parent().unwrap().join(format!(
        "korppi_prj01_outside_secret_{}.txt",
        std::process::id()
    ));
    fs::write(&outside_file, b"SECRET_HOST_FILE").unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&outside_file, dir.join("assets").join("leak.txt"))
            .expect("create file symlink");
        // Also create a directory symlink cycle to ensure traversal does not infinite-loop
        std::os::unix::fs::symlink(&dir, dir.join("assets").join("cycle_dir"))
            .expect("create dir symlink");
    }

    let packed_a = AssetPacker::new().pack(&project).expect("pack first run");
    let packed_b = AssetPacker::new().pack(&project).expect("pack second run");

    // Byte-for-byte deterministic across runs
    assert_eq!(
        packed_a.data, packed_b.data,
        "AssetPacker::pack must produce byte-identical .pak archives across runs"
    );
    assert_eq!(packed_a.entries, packed_b.entries);

    // Strictly sorted lexicographical entry table
    let mut sorted_entries = packed_a.entries.clone();
    sorted_entries.sort();
    assert_eq!(
        packed_a.entries, sorted_entries,
        "Packed entries must be in strict lexicographical order"
    );

    // Custom vfs_mount and entry_points must be included
    assert!(
        packed_a.entries.iter().any(|e| e == "addons/extra/mod.txt"),
        "Custom vfs_mount physical path file must be packed, got {:?}",
        packed_a.entries
    );
    assert!(
        packed_a.entries.iter().any(|e| e == "scripts/main.lua"),
        "Declared main_script entry point must be packed, got {:?}",
        packed_a.entries
    );
    assert!(
        packed_a.entries.iter().any(|e| e == "branding/splash.png"),
        "Declared splash_screen entry point must be packed, got {:?}",
        packed_a.entries
    );

    // External symlink must NOT be packed into the archive
    assert!(
        !packed_a.entries.iter().any(|e| e.contains("leak.txt")),
        "External symlink must be rejected/skipped during packing, got {:?}",
        packed_a.entries
    );

    // ArchiveFileSystem must read back the packed files accurately
    let archive_fs = ArchiveFileSystem::from_bytes(&packed_a.data).expect("valid EARC archive");
    assert_eq!(
        archive_fs.read("scripts/main.lua").unwrap(),
        b"-- main script"
    );
    assert_eq!(
        archive_fs.read("branding/splash.png").unwrap(),
        b"SPLASH_PNG"
    );

    let _ = fs::remove_file(outside_file);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn test_prj_02_exporter_embed_archive_standalone_executable_and_roundtrip() {
    let dir = create_temp_dir("prj_02");
    let project = Project::create_empty(&dir, "ExportBin", "Auditor").expect("create_empty");
    fs::write(dir.join("assets").join("hello.txt"), b"embedded_hello").unwrap();

    // 1. Export with embed_archive = true
    let out_embedded_dir = dir.join("dist_embedded");
    let embedded_cfg = ExportConfig {
        output_dir: out_embedded_dir.clone(),
        target_name: "ExportBin".to_string(),
        embed_archive: true,
    };
    let embedded_out =
        ProjectExporter::export(&project, &embedded_cfg).expect("export embed_archive=true");

    let exe_name = if cfg!(windows) {
        "ExportBin.exe"
    } else {
        "ExportBin"
    };
    let bin_path = out_embedded_dir.join(exe_name);
    assert!(
        bin_path.is_file(),
        "Standalone executable binary must exist at {:?}",
        bin_path
    );
    assert_eq!(
        embedded_out.package_path, bin_path,
        "When embed_archive is true, package_path must point to the self-contained executable"
    );
    assert!(
        !out_embedded_dir.join("ExportBin.pak").exists(),
        "When embed_archive is true, no separate sidecar .pak file should be emitted"
    );

    // Verify the embedded executable can be verified and loaded into an EngineApp
    ProjectExporter::verify_exported_bundle(&embedded_out)
        .expect("verify_exported_bundle must succeed on embedded executable output");
    let app = ProjectExporter::load_exported_app(&embedded_out)
        .expect("load_exported_app must succeed on embedded executable output");
    assert_eq!(app.config.app_name, "ExportBin");

    // 2. Export with embed_archive = false (sidecar .pak + manifest + launcher)
    let out_sidecar_dir = dir.join("dist_sidecar");
    let sidecar_cfg = ExportConfig {
        output_dir: out_sidecar_dir.clone(),
        target_name: "ExportBin".to_string(),
        embed_archive: false,
    };
    let sidecar_out =
        ProjectExporter::export(&project, &sidecar_cfg).expect("export embed_archive=false");
    assert_eq!(
        sidecar_out.package_path,
        out_sidecar_dir.join("ExportBin.pak")
    );
    assert!(sidecar_out.package_path.is_file());
    assert!(sidecar_out.manifest_path.is_file());
    ProjectExporter::verify_exported_bundle(&sidecar_out)
        .expect("verify_exported_bundle must succeed on sidecar .pak output");

    let _ = fs::remove_dir_all(dir);
}
