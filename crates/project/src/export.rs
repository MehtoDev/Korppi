//! Asset packing and standalone release binary exporter.

use crate::error::ProjectError;
use crate::manifest::ProjectManifest;
use crate::project::{Project, MANIFEST_FILENAME};
use korppi_asset::vfs::normalize_virtual_path;
use korppi_asset::{
    ArchiveFileSystem, AssetServer, FileSystem, MeshLoader, ShaderLoader, TextureLoader, Vfs,
    VfsError,
};
use korppi_audio::{AudioClipLoader, OggVorbisLoader, WavAudioLoader};
use korppi_engine::{EngineApp, EngineConfig};
use korppi_scene::{Scene, SceneRegistry};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Sub-directory view into an [`ArchiveFileSystem`] used to route custom `vfs_mounts` (`PRJ-03`).
#[derive(Debug, Clone)]
struct SubArchiveFileSystem {
    archive: Arc<ArchiveFileSystem>,
    root_prefix: String,
}

impl SubArchiveFileSystem {
    fn new(archive: Arc<ArchiveFileSystem>, root_prefix: &str) -> Self {
        let clean = normalize_virtual_path(root_prefix).unwrap_or_default();
        Self {
            archive,
            root_prefix: clean,
        }
    }

    fn resolve(&self, path: &str) -> Result<String, VfsError> {
        let clean = normalize_virtual_path(path)?;
        if self.root_prefix.is_empty() {
            Ok(clean)
        } else if clean.is_empty() {
            Ok(self.root_prefix.clone())
        } else {
            Ok(format!("{}/{clean}", self.root_prefix))
        }
    }
}

impl FileSystem for SubArchiveFileSystem {
    fn read(&self, path: &str) -> Result<Vec<u8>, VfsError> {
        let resolved = self.resolve(path)?;
        self.archive.read(&resolved)
    }

    fn exists(&self, path: &str) -> bool {
        self.resolve(path)
            .map(|p| self.archive.exists(&p))
            .unwrap_or(false)
    }

    fn is_file(&self, path: &str) -> bool {
        self.resolve(path)
            .map(|p| self.archive.is_file(&p))
            .unwrap_or(false)
    }

    fn is_dir(&self, path: &str) -> bool {
        self.resolve(path)
            .map(|p| self.archive.is_dir(&p))
            .unwrap_or(false)
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>, VfsError> {
        let resolved = self.resolve(path)?;
        self.archive.list_dir(&resolved)
    }
}

const EMBEDDED_ARCHIVE_MAGIC: &[u8; 4] = b"KPAK";
const EMBEDDED_TRAILER_LEN: usize = 20;

fn has_unsafe_path_components(path_str: &str) -> bool {
    let trimmed = path_str.trim();
    if trimmed.is_empty() || trimmed.starts_with('/') || trimmed.starts_with('\\') {
        return true;
    }
    Path::new(trimmed).components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
}

fn executable_file_name(target_name: &str) -> String {
    if cfg!(windows) && !target_name.ends_with(".exe") {
        format!("{target_name}.exe")
    } else {
        target_name.to_string()
    }
}

fn build_runner_stub(manifest: &ProjectManifest, embed_archive: bool) -> Vec<u8> {
    let mode = if embed_archive { "embedded" } else { "sidecar" };
    format!(
        "#!/bin/sh\n# Korppi Standalone Runtime Bundle ({mode})\n# App: {} ({})\n# StartScene: {}\nexec korppi-runtime \"$0\" \"$@\"\n",
        manifest.metadata.name,
        manifest.metadata.version,
        manifest.entry_points.start_scene
    )
    .into_bytes()
}

#[cfg(unix)]
fn mark_executable(path: &Path) -> Result<(), ProjectError> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn mark_executable(_path: &Path) -> Result<(), ProjectError> {
    Ok(())
}

/// Serialized packed asset archive.
#[derive(Debug, Clone)]
pub struct PackedArchive {
    /// Raw binary bytes of the packed archive (`.pak`).
    pub data: Vec<u8>,
    /// List of relative virtual file paths contained within the archive.
    pub entries: Vec<String>,
}

/// Asset packaging utility for bundling project files into optimized `.pak` archives.
#[derive(Debug, Default)]
pub struct AssetPacker {}

impl AssetPacker {
    /// Creates a new asset packer instance.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Recursively packages all assets, scenes, configs, custom VFS mounts, entry points,
    /// and the project manifest into a deterministic [`PackedArchive`].
    pub fn pack(&self, project: &Project) -> Result<PackedArchive, ProjectError> {
        let canonical_root = fs::canonicalize(&project.root_dir)?;
        let mut collected: BTreeMap<String, Vec<u8>> = BTreeMap::new();

        // 1. Insert serialized manifest
        let manifest_content = project.manifest.to_toml()?;
        collected.insert(MANIFEST_FILENAME.to_string(), manifest_content.into_bytes());

        // 2. Helper to validate and insert a single file relative to root_dir
        let add_sandboxed_file = |collected: &mut BTreeMap<String, Vec<u8>>,
                                  rel_str: &str|
         -> Result<(), ProjectError> {
            if has_unsafe_path_components(rel_str) {
                return Ok(());
            }
            let candidate = project.root_dir.join(rel_str);
            let Ok(meta) = fs::symlink_metadata(&candidate) else {
                return Ok(());
            };
            if meta.file_type().is_symlink() || !meta.is_file() {
                return Ok(());
            }
            let Ok(canon) = fs::canonicalize(&candidate) else {
                return Ok(());
            };
            if !canon.starts_with(&canonical_root) {
                return Ok(());
            }
            let rel = canon
                .strip_prefix(&canonical_root)
                .map_err(|e| ProjectError::Packing(e.to_string()))?
                .to_string_lossy()
                .replace('\\', "/");
            if let Ok(clean) = normalize_virtual_path(&rel) {
                if !clean.is_empty() && !collected.contains_key(&clean) {
                    collected.insert(clean, fs::read(&canon)?);
                }
            }
            Ok(())
        };

        // 3. Helper to recursively scan a directory inside canonical_root without following symlinks
        let scan_dir = |collected: &mut BTreeMap<String, Vec<u8>>,
                        dir_rel: &str|
         -> Result<(), ProjectError> {
            if has_unsafe_path_components(dir_rel) {
                return Ok(());
            }
            let physical_dir = project.root_dir.join(dir_rel);
            let Ok(dir_meta) = fs::symlink_metadata(&physical_dir) else {
                return Ok(());
            };
            if dir_meta.file_type().is_symlink() || !dir_meta.is_dir() {
                return Ok(());
            }
            let Ok(canon_dir) = fs::canonicalize(&physical_dir) else {
                return Ok(());
            };
            if !canon_dir.starts_with(&canonical_root) {
                return Ok(());
            }

            let mut stack = vec![canon_dir];
            while let Some(current) = stack.pop() {
                let mut dir_entries: Vec<fs::DirEntry> =
                    fs::read_dir(&current)?.collect::<Result<Vec<_>, _>>()?;
                dir_entries.sort_by_key(|e| e.file_name());

                for entry in dir_entries {
                    let path = entry.path();
                    let meta = fs::symlink_metadata(&path)?;
                    if meta.file_type().is_symlink() {
                        // Reject/skip symlinks so external files cannot leak and directory cycles cannot hang.
                        continue;
                    }
                    let Ok(canon) = fs::canonicalize(&path) else {
                        continue;
                    };
                    if !canon.starts_with(&canonical_root) {
                        continue;
                    }
                    if meta.is_dir() {
                        stack.push(canon);
                    } else if meta.is_file() {
                        let rel = canon
                            .strip_prefix(&canonical_root)
                            .map_err(|e| ProjectError::Packing(e.to_string()))?
                            .to_string_lossy()
                            .replace('\\', "/");
                        if let Ok(clean) = normalize_virtual_path(&rel) {
                            if !clean.is_empty() && !collected.contains_key(&clean) {
                                collected.insert(clean, fs::read(&canon)?);
                            }
                        }
                    }
                }
            }
            Ok(())
        };

        // Scan canonical directories
        scan_dir(&mut collected, "assets")?;
        scan_dir(&mut collected, "scenes")?;
        scan_dir(&mut collected, "config")?;

        // Scan all declared custom VFS mounts
        for mount in &project.manifest.subsystems.vfs_mounts {
            scan_dir(&mut collected, &mount.physical_path)?;
        }

        // Pack declared entry point files even if located outside assets/scenes/config
        add_sandboxed_file(&mut collected, &project.manifest.entry_points.start_scene)?;
        if let Some(ref script) = project.manifest.entry_points.main_script {
            add_sandboxed_file(&mut collected, script)?;
        }
        if let Some(ref splash) = project.manifest.entry_points.splash_screen {
            add_sandboxed_file(&mut collected, splash)?;
        }

        // Build archive in strict lexicographical order from BTreeMap
        let mut builder = ArchiveFileSystem::builder();
        let mut entries = Vec::with_capacity(collected.len());
        for (rel_path, content) in collected {
            builder.add_file(&rel_path, &content);
            entries.push(rel_path);
        }

        let data = builder.build();
        Ok(PackedArchive { data, entries })
    }
}

/// Configuration parameters for exporting a project.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Destination directory for exported release artifacts.
    pub output_dir: PathBuf,
    /// Base name of the target binary / package.
    pub target_name: String,
    /// Whether to embed the asset archive into the binary executable.
    pub embed_archive: bool,
}

/// Output metadata and paths generated by the export process.
#[derive(Debug, Clone)]
pub struct ExportOutput {
    /// Path to the exported asset package file (`.pak`) or self-contained executable when `embed_archive` is `true`.
    pub package_path: PathBuf,
    /// Path to the exported manifest file (or self-contained executable when `embed_archive` is `true`).
    pub manifest_path: PathBuf,
    /// Snapshot of the project manifest used during export.
    pub manifest: ProjectManifest,
}

/// Orchestrator for building standalone release distributions.
pub struct ProjectExporter {}

impl ProjectExporter {
    /// Exports the project to a standalone release package according to [`ExportConfig`].
    pub fn export(project: &Project, config: &ExportConfig) -> Result<ExportOutput, ProjectError> {
        project.validate()?;
        if has_unsafe_path_components(&config.target_name) {
            return Err(ProjectError::Validation(format!(
                "Invalid export target_name: '{}'",
                config.target_name
            )));
        }

        fs::create_dir_all(&config.output_dir)?;

        // 1. Pack assets deterministically
        let packer = AssetPacker::new();
        let packed = packer.pack(project)?;
        let exe_name = executable_file_name(&config.target_name);
        let exe_path = config.output_dir.join(&exe_name);

        if config.embed_archive {
            // Emit a single self-contained standalone executable embedding the .pak archive + trailer
            let mut bundle = build_runner_stub(&project.manifest, true);
            let pak_offset = bundle.len() as u64;
            let pak_len = packed.data.len() as u64;
            bundle.extend_from_slice(&packed.data);
            bundle.extend_from_slice(&pak_offset.to_le_bytes());
            bundle.extend_from_slice(&pak_len.to_le_bytes());
            bundle.extend_from_slice(EMBEDDED_ARCHIVE_MAGIC);

            fs::write(&exe_path, bundle)?;
            mark_executable(&exe_path)?;

            Ok(ExportOutput {
                package_path: exe_path.clone(),
                manifest_path: exe_path,
                manifest: project.manifest.clone(),
            })
        } else {
            // Emit sidecar .pak archive, manifest copy, and launcher executable
            let package_path = config
                .output_dir
                .join(format!("{}.pak", config.target_name));
            fs::write(&package_path, packed.data)?;

            let manifest_path = config.output_dir.join(MANIFEST_FILENAME);
            let manifest_toml = project.manifest.to_toml()?;
            fs::write(&manifest_path, manifest_toml)?;

            let stub = build_runner_stub(&project.manifest, false);
            fs::write(&exe_path, stub)?;
            mark_executable(&exe_path)?;

            Ok(ExportOutput {
                package_path,
                manifest_path,
                manifest: project.manifest.clone(),
            })
        }
    }

    fn extract_archive_bytes(raw_bytes: &[u8]) -> Result<&[u8], ProjectError> {
        if raw_bytes.len() >= 8 && &raw_bytes[0..4] == b"EARC" {
            return Ok(raw_bytes);
        }
        if raw_bytes.len() >= EMBEDDED_TRAILER_LEN
            && &raw_bytes[raw_bytes.len() - 4..] == EMBEDDED_ARCHIVE_MAGIC
        {
            let trailer_start = raw_bytes.len() - EMBEDDED_TRAILER_LEN;
            let pak_offset_u64 = u64::from_le_bytes(
                raw_bytes[trailer_start..trailer_start + 8]
                    .try_into()
                    .unwrap(),
            );
            let pak_len_u64 = u64::from_le_bytes(
                raw_bytes[trailer_start + 8..trailer_start + 16]
                    .try_into()
                    .unwrap(),
            );
            let offset = usize::try_from(pak_offset_u64).map_err(|_| {
                ProjectError::Export("Embedded archive offset overflows usize".to_string())
            })?;
            let len = usize::try_from(pak_len_u64).map_err(|_| {
                ProjectError::Export("Embedded archive length overflows usize".to_string())
            })?;
            let end = offset.checked_add(len).ok_or_else(|| {
                ProjectError::Export("Embedded archive offset + length overflow".to_string())
            })?;
            if end > trailer_start {
                return Err(ProjectError::Export(
                    "Embedded archive slice extends past bundle trailer".to_string(),
                ));
            }
            return Ok(&raw_bytes[offset..end]);
        }
        Err(ProjectError::Export(
            "File is neither a valid .pak archive nor an embedded standalone executable"
                .to_string(),
        ))
    }

    /// Instantiates and configures a headless [`EngineApp`] runtime loaded from an exported distribution bundle.
    pub fn load_exported_app(output: &ExportOutput) -> Result<EngineApp, ProjectError> {
        let registry = EngineApp::default_scene_registry();
        Self::load_exported_app_with_registry(output, &registry)
    }

    /// Instantiates and configures a headless [`EngineApp`] runtime loaded from an exported distribution bundle
    /// using the provided [`SceneRegistry`] to deserialize `start_scene` entities.
    pub fn load_exported_app_with_registry(
        output: &ExportOutput,
        registry: &SceneRegistry,
    ) -> Result<EngineApp, ProjectError> {
        output.manifest.validate()?;

        let raw_bytes = fs::read(&output.package_path)?;
        let archive_bytes = Self::extract_archive_bytes(&raw_bytes)?;
        let archive_fs = Arc::new(ArchiveFileSystem::from_bytes(archive_bytes)?);

        // Configure VFS with root archive and all declared custom vfs_mounts (`PRJ-03`)
        let mut vfs = Vfs::new();
        vfs.mount(
            "",
            Box::new(SubArchiveFileSystem::new(Arc::clone(&archive_fs), "")),
        );
        for mount in &output.manifest.subsystems.vfs_mounts {
            vfs.mount(
                &mount.virtual_mount,
                Box::new(SubArchiveFileSystem::new(
                    Arc::clone(&archive_fs),
                    &mount.physical_path,
                )),
            );
        }

        // Verify start scene exists in VFS and read + parse it (`PRJ-03`)
        let start_scene = &output.manifest.entry_points.start_scene;
        if !vfs.exists(start_scene) {
            return Err(ProjectError::Validation(format!(
                "Exported package is missing declared start scene: {start_scene}"
            )));
        }
        let scene_bytes = vfs.read(start_scene)?;
        let parsed_scene = if start_scene.ends_with(".bin") || start_scene.ends_with(".cbor") {
            Scene::from_binary(&scene_bytes).map_err(|e| {
                ProjectError::Parse(format!(
                    "Failed to decode binary start_scene '{start_scene}': {e}"
                ))
            })?
        } else {
            let scene_str = std::str::from_utf8(&scene_bytes).map_err(|e| {
                ProjectError::Parse(format!(
                    "Start scene '{start_scene}' is not valid UTF-8 JSON: {e}"
                ))
            })?;
            Scene::from_json(scene_str).map_err(|e| {
                ProjectError::Parse(format!(
                    "Failed to parse start_scene '{start_scene}' JSON: {e}"
                ))
            })?
        };

        // Initialize EngineApp with exported configuration
        let engine_config = EngineConfig {
            app_name: output.manifest.metadata.name.clone(),
            tick_rate: output.manifest.subsystems.fixed_tick_rate,
            max_sub_steps: output.manifest.subsystems.max_sub_steps,
            backend: output.manifest.subsystems.default_backend,
            window_size: korppi_math::Vec2::new(
                output.manifest.metadata.default_resolution[0] as f32,
                output.manifest.metadata.default_resolution[1] as f32,
            ),
            ..Default::default()
        };

        let mut app = EngineApp::with_config(engine_config)?;
        app.physics_3d.set_gravity(korppi_math::Vec3::from_array(
            output.manifest.subsystems.physics_gravity_3d,
        ));
        app.physics_2d.set_gravity(korppi_math::Vec2::from_array(
            output.manifest.subsystems.physics_gravity_2d,
        ));
        if let Some(settings) = app.world.resource_mut::<korppi_audio::AudioSettings>() {
            settings.master_volume = output.manifest.subsystems.audio_master_volume;
        }

        // Register all standard engine asset loaders (`PRJ-03`)
        let assets = AssetServer::new(vfs);
        assets.register_loader(TextureLoader);
        assets.register_loader(MeshLoader);
        assets.register_loader(ShaderLoader);
        assets.register_loader(AudioClipLoader);
        assets.register_loader(OggVorbisLoader);
        assets.register_loader(WavAudioLoader);
        app.world.insert_resource(assets);

        // Instantiate start_scene entities into app.world (`PRJ-03`)
        parsed_scene
            .spawn(&mut app.world, registry)
            .map_err(|e| ProjectError::Validation(format!("Failed to spawn start_scene: {e}")))?;

        Ok(app)
    }

    /// Verifies that an exported bundle can be loaded, read, and headlessly simulated.
    pub fn verify_exported_bundle(output: &ExportOutput) -> Result<(), ProjectError> {
        let mut app = Self::load_exported_app(output)?;

        // Step simulation headlessly for 5 frames
        let dt = app.clock.timestep();
        for _ in 0..5 {
            app.update(dt);
        }

        if app.metrics.total_fixed_ticks < 5 {
            return Err(ProjectError::Validation(
                "Exported app failed to simulate fixed ticks".to_string(),
            ));
        }

        Ok(())
    }
}
