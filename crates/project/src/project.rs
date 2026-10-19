//! Canonical Korppi Project manager and VFS mount coordinator.

use crate::error::ProjectError;
use crate::manifest::ProjectManifest;
use korppi_asset::{PhysicalFileSystem, Vfs};
use std::fs;
use std::path::{Path, PathBuf};

/// Name of the standard project manifest file.
pub const MANIFEST_FILENAME: &str = "korppi.project.toml";

/// Represents an active, on-disk Korppi game engine project.
#[derive(Debug, Clone)]
pub struct Project {
    /// Physical root directory of the project.
    pub root_dir: PathBuf,
    /// Loaded project manifest configuration.
    pub manifest: ProjectManifest,
}

impl Project {
    /// Creates a new canonical project directory layout and initial manifest on disk.
    pub fn create_empty(root: &Path, name: &str, author: &str) -> Result<Self, ProjectError> {
        fs::create_dir_all(root)?;

        // Canonical project directories
        fs::create_dir_all(root.join("assets"))?;
        fs::create_dir_all(root.join("scenes"))?;
        fs::create_dir_all(root.join("config"))?;
        fs::create_dir_all(root.join("build"))?;

        // Initial manifest
        let manifest = ProjectManifest::new(name, author);
        let manifest_content = manifest.to_toml()?;
        fs::write(root.join(MANIFEST_FILENAME), manifest_content)?;

        // Starter scene file if start_scene doesn't exist
        let start_scene_path = root.join(&manifest.entry_points.start_scene);
        if let Some(parent) = start_scene_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if !start_scene_path.exists() {
            fs::write(&start_scene_path, b"{\n  \"entities\": []\n}\n")?;
        }

        Ok(Self {
            root_dir: root.to_path_buf(),
            manifest,
        })
    }

    /// Loads an existing Korppi project from its root directory.
    pub fn load(root: &Path) -> Result<Self, ProjectError> {
        let manifest_path = root.join(MANIFEST_FILENAME);
        if !manifest_path.is_file() {
            return Err(ProjectError::Validation(format!(
                "Manifest file not found at: {}",
                manifest_path.display()
            )));
        }

        let content = fs::read_to_string(&manifest_path)?;
        let manifest = ProjectManifest::from_toml(&content)?;

        Ok(Self {
            root_dir: root.to_path_buf(),
            manifest,
        })
    }

    /// Alias for [`Self::load`].
    pub fn open(root: &Path) -> Result<Self, ProjectError> {
        Self::load(root)
    }

    /// Validates the project directory structure, manifest settings, and critical files (`PRJ-04`).
    pub fn validate(&self) -> Result<(), ProjectError> {
        if !self.root_dir.is_dir() {
            return Err(ProjectError::Validation(format!(
                "Project root directory does not exist: {}",
                self.root_dir.display()
            )));
        }
        let canonical_root = fs::canonicalize(&self.root_dir)?;
        self.manifest.validate()?;

        let validate_on_disk_path =
            |rel: &str, must_exist_file: bool, label: &str| -> Result<(), ProjectError> {
                let candidate = self.root_dir.join(rel);
                match fs::symlink_metadata(&candidate) {
                    Ok(meta) => {
                        if meta.file_type().is_symlink() {
                            return Err(ProjectError::Validation(format!(
                                "Declared {label} must not be a symlink: {}",
                                candidate.display()
                            )));
                        }
                        if must_exist_file && !meta.is_file() {
                            return Err(ProjectError::Validation(format!(
                                "Declared {label} is not a regular file: {}",
                                candidate.display()
                            )));
                        }
                        let canon = fs::canonicalize(&candidate)?;
                        if !canon.starts_with(&canonical_root) {
                            return Err(ProjectError::Validation(format!(
                                "Declared {label} escapes project root: {}",
                                candidate.display()
                            )));
                        }
                        Ok(())
                    }
                    Err(_) if must_exist_file => Err(ProjectError::Validation(format!(
                        "Declared {label} not found: {}",
                        candidate.display()
                    ))),
                    Err(_) => Ok(()),
                }
            };

        validate_on_disk_path(&self.manifest.entry_points.start_scene, true, "start scene")?;
        if let Some(ref script) = self.manifest.entry_points.main_script {
            validate_on_disk_path(script, false, "main_script")?;
        }
        if let Some(ref splash) = self.manifest.entry_points.splash_screen {
            validate_on_disk_path(splash, false, "splash_screen")?;
        }
        for mount in &self.manifest.subsystems.vfs_mounts {
            validate_on_disk_path(&mount.physical_path, false, "vfs_mount.physical_path")?;
        }

        Ok(())
    }

    /// Binds project directories and custom mounts into the engine Virtual File System (`Vfs`).
    pub fn bind_vfs(&self, vfs: &mut Vfs) -> Result<(), ProjectError> {
        let assets_dir = self.root_dir.join("assets");
        if assets_dir.exists() {
            vfs.mount("assets", Box::new(PhysicalFileSystem::new(assets_dir)?));
        }

        let scenes_dir = self.root_dir.join("scenes");
        if scenes_dir.exists() {
            vfs.mount("scenes", Box::new(PhysicalFileSystem::new(scenes_dir)?));
        }

        let config_dir = self.root_dir.join("config");
        if config_dir.exists() {
            vfs.mount("config", Box::new(PhysicalFileSystem::new(config_dir)?));
        }

        // Custom mounts from manifest
        for mount in &self.manifest.subsystems.vfs_mounts {
            let physical = self.root_dir.join(&mount.physical_path);
            if physical.exists() {
                vfs.mount(
                    &mount.virtual_mount,
                    Box::new(PhysicalFileSystem::new(physical)?),
                );
            }
        }

        Ok(())
    }

    /// Returns the absolute path to the project manifest file.
    #[must_use]
    pub fn manifest_path(&self) -> PathBuf {
        self.root_dir.join(MANIFEST_FILENAME)
    }
}
