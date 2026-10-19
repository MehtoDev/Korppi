//! Declarative project manifest format (`korppi.project.toml`).

use crate::error::ProjectError;
use korppi_engine::GraphicsBackend;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Component, Path};

/// Nil UUID constant rejected by manifest validation (`PRJ-04`).
pub const NIL_UUID: &str = "00000000-0000-0000-0000-000000000000";

/// Generates a deterministic, non-nil RFC 4122 v5-formatted UUID string from `(name, author)`.
#[must_use]
pub fn generate_project_uuid(name: &str, author: &str) -> String {
    fn fnv1a_64(seed: u64, bytes: &[u8]) -> u64 {
        let mut h = seed;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        h
    }

    let mut hi = fnv1a_64(0xcbf2_9ce4_8422_2325, name.as_bytes());
    hi = fnv1a_64(hi ^ 0xFF, author.as_bytes());
    let mut lo = fnv1a_64(0x8422_2325_cbf2_9ce4, author.as_bytes());
    lo = fnv1a_64(lo ^ 0xA5, name.as_bytes());

    // Set RFC 4122 version 5 (0101) and variant 1 (10xx) bits so it is never nil.
    hi = (hi & !0x0000_0000_0000_F000) | 0x0000_0000_0000_5000;
    lo = (lo & !0xC000_0000_0000_0000) | 0x8000_0000_0000_0000;

    let d1 = (hi >> 32) as u32;
    let d2 = ((hi >> 16) & 0xFFFF) as u16;
    let d3 = (hi & 0xFFFF) as u16;
    let d4 = ((lo >> 48) & 0xFFFF) as u16;
    let d5 = lo & 0x0000_FFFF_FFFF_FFFF;

    format!("{d1:08x}-{d2:04x}-{d3:04x}-{d4:04x}-{d5:012x}")
}

/// Validates that a manifest path string is relative and does not escape the project root.
pub(crate) fn validate_relative_manifest_path(
    path_str: &str,
    field_name: &str,
) -> Result<(), ProjectError> {
    let trimmed = path_str.trim();
    if trimmed.is_empty() {
        return Err(ProjectError::Validation(format!(
            "Manifest field '{field_name}' must not be empty"
        )));
    }
    if trimmed.starts_with('/') || trimmed.starts_with('\\') || trimmed.contains('\0') {
        return Err(ProjectError::Validation(format!(
            "Manifest field '{field_name}' must be a relative path inside the project root: '{path_str}'"
        )));
    }
    // Check both forward-slash and backslash segments for '..' or Windows drive prefixes
    for seg in trimmed.split(['/', '\\']) {
        if seg == ".." || seg.contains(':') {
            return Err(ProjectError::Validation(format!(
                "Manifest field '{field_name}' contains forbidden traversal or prefix segment: '{path_str}'"
            )));
        }
    }
    if Path::new(trimmed).components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(ProjectError::Validation(format!(
            "Manifest field '{field_name}' escapes project sandbox: '{path_str}'"
        )));
    }
    Ok(())
}

/// Project manifest metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetadataConfig {
    /// Human-readable project name.
    pub name: String,
    /// Semantic project version string (e.g. "0.1.0").
    pub version: String,
    /// Unique project identifier UUID.
    pub uuid: String,
    /// Author or organization name.
    pub author: String,
    /// Target Korppi engine version.
    pub engine_version: String,
    /// Window title displayed by the OS window manager.
    pub window_title: String,
    /// Default window resolution `[width, height]`.
    pub default_resolution: [u32; 2],
}

impl Default for MetadataConfig {
    fn default() -> Self {
        let name = "Korppi Game Project";
        let author = "Anonymous";
        Self {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            uuid: generate_project_uuid(name, author),
            author: author.to_string(),
            engine_version: "0.1.0".to_string(),
            window_title: "Korppi Game".to_string(),
            default_resolution: [1280, 720],
        }
    }
}

/// Project entry points configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntryPointsConfig {
    /// Relative virtual or physical path to the initial scene.
    pub start_scene: String,
    /// Optional main gameplay script or assembly path.
    pub main_script: Option<String>,
    /// Optional splash screen texture path.
    pub splash_screen: Option<String>,
}

impl Default for EntryPointsConfig {
    fn default() -> Self {
        Self {
            start_scene: "scenes/main.scene.json".to_string(),
            main_script: None,
            splash_screen: None,
        }
    }
}

/// Custom virtual filesystem mount mapping.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VfsMountConfig {
    /// Virtual prefix path (e.g. "mods/").
    pub virtual_mount: String,
    /// Physical directory relative to project root.
    pub physical_path: String,
}

/// Subsystem defaults and physics/audio/rendering settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubsystemsConfig {
    /// Preferred hardware RHI backend.
    pub default_backend: GraphicsBackend,
    /// 3D gravity acceleration vector `[x, y, z]`.
    pub physics_gravity_3d: [f32; 3],
    /// 2D gravity acceleration vector `[x, y]`.
    pub physics_gravity_2d: [f32; 2],
    /// Simulation fixed update rate in Hertz.
    pub fixed_tick_rate: u32,
    /// Maximum fixed sub-steps per frame clamp.
    pub max_sub_steps: u32,
    /// Global audio master volume gain `(0.0 to 1.0+)`.
    pub audio_master_volume: f32,
    /// Custom virtual filesystem mount routing.
    pub vfs_mounts: Vec<VfsMountConfig>,
}

impl Default for SubsystemsConfig {
    fn default() -> Self {
        Self {
            default_backend: GraphicsBackend::Headless,
            physics_gravity_3d: [0.0, -9.81, 0.0],
            physics_gravity_2d: [0.0, -9.81],
            fixed_tick_rate: 60,
            max_sub_steps: 8,
            audio_master_volume: 1.0,
            vfs_mounts: Vec::new(),
        }
    }
}

/// Declarative project manifest describing metadata, entry points, and subsystem settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProjectManifest {
    /// Metadata section.
    pub metadata: MetadataConfig,
    /// Entry points section.
    pub entry_points: EntryPointsConfig,
    /// Subsystem configuration section.
    pub subsystems: SubsystemsConfig,
}

impl ProjectManifest {
    /// Creates a new manifest with reasonable defaults and a deterministic non-nil UUID for the given name and author.
    #[must_use]
    pub fn new(name: &str, author: &str) -> Self {
        let mut manifest = Self::default();
        manifest.metadata.name = name.to_string();
        manifest.metadata.author = author.to_string();
        manifest.metadata.window_title = name.to_string();
        manifest.metadata.uuid = generate_project_uuid(name, author);
        manifest
    }

    /// Validates all manifest metadata, entry point paths, and subsystem settings (`PRJ-04`).
    pub fn validate(&self) -> Result<(), ProjectError> {
        if self.metadata.name.trim().is_empty() {
            return Err(ProjectError::Validation(
                "Project metadata.name must not be empty".to_string(),
            ));
        }
        let uuid_trimmed = self.metadata.uuid.trim();
        if uuid_trimmed.is_empty() || uuid_trimmed == NIL_UUID {
            return Err(ProjectError::Validation(
                "Project metadata.uuid must be a non-empty, non-nil UUID".to_string(),
            ));
        }
        if self.metadata.default_resolution[0] == 0 || self.metadata.default_resolution[1] == 0 {
            return Err(ProjectError::Validation(format!(
                "Project metadata.default_resolution must have non-zero width and height, got {:?}",
                self.metadata.default_resolution
            )));
        }

        validate_relative_manifest_path(
            &self.entry_points.start_scene,
            "entry_points.start_scene",
        )?;
        if let Some(ref script) = self.entry_points.main_script {
            validate_relative_manifest_path(script, "entry_points.main_script")?;
        }
        if let Some(ref splash) = self.entry_points.splash_screen {
            validate_relative_manifest_path(splash, "entry_points.splash_screen")?;
        }

        if self.subsystems.fixed_tick_rate == 0 || self.subsystems.fixed_tick_rate > 1_000_000_000 {
            return Err(ProjectError::Validation(format!(
                "Project subsystems.fixed_tick_rate must be in 1..=1_000_000_000, got {}",
                self.subsystems.fixed_tick_rate
            )));
        }
        if self.subsystems.max_sub_steps == 0 {
            return Err(ProjectError::Validation(
                "Project subsystems.max_sub_steps must be greater than zero".to_string(),
            ));
        }
        if !self
            .subsystems
            .physics_gravity_3d
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(ProjectError::Validation(format!(
                "Project subsystems.physics_gravity_3d must be finite, got {:?}",
                self.subsystems.physics_gravity_3d
            )));
        }
        if !self
            .subsystems
            .physics_gravity_2d
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(ProjectError::Validation(format!(
                "Project subsystems.physics_gravity_2d must be finite, got {:?}",
                self.subsystems.physics_gravity_2d
            )));
        }
        if !self.subsystems.audio_master_volume.is_finite()
            || self.subsystems.audio_master_volume < 0.0
        {
            return Err(ProjectError::Validation(format!(
                "Project subsystems.audio_master_volume must be finite and non-negative, got {}",
                self.subsystems.audio_master_volume
            )));
        }

        for mount in &self.subsystems.vfs_mounts {
            validate_relative_manifest_path(
                &mount.virtual_mount,
                "subsystems.vfs_mounts.virtual_mount",
            )?;
            validate_relative_manifest_path(
                &mount.physical_path,
                "subsystems.vfs_mounts.physical_path",
            )?;
        }

        Ok(())
    }

    /// Serializes this manifest to a standard TOML string.
    pub fn to_toml(&self) -> Result<String, ProjectError> {
        toml::to_string_pretty(self).map_err(|e| ProjectError::Parse(e.to_string()))
    }

    /// Deserializes a manifest from a TOML string.
    pub fn from_toml(content: &str) -> Result<Self, ProjectError> {
        toml::from_str(content).map_err(|e| ProjectError::Parse(e.to_string()))
    }

    /// Alias for [`Self::from_toml`].
    pub fn from_toml_str(content: &str) -> Result<Self, ProjectError> {
        Self::from_toml(content)
    }

    /// Loads and parses a [`ProjectManifest`] from a file on disk.
    pub fn load(path: &Path) -> Result<Self, ProjectError> {
        let content = fs::read_to_string(path)?;
        Self::from_toml(&content)
    }

    /// Alias for [`Self::load`].
    pub fn load_from_file(path: &Path) -> Result<Self, ProjectError> {
        Self::load(path)
    }
}
