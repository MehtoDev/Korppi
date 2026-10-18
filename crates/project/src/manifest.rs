//! Declarative project manifest format (`korppi.project.toml`).

use crate::error::ProjectError;
use korppi_engine::GraphicsBackend;
use serde::{Deserialize, Serialize};

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
        Self {
            name: "Korppi Game Project".to_string(),
            version: "0.1.0".to_string(),
            uuid: "00000000-0000-0000-0000-000000000000".to_string(),
            author: "Anonymous".to_string(),
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
    /// Creates a new manifest with reasonable defaults for the given name and author.
    #[must_use]
    pub fn new(name: &str, author: &str) -> Self {
        let mut manifest = Self::default();
        manifest.metadata.name = name.to_string();
        manifest.metadata.author = author.to_string();
        manifest.metadata.window_title = name.to_string();
        manifest
    }

    /// Serializes this manifest to a standard TOML string.
    pub fn to_toml(&self) -> Result<String, ProjectError> {
        toml::to_string_pretty(self).map_err(|e| ProjectError::Parse(e.to_string()))
    }

    /// Deserializes a manifest from a TOML string.
    pub fn from_toml(content: &str) -> Result<Self, ProjectError> {
        toml::from_str(content).map_err(|e| ProjectError::Parse(e.to_string()))
    }
}
