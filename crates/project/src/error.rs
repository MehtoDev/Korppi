//! Comprehensive error types for the project system and exporter.

use korppi_asset::VfsError;
use korppi_rhi::RhiError;
use std::fmt;

/// Errors occurring during project operations and export pipelines.
#[derive(Debug)]
pub enum ProjectError {
    /// Underlying filesystem IO error.
    Io(std::io::Error),
    /// Manifest parsing or serialization error.
    Parse(String),
    /// Project structure or configuration validation failure.
    Validation(String),
    /// Virtual File System error.
    Vfs(VfsError),
    /// Render Hardware Interface error during export verification.
    Rhi(RhiError),
    /// Asset packing or bundling failure.
    Packing(String),
    /// Export pipeline failure.
    Export(String),
}

impl fmt::Display for ProjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "IO error: {err}"),
            Self::Parse(msg) => write!(f, "Manifest parse error: {msg}"),
            Self::Validation(msg) => write!(f, "Project validation error: {msg}"),
            Self::Vfs(err) => write!(f, "VFS error: {err}"),
            Self::Rhi(err) => write!(f, "RHI error: {err}"),
            Self::Packing(msg) => write!(f, "Asset packing error: {msg}"),
            Self::Export(msg) => write!(f, "Export error: {msg}"),
        }
    }
}

impl std::error::Error for ProjectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Vfs(err) => Some(err),
            Self::Rhi(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ProjectError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<VfsError> for ProjectError {
    fn from(err: VfsError) -> Self {
        Self::Vfs(err)
    }
}

impl From<RhiError> for ProjectError {
    fn from(err: RhiError) -> Self {
        Self::Rhi(err)
    }
}
