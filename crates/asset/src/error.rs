//! Asset errors.

use crate::vfs::VfsError;
use std::fmt;

/// Errors returned by asset loading, decoding, and caching operations.
#[derive(Debug)]
pub enum AssetError {
    /// Virtual File System failure.
    Vfs(VfsError),

    /// Decoding failure during asset parsing.
    Decode(String),

    /// Asset was not found or unmanaged.
    NotFound(String),

    /// Unknown or unsupported file extension.
    UnsupportedFormat(String),

    /// Missing loader for asset type.
    MissingLoader(&'static str),
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vfs(err) => write!(f, "VFS error: {err}"),
            Self::Decode(msg) => write!(f, "Asset decode error: {msg}"),
            Self::NotFound(msg) => write!(f, "Asset not found: {msg}"),
            Self::UnsupportedFormat(msg) => write!(f, "Unsupported asset format: {msg}"),
            Self::MissingLoader(ty) => write!(f, "No loader registered for type: {ty}"),
        }
    }
}

impl std::error::Error for AssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vfs(err) => Some(err),
            _ => None,
        }
    }
}

impl From<VfsError> for AssetError {
    fn from(err: VfsError) -> Self {
        Self::Vfs(err)
    }
}
