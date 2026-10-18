//! Extensible asset loaders and decoder interfaces.

use crate::error::AssetError;
use crate::handle::Asset;

/// Trait defining a file format decoder for a specific asset type.
pub trait AssetLoader: Send + Sync + 'static {
    /// The target asset type produced by this decoder.
    type Asset: Asset;

    /// File extensions supported by this loader (e.g. `&["png", "jpg"]`).
    fn extensions(&self) -> &[&str];

    /// Decodes raw file bytes into an asset instance.
    fn load(&self, bytes: &[u8], path: &str) -> Result<Self::Asset, AssetError>;
}
