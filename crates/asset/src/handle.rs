//! Generational asset identifiers, lifecycle states, and handles.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU32;

/// Trait implemented by all loadable and cacheable asset types.
pub trait Asset: Send + Sync + 'static {
    /// Human-readable type identifier for diagnostics and error reporting.
    const TYPE_NAME: &'static str;
}

/// A generational identifier assigned to an asset in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetId {
    index: u32,
    generation: NonZeroU32,
}

impl AssetId {
    /// Creates a new `AssetId`.
    #[must_use]
    pub const fn new(index: u32, generation: NonZeroU32) -> Self {
        Self { index, generation }
    }

    /// Returns the dense cache index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the generation number to prevent ABA stale references.
    #[must_use]
    pub const fn generation(self) -> NonZeroU32 {
        self.generation
    }
}

/// The current loading lifecycle state of an asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetLoadState {
    /// Asset has not been requested or loaded.
    NotLoaded,
    /// Asset is currently loading asynchronously.
    Loading,
    /// Asset is fully decoded and resident in cache.
    Loaded,
    /// Asset loading or decoding failed.
    Failed(String),
}

/// A strongly typed generational handle referencing an asset resident in cache.
pub struct AssetHandle<T: Asset> {
    id: AssetId,
    path: Option<String>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Asset> AssetHandle<T> {
    /// Creates a new `AssetHandle`.
    #[must_use]
    pub const fn new(id: AssetId, path: Option<String>) -> Self {
        Self {
            id,
            path,
            _marker: PhantomData,
        }
    }

    /// Returns the generational ID of the asset.
    #[must_use]
    pub const fn id(&self) -> AssetId {
        self.id
    }

    /// Returns the original VFS path of the asset, if loaded from a file.
    #[must_use]
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }
}

impl<T: Asset> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            path: self.path.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: Asset> PartialEq for AssetHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Asset> Eq for AssetHandle<T> {}

impl<T: Asset> Hash for AssetHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T: Asset> fmt::Debug for AssetHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssetHandle")
            .field("id", &self.id)
            .field("path", &self.path)
            .field("type", &T::TYPE_NAME)
            .finish()
    }
}
