//! Asset management, Virtual File System (VFS), format decoders, and hot reloading.

pub mod error;
pub mod handle;
pub mod loader;
pub mod mesh;
pub mod server;
pub mod shader;
pub mod texture;
pub mod vfs;
pub mod watcher;

pub use error::AssetError;
pub use handle::{Asset, AssetHandle, AssetId, AssetLoadState};
pub use loader::AssetLoader;
pub use mesh::{Mesh, MeshLoader, Submesh};
pub use server::AssetServer;
pub use shader::{ShaderAsset, ShaderLoader, ShaderStage};
pub use texture::{Texture, TextureFormat, TextureLoader};
pub use vfs::{ArchiveFileSystem, FileSystem, MemoryFileSystem, PhysicalFileSystem, Vfs, VfsError};
pub use watcher::{AssetEvent, FileWatcher};
