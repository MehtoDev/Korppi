//! Test suite for AssetServer, Generational Handles, and Caching.

use korppi_asset::error::AssetError;
use korppi_asset::handle::{Asset, AssetHandle, AssetLoadState};
use korppi_asset::loader::AssetLoader;
use korppi_asset::server::AssetServer;
use korppi_asset::vfs::{MemoryFileSystem, Vfs};
use std::thread;
use std::time::Duration;

#[derive(Debug, PartialEq, Eq)]
struct TextAsset {
    content: String,
}

impl Asset for TextAsset {
    const TYPE_NAME: &'static str = "TextAsset";
}

struct TextLoader;

impl AssetLoader for TextLoader {
    type Asset = TextAsset;

    fn extensions(&self) -> &[&str] {
        &["txt", "md"]
    }

    fn load(&self, bytes: &[u8], _path: &str) -> Result<Self::Asset, AssetError> {
        let text =
            String::from_utf8(bytes.to_vec()).map_err(|e| AssetError::Decode(e.to_string()))?;
        Ok(TextAsset { content: text })
    }
}

#[test]
fn test_sync_asset_loading_and_caching() {
    let mut vfs = Vfs::new();
    let mut mem = MemoryFileSystem::new();
    mem.insert("greeting.txt", b"Hello, Engine!".to_vec());
    vfs.mount("", Box::new(mem));

    let server = AssetServer::new(vfs);
    server.register_loader(TextLoader);

    // Initial synchronous load
    let handle: AssetHandle<TextAsset> = server.load_sync::<TextLoader>("greeting.txt").unwrap();
    assert_eq!(server.load_state(&handle), AssetLoadState::Loaded);

    let asset = server.get(&handle).unwrap();
    assert_eq!(asset.content, "Hello, Engine!");

    // Secondary load returns the exact same cached handle
    let handle2: AssetHandle<TextAsset> = server.load_sync::<TextLoader>("greeting.txt").unwrap();
    assert_eq!(handle, handle2);

    // Non-existent asset error
    let err = server.load_sync::<TextLoader>("missing.txt");
    assert!(err.is_err());
}

#[test]
fn test_async_asset_loading() {
    let mut vfs = Vfs::new();
    let mut mem = MemoryFileSystem::new();
    mem.insert("async_test.txt", b"Asynchronous Data".to_vec());
    vfs.mount("", Box::new(mem));

    let server = AssetServer::new(vfs);
    server.register_loader(TextLoader);

    // Load async returns handle immediately
    let handle: AssetHandle<TextAsset> = server.load_async::<TextLoader>("async_test.txt");

    // Wait for worker thread to complete load
    let mut retries = 0;
    while server.load_state(&handle) == AssetLoadState::Loading && retries < 100 {
        thread::sleep(Duration::from_millis(10));
        retries += 1;
    }

    assert_eq!(server.load_state(&handle), AssetLoadState::Loaded);
    let asset = server.get(&handle).unwrap();
    assert_eq!(asset.content, "Asynchronous Data");
}

#[test]
fn test_in_place_asset_reloading() {
    let mut vfs = Vfs::new();
    let mut mem = MemoryFileSystem::new();
    mem.insert("mutable.txt", b"Version 1".to_vec());
    vfs.mount("", Box::new(mem.clone()));

    let server = AssetServer::new(vfs);
    server.register_loader(TextLoader);

    let handle: AssetHandle<TextAsset> = server.load_sync::<TextLoader>("mutable.txt").unwrap();
    assert_eq!(server.get(&handle).unwrap().content, "Version 1");

    // Mutate file in VFS
    mem.insert("mutable.txt", b"Version 2".to_vec());

    // Trigger reload
    server.reload::<TextLoader>("mutable.txt").unwrap();

    // The SAME handle now observes Version 2!
    assert_eq!(server.get(&handle).unwrap().content, "Version 2");
}
