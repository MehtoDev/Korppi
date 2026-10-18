//! Test suite for FileWatcher and Live Asset Hot-Reloading.

use korppi_asset::loader::AssetLoader;
use korppi_asset::server::AssetServer;
use korppi_asset::vfs::{PhysicalFileSystem, Vfs};
use korppi_asset::watcher::{AssetEvent, FileWatcher};
use korppi_asset::{Asset, AssetError, AssetHandle};
use std::fs;
use std::thread;
use std::time::Duration;

#[derive(Debug, PartialEq, Eq)]
struct ConfigAsset {
    data: String,
}

impl Asset for ConfigAsset {
    const TYPE_NAME: &'static str = "Config";
}

struct ConfigLoader;

impl AssetLoader for ConfigLoader {
    type Asset = ConfigAsset;

    fn extensions(&self) -> &[&str] {
        &["cfg"]
    }

    fn load(&self, bytes: &[u8], _path: &str) -> Result<Self::Asset, AssetError> {
        let s = String::from_utf8(bytes.to_vec()).map_err(|e| AssetError::Decode(e.to_string()))?;
        Ok(ConfigAsset { data: s })
    }
}

#[test]
fn test_file_watcher_detects_modification() {
    let temp_dir = std::env::temp_dir().join("korppi_asset_hot_reload_test");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let file_path = temp_dir.join("test.cfg");
    fs::write(&file_path, "initial_value").unwrap();

    let mut watcher = FileWatcher::new(Duration::from_millis(20));
    watcher.watch_directory(&temp_dir).unwrap();

    // No events initially
    assert!(watcher.poll_events().is_empty());

    // Sleep briefly to ensure mtime changes
    thread::sleep(Duration::from_millis(50));

    // Modify file
    fs::write(&file_path, "modified_value").unwrap();

    // Allow watcher polling tick to pick up changes
    thread::sleep(Duration::from_millis(60));

    let events = watcher.poll_events();
    assert!(!events.is_empty());
    assert!(events
        .iter()
        .any(|e| matches!(e, AssetEvent::Modified(p) if p.ends_with("test.cfg"))));

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_asset_server_automated_hot_reload() {
    let temp_dir = std::env::temp_dir().join("korppi_asset_server_reload_test");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let file_path = temp_dir.join("game.cfg");
    fs::write(&file_path, "difficulty = normal").unwrap();

    let mut vfs = Vfs::new();
    vfs.mount("", Box::new(PhysicalFileSystem::new(&temp_dir).unwrap()));

    let server = AssetServer::new(vfs);
    server.register_loader(ConfigLoader);

    let handle: AssetHandle<ConfigAsset> = server.load_sync::<ConfigLoader>("game.cfg").unwrap();
    assert_eq!(server.get(&handle).unwrap().data, "difficulty = normal");

    // Modify the file on disk
    thread::sleep(Duration::from_millis(50));
    fs::write(&file_path, "difficulty = hard").unwrap();

    // Trigger reload on server
    server.reload::<ConfigLoader>("game.cfg").unwrap();

    // Handle immediately observes the hard difficulty
    assert_eq!(server.get(&handle).unwrap().data, "difficulty = hard");

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
}
