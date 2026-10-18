//! Test suite for the Virtual File System (VFS).

use korppi_asset::vfs::{
    ArchiveFileSystem, FileSystem, MemoryFileSystem, PhysicalFileSystem, Vfs, VfsError,
};
use std::fs;

#[test]
fn test_memory_filesystem_read_write_list() {
    let mut mem_fs = MemoryFileSystem::new();
    mem_fs.insert("hello.txt", b"Hello, World!".to_vec());
    mem_fs.insert("scripts/main.lua", b"print('Hello')".to_vec());
    mem_fs.insert("scripts/util.lua", b"return {}".to_vec());

    assert!(mem_fs.exists("hello.txt"));
    assert!(mem_fs.is_file("hello.txt"));
    assert!(!mem_fs.is_dir("hello.txt"));

    assert_eq!(mem_fs.read_to_string("hello.txt").unwrap(), "Hello, World!");
    assert_eq!(mem_fs.read("scripts/main.lua").unwrap(), b"print('Hello')");

    assert!(mem_fs.exists("scripts"));
    assert!(mem_fs.is_dir("scripts"));

    let scripts = mem_fs.list_dir("scripts").unwrap();
    assert_eq!(scripts.len(), 2);
    assert!(scripts.contains(&"scripts/main.lua".to_string()));
    assert!(scripts.contains(&"scripts/util.lua".to_string()));

    // Non-existent file
    assert_eq!(
        mem_fs.read("missing.txt").unwrap_err(),
        VfsError::NotFound("missing.txt".into())
    );
}

#[test]
fn test_physical_filesystem_sandboxing() {
    let temp_dir = std::env::temp_dir().join("korppi_asset_test_sandbox");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(temp_dir.join("sub")).unwrap();

    fs::write(temp_dir.join("test.txt"), "Physics & Graphics").unwrap();
    fs::write(temp_dir.join("sub/child.txt"), "Nested content").unwrap();

    let phys_fs = PhysicalFileSystem::new(&temp_dir).unwrap();

    assert_eq!(
        phys_fs.read_to_string("test.txt").unwrap(),
        "Physics & Graphics"
    );
    assert_eq!(
        phys_fs.read_to_string("sub/child.txt").unwrap(),
        "Nested content"
    );

    // Attempt traversal attack
    let traversal_err = phys_fs.read("../secret.txt");
    assert!(matches!(
        traversal_err,
        Err(VfsError::SecurityViolation(_)) | Err(VfsError::NotFound(_))
    ));

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_vfs_mount_routing_and_priority() {
    let mut vfs = Vfs::new();

    let mut base_mem = MemoryFileSystem::new();
    base_mem.insert("config.json", b"{\"version\": 1}".to_vec());
    base_mem.insert("textures/grass.png", b"BASE_GRASS".to_vec());

    let mut mod_mem = MemoryFileSystem::new();
    mod_mem.insert("textures/grass.png", b"MODDED_GRASS".to_vec());
    mod_mem.insert("textures/stone.png", b"MODDED_STONE".to_vec());

    // Mount base filesystem at root
    vfs.mount("", Box::new(base_mem));
    // Mount mod filesystem with higher priority at root
    vfs.mount_priority("", Box::new(mod_mem), 10);

    // Modded grass overrides base grass
    assert_eq!(
        vfs.read_to_string("textures/grass.png").unwrap(),
        "MODDED_GRASS"
    );
    // Modded stone exists
    assert_eq!(
        vfs.read_to_string("textures/stone.png").unwrap(),
        "MODDED_STONE"
    );
    // Fallback to base config
    assert_eq!(
        vfs.read_to_string("config.json").unwrap(),
        "{\"version\": 1}"
    );
}

#[test]
fn test_packed_archive_filesystem() {
    let mut archive_builder = ArchiveFileSystem::builder();
    archive_builder.add_file("shaders/pbr.vert", b"void main() {}");
    archive_builder.add_file("shaders/pbr.frag", b"out vec4 fragColor;");
    let archive_bytes = archive_builder.build();

    let archive_fs = ArchiveFileSystem::from_bytes(&archive_bytes).unwrap();

    assert!(archive_fs.exists("shaders/pbr.vert"));
    assert_eq!(
        archive_fs.read_to_string("shaders/pbr.vert").unwrap(),
        "void main() {}"
    );
    assert_eq!(
        archive_fs.read_to_string("shaders/pbr.frag").unwrap(),
        "out vec4 fragColor;"
    );

    let shaders = archive_fs.list_dir("shaders").unwrap();
    assert_eq!(shaders.len(), 2);
    assert!(shaders.contains(&"shaders/pbr.vert".to_string()));
    assert!(shaders.contains(&"shaders/pbr.frag".to_string()));
}
