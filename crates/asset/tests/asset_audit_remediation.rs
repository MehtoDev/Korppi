//! Regression tests for `FIX-13.5` (`AST-01`–`AST-05` from the `FIX-12.5` asset audit).

use korppi_asset::{
    ArchiveFileSystem, Asset, AssetError, AssetEvent, AssetLoadState, AssetLoader, AssetServer,
    FileSystem, FileWatcher, MemoryFileSystem, MeshLoader, PhysicalFileSystem, ShaderAsset,
    ShaderLoader, Texture, TextureFormat, TextureLoader, Vfs, VfsError,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "korppi_asset_audit_{label}_{}_{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[derive(Debug, PartialEq)]
struct DummyText(String);
impl Asset for DummyText {
    const TYPE_NAME: &'static str = "DummyText";
}

struct DummyLoader;
impl AssetLoader for DummyLoader {
    type Asset = DummyText;

    fn load(&self, bytes: &[u8], _path: &str) -> Result<Self::Asset, AssetError> {
        let s = std::str::from_utf8(bytes).map_err(|e| AssetError::Decode(e.to_string()))?;
        Ok(DummyText(s.to_owned()))
    }

    fn extensions(&self) -> &[&str] {
        &["txt"]
    }
}

struct AltDummyLoader;
impl AssetLoader for AltDummyLoader {
    type Asset = DummyText;

    fn load(&self, bytes: &[u8], _path: &str) -> Result<Self::Asset, AssetError> {
        let s = std::str::from_utf8(bytes).map_err(|e| AssetError::Decode(e.to_string()))?;
        Ok(DummyText(format!("ALT:{s}")))
    }

    fn extensions(&self) -> &[&str] {
        &["alt"]
    }
}

/// AST-01 (V1): `AssetServer` normalizes virtual paths across `load_sync`, `load_async`,
/// and `reload`, recovers from failed loads on subsequent `load_sync`/`load_async`,
/// supports multiple loaders for the same `Asset` type, and validates loader extensions.
#[test]
fn ast_01_server_deduplication_recovery_and_multi_loader() {
    let mut mem = MemoryFileSystem::new();
    mem.insert("hello.txt", b"world".to_vec());
    mem.insert("hello.alt", b"world".to_vec());
    mem.insert("hello.wrong", b"world".to_vec());

    let mut vfs = Vfs::new();
    vfs.mount("/", Box::new(mem));

    let server = Arc::new(AssetServer::new(vfs));
    server.register_loader(DummyLoader);
    server.register_loader(AltDummyLoader);

    // 1. Canonical path normalization across load_sync
    let h1 = server.load_sync::<DummyLoader>("hello.txt").unwrap();
    let h2 = server.load_sync::<DummyLoader>("./hello.txt").unwrap();
    let h3 = server.load_sync::<DummyLoader>("/hello.txt").unwrap();
    assert_eq!(h1.id(), h2.id(), "load_sync must normalize './hello.txt'");
    assert_eq!(h1.id(), h3.id(), "load_sync must normalize '/hello.txt'");

    // 2. Multiple loaders targeting the same Asset type must coexist
    let h_alt = server.load_sync::<AltDummyLoader>("hello.alt").unwrap();
    assert_eq!(server.get(&h_alt).unwrap().0, "ALT:world");
    assert_eq!(server.get(&h1).unwrap().0, "world");

    // 3. Extension validation
    let bad_ext = server.load_sync::<DummyLoader>("hello.wrong");
    assert!(
        matches!(bad_ext, Err(AssetError::UnsupportedFormat(_))),
        "expected UnsupportedFormat for .wrong with DummyLoader, got {bad_ext:?}"
    );

    // 4. Failed async load must NOT return Ok(handle with None data) on subsequent load_sync
    let h_missing_async = server.load_async::<DummyLoader>("missing_file.txt");
    for _ in 0..100 {
        if matches!(
            server.load_state(&h_missing_async),
            AssetLoadState::Failed(_)
        ) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(matches!(
        server.load_state(&h_missing_async),
        AssetLoadState::Failed(_)
    ));

    let sync_retry = server.load_sync::<DummyLoader>("missing_file.txt");
    assert!(
        sync_retry.is_err(),
        "load_sync after failed load_async must retry and return Err, not Ok with None data"
    );

    // Invalid path in load_async transitions immediately to Failed
    let h_bad_async = server.load_async::<DummyLoader>("../escape.txt");
    for _ in 0..50 {
        if matches!(server.load_state(&h_bad_async), AssetLoadState::Failed(_)) {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(matches!(
        server.load_state(&h_bad_async),
        AssetLoadState::Failed(_)
    ));

    // 5. Explicit unload evicts cached asset and invalidates handle
    assert!(server.unload(&h_alt));
    assert!(server.get(&h_alt).is_none());
    assert_eq!(server.load_state(&h_alt), AssetLoadState::NotLoaded);
}

/// AST-01 (V1): `AssetServer` releases `loaders` lock before calling `loader.load`,
/// so registering a loader while another thread is decoding does not block.
#[test]
fn ast_01_loader_lock_not_held_during_decode() {
    struct SlowLoader {
        entered: Arc<AtomicBool>,
        proceed: Arc<AtomicBool>,
    }
    impl AssetLoader for SlowLoader {
        type Asset = DummyText;
        fn load(&self, bytes: &[u8], _path: &str) -> Result<Self::Asset, AssetError> {
            self.entered.store(true, Ordering::SeqCst);
            while !self.proceed.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(DummyText(String::from_utf8_lossy(bytes).into_owned()))
        }
        fn extensions(&self) -> &[&str] {
            &["slow"]
        }
    }

    let mut mem = MemoryFileSystem::new();
    mem.insert("slow.slow", b"data".to_vec());
    let mut vfs = Vfs::new();
    vfs.mount("/", Box::new(mem));
    let server = Arc::new(AssetServer::new(vfs));

    let entered = Arc::new(AtomicBool::new(false));
    let proceed = Arc::new(AtomicBool::new(false));
    server.register_loader(SlowLoader {
        entered: entered.clone(),
        proceed: proceed.clone(),
    });

    let server_bg = server.clone();
    let loader_thread =
        std::thread::spawn(move || server_bg.load_sync::<SlowLoader>("slow.slow").unwrap());

    while !entered.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(1));
    }

    let registered = Arc::new(AtomicBool::new(false));
    let reg_flag = registered.clone();
    let server_reg = server.clone();
    let reg_thread = std::thread::spawn(move || {
        server_reg.register_loader(DummyLoader);
        reg_flag.store(true, Ordering::SeqCst);
    });

    std::thread::sleep(Duration::from_millis(20));
    let finished_while_decoding = registered.load(Ordering::SeqCst);
    proceed.store(true, Ordering::SeqCst);

    reg_thread.join().unwrap();
    let handle = loader_thread.join().unwrap();
    assert!(
        finished_while_decoding,
        "register_loader was blocked while SlowLoader::load was decoding"
    );
    assert_eq!(server.get(&handle).unwrap().0, "data");
}

/// AST-02 (V2): `PhysicalFileSystem` blocks symlink sandbox escape, `MemoryFileSystem`
/// and `ArchiveFileSystem` `list_dir` return immediate children only (matching `PhysicalFileSystem`),
/// and `Vfs::read` propagates `CorruptArchive` and `SecurityViolation` instead of swallowing them.
#[test]
fn ast_02_vfs_symlink_escape_list_dir_parity_and_error_propagation() {
    let sandbox_dir = unique_temp_dir("sandbox");
    let secret_dir = unique_temp_dir("secret");
    let secret_file = secret_dir.join("secret.txt");
    std::fs::write(&secret_file, b"TOP_SECRET_DATA").unwrap();

    #[cfg(unix)]
    {
        let link_path = sandbox_dir.join("escape_link.txt");
        std::os::unix::fs::symlink(&secret_file, &link_path).unwrap();

        let subdir_link = sandbox_dir.join("escape_dir");
        std::os::unix::fs::symlink(&secret_dir, &subdir_link).unwrap();

        let phys = PhysicalFileSystem::new(&sandbox_dir).unwrap();
        let read_link = phys.read("escape_link.txt");
        let read_subdir = phys.read("escape_dir/secret.txt");
        let exists_link = phys.exists("escape_link.txt");

        let _ = std::fs::remove_file(&link_path);
        let _ = std::fs::remove_file(&subdir_link);

        assert!(
            matches!(read_link, Err(VfsError::SecurityViolation(_))),
            "symlink to file outside root must return SecurityViolation, got {read_link:?}"
        );
        assert!(
            matches!(read_subdir, Err(VfsError::SecurityViolation(_))),
            "symlinked directory outside root must return SecurityViolation, got {read_subdir:?}"
        );
        assert!(
            !exists_link,
            "exists() must return false for symlink escaping sandbox"
        );
    }

    // Directory listing parity across PhysicalFileSystem, MemoryFileSystem, ArchiveFileSystem
    std::fs::create_dir_all(sandbox_dir.join("a/b/c")).unwrap();
    std::fs::write(sandbox_dir.join("a/file1.txt"), b"1").unwrap();
    std::fs::write(sandbox_dir.join("a/b/file2.txt"), b"2").unwrap();
    std::fs::write(sandbox_dir.join("a/b/c/file3.txt"), b"3").unwrap();

    let phys_fs = PhysicalFileSystem::new(&sandbox_dir).unwrap();
    let mut mem_fs = MemoryFileSystem::new();
    mem_fs.insert("a/file1.txt", b"1".to_vec());
    mem_fs.insert("a/b/file2.txt", b"2".to_vec());
    mem_fs.insert("a/b/c/file3.txt", b"3".to_vec());

    let mut archive_builder = ArchiveFileSystem::builder();
    archive_builder.add_file("a/file1.txt", b"1");
    archive_builder.add_file("a/b/file2.txt", b"2");
    archive_builder.add_file("a/b/c/file3.txt", b"3");
    let archive_fs = ArchiveFileSystem::from_bytes(&archive_builder.build()).unwrap();

    let mut phys_list = phys_fs.list_dir("a").unwrap();
    let mut mem_list = mem_fs.list_dir("a").unwrap();
    let mut arch_list = archive_fs.list_dir("a").unwrap();
    phys_list.sort();
    mem_list.sort();
    arch_list.sort();

    let _ = std::fs::remove_dir_all(&sandbox_dir);
    let _ = std::fs::remove_dir_all(&secret_dir);

    let expected = vec!["a/b".to_string(), "a/file1.txt".to_string()];
    assert_eq!(phys_list, expected);
    assert_eq!(
        mem_list, expected,
        "MemoryFileSystem::list_dir must list immediate children only"
    );
    assert_eq!(
        arch_list, expected,
        "ArchiveFileSystem::list_dir must list immediate children only"
    );

    // Vfs::read must propagate CorruptArchive instead of falling back or returning NotFound
    let mut bad_earc = Vec::new();
    bad_earc.extend_from_slice(b"EARC");
    bad_earc.extend_from_slice(&1u32.to_le_bytes());
    bad_earc.extend_from_slice(&1u32.to_le_bytes());
    let p = b"data.bin";
    bad_earc.extend_from_slice(&(p.len() as u16).to_le_bytes());
    bad_earc.extend_from_slice(p);
    bad_earc.extend_from_slice(&9999u64.to_le_bytes());
    bad_earc.extend_from_slice(&100u64.to_le_bytes());

    let corrupt_archive = ArchiveFileSystem::from_bytes(&bad_earc).unwrap();
    let mut vfs = Vfs::new();
    vfs.mount("/", Box::new(corrupt_archive));
    assert!(
        matches!(vfs.read("data.bin"), Err(VfsError::CorruptArchive(_))),
        "Vfs::read must propagate CorruptArchive"
    );
}

/// AST-03 (V3): `ArchiveFileSystem` and `TextureLoader::decode_ktx2` reject adversarial
/// headers (huge entry count, offset + len overflow, level count overflow, 0x0 dimensions,
/// supercompression != 0, unsupported VkFormat, and R16F/Rgba16F/Rgba32F mapping) without
/// panicking or OOMing.
#[test]
fn ast_03_archive_and_ktx2_adversarial_headers() {
    // 1. Archive with count = u32::MAX on tiny buffer must return Err(CorruptArchive) without OOM
    let mut huge_count_earc = Vec::new();
    huge_count_earc.extend_from_slice(b"EARC");
    huge_count_earc.extend_from_slice(&1u32.to_le_bytes());
    huge_count_earc.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        ArchiveFileSystem::from_bytes(&huge_count_earc),
        Err(VfsError::CorruptArchive(_))
    ));

    // 2. Archive entry with offset = u64::MAX, len = 1 must return Err(CorruptArchive), not panic
    let mut overflow_earc = Vec::new();
    overflow_earc.extend_from_slice(b"EARC");
    overflow_earc.extend_from_slice(&1u32.to_le_bytes());
    overflow_earc.extend_from_slice(&1u32.to_le_bytes());
    overflow_earc.extend_from_slice(&1u16.to_le_bytes());
    overflow_earc.extend_from_slice(b"a");
    overflow_earc.extend_from_slice(&u64::MAX.to_le_bytes());
    overflow_earc.extend_from_slice(&1u64.to_le_bytes());
    if let Ok(archive) = ArchiveFileSystem::from_bytes(&overflow_earc) {
        let res = std::panic::catch_unwind(|| archive.read("a"));
        assert!(
            res.is_ok(),
            "ArchiveFileSystem::read panicked on u64::MAX + 1"
        );
        assert!(matches!(res.unwrap(), Err(VfsError::CorruptArchive(_))));
    }

    let loader = TextureLoader;

    // 3. KTX2 with 0x0 dimensions and supercompression = 1 must be rejected
    let mut ktx_zero_super = vec![0u8; 104];
    ktx_zero_super[0..12].copy_from_slice(&[
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ]);
    ktx_zero_super[12..16].copy_from_slice(&37u32.to_le_bytes()); // Rgba8
    ktx_zero_super[20..24].copy_from_slice(&0u32.to_le_bytes()); // width = 0
    ktx_zero_super[24..28].copy_from_slice(&0u32.to_le_bytes()); // height = 0
    ktx_zero_super[40..44].copy_from_slice(&1u32.to_le_bytes()); // level_count = 1
    ktx_zero_super[44..48].copy_from_slice(&1u32.to_le_bytes()); // supercompression = BasisLZ
    assert!(
        loader.load(&ktx_zero_super, "bad.ktx2").is_err(),
        "0x0 supercompressed KTX2 must be rejected"
    );

    // 4. KTX2 with valid dimensions but supercompression = 2 (Zstd) must be rejected
    let mut ktx_super = ktx_zero_super.clone();
    ktx_super[20..24].copy_from_slice(&4u32.to_le_bytes());
    ktx_super[24..28].copy_from_slice(&4u32.to_le_bytes());
    ktx_super[44..48].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(
        loader.load(&ktx_super, "zstd.ktx2"),
        Err(AssetError::UnsupportedFormat(_))
    ));

    // 5. KTX2 with unknown vk_format = 999999 must be rejected
    let mut ktx_bad_fmt = vec![0u8; 104 + 16];
    ktx_bad_fmt[0..12].copy_from_slice(&[
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ]);
    ktx_bad_fmt[12..16].copy_from_slice(&999_999u32.to_le_bytes());
    ktx_bad_fmt[20..24].copy_from_slice(&2u32.to_le_bytes());
    ktx_bad_fmt[24..28].copy_from_slice(&2u32.to_le_bytes());
    ktx_bad_fmt[40..44].copy_from_slice(&1u32.to_le_bytes());
    ktx_bad_fmt[80..88].copy_from_slice(&104u64.to_le_bytes());
    ktx_bad_fmt[88..96].copy_from_slice(&16u64.to_le_bytes());
    assert!(matches!(
        loader.load(&ktx_bad_fmt, "unknown_fmt.ktx2"),
        Err(AssetError::UnsupportedFormat(_))
    ));

    // 6. KTX2 with byte_offset = u64::MAX, byte_length = 1 must return Err, not panic
    let mut ktx_overflow = ktx_bad_fmt.clone();
    ktx_overflow[12..16].copy_from_slice(&37u32.to_le_bytes());
    ktx_overflow[80..88].copy_from_slice(&u64::MAX.to_le_bytes());
    ktx_overflow[88..96].copy_from_slice(&1u64.to_le_bytes());
    let ktx_res = std::panic::catch_unwind(|| loader.load(&ktx_overflow, "overflow.ktx2"));
    assert!(ktx_res.is_ok(), "decode_ktx2 panicked on u64::MAX + 1");
    assert!(ktx_res.unwrap().is_err());

    // 7. KTX2 float formats (R16F = 76, Rgba16F = 97, Rgba32F = 109) must decode properly
    for (vk_fmt, expected_fmt) in [
        (76u32, TextureFormat::R16F),
        (97u32, TextureFormat::Rgba16F),
        (109u32, TextureFormat::Rgba32F),
    ] {
        let mut ktx_float = ktx_bad_fmt.clone();
        ktx_float[12..16].copy_from_slice(&vk_fmt.to_le_bytes());
        let tex: Texture = loader.load(&ktx_float, "float.ktx2").unwrap();
        assert_eq!(tex.format, expected_fmt);
    }
}

/// AST-04 (V4): `MeshLoader` and `ShaderLoader` reject huge unvalidated glTF accessor counts,
/// out-of-bounds mesh vertex indices, `i32::MIN` OBJ indices, empty glTF primitives,
/// unknown file extensions, and invalid SPIR-V magic headers without panicking or OOMing.
#[test]
fn ast_04_mesh_and_shader_validation() {
    let mesh_loader = MeshLoader;

    // 1. OBJ with -2147483648 (i32::MIN) index must return Err(Decode), not panic
    let obj_min_idx = b"v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf -2147483648 1 2\n";
    let obj_res = std::panic::catch_unwind(|| mesh_loader.load(obj_min_idx, "bad.obj"));
    assert!(obj_res.is_ok(), "MeshLoader::load_obj panicked on i32::MIN");
    assert!(obj_res.unwrap().is_err());

    // 2. OBJ with out-of-bounds positive index must return Err(Decode)
    let obj_oob_idx = b"v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 99\n";
    assert!(mesh_loader.load(obj_oob_idx, "oob.obj").is_err());

    // 3. glTF with huge accessor count on tiny buffer must fail bounds check before allocating
    let gltf_huge = br#"{
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "count": 100000, "componentType": 5126, "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12}]
    }"#;
    assert!(mesh_loader.load(gltf_huge, "huge.gltf").is_err());

    // 4. glTF with out-of-bounds vertex index (9999 >= 3 vertices) must be rejected
    let mut bin = Vec::new();
    for _ in 0..9 {
        bin.extend_from_slice(&0.0f32.to_le_bytes());
    }
    for idx in [0u16, 1, 9999] {
        bin.extend_from_slice(&idx.to_le_bytes());
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let json = br#"{
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "count": 3, "componentType": 5126, "type": "VEC3"},
            {"bufferView": 1, "count": 3, "componentType": 5123, "type": "SCALAR"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6}
        ]
    }"#;
    let mut json_padded = json.to_vec();
    while !json_padded.len().is_multiple_of(4) {
        json_padded.push(b' ');
    }
    let total_len = 12 + 8 + json_padded.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total_len);
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());
    glb.extend_from_slice(&(json_padded.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    glb.extend_from_slice(&json_padded);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
    glb.extend_from_slice(&bin);
    assert!(
        mesh_loader.load(&glb, "oob_indices.glb").is_err(),
        "glTF with index >= vertex count must be rejected"
    );

    // 5. Unknown extension on MeshLoader must return UnsupportedFormat
    assert!(matches!(
        mesh_loader.load(b"{}", "not_a_mesh.unknown"),
        Err(AssetError::UnsupportedFormat(_))
    ));

    // 6. ShaderLoader must validate SPIR-V magic & length and reject unknown extensions
    let shader_loader = ShaderLoader;
    assert!(
        shader_loader.load(b"1234", "bad.spv").is_err(),
        "4-byte non-SPIR-V data must be rejected"
    );
    assert!(matches!(
        shader_loader.load(b"void main() {}", "shader.txt"),
        Err(AssetError::UnsupportedFormat(_))
    ));

    // Valid 20-byte minimal SPIR-V header (magic 0x07230203, version, generator, bound, schema)
    let mut valid_spv = Vec::new();
    valid_spv.extend_from_slice(&0x0723_0203u32.to_le_bytes());
    valid_spv.extend_from_slice(&0x0001_0000u32.to_le_bytes());
    valid_spv.extend_from_slice(&0u32.to_le_bytes());
    valid_spv.extend_from_slice(&1u32.to_le_bytes());
    valid_spv.extend_from_slice(&0u32.to_le_bytes());
    let spv_asset: ShaderAsset = shader_loader.load(&valid_spv, "ok.spv").unwrap();
    assert_eq!(spv_asset.spirv.len(), 5);
}

/// AST-05 (V5): `FileWatcher` does not infinite-loop on symlink cycles, enforces its
/// configured poll `interval`, detects rolled-back `mtime` (`mtime != old_mtime`),
/// and emits root-relative virtual paths suitable for `AssetServer::reload`.
#[test]
fn ast_05_file_watcher_symlink_cycle_interval_and_relative_paths() {
    let dir = unique_temp_dir("watcher");
    let file_path = dir.join("nested").join("test.txt");
    std::fs::create_dir_all(file_path.parent().unwrap()).unwrap();
    std::fs::write(&file_path, b"hello").unwrap();

    #[cfg(unix)]
    {
        // Self-referential symlink cycle inside watched directory
        let loop_link = dir.join("nested").join("loop_self");
        std::os::unix::fs::symlink(&dir, &loop_link).unwrap();
    }

    let mut watcher = FileWatcher::new(Duration::from_secs(60));
    watcher
        .watch_directory(&dir)
        .expect("watch_directory must not recurse infinitely on symlink cycles");

    // Roll back mtime to an older timestamp while keeping file length identical (5 bytes)
    let f = std::fs::File::options()
        .write(true)
        .open(&file_path)
        .unwrap();
    let older = UNIX_EPOCH + Duration::from_secs(1_000_000_000);
    f.set_modified(older).unwrap();
    drop(f);

    // First poll_events() executes immediately and detects the rolled-back mtime
    // with a normalized relative virtual path ("nested/test.txt").
    let events = watcher.poll_events();

    // Second poll_events() within the 60s interval must be throttled
    std::fs::write(&file_path, b"hello world").unwrap();
    let throttled = watcher.poll_events();

    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        events,
        vec![AssetEvent::Modified("nested/test.txt".to_string())],
        "FileWatcher must detect older mtime and emit root-relative virtual path"
    );
    assert!(
        throttled.is_empty(),
        "FileWatcher must respect configured poll interval, got {throttled:?}"
    );
}
