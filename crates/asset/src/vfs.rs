//! Virtual File System (VFS) with mount routing, physical sandboxing, and packed archives.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Errors encountered during Virtual File System operations.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VfsError {
    /// The requested file or directory does not exist.
    NotFound(String),
    /// Attempted path traversal or sandboxing violation.
    SecurityViolation(String),
    /// IO failure when accessing underlying storage.
    Io(String),
    /// Archive metadata or payload was corrupt or invalid.
    CorruptArchive(String),
    /// Invalid path formatting.
    InvalidPath(String),
}

impl fmt::Display for VfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "Path not found: {msg}"),
            Self::SecurityViolation(msg) => {
                write!(f, "Security violation: path escapes sandbox: {msg}")
            }
            Self::Io(msg) => write!(f, "IO error: {msg}"),
            Self::CorruptArchive(msg) => write!(f, "Corrupt archive: {msg}"),
            Self::InvalidPath(msg) => write!(f, "Invalid path: {msg}"),
        }
    }
}

impl std::error::Error for VfsError {}

/// Abstract filesystem trait for reading files and querying directories.
pub trait FileSystem: Send + Sync {
    /// Reads the entire contents of a file into a byte vector.
    fn read(&self, path: &str) -> Result<Vec<u8>, VfsError>;

    /// Reads the entire contents of a file as a UTF-8 string.
    fn read_to_string(&self, path: &str) -> Result<String, VfsError> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes).map_err(|e| VfsError::Io(e.to_string()))
    }

    /// Checks if a file or directory exists at `path`.
    fn exists(&self, path: &str) -> bool;

    /// Checks if `path` is an existing regular file.
    fn is_file(&self, path: &str) -> bool;

    /// Checks if `path` is an existing directory.
    fn is_dir(&self, path: &str) -> bool;

    /// Lists all entries inside directory `path`.
    fn list_dir(&self, path: &str) -> Result<Vec<String>, VfsError>;
}

/// Normalizes a virtual path, preventing directory traversal escaping.
pub fn normalize_virtual_path(path: &str) -> Result<String, VfsError> {
    let raw = path.replace('\\', "/");
    let mut parts: Vec<&str> = Vec::new();

    for segment in raw.split('/') {
        match segment {
            "" | "." => continue,
            ".." => {
                if parts.pop().is_none() {
                    return Err(VfsError::SecurityViolation(path.to_string()));
                }
            }
            s => parts.push(s),
        }
    }

    Ok(parts.join("/"))
}

/// In-memory filesystem storing files in memory.
#[derive(Debug, Default, Clone)]
pub struct MemoryFileSystem {
    files: Arc<RwLock<BTreeMap<String, Vec<u8>>>>,
}

impl MemoryFileSystem {
    /// Creates an empty in-memory filesystem.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a file into the in-memory filesystem.
    pub fn insert(&mut self, path: &str, data: Vec<u8>) {
        if let Ok(clean) = normalize_virtual_path(path) {
            self.files.write().unwrap().insert(clean, data);
        }
    }
}

impl FileSystem for MemoryFileSystem {
    fn read(&self, path: &str) -> Result<Vec<u8>, VfsError> {
        let clean = normalize_virtual_path(path)?;
        let files = self.files.read().unwrap();
        files
            .get(&clean)
            .cloned()
            .ok_or_else(|| VfsError::NotFound(path.to_string()))
    }

    fn exists(&self, path: &str) -> bool {
        self.is_file(path) || self.is_dir(path)
    }

    fn is_file(&self, path: &str) -> bool {
        if let Ok(clean) = normalize_virtual_path(path) {
            self.files.read().unwrap().contains_key(&clean)
        } else {
            false
        }
    }

    fn is_dir(&self, path: &str) -> bool {
        if let Ok(clean) = normalize_virtual_path(path) {
            let files = self.files.read().unwrap();
            let prefix = if clean.is_empty() {
                String::new()
            } else {
                format!("{clean}/")
            };
            files.keys().any(|k| k.starts_with(&prefix))
        } else {
            false
        }
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>, VfsError> {
        let clean = normalize_virtual_path(path)?;
        let files = self.files.read().unwrap();
        let prefix = if clean.is_empty() {
            String::new()
        } else {
            format!("{clean}/")
        };

        let mut seen = BTreeSet::new();
        for k in files.keys() {
            if let Some(rest) = k.strip_prefix(&prefix) {
                if !rest.is_empty() {
                    let child_name = rest.split('/').next().unwrap_or("");
                    if !child_name.is_empty() {
                        let entry_path = if clean.is_empty() {
                            child_name.to_string()
                        } else {
                            format!("{clean}/{child_name}")
                        };
                        seen.insert(entry_path);
                    }
                }
            }
        }

        if seen.is_empty() && !clean.is_empty() {
            return Err(VfsError::NotFound(path.to_string()));
        }

        Ok(seen.into_iter().collect())
    }
}

/// Physical filesystem accessing the operating system's filesystem within a sandboxed directory.
#[derive(Debug)]
pub struct PhysicalFileSystem {
    root: PathBuf,
}

impl PhysicalFileSystem {
    /// Creates a new physical filesystem sandboxed to `root`.
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self, VfsError> {
        let canonical = fs::canonicalize(&root).map_err(|e| VfsError::Io(e.to_string()))?;
        Ok(Self { root: canonical })
    }

    fn resolve(&self, path: &str) -> Result<PathBuf, VfsError> {
        let clean = normalize_virtual_path(path)?;
        if clean.is_empty() {
            return Ok(self.root.clone());
        }

        let mut current = self.root.clone();
        for segment in clean.split('/') {
            current.push(segment);
            match fs::symlink_metadata(&current) {
                Ok(meta) => {
                    if meta.file_type().is_symlink() {
                        let canon = fs::canonicalize(&current)
                            .map_err(|_| VfsError::SecurityViolation(path.to_string()))?;
                        if !canon.starts_with(&self.root) {
                            return Err(VfsError::SecurityViolation(path.to_string()));
                        }
                        current = canon;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(VfsError::NotFound(path.to_string()));
                }
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    return Err(VfsError::SecurityViolation(path.to_string()));
                }
                Err(e) => return Err(VfsError::Io(e.to_string())),
            }
        }

        let canonical = fs::canonicalize(&current)
            .map_err(|_| VfsError::SecurityViolation(path.to_string()))?;
        if !canonical.starts_with(&self.root) {
            return Err(VfsError::SecurityViolation(path.to_string()));
        }

        Ok(canonical)
    }
}

impl FileSystem for PhysicalFileSystem {
    fn read(&self, path: &str) -> Result<Vec<u8>, VfsError> {
        let resolved = self.resolve(path)?;
        fs::read(&resolved).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => VfsError::NotFound(path.to_string()),
            std::io::ErrorKind::PermissionDenied => VfsError::SecurityViolation(path.to_string()),
            _ => VfsError::Io(e.to_string()),
        })
    }

    fn exists(&self, path: &str) -> bool {
        self.resolve(path).map(|p| p.exists()).unwrap_or(false)
    }

    fn is_file(&self, path: &str) -> bool {
        self.resolve(path).map(|p| p.is_file()).unwrap_or(false)
    }

    fn is_dir(&self, path: &str) -> bool {
        self.resolve(path).map(|p| p.is_dir()).unwrap_or(false)
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>, VfsError> {
        let clean = normalize_virtual_path(path)?;
        let resolved = self.resolve(path)?;
        let entries = fs::read_dir(&resolved).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => VfsError::NotFound(path.to_string()),
            _ => VfsError::Io(e.to_string()),
        })?;

        let mut results = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| VfsError::Io(e.to_string()))?;
            let name = entry.file_name().to_string_lossy().replace('\\', "/");
            let rel = if clean.is_empty() {
                name
            } else {
                format!("{clean}/{name}")
            };
            results.push(rel);
        }

        Ok(results)
    }
}

/// Packed archive filesystem for bundled assets.
#[derive(Debug, Clone)]
pub struct ArchiveFileSystem {
    entries: HashMap<String, (usize, usize)>,
    data: Vec<u8>,
}

impl ArchiveFileSystem {
    /// Creates an archive builder for creating packed archives.
    #[must_use]
    pub fn builder() -> ArchiveBuilder {
        ArchiveBuilder::new()
    }

    /// Loads an archive from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VfsError> {
        if bytes.len() < 8 || &bytes[0..4] != b"EARC" {
            return Err(VfsError::CorruptArchive("Invalid magic header".into()));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 {
            return Err(VfsError::CorruptArchive(format!(
                "Unsupported version {version}"
            )));
        }

        let mut cursor = 8;
        if cursor + 4 > bytes.len() {
            return Err(VfsError::CorruptArchive("Truncated entry count".into()));
        }

        let count = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        // Each entry header is at least 18 bytes (2B path_len + 8B offset + 8B len).
        if count > (bytes.len() - cursor) / 18 {
            return Err(VfsError::CorruptArchive(
                "Archive entry count exceeds remaining buffer size".into(),
            ));
        }

        let mut entries = HashMap::with_capacity(count);
        for _ in 0..count {
            if cursor + 2 > bytes.len() {
                return Err(VfsError::CorruptArchive("Truncated path length".into()));
            }
            let path_len =
                u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as usize;
            cursor += 2;

            if cursor + path_len > bytes.len() {
                return Err(VfsError::CorruptArchive("Truncated path string".into()));
            }
            let path_str = std::str::from_utf8(&bytes[cursor..cursor + path_len])
                .map_err(|e| VfsError::CorruptArchive(e.to_string()))?;
            let clean_path = normalize_virtual_path(path_str)?;
            cursor += path_len;

            if cursor + 16 > bytes.len() {
                return Err(VfsError::CorruptArchive("Truncated offset/len".into()));
            }
            let offset_u64 = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;
            let len_u64 = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;

            let offset = usize::try_from(offset_u64)
                .map_err(|_| VfsError::CorruptArchive("Entry offset overflows usize".into()))?;
            let len = usize::try_from(len_u64)
                .map_err(|_| VfsError::CorruptArchive("Entry length overflows usize".into()))?;

            entries.insert(clean_path, (offset, len));
        }

        let data = bytes.to_vec();
        Ok(Self { entries, data })
    }
}

impl FileSystem for ArchiveFileSystem {
    fn read(&self, path: &str) -> Result<Vec<u8>, VfsError> {
        let clean = normalize_virtual_path(path)?;
        let &(offset, len) = self
            .entries
            .get(&clean)
            .ok_or_else(|| VfsError::NotFound(path.to_string()))?;

        let end = offset
            .checked_add(len)
            .ok_or_else(|| VfsError::CorruptArchive("Slice offset + length overflow".into()))?;
        if end > self.data.len() {
            return Err(VfsError::CorruptArchive(
                "Slice extends past archive end".into(),
            ));
        }

        Ok(self.data[offset..end].to_vec())
    }

    fn exists(&self, path: &str) -> bool {
        self.is_file(path) || self.is_dir(path)
    }

    fn is_file(&self, path: &str) -> bool {
        if let Ok(clean) = normalize_virtual_path(path) {
            self.entries.contains_key(&clean)
        } else {
            false
        }
    }

    fn is_dir(&self, path: &str) -> bool {
        if let Ok(clean) = normalize_virtual_path(path) {
            let prefix = if clean.is_empty() {
                String::new()
            } else {
                format!("{clean}/")
            };
            self.entries.keys().any(|k| k.starts_with(&prefix))
        } else {
            false
        }
    }

    fn list_dir(&self, path: &str) -> Result<Vec<String>, VfsError> {
        let clean = normalize_virtual_path(path)?;
        let prefix = if clean.is_empty() {
            String::new()
        } else {
            format!("{clean}/")
        };

        let mut seen = BTreeSet::new();
        for k in self.entries.keys() {
            if let Some(rest) = k.strip_prefix(&prefix) {
                if !rest.is_empty() {
                    let child_name = rest.split('/').next().unwrap_or("");
                    if !child_name.is_empty() {
                        let entry_path = if clean.is_empty() {
                            child_name.to_string()
                        } else {
                            format!("{clean}/{child_name}")
                        };
                        seen.insert(entry_path);
                    }
                }
            }
        }

        if seen.is_empty() && !clean.is_empty() {
            return Err(VfsError::NotFound(path.to_string()));
        }

        Ok(seen.into_iter().collect())
    }
}

/// Builder for creating packed archives.
#[derive(Debug, Default)]
pub struct ArchiveBuilder {
    files: Vec<(String, Vec<u8>)>,
}

impl ArchiveBuilder {
    /// Creates an empty archive builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a file with content into the archive.
    pub fn add_file(&mut self, path: &str, content: &[u8]) {
        if let Ok(clean) = normalize_virtual_path(path) {
            self.files.push((clean, content.to_vec()));
        }
    }

    /// Encodes and packs all files into archive bytes.
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        // Magic
        buffer.extend_from_slice(b"EARC");
        // Version 1
        buffer.extend_from_slice(&1u32.to_le_bytes());
        // Entry count
        buffer.extend_from_slice(&(self.files.len() as u32).to_le_bytes());

        // Header entries
        // Calculate header size first to know payload offsets
        let mut header_size = 8 + 4; // Magic + version + count
        for (path, _) in &self.files {
            header_size += 2 + path.len() + 8 + 8; // path_len + path + offset + len
        }

        let mut current_offset = header_size;
        for (path, content) in &self.files {
            buffer.extend_from_slice(&(path.len() as u16).to_le_bytes());
            buffer.extend_from_slice(path.as_bytes());
            buffer.extend_from_slice(&(current_offset as u64).to_le_bytes());
            buffer.extend_from_slice(&(content.len() as u64).to_le_bytes());
            current_offset += content.len();
        }

        // Payloads
        for (_, content) in &self.files {
            buffer.extend_from_slice(content);
        }

        buffer
    }
}

struct MountEntry {
    prefix: String,
    fs: Box<dyn FileSystem>,
    priority: i32,
}

/// The Virtual File System (VFS) router managing hierarchical and prioritized mounts.
pub struct Vfs {
    mounts: Vec<MountEntry>,
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

impl Vfs {
    /// Creates an empty VFS router.
    #[must_use]
    pub fn new() -> Self {
        Self { mounts: Vec::new() }
    }

    /// Mounts a filesystem at `prefix` with default priority 0.
    pub fn mount(&mut self, prefix: &str, fs: Box<dyn FileSystem>) {
        self.mount_priority(prefix, fs, 0);
    }

    /// Mounts a filesystem at `prefix` with a custom priority. Higher priority mounts take precedence.
    pub fn mount_priority(&mut self, prefix: &str, fs: Box<dyn FileSystem>, priority: i32) {
        let clean_prefix = normalize_virtual_path(prefix).unwrap_or_default();
        self.mounts.push(MountEntry {
            prefix: clean_prefix,
            fs,
            priority,
        });

        // Sort descending by priority, then by prefix length descending
        self.mounts.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| b.prefix.len().cmp(&a.prefix.len()))
        });
    }

    /// Reads a file from the VFS by path.
    pub fn read(&self, path: &str) -> Result<Vec<u8>, VfsError> {
        let clean = normalize_virtual_path(path)?;
        for mount in &self.mounts {
            let sub_path = if mount.prefix.is_empty() {
                Some(clean.as_str())
            } else if clean == mount.prefix {
                Some("")
            } else {
                clean
                    .strip_prefix(&mount.prefix)
                    .and_then(|rest| rest.strip_prefix('/'))
            };

            if let Some(sub) = sub_path {
                match mount.fs.read(sub) {
                    Ok(bytes) => return Ok(bytes),
                    Err(VfsError::NotFound(_)) => continue,
                    Err(err) => return Err(err),
                }
            }
        }

        Err(VfsError::NotFound(path.to_string()))
    }

    /// Reads a file from the VFS as a string.
    pub fn read_to_string(&self, path: &str) -> Result<String, VfsError> {
        let bytes = self.read(path)?;
        String::from_utf8(bytes).map_err(|e| VfsError::Io(e.to_string()))
    }

    /// Checks if a file or directory exists in the VFS.
    pub fn exists(&self, path: &str) -> bool {
        let Ok(clean) = normalize_virtual_path(path) else {
            return false;
        };

        for mount in &self.mounts {
            if mount.prefix.is_empty() {
                if mount.fs.exists(&clean) {
                    return true;
                }
            } else if clean == mount.prefix || clean.starts_with(&format!("{}/", mount.prefix)) {
                let sub_path = if clean == mount.prefix {
                    ""
                } else {
                    &clean[mount.prefix.len() + 1..]
                };
                if mount.fs.exists(sub_path) {
                    return true;
                }
            }
        }

        false
    }
}
