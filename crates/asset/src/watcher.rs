//! File system watcher for live asset hot-reloading.

use crate::error::AssetError;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Events emitted when files change on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetEvent {
    /// A new asset file was created.
    Added(String),
    /// An existing asset file was modified on disk.
    Modified(String),
    /// An asset file was deleted from disk.
    Removed(String),
}

/// File system change detector monitoring watched asset directories.
#[derive(Debug)]
pub struct FileWatcher {
    interval: Duration,
    watched_dirs: Vec<PathBuf>,
    file_records: HashMap<PathBuf, (SystemTime, u64)>,
}

impl FileWatcher {
    /// Creates a new `FileWatcher` with a specific polling interval.
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            watched_dirs: Vec::new(),
            file_records: HashMap::new(),
        }
    }

    /// Registers a directory to be recursively monitored for asset modifications.
    pub fn watch_directory<P: AsRef<Path>>(&mut self, dir: P) -> Result<(), AssetError> {
        let canonical = fs::canonicalize(&dir)
            .map_err(|e| AssetError::Vfs(crate::vfs::VfsError::Io(e.to_string())))?;
        if !self.watched_dirs.contains(&canonical) {
            self.watched_dirs.push(canonical.clone());
            Self::scan_dir_into(&canonical, &mut self.file_records);
        }
        Ok(())
    }

    fn scan_dir_into(dir: &Path, records: &mut HashMap<PathBuf, (SystemTime, u64)>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::scan_dir_into(&path, records);
            } else if path.is_file() {
                if let Ok(meta) = path.metadata() {
                    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    let len = meta.len();
                    records.insert(path, (mtime, len));
                }
            }
        }
    }

    /// Polls the filesystem for changes since the last poll, returning newly detected events.
    pub fn poll_events(&mut self) -> Vec<AssetEvent> {
        let mut current_scan = HashMap::new();
        for dir in &self.watched_dirs.clone() {
            Self::scan_dir_into(dir, &mut current_scan);
        }

        let mut events = Vec::new();

        // Detect Added and Modified files
        for (path, &(mtime, len)) in &current_scan {
            if let Some(&(old_mtime, old_len)) = self.file_records.get(path) {
                if mtime > old_mtime || len != old_len {
                    events.push(AssetEvent::Modified(path.to_string_lossy().to_string()));
                }
            } else {
                events.push(AssetEvent::Added(path.to_string_lossy().to_string()));
            }
        }

        // Detect Removed files
        for old_path in self.file_records.keys() {
            if !current_scan.contains_key(old_path) {
                events.push(AssetEvent::Removed(old_path.to_string_lossy().to_string()));
            }
        }

        self.file_records = current_scan;
        events
    }

    /// Returns the polling interval configured for this watcher.
    #[must_use]
    pub fn interval(&self) -> Duration {
        self.interval
    }
}
