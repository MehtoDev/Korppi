//! Program binaries (finding RGL-12, RVK-16 for the GL side): linked programs are kept as
//! `glGetProgramBinary` data keyed by their shader sources, reused by later pipelines with the
//! same sources, and serialized by `GraphicsDevice::pipeline_cache_data` for later runs. Data
//! is tagged with the driver (vendor, renderer, version), and a binary the driver rejects is
//! dropped and the program compiled from source.

use glow::HasContext;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Tag in front of serialized binaries.
const MAGIC: &[u8] = b"KORPPI-GL-PROGRAM-BINARIES\0";
/// Version of the format.
const VERSION: u32 = 1;

/// Program binaries by source key.
pub(crate) struct ProgramBinaryCache {
    /// Whether the driver offers any binary format.
    enabled: bool,
    /// Vendor, renderer and version of the driver.
    driver: String,
    binaries: HashMap<u64, glow::ProgramBinary>,
}

/// Key of a program linked from shaders with these `(stage, source hash)` pairs.
pub(crate) fn program_key(stages: impl IntoIterator<Item = (u32, u64)>) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    for (stage, source) in stages {
        stage.hash(&mut hasher);
        source.hash(&mut hasher);
    }
    hasher.finish()
}

/// Hash of a shader source.
pub(crate) fn source_hash(source: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

impl ProgramBinaryCache {
    /// Queries the driver's binary support.
    ///
    /// # Safety
    /// The context must be current.
    pub(crate) unsafe fn new(gl: &glow::Context) -> Self {
        // SAFETY: forwarded from the caller.
        let (formats, driver) = unsafe {
            (
                gl.get_parameter_i32(glow::NUM_PROGRAM_BINARY_FORMATS),
                format!(
                    "{}\n{}\n{}",
                    gl.get_parameter_string(glow::VENDOR),
                    gl.get_parameter_string(glow::RENDERER),
                    gl.get_parameter_string(glow::VERSION)
                ),
            )
        };
        Self {
            enabled: formats > 0,
            driver,
            binaries: HashMap::new(),
        }
    }

    /// Whether binaries are kept.
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn get(&self, key: u64) -> Option<&glow::ProgramBinary> {
        self.binaries.get(&key)
    }

    pub(crate) fn insert(&mut self, key: u64, binary: glow::ProgramBinary) {
        self.binaries.insert(key, binary);
    }

    pub(crate) fn remove(&mut self, key: u64) {
        self.binaries.remove(&key);
    }

    /// Number of binaries kept.
    pub(crate) fn len(&self) -> usize {
        self.binaries.len()
    }

    /// The binaries behind the tag and the driver identity.
    pub(crate) fn serialize(&self) -> Vec<u8> {
        if !self.enabled {
            return Vec::new();
        }
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.driver.len() as u32).to_le_bytes());
        out.extend_from_slice(self.driver.as_bytes());
        out.extend_from_slice(&(self.binaries.len() as u32).to_le_bytes());
        for (key, binary) in &self.binaries {
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&binary.format.to_le_bytes());
            out.extend_from_slice(&(binary.buffer.len() as u32).to_le_bytes());
            out.extend_from_slice(&binary.buffer);
        }
        out
    }

    /// Adds the binaries of serialized `data` if it was made by this driver; returns whether
    /// it was.
    pub(crate) fn load(&mut self, data: &[u8]) -> bool {
        if !self.enabled {
            return false;
        }
        let parsed = (|| {
            let mut rest = data.strip_prefix(MAGIC)?;
            fn take<'a>(rest: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
                let (head, tail) = rest.split_at_checked(n)?;
                *rest = tail;
                Some(head)
            }
            let u32_at = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().unwrap());
            if u32_at(take(&mut rest, 4)?) != VERSION {
                return None;
            }
            let driver_len = u32_at(take(&mut rest, 4)?) as usize;
            if take(&mut rest, driver_len)? != self.driver.as_bytes() {
                return None;
            }
            let count = u32_at(take(&mut rest, 4)?) as usize;
            if count > rest.len() / 16 {
                return None;
            }
            let mut binaries = Vec::with_capacity(count);
            for _ in 0..count {
                let key = u64::from_le_bytes(take(&mut rest, 8)?.try_into().unwrap());
                let format = u32_at(take(&mut rest, 4)?);
                let len = u32_at(take(&mut rest, 4)?) as usize;
                let buffer = take(&mut rest, len)?.to_vec();
                binaries.push((key, glow::ProgramBinary { buffer, format }));
            }
            Some(binaries)
        })();
        match parsed {
            Some(binaries) => {
                self.binaries.extend(binaries);
                true
            }
            None => false,
        }
    }
}
