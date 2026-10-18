//! Shader asset representation and bytecode/source loader.

use crate::error::AssetError;
use crate::handle::Asset;
use crate::loader::AssetLoader;

/// Pipeline shader execution stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex shading stage.
    Vertex,
    /// Fragment / pixel shading stage.
    Fragment,
    /// Compute shading stage.
    Compute,
}

/// In-memory representation of compiled shader bytecode or high-level source text.
#[derive(Debug, Clone)]
pub struct ShaderAsset {
    /// Target execution stage.
    pub stage: ShaderStage,
    /// Compiled 32-bit SPIR-V binary words.
    pub spirv: Vec<u32>,
    /// High-level source code (GLSL, WGSL) if available.
    pub source: Option<String>,
}

impl Asset for ShaderAsset {
    const TYPE_NAME: &'static str = "Shader";
}

/// Loader for compiled SPIR-V bytecode (`.spv`) and shader source files (`.vert`, `.frag`, `.comp`).
#[derive(Debug, Default, Clone, Copy)]
pub struct ShaderLoader;

impl ShaderLoader {
    /// Creates a new `ShaderLoader`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl AssetLoader for ShaderLoader {
    type Asset = ShaderAsset;

    fn extensions(&self) -> &[&str] {
        &["spv", "vert", "frag", "comp"]
    }

    fn load(&self, bytes: &[u8], path: &str) -> Result<Self::Asset, AssetError> {
        let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();

        let stage = match ext.as_str() {
            "vert" => ShaderStage::Vertex,
            "frag" => ShaderStage::Fragment,
            "comp" => ShaderStage::Compute,
            _ => {
                // If .spv, inspect path for sub-extension (e.g., "pbr.vert.spv")
                if path.contains(".vert.") {
                    ShaderStage::Vertex
                } else if path.contains(".comp.") {
                    ShaderStage::Compute
                } else {
                    ShaderStage::Fragment
                }
            }
        };

        if ext == "spv" {
            if !bytes.len().is_multiple_of(4) {
                return Err(AssetError::Decode(
                    "SPIR-V byte length must be a multiple of 4".into(),
                ));
            }
            let mut words = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.as_chunks::<4>().0 {
                words.push(u32::from_le_bytes(*chunk));
            }
            Ok(ShaderAsset {
                stage,
                spirv: words,
                source: None,
            })
        } else {
            let source_text = std::str::from_utf8(bytes)
                .map_err(|e| AssetError::Decode(format!("Invalid UTF-8 in shader source: {e}")))?;
            Ok(ShaderAsset {
                stage,
                spirv: Vec::new(),
                source: Some(source_text.to_string()),
            })
        }
    }
}
