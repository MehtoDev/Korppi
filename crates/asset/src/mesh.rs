//! 3D Mesh asset types and geometry loaders (Wavefront OBJ, glTF 2.0 / GLB).

use crate::error::AssetError;
use crate::handle::Asset;
use crate::loader::AssetLoader;
use korppi_math::{Aabb, Vec2, Vec3, Vec4};
use serde_json::Value;
use std::collections::HashMap;

/// Submesh partition referencing a continuous range of indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submesh {
    /// Starting index offset in the parent mesh's index buffer.
    pub index_start: u32,
    /// Number of indices belonging to this submesh.
    pub index_count: u32,
    /// Optional index into the material library.
    pub material_index: Option<usize>,
}

/// In-memory representation of a 3D polygonal mesh with GPU-ready vertex buffers.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex spatial positions.
    pub positions: Vec<Vec3>,
    /// Vertex surface normal vectors.
    pub normals: Vec<Vec3>,
    /// Vertex UV texture coordinates (channel 0).
    pub uvs: Vec<Vec2>,
    /// Vertex tangent vectors with handedness sign in w.
    pub tangents: Vec<Vec4>,
    /// Vertex color attributes (RGBA).
    pub colors: Vec<Vec4>,
    /// 32-bit triangle index buffer.
    pub indices: Vec<u32>,
    /// Submesh partitions for multi-material drawing.
    pub submeshes: Vec<Submesh>,
    /// Local Axis-Aligned Bounding Box (AABB).
    pub bounds: Aabb,
}

impl Asset for Mesh {
    const TYPE_NAME: &'static str = "Mesh";
}

/// Mesh loader capable of decoding Wavefront OBJ and glTF 2.0 / GLB models.
#[derive(Debug, Default, Clone, Copy)]
pub struct MeshLoader;

impl MeshLoader {
    /// Creates a new `MeshLoader`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn load_obj(&self, text: &str) -> Result<Mesh, AssetError> {
        let mut raw_positions: Vec<Vec3> = Vec::new();
        let mut raw_uvs: Vec<Vec2> = Vec::new();
        let mut raw_normals: Vec<Vec3> = Vec::new();

        let mut positions: Vec<Vec3> = Vec::new();
        let mut uvs: Vec<Vec2> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let mut vertex_lut: HashMap<(i32, i32, i32), u32> = HashMap::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut tokens = line.split_whitespace();
            let Some(cmd) = tokens.next() else {
                continue;
            };

            match cmd {
                "v" => {
                    let x: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let y: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let z: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    raw_positions.push(Vec3::new(x, y, z));
                }
                "vt" => {
                    let u: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let v: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    raw_uvs.push(Vec2::new(u, v));
                }
                "vn" => {
                    let x: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let y: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    let z: f32 = tokens.next().unwrap_or("0").parse().unwrap_or(0.0);
                    raw_normals.push(Vec3::new(x, y, z).normalize_or_zero());
                }
                "f" => {
                    let face_tokens: Vec<&str> = tokens.collect();
                    if face_tokens.len() < 3 {
                        continue;
                    }

                    let mut face_indices = Vec::new();
                    for token in face_tokens {
                        let parts: Vec<&str> = token.split('/').collect();
                        let p_idx: i32 = parts[0].parse().unwrap_or(0);
                        let t_idx: i32 = if parts.len() > 1 && !parts[1].is_empty() {
                            parts[1].parse().unwrap_or(0)
                        } else {
                            0
                        };
                        let n_idx: i32 = if parts.len() > 2 && !parts[2].is_empty() {
                            parts[2].parse().unwrap_or(0)
                        } else {
                            0
                        };

                        let key = (p_idx, t_idx, n_idx);
                        let final_index = if let Some(&existing) = vertex_lut.get(&key) {
                            existing
                        } else {
                            let idx = positions.len() as u32;

                            // Resolve position (1-based, supports negative)
                            let p = if p_idx > 0 {
                                raw_positions.get((p_idx - 1) as usize).copied()
                            } else if p_idx < 0 {
                                raw_positions
                                    .get((raw_positions.len() as i32 + p_idx) as usize)
                                    .copied()
                            } else {
                                None
                            }
                            .unwrap_or(Vec3::ZERO);
                            positions.push(p);

                            // Resolve UV
                            let uv = if t_idx > 0 {
                                raw_uvs.get((t_idx - 1) as usize).copied()
                            } else if t_idx < 0 {
                                raw_uvs
                                    .get((raw_uvs.len() as i32 + t_idx) as usize)
                                    .copied()
                            } else {
                                None
                            }
                            .unwrap_or(Vec2::ZERO);
                            uvs.push(uv);

                            // Resolve Normal
                            let n = if n_idx > 0 {
                                raw_normals.get((n_idx - 1) as usize).copied()
                            } else if n_idx < 0 {
                                raw_normals
                                    .get((raw_normals.len() as i32 + n_idx) as usize)
                                    .copied()
                            } else {
                                None
                            }
                            .unwrap_or(Vec3::ZERO);
                            normals.push(n);

                            vertex_lut.insert(key, idx);
                            idx
                        };

                        face_indices.push(final_index);
                    }

                    // Triangulate face using triangle fan
                    for i in 1..face_indices.len() - 1 {
                        indices.push(face_indices[0]);
                        indices.push(face_indices[i]);
                        indices.push(face_indices[i + 1]);
                    }
                }
                _ => {}
            }
        }

        // Auto-generate normals if missing
        let has_normals = normals.iter().any(|n| n.length_squared() > 1e-4);
        if !has_normals && !positions.is_empty() {
            normals = vec![Vec3::ZERO; positions.len()];
            for chunk in indices.as_chunks::<3>().0 {
                let i0 = chunk[0] as usize;
                let i1 = chunk[1] as usize;
                let i2 = chunk[2] as usize;
                let p0 = positions[i0];
                let p1 = positions[i1];
                let p2 = positions[i2];
                let normal = (p1 - p0).cross(p2 - p0);
                normals[i0] += normal;
                normals[i1] += normal;
                normals[i2] += normal;
            }
            for n in &mut normals {
                *n = n.normalize_or_zero();
            }
        }

        let bounds = calculate_bounds(&positions);
        let submeshes = vec![Submesh {
            index_start: 0,
            index_count: indices.len() as u32,
            material_index: None,
        }];

        Ok(Mesh {
            positions,
            normals,
            uvs,
            tangents: Vec::new(),
            colors: Vec::new(),
            indices,
            submeshes,
            bounds,
        })
    }

    fn load_glb(&self, bytes: &[u8]) -> Result<Mesh, AssetError> {
        if bytes.len() < 20 {
            return Err(AssetError::Decode("GLB data too short".into()));
        }

        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != 0x46546C67 {
            return Err(AssetError::Decode("Invalid GLB magic header".into()));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 2 {
            return Err(AssetError::Decode(format!(
                "Unsupported glTF version {version}"
            )));
        }

        let mut cursor = 12;
        let mut json_chunk = None;
        let mut bin_chunk = None;

        while cursor + 8 <= bytes.len() {
            let chunk_len =
                u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            let chunk_type = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap());
            cursor += 8;

            if cursor + chunk_len > bytes.len() {
                return Err(AssetError::Decode("Truncated GLB chunk".into()));
            }

            let chunk_data = &bytes[cursor..cursor + chunk_len];
            cursor += chunk_len;

            if chunk_type == 0x4E4F534A {
                // "JSON"
                json_chunk = Some(chunk_data);
            } else if chunk_type == 0x004E4942 {
                // "BIN\0"
                bin_chunk = Some(chunk_data);
            }
        }

        let json_bytes =
            json_chunk.ok_or_else(|| AssetError::Decode("Missing JSON chunk in GLB".into()))?;
        let bin_bytes = bin_chunk.unwrap_or(&[]);

        let json: Value = serde_json::from_slice(json_bytes)
            .map_err(|e| AssetError::Decode(format!("glTF JSON parse error: {e}")))?;

        self.parse_gltf_json(&json, bin_bytes)
    }

    fn parse_gltf_json(&self, json: &Value, bin_data: &[u8]) -> Result<Mesh, AssetError> {
        let meshes = json["meshes"]
            .as_array()
            .ok_or_else(|| AssetError::Decode("No meshes array in glTF".into()))?;

        if meshes.is_empty() {
            return Err(AssetError::Decode("Empty meshes array in glTF".into()));
        }

        let primitives = meshes[0]["primitives"]
            .as_array()
            .ok_or_else(|| AssetError::Decode("No primitives in mesh".into()))?;

        if primitives.is_empty() {
            return Err(AssetError::Decode("Empty primitives in mesh".into()));
        }

        let prim = &primitives[0];
        let pos_acc_idx = prim["attributes"]["POSITION"]
            .as_u64()
            .ok_or_else(|| AssetError::Decode("Missing POSITION attribute in primitive".into()))?
            as usize;

        let positions = self.read_vec3_accessor(json, bin_data, pos_acc_idx)?;

        let normals = if let Some(n_acc_idx) = prim["attributes"]["NORMAL"].as_u64() {
            self.read_vec3_accessor(json, bin_data, n_acc_idx as usize)?
        } else {
            vec![Vec3::Y; positions.len()]
        };

        let uvs = if let Some(uv_acc_idx) = prim["attributes"]["TEXCOORD_0"].as_u64() {
            self.read_vec2_accessor(json, bin_data, uv_acc_idx as usize)?
        } else {
            vec![Vec2::ZERO; positions.len()]
        };

        let indices = if let Some(idx_acc_idx) = prim["indices"].as_u64() {
            self.read_scalar_indices(json, bin_data, idx_acc_idx as usize)?
        } else {
            (0..positions.len() as u32).collect()
        };

        let bounds = calculate_bounds(&positions);
        let submeshes = vec![Submesh {
            index_start: 0,
            index_count: indices.len() as u32,
            material_index: None,
        }];

        Ok(Mesh {
            positions,
            normals,
            uvs,
            tangents: Vec::new(),
            colors: Vec::new(),
            indices,
            submeshes,
            bounds,
        })
    }

    fn read_vec3_accessor(
        &self,
        json: &Value,
        bin_data: &[u8],
        accessor_idx: usize,
    ) -> Result<Vec<Vec3>, AssetError> {
        let accessor = &json["accessors"][accessor_idx];
        let count = accessor["count"].as_u64().unwrap_or(0) as usize;
        let buffer_view_idx = accessor["bufferView"].as_u64().unwrap_or(0) as usize;
        let acc_offset = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;

        let buffer_view = &json["bufferViews"][buffer_view_idx];
        let view_offset = buffer_view["byteOffset"].as_u64().unwrap_or(0) as usize;

        let start = view_offset + acc_offset;
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let offset = start + i * 12;
            if offset + 12 > bin_data.len() {
                break;
            }
            let x = f32::from_le_bytes(bin_data[offset..offset + 4].try_into().unwrap());
            let y = f32::from_le_bytes(bin_data[offset + 4..offset + 8].try_into().unwrap());
            let z = f32::from_le_bytes(bin_data[offset + 8..offset + 12].try_into().unwrap());
            results.push(Vec3::new(x, y, z));
        }

        Ok(results)
    }

    fn read_vec2_accessor(
        &self,
        json: &Value,
        bin_data: &[u8],
        accessor_idx: usize,
    ) -> Result<Vec<Vec2>, AssetError> {
        let accessor = &json["accessors"][accessor_idx];
        let count = accessor["count"].as_u64().unwrap_or(0) as usize;
        let buffer_view_idx = accessor["bufferView"].as_u64().unwrap_or(0) as usize;
        let acc_offset = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;

        let buffer_view = &json["bufferViews"][buffer_view_idx];
        let view_offset = buffer_view["byteOffset"].as_u64().unwrap_or(0) as usize;

        let start = view_offset + acc_offset;
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let offset = start + i * 8;
            if offset + 8 > bin_data.len() {
                break;
            }
            let u = f32::from_le_bytes(bin_data[offset..offset + 4].try_into().unwrap());
            let v = f32::from_le_bytes(bin_data[offset + 4..offset + 8].try_into().unwrap());
            results.push(Vec2::new(u, v));
        }

        Ok(results)
    }

    fn read_scalar_indices(
        &self,
        json: &Value,
        bin_data: &[u8],
        accessor_idx: usize,
    ) -> Result<Vec<u32>, AssetError> {
        let accessor = &json["accessors"][accessor_idx];
        let count = accessor["count"].as_u64().unwrap_or(0) as usize;
        let comp_type = accessor["componentType"].as_u64().unwrap_or(5123) as u32;
        let buffer_view_idx = accessor["bufferView"].as_u64().unwrap_or(0) as usize;
        let acc_offset = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;

        let buffer_view = &json["bufferViews"][buffer_view_idx];
        let view_offset = buffer_view["byteOffset"].as_u64().unwrap_or(0) as usize;

        let start = view_offset + acc_offset;
        let mut indices = Vec::with_capacity(count);

        match comp_type {
            5121 => {
                // UNSIGNED_BYTE
                for i in 0..count {
                    if start + i < bin_data.len() {
                        indices.push(bin_data[start + i] as u32);
                    }
                }
            }
            5123 => {
                // UNSIGNED_SHORT
                for i in 0..count {
                    let offset = start + i * 2;
                    if offset + 2 <= bin_data.len() {
                        let val =
                            u16::from_le_bytes(bin_data[offset..offset + 2].try_into().unwrap());
                        indices.push(val as u32);
                    }
                }
            }
            5125 => {
                // UNSIGNED_INT
                for i in 0..count {
                    let offset = start + i * 4;
                    if offset + 4 <= bin_data.len() {
                        let val =
                            u32::from_le_bytes(bin_data[offset..offset + 4].try_into().unwrap());
                        indices.push(val);
                    }
                }
            }
            _ => {
                return Err(AssetError::Decode(format!(
                    "Unsupported index component type {comp_type}"
                )))
            }
        }

        Ok(indices)
    }
}

fn calculate_bounds(positions: &[Vec3]) -> Aabb {
    if positions.is_empty() {
        return Aabb::from_min_max(Vec3::ZERO, Vec3::ZERO);
    }
    let mut min = positions[0];
    let mut max = positions[0];
    for &p in &positions[1..] {
        min = min.min(p);
        max = max.max(p);
    }
    Aabb::from_min_max(min, max)
}

impl AssetLoader for MeshLoader {
    type Asset = Mesh;

    fn extensions(&self) -> &[&str] {
        &["obj", "glb", "gltf"]
    }

    fn load(&self, bytes: &[u8], path: &str) -> Result<Self::Asset, AssetError> {
        let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();

        if ext == "glb" {
            self.load_glb(bytes)
        } else if ext == "gltf" {
            let json: Value = serde_json::from_slice(bytes)
                .map_err(|e| AssetError::Decode(format!("glTF JSON parse error: {e}")))?;
            self.parse_gltf_json(&json, &[])
        } else {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| AssetError::Decode(format!("Invalid UTF-8 in OBJ: {e}")))?;
            self.load_obj(text)
        }
    }
}
