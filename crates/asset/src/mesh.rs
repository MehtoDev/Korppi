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
                        let p_idx: i32 = parts[0].parse().map_err(|_| {
                            AssetError::Decode(format!("Invalid OBJ position index '{}'", parts[0]))
                        })?;
                        if p_idx == 0 {
                            return Err(AssetError::Decode(
                                "OBJ position index cannot be 0".into(),
                            ));
                        }
                        let t_idx: i32 = if parts.len() > 1 && !parts[1].is_empty() {
                            let val: i32 = parts[1].parse().map_err(|_| {
                                AssetError::Decode(format!("Invalid OBJ UV index '{}'", parts[1]))
                            })?;
                            if val == 0 {
                                return Err(AssetError::Decode("OBJ UV index cannot be 0".into()));
                            }
                            val
                        } else {
                            0
                        };
                        let n_idx: i32 = if parts.len() > 2 && !parts[2].is_empty() {
                            let val: i32 = parts[2].parse().map_err(|_| {
                                AssetError::Decode(format!(
                                    "Invalid OBJ normal index '{}'",
                                    parts[2]
                                ))
                            })?;
                            if val == 0 {
                                return Err(AssetError::Decode(
                                    "OBJ normal index cannot be 0".into(),
                                ));
                            }
                            val
                        } else {
                            0
                        };

                        let key = (p_idx, t_idx, n_idx);
                        let final_index = if let Some(&existing) = vertex_lut.get(&key) {
                            existing
                        } else {
                            let idx = positions.len() as u32;

                            let p = resolve_obj_index(&raw_positions, p_idx)?;
                            let uv = if t_idx != 0 {
                                resolve_obj_index(&raw_uvs, t_idx)?
                            } else {
                                Vec2::ZERO
                            };
                            let n = if n_idx != 0 {
                                resolve_obj_index(&raw_normals, n_idx)?
                            } else {
                                Vec3::ZERO
                            };

                            positions.push(p);
                            uvs.push(uv);
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

        let mut cursor: usize = 12;
        let mut json_chunk = None;
        let mut bin_chunk = None;

        while cursor + 8 <= bytes.len() {
            let chunk_len =
                u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
            let chunk_type = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap());
            cursor += 8;

            let end = cursor
                .checked_add(chunk_len)
                .ok_or_else(|| AssetError::Decode("GLB chunk length overflow".into()))?;
            if end > bytes.len() {
                return Err(AssetError::Decode("Truncated GLB chunk".into()));
            }

            let chunk_data = &bytes[cursor..end];
            cursor = end;

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
        if positions.is_empty() {
            return Err(AssetError::Decode(
                "glTF primitive has 0 vertex positions".into(),
            ));
        }

        let normals = if let Some(n_acc_idx) = prim["attributes"]["NORMAL"].as_u64() {
            let n = self.read_vec3_accessor(json, bin_data, n_acc_idx as usize)?;
            if n.len() != positions.len() {
                return Err(AssetError::Decode(
                    "glTF NORMAL count does not match POSITION count".into(),
                ));
            }
            n
        } else {
            vec![Vec3::Y; positions.len()]
        };

        let uvs = if let Some(uv_acc_idx) = prim["attributes"]["TEXCOORD_0"].as_u64() {
            let u = self.read_vec2_accessor(json, bin_data, uv_acc_idx as usize)?;
            if u.len() != positions.len() {
                return Err(AssetError::Decode(
                    "glTF TEXCOORD_0 count does not match POSITION count".into(),
                ));
            }
            u
        } else {
            vec![Vec2::ZERO; positions.len()]
        };

        let indices = if let Some(idx_acc_idx) = prim["indices"].as_u64() {
            let indices = self.read_scalar_indices(json, bin_data, idx_acc_idx as usize)?;
            let vertex_count = positions.len();
            if let Some(&bad_idx) = indices.iter().find(|&&idx| (idx as usize) >= vertex_count) {
                return Err(AssetError::Decode(format!(
                    "glTF index {bad_idx} out of bounds for {vertex_count} vertices"
                )));
            }
            indices
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

    fn validate_accessor_slice(
        json: &Value,
        bin_data: &[u8],
        accessor_idx: usize,
        elem_size: usize,
    ) -> Result<(usize, usize, u32), AssetError> {
        let accessors = json["accessors"]
            .as_array()
            .ok_or_else(|| AssetError::Decode("Missing accessors array in glTF".into()))?;
        let accessor = accessors
            .get(accessor_idx)
            .ok_or_else(|| AssetError::Decode(format!("Accessor {accessor_idx} out of bounds")))?;

        let count_u64 = accessor["count"].as_u64().unwrap_or(0);
        let count = usize::try_from(count_u64)
            .map_err(|_| AssetError::Decode("Accessor count overflows usize".into()))?;
        let comp_type = accessor["componentType"].as_u64().unwrap_or(5123) as u32;
        let buffer_view_idx = accessor["bufferView"]
            .as_u64()
            .ok_or_else(|| AssetError::Decode("Accessor missing bufferView".into()))?
            as usize;
        let acc_offset = usize::try_from(accessor["byteOffset"].as_u64().unwrap_or(0))
            .map_err(|_| AssetError::Decode("Accessor byteOffset overflow".into()))?;

        let buffer_views = json["bufferViews"]
            .as_array()
            .ok_or_else(|| AssetError::Decode("Missing bufferViews array in glTF".into()))?;
        let buffer_view = buffer_views.get(buffer_view_idx).ok_or_else(|| {
            AssetError::Decode(format!("bufferView {buffer_view_idx} out of bounds"))
        })?;
        let view_offset = usize::try_from(buffer_view["byteOffset"].as_u64().unwrap_or(0))
            .map_err(|_| AssetError::Decode("bufferView byteOffset overflow".into()))?;

        let start = view_offset
            .checked_add(acc_offset)
            .ok_or_else(|| AssetError::Decode("glTF accessor start offset overflow".into()))?;
        let byte_len = count
            .checked_mul(elem_size)
            .ok_or_else(|| AssetError::Decode("glTF accessor byte length overflow".into()))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| AssetError::Decode("glTF accessor end offset overflow".into()))?;

        if end > bin_data.len() {
            return Err(AssetError::Decode(
                "glTF accessor range extends past binary buffer".into(),
            ));
        }

        if let Some(view_len_u64) = buffer_view["byteLength"].as_u64() {
            let view_len = usize::try_from(view_len_u64)
                .map_err(|_| AssetError::Decode("bufferView byteLength overflow".into()))?;
            let view_end = view_offset
                .checked_add(view_len)
                .ok_or_else(|| AssetError::Decode("bufferView end offset overflow".into()))?;
            if end > view_end {
                return Err(AssetError::Decode(
                    "glTF accessor extends past bufferView byteLength".into(),
                ));
            }
        }

        Ok((start, count, comp_type))
    }

    fn read_vec3_accessor(
        &self,
        json: &Value,
        bin_data: &[u8],
        accessor_idx: usize,
    ) -> Result<Vec<Vec3>, AssetError> {
        let (start, count, _) = Self::validate_accessor_slice(json, bin_data, accessor_idx, 12)?;
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let offset = start + i * 12;
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
        let (start, count, _) = Self::validate_accessor_slice(json, bin_data, accessor_idx, 8)?;
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let offset = start + i * 8;
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
        let comp_type = json["accessors"][accessor_idx]["componentType"]
            .as_u64()
            .unwrap_or(5123) as u32;
        let elem_size = match comp_type {
            5121 => 1,
            5123 => 2,
            5125 => 4,
            _ => {
                return Err(AssetError::Decode(format!(
                    "Unsupported index component type {comp_type}"
                )))
            }
        };

        let (start, count, _) =
            Self::validate_accessor_slice(json, bin_data, accessor_idx, elem_size)?;
        let mut indices = Vec::with_capacity(count);

        match comp_type {
            5121 => {
                for i in 0..count {
                    indices.push(bin_data[start + i] as u32);
                }
            }
            5123 => {
                for i in 0..count {
                    let offset = start + i * 2;
                    let val = u16::from_le_bytes(bin_data[offset..offset + 2].try_into().unwrap());
                    indices.push(val as u32);
                }
            }
            5125 => {
                for i in 0..count {
                    let offset = start + i * 4;
                    let val = u32::from_le_bytes(bin_data[offset..offset + 4].try_into().unwrap());
                    indices.push(val);
                }
            }
            _ => unreachable!(),
        }

        Ok(indices)
    }
}

fn resolve_obj_index<T: Copy>(slice: &[T], idx: i32) -> Result<T, AssetError> {
    let resolved = if idx > 0 {
        usize::try_from(idx - 1).ok()
    } else if idx < 0 {
        isize::try_from(slice.len())
            .ok()
            .and_then(|len| len.checked_add(idx as isize))
            .and_then(|pos| usize::try_from(pos).ok())
    } else {
        None
    };

    resolved.and_then(|i| slice.get(i).copied()).ok_or_else(|| {
        AssetError::Decode(format!(
            "OBJ index {idx} out of bounds for attribute list of length {}",
            slice.len()
        ))
    })
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
        let ext = path
            .rsplit_once('.')
            .map(|(_, e)| e.to_ascii_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "glb" => self.load_glb(bytes),
            "gltf" => {
                let json: Value = serde_json::from_slice(bytes)
                    .map_err(|e| AssetError::Decode(format!("glTF JSON parse error: {e}")))?;
                self.parse_gltf_json(&json, &[])
            }
            "obj" => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|e| AssetError::Decode(format!("Invalid UTF-8 in OBJ: {e}")))?;
                self.load_obj(text)
            }
            _ => Err(AssetError::UnsupportedFormat(ext)),
        }
    }
}
