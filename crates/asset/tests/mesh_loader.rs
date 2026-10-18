//! Test suite for Mesh Loader (OBJ & glTF 2.0).

use korppi_asset::loader::AssetLoader;
use korppi_asset::mesh::MeshLoader;
use korppi_math::Vec3;

#[test]
fn test_obj_mesh_loading() {
    let obj_data = b"
# Simple Quad with Normals and UVs
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0

vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0

vn 0.0 0.0 1.0

# Quad face decomposed into two triangles
f 1/1/1 2/2/1 3/3/1 4/4/1
";

    let loader = MeshLoader::new();
    let mesh = loader.load(obj_data, "quad.obj").unwrap();

    assert_eq!(mesh.positions.len(), 4);
    assert_eq!(mesh.normals.len(), 4);
    assert_eq!(mesh.uvs.len(), 4);
    // Triangulated quad = 2 triangles = 6 indices
    assert_eq!(mesh.indices.len(), 6);
    assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);

    // Check bounds
    assert_eq!(mesh.bounds.min, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(mesh.bounds.max, Vec3::new(1.0, 1.0, 0.0));
}

#[test]
fn test_obj_auto_normal_generation_when_missing() {
    let obj_data = b"
# Triangle without normals
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0

f 1 2 3
";

    let loader = MeshLoader::new();
    let mesh = loader.load(obj_data, "tri.obj").unwrap();

    assert_eq!(mesh.positions.len(), 3);
    assert_eq!(mesh.indices.len(), 3);
    // Normals should have been generated automatically
    assert_eq!(mesh.normals.len(), 3);
    // Counter-clockwise triangle in XY plane points in +Z
    assert!((mesh.normals[0].z - 1.0).abs() < 1e-4);
}

#[test]
fn test_glb_binary_gltf_loading() {
    // Construct a minimal valid GLB binary containing a single triangle
    let json_text = r#"{
        "asset": {"version": "2.0"},
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0
                },
                "indices": 1
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "max": [1.0, 1.0, 0.0],
                "min": [0.0, 0.0, 0.0]
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": 36
            },
            {
                "buffer": 0,
                "byteOffset": 36,
                "byteLength": 6
            }
        ],
        "buffers": [{
            "byteLength": 44
        }]
    }"#;

    let json_bytes = json_text.as_bytes();
    // Padding JSON to 4-byte alignment with spaces
    let json_padding = (4 - (json_bytes.len() % 4)) % 4;
    let total_json_len = json_bytes.len() + json_padding;

    // Binary payload: 3 vertices (3 * 3 * 4 = 36 bytes) + 3 indices (3 * 2 = 6 bytes) + 2 bytes pad = 44 bytes
    let mut bin_data = Vec::new();
    // v0: (0.0, 0.0, 0.0)
    bin_data.extend_from_slice(&0.0f32.to_le_bytes());
    bin_data.extend_from_slice(&0.0f32.to_le_bytes());
    bin_data.extend_from_slice(&0.0f32.to_le_bytes());
    // v1: (1.0, 0.0, 0.0)
    bin_data.extend_from_slice(&1.0f32.to_le_bytes());
    bin_data.extend_from_slice(&0.0f32.to_le_bytes());
    bin_data.extend_from_slice(&0.0f32.to_le_bytes());
    // v2: (0.0, 1.0, 0.0)
    bin_data.extend_from_slice(&0.0f32.to_le_bytes());
    bin_data.extend_from_slice(&1.0f32.to_le_bytes());
    bin_data.extend_from_slice(&0.0f32.to_le_bytes());

    // Indices: [0, 1, 2] as u16
    bin_data.extend_from_slice(&0u16.to_le_bytes());
    bin_data.extend_from_slice(&1u16.to_le_bytes());
    bin_data.extend_from_slice(&2u16.to_le_bytes());
    // Padding
    bin_data.extend_from_slice(&[0, 0]);

    let bin_padding = (4 - (bin_data.len() % 4)) % 4;
    let total_bin_len = bin_data.len() + bin_padding;

    let total_glb_len = 12 + 8 + total_json_len + 8 + total_bin_len;

    let mut glb = Vec::new();
    // Header: magic (0x46546C67), version (2), length
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_glb_len as u32).to_le_bytes());

    // Chunk 0: JSON (type 0x4E4F534A)
    glb.extend_from_slice(&(total_json_len as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    glb.extend_from_slice(json_bytes);
    glb.extend(std::iter::repeat_n(b' ', json_padding));

    // Chunk 1: BIN (type 0x004E4942)
    glb.extend_from_slice(&(total_bin_len as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
    glb.extend_from_slice(&bin_data);
    glb.extend(std::iter::repeat_n(0, bin_padding));

    let loader = MeshLoader::new();
    let mesh = loader.load(&glb, "triangle.glb").unwrap();

    assert_eq!(mesh.positions.len(), 3);
    assert_eq!(mesh.indices, vec![0, 1, 2]);
    assert_eq!(mesh.bounds.min, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(mesh.bounds.max, Vec3::new(1.0, 1.0, 0.0));
}
