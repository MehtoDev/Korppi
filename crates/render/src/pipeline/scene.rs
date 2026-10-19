//! Scene, mesh, material, camera, and light descriptors for [`ForwardPipeline`](super::ForwardPipeline).

use crate::clustered::ReverseZ;
use crate::pbr::PbrMaterial;
use korppi_math::{Mat4, Vec3};
use korppi_rhi::{
    BufferDescriptor, BufferHandle, BufferUsage, GraphicsDevice, MemoryUsage, RhiError,
};

/// Interleaved 3D vertex format (`32` bytes: position `vec3`, normal `vec3`, uv `vec2`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex3d {
    /// Object-space vertex position.
    pub position: [f32; 3],
    /// Object-space unit surface normal.
    pub normal: [f32; 3],
    /// Texture coordinates in `[0, 1]`.
    pub uv: [f32; 2],
}

gpu_pod!(Vertex3d = 8 * 4);

/// CPU-side indexed triangle mesh geometry.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    /// Interleaved vertices.
    pub vertices: Vec<Vertex3d>,
    /// Triangle list indices (`u32`, counter-clockwise front-facing).
    pub indices: Vec<u32>,
}

impl MeshData {
    /// Constructs a quad in the XY plane (`z = 0`) centered at the origin with normal `+Z`.
    pub fn facing_quad(width: f32, height: f32) -> Self {
        let hx = width * 0.5;
        let hy = height * 0.5;
        let n = [0.0, 0.0, 1.0];
        Self {
            vertices: vec![
                Vertex3d {
                    position: [-hx, -hy, 0.0],
                    normal: n,
                    uv: [0.0, 1.0],
                },
                Vertex3d {
                    position: [hx, -hy, 0.0],
                    normal: n,
                    uv: [1.0, 1.0],
                },
                Vertex3d {
                    position: [hx, hy, 0.0],
                    normal: n,
                    uv: [1.0, 0.0],
                },
                Vertex3d {
                    position: [-hx, hy, 0.0],
                    normal: n,
                    uv: [0.0, 0.0],
                },
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    /// Constructs a horizontal plane in the XZ plane (`y = 0`) centered at the origin with normal `+Y`.
    pub fn plane(width: f32, depth: f32) -> Self {
        let hx = width * 0.5;
        let hz = depth * 0.5;
        let n = [0.0, 1.0, 0.0];
        Self {
            vertices: vec![
                Vertex3d {
                    position: [-hx, 0.0, hz],
                    normal: n,
                    uv: [0.0, 1.0],
                },
                Vertex3d {
                    position: [hx, 0.0, hz],
                    normal: n,
                    uv: [1.0, 1.0],
                },
                Vertex3d {
                    position: [hx, 0.0, -hz],
                    normal: n,
                    uv: [1.0, 0.0],
                },
                Vertex3d {
                    position: [-hx, 0.0, -hz],
                    normal: n,
                    uv: [0.0, 0.0],
                },
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    /// Constructs a unit cube (`[-0.5, 0.5]^3`) with flat per-face normals.
    pub fn unit_cube() -> Self {
        let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
            // +Z
            (
                [0.0, 0.0, 1.0],
                [
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ],
            ),
            // -Z
            (
                [0.0, 0.0, -1.0],
                [
                    [0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                ],
            ),
            // +X
            (
                [1.0, 0.0, 0.0],
                [
                    [0.5, -0.5, 0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, 0.5, 0.5],
                ],
            ),
            // -X
            (
                [-1.0, 0.0, 0.0],
                [
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                ],
            ),
            // +Y
            (
                [0.0, 1.0, 0.0],
                [
                    [-0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                ],
            ),
            // -Y
            (
                [0.0, -1.0, 0.0],
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, -0.5, 0.5],
                ],
            ),
        ];

        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for (normal, positions) in faces {
            let base = vertices.len() as u32;
            for (pos, uv) in positions.into_iter().zip(uvs) {
                vertices.push(Vertex3d {
                    position: pos,
                    normal,
                    uv,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self { vertices, indices }
    }

    /// Constructs a UV sphere centered at the origin with outward smooth normals.
    pub fn uv_sphere(radius: f32, stacks: u32, slices: u32) -> Self {
        let stacks = stacks.max(3);
        let slices = slices.max(3);
        let mut vertices = Vec::with_capacity(((stacks + 1) * (slices + 1)) as usize);
        let mut indices = Vec::with_capacity((stacks * slices * 6) as usize);

        for i in 0..=stacks {
            let v = i as f32 / stacks as f32;
            let phi = v * std::f32::consts::PI;
            let (sin_phi, cos_phi) = phi.sin_cos();

            for j in 0..=slices {
                let u = j as f32 / slices as f32;
                let theta = u * std::f32::consts::TAU;
                let (sin_theta, cos_theta) = theta.sin_cos();

                let nx = sin_phi * cos_theta;
                let ny = cos_phi;
                let nz = sin_phi * sin_theta;

                vertices.push(Vertex3d {
                    position: [nx * radius, ny * radius, nz * radius],
                    normal: [nx, ny, nz],
                    uv: [u, v],
                });
            }
        }

        let ring = slices + 1;
        for i in 0..stacks {
            for j in 0..slices {
                let a = i * ring + j;
                let b = (i + 1) * ring + j;
                let c = (i + 1) * ring + (j + 1);
                let d = i * ring + (j + 1);
                indices.extend_from_slice(&[a, c, b, a, d, c]);
            }
        }

        Self { vertices, indices }
    }
}

/// GPU-uploaded vertex and index buffer pair for a 3D mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMesh {
    /// Vertex buffer handle (`Vertex3d` stride `32` bytes).
    pub vertex_buffer: BufferHandle,
    /// Index buffer handle (`u32`).
    pub index_buffer: BufferHandle,
    /// Number of indices to draw.
    pub index_count: u32,
}

impl GpuMesh {
    /// Uploads `mesh` into device-local vertex and index buffers.
    pub fn upload(
        device: &dyn GraphicsDevice,
        mesh: &MeshData,
        label: Option<&'static str>,
    ) -> Result<Self, RhiError> {
        let vb_bytes: &[u8] = korppi_core::pod::cast_slice(&mesh.vertices);
        let ib_bytes: &[u8] = korppi_core::pod::cast_slice(&mesh.indices);

        let vertex_buffer = device.create_buffer(
            &BufferDescriptor {
                size: (vb_bytes.len().max(32)) as u64,
                usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuOnly,
                label,
            },
            Some(vb_bytes),
        )?;
        let index_buffer = device.create_buffer(
            &BufferDescriptor {
                size: (ib_bytes.len().max(4)) as u64,
                usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuOnly,
                label,
            },
            Some(ib_bytes),
        )?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        })
    }

    /// Destroys the GPU vertex and index buffers.
    pub fn destroy(self, device: &dyn GraphicsDevice) -> Result<(), RhiError> {
        device.destroy_buffer(self.vertex_buffer)?;
        device.destroy_buffer(self.index_buffer)?;
        Ok(())
    }
}

/// Material parameters for a 3D surface rendered by [`ForwardPipeline`](super::ForwardPipeline).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForwardMaterial {
    /// Base linear RGB reflectance/albedo.
    pub albedo: Vec3,
    /// Linear HDR emissive radiance added to outgoing light.
    pub emissive: Vec3,
    /// Physically Based Rendering surface parameters.
    pub pbr: PbrMaterial,
}

impl Default for ForwardMaterial {
    fn default() -> Self {
        Self {
            albedo: Vec3::ONE,
            emissive: Vec3::ZERO,
            pbr: PbrMaterial::default(),
        }
    }
}

/// 3D perspective camera with Reverse-Z projection (`near -> 1.0`, `far -> 0.0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera3d {
    /// Eye position in world space.
    pub position: Vec3,
    /// Look-at target in world space.
    pub target: Vec3,
    /// Up vector in world space.
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// Viewport aspect ratio (`width / height`).
    pub aspect_ratio: f32,
    /// Near clip distance (`> 0.0`).
    pub z_near: f32,
    /// Far clip distance (`> z_near`).
    pub z_far: f32,
}

impl Default for Camera3d {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            aspect_ratio: 16.0 / 9.0,
            z_near: 0.1,
            z_far: 100.0,
        }
    }
}

impl Camera3d {
    /// Right-handed world-to-view matrix (`-Z` forward, `RND-25`).
    #[inline]
    pub fn view_matrix(&self) -> Mat4 {
        let pos = if self.position.is_finite() {
            self.position
        } else {
            Vec3::new(0.0, 0.0, 5.0)
        };
        let raw_dir = if self.target.is_finite() {
            self.target - pos
        } else {
            Vec3::NEG_Z
        };
        let forward = raw_dir.try_normalize().unwrap_or(Vec3::NEG_Z);
        let raw_up = if self.up.is_finite() {
            self.up.try_normalize().unwrap_or(Vec3::Y)
        } else {
            Vec3::Y
        };
        let up = if forward.dot(raw_up).abs() > 0.999 {
            if forward.y.abs() < 0.9 {
                Vec3::Y
            } else {
                Vec3::Z
            }
        } else {
            raw_up
        };
        Mat4::look_at_rh(pos, pos + forward, up)
    }

    /// Right-handed Reverse-Z perspective projection matrix (`near -> 1.0`, `far -> 0.0`, `RND-25`).
    #[inline]
    pub fn projection_matrix(&self) -> Mat4 {
        let fov = if self.fov_y_radians.is_finite() {
            self.fov_y_radians.clamp(0.01, 3.10)
        } else {
            std::f32::consts::FRAC_PI_3
        };
        let aspect = if self.aspect_ratio.is_finite() && self.aspect_ratio > 1e-4 {
            self.aspect_ratio
        } else {
            1.0
        };
        let near = if self.z_near.is_finite() && self.z_near > 1e-4 {
            self.z_near
        } else {
            0.1
        };
        let far = if self.z_far.is_finite() && self.z_far > near + 1e-3 {
            self.z_far
        } else {
            near + 100.0
        };
        ReverseZ::perspective_rh(fov, aspect, near, far)
    }
}

/// Single mesh instance draw call in a [`ForwardScene`].
#[derive(Clone, Debug)]
pub struct MeshDraw {
    /// Uploaded mesh buffers.
    pub mesh: GpuMesh,
    /// Object-to-world transformation matrix.
    pub transform: Mat4,
    /// Surface material.
    pub material: ForwardMaterial,
    /// Whether this draw participates in shadow-map passes.
    pub casts_shadows: bool,
}

/// World-space directional light (sun/moon) with optional Cascaded Shadow Maps (CSM).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldDirectionalLight {
    /// Propagation direction of the light rays in world space (pointing from the light toward surfaces).
    pub direction: Vec3,
    /// Linear RGB color.
    pub color: Vec3,
    /// Direct illuminance scalar.
    pub illuminance: f32,
    /// Whether this directional light renders CSM cascades into the shadow atlas.
    pub casts_shadows: bool,
}

/// World-space omnidirectional point light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPointLight {
    /// World-space center position.
    pub position: Vec3,
    /// Maximum attenuation cutoff radius (`> 0.0`).
    pub radius: f32,
    /// Linear RGB color.
    pub color: Vec3,
    /// Luminous intensity scalar.
    pub intensity: f32,
    /// Whether this point light renders an octahedral omnidirectional shadow tile.
    pub casts_shadows: bool,
}

/// World-space cone spotlight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSpotLight {
    /// World-space apex position.
    pub position: Vec3,
    /// World-space forward cone axis direction.
    pub direction: Vec3,
    /// Maximum range cutoff distance (`> 0.0`).
    pub range: f32,
    /// Outer cone half-angle in radians.
    pub half_angle: f32,
    /// Linear RGB color.
    pub color: Vec3,
    /// Luminous intensity scalar.
    pub intensity: f32,
    /// Whether this spotlight renders a perspective shadow tile in the shadow atlas.
    pub casts_shadows: bool,
}

/// Complete frame scene description submitted to [`ForwardPipeline`](super::ForwardPipeline).
#[derive(Clone, Debug, Default)]
pub struct ForwardScene {
    /// Active 3D camera.
    pub camera: Camera3d,
    /// Optional primary directional light.
    pub directional_light: Option<WorldDirectionalLight>,
    /// Uniform linear RGB ambient irradiance (modulated by SSAO and surface albedo).
    pub ambient_light: Vec3,
    /// Linear HDR clear color for the sky/background.
    pub clear_color: [f32; 4],
    /// Opaque mesh draw calls.
    pub draws: Vec<MeshDraw>,
    /// Active omnidirectional point lights.
    pub point_lights: Vec<WorldPointLight>,
    /// Active cone spotlights.
    pub spot_lights: Vec<WorldSpotLight>,
}
