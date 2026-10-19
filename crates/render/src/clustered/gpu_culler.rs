//! Asynchronous GPU compute shader froxel light culler and buffer structures.

use crate::clustered::culler::{ClusterRecord, CullingOutput, MAX_FEATURES_PER_FROXEL};
use crate::clustered::decal::Decal;
use crate::clustered::grid::FroxelAabb;
use crate::clustered::light::{PointLight, SpotLight};
use crate::clustered::probe::{ProbeShape, ReflectionProbe};
use korppi_math::Mat3;
use korppi_rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupHandle, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutHandle, BindingResource, BindingType, BufferDescriptor,
    BufferHandle, BufferUsage, CommandBufferRecorder, ComputePipelineDescriptor, FrameIndex,
    GraphicsDevice, MemoryUsage, PipelineHandle, RhiError, ShaderDescriptor, ShaderHandle,
    ShaderSource, ShaderStage, ShaderStages, FRAMES_IN_FLIGHT,
};

/// Storage buffers of set 1, by binding: whether the shader only reads them.
const STORAGE_BINDINGS: [bool; 10] = [
    true,  // 0: froxel AABBs
    true,  // 1: point lights
    true,  // 2: spot lights
    false, // 3: cluster records
    false, // 4: light indices
    false, // 5: atomic counters
    true,  // 6: decals
    true,  // 7: probes
    false, // 8: decal indices
    false, // 9: probe indices
];

/// Pre-compiled SPIR-V bytecode for `clustered_light_cull.comp`.
pub const CLUSTERED_LIGHT_CULL_SPIRV: &[u8] = include_bytes!("shaders/clustered_light_cull.spv");

/// GLSL compute shader source code for `clustered_light_cull.comp`.
pub const CLUSTERED_LIGHT_CULL_GLSL: &str = include_str!("shaders/clustered_light_cull.comp");

/// GPU representation of uniform parameters for clustered culling.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullingParamsGpu {
    /// Total number of 3D froxel cells.
    pub total_clusters: u32,
    /// Number of active view-space point lights.
    pub point_light_count: u32,
    /// Number of active view-space spotlights.
    pub spot_light_count: u32,
    /// Number of active view-space decals.
    pub decal_count: u32,
    /// Number of active view-space reflection probes.
    pub probe_count: u32,
    /// Maximum capacity of global light index buffer.
    pub max_light_indices: u32,
    /// Maximum capacity of global decal index buffer.
    pub max_decal_indices: u32,
    /// Maximum capacity of global probe index buffer.
    pub max_probe_indices: u32,
}

/// GPU std430 alignment struct for a froxel bounding volume.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct FroxelAabbGpu {
    /// Axis-Aligned Bounding Box minimum corner in view space.
    pub aabb_min: [f32; 4],
    /// Axis-Aligned Bounding Box maximum corner in view space.
    pub aabb_max: [f32; 4],
    /// Circumscribed bounding sphere center and radius in view space.
    pub sphere_center_radius: [f32; 4],
}

/// GPU std430 alignment struct for a point light.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct PointLightGpu {
    /// Position and radius in view space: `[x, y, z, radius]`.
    pub position_radius: [f32; 4],
    /// Color and intensity: `[r, g, b, intensity]`.
    pub color_intensity: [f32; 4],
}

/// GPU std430 alignment struct for a spot light.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpotLightGpu {
    /// Apex position and range in view space: `[x, y, z, range]`.
    pub position_range: [f32; 4],
    /// Forward direction and outer half-angle: `[dx, dy, dz, half_angle]`.
    pub direction_half_angle: [f32; 4],
    /// Color and intensity: `[r, g, b, intensity]`.
    pub color_intensity: [f32; 4],
}

/// GPU std430 alignment struct for an oriented decal volume.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct DecalGpu {
    /// View-space center position: `[x, y, z, 0.0]`.
    pub position_vs: [f32; 4],
    /// Local half-extents: `[hx, hy, hz, 0.0]`.
    pub half_extents: [f32; 4],
    /// Local X axis direction: `[b0.x, b0.y, b0.z, 0.0]`.
    pub axis0: [f32; 4],
    /// Local Y axis direction: `[b1.x, b1.y, b1.z, 0.0]`.
    pub axis1: [f32; 4],
    /// Local Z axis direction: `[b2.x, b2.y, b2.z, 0.0]`.
    pub axis2: [f32; 4],
}

/// GPU std430 alignment struct for a localized reflection probe.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct ProbeGpu {
    /// View-space position and shape type: `[x, y, z, type]` (0.0 = Sphere, 1.0 = Box).
    pub position_type: [f32; 4],
    /// Shape parameters: radius for Sphere, half-extents for Box.
    pub param: [f32; 4],
}

gpu_pod!(
    CullingParamsGpu = 8 * 4,
    FroxelAabbGpu = 3 * 16,
    PointLightGpu = 2 * 16,
    SpotLightGpu = 3 * 16,
    DecalGpu = 5 * 16,
    ProbeGpu = 2 * 16,
);

/// Persistent GPU storage buffers holding culled records and indices for direct forward shading binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCullingBuffers {
    /// SSBO holding `ClusterRecord` entries per froxel.
    pub records_buffer: BufferHandle,
    /// Byte size of `records_buffer`.
    pub records_size: u64,
    /// SSBO holding flattened light indices.
    pub light_indices_buffer: BufferHandle,
    /// Byte size of `light_indices_buffer`.
    pub light_indices_size: u64,
    /// SSBO holding flattened decal indices.
    pub decal_indices_buffer: BufferHandle,
    /// SSBO holding flattened reflection probe indices.
    pub probe_indices_buffer: BufferHandle,
    /// SSBO holding the frame's uploaded view-space point lights.
    pub points_buffer: BufferHandle,
    /// Byte size of `points_buffer`.
    pub points_size: u64,
    /// SSBO holding the frame's uploaded view-space spot lights.
    pub spots_buffer: BufferHandle,
    /// Byte size of `spots_buffer`.
    pub spots_size: u64,
    /// Total froxels in the cluster grid.
    pub total_clusters: u32,
}

/// GPU compute pipeline manager for dispatching clustered culling shaders.
///
/// Host-written input buffers (`ubo`, `froxels`, `points`, `spots`, `decals`, `probes`, `counter`)
/// are duplicated across [`FRAMES_IN_FLIGHT`] slots so concurrent in-flight frames never race.
/// Temporary conversion vectors are retained across frames so steady-state dispatch performs zero
/// heap allocations.
pub struct GpuLightCuller {
    pipeline: Option<PipelineHandle>,
    /// Set 0 (parameters) and set 1 (storage buffers) layouts.
    layouts: Option<[BindGroupLayoutHandle; 2]>,
    /// Set 0 group per frame-in-flight slot, created with the uniform buffer.
    params_group: [Option<BindGroupHandle>; FRAMES_IN_FLIGHT],
    /// Set 1 group per frame-in-flight slot and the buffer ranges it was created with.
    buffers_group: [Option<(BindGroupHandle, [BindGroupEntry; 10])>; FRAMES_IN_FLIGHT],
    shader: Option<ShaderHandle>,
    ubo: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    froxels_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    points_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    spots_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    decals_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    probes_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    counter_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    records_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    indices_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    decal_indices_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    probe_indices_buf: [Option<BufferHandle>; FRAMES_IN_FLIGHT],
    allocated_froxels: [u32; FRAMES_IN_FLIGHT],
    allocated_points: [usize; FRAMES_IN_FLIGHT],
    allocated_spots: [usize; FRAMES_IN_FLIGHT],
    allocated_decals: [usize; FRAMES_IN_FLIGHT],
    allocated_probes: [usize; FRAMES_IN_FLIGHT],
    allocated_clusters: [u32; FRAMES_IN_FLIGHT],
    max_light_indices: [u32; FRAMES_IN_FLIGHT],
    max_decal_indices: [u32; FRAMES_IN_FLIGHT],
    max_probe_indices: [u32; FRAMES_IN_FLIGHT],
    scratch_froxels: Vec<FroxelAabbGpu>,
    scratch_points: Vec<PointLightGpu>,
    scratch_spots: Vec<SpotLightGpu>,
    scratch_decals: Vec<DecalGpu>,
    scratch_probes: Vec<ProbeGpu>,
}

impl Default for GpuLightCuller {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuLightCuller {
    /// Creates an uninitialized GPU light culler with no allocated buffers.
    pub fn new() -> Self {
        Self {
            pipeline: None,
            layouts: None,
            params_group: [None; FRAMES_IN_FLIGHT],
            buffers_group: [None; FRAMES_IN_FLIGHT],
            shader: None,
            ubo: [None; FRAMES_IN_FLIGHT],
            froxels_buf: [None; FRAMES_IN_FLIGHT],
            points_buf: [None; FRAMES_IN_FLIGHT],
            spots_buf: [None; FRAMES_IN_FLIGHT],
            decals_buf: [None; FRAMES_IN_FLIGHT],
            probes_buf: [None; FRAMES_IN_FLIGHT],
            counter_buf: [None; FRAMES_IN_FLIGHT],
            records_buf: [None; FRAMES_IN_FLIGHT],
            indices_buf: [None; FRAMES_IN_FLIGHT],
            decal_indices_buf: [None; FRAMES_IN_FLIGHT],
            probe_indices_buf: [None; FRAMES_IN_FLIGHT],
            allocated_froxels: [0; FRAMES_IN_FLIGHT],
            allocated_points: [0; FRAMES_IN_FLIGHT],
            allocated_spots: [0; FRAMES_IN_FLIGHT],
            allocated_decals: [0; FRAMES_IN_FLIGHT],
            allocated_probes: [0; FRAMES_IN_FLIGHT],
            allocated_clusters: [0; FRAMES_IN_FLIGHT],
            max_light_indices: [0; FRAMES_IN_FLIGHT],
            max_decal_indices: [0; FRAMES_IN_FLIGHT],
            max_probe_indices: [0; FRAMES_IN_FLIGHT],
            scratch_froxels: Vec::new(),
            scratch_points: Vec::new(),
            scratch_spots: Vec::new(),
            scratch_decals: Vec::new(),
            scratch_probes: Vec::new(),
        }
    }

    /// Compiles the compute shader and creates the compute pipeline on the graphics device.
    pub fn init(&mut self, device: &dyn GraphicsDevice) -> Result<(), RhiError> {
        let shader_desc = ShaderDescriptor {
            stage: ShaderStage::Compute,
            entry_point: "main",
            source: ShaderSource::SpirvBytes(CLUSTERED_LIGHT_CULL_SPIRV),
            label: Some("clustered_light_cull_comp"),
        };
        let shader = device.create_shader(&shader_desc)?;
        let params_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                },
                stages: ShaderStages::COMPUTE,
            }],
            label: Some("clustered_light_cull_params"),
        })?;
        let buffers_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: STORAGE_BINDINGS
                .iter()
                .enumerate()
                .map(|(binding, &read_only)| BindGroupLayoutEntry {
                    binding: binding as u32,
                    ty: BindingType::StorageBuffer {
                        read_only,
                        dynamic_offset: false,
                    },
                    stages: ShaderStages::COMPUTE,
                })
                .collect(),
            label: Some("clustered_light_cull_buffers"),
        })?;
        self.layouts = Some([params_layout, buffers_layout]);

        let pipe_desc = ComputePipelineDescriptor {
            compute_shader: shader,
            bind_group_layouts: vec![params_layout, buffers_layout],
            push_constant_ranges: vec![],
            label: Some("clustered_light_cull_pipeline"),
        };
        let pipeline = device.create_compute_pipeline(&pipe_desc)?;

        self.shader = Some(shader);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    /// Returns the persistent GPU buffer handle holding cluster records (slot 0).
    #[inline]
    pub fn records_buffer(&self) -> Option<BufferHandle> {
        self.records_buf[0]
    }

    /// Returns the persistent GPU buffer handle holding flattened light indices (slot 0).
    #[inline]
    pub fn light_indices_buffer(&self) -> Option<BufferHandle> {
        self.indices_buf[0]
    }

    /// Returns the persistent GPU buffer handle holding flattened decal indices (slot 0).
    #[inline]
    pub fn decal_indices_buffer(&self) -> Option<BufferHandle> {
        self.decal_indices_buf[0]
    }

    /// Returns the persistent GPU buffer handle holding flattened probe indices (slot 0).
    #[inline]
    pub fn probe_indices_buffer(&self) -> Option<BufferHandle> {
        self.probe_indices_buf[0]
    }

    /// Ensures all persistent GPU buffers for `slot` are created with sufficient capacity without per-frame churn (`RND-23`).
    #[allow(clippy::too_many_arguments)]
    fn ensure_buffers(
        &mut self,
        device: &dyn GraphicsDevice,
        slot: usize,
        total_clusters: u32,
        point_count: usize,
        spot_count: usize,
        decal_count: usize,
        probe_count: usize,
    ) -> Result<(), RhiError> {
        let max_light_indices =
            (total_clusters.saturating_mul((MAX_FEATURES_PER_FROXEL as u32) * 2)).max(1024);
        let max_decal_indices =
            (total_clusters.saturating_mul(MAX_FEATURES_PER_FROXEL as u32)).max(512);
        let max_probe_indices =
            (total_clusters.saturating_mul(MAX_FEATURES_PER_FROXEL as u32)).max(256);

        if self.ubo[slot].is_none() {
            let ubo = device.create_buffer(
                &BufferDescriptor {
                    size: std::mem::size_of::<CullingParamsGpu>() as u64,
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("culling_params_ubo"),
                },
                None,
            )?;
            self.ubo[slot] = Some(ubo);
            let [layout, _] = self.layouts.ok_or_else(|| {
                RhiError::ValidationError("GpuLightCuller pipeline not initialized".to_string())
            })?;
            self.params_group[slot] = Some(device.create_bind_group(&BindGroupDescriptor {
                layout,
                entries: vec![BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: ubo,
                        offset: 0,
                        size: size_of::<CullingParamsGpu>() as u64,
                    },
                }],
                label: Some("clustered_light_cull_params"),
            })?);
        }

        if self.froxels_buf[slot].is_none() || self.allocated_froxels[slot] < total_clusters {
            if let Some(old) = self.froxels_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = (total_clusters.max(1) as usize) * std::mem::size_of::<FroxelAabbGpu>();
            self.froxels_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("froxels_input_ssbo"),
                },
                None,
            )?);
            self.allocated_froxels[slot] = total_clusters.max(1);
        }

        let pt_cap = point_count.max(16);
        if self.points_buf[slot].is_none() || self.allocated_points[slot] < pt_cap {
            if let Some(old) = self.points_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = pt_cap * std::mem::size_of::<PointLightGpu>();
            self.points_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("point_lights_input_ssbo"),
                },
                None,
            )?);
            self.allocated_points[slot] = pt_cap;
        }

        let sp_cap = spot_count.max(16);
        if self.spots_buf[slot].is_none() || self.allocated_spots[slot] < sp_cap {
            if let Some(old) = self.spots_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = sp_cap * std::mem::size_of::<SpotLightGpu>();
            self.spots_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("spot_lights_input_ssbo"),
                },
                None,
            )?);
            self.allocated_spots[slot] = sp_cap;
        }

        let dec_cap = decal_count.max(16);
        if self.decals_buf[slot].is_none() || self.allocated_decals[slot] < dec_cap {
            if let Some(old) = self.decals_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = dec_cap * std::mem::size_of::<DecalGpu>();
            self.decals_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("decals_input_ssbo"),
                },
                None,
            )?);
            self.allocated_decals[slot] = dec_cap;
        }

        let prb_cap = probe_count.max(16);
        if self.probes_buf[slot].is_none() || self.allocated_probes[slot] < prb_cap {
            if let Some(old) = self.probes_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = prb_cap * std::mem::size_of::<ProbeGpu>();
            self.probes_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("probes_input_ssbo"),
                },
                None,
            )?);
            self.allocated_probes[slot] = prb_cap;
        }

        let rec_clusters = total_clusters.max(1);
        if self.records_buf[slot].is_none() || self.allocated_clusters[slot] < rec_clusters {
            if let Some(old) = self.records_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = rec_clusters as usize * std::mem::size_of::<ClusterRecord>();
            self.records_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    memory: MemoryUsage::GpuToCpu,
                    label: Some("cluster_records_output_ssbo"),
                },
                None,
            )?);
            self.allocated_clusters[slot] = rec_clusters;
        }

        if self.indices_buf[slot].is_none() || self.max_light_indices[slot] < max_light_indices {
            if let Some(old) = self.indices_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = max_light_indices as usize * std::mem::size_of::<u32>();
            self.indices_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    memory: MemoryUsage::GpuToCpu,
                    label: Some("global_light_indices_ssbo"),
                },
                None,
            )?);
            self.max_light_indices[slot] = max_light_indices;
        }

        if self.decal_indices_buf[slot].is_none()
            || self.max_decal_indices[slot] < max_decal_indices
        {
            if let Some(old) = self.decal_indices_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = max_decal_indices as usize * std::mem::size_of::<u32>();
            self.decal_indices_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    memory: MemoryUsage::GpuToCpu,
                    label: Some("global_decal_indices_ssbo"),
                },
                None,
            )?);
            self.max_decal_indices[slot] = max_decal_indices;
        }

        if self.probe_indices_buf[slot].is_none()
            || self.max_probe_indices[slot] < max_probe_indices
        {
            if let Some(old) = self.probe_indices_buf[slot].take() {
                device.destroy_buffer(old)?;
            }
            let size = max_probe_indices as usize * std::mem::size_of::<u32>();
            self.probe_indices_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: size as u64,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    memory: MemoryUsage::GpuToCpu,
                    label: Some("global_probe_indices_ssbo"),
                },
                None,
            )?);
            self.max_probe_indices[slot] = max_probe_indices;
        }

        if self.counter_buf[slot].is_none() {
            self.counter_buf[slot] = Some(device.create_buffer(
                &BufferDescriptor {
                    size: 16,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    memory: MemoryUsage::GpuToCpu,
                    label: Some("atomic_counter_ssbo"),
                },
                None,
            )?);
        }

        Ok(())
    }

    /// Uploads feature data to the persistent GPU buffers for `frame_index` and records compute
    /// dispatch commands into `recorder` with zero steady-state heap allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_culling_for_frame(
        &mut self,
        device: &dyn GraphicsDevice,
        recorder: &mut dyn CommandBufferRecorder,
        frame_index: FrameIndex,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
        decals: &[Decal],
        probes: &[ReflectionProbe],
    ) -> Result<GpuCullingBuffers, RhiError> {
        let pipeline = self.pipeline.ok_or_else(|| {
            RhiError::ValidationError("GpuLightCuller pipeline not initialized".to_string())
        })?;
        let slot = frame_index.index() % FRAMES_IN_FLIGHT;
        let total_clusters = froxels.len() as u32;
        let point_count = point_lights.len();
        let spot_count = spot_lights.len();
        let decal_count = decals.len();
        let probe_count = probes.len();

        self.ensure_buffers(
            device,
            slot,
            total_clusters,
            point_count,
            spot_count,
            decal_count,
            probe_count,
        )?;

        let max_light_indices = self.max_light_indices[slot];
        let max_decal_indices = self.max_decal_indices[slot];
        let max_probe_indices = self.max_probe_indices[slot];

        // 1. Upload parameters UBO
        let params = CullingParamsGpu {
            total_clusters,
            point_light_count: point_count as u32,
            spot_light_count: spot_count as u32,
            decal_count: decal_count as u32,
            probe_count: probe_count as u32,
            max_light_indices,
            max_decal_indices,
            max_probe_indices,
        };
        let params_bytes = korppi_core::pod::bytes_of(&params);
        device.write_buffer(self.ubo[slot].unwrap(), 0, params_bytes)?;

        // 2. Upload froxels using reusable scratch vector
        self.scratch_froxels.clear();
        if self.scratch_froxels.capacity() < froxels.len() {
            self.scratch_froxels
                .reserve(froxels.len() - self.scratch_froxels.capacity());
        }
        for f in froxels {
            self.scratch_froxels.push(FroxelAabbGpu {
                aabb_min: [f.aabb_min.x, f.aabb_min.y, f.aabb_min.z, 0.0],
                aabb_max: [f.aabb_max.x, f.aabb_max.y, f.aabb_max.z, 0.0],
                sphere_center_radius: [
                    f.sphere_center.x,
                    f.sphere_center.y,
                    f.sphere_center.z,
                    f.sphere_radius,
                ],
            });
        }
        let froxels_bytes: &[u8] = korppi_core::pod::cast_slice(&self.scratch_froxels);
        if !froxels_bytes.is_empty() {
            device.write_buffer(self.froxels_buf[slot].unwrap(), 0, froxels_bytes)?;
        }

        // 3. Upload point lights using reusable scratch vector
        self.scratch_points.clear();
        let pt_needed = point_lights.len().max(1);
        if self.scratch_points.capacity() < pt_needed {
            self.scratch_points
                .reserve(pt_needed - self.scratch_points.capacity());
        }
        for p in point_lights {
            self.scratch_points.push(PointLightGpu {
                position_radius: [p.position_vs.x, p.position_vs.y, p.position_vs.z, p.radius],
                color_intensity: [p.color.x, p.color.y, p.color.z, p.intensity],
            });
        }
        if self.scratch_points.is_empty() {
            self.scratch_points.push(PointLightGpu::default());
        }
        let points_bytes: &[u8] = korppi_core::pod::cast_slice(&self.scratch_points);
        device.write_buffer(self.points_buf[slot].unwrap(), 0, points_bytes)?;

        // 4. Upload spot lights using reusable scratch vector
        self.scratch_spots.clear();
        let sp_needed = spot_lights.len().max(1);
        if self.scratch_spots.capacity() < sp_needed {
            self.scratch_spots
                .reserve(sp_needed - self.scratch_spots.capacity());
        }
        for s in spot_lights {
            self.scratch_spots.push(SpotLightGpu {
                position_range: [s.position_vs.x, s.position_vs.y, s.position_vs.z, s.range],
                direction_half_angle: [
                    s.direction_vs.x,
                    s.direction_vs.y,
                    s.direction_vs.z,
                    s.half_angle,
                ],
                color_intensity: [s.color.x, s.color.y, s.color.z, s.intensity],
            });
        }
        if self.scratch_spots.is_empty() {
            self.scratch_spots.push(SpotLightGpu::default());
        }
        let spots_bytes: &[u8] = korppi_core::pod::cast_slice(&self.scratch_spots);
        device.write_buffer(self.spots_buf[slot].unwrap(), 0, spots_bytes)?;

        // 5. Upload decals using reusable scratch vector
        self.scratch_decals.clear();
        let dec_needed = decals.len().max(1);
        if self.scratch_decals.capacity() < dec_needed {
            self.scratch_decals
                .reserve(dec_needed - self.scratch_decals.capacity());
        }
        for d in decals {
            let r_mat = Mat3::from_quat(d.orientation_vs);
            let b0 = r_mat.col(0);
            let b1 = r_mat.col(1);
            let b2 = r_mat.col(2);
            self.scratch_decals.push(DecalGpu {
                position_vs: [d.position_vs.x, d.position_vs.y, d.position_vs.z, 0.0],
                half_extents: [d.half_extents.x, d.half_extents.y, d.half_extents.z, 0.0],
                axis0: [b0.x, b0.y, b0.z, 0.0],
                axis1: [b1.x, b1.y, b1.z, 0.0],
                axis2: [b2.x, b2.y, b2.z, 0.0],
            });
        }
        if self.scratch_decals.is_empty() {
            self.scratch_decals.push(DecalGpu::default());
        }
        let decals_bytes: &[u8] = korppi_core::pod::cast_slice(&self.scratch_decals);
        device.write_buffer(self.decals_buf[slot].unwrap(), 0, decals_bytes)?;

        // 6. Upload probes using reusable scratch vector
        self.scratch_probes.clear();
        let prb_needed = probes.len().max(1);
        if self.scratch_probes.capacity() < prb_needed {
            self.scratch_probes
                .reserve(prb_needed - self.scratch_probes.capacity());
        }
        for prb in probes {
            let (shape_type, param) = match prb.shape {
                ProbeShape::Sphere { radius } => (0.0f32, [radius, 0.0, 0.0, 0.0]),
                ProbeShape::Box { half_extents } => (
                    1.0f32,
                    [half_extents.x, half_extents.y, half_extents.z, 0.0],
                ),
            };
            self.scratch_probes.push(ProbeGpu {
                position_type: [
                    prb.position_vs.x,
                    prb.position_vs.y,
                    prb.position_vs.z,
                    shape_type,
                ],
                param,
            });
        }
        if self.scratch_probes.is_empty() {
            self.scratch_probes.push(ProbeGpu::default());
        }
        let probes_bytes: &[u8] = korppi_core::pod::cast_slice(&self.scratch_probes);
        device.write_buffer(self.probes_buf[slot].unwrap(), 0, probes_bytes)?;

        // 7. Reset atomic counters to 0
        let zero_counters = [0u32; 4];
        let counter_bytes: &[u8] = korppi_core::pod::cast_slice(&zero_counters);
        device.write_buffer(self.counter_buf[slot].unwrap(), 0, counter_bytes)?;

        // Bind and dispatch without heap allocation in steady state
        let bind_clusters = total_clusters.max(1) as usize;
        recorder.bind_pipeline(pipeline);
        let ranges: [(Option<BufferHandle>, usize); 10] = [
            (
                self.froxels_buf[slot],
                bind_clusters * size_of::<FroxelAabbGpu>(),
            ),
            (
                self.points_buf[slot],
                self.allocated_points[slot] * size_of::<PointLightGpu>(),
            ),
            (
                self.spots_buf[slot],
                self.allocated_spots[slot] * size_of::<SpotLightGpu>(),
            ),
            (
                self.records_buf[slot],
                bind_clusters * size_of::<ClusterRecord>(),
            ),
            (
                self.indices_buf[slot],
                max_light_indices as usize * size_of::<u32>(),
            ),
            (self.counter_buf[slot], 16),
            (
                self.decals_buf[slot],
                self.allocated_decals[slot] * size_of::<DecalGpu>(),
            ),
            (
                self.probes_buf[slot],
                self.allocated_probes[slot] * size_of::<ProbeGpu>(),
            ),
            (
                self.decal_indices_buf[slot],
                max_decal_indices as usize * size_of::<u32>(),
            ),
            (
                self.probe_indices_buf[slot],
                max_probe_indices as usize * size_of::<u32>(),
            ),
        ];
        let entries: [BindGroupEntry; 10] = std::array::from_fn(|binding| {
            let (buffer, size) = ranges[binding];
            BindGroupEntry {
                binding: binding as u32,
                resource: BindingResource::Buffer {
                    buffer: buffer.expect("allocated by ensure_buffers"),
                    offset: 0,
                    size: size as u64,
                },
            }
        });
        let buffers_group = self.buffers_group(device, slot, entries)?;
        let params_group = self.params_group[slot].ok_or_else(|| {
            RhiError::ValidationError("GpuLightCuller parameters group missing".to_string())
        })?;
        recorder.set_bind_group(0, params_group, &[]);
        recorder.set_bind_group(1, buffers_group, &[]);

        let group_count_x = total_clusters.div_ceil(64);
        recorder.dispatch(group_count_x, 1, 1);

        Ok(GpuCullingBuffers {
            records_buffer: self.records_buf[slot].unwrap(),
            records_size: (bind_clusters * size_of::<ClusterRecord>()) as u64,
            light_indices_buffer: self.indices_buf[slot].unwrap(),
            light_indices_size: (max_light_indices as usize * size_of::<u32>()) as u64,
            decal_indices_buffer: self.decal_indices_buf[slot].unwrap(),
            probe_indices_buffer: self.probe_indices_buf[slot].unwrap(),
            points_buffer: self.points_buf[slot].unwrap(),
            points_size: (self.allocated_points[slot] * size_of::<PointLightGpu>()) as u64,
            spots_buffer: self.spots_buf[slot].unwrap(),
            spots_size: (self.allocated_spots[slot] * size_of::<SpotLightGpu>()) as u64,
            total_clusters,
        })
    }

    /// Uploads feature data to persistent GPU buffers (slot 0) and records compute dispatch commands into `recorder`.
    ///
    /// This method performs no CPU stalls and no GPU readback, piping SSBO handles directly to subsequent passes.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_culling_async(
        &mut self,
        device: &dyn GraphicsDevice,
        recorder: &mut dyn CommandBufferRecorder,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
        decals: &[Decal],
        probes: &[ReflectionProbe],
    ) -> Result<GpuCullingBuffers, RhiError> {
        self.dispatch_culling_for_frame(
            device,
            recorder,
            FrameIndex::new(0),
            froxels,
            point_lights,
            spot_lights,
            decals,
            probes,
        )
    }

    /// Dispatches compute culling across froxels and reads back culled feature records to CPU.
    pub fn cull_features(
        &mut self,
        device: &dyn GraphicsDevice,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
        decals: &[Decal],
        probes: &[ReflectionProbe],
    ) -> Result<CullingOutput, RhiError> {
        let mut recorder = device.create_command_recorder()?;
        recorder.begin()?;

        let buffers = self.dispatch_culling_for_frame(
            device,
            &mut *recorder,
            FrameIndex::new(0),
            froxels,
            point_lights,
            spot_lights,
            decals,
            probes,
        )?;

        let cmd = recorder.finish()?;
        device.submit(cmd)?;
        device.wait_idle()?;

        let total_clusters = buffers.total_clusters;
        let records_size = total_clusters as usize * std::mem::size_of::<ClusterRecord>();
        let records_bytes = if records_size > 0 {
            device.read_buffer(self.records_buf[0].unwrap(), 0, records_size as u64)?
        } else {
            Vec::new()
        };
        let counter_bytes = device.read_buffer(self.counter_buf[0].unwrap(), 0, 16)?;

        let total_lights = u32::from_ne_bytes(counter_bytes[0..4].try_into().unwrap())
            .min(self.max_light_indices[0]);
        let total_decals = u32::from_ne_bytes(counter_bytes[4..8].try_into().unwrap())
            .min(self.max_decal_indices[0]);
        let total_probes = u32::from_ne_bytes(counter_bytes[8..12].try_into().unwrap())
            .min(self.max_probe_indices[0]);

        let mut global_light_indices = Vec::with_capacity(total_lights as usize);
        if total_lights > 0 {
            let indices_bytes =
                device.read_buffer(self.indices_buf[0].unwrap(), 0, (total_lights * 4) as u64)?;
            for chunk in indices_bytes.as_chunks::<4>().0 {
                global_light_indices.push(u32::from_ne_bytes(*chunk));
            }
        }

        let mut global_decal_indices = Vec::with_capacity(total_decals as usize);
        if total_decals > 0 {
            let indices_bytes = device.read_buffer(
                self.decal_indices_buf[0].unwrap(),
                0,
                (total_decals * 4) as u64,
            )?;
            for chunk in indices_bytes.as_chunks::<4>().0 {
                global_decal_indices.push(u32::from_ne_bytes(*chunk));
            }
        }

        let mut global_probe_indices = Vec::with_capacity(total_probes as usize);
        if total_probes > 0 {
            let indices_bytes = device.read_buffer(
                self.probe_indices_buf[0].unwrap(),
                0,
                (total_probes * 4) as u64,
            )?;
            for chunk in indices_bytes.as_chunks::<4>().0 {
                global_probe_indices.push(u32::from_ne_bytes(*chunk));
            }
        }

        let mut cluster_records = Vec::with_capacity(total_clusters as usize);
        for chunk in records_bytes
            .as_chunks::<{ std::mem::size_of::<ClusterRecord>() }>()
            .0
        {
            let record: ClusterRecord = korppi_core::pod::pod_read_unaligned(chunk);
            cluster_records.push(record);
        }

        Ok(CullingOutput {
            cluster_records,
            global_light_indices,
            global_decal_indices,
            global_probe_indices,
        })
    }

    /// Dispatches compute light culling and reads back results (convenience method without decals/probes).
    #[inline]
    pub fn cull_lights(
        &mut self,
        device: &dyn GraphicsDevice,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
    ) -> Result<CullingOutput, RhiError> {
        self.cull_features(device, froxels, point_lights, spot_lights, &[], &[])
    }

    /// The set 1 group binding `entries` for `slot`, created again only when buffers or ranges changed.
    fn buffers_group(
        &mut self,
        device: &dyn GraphicsDevice,
        slot: usize,
        entries: [BindGroupEntry; 10],
    ) -> Result<BindGroupHandle, RhiError> {
        if let Some((group, current)) = &self.buffers_group[slot] {
            if *current == entries {
                return Ok(*group);
            }
        }
        if let Some((old, _)) = self.buffers_group[slot].take() {
            device.destroy_bind_group(old)?;
        }
        let [_, layout] = self.layouts.ok_or_else(|| {
            RhiError::ValidationError("GpuLightCuller pipeline not initialized".to_string())
        })?;
        let group = device.create_bind_group(&BindGroupDescriptor {
            layout,
            entries: entries.to_vec(),
            label: Some("clustered_light_cull_buffers"),
        })?;
        self.buffers_group[slot] = Some((group, entries));
        Ok(group)
    }

    /// Destroys all persistent GPU pipeline and buffer resources.
    pub fn destroy(&mut self, device: &dyn GraphicsDevice) -> Result<(), RhiError> {
        for slot in 0..FRAMES_IN_FLIGHT {
            if let Some((group, _)) = self.buffers_group[slot].take() {
                device.destroy_bind_group(group)?;
            }
            if let Some(group) = self.params_group[slot].take() {
                device.destroy_bind_group(group)?;
            }
            if let Some(buf) = self.ubo[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.froxels_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.points_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.spots_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.decals_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.probes_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.counter_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.records_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.indices_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.decal_indices_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            if let Some(buf) = self.probe_indices_buf[slot].take() {
                device.destroy_buffer(buf)?;
            }
            self.allocated_froxels[slot] = 0;
            self.allocated_points[slot] = 0;
            self.allocated_spots[slot] = 0;
            self.allocated_decals[slot] = 0;
            self.allocated_probes[slot] = 0;
            self.allocated_clusters[slot] = 0;
            self.max_light_indices[slot] = 0;
            self.max_decal_indices[slot] = 0;
            self.max_probe_indices[slot] = 0;
        }
        for layout in self.layouts.take().into_iter().flatten() {
            device.destroy_bind_group_layout(layout)?;
        }
        if let Some(pipe) = self.pipeline.take() {
            device.destroy_pipeline(pipe)?;
        }
        if let Some(shader) = self.shader.take() {
            device.destroy_shader(shader)?;
        }
        Ok(())
    }
}
