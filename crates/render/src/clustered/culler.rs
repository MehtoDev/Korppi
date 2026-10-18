//! CPU clustered light, decal, and reflection probe culler and GPU buffer packing structures.

use crate::clustered::decal::{test_decal_vs_aabb, Decal};
use crate::clustered::grid::{ClusterGridConfig, FroxelAabb};
use crate::clustered::light::{test_cone_vs_sphere, test_sphere_vs_aabb, PointLight, SpotLight};
use crate::clustered::probe::{test_probe_vs_aabb, ReflectionProbe};

/// Bookkeeping record storing offsets and counts into the global entity index buffers for a froxel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterRecord {
    /// Starting offset in the global index buffer for point light indices.
    pub point_offset: u32,
    /// Number of active point lights intersecting this cluster.
    pub point_count: u32,
    /// Starting offset in the global index buffer for spot light indices.
    pub spot_offset: u32,
    /// Number of active spot lights intersecting this cluster.
    pub spot_count: u32,
    /// Starting offset in the global index buffer for decal indices.
    pub decal_offset: u32,
    /// Number of active decals intersecting this cluster.
    pub decal_count: u32,
    /// Starting offset in the global index buffer for reflection probe indices.
    pub probe_offset: u32,
    /// Number of active reflection probes intersecting this cluster.
    pub probe_count: u32,
}

gpu_pod!(ClusterRecord = 8 * 4);

/// Maximum number of lights, decals, or probes kept per froxel on both CPU and GPU (`RND-12`).
pub const MAX_FEATURES_PER_FROXEL: usize = 256;

#[derive(Default)]
struct CullingScratchPool {
    records: Option<Vec<ClusterRecord>>,
    light_indices: Option<Vec<u32>>,
    decal_indices: Option<Vec<u32>>,
    probe_indices: Option<Vec<u32>>,
}

thread_local! {
    static CULLING_SCRATCH: std::cell::RefCell<CullingScratchPool> =
        std::cell::RefCell::new(CullingScratchPool::default());
}

/// Output produced by the clustered culling stage.
#[derive(Clone, Debug, Default)]
pub struct CullingOutput {
    /// Array of cluster records indexed by flat cluster index.
    pub cluster_records: Vec<ClusterRecord>,
    /// Compact linear list of light indices referenced by cluster records.
    pub global_light_indices: Vec<u32>,
    /// Compact linear list of decal indices referenced by cluster records.
    pub global_decal_indices: Vec<u32>,
    /// Compact linear list of reflection probe indices referenced by cluster records.
    pub global_probe_indices: Vec<u32>,
}

impl CullingOutput {
    fn from_thread_local() -> Self {
        CULLING_SCRATCH.with(|cell| {
            let mut pool = cell.borrow_mut();
            let mut cluster_records = pool.records.take().unwrap_or_default();
            cluster_records.clear();
            let mut global_light_indices = pool.light_indices.take().unwrap_or_default();
            global_light_indices.clear();
            let mut global_decal_indices = pool.decal_indices.take().unwrap_or_default();
            global_decal_indices.clear();
            let mut global_probe_indices = pool.probe_indices.take().unwrap_or_default();
            global_probe_indices.clear();
            Self {
                cluster_records,
                global_light_indices,
                global_decal_indices,
                global_probe_indices,
            }
        })
    }

    /// Clears all output vectors while preserving their allocated capacities.
    pub fn clear(&mut self) {
        self.cluster_records.clear();
        self.global_light_indices.clear();
        self.global_decal_indices.clear();
        self.global_probe_indices.clear();
    }
}

impl Drop for CullingOutput {
    fn drop(&mut self) {
        let records = std::mem::take(&mut self.cluster_records);
        let lights = std::mem::take(&mut self.global_light_indices);
        let decals = std::mem::take(&mut self.global_decal_indices);
        let probes = std::mem::take(&mut self.global_probe_indices);
        CULLING_SCRATCH.with(|cell| {
            let mut pool = cell.borrow_mut();
            if records.capacity() > 0 && pool.records.is_none() {
                pool.records = Some(records);
            }
            if lights.capacity() > 0 && pool.light_indices.is_none() {
                pool.light_indices = Some(lights);
            }
            if decals.capacity() > 0 && pool.decal_indices.is_none() {
                pool.decal_indices = Some(decals);
            }
            if probes.capacity() > 0 && pool.probe_indices.is_none() {
                pool.probe_indices = Some(probes);
            }
        });
    }
}

/// CPU-side culler for assigning lights, decals, and reflection probes into active 3D froxel cells.
pub struct ClusteredLightCuller {
    total_clusters: usize,
    visible_points: Vec<u32>,
    visible_spots: Vec<u32>,
    visible_decals: Vec<u32>,
    visible_probes: Vec<u32>,
}

impl ClusteredLightCuller {
    /// Creates a new culler for the given grid configuration.
    pub fn new(config: &ClusterGridConfig) -> Self {
        Self {
            total_clusters: config.total_clusters() as usize,
            visible_points: Vec::with_capacity(MAX_FEATURES_PER_FROXEL),
            visible_spots: Vec::with_capacity(MAX_FEATURES_PER_FROXEL),
            visible_decals: Vec::with_capacity(MAX_FEATURES_PER_FROXEL),
            visible_probes: Vec::with_capacity(MAX_FEATURES_PER_FROXEL),
        }
    }

    /// Culls the provided point lights, spotlights, decals, and reflection probes into `out`
    /// without heap allocation once `out` and the culler's scratch buffers are warmed up (`RND-11`).
    pub fn cull_features_into(
        &mut self,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
        decals: &[Decal],
        probes: &[ReflectionProbe],
        out: &mut CullingOutput,
    ) {
        assert_eq!(
            froxels.len(),
            self.total_clusters,
            "Froxel count must match total cluster count"
        );

        out.clear();
        out.cluster_records
            .resize(self.total_clusters, ClusterRecord::default());

        for (cluster_idx, froxel) in froxels.iter().enumerate() {
            self.visible_points.clear();
            self.visible_spots.clear();
            self.visible_decals.clear();
            self.visible_probes.clear();

            for (light_idx, point_light) in point_lights.iter().enumerate() {
                if self.visible_points.len() >= MAX_FEATURES_PER_FROXEL {
                    break;
                }
                if test_sphere_vs_aabb(
                    point_light.position_vs,
                    point_light.radius,
                    froxel.aabb_min,
                    froxel.aabb_max,
                ) {
                    self.visible_points.push(light_idx as u32);
                }
            }

            for (light_idx, spot_light) in spot_lights.iter().enumerate() {
                if self.visible_spots.len() >= MAX_FEATURES_PER_FROXEL {
                    break;
                }
                if test_cone_vs_sphere(
                    spot_light.position_vs,
                    spot_light.direction_vs,
                    spot_light.range,
                    spot_light.half_angle,
                    froxel.sphere_center,
                    froxel.sphere_radius,
                ) {
                    self.visible_spots.push(light_idx as u32);
                }
            }

            for (decal_idx, decal) in decals.iter().enumerate() {
                if self.visible_decals.len() >= MAX_FEATURES_PER_FROXEL {
                    break;
                }
                if test_decal_vs_aabb(decal, froxel.aabb_min, froxel.aabb_max) {
                    self.visible_decals.push(decal_idx as u32);
                }
            }

            for (probe_idx, probe) in probes.iter().enumerate() {
                if self.visible_probes.len() >= MAX_FEATURES_PER_FROXEL {
                    break;
                }
                if test_probe_vs_aabb(probe, froxel.aabb_min, froxel.aabb_max) {
                    self.visible_probes.push(probe_idx as u32);
                }
            }

            let point_offset = out.global_light_indices.len() as u32;
            let point_count = self.visible_points.len() as u32;
            out.global_light_indices
                .extend_from_slice(&self.visible_points);

            let spot_offset = out.global_light_indices.len() as u32;
            let spot_count = self.visible_spots.len() as u32;
            out.global_light_indices
                .extend_from_slice(&self.visible_spots);

            let decal_offset = out.global_decal_indices.len() as u32;
            let decal_count = self.visible_decals.len() as u32;
            out.global_decal_indices
                .extend_from_slice(&self.visible_decals);

            let probe_offset = out.global_probe_indices.len() as u32;
            let probe_count = self.visible_probes.len() as u32;
            out.global_probe_indices
                .extend_from_slice(&self.visible_probes);

            out.cluster_records[cluster_idx] = ClusterRecord {
                point_offset,
                point_count,
                spot_offset,
                spot_count,
                decal_offset,
                decal_count,
                probe_offset,
                probe_count,
            };
        }
    }

    /// Culls the provided point and spot lights into `out` without heap allocation when `out` has capacity.
    #[inline]
    pub fn cull_lights_into(
        &mut self,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
        out: &mut CullingOutput,
    ) {
        self.cull_features_into(froxels, point_lights, spot_lights, &[], &[], out);
    }

    /// Culls the provided point lights, spotlights, decals, and reflection probes against view-space froxels.
    pub fn cull_features(
        &mut self,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
        decals: &[Decal],
        probes: &[ReflectionProbe],
    ) -> CullingOutput {
        let mut out = CullingOutput::from_thread_local();
        self.cull_features_into(froxels, point_lights, spot_lights, decals, probes, &mut out);
        out
    }

    /// Culls the provided point and spot lights against precomputed view-space froxel AABBs.
    #[inline]
    pub fn cull_lights(
        &mut self,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
    ) -> CullingOutput {
        self.cull_features(froxels, point_lights, spot_lights, &[], &[])
    }

    /// Dispatches GPU compute light culling when a device and GPU pipeline are available;
    /// otherwise seamlessly falls back to CPU culling.
    pub fn cull_lights_gpu_or_fallback(
        &mut self,
        device: Option<&dyn korppi_rhi::GraphicsDevice>,
        gpu_culler: Option<&mut crate::clustered::gpu_culler::GpuLightCuller>,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
    ) -> CullingOutput {
        self.cull_features_gpu_or_fallback(
            device,
            gpu_culler,
            froxels,
            point_lights,
            spot_lights,
            &[],
            &[],
        )
    }

    /// Dispatches GPU compute feature culling (lights, decals, probes) when available;
    /// otherwise seamlessly falls back to CPU culling.
    #[allow(clippy::too_many_arguments)]
    pub fn cull_features_gpu_or_fallback(
        &mut self,
        device: Option<&dyn korppi_rhi::GraphicsDevice>,
        gpu_culler: Option<&mut crate::clustered::gpu_culler::GpuLightCuller>,
        froxels: &[FroxelAabb],
        point_lights: &[PointLight],
        spot_lights: &[SpotLight],
        decals: &[Decal],
        probes: &[ReflectionProbe],
    ) -> CullingOutput {
        if let (Some(dev), Some(gpu)) = (device, gpu_culler) {
            if let Ok(gpu_out) =
                gpu.cull_features(dev, froxels, point_lights, spot_lights, decals, probes)
            {
                return gpu_out;
            }
        }
        self.cull_features(froxels, point_lights, spot_lights, decals, probes)
    }
}
