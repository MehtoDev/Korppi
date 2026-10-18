//! Cascaded Shadow Maps (CSM), Practical Split Scheme, and Texel Snapping (`RND-07`).

use korppi_math::{Mat4, Vec3, Vec4};
use korppi_rhi::SmallVec;

/// Configuration parameters for Cascaded Shadow Maps (CSM).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CascadeConfig {
    /// Number of shadow cascades (typically 3 or 4).
    pub cascade_count: usize,
    /// Split distribution blending factor: 1.0 = logarithmic, 0.0 = uniform.
    pub split_lambda: f32,
    /// Shadow map resolution per cascade in texels (e.g. 1024, 2048).
    pub shadow_map_resolution: u32,
    /// Camera near plane distance.
    pub near_plane: f32,
    /// Camera far plane distance for shadow casting.
    pub far_plane: f32,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            cascade_count: 4,
            split_lambda: 0.85,
            shadow_map_resolution: 2048,
            near_plane: 0.5,
            far_plane: 100.0,
        }
    }
}

impl CascadeConfig {
    /// Computes the cascade split depths using the Practical Split Scheme (PSSM).
    ///
    /// Blends logarithmic and uniform partition functions:
    /// $$z_{split}(i) = \lambda \cdot z_{log}(i) + (1 - \lambda) \cdot z_{uniform}(i)$$
    pub fn compute_split_depths(&self) -> Vec<f32> {
        let mut splits = Vec::with_capacity(self.cascade_count + 1);
        self.compute_split_depths_into(&mut splits);
        splits
    }

    /// Computes cascade split depths into `out` without heap allocation when `out` has capacity.
    pub fn compute_split_depths_into(&self, out: &mut Vec<f32>) {
        out.clear();
        let n = self.cascade_count as f32;
        out.push(self.near_plane);

        let log_ratio = self.far_plane / self.near_plane;

        for i in 1..self.cascade_count {
            let frac = i as f32 / n;
            let z_log = self.near_plane * log_ratio.powf(frac);
            let z_uniform = self.near_plane + (self.far_plane - self.near_plane) * frac;
            let z_split = self.split_lambda * z_log + (1.0 - self.split_lambda) * z_uniform;
            out.push(z_split);
        }

        out.push(self.far_plane);
    }
}

/// A computed shadow cascade with light projection matrix and texel metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cascade {
    /// Combined view-projection matrix for rendering the cascade depth map.
    pub view_proj_matrix: Mat4,
    /// Far split depth distance in view space for this cascade.
    pub split_depth: f32,
    /// World-space footprint of a single shadow map texel.
    pub texel_size_world: f32,
    /// World-space center of the bounding sphere enclosing this cascade partition.
    pub sphere_center: Vec3,
    /// World-space radius of the enclosing bounding sphere.
    pub sphere_radius: f32,
    /// Snapped center in fixed-origin light view space to prevent sub-texel edge shimmering (`RND-07`).
    pub snapped_light_center: Vec3,
}

/// Computes stabilized Cascaded Shadow Maps into `out` without allocating when `out` has capacity.
pub fn compute_cascades_into(
    camera_view: Mat4,
    camera_proj: Mat4,
    light_direction: Vec3,
    config: &CascadeConfig,
    out: &mut Vec<Cascade>,
) {
    out.clear();
    let mut splits: SmallVec<[f32; 8]> = SmallVec::new();
    let n = config.cascade_count as f32;
    splits.push(config.near_plane);
    let log_ratio = config.far_plane / config.near_plane;
    for i in 1..config.cascade_count {
        let frac = i as f32 / n;
        let z_log = config.near_plane * log_ratio.powf(frac);
        let z_uniform = config.near_plane + (config.far_plane - config.near_plane) * frac;
        splits.push(config.split_lambda * z_log + (1.0 - config.split_lambda) * z_uniform);
    }
    splits.push(config.far_plane);

    let light_dir = light_direction.normalize();
    let inv_cam_view = camera_view.inverse();
    let inv_cam_proj = camera_proj.inverse();

    let unproject_ndc_to_view = |ndc: Vec4| -> Vec3 {
        let view_space = inv_cam_proj * ndc;
        Vec3::new(
            view_space.x / view_space.w,
            view_space.y / view_space.w,
            view_space.z / view_space.w,
        )
    };

    // Fixed-origin light view matrix so light-space coordinates of static world points
    // do not drift when the camera moves (RND-07).
    let up = if light_dir.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let base_light_view = Mat4::look_to_rh(Vec3::ZERO, light_dir, up);
    let inv_base_light_view = base_light_view.inverse();
    let res = config.shadow_map_resolution.max(1) as f32;

    for i in 0..config.cascade_count {
        let z_near = splits[i];
        let z_far = splits[i + 1];

        let clip_near = camera_proj * Vec4::new(0.0, 0.0, -z_near, 1.0);
        let clip_far = camera_proj * Vec4::new(0.0, 0.0, -z_far, 1.0);
        let ndc_near_z = clip_near.z / clip_near.w;
        let ndc_far_z = clip_far.z / clip_far.w;

        // Compute view-space frustum corners first so the bounding sphere radius is strictly
        // invariant to camera translation and rotation.
        let view_corners = [
            unproject_ndc_to_view(Vec4::new(-1.0, -1.0, ndc_near_z, 1.0)),
            unproject_ndc_to_view(Vec4::new(1.0, -1.0, ndc_near_z, 1.0)),
            unproject_ndc_to_view(Vec4::new(-1.0, 1.0, ndc_near_z, 1.0)),
            unproject_ndc_to_view(Vec4::new(1.0, 1.0, ndc_near_z, 1.0)),
            unproject_ndc_to_view(Vec4::new(-1.0, -1.0, ndc_far_z, 1.0)),
            unproject_ndc_to_view(Vec4::new(1.0, -1.0, ndc_far_z, 1.0)),
            unproject_ndc_to_view(Vec4::new(-1.0, 1.0, ndc_far_z, 1.0)),
            unproject_ndc_to_view(Vec4::new(1.0, 1.0, ndc_far_z, 1.0)),
        ];

        let mut view_center = Vec3::ZERO;
        for c in &view_corners {
            view_center += *c;
        }
        view_center /= 8.0;

        let mut sphere_radius = 0.0f32;
        for c in &view_corners {
            sphere_radius = sphere_radius.max((view_center - *c).length());
        }
        // Quantize sphere radius to 1/16m increments for extra numerical stability
        sphere_radius = (sphere_radius * 16.0).ceil() / 16.0;

        let world_center4 =
            inv_cam_view * Vec4::new(view_center.x, view_center.y, view_center.z, 1.0);
        let sphere_center = Vec3::new(world_center4.x, world_center4.y, world_center4.z);

        let texel_size_world = (2.0 * sphere_radius) / res;

        // Transform cascade sphere center into fixed-origin light space and snap X/Y to texel increments
        let center_in_light =
            base_light_view * Vec4::new(sphere_center.x, sphere_center.y, sphere_center.z, 1.0);
        let snapped_x = (center_in_light.x / texel_size_world).floor() * texel_size_world;
        let snapped_y = (center_in_light.y / texel_size_world).floor() * texel_size_world;
        let snapped_z = center_in_light.z;
        let snapped_light_center = Vec3::new(snapped_x, snapped_y, snapped_z);

        let snapped_world4 = inv_base_light_view * Vec4::new(snapped_x, snapped_y, snapped_z, 1.0);
        let snapped_sphere_center = Vec3::new(snapped_world4.x, snapped_world4.y, snapped_world4.z);

        // Build orthographic light view-projection directly in fixed-origin light space around snapped center
        let r = sphere_radius;
        let z_mult = 4.0;
        let light_translate =
            Mat4::from_translation(Vec3::new(-snapped_x, -snapped_y, -(snapped_z + r * 2.0)));
        let light_view = light_translate * base_light_view;
        let light_proj = Mat4::orthographic_rh(-r, r, -r, r, 0.0, r * z_mult);

        out.push(Cascade {
            view_proj_matrix: light_proj * light_view,
            split_depth: z_far,
            texel_size_world,
            sphere_center: snapped_sphere_center,
            sphere_radius,
            snapped_light_center,
        });
    }
}

/// Computes stabilized Cascaded Shadow Maps for the camera frustum and light direction (`RND-07`).
pub fn compute_cascades(
    camera_view: Mat4,
    camera_proj: Mat4,
    light_direction: Vec3,
    config: &CascadeConfig,
) -> Vec<Cascade> {
    let mut cascades = Vec::with_capacity(config.cascade_count);
    compute_cascades_into(
        camera_view,
        camera_proj,
        light_direction,
        config,
        &mut cascades,
    );
    cascades
}
