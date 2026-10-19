//! Clustered frustum slicing and view-space Froxel AABB grid generation (`RND-08`, `RND-15`).
//!
//! # Screen-Space Coordinate Convention
//! Following the engine's RHI coordinate contract (`docs/architecture/coordinate-systems.md`),
//! pixel `(0, 0)` is the **top-left** of the viewport (`ndc_x = -1.0, ndc_y = +1.0`), and pixel
//! `(screen_width, screen_height)` is the **bottom-right** (`ndc_x = +1.0, ndc_y = -1.0`).
//! Tile row `y = 0` therefore corresponds to the top of the screen (`+Y` in right-handed view space).

use korppi_math::{Mat4, Vec2, Vec3, Vec4};

/// Spatial configuration and dimensions for the 3D clustered froxel grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterGridConfig {
    screen_width: u32,
    screen_height: u32,
    tile_size: u32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    z_near: f32,
    z_far: f32,
    scale_z: f32,
    bias_z: f32,
}

impl ClusterGridConfig {
    /// Creates a new clustered grid configuration with exponential depth slicing.
    pub fn new(
        screen_width: u32,
        screen_height: u32,
        tile_size: u32,
        dim_z: u32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        assert!(tile_size > 0, "Tile size must be greater than zero");
        assert!(dim_z > 0, "Depth slice count must be greater than zero");
        assert!(z_near > 0.0, "z_near must be positive and non-zero");
        assert!(z_far > z_near, "z_far must be strictly greater than z_near");

        let screen_width = screen_width.max(1);
        let screen_height = screen_height.max(1);
        let dim_x = screen_width.div_ceil(tile_size).max(1);
        let dim_y = screen_height.div_ceil(tile_size).max(1);

        // Exponential depth slicing constants:
        // slice = floor(log2(depth / z_near) / log2(z_far / z_near) * dim_z)
        //       = floor(log2(depth) * scale_z + bias_z)
        let log_ratio = (z_far / z_near).log2();
        let scale_z = (dim_z as f32) / log_ratio;
        let bias_z = -scale_z * z_near.log2();

        Self {
            screen_width,
            screen_height,
            tile_size,
            dim_x,
            dim_y,
            dim_z,
            z_near,
            z_far,
            scale_z,
            bias_z,
        }
    }

    /// Screen width in pixels.
    #[inline]
    pub fn screen_width(&self) -> u32 {
        self.screen_width
    }

    /// Screen height in pixels.
    #[inline]
    pub fn screen_height(&self) -> u32 {
        self.screen_height
    }

    /// Screen-space tile dimension in pixels.
    #[inline]
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// Number of clusters along the horizontal screen axis.
    #[inline]
    pub fn dim_x(&self) -> u32 {
        self.dim_x
    }

    /// Number of clusters along the vertical screen axis.
    #[inline]
    pub fn dim_y(&self) -> u32 {
        self.dim_y
    }

    /// Number of exponential depth slices along the camera view axis.
    #[inline]
    pub fn dim_z(&self) -> u32 {
        self.dim_z
    }

    /// Total number of 3D froxels ($N_x \times N_y \times N_z$).
    #[inline]
    pub fn total_clusters(&self) -> u32 {
        self.dim_x * self.dim_y * self.dim_z
    }

    /// Near plane depth distance in view space.
    #[inline]
    pub fn z_near(&self) -> f32 {
        self.z_near
    }

    /// Far plane depth distance in view space.
    #[inline]
    pub fn z_far(&self) -> f32 {
        self.z_far
    }

    /// Computes the exponential depth slice index $i_z \in [0, N_z - 1]$ for a given view depth.
    #[inline]
    pub fn compute_slice_z(&self, view_depth: f32) -> u32 {
        if view_depth <= self.z_near {
            return 0;
        }
        if view_depth >= self.z_far {
            return self.dim_z - 1;
        }

        let slice = view_depth.log2() * self.scale_z + self.bias_z;
        let slice_clamped = slice.floor().max(0.0) as u32;
        slice_clamped.min(self.dim_z - 1)
    }

    /// Computes the flat 1D cluster index for a top-down pixel position (`(0, 0)` = top-left) and view depth.
    #[inline]
    pub fn compute_cluster_index(&self, screen_pos: Vec2, view_depth: f32) -> u32 {
        let x_tile =
            ((screen_pos.x / self.tile_size as f32).floor().max(0.0) as u32).min(self.dim_x - 1);
        let y_tile =
            ((screen_pos.y / self.tile_size as f32).floor().max(0.0) as u32).min(self.dim_y - 1);
        let z_slice = self.compute_slice_z(view_depth);

        x_tile + y_tile * self.dim_x + z_slice * (self.dim_x * self.dim_y)
    }

    /// Constructs view-space bounding volumes (AABB and circumscribed sphere) for all froxels
    /// into `out` without allocating when `out` has capacity (`RND-08`, `RND-15`).
    pub fn build_froxel_aabbs_into(&self, inverse_projection: Mat4, out: &mut Vec<FroxelAabb>) {
        let total = self.total_clusters() as usize;
        out.clear();
        out.reserve(total);

        let sw = self.screen_width as f32;
        let sh = self.screen_height as f32;
        let ts = self.tile_size as f32;
        let log_ratio = (self.z_far / self.z_near).ln();

        // Unproject an NDC (x, y) line at two clip depths and interpolate linearly in view space
        // to target view-space z = -z_depth. This works identically for perspective and orthographic
        // projections (both standard-Z and reverse-Z; RND-15).
        let unproject_ndc_at_depth = |ndc_x: f32, ndc_y: f32, z_depth: f32| -> Vec3 {
            let clip_a = inverse_projection * Vec4::new(ndc_x, ndc_y, 0.25, 1.0);
            let clip_b = inverse_projection * Vec4::new(ndc_x, ndc_y, 0.75, 1.0);
            let va = Vec3::new(
                clip_a.x / clip_a.w,
                clip_a.y / clip_a.w,
                clip_a.z / clip_a.w,
            );
            let vb = Vec3::new(
                clip_b.x / clip_b.w,
                clip_b.y / clip_b.w,
                clip_b.z / clip_b.w,
            );
            let dz = vb.z - va.z;
            if dz.abs() > 1e-6 {
                let t = (-z_depth - va.z) / dz;
                va + (vb - va) * t
            } else {
                Vec3::new(va.x, va.y, -z_depth)
            }
        };

        for z in 0..self.dim_z {
            let z_min = self.z_near * (log_ratio * (z as f32 / self.dim_z as f32)).exp();
            let z_max = self.z_near * (log_ratio * ((z + 1) as f32 / self.dim_z as f32)).exp();

            for y in 0..self.dim_y {
                let px_y0 = (y as f32 * ts).min(sh);
                let px_y1 = ((y + 1) as f32 * ts).min(sh);

                // Top-down pixel rows: row 0 -> ndc_y = +1.0 (top), row height -> ndc_y = -1.0 (bottom) (RND-08)
                let ndc_y0 = 1.0 - (px_y0 / sh) * 2.0;
                let ndc_y1 = 1.0 - (px_y1 / sh) * 2.0;

                for x in 0..self.dim_x {
                    let px_x0 = (x as f32 * ts).min(sw);
                    let px_x1 = ((x + 1) as f32 * ts).min(sw);

                    let ndc_x0 = (px_x0 / sw) * 2.0 - 1.0;
                    let ndc_x1 = (px_x1 / sw) * 2.0 - 1.0;

                    let corners = [
                        unproject_ndc_at_depth(ndc_x0, ndc_y0, z_min),
                        unproject_ndc_at_depth(ndc_x1, ndc_y0, z_min),
                        unproject_ndc_at_depth(ndc_x0, ndc_y1, z_min),
                        unproject_ndc_at_depth(ndc_x1, ndc_y1, z_min),
                        unproject_ndc_at_depth(ndc_x0, ndc_y0, z_max),
                        unproject_ndc_at_depth(ndc_x1, ndc_y0, z_max),
                        unproject_ndc_at_depth(ndc_x0, ndc_y1, z_max),
                        unproject_ndc_at_depth(ndc_x1, ndc_y1, z_max),
                    ];

                    let mut aabb_min = Vec3::splat(f32::INFINITY);
                    let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

                    for c in &corners {
                        aabb_min = aabb_min.min(*c);
                        aabb_max = aabb_max.max(*c);
                    }

                    let sphere_center = (aabb_min + aabb_max) * 0.5;
                    let half_extents = (aabb_max - aabb_min) * 0.5;
                    let sphere_radius = half_extents.length();

                    out.push(FroxelAabb {
                        aabb_min,
                        aabb_max,
                        sphere_center,
                        sphere_radius,
                    });
                }
            }
        }
    }

    /// Constructs view-space bounding volumes (AABB and circumscribed sphere) for all froxels.
    pub fn build_froxel_aabbs(&self, inverse_projection: Mat4) -> Vec<FroxelAabb> {
        let mut froxels = Vec::with_capacity(self.total_clusters() as usize);
        self.build_froxel_aabbs_into(inverse_projection, &mut froxels);
        froxels
    }
}

/// View-space bounding volume for an individual froxel sub-volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FroxelAabb {
    /// Axis-Aligned Bounding Box minimum corner in view space.
    pub aabb_min: Vec3,
    /// Axis-Aligned Bounding Box maximum corner in view space.
    pub aabb_max: Vec3,
    /// Circumscribed bounding sphere center in view space.
    pub sphere_center: Vec3,
    /// Circumscribed bounding sphere radius in view space.
    pub sphere_radius: f32,
}
