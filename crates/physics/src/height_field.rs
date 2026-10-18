//! Box3D's height field terrain (`height_field.c`, `b3HeightFieldData`).
//!
//! A [`HeightField3d`] represents terrain on a regular 2D grid of heights. Each grid cell has
//! two triangles. Edge flags (`Concave`, `Convex`, `Flat`, `Open`) are precomputed for internal
//! edges to prevent ghost bumps when sliding or rolling across cell boundaries. Queries walk
//! the grid in O(overlapping cells) time without allocations.

use crate::mesh::{
    bounds_triangle_overlap, intersect_ray_triangle, MeshRayHit, MeshTriangle, CONCAVE_EDGE1,
    CONCAVE_EDGE2, CONCAVE_EDGE3, INVERSE_CONCAVE_EDGE1, INVERSE_CONCAVE_EDGE2,
    INVERSE_CONCAVE_EDGE3,
};
use korppi_math::Vec3;
use serde::{Deserialize, Serialize};

/// The material index that represents a hole in the height field.
pub const HEIGHT_FIELD_HOLE: u8 = 0xFF;

/// Cosine of 5 degrees, used to classify edges as flat.
const COS_5_DEG: f32 = 0.9962;

/// Configuration options for building a [`HeightField3d`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HeightFieldOptions {
    /// Optional per-cell material indices; cells marked [`HEIGHT_FIELD_HOLE`] have no collision.
    pub materials: Option<Vec<u8>>,
    /// Winding order: false for counter-clockwise seen from above (+Y), true for clockwise.
    pub clockwise: bool,
}

/// A 3D terrain height field with Box3D's edge classification and grid queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeightField3d {
    cols: usize,
    rows: usize,
    scale: Vec3,
    heights: Vec<f32>,
    materials: Option<Vec<u8>>,
    flags: Vec<u8>,
    lower: Vec3,
    upper: Vec3,
    clockwise: bool,
}

impl HeightField3d {
    /// Builds a flat or contoured height field over `cols` by `rows` grid vertices.
    /// Returns `None` if `cols < 2`, `rows < 2`, or `heights.len() != cols * rows`.
    #[must_use]
    pub fn new(cols: usize, rows: usize, scale: Vec3, heights: &[f32]) -> Option<Self> {
        Self::with_options(cols, rows, scale, heights, HeightFieldOptions::default())
    }

    /// Builds a height field with custom options such as holes or winding order.
    #[must_use]
    pub fn with_options(
        cols: usize,
        rows: usize,
        scale: Vec3,
        heights: &[f32],
        options: HeightFieldOptions,
    ) -> Option<Self> {
        if cols < 2 || rows < 2 || heights.len() != cols * rows {
            return None;
        }
        if scale.x <= 0.0 || scale.z <= 0.0 {
            return None;
        }

        let cell_count = (cols - 1) * (rows - 1);
        if let Some(ref mats) = options.materials {
            if mats.len() != cell_count {
                return None;
            }
        }

        let mut min_h = f32::MAX;
        let mut max_h = -f32::MAX;
        for &h in heights {
            if !h.is_finite() {
                return None;
            }
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }

        let lower = Vec3::new(0.0, (min_h * scale.y).min(max_h * scale.y), 0.0);
        let upper = Vec3::new(
            (cols - 1) as f32 * scale.x,
            (min_h * scale.y).max(max_h * scale.y),
            (rows - 1) as f32 * scale.z,
        );

        let mut flags = vec![0u8; 2 * cell_count];
        let material_slice = options.materials.as_deref();

        for row in 0..rows - 1 {
            for col in 0..cols - 1 {
                let cell = row * (cols - 1) + col;
                if material_slice.is_some_and(|m| m[cell] == HEIGHT_FIELD_HOLE) {
                    continue;
                }

                let tri0 = 2 * cell;
                let tri1 = tri0 + 1;

                let (corners, [plane1, plane2]) = get_cell_geometry(col, row, cols, scale, heights);

                let mut flags1 = 0u8;
                let mut flags2 = 0u8;

                // Internal diagonal edge (edge 2 in Box3D: bit 1)
                let sep = plane1.separation(corners[3]); // corners[3] is vs1[0] (p22)
                let cos_angle = plane1.normal.dot(plane2.normal);
                if sep > 0.0 || cos_angle > COS_5_DEG {
                    flags1 |= CONCAVE_EDGE2;
                    flags2 |= CONCAVE_EDGE2;
                }
                if sep < 0.0 || cos_angle > COS_5_DEG {
                    flags1 |= INVERSE_CONCAVE_EDGE2;
                    flags2 |= INVERSE_CONCAVE_EDGE2;
                }

                // Top neighbour (row - 1) -> edge 3 in Box3D (bit 2)
                if row > 0 {
                    let top_cell = (row - 1) * (cols - 1) + col;
                    if !material_slice.is_some_and(|m| m[top_cell] == HEIGHT_FIELD_HOLE) {
                        let (top_corners, [_, top_plane2]) =
                            get_cell_geometry(col, row - 1, cols, scale, heights);
                        let sep = plane1.separation(top_corners[1]);
                        let cos_angle = plane1.normal.dot(top_plane2.normal);
                        if sep > 0.0 || cos_angle > COS_5_DEG {
                            flags1 |= CONCAVE_EDGE3;
                        }
                        if sep < 0.0 || cos_angle > COS_5_DEG {
                            flags1 |= INVERSE_CONCAVE_EDGE3;
                        }
                    }
                }

                // Bottom neighbour (row + 1) -> edge 3 in Box3D (bit 2)
                if row + 1 < rows - 1 {
                    let bot_cell = (row + 1) * (cols - 1) + col;
                    if !material_slice.is_some_and(|m| m[bot_cell] == HEIGHT_FIELD_HOLE) {
                        let (bot_corners, [bot_plane1, _]) =
                            get_cell_geometry(col, row + 1, cols, scale, heights);
                        let sep = plane2.separation(bot_corners[2]);
                        let cos_angle = plane2.normal.dot(bot_plane1.normal);
                        if sep > 0.0 || cos_angle > COS_5_DEG {
                            flags2 |= CONCAVE_EDGE3;
                        }
                        if sep < 0.0 || cos_angle > COS_5_DEG {
                            flags2 |= INVERSE_CONCAVE_EDGE3;
                        }
                    }
                }

                // Left neighbour (col - 1) -> edge 1 in Box3D (bit 0)
                if col > 0 {
                    let left_cell = row * (cols - 1) + col - 1;
                    if !material_slice.is_some_and(|m| m[left_cell] == HEIGHT_FIELD_HOLE) {
                        let (left_corners, [_, left_plane2]) =
                            get_cell_geometry(col - 1, row, cols, scale, heights);
                        let sep = plane1.separation(left_corners[2]);
                        let cos_angle = plane1.normal.dot(left_plane2.normal);
                        if sep > 0.0 || cos_angle > COS_5_DEG {
                            flags1 |= CONCAVE_EDGE1;
                        }
                        if sep < 0.0 || cos_angle > COS_5_DEG {
                            flags1 |= INVERSE_CONCAVE_EDGE1;
                        }
                    }
                }

                // Right neighbour (col + 1) -> edge 1 in Box3D (bit 0)
                if col + 1 < cols - 1 {
                    let right_cell = row * (cols - 1) + col + 1;
                    if !material_slice.is_some_and(|m| m[right_cell] == HEIGHT_FIELD_HOLE) {
                        let (right_corners, [right_plane1, _]) =
                            get_cell_geometry(col + 1, row, cols, scale, heights);
                        let sep = plane2.separation(right_corners[1]);
                        let cos_angle = plane2.normal.dot(right_plane1.normal);
                        if sep > 0.0 || cos_angle > COS_5_DEG {
                            flags2 |= CONCAVE_EDGE1;
                        }
                        if sep < 0.0 || cos_angle > COS_5_DEG {
                            flags2 |= INVERSE_CONCAVE_EDGE1;
                        }
                    }
                }

                if options.clockwise {
                    flags1 = swap_edges_1_and_3(flags1);
                    flags2 = swap_edges_1_and_3(flags2);
                }

                flags[tri0] = flags1;
                flags[tri1] = flags2;
            }
        }

        Some(Self {
            cols,
            rows,
            scale,
            heights: heights.to_vec(),
            materials: options.materials,
            flags,
            lower,
            upper,
            clockwise: options.clockwise,
        })
    }

    /// Creates a flat grid of heights at y = 0.
    #[must_use]
    pub fn grid(cols: usize, rows: usize, scale: Vec3) -> Option<Self> {
        let heights = vec![0.0f32; cols * rows];
        Self::new(cols, rows, scale, &heights)
    }

    /// Creates a sinusoidal wavy grid matching Box3D's `b3CreateWave`.
    #[must_use]
    pub fn wave(
        cols: usize,
        rows: usize,
        scale: Vec3,
        row_freq: f32,
        col_freq: f32,
    ) -> Option<Self> {
        let mut heights = Vec::with_capacity(cols * rows);
        let omega_z = 2.0 * std::f32::consts::PI * row_freq;
        let omega_x = 2.0 * std::f32::consts::PI * col_freq;

        for r in 0..rows {
            let row_h = (omega_z * r as f32).sin();
            for c in 0..cols {
                let col_h = (omega_x * c as f32).sin();
                heights.push(row_h * col_h);
            }
        }
        Self::new(cols, rows, scale, &heights)
    }

    /// Number of grid columns along local X.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of grid rows along local Z.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Scaling along local axes.
    #[must_use]
    pub fn scale(&self) -> Vec3 {
        self.scale
    }

    /// Total triangle count: `2 * (rows - 1) * (cols - 1)`.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        2 * (self.rows - 1) * (self.cols - 1)
    }

    /// Local axis-aligned bounding box.
    #[must_use]
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.lower, self.upper)
    }

    /// The 4 corner coordinates of cell `(col, row)`.
    #[must_use]
    pub fn cell_corners(&self, col: usize, row: usize) -> [Vec3; 4] {
        let i11 = row * self.cols + col;
        let i12 = i11 + 1;
        let i21 = (row + 1) * self.cols + col;
        let i22 = i21 + 1;

        let x1 = col as f32 * self.scale.x;
        let x2 = (col + 1) as f32 * self.scale.x;
        let z1 = row as f32 * self.scale.z;
        let z2 = (row + 1) as f32 * self.scale.z;

        [
            Vec3::new(x1, self.heights[i11] * self.scale.y, z1),
            Vec3::new(x2, self.heights[i12] * self.scale.y, z1),
            Vec3::new(x1, self.heights[i21] * self.scale.y, z2),
            Vec3::new(x2, self.heights[i22] * self.scale.y, z2),
        ]
    }

    /// Whether the cell at `(col, row)` is a hole.
    #[must_use]
    pub fn is_hole(&self, col: usize, row: usize) -> bool {
        let cell = row * (self.cols - 1) + col;
        self.materials
            .as_ref()
            .is_some_and(|m| m[cell] == HEIGHT_FIELD_HOLE)
    }

    /// Returns the material index for the given triangle index (cell quad = index / 2).
    #[must_use]
    pub fn material_index(&self, index: usize) -> u8 {
        let quad = index / 2;
        self.materials
            .as_ref()
            .and_then(|m| m.get(quad).copied())
            .unwrap_or(0)
    }

    /// Returns triangle `index` as a [`MeshTriangle`].
    #[must_use]
    pub fn triangle(&self, index: usize) -> MeshTriangle {
        let quad = index / 2;
        let sub = index % 2;
        let row = quad / (self.cols - 1);
        let col = quad % (self.cols - 1);

        let corners = self.cell_corners(col, row);
        let i11 = (row * self.cols + col) as u32;
        let i12 = i11 + 1;
        let i21 = ((row + 1) * self.cols + col) as u32;
        let i22 = i21 + 1;

        let (vertices, indices) = if self.clockwise {
            if sub == 0 {
                ([corners[0], corners[1], corners[2]], [i11, i12, i21])
            } else {
                ([corners[3], corners[2], corners[1]], [i22, i21, i12])
            }
        } else if sub == 0 {
            ([corners[0], corners[2], corners[1]], [i11, i21, i12])
        } else {
            ([corners[3], corners[1], corners[2]], [i22, i12, i21])
        };

        MeshTriangle {
            vertices,
            indices,
            flags: self.flags[index],
        }
    }

    /// Queries all triangles whose bounding box overlaps `[lower, upper]`.
    pub fn query(&self, lower: Vec3, upper: Vec3, mut visit: impl FnMut(usize) -> bool) {
        let min_col = (lower.x / self.scale.x).floor().max(0.0) as usize;
        let max_col = ((upper.x / self.scale.x).ceil().max(0.0) as usize).min(self.cols - 1);
        let min_row = (lower.z / self.scale.z).floor().max(0.0) as usize;
        let max_row = ((upper.z / self.scale.z).ceil().max(0.0) as usize).min(self.rows - 1);

        let center = (lower + upper) * 0.5;
        let extent = upper - center;

        for row in min_row..max_row {
            for col in min_col..max_col {
                if self.is_hole(col, row) {
                    continue;
                }

                let corners = self.cell_corners(col, row);
                let cell_min = corners[0].min(corners[1]).min(corners[2]).min(corners[3]);
                let cell_max = corners[0].max(corners[1]).max(corners[2]).max(corners[3]);

                if (lower - cell_max)
                    .max(cell_min - upper)
                    .cmple(Vec3::ZERO)
                    .all()
                {
                    let quad = row * (self.cols - 1) + col;
                    let tri0 = 2 * quad;
                    let tri1 = tri0 + 1;

                    let t0 = self.triangle(tri0);
                    if bounds_triangle_overlap(center, extent, t0.vertices) && !visit(tri0) {
                        return;
                    }
                    let t1 = self.triangle(tri1);
                    if bounds_triangle_overlap(center, extent, t1.vertices) && !visit(tri1) {
                        return;
                    }
                }
            }
        }
    }

    /// Casts a ray against front-facing triangles of the height field.
    #[must_use]
    pub fn ray_cast(
        &self,
        origin: Vec3,
        translation: Vec3,
        max_fraction: f32,
    ) -> Option<MeshRayHit> {
        let delta = translation * max_fraction;
        let start = origin;
        let end = origin + delta;

        let ray_min = start.min(end);
        let ray_max = start.max(end);

        let min_col = (ray_min.x / self.scale.x).floor().max(0.0) as usize;
        let max_col = ((ray_max.x / self.scale.x).ceil().max(0.0) as usize).min(self.cols - 1);
        let min_row = (ray_min.z / self.scale.z).floor().max(0.0) as usize;
        let max_row = ((ray_max.z / self.scale.z).ceil().max(0.0) as usize).min(self.rows - 1);

        let mut best: Option<MeshRayHit> = None;
        let mut best_frac = 1.0f32;

        for row in min_row..max_row {
            for col in min_col..max_col {
                if self.is_hole(col, row) {
                    continue;
                }
                let quad = row * (self.cols - 1) + col;
                for sub in 0..2 {
                    let tri_idx = 2 * quad + sub;
                    let tri = self.triangle(tri_idx);
                    let [v1, v2, v3] = tri.vertices;
                    let frac = intersect_ray_triangle(origin, translation, v1, v2, v3);
                    if frac < best_frac && frac <= max_fraction {
                        best_frac = frac;
                        let e1 = v2 - v1;
                        let e2 = v3 - v1;
                        let normal = crate::contact_solver::normalize(e1.cross(e2));
                        best = Some(MeshRayHit {
                            fraction: frac,
                            point: origin + translation * frac,
                            normal,
                            triangle: tri_idx,
                        });
                    }
                }
            }
        }

        best
    }
}

#[derive(Clone, Copy)]
struct LocalPlane {
    normal: Vec3,
    distance: f32,
}

impl LocalPlane {
    fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Self {
        let normal = crate::contact_solver::normalize((b - a).cross(c - a));
        let distance = normal.dot(a);
        Self { normal, distance }
    }

    fn separation(&self, p: Vec3) -> f32 {
        self.normal.dot(p) - self.distance
    }
}

fn get_cell_geometry(
    col: usize,
    row: usize,
    cols: usize,
    scale: Vec3,
    heights: &[f32],
) -> ([Vec3; 4], [LocalPlane; 2]) {
    let i11 = row * cols + col;
    let i12 = i11 + 1;
    let i21 = (row + 1) * cols + col;
    let i22 = i21 + 1;

    let x1 = col as f32 * scale.x;
    let x2 = (col + 1) as f32 * scale.x;
    let z1 = row as f32 * scale.z;
    let z2 = (row + 1) as f32 * scale.z;

    let p11 = Vec3::new(x1, heights[i11] * scale.y, z1);
    let p12 = Vec3::new(x2, heights[i12] * scale.y, z1);
    let p21 = Vec3::new(x1, heights[i21] * scale.y, z2);
    let p22 = Vec3::new(x2, heights[i22] * scale.y, z2);

    // Box3D Tri 0 (CCW): p11, p21, p12
    let plane1 = LocalPlane::from_points(p11, p21, p12);
    // Box3D Tri 1 (CCW): p22, p12, p21
    let plane2 = LocalPlane::from_points(p22, p12, p21);

    ([p11, p12, p21, p22], [plane1, plane2])
}

fn swap_edges_1_and_3(flags: u8) -> u8 {
    let e1 = flags & (CONCAVE_EDGE1 | INVERSE_CONCAVE_EDGE1);
    let e3 = flags & (CONCAVE_EDGE3 | INVERSE_CONCAVE_EDGE3);
    (flags & !(CONCAVE_EDGE1 | CONCAVE_EDGE3 | INVERSE_CONCAVE_EDGE1 | INVERSE_CONCAVE_EDGE3))
        | (e1 << 2)
        | (e3 >> 2)
}
