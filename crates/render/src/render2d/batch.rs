//! Batched 2D quad and sprite rendering structures.

use crate::render2d::atlas::{compute_9patch_quads, NinePatchMargins, Rect2d, TextureAtlas2d};
use crate::render2d::text::BitmapFont2d;
use korppi_math::{Vec2, Vec4};
use korppi_rhi::TextureHandle;

/// Vertex structure for batched 2D quads and UI elements.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadVertex2d {
    /// 2D position in canvas / screen coordinates.
    pub position: [f32; 2],
    /// Normalized UV texture coordinates.
    pub uv: [f32; 2],
    /// RGBA tint color multiplier.
    pub color: [f32; 4],
}

gpu_pod!(QuadVertex2d = (2 + 2 + 4) * 4);

/// Contiguous index-buffer run sharing a single bound texture (`None` for the default white texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuadBatchDraw {
    /// Bound texture handle, or `None` to bind the default 1x1 white texture.
    pub texture: Option<TextureHandle>,
    /// Starting index in [`QuadBatch2d::indices`].
    pub first_index: u32,
    /// Number of indices in this draw run.
    pub index_count: u32,
}

/// Dynamic batch builder collecting 2D quads into shared vertex and index buffers.
#[derive(Clone, Debug)]
pub struct QuadBatch2d {
    vertices: Vec<QuadVertex2d>,
    indices: Vec<u32>,
    draws: Vec<QuadBatchDraw>,
    max_quads: usize,
}

impl QuadBatch2d {
    /// Creates a new quad batch with pre-allocated capacity for `max_quads`.
    pub fn new(max_quads: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(max_quads * 4),
            indices: Vec::with_capacity(max_quads * 6),
            draws: Vec::with_capacity(16),
            max_quads,
        }
    }

    fn record_draw_indices(
        &mut self,
        texture: Option<TextureHandle>,
        first_index: u32,
        index_count: u32,
    ) {
        if let Some(last) = self.draws.last_mut() {
            if last.texture == texture && last.first_index + last.index_count == first_index {
                last.index_count += index_count;
                return;
            }
        }
        self.draws.push(QuadBatchDraw {
            texture,
            first_index,
            index_count,
        });
    }

    /// Appends an axis-aligned 2D quad with an optional texture handle.
    pub fn push_quad_with_texture(
        &mut self,
        texture: Option<TextureHandle>,
        min: Vec2,
        max: Vec2,
        uv_min: Vec2,
        uv_max: Vec2,
        color: [f32; 4],
    ) {
        if self.quad_count() >= self.max_quads {
            return;
        }

        let base_index = self.vertices.len() as u32;
        let first_index = self.indices.len() as u32;

        // 4 Quad Vertices: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.vertices.push(QuadVertex2d {
            position: [min.x, min.y],
            uv: [uv_min.x, uv_min.y],
            color,
        });
        self.vertices.push(QuadVertex2d {
            position: [max.x, min.y],
            uv: [uv_max.x, uv_min.y],
            color,
        });
        self.vertices.push(QuadVertex2d {
            position: [max.x, max.y],
            uv: [uv_max.x, uv_max.y],
            color,
        });
        self.vertices.push(QuadVertex2d {
            position: [min.x, max.y],
            uv: [uv_min.x, uv_max.y],
            color,
        });

        // 6 Indices per Quad: [0, 1, 2, 2, 3, 0]
        self.indices.extend_from_slice(&[
            base_index,
            base_index + 1,
            base_index + 2,
            base_index + 2,
            base_index + 3,
            base_index,
        ]);

        self.record_draw_indices(texture, first_index, 6);
    }

    /// Appends an untextured (solid-colour) axis-aligned 2D quad to the batch.
    pub fn push_quad(&mut self, min: Vec2, max: Vec2, uv_min: Vec2, uv_max: Vec2, color: [f32; 4]) {
        self.push_quad_with_texture(None, min, max, uv_min, uv_max, color);
    }

    /// Appends a textured axis-aligned 2D quad to the batch.
    pub fn push_textured_quad(
        &mut self,
        texture: TextureHandle,
        min: Vec2,
        max: Vec2,
        uv_min: Vec2,
        uv_max: Vec2,
        color: [f32; 4],
    ) {
        self.push_quad_with_texture(Some(texture), min, max, uv_min, uv_max, color);
    }

    /// Appends an arbitrary oriented 4-vertex quad with an optional texture handle.
    pub fn push_quad_vertices_with_texture(
        &mut self,
        texture: Option<TextureHandle>,
        points: [Vec2; 4],
        uvs: [[f32; 2]; 4],
        color: [f32; 4],
    ) {
        if self.quad_count() >= self.max_quads {
            return;
        }

        let base_index = self.vertices.len() as u32;
        let first_index = self.indices.len() as u32;

        for i in 0..4 {
            self.vertices.push(QuadVertex2d {
                position: [points[i].x, points[i].y],
                uv: uvs[i],
                color,
            });
        }

        self.indices.extend_from_slice(&[
            base_index,
            base_index + 1,
            base_index + 2,
            base_index + 2,
            base_index + 3,
            base_index,
        ]);

        self.record_draw_indices(texture, first_index, 6);
    }

    /// Appends an arbitrary oriented 4-vertex untextured quad to the batch.
    pub fn push_quad_vertices(&mut self, points: [Vec2; 4], uvs: [[f32; 2]; 4], color: [f32; 4]) {
        self.push_quad_vertices_with_texture(None, points, uvs, color);
    }

    /// Appends an arbitrary oriented 4-vertex textured quad to the batch.
    pub fn push_textured_quad_vertices(
        &mut self,
        texture: TextureHandle,
        points: [Vec2; 4],
        uvs: [[f32; 2]; 4],
        color: [f32; 4],
    ) {
        self.push_quad_vertices_with_texture(Some(texture), points, uvs, color);
    }

    /// Appends a sprite sub-region from `atlas` sampled from `texture`.
    ///
    /// Returns `false` if `region_id` does not exist in `atlas`.
    pub fn push_atlas_region(
        &mut self,
        texture: TextureHandle,
        atlas: &TextureAtlas2d,
        region_id: usize,
        min: Vec2,
        max: Vec2,
        color: [f32; 4],
    ) -> bool {
        let Some(uv) = atlas.get_uv(region_id) else {
            return false;
        };
        self.push_textured_quad(
            texture,
            min,
            max,
            Vec2::new(uv.x, uv.y),
            Vec2::new(uv.z, uv.w),
            color,
        );
        true
    }

    /// Decomposes a 9-patch rectangle and pushes all non-degenerate sub-quads into the batch.
    pub fn push_9patch(
        &mut self,
        texture: Option<TextureHandle>,
        src_rect: Rect2d,
        src_uv: Vec4,
        margins: NinePatchMargins,
        dest: Rect2d,
        color: [f32; 4],
    ) {
        let quads = compute_9patch_quads(src_rect, src_uv, margins, dest);
        for sub in quads {
            if sub.dest.width > 0.0 && sub.dest.height > 0.0 {
                let min = Vec2::new(sub.dest.x, sub.dest.y);
                let max = Vec2::new(sub.dest.x + sub.dest.width, sub.dest.y + sub.dest.height);
                let uv_min = Vec2::new(sub.uv.x, sub.uv.y);
                let uv_max = Vec2::new(sub.uv.z, sub.uv.w);
                self.push_quad_with_texture(texture, min, max, uv_min, uv_max, color);
            }
        }
    }

    /// Appends a string of printable ASCII text rendered with `font` at `origin` and `scale`.
    pub fn push_text(
        &mut self,
        font: &BitmapFont2d,
        text: &str,
        origin: Vec2,
        scale: f32,
        color: [f32; 4],
    ) {
        if scale <= 0.0 || text.is_empty() {
            return;
        }
        let cell = font.glyph_size(scale);
        let tex = font.texture();
        let mut cursor = origin;
        for ch in text.chars() {
            if ch == '\n' {
                cursor.x = origin.x;
                cursor.y += cell.y;
                continue;
            }
            if ch != ' ' {
                if let Some(uv) = font.glyph_uv(ch) {
                    self.push_textured_quad(
                        tex,
                        cursor,
                        cursor + cell,
                        Vec2::new(uv.x, uv.y),
                        Vec2::new(uv.z, uv.w),
                        color,
                    );
                }
            }
            cursor.x += cell.x;
        }
    }

    /// Appends a thick line segment as an oriented quad.
    pub fn push_line(&mut self, p1: Vec2, p2: Vec2, thickness: f32, color: [f32; 4]) {
        let dir = p2 - p1;
        let len = dir.length();
        if len < 1e-6 {
            return;
        }
        let u = dir / len;
        let perp = Vec2::new(-u.y, u.x) * (thickness * 0.5);

        let v0 = p1 + perp;
        let v1 = p2 + perp;
        let v2 = p2 - perp;
        let v3 = p1 - perp;

        self.push_quad_vertices(
            [v0, v1, v2, v3],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            color,
        );
    }

    /// Returns the total number of quads currently buffered in this batch.
    #[inline]
    pub fn quad_count(&self) -> usize {
        self.vertices.len() / 4
    }

    /// Returns the slice of buffered vertices.
    #[inline]
    pub fn vertices(&self) -> &[QuadVertex2d] {
        &self.vertices
    }

    /// Returns the slice of generated index buffer indices.
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Returns the slice of coalesced draw runs in submission order.
    #[inline]
    pub fn draws(&self) -> &[QuadBatchDraw] {
        &self.draws
    }

    /// Clears the batch, preserving underlying allocated buffer capacities.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.draws.clear();
    }
}
