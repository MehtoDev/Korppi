//! 2D texture atlasing, sub-region UV transforms, and 9-patch slicing.

use korppi_math::Vec4;

/// 2D axis-aligned rectangle in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect2d {
    /// Horizontal origin coordinate.
    pub x: f32,
    /// Vertical origin coordinate.
    pub y: f32,
    /// Width dimension.
    pub width: f32,
    /// Height dimension.
    pub height: f32,
}

/// 9-patch margin insets (left, right, top, bottom) in source pixels.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct NinePatchMargins {
    /// Left margin border width.
    pub left: f32,
    /// Right margin border width.
    pub right: f32,
    /// Top margin border height.
    pub top: f32,
    /// Bottom margin border height.
    pub bottom: f32,
}

/// Represents a single sub-quad produced by 9-patch decomposition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NinePatchQuad {
    /// Destination rectangle in screen coordinates.
    pub dest: Rect2d,
    /// Normalized UV rectangle: `[u_min, v_min, u_max, v_max]`.
    pub uv: Vec4,
}

/// 2D texture atlas storing named sprite regions and calculating normalized UVs.
#[derive(Clone, Debug)]
pub struct TextureAtlas2d {
    width: u32,
    height: u32,
    regions: Vec<Rect2d>,
    names: Vec<String>,
}

impl TextureAtlas2d {
    /// Creates a new texture atlas with given pixel dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            regions: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Registers a sprite sub-region rectangle and returns its unique ID.
    pub fn add_region(&mut self, name: &str, rect: Rect2d) -> usize {
        let id = self.regions.len();
        self.regions.push(rect);
        self.names.push(name.to_string());
        id
    }

    /// Calculates normalized UV coordinates `[u_min, v_min, u_max, v_max]` for region `id`.
    pub fn get_uv(&self, id: usize) -> Option<Vec4> {
        let rect = self.regions.get(id)?;
        let w = self.width as f32;
        let h = self.height as f32;

        let u_min = rect.x / w;
        let v_min = rect.y / h;
        let u_max = (rect.x + rect.width) / w;
        let v_max = (rect.y + rect.height) / h;

        Some(Vec4::new(u_min, v_min, u_max, v_max))
    }

    /// Returns the pixel rectangle for region `id`.
    pub fn get_region(&self, id: usize) -> Option<Rect2d> {
        self.regions.get(id).copied()
    }

    /// Finds a region index by name.
    pub fn find_region(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
}

/// Decomposes a source rectangle and margins into 9 scalable sub-quads.
pub fn compute_9patch_quads(
    src_rect: Rect2d,
    src_uv: Vec4,
    margins: NinePatchMargins,
    dest: Rect2d,
) -> [NinePatchQuad; 9] {
    let uv_span_x = src_uv.z - src_uv.x;
    let uv_span_y = src_uv.w - src_uv.y;

    // UV splits
    let u0 = src_uv.x;
    let u1 = src_uv.x + (margins.left / src_rect.width) * uv_span_x;
    let u2 = src_uv.z - (margins.right / src_rect.width) * uv_span_x;
    let u3 = src_uv.z;

    let v0 = src_uv.y;
    let v1 = src_uv.y + (margins.top / src_rect.height) * uv_span_y;
    let v2 = src_uv.w - (margins.bottom / src_rect.height) * uv_span_y;
    let v3 = src_uv.w;

    // Destination coordinate splits
    let x0 = dest.x;
    let x1 = dest.x + margins.left;
    let x2 = dest.x + dest.width - margins.right;

    let y0 = dest.y;
    let y1 = dest.y + margins.top;
    let y2 = dest.y + dest.height - margins.bottom;

    let w_left = margins.left;
    let w_center = (dest.width - margins.left - margins.right).max(0.0);
    let w_right = margins.right;

    let h_top = margins.top;
    let h_center = (dest.height - margins.top - margins.bottom).max(0.0);
    let h_bottom = margins.bottom;

    [
        // Row 0: Top-Left, Top-Center, Top-Right
        NinePatchQuad {
            dest: Rect2d {
                x: x0,
                y: y0,
                width: w_left,
                height: h_top,
            },
            uv: Vec4::new(u0, v0, u1, v1),
        },
        NinePatchQuad {
            dest: Rect2d {
                x: x1,
                y: y0,
                width: w_center,
                height: h_top,
            },
            uv: Vec4::new(u1, v0, u2, v1),
        },
        NinePatchQuad {
            dest: Rect2d {
                x: x2,
                y: y0,
                width: w_right,
                height: h_top,
            },
            uv: Vec4::new(u2, v0, u3, v1),
        },
        // Row 1: Middle-Left, Center, Middle-Right
        NinePatchQuad {
            dest: Rect2d {
                x: x0,
                y: y1,
                width: w_left,
                height: h_center,
            },
            uv: Vec4::new(u0, v1, u1, v2),
        },
        NinePatchQuad {
            dest: Rect2d {
                x: x1,
                y: y1,
                width: w_center,
                height: h_center,
            },
            uv: Vec4::new(u1, v1, u2, v2),
        },
        NinePatchQuad {
            dest: Rect2d {
                x: x2,
                y: y1,
                width: w_right,
                height: h_center,
            },
            uv: Vec4::new(u2, v1, u3, v2),
        },
        // Row 2: Bottom-Left, Bottom-Center, Bottom-Right
        NinePatchQuad {
            dest: Rect2d {
                x: x0,
                y: y2,
                width: w_left,
                height: h_bottom,
            },
            uv: Vec4::new(u0, v2, u1, v3),
        },
        NinePatchQuad {
            dest: Rect2d {
                x: x1,
                y: y2,
                width: w_center,
                height: h_bottom,
            },
            uv: Vec4::new(u1, v2, u2, v3),
        },
        NinePatchQuad {
            dest: Rect2d {
                x: x2,
                y: y2,
                width: w_right,
                height: h_bottom,
            },
            uv: Vec4::new(u2, v2, u3, v3),
        },
    ]
}
