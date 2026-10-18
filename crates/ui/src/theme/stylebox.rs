//! Vector and 9-patch stylebox rendering descriptors.

use crate::layout::rect::Rect2d;
use korppi_math::Vec2;

/// Axis stretch mode for 9-patch texture slicing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AxisStretchMode {
    /// Texture is stretched to fit the edge/center area.
    #[default]
    Stretch,
    /// Texture is tiled repeatedly.
    Tile,
    /// Texture is tiled and scaled so whole tiles fit cleanly.
    TileFit,
}

/// Vector-rendered panel stylebox with background fill, borders, rounded corners, and drop shadow.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleBoxFlat {
    /// Background interior fill color (RGBA).
    pub bg_color: [f32; 4],
    /// Border stroke outline color (RGBA).
    pub border_color: [f32; 4],
    /// Border stroke widths: `[left, top, right, bottom]`.
    pub border_width: [f32; 4],
    /// Corner radii: `[top_left, top_right, bottom_right, bottom_left]`.
    pub corner_radius: [f32; 4],
    /// Expand margins extending outer visual bounds beyond control rect: `[left, top, right, bottom]`.
    pub expand_margin: [f32; 4],
    /// Content insets for child controls: `[left, top, right, bottom]`.
    pub content_margin: [f32; 4],
    /// Drop shadow color (RGBA).
    pub shadow_color: [f32; 4],
    /// Blur / size distance of the drop shadow.
    pub shadow_size: f32,
    /// Pixel offset of the drop shadow.
    pub shadow_offset: Vec2,
    /// Whether to draw the center interior background fill.
    pub draw_center: bool,
    /// Number of segments per rounded corner curve.
    pub corner_detail: u32,
}

impl Default for StyleBoxFlat {
    fn default() -> Self {
        Self {
            bg_color: [0.6, 0.6, 0.6, 1.0],
            border_color: [0.8, 0.8, 0.8, 1.0],
            border_width: [0.0; 4],
            corner_radius: [0.0; 4],
            expand_margin: [0.0; 4],
            content_margin: [0.0; 4],
            shadow_color: [0.0, 0.0, 0.0, 0.6],
            shadow_size: 0.0,
            shadow_offset: Vec2::ZERO,
            draw_center: true,
            corner_detail: 8,
        }
    }
}

impl StyleBoxFlat {
    /// Calculates the expanded outer bounding rectangle including `expand_margin`.
    #[must_use]
    pub fn get_draw_rect(&self, rect: Rect2d) -> Rect2d {
        Rect2d {
            x: rect.x - self.expand_margin[0],
            y: rect.y - self.expand_margin[1],
            width: rect.width + self.expand_margin[0] + self.expand_margin[2],
            height: rect.height + self.expand_margin[1] + self.expand_margin[3],
        }
    }

    /// Returns the minimum size required by the content margins.
    #[must_use]
    pub fn get_minimum_size(&self) -> Vec2 {
        Vec2::new(
            self.content_margin[0] + self.content_margin[2],
            self.content_margin[1] + self.content_margin[3],
        )
    }
}

/// 9-slice scalable texture stylebox.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleBoxTexture {
    /// Sub-region of texture atlas to render.
    pub region_rect: Rect2d,
    /// 9-patch margin slices: `[left, top, right, bottom]`.
    pub texture_margin: [f32; 4],
    /// Expand margins: `[left, top, right, bottom]`.
    pub expand_margin: [f32; 4],
    /// Content insets for child controls: `[left, top, right, bottom]`.
    pub content_margin: [f32; 4],
    /// Color tint / modulation.
    pub modulate: [f32; 4],
    /// Horizontal axis stretch mode.
    pub axis_h: AxisStretchMode,
    /// Vertical axis stretch mode.
    pub axis_v: AxisStretchMode,
    /// Whether to render the central sliced quad.
    pub draw_center: bool,
}

impl Default for StyleBoxTexture {
    fn default() -> Self {
        Self {
            region_rect: Rect2d::default(),
            texture_margin: [0.0; 4],
            expand_margin: [0.0; 4],
            content_margin: [0.0; 4],
            modulate: [1.0, 1.0, 1.0, 1.0],
            axis_h: AxisStretchMode::Stretch,
            axis_v: AxisStretchMode::Stretch,
            draw_center: true,
        }
    }
}

impl StyleBoxTexture {
    /// Calculates the expanded outer bounding rectangle including `expand_margin`.
    #[must_use]
    pub fn get_draw_rect(&self, rect: Rect2d) -> Rect2d {
        Rect2d {
            x: rect.x - self.expand_margin[0],
            y: rect.y - self.expand_margin[1],
            width: rect.width + self.expand_margin[0] + self.expand_margin[2],
            height: rect.height + self.expand_margin[1] + self.expand_margin[3],
        }
    }

    /// Returns the minimum size required by the texture/content margins.
    #[must_use]
    pub fn get_minimum_size(&self) -> Vec2 {
        Vec2::new(
            (self.content_margin[0] + self.content_margin[2])
                .max(self.texture_margin[0] + self.texture_margin[2]),
            (self.content_margin[1] + self.content_margin[3])
                .max(self.texture_margin[1] + self.texture_margin[3]),
        )
    }
}

/// Invisible stylebox providing spacing and content margins.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct StyleBoxEmpty {
    /// Content insets for child controls: `[left, top, right, bottom]`.
    pub content_margin: [f32; 4],
}

impl StyleBoxEmpty {
    /// Returns the input rect without expansion.
    #[must_use]
    pub fn get_draw_rect(&self, rect: Rect2d) -> Rect2d {
        rect
    }

    /// Returns the minimum size required by content margins.
    #[must_use]
    pub fn get_minimum_size(&self) -> Vec2 {
        Vec2::new(
            self.content_margin[0] + self.content_margin[2],
            self.content_margin[1] + self.content_margin[3],
        )
    }
}

/// Polymorphic StyleBox enum containing Flat, Texture, or Empty style definitions.
#[derive(Clone, Debug, PartialEq)]
pub enum StyleBox {
    /// Vector-rendered flat stylebox.
    Flat(StyleBoxFlat),
    /// 9-patch sliced texture stylebox.
    Texture(StyleBoxTexture),
    /// Invisible margin-only stylebox.
    Empty(StyleBoxEmpty),
}

impl StyleBox {
    /// Calculates the outer draw rect including expand margins.
    #[must_use]
    pub fn get_draw_rect(&self, rect: Rect2d) -> Rect2d {
        match self {
            Self::Flat(flat) => flat.get_draw_rect(rect),
            Self::Texture(tex) => tex.get_draw_rect(rect),
            Self::Empty(empty) => empty.get_draw_rect(rect),
        }
    }

    /// Returns the content padding insets: `[left, top, right, bottom]`.
    #[must_use]
    pub fn get_content_margin(&self) -> [f32; 4] {
        match self {
            Self::Flat(flat) => flat.content_margin,
            Self::Texture(tex) => tex.content_margin,
            Self::Empty(empty) => empty.content_margin,
        }
    }

    /// Returns the minimum size enforced by this stylebox.
    #[must_use]
    pub fn get_minimum_size(&self) -> Vec2 {
        match self {
            Self::Flat(flat) => flat.get_minimum_size(),
            Self::Texture(tex) => tex.get_minimum_size(),
            Self::Empty(empty) => empty.get_minimum_size(),
        }
    }
}
