//! Canvas scaling modes, coordinate remapping, and orthographic projection.

use korppi_math::{Mat4, Vec2};

/// Canvas resolution scaling strategies for window resizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CanvasScaleMode {
    /// Stretches virtual canvas to fill window regardless of aspect ratio distortion.
    Stretch,
    /// Preserves aspect ratio, letterboxing or pillarboxing with centered viewports.
    #[default]
    KeepAspect,
    /// Restricts canvas scale to integer multipliers ($1\times, 2\times, \dots$) for pixel art.
    IntegerRatio,
}

/// Resolved canvas viewport and coordinate space mapping parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasTransform {
    /// Virtual base game canvas resolution.
    pub virtual_size: Vec2,
    /// Current physical window pixel resolution.
    pub window_size: Vec2,
    /// Physical pixel offset of the canvas within the window.
    pub viewport_pos: Vec2,
    /// Physical pixel dimensions of the canvas viewport.
    pub viewport_size: Vec2,
    /// Effective scaling factor.
    pub scale: f32,
}

impl CanvasTransform {
    /// Converts a virtual canvas coordinate point into physical window screen space.
    #[inline]
    pub fn virtual_to_screen(&self, virtual_pos: Vec2) -> Vec2 {
        self.viewport_pos + virtual_pos * self.scale
    }

    /// Converts a physical window screen coordinate into virtual canvas space.
    #[inline]
    pub fn screen_to_virtual(&self, screen_pos: Vec2) -> Vec2 {
        (screen_pos - self.viewport_pos) / self.scale.max(1e-4)
    }

    /// Generates an orthographic projection matrix mapping virtual canvas bounds into NDC.
    pub fn orthographic_projection(&self) -> Mat4 {
        Mat4::orthographic_rh(0.0, self.virtual_size.x, self.virtual_size.y, 0.0, 0.0, 1.0)
    }
}

/// Computes canvas viewport placement and scaling parameters for a target window size.
pub fn compute_canvas_transform(
    virtual_size: Vec2,
    window_size: Vec2,
    mode: CanvasScaleMode,
) -> CanvasTransform {
    match mode {
        CanvasScaleMode::Stretch => {
            let scale_x = window_size.x / virtual_size.x.max(1e-4);
            let scale_y = window_size.y / virtual_size.y.max(1e-4);
            CanvasTransform {
                virtual_size,
                window_size,
                viewport_pos: Vec2::ZERO,
                viewport_size: window_size,
                scale: (scale_x + scale_y) * 0.5,
            }
        }
        CanvasScaleMode::KeepAspect => {
            let scale_x = window_size.x / virtual_size.x.max(1e-4);
            let scale_y = window_size.y / virtual_size.y.max(1e-4);
            let scale = scale_x.min(scale_y);

            let viewport_size = virtual_size * scale;
            let viewport_pos = (window_size - viewport_size) * 0.5;

            CanvasTransform {
                virtual_size,
                window_size,
                viewport_pos,
                viewport_size,
                scale,
            }
        }
        CanvasScaleMode::IntegerRatio => {
            let scale_x = (window_size.x / virtual_size.x.max(1e-4)).floor().max(1.0);
            let scale_y = (window_size.y / virtual_size.y.max(1e-4)).floor().max(1.0);
            let scale = scale_x.min(scale_y);

            let viewport_size = virtual_size * scale;
            let viewport_pos = (window_size - viewport_size) * 0.5;

            CanvasTransform {
                virtual_size,
                window_size,
                viewport_pos,
                viewport_size,
                scale,
            }
        }
    }
}
