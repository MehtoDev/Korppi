//! Intermediate draw commands for UI canvas rendering and batched decomposition.

use crate::layout::Rect2d;
use crate::theme::{StyleBoxFlat, StyleBoxTexture};
use crate::widgets::{
    ButtonState, HorizontalAlignment, ProgressFillMode, SliderOrientation, VerticalAlignment,
};
use korppi_math::Vec2;

/// Intermediate drawing command generated during UI hierarchy traversal.
#[derive(Clone, Debug, PartialEq)]
pub enum UiDrawCommand {
    /// Generic textured or solid color 2D quad.
    Quad {
        /// Screen destination rectangle.
        rect: Rect2d,
        /// Top-left UV coordinates.
        uv_min: Vec2,
        /// Bottom-right UV coordinates.
        uv_max: Vec2,
        /// RGBA tint multiplier.
        color: [f32; 4],
        /// Optional hardware scissor rectangle constraint.
        scissor: Option<Rect2d>,
    },
    /// Flat vector-styled box with fill and borders.
    StyleBoxFlat {
        /// Screen destination rectangle.
        rect: Rect2d,
        /// Flat styling parameters.
        style: StyleBoxFlat,
        /// Optional scissor rectangle constraint.
        scissor: Option<Rect2d>,
    },
    /// 9-slice scalable texture styled box.
    StyleBoxTexture {
        /// Screen destination rectangle.
        rect: Rect2d,
        /// Texture styling parameters.
        style: StyleBoxTexture,
        /// Optional scissor rectangle constraint.
        scissor: Option<Rect2d>,
    },
    /// Text label string.
    Text {
        /// Screen destination rectangle.
        rect: Rect2d,
        /// Text string to render.
        text: String,
        /// Font RGBA color.
        color: [f32; 4],
        /// Text horizontal and vertical alignments.
        alignment: (HorizontalAlignment, VerticalAlignment),
        /// Optional scissor rectangle constraint.
        scissor: Option<Rect2d>,
    },
    /// Interactive button widget background and text.
    Button {
        /// Screen destination rectangle.
        rect: Rect2d,
        /// Button label text.
        text: String,
        /// Current visual interaction state.
        state: ButtonState,
        /// Optional scissor rectangle constraint.
        scissor: Option<Rect2d>,
    },
    /// Progress bar indicator track and filled bar.
    ProgressBar {
        /// Screen destination rectangle.
        rect: Rect2d,
        /// Normalized fill ratio $[0.0, 1.0]$.
        ratio: f32,
        /// Fill presentation direction.
        fill_mode: ProgressFillMode,
        /// Optional scissor rectangle constraint.
        scissor: Option<Rect2d>,
    },
    /// Value range slider track and thumb handle.
    Slider {
        /// Screen destination rectangle.
        rect: Rect2d,
        /// Normalized slider ratio $[0.0, 1.0]$.
        ratio: f32,
        /// Slider track orientation.
        orientation: SliderOrientation,
        /// Optional scissor rectangle constraint.
        scissor: Option<Rect2d>,
    },
}
