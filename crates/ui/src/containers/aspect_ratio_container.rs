//! Aspect ratio container maintaining fixed proportional aspect ratios.

use crate::containers::box_container::BoxAlignment;
use korppi_ecs::Component;

/// Stretch mode governing how the aspect ratio is enforced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AspectStretchMode {
    /// Width dictates height based on the ratio.
    WidthControlsHeight,
    /// Height dictates width based on the ratio.
    HeightControlsWidth,
    /// Scales child to fit within the container while maintaining ratio.
    #[default]
    Fit,
    /// Scales child to fully cover the container while maintaining ratio.
    Cover,
}

/// Component that preserves a specific aspect ratio for its child controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AspectRatioContainer {
    /// Target aspect ratio (width / height).
    pub ratio: f32,
    /// Scaling behavior mode.
    pub stretch_mode: AspectStretchMode,
    /// Horizontal alignment of the scaled child within the container.
    pub alignment_horizontal: BoxAlignment,
    /// Vertical alignment of the scaled child within the container.
    pub alignment_vertical: BoxAlignment,
}

impl Component for AspectRatioContainer {}

impl Default for AspectRatioContainer {
    fn default() -> Self {
        Self {
            ratio: 1.0,
            stretch_mode: AspectStretchMode::Fit,
            alignment_horizontal: BoxAlignment::Center,
            alignment_vertical: BoxAlignment::Center,
        }
    }
}

impl AspectRatioContainer {
    /// Creates a new `AspectRatioContainer` with the specified ratio and stretch mode.
    #[must_use]
    pub const fn new(ratio: f32, stretch_mode: AspectStretchMode) -> Self {
        Self {
            ratio,
            stretch_mode,
            alignment_horizontal: BoxAlignment::Center,
            alignment_vertical: BoxAlignment::Center,
        }
    }
}
