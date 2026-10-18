//! Margin container insetting child controls by customizable edge margins.

use korppi_ecs::Component;

/// Component that adds margin padding insets around its child controls.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MarginContainer {
    /// Inset padding from the left edge.
    pub margin_left: f32,
    /// Inset padding from the top edge.
    pub margin_top: f32,
    /// Inset padding from the right edge.
    pub margin_right: f32,
    /// Inset padding from the bottom edge.
    pub margin_bottom: f32,
}

impl Component for MarginContainer {}

impl MarginContainer {
    /// Creates a new `MarginContainer` with individual edge insets.
    #[must_use]
    pub const fn new(
        margin_left: f32,
        margin_top: f32,
        margin_right: f32,
        margin_bottom: f32,
    ) -> Self {
        Self {
            margin_left,
            margin_top,
            margin_right,
            margin_bottom,
        }
    }

    /// Creates a new `MarginContainer` with uniform insets on all 4 sides.
    #[must_use]
    pub const fn all(margin: f32) -> Self {
        Self {
            margin_left: margin,
            margin_top: margin,
            margin_right: margin,
            margin_bottom: margin,
        }
    }

    /// Creates a new `MarginContainer` with symmetric horizontal and vertical insets.
    #[must_use]
    pub const fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self {
            margin_left: horizontal,
            margin_top: vertical,
            margin_right: horizontal,
            margin_bottom: vertical,
        }
    }
}
