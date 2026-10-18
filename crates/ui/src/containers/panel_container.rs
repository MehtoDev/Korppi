//! Panel container framing child controls with theme/stylebox padding.

use korppi_ecs::Component;

/// Component that frames its child controls, applying inner style padding.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct PanelContainer {
    /// Inset padding from the left edge.
    pub padding_left: f32,
    /// Inset padding from the top edge.
    pub padding_top: f32,
    /// Inset padding from the right edge.
    pub padding_right: f32,
    /// Inset padding from the bottom edge.
    pub padding_bottom: f32,
}

impl Component for PanelContainer {}

impl PanelContainer {
    /// Creates a new `PanelContainer` with individual padding values.
    #[must_use]
    pub const fn new(
        padding_left: f32,
        padding_top: f32,
        padding_right: f32,
        padding_bottom: f32,
    ) -> Self {
        Self {
            padding_left,
            padding_top,
            padding_right,
            padding_bottom,
        }
    }

    /// Creates a new `PanelContainer` with uniform padding on all sides.
    #[must_use]
    pub const fn all(padding: f32) -> Self {
        Self {
            padding_left: padding,
            padding_top: padding,
            padding_right: padding,
            padding_bottom: padding,
        }
    }

    /// Creates a new `PanelContainer` with symmetric padding.
    #[must_use]
    pub const fn symmetric(horizontal: f32, vertical: f32) -> Self {
        Self {
            padding_left: horizontal,
            padding_top: vertical,
            padding_right: horizontal,
            padding_bottom: vertical,
        }
    }
}
