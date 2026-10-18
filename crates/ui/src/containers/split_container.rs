//! Two-pane split container separated by an adjustable draggable split bar.

use crate::containers::box_container::Orientation;
use korppi_ecs::Component;

/// Dragger bar visual visibility mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DraggerVisibility {
    /// Dragger bar is visible and interactive.
    #[default]
    Visible,
    /// Dragger bar is hidden.
    Hidden,
}

/// Component that splits its interior into two resizable panes with an adjustable divider bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitContainer {
    /// Direction of splitting (Horizontal or Vertical).
    pub orientation: Orientation,
    /// Pixel offset from the default proportional split point.
    pub split_offset: f32,
    /// Thickness of the divider bar in pixels.
    pub dragger_thickness: f32,
    /// Whether one pane is collapsed.
    pub collapsed: bool,
    /// If collapsed, whether the first pane (true) or second pane (false) is collapsed.
    pub first_collapsed: bool,
    /// Divider bar visibility mode.
    pub dragger_visibility: DraggerVisibility,
}

impl Component for SplitContainer {}

impl Default for SplitContainer {
    fn default() -> Self {
        Self {
            orientation: Orientation::Horizontal,
            split_offset: 0.0,
            dragger_thickness: 8.0,
            collapsed: false,
            first_collapsed: false,
            dragger_visibility: DraggerVisibility::Visible,
        }
    }
}

impl SplitContainer {
    /// Creates a new `SplitContainer` with the specified orientation and dragger thickness.
    #[must_use]
    pub const fn new(orientation: Orientation, dragger_thickness: f32) -> Self {
        Self {
            orientation,
            split_offset: 0.0,
            dragger_thickness,
            collapsed: false,
            first_collapsed: false,
            dragger_visibility: DraggerVisibility::Visible,
        }
    }
}

/// Helper builder for horizontal split containers.
pub struct HSplitContainer;

impl HSplitContainer {
    /// Creates a new horizontal `SplitContainer`.
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub const fn new(dragger_thickness: f32) -> SplitContainer {
        SplitContainer::new(Orientation::Horizontal, dragger_thickness)
    }
}

/// Helper builder for vertical split containers.
pub struct VSplitContainer;

impl VSplitContainer {
    /// Creates a new vertical `SplitContainer`.
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub const fn new(dragger_thickness: f32) -> SplitContainer {
        SplitContainer::new(Orientation::Vertical, dragger_thickness)
    }
}
