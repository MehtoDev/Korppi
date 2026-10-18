//! Base Godot-inspired Control node component and layout presets.

use crate::layout::flags::SizeFlags;
use korppi_ecs::{Component, Entity};
use korppi_math::Vec2;

/// Growth direction when text or minimum size expands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GrowDirection {
    /// Expands towards negative coordinates (left or top).
    Begin,
    /// Expands towards positive coordinates (right or bottom).
    #[default]
    End,
    /// Expands symmetrically in both directions.
    Both,
}

/// Mouse interaction filtering strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MouseFilter {
    /// Stops mouse event propagation; consumes the click/hover event.
    #[default]
    Stop,
    /// Responds to mouse events and passes them down to underlying controls.
    Pass,
    /// Completely ignores mouse events; clicks pass directly through.
    Ignore,
}

/// Keyboard and gamepad focus reception mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FocusMode {
    /// Control cannot receive keyboard/gamepad focus.
    #[default]
    None,
    /// Control only receives focus when clicked with the mouse.
    Click,
    /// Control receives focus on mouse click and keyboard Tab cycling.
    All,
}

/// Standard layout preset anchoring shortcuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutPreset {
    /// Anchors to top-left corner.
    TopLeft,
    /// Anchors to top-right corner.
    TopRight,
    /// Anchors to bottom-left corner.
    BottomLeft,
    /// Anchors to bottom-right corner.
    BottomRight,
    /// Anchors to center-left edge.
    CenterLeft,
    /// Anchors to center-top edge.
    CenterTop,
    /// Anchors to center-right edge.
    CenterRight,
    /// Anchors to center-bottom edge.
    CenterBottom,
    /// Anchors to parent center point.
    Center,
    /// Anchors vertically along the full left edge.
    LeftWide,
    /// Anchors horizontally along the full top edge.
    TopWide,
    /// Anchors vertically along the full right edge.
    RightWide,
    /// Anchors horizontally along the full bottom edge.
    BottomWide,
    /// Anchors along the vertical center line with full height.
    VCenterWide,
    /// Anchors along the horizontal center line with full width.
    HCenterWide,
    /// Anchors to all 4 corners, filling the parent rectangle.
    FullRect,
}

/// Base UI control node component storing anchors, offsets, size flags, and geometry constraints.
#[derive(Clone, Debug, PartialEq)]
pub struct Control {
    /// Normalized horizontal anchor for the left edge ($\in [0.0, 1.0]$).
    pub anchor_left: f32,
    /// Normalized vertical anchor for the top edge ($\in [0.0, 1.0]$).
    pub anchor_top: f32,
    /// Normalized horizontal anchor for the right edge ($\in [0.0, 1.0]$).
    pub anchor_right: f32,
    /// Normalized vertical anchor for the bottom edge ($\in [0.0, 1.0]$).
    pub anchor_bottom: f32,

    /// Pixel offset from `anchor_left`.
    pub offset_left: f32,
    /// Pixel offset from `anchor_top`.
    pub offset_top: f32,
    /// Pixel offset from `anchor_right`.
    pub offset_right: f32,
    /// Pixel offset from `anchor_bottom`.
    pub offset_bottom: f32,

    /// Horizontal expansion growth direction.
    pub grow_horizontal: GrowDirection,
    /// Vertical expansion growth direction.
    pub grow_vertical: GrowDirection,

    /// Horizontal container space distribution flags.
    pub size_flags_horizontal: SizeFlags,
    /// Vertical container space distribution flags.
    pub size_flags_vertical: SizeFlags,
    /// Container stretch ratio weighting during expansion.
    pub stretch_ratio: f32,

    /// Explicit entity neighbor for directional focus navigation to the left.
    pub focus_neighbor_left: Option<Entity>,
    /// Explicit entity neighbor for directional focus navigation upwards.
    pub focus_neighbor_top: Option<Entity>,
    /// Explicit entity neighbor for directional focus navigation to the right.
    pub focus_neighbor_right: Option<Entity>,
    /// Explicit entity neighbor for directional focus navigation downwards.
    pub focus_neighbor_bottom: Option<Entity>,
    /// Explicit entity to receive focus on Tab key navigation.
    pub focus_next: Option<Entity>,
    /// Explicit entity to receive focus on Shift+Tab key navigation.
    pub focus_previous: Option<Entity>,

    /// User-defined minimum size constraint.
    pub custom_minimum_size: Vec2,
    /// Mouse interaction filter.
    pub mouse_filter: MouseFilter,
    /// Keyboard/gamepad focus mode.
    pub focus_mode: FocusMode,
    /// Whether this control and its descendants are rendered and interactive.
    pub visible: bool,
    /// Whether children are clipped to this control's bounding box.
    pub clip_contents: bool,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            anchor_left: 0.0,
            anchor_top: 0.0,
            anchor_right: 0.0,
            anchor_bottom: 0.0,

            offset_left: 0.0,
            offset_top: 0.0,
            offset_right: 0.0,
            offset_bottom: 0.0,

            grow_horizontal: GrowDirection::End,
            grow_vertical: GrowDirection::End,

            size_flags_horizontal: SizeFlags::FILL,
            size_flags_vertical: SizeFlags::FILL,
            stretch_ratio: 1.0,

            focus_neighbor_left: None,
            focus_neighbor_top: None,
            focus_neighbor_right: None,
            focus_neighbor_bottom: None,
            focus_next: None,
            focus_previous: None,

            custom_minimum_size: Vec2::ZERO,
            mouse_filter: MouseFilter::Stop,
            focus_mode: FocusMode::None,
            visible: true,
            clip_contents: false,
        }
    }
}

impl Control {
    /// Configures the 4 anchor values to match a canonical `LayoutPreset`.
    ///
    /// If `keep_offsets` is false, resets all offsets to zero.
    pub fn set_anchors_preset(&mut self, preset: LayoutPreset, keep_offsets: bool) {
        match preset {
            LayoutPreset::TopLeft => {
                self.anchor_left = 0.0;
                self.anchor_top = 0.0;
                self.anchor_right = 0.0;
                self.anchor_bottom = 0.0;
            }
            LayoutPreset::TopRight => {
                self.anchor_left = 1.0;
                self.anchor_top = 0.0;
                self.anchor_right = 1.0;
                self.anchor_bottom = 0.0;
            }
            LayoutPreset::BottomLeft => {
                self.anchor_left = 0.0;
                self.anchor_top = 1.0;
                self.anchor_right = 0.0;
                self.anchor_bottom = 1.0;
            }
            LayoutPreset::BottomRight => {
                self.anchor_left = 1.0;
                self.anchor_top = 1.0;
                self.anchor_right = 1.0;
                self.anchor_bottom = 1.0;
            }
            LayoutPreset::CenterLeft => {
                self.anchor_left = 0.0;
                self.anchor_top = 0.5;
                self.anchor_right = 0.0;
                self.anchor_bottom = 0.5;
            }
            LayoutPreset::CenterTop => {
                self.anchor_left = 0.5;
                self.anchor_top = 0.0;
                self.anchor_right = 0.5;
                self.anchor_bottom = 0.0;
            }
            LayoutPreset::CenterRight => {
                self.anchor_left = 1.0;
                self.anchor_top = 0.5;
                self.anchor_right = 1.0;
                self.anchor_bottom = 0.5;
            }
            LayoutPreset::CenterBottom => {
                self.anchor_left = 0.5;
                self.anchor_top = 1.0;
                self.anchor_right = 0.5;
                self.anchor_bottom = 1.0;
            }
            LayoutPreset::Center => {
                self.anchor_left = 0.5;
                self.anchor_top = 0.5;
                self.anchor_right = 0.5;
                self.anchor_bottom = 0.5;
            }
            LayoutPreset::LeftWide => {
                self.anchor_left = 0.0;
                self.anchor_top = 0.0;
                self.anchor_right = 0.0;
                self.anchor_bottom = 1.0;
            }
            LayoutPreset::TopWide => {
                self.anchor_left = 0.0;
                self.anchor_top = 0.0;
                self.anchor_right = 1.0;
                self.anchor_bottom = 0.0;
            }
            LayoutPreset::RightWide => {
                self.anchor_left = 1.0;
                self.anchor_top = 0.0;
                self.anchor_right = 1.0;
                self.anchor_bottom = 1.0;
            }
            LayoutPreset::BottomWide => {
                self.anchor_left = 0.0;
                self.anchor_top = 1.0;
                self.anchor_right = 1.0;
                self.anchor_bottom = 1.0;
            }
            LayoutPreset::VCenterWide => {
                self.anchor_left = 0.5;
                self.anchor_top = 0.0;
                self.anchor_right = 0.5;
                self.anchor_bottom = 1.0;
            }
            LayoutPreset::HCenterWide => {
                self.anchor_left = 0.0;
                self.anchor_top = 0.5;
                self.anchor_right = 1.0;
                self.anchor_bottom = 0.5;
            }
            LayoutPreset::FullRect => {
                self.anchor_left = 0.0;
                self.anchor_top = 0.0;
                self.anchor_right = 1.0;
                self.anchor_bottom = 1.0;
            }
        }

        if !keep_offsets {
            self.offset_left = 0.0;
            self.offset_top = 0.0;
            self.offset_right = 0.0;
            self.offset_bottom = 0.0;
        }
    }
}

impl Component for Control {}
