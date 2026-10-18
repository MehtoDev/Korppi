//! Scroll container managing virtual clipping viewports and scroll offsets.

use korppi_ecs::Component;

/// Scrollbar display and scrolling behavior mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScrollMode {
    /// Scrolling is disabled along this axis.
    Disabled,
    /// Scrollbars appear automatically when content exceeds viewport size.
    #[default]
    Auto,
    /// Scrollbars are always visible.
    Always,
}

/// Component providing a clipped scrollable viewport for larger child content.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollContainer {
    /// Current horizontal scroll offset in pixels.
    pub scroll_horizontal: f32,
    /// Current vertical scroll offset in pixels.
    pub scroll_vertical: f32,
    /// Horizontal scrolling mode.
    pub horizontal_scroll_mode: ScrollMode,
    /// Vertical scrolling mode.
    pub vertical_scroll_mode: ScrollMode,
    /// Pixel thickness of the scrollbars.
    pub scrollbar_thickness: f32,
}

impl Component for ScrollContainer {}

impl Default for ScrollContainer {
    fn default() -> Self {
        Self {
            scroll_horizontal: 0.0,
            scroll_vertical: 0.0,
            horizontal_scroll_mode: ScrollMode::Auto,
            vertical_scroll_mode: ScrollMode::Auto,
            scrollbar_thickness: 12.0,
        }
    }
}
