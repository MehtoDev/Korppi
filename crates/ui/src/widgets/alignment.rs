//! Horizontal and vertical alignment enums for UI widgets.

/// Horizontal text and content alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HorizontalAlignment {
    /// Aligns content to the left edge.
    #[default]
    Left,
    /// Centers content horizontally.
    Center,
    /// Aligns content to the right edge.
    Right,
    /// Expands content to fill available horizontal space.
    Fill,
}

/// Vertical text and content alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerticalAlignment {
    /// Aligns content to the top edge.
    #[default]
    Top,
    /// Centers content vertically.
    Center,
    /// Aligns content to the bottom edge.
    Bottom,
    /// Expands content to fill available vertical space.
    Fill,
}
