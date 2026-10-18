//! Linear box container laying out child controls horizontally or vertically.

use korppi_ecs::Component;

/// Orientation of linear layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orientation {
    /// Children are laid out left-to-right.
    #[default]
    Horizontal,
    /// Children are laid out top-to-bottom.
    Vertical,
}

/// Alignment of children within the box when extra space remains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BoxAlignment {
    /// Aligns children towards the start of the container.
    #[default]
    Begin,
    /// Centers children within the container.
    Center,
    /// Aligns children towards the end of the container.
    End,
}

/// Component that lays out child controls sequentially in a single axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxContainer {
    /// Direction of child arrangement.
    pub orientation: Orientation,
    /// Pixel distance between consecutive children.
    pub separation: f32,
    /// Alignment mode along the primary layout axis.
    pub alignment: BoxAlignment,
}

impl Component for BoxContainer {}

impl Default for BoxContainer {
    fn default() -> Self {
        Self {
            orientation: Orientation::Horizontal,
            separation: 4.0,
            alignment: BoxAlignment::Begin,
        }
    }
}

impl BoxContainer {
    /// Creates a new `BoxContainer` with the specified orientation and separation.
    #[must_use]
    pub const fn new(orientation: Orientation, separation: f32) -> Self {
        Self {
            orientation,
            separation,
            alignment: BoxAlignment::Begin,
        }
    }
}

/// Builder helper for horizontal box containers.
pub struct HBoxContainer;

impl HBoxContainer {
    /// Creates a new horizontal `BoxContainer` with the given separation.
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub const fn new(separation: f32) -> BoxContainer {
        BoxContainer::new(Orientation::Horizontal, separation)
    }
}

/// Builder helper for vertical box containers.
pub struct VBoxContainer;

impl VBoxContainer {
    /// Creates a new vertical `BoxContainer` with the given separation.
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub const fn new(separation: f32) -> BoxContainer {
        BoxContainer::new(Orientation::Vertical, separation)
    }
}
