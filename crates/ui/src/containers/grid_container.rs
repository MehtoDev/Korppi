//! Grid container laying out child controls in configurable column/row grids.

use korppi_ecs::Component;

/// Component that organizes child controls into a two-dimensional grid of cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridContainer {
    /// Number of columns in the grid. Must be at least 1.
    pub columns: usize,
    /// Horizontal separation between columns in pixels.
    pub h_separation: f32,
    /// Vertical separation between rows in pixels.
    pub v_separation: f32,
}

impl Component for GridContainer {}

impl Default for GridContainer {
    fn default() -> Self {
        Self {
            columns: 1,
            h_separation: 4.0,
            v_separation: 4.0,
        }
    }
}

impl GridContainer {
    /// Creates a new `GridContainer` with specified columns and separations.
    #[must_use]
    pub fn new(columns: usize, h_separation: f32, v_separation: f32) -> Self {
        Self {
            columns: columns.max(1),
            h_separation,
            v_separation,
        }
    }
}
