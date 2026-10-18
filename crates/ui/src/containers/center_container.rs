//! Center container keeping child controls centered within its bounding area.

use korppi_ecs::Component;

/// Component that centers its children within the allocated space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CenterContainer {
    /// If true, children are centered relative to the top-left origin instead of the container center.
    pub use_top_left: bool,
}

impl Component for CenterContainer {}

impl CenterContainer {
    /// Creates a new `CenterContainer`.
    #[must_use]
    pub const fn new(use_top_left: bool) -> Self {
        Self { use_top_left }
    }
}
