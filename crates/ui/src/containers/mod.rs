//! Automated layout container suite inspired by Godot's Container nodes.

pub mod aspect_ratio_container;
pub mod box_container;
pub mod center_container;
pub mod fitting;
pub mod grid_container;
pub mod layout;
pub mod margin_container;
pub mod panel_container;
pub mod scroll_container;
pub mod split_container;

pub use aspect_ratio_container::{AspectRatioContainer, AspectStretchMode};
pub use box_container::{BoxAlignment, BoxContainer, HBoxContainer, Orientation, VBoxContainer};
pub use center_container::CenterContainer;
pub use fitting::fit_child_in_rect;
pub use grid_container::GridContainer;
pub(crate) use layout::panel_effective_padding;
pub use layout::resolve_container_layout;
pub use margin_container::MarginContainer;
pub use panel_container::PanelContainer;
pub use scroll_container::{ScrollContainer, ScrollMode};
pub use split_container::{DraggerVisibility, HSplitContainer, SplitContainer, VSplitContainer};
