//! UI layout primitives, geometry, flags, and hierarchy resolution.

pub mod flags;
pub mod hierarchy;
pub mod rect;
pub mod resolver;

pub use flags::SizeFlags;
pub use hierarchy::{
    compute_combined_minimum_size, is_effectively_visible, resolve_control_tree,
    resolve_ui_layout_system, ResolvedRect,
};
pub use rect::Rect2d;
pub use resolver::resolve_control_rect;
