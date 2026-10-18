//! Canonical Godot-inspired child fitting algorithm inside an allocated rectangle.

use crate::control::Control;
use crate::layout::flags::SizeFlags;
use crate::layout::rect::Rect2d;

/// Fits a child control into an allocated container cell rectangle adhering to
/// `SizeFlags` (`FILL`, `SHRINK_CENTER`, `SHRINK_END`, and default `SHRINK_BEGIN`).
#[must_use]
pub fn fit_child_in_rect(child: &Control, available_rect: Rect2d) -> Rect2d {
    let min_size = child.custom_minimum_size;

    // Horizontal dimension
    let (x, width) = if child.size_flags_horizontal.contains(SizeFlags::FILL) {
        let w = available_rect.width.max(min_size.x);
        (available_rect.x, w)
    } else {
        let w = min_size.x.min(available_rect.width);
        let x_offset = if child.size_flags_horizontal.contains(SizeFlags::SHRINK_END) {
            available_rect.width - w
        } else if child
            .size_flags_horizontal
            .contains(SizeFlags::SHRINK_CENTER)
        {
            ((available_rect.width - w) * 0.5).floor()
        } else {
            0.0
        };
        (available_rect.x + x_offset, w)
    };

    // Vertical dimension
    let (y, height) = if child.size_flags_vertical.contains(SizeFlags::FILL) {
        let h = available_rect.height.max(min_size.y);
        (available_rect.y, h)
    } else {
        let h = min_size.y.min(available_rect.height);
        let y_offset = if child.size_flags_vertical.contains(SizeFlags::SHRINK_END) {
            available_rect.height - h
        } else if child.size_flags_vertical.contains(SizeFlags::SHRINK_CENTER) {
            ((available_rect.height - h) * 0.5).floor()
        } else {
            0.0
        };
        (available_rect.y + y_offset, h)
    };

    Rect2d {
        x,
        y,
        width,
        height,
    }
}
