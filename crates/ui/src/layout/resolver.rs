//! Layout rectangle calculation from normalized anchors and pixel offsets.

use crate::control::{Control, GrowDirection};
use crate::layout::rect::Rect2d;

/// Resolves a single `Control`'s rectangle given its parent container's bounding rectangle.
///
/// Computes four edge positions from normalized anchors and pixel offsets:
/// - $x_0 = \text{parent}.x + (\text{anchor\_left} \times \text{parent}.\text{width}) + \text{offset\_left}$
/// - $y_0 = \text{parent}.y + (\text{anchor\_top} \times \text{parent}.\text{height}) + \text{offset\_top}$
/// - $x_1 = \text{parent}.x + (\text{anchor\_right} \times \text{parent}.\text{width}) + \text{offset\_right}$
/// - $y_1 = \text{parent}.y + (\text{anchor\_bottom} \times \text{parent}.\text{height}) + \text{offset\_bottom}$
///
/// Enforces `custom_minimum_size` clamping on the computed width and height, shifting the origin
/// according to `grow_horizontal` and `grow_vertical` when minimum size expands the control, and
/// snaps the resulting geometry to integer pixel coordinates.
pub fn resolve_control_rect(control: &Control, parent_rect: Rect2d) -> Rect2d {
    let x0 = parent_rect.x + (control.anchor_left * parent_rect.width) + control.offset_left;
    let y0 = parent_rect.y + (control.anchor_top * parent_rect.height) + control.offset_top;
    let x1 = parent_rect.x + (control.anchor_right * parent_rect.width) + control.offset_right;
    let y1 = parent_rect.y + (control.anchor_bottom * parent_rect.height) + control.offset_bottom;

    let raw_w = (x1 - x0).max(0.0);
    let raw_h = (y1 - y0).max(0.0);
    let min_w = control.custom_minimum_size.x.max(0.0);
    let min_h = control.custom_minimum_size.y.max(0.0);

    let width = raw_w.max(min_w);
    let height = raw_h.max(min_h);

    let x = if width > raw_w {
        let deficit = width - raw_w;
        match control.grow_horizontal {
            GrowDirection::End => x0,
            GrowDirection::Begin => x0 - deficit,
            GrowDirection::Both => x0 - deficit * 0.5,
        }
    } else {
        x0
    };

    let y = if height > raw_h {
        let deficit = height - raw_h;
        match control.grow_vertical {
            GrowDirection::End => y0,
            GrowDirection::Begin => y0 - deficit,
            GrowDirection::Both => y0 - deficit * 0.5,
        }
    } else {
        y0
    };

    Rect2d {
        x: x.round(),
        y: y.round(),
        width: width.round(),
        height: height.round(),
    }
}
