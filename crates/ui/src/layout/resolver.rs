//! Layout rectangle calculation from normalized anchors and pixel offsets.

use crate::control::Control;
use crate::layout::rect::Rect2d;

/// Resolves a single `Control`'s rectangle given its parent container's bounding rectangle.
///
/// Computes four edge positions from normalized anchors and pixel offsets:
/// - $x_0 = \text{parent}.x + (\text{anchor\_left} \times \text{parent}.\text{width}) + \text{offset\_left}$
/// - $y_0 = \text{parent}.y + (\text{anchor\_top} \times \text{parent}.\text{height}) + \text{offset\_top}$
/// - $x_1 = \text{parent}.x + (\text{anchor\_right} \times \text{parent}.\text{width}) + \text{offset\_right}$
/// - $y_1 = \text{parent}.y + (\text{anchor\_bottom} \times \text{parent}.\text{height}) + \text{offset\_bottom}$
///
/// Enforces `custom_minimum_size` clamping on the computed width and height.
pub fn resolve_control_rect(control: &Control, parent_rect: Rect2d) -> Rect2d {
    let x0 = parent_rect.x + (control.anchor_left * parent_rect.width) + control.offset_left;
    let y0 = parent_rect.y + (control.anchor_top * parent_rect.height) + control.offset_top;
    let x1 = parent_rect.x + (control.anchor_right * parent_rect.width) + control.offset_right;
    let y1 = parent_rect.y + (control.anchor_bottom * parent_rect.height) + control.offset_bottom;

    let width = (x1 - x0).max(control.custom_minimum_size.x);
    let height = (y1 - y0).max(control.custom_minimum_size.y);

    Rect2d {
        x: x0,
        y: y0,
        width,
        height,
    }
}
