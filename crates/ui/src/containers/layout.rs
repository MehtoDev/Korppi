//! Automated layout calculation and child rectangle fitting for UI containers.

use crate::containers::aspect_ratio_container::{AspectRatioContainer, AspectStretchMode};
use crate::containers::box_container::{BoxAlignment, BoxContainer, Orientation};
use crate::containers::center_container::CenterContainer;
use crate::containers::fitting::fit_child_in_rect;
use crate::containers::grid_container::GridContainer;
use crate::containers::margin_container::MarginContainer;
use crate::containers::panel_container::PanelContainer;
use crate::containers::scroll_container::ScrollContainer;
use crate::containers::split_container::SplitContainer;
use crate::control::Control;
use crate::layout::flags::SizeFlags;
use crate::layout::hierarchy::ResolvedRect;
use crate::layout::rect::Rect2d;
use korppi_ecs::{Entity, World};

/// Attempts to sort and fit the children of a container entity.
/// Returns `true` if the entity was a recognized container and handled child layout.
pub fn resolve_container_layout(
    world: &mut World,
    container_entity: Entity,
    container_rect: Rect2d,
    children: &[Entity],
) -> bool {
    // 1. BoxContainer (HBox / VBox)
    if let Some(box_container) = world.get::<BoxContainer>(container_entity).copied() {
        resolve_box_layout(world, box_container, container_rect, children);
        return true;
    }

    // 2. GridContainer
    if let Some(grid) = world.get::<GridContainer>(container_entity).copied() {
        resolve_grid_layout(world, grid, container_rect, children);
        return true;
    }

    // 3. MarginContainer
    if let Some(margin) = world.get::<MarginContainer>(container_entity).copied() {
        resolve_margin_layout(world, margin, container_rect, children);
        return true;
    }

    // 4. PanelContainer
    if let Some(panel) = world.get::<PanelContainer>(container_entity).copied() {
        resolve_panel_layout(world, panel, container_rect, children);
        return true;
    }

    // 5. ScrollContainer
    if let Some(scroll) = world.get::<ScrollContainer>(container_entity).copied() {
        resolve_scroll_layout(world, container_entity, scroll, container_rect, children);
        return true;
    }

    // 6. CenterContainer
    if let Some(center) = world.get::<CenterContainer>(container_entity).copied() {
        resolve_center_layout(world, center, container_rect, children);
        return true;
    }

    // 7. AspectRatioContainer
    if let Some(arc) = world.get::<AspectRatioContainer>(container_entity).copied() {
        resolve_aspect_ratio_layout(world, arc, container_rect, children);
        return true;
    }

    // 8. SplitContainer
    if let Some(split) = world.get::<SplitContainer>(container_entity).copied() {
        resolve_split_layout(world, split, container_rect, children);
        return true;
    }

    false
}

fn set_child_resolved_rect(world: &mut World, child: Entity, global: Rect2d, parent_rect: Rect2d) {
    let local = Rect2d {
        x: global.x - parent_rect.x,
        y: global.y - parent_rect.y,
        width: global.width,
        height: global.height,
    };
    if let Some(r) = world.get_mut::<ResolvedRect>(child) {
        r.global = global;
        r.local = local;
    } else {
        world.add_component(child, ResolvedRect { global, local });
    }
}

fn resolve_box_layout(
    world: &mut World,
    box_container: BoxContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    let count = valid_children.len();
    if count == 0 {
        return;
    }

    let is_horizontal = box_container.orientation == Orientation::Horizontal;
    let separation = box_container.separation;
    let total_separation = separation * (count - 1) as f32;

    let total_available = if is_horizontal {
        parent_rect.width
    } else {
        parent_rect.height
    };

    // Calculate minimum sizes and expand weights
    let mut sum_fixed_min = 0.0f32;
    let mut total_stretch_ratio = 0.0f32;
    struct ChildStretch {
        will_stretch: bool,
        ratio: f32,
        min_s: f32,
        final_s: f32,
    }
    let mut items = Vec::with_capacity(count);

    for (_, c) in &valid_children {
        let (min_s, expand, ratio) = if is_horizontal {
            (
                c.custom_minimum_size.x,
                c.size_flags_horizontal.contains(SizeFlags::EXPAND),
                c.stretch_ratio,
            )
        } else {
            (
                c.custom_minimum_size.y,
                c.size_flags_vertical.contains(SizeFlags::EXPAND),
                c.stretch_ratio,
            )
        };
        if expand && ratio > 0.0 {
            total_stretch_ratio += ratio;
            items.push(ChildStretch {
                will_stretch: true,
                ratio,
                min_s,
                final_s: min_s,
            });
        } else {
            sum_fixed_min += min_s;
            items.push(ChildStretch {
                will_stretch: false,
                ratio,
                min_s,
                final_s: min_s,
            });
        }
    }

    let mut stretch_space = (total_available - total_separation - sum_fixed_min).max(0.0);

    // Iteratively distribute stretch space, clamping to min_s if needed
    loop {
        if total_stretch_ratio <= 0.0 {
            break;
        }
        let mut refit_needed = false;
        for item in &mut items {
            if item.will_stretch {
                let candidate = stretch_space * (item.ratio / total_stretch_ratio);
                if candidate < item.min_s {
                    item.will_stretch = false;
                    item.final_s = item.min_s;
                    stretch_space = (stretch_space - item.min_s).max(0.0);
                    total_stretch_ratio -= item.ratio;
                    refit_needed = true;
                    break;
                } else {
                    item.final_s = candidate;
                }
            }
        }
        if !refit_needed {
            break;
        }
    }

    let allocated_sizes: Vec<f32> = items.iter().map(|item| item.final_s).collect();

    // Compute initial offset from alignment
    let total_used: f32 = allocated_sizes.iter().sum::<f32>() + total_separation;
    let remaining = (total_available - total_used).max(0.0);
    let mut offset = match box_container.alignment {
        BoxAlignment::Begin => 0.0,
        BoxAlignment::Center => (remaining * 0.5).floor(),
        BoxAlignment::End => remaining,
    };

    // Fit each child
    for (i, (entity, control)) in valid_children.iter().enumerate() {
        let size = allocated_sizes[i];
        let available_cell = if is_horizontal {
            Rect2d {
                x: parent_rect.x + offset,
                y: parent_rect.y,
                width: size,
                height: parent_rect.height,
            }
        } else {
            Rect2d {
                x: parent_rect.x,
                y: parent_rect.y + offset,
                width: parent_rect.width,
                height: size,
            }
        };

        let fitted = fit_child_in_rect(control, available_cell);
        set_child_resolved_rect(world, *entity, fitted, parent_rect);
        offset += size + separation;
    }
}

fn resolve_grid_layout(
    world: &mut World,
    grid: GridContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    let count = valid_children.len();
    if count == 0 {
        return;
    }

    let cols = grid.columns.max(1);
    let rows = count.div_ceil(cols);

    let mut col_min_w = vec![0.0f32; cols];
    let mut row_min_h = vec![0.0f32; rows];
    let mut col_expand = vec![false; cols];
    let mut row_expand = vec![false; rows];

    for (idx, (_, c)) in valid_children.iter().enumerate() {
        let c_idx = idx % cols;
        let r_idx = idx / cols;

        col_min_w[c_idx] = col_min_w[c_idx].max(c.custom_minimum_size.x);
        row_min_h[r_idx] = row_min_h[r_idx].max(c.custom_minimum_size.y);

        if c.size_flags_horizontal.contains(SizeFlags::EXPAND) {
            col_expand[c_idx] = true;
        }
        if c.size_flags_vertical.contains(SizeFlags::EXPAND) {
            row_expand[r_idx] = true;
        }
    }

    // Allocate widths
    let total_h_sep = grid.h_separation * (cols - 1) as f32;
    let sum_col_min: f32 = col_min_w.iter().sum();
    let extra_w = (parent_rect.width - sum_col_min - total_h_sep).max(0.0);
    let num_col_expand = col_expand.iter().filter(|&&e| e).count();

    let mut col_widths = col_min_w.clone();
    if num_col_expand > 0 && extra_w > 0.0 {
        let add_w = extra_w / num_col_expand as f32;
        for c in 0..cols {
            if col_expand[c] {
                col_widths[c] += add_w;
            }
        }
    }

    // Allocate heights
    let total_v_sep = grid.v_separation * (rows - 1) as f32;
    let sum_row_min: f32 = row_min_h.iter().sum();
    let extra_h = (parent_rect.height - sum_row_min - total_v_sep).max(0.0);
    let num_row_expand = row_expand.iter().filter(|&&e| e).count();

    let mut row_heights = row_min_h.clone();
    if num_row_expand > 0 && extra_h > 0.0 {
        let add_h = extra_h / num_row_expand as f32;
        for r in 0..rows {
            if row_expand[r] {
                row_heights[r] += add_h;
            }
        }
    }

    // Compute column X offsets and row Y offsets
    let mut col_x = vec![0.0f32; cols];
    let mut cur_x = parent_rect.x;
    for c in 0..cols {
        col_x[c] = cur_x;
        cur_x += col_widths[c] + grid.h_separation;
    }

    let mut row_y = vec![0.0f32; rows];
    let mut cur_y = parent_rect.y;
    for r in 0..rows {
        row_y[r] = cur_y;
        cur_y += row_heights[r] + grid.v_separation;
    }

    // Fit each child
    for (idx, (entity, control)) in valid_children.iter().enumerate() {
        let c = idx % cols;
        let r = idx / cols;

        let cell = Rect2d {
            x: col_x[c],
            y: row_y[r],
            width: col_widths[c],
            height: row_heights[r],
        };

        let fitted = fit_child_in_rect(control, cell);
        set_child_resolved_rect(world, *entity, fitted, parent_rect);
    }
}

fn resolve_margin_layout(
    world: &mut World,
    margin: MarginContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    let inner_rect = Rect2d {
        x: parent_rect.x + margin.margin_left,
        y: parent_rect.y + margin.margin_top,
        width: (parent_rect.width - margin.margin_left - margin.margin_right).max(0.0),
        height: (parent_rect.height - margin.margin_top - margin.margin_bottom).max(0.0),
    };

    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    for (entity, control) in valid_children {
        let fitted = fit_child_in_rect(&control, inner_rect);
        set_child_resolved_rect(world, entity, fitted, parent_rect);
    }
}

fn resolve_panel_layout(
    world: &mut World,
    panel: PanelContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    let inner_rect = Rect2d {
        x: parent_rect.x + panel.padding_left,
        y: parent_rect.y + panel.padding_top,
        width: (parent_rect.width - panel.padding_left - panel.padding_right).max(0.0),
        height: (parent_rect.height - panel.padding_top - panel.padding_bottom).max(0.0),
    };

    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    for (entity, control) in valid_children {
        let fitted = fit_child_in_rect(&control, inner_rect);
        set_child_resolved_rect(world, entity, fitted, parent_rect);
    }
}

fn resolve_scroll_layout(
    world: &mut World,
    container_entity: Entity,
    scroll: ScrollContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    // Ensure clip_contents is enabled
    if let Some(ctl) = world.get_mut::<Control>(container_entity) {
        ctl.clip_contents = true;
    }

    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    for (entity, control) in valid_children {
        let child_w = control.custom_minimum_size.x.max(parent_rect.width);
        let child_h = control.custom_minimum_size.y.max(parent_rect.height);

        let global = Rect2d {
            x: parent_rect.x - scroll.scroll_horizontal,
            y: parent_rect.y - scroll.scroll_vertical,
            width: child_w,
            height: child_h,
        };

        set_child_resolved_rect(world, entity, global, parent_rect);
    }
}

fn resolve_center_layout(
    world: &mut World,
    center: CenterContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    for (entity, control) in valid_children {
        let min_s = control.custom_minimum_size;
        let ofs_x = if center.use_top_left {
            (-min_s.x * 0.5).floor()
        } else {
            ((parent_rect.width - min_s.x) * 0.5).floor()
        };
        let ofs_y = if center.use_top_left {
            (-min_s.y * 0.5).floor()
        } else {
            ((parent_rect.height - min_s.y) * 0.5).floor()
        };

        let cell = Rect2d {
            x: parent_rect.x + ofs_x,
            y: parent_rect.y + ofs_y,
            width: min_s.x,
            height: min_s.y,
        };

        let fitted = fit_child_in_rect(&control, cell);
        set_child_resolved_rect(world, entity, fitted, parent_rect);
    }
}

fn resolve_aspect_ratio_layout(
    world: &mut World,
    arc: AspectRatioContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    let ratio = if arc.ratio <= 0.0 { 1.0 } else { arc.ratio };
    let scale_factor = match arc.stretch_mode {
        AspectStretchMode::WidthControlsHeight => parent_rect.width / ratio,
        AspectStretchMode::HeightControlsWidth => parent_rect.height,
        AspectStretchMode::Fit => (parent_rect.width / ratio).min(parent_rect.height),
        AspectStretchMode::Cover => (parent_rect.width / ratio).max(parent_rect.height),
    };

    let target_w = ratio * scale_factor;
    let target_h = scale_factor;

    let align_x = match arc.alignment_horizontal {
        BoxAlignment::Begin => 0.0,
        BoxAlignment::Center => 0.5,
        BoxAlignment::End => 1.0,
    };
    let align_y = match arc.alignment_vertical {
        BoxAlignment::Begin => 0.0,
        BoxAlignment::Center => 0.5,
        BoxAlignment::End => 1.0,
    };

    let ofs_x = ((parent_rect.width - target_w) * align_x).floor();
    let ofs_y = ((parent_rect.height - target_h) * align_y).floor();

    let cell = Rect2d {
        x: parent_rect.x + ofs_x,
        y: parent_rect.y + ofs_y,
        width: target_w,
        height: target_h,
    };

    for (entity, control) in valid_children {
        let fitted = fit_child_in_rect(&control, cell);
        set_child_resolved_rect(world, entity, fitted, parent_rect);
    }
}

fn resolve_split_layout(
    world: &mut World,
    split: SplitContainer,
    parent_rect: Rect2d,
    children: &[Entity],
) {
    let valid_children: Vec<(Entity, Control)> = children
        .iter()
        .filter_map(|&e| world.get::<Control>(e).cloned().map(|c| (e, c)))
        .filter(|(_, c)| c.visible)
        .collect();

    if valid_children.is_empty() {
        return;
    }

    if valid_children.len() == 1 {
        let (entity, control) = &valid_children[0];
        let fitted = fit_child_in_rect(control, parent_rect);
        set_child_resolved_rect(world, *entity, fitted, parent_rect);
        return;
    }

    let is_horizontal = split.orientation == Orientation::Horizontal;
    let (c0_entity, c0_control) = &valid_children[0];
    let (c1_entity, c1_control) = &valid_children[1];

    if split.collapsed {
        if split.first_collapsed {
            let zero_rect = Rect2d {
                x: parent_rect.x,
                y: parent_rect.y,
                width: 0.0,
                height: 0.0,
            };
            set_child_resolved_rect(world, *c0_entity, zero_rect, parent_rect);
            let fitted1 = fit_child_in_rect(c1_control, parent_rect);
            set_child_resolved_rect(world, *c1_entity, fitted1, parent_rect);
        } else {
            let fitted0 = fit_child_in_rect(c0_control, parent_rect);
            set_child_resolved_rect(world, *c0_entity, fitted0, parent_rect);
            let zero_rect = Rect2d {
                x: parent_rect.x + parent_rect.width,
                y: parent_rect.y + parent_rect.height,
                width: 0.0,
                height: 0.0,
            };
            set_child_resolved_rect(world, *c1_entity, zero_rect, parent_rect);
        }
        return;
    }

    let total_space = if is_horizontal {
        parent_rect.width
    } else {
        parent_rect.height
    };
    let dragger = split.dragger_thickness;
    let available = (total_space - dragger).max(0.0);

    let min0 = if is_horizontal {
        c0_control.custom_minimum_size.x
    } else {
        c0_control.custom_minimum_size.y
    };
    let min1 = if is_horizontal {
        c1_control.custom_minimum_size.x
    } else {
        c1_control.custom_minimum_size.y
    };

    let base_split = available * 0.5 + split.split_offset;
    let clamped_split = base_split.clamp(min0, (available - min1).max(min0));

    if is_horizontal {
        let cell0 = Rect2d {
            x: parent_rect.x,
            y: parent_rect.y,
            width: clamped_split,
            height: parent_rect.height,
        };
        let cell1 = Rect2d {
            x: parent_rect.x + clamped_split + dragger,
            y: parent_rect.y,
            width: (available - clamped_split).max(0.0),
            height: parent_rect.height,
        };
        let fitted0 = fit_child_in_rect(c0_control, cell0);
        let fitted1 = fit_child_in_rect(c1_control, cell1);
        set_child_resolved_rect(world, *c0_entity, fitted0, parent_rect);
        set_child_resolved_rect(world, *c1_entity, fitted1, parent_rect);
    } else {
        let cell0 = Rect2d {
            x: parent_rect.x,
            y: parent_rect.y,
            width: parent_rect.width,
            height: clamped_split,
        };
        let cell1 = Rect2d {
            x: parent_rect.x,
            y: parent_rect.y + clamped_split + dragger,
            width: parent_rect.width,
            height: (available - clamped_split).max(0.0),
        };
        let fitted0 = fit_child_in_rect(c0_control, cell0);
        let fitted1 = fit_child_in_rect(c1_control, cell1);
        set_child_resolved_rect(world, *c0_entity, fitted0, parent_rect);
        set_child_resolved_rect(world, *c1_entity, fitted1, parent_rect);
    }
}
