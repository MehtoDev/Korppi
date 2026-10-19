//! Recursive ECS hierarchy traversal and layout resolution system.

use crate::containers::{
    panel_effective_padding, resolve_container_layout, AspectRatioContainer, BoxContainer,
    CenterContainer, DraggerVisibility, GridContainer, MarginContainer, Orientation,
    PanelContainer, ScrollContainer, ScrollMode, SplitContainer,
};
use crate::control::Control;
use crate::layout::rect::Rect2d;
use crate::layout::resolver::resolve_control_rect;
use crate::theme::ThemeOverride;
use crate::widgets::Label;
use korppi_ecs::{Component, Entity, World};
use korppi_math::Vec2;

/// Resolved geometric layout rectangle computed for a `Control` node.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ResolvedRect {
    /// Global screen-space axis-aligned bounding box.
    pub global: Rect2d,
    /// Local rectangle relative to parent control origin.
    pub local: Rect2d,
}

impl Component for ResolvedRect {}

/// Returns `true` if `entity` is alive, has a visible `Control`, all ancestor `Control`s are
/// visible, and it is not inside a collapsed pane of an ancestor `SplitContainer`.
#[must_use]
pub fn is_effectively_visible(world: &World, entity: Entity) -> bool {
    if !world.is_alive(entity) {
        return false;
    }
    let Some(ctl) = world.get::<Control>(entity) else {
        return false;
    };
    if !ctl.visible {
        return false;
    }

    let mut curr = entity;
    while let Some(parent) = world.parent_of(curr) {
        if let Some(p_ctl) = world.get::<Control>(parent) {
            if !p_ctl.visible {
                return false;
            }
        }
        if let Some(split) = world.get::<SplitContainer>(parent) {
            if split.collapsed {
                let mut vis_idx = 0usize;
                for &ch in world.children(parent) {
                    if world.get::<Control>(ch).is_some_and(|c| c.visible) {
                        if ch == curr {
                            if (split.first_collapsed && vis_idx == 0)
                                || (!split.first_collapsed && vis_idx == 1)
                            {
                                return false;
                            }
                            break;
                        }
                        vis_idx += 1;
                    }
                }
            }
        }
        curr = parent;
    }
    true
}

/// Computes the combined bottom-up minimum size for `entity`, aggregating container children,
/// stylebox margins, and `Control::custom_minimum_size`.
#[must_use]
pub fn compute_combined_minimum_size(world: &World, entity: Entity) -> Vec2 {
    let Some(ctl) = world.get::<Control>(entity) else {
        return Vec2::ZERO;
    };
    if !ctl.visible {
        return Vec2::ZERO;
    }

    let visible_child_mins: korppi_core::SmallVec<[Vec2; 16]> = world
        .children(entity)
        .iter()
        .copied()
        .filter(|&ch| world.get::<Control>(ch).is_some_and(|c| c.visible))
        .map(|ch| compute_combined_minimum_size(world, ch))
        .collect();

    let max_child_min = visible_child_mins
        .iter()
        .copied()
        .fold(Vec2::ZERO, Vec2::max);

    let container_min = if let Some(box_c) = world.get::<BoxContainer>(entity) {
        let n = visible_child_mins.len();
        if n == 0 {
            Vec2::ZERO
        } else {
            let sep = box_c.separation * (n - 1) as f32;
            match box_c.orientation {
                Orientation::Horizontal => {
                    let sum_w: f32 = visible_child_mins.iter().map(|v| v.x).sum();
                    Vec2::new(sum_w + sep, max_child_min.y)
                }
                Orientation::Vertical => {
                    let sum_h: f32 = visible_child_mins.iter().map(|v| v.y).sum();
                    Vec2::new(max_child_min.x, sum_h + sep)
                }
            }
        }
    } else if let Some(grid) = world.get::<GridContainer>(entity) {
        let n = visible_child_mins.len();
        if n == 0 {
            Vec2::ZERO
        } else {
            let cols = grid.columns.max(1);
            let rows = n.div_ceil(cols);
            let mut col_w: korppi_core::SmallVec<[f32; 16]> = korppi_core::smallvec![0.0f32; cols];
            let mut row_h: korppi_core::SmallVec<[f32; 16]> = korppi_core::smallvec![0.0f32; rows];
            for (idx, m) in visible_child_mins.iter().enumerate() {
                let c = idx % cols;
                let r = idx / cols;
                col_w[c] = col_w[c].max(m.x);
                row_h[r] = row_h[r].max(m.y);
            }
            let used_cols = cols.min(n);
            let w: f32 = col_w.iter().sum::<f32>() + grid.h_separation * (used_cols - 1) as f32;
            let h: f32 = row_h.iter().sum::<f32>() + grid.v_separation * (rows - 1) as f32;
            Vec2::new(w, h)
        }
    } else if let Some(margin) = world.get::<MarginContainer>(entity) {
        max_child_min
            + Vec2::new(
                margin.margin_left + margin.margin_right,
                margin.margin_top + margin.margin_bottom,
            )
    } else if let Some(panel) = world.get::<PanelContainer>(entity).copied() {
        let pad = panel_effective_padding(world, entity, panel);
        max_child_min + Vec2::new(pad[0] + pad[2], pad[1] + pad[3])
    } else if world.has::<CenterContainer>(entity) || world.has::<AspectRatioContainer>(entity) {
        max_child_min
    } else if let Some(scroll) = world.get::<ScrollContainer>(entity) {
        let w = if scroll.horizontal_scroll_mode == ScrollMode::Disabled {
            max_child_min.x
        } else {
            0.0
        };
        let h = if scroll.vertical_scroll_mode == ScrollMode::Disabled {
            max_child_min.y
        } else {
            0.0
        };
        Vec2::new(w, h)
    } else if let Some(split) = world.get::<SplitContainer>(entity) {
        match visible_child_mins.as_slice() {
            [] => Vec2::ZERO,
            [m0] => *m0,
            [m0, m1, ..] => {
                if split.collapsed {
                    if split.first_collapsed {
                        *m1
                    } else {
                        *m0
                    }
                } else {
                    let dragger = if split.dragger_visibility == DraggerVisibility::Hidden {
                        0.0
                    } else {
                        split.dragger_thickness
                    };
                    match split.orientation {
                        Orientation::Horizontal => Vec2::new(m0.x + m1.x + dragger, m0.y.max(m1.y)),
                        Orientation::Vertical => Vec2::new(m0.x.max(m1.x), m0.y + m1.y + dragger),
                    }
                }
            }
        }
    } else {
        Vec2::ZERO
    };

    let mut widget_min = Vec2::ZERO;
    if let Some(lbl) = world.get::<Label>(entity) {
        if !lbl.text().is_empty() {
            let max_chars = (0..lbl.line_count())
                .filter_map(|i| lbl.get_line(i))
                .map(|l| l.chars().count())
                .max()
                .unwrap_or(0);
            widget_min = widget_min.max(Vec2::new(
                max_chars as f32 * 8.0,
                lbl.line_count().max(1) as f32 * 16.0,
            ));
        }
    }
    if let Some(ov) = world.get::<ThemeOverride>(entity) {
        if let Some(sb) = ov.get_stylebox("panel") {
            widget_min += sb.get_minimum_size();
        }
    }

    ctl.custom_minimum_size.max(container_min).max(widget_min)
}

/// Returns a copy of `entity`'s `Control` with `custom_minimum_size` replaced by its combined
/// bottom-up minimum size.
#[must_use]
pub(crate) fn effective_control(world: &World, entity: Entity) -> Option<Control> {
    let mut ctl = world.get::<Control>(entity)?.clone();
    ctl.custom_minimum_size = compute_combined_minimum_size(world, entity);
    Some(ctl)
}

fn zero_subtree_rects(world: &mut World, entity: Entity) {
    if let Some(r) = world.get_mut::<ResolvedRect>(entity) {
        *r = ResolvedRect::default();
    } else if world.has::<Control>(entity) {
        world.add_component(entity, ResolvedRect::default());
    }
    let children: Vec<Entity> = world.children(entity).to_vec();
    for ch in children {
        zero_subtree_rects(world, ch);
    }
}

#[derive(Default)]
struct UiLayoutScratch {
    candidates: Vec<Entity>,
    roots: Vec<Entity>,
    children: Vec<Entity>,
}

thread_local! {
    static UI_LAYOUT_SCRATCH: std::cell::RefCell<UiLayoutScratch> =
        std::cell::RefCell::new(UiLayoutScratch::default());
}

/// Recursively resolves the UI layout for an entity and all its child control descendants.
pub fn resolve_control_tree(world: &mut World, entity: Entity, parent_rect: Rect2d) {
    UI_LAYOUT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.children.clear();
        resolve_control_tree_inner(world, entity, parent_rect, false, &mut scratch.children);
    });
}

fn resolve_control_tree_inner(
    world: &mut World,
    entity: Entity,
    parent_rect: Rect2d,
    skip_self_calc: bool,
    children_scratch: &mut Vec<Entity>,
) {
    if !is_effectively_visible(world, entity) {
        zero_subtree_rects(world, entity);
        return;
    }

    let resolved_rect = if skip_self_calc {
        if let Some(r) = world.get::<ResolvedRect>(entity) {
            r.global
        } else {
            return;
        }
    } else if let Some(control) = effective_control(world, entity) {
        let global = resolve_control_rect(&control, parent_rect);
        let local = Rect2d {
            x: global.x - parent_rect.x,
            y: global.y - parent_rect.y,
            width: global.width,
            height: global.height,
        };

        if let Some(r) = world.get_mut::<ResolvedRect>(entity) {
            r.global = global;
            r.local = local;
        } else {
            world.add_component(entity, ResolvedRect { global, local });
        }
        global
    } else {
        return;
    };

    let start = children_scratch.len();
    children_scratch.extend_from_slice(world.children(entity));
    let end = children_scratch.len();

    let is_container =
        resolve_container_layout(world, entity, resolved_rect, &children_scratch[start..end]);
    if is_container {
        for idx in start..end {
            let child = children_scratch[idx];
            if !is_effectively_visible(world, child) {
                zero_subtree_rects(world, child);
                continue;
            }
            let child_rect = world
                .get::<ResolvedRect>(child)
                .map(|r| r.global)
                .unwrap_or(resolved_rect);
            resolve_control_tree_inner(world, child, child_rect, true, children_scratch);
        }
    } else {
        for idx in start..end {
            let child = children_scratch[idx];
            resolve_control_tree_inner(world, child, resolved_rect, false, children_scratch);
        }
    }

    children_scratch.truncate(start);
}

/// System that identifies all root controls and recursively resolves the full UI hierarchy.
pub fn resolve_ui_layout_system(world: &mut World, screen_rect: Rect2d) {
    UI_LAYOUT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.candidates.clear();
        scratch.roots.clear();
        scratch.children.clear();

        {
            let mut query = world.query_mut::<(Entity, &Control)>();
            for (entity, _) in &mut query {
                scratch.candidates.push(entity);
            }
        }

        for i in 0..scratch.candidates.len() {
            let entity = scratch.candidates[i];
            let is_root = world
                .parent_of(entity)
                .is_none_or(|parent| !world.has::<Control>(parent));
            if is_root {
                scratch.roots.push(entity);
            }
        }

        scratch.roots.sort_by_key(|&e| {
            let z = world.get::<Control>(e).map_or(0, |c| c.z_index);
            (z, e.index())
        });

        let UiLayoutScratch {
            roots, children, ..
        } = &mut *scratch;
        for &root in roots.iter() {
            resolve_control_tree_inner(world, root, screen_rect, false, children);
        }
    });
}
