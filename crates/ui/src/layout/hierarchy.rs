//! Recursive ECS hierarchy traversal and layout resolution system.

use crate::containers::resolve_container_layout;
use crate::control::Control;
use crate::layout::rect::Rect2d;
use crate::layout::resolver::resolve_control_rect;
use korppi_ecs::{Component, Entity, World};

/// Resolved geometric layout rectangle computed for a `Control` node.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ResolvedRect {
    /// Global screen-space axis-aligned bounding box.
    pub global: Rect2d,
    /// Local rectangle relative to parent control origin.
    pub local: Rect2d,
}

impl Component for ResolvedRect {}

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
    let resolved_rect = if skip_self_calc {
        if let Some(r) = world.get::<ResolvedRect>(entity) {
            r.global
        } else {
            return;
        }
    } else if let Some(control) = world.get::<Control>(entity) {
        let global = resolve_control_rect(control, parent_rect);
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

        let UiLayoutScratch {
            roots, children, ..
        } = &mut *scratch;
        for &root in roots.iter() {
            resolve_control_tree_inner(world, root, screen_rect, false, children);
        }
    });
}
