//! Front-to-back point-in-rect hit testing respecting Godot-style mouse filtering.

use crate::control::{Control, MouseFilter};
use crate::layout::hierarchy::ResolvedRect;
use korppi_ecs::{Entity, World};
use korppi_math::Vec2;

/// Collects entities with `Control` in front-to-back order (reverse depth-first rendering order).
pub fn collect_front_to_back_order(world: &World) -> Vec<Entity> {
    let mut roots = Vec::new();
    for entity in world.entities() {
        if world.has::<Control>(entity) {
            let is_root = if let Some(parent) = world.parent_of(entity) {
                !world.has::<Control>(parent)
            } else {
                true
            };
            if is_root {
                roots.push(entity);
            }
        }
    }

    let mut result = Vec::new();
    // Traverse roots in reverse order so later spawned root controls are tested first
    for &root in roots.iter().rev() {
        collect_node_front_to_back(world, root, &mut result);
    }
    result
}

fn collect_node_front_to_back(world: &World, entity: Entity, out: &mut Vec<Entity>) {
    // In GUI rendering, children are drawn on top of their parent.
    // Later children in the array are drawn on top of earlier children.
    // Therefore, front-to-back traversal visits children in reverse order, then self.
    for &child in world.children(entity).iter().rev() {
        collect_node_front_to_back(world, child, out);
    }
    out.push(entity);
}

/// Checks if an entity is clipped by an ancestor with `clip_contents = true`.
fn is_clipped_by_ancestor(world: &World, entity: Entity, position: Vec2) -> bool {
    let mut current = entity;
    while let Some(parent) = world.parent_of(current) {
        if let (Some(ctl), Some(rect)) = (
            world.get::<Control>(parent),
            world.get::<ResolvedRect>(parent),
        ) {
            if ctl.clip_contents && !rect.global.contains_point(position) {
                return true;
            }
        }
        current = parent;
    }
    false
}

/// Tests a screen position against controls in front-to-back Z-order.
///
/// Respects `MouseFilter`:
/// - `MouseFilter::Ignore`: The control is ignored and does not block controls beneath it.
/// - `MouseFilter::Pass`: The control receives events and passes them down to controls beneath it.
/// - `MouseFilter::Stop`: The control receives events and stops further propagation underneath.
pub fn hit_test_controls(world: &World, position: Vec2) -> Vec<Entity> {
    let order = collect_front_to_back_order(world);
    let mut hits = Vec::new();

    for entity in order {
        let (control, rect) = match (
            world.get::<Control>(entity),
            world.get::<ResolvedRect>(entity),
        ) {
            (Some(c), Some(r)) => (c, r),
            _ => continue,
        };

        if !control.visible {
            continue;
        }

        if is_clipped_by_ancestor(world, entity, position) {
            continue;
        }

        if rect.global.contains_point(position) {
            match control.mouse_filter {
                MouseFilter::Ignore => {
                    // Transparent to mouse interaction
                }
                MouseFilter::Pass => {
                    hits.push(entity);
                    // Continues propagation underneath
                }
                MouseFilter::Stop => {
                    hits.push(entity);
                    // Consumes event; stops further propagation
                    break;
                }
            }
        }
    }

    hits
}
