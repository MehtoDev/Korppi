//! Focus management, Tab cycling, and 4-way directional navigation.

use crate::control::{Control, FocusMode};
use crate::input::event::{UiEvent, UiReceiver};
use crate::layout::hierarchy::ResolvedRect;
use korppi_ecs::{Entity, World};
use korppi_math::Vec2;

/// Direction for directional focus navigation (D-pad, Arrow keys).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusDirection {
    /// Leftwards direction (-X).
    Left,
    /// Rightwards direction (+X).
    Right,
    /// Upwards direction (-Y).
    Up,
    /// Downwards direction (+Y).
    Down,
}

impl FocusDirection {
    /// Returns the unit direction vector in screen coordinates (+Y downwards).
    #[inline]
    pub fn vector(&self) -> Vec2 {
        match self {
            FocusDirection::Left => Vec2::new(-1.0, 0.0),
            FocusDirection::Right => Vec2::new(1.0, 0.0),
            FocusDirection::Up => Vec2::new(0.0, -1.0),
            FocusDirection::Down => Vec2::new(0.0, 1.0),
        }
    }
}

/// Manages currently focused UI control and handles focus transitions.
#[derive(Clone, Debug, Default)]
pub struct FocusManager {
    focused_control: Option<Entity>,
}

impl FocusManager {
    /// Returns the currently focused control entity, if any.
    #[inline]
    pub fn focused_control(&self) -> Option<Entity> {
        self.focused_control
    }

    /// Sets the focused control, dispatching `FocusLost` and `FocusGained` events.
    pub fn set_focus(&mut self, world: &mut World, target: Option<Entity>) {
        if self.focused_control == target {
            return;
        }

        if let Some(old) = self.focused_control {
            if let Some(receiver) = world.get_mut::<UiReceiver>(old) {
                receiver.push(UiEvent::FocusLost);
            }
        }

        self.focused_control = target;

        if let Some(new_target) = target {
            if let Some(receiver) = world.get_mut::<UiReceiver>(new_target) {
                receiver.push(UiEvent::FocusGained);
            }
        }
    }

    /// Clears the currently focused control.
    pub fn clear_focus(&mut self, world: &mut World) {
        self.set_focus(world, None);
    }

    /// Cycles focus forwards or backwards using Tab / Shift+Tab.
    pub fn navigate_tab(&mut self, world: &mut World, reverse: bool) {
        if let Some(current) = self.focused_control {
            if let Some(control) = world.get::<Control>(current) {
                if !reverse {
                    if let Some(next) = control.focus_next {
                        if is_tab_focusable(world, next) {
                            self.set_focus(world, Some(next));
                            return;
                        }
                    }
                } else if let Some(prev) = control.focus_previous {
                    if is_tab_focusable(world, prev) {
                        self.set_focus(world, Some(prev));
                        return;
                    }
                }
            }
        }

        let focusables = collect_tab_focusable_controls(world);
        if focusables.is_empty() {
            return;
        }

        let next_entity = match self.focused_control {
            Some(cur) => {
                if let Some(idx) = focusables.iter().position(|&e| e == cur) {
                    if !reverse {
                        focusables[(idx + 1) % focusables.len()]
                    } else {
                        focusables[(idx + focusables.len() - 1) % focusables.len()]
                    }
                } else if !reverse {
                    focusables[0]
                } else {
                    focusables[focusables.len() - 1]
                }
            }
            None => {
                if !reverse {
                    focusables[0]
                } else {
                    focusables[focusables.len() - 1]
                }
            }
        };

        self.set_focus(world, Some(next_entity));
    }

    /// Navigates focus in the given 4-way direction.
    pub fn navigate_direction(&mut self, world: &mut World, direction: FocusDirection) {
        let current = match self.focused_control {
            Some(e) => e,
            None => {
                let focusables = collect_tab_focusable_controls(world);
                if let Some(&first) = focusables.first() {
                    self.set_focus(world, Some(first));
                }
                return;
            }
        };

        // 1. Check explicit neighbor override
        if let Some(control) = world.get::<Control>(current) {
            let neighbor = match direction {
                FocusDirection::Left => control.focus_neighbor_left,
                FocusDirection::Right => control.focus_neighbor_right,
                FocusDirection::Up => control.focus_neighbor_top,
                FocusDirection::Down => control.focus_neighbor_bottom,
            };
            if let Some(target) = neighbor {
                if is_directional_focusable(world, target) {
                    self.set_focus(world, Some(target));
                    return;
                }
            }
        }

        // 2. Spatial geometric search
        let current_rect = match world.get::<ResolvedRect>(current) {
            Some(r) => r.global,
            None => return,
        };

        let current_center = current_rect.center();
        let dir_vec = direction.vector();

        let mut best_candidate: Option<Entity> = None;
        let mut best_score = f32::MAX;

        for entity in world.entities() {
            if entity == current || !is_directional_focusable(world, entity) {
                continue;
            }

            let cand_rect = match world.get::<ResolvedRect>(entity) {
                Some(r) => r.global,
                None => continue,
            };

            let cand_center = cand_rect.center();
            let delta = cand_center - current_center;

            // Must be forward in the target direction
            let forward = delta.dot(dir_vec);
            if forward <= 0.0 {
                continue;
            }

            // Lateral distance perpendicular to direction vector
            let lateral = (delta - dir_vec * forward).length();
            // Score prioritizing primary direction alignment
            let score = forward * forward + lateral * lateral * 2.0;

            if score < best_score {
                best_score = score;
                best_candidate = Some(entity);
            }
        }

        if let Some(best) = best_candidate {
            self.set_focus(world, Some(best));
        }
    }
}

fn collect_tab_focusable_controls(world: &World) -> Vec<Entity> {
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
    for root in roots {
        collect_pre_order(world, root, &mut result);
    }

    // Include any disconnected controls
    for entity in world.entities() {
        if world.has::<Control>(entity) && !result.contains(&entity) {
            result.push(entity);
        }
    }

    result
        .into_iter()
        .filter(|&e| is_tab_focusable(world, e))
        .collect()
}

fn collect_pre_order(world: &World, entity: Entity, out: &mut Vec<Entity>) {
    out.push(entity);
    for &child in world.children(entity) {
        collect_pre_order(world, child, out);
    }
}

fn is_tab_focusable(world: &World, entity: Entity) -> bool {
    if let Some(control) = world.get::<Control>(entity) {
        control.visible && control.focus_mode == FocusMode::All
    } else {
        false
    }
}

fn is_directional_focusable(world: &World, entity: Entity) -> bool {
    if let Some(control) = world.get::<Control>(entity) {
        control.visible && control.focus_mode != FocusMode::None
    } else {
        false
    }
}
