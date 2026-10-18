//! Hierarchical theme resolution cascade across the ECS scene graph.

use crate::theme::definition::{Theme, ThemeOverride};
use crate::theme::stylebox::StyleBox;
use korppi_ecs::{Entity, World};

/// Resolves a theme color for an entity, checking local overrides, ancestor themes, and default theme.
#[must_use]
pub fn get_theme_color(
    world: &World,
    entity: Entity,
    theme_type: &str,
    item: &str,
) -> Option<[f32; 4]> {
    // 1. Local ThemeOverride on entity
    if let Some(ov) = world.get::<ThemeOverride>(entity) {
        if let Some(color) = ov.get_color(item) {
            return Some(color);
        }
    }

    // 2. Ancestor Theme traversal
    let mut current = Some(entity);
    while let Some(cur_entity) = current {
        if let Some(theme) = world.get::<Theme>(cur_entity) {
            if let Some(color) = theme.get_color(theme_type, item) {
                return Some(color);
            }
        }
        current = world.parent_of(cur_entity);
    }

    // 3. Fallback to default engine dark theme
    Theme::dark().get_color(theme_type, item)
}

/// Resolves a numeric constant for an entity through the theme hierarchy.
#[must_use]
pub fn get_theme_constant(
    world: &World,
    entity: Entity,
    theme_type: &str,
    item: &str,
) -> Option<i32> {
    if let Some(ov) = world.get::<ThemeOverride>(entity) {
        if let Some(val) = ov.get_constant(item) {
            return Some(val);
        }
    }

    let mut current = Some(entity);
    while let Some(cur_entity) = current {
        if let Some(theme) = world.get::<Theme>(cur_entity) {
            if let Some(val) = theme.get_constant(theme_type, item) {
                return Some(val);
            }
        }
        current = world.parent_of(cur_entity);
    }

    Theme::dark().get_constant(theme_type, item)
}

/// Resolves a font size for an entity through the theme hierarchy.
#[must_use]
pub fn get_theme_font_size(
    world: &World,
    entity: Entity,
    theme_type: &str,
    item: &str,
) -> Option<u32> {
    if let Some(ov) = world.get::<ThemeOverride>(entity) {
        if let Some(size) = ov.get_font_size(item) {
            return Some(size);
        }
    }

    let mut current = Some(entity);
    while let Some(cur_entity) = current {
        if let Some(theme) = world.get::<Theme>(cur_entity) {
            if let Some(size) = theme.get_font_size(theme_type, item) {
                return Some(size);
            }
        }
        current = world.parent_of(cur_entity);
    }

    Theme::dark().get_font_size(theme_type, item)
}

/// Resolves a StyleBox for an entity and state through the theme hierarchy.
#[must_use]
pub fn get_theme_stylebox(
    world: &World,
    entity: Entity,
    theme_type: &str,
    item: &str,
) -> Option<StyleBox> {
    if let Some(ov) = world.get::<ThemeOverride>(entity) {
        if let Some(sb) = ov.get_stylebox(item) {
            return Some(sb.clone());
        }
    }

    let mut current = Some(entity);
    while let Some(cur_entity) = current {
        if let Some(theme) = world.get::<Theme>(cur_entity) {
            if let Some(sb) = theme.get_stylebox(theme_type, item) {
                return Some(sb.clone());
            }
        }
        current = world.parent_of(cur_entity);
    }

    Theme::dark().get_stylebox(theme_type, item).cloned()
}
