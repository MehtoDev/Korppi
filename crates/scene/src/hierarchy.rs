//! Scene hierarchy operations over the ECS's `ChildOf` relation (ADR-048).
//!
//! The hierarchy lives in the ECS: a child holds the exclusive pair `(ChildOf, parent)`, and
//! [`World::children`] lists a parent's children in the order they were attached, which UI
//! layout and draw order rely on. Despawning a parent despawns its descendants. These functions
//! add the scene's error type and traversals on top.

use korppi_ecs::{ChildOf, Entity, RelationshipError, World};
use std::collections::VecDeque;
use std::fmt;

/// Error type for scene hierarchy mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchyError {
    /// An entity cannot be made its own parent.
    SelfParenting(Entity),

    /// Cycle detected in entity hierarchy.
    CycleDetected {
        /// Attempted parent entity.
        parent: Entity,
        /// Attempted child entity.
        child: Entity,
    },

    /// The requested entity is not alive.
    EntityNotFound(Entity),
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SelfParenting(entity) => {
                write!(f, "An entity cannot be its own parent: {entity:?}")
            }
            Self::CycleDetected { parent, child } => write!(
                f,
                "Cycle detected: entity {parent:?} is already a descendant of {child:?}"
            ),
            Self::EntityNotFound(entity) => write!(f, "Entity is not alive: {entity:?}"),
        }
    }
}

impl std::error::Error for HierarchyError {}

/// Attaches `child` to `parent`, after the parent's existing children.
///
/// If `child` was attached to a different parent, it moves; attaching it to its own parent again
/// changes nothing.
///
/// # Errors
/// Returns [`HierarchyError::SelfParenting`] if `parent == child`.
/// Returns [`HierarchyError::CycleDetected`] if `parent` is a descendant of `child`.
/// Returns [`HierarchyError::EntityNotFound`] if either entity is not alive.
pub fn attach_child(
    world: &mut World,
    parent: Entity,
    child: Entity,
) -> Result<(), HierarchyError> {
    if !world.is_alive(parent) {
        return Err(HierarchyError::EntityNotFound(parent));
    }
    if !world.is_alive(child) {
        return Err(HierarchyError::EntityNotFound(child));
    }
    if parent == child {
        return Err(HierarchyError::SelfParenting(parent));
    }
    match world.try_add_pair::<ChildOf>(child, parent) {
        Ok(()) => Ok(()),
        Err(RelationshipError::CycleDetected) => {
            Err(HierarchyError::CycleDetected { parent, child })
        }
        Err(RelationshipError::EntityNotFound) => Err(HierarchyError::EntityNotFound(child)),
        Err(RelationshipError::TargetNotFound) => Err(HierarchyError::EntityNotFound(parent)),
        Err(RelationshipError::PayloadRequired) => unreachable!("ChildOf carries no data"),
    }
}

/// Detaches `child` from `parent`; does nothing if `parent` is not its parent.
///
/// # Errors
/// Returns [`HierarchyError::EntityNotFound`] if either entity is not alive.
pub fn detach_child(
    world: &mut World,
    parent: Entity,
    child: Entity,
) -> Result<(), HierarchyError> {
    if !world.is_alive(parent) {
        return Err(HierarchyError::EntityNotFound(parent));
    }
    if !world.is_alive(child) {
        return Err(HierarchyError::EntityNotFound(child));
    }
    world.remove_pair::<ChildOf>(child, parent);
    Ok(())
}

/// Returns the root of the hierarchy `entity` belongs to (itself if it has no parent).
#[must_use]
pub fn get_root(world: &World, entity: Entity) -> Entity {
    let mut current = entity;
    while let Some(parent) = world.parent_of(current) {
        current = parent;
    }
    current
}

/// Returns the ancestors of `entity`, from its parent up to the root.
#[must_use]
pub fn get_ancestors(world: &World, entity: Entity) -> Vec<Entity> {
    world.ancestors(entity).collect()
}

/// Returns the descendants of `entity` in breadth-first order, each parent's children in the
/// order they were attached.
#[must_use]
pub fn get_descendants(world: &World, entity: Entity) -> Vec<Entity> {
    let mut descendants = Vec::new();
    let mut queue: VecDeque<Entity> = world.children(entity).iter().copied().collect();
    while let Some(current) = queue.pop_front() {
        descendants.push(current);
        queue.extend(world.children(current).iter().copied());
    }
    descendants
}

/// Despawns `entity` and all of its descendants, detaching it from its parent.
pub fn despawn_recursive(world: &mut World, entity: Entity) {
    // `ChildOf` deletes the children of a despawned parent.
    world.despawn(entity);
}
