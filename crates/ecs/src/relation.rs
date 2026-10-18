//! First-class entity relationship modeling, graph queries, and cascade policies.
//!
//! Inspired by Flecs (`.external_src/flecs`), relationships allow entities to form directed,
//! typed graph edges with other entities or components (e.g., `(ChildOf, ParentEntity)`,
//! `(IsA, Prefab)`). Relationships are stored as first-class component IDs directly within
//! ECS archetype tables, enabling cache-coherent columnar storage and zero-cost graph queries.

use crate::archetype::Archetype;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::Entity;
use crate::fast_hash::FastHashMap;
use std::fmt;

/// Policy dictating the action performed on source entities when the target entity is despawned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CleanupPolicy {
    /// Automatically remove the relationship pair from the source entity.
    #[default]
    Remove,
    /// Cascade delete the source entity that possessed the relationship pair.
    Delete,
    /// Remove the relationship pair from the source entity, as [`Remove`](Self::Remove) does.
    Clear,
}

/// Trait defining an entity relationship type.
///
/// Relationships can be zero-sized tags (like `ChildOf` or `IsA`) or carry payload data.
pub trait Relation: Component {
    /// Whether an entity can have at most one target for this relation.
    ///
    /// If `true`, adding a new pair with this relation will automatically replace any existing pair
    /// of the same relation on the source entity.
    const EXCLUSIVE: bool = false;

    /// The action to take on source entities when the target entity is despawned.
    const CLEANUP: CleanupPolicy = CleanupPolicy::Remove;

    /// Whether pairs of this relation must not form a cycle (checked when one is added).
    /// Exclusive relations are always checked; `IsA` is acyclic so that inheritance terminates.
    const ACYCLIC: bool = false;
}

/// Built-in hierarchical parent-child relationship.
///
/// Exclusive: `true` (an entity has at most one parent).
/// Cleanup: `CleanupPolicy::Delete` (despawning parent cascade-despawns children).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ChildOf;
impl Component for ChildOf {}
impl Relation for ChildOf {
    const EXCLUSIVE: bool = true;
    const CLEANUP: CleanupPolicy = CleanupPolicy::Delete;
}

/// Built-in prefab / template inheritance relationship.
///
/// Exclusive: `false` (an entity can inherit from multiple prefabs/traits).
/// Cleanup: `CleanupPolicy::Remove` (despawning prefab removes inheritance from instances).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IsA;
impl Component for IsA {}
impl Relation for IsA {
    const EXCLUSIVE: bool = false;
    const CLEANUP: CleanupPolicy = CleanupPolicy::Remove;
    const ACYCLIC: bool = true;
}

/// Errors returned by relationship operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationshipError {
    /// The operation would introduce a cycle in a hierarchical graph.
    CycleDetected,
    /// The source entity does not exist or has been despawned.
    EntityNotFound,
    /// The target entity does not exist or has been despawned.
    TargetNotFound,
    /// The relationship requires payload data and cannot be added via `add_pair`.
    PayloadRequired,
}

impl fmt::Display for RelationshipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "Relationship operation would introduce a cycle"),
            Self::EntityNotFound => write!(f, "Source entity does not exist or is dead"),
            Self::TargetNotFound => write!(f, "Target entity does not exist or is dead"),
            Self::PayloadRequired => write!(
                f,
                "Relation requires payload data; use set_pair instead of add_pair"
            ),
        }
    }
}

impl std::error::Error for RelationshipError {}

/// An outgoing relationship edge from a source entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutgoingEdge {
    /// Component ID of the relation type.
    pub relation: ComponentId,
    /// Target entity of the relationship.
    pub target: Entity,
}

impl Default for OutgoingEdge {
    fn default() -> Self {
        Self {
            relation: ComponentId(usize::MAX),
            target: Entity::new(u32::MAX, std::num::NonZeroU32::new(1).unwrap()),
        }
    }
}

/// Compact outgoing relationships stored densely per entity.
///
/// In 99.9% of game engine scenarios, an entity has 0, 1, or 2 relationships (e.g. `ChildOf`, `IsA`).
/// Storing up to 2 edges inline eliminates heap allocation and hash map lookups.
#[derive(Debug, Clone, Default)]
pub struct EntityRelations {
    inline: [OutgoingEdge; 2],
    len: u8,
    extra: Option<Vec<OutgoingEdge>>,
}

impl EntityRelations {
    /// Returns `true` if there are no outgoing relationships.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of outgoing relationships.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize + self.extra.as_ref().map_or(0, |v| v.len())
    }

    /// Returns an iterator over all outgoing relationship edges.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &OutgoingEdge> {
        let inline_slice = &self.inline[..self.len as usize];
        let extra_slice = self.extra.as_deref().unwrap_or(&[]);
        inline_slice.iter().chain(extra_slice.iter())
    }

    /// Returns the first target entity for `relation`, if present.
    #[inline]
    #[must_use]
    pub fn target_of(&self, relation: ComponentId) -> Option<Entity> {
        for edge in &self.inline[..self.len as usize] {
            if edge.relation == relation {
                return Some(edge.target);
            }
        }
        if let Some(extra) = &self.extra {
            for edge in extra {
                if edge.relation == relation {
                    return Some(edge.target);
                }
            }
        }
        None
    }

    /// Returns an iterator over all target entities for `relation`.
    #[inline]
    pub fn targets(&self, relation: ComponentId) -> impl Iterator<Item = Entity> + '_ {
        self.iter()
            .filter(move |e| e.relation == relation)
            .map(|e| e.target)
    }

    /// Returns `true` if this entity has a relationship pair `(relation, target)`.
    #[inline]
    #[must_use]
    pub fn has_pair(&self, relation: ComponentId, target: Entity) -> bool {
        for edge in &self.inline[..self.len as usize] {
            if edge.relation == relation && edge.target == target {
                return true;
            }
        }
        if let Some(extra) = &self.extra {
            for edge in extra {
                if edge.relation == relation && edge.target == target {
                    return true;
                }
            }
        }
        false
    }

    /// Returns `true` if this entity has any pair with `relation`.
    #[inline]
    #[must_use]
    pub fn has_relation(&self, relation: ComponentId) -> bool {
        for edge in &self.inline[..self.len as usize] {
            if edge.relation == relation {
                return true;
            }
        }
        if let Some(extra) = &self.extra {
            for edge in extra {
                if edge.relation == relation {
                    return true;
                }
            }
        }
        false
    }

    /// Adds an outgoing relationship edge `(relation, target)`.
    pub fn add(&mut self, relation: ComponentId, target: Entity) {
        if (self.len as usize) < self.inline.len() {
            self.inline[self.len as usize] = OutgoingEdge { relation, target };
            self.len += 1;
        } else {
            self.extra
                .get_or_insert_with(Vec::new)
                .push(OutgoingEdge { relation, target });
        }
    }

    /// Replaces an existing exclusive relation's target, returning the old target if present.
    pub fn replace_exclusive(&mut self, relation: ComponentId, target: Entity) -> Option<Entity> {
        for edge in &mut self.inline[..self.len as usize] {
            if edge.relation == relation {
                let old = edge.target;
                edge.target = target;
                return Some(old);
            }
        }
        if let Some(extra) = &mut self.extra {
            for edge in extra.iter_mut() {
                if edge.relation == relation {
                    let old = edge.target;
                    edge.target = target;
                    return Some(old);
                }
            }
        }
        self.add(relation, target);
        None
    }

    /// Removes a relationship pair `(relation, target)`.
    pub fn remove(&mut self, relation: ComponentId, target: Entity) -> bool {
        let inline_len = self.len as usize;
        for i in 0..inline_len {
            if self.inline[i].relation == relation && self.inline[i].target == target {
                if let Some(extra) = &mut self.extra {
                    if let Some(last) = extra.pop() {
                        self.inline[i] = last;
                        return true;
                    }
                }
                if i < inline_len - 1 {
                    self.inline[i] = self.inline[inline_len - 1];
                }
                self.len -= 1;
                return true;
            }
        }

        if let Some(extra) = &mut self.extra {
            if let Some(pos) = extra
                .iter()
                .position(|e| e.relation == relation && e.target == target)
            {
                extra.swap_remove(pos);
                return true;
            }
        }

        false
    }

    /// Clears all outgoing relationship edges.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        if let Some(extra) = &mut self.extra {
            extra.clear();
        }
    }
}

/// An incoming relation targeting an entity, recording policy and source entities.
#[derive(Debug, Clone, Default)]
pub struct IncomingRelation {
    /// Relation component ID.
    pub relation: ComponentId,
    /// Associated cleanup policy.
    pub policy: CleanupPolicy,
    /// Source entities pointing to the target, in the order they were added.
    pub sources: Vec<Entity>,
}

/// Incoming relationships pointing to a target entity.
#[derive(Debug, Clone, Default)]
pub struct IncomingTarget {
    /// List of incoming relations and their source entities.
    pub relations: Vec<IncomingRelation>,
    active_count: usize,
}

impl IncomingTarget {
    /// Returns `true` if there are no incoming relationships.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Returns the active incoming relations.
    #[inline]
    pub fn active_relations(&self) -> &[IncomingRelation] {
        &self.relations[..self.active_count]
    }

    /// Returns the slice of source entities for `relation`.
    #[inline]
    #[must_use]
    pub fn sources(&self, relation: ComponentId) -> &[Entity] {
        for rel in &self.relations[..self.active_count] {
            if rel.relation == relation {
                return rel.sources.as_slice();
            }
        }
        &[]
    }

    /// Records an incoming relationship from `source` under `relation` with `policy`, after the
    /// sources recorded before it. The caller guarantees `source` is not recorded yet.
    pub fn add_source(&mut self, relation: ComponentId, policy: CleanupPolicy, source: Entity) {
        for rel in &mut self.relations[..self.active_count] {
            if rel.relation == relation {
                rel.policy = policy;
                rel.sources.push(source);
                return;
            }
        }
        if self.active_count < self.relations.len() {
            let rel = &mut self.relations[self.active_count];
            rel.relation = relation;
            rel.policy = policy;
            rel.sources.clear();
            rel.sources.push(source);
        } else {
            self.relations.push(IncomingRelation {
                relation,
                policy,
                sources: vec![source],
            });
        }
        self.active_count += 1;
    }

    /// Removes an incoming relationship from `source` under `relation`, keeping the order of the
    /// other sources.
    pub fn remove_source(&mut self, relation: ComponentId, source: Entity) -> bool {
        for rel in &mut self.relations[..self.active_count] {
            if rel.relation == relation {
                let Some(position) = rel.sources.iter().position(|&s| s == source) else {
                    return false;
                };
                rel.sources.remove(position);
                return true;
            }
        }
        false
    }

    /// Clears all incoming relationships, retaining allocated vectors for recycled entities.
    #[inline]
    pub fn clear(&mut self) {
        for i in 0..self.active_count {
            self.relations[i].sources.clear();
        }
        self.active_count = 0;
    }
}

/// Bidirectional index maintaining relationship connections across entities.
#[derive(Debug, Default)]
pub struct RelationshipIndex {
    /// Dense array of outgoing relations indexed by source entity index
    outgoing: Vec<EntityRelations>,
    /// Dense array of incoming relations indexed by target entity index
    incoming: Vec<IncomingTarget>,
    /// relation_id -> CleanupPolicy
    cleanup_policies: FastHashMap<ComponentId, CleanupPolicy>,
    /// relation_id -> exclusive
    exclusive_flags: FastHashMap<ComponentId, bool>,
    /// Depth-first search stack of the cycle check.
    cycle_stack: Vec<Entity>,
    /// Per entity index: the check that last visited it.
    cycle_seen: Vec<u32>,
    cycle_stamp: u32,
}

impl RelationshipIndex {
    /// Creates a new, empty `RelationshipIndex`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_outgoing(&mut self, entity: Entity) -> &mut EntityRelations {
        let idx = entity.index() as usize;
        if idx >= self.outgoing.len() {
            self.outgoing.resize_with(idx + 1, EntityRelations::default);
        }
        &mut self.outgoing[idx]
    }

    fn ensure_incoming(&mut self, entity: Entity) -> &mut IncomingTarget {
        let idx = entity.index() as usize;
        if idx >= self.incoming.len() {
            self.incoming.resize_with(idx + 1, IncomingTarget::default);
        }
        &mut self.incoming[idx]
    }

    /// Records a relationship pair between `source` and `target`.
    pub fn insert(
        &mut self,
        source: Entity,
        relation: ComponentId,
        target: Entity,
        exclusive: bool,
        cleanup: CleanupPolicy,
    ) -> Option<Entity> {
        self.cleanup_policies.entry(relation).or_insert(cleanup);
        self.exclusive_flags.entry(relation).or_insert(exclusive);

        let mut replaced_target = None;
        if exclusive {
            let out = self.ensure_outgoing(source);
            if let Some(old_target) = out.replace_exclusive(relation, target) {
                if old_target == target {
                    return None;
                }
                replaced_target = Some(old_target);
                let old_idx = old_target.index() as usize;
                if let Some(in_target) = self.incoming.get_mut(old_idx) {
                    in_target.remove_source(relation, source);
                }
            }
        } else {
            let out = self.ensure_outgoing(source);
            if out.has_pair(relation, target) {
                return None;
            }
            out.add(relation, target);
        }

        let in_target = self.ensure_incoming(target);
        in_target.add_source(relation, cleanup, source);

        replaced_target
    }

    /// Removes a relationship pair between `source` and `target`.
    pub fn remove(&mut self, source: Entity, relation: ComponentId, target: Entity) -> bool {
        let s_idx = source.index() as usize;
        let removed = if let Some(out) = self.outgoing.get_mut(s_idx) {
            out.remove(relation, target)
        } else {
            false
        };

        if removed {
            let t_idx = target.index() as usize;
            if let Some(in_target) = self.incoming.get_mut(t_idx) {
                in_target.remove_source(relation, source);
            }
        }

        removed
    }

    /// Returns the single target of relation for `source`, or `None` if absent.
    #[must_use]
    pub fn target_of(&self, source: Entity, relation: ComponentId) -> Option<Entity> {
        self.outgoing
            .get(source.index() as usize)
            .and_then(|out| out.target_of(relation))
    }

    /// Returns all target entities for `source` under `relation`.
    pub fn targets(
        &self,
        source: Entity,
        relation: ComponentId,
    ) -> impl Iterator<Item = Entity> + '_ {
        self.outgoing
            .get(source.index() as usize)
            .into_iter()
            .flat_map(move |out| out.targets(relation))
    }

    /// Returns all source entities targeting `target` under `relation`.
    #[must_use]
    pub fn sources(&self, target: Entity, relation: ComponentId) -> &[Entity] {
        self.incoming
            .get(target.index() as usize)
            .map_or(&[], |in_target| in_target.sources(relation))
    }

    /// Returns `true` if `source` has a relationship `(relation, target)`.
    #[must_use]
    pub fn has_pair(&self, source: Entity, relation: ComponentId, target: Entity) -> bool {
        self.outgoing
            .get(source.index() as usize)
            .is_some_and(|out| out.has_pair(relation, target))
    }

    /// Returns `true` if `source` possesses any pair with `relation`.
    #[must_use]
    pub fn has_relation(&self, source: Entity, relation: ComponentId) -> bool {
        self.outgoing
            .get(source.index() as usize)
            .is_some_and(|out| out.has_relation(relation))
    }

    /// Returns the cleanup policy for `relation`.
    #[must_use]
    pub fn cleanup_policy(&self, relation: ComponentId) -> CleanupPolicy {
        self.cleanup_policies
            .get(&relation)
            .copied()
            .unwrap_or(CleanupPolicy::Remove)
    }

    /// Returns the `IncomingTarget` descriptor for `target` if it has active incoming relationships.
    #[inline]
    #[must_use]
    pub fn incoming_target(&self, target: Entity) -> Option<&IncomingTarget> {
        self.incoming
            .get(target.index() as usize)
            .filter(|in_target| !in_target.is_empty())
    }

    /// Returns `true` if `entity` has any incoming or outgoing relationships.
    #[must_use]
    pub fn has_any_relationships(&self, entity: Entity) -> bool {
        let idx = entity.index() as usize;
        let has_out = self.outgoing.get(idx).is_some_and(|r| !r.is_empty());
        let has_in = self.incoming.get(idx).is_some_and(|r| !r.is_empty());
        has_out || has_in
    }

    /// Returns `true` if `target` has any incoming relationships.
    #[must_use]
    #[inline]
    pub fn has_incoming(&self, target: Entity) -> bool {
        self.incoming
            .get(target.index() as usize)
            .is_some_and(|in_target| !in_target.is_empty())
    }

    /// Clears the outgoing edges of `entity`, removing it as a source from every target for which
    /// `target_dies` is `false` (the incoming lists of dying targets are cleared whole).
    #[inline]
    pub fn clear_outgoing(&mut self, entity: Entity, target_dies: impl Fn(Entity) -> bool) {
        let e_idx = entity.index() as usize;
        if let Some(out) = self.outgoing.get_mut(e_idx) {
            if !out.is_empty() {
                for edge in out.iter() {
                    if !target_dies(edge.target) {
                        if let Some(in_target) = self.incoming.get_mut(edge.target.index() as usize)
                        {
                            in_target.remove_source(edge.relation, entity);
                        }
                    }
                }
                out.clear();
            }
        }
    }

    /// Clears all incoming edges for `entity`.
    #[inline]
    pub fn clear_incoming(&mut self, entity: Entity) {
        if let Some(in_target) = self.incoming.get_mut(entity.index() as usize) {
            if !in_target.is_empty() {
                in_target.clear();
            }
        }
    }

    /// Checks if setting `target` as a relation target for `source` would create a cycle: if
    /// `source` is `target` or can be reached from it along `relation`. Allocates nothing once
    /// its buffers have grown.
    pub fn would_create_cycle(
        &mut self,
        source: Entity,
        target: Entity,
        relation: ComponentId,
    ) -> bool {
        if source == target {
            return true;
        }
        if self
            .outgoing
            .get(target.index() as usize)
            .is_none_or(|out| !out.has_relation(relation))
        {
            return false;
        }
        self.cycle_stamp = self.cycle_stamp.wrapping_add(1);
        if self.cycle_stamp == 0 {
            self.cycle_seen.fill(0);
            self.cycle_stamp = 1;
        }
        let stamp = self.cycle_stamp;
        let mut stack = std::mem::take(&mut self.cycle_stack);
        stack.clear();
        stack.push(target);
        let mut found = false;
        while let Some(current) = stack.pop() {
            if current == source {
                found = true;
                break;
            }
            let index = current.index() as usize;
            if self.cycle_seen.len() <= index {
                self.cycle_seen.resize(index + 1, 0);
            }
            if self.cycle_seen[index] == stamp {
                continue;
            }
            self.cycle_seen[index] = stamp;
            if let Some(out) = self.outgoing.get(index) {
                stack.extend(
                    out.iter()
                        .filter(|edge| edge.relation == relation)
                        .map(|edge| edge.target),
                );
            }
        }
        self.cycle_stack = stack;
        found
    }
}

crate::query::archetype_filter!(
    /// Filter matching only archetypes that possess any pair of relation `R`.
    WithPair<R: Relation>,
    |registry: &mut ComponentRegistry| registry.register::<R>(),
    |id, archetype: &Archetype| archetype.has_relation(id)
);

crate::query::archetype_filter!(
    /// Filter matching only archetypes that possess any pair with relation `R` (wildcard).
    WithRelation<R: Relation>,
    |registry: &mut ComponentRegistry| registry.register::<R>(),
    |id, archetype: &Archetype| archetype.has_relation(id)
);

crate::query::archetype_filter!(
    /// Filter matching only archetypes that DO NOT possess any pair of relation `R`.
    WithoutRelation<R: Relation>,
    |registry: &mut ComponentRegistry| registry.register::<R>(),
    |id, archetype: &Archetype| !archetype.has_relation(id)
);
