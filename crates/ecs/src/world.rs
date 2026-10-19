//! Central ECS World orchestrating entities, archetypes, and queries.

use crate::archetype::Archetype;
use crate::bundle::Bundle;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::{Entity, EntityLocation, EntityManager};
use crate::fast_hash::FastHashMap;
use crate::observer::{EventKind, Observers};
use crate::query::{QueryIter, QueryState, WorldQuery};
use crate::relation::{CleanupPolicy, Relation, RelationshipError, RelationshipIndex};
use crate::storage::Column;
use crate::tick::{ChangeTicks, Tick, CHECK_TICK_THRESHOLD, MAX_CHANGE_AGE};
use std::any::{Any, TypeId};
use std::sync::atomic::{AtomicU64, Ordering};

/// Source of [`World::id`] values.
static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);

/// The central coordinator managing all entities, archetype tables, component data, and relationships.
pub struct World {
    /// Unique among the worlds of the process; binds a [`QueryState`] to its world.
    id: u64,
    /// Changes whenever an archetype slot is freed or reused, so cached matches must be redone.
    archetype_epoch: u64,
    /// The tick changes are stamped with; each query run takes it and advances it.
    change_tick: Tick,
    /// `change_tick` when all ticks were last clamped to [`MAX_CHANGE_AGE`].
    last_tick_check: Tick,
    entities: EntityManager,
    components: ComponentRegistry,
    archetypes: Vec<Archetype>,
    archetype_map: FastHashMap<Vec<ComponentId>, usize>,
    relationships: RelationshipIndex,
    free_archetypes: Vec<usize>,
    /// Per pair component ID (by index): the archetypes holding that pair.
    pair_archetypes: Vec<Vec<usize>>,
    /// The archetype each bundle type spawns into.
    bundle_archetypes: FastHashMap<TypeId, usize>,
    /// Keys of freed `archetype_map` entries, reused for new archetypes.
    free_keys: Vec<Vec<ComponentId>>,
    /// Component list of an archetype being looked up from an edge.
    scratch_components: Vec<ComponentId>,
    /// Typed singletons ([`insert_resource`](Self::insert_resource)).
    resources: FastHashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Observers and the events waiting for them ([`observe`](Self::observe)).
    observers: Observers,
    /// Prefabs waiting for an instance in [`instantiate`](Self::instantiate), with the instance
    /// to attach theirs to.
    pub(crate) instantiate_queue: Vec<(Entity, Option<Entity>)>,
    despawn: DespawnScratch,
}

/// Buffers a cascading despawn reuses, so that it allocates nothing once they have grown.
#[derive(Default)]
struct DespawnScratch {
    /// Entities to delete, in deletion order; each is marked in the entity manager while queued.
    queue: Vec<Entity>,
    /// Queued entities that are the target of at least one relation.
    dead_targets: Vec<Entity>,
    /// Relations to one dead target whose sources survive it (`Remove`, `Clear`).
    relations: Vec<ComponentId>,
    /// Sources of one relation to one dead target.
    sources: Vec<Entity>,
    /// Archetypes holding a pair to a dead target, sorted and deduplicated.
    dying_archetypes: Vec<usize>,
    /// Per archetype ID: listed in `dying_archetypes`.
    dying: Vec<bool>,
    /// Pair IDs one dead target releases.
    pair_ids: Vec<ComponentId>,
    /// Edges of the archetype being freed.
    edges: Vec<(ComponentId, usize)>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Creates a new, empty ECS `World`.
    #[must_use]
    pub fn new() -> Self {
        let mut world = Self {
            id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            archetype_epoch: 0,
            change_tick: Tick::new(1),
            last_tick_check: Tick::new(1),
            entities: EntityManager::new(),
            components: ComponentRegistry::new(),
            archetypes: Vec::new(),
            archetype_map: FastHashMap::default(),
            relationships: RelationshipIndex::new(),
            free_archetypes: Vec::new(),
            pair_archetypes: Vec::new(),
            bundle_archetypes: FastHashMap::default(),
            free_keys: Vec::new(),
            scratch_components: Vec::new(),
            resources: FastHashMap::default(),
            observers: Observers::default(),
            instantiate_queue: Vec::new(),
            despawn: DespawnScratch::default(),
        };

        // Create empty archetype at index 0
        let empty_id = world.archetype_for(&[]);
        debug_assert_eq!(empty_id, 0);
        world
    }

    /// Returns the total number of alive entities in the world.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.alive_count()
    }

    /// Returns the total number of archetype tables in the world.
    #[must_use]
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Returns this world's identifier, unique among the worlds of the process.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Returns a counter that changes whenever an archetype slot is freed or reused; appending new
    /// archetypes leaves it unchanged. [`QueryState`] re-matches every archetype when it changes.
    #[must_use]
    pub const fn archetype_epoch(&self) -> u64 {
        self.archetype_epoch
    }

    /// Returns the tick that changes made now are stamped with.
    #[must_use]
    pub const fn change_tick(&self) -> Tick {
        self.change_tick
    }

    /// Starts a query run: returns the current change tick as the run's `this_run` and advances
    /// the counter, so that later changes are newer than the run.
    pub(crate) fn increment_change_tick(&mut self) -> Tick {
        let this_run = self.change_tick;
        self.advance_change_tick(1);
        this_run
    }

    /// Advances the change tick by `runs`, as that many query runs would, clamping old ticks
    /// whenever [`CHECK_TICK_THRESHOLD`] ticks have passed since the last clamp.
    pub fn advance_change_tick(&mut self, runs: u32) {
        let mut remaining = runs;
        while remaining > 0 {
            // Clamping relies on at most `CHECK_TICK_THRESHOLD` ticks passing between checks.
            let step = remaining.min(CHECK_TICK_THRESHOLD);
            remaining -= step;
            self.change_tick = Tick::new(self.change_tick.get().wrapping_add(step));
            if self
                .change_tick
                .get()
                .wrapping_sub(self.last_tick_check.get())
                >= CHECK_TICK_THRESHOLD
            {
                self.check_change_ticks();
            }
        }
    }

    /// Clamps every component tick to at most [`MAX_CHANGE_AGE`] before the current tick, so
    /// that comparisons stay correct when the counter wraps around.
    fn check_change_ticks(&mut self) {
        let now = self.change_tick;
        for archetype in &mut self.archetypes {
            for column in archetype.columns_mut() {
                column.clamp_ticks(now);
            }
        }
        self.last_tick_check = now;
    }

    /// Checks if the given entity handle is currently alive.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Iterates over all active entities across all archetypes.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.archetypes
            .iter()
            .flat_map(|arch| arch.entities().iter().copied())
    }

    /// The archetype holding exactly `components` (sorted, without duplicates), created on first
    /// use.
    fn archetype_for(&mut self, components: &[ComponentId]) -> usize {
        debug_assert!(components.windows(2).all(|w| w[0] < w[1]));
        if let Some(&id) = self.archetype_map.get(components) {
            return id;
        }
        let id = self.create_archetype(components);
        let mut key = self.free_keys.pop().unwrap_or_default();
        key.clear();
        key.extend_from_slice(components);
        self.archetype_map.insert(key, id);
        id
    }

    /// Creates the archetype for `components` in a new or freed slot, and records the pairs it
    /// holds.
    ///
    /// A freed slot whose columns have the same layouts, in the same order, keeps its buffers, so
    /// that a parent replacing a dead one (with a recycled pair ID) allocates nothing.
    fn create_archetype(&mut self, components: &[ComponentId]) -> usize {
        let fitting = self.free_archetypes.iter().rposition(|&slot| {
            self.archetypes[slot].fits(components.iter().map(|&c| self.info(c).layout()))
        });
        let id = if let Some(position) = fitting {
            let slot = self.free_archetypes.remove(position);
            let (archetypes, registry) = (&mut self.archetypes, &self.components);
            archetypes[slot].reuse(
                components,
                |c| *registry.get_info(c).unwrap(),
                |c| registry.id_to_pair(c),
            );
            self.archetype_epoch += 1;
            slot
        } else {
            let columns = components
                .iter()
                .map(|&component| Column::new(*self.info(component)))
                .collect();
            let pairs = components
                .iter()
                .filter_map(|&component| self.components.id_to_pair(component))
                .collect();
            let archetype = |slot| Archetype::with_pairs(slot, components.to_vec(), pairs, columns);
            if let Some(slot) = self.free_archetypes.pop() {
                self.archetypes[slot] = archetype(slot);
                self.archetype_epoch += 1;
                slot
            } else {
                let slot = self.archetypes.len();
                self.archetypes.push(archetype(slot));
                slot
            }
        };

        let observed = components.iter().fold(0, |mask, &c| {
            let rel_mask = self
                .components
                .id_to_pair(c)
                .map_or(0, |(rel, _)| self.observers.mask(rel));
            mask | self.observers.mask(c) | rel_mask
        });
        self.archetypes[id].set_observed(observed);
        // `Prefab` and `IsA` are looked up rather than registered up front, so that the world
        // assigns component IDs in the order the game registers them.
        if let Some(prefab) = self.prefab_id() {
            self.archetypes[id].set_prefab(components.binary_search(&prefab).is_ok());
        }
        if self
            .isa_id()
            .is_some_and(|isa| self.archetypes[id].has_relation(isa))
        {
            self.refresh_inherited(id);
        }
        for &component in components {
            if self.components.id_to_pair(component).is_some() {
                let index = component.index();
                if self.pair_archetypes.len() <= index {
                    self.pair_archetypes.resize_with(index + 1, Vec::new);
                }
                self.pair_archetypes[index].push(id);
            }
        }
        id
    }

    fn info(&self, component: ComponentId) -> &crate::component::ComponentInfo {
        self.components
            .get_info(component)
            .expect("Unregistered component info")
    }

    /// Spawns a new entity populated with the given bundle of components.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.entities.allocate();
        self.place_spawned(entity, bundle);
        entity
    }

    /// Writes `bundle` into its archetype for the allocated, unplaced `entity`, records the
    /// spawn's events and runs their observers.
    fn place_spawned<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        let arch_id = self.bundle_archetype::<B>();
        let arch = &mut self.archetypes[arch_id];
        let row = arch.push_entity(entity);
        bundle.write_to_archetype(arch, &self.components, self.change_tick);
        self.entities.set_location(
            entity,
            EntityLocation {
                archetype_id: arch_id,
                row,
            },
        );
        let arch = &self.archetypes[arch_id];
        if arch.observed() & (EventKind::Add.bit() | EventKind::Set.bit()) != 0 {
            for kind in [EventKind::Add, EventKind::Set] {
                for &component in arch.components() {
                    self.observers.record(kind, component, entity);
                }
            }
            self.flush_observers();
        }
    }

    /// Places `entity`, reserved by commands and allocated since, with `bundle`.
    pub(crate) fn spawn_reserved<B: Bundle>(&mut self, entity: Entity, bundle: B) {
        self.entities.flush_reserved();
        assert!(
            self.entities.is_unplaced(entity),
            "{entity:?} was not reserved by commands"
        );
        self.place_spawned(entity, bundle);
    }

    /// Allocates the entities commands reserved, so that commands can place them.
    pub(crate) fn flush_reserved_entities(&mut self) {
        self.entities.flush_reserved();
    }

    /// Frees any entities reserved by commands that were never placed into an archetype.
    pub(crate) fn reclaim_unplaced_entities(&mut self) {
        self.entities.reclaim_unplaced();
    }

    /// The observers, for the observer module.
    pub(crate) fn observers_mut(&mut self) -> &mut Observers {
        &mut self.observers
    }

    /// Registers `T` and returns its ID.
    pub(crate) fn register_component<T: Component>(&mut self) -> ComponentId {
        self.components.register::<T>()
    }

    /// Recomputes each archetype's mask of observed events.
    pub(crate) fn refresh_observed_masks(&mut self) {
        let components = &self.components;
        let observers = &self.observers;
        for archetype in &mut self.archetypes {
            let mask = archetype.components().iter().fold(0, |mask, &c| {
                let rel_mask = components
                    .id_to_pair(c)
                    .map_or(0, |(rel, _)| observers.mask(rel));
                mask | observers.mask(c) | rel_mask
            });
            archetype.set_observed(mask);
        }
    }

    /// The entity records, for parameters that reserve entities or look up locations.
    pub(crate) const fn entity_manager(&self) -> &EntityManager {
        &self.entities
    }

    /// The archetype and row of `entity`, if it is alive.
    #[must_use]
    pub fn entity_location(&self, entity: Entity) -> Option<EntityLocation> {
        self.entities.location(entity)
    }

    // =========================================================================
    // Resources
    // =========================================================================

    /// Stores `value` as the world's resource of type `R`, replacing any previous one.
    pub fn insert_resource<R: Send + Sync + 'static>(&mut self, value: R) {
        self.resources.insert(TypeId::of::<R>(), Box::new(value));
    }

    /// Returns the resource of type `R`, if there is one.
    #[must_use]
    pub fn resource<R: Send + Sync + 'static>(&self) -> Option<&R> {
        self.resources.get(&TypeId::of::<R>())?.downcast_ref()
    }

    /// Returns the resource of type `R` mutably, if there is one.
    pub fn resource_mut<R: Send + Sync + 'static>(&mut self) -> Option<&mut R> {
        self.resources.get_mut(&TypeId::of::<R>())?.downcast_mut()
    }

    /// Removes and returns the resource of type `R`.
    pub fn remove_resource<R: Send + Sync + 'static>(&mut self) -> Option<R> {
        let boxed = self.resources.remove(&TypeId::of::<R>())?;
        Some(
            *boxed
                .downcast::<R>()
                .expect("resource stored under its own TypeId"),
        )
    }

    /// Returns `true` if the world has a resource of type `R`.
    #[must_use]
    pub fn contains_resource<R: Send + Sync + 'static>(&self) -> bool {
        self.resources.contains_key(&TypeId::of::<R>())
    }

    /// The resource of type `R` of the world behind `world`, without borrowing anything else.
    ///
    /// # Safety
    /// `world` must be valid, and no other reference to that resource may be used while the
    /// result is.
    pub(crate) unsafe fn resource_raw<'a, R: Send + Sync + 'static>(
        world: *mut World,
    ) -> Option<&'a mut R> {
        // SAFETY: the caller guarantees `world` is valid; only the `resources` field is borrowed,
        // and only for the lookup.
        let resources = unsafe { &mut (*world).resources };
        let value: *mut R = resources.get_mut(&TypeId::of::<R>())?.downcast_mut::<R>()?;
        // SAFETY: the value lives in its own heap allocation, which the caller borrows
        // exclusively.
        Some(unsafe { &mut *value })
    }

    /// The archetype a bundle of type `B` spawns into, cached after the first spawn.
    fn bundle_archetype<B: Bundle>(&mut self) -> usize {
        if let Some(&id) = self.bundle_archetypes.get(&TypeId::of::<B>()) {
            return id;
        }
        let components = B::register(&mut self.components);
        let id = self.archetype_for(&components);
        self.bundle_archetypes.insert(TypeId::of::<B>(), id);
        id
    }

    /// Spawns an empty entity without components.
    pub fn spawn_empty(&mut self) -> Entity {
        self.spawn(())
    }

    /// Despawns an entity, recursively applying cascade cleanup policies to connected relationships.
    ///
    /// Returns `false` if `entity` is not alive. Entities are deleted in a deterministic order:
    /// breadth first from `entity`, each target's incoming relations in the order they were
    /// first added, and each relation's sources in the order they were added.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };
        if self.relationships.has_any_relationships(entity) {
            self.despawn_cascade(entity);
        } else {
            self.entities.free(entity);
            self.record_removals(loc.archetype_id, entity);
            if let Some(swapped) = self.archetypes[loc.archetype_id].swap_remove(loc.row) {
                self.entities.set_location(swapped, loc);
            }
        }
        self.flush_observers();
        true
    }

    /// Records an `OnRemove` event for each observed component of archetype `archetype` that
    /// `entity` loses.
    #[inline]
    fn record_removals(&mut self, archetype: usize, entity: Entity) {
        let arch = &self.archetypes[archetype];
        if arch.observed() & EventKind::Remove.bit() != 0 {
            for &component in arch.components() {
                self.observers.record(EventKind::Remove, component, entity);
                if let Some((rel, _)) = self.components.id_to_pair(component) {
                    self.observers.record(EventKind::Remove, rel, entity);
                }
            }
        }
    }

    /// Deletes `root` and, transitively, the sources of every `Delete` relation to a deleted
    /// entity; removes the pairs of surviving sources (`Remove`, `Clear`); then frees the
    /// archetypes and pair IDs of the dead targets (ECS-07).
    fn despawn_cascade(&mut self, root: Entity) {
        let mut scratch = std::mem::take(&mut self.despawn);
        let DespawnScratch {
            queue,
            dead_targets,
            relations,
            sources,
            dying_archetypes,
            dying,
            pair_ids,
            edges,
        } = &mut scratch;
        queue.clear();
        dead_targets.clear();
        dying_archetypes.clear();

        // 1. Breadth-first collection of the entities to delete.
        queue.push(root);
        self.entities.mark(root);
        let mut next = 0;
        while next < queue.len() {
            let current = queue[next];
            next += 1;
            let Some(incoming) = self.relationships.incoming_target(current) else {
                continue;
            };
            dead_targets.push(current);
            for relation in incoming.active_relations() {
                if relation.policy != CleanupPolicy::Delete {
                    continue;
                }
                for &source in &relation.sources {
                    if self.entities.mark(source) {
                        queue.push(source);
                    }
                }
            }
        }

        // 2. Surviving sources of `Remove` and `Clear` relations lose their pairs to dead targets.
        for &target in dead_targets.iter() {
            relations.clear();
            if let Some(incoming) = self.relationships.incoming_target(target) {
                relations.extend(
                    incoming
                        .active_relations()
                        .iter()
                        .filter(|relation| relation.policy != CleanupPolicy::Delete)
                        .map(|relation| relation.relation),
                );
            }
            for &relation in relations.iter() {
                let Some(pair_id) = self.components.pair_to_id(relation, target) else {
                    continue;
                };
                sources.clear();
                sources.extend_from_slice(self.relationships.sources(target, relation));
                for &source in sources.iter() {
                    if !self.entities.is_marked(source) {
                        self.remove_component_by_id_raw(source, pair_id);
                        self.relationships.remove(source, relation, target);
                        self.observers.record(EventKind::Remove, pair_id, source);
                        self.observers.record(EventKind::Remove, relation, source);
                    }
                }
            }
        }

        // 3. Every archetype holding a pair to a dead target now holds only entities being deleted.
        for &target in dead_targets.iter() {
            for pair_id in self.components.pairs_for_target(target) {
                if let Some(archetypes) = self.pair_archetypes.get(pair_id.index()) {
                    dying_archetypes.extend_from_slice(archetypes);
                }
            }
        }
        dying_archetypes.sort_unstable();
        dying_archetypes.dedup();
        if dying.len() < self.archetypes.len() {
            dying.resize(self.archetypes.len(), false);
        }
        for &archetype in dying_archetypes.iter() {
            dying[archetype] = true;
        }

        // 4. Delete the queued entities. Rows in dying archetypes are dropped with the whole table.
        for &e in queue.iter() {
            let entities = &self.entities;
            self.relationships
                .clear_outgoing(e, |target| entities.is_marked(target));
            let Some(loc) = self.entities.free(e) else {
                continue;
            };
            self.record_removals(loc.archetype_id, e);
            if !dying[loc.archetype_id] {
                if let Some(swapped) = self.archetypes[loc.archetype_id].swap_remove(loc.row) {
                    self.entities.set_location(swapped, loc);
                }
            }
        }
        for &target in dead_targets.iter() {
            self.relationships.clear_incoming(target);
        }

        // 5. Free the dying archetypes, then release the dead targets' pair IDs for reuse: no
        //    archetype holds them any more.
        for &archetype in dying_archetypes.iter() {
            let entities = self.archetypes[archetype].entities();
            assert!(
                entities.iter().all(|&e| !self.entities.is_alive(e)),
                "a live entity holds a pair to a dead target"
            );
            self.free_archetype(archetype, edges);
            dying[archetype] = false;
        }
        for &target in dead_targets.iter() {
            self.components
                .unregister_all_pairs_for_target_into(target, pair_ids);
            for pair_id in pair_ids.iter() {
                self.pair_archetypes[pair_id.index()].clear();
            }
        }

        for &e in queue.iter() {
            self.entities.unmark(e);
        }
        self.despawn = scratch;
    }

    /// Frees archetype `archetype_id`: drops its rows, unlinks every edge into it, and makes its
    /// slot reusable. `edges` is scratch space.
    fn free_archetype(&mut self, archetype_id: usize, edges: &mut Vec<(ComponentId, usize)>) {
        if let Some((key, _)) = self
            .archetype_map
            .remove_entry(self.archetypes[archetype_id].components())
        {
            self.free_keys.push(key);
        }
        self.bundle_archetypes
            .retain(|_, &mut id| id != archetype_id);
        for &component in self.archetypes[archetype_id].components() {
            if let Some(holders) = self.pair_archetypes.get_mut(component.index()) {
                holders.retain(|&id| id != archetype_id);
            }
        }

        // Each edge out of the archetype has a reverse edge in its neighbour pointing back here.
        edges.clear();
        edges.extend(
            self.archetypes[archetype_id]
                .edges()
                .remove
                .iter()
                .map(|(&c, &dst)| (c, dst)),
        );
        for &(component, neighbour) in edges.iter() {
            let add = &mut self.archetypes[neighbour].edges_mut().add;
            if add.get(&component) == Some(&archetype_id) {
                add.remove(&component);
            }
        }
        edges.clear();
        edges.extend(
            self.archetypes[archetype_id]
                .edges()
                .add
                .iter()
                .map(|(&c, &dst)| (c, dst)),
        );
        for &(component, neighbour) in edges.iter() {
            let remove = &mut self.archetypes[neighbour].edges_mut().remove;
            if remove.get(&component) == Some(&archetype_id) {
                remove.remove(&component);
            }
        }

        self.archetypes[archetype_id].release();
        self.free_archetypes.push(archetype_id);
        self.archetype_epoch += 1;
    }

    /// Returns an immutable reference to an entity's component, or `None` if not present.
    #[must_use]
    ///
    /// An instance of a prefab that does not own `T` reads the prefab's value (see
    /// [`crate::prefab`]).
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let loc = self.entities.location(entity)?;
        let id = self.components.id_of::<T>()?;
        let arch = &self.archetypes[loc.archetype_id];
        let Some(col) = arch.column(id) else {
            return self.inherited_component(loc.archetype_id, id);
        };
        // SAFETY: `id` corresponds to `T: Component` in `self.components`, and `loc.row` is valid for `arch`.
        unsafe { Some(col.get::<T>(loc.row)) }
    }

    /// Returns an immutable reference to a component owned directly by `entity`, ignoring any
    /// inherited prefab components.
    #[must_use]
    pub fn get_owned<T: Component>(&self, entity: Entity) -> Option<&T> {
        let loc = self.entities.location(entity)?;
        let id = self.components.id_of::<T>()?;
        let arch = &self.archetypes[loc.archetype_id];
        let col = arch.column(id)?;
        // SAFETY: `id` corresponds to `T: Component` in `self.components`, and `loc.row` is valid for `arch`.
        unsafe { Some(col.get::<T>(loc.row)) }
    }

    /// Returns a mutable reference to an entity's component, or `None` if not present. Only
    /// owned components are reachable: an instance never writes its prefab's value.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let loc = self.entities.location(entity)?;
        let id = self.components.id_of::<T>()?;
        let arch = &mut self.archetypes[loc.archetype_id];
        let col = arch.column_mut(id)?;
        col.mark_changed(loc.row, self.change_tick);
        // SAFETY: `id` corresponds to `T: Component` in `self.components`, and `loc.row` is valid for `arch`.
        unsafe { Some(col.get_mut::<T>(loc.row)) }
    }

    /// Returns `true` if the entity owns component `T` or inherits it from a prefab.
    #[must_use]
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };
        let Some(id) = self.components.id_of::<T>() else {
            return false;
        };
        self.archetypes[loc.archetype_id].has_or_inherits(id)
    }

    /// Returns `true` if the entity owns or inherits the component corresponding to `comp_id`.
    #[must_use]
    pub fn has_component_id(&self, entity: Entity, comp_id: ComponentId) -> bool {
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };
        self.archetypes[loc.archetype_id].has_or_inherits(comp_id)
    }

    /// Returns an immutable reference to a component by its raw [`ComponentId`], or `None` if the
    /// entity lacks it or `T` is not its type.
    ///
    /// An instance of a prefab that does not own `comp_id` reads the prefab's value.
    #[must_use]
    pub fn get_by_id<T: 'static>(&self, entity: Entity, comp_id: ComponentId) -> Option<&T> {
        let loc = self.entities.location(entity)?;
        let Some(col) = self.archetypes[loc.archetype_id].column(comp_id) else {
            return self.inherited_component(loc.archetype_id, comp_id);
        };
        if col.info().type_id() != Some(TypeId::of::<T>()) {
            return None;
        }
        // SAFETY: the column stores `T` (checked against the column's own `TypeId`), and `loc.row`
        // is the entity's row in this archetype.
        unsafe { Some(col.get::<T>(loc.row)) }
    }

    /// Returns a mutable reference to a component by its raw [`ComponentId`], or `None` if the
    /// entity lacks it or `T` is not its type.
    pub fn get_mut_by_id<T: 'static>(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
    ) -> Option<&mut T> {
        let loc = self.entities.location(entity)?;
        let col = self.archetypes[loc.archetype_id].column_mut(comp_id)?;
        if col.info().type_id() != Some(TypeId::of::<T>()) {
            return None;
        }
        col.mark_changed(loc.row, self.change_tick);
        // SAFETY: the column stores `T` (checked against the column's own `TypeId`), and `loc.row`
        // is the entity's row in this archetype.
        unsafe { Some(col.get_mut::<T>(loc.row)) }
    }

    /// Adds `component` to `entity` under the raw [`ComponentId`] `added_id`, moving the entity to
    /// the archetype with that component, or overwrites the value the entity already has.
    ///
    /// # Panics
    /// If `entity` is dead, `added_id` is not registered or not `T`'s ID, or `added_id` is a
    /// relationship pair: pairs are added with [`add_pair`](Self::add_pair) or
    /// [`set_pair`](Self::set_pair), which keep the relationship index in step.
    pub fn add_component_by_id<T: 'static>(
        &mut self,
        entity: Entity,
        added_id: ComponentId,
        component: T,
    ) {
        let info = self
            .components
            .get_info(added_id)
            .expect("Component ID not registered");
        assert_eq!(
            info.type_id(),
            Some(TypeId::of::<T>()),
            "Component type mismatch for ComponentId {added_id:?}"
        );
        assert!(
            self.components.id_to_pair(added_id).is_none(),
            "ComponentId {added_id:?} is a relationship pair; add it with add_pair or set_pair"
        );
        self.insert_by_id(entity, added_id, component);
        self.flush_observers();
    }

    /// Dynamically attaches a component to an entity, migrating it to the target archetype.
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        let added_id = self.components.register::<T>();
        self.insert_by_id(entity, added_id, component);
        self.flush_observers();
    }

    /// Removes the component `comp_id` from `entity` and returns it, or `None` if the entity lacks
    /// it, `T` is not its type, or `comp_id` is a relationship pair (removed with
    /// [`remove_pair`](Self::remove_pair) or [`remove_pair_data`](Self::remove_pair_data)).
    pub fn remove_component_by_id<T: 'static>(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
    ) -> Option<T> {
        if self.components.id_to_pair(comp_id).is_some() {
            return None;
        }
        let removed = self.take_by_id(entity, comp_id);
        self.flush_observers();
        removed
    }

    /// Dynamically removes a component from an entity, migrating it to the target archetype.
    pub fn remove_component<T: Component>(&mut self, entity: Entity) -> Option<T> {
        let removed_id = self.components.id_of::<T>()?;
        let removed = self.take_by_id(entity, removed_id);
        self.flush_observers();
        removed
    }

    /// Removes a component from an entity by its [`ComponentId`], dropping it in place.
    ///
    /// Crate-private: it also removes pairs, which the caller must remove from the relationship
    /// index too.
    pub(crate) fn remove_component_by_id_raw(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
    ) -> bool {
        let Some(loc) = self.entities.location(entity) else {
            return false;
        };
        let Some(rem_idx) = self.archetypes[loc.archetype_id].column_index(comp_id) else {
            return false;
        };
        let dst = self.remove_target(loc.archetype_id, comp_id);
        let (src_arch, dst_arch) = get_two_mut(&mut self.archetypes, loc.archetype_id, dst);
        let dst_row = dst_arch.push_entity(entity);
        Archetype::move_matching_columns(src_arch, loc.row, dst_arch, self.change_tick);
        src_arch
            .column_by_index_mut(rem_idx)
            .unwrap()
            .swap_remove(loc.row);
        self.finish_move(entity, loc, dst, dst_row);
        true
    }

    /// Adds or overwrites a component or pair `T` under `added_id`; the caller has checked that
    /// `added_id` is registered with `T`'s type.
    fn insert_by_id<T: 'static>(&mut self, entity: Entity, added_id: ComponentId, component: T) {
        let loc = self.entities.location(entity).expect("Entity not found");
        if self.archetypes[loc.archetype_id].has_component(added_id) {
            let slot = self
                .get_mut_by_id::<T>(entity, added_id)
                .expect("column type differs from the registry");
            *slot = component;
            self.observers.record(EventKind::Set, added_id, entity);
            if let Some((rel, _)) = self.components.id_to_pair(added_id) {
                self.observers.record(EventKind::Set, rel, entity);
            }
            return;
        }

        let dst = self.add_target(loc.archetype_id, added_id);
        // The destination column must store `T` before anything moves.
        assert_eq!(
            self.archetypes[dst]
                .column(added_id)
                .unwrap()
                .info()
                .type_id(),
            Some(TypeId::of::<T>()),
            "column type differs from the registry"
        );
        let tick = self.change_tick;
        // SAFETY: the column stores `T` (checked above), so pushing `component` appends one valid
        // value.
        unsafe {
            self.move_with_added(entity, loc, added_id, dst, |column| {
                column.push(component, tick);
            });
        }
        self.observers.record(EventKind::Add, added_id, entity);
        self.observers.record(EventKind::Set, added_id, entity);
        if let Some((rel, _)) = self.components.id_to_pair(added_id) {
            self.observers.record(EventKind::Add, rel, entity);
            self.observers.record(EventKind::Set, rel, entity);
        }
    }

    /// Moves `entity` (at `location`) to archetype `dst`, which is its archetype plus `added`,
    /// carrying every shared column over, and lets `push` append the value of `added`.
    ///
    /// # Safety
    /// `push` must append exactly one valid value of `added` to the column it is given.
    pub(crate) unsafe fn move_with_added(
        &mut self,
        entity: Entity,
        location: EntityLocation,
        added: ComponentId,
        dst: usize,
        push: impl FnOnce(&mut Column),
    ) {
        let (src_arch, dst_arch) = get_two_mut(&mut self.archetypes, location.archetype_id, dst);
        let added_idx = dst_arch.column_index(added).unwrap();
        let dst_row = dst_arch.push_entity(entity);
        Archetype::move_matching_columns(src_arch, location.row, dst_arch, self.change_tick);
        push(dst_arch.column_by_index_mut(added_idx).unwrap());
        self.finish_move(entity, location, dst, dst_row);
    }

    /// The archetype `entity`'s archetype leads to when `added` is added.
    pub(crate) fn archetype_with_added(
        &mut self,
        location: EntityLocation,
        added: ComponentId,
    ) -> usize {
        self.add_target(location.archetype_id, added)
    }

    /// The component registry, mutably, for registrations other modules make.
    pub(crate) fn registry_mut(&mut self) -> &mut ComponentRegistry {
        &mut self.components
    }

    /// Removes the component `comp_id` from `entity` and returns it if `T` is its type.
    fn take_by_id<T: 'static>(&mut self, entity: Entity, comp_id: ComponentId) -> Option<T> {
        let loc = self.entities.location(entity)?;
        let src = &self.archetypes[loc.archetype_id];
        let rem_idx = src.column_index(comp_id)?;
        if src.column_by_index(rem_idx)?.info().type_id() != Some(TypeId::of::<T>()) {
            return None;
        }
        let dst = self.remove_target(loc.archetype_id, comp_id);
        let (src_arch, dst_arch) = get_two_mut(&mut self.archetypes, loc.archetype_id, dst);
        let dst_row = dst_arch.push_entity(entity);
        Archetype::move_matching_columns(src_arch, loc.row, dst_arch, self.change_tick);
        let removed_col = src_arch.column_by_index_mut(rem_idx).unwrap();
        // SAFETY: the column stores `T` (checked above) and `loc.row` is below its length.
        let (removed, _) = unsafe { removed_col.swap_remove_and_take::<T>(loc.row) };
        self.finish_move(entity, loc, dst, dst_row);
        self.observers.record(EventKind::Remove, comp_id, entity);
        if let Some((rel, _)) = self.components.id_to_pair(comp_id) {
            self.observers.record(EventKind::Remove, rel, entity);
        }
        Some(removed)
    }

    /// The archetype reached from `src` by adding `id`, created and linked both ways on first use.
    fn add_target(&mut self, src: usize, id: ComponentId) -> usize {
        if let Some(&dst) = self.archetypes[src].edges().add.get(&id) {
            return dst;
        }
        let mut components = std::mem::take(&mut self.scratch_components);
        components.clear();
        components.extend_from_slice(self.archetypes[src].components());
        if let Err(position) = components.binary_search(&id) {
            components.insert(position, id);
        }
        let dst = self.archetype_for(&components);
        self.scratch_components = components;
        self.archetypes[src].edges_mut().add.insert(id, dst);
        self.archetypes[dst].edges_mut().remove.insert(id, src);
        dst
    }

    /// The archetype reached from `src` by removing `id`, created and linked both ways on first use.
    fn remove_target(&mut self, src: usize, id: ComponentId) -> usize {
        if let Some(&dst) = self.archetypes[src].edges().remove.get(&id) {
            return dst;
        }
        let mut components = std::mem::take(&mut self.scratch_components);
        components.clear();
        components.extend(
            self.archetypes[src]
                .components()
                .iter()
                .copied()
                .filter(|&c| c != id),
        );
        let dst = self.archetype_for(&components);
        self.scratch_components = components;
        self.archetypes[src].edges_mut().remove.insert(id, dst);
        self.archetypes[dst].edges_mut().add.insert(id, src);
        dst
    }

    /// Completes a move of `entity` from `from` to row `dst_row` of archetype `dst`, after every
    /// column of the source has dropped the entity's row: swap-removes it from the source's entity
    /// list and updates both locations.
    fn finish_move(&mut self, entity: Entity, from: EntityLocation, dst: usize, dst_row: usize) {
        let src = &mut self.archetypes[from.archetype_id];
        src.entities_mut().swap_remove(from.row);
        if let Some(&swapped) = src.entities().get(from.row) {
            self.entities.set_location(swapped, from);
        }
        self.entities.set_location(
            entity,
            EntityLocation {
                archetype_id: dst,
                row: dst_row,
            },
        );
        if self.entities.is_isa_target(entity) {
            self.refresh_instances_of(entity);
        }
    }

    /// The ID of the [`Prefab`](crate::prefab::Prefab) tag, once registered.
    pub(crate) fn prefab_id(&self) -> Option<ComponentId> {
        self.components.id_of::<crate::prefab::Prefab>()
    }

    /// The ID of the [`IsA`](crate::relation::IsA) relation, once registered.
    pub(crate) fn isa_id(&self) -> Option<ComponentId> {
        self.components.id_of::<crate::relation::IsA>()
    }

    /// The archetypes holding pair `pair`.
    pub(crate) fn pair_archetypes_of(&self, pair: ComponentId) -> &[usize] {
        self.pair_archetypes
            .get(pair.index())
            .map_or(&[], Vec::as_slice)
    }

    /// Records that cached query matches must be redone.
    pub(crate) fn bump_archetype_epoch(&mut self) {
        self.archetype_epoch += 1;
    }

    // =========================================================================
    // Relationship API (Flecs Alignment)
    // =========================================================================

    /// Attempts to add a relationship pair `(R, target)` to `entity`.
    ///
    /// Validates entity liveness and checks for cycle creation on exclusive relations.
    pub fn try_add_pair<R: Relation>(
        &mut self,
        entity: Entity,
        target: Entity,
    ) -> Result<(), RelationshipError> {
        if !self.is_alive(entity) {
            return Err(RelationshipError::EntityNotFound);
        }
        if !self.is_alive(target) {
            return Err(RelationshipError::TargetNotFound);
        }
        if std::mem::size_of::<R>() != 0 {
            return Err(RelationshipError::PayloadRequired);
        }

        let rel_id = self.components.register::<R>();

        if (R::EXCLUSIVE || R::ACYCLIC)
            && self
                .relationships
                .would_create_cycle(entity, target, rel_id)
        {
            return Err(RelationshipError::CycleDetected);
        }

        let pair_id = self.components.register_pair(rel_id, target);
        if Some(rel_id) == self.isa_id() {
            self.entities.set_isa_target(target);
        }

        if R::EXCLUSIVE {
            if let Some(replaced) =
                self.relationships
                    .insert(entity, rel_id, target, R::EXCLUSIVE, R::CLEANUP)
            {
                if replaced != target {
                    if let Some(old_pair_id) = self.components.pair_to_id(rel_id, replaced) {
                        self.remove_component_by_id_raw(entity, old_pair_id);
                        self.observers
                            .record(EventKind::Remove, old_pair_id, entity);
                        self.observers.record(EventKind::Remove, rel_id, entity);
                    }
                }
            }
        } else {
            self.relationships
                .insert(entity, rel_id, target, R::EXCLUSIVE, R::CLEANUP);
        }

        // SAFETY: size_of::<R>() is checked above to be 0, so zeroed bytes represent a valid instance of zero-sized type R.
        let value: R = unsafe { std::mem::zeroed() };
        self.insert_by_id(entity, pair_id, value);
        self.flush_observers();

        Ok(())
    }

    /// Adds a relationship pair `(R, target)` to `entity`.
    ///
    /// # Panics
    /// Panics if either entity is dead or if a cycle is detected.
    pub fn add_pair<R: Relation>(&mut self, entity: Entity, target: Entity) {
        self.try_add_pair::<R>(entity, target)
            .expect("Failed to add relationship pair");
    }

    /// Sets a relationship pair `(R, target)` with payload data on `entity`.
    pub fn set_pair<R: Relation>(&mut self, entity: Entity, target: Entity, data: R) {
        self.try_set_pair(entity, target, data)
            .expect("Failed to set relationship pair");
    }

    /// Attempts to set a relationship pair `(R, target)` with payload data on `entity`.
    pub fn try_set_pair<R: Relation>(
        &mut self,
        entity: Entity,
        target: Entity,
        data: R,
    ) -> Result<(), RelationshipError> {
        if !self.is_alive(entity) {
            return Err(RelationshipError::EntityNotFound);
        }
        if !self.is_alive(target) {
            return Err(RelationshipError::TargetNotFound);
        }

        let rel_id = self.components.register::<R>();

        if (R::EXCLUSIVE || R::ACYCLIC)
            && self
                .relationships
                .would_create_cycle(entity, target, rel_id)
        {
            return Err(RelationshipError::CycleDetected);
        }

        let pair_id = self.components.register_pair(rel_id, target);
        if Some(rel_id) == self.isa_id() {
            self.entities.set_isa_target(target);
        }

        if R::EXCLUSIVE {
            if let Some(replaced) =
                self.relationships
                    .insert(entity, rel_id, target, R::EXCLUSIVE, R::CLEANUP)
            {
                if replaced != target {
                    if let Some(old_pair_id) = self.components.pair_to_id(rel_id, replaced) {
                        self.remove_component_by_id_raw(entity, old_pair_id);
                        self.observers
                            .record(EventKind::Remove, old_pair_id, entity);
                        self.observers.record(EventKind::Remove, rel_id, entity);
                    }
                }
            }
        } else {
            self.relationships
                .insert(entity, rel_id, target, R::EXCLUSIVE, R::CLEANUP);
        }

        self.insert_by_id(entity, pair_id, data);
        self.flush_observers();

        Ok(())
    }

    /// Returns an immutable reference to the data of pair `(R, target)` on `entity`.
    #[must_use]
    pub fn get_pair<R: Relation>(&self, entity: Entity, target: Entity) -> Option<&R> {
        if !self.is_alive(target) {
            return None;
        }
        let rel_id = self.components.id_of::<R>()?;
        let pair_id = self.components.pair_to_id(rel_id, target)?;
        self.get_by_id::<R>(entity, pair_id)
    }

    /// Returns a mutable reference to the data of pair `(R, target)` on `entity`.
    pub fn get_pair_mut<R: Relation>(&mut self, entity: Entity, target: Entity) -> Option<&mut R> {
        if !self.is_alive(target) {
            return None;
        }
        let rel_id = self.components.id_of::<R>()?;
        let pair_id = self.components.pair_to_id(rel_id, target)?;
        self.get_mut_by_id::<R>(entity, pair_id)
    }

    /// Removes a relationship pair `(R, target)` from `entity`, returning `true` if it was present.
    pub fn remove_pair<R: Relation>(&mut self, entity: Entity, target: Entity) -> bool {
        if !self.is_alive(entity) || !self.is_alive(target) {
            return false;
        }
        let Some(rel_id) = self.components.id_of::<R>() else {
            return false;
        };
        let Some(pair_id) = self.components.pair_to_id(rel_id, target) else {
            return false;
        };
        if self.relationships.remove(entity, rel_id, target) {
            let removed = self.remove_component_by_id_raw(entity, pair_id);
            if removed {
                self.observers.record(EventKind::Remove, pair_id, entity);
                self.observers.record(EventKind::Remove, rel_id, entity);
                self.flush_observers();
            }
            removed
        } else {
            false
        }
    }

    /// Removes a relationship pair `(R, target)` from `entity` and returns its data payload.
    pub fn remove_pair_data<R: Relation>(&mut self, entity: Entity, target: Entity) -> Option<R> {
        if !self.is_alive(entity) || !self.is_alive(target) {
            return None;
        }
        let rel_id = self.components.id_of::<R>()?;
        let pair_id = self.components.pair_to_id(rel_id, target)?;
        if self.relationships.remove(entity, rel_id, target) {
            let removed = self.take_by_id::<R>(entity, pair_id);
            self.flush_observers();
            removed
        } else {
            None
        }
    }

    /// Returns `true` if `entity` possesses the relationship pair `(R, target)`.
    #[must_use]
    pub fn has_pair<R: Relation>(&self, entity: Entity, target: Entity) -> bool {
        if !self.is_alive(entity) || !self.is_alive(target) {
            return false;
        }
        let Some(rel_id) = self.components.id_of::<R>() else {
            return false;
        };
        self.relationships.has_pair(entity, rel_id, target)
    }

    /// Returns `true` if `entity` possesses any relationship with relation `R` (wildcard check).
    #[must_use]
    pub fn has_relation<R: Relation>(&self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let Some(rel_id) = self.components.id_of::<R>() else {
            return false;
        };
        self.relationships.has_relation(entity, rel_id)
    }

    /// Returns the single target of relation `R` for `entity`, or `None` if absent.
    #[must_use]
    pub fn target_of<R: Relation>(&self, entity: Entity) -> Option<Entity> {
        if !self.is_alive(entity) {
            return None;
        }
        let rel_id = self.components.id_of::<R>()?;
        self.relationships.target_of(entity, rel_id)
    }

    /// Returns all target entities for `entity` under relation `R`.
    #[must_use]
    pub fn targets_of<R: Relation>(&self, entity: Entity) -> Vec<Entity> {
        if !self.is_alive(entity) {
            return Vec::new();
        }
        let Some(rel_id) = self.components.id_of::<R>() else {
            return Vec::new();
        };
        self.relationships.targets(entity, rel_id).collect()
    }

    /// Returns all source entities targeting `target` under relation `R`.
    #[must_use]
    pub fn sources_of<R: Relation>(&self, target: Entity) -> Vec<Entity> {
        if !self.is_alive(target) {
            return Vec::new();
        }
        let Some(rel_id) = self.components.id_of::<R>() else {
            return Vec::new();
        };
        self.relationships.sources(target, rel_id).to_vec()
    }

    // =========================================================================
    // Hierarchical Shortcuts
    // =========================================================================

    /// Returns the parent entity of `entity`, if one exists via [`crate::relation::ChildOf`].
    #[must_use]
    pub fn parent_of(&self, entity: Entity) -> Option<Entity> {
        self.target_of::<crate::relation::ChildOf>(entity)
    }

    /// The children of `parent` (its [`ChildOf`](crate::relation::ChildOf) sources), in the
    /// order they were attached; empty if it has none.
    #[must_use]
    pub fn children(&self, parent: Entity) -> &[Entity] {
        if !self.is_alive(parent) {
            return &[];
        }
        match self.components.id_of::<crate::relation::ChildOf>() {
            Some(child_of) => self.relationships.sources(parent, child_of),
            None => &[],
        }
    }

    /// Returns all direct child entities of `entity` via [`crate::relation::ChildOf`], in the
    /// order they were attached.
    #[must_use]
    pub fn children_of(&self, entity: Entity) -> Vec<Entity> {
        self.sources_of::<crate::relation::ChildOf>(entity)
    }

    /// Returns an iterator walking up the ancestor chain of `entity` via [`crate::relation::ChildOf`].
    #[must_use]
    pub fn ancestors(&self, entity: Entity) -> AncestorIter<'_> {
        AncestorIter {
            world: self,
            current: self.is_alive(entity).then_some(entity),
        }
    }

    /// Returns an iterator walking down the descendant subtree of `entity` via [`crate::relation::ChildOf`].
    #[must_use]
    pub fn descendants(&self, entity: Entity) -> DescendantIter<'_> {
        let mut stack = Vec::new();
        stack.extend(self.children(entity).iter().rev().copied());
        DescendantIter { world: self, stack }
    }

    /// Returns the slice of all archetypes in the world.
    #[must_use]
    pub fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    /// Accesses the component registry.
    #[must_use]
    pub const fn components(&self) -> &ComponentRegistry {
        &self.components
    }

    /// Creates a cached query state for `Q`, bound to this world.
    ///
    /// # Panics
    /// If two terms of `Q` access one component and at least one of them writes it.
    pub fn query_state<Q: WorldQuery>(&mut self) -> QueryState<Q> {
        let mut state = QueryState::new(&mut self.components, self.id, self.change_tick);
        state.update(self);
        state
    }

    /// The archetype tables and entity records, for iterators built by a [`QueryState`].
    pub(crate) fn archetypes_and_entities(&mut self) -> (&mut [Archetype], &EntityManager) {
        (&mut self.archetypes, &self.entities)
    }

    /// The archetype tables, for iterators built by a [`QueryState`] of this world.
    pub(crate) fn archetypes_mut(&mut self) -> &mut [Archetype] {
        &mut self.archetypes
    }

    /// Queries matching archetypes with mutable access to requested components.
    ///
    /// # Panics
    /// If two terms of `Q` access one component and at least one of them writes it.
    pub fn query_mut<Q: WorldQuery>(&mut self) -> QueryIter<'_, Q> {
        let state = Q::init_state(&mut self.components);
        crate::query::validate_query_access::<Q>(&state, &self.components);
        // A one-off query has no previous run: every component counts as added and changed.
        let this_run = self.increment_change_tick();
        let ticks = ChangeTicks {
            last_run: Tick::new(this_run.get().wrapping_sub(MAX_CHANGE_AGE)),
            this_run,
        };
        let include_prefabs = crate::query::includes_prefabs::<Q>(&state, &self.components);
        QueryIter::new(
            state,
            &mut self.archetypes,
            &self.entities,
            ticks,
            include_prefabs,
        )
    }
}

/// Iterator traversing entity ancestors upward through [`crate::relation::ChildOf`].
pub struct AncestorIter<'a> {
    world: &'a World,
    current: Option<Entity>,
}

impl<'a> Iterator for AncestorIter<'a> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.current?;
        let parent = self.world.parent_of(curr);
        self.current = parent;
        parent
    }
}

/// Iterator traversing entity descendants downward through [`crate::relation::ChildOf`].
pub struct DescendantIter<'a> {
    world: &'a World,
    stack: Vec<Entity>,
}

impl<'a> Iterator for DescendantIter<'a> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.stack.pop()?;
        self.stack
            .extend(self.world.children(next).iter().rev().copied());
        Some(next)
    }
}

fn get_two_mut<T>(slice: &mut [T], idx_a: usize, idx_b: usize) -> (&mut T, &mut T) {
    assert_ne!(idx_a, idx_b);
    if idx_a < idx_b {
        let (left, right) = slice.split_at_mut(idx_b);
        (&mut left[idx_a], &mut right[0])
    } else {
        let (left, right) = slice.split_at_mut(idx_a);
        (&mut right[0], &mut left[idx_b])
    }
}
