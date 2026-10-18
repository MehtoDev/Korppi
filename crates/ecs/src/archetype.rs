//! Archetype table storage and transition graph edges.
//!
//! Only the [`World`](crate::World) creates and changes archetypes: their columns, entity list
//! and edges must stay in step for the query iterator's `unsafe` reads to be valid. Outside the
//! crate an archetype is read-only.

use crate::component::ComponentId;
use crate::entity::Entity;
use crate::fast_hash::FastHashMap;
use crate::storage::Column;

/// Cached transition edges between archetypes for adding and removing components.
#[derive(Default)]
pub struct ArchetypeEdges {
    /// Target archetype ID when adding a component.
    pub add: FastHashMap<ComponentId, usize>,
    /// Target archetype ID when removing a component.
    pub remove: FastHashMap<ComponentId, usize>,
}

impl ArchetypeEdges {
    /// Clears all transition edges.
    pub(crate) fn clear(&mut self) {
        self.add.clear();
        self.remove.clear();
    }
}

/// An archetype represents a unique combination of component types stored in contiguous columnar tables.
pub struct Archetype {
    id: usize,
    components: Vec<ComponentId>,
    relations: Vec<ComponentId>,
    /// `(relation, target)` of each pair component, in component order.
    pairs: Vec<(ComponentId, Entity)>,
    columns: Vec<Column>,
    entities: Vec<Entity>,
    edges: ArchetypeEdges,
    /// Freed by the world and waiting for reuse: empty, and matched by no query state.
    free: bool,
    /// The observer event kinds watched for any of its components (`EventKind::bit`).
    observed: u8,
    /// Components its entities inherit through `IsA` and do not own, sorted by ID, each with the
    /// prefab entity that owns the value.
    inherited: Vec<(ComponentId, Entity)>,
    /// Holds the `Prefab` tag: skipped by queries that do not name it.
    prefab: bool,
}

impl Archetype {
    /// Creates a new archetype; `pairs` lists the `(relation, target)` of its pair components.
    #[must_use]
    pub(crate) fn with_pairs(
        id: usize,
        components: Vec<ComponentId>,
        pairs: Vec<(ComponentId, Entity)>,
        columns: Vec<Column>,
    ) -> Self {
        let mut relations = Vec::new();
        for &(relation, _) in &pairs {
            if !relations.contains(&relation) {
                relations.push(relation);
            }
        }
        Self {
            id,
            components,
            relations,
            pairs,
            columns,
            entities: Vec::new(),
            edges: ArchetypeEdges::default(),
            free: false,
            observed: 0,
            inherited: Vec::new(),
            prefab: false,
        }
    }

    /// Returns the unique archetype ID.
    #[must_use]
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Returns `true` for a slot the world freed (a dead target's archetype) and has not reused.
    #[must_use]
    pub const fn is_free(&self) -> bool {
        self.free
    }

    /// The prefab entity whose value of `id` this archetype's entities inherit, if they inherit it.
    #[must_use]
    pub fn inherits(&self, id: ComponentId) -> Option<Entity> {
        self.inherited
            .binary_search_by_key(&id, |&(c, _)| c)
            .ok()
            .map(|i| self.inherited[i].1)
    }

    /// Returns `true` if this archetype's entities own or inherit component `id`.
    #[must_use]
    pub fn has_or_inherits(&self, id: ComponentId) -> bool {
        self.has_component(id) || self.inherits(id).is_some()
    }

    /// The inherited components and their owners, sorted by component ID.
    #[must_use]
    pub fn inherited(&self) -> &[(ComponentId, Entity)] {
        &self.inherited
    }

    /// Returns `true` if this archetype holds the `Prefab` tag.
    #[must_use]
    pub const fn is_prefab(&self) -> bool {
        self.prefab
    }

    pub(crate) fn set_prefab(&mut self, prefab: bool) {
        self.prefab = prefab;
    }

    /// Takes the inherited list out for rebuilding (keeping its buffer).
    pub(crate) fn take_inherited(&mut self) -> Vec<(ComponentId, Entity)> {
        std::mem::take(&mut self.inherited)
    }

    pub(crate) fn set_inherited(&mut self, inherited: Vec<(ComponentId, Entity)>) {
        self.inherited = inherited;
    }

    /// The `(relation, target)` of each pair component, in component order.
    #[must_use]
    pub fn pairs(&self) -> &[(ComponentId, Entity)] {
        &self.pairs
    }

    /// The observer event kinds watched for any of this archetype's components.
    pub(crate) const fn observed(&self) -> u8 {
        self.observed
    }

    pub(crate) fn set_observed(&mut self, mask: u8) {
        self.observed = mask;
    }

    /// Returns the number of entities residing in this archetype table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Returns `true` if this archetype currently contains no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Returns the sorted list of component IDs belonging to this archetype.
    #[must_use]
    pub fn components(&self) -> &[ComponentId] {
        &self.components
    }

    /// Locates the column index for the given component ID via binary search.
    #[must_use]
    pub fn column_index(&self, id: ComponentId) -> Option<usize> {
        self.components.binary_search(&id).ok()
    }

    /// Returns `true` if this archetype contains the specified component ID.
    #[must_use]
    pub fn has_component(&self, id: ComponentId) -> bool {
        self.column_index(id).is_some()
    }

    /// Returns the relation IDs present on this archetype.
    #[must_use]
    pub fn relations(&self) -> &[ComponentId] {
        &self.relations
    }

    /// Returns the target of the first pair of `relation` in this archetype.
    #[must_use]
    pub fn target(&self, relation: ComponentId) -> Option<Entity> {
        self.pairs
            .iter()
            .find(|&&(r, _)| r == relation)
            .map(|&(_, target)| target)
    }

    /// Returns `true` if this archetype contains any pair of the specified relation ID.
    #[must_use]
    pub fn has_relation(&self, id: ComponentId) -> bool {
        self.relations.contains(&id)
    }

    /// Returns the slice of entities stored in this archetype.
    #[must_use]
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Pushes an entity into the archetype table and returns its allocated row index.
    pub(crate) fn push_entity(&mut self, entity: Entity) -> usize {
        let row = self.entities.len();
        self.entities.push(entity);
        row
    }

    /// Retrieves an immutable reference to the column corresponding to `id`.
    #[must_use]
    pub fn column(&self, id: ComponentId) -> Option<&Column> {
        self.column_index(id).map(|idx| &self.columns[idx])
    }

    /// Retrieves a mutable reference to the column corresponding to `id`.
    pub(crate) fn column_mut(&mut self, id: ComponentId) -> Option<&mut Column> {
        self.column_index(id).map(|idx| &mut self.columns[idx])
    }

    /// Retrieves an immutable reference to the column at the given index.
    #[must_use]
    pub fn column_by_index(&self, index: usize) -> Option<&Column> {
        self.columns.get(index)
    }

    /// Retrieves a mutable reference to the column at the given index.
    pub(crate) fn column_by_index_mut(&mut self, index: usize) -> Option<&mut Column> {
        self.columns.get_mut(index)
    }

    /// Returns the raw pointer to the base of the column buffer at `index`.
    #[must_use]
    pub fn column_raw_ptr(&self, index: usize) -> *mut u8 {
        self.columns[index].raw_ptr()
    }

    /// Returns a typed immutable slice of components at `index`.
    ///
    /// # Safety
    /// `T` must match the component type at `index`.
    #[must_use]
    pub unsafe fn column_slice<T>(&self, index: usize) -> &[T] {
        let col = &self.columns[index];
        // SAFETY: caller guarantees T matches column component type.
        unsafe { col.as_slice::<T>() }
    }

    /// Accesses the mutable entity list.
    pub(crate) fn entities_mut(&mut self) -> &mut Vec<Entity> {
        &mut self.entities
    }

    /// Accesses the slice of columnar storages.
    #[must_use]
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Accesses the columns mutably, for tick maintenance.
    pub(crate) fn columns_mut(&mut self) -> &mut [Column] {
        &mut self.columns
    }

    /// Accesses the cached transition edges for this archetype.
    pub fn edges(&self) -> &ArchetypeEdges {
        &self.edges
    }

    /// Accesses the mutable transition edges for this archetype.
    pub(crate) fn edges_mut(&mut self) -> &mut ArchetypeEdges {
        &mut self.edges
    }

    /// Moves common components from `src_row` in `src_arch` to `dst_arch` at a newly pushed row without heap allocation.
    pub(crate) fn move_matching_columns(
        src_arch: &mut Archetype,
        src_row: usize,
        dst_arch: &mut Archetype,
        now: crate::tick::Tick,
    ) {
        let mut src_idx = 0;
        let mut dst_idx = 0;
        let src_len = src_arch.components.len();
        let dst_len = dst_arch.components.len();

        while src_idx < src_len && dst_idx < dst_len {
            let c_src = src_arch.components[src_idx];
            let c_dst = dst_arch.components[dst_idx];
            if c_src == c_dst {
                let src_col = &mut src_arch.columns[src_idx];
                let dst_col = &mut dst_arch.columns[dst_idx];
                // SAFETY: component types match; src_row < src_col.len(); dst_col is at the new row.
                unsafe {
                    src_col.move_row_to(src_row, dst_col, now);
                }
                src_idx += 1;
                dst_idx += 1;
            } else if c_src < c_dst {
                src_idx += 1;
            } else {
                dst_idx += 1;
            }
        }
    }

    /// Removes an entity at `row` via swap-remove across all columns and entity lists in deterministic order.
    ///
    /// If an entity was swapped into `row`, returns that swapped `Entity` so its location can be updated.
    pub(crate) fn swap_remove(&mut self, row: usize) -> Option<Entity> {
        assert!(row < self.entities.len());
        for column in &mut self.columns {
            column.swap_remove(row);
        }
        self.entities.swap_remove(row);
        self.entities.get(row).copied()
    }

    /// Drops every row and empties the archetype's component set, relations and edges, leaving a
    /// free slot for the world to reuse. The empty columns keep their buffers for
    /// [`reuse`](Self::reuse); with no components, no lookup reaches them.
    pub(crate) fn release(&mut self) {
        self.clear();
        self.components.clear();
        self.relations.clear();
        self.pairs.clear();
        self.inherited.clear();
        self.prefab = false;
        self.edges.clear();
        self.free = true;
    }

    /// Returns `true` if this free slot's columns have exactly `layouts`, in order.
    pub(crate) fn fits(&self, layouts: impl ExactSizeIterator<Item = std::alloc::Layout>) -> bool {
        self.free
            && self.columns.len() == layouts.len()
            && self
                .columns
                .iter()
                .zip(layouts)
                .all(|(column, layout)| column.info().layout() == layout)
    }

    /// Turns this free slot into the archetype for `components`, keeping every buffer. `info` and
    /// `pair_of` look up each component's metadata and, for a pair, its relation and target.
    pub(crate) fn reuse(
        &mut self,
        components: &[ComponentId],
        info: impl Fn(ComponentId) -> crate::component::ComponentInfo,
        pair_of: impl Fn(ComponentId) -> Option<(ComponentId, Entity)>,
    ) {
        debug_assert!(self.free && self.entities.is_empty());
        self.components.extend_from_slice(components);
        for (column, &component) in self.columns.iter_mut().zip(components) {
            column.reset(info(component));
            if let Some((relation, target)) = pair_of(component) {
                self.pairs.push((relation, target));
                if !self.relations.contains(&relation) {
                    self.relations.push(relation);
                }
            }
        }
        self.free = false;
    }

    /// Clears all entities and component columns in this archetype.
    pub(crate) fn clear(&mut self) {
        for column in &mut self.columns {
            column.clear();
        }
        self.entities.clear();
    }
}
