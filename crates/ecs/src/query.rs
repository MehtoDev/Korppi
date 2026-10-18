//! Type-safe archetype queries: query terms, iterators and cached query state.
//!
//! A query `Q` is a [`WorldQuery`]: a component reference (`&T`, `&mut T`), [`Entity`], a filter
//! ([`With`], [`Without`], the relation filters in [`crate::relation`]) or a tuple of up to eight
//! of them; [`Option`] of a reference, [`Has`], [`Target`], and the change filters [`Added`] and
//! [`Changed`] widen what a term can express. [`World::query_mut`](crate::World::query_mut)
//! builds a [`QueryIter`] that matches
//! archetypes as it walks them; [`World::query_state`](crate::World::query_state) builds a
//! [`QueryState`] that keeps the matching archetypes between runs and only checks archetypes
//! created since. Both yield one item per entity (`next`, `for_each`) or one tuple of slices per
//! archetype (`for_each_chunk`). Neither allocates while it runs.
//!
//! Terms that access one component twice, at least once mutably, are rejected with a panic when
//! the query is built.

use crate::archetype::Archetype;
use crate::component::{Component, ComponentId, ComponentRegistry};
use crate::entity::{Entity, EntityManager};
use crate::relation::Relation;
use crate::tick::{ChangeTicks, ComponentTicks, Tick, MAX_CHANGE_AGE};
use crate::world::World;
use std::marker::PhantomData;
use std::mem::size_of;

/// Query component access kind for access-conflict validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentAccess {
    /// Read-only access to a component.
    Read(ComponentId),
    /// Exclusive read-write access to a component.
    Write(ComponentId),
}

impl ComponentAccess {
    /// The component both accesses touch, if at least one of them writes it.
    fn conflict(self, other: Self) -> Option<ComponentId> {
        match (self, other) {
            (Self::Write(a), Self::Write(b) | Self::Read(b)) | (Self::Read(a), Self::Write(b))
                if a == b =>
            {
                Some(a)
            }
            _ => None,
        }
    }
}

/// What a query term may consult when it binds an archetype: the run's change ticks, and the
/// world's archetypes and entity records, to find the prefab row an inherited term reads.
#[derive(Clone, Copy)]
pub struct FetchContext {
    /// The run's change ticks.
    pub ticks: ChangeTicks,
    archetypes: *const [Archetype],
    entities: *const EntityManager,
}

impl FetchContext {
    /// A context over `archetypes` and `entities` for a run with `ticks`.
    pub(crate) fn new(
        archetypes: *const [Archetype],
        entities: *const EntityManager,
        ticks: ChangeTicks,
    ) -> Self {
        Self {
            ticks,
            archetypes,
            entities,
        }
    }

    /// The row of component `id` that entities of `archetype` inherit, as a pointer, if they
    /// inherit it.
    ///
    /// # Safety
    /// The context's pointers must be valid for reads, and `archetype` one of its archetypes.
    unsafe fn inherited_row<T>(self, archetype: &Archetype, id: ComponentId) -> Option<*mut T> {
        let owner = archetype.inherits(id)?;
        // SAFETY: the caller guarantees the entity records are valid for reads; nothing writes
        // them while a query runs.
        let location = unsafe { &*self.entities }.location(owner)?;
        // SAFETY: as above, for the archetype list; `location` names one of its archetypes.
        let source = unsafe { &(*self.archetypes)[location.archetype_id] };
        let index = source.column_index(id)?;
        // SAFETY: the owner's row is below the source column's length.
        Some(unsafe { source.column_raw_ptr(index).cast::<T>().add(location.row) })
    }
}

/// Panics if two terms of `Q` access one component and at least one of them writes it.
///
/// Compares every pair of reported accesses without collecting them, so building a query does
/// not allocate.
pub(crate) fn validate_query_access<Q: WorldQuery>(state: &Q::State, registry: &ComponentRegistry) {
    let mut i = 0;
    Q::access_set(state, &mut |a| {
        let mut j = 0;
        Q::access_set(state, &mut |b| {
            if j > i {
                if let Some(id) = a.conflict(b) {
                    let name = registry.get_info(id).map_or("unknown", |info| info.name());
                    panic!(
                        "Conflicting access in query: component `{name}` cannot be borrowed mutably more than once or concurrently with immutable accesses"
                    );
                }
            }
            j += 1;
        });
        i += 1;
    });
}

/// Returns `true` if a term of `Q` names `Prefab`, so that the query matches prefabs.
pub(crate) fn includes_prefabs<Q: WorldQuery>(
    state: &Q::State,
    registry: &ComponentRegistry,
) -> bool {
    registry
        .id_of::<crate::prefab::Prefab>()
        .is_some_and(|prefab| Q::mentions(state, prefab))
}

/// Trait defining a query specification over archetype component columns.
///
/// # Safety
/// The query iterator hands out the references `fetch_next` and `fetch_slice` produce after
/// checking only what `access_set` reports. An implementation must:
/// - report through `access_set` every component whose values it reads (`Read`) or writes
///   (`Write`); reading only a column's change ticks is not an access;
/// - return `true` from `matches_archetype` only for archetypes that hold every column
///   `init_fetch` requires;
/// - read and write only those columns, as their registered type, in `init_fetch`,
///   `fetch_next`, `fetch_slice` and `finish_fetch`, and write change ticks only in
///   `finish_fetch`;
/// - set `DENSE` to `false` if `filter` can return `false`.
pub unsafe trait WorldQuery {
    /// The item yielded for each matching entity row.
    type Item<'w>;
    /// The rows of one archetype at once: a slice per component term, `()` per filter.
    type Slice<'w>;
    /// Cached state (such as resolved [`ComponentId`]s).
    type State: Copy + Send + 'static;
    /// Per-archetype fetch state: pointers to the current row of each column the query reads.
    type Fetch<'w>: Copy;

    /// `true` if every row of a matching archetype yields an item (`filter` is always `true`).
    /// Chunk iteration needs it.
    const DENSE: bool = true;

    /// Produces a default inert fetch state before any archetype is bound.
    fn empty_fetch<'w>() -> Self::Fetch<'w>;

    /// Initializes and caches component IDs from the registry.
    fn init_state(registry: &mut ComponentRegistry) -> Self::State;

    /// Reports each component access of this query to `report`, for access-conflict checks.
    fn access_set(state: &Self::State, report: &mut dyn FnMut(ComponentAccess));

    /// Returns `true` if the archetype contains all components required by this query.
    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool;

    /// Prepares fetching for an archetype table once per archetype.
    ///
    /// # Safety
    /// Caller must guarantee `archetype` is a valid pointer derived from exclusive World access,
    /// `archetype` matches this query, and conflicting accesses have been validated.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *mut Archetype,
        ctx: FetchContext,
    ) -> Self::Fetch<'w>;

    /// Returns `true` if a term of this query names component `id` (a query naming `Prefab`
    /// sees prefabs).
    fn mentions(_state: &Self::State, _id: ComponentId) -> bool {
        false
    }

    /// Returns `true` if the row at the current position of `fetch` yields an item.
    ///
    /// # Safety
    /// `fetch` must be valid for the current row within the active archetype.
    #[inline(always)]
    unsafe fn filter<'w>(_fetch: Self::Fetch<'w>) -> bool {
        true
    }

    /// Fetches the query item at the current position and advances the fetch state to the next row.
    ///
    /// # Safety
    /// `fetch` must be valid for the current row within the active archetype.
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>);

    /// Returns `true` if every term of `fetch` steps one row at a time (none reads an inherited
    /// value), so that [`fetch_next_dense`](Self::fetch_next_dense) may be used for the archetype.
    #[inline(always)]
    fn is_dense<'w>(_fetch: Self::Fetch<'w>) -> bool {
        true
    }

    /// As [`fetch_next`](Self::fetch_next), for a fetch that [`is_dense`](Self::is_dense): the
    /// iterator's fast path, with the per-row step known at compile time.
    ///
    /// # Safety
    /// As for `fetch_next`, and `is_dense(fetch)` holds.
    #[inline(always)]
    unsafe fn fetch_next_dense<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        // SAFETY: the caller's guarantee covers `fetch_next`.
        unsafe { Self::fetch_next(fetch) }
    }

    /// Returns the `len` rows starting at the current position of `fetch` as slices.
    ///
    /// # Safety
    /// `fetch` must be valid for the current row within the active archetype, the archetype must
    /// have at least `len` rows from there on, and no item fetched from those rows may be in use.
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, len: usize) -> Self::Slice<'w>;

    /// Advances `fetch` by `rows` rows without fetching them.
    ///
    /// # Safety
    /// The active archetype must have at least `rows` rows from the current position on.
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w>;

    /// Called after one row was fetched by entity (not by iteration): mutable terms mark that row
    /// changed at `ticks.this_run`.
    ///
    /// # Safety
    /// As for [`finish_fetch`](Self::finish_fetch); `row` is below the archetype's length.
    #[inline(always)]
    unsafe fn mark_row(
        _state: &Self::State,
        _archetype: *mut Archetype,
        _row: usize,
        _ticks: ChangeTicks,
    ) {
    }

    /// Called when the iterator leaves an archetype `init_fetch` prepared: mutable terms mark
    /// the rows of their column changed at `ticks.this_run`.
    ///
    /// # Safety
    /// As for [`init_fetch`](Self::init_fetch); no fetch of this archetype is used afterwards.
    #[inline(always)]
    unsafe fn finish_fetch(_state: &Self::State, _archetype: *mut Archetype, _ticks: ChangeTicks) {}
}

// SAFETY: reads only the archetype's entity list, which no query term can write.
unsafe impl WorldQuery for Entity {
    type Item<'w> = Entity;
    type Slice<'w> = &'w [Entity];
    type State = ();
    type Fetch<'w> = *const Entity;

    fn empty_fetch<'w>() -> Self::Fetch<'w> {
        std::ptr::null()
    }

    fn init_state(_registry: &mut ComponentRegistry) -> Self::State {}

    fn access_set(_state: &Self::State, _report: &mut dyn FnMut(ComponentAccess)) {}

    fn matches_archetype(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    #[inline(always)]
    unsafe fn init_fetch<'w>(
        _state: &Self::State,
        archetype: *mut Archetype,
        _ctx: FetchContext,
    ) -> Self::Fetch<'w> {
        // SAFETY: caller guarantees archetype is valid for reads.
        let arch = unsafe { &*archetype };
        arch.entities().as_ptr()
    }

    #[inline(always)]
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        // SAFETY: fetch points to valid entity in bounds.
        let entity = unsafe { *fetch };
        // SAFETY: advancing pointer stays within valid entity buffer.
        let next = unsafe { fetch.add(1) };
        (entity, next)
    }

    #[inline(always)]
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
        // SAFETY: the caller guarantees `len` initialized entities from `fetch` on.
        unsafe { std::slice::from_raw_parts(fetch, len) }
    }

    #[inline(always)]
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w> {
        // SAFETY: the caller guarantees `rows` rows from the current position.
        unsafe { fetch.add(rows) }
    }
}

/// Implements [`WorldQuery`] for an archetype-level filter whose item is `()`.
macro_rules! archetype_filter {
    ($(#[$doc:meta])* $name:ident<$t:ident: $bound:path>, $init:expr, $matches:expr) => {
        $(#[$doc])*
        pub struct $name<$t: $bound>(::std::marker::PhantomData<$t>);

        // SAFETY: a filter on the archetype's component set; it reads no column.
        unsafe impl<$t: $bound> $crate::query::WorldQuery for $name<$t> {
            type Item<'w> = ();
            type Slice<'w> = ();
            type State = $crate::component::ComponentId;
            type Fetch<'w> = ();

            fn empty_fetch<'w>() -> Self::Fetch<'w> {}

            fn init_state(registry: &mut $crate::component::ComponentRegistry) -> Self::State {
                $init(registry)
            }

            fn access_set(
                _state: &Self::State,
                _report: &mut dyn FnMut($crate::query::ComponentAccess),
            ) {
            }

            fn matches_archetype(state: &Self::State, archetype: &$crate::archetype::Archetype) -> bool {
                $matches(*state, archetype)
            }

            fn mentions(state: &Self::State, id: $crate::component::ComponentId) -> bool {
                *state == id
            }

            #[inline(always)]
            unsafe fn init_fetch<'w>(
                _state: &Self::State,
                _archetype: *mut $crate::archetype::Archetype,
                _ctx: $crate::query::FetchContext,
            ) -> Self::Fetch<'w> {
            }

            #[inline(always)]
            unsafe fn fetch_next<'w>(_fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
                ((), ())
            }

            #[inline(always)]
            unsafe fn fetch_slice<'w>(_fetch: Self::Fetch<'w>, _len: usize) -> Self::Slice<'w> {}

            #[inline(always)]
            unsafe fn seek<'w>(_fetch: Self::Fetch<'w>, _rows: usize) -> Self::Fetch<'w> {}
        }
    };
}
pub(crate) use archetype_filter;

archetype_filter!(
    /// Filter matching only entities that own or inherit component `T`.
    With<T: Component>,
    |registry: &mut ComponentRegistry| registry.register::<T>(),
    |id, archetype: &Archetype| archetype.has_or_inherits(id)
);

archetype_filter!(
    /// Filter matching only entities that neither own nor inherit component `T`.
    Without<T: Component>,
    |registry: &mut ComponentRegistry| registry.register::<T>(),
    |id, archetype: &Archetype| !archetype.has_or_inherits(id)
);

/// The column of `id` in `archetype` and its base pointer as `*mut T`.
///
/// # Safety
/// `archetype` must be valid for reads and hold a column for `id` storing `T`.
#[inline(always)]
unsafe fn column_base<T>(archetype: *mut Archetype, id: ComponentId) -> *mut T {
    // SAFETY: the caller guarantees the archetype is valid for reads.
    let arch = unsafe { &*archetype };
    let index = arch
        .column_index(id)
        .expect("Missing column in matching archetype");
    arch.column_raw_ptr(index).cast::<T>()
}

/// Marks every row of the column of `id` in `archetype` changed at `tick`, if it has one.
///
/// # Safety
/// `archetype` must be valid for writes, and no reference into its change ticks may be live.
#[inline(always)]
unsafe fn mark_column_changed(archetype: *mut Archetype, id: ComponentId, tick: Tick) {
    // SAFETY: the caller guarantees exclusive access to the archetype's change ticks.
    let arch = unsafe { &mut *archetype };
    if let Some(column) = arch.column_mut(id) {
        column.mark_all_changed(tick);
    }
}

/// Marks the row `row` of the column of `id` in `archetype` changed at `tick`, if it has one.
///
/// # Safety
/// As for [`mark_column_changed`]; `row` is below the archetype's length.
#[inline(always)]
unsafe fn mark_row_changed(archetype: *mut Archetype, id: ComponentId, row: usize, tick: Tick) {
    // SAFETY: the caller guarantees exclusive access to the archetype's change ticks.
    let arch = unsafe { &mut *archetype };
    if let Some(column) = arch.column_mut(id) {
        column.mark_changed(row, tick);
    }
}

// SAFETY: reports `Read(T)`, matches only archetypes with a `T` column, and reads only that
// column as `T` (its ID comes from the registry's `TypeId` mapping for `T`).
unsafe impl<T: Component> WorldQuery for &T {
    type Item<'w> = &'w T;
    type Slice<'w> = &'w [T];
    type State = ComponentId;
    /// The current row's value and the step to the next row in bytes: `size_of::<T>()` for an
    /// owned column, 0 for a value inherited from a prefab, which every row of the archetype
    /// shares.
    type Fetch<'w> = (*const T, usize);

    fn empty_fetch<'w>() -> Self::Fetch<'w> {
        (std::ptr::null(), size_of::<T>())
    }

    fn init_state(registry: &mut ComponentRegistry) -> Self::State {
        registry.register::<T>()
    }

    fn access_set(state: &Self::State, report: &mut dyn FnMut(ComponentAccess)) {
        report(ComponentAccess::Read(*state));
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        archetype.has_or_inherits(*state)
    }

    fn mentions(state: &Self::State, id: ComponentId) -> bool {
        *state == id
    }

    #[inline(always)]
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *mut Archetype,
        ctx: FetchContext,
    ) -> Self::Fetch<'w> {
        // SAFETY: the caller guarantees the archetype is valid for reads.
        let arch = unsafe { &*archetype };
        if arch.has_component(*state) {
            // SAFETY: the archetype has a `T` column.
            (
                unsafe { column_base::<T>(archetype, *state) },
                size_of::<T>(),
            )
        } else {
            // SAFETY: the context's pointers are valid; the archetype matches, so it inherits `T`.
            let row = unsafe { ctx.inherited_row::<T>(arch, *state) }
                .expect("matching archetype owns or inherits the component");
            (row, 0)
        }
    }

    #[inline(always)]
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        let (row, step) = fetch;
        // SAFETY: caller guarantees fetch is at a live row and access-check verified no aliasing
        // mut borrows; a prefab's value is never borrowed mutably through an instance.
        let item = unsafe { &*row };
        // SAFETY: an owned column advances within its buffer; an inherited value stays put.
        (item, (unsafe { row.byte_add(step) }, step))
    }

    #[inline(always)]
    fn is_dense<'w>(fetch: Self::Fetch<'w>) -> bool {
        fetch.1 == size_of::<T>()
    }

    #[inline(always)]
    unsafe fn fetch_next_dense<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        let (row, step) = fetch;
        // SAFETY: as for `fetch_next`; a dense fetch steps by one row.
        unsafe { (&*row, (row.add(1), step)) }
    }

    #[inline(always)]
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
        let (row, step) = fetch;
        assert!(
            step == size_of::<T>(),
            "for_each_chunk cannot hand out an inherited component as a slice; iterate with for_each"
        );
        // SAFETY: the caller guarantees `len` initialized values from `row` on, none of them
        // borrowed mutably (access check); for a zero-sized `T` the pointer is aligned and dangling.
        unsafe { std::slice::from_raw_parts(row, len) }
    }

    #[inline(always)]
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w> {
        let (row, step) = fetch;
        // SAFETY: the caller guarantees `rows` rows from the current position.
        (unsafe { row.byte_add(step * rows) }, step)
    }
}

// SAFETY: reports `Write(T)`, matches only archetypes with a `T` column, reads and writes only
// that column as `T` (its ID comes from the registry's `TypeId` mapping for `T`), and writes its
// change ticks only in `finish_fetch`.
unsafe impl<T: Component> WorldQuery for &mut T {
    type Item<'w> = &'w mut T;
    type Slice<'w> = &'w mut [T];
    type State = ComponentId;
    type Fetch<'w> = *mut T;

    fn empty_fetch<'w>() -> Self::Fetch<'w> {
        std::ptr::null_mut()
    }

    fn init_state(registry: &mut ComponentRegistry) -> Self::State {
        registry.register::<T>()
    }

    fn access_set(state: &Self::State, report: &mut dyn FnMut(ComponentAccess)) {
        report(ComponentAccess::Write(*state));
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        archetype.has_component(*state)
    }

    fn mentions(state: &Self::State, id: ComponentId) -> bool {
        *state == id
    }

    #[inline(always)]
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *mut Archetype,
        _ctx: FetchContext,
    ) -> Self::Fetch<'w> {
        // SAFETY: the archetype matches, so it has a `T` column.
        unsafe { column_base::<T>(archetype, *state) }
    }

    #[inline(always)]
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        // SAFETY: caller guarantees fetch is within column buffer and access-check verified no concurrent borrows.
        let item = unsafe { &mut *fetch };
        // SAFETY: advancing pointer stays within valid column buffer.
        let next = unsafe { fetch.add(1) };
        (item, next)
    }

    #[inline(always)]
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
        // SAFETY: the caller guarantees `len` initialized values from `fetch` on that nothing else
        // borrows (access check); for a zero-sized `T` the pointer is aligned and dangling.
        unsafe { std::slice::from_raw_parts_mut(fetch, len) }
    }

    #[inline(always)]
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w> {
        // SAFETY: the caller guarantees `rows` rows from the current position.
        unsafe { fetch.add(rows) }
    }

    #[inline(always)]
    unsafe fn mark_row(
        state: &Self::State,
        archetype: *mut Archetype,
        row: usize,
        ticks: ChangeTicks,
    ) {
        // SAFETY: as for `finish_fetch`.
        unsafe { mark_row_changed(archetype, *state, row, ticks.this_run) };
    }

    #[inline(always)]
    unsafe fn finish_fetch(state: &Self::State, archetype: *mut Archetype, ticks: ChangeTicks) {
        // SAFETY: the caller guarantees exclusive access; only this term writes the ticks.
        unsafe { mark_column_changed(archetype, *state, ticks.this_run) };
    }
}

// SAFETY: as for `&T`, on archetypes that have the column; `None` elsewhere, reading nothing.
unsafe impl<T: Component> WorldQuery for Option<&T> {
    type Item<'w> = Option<&'w T>;
    type Slice<'w> = Option<&'w [T]>;
    type State = ComponentId;
    /// As for `&T`; a null pointer where the entity neither owns nor inherits `T`.
    type Fetch<'w> = (*const T, usize);

    fn empty_fetch<'w>() -> Self::Fetch<'w> {
        (std::ptr::null(), 0)
    }

    fn init_state(registry: &mut ComponentRegistry) -> Self::State {
        registry.register::<T>()
    }

    fn access_set(state: &Self::State, report: &mut dyn FnMut(ComponentAccess)) {
        report(ComponentAccess::Read(*state));
    }

    fn matches_archetype(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn mentions(state: &Self::State, id: ComponentId) -> bool {
        *state == id
    }

    #[inline(always)]
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *mut Archetype,
        ctx: FetchContext,
    ) -> Self::Fetch<'w> {
        // SAFETY: the caller guarantees the archetype is valid for reads.
        let arch = unsafe { &*archetype };
        if arch.has_component(*state) {
            // SAFETY: the archetype has a `T` column.
            (
                unsafe { column_base::<T>(archetype, *state) },
                size_of::<T>(),
            )
        } else {
            // SAFETY: the context's pointers are valid for reads.
            match unsafe { ctx.inherited_row::<T>(arch, *state) } {
                Some(row) => (row, 0),
                None => (std::ptr::null(), 0),
            }
        }
    }

    #[inline(always)]
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        let (row, step) = fetch;
        if row.is_null() {
            return (None, fetch);
        }
        // SAFETY: as for `&T`.
        unsafe { (Some(&*row), (row.byte_add(step), step)) }
    }

    #[inline(always)]
    fn is_dense<'w>(fetch: Self::Fetch<'w>) -> bool {
        fetch.0.is_null() || fetch.1 == size_of::<T>()
    }

    #[inline(always)]
    unsafe fn fetch_next_dense<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        let (row, step) = fetch;
        if row.is_null() {
            return (None, fetch);
        }
        // SAFETY: as for `&T`; a dense fetch steps by one row.
        unsafe { (Some(&*row), (row.add(1), step)) }
    }

    #[inline(always)]
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
        let (row, step) = fetch;
        if row.is_null() {
            return None;
        }
        assert!(
            step == size_of::<T>(),
            "for_each_chunk cannot hand out an inherited component as a slice; iterate with for_each"
        );
        // SAFETY: as for `&T`.
        Some(unsafe { std::slice::from_raw_parts(row, len) })
    }

    #[inline(always)]
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w> {
        let (row, step) = fetch;
        if row.is_null() {
            return fetch;
        }
        // SAFETY: the caller guarantees `rows` rows from the current position.
        (unsafe { row.byte_add(step * rows) }, step)
    }
}

// SAFETY: as for `&mut T`, on archetypes that have the column; `None` elsewhere, touching nothing.
unsafe impl<T: Component> WorldQuery for Option<&mut T> {
    type Item<'w> = Option<&'w mut T>;
    type Slice<'w> = Option<&'w mut [T]>;
    type State = ComponentId;
    type Fetch<'w> = *mut T;

    fn empty_fetch<'w>() -> Self::Fetch<'w> {
        std::ptr::null_mut()
    }

    fn init_state(registry: &mut ComponentRegistry) -> Self::State {
        registry.register::<T>()
    }

    fn access_set(state: &Self::State, report: &mut dyn FnMut(ComponentAccess)) {
        report(ComponentAccess::Write(*state));
    }

    fn matches_archetype(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn mentions(state: &Self::State, id: ComponentId) -> bool {
        *state == id
    }

    #[inline(always)]
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *mut Archetype,
        _ctx: FetchContext,
    ) -> Self::Fetch<'w> {
        // SAFETY: the caller guarantees the archetype is valid for reads.
        let arch = unsafe { &*archetype };
        if arch.has_component(*state) {
            // SAFETY: the archetype has a `T` column.
            unsafe { column_base::<T>(archetype, *state) }
        } else {
            std::ptr::null_mut()
        }
    }

    #[inline(always)]
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        if fetch.is_null() {
            return (None, fetch);
        }
        // SAFETY: as for `&mut T`.
        unsafe { (Some(&mut *fetch), fetch.add(1)) }
    }

    #[inline(always)]
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
        // SAFETY: as for `&mut T` when the column exists.
        (!fetch.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(fetch, len) })
    }

    #[inline(always)]
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w> {
        if fetch.is_null() {
            return fetch;
        }
        // SAFETY: the caller guarantees `rows` rows from the current position.
        unsafe { fetch.add(rows) }
    }

    #[inline(always)]
    unsafe fn mark_row(
        state: &Self::State,
        archetype: *mut Archetype,
        row: usize,
        ticks: ChangeTicks,
    ) {
        // SAFETY: as for `&mut T`; a missing column is skipped.
        unsafe { mark_row_changed(archetype, *state, row, ticks.this_run) };
    }

    #[inline(always)]
    unsafe fn finish_fetch(state: &Self::State, archetype: *mut Archetype, ticks: ChangeTicks) {
        // SAFETY: as for `&mut T`; a missing column is skipped.
        unsafe { mark_column_changed(archetype, *state, ticks.this_run) };
    }
}

/// Query term that yields whether the entity has component `T`, without borrowing it.
pub struct Has<T: Component>(PhantomData<T>);

// SAFETY: reads only the archetype's component set.
unsafe impl<T: Component> WorldQuery for Has<T> {
    type Item<'w> = bool;
    type Slice<'w> = bool;
    type State = ComponentId;
    type Fetch<'w> = bool;

    fn empty_fetch<'w>() -> Self::Fetch<'w> {
        false
    }

    fn init_state(registry: &mut ComponentRegistry) -> Self::State {
        registry.register::<T>()
    }

    fn access_set(_state: &Self::State, _report: &mut dyn FnMut(ComponentAccess)) {}

    fn matches_archetype(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn mentions(state: &Self::State, id: ComponentId) -> bool {
        *state == id
    }

    #[inline(always)]
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *mut Archetype,
        _ctx: FetchContext,
    ) -> Self::Fetch<'w> {
        // SAFETY: the caller guarantees the archetype is valid for reads.
        unsafe { &*archetype }.has_or_inherits(*state)
    }

    #[inline(always)]
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        (fetch, fetch)
    }

    #[inline(always)]
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, _len: usize) -> Self::Slice<'w> {
        fetch
    }

    #[inline(always)]
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, _rows: usize) -> Self::Fetch<'w> {
        fetch
    }
}

/// Query term that yields the target of the entity's pair of the exclusive relation `R`, such
/// as its `ChildOf` parent; it matches only entities that have such a pair.
pub struct Target<R: Relation>(PhantomData<R>);

// SAFETY: reads only the archetype's pair list.
unsafe impl<R: Relation> WorldQuery for Target<R> {
    type Item<'w> = Entity;
    type Slice<'w> = Entity;
    type State = ComponentId;
    type Fetch<'w> = Entity;

    fn empty_fetch<'w>() -> Self::Fetch<'w> {
        Entity::new(u32::MAX, std::num::NonZeroU32::MIN)
    }

    fn init_state(registry: &mut ComponentRegistry) -> Self::State {
        const {
            assert!(
                R::EXCLUSIVE,
                "Target<R> needs an exclusive relation: an entity has at most one target"
            );
        }
        registry.register::<R>()
    }

    fn access_set(_state: &Self::State, _report: &mut dyn FnMut(ComponentAccess)) {}

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        archetype.has_relation(*state)
    }

    #[inline(always)]
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *mut Archetype,
        _ctx: FetchContext,
    ) -> Self::Fetch<'w> {
        // SAFETY: the caller guarantees the archetype is valid for reads.
        let arch = unsafe { &*archetype };
        arch.target(*state)
            .expect("matching archetype has a pair of the relation")
    }

    #[inline(always)]
    unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
        (fetch, fetch)
    }

    #[inline(always)]
    unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, _len: usize) -> Self::Slice<'w> {
        fetch
    }

    #[inline(always)]
    unsafe fn seek<'w>(fetch: Self::Fetch<'w>, _rows: usize) -> Self::Fetch<'w> {
        fetch
    }
}

/// Implements [`WorldQuery`] for a per-row change filter over a column's ticks. `$bulk` is
/// whether the column's bulk change counts (it does for changes, not for additions).
macro_rules! change_filter {
    ($(#[$doc:meta])* $name:ident, $field:ident, $bulk:expr) => {
        $(#[$doc])*
        pub struct $name<T: Component>(PhantomData<T>);

        // SAFETY: matches archetypes with a `T` column and reads only that column's change ticks,
        // which is not an access to `T`'s values.
        unsafe impl<T: Component> WorldQuery for $name<T> {
            type Item<'w> = ();
            type Slice<'w> = ();
            type State = ComponentId;
            /// The current row's ticks, the run's ticks, and whether the column's bulk change
            /// is newer than the previous run (every row passes then).
            type Fetch<'w> = (*const ComponentTicks, ChangeTicks, bool);

            const DENSE: bool = false;

            fn empty_fetch<'w>() -> Self::Fetch<'w> {
                (
                    std::ptr::null(),
                    ChangeTicks {
                        last_run: Tick::default(),
                        this_run: Tick::default(),
                    },
                    false,
                )
            }

            fn init_state(registry: &mut ComponentRegistry) -> Self::State {
                registry.register::<T>()
            }

            fn access_set(_state: &Self::State, _report: &mut dyn FnMut(ComponentAccess)) {}

            fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
                archetype.has_component(*state)
            }

            fn mentions(state: &Self::State, id: ComponentId) -> bool {
                *state == id
            }

            #[inline(always)]
            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *mut Archetype,
                ctx: FetchContext,
            ) -> Self::Fetch<'w> {
                let ticks = ctx.ticks;
                // SAFETY: the caller guarantees the archetype is valid for reads.
                let arch = unsafe { &*archetype };
                let column = arch
                    .column(*state)
                    .expect("Missing column in matching archetype");
                let bulk = $bulk
                    && column
                        .bulk_changed()
                        .is_some_and(|tick| tick.is_newer_than(ticks.last_run, ticks.this_run));
                (column.ticks().as_ptr(), ticks, bulk)
            }

            #[inline(always)]
            unsafe fn filter<'w>(fetch: Self::Fetch<'w>) -> bool {
                let (row, ticks, bulk) = fetch;
                // SAFETY: `row` points at the current row's ticks.
                bulk || unsafe { (*row).$field }.is_newer_than(ticks.last_run, ticks.this_run)
            }

            #[inline(always)]
            unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
                // SAFETY: advancing stays within the column's ticks (one per row).
                ((), (unsafe { fetch.0.add(1) }, fetch.1, fetch.2))
            }

            #[inline(always)]
            unsafe fn fetch_slice<'w>(_fetch: Self::Fetch<'w>, _len: usize) -> Self::Slice<'w> {}

            #[inline(always)]
            unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w> {
                // SAFETY: the caller guarantees `rows` rows from the current position.
                (unsafe { fetch.0.add(rows) }, fetch.1, fetch.2)
            }
        }
    };
}

change_filter!(
    /// Filter matching entities whose `T` was added since the query's previous run.
    Added,
    added,
    false
);

change_filter!(
    /// Filter matching entities whose `T` was added or mutably accessed since the query's
    /// previous run. A mutable query marks every row of an archetype it visits; `get_mut` and
    /// overwriting with `add_component` mark one row.
    Changed,
    changed,
    true
);

macro_rules! impl_query_tuple {
    ($($name:ident),*) => {
        // SAFETY: each term upholds the contract; the tuple reports the union of their accesses,
        // matches the intersection of their archetypes, filters on all of them and fetches and
        // finishes through each term unchanged.
        unsafe impl<$($name: WorldQuery),*> WorldQuery for ($($name,)*) {
            type Item<'w> = ($($name::Item<'w>,)*);
            type Slice<'w> = ($($name::Slice<'w>,)*);
            type State = ($($name::State,)*);
            type Fetch<'w> = ($($name::Fetch<'w>,)*);

            const DENSE: bool = $($name::DENSE &&)* true;

            fn empty_fetch<'w>() -> Self::Fetch<'w> {
                ($($name::empty_fetch(),)*)
            }

            fn init_state(registry: &mut ComponentRegistry) -> Self::State {
                ($($name::init_state(registry),)*)
            }

            #[allow(non_snake_case)]
            fn access_set(state: &Self::State, report: &mut dyn FnMut(ComponentAccess)) {
                let ($($name,)*) = state;
                $($name::access_set($name, report);)*
            }

            #[allow(non_snake_case)]
            fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
                let ($($name,)*) = state;
                $($name::matches_archetype($name, archetype) &&)* true
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *mut Archetype,
                ctx: FetchContext,
            ) -> Self::Fetch<'w> {
                let ($($name,)*) = state;
                // SAFETY: caller guarantees archetype matches query.
                unsafe {
                    ($($name::init_fetch($name, archetype, ctx),)*)
                }
            }

            #[allow(non_snake_case)]
            fn mentions(state: &Self::State, id: ComponentId) -> bool {
                let ($($name,)*) = state;
                $($name::mentions($name, id) ||)* false
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn filter<'w>(fetch: Self::Fetch<'w>) -> bool {
                let ($($name,)*) = fetch;
                // SAFETY: the caller's guarantee holds for each term's fetch.
                unsafe { $($name::filter($name) &&)* true }
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn fetch_next<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
                let ($($name,)*) = fetch;
                // SAFETY: caller guarantees each term fetch pointer is valid for current row in active archetype.
                $(let $name = unsafe { $name::fetch_next($name) };)*
                (($($name.0,)*), ($($name.1,)*))
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            fn is_dense<'w>(fetch: Self::Fetch<'w>) -> bool {
                let ($($name,)*) = fetch;
                $($name::is_dense($name) &&)* true
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn fetch_next_dense<'w>(fetch: Self::Fetch<'w>) -> (Self::Item<'w>, Self::Fetch<'w>) {
                let ($($name,)*) = fetch;
                // SAFETY: the caller's guarantee holds for each term's fetch.
                $(let $name = unsafe { $name::fetch_next_dense($name) };)*
                (($($name.0,)*), ($($name.1,)*))
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn fetch_slice<'w>(fetch: Self::Fetch<'w>, len: usize) -> Self::Slice<'w> {
                let ($($name,)*) = fetch;
                // SAFETY: the caller's guarantee holds for each term's fetch.
                unsafe { ($($name::fetch_slice($name, len),)*) }
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn seek<'w>(fetch: Self::Fetch<'w>, rows: usize) -> Self::Fetch<'w> {
                let ($($name,)*) = fetch;
                // SAFETY: the caller's guarantee holds for each term's fetch.
                unsafe { ($($name::seek($name, rows),)*) }
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn mark_row(
                state: &Self::State,
                archetype: *mut Archetype,
                row: usize,
                ticks: ChangeTicks,
            ) {
                let ($($name,)*) = state;
                // SAFETY: the caller's guarantee holds for each term.
                unsafe { $($name::mark_row($name, archetype, row, ticks);)* }
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            unsafe fn finish_fetch(state: &Self::State, archetype: *mut Archetype, ticks: ChangeTicks) {
                let ($($name,)*) = state;
                // SAFETY: the caller's guarantee holds for each term.
                unsafe { $($name::finish_fetch($name, archetype, ticks);)* }
            }
        }
    };
}

impl_query_tuple!(A, B);
impl_query_tuple!(A, B, C);
impl_query_tuple!(A, B, C, D);
impl_query_tuple!(A, B, C, D, E);
impl_query_tuple!(A, B, C, D, E, F);
impl_query_tuple!(A, B, C, D, E, F, G);
impl_query_tuple!(A, B, C, D, E, F, G, H);

/// Iterator yielding query items across the matching archetypes.
///
/// Built by [`World::query_mut`](crate::World::query_mut), which visits every archetype and
/// matches it, or by [`QueryState::iter_mut`], which visits only the state's matched archetypes.
/// When it leaves an archetype (or is dropped) its mutable terms mark that archetype's rows
/// changed.
pub struct QueryIter<'w, Q: WorldQuery> {
    state: Q::State,
    ticks: ChangeTicks,
    archetypes: *mut [Archetype],
    /// The world's entity records, for inherited terms.
    entities: *const EntityManager,
    /// Visit archetypes holding `Prefab` (when `matched` is `None`).
    include_prefabs: bool,
    /// The active archetype's fetch steps every term by one row (nothing inherited).
    dense: bool,
    /// The archetypes to visit, already matched; `None` visits every archetype and matches it.
    matched: Option<&'w [usize]>,
    /// Next position in `matched`, or the next archetype index when `matched` is `None`.
    cursor: usize,
    /// The archetype `fetch` was prepared for, until `finish_fetch` has run for it.
    current: Option<*mut Archetype>,
    /// Rows of the active archetype already visited, and its row count.
    row: usize,
    len: usize,
    fetch: Q::Fetch<'w>,
    _marker: PhantomData<&'w mut [Archetype]>,
}

impl<'w, Q: WorldQuery> QueryIter<'w, Q> {
    /// Creates a query iterator that visits and matches every archetype.
    ///
    /// Crate-private: `state` must come from the registry that built `archetypes`, and its
    /// accesses must have passed [`validate_query_access`].
    pub(crate) fn new(
        state: Q::State,
        archetypes: &'w mut [Archetype],
        entities: &'w EntityManager,
        ticks: ChangeTicks,
        include_prefabs: bool,
    ) -> Self {
        // SAFETY: the borrows outlive the iterator; the caller guarantees the rest.
        unsafe { Self::from_raw(state, archetypes, entities, None, ticks, include_prefabs) }
    }

    /// Creates a query iterator that visits only `matched`.
    ///
    /// Crate-private: as for [`new`](Self::new), and every index in `matched` must name an
    /// archetype of `archetypes` that matches `Q`.
    pub(crate) fn with_matched(
        state: Q::State,
        archetypes: &'w mut [Archetype],
        entities: &'w EntityManager,
        matched: &'w [usize],
        ticks: ChangeTicks,
    ) -> Self {
        // SAFETY: the borrows outlive the iterator; the caller guarantees the rest.
        unsafe { Self::from_raw(state, archetypes, entities, Some(matched), ticks, false) }
    }

    /// Creates a query iterator over the archetypes behind `archetypes`.
    ///
    /// # Safety
    /// As for [`with_matched`](Self::with_matched) (or [`new`](Self::new) when `matched` is
    /// `None`), and for `'w` nothing may access the columns `Q` reads or writes except through
    /// this iterator (system parameters share one pointer, checked by their access sets).
    pub(crate) unsafe fn from_raw(
        state: Q::State,
        archetypes: *mut [Archetype],
        entities: *const EntityManager,
        matched: Option<&'w [usize]>,
        ticks: ChangeTicks,
        include_prefabs: bool,
    ) -> Self {
        Self {
            state,
            ticks,
            archetypes,
            entities,
            include_prefabs,
            dense: true,
            matched,
            cursor: 0,
            current: None,
            row: 0,
            len: 0,
            fetch: Q::empty_fetch(),
            _marker: PhantomData,
        }
    }

    /// Runs `finish_fetch` for the active archetype, if any.
    #[inline(always)]
    fn finish_current(&mut self) {
        if let Some(archetype) = self.current.take() {
            // SAFETY: `fetch` is not used for this archetype again; the items handed out point
            // into value columns and entity lists, not into the change ticks it writes.
            unsafe { Q::finish_fetch(&self.state, archetype, self.ticks) };
        }
    }

    /// Binds the next non-empty matching archetype; returns `false` when there is none.
    #[inline(always)]
    fn advance(&mut self) -> bool {
        self.finish_current();
        let count = self.matched.map_or(self.archetypes.len(), <[usize]>::len);
        while self.cursor < count {
            let index = self
                .matched
                .map_or(self.cursor, |matched| matched[self.cursor]);
            self.cursor += 1;
            assert!(
                index < self.archetypes.len(),
                "matched archetype out of range"
            );
            // SAFETY: `index` is in bounds (checked above).
            let archetype = unsafe { (self.archetypes as *mut Archetype).add(index) };
            // SAFETY: the archetype is valid for reads; the items handed out so far point into
            // column buffers and entity lists, not into the `Archetype` value itself.
            let arch = unsafe { &*archetype };
            let len = arch.len();
            if len == 0
                || (self.matched.is_none()
                    && (!Q::matches_archetype(&self.state, arch)
                        || (arch.is_prefab() && !self.include_prefabs)))
            {
                continue;
            }
            let ctx = FetchContext::new(self.archetypes, self.entities, self.ticks);
            // SAFETY: the archetype matches `Q` (checked here, or by the `QueryState` that built
            // `matched`) and the query's accesses were validated when it was built.
            self.fetch = unsafe { Q::init_fetch(&self.state, archetype, ctx) };
            self.dense = Q::is_dense(self.fetch);
            self.current = Some(archetype);
            self.row = 0;
            self.len = len;
            return true;
        }
        self.row = 0;
        self.len = 0;
        false
    }

    /// Executes `f` on every remaining query item.
    #[inline(always)]
    pub fn for_each<F>(mut self, mut f: F)
    where
        F: FnMut(Q::Item<'w>),
    {
        loop {
            let mut fetch = self.fetch;
            if self.dense {
                for _ in self.row..self.len {
                    let current = fetch;
                    // SAFETY: `fetch` stays within the active archetype's rows and is dense.
                    let (item, next) = unsafe { Q::fetch_next_dense(fetch) };
                    fetch = next;
                    // SAFETY: `current` is at a row of the active archetype.
                    if Q::DENSE || unsafe { Q::filter(current) } {
                        f(item);
                    }
                }
            } else {
                for _ in self.row..self.len {
                    let current = fetch;
                    // SAFETY: `fetch` stays within the active archetype's rows.
                    let (item, next) = unsafe { Q::fetch_next(fetch) };
                    fetch = next;
                    // SAFETY: `current` is at a row of the active archetype.
                    if Q::DENSE || unsafe { Q::filter(current) } {
                        f(item);
                    }
                }
            }
            if !self.advance() {
                return;
            }
        }
    }

    /// Executes `f` once per remaining archetype with the rows not yet yielded, as slices.
    ///
    /// Slices let the loop body run over contiguous arrays (and vectorize); an archetype's chunk
    /// holds every matching entity of that archetype. Queries with per-row filters (`Added`,
    /// `Changed`) cannot be iterated in chunks.
    pub fn for_each_chunk<F>(mut self, mut f: F)
    where
        F: FnMut(Q::Slice<'w>),
    {
        const {
            assert!(
                Q::DENSE,
                "for_each_chunk needs a query without per-row filters"
            );
        }
        loop {
            if self.row < self.len {
                // SAFETY: `fetch` is at row `row` of the active archetype, which has `len` rows;
                // the rows before `row` were yielded by `next` and are not part of the chunk.
                f(unsafe { Q::fetch_slice(self.fetch, self.len - self.row) });
            }
            if !self.advance() {
                return;
            }
        }
    }
}

impl<'w, Q: WorldQuery> Iterator for QueryIter<'w, Q> {
    type Item = Q::Item<'w>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.row == self.len && !self.advance() {
                return None;
            }
            self.row += 1;
            let current = self.fetch;
            // SAFETY: `row <= len`, so `fetch` is at a row of the active archetype.
            let (item, next) = unsafe { Q::fetch_next(current) };
            self.fetch = next;
            // SAFETY: `current` is at a row of the active archetype.
            if Q::DENSE || unsafe { Q::filter(current) } {
                return Some(item);
            }
        }
    }
}

impl<Q: WorldQuery> Drop for QueryIter<'_, Q> {
    fn drop(&mut self) {
        self.finish_current();
    }
}

/// A query kept between runs: its resolved state, the archetypes that match it, and the tick of
/// its previous run for [`Added`] and [`Changed`].
///
/// Built by [`World::query_state`](crate::World::query_state) and bound to that world. Each run
/// first checks the archetypes created since the previous one; when the world has freed or reused
/// an archetype slot since, it matches every archetype again.
pub struct QueryState<Q: WorldQuery> {
    state: Q::State,
    world_id: u64,
    matched_archetypes: Vec<usize>,
    /// Archetypes below this index are in `matched_archetypes` or do not match.
    scanned: usize,
    /// The world's archetype epoch when `matched_archetypes` was last brought up to date against
    /// it; `None` after [`update_archetypes`](Self::update_archetypes).
    epoch: Option<u64>,
    /// `this_run` of the previous run.
    last_run: Tick,
    /// A term names `Prefab`, so prefabs match.
    include_prefabs: bool,
}

impl<Q: WorldQuery> QueryState<Q> {
    /// Creates a cached query state for `Q` against the world identified by `world_id`; its
    /// first run reports every component as added and changed.
    pub(crate) fn new(registry: &mut ComponentRegistry, world_id: u64, created: Tick) -> Self {
        let state = Q::init_state(registry);
        validate_query_access::<Q>(&state, registry);
        let include_prefabs = includes_prefabs::<Q>(&state, registry);
        Self {
            state,
            include_prefabs,
            world_id,
            matched_archetypes: Vec::new(),
            scanned: 0,
            epoch: None,
            last_run: Tick::new(created.get().wrapping_sub(MAX_CHANGE_AGE)),
        }
    }

    /// Matches every archetype of `archetypes` again.
    ///
    /// The next run against the world re-matches from scratch as well, since `archetypes` need
    /// not be that world's.
    pub fn update_archetypes(&mut self, archetypes: &[Archetype]) {
        self.matched_archetypes.clear();
        self.scanned = 0;
        self.scan(archetypes);
        self.epoch = None;
    }

    /// Brings the matched archetypes up to date with `world`.
    ///
    /// # Panics
    /// If `world` is not the world this state was created from.
    pub fn update(&mut self, world: &World) {
        assert_eq!(
            self.world_id,
            world.id(),
            "QueryState used with a World it was not created from"
        );
        if self.epoch != Some(world.archetype_epoch()) {
            self.matched_archetypes.clear();
            self.scanned = 0;
            self.epoch = Some(world.archetype_epoch());
        }
        self.scan(world.archetypes());
    }

    fn scan(&mut self, archetypes: &[Archetype]) {
        for (index, archetype) in archetypes.iter().enumerate().skip(self.scanned) {
            if !archetype.is_free()
                && (!archetype.is_prefab() || self.include_prefabs)
                && Q::matches_archetype(&self.state, archetype)
            {
                self.matched_archetypes.push(index);
            }
        }
        self.scanned = archetypes.len();
    }

    /// Iterates the matching entities of `world`; starts a new run for [`Added`] and
    /// [`Changed`].
    ///
    /// # Panics
    /// If `world` is not the world this state was created from.
    pub fn iter_mut<'w>(&'w mut self, world: &'w mut World) -> QueryIter<'w, Q> {
        self.update(world);
        let this_run = world.increment_change_tick();
        self.last_run.clamp_age(this_run);
        let ticks = ChangeTicks {
            last_run: std::mem::replace(&mut self.last_run, this_run),
            this_run,
        };
        let (archetypes, entities) = world.archetypes_and_entities();
        QueryIter::with_matched(
            self.state,
            archetypes,
            entities,
            &self.matched_archetypes,
            ticks,
        )
    }

    /// Executes `f` on every matching entity of `world`.
    ///
    /// # Panics
    /// If `world` is not the world this state was created from.
    pub fn for_each<'w>(&'w mut self, world: &'w mut World, f: impl FnMut(Q::Item<'w>)) {
        self.iter_mut(world).for_each(f);
    }

    /// Executes `f` once per matching archetype of `world`, with its rows as slices.
    ///
    /// # Panics
    /// If `world` is not the world this state was created from.
    pub fn for_each_chunk<'w>(&'w mut self, world: &'w mut World, f: impl FnMut(Q::Slice<'w>)) {
        self.iter_mut(world).for_each_chunk(f);
    }

    /// Returns the resolved state for this query.
    #[must_use]
    pub const fn state(&self) -> &Q::State {
        &self.state
    }

    /// Returns `true` if a term names `Prefab`, so that prefabs match.
    #[must_use]
    pub const fn includes_prefabs(&self) -> bool {
        self.include_prefabs
    }

    /// Returns the indices of the matching archetypes, as of the last update.
    #[must_use]
    pub fn matched_archetypes(&self) -> &[usize] {
        &self.matched_archetypes
    }
}
