//! Generational entity identifiers and lifecycle allocation.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};

/// A lightweight, generational entity identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Entity {
    index: u32,
    generation: NonZeroU32,
}

impl Entity {
    /// Creates a new `Entity` with the specified index and generation.
    #[must_use]
    pub const fn new(index: u32, generation: NonZeroU32) -> Self {
        Self { index, generation }
    }

    /// Returns the dense index of the entity.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the generation of the entity to detect stale references.
    #[must_use]
    pub const fn generation(self) -> NonZeroU32 {
        self.generation
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index, self.generation.get())
    }
}

/// Location of an entity inside archetype table storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityLocation {
    /// Archetype index within the world.
    pub archetype_id: usize,
    /// Row index within the archetype table.
    pub row: usize,
}

/// Generation, location and flags of one entity index, kept together so that a structural
/// change touches one cache line per entity.
#[derive(Debug, Clone, Copy)]
struct EntityRecord {
    generation: NonZeroU32,
    /// Archetype index, or [`NO_ARCHETYPE`] while the index is free.
    archetype: u32,
    row: u32,
    /// [`MARKED`] and [`ISA_TARGET`].
    flags: u8,
}

const NO_ARCHETYPE: u32 = u32::MAX;
/// Set by [`EntityManager::mark`] for the world's own bookkeeping (the despawn queue).
const MARKED: u8 = 1;
/// The entity is the target of an `IsA` pair: instances inherit from it.
const ISA_TARGET: u8 = 2;

/// Manages entity generation tracking and index recycling.
///
/// Entities can also be reserved through a shared reference ([`reserve`](Self::reserve)), as
/// commands do while a system runs: a reservation predicts the entity the next allocation will
/// return, and [`flush_reserved`](Self::flush_reserved) performs the reserved allocations in
/// order before anything else allocates.
#[derive(Debug, Default)]
pub struct EntityManager {
    records: Vec<EntityRecord>,
    free_indices: Vec<u32>,
    alive_count: usize,
    reserved: AtomicU32,
}

impl EntityManager {
    /// Creates a new, empty `EntityManager`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of currently alive entities.
    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.alive_count
    }

    /// Returns the total entity index capacity (number of allocated indices).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.records.len()
    }

    /// Allocates a new entity or recycles a freed entity index with an incremented generation.
    ///
    /// The entity has no location until [`set_location`](Self::set_location) gives it one.
    /// Pending reservations are allocated first, so they keep the entities they predicted.
    pub fn allocate(&mut self) -> Entity {
        self.flush_reserved();
        self.allocate_now()
    }

    fn allocate_now(&mut self) -> Entity {
        self.alive_count += 1;
        if let Some(index) = self.free_indices.pop() {
            Entity::new(index, self.records[index as usize].generation)
        } else {
            let index = u32::try_from(self.records.len()).expect("entity index space exhausted");
            let generation = NonZeroU32::MIN;
            self.records.push(EntityRecord {
                generation,
                archetype: NO_ARCHETYPE,
                row: 0,
                flags: 0,
            });
            Entity::new(index, generation)
        }
    }

    /// Reserves the entity a later allocation will return, without changing the manager.
    pub(crate) fn reserve(&self) -> Entity {
        let k = self.reserved.fetch_add(1, Ordering::Relaxed) as usize;
        let free = self.free_indices.len();
        if k < free {
            // `allocate` pops the free list from its end.
            let index = self.free_indices[free - 1 - k];
            Entity::new(index, self.records[index as usize].generation)
        } else {
            let index = u32::try_from(self.records.len() + (k - free))
                .expect("entity index space exhausted");
            Entity::new(index, NonZeroU32::MIN)
        }
    }

    /// Performs the pending reservations' allocations, in the order they were made. The
    /// entities get no location; whoever reserved them places them.
    pub(crate) fn flush_reserved(&mut self) {
        let pending = std::mem::take(self.reserved.get_mut());
        for _ in 0..pending {
            self.allocate_now();
        }
    }

    /// Returns `true` if `entity` is allocated (current generation) but has no location yet.
    pub(crate) fn is_unplaced(&self, entity: Entity) -> bool {
        self.record(entity)
            .is_some_and(|record| record.archetype == NO_ARCHETYPE)
    }

    /// The record of `entity` if `entity` is its index's current generation.
    #[inline]
    fn record(&self, entity: Entity) -> Option<&EntityRecord> {
        self.records
            .get(entity.index as usize)
            .filter(|record| record.generation == entity.generation)
    }

    /// Checks if the given entity handle is currently alive.
    #[must_use]
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.record(entity)
            .is_some_and(|record| record.archetype != NO_ARCHETYPE)
    }

    /// Retrieves the archetype storage location for the given entity.
    #[must_use]
    #[inline]
    pub fn location(&self, entity: Entity) -> Option<EntityLocation> {
        let record = self.record(entity)?;
        (record.archetype != NO_ARCHETYPE).then_some(EntityLocation {
            archetype_id: record.archetype as usize,
            row: record.row as usize,
        })
    }

    /// Updates the storage location of an entity.
    ///
    /// # Panics
    /// If the archetype index or the row does not fit in 32 bits.
    #[inline]
    pub fn set_location(&mut self, entity: Entity, location: EntityLocation) {
        let archetype = u32::try_from(location.archetype_id)
            .ok()
            .filter(|&a| a != NO_ARCHETYPE)
            .expect("archetype index out of range");
        let row = u32::try_from(location.row).expect("row out of range");
        if let Some(record) = self.records.get_mut(entity.index as usize) {
            if record.generation == entity.generation {
                record.archetype = archetype;
                record.row = row;
            }
        }
    }

    /// Deallocates an entity, incrementing its generation and freeing its index for reuse; an
    /// index at its last generation is retired instead.
    #[inline]
    pub fn free(&mut self, entity: Entity) -> Option<EntityLocation> {
        let record = self.records.get_mut(entity.index as usize)?;
        if record.generation != entity.generation || record.archetype == NO_ARCHETYPE {
            return None;
        }
        let location = EntityLocation {
            archetype_id: record.archetype as usize,
            row: record.row as usize,
        };
        record.archetype = NO_ARCHETYPE;
        // The despawn cascade still reads the mark of entities it has freed.
        record.flags &= MARKED;
        // An index at its last generation is retired rather than wrapped back to 1, which would
        // make old handles to it valid again (as Flecs does at its generation limit).
        if let Some(next) = record.generation.checked_add(1) {
            record.generation = next;
            self.free_indices.push(entity.index);
        }
        self.alive_count -= 1;
        Some(location)
    }

    /// Sets the mark of `entity`'s index; returns `false` if it was already set.
    #[inline]
    pub(crate) fn mark(&mut self, entity: Entity) -> bool {
        let record = &mut self.records[entity.index as usize];
        let was_marked = record.flags & MARKED != 0;
        record.flags |= MARKED;
        !was_marked
    }

    /// Returns `true` if the index of `entity` is marked.
    #[inline]
    pub(crate) fn is_marked(&self, entity: Entity) -> bool {
        self.records
            .get(entity.index as usize)
            .is_some_and(|record| record.flags & MARKED != 0)
    }

    /// Clears the mark of `entity`'s index.
    #[inline]
    pub(crate) fn unmark(&mut self, entity: Entity) {
        self.records[entity.index as usize].flags &= !MARKED;
    }

    /// Records that `entity` is the target of an `IsA` pair (until it is freed).
    pub(crate) fn set_isa_target(&mut self, entity: Entity) {
        if let Some(record) = self.records.get_mut(entity.index as usize) {
            if record.generation == entity.generation {
                record.flags |= ISA_TARGET;
            }
        }
    }

    /// Returns `true` if `entity` is, or was since it was allocated, the target of an `IsA` pair.
    #[inline]
    pub(crate) fn is_isa_target(&self, entity: Entity) -> bool {
        self.record(entity)
            .is_some_and(|record| record.flags & ISA_TARGET != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_index_that_reaches_the_last_generation_is_retired() {
        let mut manager = EntityManager::new();
        let first = manager.allocate();
        manager.set_location(
            first,
            EntityLocation {
                archetype_id: 0,
                row: 0,
            },
        );
        // Fast-forward the index to its last generation, as 2^32 - 1 reuses would.
        manager.records[first.index as usize].generation = NonZeroU32::MAX;
        let last = Entity::new(first.index, NonZeroU32::MAX);
        assert!(manager.free(last).is_some());

        // Reusing the index would wrap its generation back to 1 and revive old handles.
        let next = manager.allocate();
        assert_ne!(next.index(), first.index());
        assert!(!manager.is_alive(Entity::new(first.index, NonZeroU32::MIN)));
        assert!(!manager.is_alive(last));
        assert_eq!(manager.alive_count(), 1);
    }
}
