//! Generational slot map and handle system for safe, cache-friendly object references.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU32;

/// A lightweight, strongly typed generational handle referencing an element in a [`SlotMap`].
pub struct Handle<T> {
    index: u32,
    generation: NonZeroU32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    /// Internal constructor for handle creation.
    pub const fn new(index: u32, generation: NonZeroU32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// Returns the dense slot index of this handle.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the generation number of this handle.
    #[must_use]
    pub const fn generation(self) -> NonZeroU32 {
        self.generation
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> From<u32> for Handle<T> {
    fn from(index: u32) -> Self {
        Self::new(index, NonZeroU32::MIN)
    }
}

impl<T> From<i32> for Handle<T> {
    fn from(index: i32) -> Self {
        Self::new(index as u32, NonZeroU32::MIN)
    }
}

impl<T> From<Handle<T>> for u32 {
    fn from(handle: Handle<T>) -> Self {
        handle.index()
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> PartialEq<u32> for Handle<T> {
    fn eq(&self, other: &u32) -> bool {
        self.index == *other
    }
}

impl<T> PartialEq<Handle<T>> for u32 {
    fn eq(&self, other: &Handle<T>) -> bool {
        *self == other.index
    }
}

impl<T> Eq for Handle<T> {}

impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index
            .cmp(&other.index)
            .then_with(|| self.generation.cmp(&other.generation))
    }
}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("index", &self.index)
            .field("generation", &self.generation.get())
            .finish()
    }
}

enum SlotState<T> {
    Occupied(T),
    Free { next_free: Option<u32> },
}

struct Slot<T> {
    generation: NonZeroU32,
    state: SlotState<T>,
}

/// An associative container with generational indices preventing ABA dangling reference bugs.
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
    len: usize,
}

impl<T> Default for SlotMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SlotMap<T> {
    /// Creates an empty `SlotMap`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    /// Creates an empty `SlotMap` pre-allocated for the given capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
        }
    }

    /// Returns the number of live elements in the slot map.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the slot map contains no live elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts a new value into the slot map, returning a typed [`Handle`].
    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.len += 1;

        if let Some(free_idx) = self.free_head {
            let slot = &mut self.slots[free_idx as usize];
            let next_free = match slot.state {
                SlotState::Free { next_free } => next_free,
                SlotState::Occupied(_) => unreachable!("Corrupted free list in SlotMap"),
            };
            self.free_head = next_free;
            slot.state = SlotState::Occupied(value);

            Handle::new(free_idx, slot.generation)
        } else {
            let new_idx = self.slots.len() as u32;
            let generation = NonZeroU32::new(1).unwrap();
            self.slots.push(Slot {
                generation,
                state: SlotState::Occupied(value),
            });

            Handle::new(new_idx, generation)
        }
    }

    /// Returns a reference to the element corresponding to the handle, or `None` if stale or invalid.
    #[must_use]
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation == handle.generation {
            if let SlotState::Occupied(ref val) = slot.state {
                return Some(val);
            }
        }
        None
    }

    /// Returns a mutable reference to the element corresponding to the handle, or `None` if stale or invalid.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation == handle.generation {
            if let SlotState::Occupied(ref mut val) = slot.state {
                return Some(val);
            }
        }
        None
    }

    /// Removes and returns the element corresponding to the handle, or `None` if stale or invalid.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot_idx = handle.index as usize;
        let slot = self.slots.get_mut(slot_idx)?;

        if slot.generation != handle.generation {
            return None;
        }

        // Increment generation on removal so older handles are instantly invalidated
        let next_generation = slot
            .generation
            .get()
            .checked_add(1)
            .and_then(NonZeroU32::new)
            .unwrap_or_else(|| NonZeroU32::new(1).unwrap());
        slot.generation = next_generation;

        let prev_state = std::mem::replace(
            &mut slot.state,
            SlotState::Free {
                next_free: self.free_head,
            },
        );
        self.free_head = Some(handle.index);
        self.len -= 1;

        match prev_state {
            SlotState::Occupied(val) => Some(val),
            SlotState::Free { .. } => None,
        }
    }

    /// Clears all entries from the slot map while retaining memory.
    pub fn clear(&mut self) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if let SlotState::Occupied(_) = slot.state {
                slot.generation = slot
                    .generation
                    .get()
                    .checked_add(1)
                    .and_then(NonZeroU32::new)
                    .unwrap_or_else(|| NonZeroU32::new(1).unwrap());
                slot.state = SlotState::Free {
                    next_free: self.free_head,
                };
                self.free_head = Some(i as u32);
            }
        }
        self.len = 0;
    }
}
