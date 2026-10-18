//! Generational ids for objects kept in parallel arrays (Box2D `b2IdPool`).
//!
//! A [`SlotMap`](crate::SlotMap) owns its values; an [`IdPool`] owns only the ids. A system that
//! stores its objects as structure-of-arrays indexes every array by [`Handle::index`] and uses
//! the pool to hand out ids, recycle freed slots and reject stale handles: freeing a slot
//! advances its generation, so a handle to a freed object stays invalid after the slot is reused.

use crate::slot_map::Handle;
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroU32;

/// Allocates and validates generational [`Handle`]s for slots of parallel arrays.
pub struct IdPool<T> {
    /// Current generation of each slot.
    generations: Vec<NonZeroU32>,
    /// Whether each slot holds a live object.
    live: Vec<bool>,
    /// Freed slots, reused last-freed first.
    free: Vec<u32>,
    len: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> IdPool<T> {
    /// Creates an empty pool.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            generations: Vec::new(),
            live: Vec::new(),
            free: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Allocates a handle, reusing the most recently freed slot if there is one. When the
    /// returned index equals the previous [`Self::slot_count`], the caller's arrays must grow.
    pub fn alloc(&mut self) -> Handle<T> {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            self.live[index as usize] = true;
            return Handle::new(index, self.generations[index as usize]);
        }
        let index = u32::try_from(self.generations.len()).expect("IdPool holds at most 2^32 slots");
        self.generations.push(NonZeroU32::MIN);
        self.live.push(true);
        Handle::new(index, NonZeroU32::MIN)
    }

    /// Frees `handle`'s slot and advances its generation. Returns `false`, changing nothing, if
    /// the handle is stale or foreign.
    pub fn free(&mut self, handle: Handle<T>) -> bool {
        if !self.contains(handle) {
            return false;
        }
        let index = handle.index() as usize;
        let generation = &mut self.generations[index];
        *generation = generation.checked_add(1).unwrap_or(NonZeroU32::MIN);
        self.live[index] = false;
        self.free.push(handle.index());
        self.len -= 1;
        true
    }

    /// Whether `handle` refers to a live object.
    #[must_use]
    pub fn contains(&self, handle: Handle<T>) -> bool {
        let index = handle.index() as usize;
        index < self.live.len()
            && self.live[index]
            && self.generations[index] == handle.generation()
    }

    /// The live handle of slot `index`, or `None` if the slot is free or out of range.
    #[must_use]
    pub fn handle_at(&self, index: u32) -> Option<Handle<T>> {
        let i = index as usize;
        (i < self.live.len() && self.live[i]).then(|| Handle::new(index, self.generations[i]))
    }

    /// Number of slots, live or free: the length the caller's arrays need.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.generations.len()
    }

    /// Number of live handles.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no handle is live.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The live handles in slot order.
    pub fn iter(&self) -> impl Iterator<Item = Handle<T>> + '_ {
        (0..self.generations.len() as u32).filter_map(|index| self.handle_at(index))
    }

    /// The pool as plain data, for saving it: each slot's generation and liveness, and the
    /// free list in reuse order.
    #[must_use]
    pub fn to_parts(&self) -> IdPoolParts {
        IdPoolParts {
            generations: self.generations.iter().map(|g| g.get()).collect(),
            live: self.live.clone(),
            free: self.free.clone(),
        }
    }

    /// The pool saved by [`Self::to_parts`], or `None` if the parts are inconsistent: lengths
    /// differ, a generation is zero, or the free list is not exactly the dead slots.
    #[must_use]
    pub fn from_parts(parts: IdPoolParts) -> Option<Self> {
        let IdPoolParts {
            generations,
            live,
            free,
        } = parts;
        if generations.len() != live.len() {
            return None;
        }
        let generations: Vec<NonZeroU32> = generations
            .into_iter()
            .map(NonZeroU32::new)
            .collect::<Option<_>>()?;
        let dead = live.iter().filter(|&&l| !l).count();
        let mut seen = vec![false; live.len()];
        for &slot in &free {
            let i = slot as usize;
            if i >= live.len() || live[i] || std::mem::replace(&mut seen[i], true) {
                return None;
            }
        }
        if free.len() != dead {
            return None;
        }
        Some(Self {
            len: live.len() - dead,
            generations,
            live,
            free,
            _marker: PhantomData,
        })
    }
}

/// An [`IdPool`] as plain data ([`IdPool::to_parts`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IdPoolParts {
    /// Each slot's generation, never zero.
    pub generations: Vec<u32>,
    /// Whether each slot holds a live object.
    pub live: Vec<bool>,
    /// The free slots, reused from the end first.
    pub free: Vec<u32>,
}

impl<T> Default for IdPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for IdPool<T> {
    fn clone(&self) -> Self {
        Self {
            generations: self.generations.clone(),
            live: self.live.clone(),
            free: self.free.clone(),
            len: self.len,
            _marker: PhantomData,
        }
    }

    /// Copies `source` into this pool, reusing its allocations.
    fn clone_from(&mut self, source: &Self) {
        self.generations.clone_from(&source.generations);
        self.live.clone_from(&source.live);
        self.free.clone_from(&source.free);
        self.len = source.len;
    }
}

impl<T> fmt::Debug for IdPool<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IdPool")
            .field("len", &self.len)
            .field("slot_count", &self.generations.len())
            .finish()
    }
}

impl<T> PartialEq for IdPool<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.generations == other.generations
            && self.live == other.live
            && self.free == other.free
    }
}

impl<T> Eq for IdPool<T> {}
