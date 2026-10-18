//! Slices that the step's workers read and write in disjoint parts.
//!
//! The parallel parts of a step hand each worker ranges of the same arrays: bodies, motions,
//! constraints, contacts. Which elements a worker may touch follows from the step's structure
//! (blocks claimed through atomics, graph colors whose constraints share no dynamic body), not
//! from the borrow checker, so [`Shared`] gives element access through `unsafe` methods whose
//! callers prove that structure. Indices are bounds-checked; only aliasing is the caller's to
//! prove.

use std::marker::PhantomData;

/// A mutable slice whose elements several threads access at once, each element by at most one
/// writer at a time.
pub(crate) struct Shared<'a, T> {
    ptr: *mut T,
    len: usize,
    _slice: PhantomData<&'a mut [T]>,
}

impl<T> Clone for Shared<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Shared<'_, T> {}

// SAFETY: a `Shared` hands out `&mut T` on other threads, so `T` must be `Send`; and `&T` on
// several threads at once, so `T` must be `Sync`. Callers of the accessors prove that no two
// threads access one element at once unless both only read.
unsafe impl<T: Send + Sync> Send for Shared<'_, T> {}

// SAFETY: as for `Send`.
unsafe impl<T: Send + Sync> Sync for Shared<'_, T> {}

impl<'a, T> Shared<'a, T> {
    /// Shares `slice` for the lifetime `'a`, during which it is reached only through this view.
    pub fn new(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _slice: PhantomData,
        }
    }

    /// Length of the shared slice.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// The element at `index`, mutably.
    ///
    /// # Safety
    ///
    /// While the reference lives, no other reference to the element may exist, on any thread.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut(&self, index: usize) -> &mut T {
        assert!(index < self.len, "index {index} out of {}", self.len);
        // SAFETY: `index` is in bounds of the slice borrowed for `'a`, and the caller
        // guarantees the reference is the only one to the element.
        unsafe { &mut *self.ptr.add(index) }
    }

    /// The element at `index`.
    ///
    /// # Safety
    ///
    /// While the reference lives, no mutable reference to the element may exist, on any
    /// thread.
    pub unsafe fn get(&self, index: usize) -> &T {
        assert!(index < self.len, "index {index} out of {}", self.len);
        // SAFETY: `index` is in bounds, and the caller guarantees no concurrent writer.
        unsafe { &*self.ptr.add(index) }
    }

    /// A copy of the element at `index`.
    ///
    /// # Safety
    ///
    /// No other thread may write the element during the call.
    pub unsafe fn read(&self, index: usize) -> T
    where
        T: Copy,
    {
        // SAFETY: forwarded from the caller.
        unsafe { *self.get(index) }
    }
}
