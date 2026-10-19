//! Linear bump allocator (`FrameArena`) for fast, zero-fragmentation temporary per-frame allocations.
//!
//! # Inputs / Outputs
//! Takes a byte capacity at initialization and allocates aligned, default-initialized mutable
//! slices (`&mut [T]`) in $O(1)$ time without per-allocation heap calls. Calling [`FrameArena::reset`]
//! rewinds the bump offset to `0` in $O(1)$ time while retaining the backing allocation.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::cell::Cell;
use std::ptr::NonNull;

/// Base alignment (64 bytes, one cache line) guaranteed for the arena's backing buffer.
const BASE_ALIGN: usize = 64;

/// A fast bump-pointer arena allocator for temporary frame data.
///
/// The backing buffer is allocated once with 64-byte cache-line alignment and held via a raw
/// [`NonNull<u8>`] pointer so that sequential [`FrameArena::alloc_slice`] calls derive disjoint
/// mutable slices directly from the root allocation provenance without aliasing live slices.
pub struct FrameArena {
    ptr: NonNull<u8>,
    capacity: usize,
    offset: Cell<usize>,
}

// SAFETY: `FrameArena` uniquely owns its heap buffer (`ptr..ptr + capacity`) and does not share it
// across threads without synchronization (`Cell<usize>` makes `FrameArena` `!Sync`).
unsafe impl Send for FrameArena {}

impl FrameArena {
    /// Creates a new `FrameArena` with the specified capacity in bytes, aligned to 64 bytes.
    ///
    /// # Panics
    /// Panics if `capacity` exceeds `isize::MAX` when rounded to 64-byte alignment.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::without_provenance(
                    std::num::NonZeroUsize::new(BASE_ALIGN).expect("non-zero base align"),
                ),
                capacity: 0,
                offset: Cell::new(0),
            };
        }

        let layout =
            Layout::from_size_align(capacity, BASE_ALIGN).expect("FrameArena capacity overflow");
        // SAFETY: `capacity > 0` and `BASE_ALIGN` is a non-zero power of two, so `layout` has
        // non-zero size. Null return is handled via `handle_alloc_error`.
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            handle_alloc_error(layout);
        };

        Self {
            ptr,
            capacity,
            offset: Cell::new(0),
        }
    }

    /// Returns the total byte capacity of the arena.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of bytes currently allocated in the arena.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.offset.get()
    }

    /// Resets the arena's allocation offset back to zero in $O(1)$ time without freeing memory.
    ///
    /// Requiring `&mut self` guarantees at compile time that all previously allocated `&mut [T]`
    /// slices (whose lifetimes are tied to `&self`) have expired before the bump pointer rewinds.
    pub fn reset(&mut self) {
        self.offset.set(0);
    }

    /// Allocates a contiguous, aligned mutable slice of `count` elements of type `T`, initialized
    /// to `T::default()`.
    ///
    /// Returns `None` if the arena does not have enough remaining capacity to satisfy the
    /// alignment and size requirements.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice<T: Copy + Default>(&self, count: usize) -> Option<&mut [T]> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        if size == 0 || count == 0 {
            if count > isize::MAX as usize {
                return None;
            }
            let dangling = NonNull::<T>::dangling().as_ptr();
            // SAFETY: When `size == 0` or `count == 0`, the slice spans 0 bytes, `dangling` is
            // non-null and aligned to `align_of::<T>()`, and `count <= isize::MAX`.
            return Some(unsafe { std::slice::from_raw_parts_mut(dangling, count) });
        }

        let total_bytes = size.checked_mul(count)?;
        if total_bytes > isize::MAX as usize {
            return None;
        }

        let current_offset = self.offset.get();
        let current_addr = self.ptr.as_ptr().addr().checked_add(current_offset)?;
        let padding = (align - (current_addr & (align - 1))) & (align - 1);

        let start_offset = current_offset.checked_add(padding)?;
        let next_offset = start_offset.checked_add(total_bytes)?;
        if next_offset > self.capacity {
            return None;
        }

        self.offset.set(next_offset);

        // SAFETY: `start_offset + total_bytes <= self.capacity`, so `start_offset..next_offset` lies
        // within the owned allocation `self.ptr` and is disjoint from all previous allocations
        // (`0..current_offset`) and future allocations (`next_offset..self.capacity`). Adding
        // `padding` guarantees that `ptr` is aligned to `align_of::<T>()`. Each element is
        // initialized with `T::default()` before forming the mutable slice tied to `&self`.
        unsafe {
            let ptr = self.ptr.as_ptr().add(start_offset).cast::<T>();
            let default_val = T::default();
            for i in 0..count {
                ptr.add(i).write(default_val);
            }
            Some(std::slice::from_raw_parts_mut(ptr, count))
        }
    }
}

impl Drop for FrameArena {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Layout::from_size_align(self.capacity, BASE_ALIGN)
                .expect("FrameArena layout was validated in with_capacity");
            // SAFETY: `self.ptr` was allocated with `alloc_zeroed(layout)` in `with_capacity` and
            // all borrowed slices (`&self`) have ended before `drop(&mut self)`.
            unsafe {
                dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}
