//! Linear bump frame allocator for zero-allocation scratchpad operations.

use std::cell::Cell;
use std::mem;

/// Linear bump allocator designed for per-frame scratch memory.
///
/// Disallows individual deallocations; all memory is reclaimed instantaneously via [`reset`](Self::reset).
#[derive(Debug)]
pub struct FrameArena {
    buffer: Box<[u8]>,
    offset: Cell<usize>,
}

impl FrameArena {
    /// Creates a new `FrameArena` with the specified byte capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let buffer = vec![0u8; capacity].into_boxed_slice();
        Self {
            buffer,
            offset: Cell::new(0),
        }
    }

    /// Returns the total capacity of the arena in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of bytes currently allocated in this frame.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.offset.get()
    }

    /// Allocates a contiguous mutable slice of `count` elements of type `T`.
    /// Elements are initialized to their default values.
    ///
    /// Returns `None` if the arena does not have enough remaining capacity.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice<T: Copy + Default>(&self, count: usize) -> Option<&mut [T]> {
        if count == 0 {
            return Some(&mut []);
        }

        let align = mem::align_of::<T>();
        let size = mem::size_of::<T>();
        let total_bytes = size.checked_mul(count)?;

        let current_offset = self.offset.get();
        let base_ptr = self.buffer.as_ptr() as usize;
        let current_ptr = base_ptr + current_offset;

        // Calculate alignment padding
        let padding = (align - (current_ptr % align)) % align;
        let start_offset = current_offset.checked_add(padding)?;
        let end_offset = start_offset.checked_add(total_bytes)?;

        if end_offset > self.buffer.len() {
            return None;
        }

        self.offset.set(end_offset);

        // SAFETY:
        // 1. `start_offset` to `end_offset` is within the bounds of `self.buffer`.
        // 2. `start_offset` is properly aligned for type `T`.
        // 3. Monotonically increasing `offset` guarantees disjoint slices across allocations.
        // 4. `reset(&mut self)` requires an exclusive reference to `Self`, ensuring all allocated
        //    slices have expired before memory can be overwritten.
        unsafe {
            let slice_ptr = self.buffer.as_ptr().add(start_offset) as *mut T;
            for i in 0..count {
                slice_ptr.add(i).write(T::default());
            }
            Some(std::slice::from_raw_parts_mut(slice_ptr, count))
        }
    }

    /// Resets the bump allocator offset back to zero.
    ///
    /// Requires an exclusive `&mut self` borrow, ensuring no previously allocated slices are aliased.
    pub fn reset(&mut self) {
        self.offset.set(0);
    }
}
