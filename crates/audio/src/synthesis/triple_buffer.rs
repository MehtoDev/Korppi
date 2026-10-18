//! Wait-free triple-buffer synchronization for transferring acoustic metadata across threads.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

const DIRTY_BIT: u8 = 0x04;
const INDEX_MASK: u8 = 0x03;

/// Shared internal storage holding 3 preallocated buffers and an atomic state register.
#[repr(align(64))]
struct SharedBuffer<T> {
    slots: [UnsafeCell<T>; 3],
    state: AtomicU8,
}

// SAFETY: Synchronization is guaranteed by the atomic state machine in `state`.
// At any time, the producer slot (`back`), consumer slot (`front`), and exchange slot (`shared`)
// are guaranteed to be mutually exclusive distinct indices from {0, 1, 2}.
unsafe impl<T: Send> Send for SharedBuffer<T> {}
// SAFETY: as above: shared references only reach the slots through the producer and consumer,
// which own distinct slots and need `&mut self` to touch them; values move between threads, so
// `T: Send` suffices. The handles are `Send` through their `Arc` of this state.
unsafe impl<T: Send> Sync for SharedBuffer<T> {}

/// Producer endpoint of the wait-free triple buffer.
pub struct TripleBufferProducer<T> {
    shared: Arc<SharedBuffer<T>>,
    back: usize,
}

impl<T> TripleBufferProducer<T> {
    /// Writes a new value into the back buffer and atomically publishes it to the shared slot.
    ///
    /// This operation is strictly wait-free ($O(1)$ atomic swap) and performs zero allocations.
    pub fn write(&mut self, value: T) {
        // SAFETY: The atomic state protocol guarantees that `self.back` is owned exclusively
        // by the producer and is not accessible by the consumer while being written.
        unsafe {
            *self.shared.slots[self.back].get() = value;
        }

        // Atomically publish `self.back` into `shared` and mark as dirty
        let publish_val = (self.back as u8 & INDEX_MASK) | DIRTY_BIT;
        let old_state = self.shared.state.swap(publish_val, Ordering::AcqRel);

        // The previous shared index now becomes the producer's private back buffer
        self.back = (old_state & INDEX_MASK) as usize;
    }
}

impl<T> std::fmt::Debug for TripleBufferProducer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TripleBufferProducer")
            .field("back", &self.back)
            .finish()
    }
}

/// Consumer endpoint of the wait-free triple buffer.
pub struct TripleBufferConsumer<T> {
    shared: Arc<SharedBuffer<T>>,
    front: usize,
}

impl<T> std::fmt::Debug for TripleBufferConsumer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TripleBufferConsumer")
            .field("front", &self.front)
            .finish()
    }
}

impl<T> TripleBufferConsumer<T> {
    /// Reads the latest available value from the buffer.
    ///
    /// If an update was published by the producer, the consumer atomically swaps its front slot
    /// with the shared slot and clears the dirty flag.
    ///
    /// This operation is wait-free, never blocks, and performs zero heap allocations.
    pub fn read(&mut self) -> &T {
        let mut current_state = self.shared.state.load(Ordering::Acquire);

        // Check if new data was published
        while (current_state & DIRTY_BIT) != 0 {
            // New state: put our old front into shared, clear dirty bit
            let new_state = self.front as u8 & INDEX_MASK;

            match self.shared.state.compare_exchange_weak(
                current_state,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Update our front to the newly acquired slot
                    self.front = (current_state & INDEX_MASK) as usize;
                    break;
                }
                Err(actual) => {
                    current_state = actual;
                }
            }
        }

        // SAFETY: The atomic state machine guarantees that `self.front` is exclusively held
        // by this consumer and cannot be overwritten by the producer until consumer swaps it away.
        unsafe { &*self.shared.slots[self.front].get() }
    }
}

/// Creates a new wait-free triple buffer initialized with default values.
///
/// Returns a tuple of `(producer, consumer)` endpoints.
#[allow(clippy::new_ret_no_self)]
#[must_use]
pub fn triple_buffer<T: Default + Clone>() -> (TripleBufferProducer<T>, TripleBufferConsumer<T>) {
    let initial_val = T::default();
    triple_buffer_with(initial_val)
}

/// Creates a new wait-free triple buffer initialized with copies of `initial_value`.
#[must_use]
pub fn triple_buffer_with<T: Clone>(
    initial_value: T,
) -> (TripleBufferProducer<T>, TripleBufferConsumer<T>) {
    // Initial assignment:
    // Slot 0: Producer back buffer
    // Slot 1: Shared exchange buffer
    // Slot 2: Consumer front buffer
    // Initial state: shared index = 1, dirty = 0
    let shared = Arc::new(SharedBuffer {
        slots: [
            UnsafeCell::new(initial_value.clone()),
            UnsafeCell::new(initial_value.clone()),
            UnsafeCell::new(initial_value),
        ],
        state: AtomicU8::new(1),
    });

    let producer = TripleBufferProducer {
        shared: shared.clone(),
        back: 0,
    };

    let consumer = TripleBufferConsumer { shared, front: 2 };

    (producer, consumer)
}
