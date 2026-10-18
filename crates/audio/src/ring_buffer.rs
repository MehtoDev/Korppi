//! Wait-free, lock-free Single-Producer Single-Consumer (SPSC) ring buffer
//! designed for hard real-time audio thread communication.
//!
//! One thread pushes through the [`Producer`] and one thread pops through the [`Consumer`]:
//! both take `&mut self`, so a handle cannot be shared between threads (finding AUD-02, where
//! two threads pushing through one producer lost and duplicated items):
//!
//! ```compile_fail
//! use korppi_audio::ring_buffer::SpscRingBuffer;
//! let (mut producer, _consumer) = SpscRingBuffer::<u32>::new(4);
//! let shared = &mut producer;
//! std::thread::scope(|s| {
//!     s.spawn(|| shared.push(1));
//!     s.spawn(|| shared.push(2)); // a second mutable borrow of the one producer
//! });
//! ```
//!
//! ```compile_fail
//! use korppi_audio::ring_buffer::SpscRingBuffer;
//! let (_producer, consumer) = SpscRingBuffer::<u32>::new(4);
//! let shared = &consumer;
//! std::thread::scope(|s| {
//!     s.spawn(|| shared.pop()); // `pop` needs `&mut Consumer`
//! });
//! ```
//!
//! Moving each handle to its own thread works:
//!
//! ```
//! use korppi_audio::ring_buffer::SpscRingBuffer;
//! let (mut producer, mut consumer) = SpscRingBuffer::<u32>::new(4);
//! std::thread::scope(|s| {
//!     s.spawn(move || producer.push(7).unwrap());
//! });
//! assert_eq!(consumer.pop(), Some(7));
//! ```

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Alignment wrapper guaranteeing that cache-sensitive atomic variables
/// sit on distinct 64-byte cache lines, eliminating false sharing between
/// producer and consumer threads.
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    const fn new(value: T) -> Self {
        Self { value }
    }
}

/// Internal shared state of the wait-free SPSC ring buffer.
struct RingBufferInner<T> {
    /// Monotonically increasing write cursor (written only by Producer).
    head: CachePadded<AtomicUsize>,
    /// Monotonically increasing read cursor (written only by Consumer).
    tail: CachePadded<AtomicUsize>,
    /// Contiguous pre-allocated buffer storage.
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Usable element capacity.
    capacity: usize,
}

// SAFETY: values of `T` move from the producer's thread to the consumer's, so `T: Send`
// suffices; the cursors are atomics.
unsafe impl<T: Send> Send for RingBufferInner<T> {}
// SAFETY: the one `Producer` and the one `Consumer` share the inner state. Only the producer
// writes `head` and the slots in `head..tail + capacity`, only the consumer writes `tail` and
// takes the slots in `tail..head`, and each half needs `&mut self` to do so, so one thread at a
// time acts for each half (AUD-02). Acquire/Release on the cursors orders the slot accesses.
unsafe impl<T: Send> Sync for RingBufferInner<T> {}

impl<T> Drop for RingBufferInner<T> {
    fn drop(&mut self) {
        let mut tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Relaxed);

        while tail < head {
            let slot_idx = tail % self.capacity;
            // SAFETY: Elements between `tail` and `head` are valid initialized instances of `T`
            // that must be dropped during ring buffer destruction.
            unsafe {
                let elem_ptr = (*self.buffer[slot_idx].get()).as_mut_ptr();
                std::ptr::drop_in_place(elem_ptr);
            }
            tail += 1;
        }
    }
}

/// Producer half of the SPSC ring buffer. Runs on gameplay/ECS threads.
///
/// `Send` (for `T: Send`) through its `Arc` of the shared state; it needs no `unsafe impl`.
pub struct Producer<T> {
    inner: Arc<RingBufferInner<T>>,
}

impl<T> Producer<T> {
    /// Pushes an item into the ring buffer.
    ///
    /// This operation is strictly wait-free and guarantees zero heap allocations.
    /// If the buffer is full, the unwritten item is returned in the `Err` variant.
    /// `&mut self`: one thread at a time pushes (AUD-02).
    pub fn push(&mut self, item: T) -> Result<(), T> {
        let head = self.inner.head.value.load(Ordering::Relaxed);
        let tail = self.inner.tail.value.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.inner.capacity {
            return Err(item);
        }

        let slot_idx = head % self.inner.capacity;

        // SAFETY: `head - tail < capacity`, so the consumer has already taken this slot's value
        // (or it was never written). `&mut self` makes this the only producer access, and only
        // the producer writes `head` or slots at or beyond it.
        unsafe {
            let elem_ptr = self.inner.buffer[slot_idx].get();
            (*elem_ptr).write(item);
        }

        // Release store publishes the newly written slot to the consumer.
        self.inner.head.value.store(head + 1, Ordering::Release);
        Ok(())
    }

    /// Returns the total capacity of the ring buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Returns the current number of unconsumed elements in the ring buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        let head = self.inner.head.value.load(Ordering::Relaxed);
        let tail = self.inner.tail.value.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the ring buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() >= self.inner.capacity
    }

    /// Returns `true` if the ring buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Consumer half of the SPSC ring buffer. Runs on the real-time audio thread.
pub struct Consumer<T> {
    inner: Arc<RingBufferInner<T>>,
}

impl<T> Consumer<T> {
    /// Pops an item from the ring buffer.
    ///
    /// This operation is strictly wait-free and guarantees zero heap allocations.
    /// Returns `None` if the queue is empty. `&mut self`: one thread at a time pops (AUD-02).
    pub fn pop(&mut self) -> Option<T> {
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        let head = self.inner.head.value.load(Ordering::Acquire);

        if tail == head {
            return None;
        }

        let slot_idx = tail % self.inner.capacity;

        // SAFETY: `tail < head`, so the producer initialized this slot and published it with
        // the Release store of `head` this thread acquired. `&mut self` makes this the only
        // consumer access, so the value is read out exactly once.
        let item = unsafe {
            let elem_ptr = self.inner.buffer[slot_idx].get();
            (*elem_ptr).assume_init_read()
        };

        // Release store informs the producer that the slot is now free.
        self.inner.tail.value.store(tail + 1, Ordering::Release);
        Some(item)
    }

    /// Returns the current number of unconsumed elements.
    #[must_use]
    pub fn len(&self) -> usize {
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        let head = self.inner.head.value.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the ring buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the total capacity of the ring buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }
}

/// Factory for creating lock-free Single-Producer Single-Consumer ring buffers.
pub struct SpscRingBuffer<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> SpscRingBuffer<T> {
    /// Constructs a new lock-free SPSC ring buffer with the specified capacity.
    ///
    /// Allocates contiguous uninitialized memory once during initialization;
    /// subsequent pushes and pops perform zero allocations.
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(capacity: usize) -> (Producer<T>, Consumer<T>) {
        assert!(
            capacity > 0,
            "Ring buffer capacity must be strictly greater than zero"
        );

        let mut storage = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            storage.push(UnsafeCell::new(MaybeUninit::uninit()));
        }

        let inner = Arc::new(RingBufferInner {
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            buffer: storage.into_boxed_slice(),
            capacity,
        });

        (
            Producer {
                inner: Arc::clone(&inner),
            },
            Consumer { inner },
        )
    }
}
