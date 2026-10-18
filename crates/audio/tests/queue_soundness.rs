//! The queues between the game and audio threads (FIX-7.1, finding AUD-02, probe A6).
//!
//! Before FIX-7.1, `Producer::push` and `Consumer::pop` took `&self` on handles that were `Sync`,
//! so two threads could push through one producer: a data race that lost items (A6 delivered
//! 35,366 of 60,000) and, with items owning heap memory, leaked or dropped slots twice. The
//! handles now need `&mut self`, so sharing one between threads does not compile (the
//! `compile_fail` examples in `korppi_audio::ring_buffer`). These tests check the outcome the
//! queues exist for: every item arrives once, in order, and is dropped exactly once. They also
//! run under Miri (`cargo +nightly miri test -p korppi_audio --test queue_soundness`).

use korppi_audio::ring_buffer::SpscRingBuffer;
use korppi_audio::{triple_buffer_with, TripleBufferConsumer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Items per test: fewer under Miri, which is slow.
const ITEMS: usize = if cfg!(miri) { 300 } else { 50_000 };

/// A value owning heap memory that counts its drops.
struct Tracked {
    value: Box<usize>,
    drops: Arc<AtomicUsize>,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn items_cross_threads_once_in_order_and_are_dropped_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (mut producer, mut consumer) = SpscRingBuffer::<Tracked>::new(16);
    let producer_drops = Arc::clone(&drops);
    std::thread::scope(|s| {
        s.spawn(move || {
            for i in 0..ITEMS {
                let mut item = Tracked {
                    value: Box::new(i),
                    drops: Arc::clone(&producer_drops),
                };
                loop {
                    match producer.push(item) {
                        Ok(()) => break,
                        Err(back) => {
                            item = back;
                            std::thread::yield_now();
                        }
                    }
                }
            }
        });
        let mut next = 0;
        while next < ITEMS {
            match consumer.pop() {
                Some(item) => {
                    assert_eq!(*item.value, next, "items arrive once and in order");
                    next += 1;
                }
                None => std::thread::yield_now(),
            }
        }
        assert!(consumer.pop().is_none(), "nothing arrives twice");
    });
    assert_eq!(
        drops.load(Ordering::Relaxed),
        ITEMS,
        "every item is dropped exactly once"
    );
}

#[test]
fn items_left_in_the_queue_are_dropped_with_it() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (mut producer, mut consumer) = SpscRingBuffer::<Tracked>::new(4);
    let item = |value| Tracked {
        value: Box::new(value),
        drops: Arc::clone(&drops),
    };
    for value in 0..4 {
        assert!(producer.push(item(value)).is_ok());
    }
    // Refused when full: the item comes back and is the caller's to drop.
    let refused = producer.push(item(4)).expect_err("the queue is full");
    drop(refused);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    // One popped; three still queued when both halves go away.
    drop(consumer.pop());
    drop(producer);
    drop(consumer);
    assert_eq!(
        drops.load(Ordering::Relaxed),
        5,
        "queued items are dropped exactly once"
    );
}

/// The triple buffer hands the latest published value across threads: a reader only ever sees
/// values the writer wrote, never going back, and every value is dropped exactly once.
#[test]
fn triple_buffer_reads_only_published_values_and_drops_each_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let tracked = |value| Tracked {
        value: Box::new(value),
        drops: Arc::clone(&drops),
    };
    let (mut producer, mut consumer): (_, TripleBufferConsumer<Tracked>) =
        triple_buffer_with(tracked(0));
    // `triple_buffer_with` clones the initial value into all three slots.
    let writes = ITEMS / 10;
    std::thread::scope(|s| {
        let drops = Arc::clone(&drops);
        s.spawn(move || {
            for i in 1..=writes {
                producer.write(Tracked {
                    value: Box::new(i),
                    drops: Arc::clone(&drops),
                });
            }
        });
        let mut last = 0;
        while last < writes {
            let seen = *consumer.read().value;
            assert!(seen >= last, "read {seen} after {last}");
            last = seen;
            std::thread::yield_now();
        }
    });
    drop(consumer);
    // Three initial values (the original was moved into a slot after two clones) and every write
    // were dropped: overwritten in a slot or with the buffer.
    assert_eq!(drops.load(Ordering::Relaxed), 3 + writes);
}

impl Clone for Tracked {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            drops: Arc::clone(&self.drops),
        }
    }
}
