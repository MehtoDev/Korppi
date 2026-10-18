//! Reuse of boxed per-frame objects (finding RHI-11).
//!
//! Recorders and command buffers are handed out as boxed trait objects. Allocating a box per
//! frame breaks the rule that frame loops do not allocate, so backends keep finished boxes in a
//! [`BoxPool`] and hand them out again. Pooled objects refer back to their pool through a
//! `Weak`, so pools and objects never keep each other alive.

use std::sync::{Arc, Mutex, Weak};

/// Boxes kept per pool; more returned boxes are dropped.
const MAX_POOLED: usize = 32;

/// A pool of boxed objects reused instead of reallocated.
#[derive(Debug)]
pub struct BoxPool<T> {
    boxes: Mutex<Vec<Box<T>>>,
}

impl<T> BoxPool<T> {
    /// An empty pool.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            boxes: Mutex::new(Vec::with_capacity(MAX_POOLED)),
        })
    }

    /// A pooled box, if one is available.
    pub fn take(&self) -> Option<Box<T>> {
        self.boxes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
    }

    /// Returns a box for reuse (dropped if the pool is full).
    pub fn give(&self, item: Box<T>) {
        let mut boxes = self
            .boxes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if boxes.len() < MAX_POOLED {
            boxes.push(item);
        }
    }

    /// Returns `item` to the pool behind `pool`, if it still exists.
    pub fn give_back(pool: &Weak<Self>, item: Box<T>) {
        if let Some(pool) = pool.upgrade() {
            pool.give(item);
        }
    }

    /// Number of pooled boxes.
    pub fn len(&self) -> usize {
        self.boxes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// True if no box is pooled.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_are_reused_and_weak_owners_do_not_keep_pools_alive() {
        let pool = BoxPool::<u64>::new();
        let first = Box::new(7);
        let address = &*first as *const u64;
        pool.give(first);
        let again = pool.take().expect("pooled");
        assert_eq!(
            &*again as *const u64, address,
            "the same allocation comes back"
        );
        let weak = Arc::downgrade(&pool);
        drop(pool);
        BoxPool::give_back(&weak, again); // dropped: the pool is gone
        assert!(weak.upgrade().is_none());
    }
}
