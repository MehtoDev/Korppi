//! `IdPool` hands out generational handles for objects kept in parallel arrays, and rejects a
//! handle once its object is freed, even after the slot is reused (Box2D `b2IdPool` plus the
//! generation of `b2BodyId`).

use korppi_core::{Handle, IdPool, IdPoolParts};

struct Body;

#[test]
fn freed_handles_are_rejected_after_their_slot_is_reused() {
    let mut pool: IdPool<Body> = IdPool::new();
    let a = pool.alloc();
    let b = pool.alloc();
    assert_eq!((a.index(), b.index()), (0, 1));
    assert_eq!(pool.len(), 2);
    assert_eq!(pool.slot_count(), 2);

    assert!(pool.free(a));
    assert!(!pool.free(a), "a handle is freed only once");
    assert!(!pool.contains(a));
    assert!(pool.contains(b));
    assert_eq!(pool.len(), 1);
    assert_eq!(pool.handle_at(0), None);

    // The freed slot is reused with a new generation.
    let c = pool.alloc();
    assert_eq!(c.index(), a.index());
    assert_ne!(c, a);
    assert!(!pool.contains(a), "the stale handle must stay invalid");
    assert!(!pool.free(a), "a stale handle must not free the new object");
    assert!(pool.contains(c));
    assert_eq!(pool.handle_at(0), Some(c));
    assert_eq!(pool.slot_count(), 2, "reuse must not grow the arrays");
}

#[test]
fn iteration_visits_live_handles_in_slot_order() {
    let mut pool: IdPool<Body> = IdPool::new();
    let handles: Vec<Handle<Body>> = (0..5).map(|_| pool.alloc()).collect();
    pool.free(handles[1]);
    pool.free(handles[3]);
    assert_eq!(
        pool.iter().collect::<Vec<_>>(),
        vec![handles[0], handles[2], handles[4]]
    );
    assert!(!pool.is_empty());

    // Freed slots are reused last-freed first, as Box2D's id pool does.
    assert_eq!(pool.alloc().index(), 3);
    assert_eq!(pool.alloc().index(), 1);
    assert_eq!(pool.alloc().index(), 5);
}

#[test]
fn a_handle_from_another_slot_count_is_rejected() {
    let mut pool: IdPool<Body> = IdPool::new();
    let mut other: IdPool<Body> = IdPool::new();
    for _ in 0..3 {
        other.alloc();
    }
    let foreign = other.alloc();
    assert!(
        !pool.contains(foreign),
        "a handle past the pool's slots is invalid"
    );
    assert!(!pool.free(foreign));
    pool.alloc();
    assert_eq!(pool.handle_at(7), None);
}

#[test]
fn a_pool_saved_as_parts_comes_back_the_same() {
    // A saved pool keeps its handles, its stale ones and its reuse order.
    let mut pool: IdPool<Body> = IdPool::new();
    let handles: Vec<_> = (0..4).map(|_| pool.alloc()).collect();
    pool.free(handles[2]);
    pool.free(handles[0]);
    let mut copy = IdPool::<Body>::from_parts(pool.to_parts()).expect("consistent parts");
    assert_eq!(copy.len(), 2);
    assert!(copy.contains(handles[1]) && copy.contains(handles[3]));
    assert!(!copy.contains(handles[0]) && !copy.contains(handles[2]));
    assert_eq!(copy.alloc(), pool.alloc());
    assert_eq!(copy.alloc(), pool.alloc());

    // Inconsistent parts are refused.
    let refused = |parts: IdPoolParts| IdPool::<Body>::from_parts(parts).is_none();
    let valid = IdPoolParts {
        generations: vec![1, 2],
        live: vec![true, false],
        free: vec![1],
    };
    assert!(IdPool::<Body>::from_parts(valid.clone()).is_some());
    assert!(refused(IdPoolParts {
        generations: vec![0, 2],
        ..valid.clone()
    }));
    assert!(refused(IdPoolParts {
        live: vec![true],
        ..valid.clone()
    }));
    assert!(refused(IdPoolParts {
        free: vec![0],
        ..valid.clone()
    }));
    assert!(refused(IdPoolParts {
        free: vec![1, 1],
        ..valid
    }));
}
