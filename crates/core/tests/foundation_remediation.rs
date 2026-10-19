//! Regression tests for Milestone FIX-13.1 (`COR-01` through `COR-06`).

use korppi_core::pod::{cast_slice, cast_slice_mut};
use korppi_core::task::{parallel_for, WorkerPool};
use korppi_core::{FixedClock, FrameArena, Handle, SlotMap};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Align64([u8; 64]);

impl Default for Align64 {
    fn default() -> Self {
        Self([0u8; 64])
    }
}

#[repr(C, align(128))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Align128([u8; 128]);

impl Default for Align128 {
    fn default() -> Self {
        Self([0u8; 128])
    }
}

/// COR-01 (Probe F1a): Concurrent mutable slices from `&FrameArena` must not alias or
/// write through a `SharedReadOnly` tag under Miri (Stacked Borrows and Tree Borrows).
#[test]
fn cor01_frame_arena_concurrent_mutable_slices_and_zst() {
    let mut arena = FrameArena::with_capacity(1024);

    let s1 = arena.alloc_slice::<u32>(4).expect("first slice alloc");
    s1[0] = 0xDEAD_BEEF;
    s1[3] = 11;

    // Allocating a second slice while `s1` is still live must not invalidate `s1`.
    let s2 = arena.alloc_slice::<u32>(4).expect("second slice alloc");
    s2[0] = 0xCAFE_BABE;
    s1[1] = 0x1234_5678;
    s2[1] = s1[0] ^ s1[1];

    assert_eq!(s1[0], 0xDEAD_BEEF);
    assert_eq!(s1[1], 0x1234_5678);
    assert_eq!(s1[3], 11);
    assert_eq!(s2[0], 0xCAFE_BABE);
    assert_eq!(s2[1], 0xDEAD_BEEF ^ 0x1234_5678);

    // Zero-sized types must succeed without advancing the arena offset.
    let before = arena.allocated_bytes();
    let zst = arena.alloc_slice::<()>(100).expect("ZST slice alloc");
    assert_eq!(zst.len(), 100);
    assert_eq!(arena.allocated_bytes(), before);

    arena.reset();
    assert_eq!(arena.allocated_bytes(), 0);
}

/// COR-01 (Probe F1b): `FrameArena::with_capacity(64)` must have 64-byte base alignment so a
/// single `Align64` allocation always succeeds without wasting capacity on base misalignment.
#[test]
fn cor01_frame_arena_exact_capacity_64_byte_alignment() {
    for _ in 0..32 {
        let arena = FrameArena::with_capacity(64);
        let slice = arena
            .alloc_slice::<Align64>(1)
            .expect("64-byte aligned allocation in 64-byte arena must succeed");
        assert_eq!((slice.as_ptr() as usize) % 64, 0);
        slice[0].0[0] = 42;
        assert_eq!(slice[0].0[0], 42);
    }

    let arena128 = FrameArena::with_capacity(256);
    let slice128 = arena128
        .alloc_slice::<Align128>(1)
        .expect("128-byte aligned allocation in 256-byte arena must succeed");
    assert_eq!((slice128.as_ptr() as usize) % 128, 0);
}

/// COR-02 (Probe F2): `SlotMap::remove` on a `Free` slot whose generation matches the passed
/// handle's generation must be a safe no-op (`None`) without underflowing `len` or corrupting `free_head`.
#[test]
fn cor02_slot_map_remove_on_free_slot_with_matching_generation() {
    // F2a: Empty map after removing slot 0 (which advances slot 0 to generation 2).
    let mut map = SlotMap::<u32>::new();
    let h1 = map.insert(100);
    assert_eq!(map.remove(h1), Some(100));
    assert_eq!(map.len(), 0);

    let forged_free_gen = Handle::<u32>::new(h1.index(), NonZeroU32::new(2).unwrap());
    assert_eq!(map.remove(forged_free_gen), None);
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());

    // F2b & F2c: Non-empty map must preserve `len` and free-list integrity across subsequent inserts.
    let mut map2 = SlotMap::<&'static str>::new();
    let a = map2.insert("a");
    let _b = map2.insert("b");
    assert_eq!(map2.remove(a), Some("a"));
    assert_eq!(map2.len(), 1);

    let forged_a_gen2 = Handle::new(a.index(), NonZeroU32::new(2).unwrap());
    assert_eq!(map2.remove(forged_a_gen2), None);
    assert_eq!(map2.len(), 1);

    let c = map2.insert("c");
    let d = map2.insert("d");
    assert_ne!(c.index(), d.index());
    assert_eq!(map2.len(), 3);
    assert_eq!(map2.get(c), Some(&"c"));
    assert_eq!(map2.get(d), Some(&"d"));
}

/// COR-03 (Probe F3a/F3b): `FixedClock::new` must reject `tick_rate > 1_000_000_000` so `timestep`
/// can never truncate to `Duration::ZERO`.
#[test]
#[should_panic(expected = "tick_rate")]
fn cor03_fixed_clock_rejects_sub_nanosecond_tick_rate() {
    let _ = FixedClock::new(2_000_000_000);
}

/// COR-03 (Probe F3c/F3d): `FixedClock::advance` saturates on huge durations/sub-steps and
/// `FixedClock::alpha()` stays strictly in `[0.0, 1.0)`.
#[test]
fn cor03_fixed_clock_saturates_and_alpha_is_strictly_less_than_one() {
    let mut clock = FixedClock::new(60);
    clock.set_max_sub_steps(u32::MAX);
    clock.advance(Duration::MAX);
    assert!(clock.alpha() >= 0.0 && clock.alpha() < 1.0);

    let mut clock2 = FixedClock::new(60);
    clock2.advance(clock2.timestep());
    assert!(clock2.should_tick());
    let alpha_at_step = clock2.alpha();
    assert!(
        (0.0..1.0).contains(&alpha_at_step),
        "alpha() must be in [0.0, 1.0), got {alpha_at_step}"
    );

    let mut clock3 = FixedClock::new(60);
    clock3.advance(clock3.timestep() - Duration::from_nanos(1));
    let alpha_near_step = clock3.alpha();
    assert!(
        (0.0..1.0).contains(&alpha_near_step),
        "alpha() near step must be in [0.0, 1.0), got {alpha_near_step}"
    );
}

/// COR-04 (Probe F4): `parallel_for` must not overflow `min_range * max_blocks` when `min_range` is large.
#[test]
fn cor04_parallel_for_large_min_range_does_not_overflow() {
    let pool = WorkerPool::new(4);
    let blocks = AtomicUsize::new(0);
    let items = AtomicUsize::new(0);

    // `(usize::MAX / 32) + 1` previously wrapped `min_range * max_blocks` (32) to 0 in release
    // or panicked with overflow in debug.
    parallel_for(&pool, 10, (usize::MAX / 32) + 1, &|start, end, _worker| {
        blocks.fetch_add(1, Ordering::Relaxed);
        items.fetch_add(end - start, Ordering::Relaxed);
    });
    assert_eq!(blocks.load(Ordering::Relaxed), 1);
    assert_eq!(items.load(Ordering::Relaxed), 10);

    blocks.store(0, Ordering::Relaxed);
    items.store(0, Ordering::Relaxed);
    parallel_for(&pool, 10, usize::MAX, &|start, end, _worker| {
        blocks.fetch_add(1, Ordering::Relaxed);
        items.fetch_add(end - start, Ordering::Relaxed);
    });
    assert_eq!(blocks.load(Ordering::Relaxed), 1);
    assert_eq!(items.load(Ordering::Relaxed), 10);
}

/// COR-04: `WorkerPool::new` must reject `worker_count > 0xFFFF` (`BUSY_MASK`).
#[test]
#[should_panic(expected = "worker_count")]
fn cor04_worker_pool_rejects_count_above_busy_mask() {
    let _ = WorkerPool::new(0x1_0000);
}

/// COR-05 (Probe F5c): `pod::cast_slice` must panic when casting a non-empty ZST slice to a non-ZST type.
#[test]
#[should_panic(expected = "zero-sized")]
fn cor05_pod_cast_slice_panics_on_non_empty_zst_to_non_zst() {
    let zst = [(), (), ()];
    let _: &[u32] = cast_slice(&zst);
}

/// COR-05 (Probe F5c): `pod::cast_slice_mut` must panic when casting a non-empty ZST slice to a non-ZST type.
#[test]
#[should_panic(expected = "zero-sized")]
fn cor05_pod_cast_slice_mut_panics_on_non_empty_zst_to_non_zst() {
    let mut zst = [(), ()];
    let _: &mut [u32] = cast_slice_mut(&mut zst);
}

/// COR-05 (Probe F5a/F5b): `Handle` comparisons against `u32` must check `generation == 1`
/// (never equating handles of different generations via a `u32`), and `Handle::from(-1i32)`
/// must panic rather than silently wrapping to `u32::MAX`.
#[test]
#[should_panic]
fn cor05_handle_from_negative_i32_panics() {
    let h_gen1 = Handle::<u32>::new(7, NonZeroU32::new(1).unwrap());
    let h_gen99 = Handle::<u32>::new(7, NonZeroU32::new(99).unwrap());
    assert_ne!(h_gen1, h_gen99);
    assert!(h_gen1 == 7u32);
    assert!(h_gen99 != 7u32);
    assert!(7u32 != h_gen99);

    let _ = Handle::<u32>::from(-1i32);
}

/// COR-06: `SlotMap` provides `contains`, `capacity`, `iter`, `iter_mut`, `values`, and `values_mut`.
#[test]
fn cor06_slot_map_contains_capacity_and_iterators() {
    let mut map = SlotMap::with_capacity(8);
    assert!(map.capacity() >= 8);

    let h1 = map.insert(10u32);
    let h2 = map.insert(20u32);
    assert!(map.contains(h1));
    assert!(map.contains(h2));

    map.remove(h1);
    assert!(!map.contains(h1));
    assert!(map.contains(h2));

    let h3 = map.insert(30u32);
    assert!(map.contains(h3));

    for (_handle, val) in map.iter_mut() {
        *val += 5;
    }

    let pairs: Vec<(Handle<u32>, u32)> = map.iter().map(|(h, &v)| (h, v)).collect();
    assert_eq!(pairs, vec![(h3, 35), (h2, 25)]);

    for val in map.values_mut() {
        *val *= 2;
    }
    let vals: Vec<u32> = map.values().copied().collect();
    assert_eq!(vals, vec![70, 50]);
}
