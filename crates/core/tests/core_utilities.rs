//! Integration tests for korppi_core utilities.

use korppi_core::{FixedClock, FrameArena, Handle, SlotMap};
use std::time::Duration;

#[test]
fn fixed_clock_accumulates_and_yields_exact_ticks() {
    let tick_rate = 60; // 60 Hz
    let mut clock = FixedClock::new(tick_rate);
    assert_eq!(clock.tick_rate(), 60);
    assert_eq!(clock.timestep(), Duration::from_nanos(1_000_000_000 / 60));

    // Initially, no ticks should be queued
    assert!(!clock.should_tick());
    assert_eq!(clock.alpha(), 0.0);

    // Step by half a timestep: no tick yet, alpha is ~0.5
    let half_step = clock.timestep() / 2;
    clock.advance(half_step);
    assert!(!clock.should_tick());
    assert!((clock.alpha() - 0.5).abs() < 1e-3);

    // Step by another half timestep + 10%: exactly 1 tick queued, alpha is ~0.1
    let remainder = (clock.timestep() / 2) + (clock.timestep() / 10);
    clock.advance(remainder);
    assert!(clock.should_tick());
    clock.tick();

    assert!(!clock.should_tick());
    assert!((clock.alpha() - 0.1).abs() < 1e-3);
    assert_eq!(clock.current_tick(), 1);
}

#[test]
fn fixed_clock_clamps_accumulator_to_prevent_spiral_of_death() {
    let mut clock = FixedClock::new(60);
    clock.set_max_sub_steps(4);

    // Advance by 1 second (which would be 60 ticks without clamping)
    clock.advance(Duration::from_secs(1));

    let mut ticks = 0;
    while clock.should_tick() {
        clock.tick();
        ticks += 1;
    }

    // Should be clamped to max_sub_steps (4)
    assert_eq!(ticks, 4);
    assert_eq!(clock.current_tick(), 4);
}

#[test]
fn frame_arena_allocates_and_resets() {
    let mut arena = FrameArena::with_capacity(1024);
    assert_eq!(arena.capacity(), 1024);
    assert_eq!(arena.allocated_bytes(), 0);

    let slice1 = arena.alloc_slice::<u32>(10).expect("allocation failed");
    assert_eq!(slice1.len(), 10);
    for (i, item) in slice1.iter_mut().enumerate() {
        *item = i as u32 * 10;
    }
    assert_eq!(slice1[5], 50);

    let slice2 = arena.alloc_slice::<f32>(5).expect("allocation failed");
    assert_eq!(slice2.len(), 5);
    assert!(arena.allocated_bytes() >= (10 * 4 + 5 * 4));

    // Reset clears allocated bytes without dropping memory capacity
    arena.reset();
    assert_eq!(arena.allocated_bytes(), 0);
    assert_eq!(arena.capacity(), 1024);

    // Can allocate again up to full capacity
    let slice3 = arena.alloc_slice::<u8>(1024).expect("allocation failed");
    assert_eq!(slice3.len(), 1024);
}

#[test]
fn generational_slot_map_handles_insert_get_remove_and_reuse() {
    let mut map: SlotMap<String> = SlotMap::new();

    let h1: Handle<String> = map.insert("Entity_Alpha".to_string());
    let h2: Handle<String> = map.insert("Entity_Beta".to_string());

    assert_eq!(map.len(), 2);
    assert!(!map.is_empty());

    assert_eq!(map.get(h1).map(|s| s.as_str()), Some("Entity_Alpha"));
    assert_eq!(map.get(h2).map(|s| s.as_str()), Some("Entity_Beta"));

    // Mutation
    if let Some(val) = map.get_mut(h1) {
        val.push_str("_Modified");
    }
    assert_eq!(
        map.get(h1).map(|s| s.as_str()),
        Some("Entity_Alpha_Modified")
    );

    // Remove h1
    let removed = map.remove(h1);
    assert_eq!(removed, Some("Entity_Alpha_Modified".to_string()));
    assert_eq!(map.len(), 1);

    // Stale handle should now return None
    assert_eq!(map.get(h1), None);
    assert_eq!(map.remove(h1), None);

    // Insert new item: it should reuse index of h1 but with a newer generation
    let h3 = map.insert("Entity_Gamma".to_string());
    assert_eq!(map.len(), 2);
    assert_eq!(h3.index(), h1.index());
    assert_ne!(h3.generation(), h1.generation());

    // Old handle h1 remains invalid, new handle h3 works
    assert_eq!(map.get(h1), None);
    assert_eq!(map.get(h3).map(|s| s.as_str()), Some("Entity_Gamma"));
}
