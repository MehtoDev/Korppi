//! Runtime-defined components (ADR-048): components a script or the editor defines by name and
//! layout, read and written as bytes.

use korppi_ecs::{Component, Entity, World};
use std::alloc::Layout;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Position(f32);
impl Component for Position {}

fn health_bytes(value: u32) -> [u8; 4] {
    value.to_ne_bytes()
}

#[test]
fn a_runtime_component_lives_on_entities_as_bytes() {
    let mut world = World::new();
    let health = world.register_dynamic_component("script.Health", Layout::new::<u32>());
    assert_eq!(
        world.register_dynamic_component("script.Health", Layout::new::<u32>()),
        health,
        "registering a name again returns its component"
    );
    assert_eq!(world.component_name(health), Some("script.Health"));

    let a = world.spawn((Position(1.0),));
    let b = world.spawn((Position(2.0),));
    world.insert_bytes(a, health, &health_bytes(100));
    world.insert_bytes(b, health, &health_bytes(50));
    assert_eq!(world.get_bytes(a, health), Some(&health_bytes(100)[..]));

    // Overwrite, mutate in place, and move to another archetype: the bytes follow the entity.
    world.insert_bytes(b, health, &health_bytes(40));
    world.get_bytes_mut(a, health).unwrap()[0] = 99;
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Marker;
    impl Component for Marker {}
    world.add_component(a, Marker);
    let mut expected_a = health_bytes(100);
    expected_a[0] = 99;
    assert_eq!(world.get_bytes(a, health), Some(&expected_a[..]));
    assert_eq!(world.get_bytes(b, health), Some(&health_bytes(40)[..]));
    assert_eq!(world.get::<Position>(a), Some(&Position(1.0)));

    // The inspector's view: which entities have it, and what an entity has.
    let mut holders: Vec<Entity> = world.entities_with(health).collect();
    holders.sort();
    assert_eq!(holders, vec![a, b]);
    assert!(world.entity_components(a).unwrap().contains(&health));

    // Typed access refuses a component without a Rust type.
    assert_eq!(world.get_by_id::<u32>(a, health), None);

    assert!(world.remove_by_id(b, health));
    assert!(!world.remove_by_id(b, health));
    assert_eq!(world.get_bytes(b, health), None);
    assert_eq!(world.entities_with(health).collect::<Vec<_>>(), vec![a]);
}

#[test]
#[should_panic(expected = "layout")]
fn a_name_cannot_be_registered_with_another_layout() {
    let mut world = World::new();
    world.register_dynamic_component("script.Speed", Layout::new::<f32>());
    world.register_dynamic_component("script.Speed", Layout::new::<f64>());
}

#[test]
#[should_panic(expected = "bytes")]
fn inserted_bytes_must_match_the_layout() {
    let mut world = World::new();
    let id = world.register_dynamic_component("script.Speed", Layout::new::<f32>());
    let e = world.spawn(());
    world.insert_bytes(e, id, &[1, 2, 3]);
}

#[test]
#[should_panic(expected = "runtime-defined")]
fn byte_access_refuses_typed_components() {
    let mut world = World::new();
    let e = world.spawn((Position(1.0),));
    let id = world.components().id_of::<Position>().unwrap();
    // A typed value's padding bytes may be uninitialized, so its bytes are not exposed.
    world.insert_bytes(e, id, &1.0f32.to_ne_bytes());
}

static DROPS: AtomicUsize = AtomicUsize::new(0);

/// Drop function of the runtime component in the next test.
///
/// # Safety
/// `ptr` points at a value of that component.
unsafe fn count_drop(_ptr: *mut u8) {
    DROPS.fetch_add(1, Ordering::Relaxed);
}

#[test]
fn a_runtime_component_with_a_drop_function_is_dropped_once_per_value() {
    let mut world = World::new();
    // SAFETY: `count_drop` accepts any value of the layout.
    let handle = unsafe {
        world.register_dynamic_component_with_drop(
            "script.Handle",
            Layout::new::<u64>(),
            count_drop,
        )
    };
    let a = world.spawn((Position(0.0),));
    let b = world.spawn((Position(0.0),));
    let c = world.spawn((Position(0.0),));
    for (e, v) in [(a, 1u64), (b, 2), (c, 3)] {
        // SAFETY: any `u64` is a valid value of the component.
        unsafe { world.insert_bytes_unchecked(e, handle, &v.to_ne_bytes()) };
    }
    // Writing bytes into a value with a drop function is not offered.
    assert!(world.get_bytes_mut(a, handle).is_none());

    // SAFETY: as above.
    unsafe { world.insert_bytes_unchecked(a, handle, &7u64.to_ne_bytes()) };
    assert_eq!(
        DROPS.load(Ordering::Relaxed),
        1,
        "overwrite drops the old value"
    );
    world.remove_by_id(b, handle);
    assert_eq!(DROPS.load(Ordering::Relaxed), 2);
    world.despawn(c);
    assert_eq!(DROPS.load(Ordering::Relaxed), 3);
    drop(world);
    assert_eq!(DROPS.load(Ordering::Relaxed), 4);
}
