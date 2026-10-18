//! Integration tests for Scene serialization, entity remapping, and custom components.

use korppi_ecs::prelude::*;
use korppi_math::prelude::*;
use korppi_scene::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Health {
    pub current: f32,
    pub max: f32,
}

impl Component for Health {}

#[test]
fn scene_roundtrip_json_serialization_and_deserialization() {
    let mut world = World::new();
    let registry = SceneRegistry::default();

    let root = world.spawn((
        Name::new("Player"),
        LocalTransform::from_translation(Vec3::new(10.0, 20.0, 30.0)),
    ));
    let weapon = world.spawn((
        Name::new("Sword"),
        LocalTransform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
    ));

    attach_child(&mut world, root, weapon).unwrap();

    let scene = Scene::from_world(&world, &[root, weapon], &registry).unwrap();
    let json = scene.to_json().unwrap();

    assert!(json.contains("Player"));
    assert!(json.contains("Sword"));
    assert!(json.contains("LocalTransform"));

    let deserialized_scene = Scene::from_json(&json).unwrap();
    assert_eq!(scene.entities.len(), deserialized_scene.entities.len());
}

#[test]
fn scene_roundtrip_binary_serialization_and_deserialization() {
    let mut world = World::new();
    let registry = SceneRegistry::default();

    let entity = world.spawn((
        Name::new("StaticRock"),
        LocalTransform::from_scale(Vec3::new(2.5, 2.5, 2.5)),
    ));

    let scene = Scene::from_world(&world, &[entity], &registry).unwrap();
    let binary = scene.to_binary().unwrap();

    assert!(!binary.is_empty());

    let deserialized_scene = Scene::from_binary(&binary).unwrap();
    assert_eq!(scene, deserialized_scene);
}

#[test]
fn scene_spawning_remaps_entity_ids_in_hierarchy() {
    let mut source_world = World::new();
    let registry = SceneRegistry::default();

    let orig_parent = source_world.spawn((
        Name::new("Vehicle"),
        LocalTransform::from_translation(Vec3::new(5.0, 0.0, 0.0)),
    ));
    let orig_wheel = source_world.spawn((
        Name::new("WheelFL"),
        LocalTransform::from_translation(Vec3::new(1.0, -0.5, 1.0)),
    ));

    attach_child(&mut source_world, orig_parent, orig_wheel).unwrap();

    let scene = Scene::from_world(&source_world, &[orig_parent, orig_wheel], &registry).unwrap();

    // Create target world with some dummy entities to shift indices
    let mut target_world = World::new();
    let _dummy1 = target_world.spawn((Name::new("Dummy1"),));
    let _dummy2 = target_world.spawn((Name::new("Dummy2"),));
    let _dummy3 = target_world.spawn((Name::new("Dummy3"),));

    let spawned_entities = scene.spawn(&mut target_world, &registry).unwrap();
    assert_eq!(spawned_entities.len(), 2);

    let spawned_parent = spawned_entities[0];
    let spawned_wheel = spawned_entities[1];

    assert_ne!(spawned_parent, orig_parent);
    assert_ne!(spawned_wheel, orig_wheel);

    // Verify Name & LocalTransform
    assert_eq!(
        target_world.get::<Name>(spawned_parent).unwrap().as_str(),
        "Vehicle"
    );
    assert_eq!(
        target_world.get::<Name>(spawned_wheel).unwrap().as_str(),
        "WheelFL"
    );
    assert_eq!(
        target_world
            .get::<LocalTransform>(spawned_parent)
            .unwrap()
            .translation,
        Vec3::new(5.0, 0.0, 0.0)
    );

    // Verify the hierarchy was rebuilt between the new entities
    assert_eq!(target_world.children(spawned_parent), [spawned_wheel]);
    assert_eq!(target_world.parent_of(spawned_wheel), Some(spawned_parent));
}

#[test]
fn scene_spawning_as_child_attaches_roots_to_parent() {
    let mut world = World::new();
    let registry = SceneRegistry::default();

    let scene_root = world.spawn((Name::new("SubSceneRoot"),));
    let scene_child = world.spawn((Name::new("SubSceneChild"),));
    attach_child(&mut world, scene_root, scene_child).unwrap();

    let scene = Scene::from_world(&world, &[scene_root, scene_child], &registry).unwrap();

    let mut dest_world = World::new();
    let dest_parent = dest_world.spawn((Name::new("WorldRoot"),));

    let spawned = scene
        .spawn_as_child(&mut dest_world, &registry, dest_parent)
        .unwrap();

    let spawned_root = spawned[0];
    let spawned_child = spawned[1];

    // spawned_root should now be a child of dest_parent
    assert_eq!(dest_world.parent_of(spawned_root), Some(dest_parent));
    assert!(dest_world.children(dest_parent).contains(&spawned_root));

    // internal relationship between spawned_root and spawned_child should remain
    assert_eq!(dest_world.parent_of(spawned_child), Some(spawned_root));
}

#[test]
fn custom_component_registration_and_serialization() {
    let mut registry = SceneRegistry::default();
    registry.register::<Health>("Health");

    let mut world = World::new();
    let hero = world.spawn((
        Name::new("Hero"),
        Health {
            current: 75.0,
            max: 100.0,
        },
    ));

    let scene = Scene::from_world(&world, &[hero], &registry).unwrap();
    let json = scene.to_json().unwrap();
    assert!(json.contains("\"current\":75.0"));

    let mut new_world = World::new();
    let spawned = scene.spawn(&mut new_world, &registry).unwrap();
    let new_hero = spawned[0];

    let health = new_world.get::<Health>(new_hero).unwrap();
    assert_eq!(
        *health,
        Health {
            current: 75.0,
            max: 100.0
        }
    );
}
