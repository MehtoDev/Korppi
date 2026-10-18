//! Integration tests for Prefab templates, overrides, and instantiation.

use korppi_ecs::prelude::*;
use korppi_math::prelude::*;
use korppi_scene::prefab::{Prefab, PrefabBuilder, PrefabOverride};
use korppi_scene::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Armor {
    pub rating: f32,
}

impl Component for Armor {}

#[test]
fn prefab_instantiation_spawns_hierarchy() {
    let mut world = World::new();
    let registry = SceneRegistry::default();

    let root = world.spawn((
        Name::new("Goblin"),
        LocalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0)),
    ));
    let weapon = world.spawn((
        Name::new("Dagger"),
        LocalTransform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
    ));
    attach_child(&mut world, root, weapon).unwrap();

    let prefab = Prefab::from_world("GoblinPrefab", &world, &[root, weapon], &registry).unwrap();

    let mut game_world = World::new();
    let spawned = prefab.instantiate(&mut game_world, &registry).unwrap();
    assert_eq!(spawned.len(), 2);

    let spawned_root = spawned[0];
    let spawned_weapon = spawned[1];

    assert_eq!(
        game_world.get::<Name>(spawned_root).unwrap().as_str(),
        "Goblin"
    );
    assert_eq!(
        game_world.get::<Name>(spawned_weapon).unwrap().as_str(),
        "Dagger"
    );

    // Verify hierarchy
    assert_eq!(game_world.children(spawned_root), [spawned_weapon]);
    assert_eq!(game_world.parent_of(spawned_weapon), Some(spawned_root));
}

#[test]
fn prefab_instantiation_with_component_overrides() {
    let mut world = World::new();
    let mut registry = SceneRegistry::default();
    registry.register::<Armor>("Armor");

    let root = world.spawn((
        Name::new("Soldier"),
        LocalTransform::from_translation(Vec3::ZERO),
        Armor { rating: 10.0 },
    ));

    let prefab = Prefab::from_world("SoldierPrefab", &world, &[root], &registry).unwrap();

    // Create an override changing position and armor
    let overrides = vec![PrefabOverride::new(0)
        .with_component(
            "LocalTransform",
            &LocalTransform::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        )
        .unwrap()
        .with_component("Armor", &Armor { rating: 50.0 })
        .unwrap()];

    let mut game_world = World::new();
    let spawned = prefab
        .instantiate_with_overrides(&mut game_world, &registry, &overrides)
        .unwrap();

    let soldier = spawned[0];
    assert_eq!(
        game_world
            .get::<LocalTransform>(soldier)
            .unwrap()
            .translation,
        Vec3::new(50.0, 0.0, 0.0)
    );
    assert_eq!(game_world.get::<Armor>(soldier).unwrap().rating, 50.0);
    assert_eq!(game_world.get::<Name>(soldier).unwrap().as_str(), "Soldier");
}

#[test]
fn prefab_builder_api_overrides_components() {
    let mut world = World::new();
    let registry = SceneRegistry::default();

    let root = world.spawn((
        Name::new("BaseTurret"),
        LocalTransform::from_translation(Vec3::ZERO),
    ));

    let prefab = Prefab::from_world("TurretPrefab", &world, &[root], &registry).unwrap();

    let mut game_world = World::new();
    let spawned = PrefabBuilder::new(&prefab)
        .override_component(0, "Name", &Name::new("EliteTurret"))
        .unwrap()
        .override_component(
            0,
            "LocalTransform",
            &LocalTransform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        )
        .unwrap()
        .spawn(&mut game_world, &registry)
        .unwrap();

    let turret = spawned[0];
    assert_eq!(
        game_world.get::<Name>(turret).unwrap().as_str(),
        "EliteTurret"
    );
    assert_eq!(
        game_world
            .get::<LocalTransform>(turret)
            .unwrap()
            .translation,
        Vec3::new(0.0, 10.0, 0.0)
    );
}

#[test]
fn prefab_instantiation_as_child_into_existing_hierarchy() {
    let mut world = World::new();
    let registry = SceneRegistry::default();

    let root = world.spawn((Name::new("SubUnit"),));
    let prefab = Prefab::from_world("SubUnitPrefab", &world, &[root], &registry).unwrap();

    let mut game_world = World::new();
    let boss = game_world.spawn((Name::new("Boss"),));

    let spawned = prefab
        .instantiate_as_child(&mut game_world, &registry, boss)
        .unwrap();

    let subunit = spawned[0];
    assert_eq!(game_world.parent_of(subunit), Some(boss));
    assert!(game_world.children(boss).contains(&subunit));
}

#[test]
fn prefab_serialization_roundtrip() {
    let mut world = World::new();
    let registry = SceneRegistry::default();

    let root = world.spawn((
        Name::new("TemplateNode"),
        LocalTransform::from_scale(Vec3::new(3.0, 3.0, 3.0)),
    ));

    let prefab = Prefab::from_world("Template", &world, &[root], &registry).unwrap();

    let json = prefab.to_json().unwrap();
    assert!(json.contains("TemplateNode"));

    let deser_json = Prefab::from_json(&json).unwrap();
    assert_eq!(prefab, deser_json);

    let binary = prefab.to_binary().unwrap();
    let deser_bin = Prefab::from_binary(&binary).unwrap();
    assert_eq!(prefab, deser_bin);
}
