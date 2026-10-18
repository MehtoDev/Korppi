//! Prefab templates and instantiation with property overrides.

use crate::registry::SceneRegistry;
use crate::scene::{Scene, SceneError};
use korppi_ecs::{Component, Entity, World};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Property overrides targeting a specific entity in a prefab instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefabOverride {
    /// Index of the target entity in the prefab template.
    pub entity_index: usize,
    /// Overridden component JSON payloads keyed by component type name.
    pub component_overrides: BTreeMap<String, serde_json::Value>,
}

impl PrefabOverride {
    /// Creates a new override targeting the entity at `entity_index`.
    #[must_use]
    pub fn new(entity_index: usize) -> Self {
        Self {
            entity_index,
            component_overrides: BTreeMap::new(),
        }
    }

    /// Adds or replaces a component override using a typed component value.
    pub fn with_component<T: Component + Serialize>(
        mut self,
        name: &str,
        component: &T,
    ) -> Result<Self, SceneError> {
        let value = serde_json::to_value(component).map_err(SceneError::from)?;
        self.component_overrides.insert(name.to_string(), value);
        Ok(self)
    }

    /// Adds or replaces a component override using a raw JSON value.
    #[must_use]
    pub fn with_json(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.component_overrides.insert(name.into(), value);
        self
    }
}

/// A reusable template definition representing a hierarchy of entities and components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prefab {
    /// Identifier or name of the prefab.
    pub name: String,
    /// The template scene hierarchy.
    pub scene: Scene,
}

impl Prefab {
    /// Creates a new prefab with the given name and template scene.
    #[must_use]
    pub fn new(name: impl Into<String>, scene: Scene) -> Self {
        Self {
            name: name.into(),
            scene,
        }
    }

    /// Captures a prefab template from a subset of entities in `world`.
    pub fn from_world(
        name: impl Into<String>,
        world: &World,
        entities: &[Entity],
        registry: &SceneRegistry,
    ) -> Result<Self, SceneError> {
        let scene = Scene::from_world(world, entities, registry)?;
        Ok(Self::new(name, scene))
    }

    /// Instantiates the prefab into `world` without overrides.
    pub fn instantiate(
        &self,
        world: &mut World,
        registry: &SceneRegistry,
    ) -> Result<Vec<Entity>, SceneError> {
        self.scene.spawn(world, registry)
    }

    /// Instantiates the prefab into `world` as a child of `parent`.
    pub fn instantiate_as_child(
        &self,
        world: &mut World,
        registry: &SceneRegistry,
        parent: Entity,
    ) -> Result<Vec<Entity>, SceneError> {
        self.scene.spawn_as_child(world, registry, parent)
    }

    /// Instantiates the prefab into `world` with the specified overrides applied.
    pub fn instantiate_with_overrides(
        &self,
        world: &mut World,
        registry: &SceneRegistry,
        overrides: &[PrefabOverride],
    ) -> Result<Vec<Entity>, SceneError> {
        let mut scene = self.scene.clone();

        for over in overrides {
            if over.entity_index >= scene.entities.len() {
                return Err(SceneError::UnregisteredComponent(format!(
                    "Invalid entity index {} for prefab with {} entities",
                    over.entity_index,
                    scene.entities.len()
                )));
            }
            let entity_entry = &mut scene.entities[over.entity_index];
            for (comp_name, comp_value) in &over.component_overrides {
                entity_entry
                    .components
                    .insert(comp_name.clone(), comp_value.clone());
            }
        }

        scene.spawn(world, registry)
    }

    /// Instantiates the prefab into `world` with overrides as a child of `parent`.
    pub fn instantiate_with_overrides_as_child(
        &self,
        world: &mut World,
        registry: &SceneRegistry,
        overrides: &[PrefabOverride],
        parent: Entity,
    ) -> Result<Vec<Entity>, SceneError> {
        let mut scene = self.scene.clone();

        for over in overrides {
            if over.entity_index >= scene.entities.len() {
                return Err(SceneError::UnregisteredComponent(format!(
                    "Invalid entity index {} for prefab with {} entities",
                    over.entity_index,
                    scene.entities.len()
                )));
            }
            let entity_entry = &mut scene.entities[over.entity_index];
            for (comp_name, comp_value) in &over.component_overrides {
                entity_entry
                    .components
                    .insert(comp_name.clone(), comp_value.clone());
            }
        }

        scene.spawn_as_child(world, registry, parent)
    }

    /// Serializes the prefab to JSON string.
    pub fn to_json(&self) -> Result<String, SceneError> {
        serde_json::to_string(self).map_err(SceneError::from)
    }

    /// Deserializes a prefab from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, SceneError> {
        serde_json::from_str(json).map_err(SceneError::from)
    }

    /// Serializes the prefab to compact CBOR binary format.
    pub fn to_binary(&self) -> Result<Vec<u8>, SceneError> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes)
            .map_err(|e| SceneError::BinarySerializationError(e.to_string()))?;
        Ok(bytes)
    }

    /// Deserializes a prefab from compact CBOR binary format.
    pub fn from_binary(bytes: &[u8]) -> Result<Self, SceneError> {
        ciborium::from_reader(bytes)
            .map_err(|e| SceneError::BinaryDeserializationError(e.to_string()))
    }
}

/// Fluent builder for configuring overrides on a prefab before spawning.
pub struct PrefabBuilder<'a> {
    prefab: &'a Prefab,
    overrides: HashMap<usize, BTreeMap<String, serde_json::Value>>,
}

impl<'a> PrefabBuilder<'a> {
    /// Creates a new builder for the given prefab.
    #[must_use]
    pub fn new(prefab: &'a Prefab) -> Self {
        Self {
            prefab,
            overrides: HashMap::new(),
        }
    }

    /// Overrides a component on the entity at `entity_index`.
    pub fn override_component<T: Component + Serialize>(
        mut self,
        entity_index: usize,
        name: &str,
        component: &T,
    ) -> Result<Self, SceneError> {
        let val = serde_json::to_value(component).map_err(SceneError::from)?;
        self.overrides
            .entry(entity_index)
            .or_default()
            .insert(name.to_string(), val);
        Ok(self)
    }

    /// Overrides a component with a raw JSON value.
    #[must_use]
    pub fn override_json(
        mut self,
        entity_index: usize,
        name: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.overrides
            .entry(entity_index)
            .or_default()
            .insert(name.into(), value);
        self
    }

    /// Spawns the prefab instance with all configured overrides.
    pub fn spawn(
        self,
        world: &mut World,
        registry: &SceneRegistry,
    ) -> Result<Vec<Entity>, SceneError> {
        let override_vec: Vec<PrefabOverride> = self
            .overrides
            .into_iter()
            .map(|(entity_index, component_overrides)| PrefabOverride {
                entity_index,
                component_overrides,
            })
            .collect();
        self.prefab
            .instantiate_with_overrides(world, registry, &override_vec)
    }

    /// Spawns the prefab instance with overrides as a child of `parent`.
    pub fn spawn_as_child(
        self,
        world: &mut World,
        registry: &SceneRegistry,
        parent: Entity,
    ) -> Result<Vec<Entity>, SceneError> {
        let override_vec: Vec<PrefabOverride> = self
            .overrides
            .into_iter()
            .map(|(entity_index, component_overrides)| PrefabOverride {
                entity_index,
                component_overrides,
            })
            .collect();
        self.prefab
            .instantiate_with_overrides_as_child(world, registry, &override_vec, parent)
    }
}
