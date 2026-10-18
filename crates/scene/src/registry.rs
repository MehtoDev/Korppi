//! Component serialization registry and entity reference remapping.

use crate::name::Name;
use crate::scene::SceneError;
use crate::transform::{GlobalTransform, LocalTransform};
use korppi_ecs::{Component, Entity, World};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Trait for components that contain internal [`Entity`] references that require
/// remapping when a scene or prefab is instantiated into a new [`World`].
pub trait EntityRemap {
    /// Remaps all internal entity references according to the provided translation table.
    fn remap_entities(&mut self, map: &HashMap<Entity, Entity>);
}

type SerializeFn = Arc<dyn Fn(&World, Entity) -> Option<serde_json::Value> + Send + Sync>;
type SpawnFn =
    Arc<dyn Fn(&mut World, Entity, &serde_json::Value) -> Result<(), SceneError> + Send + Sync>;
type RemapFn = Arc<dyn Fn(&mut World, Entity, &HashMap<Entity, Entity>) + Send + Sync>;

#[derive(Clone)]
struct ComponentEntry {
    serialize: SerializeFn,
    spawn: SpawnFn,
    remap: Option<RemapFn>,
}

/// Registry storing serialization, deserialization, and entity remapping callbacks for components.
#[derive(Clone)]
pub struct SceneRegistry {
    entries: HashMap<String, ComponentEntry>,
}

impl Default for SceneRegistry {
    fn default() -> Self {
        let mut registry = Self {
            entries: HashMap::new(),
        };

        registry.register::<Name>("Name");
        registry.register::<LocalTransform>("LocalTransform");
        registry.register::<GlobalTransform>("GlobalTransform");

        registry
    }
}

impl SceneRegistry {
    /// Creates an empty registry without pre-registered components.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Registers a standard component type for serialization and deserialization.
    pub fn register<T>(&mut self, name: &'static str)
    where
        T: Component + Serialize + DeserializeOwned + 'static,
    {
        self.register_custom::<T>(name, None);
    }

    /// Registers a component type that contains entity references requiring remapping.
    pub fn register_remappable<T>(&mut self, name: &'static str)
    where
        T: Component + Serialize + DeserializeOwned + EntityRemap + 'static,
    {
        let remap: RemapFn = Arc::new(|world, entity, map| {
            if let Some(comp) = world.get_mut::<T>(entity) {
                comp.remap_entities(map);
            }
        });

        self.register_custom::<T>(name, Some(remap));
    }

    /// Registers a component type with an optional custom remapping function.
    pub fn register_custom<T>(&mut self, name: &'static str, remap: Option<RemapFn>)
    where
        T: Component + Serialize + DeserializeOwned + 'static,
    {
        let serialize: SerializeFn = Arc::new(|world, entity| {
            world
                .get::<T>(entity)
                .and_then(|comp| serde_json::to_value(comp).ok())
        });

        let spawn_name = name.to_string();
        let spawn: SpawnFn = Arc::new(move |world, entity, value| {
            let comp: T = serde_json::from_value(value.clone()).map_err(|err| {
                SceneError::ComponentDeserializationFailed {
                    component: spawn_name.clone(),
                    reason: err.to_string(),
                }
            })?;
            world.add_component(entity, comp);
            Ok(())
        });

        self.entries.insert(
            name.to_string(),
            ComponentEntry {
                serialize,
                spawn,
                remap,
            },
        );
    }

    /// Serializes all registered components present on `entity` in `world`.
    #[must_use]
    pub fn serialize_entity(
        &self,
        world: &World,
        entity: Entity,
    ) -> BTreeMap<String, serde_json::Value> {
        let mut map = BTreeMap::new();
        for (name, entry) in &self.entries {
            if let Some(value) = (entry.serialize)(world, entity) {
                map.insert(name.clone(), value);
            }
        }
        map
    }

    /// Deserializes and attaches a named component value to `entity` in `world`.
    pub fn spawn_component(
        &self,
        world: &mut World,
        entity: Entity,
        name: &str,
        value: &serde_json::Value,
    ) -> Result<(), SceneError> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| SceneError::UnregisteredComponent(name.to_string()))?;
        (entry.spawn)(world, entity, value)
    }

    /// Remaps entity references on all components attached to `entity`.
    pub fn remap_components(
        &self,
        world: &mut World,
        entity: Entity,
        map: &HashMap<Entity, Entity>,
    ) {
        for entry in self.entries.values() {
            if let Some(remap_fn) = &entry.remap {
                remap_fn(world, entity, map);
            }
        }
    }
}
