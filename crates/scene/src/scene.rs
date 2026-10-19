//! Scene data structure, JSON and binary serialization, and entity spawning/remapping.

use crate::hierarchy::{attach_child, HierarchyError};
use crate::registry::SceneRegistry;
use korppi_ecs::{Entity, IsA, World};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// Error conditions encountered during scene operations.
#[derive(Debug)]
pub enum SceneError {
    /// Attempted to serialize or deserialize an unregistered component name.
    UnregisteredComponent(String),

    /// Deserialization of a component payload failed.
    ComponentDeserializationFailed {
        /// Name of the component that failed.
        component: String,
        /// Underlying error description.
        reason: String,
    },

    /// An expected entity was not found in the source world.
    EntityNotFound(Entity),

    /// Duplicate entity handle encountered in a scene entity list.
    DuplicateEntity(Entity),

    /// Out-of-bounds entity index in a prefab override.
    InvalidEntityIndex {
        /// The requested entity index.
        index: usize,
        /// Total number of entities in the prefab template.
        len: usize,
    },

    /// A hierarchy tree error occurred during parenting or detachment.
    HierarchyError(HierarchyError),

    /// Serialization or deserialization of JSON failed.
    JsonError(serde_json::Error),

    /// Serialization of binary payload failed.
    BinarySerializationError(String),

    /// Deserialization of binary payload failed.
    BinaryDeserializationError(String),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnregisteredComponent(name) => write!(f, "Unregistered component: {name}"),
            Self::ComponentDeserializationFailed { component, reason } => {
                write!(f, "Failed to deserialize component '{component}': {reason}")
            }
            Self::EntityNotFound(entity) => write!(f, "Entity not found: {entity:?}"),
            Self::DuplicateEntity(entity) => {
                write!(f, "Duplicate entity in scene: {entity:?}")
            }
            Self::InvalidEntityIndex { index, len } => {
                write!(
                    f,
                    "Invalid entity index {index} for prefab with {len} entities"
                )
            }
            Self::HierarchyError(err) => write!(f, "Hierarchy error: {err}"),
            Self::JsonError(err) => write!(f, "JSON serialization error: {err}"),
            Self::BinarySerializationError(msg) => write!(f, "Binary serialization error: {msg}"),
            Self::BinaryDeserializationError(msg) => {
                write!(f, "Binary deserialization error: {msg}")
            }
        }
    }
}

impl std::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::HierarchyError(err) => Some(err),
            Self::JsonError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<HierarchyError> for SceneError {
    fn from(err: HierarchyError) -> Self {
        Self::HierarchyError(err)
    }
}

impl From<serde_json::Error> for SceneError {
    fn from(err: serde_json::Error) -> Self {
        Self::JsonError(err)
    }
}

/// A serialized representation of a single entity, its registered components and its children.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneEntity {
    /// The original entity handle when the scene was captured.
    pub entity: Entity,
    /// Key-value mapping of component type names to serialized JSON values.
    pub components: BTreeMap<String, serde_json::Value>,
    /// The entity's `ChildOf` children when the scene was captured, in attach order. Spawning
    /// links those that are part of the scene.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Entity>,
    /// The entity's `IsA` prefab inheritance targets when the scene was captured.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub isa_targets: Vec<Entity>,
}

/// A self-contained, serializable snapshot of entities, components, and hierarchies.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Scene {
    /// All entities captured in this scene.
    pub entities: Vec<SceneEntity>,
}

impl Scene {
    /// Creates a new, empty `Scene`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    /// Captures the specified entities from `world` into a `Scene`.
    pub fn from_world(
        world: &World,
        entities: &[Entity],
        registry: &SceneRegistry,
    ) -> Result<Self, SceneError> {
        let mut scene_entities = Vec::with_capacity(entities.len());
        let mut seen = HashSet::with_capacity(entities.len());

        for &entity in entities {
            if !world.is_alive(entity) {
                return Err(SceneError::EntityNotFound(entity));
            }
            if !seen.insert(entity) {
                return Err(SceneError::DuplicateEntity(entity));
            }
            let components = registry.serialize_entity(world, entity);
            let children = world.children(entity).to_vec();
            let isa_targets = world.targets_of::<IsA>(entity).to_vec();
            scene_entities.push(SceneEntity {
                entity,
                components,
                children,
                isa_targets,
            });
        }

        Ok(Self {
            entities: scene_entities,
        })
    }

    /// Captures all alive entities from `world` into a `Scene`.
    pub fn from_entire_world(world: &World, registry: &SceneRegistry) -> Result<Self, SceneError> {
        let all_entities: Vec<Entity> = world.entities().collect();
        Self::from_world(world, &all_entities, registry)
    }

    /// Serializes the scene into a compact JSON string.
    pub fn to_json(&self) -> Result<String, SceneError> {
        serde_json::to_string(self).map_err(SceneError::from)
    }

    /// Serializes the scene into a human-readable, pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, SceneError> {
        serde_json::to_string_pretty(self).map_err(SceneError::from)
    }

    /// Deserializes a scene from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, SceneError> {
        serde_json::from_str(json).map_err(SceneError::from)
    }

    /// Serializes the scene into a compact binary format using CBOR.
    pub fn to_binary(&self) -> Result<Vec<u8>, SceneError> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes)
            .map_err(|e| SceneError::BinarySerializationError(e.to_string()))?;
        Ok(bytes)
    }

    /// Deserializes a scene from compact binary CBOR data.
    pub fn from_binary(bytes: &[u8]) -> Result<Self, SceneError> {
        ciborium::from_reader(bytes)
            .map_err(|e| SceneError::BinaryDeserializationError(e.to_string()))
    }

    /// Instantiates all entities and components from this scene into `world`.
    ///
    /// Returns the newly allocated [`Entity`] handles in the same order as `self.entities`.
    /// The hierarchy between scene entities is rebuilt with `ChildOf`, in the captured child
    /// order, `IsA` inheritance links are restored, and component entity references are remapped.
    /// If any step fails, all newly allocated entities are despawned so `world` remains clean.
    pub fn spawn(
        &self,
        world: &mut World,
        registry: &SceneRegistry,
    ) -> Result<Vec<Entity>, SceneError> {
        // Pre-validate duplicate entity IDs and unregistered component names before mutating `world`.
        let mut seen = HashSet::with_capacity(self.entities.len());
        for scene_entity in &self.entities {
            if !seen.insert(scene_entity.entity) {
                return Err(SceneError::DuplicateEntity(scene_entity.entity));
            }
            for comp_name in scene_entity.components.keys() {
                if !registry.is_registered(comp_name) {
                    return Err(SceneError::UnregisteredComponent(comp_name.clone()));
                }
            }
        }

        let mut entity_map: HashMap<Entity, Entity> = HashMap::with_capacity(self.entities.len());
        let mut spawned: Vec<Entity> = Vec::with_capacity(self.entities.len());

        // Step 1: Allocate new empty entities in target world
        for scene_entity in &self.entities {
            let new_entity = world.spawn_empty();
            entity_map.insert(scene_entity.entity, new_entity);
            spawned.push(new_entity);
        }

        let populate = |world: &mut World| -> Result<(), SceneError> {
            // Step 2: Spawn each component onto allocated entities
            for (i, scene_entity) in self.entities.iter().enumerate() {
                let new_entity = spawned[i];
                for (comp_name, comp_value) in &scene_entity.components {
                    registry.spawn_component(world, new_entity, comp_name, comp_value)?;
                }
            }

            // Step 3: Remap internal entity references held by components
            for &new_entity in &spawned {
                registry.remap_components(world, new_entity, &entity_map);
            }

            // Step 4: Rebuild the hierarchy between scene entities
            for (i, scene_entity) in self.entities.iter().enumerate() {
                for child in &scene_entity.children {
                    if let Some(&new_child) = entity_map.get(child) {
                        attach_child(world, spawned[i], new_child)?;
                    }
                }
            }

            // Step 5: Rebuild IsA prefab inheritance relationships between scene entities
            for (i, scene_entity) in self.entities.iter().enumerate() {
                for base in &scene_entity.isa_targets {
                    if let Some(&new_base) = entity_map.get(base) {
                        if new_base == spawned[i] {
                            return Err(SceneError::HierarchyError(HierarchyError::SelfParenting(
                                spawned[i],
                            )));
                        }
                        world.add_pair::<IsA>(spawned[i], new_base);
                    }
                }
            }

            Ok(())
        };

        if let Err(err) = populate(world) {
            for entity in spawned {
                world.despawn(entity);
            }
            return Err(err);
        }

        Ok(spawned)
    }

    /// Instantiates this scene into `world` and parents any root entities in the scene
    /// to the specified `parent` entity.
    pub fn spawn_as_child(
        &self,
        world: &mut World,
        registry: &SceneRegistry,
        parent: Entity,
    ) -> Result<Vec<Entity>, SceneError> {
        if !world.is_alive(parent) {
            return Err(SceneError::EntityNotFound(parent));
        }

        let scene_entities_set: HashSet<Entity> =
            self.entities.iter().map(|se| se.entity).collect();

        let spawned = self.spawn(world, registry)?;

        // The scene's roots: entities no scene entity lists as a child.
        let children: HashSet<Entity> = self
            .entities
            .iter()
            .flat_map(|se| se.children.iter().copied())
            .filter(|child| scene_entities_set.contains(child))
            .collect();
        for (i, scene_entity) in self.entities.iter().enumerate() {
            if !children.contains(&scene_entity.entity) {
                if let Err(err) = attach_child(world, parent, spawned[i]) {
                    for entity in spawned {
                        world.despawn(entity);
                    }
                    return Err(SceneError::from(err));
                }
            }
        }

        Ok(spawned)
    }
}
