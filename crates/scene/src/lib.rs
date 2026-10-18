//! Scene graph, entity hierarchy, transform propagation, and scene serialization.

pub mod hierarchy;
pub mod name;
pub mod prefab;
pub mod prelude;
pub mod registry;
pub mod scene;
pub mod transform;

pub use hierarchy::{
    attach_child, despawn_recursive, detach_child, get_ancestors, get_descendants, get_root,
    HierarchyError,
};
pub use name::Name;
pub use prefab::{Prefab, PrefabBuilder, PrefabOverride};
pub use registry::{EntityRemap, SceneRegistry};
pub use scene::{Scene, SceneEntity, SceneError};
pub use transform::{
    propagate_transforms, propagate_transforms_with_context, GlobalTransform, LocalTransform,
    TransformPropagationContext,
};
