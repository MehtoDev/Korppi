//! Convenience prelude re-exporting common scene types and functions.

pub use crate::hierarchy::{
    attach_child, despawn_recursive, detach_child, get_ancestors, get_descendants, get_root,
    HierarchyError,
};
pub use crate::name::Name;
pub use crate::prefab::{Prefab, PrefabBuilder, PrefabOverride};
pub use crate::registry::{EntityRemap, SceneRegistry};
pub use crate::scene::{Scene, SceneEntity, SceneError};
pub use crate::transform::{
    propagate_transforms, propagate_transforms_with_context, GlobalTransform, LocalTransform,
    TransformPropagationContext,
};
