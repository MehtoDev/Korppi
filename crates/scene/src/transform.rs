//! Local and Global transform components and hierarchical propagation system.

use korppi_ecs::{Component, Entity, World};
use korppi_math::{Mat3, Mat4, Quat, Transform, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Local 3D affine transform component relative to parent entity (or world space if root).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalTransform {
    /// 3D translation offset.
    pub translation: Vec3,
    /// 3D rotation orientation quaternion.
    pub rotation: Quat,
    /// 3D non-uniform scale.
    pub scale: Vec3,
}

impl Component for LocalTransform {}

impl Default for LocalTransform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl LocalTransform {
    /// Creates a local transform from an affine `Transform`.
    #[must_use]
    pub const fn from_transform(t: Transform) -> Self {
        Self {
            translation: t.translation,
            rotation: t.rotation,
            scale: t.scale,
        }
    }

    /// Creates a local transform with a translation offset.
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Creates a local transform with an orientation quaternion.
    #[must_use]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation,
            scale: Vec3::ONE,
        }
    }

    /// Creates a local transform with a scale vector.
    #[must_use]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale,
        }
    }

    /// Computes the 4x4 affine transform matrix `T * R * S`.
    #[must_use]
    pub fn compute_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Converts to the underlying `Transform` primitive.
    #[must_use]
    pub const fn as_transform(&self) -> Transform {
        Transform {
            translation: self.translation,
            rotation: self.rotation,
            scale: self.scale,
        }
    }
}

/// Global world-space transform component computed by multiplying ancestor local transforms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GlobalTransform {
    /// 4x4 world affine transform matrix.
    pub matrix: Mat4,
}

impl Component for GlobalTransform {}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
        }
    }
}

impl GlobalTransform {
    /// Creates a global transform from a 4x4 matrix.
    #[must_use]
    pub const fn from_matrix(matrix: Mat4) -> Self {
        Self { matrix }
    }

    /// Creates a global transform from an affine `Transform`.
    #[must_use]
    pub fn from_transform(t: Transform) -> Self {
        Self {
            matrix: t.to_matrix(),
        }
    }

    /// Creates a global transform with a translation offset.
    #[must_use]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            matrix: Mat4::from_translation(translation),
        }
    }

    /// Extracts world translation from the matrix column 3.
    #[must_use]
    pub fn translation(&self) -> Vec3 {
        let col3 = self.matrix.col(3);
        Vec3::new(col3.x, col3.y, col3.z)
    }

    /// Extracts world non-uniform scale by computing matrix column lengths.
    #[must_use]
    pub fn scale(&self) -> Vec3 {
        let col0 = self.matrix.col(0);
        let col1 = self.matrix.col(1);
        let col2 = self.matrix.col(2);
        Vec3::new(
            Vec3::new(col0.x, col0.y, col0.z).length(),
            Vec3::new(col1.x, col1.y, col1.z).length(),
            Vec3::new(col2.x, col2.y, col2.z).length(),
        )
    }

    /// Extracts world rotation quaternion.
    #[must_use]
    pub fn rotation(&self) -> Quat {
        let scale = self.scale();
        if scale.x.abs() < 1e-6 || scale.y.abs() < 1e-6 || scale.z.abs() < 1e-6 {
            return Quat::IDENTITY;
        }
        let inv_scale = Vec3::new(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
        let col0 = self.matrix.col(0) * inv_scale.x;
        let col1 = self.matrix.col(1) * inv_scale.y;
        let col2 = self.matrix.col(2) * inv_scale.z;
        let rot_mat = Mat3::from_cols(
            Vec3::new(col0.x, col0.y, col0.z),
            Vec3::new(col1.x, col1.y, col1.z),
            Vec3::new(col2.x, col2.y, col2.z),
        );
        Quat::from_mat3(&rot_mat)
    }
}

/// Reusable scratch buffers for local-to-world transform propagation,
/// eliminating dynamic heap allocations on frame loops.
#[derive(Debug, Default)]
pub struct TransformPropagationContext {
    roots: Vec<(Entity, Mat4)>,
    queue: VecDeque<(Entity, Mat4)>,
}

impl TransformPropagationContext {
    /// Creates an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty context with specified capacity hint.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            roots: Vec::with_capacity(capacity),
            queue: VecDeque::with_capacity(capacity),
        }
    }

    /// Clears the scratch buffers while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.roots.clear();
        self.queue.clear();
    }
}

thread_local! {
    static DEFAULT_PROPAGATION_CONTEXT: std::cell::RefCell<TransformPropagationContext> =
        std::cell::RefCell::new(TransformPropagationContext::new());
}

/// Recursively propagates local transforms down the scene hierarchy into global transform matrices,
/// reusing internal thread-local scratch buffers to ensure zero frame allocations.
///
/// Designed to be executed in the [`korppi_ecs::Stage::PostUpdate`] schedule stage.
pub fn propagate_transforms(world: &mut World) {
    DEFAULT_PROPAGATION_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        propagate_transforms_with_context(world, &mut ctx);
    });
}

/// Propagates local transforms down the scene hierarchy using an explicitly provided scratch context.
pub fn propagate_transforms_with_context(world: &mut World, ctx: &mut TransformPropagationContext) {
    ctx.clear();

    // 1. Gather all entities with LocalTransform
    {
        let mut query = world.query_mut::<(Entity, &LocalTransform)>();
        for (entity, local) in &mut query {
            ctx.roots.push((entity, local.compute_matrix()));
        }
    }

    // 2. Identify roots: entities without a `ChildOf` parent
    for &(entity, local_mat) in &ctx.roots {
        if world.parent_of(entity).is_none() {
            ctx.queue.push_back((entity, local_mat));
        }
    }

    // 3. Breadth-first traversal down children hierarchies (`ChildOf` forms no cycles)
    while let Some((entity, world_matrix)) = ctx.queue.pop_front() {
        if let Some(global) = world.get_mut::<GlobalTransform>(entity) {
            global.matrix = world_matrix;
        } else {
            world.add_component(entity, GlobalTransform::from_matrix(world_matrix));
        }

        // Traverse children without cloning the entities collection
        for &child in world.children(entity) {
            if let Some(child_local) = world.get::<LocalTransform>(child) {
                let child_world_mat = world_matrix * child_local.compute_matrix();
                ctx.queue.push_back((child, child_world_mat));
            }
        }
    }
}
