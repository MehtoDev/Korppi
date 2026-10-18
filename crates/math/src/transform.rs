//! 3D affine transform combining translation, rotation, and non-uniform scale.

use crate::{Mat4, Quat, Vec3};

/// An affine transform in 3-dimensional space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Position translation vector.
    pub translation: Vec3,
    /// Orientation quaternion.
    pub rotation: Quat,
    /// Scale factor vector.
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// Identity transform with zero translation, identity rotation, and unit scale.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Creates a transform with only a translation.
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Creates a transform with only a rotation.
    #[must_use]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation,
            scale: Vec3::ONE,
        }
    }

    /// Creates a transform with only a non-uniform scale.
    #[must_use]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale,
        }
    }

    /// Computes the 4x4 affine model matrix corresponding to this transform.
    #[must_use]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Combines this parent transform with a child transform (`parent * child`).
    #[must_use]
    pub fn combine(&self, child: &Self) -> Self {
        Self {
            translation: self.translation + self.rotation * (self.scale * child.translation),
            rotation: self.rotation * child.rotation,
            scale: self.scale * child.scale,
        }
    }

    /// Transforms a 3D point in local space to world space.
    #[must_use]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * (self.scale * point)
    }

    /// Transforms a direction vector from local space to world space (ignoring translation).
    #[must_use]
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation * (self.scale * vector)
    }

    /// Computes the inverse transform.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv_scale = Vec3::new(1.0 / self.scale.x, 1.0 / self.scale.y, 1.0 / self.scale.z);
        let inv_rotation = self.rotation.inverse();
        let inv_translation = inv_rotation * (-self.translation * inv_scale);

        Self {
            translation: inv_translation,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }
}
