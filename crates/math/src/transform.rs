//! 3D affine transform combining translation, rotation, and non-uniform scale.

use crate::{Mat3, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// An affine transform in 3-dimensional space (`translation`, `rotation`, `scale`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

    /// Creates a transform at coordinates `(x, y, z)` with identity rotation and unit scale.
    #[must_use]
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
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

    /// Returns the local forward direction vector (`rotation * -Z`).
    #[inline]
    #[must_use]
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Returns the local right direction vector (`rotation * +X`).
    #[inline]
    #[must_use]
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Returns the local up direction vector (`rotation * +Y`).
    #[inline]
    #[must_use]
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Rotates this transform so its local `-Z` forward axis points toward `target` with `up`.
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let back = (self.translation - target)
            .try_normalize()
            .unwrap_or(Vec3::Z);
        let up_norm = up.try_normalize().unwrap_or(Vec3::Y);
        let right = up_norm
            .cross(back)
            .try_normalize()
            .unwrap_or_else(|| back.any_orthonormal_vector());
        let up_ortho = back.cross(right);
        self.rotation = Quat::from_mat3(&Mat3::from_cols(right, up_ortho, back));
    }

    /// Returns a copy of this transform rotated to look at `target` with `up`.
    #[must_use]
    pub fn looking_at(mut self, target: Vec3, up: Vec3) -> Self {
        self.look_at(target, up);
        self
    }

    /// Computes the 4x4 affine model matrix corresponding to this transform.
    #[must_use]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Computes the 4x4 affine model matrix corresponding to this transform (alias for [`Self::to_matrix`]).
    #[inline]
    #[must_use]
    pub fn compute_matrix(&self) -> Mat4 {
        self.to_matrix()
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

    /// Transforms a 3D point in local space to world space (`translation + rotation * (scale * point)`).
    #[must_use]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * (self.scale * point)
    }

    /// Transforms a direction vector from local space to world space (`rotation * (scale * vector)`).
    #[must_use]
    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation * (self.scale * vector)
    }

    /// Transforms a 3D point from world space back into local space (`scale^-1 * (rotation^-1 * (point - translation))`).
    ///
    /// Unlike [`Self::inverse`], this method is exact for all points even when `self` combines
    /// a non-identity `rotation` with non-uniform `scale`.
    #[inline]
    #[must_use]
    pub fn inverse_transform_point(&self, point: Vec3) -> Vec3 {
        (self.rotation.inverse() * (point - self.translation)) / self.scale
    }

    /// Transforms a 3D direction vector from world space back into local space (`scale^-1 * (rotation^-1 * vector)`).
    #[inline]
    #[must_use]
    pub fn inverse_transform_vector(&self, vector: Vec3) -> Vec3 {
        (self.rotation.inverse() * vector) / self.scale
    }

    /// Computes the TRS inverse transform.
    ///
    /// Note that a decomposed `(translation, rotation, scale)` transform can only represent
    /// the exact inverse of another TRS transform over all points when `scale` is uniform (or
    /// when `rotation` is identity), because $S^{-1} R^{-1}$ introduces shear when $S$ is
    /// non-uniform. At the origin (`Vec3::ZERO`), `t.transform_point(t.inverse().transform_point(Vec3::ZERO))`
    /// is always exact; for arbitrary points under non-uniform scale, use [`Self::inverse_transform_point`]
    /// or `self.to_matrix().inverse()`.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv_scale = Vec3::new(1.0 / self.scale.x, 1.0 / self.scale.y, 1.0 / self.scale.z);
        let inv_rotation = self.rotation.inverse();
        let inv_translation = (inv_rotation * -self.translation) * inv_scale;

        Self {
            translation: inv_translation,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }
}
