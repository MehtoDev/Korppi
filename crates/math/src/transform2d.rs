//! 2D affine transform combining translation and rotation.

use crate::rot2::Rot2;
use crate::Vec2;
use serde::{Deserialize, Serialize};

/// A 2D rigid transform combining position translation and 2D rotation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform2d {
    /// Position translation vector.
    pub translation: Vec2,
    /// Orientation complex unit.
    pub rotation: Rot2,
}

impl Default for Transform2d {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform2d {
    /// Identity 2D transform with zero translation and identity rotation.
    pub const IDENTITY: Self = Self {
        translation: Vec2::ZERO,
        rotation: Rot2::IDENTITY,
    };

    /// Creates a transform with only a translation.
    #[must_use]
    pub const fn from_translation(translation: Vec2) -> Self {
        Self {
            translation,
            rotation: Rot2::IDENTITY,
        }
    }

    /// Creates a transform with only a rotation.
    #[must_use]
    pub const fn from_rotation(rotation: Rot2) -> Self {
        Self {
            translation: Vec2::ZERO,
            rotation,
        }
    }

    /// Creates a transform with translation and rotation.
    #[must_use]
    pub const fn from_translation_rotation(translation: Vec2, rotation: Rot2) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// Transforms a point by rotating then translating.
    #[must_use]
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        self.translation + self.rotation.rotate(point)
    }

    /// Inverse transforms a point (un-translates then un-rotates).
    #[must_use]
    pub fn inv_transform_point(&self, point: Vec2) -> Vec2 {
        self.rotation.inv_rotate(point - self.translation)
    }

    /// Transforms a direction vector by rotating only.
    #[must_use]
    pub fn transform_vector(&self, vector: Vec2) -> Vec2 {
        self.rotation.rotate(vector)
    }

    /// Inverse transforms a direction vector by inverse rotating only.
    #[must_use]
    pub fn inv_transform_vector(&self, vector: Vec2) -> Vec2 {
        self.rotation.inv_rotate(vector)
    }

    /// Combines this parent transform with a child transform (`self * child`).
    #[must_use]
    pub fn combine(&self, child: &Self) -> Self {
        Self {
            translation: self.transform_point(child.translation),
            rotation: self.rotation * child.rotation,
        }
    }

    /// `other` relative to this transform, `self⁻¹ · other` (Box2D `b2InvMulTransforms`).
    ///
    /// The translations are subtracted before rotating, so two nearby poses far from the origin
    /// keep their exact offset, unlike `self.inverse().combine(other)`.
    #[must_use]
    pub fn inv_mul(&self, other: &Self) -> Self {
        Self {
            translation: self
                .rotation
                .inv_rotate(other.translation - self.translation),
            rotation: self.rotation.inverse() * other.rotation,
        }
    }

    /// Computes the inverse transform.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv_rot = self.rotation.inverse();
        Self {
            translation: inv_rot.rotate(-self.translation),
            rotation: inv_rot,
        }
    }
}
