//! 3D bounding sphere volume.

use crate::{Aabb, Vec3};
use serde::{Deserialize, Serialize};

/// A 3D bounding sphere defined by a center position and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sphere {
    /// Center of the sphere.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
}

impl Sphere {
    /// Creates a new sphere.
    #[must_use]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            radius: radius.abs(),
        }
    }

    /// Returns `true` if this sphere intersects another sphere.
    #[must_use]
    pub fn intersects_sphere(&self, other: &Self) -> bool {
        let distance_squared = (self.center - other.center).length_squared();
        let radius_sum = self.radius + other.radius;
        distance_squared <= radius_sum * radius_sum
    }

    /// Returns `true` if this sphere intersects an AABB.
    #[must_use]
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        if aabb.is_empty() {
            return false;
        }
        let closest_point = self.center.clamp(aabb.min, aabb.max);
        let distance_squared = (self.center - closest_point).length_squared();
        distance_squared <= self.radius * self.radius
    }

    /// Computes the minimal enclosing AABB for this sphere.
    #[must_use]
    pub fn to_aabb(&self) -> Aabb {
        Aabb {
            min: self.center - Vec3::splat(self.radius),
            max: self.center + Vec3::splat(self.radius),
        }
    }
}
