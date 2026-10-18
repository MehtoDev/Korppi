//! Axis-Aligned Bounding Box (AABB) volume for broadphase spatial trees and culling.

use crate::{Ray, Vec3};
use serde::{Deserialize, Serialize};

/// An Axis-Aligned Bounding Box defined by minimum and maximum coordinate bounds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    /// Minimum corner coordinate.
    pub min: Vec3,
    /// Maximum corner coordinate.
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }
}

impl Aabb {
    /// Creates an AABB from explicit minimum and maximum points.
    #[must_use]
    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self {
            min: min.min(max),
            max: min.max(max),
        }
    }

    /// Creates an AABB from a center point and half-extents.
    #[must_use]
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents.abs(),
            max: center + half_extents.abs(),
        }
    }

    /// Computes the center point of the bounding box.
    #[must_use]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Computes the full extents (dimensions) along each axis.
    #[must_use]
    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }

    /// Computes half of the dimensions along each axis.
    #[must_use]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Computes the surface area of the bounding box (Surface Area Heuristic / SAH).
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let d = self.extents();
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Returns `true` if this AABB overlaps another AABB.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Returns `true` if this AABB completely contains another AABB.
    #[must_use]
    pub fn contains(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && self.max.x >= other.max.x
            && self.min.y <= other.min.y
            && self.max.y >= other.max.y
            && self.min.z <= other.min.z
            && self.max.z >= other.max.z
    }

    /// Returns `true` if this AABB contains a point.
    #[must_use]
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Computes the minimum bounding box encompassing both `self` and `other`.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Expands the AABB uniformly by `margin` on all sides.
    #[must_use]
    pub fn expand(&self, margin: f32) -> Self {
        Self {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }

    /// Expands the AABB by a safety margin and predicts movement along `displacement`.
    /// Matches `box3d`'s fat AABB strategy for BVH dynamic broadphase caching.
    #[must_use]
    pub fn fatten(&self, displacement: Vec3, margin: f32) -> Self {
        let mut fat = self.expand(margin);
        if displacement.x < 0.0 {
            fat.min.x += displacement.x;
        } else {
            fat.max.x += displacement.x;
        }
        if displacement.y < 0.0 {
            fat.min.y += displacement.y;
        } else {
            fat.max.y += displacement.y;
        }
        if displacement.z < 0.0 {
            fat.min.z += displacement.z;
        } else {
            fat.max.z += displacement.z;
        }
        fat
    }

    /// Tests for ray intersection using the Kay-Kajiya slab method within `[t_min, t_max]`.
    /// Returns the entry distance parameter `t` along the ray if hit.
    #[must_use]
    pub fn intersect_ray(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<f32> {
        let mut t0 = t_min;
        let mut t1 = t_max;

        for i in 0..3 {
            let inv_d = 1.0 / ray.direction[i];
            let mut t_near = (self.min[i] - ray.origin[i]) * inv_d;
            let mut t_far = (self.max[i] - ray.origin[i]) * inv_d;

            if inv_d < 0.0 {
                std::mem::swap(&mut t_near, &mut t_far);
            }

            t0 = t0.max(t_near);
            t1 = t1.min(t_far);

            if t0 > t1 {
                return None;
            }
        }

        Some(t0)
    }
}
