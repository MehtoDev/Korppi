//! 2D Axis-Aligned Bounding Box (AABB) volume for broadphase and queries.

use crate::Vec2;
use serde::{Deserialize, Serialize};

/// A 2-dimensional Axis-Aligned Bounding Box defined by minimum and maximum extents.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb2d {
    /// Minimum corner of the box.
    pub min: Vec2,
    /// Maximum corner of the box.
    pub max: Vec2,
}

impl Default for Aabb2d {
    fn default() -> Self {
        Self::empty()
    }
}

impl Aabb2d {
    /// Creates a new 2D AABB from minimum and maximum extents.
    #[must_use]
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self {
            min: min.min(max),
            max: min.max(max),
        }
    }

    /// Creates an empty / inverted bounding box for accumulation.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            min: Vec2::splat(f32::INFINITY),
            max: Vec2::splat(f32::NEG_INFINITY),
        }
    }

    /// Creates an AABB containing a single point.
    #[must_use]
    pub const fn from_point(point: Vec2) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    /// Computes the center of the bounding box.
    #[must_use]
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Computes half extents (half-width, half-height).
    #[must_use]
    pub fn half_extents(&self) -> Vec2 {
        (self.max - self.min) * 0.5
    }

    /// Computes total extents (width, height).
    #[must_use]
    pub fn extents(&self) -> Vec2 {
        self.max - self.min
    }

    /// Computes perimeter of the bounding box: `2 * (w + h)`.
    #[must_use]
    pub fn perimeter(&self) -> f32 {
        let d = self.max - self.min;
        2.0 * (d.x + d.y)
    }

    /// Computes area of the bounding box: `w * h`.
    #[must_use]
    pub fn area(&self) -> f32 {
        let d = self.max - self.min;
        d.x * d.y
    }

    /// Checks if this AABB contains a point.
    #[must_use]
    pub fn contains_point(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// Checks if this AABB completely contains another AABB.
    #[must_use]
    pub fn contains_aabb(&self, other: &Self) -> bool {
        self.min.x <= other.min.x
            && self.min.y <= other.min.y
            && self.max.x >= other.max.x
            && self.max.y >= other.max.y
    }

    /// Checks if this AABB intersects / overlaps another AABB.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Computes the union containing both bounding boxes.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Computes the intersection overlap of two bounding boxes.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        if min.x <= max.x && min.y <= max.y {
            Some(Self { min, max })
        } else {
            None
        }
    }

    /// Expands the bounding box uniformly by a margin in all directions.
    #[must_use]
    pub fn fatten(&self, margin: f32) -> Self {
        Self {
            min: self.min - Vec2::splat(margin),
            max: self.max + Vec2::splat(margin),
        }
    }

    /// Expands the bounding box to enclose a displaced version (velocity expansion).
    #[must_use]
    pub fn enlarge_by_displacement(&self, displacement: Vec2) -> Self {
        let min = self.min.min(self.min + displacement);
        let max = self.max.max(self.max + displacement);
        Self { min, max }
    }
}
