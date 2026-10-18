//! 3D infinite ray definition for physics queries and intersection tests.

use crate::Vec3;

/// A half-line starting at `origin` and pointing in normalized `direction`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// The ray starting point.
    pub origin: Vec3,
    /// The normalized direction vector.
    pub direction: Vec3,
}

impl Ray {
    /// Creates a new ray with the given origin and normalized direction.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Evaluates the point along the ray at parameter `t`: `origin + direction * t`.
    #[must_use]
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}
