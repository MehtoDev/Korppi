//! 3D infinite ray definition for physics queries and intersection tests.

use crate::Vec3;
use serde::{Deserialize, Serialize};

/// A half-line starting at `origin` and pointing in normalized `direction`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ray {
    /// The ray starting point.
    pub origin: Vec3,
    /// The normalized direction vector.
    pub direction: Vec3,
}

impl Ray {
    /// Creates a new ray with the given `origin` and normalized `direction`.
    ///
    /// If `direction` is zero or too small to normalize cleanly, falls back to [`Vec3::X`]
    /// (matching [`crate::Ray2d::new`]) so `direction` is never `NaN`.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.try_normalize().unwrap_or(Vec3::X),
        }
    }

    /// Evaluates the point along the ray at parameter `t`: `origin + direction * t`.
    #[must_use]
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}
