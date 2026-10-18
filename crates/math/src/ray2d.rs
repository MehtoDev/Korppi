//! 2D Ray primitive for line and ray casting queries.

use crate::aabb2d::Aabb2d;
use crate::Vec2;
use serde::{Deserialize, Serialize};

/// A 2-dimensional ray defined by an origin and normalized direction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ray2d {
    /// Origin point of the ray.
    pub origin: Vec2,
    /// Normalized direction vector of the ray.
    pub direction: Vec2,
}

impl Ray2d {
    /// Creates a new 2D ray. Direction is normalized if non-zero.
    #[must_use]
    pub fn new(origin: Vec2, direction: Vec2) -> Self {
        let dir_len = direction.length();
        let direction = if dir_len > 1e-6 {
            direction / dir_len
        } else {
            Vec2::X
        };
        Self { origin, direction }
    }

    /// Evaluates the point along the ray at distance `t`: `origin + direction * t`.
    #[must_use]
    pub fn point_at(&self, t: f32) -> Vec2 {
        self.origin + self.direction * t
    }

    /// Intersects this ray with a 2D AABB using the slab method.
    ///
    /// Returns the distance parameter `t >= 0.0` along the ray to the entry hit point,
    /// or `None` if the ray misses or points away.
    #[must_use]
    pub fn intersects_aabb(&self, aabb: &Aabb2d) -> Option<f32> {
        let mut tmin = f32::NEG_INFINITY;
        let mut tmax = f32::INFINITY;

        for i in 0..2 {
            let (origin_i, dir_i, min_i, max_i) = if i == 0 {
                (self.origin.x, self.direction.x, aabb.min.x, aabb.max.x)
            } else {
                (self.origin.y, self.direction.y, aabb.min.y, aabb.max.y)
            };

            if dir_i.abs() < 1e-8 {
                if origin_i < min_i || origin_i > max_i {
                    return None;
                }
            } else {
                let inv_d = 1.0 / dir_i;
                let mut t1 = (min_i - origin_i) * inv_d;
                let mut t2 = (max_i - origin_i) * inv_d;

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }

                tmin = tmin.max(t1);
                tmax = tmax.min(t2);

                if tmin > tmax {
                    return None;
                }
            }
        }

        if tmax < 0.0 {
            None
        } else if tmin < 0.0 {
            Some(0.0)
        } else {
            Some(tmin)
        }
    }
}
