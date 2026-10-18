//! 2D physics and gameplay debug drawing utilities compiling to QuadBatch2d.

use crate::render2d::batch::QuadBatch2d;
use korppi_math::Vec2;
use std::f32::consts::PI;

/// Helper for drawing 2D debug wireframes, colliders, contact points, and vectors into a `QuadBatch2d`.
pub struct DebugDraw2d<'a> {
    /// Target quad batch receiving generated geometry.
    pub batch: &'a mut QuadBatch2d,
}

impl<'a> DebugDraw2d<'a> {
    /// Creates a new debug drawer referencing an active quad batch.
    pub fn new(batch: &'a mut QuadBatch2d) -> Self {
        Self { batch }
    }

    /// Draws a line segment with the given thickness.
    pub fn draw_line(&mut self, p1: Vec2, p2: Vec2, thickness: f32, color: [f32; 4]) {
        self.batch.push_line(p1, p2, thickness, color);
    }

    /// Draws a wireframe circle composed of `segments` linear chords.
    pub fn draw_circle(
        &mut self,
        center: Vec2,
        radius: f32,
        segments: usize,
        thickness: f32,
        color: [f32; 4],
    ) {
        let seg_count = segments.max(3);
        let step = (2.0 * PI) / seg_count as f32;

        let mut prev = center + Vec2::new(radius, 0.0);
        for i in 1..=seg_count {
            let angle = i as f32 * step;
            let next = center + Vec2::new(radius * angle.cos(), radius * angle.sin());
            self.draw_line(prev, next, thickness, color);
            prev = next;
        }
    }

    /// Draws a wireframe capsule defined by two center points and a radius.
    pub fn draw_capsule(
        &mut self,
        p1: Vec2,
        p2: Vec2,
        radius: f32,
        segments: usize,
        thickness: f32,
        color: [f32; 4],
    ) {
        let axis = p2 - p1;
        let len = axis.length();
        let u = if len > 1e-6 { axis / len } else { Vec2::Y };
        let perp = Vec2::new(-u.y, u.x);

        // Two side lines
        self.draw_line(p1 + perp * radius, p2 + perp * radius, thickness, color);
        self.draw_line(p1 - perp * radius, p2 - perp * radius, thickness, color);

        // Hemispherical caps
        let cap_segs = (segments / 2).max(2);
        let step = PI / cap_segs as f32;
        let base_angle = u.y.atan2(u.x);

        // Cap at p2 (facing along u)
        let mut prev = p2 + perp * radius;
        for i in 1..=cap_segs {
            let a = base_angle - (PI * 0.5) + i as f32 * step;
            let next = p2 + Vec2::new(radius * a.cos(), radius * a.sin());
            self.draw_line(prev, next, thickness, color);
            prev = next;
        }

        // Cap at p1 (facing along -u)
        let mut prev = p1 - perp * radius;
        for i in 1..=cap_segs {
            let a = base_angle + (PI * 0.5) + i as f32 * step;
            let next = p1 + Vec2::new(radius * a.cos(), radius * a.sin());
            self.draw_line(prev, next, thickness, color);
            prev = next;
        }
    }

    /// Draws a closed wireframe polygon from an array of vertices.
    pub fn draw_polygon(&mut self, vertices: &[Vec2], thickness: f32, color: [f32; 4]) {
        let n = vertices.len();
        if n < 2 {
            return;
        }
        for i in 0..n {
            let p1 = vertices[i];
            let p2 = vertices[(i + 1) % n];
            self.draw_line(p1, p2, thickness, color);
        }
    }

    /// Draws an axis-aligned box wireframe.
    pub fn draw_aabb(&mut self, min: Vec2, max: Vec2, thickness: f32, color: [f32; 4]) {
        let p0 = min;
        let p1 = Vec2::new(max.x, min.y);
        let p2 = max;
        let p3 = Vec2::new(min.x, max.y);
        self.draw_polygon(&[p0, p1, p2, p3], thickness, color);
    }

    /// Draws a point indicator cross with diameter `size`.
    pub fn draw_point(&mut self, point: Vec2, size: f32, color: [f32; 4]) {
        let half = size * 0.5;
        self.batch.push_quad(
            point - Vec2::splat(half),
            point + Vec2::splat(half),
            Vec2::ZERO,
            Vec2::ONE,
            color,
        );
    }

    /// Draws a surface normal indicator line.
    pub fn draw_normal(
        &mut self,
        point: Vec2,
        normal: Vec2,
        length: f32,
        thickness: f32,
        color: [f32; 4],
    ) {
        let dir = normal.normalize_or_zero();
        self.draw_line(point, point + dir * length, thickness, color);
    }

    /// Draws a directional ray up to `length`.
    pub fn draw_ray(
        &mut self,
        origin: Vec2,
        direction: Vec2,
        length: f32,
        thickness: f32,
        color: [f32; 4],
    ) {
        let dir = direction.normalize_or_zero();
        self.draw_line(origin, origin + dir * length, thickness, color);
    }
}
