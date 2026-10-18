//! 3D physics and world debug drawing utilities compiling wireframe lines and crosses.

use korppi_math::{Quat, Vec3};
use std::f32::consts::PI;

/// Vertex structure for 3D debug lines.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineVertex3d {
    /// World position (x, y, z).
    pub position: [f32; 3],
    /// RGBA color.
    pub color: [f32; 4],
}

/// Dynamic 3D debug drawer buffering wireframe boxes, sphere circles, capsule rims, contact crosses, and rays.
#[derive(Clone, Debug, Default)]
pub struct DebugDraw3d {
    vertices: Vec<LineVertex3d>,
    indices: Vec<u32>,
}

impl DebugDraw3d {
    /// Creates an empty 3D debug drawer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(256),
            indices: Vec::with_capacity(256),
        }
    }

    /// Appends a 3D line segment.
    pub fn draw_line(&mut self, p1: Vec3, p2: Vec3, color: [f32; 4]) {
        let base = self.vertices.len() as u32;
        self.vertices.push(LineVertex3d {
            position: [p1.x, p1.y, p1.z],
            color,
        });
        self.vertices.push(LineVertex3d {
            position: [p2.x, p2.y, p2.z],
            color,
        });
        self.indices.push(base);
        self.indices.push(base + 1);
    }

    /// Draws an oriented 3D bounding box wireframe (12 edges).
    pub fn draw_box(&mut self, center: Vec3, half_extents: Vec3, rotation: Quat, color: [f32; 4]) {
        let h = half_extents;
        let local_corners = [
            Vec3::new(-h.x, -h.y, -h.z),
            Vec3::new(h.x, -h.y, -h.z),
            Vec3::new(h.x, h.y, -h.z),
            Vec3::new(-h.x, h.y, -h.z),
            Vec3::new(-h.x, -h.y, h.z),
            Vec3::new(h.x, -h.y, h.z),
            Vec3::new(h.x, h.y, h.z),
            Vec3::new(-h.x, h.y, h.z),
        ];

        let mut world_corners = [Vec3::ZERO; 8];
        for i in 0..8 {
            world_corners[i] = center + rotation * local_corners[i];
        }

        // Bottom 4 edges
        self.draw_line(world_corners[0], world_corners[1], color);
        self.draw_line(world_corners[1], world_corners[2], color);
        self.draw_line(world_corners[2], world_corners[3], color);
        self.draw_line(world_corners[3], world_corners[0], color);

        // Top 4 edges
        self.draw_line(world_corners[4], world_corners[5], color);
        self.draw_line(world_corners[5], world_corners[6], color);
        self.draw_line(world_corners[6], world_corners[7], color);
        self.draw_line(world_corners[7], world_corners[4], color);

        // Vertical 4 edges
        self.draw_line(world_corners[0], world_corners[4], color);
        self.draw_line(world_corners[1], world_corners[5], color);
        self.draw_line(world_corners[2], world_corners[6], color);
        self.draw_line(world_corners[3], world_corners[7], color);
    }

    /// Draws an axis-aligned bounding box wireframe.
    pub fn draw_aabb(&mut self, min: Vec3, max: Vec3, color: [f32; 4]) {
        let center = (min + max) * 0.5;
        let half_extents = (max - min) * 0.5;
        self.draw_box(center, half_extents, Quat::IDENTITY, color);
    }

    /// Draws a wireframe sphere composed of 3 great circles along XY, XZ, YZ planes.
    pub fn draw_sphere(&mut self, center: Vec3, radius: f32, segments: usize, color: [f32; 4]) {
        let n = segments.max(4);
        let step = (2.0 * PI) / n as f32;

        // XY plane circle
        for i in 0..n {
            let a1 = i as f32 * step;
            let a2 = (i + 1) as f32 * step;
            let p1 = center + Vec3::new(radius * a1.cos(), radius * a1.sin(), 0.0);
            let p2 = center + Vec3::new(radius * a2.cos(), radius * a2.sin(), 0.0);
            self.draw_line(p1, p2, color);
        }

        // XZ plane circle
        for i in 0..n {
            let a1 = i as f32 * step;
            let a2 = (i + 1) as f32 * step;
            let p1 = center + Vec3::new(radius * a1.cos(), 0.0, radius * a1.sin());
            let p2 = center + Vec3::new(radius * a2.cos(), 0.0, radius * a2.sin());
            self.draw_line(p1, p2, color);
        }

        // YZ plane circle
        for i in 0..n {
            let a1 = i as f32 * step;
            let a2 = (i + 1) as f32 * step;
            let p1 = center + Vec3::new(0.0, radius * a1.cos(), radius * a1.sin());
            let p2 = center + Vec3::new(0.0, radius * a2.cos(), radius * a2.sin());
            self.draw_line(p1, p2, color);
        }
    }

    /// Draws a wireframe capsule along segment `(p1, p2)` with spherical ends.
    pub fn draw_capsule(
        &mut self,
        p1: Vec3,
        p2: Vec3,
        radius: f32,
        segments: usize,
        color: [f32; 4],
    ) {
        self.draw_sphere(p1, radius, segments, color);
        self.draw_sphere(p2, radius, segments, color);

        let axis = p2 - p1;
        let len = axis.length();
        if len > 1e-6 {
            let u = axis / len;
            let mut perp1 = u.cross(Vec3::Y);
            if perp1.length_squared() < 1e-4 {
                perp1 = u.cross(Vec3::X);
            }
            let perp1 = perp1.normalize() * radius;
            let perp2 = u.cross(perp1).normalize() * radius;

            // 4 connecting lines along cylinder body
            self.draw_line(p1 + perp1, p2 + perp1, color);
            self.draw_line(p1 - perp1, p2 - perp1, color);
            self.draw_line(p1 + perp2, p2 + perp2, color);
            self.draw_line(p1 - perp2, p2 - perp2, color);
        }
    }

    /// Draws a 3D contact point cross (3 orthogonal axis lines of length `size`).
    pub fn draw_cross(&mut self, position: Vec3, size: f32, color: [f32; 4]) {
        let h = size * 0.5;
        self.draw_line(position - Vec3::X * h, position + Vec3::X * h, color);
        self.draw_line(position - Vec3::Y * h, position + Vec3::Y * h, color);
        self.draw_line(position - Vec3::Z * h, position + Vec3::Z * h, color);
    }

    /// Draws a normal vector indicator line.
    pub fn draw_normal(&mut self, point: Vec3, normal: Vec3, length: f32, color: [f32; 4]) {
        let dir = normal.normalize_or_zero();
        self.draw_line(point, point + dir * length, color);
    }

    /// Draws a directional ray indicator line.
    pub fn draw_ray(&mut self, origin: Vec3, direction: Vec3, length: f32, color: [f32; 4]) {
        let dir = direction.normalize_or_zero();
        self.draw_line(origin, origin + dir * length, color);
    }

    /// Returns the number of line segments buffered.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.indices.len() / 2
    }

    /// Returns the slice of line vertices.
    #[must_use]
    pub fn vertices(&self) -> &[LineVertex3d] {
        &self.vertices
    }

    /// Returns the slice of line indices.
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Clears the debug line buffer.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}
