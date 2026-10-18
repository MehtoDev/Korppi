//! Acoustic scene representation, geometric primitives, and ray intersection routines.

use super::material::AcousticMaterial;
use korppi_math::Vec3;

/// 3D ray for acoustic path tracing and occlusion queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticRay {
    /// Ray start origin in meters.
    pub origin: Vec3,
    /// Normalized unit direction vector.
    pub direction: Vec3,
}

impl AcousticRay {
    /// Creates a new acoustic ray, automatically normalizing direction.
    #[must_use]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
        }
    }

    /// Evaluates point along ray at distance `t`: `origin + direction * t`.
    #[must_use]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Ray intersection result against an acoustic surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticHit {
    /// Distance along the ray in meters.
    pub distance: f32,
    /// World position of surface hit point.
    pub point: Vec3,
    /// Surface normal oriented towards the incident ray direction.
    pub normal: Vec3,
    /// Acoustic material properties of the intersected surface.
    pub material: AcousticMaterial,
}

/// 3D triangle primitive with acoustic material properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticTriangle {
    /// First vertex.
    pub v0: Vec3,
    /// Second vertex.
    pub v1: Vec3,
    /// Third vertex.
    pub v2: Vec3,
    /// Geometric face normal.
    pub normal: Vec3,
    /// Acoustic surface material.
    pub material: AcousticMaterial,
}

impl AcousticTriangle {
    /// Creates an acoustic triangle from 3 vertices and a material.
    #[must_use]
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material: AcousticMaterial) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalize_or_zero();
        Self {
            v0,
            v1,
            v2,
            normal,
            material,
        }
    }

    /// Tests ray intersection using the Möller-Trumbore algorithm.
    ///
    /// Surface normal is oriented to always oppose incident ray direction.
    #[must_use]
    pub fn intersect(&self, ray: &AcousticRay) -> Option<AcousticHit> {
        const EPSILON: f32 = 1e-6;
        let edge1 = self.v1 - self.v0;
        let edge2 = self.v2 - self.v0;
        let h = ray.direction.cross(edge2);
        let a = edge1.dot(h);

        if a.abs() < EPSILON {
            return None; // Parallel to triangle
        }

        let f = 1.0 / a;
        let s = ray.origin - self.v0;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(edge1);
        let v = f * ray.direction.dot(q);
        if v < 0.0 || (u + v) > 1.0 {
            return None;
        }

        let t = f * edge2.dot(q);
        if t <= EPSILON {
            return None; // Hit behind or right at ray origin
        }

        let point = ray.at(t);
        let mut normal = self.normal;
        if normal.dot(ray.direction) > 0.0 {
            normal = -normal;
        }

        Some(AcousticHit {
            distance: t,
            point,
            normal,
            material: self.material,
        })
    }
}

/// 3D box primitive representing acoustic rooms or obstacles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticBox {
    /// World center position.
    pub center: Vec3,
    /// Total dimensions (width along X, height along Y, depth along Z).
    pub size: Vec3,
    /// Acoustic material of the box surfaces.
    pub material: AcousticMaterial,
}

impl AcousticBox {
    /// Creates a new acoustic box.
    #[must_use]
    pub const fn new(center: Vec3, size: Vec3, material: AcousticMaterial) -> Self {
        Self {
            center,
            size,
            material,
        }
    }

    /// Converts this box into 12 triangles forming its 6 faces.
    #[must_use]
    pub fn to_triangles(&self) -> [AcousticTriangle; 12] {
        let half = self.size * 0.5;
        let c = self.center;

        // 8 corner points
        let p000 = c + Vec3::new(-half.x, -half.y, -half.z);
        let p001 = c + Vec3::new(-half.x, -half.y, half.z);
        let p010 = c + Vec3::new(-half.x, half.y, -half.z);
        let p011 = c + Vec3::new(-half.x, half.y, half.z);
        let p100 = c + Vec3::new(half.x, -half.y, -half.z);
        let p101 = c + Vec3::new(half.x, -half.y, half.z);
        let p110 = c + Vec3::new(half.x, half.y, -half.z);
        let p111 = c + Vec3::new(half.x, half.y, half.z);

        let mat = self.material;

        [
            // +X face
            AcousticTriangle::new(p100, p110, p111, mat),
            AcousticTriangle::new(p100, p111, p101, mat),
            // -X face
            AcousticTriangle::new(p000, p001, p011, mat),
            AcousticTriangle::new(p000, p011, p010, mat),
            // +Y face
            AcousticTriangle::new(p010, p011, p111, mat),
            AcousticTriangle::new(p010, p111, p110, mat),
            // -Y face
            AcousticTriangle::new(p000, p100, p101, mat),
            AcousticTriangle::new(p000, p101, p001, mat),
            // +Z face
            AcousticTriangle::new(p001, p101, p111, mat),
            AcousticTriangle::new(p001, p111, p011, mat),
            // -Z face
            AcousticTriangle::new(p000, p010, p110, mat),
            AcousticTriangle::new(p000, p110, p100, mat),
        ]
    }
}

/// Collection of acoustic geometric surfaces and spatial acceleration.
#[derive(Debug, Default, Clone)]
pub struct AcousticScene {
    triangles: Vec<AcousticTriangle>,
}

impl AcousticScene {
    /// Creates an empty acoustic scene.
    #[must_use]
    pub fn new() -> Self {
        Self {
            triangles: Vec::new(),
        }
    }

    /// Adds a single acoustic triangle to the scene.
    pub fn add_triangle(&mut self, triangle: AcousticTriangle) {
        self.triangles.push(triangle);
    }

    /// Adds an acoustic box (expanding it to 12 triangles) to the scene.
    pub fn add_box(&mut self, acoustic_box: AcousticBox) {
        for tri in acoustic_box.to_triangles() {
            self.triangles.push(tri);
        }
    }

    /// Returns the total number of triangles in the acoustic scene.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Finds the closest acoustic surface intersection along the ray.
    #[must_use]
    pub fn raycast_closest(&self, ray: &AcousticRay) -> Option<AcousticHit> {
        let mut closest: Option<AcousticHit> = None;

        for tri in &self.triangles {
            if let Some(hit) = tri.intersect(ray) {
                match &closest {
                    Some(prev) if hit.distance < prev.distance => {
                        closest = Some(hit);
                    }
                    None => {
                        closest = Some(hit);
                    }
                    _ => {}
                }
            }
        }

        closest
    }

    /// Evaluates multi-wall transmission across all obstacles up to `max_distance`.
    ///
    /// Multiplies transmission coefficients across all intersecting barriers:
    /// $$\mathbf{T}(b) = \prod_{k=1}^M \boldsymbol{\tau}_{m_k}(b)$$
    #[must_use]
    pub fn raycast_transmission(&self, ray: &AcousticRay, max_distance: f32) -> [f32; 3] {
        const EPSILON: f32 = 1e-4;
        let mut transmission = [1.0f32; 3];
        let mut hit_distances = [0.0f32; 32];
        let mut hit_count = 0;

        for tri in &self.triangles {
            if let Some(hit) = tri.intersect(ray) {
                if hit.distance > EPSILON && hit.distance < (max_distance - EPSILON) {
                    let is_duplicate = hit_distances[..hit_count]
                        .iter()
                        .any(|&d| (d - hit.distance).abs() < 1e-3);
                    if !is_duplicate {
                        if hit_count < hit_distances.len() {
                            hit_distances[hit_count] = hit.distance;
                            hit_count += 1;
                        }
                        transmission[0] *= hit.material.transmission[0];
                        transmission[1] *= hit.material.transmission[1];
                        transmission[2] *= hit.material.transmission[2];
                    }
                }
            }
        }

        transmission
    }

    /// Computes statistical reverberation time RT60 across 3 bands using Eyring's formula
    /// based on total enclosing surface area, bounding volume, and area-weighted absorption.
    #[must_use]
    pub fn estimate_rt60_eyring(&self) -> [f32; 3] {
        if self.triangles.is_empty() {
            return [0.5, 0.5, 0.5];
        }

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut total_area = 0.0f32;
        let mut weighted_alpha = [0.0f32; 3];

        for tri in &self.triangles {
            min = min.min(tri.v0).min(tri.v1).min(tri.v2);
            max = max.max(tri.v0).max(tri.v1).max(tri.v2);

            let edge1 = tri.v1 - tri.v0;
            let edge2 = tri.v2 - tri.v0;
            let area = 0.5 * edge1.cross(edge2).length();
            total_area += area;

            weighted_alpha[0] += area * tri.material.absorption[0];
            weighted_alpha[1] += area * tri.material.absorption[1];
            weighted_alpha[2] += area * tri.material.absorption[2];
        }

        if total_area < 1e-4 {
            return [0.5, 0.5, 0.5];
        }

        let avg_alpha = [
            (weighted_alpha[0] / total_area).clamp(0.001, 0.999),
            (weighted_alpha[1] / total_area).clamp(0.001, 0.999),
            (weighted_alpha[2] / total_area).clamp(0.001, 0.999),
        ];

        let dim = (max - min).max(Vec3::splat(1.0));
        let volume = dim.x * dim.y * dim.z;

        [
            super::energy_decay::eyring_rt60(volume, total_area, avg_alpha[0]).clamp(0.05, 10.0),
            super::energy_decay::eyring_rt60(volume, total_area, avg_alpha[1]).clamp(0.05, 10.0),
            super::energy_decay::eyring_rt60(volume, total_area, avg_alpha[2]).clamp(0.05, 10.0),
        ]
    }
}

impl crate::spatial::AcousticRaycaster for AcousticScene {
    fn raycast_occlusion(&self, from: Vec3, to: Vec3) -> Option<f32> {
        let delta = to - from;
        let dist = delta.length();
        if dist < 1e-4 {
            return None;
        }

        let ray = AcousticRay::new(from, delta / dist);
        if let Some(hit) = self.raycast_closest(&ray) {
            if hit.distance > 1e-4 && hit.distance < (dist - 1e-4) {
                let trans = self.raycast_transmission(&ray, dist);
                let avg_trans = (trans[0] + trans[1] + trans[2]) / 3.0;
                let occ = (1.0 - avg_trans).clamp(0.0, 1.0);
                if occ > 1e-4 {
                    return Some(occ);
                }
            }
        }

        None
    }
}
