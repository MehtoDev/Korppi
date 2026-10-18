//! Localized reflection probe representations and view-space intersection tests.

use crate::clustered::light::test_sphere_vs_aabb;
use korppi_math::Vec3;

/// Geometric shape used for localized reflection probe boundaries and parallax correction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbeShape {
    /// Omnidirectional spherical reflection probe.
    Sphere {
        /// Cutoff radius.
        radius: f32,
    },
    /// Box-projected cubemap reflection probe.
    Box {
        /// Half-extents along view-space axes.
        half_extents: Vec3,
    },
}

/// Localized reflection probe source in view space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionProbe {
    /// Position in view space.
    pub position_vs: Vec3,
    /// Probe bounding shape.
    pub shape: ProbeShape,
    /// Specular reflection intensity scale.
    pub intensity: f32,
}

impl ReflectionProbe {
    /// Creates a spherical reflection probe.
    #[inline]
    pub fn sphere(position_vs: Vec3, radius: f32, intensity: f32) -> Self {
        Self {
            position_vs,
            shape: ProbeShape::Sphere { radius },
            intensity,
        }
    }

    /// Creates a box-projected reflection probe.
    #[inline]
    pub fn box_projected(position_vs: Vec3, half_extents: Vec3, intensity: f32) -> Self {
        Self {
            position_vs,
            shape: ProbeShape::Box { half_extents },
            intensity,
        }
    }
}

/// Tests intersection between a reflection probe bounding volume and a froxel AABB.
#[inline]
pub fn test_probe_vs_aabb(probe: &ReflectionProbe, aabb_min: Vec3, aabb_max: Vec3) -> bool {
    match probe.shape {
        ProbeShape::Sphere { radius } => {
            test_sphere_vs_aabb(probe.position_vs, radius, aabb_min, aabb_max)
        }
        ProbeShape::Box { half_extents } => {
            let box_min = probe.position_vs - half_extents;
            let box_max = probe.position_vs + half_extents;
            aabb_min.x <= box_max.x
                && aabb_max.x >= box_min.x
                && aabb_min.y <= box_max.y
                && aabb_max.y >= box_min.y
                && aabb_min.z <= box_max.z
                && aabb_max.z >= box_min.z
        }
    }
}
