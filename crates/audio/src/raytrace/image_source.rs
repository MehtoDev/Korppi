//! Early reflections as image sources (FIX-7.5, ADR-051; finding AUD-08).
//!
//! A specular reflection off a chain of surfaces reaches an emitter as if from the emitter's image,
//! mirrored across the surfaces' planes from the last to the first. The path tracer finds which
//! chains of surfaces its rays reflect off ([`super::PathTracer::chains`]); for each emitter,
//! [`SpecularChain::image_path`] then gives the exact path through a chain, after checking that it
//! exists: that each mirror point lies on its surface and that nothing blocks a leg. A path found
//! this way does not depend on which rays happened to pass the emitter, so a few rays give
//! steady early reflections.

use super::acoustic_geometry::AcousticGeometry;
use super::geometry::AcousticRay;
use korppi_math::Vec3;

/// The reflections a chain holds at most.
pub const MAX_IMAGE_ORDER: usize = 3;
/// Two normals at least this close in cosine are the same plane's.
const SAME_NORMAL: f32 = 0.9999;
/// Two plane offsets at most this far apart, in meters, are the same plane's.
const SAME_OFFSET: f32 = 1e-3;
/// How far a reflected leg starts off its surface, in meters (as the tracer's).
const SURFACE_OFFSET: f32 = 1e-4;

/// The plane of a reflecting surface: its normal, facing where the sound comes from, and its
/// offset along it.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ReflectingPlane {
    /// The unit normal, facing the incoming sound.
    pub normal: Vec3,
    /// `normal · p` for any point `p` on the plane.
    pub offset: f32,
}

impl ReflectingPlane {
    /// The plane through `point` with unit `normal`.
    #[must_use]
    pub fn through(point: Vec3, normal: Vec3) -> Self {
        Self {
            normal,
            offset: normal.dot(point),
        }
    }

    /// `point` mirrored across the plane.
    #[must_use]
    pub fn mirror(&self, point: Vec3) -> Vec3 {
        point - self.normal * (2.0 * (self.normal.dot(point) - self.offset))
    }

    /// Whether a surface at `point` facing `normal` lies in this plane, facing the same way.
    fn holds(&self, point: Vec3, normal: Vec3) -> bool {
        normal.dot(self.normal) > SAME_NORMAL
            && (self.normal.dot(point) - self.offset).abs() < SAME_OFFSET
    }

    fn same(&self, other: &Self) -> bool {
        self.normal.dot(other.normal) > SAME_NORMAL
            && (self.offset - other.offset).abs() < SAME_OFFSET
    }
}

/// The planes a specular path reflects off, in order from the listener.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SpecularChain {
    planes: [ReflectingPlane; MAX_IMAGE_ORDER],
    order: usize,
}

/// An exact specular path from the listener to an emitter.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ImagePath {
    /// Its length, in meters: the distance from the listener to the emitter's image.
    pub length: f32,
    /// The unit direction it arrives from, seen from the listener.
    pub arrival: Vec3,
    /// The unit direction it leaves the emitter in.
    pub departure: Vec3,
    /// The share of the energy its surfaces reflect specularly per band (`AIR_ABSORPTION_COEFFS`'
    /// bands): the product of `(1 − absorption)(1 − scattering)` over its reflections.
    pub reflectance: [f32; 3],
    /// Its reflections.
    pub order: usize,
}

impl SpecularChain {
    /// The chain of the first [`MAX_IMAGE_ORDER`] of `planes`.
    #[must_use]
    pub fn new(planes: &[ReflectingPlane]) -> Self {
        let order = planes.len().min(MAX_IMAGE_ORDER);
        let mut chain = Self {
            order,
            ..Self::default()
        };
        chain.planes[..order].copy_from_slice(&planes[..order]);
        chain
    }

    /// Its planes, from the listener.
    #[must_use]
    pub fn planes(&self) -> &[ReflectingPlane] {
        &self.planes[..self.order]
    }

    /// Its reflections.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Whether `other` reflects off the same planes in the same order.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        self.order == other.order
            && self
                .planes()
                .iter()
                .zip(other.planes())
                .all(|(a, b)| a.same(b))
    }

    /// The specular path from `listener` to `emitter` off this chain's planes, if `geometry`
    /// has it: each leg reaches its plane's surface first, and the last leg reaches the emitter
    /// unblocked.
    #[must_use]
    pub fn image_path(
        &self,
        geometry: &AcousticGeometry,
        listener: Vec3,
        emitter: Vec3,
    ) -> Option<ImagePath> {
        if self.order == 0 {
            return None;
        }
        let image = self
            .planes()
            .iter()
            .rev()
            .fold(emitter, |image, plane| plane.mirror(image));
        let length = (image - listener).length();
        if length < 1e-4 {
            return None;
        }
        let arrival = (image - listener) / length;
        let (mut origin, mut direction) = (listener, arrival);
        let mut reflectance = [1.0f32; 3];
        let mut travelled = 0.0;
        for plane in self.planes() {
            let hit = geometry.raycast_closest(
                &AcousticRay { origin, direction },
                length - travelled + 0.01,
            )?;
            if !plane.holds(hit.point, hit.normal) {
                return None;
            }
            for (band, share) in reflectance.iter_mut().enumerate() {
                *share *= (1.0 - hit.material.absorption[band]).max(0.0)
                    * (1.0 - hit.material.scattering).max(0.0);
            }
            travelled += hit.distance;
            direction = (direction - 2.0 * direction.dot(hit.normal) * hit.normal).normalize();
            origin = hit.point + hit.normal * SURFACE_OFFSET;
        }
        let rest = emitter - origin;
        let remaining = rest.length();
        if remaining < 1e-4 {
            return None;
        }
        // Every mirror point on its plane, the last leg points at the emitter.
        let toward = rest / remaining;
        if geometry
            .raycast_closest(
                &AcousticRay {
                    origin,
                    direction: toward,
                },
                remaining,
            )
            .is_some()
        {
            return None;
        }
        Some(ImagePath {
            length,
            arrival,
            departure: -toward,
            reflectance,
            order: self.order,
        })
    }
}
