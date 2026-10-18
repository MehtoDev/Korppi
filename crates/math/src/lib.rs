//! SIMD-accelerated 2D/3D mathematics, transforms, and bounding volumes.
//!
//! # Inputs / Outputs
//! Provides native data-oriented vectors ([`Vec2`], [`Vec3`], [`Vec4`]), rotations ([`Rot2`], [`Quat`]),
//! column-major matrices ([`Mat3`], [`Mat4`], [`Affine3A`]), affine transforms ([`Transform`], [`Transform2d`]),
//! bounding volumes ([`Aabb`], [`Aabb2d`], [`Sphere`]), rays ([`Ray`], [`Ray2d`]), and 4-wide SIMD
//! solver lanes ([`FloatW`]).

pub mod aabb;
pub mod aabb2d;
pub mod mat3;
pub mod mat4;
pub mod quat;
pub mod ray;
pub mod ray2d;
pub mod rot2;
pub mod simd;
pub mod sphere;
pub mod transform;
pub mod transform2d;
pub mod vec2;
pub mod vec3;
pub mod vec4;

pub use aabb::Aabb;
pub use aabb2d::Aabb2d;
pub use mat3::Mat3;
pub use mat4::{Affine3A, Mat4};
pub use quat::Quat;
pub use ray::Ray;
pub use ray2d::Ray2d;
pub use rot2::{unwind_angle, Rot2};
pub use simd::FloatW;
pub use sphere::Sphere;
pub use transform::Transform;
pub use transform2d::Transform2d;
pub use vec2::{BVec2, Vec2};
pub use vec3::{BVec3, Vec3};
pub use vec4::{BVec4, Vec4};

/// Common math types re-exported for ergonomic imports.
pub mod prelude {
    pub use crate::aabb::Aabb;
    pub use crate::aabb2d::Aabb2d;
    pub use crate::mat3::Mat3;
    pub use crate::mat4::{Affine3A, Mat4};
    pub use crate::quat::Quat;
    pub use crate::ray::Ray;
    pub use crate::ray2d::Ray2d;
    pub use crate::rot2::{unwind_angle, Rot2};
    pub use crate::sphere::Sphere;
    pub use crate::transform::Transform;
    pub use crate::transform2d::Transform2d;
    pub use crate::vec2::{BVec2, Vec2};
    pub use crate::vec3::{BVec3, Vec3};
    pub use crate::vec4::{BVec4, Vec4};
}
