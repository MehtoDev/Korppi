//! High-precision Reverse-Z depth buffer mathematics and conventions.

use korppi_math::Mat4;
use korppi_rhi::CompareFunction;

/// Helper utilities and standard conventions for high-precision Reverse-Z depth buffers.
pub struct ReverseZ;

impl ReverseZ {
    /// Depth compare function for Reverse-Z (`GreaterEqual`).
    pub const DEPTH_COMPARE: CompareFunction = CompareFunction::GreaterEqual;

    /// Clear depth value for Reverse-Z ($0.0$, representing the far plane).
    pub const CLEAR_DEPTH: f32 = 0.0;

    /// Generates a right-handed perspective projection matrix with Reverse-Z mapping:
    /// $Z_{near}$ maps to $1.0$ and $Z_{far}$ maps to $0.0$.
    #[inline]
    pub fn perspective_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Mat4 {
        Mat4::perspective_rh(fov_y_radians, aspect_ratio, z_far, z_near)
    }

    /// Generates an infinite far-plane Reverse-Z perspective projection matrix.
    #[inline]
    pub fn perspective_infinite_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(fov_y_radians, aspect_ratio, z_near)
    }
}
