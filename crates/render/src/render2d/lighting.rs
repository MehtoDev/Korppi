//! 2D lighting and dynamic shadow fin extrusion.

use korppi_math::Vec2;

/// 2D point light source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light2d {
    /// 2D position in canvas space.
    pub position: Vec2,
    /// Light attenuation radius.
    pub radius: f32,
    /// RGBA light tint color.
    pub color: [f32; 4],
    /// Radiance energy multiplier.
    pub energy: f32,
}

impl Default for Light2d {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            radius: 200.0,
            color: [1.0, 1.0, 1.0, 1.0],
            energy: 1.0,
        }
    }
}

/// 2D line segment shadow occluder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occluder2d {
    /// First vertex of occluder edge.
    pub p0: Vec2,
    /// Second vertex of occluder edge.
    pub p1: Vec2,
}

impl Occluder2d {
    /// Extrudes the occluder edge away from `light` to form a 4-vertex shadow fin quad.
    ///
    /// Produces $[p_0, p_1, p_1', p_0']$ where extruded vertices extend to `light.radius`.
    pub fn extrude_shadow(&self, light: &Light2d) -> [Vec2; 4] {
        let dir0 = (self.p0 - light.position).normalize_or_zero();
        let dir1 = (self.p1 - light.position).normalize_or_zero();

        let ext0 = light.position + dir0 * light.radius;
        let ext1 = light.position + dir1 * light.radius;

        [self.p0, self.p1, ext1, ext0]
    }
}
