//! Surface materials, conveyor belt velocities, and friction/restitution callbacks.

use korppi_math::Vec3;
use serde::{Deserialize, Serialize};

/// Surface material properties for shapes, per-triangle mesh materials, and height fields (Box3D `b3SurfaceMaterial`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMaterial3d {
    /// Coulomb friction coefficient, usually [0, 1].
    pub friction: f32,
    /// Restitution (bounce) coefficient, usually [0, 1].
    pub restitution: f32,
    /// Rolling resistance coefficient, usually [0, 1].
    pub rolling_resistance: f32,
    /// Local tangent velocity for conveyor belts, projected onto the contact plane.
    pub tangent_velocity: Vec3,
    /// User material identifier passed to callbacks and query hits.
    pub user_material_id: u64,
    /// Custom debug draw color (low 24 bits RGB, high 8 bits debug preset).
    pub custom_color: u32,
}

impl Default for SurfaceMaterial3d {
    fn default() -> Self {
        Self {
            friction: 0.6,
            restitution: 0.0,
            rolling_resistance: 0.0,
            tangent_velocity: Vec3::ZERO,
            user_material_id: 0,
            custom_color: 0,
        }
    }
}

impl SurfaceMaterial3d {
    /// Constructs a surface material with default values (friction 0.6, no bounce, no conveyor).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            friction: 0.6,
            restitution: 0.0,
            rolling_resistance: 0.0,
            tangent_velocity: Vec3::ZERO,
            user_material_id: 0,
            custom_color: 0,
        }
    }

    /// Sets the Coulomb friction coefficient.
    #[must_use]
    pub const fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    /// Sets the coefficient of restitution.
    #[must_use]
    pub const fn with_restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    /// Sets the rolling resistance coefficient.
    #[must_use]
    pub const fn with_rolling_resistance(mut self, rolling_resistance: f32) -> Self {
        self.rolling_resistance = rolling_resistance;
        self
    }

    /// Sets the surface tangent velocity (conveyor belt).
    #[must_use]
    pub const fn with_tangent_velocity(mut self, tangent_velocity: Vec3) -> Self {
        self.tangent_velocity = tangent_velocity;
        self
    }

    /// Sets the user material identifier.
    #[must_use]
    pub const fn with_user_material_id(mut self, user_material_id: u64) -> Self {
        self.user_material_id = user_material_id;
        self
    }

    /// Sets the custom debug draw color.
    #[must_use]
    pub const fn with_custom_color(mut self, custom_color: u32) -> Self {
        self.custom_color = custom_color;
        self
    }
}

/// Callback for combining friction of two colliding surfaces (Box3D `b3FrictionCallback`).
pub type FrictionCallback3d =
    fn(friction_a: f32, material_id_a: u64, friction_b: f32, material_id_b: u64) -> f32;

/// Default friction combining function: geometric mean `sqrt(fA * fB)` (Box3D `b3DefaultFrictionCallback`).
pub fn default_friction_callback(f_a: f32, _: u64, f_b: f32, _: u64) -> f32 {
    (f_a * f_b).sqrt()
}

/// Callback for combining restitution of two colliding surfaces (Box3D `b3RestitutionCallback`).
pub type RestitutionCallback3d =
    fn(restitution_a: f32, material_id_a: u64, restitution_b: f32, material_id_b: u64) -> f32;

/// Default restitution combining function: maximum `max(rA, rB)` (Box3D `b3DefaultRestitutionCallback`).
pub fn default_restitution_callback(r_a: f32, _: u64, r_b: f32, _: u64) -> f32 {
    r_a.max(r_b)
}

/// Returns the default friction callback function pointer.
#[must_use]
pub const fn default_friction_callback_fn() -> FrictionCallback3d {
    default_friction_callback
}

/// Returns the default restitution callback function pointer.
#[must_use]
pub const fn default_restitution_callback_fn() -> RestitutionCallback3d {
    default_restitution_callback
}
