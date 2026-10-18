//! 2D Collider attachment defining shape geometry and physical surface properties.

use crate::shape::Shape2d;
use korppi_ecs::Component;
use serde::{Deserialize, Serialize};

/// Which colliders collide with each other (Box2D `b2Filter`). Two colliders in the same
/// non-zero group always collide if the group is positive and never if it is negative;
/// otherwise each one's `mask_bits` must include the other's `category_bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Filter2d {
    /// The categories the collider belongs to, usually one bit.
    pub category_bits: u64,
    /// The categories the collider collides with.
    pub mask_bits: u64,
    /// Colliders sharing a non-zero group always (positive) or never (negative) collide,
    /// whatever their bits; ragdolls use a negative group to keep their parts apart.
    pub group_index: i32,
}

impl Default for Filter2d {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Filter2d {
    /// Category 1, colliding with everything, no group (Box2D `b2DefaultFilter`).
    pub const DEFAULT: Self = Self {
        category_bits: 1,
        mask_bits: u64::MAX,
        group_index: 0,
    };

    /// A filter with these bits and no group.
    #[must_use]
    pub const fn new(category_bits: u64, mask_bits: u64) -> Self {
        Self {
            category_bits,
            mask_bits,
            group_index: 0,
        }
    }

    /// Sets the group index in builder style.
    #[must_use]
    pub const fn with_group(mut self, group_index: i32) -> Self {
        self.group_index = group_index;
        self
    }

    /// Whether colliders with these filters collide (Box2D `b2ShouldShapesCollide`).
    #[must_use]
    pub const fn should_collide(&self, other: &Self) -> bool {
        if self.group_index == other.group_index && self.group_index != 0 {
            return self.group_index > 0;
        }
        (self.mask_bits & other.category_bits) != 0 && (self.category_bits & other.mask_bits) != 0
    }
}

/// Defines 2D collision geometry, friction, and restitution attached to a rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Collider2d {
    /// The geometric collision primitive.
    pub shape: Shape2d,
    /// Density in kg/m² (default 1, as Box2D). Attaching the collider to a dynamic body sets the
    /// body's mass, centre of mass and rotational inertia from it; `0` gives no mass.
    pub density: f32,
    /// Friction coefficient (typically 0.0 to 1.0).
    pub friction: f32,
    /// Restitution (bounciness) coefficient (0.0 for inelastic, 1.0 for perfectly elastic).
    pub restitution: f32,
    /// If `true`, the collider makes no contacts and has no continuous collision; with
    /// `enable_sensor_events` it reports the colliders that overlap it (Box2D `isSensor`). It
    /// still adds its density to the body's mass.
    pub is_sensor: bool,
    /// Rolling resistance, usually 0 to 1 (Box2D `rollingResistance`, default 0). Scaled by
    /// the larger shape radius of a contact, it resists the relative spin of round shapes.
    #[serde(default)]
    pub rolling_resistance: f32,
    /// Surface speed along the contact tangent, for conveyor belts, in m/s (Box2D
    /// `tangentSpeed`, default 0). A contact moves its colliders at the sum of theirs.
    #[serde(default)]
    pub tangent_speed: f32,
    /// An id the world's friction and restitution mixers receive (Box2D `userMaterialId`); the
    /// world does not use it otherwise.
    #[serde(default)]
    pub user_material_id: u64,
    /// Which colliders it collides with.
    #[serde(default)]
    pub filter: Filter2d,
    /// Ask the world's custom filter whether to collide with another collider (Box2D
    /// `enableCustomFiltering`, off by default). One collider of a pair suffices.
    #[serde(default)]
    pub enable_custom_filtering: bool,
    /// Report begin and end touch events for this collider's contacts (Box2D
    /// `enableContactEvents`, off by default). One collider of a contact suffices.
    #[serde(default)]
    pub enable_contact_events: bool,
    /// Report hit events for this collider's contacts (Box2D `enableHitEvents`, off by
    /// default). One collider of a contact suffices.
    #[serde(default)]
    pub enable_hit_events: bool,
    /// Take part in sensor events (Box2D `enableSensorEvents`, off by default, even for
    /// sensors): a sensor reports a collider only if both enable them.
    #[serde(default)]
    pub enable_sensor_events: bool,
    /// Pass this collider's new manifolds to the world's pre-solve callback (Box2D
    /// `enablePreSolveEvents`, off by default). One collider of a contact suffices.
    #[serde(default)]
    pub enable_pre_solve_events: bool,
}

impl Component for Collider2d {}

impl Default for Collider2d {
    fn default() -> Self {
        Self {
            shape: Shape2d::Circle(crate::shape::Circle2d::new(korppi_math::Vec2::ZERO, 0.5)),
            density: 1.0,
            friction: 0.6,
            restitution: 0.0,
            is_sensor: false,
            rolling_resistance: 0.0,
            tangent_speed: 0.0,
            user_material_id: 0,
            filter: Filter2d::DEFAULT,
            enable_custom_filtering: false,
            enable_contact_events: false,
            enable_hit_events: false,
            enable_sensor_events: false,
            enable_pre_solve_events: false,
        }
    }
}

/// A collider's surface (Box2D `b2SurfaceMaterial`): what contacts mix from it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceMaterial2d {
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution coefficient.
    pub restitution: f32,
    /// Rolling resistance.
    pub rolling_resistance: f32,
    /// Surface speed along the contact tangent.
    pub tangent_speed: f32,
    /// An id the world's material mixers receive.
    pub user_material_id: u64,
}

impl Default for SurfaceMaterial2d {
    /// A new collider's surface.
    fn default() -> Self {
        Collider2d::default().material()
    }
}

impl Collider2d {
    /// The collider's surface.
    #[must_use]
    pub const fn material(&self) -> SurfaceMaterial2d {
        SurfaceMaterial2d {
            friction: self.friction,
            restitution: self.restitution,
            rolling_resistance: self.rolling_resistance,
            tangent_speed: self.tangent_speed,
            user_material_id: self.user_material_id,
        }
    }

    /// Sets the collider's surface.
    #[must_use]
    pub const fn with_material(mut self, material: SurfaceMaterial2d) -> Self {
        self.friction = material.friction;
        self.restitution = material.restitution;
        self.rolling_resistance = material.rolling_resistance;
        self.tangent_speed = material.tangent_speed;
        self.user_material_id = material.user_material_id;
        self
    }

    /// Creates a new collider with the given shape geometry.
    #[must_use]
    pub fn new(shape: Shape2d) -> Self {
        Self {
            shape,
            ..Default::default()
        }
    }

    /// Sets the density in kg/m² (at least `0`).
    #[must_use]
    pub const fn with_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    /// Sets the friction coefficient.
    #[must_use]
    pub const fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    /// Sets the restitution (bounciness) coefficient.
    #[must_use]
    pub const fn with_restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    /// Sets the rolling resistance.
    #[must_use]
    pub const fn with_rolling_resistance(mut self, rolling_resistance: f32) -> Self {
        self.rolling_resistance = rolling_resistance;
        self
    }

    /// Sets the tangent speed, for conveyor belts.
    #[must_use]
    pub const fn with_tangent_speed(mut self, tangent_speed: f32) -> Self {
        self.tangent_speed = tangent_speed;
        self
    }

    /// Sets the user material id.
    #[must_use]
    pub const fn with_user_material_id(mut self, id: u64) -> Self {
        self.user_material_id = id;
        self
    }

    /// Sets whether this collider acts as a sensor (trigger).
    #[must_use]
    pub const fn with_sensor(mut self, is_sensor: bool) -> Self {
        self.is_sensor = is_sensor;
        self
    }

    /// Sets whether this collider's contacts report begin and end touch events.
    #[must_use]
    pub const fn with_contact_events(mut self, enable: bool) -> Self {
        self.enable_contact_events = enable;
        self
    }

    /// Sets whether this collider's contacts report hit events.
    #[must_use]
    pub const fn with_hit_events(mut self, enable: bool) -> Self {
        self.enable_hit_events = enable;
        self
    }

    /// Sets whether this collider takes part in sensor events, as a sensor or as a visitor.
    #[must_use]
    pub const fn with_sensor_events(mut self, enable: bool) -> Self {
        self.enable_sensor_events = enable;
        self
    }

    /// Sets whether this collider's new manifolds go to the world's pre-solve callback.
    #[must_use]
    pub const fn with_pre_solve_events(mut self, enable: bool) -> Self {
        self.enable_pre_solve_events = enable;
        self
    }

    /// Sets the collision filter.
    #[must_use]
    pub const fn with_filter(mut self, filter: Filter2d) -> Self {
        self.filter = filter;
        self
    }

    /// Sets whether the world's custom filter decides this collider's pairs.
    #[must_use]
    pub const fn with_custom_filtering(mut self, enable: bool) -> Self {
        self.enable_custom_filtering = enable;
        self
    }
}
