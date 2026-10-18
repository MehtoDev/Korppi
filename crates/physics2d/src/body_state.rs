//! The solver's view of an awake body during a step (Box2D `b2BodyState`).
//!
//! At the start of a step the world copies each awake body's velocities into a contiguous
//! array of [`BodyState2d`], in the awake set's order. Every solver stage works on that array:
//! velocity integration, the joints, the contacts four at a time, and position integration,
//! which accumulates the motion since the step began (`delta_position`, `delta_rotation`)
//! instead of moving the body. Constraints refer to their bodies by awake index plus one, zero
//! standing for a body that is not awake (a static one): it reads as [`BodyState2d::IDENTITY`]
//! and takes no velocity. Only dynamic bodies' states take solved velocities, so constraints of
//! one graph color, which share no dynamic body, may be solved in parallel. The step's end
//! writes the velocities back and moves each body by its accumulated motion.

use crate::body::{BodyType, RigidBody2d};
use korppi_math::{Rot2, Vec2};

/// A body as the solver sees it during a step (Box2D `b2BodyState`): its velocities and how far
/// it moved and turned since the step began. Bodies that are not awake are the identity state;
/// only dynamic bodies take the solved velocities.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(32))]
pub(crate) struct BodyState2d {
    pub linear_velocity: Vec2,
    pub angular_velocity: f32,
    pub dynamic: bool,
    pub delta_position: Vec2,
    pub delta_rotation: Rot2,
}

impl BodyState2d {
    /// Box2D `b2_identityBodyState`: still, unmoved, not dynamic.
    pub const IDENTITY: Self = Self {
        linear_velocity: Vec2::ZERO,
        angular_velocity: 0.0,
        dynamic: false,
        delta_position: Vec2::ZERO,
        delta_rotation: Rot2::IDENTITY,
    };

    /// `body` at the start of a step, unmoved.
    pub fn at_rest(body: &RigidBody2d) -> Self {
        Self {
            linear_velocity: body.linear_velocity,
            angular_velocity: body.angular_velocity,
            dynamic: body.body_type == BodyType::Dynamic,
            ..Self::IDENTITY
        }
    }

    /// Writes the solved velocities back to a dynamic body.
    pub fn store(&self, body: &mut RigidBody2d) {
        if self.dynamic {
            body.linear_velocity = self.linear_velocity;
            body.angular_velocity = self.angular_velocity;
        }
    }

    /// Box2D `b2MulAdd` and `+=` on the velocities, for a dynamic body.
    pub fn apply(&mut self, inv_mass: f32, inv_inertia: f32, impulse: Vec2, angular: f32) {
        if self.dynamic {
            self.linear_velocity = Vec2::new(
                self.linear_velocity.x + inv_mass * impulse.x,
                self.linear_velocity.y + inv_mass * impulse.y,
            );
            self.angular_velocity += inv_inertia * angular;
        }
    }
}
