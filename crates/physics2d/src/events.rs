//! Events the world reports after each step (Box2D `b2ContactEvents`, `b2BodyEvents`), and
//! contact data for queries (Box2D `b2ContactData`).

use crate::narrowphase::ContactManifold2d;
use crate::world::{BodyId2d, ColliderId2d, JointId2d};
use korppi_math::{Transform2d, Vec2};
use serde::{Deserialize, Serialize};

/// Two colliders started touching (Box2D `b2ContactBeginTouchEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContactBeginEvent2d {
    /// Collider A of the contact.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_a: ColliderId2d,
    /// Collider B of the contact.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_b: ColliderId2d,
}

/// Two colliders stopped touching, or their contact was destroyed (Box2D
/// `b2ContactEndTouchEvent`). Either collider may have been destroyed since.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContactEndEvent2d {
    /// Collider A of the contact.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_a: ColliderId2d,
    /// Collider B of the contact.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_b: ColliderId2d,
}

/// Two colliders struck each other faster than the hit event threshold (Box2D
/// `b2ContactHitEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactHitEvent2d {
    /// Collider A of the contact.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_a: ColliderId2d,
    /// Collider B of the contact.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_b: ColliderId2d,
    /// Where they struck, midway between the surfaces; reconstructed after the step from a
    /// static body if one is involved, so a moving pair may trail the impact by up to a step.
    pub point: Vec2,
    /// Unit normal from collider A to collider B.
    pub normal: Vec2,
    /// Speed at which the colliders approached, always positive.
    pub approach_speed: f32,
}

/// The contact events of the last step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContactEvents2d {
    /// Contacts that started touching during the step.
    pub begin: Vec<ContactBeginEvent2d>,
    /// Contacts that stopped touching during the step, or were destroyed by calls made since
    /// the step before it.
    pub end: Vec<ContactEndEvent2d>,
    /// Contacts that struck during the step.
    pub hit: Vec<ContactHitEvent2d>,
}

impl ContactEvents2d {
    fn clear(&mut self) {
        self.begin.clear();
        self.end.clear();
        self.hit.clear();
    }

    /// Starts a step's events: clears them, and moves in the end events of the calls made
    /// since the last step.
    pub(crate) fn start_step(&mut self, pending_end: &mut Vec<ContactEndEvent2d>) {
        self.clear();
        std::mem::swap(&mut self.end, pending_end);
    }
}

/// A collider started overlapping a sensor (Box2D `b2SensorBeginTouchEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorBeginEvent2d {
    /// The sensor collider.
    #[serde(with = "crate::serde_ids::handle")]
    pub sensor: ColliderId2d,
    /// The collider that entered it.
    #[serde(with = "crate::serde_ids::handle")]
    pub visitor: ColliderId2d,
}

/// A collider stopped overlapping a sensor, or one of them was destroyed (Box2D
/// `b2SensorEndTouchEvent`). Either collider may have been destroyed since.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorEndEvent2d {
    /// The sensor collider.
    #[serde(with = "crate::serde_ids::handle")]
    pub sensor: ColliderId2d,
    /// The collider that left it.
    #[serde(with = "crate::serde_ids::handle")]
    pub visitor: ColliderId2d,
}

/// The sensor events of the last step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SensorEvents2d {
    /// Overlaps that began during the step.
    pub begin: Vec<SensorBeginEvent2d>,
    /// Overlaps that ended during the step, or because of calls made since the step before.
    pub end: Vec<SensorEndEvent2d>,
}

impl SensorEvents2d {
    /// Starts a step's events: clears them, and moves in the end events of the calls made
    /// since the last step.
    pub(crate) fn start_step(&mut self, pending_end: &mut Vec<SensorEndEvent2d>) {
        self.begin.clear();
        self.end.clear();
        std::mem::swap(&mut self.end, pending_end);
    }
}

/// A joint's force or torque reached its threshold during the step (Box2D `b2JointEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct JointEvent2d {
    /// The joint.
    #[serde(with = "crate::serde_ids::handle")]
    pub joint: JointId2d,
}

/// A body moved during the step (Box2D `b2BodyMoveEvent`). Every awake body reports one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyMoveEvent2d {
    /// The body.
    #[serde(with = "crate::serde_ids::handle")]
    pub body: BodyId2d,
    /// Its pose after the step.
    pub transform: Transform2d,
    /// Whether it fell asleep at the end of the step.
    pub fell_asleep: bool,
}

/// A touching contact (Box2D `b2ContactData`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactData2d {
    /// Collider A of the contact.
    pub collider_a: ColliderId2d,
    /// Collider B of the contact.
    pub collider_b: ColliderId2d,
    /// The contact manifold, its normal pointing from collider A to collider B.
    pub manifold: ContactManifold2d,
}
