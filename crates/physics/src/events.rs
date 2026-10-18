//! Events the world reports after each step (Box3D `b3SensorEvents`, `b3ContactEvents`,
//! `b3BodyEvents`, `b3JointEvents`).

use crate::world::{BodyId3d, ColliderId3d, JointId3d};
use korppi_math::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// A collider started overlapping a sensor (Box3D `b3SensorBeginTouchEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorBeginEvent3d {
    /// The collider of the sensor.
    #[serde(with = "crate::serde_ids::handle")]
    pub sensor: ColliderId3d,
    /// The collider that entered it.
    #[serde(with = "crate::serde_ids::handle")]
    pub visitor: ColliderId3d,
}

/// A collider stopped overlapping a sensor (Box3D `b3SensorEndTouchEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorEndEvent3d {
    /// The collider of the sensor.
    #[serde(with = "crate::serde_ids::handle")]
    pub sensor: ColliderId3d,
    /// The collider that left it.
    #[serde(with = "crate::serde_ids::handle")]
    pub visitor: ColliderId3d,
}

/// The sensor events of the last step (Box3D `b3SensorEvents`), ordered by sensor and then by
/// visitor collider.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorEvents3d {
    /// Overlaps that began during the step.
    pub begin: Vec<SensorBeginEvent3d>,
    /// Overlaps that ended during the step.
    pub end: Vec<SensorEndEvent3d>,
}

impl SensorEvents3d {
    /// Clears both lists, keeping their capacity.
    pub(crate) fn clear(&mut self) {
        self.begin.clear();
        self.end.clear();
    }
}

/// Two colliders began touching (Box3D `b3ContactBeginTouchEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContactBeginEvent3d {
    /// Collider A.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_a: ColliderId3d,
    /// Collider B.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_b: ColliderId3d,
    /// Body A.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_a: BodyId3d,
    /// Body B.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_b: BodyId3d,
}

/// Two colliders stopped touching (Box3D `b3ContactEndTouchEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContactEndEvent3d {
    /// Collider A.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_a: ColliderId3d,
    /// Collider B.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_b: ColliderId3d,
    /// Body A.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_a: BodyId3d,
    /// Body B.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_b: BodyId3d,
}

/// Two colliders collided with approach speed exceeding the hit threshold (Box3D `b3ContactHitEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactHitEvent3d {
    /// Collider A.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_a: ColliderId3d,
    /// Collider B.
    #[serde(with = "crate::serde_ids::handle")]
    pub collider_b: ColliderId3d,
    /// Body A.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_a: BodyId3d,
    /// Body B.
    #[serde(with = "crate::serde_ids::handle")]
    pub body_b: BodyId3d,
    /// Impact contact point in world coordinates.
    pub point: Vec3,
    /// Contact normal pointing from collider A to collider B.
    pub normal: Vec3,
    /// Speed at which the bodies approached along the contact normal (m/s).
    pub approach_speed: f32,
}

/// Contact events of the last step (Box3D `b3ContactEvents`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContactEvents3d {
    /// Contacts that began touching this step.
    pub begin: Vec<ContactBeginEvent3d>,
    /// Contacts that stopped touching this step.
    pub end: Vec<ContactEndEvent3d>,
    /// High-velocity impacts meeting the hit threshold.
    pub hit: Vec<ContactHitEvent3d>,
}

impl ContactEvents3d {
    /// Creates event lists preallocated with `capacity`.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            begin: Vec::with_capacity(capacity),
            end: Vec::with_capacity(capacity),
            hit: Vec::with_capacity(capacity),
        }
    }

    /// Clears all event lists, keeping their allocated capacity.
    pub(crate) fn clear(&mut self) {
        self.begin.clear();
        self.end.clear();
        self.hit.clear();
    }
}

/// A rigid body moved during simulation (Box3D `b3BodyMoveEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyMoveEvent3d {
    /// Rigid body identifier.
    #[serde(with = "crate::serde_ids::handle")]
    pub body: BodyId3d,
    /// Body position in world space.
    pub position: Vec3,
    /// Body rotation in world space.
    pub rotation: Quat,
    /// Whether the body fell asleep this time step.
    pub fell_asleep: bool,
}

/// Body movement events of the last step (Box3D `b3BodyEvents`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyEvents3d {
    /// Move events for bodies that moved or fell asleep.
    pub move_events: Vec<BodyMoveEvent3d>,
}

impl BodyEvents3d {
    /// Creates body move events preallocated with `capacity`.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            move_events: Vec::with_capacity(capacity),
        }
    }

    /// Clears move events list, keeping allocated capacity.
    pub(crate) fn clear(&mut self) {
        self.move_events.clear();
    }
}

/// A joint exceeded its force or torque threshold during the step (Box3D `b3JointEvent`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointEvent3d {
    /// Joint identifier.
    #[serde(with = "crate::serde_ids::handle")]
    pub joint: JointId3d,
    /// Observed linear reaction force magnitude in Newtons.
    pub linear_force: f32,
    /// Observed angular reaction torque magnitude in Newton-meters.
    pub angular_torque: f32,
}

/// Joint events of the last step (Box3D `b3JointEvents`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JointEvents3d {
    /// Triggered joint events.
    pub events: Vec<JointEvent3d>,
}

impl JointEvents3d {
    /// Creates joint events preallocated with `capacity`.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
        }
    }

    /// Clears joint events list, keeping allocated capacity.
    pub(crate) fn clear(&mut self) {
        self.events.clear();
    }
}
