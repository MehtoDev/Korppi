//! Snapshots of a whole world (Box2D v3.2 `b2World_GetSnapshot`, `b2World_Restore`,
//! `b2CreateWorldFromSnapshot`).
//!
//! A [`WorldSnapshot2d`] holds everything the simulation continues from: bodies, colliders,
//! chains and joints with their accumulated impulses, the broadphase trees, contacts with their
//! manifolds and warm-starting impulses, islands, solver sets, sleep timers, pending events
//! and the last step's events. Restoring it with
//! [`PhysicsWorld2d::restore_snapshot`](crate::world::PhysicsWorld2d::restore_snapshot)
//! continues bit for bit as the world did from the snapshot, and handles held at the snapshot
//! stay valid. Host wiring (the pre-solve, pre-continuous and custom filter callbacks and the
//! material mixers) is not part of it: an in-place restore keeps the world's,
//! [`PhysicsWorld2d::from_snapshot`](crate::world::PhysicsWorld2d::from_snapshot) starts
//! without. A snapshot is serde-serializable, for save games.

use crate::world::PhysicsWorld2d;
use serde::{Deserialize, Serialize};

/// The whole simulation state of a world at a step boundary.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorldSnapshot2d {
    /// The world's state, without host wiring and per-step scratch.
    pub(crate) world: PhysicsWorld2d,
}

impl Clone for WorldSnapshot2d {
    fn clone(&self) -> Self {
        Self {
            world: self.world.clone_state(),
        }
    }
}

impl WorldSnapshot2d {
    /// The step count of the world when the snapshot was taken.
    #[must_use]
    pub const fn step_index(&self) -> u64 {
        self.world.step_index()
    }

    /// The state hash of the world when the snapshot was taken (see
    /// [`PhysicsWorld2d::state_hash`]).
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        self.world.state_hash()
    }
}
