//! Chain definitions (Box2D `b2ChainDef`): a polyline that a world turns into one-sided chain
//! segments with ghost vertices, one collider each.

use crate::collider::{Filter2d, SurfaceMaterial2d};
use crate::shape::ChainSegment2d;
use crate::world::{BodyId2d, ColliderId2d};
use korppi_math::Vec2;
use serde::{Deserialize, Serialize};

/// A chain to create on a body with
/// [`PhysicsWorld2d::create_chain`](crate::world::PhysicsWorld2d::create_chain) (Box2D
/// `b2ChainDef`).
///
/// Each consecutive pair of points makes a segment; a loop also joins the last point to the
/// first. A segment collides on its right, so a loop wound counter-clockwise collides outside
/// and one wound clockwise inside, and ground runs right to left. An open chain's end segments
/// take the given ghost vertices for their outer neighbours.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainDef2d {
    /// The points, at least 2 for an open chain and 3 for a loop.
    pub points: Vec<Vec2>,
    /// Whether the last point joins the first.
    pub is_loop: bool,
    /// The vertex before the first point of an open chain.
    pub ghost1: Vec2,
    /// The vertex after the last point of an open chain.
    pub ghost2: Vec2,
    /// One material for every segment, or one per segment.
    pub materials: Vec<SurfaceMaterial2d>,
    /// Which colliders the segments collide with.
    pub filter: Filter2d,
    /// Whether sensors detect the segments (Box2D `enableSensorEvents`, off by default).
    pub enable_sensor_events: bool,
}

impl ChainDef2d {
    /// An open chain through `points`, between ghost vertices `ghost1` and `ghost2`.
    #[must_use]
    pub fn open(points: Vec<Vec2>, ghost1: Vec2, ghost2: Vec2) -> Self {
        Self {
            points,
            is_loop: false,
            ghost1,
            ghost2,
            materials: vec![SurfaceMaterial2d::default()],
            filter: Filter2d::DEFAULT,
            enable_sensor_events: false,
        }
    }

    /// A closed chain through `points`.
    #[must_use]
    pub fn looped(points: Vec<Vec2>) -> Self {
        Self {
            is_loop: true,
            ..Self::open(points, Vec2::ZERO, Vec2::ZERO)
        }
    }

    /// Sets one material for every segment.
    #[must_use]
    pub fn with_material(mut self, material: SurfaceMaterial2d) -> Self {
        self.materials = vec![material];
        self
    }

    /// Sets a material per segment.
    #[must_use]
    pub fn with_segment_materials(mut self, materials: Vec<SurfaceMaterial2d>) -> Self {
        self.materials = materials;
        self
    }

    /// Sets the collision filter.
    #[must_use]
    pub const fn with_filter(mut self, filter: Filter2d) -> Self {
        self.filter = filter;
        self
    }

    /// Sets whether sensors detect the segments.
    #[must_use]
    pub const fn with_sensor_events(mut self, enable: bool) -> Self {
        self.enable_sensor_events = enable;
        self
    }

    /// Number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        if self.is_loop {
            self.points.len()
        } else {
            self.points.len().saturating_sub(1)
        }
    }

    /// The segments with their ghost vertices, as Box2D's `b2CreateChain` builds them.
    pub fn segments(&self) -> impl Iterator<Item = ChainSegment2d> + '_ {
        let points = &self.points;
        let n = self.segment_count();
        (0..n).map(move |i| {
            if self.is_loop {
                let count = points.len();
                let previous = if i == 0 { n - 1 } else { i - 1 };
                ChainSegment2d::new(
                    points[previous],
                    points[i],
                    points[(i + 1) % count],
                    points[(i + 2) % count],
                )
            } else {
                ChainSegment2d::new(
                    if i == 0 { self.ghost1 } else { points[i - 1] },
                    points[i],
                    points[i + 1],
                    if i == n - 1 {
                        self.ghost2
                    } else {
                        points[i + 2]
                    },
                )
            }
        })
    }

    /// The material of segment `i`.
    #[must_use]
    pub fn material(&self, i: usize) -> SurfaceMaterial2d {
        if self.materials.len() == 1 {
            self.materials[0]
        } else {
            self.materials[i]
        }
    }
}

/// A chain a world made from a [`ChainDef2d`] (Box2D `b2ChainShape`): one chain-segment
/// collider per segment, in the definition's order. The segments are ordinary colliders, so
/// [`PhysicsWorld2d::collider_mut`](crate::world::PhysicsWorld2d::collider_mut) changes a
/// segment's material; they are destroyed with the chain, not one by one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chain2d {
    #[serde(with = "crate::serde_ids::handle")]
    pub(crate) body: BodyId2d,
    #[serde(with = "crate::serde_ids::handle_vec")]
    pub(crate) segments: Vec<ColliderId2d>,
    pub(crate) is_loop: bool,
}

impl Chain2d {
    /// The body the chain is attached to.
    #[must_use]
    pub const fn body(&self) -> BodyId2d {
        self.body
    }

    /// The segment colliders (Box2D `b2Chain_GetSegments`).
    #[must_use]
    pub fn segments(&self) -> &[ColliderId2d] {
        &self.segments
    }

    /// Number of segments (Box2D `b2Chain_GetSegmentCount`).
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Whether the last point joins the first.
    #[must_use]
    pub const fn is_loop(&self) -> bool {
        self.is_loop
    }
}
