//! Robust, deterministic 2D physics engine adapting Erin Catto's Box2D into pure Rust.

pub mod body;
mod body_state;
pub mod bvh;
pub mod chain;
pub mod collider;
pub mod config;
pub mod constraint_graph;
pub mod contact_manager;
pub mod diagnostics;
pub mod distance;
pub mod events;
mod geometry;
pub mod hull;
pub mod joint;
pub mod manifold;
mod math;
pub mod mover;
pub mod narrowphase;
mod parallel;
mod serde_ids;
pub mod shape;
pub mod snapshot;
pub mod solver;
mod solver_wide;
pub mod world;

pub use body::{BodyType, MotionLocks2d, RigidBody2d, MAX_LINEAR_SPEED, MAX_ROTATION};
pub use bvh::DynamicTree2d;
pub use chain::{Chain2d, ChainDef2d};
pub use collider::{Collider2d, Filter2d, SurfaceMaterial2d};
pub use config::WorldConfig2d;
pub use constraint_graph::{ConstraintGraph2d, GraphColor2d};
pub use contact_manager::{Contact2d, ContactManager2d};
pub use diagnostics::{Counters2d, Profile2d};
pub use distance::{
    segment_distance, shape_cast, shape_distance, time_of_impact, CastOutput2d, DistanceOutput2d,
    RayCastInput2d, ShapeCastInput2d, ShapeCastPairInput2d, ShapeProxy2d, SimplexCache2d, Sweep2d,
    ToiInput2d, ToiOutput2d, ToiState2d,
};
pub use events::{
    BodyMoveEvent2d, ContactBeginEvent2d, ContactData2d, ContactEndEvent2d, ContactEvents2d,
    ContactHitEvent2d, JointEvent2d, SensorBeginEvent2d, SensorEndEvent2d, SensorEvents2d,
};
pub use hull::Hull2d;
pub use joint::{
    DistanceJoint2d, Joint2d, MotorJoint2d, MoverJoint2d, PogoJoint2d, PrismaticJoint2d,
    RevoluteJoint2d, Softness2d, WeldJoint2d, WheelJoint2d,
};
pub use mover::{
    clip_vector, solve_planes, CollisionPlane2d, Plane2d, PlaneResult2d, PlaneSolverResult2d,
};
pub use narrowphase::{collide_shapes, ContactManifold2d, ManifoldPoint2d, TimeOfImpact2d};
pub use shape::{
    Capsule2d, ChainSegment2d, Circle2d, MassData2d, Polygon2d, Segment2d, Shape2d,
    MAX_POLYGON_VERTICES,
};
pub use snapshot::WorldSnapshot2d;
pub use solver::{
    solve_contacts_2d, ContactMaterial2d, SolverConfig2d, CONTACT_PUSH_SPEED,
    RESTITUTION_ITERATIONS, RESTITUTION_THRESHOLD,
};
pub use world::{
    BodyId2d, ChainId2d, ColliderId2d, ExplosionDef2d, JointId2d, JointInstance2d, MaterialMixer2d,
    PhysicsWorld2d, QueryFilter2d, RayHit2d,
};
