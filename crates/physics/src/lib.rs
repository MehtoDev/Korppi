//! Robust, deterministic 3D physics engine adapting Erin Catto's Box3D into pure Rust.

pub mod body;
pub mod bvh;
pub mod collider;
pub mod compound;
pub mod constraint_graph;
pub mod contact;
mod contact_solver;
pub mod distance;
pub mod events;
pub mod height_field;
pub mod hull;
pub mod island;
pub mod joint;
mod manifold;
pub mod material;
pub mod math;
pub mod mesh;
pub mod mesh_contact;
pub mod mover;
pub mod narrowphase;
pub(crate) mod parallel;
mod quickhull;
pub(crate) mod serde_ids;
pub mod shape;
pub mod snapshot;
pub mod solver;
pub(crate) mod solver_wide;
mod triangle_manifold;
pub mod world;

pub use body::{BodyType, MotionLocks3d, RigidBody, CONTINUOUS_SAFETY_FACTOR, MAX_ROTATION};
pub use bvh::DynamicTree;
pub use collider::{Collider, Filter3d, QueryFilter3d};
pub use compound::{Compound3d, CompoundChild, CompoundRayHit};
pub use constraint_graph::{
    ConstraintGraph3d, GraphColor3d, DYNAMIC_COLOR_COUNT, GRAPH_COLOR_COUNT, OVERFLOW_INDEX,
};
pub use contact::{Contact3d, ContactManager3d};
pub use contact_solver::{Softness3d, LINEAR_SLOP, SPECULATIVE_DISTANCE};
pub use events::{
    BodyEvents3d, BodyMoveEvent3d, ContactBeginEvent3d, ContactEndEvent3d, ContactEvents3d,
    ContactHitEvent3d, JointEvent3d, JointEvents3d, SensorBeginEvent3d, SensorEndEvent3d,
    SensorEvents3d,
};
pub use height_field::{HeightField3d, HeightFieldOptions, HEIGHT_FIELD_HOLE};
pub use hull::Hull3d;
pub use island::{
    BodyLinks3d, IslandManager3d, SolverSet3d, AWAKE_SET, DISABLED_SET, FIRST_SLEEPING_SET,
    STATIC_SET, TIME_TO_SLEEP,
};
pub use joint::{
    get_swing_angle, DistanceJoint, FilterJoint, Frame3d, Joint, JointBase3d, MotorJoint,
    ParallelJoint, PrismaticJoint, RevoluteJoint, SphericalJoint, WeldJoint, WheelJoint,
};
pub use manifold::SatCache;
pub use material::{
    default_friction_callback, default_restitution_callback, FrictionCallback3d,
    RestitutionCallback3d, SurfaceMaterial3d,
};
pub use mesh::{Mesh3d, MeshEdge, MeshOptions, MeshRayHit, MeshTriangle};
pub use mesh_contact::MeshContactCache;
pub use narrowphase::{
    CollisionBody, ContactCache, ContactManifold, ContactPoint, ContactPoints, MAX_MANIFOLD_POINTS,
};
pub use shape::Shape;
pub use snapshot::{BodySnapshot3d, WorldSnapshot3d};
pub use solver::{
    solve_contacts, solve_contacts_with_buffers, SolverConfig3d, SolverScratch3d,
    CONTACT_RECYCLE_DISTANCE, MAX_LINEAR_SPEED,
};
pub use world::{
    BodyId, BodyId3d, ColliderId, ColliderId3d, Counters3d, ExplosionDef3d, JointId, JointId3d,
    Joints, PhysicsWorld, RayHit3d, ShapeCastHit3d,
};
