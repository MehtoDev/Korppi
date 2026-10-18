//! What a world holds and where its last step spent its time (Box2D `b2Counters`,
//! `b2Profile`).

/// Object counts of a world (Box2D `b2World_GetCounters`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters2d {
    /// Live bodies.
    pub body_count: usize,
    /// Bodies in the awake set: dynamic and kinematic bodies that are neither asleep nor
    /// disabled (Box2D `b2World_GetAwakeBodyCount`).
    pub awake_body_count: usize,
    /// Live islands, awake or asleep.
    pub island_count: usize,
    /// Contacts the collide pass updates: those with an awake body, touching or not.
    pub awake_contact_count: usize,
    /// Contacts the last step recycled instead of colliding again (Box2D
    /// `recycledContactCount`).
    pub recycled_contact_count: usize,
    /// Live colliders.
    pub collider_count: usize,
    /// Contacts: collider pairs whose fat boxes overlap, touching or not.
    pub contact_count: usize,
    /// Live joints.
    pub joint_count: usize,
    /// Height of the dynamic or kinematic body tree, whichever is taller.
    pub tree_height: i32,
    /// Height of the static body tree.
    pub static_tree_height: i32,
    /// Colors of the constraint graph, the overflow included.
    pub color_count: usize,
    /// Contacts and joints in each graph color, the overflow last (Box2D `colorCounts`).
    pub color_counts: [usize; crate::constraint_graph::GRAPH_COLOR_COUNT],
}

/// Time the last step spent in each stage, in milliseconds (Box2D `b2World_GetProfile`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Profile2d {
    /// The whole step.
    pub step: f32,
    /// Moving proxies and finding pairs.
    pub broadphase: f32,
    /// Computing manifolds and updating contacts.
    pub narrowphase: f32,
    /// Preparing and solving the constraints over the substeps.
    pub solve: f32,
    /// Continuous collision of fast bodies.
    pub continuous: f32,
    /// Islands and sleep.
    pub islands: f32,
    /// Sensor overlaps.
    pub sensors: f32,
}
