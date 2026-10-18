//! 3D joints, ported from Box3D (`joint.c` and `*_joint.c`).
//!
//! A joint connects two bodies through a local frame on each ([`JointBase3d`]): a revolute
//! hinge turns about frame z, a prismatic slider and a wheel's suspension move along frame A's
//! x axis, a wheel spins about frame B's z axis, and a spherical joint's cone is about frame A's
//! z axis and its twist about frame B's. Each step the solver prepares every active joint (its
//! frames in world space relative to the centres of mass, effective masses and softness), then
//! each substep warm starts it, solves it with position correction, and after the positions
//! are integrated relaxes it without correction. Rigid rows are soft constraints
//! (`constraint_hertz`, default 60 Hz, clamped to a quarter of the substep rate, damping ratio
//! 2); springs, limits and motors are Box3D's, with accumulated impulses that carry over between
//! steps.

use crate::body::{BodyType, RigidBody};
use crate::contact_solver::{normalize, BodyMotion3d, Softness3d, LINEAR_SLOP};
use crate::math;
use crate::world::BodyId3d;
use korppi_math::{Mat3, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Box3D's default stiffness of a joint's rigid rows, in Hz (`constraintHertz`).
pub const DEFAULT_CONSTRAINT_HERTZ: f32 = 60.0;

/// Box3D's default damping ratio of a joint's rigid rows (`constraintDampingRatio`).
pub const DEFAULT_CONSTRAINT_DAMPING_RATIO: f32 = 2.0;

/// The largest angle limit, in radians: limits are clamped to ±0.99 π, as in Box3D.
pub const MAX_JOINT_ANGLE: f32 = 0.99 * std::f32::consts::PI;

/// The longest distance-joint length, in metres (Box3D `B3_HUGE`).
pub const HUGE_LENGTH: f32 = 100_000.0;

/// Box3D's smallest determinant treated as invertible, `1000 * FLT_MIN`.
const MIN_DETERMINANT: f32 = 1000.0 * f32::MIN_POSITIVE;

/// A position and rotation (Box3D `b3Transform`): a joint's frame on a body, in the body's
/// coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Frame3d {
    /// The frame's origin.
    pub position: Vec3,
    /// The frame's rotation.
    pub rotation: Quat,
}

impl Default for Frame3d {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Frame3d {
    /// The body's own frame.
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// A frame at `position` turned by `rotation`.
    #[must_use]
    pub const fn new(position: Vec3, rotation: Quat) -> Self {
        Self { position, rotation }
    }

    /// A frame at `position`, aligned with the body.
    #[must_use]
    pub const fn at(position: Vec3) -> Self {
        Self::new(position, Quat::IDENTITY)
    }

    /// The frame on `body` that coincides with the world-space frame `world`, for joining
    /// bodies at their current poses.
    #[must_use]
    pub fn from_world(body: &RigidBody, world: Frame3d) -> Self {
        let inverse = body.rotation.conjugate();
        Self::new(
            inverse * (world.position - body.position),
            inverse * world.rotation,
        )
    }
}

/// What every joint has (Box3D `b3JointDef`): the bodies, a frame on each, the softness of the
/// rigid rows, and whether the bodies collide.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointBase3d {
    /// Index of body A.
    pub body_a: u32,
    /// Index of body B.
    pub body_b: u32,
    /// The joint frame on body A, in A's coordinates.
    pub local_frame_a: Frame3d,
    /// The joint frame on body B, in B's coordinates.
    pub local_frame_b: Frame3d,
    /// Stiffness of the rigid rows in Hz (Box3D `constraintHertz`, default 60), clamped to a
    /// quarter of the substep rate.
    pub constraint_hertz: f32,
    /// Damping ratio of the rigid rows (Box3D `constraintDampingRatio`, default 2).
    pub constraint_damping_ratio: f32,
    /// Whether the two bodies collide with each other (Box3D `collideConnected`, off by
    /// default).
    pub collide_connected: bool,
    /// Force threshold in Newtons for triggering joint events (Box3D `forceThreshold`, default `f32::MAX`).
    #[serde(default = "default_threshold")]
    pub force_threshold: f32,
    /// Torque threshold in N·m for triggering joint events (Box3D `torqueThreshold`, default `f32::MAX`).
    #[serde(default = "default_threshold")]
    pub torque_threshold: f32,
    #[serde(skip)]
    step: JointStep,
}

fn default_threshold() -> f32 {
    f32::MAX
}

impl JointBase3d {
    /// A joint between bodies `body_a` and `body_b` at the given local frames.
    #[must_use]
    pub fn new(
        body_a: impl Into<BodyId3d>,
        body_b: impl Into<BodyId3d>,
        local_frame_a: Frame3d,
        local_frame_b: Frame3d,
    ) -> Self {
        Self {
            body_a: body_a.into().index(),
            body_b: body_b.into().index(),
            local_frame_a,
            local_frame_b,
            constraint_hertz: DEFAULT_CONSTRAINT_HERTZ,
            constraint_damping_ratio: DEFAULT_CONSTRAINT_DAMPING_RATIO,
            collide_connected: false,
            force_threshold: f32::MAX,
            torque_threshold: f32::MAX,
            step: JointStep::default(),
        }
    }

    /// The joint frames in world space for bodies at their current poses.
    fn world_frames(&self, a: &RigidBody, b: &RigidBody) -> (Frame3d, Frame3d) {
        (
            Frame3d::new(
                a.position + a.rotation * self.local_frame_a.position,
                a.rotation * self.local_frame_a.rotation,
            ),
            Frame3d::new(
                b.position + b.rotation * self.local_frame_b.position,
                b.rotation * self.local_frame_b.rotation,
            ),
        )
    }
}

/// What a joint prepares at the start of a step (Box3D `b3JointSim`'s solver fields): masses
/// and world inverse inertias, the frames with world rotations and positions relative to the
/// centres of mass (the local frames are on the body origins), the separation of the centres,
/// and the rigid rows' softness.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct JointStep {
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_i_a: Mat3,
    inv_i_b: Mat3,
    frame_a: Frame3d,
    frame_b: Frame3d,
    delta_center: Vec3,
    softness: Softness3d,
    fixed_rotation: bool,
}

impl JointStep {
    fn masses(&self) -> (f32, f32) {
        (self.inv_mass_a, self.inv_mass_b)
    }

    fn inv_inertia_sum(&self) -> Mat3 {
        self.inv_i_a + self.inv_i_b
    }

    /// The bias, mass scale and impulse scale of a rigid row with position error `c`: the
    /// constraint softness in the biased solve, a plain rigid row without correction in the
    /// relax pass (Box3D's `useBias`).
    fn rigid(&self, c: f32, use_bias: bool) -> (f32, f32, f32) {
        if use_bias {
            (
                self.softness.bias_rate * c,
                self.softness.mass_scale,
                self.softness.impulse_scale,
            )
        } else {
            (0.0, 1.0, 0.0)
        }
    }

    /// A one-sided limit row with position error `c` (positive inside the range): speculative
    /// inside, soft past it in the biased solve, rigid without correction in the relax pass.
    fn limit(&self, c: f32, inv_h: f32, use_bias: bool) -> (f32, f32, f32) {
        if c > 0.0 {
            (c * inv_h, 1.0, 0.0)
        } else {
            self.rigid(c, use_bias)
        }
    }
}

/// The velocities and step motions of a joint's bodies while it is solved (Box3D's two body
/// states; a body that is not awake reads as still and unmoved).
#[derive(Debug, Clone, Copy)]
struct Pair {
    v_a: Vec3,
    w_a: Vec3,
    v_b: Vec3,
    w_b: Vec3,
    dp_a: Vec3,
    dp_b: Vec3,
    dq_a: Quat,
    dq_b: Quat,
}

impl Pair {
    fn load(bodies: &[RigidBody], motions: &[BodyMotion3d], base: &JointBase3d) -> Self {
        let (a, b) = (base.body_a as usize, base.body_b as usize);
        let (ma, mb) = (motions[a], motions[b]);
        Self {
            v_a: bodies[a].linear_velocity,
            w_a: bodies[a].angular_velocity,
            v_b: bodies[b].linear_velocity,
            w_b: bodies[b].angular_velocity,
            dp_a: ma.delta_position,
            dp_b: mb.delta_position,
            dq_a: ma.delta_rotation,
            dq_b: mb.delta_rotation,
        }
    }

    /// # Safety
    ///
    /// No other thread may write to the dynamic bodies of `base` during the call.
    unsafe fn load_shared(
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        motions: &crate::parallel::Shared<'_, BodyMotion3d>,
        base: &JointBase3d,
    ) -> (Self, bool, bool) {
        let (a, b) = (base.body_a as usize, base.body_b as usize);
        // SAFETY: caller guarantees no concurrent write to these bodies or their motions.
        let (body_a, body_b) = unsafe { (bodies.get(a), bodies.get(b)) };
        // SAFETY: caller guarantees motions are read-only during constraint solving.
        let (ma, mb) = unsafe { (motions.read(a), motions.read(b)) };
        (
            Self {
                v_a: body_a.linear_velocity,
                w_a: body_a.angular_velocity,
                v_b: body_b.linear_velocity,
                w_b: body_b.angular_velocity,
                dp_a: ma.delta_position,
                dp_b: mb.delta_position,
                dq_a: ma.delta_rotation,
                dq_b: mb.delta_rotation,
            },
            body_a.body_type == BodyType::Dynamic,
            body_b.body_type == BodyType::Dynamic,
        )
    }

    /// Writes the solved velocities back to the dynamic bodies.
    fn store(&self, bodies: &mut [RigidBody], base: &JointBase3d) {
        let a = &mut bodies[base.body_a as usize];
        if a.body_type == BodyType::Dynamic {
            a.linear_velocity = self.v_a;
            a.angular_velocity = self.w_a;
        }
        let b = &mut bodies[base.body_b as usize];
        if b.body_type == BodyType::Dynamic {
            b.linear_velocity = self.v_b;
            b.angular_velocity = self.w_b;
        }
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `base` during the call.
    unsafe fn store_shared(
        &self,
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        base: &JointBase3d,
        dynamic_a: bool,
        dynamic_b: bool,
    ) {
        if dynamic_a {
            // SAFETY: caller guarantees exclusive access to this dynamic body.
            let a = unsafe { bodies.get_mut(base.body_a as usize) };
            a.linear_velocity = self.v_a;
            a.angular_velocity = self.w_a;
        }
        if dynamic_b {
            // SAFETY: caller guarantees exclusive access to this dynamic body.
            let b = unsafe { bodies.get_mut(base.body_b as usize) };
            b.linear_velocity = self.v_b;
            b.angular_velocity = self.w_b;
        }
    }

    /// The current anchors: the prepared frame positions turned by the step's rotations.
    fn anchors(&self, s: &JointStep) -> (Vec3, Vec3) {
        (
            self.dq_a * s.frame_a.position,
            self.dq_b * s.frame_b.position,
        )
    }

    /// The current frame rotations, with B's sign chosen so that their relative rotation is
    /// the shorter one.
    fn frame_rotations(&self, s: &JointStep) -> (Quat, Quat) {
        let quat_a = self.dq_a * s.frame_a.rotation;
        let mut quat_b = self.dq_b * s.frame_b.rotation;
        if quat_a.dot(quat_b) < 0.0 {
            quat_b = -quat_b;
        }
        (quat_a, quat_b)
    }

    /// The separation of the anchors: the prepared one plus the motion since.
    fn separation(&self, s: &JointStep, r_a: Vec3, r_b: Vec3) -> Vec3 {
        (self.dp_b - self.dp_a) + (r_b - r_a) + s.delta_center
    }

    /// The relative velocity of the anchors.
    fn anchor_velocity(&self, r_a: Vec3, r_b: Vec3) -> Vec3 {
        (self.v_b + self.w_b.cross(r_b)) - self.v_a - self.w_a.cross(r_a)
    }

    /// Applies the linear impulse `p` at the anchors.
    fn apply_linear(&mut self, s: &JointStep, p: Vec3, r_a: Vec3, r_b: Vec3) {
        self.v_a -= p * s.inv_mass_a;
        self.w_a -= s.inv_i_a * r_a.cross(p);
        self.v_b += p * s.inv_mass_b;
        self.w_b += s.inv_i_b * r_b.cross(p);
    }

    /// Applies a linear impulse `p` with explicit angular parts `l_a` and `l_b`.
    fn apply(&mut self, s: &JointStep, p: Vec3, l_a: Vec3, l_b: Vec3) {
        self.v_a -= p * s.inv_mass_a;
        self.w_a -= s.inv_i_a * l_a;
        self.v_b += p * s.inv_mass_b;
        self.w_b += s.inv_i_b * l_b;
    }

    /// Applies the angular impulse `l`.
    fn apply_angular(&mut self, s: &JointStep, l: Vec3) {
        self.w_a -= s.inv_i_a * l;
        self.w_b += s.inv_i_b * l;
    }
}

/// The substep and its inverse.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Substep {
    pub h: f32,
    pub inv_h: f32,
}

/// Box3D's collinearity axes of a hinge: the Jacobians of relative rotation `rel`'s x and y
/// components, in world space through frame A's rotation `quat_a`.
fn perp_axes(quat_a: Quat, rel: Quat) -> (Vec3, Vec3) {
    let v = Vec3::new(rel.x, rel.y, rel.z);
    (
        (quat_a * (Vec3::X * rel.w + v.cross(Vec3::X))) * 0.5,
        (quat_a * (Vec3::Y * rel.w + v.cross(Vec3::Y))) * 0.5,
    )
}

/// The 2×2 inverse-inertia matrix of two angular axes.
fn axis_pair_mass(i_sum: &Mat3, x: Vec3, y: Vec3) -> (f32, f32, f32) {
    (x.dot(*i_sum * x), x.dot(*i_sum * y), y.dot(*i_sum * y))
}

/// Solves a soft two-axis row, accumulating `impulse`; returns the impulse change.
fn solve_pair(
    k: (f32, f32, f32),
    cdot: (f32, f32),
    bias: (f32, f32),
    (mass_scale, impulse_scale): (f32, f32),
    impulse: &mut Vec2,
) -> Vec2 {
    let sol = math::solve2(k, (cdot.0 + bias.0, cdot.1 + bias.1));
    let delta = Vec2::new(
        -mass_scale * sol.0 - impulse_scale * impulse.x,
        -mass_scale * sol.1 - impulse_scale * impulse.y,
    );
    *impulse += delta;
    delta
}

/// Solves a one-sided row: accumulates `impulse`, keeps it non-negative, returns the change.
fn solve_limit_row(
    impulse: &mut f32,
    mass: f32,
    cdot: f32,
    (bias, mass_scale, impulse_scale): (f32, f32, f32),
) -> f32 {
    let old = *impulse;
    let delta = -mass_scale * mass * (cdot + bias) - impulse_scale * old;
    *impulse = (old + delta).max(0.0);
    *impulse - old
}

/// Solves a motor row clamped to `max_impulse`: returns the change.
fn solve_motor_row(impulse: &mut f32, mass: f32, cdot: f32, max_impulse: f32) -> f32 {
    let old = *impulse;
    *impulse = (old - mass * cdot).clamp(-max_impulse, max_impulse);
    *impulse - old
}

/// The effective mass of a scalar row with inverse effective mass `k`.
fn mass_of(k: f32) -> f32 {
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// Clamps a vector impulse's length to `max`.
fn clamp_length(v: Vec3, max: f32) -> Vec3 {
    if v.length_squared() > max * max {
        normalize(v) * max
    } else {
        v
    }
}

/// Body B's frame rotation relative to body A's at their current poses, the shorter way.
fn relative_rotation(base: &JointBase3d, a: &RigidBody, b: &RigidBody) -> Quat {
    let quat_a = a.rotation * base.local_frame_a.rotation;
    let mut quat_b = b.rotation * base.local_frame_b.rotation;
    if quat_a.dot(quat_b) < 0.0 {
        quat_b = -quat_b;
    }
    quat_a.conjugate() * quat_b
}

/// The point-to-point row shared by the spherical and revolute joints: the anchors coincide,
/// softly in the biased solve.
fn solve_point(s: &JointStep, p: &mut Pair, accumulated: &mut Vec3, use_bias: bool) {
    let (r_a, r_b) = p.anchors(s);
    let cdot = p.anchor_velocity(r_a, r_b);
    let (bias, mass_scale, impulse_scale) = if use_bias {
        (
            p.separation(s, r_a, r_b) * s.softness.bias_rate,
            s.softness.mass_scale,
            s.softness.impulse_scale,
        )
    } else {
        (Vec3::ZERO, 1.0, 0.0)
    };
    let k = math::point_mass(s.masses(), &s.inv_i_a, &s.inv_i_b, r_a, r_b);
    let b = math::solve3(&k, cdot + bias);
    let impulse = b * -mass_scale - *accumulated * impulse_scale;
    *accumulated += impulse;
    p.apply_linear(s, impulse, r_a, r_b);
}

/// Swing angle used for spherical joint cone limit, adapting Erin Catto's `b3GetSwingAngle`.
#[must_use]
pub fn get_swing_angle(q: Quat) -> f32 {
    math::swing_angle(q)
}

/// Implements the shared builders of a joint type.
macro_rules! joint_base_builders {
    ($ty:ident, $variant:ident) => {
        impl $ty {
            /// Applies the accumulated impulses to two bodies at their current poses, as the
            /// solver's warm start does at the start of a step.
            pub fn warm_start(&self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
                Joint::$variant(self.clone()).warm_start(body_a, body_b);
            }

            /// Sets the joint's local frames.
            #[must_use]
            pub fn with_local_frames(mut self, frame_a: Frame3d, frame_b: Frame3d) -> Self {
                self.base.local_frame_a = frame_a;
                self.base.local_frame_b = frame_b;
                self
            }

            /// Sets the stiffness and damping ratio of the rigid rows (Box3D
            /// `b3Joint_SetConstraintTuning`).
            #[must_use]
            pub fn with_constraint_tuning(mut self, hertz: f32, damping_ratio: f32) -> Self {
                self.base.constraint_hertz = hertz;
                self.base.constraint_damping_ratio = damping_ratio;
                self
            }
        }
    };
}

// ---------------------------------------------------------------- spherical

/// Spherical (ball-and-socket) joint: the frame origins coincide; optional cone limit about
/// frame A's z axis, twist limits about frame B's, a spring towards a target rotation and a
/// motor towards a relative angular velocity (Box3D `spherical_joint.c`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SphericalJoint {
    /// The bodies and frames.
    pub base: JointBase3d,
    /// Whether the rotational spring towards `target_rotation` is on.
    pub enable_spring: bool,
    /// Spring stiffness in Hz.
    pub hertz: f32,
    /// Spring damping ratio.
    pub damping_ratio: f32,
    /// Spring target: frame B's rotation relative to frame A's.
    pub target_rotation: Quat,
    /// Whether the cone limit is on.
    pub enable_cone_limit: bool,
    /// Largest swing of frame B's z axis away from frame A's, in [0, π/2].
    pub cone_angle: f32,
    /// Whether the twist limits are on.
    pub enable_twist_limit: bool,
    /// Lower twist about frame B's z axis, in radians.
    pub lower_twist_angle: f32,
    /// Upper twist, in radians.
    pub upper_twist_angle: f32,
    /// Whether the motor is on.
    pub enable_motor: bool,
    /// Largest motor torque, N·m.
    pub max_motor_torque: f32,
    /// Motor target: body B's angular velocity relative to A's, world space.
    pub motor_velocity: Vec3,
    /// Accumulated point impulse.
    pub linear_impulse: Vec3,
    /// Accumulated spring impulse.
    pub spring_impulse: Vec3,
    /// Accumulated motor impulse.
    pub motor_impulse: Vec3,
    /// Accumulated lower twist limit impulse.
    pub lower_twist_impulse: f32,
    /// Accumulated upper twist limit impulse.
    pub upper_twist_impulse: f32,
    /// Accumulated cone limit impulse.
    pub swing_impulse: f32,
    #[serde(skip)]
    swing_axis: Vec3,
    #[serde(skip)]
    twist_jacobian: Vec3,
    #[serde(skip)]
    rotation_mass: Mat3,
    #[serde(skip)]
    swing_mass: f32,
    #[serde(skip)]
    twist_mass: f32,
    #[serde(skip)]
    spring_softness: Softness3d,
}

joint_base_builders!(SphericalJoint, Spherical);

impl SphericalJoint {
    /// Joins body A and body B at the given local anchors, frames aligned with the bodies.
    #[must_use]
    pub fn new(
        body_a: impl Into<BodyId3d>,
        body_b: impl Into<BodyId3d>,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
    ) -> Self {
        Self {
            base: JointBase3d::new(
                body_a,
                body_b,
                Frame3d::at(local_anchor_a),
                Frame3d::at(local_anchor_b),
            ),
            enable_spring: false,
            hertz: 0.0,
            damping_ratio: 0.0,
            target_rotation: Quat::IDENTITY,
            enable_cone_limit: false,
            cone_angle: 0.0,
            enable_twist_limit: false,
            lower_twist_angle: 0.0,
            upper_twist_angle: 0.0,
            enable_motor: false,
            max_motor_torque: 0.0,
            motor_velocity: Vec3::ZERO,
            linear_impulse: Vec3::ZERO,
            spring_impulse: Vec3::ZERO,
            motor_impulse: Vec3::ZERO,
            lower_twist_impulse: 0.0,
            upper_twist_impulse: 0.0,
            swing_impulse: 0.0,
            swing_axis: Vec3::ZERO,
            twist_jacobian: Vec3::ZERO,
            rotation_mass: Mat3::ZERO,
            swing_mass: 0.0,
            twist_mass: 0.0,
            spring_softness: Softness3d::default(),
        }
    }

    /// Enables the cone limit with half-angle `cone_angle`, clamped to [0, π/2] as in Box3D.
    #[must_use]
    pub fn with_cone_limit(mut self, cone_angle: f32) -> Self {
        self.enable_cone_limit = true;
        self.cone_angle = cone_angle.clamp(0.0, 0.5 * std::f32::consts::PI);
        self
    }

    /// Enables twist limits, ordered and clamped to ±0.99 π.
    #[must_use]
    pub fn with_twist_limits(mut self, lower: f32, upper: f32) -> Self {
        self.enable_twist_limit = true;
        self.lower_twist_angle = lower.min(upper).clamp(-MAX_JOINT_ANGLE, MAX_JOINT_ANGLE);
        self.upper_twist_angle = lower.max(upper).clamp(-MAX_JOINT_ANGLE, MAX_JOINT_ANGLE);
        self
    }

    /// Enables the rotational spring towards `target` (frame B relative to frame A).
    #[must_use]
    pub fn with_spring(mut self, hertz: f32, damping_ratio: f32, target: Quat) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self.target_rotation = target;
        self
    }

    /// Enables the motor towards relative angular velocity `velocity` with at most
    /// `max_torque`.
    #[must_use]
    pub fn with_motor(mut self, velocity: Vec3, max_torque: f32) -> Self {
        self.enable_motor = true;
        self.motor_velocity = velocity;
        self.max_motor_torque = max_torque;
        self
    }

    /// Sets the frames' rotations on body A and body B, keeping their anchors.
    #[must_use]
    pub fn with_frames(mut self, rotation_a: Quat, rotation_b: Quat) -> Self {
        self.base.local_frame_a.rotation = rotation_a;
        self.base.local_frame_b.rotation = rotation_b;
        self
    }

    /// The current swing of frame B's z axis away from frame A's, in radians.
    #[must_use]
    pub fn current_swing_angle(&self, body_a: &RigidBody, body_b: &RigidBody) -> f32 {
        math::swing_angle(relative_rotation(&self.base, body_a, body_b))
    }

    /// The current twist about frame B's z axis, in radians.
    #[must_use]
    pub fn current_twist_angle(&self, body_a: &RigidBody, body_b: &RigidBody) -> f32 {
        math::twist_angle(relative_rotation(&self.base, body_a, body_b))
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        let i_sum = s.inv_inertia_sum();
        let cone_axis = s.frame_a.rotation * Vec3::Z;
        let twist_axis = s.frame_b.rotation * Vec3::Z;
        if self.enable_cone_limit {
            let swing_axis = normalize(cone_axis.cross(twist_axis));
            self.swing_mass = mass_of(swing_axis.dot(i_sum * swing_axis));
            self.swing_axis = swing_axis;
        } else {
            self.swing_impulse = 0.0;
        }
        if self.enable_twist_limit {
            let rel = s.frame_a.rotation.conjugate() * s.frame_b.rotation;
            let tan_half =
                ((rel.x * rel.x + rel.y * rel.y) / (rel.z * rel.z + rel.w * rel.w)).sqrt();
            let swing_axis = normalize(cone_axis.cross(twist_axis));
            let perp_axis = swing_axis.cross(cone_axis);
            let jacobian = cone_axis + perp_axis * tan_half;
            self.twist_mass = mass_of(jacobian.dot(i_sum * jacobian));
            self.twist_jacobian = jacobian;
        } else {
            self.lower_twist_impulse = 0.0;
            self.upper_twist_impulse = 0.0;
        }
        self.rotation_mass = if s.fixed_rotation {
            Mat3::ZERO
        } else {
            math::invert(&i_sum)
        };
        self.spring_softness = Softness3d::new(self.hertz, self.damping_ratio, h);
        if !self.enable_spring {
            self.spring_impulse = Vec3::ZERO;
        }
        if !self.enable_motor {
            self.motor_impulse = Vec3::ZERO;
        }
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        let (r_a, r_b) = p.anchors(s);
        let angular = (self.spring_impulse + self.motor_impulse)
            - self.swing_axis * self.swing_impulse
            + self.twist_jacobian * (self.lower_twist_impulse - self.upper_twist_impulse);
        p.apply(
            s,
            self.linear_impulse,
            r_a.cross(self.linear_impulse) + angular,
            r_b.cross(self.linear_impulse) + angular,
        );
    }

    fn solve(&mut self, p: &mut Pair, step: Substep, use_bias: bool) {
        let s = self.base.step;
        let quat_a = p.dq_a * s.frame_a.rotation;
        let quat_b = p.dq_b * s.frame_b.rotation;
        let rel = quat_a.conjugate() * quat_b;
        let rotating = !s.fixed_rotation;

        if self.enable_spring && rotating {
            let c = -(quat_a * math::delta_quat_to_rotation(rel, self.target_rotation));
            let soft = self.spring_softness;
            let cdot = p.w_b - p.w_a;
            let impulse = -(self.rotation_mass * (cdot + c * soft.bias_rate)) * soft.mass_scale
                - self.spring_impulse * soft.impulse_scale;
            self.spring_impulse += impulse;
            p.apply_angular(&s, impulse);
        }

        if self.enable_motor && rotating {
            let cdot = p.w_b - p.w_a;
            let lambda = -(self.rotation_mass * (cdot - self.motor_velocity));
            let mut new_impulse = self.motor_impulse + lambda;
            let length = new_impulse.length();
            let max_impulse = self.max_motor_torque * step.h;
            if length > max_impulse {
                new_impulse *= max_impulse / length;
            }
            let delta = new_impulse - self.motor_impulse;
            self.motor_impulse = new_impulse;
            p.apply_angular(&s, delta);
        }

        if self.enable_twist_limit && rotating {
            let twist = math::twist_angle(rel);
            let jacobian = self.twist_jacobian;
            let row = s.limit(twist - self.lower_twist_angle, step.inv_h, use_bias);
            let cdot = (p.w_b - p.w_a).dot(jacobian);
            let delta = solve_limit_row(&mut self.lower_twist_impulse, self.twist_mass, cdot, row);
            p.apply_angular(&s, jacobian * delta);
            let row = s.limit(self.upper_twist_angle - twist, step.inv_h, use_bias);
            let cdot = (p.w_a - p.w_b).dot(jacobian);
            let delta = solve_limit_row(&mut self.upper_twist_impulse, self.twist_mass, cdot, row);
            p.apply_angular(&s, -(jacobian * delta));
        }

        if self.enable_cone_limit && rotating {
            let swing = math::swing_angle(rel);
            let axis = self.swing_axis;
            let row = s.limit(self.cone_angle - swing, step.inv_h, use_bias);
            let cdot = (p.w_a - p.w_b).dot(axis);
            let delta = solve_limit_row(&mut self.swing_impulse, self.swing_mass, cdot, row);
            p.apply_angular(&s, -(axis * delta));
        }

        solve_point(&s, p, &mut self.linear_impulse, use_bias);
    }
}

// ---------------------------------------------------------------- revolute

/// Revolute (hinge) joint: the frame origins coincide and the bodies turn only about frame
/// z, with an optional spring towards a target angle, limits and a motor (Box3D
/// `revolute_joint.c`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RevoluteJoint {
    /// The bodies and frames.
    pub base: JointBase3d,
    /// Whether the spring towards `target_angle` is on.
    pub enable_spring: bool,
    /// Spring stiffness in Hz.
    pub hertz: f32,
    /// Spring damping ratio.
    pub damping_ratio: f32,
    /// Spring target angle, radians.
    pub target_angle: f32,
    /// Whether angular limits are enabled.
    pub enable_limit: bool,
    /// Lower rotation limit in radians.
    pub lower_angle: f32,
    /// Upper rotation limit in radians.
    pub upper_angle: f32,
    /// Whether rotational motor is enabled.
    pub enable_motor: bool,
    /// Target rotational motor speed in rad/s.
    pub motor_speed: f32,
    /// Maximum motor torque.
    pub max_motor_torque: f32,
    /// Accumulated point impulse.
    pub linear_impulse: Vec3,
    /// Accumulated collinearity impulse.
    pub perp_impulse: Vec2,
    /// Accumulated spring impulse.
    pub spring_impulse: f32,
    /// Accumulated motor impulse.
    pub motor_impulse: f32,
    /// Accumulated impulse of the lower angle limit.
    pub lower_impulse: f32,
    /// Accumulated impulse of the upper angle limit.
    pub upper_impulse: f32,
    #[serde(skip)]
    axial_mass: f32,
    #[serde(skip)]
    rotation_axis_z: Vec3,
    #[serde(skip)]
    perp_axis_x: Vec3,
    #[serde(skip)]
    perp_axis_y: Vec3,
    #[serde(skip)]
    spring_softness: Softness3d,
}

joint_base_builders!(RevoluteJoint, Revolute);

impl RevoluteJoint {
    /// Joins body A and body B at the given local anchors with the hinge along `local_axis_a`,
    /// the same axis in both bodies' frames: the hinge angle is zero when both bodies have the
    /// same rotation.
    #[must_use]
    pub fn new(
        body_a: impl Into<BodyId3d>,
        body_b: impl Into<BodyId3d>,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        local_axis_a: Vec3,
    ) -> Self {
        let frame = Quat::from_rotation_arc(Vec3::Z, local_axis_a.normalize());
        Self {
            base: JointBase3d::new(
                body_a,
                body_b,
                Frame3d::new(local_anchor_a, frame),
                Frame3d::new(local_anchor_b, frame),
            ),
            enable_spring: false,
            hertz: 0.0,
            damping_ratio: 0.0,
            target_angle: 0.0,
            enable_limit: false,
            lower_angle: 0.0,
            upper_angle: 0.0,
            enable_motor: false,
            motor_speed: 0.0,
            max_motor_torque: 0.0,
            linear_impulse: Vec3::ZERO,
            perp_impulse: Vec2::ZERO,
            spring_impulse: 0.0,
            motor_impulse: 0.0,
            lower_impulse: 0.0,
            upper_impulse: 0.0,
            axial_mass: 0.0,
            rotation_axis_z: Vec3::ZERO,
            perp_axis_x: Vec3::ZERO,
            perp_axis_y: Vec3::ZERO,
            spring_softness: Softness3d::default(),
        }
    }

    /// Enables limits on the hinge angle in radians. As in Box3D, the limits are ordered and
    /// clamped to ±0.99 π.
    #[must_use]
    pub fn with_limits(mut self, lower: f32, upper: f32) -> Self {
        self.enable_limit = true;
        self.lower_angle = lower.min(upper).clamp(-MAX_JOINT_ANGLE, MAX_JOINT_ANGLE);
        self.upper_angle = lower.max(upper).clamp(-MAX_JOINT_ANGLE, MAX_JOINT_ANGLE);
        self
    }

    /// Enables the motor: it drives the hinge towards `speed` rad/s with at most `max_torque`
    /// N·m.
    #[must_use]
    pub fn with_motor(mut self, speed: f32, max_torque: f32) -> Self {
        self.enable_motor = true;
        self.motor_speed = speed;
        self.max_motor_torque = max_torque;
        self
    }

    /// Enables the spring towards the target angle.
    #[must_use]
    pub fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Sets the spring's target angle, clamped to ±π.
    #[must_use]
    pub fn with_target_angle(mut self, angle: f32) -> Self {
        self.target_angle = angle.clamp(-std::f32::consts::PI, std::f32::consts::PI);
        self
    }

    /// The hinge angle in radians, in [-π, π]: frame B's twist relative to frame A about the
    /// hinge (Box3D `b3RevoluteJoint_GetAngle`).
    #[must_use]
    pub fn angle(&self, body_a: &RigidBody, body_b: &RigidBody) -> f32 {
        math::twist_angle(relative_rotation(&self.base, body_a, body_b))
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        let i_sum = s.inv_inertia_sum();
        let axis = s.frame_a.rotation * Vec3::Z;
        self.axial_mass = mass_of(axis.dot(i_sum * axis));
        self.rotation_axis_z = axis;
        let rel = s.frame_a.rotation.conjugate() * s.frame_b.rotation;
        (self.perp_axis_x, self.perp_axis_y) = perp_axes(s.frame_a.rotation, rel);
        self.spring_softness = Softness3d::new(self.hertz, self.damping_ratio, h);
        if !self.enable_spring {
            self.spring_impulse = 0.0;
        }
        if !self.enable_motor {
            self.motor_impulse = 0.0;
        }
        if !self.enable_limit {
            self.lower_impulse = 0.0;
            self.upper_impulse = 0.0;
        }
    }

    fn axial_impulse(&self) -> f32 {
        self.spring_impulse + self.motor_impulse + self.lower_impulse - self.upper_impulse
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        let (r_a, r_b) = p.anchors(s);
        let angular = self.perp_axis_x * self.perp_impulse.x
            + self.perp_axis_y * self.perp_impulse.y
            + self.rotation_axis_z * self.axial_impulse();
        p.apply(
            s,
            self.linear_impulse,
            r_a.cross(self.linear_impulse) + angular,
            r_b.cross(self.linear_impulse) + angular,
        );
    }

    fn solve(&mut self, p: &mut Pair, step: Substep, use_bias: bool) {
        let s = self.base.step;
        let (quat_a, quat_b) = p.frame_rotations(&s);
        let rel = quat_a.conjugate() * quat_b;
        let axis = self.rotation_axis_z;
        let rotating = !s.fixed_rotation;

        if self.enable_spring && rotating {
            let c = math::twist_angle(rel) - self.target_angle;
            let soft = self.spring_softness;
            let cdot = (p.w_b - p.w_a).dot(axis);
            let delta = -soft.mass_scale * self.axial_mass * (cdot + soft.bias_rate * c)
                - soft.impulse_scale * self.spring_impulse;
            self.spring_impulse += delta;
            p.apply_angular(&s, axis * delta);
        }

        if self.enable_motor && rotating {
            let cdot = (p.w_b - p.w_a).dot(axis) - self.motor_speed;
            let delta = solve_motor_row(
                &mut self.motor_impulse,
                self.axial_mass,
                cdot,
                self.max_motor_torque * step.h,
            );
            p.apply_angular(&s, axis * delta);
        }

        if self.enable_limit && rotating {
            let angle = math::twist_angle(rel);
            let row = s.limit(angle - self.lower_angle, step.inv_h, use_bias);
            let cdot = (p.w_b - p.w_a).dot(axis);
            let delta = solve_limit_row(&mut self.lower_impulse, self.axial_mass, cdot, row);
            p.apply_angular(&s, axis * delta);
            let row = s.limit(self.upper_angle - angle, step.inv_h, use_bias);
            let cdot = (p.w_a - p.w_b).dot(axis);
            let delta = solve_limit_row(&mut self.upper_impulse, self.axial_mass, cdot, row);
            p.apply_angular(&s, -(axis * delta));
        }

        // Collinearity: frame B's z axis stays along frame A's.
        if rotating {
            let (bias, mass_scale, impulse_scale) = if use_bias {
                let rate = s.softness.bias_rate;
                (
                    (rate * rel.x, rate * rel.y),
                    s.softness.mass_scale,
                    s.softness.impulse_scale,
                )
            } else {
                ((0.0, 0.0), 1.0, 0.0)
            };
            let (perp_x, perp_y) = perp_axes(quat_a, rel);
            self.perp_axis_x = perp_x;
            self.perp_axis_y = perp_y;
            let k = axis_pair_mass(&s.inv_inertia_sum(), perp_x, perp_y);
            let w_rel = p.w_b - p.w_a;
            let delta = solve_pair(
                k,
                (w_rel.dot(perp_x), w_rel.dot(perp_y)),
                bias,
                (mass_scale, impulse_scale),
                &mut self.perp_impulse,
            );
            p.apply_angular(&s, perp_x * delta.x + perp_y * delta.y);
        }

        solve_point(&s, p, &mut self.linear_impulse, use_bias);
    }
}

// ---------------------------------------------------------------- prismatic

/// Prismatic (slider) joint: body B slides along frame A's x axis without turning relative to
/// A, with an optional spring towards a target translation, limits and a motor (Box3D
/// `prismatic_joint.c`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrismaticJoint {
    /// The bodies and frames.
    pub base: JointBase3d,
    /// Whether the spring towards `target_translation` is on.
    pub enable_spring: bool,
    /// Spring stiffness in Hz.
    pub hertz: f32,
    /// Spring damping ratio.
    pub damping_ratio: f32,
    /// Spring target translation, metres.
    pub target_translation: f32,
    /// Whether translation limits are enabled.
    pub enable_limit: bool,
    /// Lower translation limit.
    pub lower_translation: f32,
    /// Upper translation limit.
    pub upper_translation: f32,
    /// Whether linear motor is enabled.
    pub enable_motor: bool,
    /// Target motor speed.
    pub motor_speed: f32,
    /// Maximum motor force.
    pub max_motor_force: f32,
    /// Accumulated point-to-line impulse.
    pub perp_impulse: Vec2,
    /// Accumulated rotation impulse.
    pub angular_impulse: Vec3,
    /// Accumulated spring impulse.
    pub spring_impulse: f32,
    /// Accumulated motor impulse.
    pub motor_impulse: f32,
    /// Accumulated impulse of the lower translation limit.
    pub lower_impulse: f32,
    /// Accumulated impulse of the upper translation limit.
    pub upper_impulse: f32,
    #[serde(skip)]
    joint_axis: Vec3,
    #[serde(skip)]
    perp_axis_y: Vec3,
    #[serde(skip)]
    perp_axis_z: Vec3,
    #[serde(skip)]
    rotation_mass: Mat3,
    #[serde(skip)]
    spring_softness: Softness3d,
}

joint_base_builders!(PrismaticJoint, Prismatic);

impl PrismaticJoint {
    /// Joins body A and body B at the given local anchors, sliding along `local_axis_a`, the
    /// same axis in both bodies' frames.
    #[must_use]
    pub fn new(
        body_a: impl Into<BodyId3d>,
        body_b: impl Into<BodyId3d>,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        local_axis_a: Vec3,
    ) -> Self {
        let frame = Quat::from_rotation_arc(Vec3::X, local_axis_a.normalize());
        Self {
            base: JointBase3d::new(
                body_a,
                body_b,
                Frame3d::new(local_anchor_a, frame),
                Frame3d::new(local_anchor_b, frame),
            ),
            enable_spring: false,
            hertz: 0.0,
            damping_ratio: 0.0,
            target_translation: 0.0,
            enable_limit: false,
            lower_translation: 0.0,
            upper_translation: 0.0,
            enable_motor: false,
            motor_speed: 0.0,
            max_motor_force: 0.0,
            perp_impulse: Vec2::ZERO,
            angular_impulse: Vec3::ZERO,
            spring_impulse: 0.0,
            motor_impulse: 0.0,
            lower_impulse: 0.0,
            upper_impulse: 0.0,
            joint_axis: Vec3::ZERO,
            perp_axis_y: Vec3::ZERO,
            perp_axis_z: Vec3::ZERO,
            rotation_mass: Mat3::ZERO,
            spring_softness: Softness3d::default(),
        }
    }

    /// Enables limits on the translation in metres, ordered.
    #[must_use]
    pub fn with_limits(mut self, lower: f32, upper: f32) -> Self {
        self.enable_limit = true;
        self.lower_translation = lower.min(upper);
        self.upper_translation = lower.max(upper);
        self
    }

    /// Enables the motor: it drives the slider towards `speed` m/s with at most `max_force` N.
    #[must_use]
    pub fn with_motor(mut self, speed: f32, max_force: f32) -> Self {
        self.enable_motor = true;
        self.motor_speed = speed;
        self.max_motor_force = max_force;
        self
    }

    /// Enables the spring towards the target translation.
    #[must_use]
    pub fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Sets the spring's target translation.
    #[must_use]
    pub fn with_target_translation(mut self, translation: f32) -> Self {
        self.target_translation = translation;
        self
    }

    /// The translation in metres: how far frame B's origin is from frame A's along the slider
    /// axis (Box3D `b3PrismaticJoint_GetTranslation`).
    #[must_use]
    pub fn translation(&self, body_a: &RigidBody, body_b: &RigidBody) -> f32 {
        let (frame_a, frame_b) = self.base.world_frames(body_a, body_b);
        (frame_b.position - frame_a.position).dot(frame_a.rotation * Vec3::X)
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        self.rotation_mass = math::invert(&s.inv_inertia_sum());
        let m = Mat3::from_quat(s.frame_a.rotation);
        self.joint_axis = m.x_axis;
        self.perp_axis_y = m.y_axis;
        self.perp_axis_z = m.z_axis;
        self.spring_softness = Softness3d::new(self.hertz, self.damping_ratio, h);
        if !self.enable_spring {
            self.spring_impulse = 0.0;
        }
        if !self.enable_motor {
            self.motor_impulse = 0.0;
        }
        if !self.enable_limit {
            self.lower_impulse = 0.0;
            self.upper_impulse = 0.0;
        }
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        let (r_a, r_b) = p.anchors(s);
        let d = p.separation(s, r_a, r_b);
        let axis = p.dq_a * self.joint_axis;
        let perp_y = p.dq_a * self.perp_axis_y;
        let perp_z = p.dq_a * self.perp_axis_z;
        let axial =
            self.spring_impulse + self.motor_impulse + self.lower_impulse - self.upper_impulse;
        let (py, pz) = (self.perp_impulse.x, self.perp_impulse.y);
        let lever_a = r_a + d;
        let impulse = axis * axial + perp_y * py + perp_z * pz;
        let l_a = lever_a.cross(axis) * axial
            + lever_a.cross(perp_y) * py
            + lever_a.cross(perp_z) * pz
            + self.angular_impulse;
        let l_b = r_b.cross(axis) * axial
            + r_b.cross(perp_y) * py
            + r_b.cross(perp_z) * pz
            + self.angular_impulse;
        p.apply(s, impulse, l_a, l_b);
    }

    fn solve(&mut self, p: &mut Pair, step: Substep, use_bias: bool) {
        let s = self.base.step;
        let (m_a, m_b) = s.masses();
        let (r_a, r_b) = p.anchors(&s);
        let d = p.separation(&s, r_a, r_b);
        let lever_a = r_a + d;
        let axis = p.dq_a * self.joint_axis;
        let s_ax = lever_a.cross(axis);
        let s_bx = r_b.cross(axis);
        let translation = d.dot(axis);
        let rotating = !s.fixed_rotation;
        // The axial effective mass must be fresh to avoid divergence when the joint is stressed.
        let axial_mass =
            mass_of(m_a + m_b + s_ax.dot(s.inv_i_a * s_ax) + s_bx.dot(s.inv_i_b * s_bx));
        let axial_velocity =
            |p: &Pair| (p.v_b + p.w_b.cross(r_b) - p.v_a - p.w_a.cross(lever_a)).dot(axis);

        if self.enable_spring && rotating {
            let c = translation - self.target_translation;
            let soft = self.spring_softness;
            let delta = -soft.mass_scale * axial_mass * (axial_velocity(p) + soft.bias_rate * c)
                - soft.impulse_scale * self.spring_impulse;
            self.spring_impulse += delta;
            p.apply(&s, axis * delta, s_ax * delta, s_bx * delta);
        }

        if self.enable_motor && rotating {
            let cdot = axial_velocity(p) - self.motor_speed;
            let max_impulse = self.max_motor_force * step.h;
            let delta = solve_motor_row(&mut self.motor_impulse, axial_mass, cdot, max_impulse);
            p.apply(&s, axis * delta, s_ax * delta, s_bx * delta);
        }

        if self.enable_limit && rotating {
            // Box3D solves a limit only within a quarter of the range of it.
            let speculative_distance = 0.25 * (self.upper_translation - self.lower_translation);
            let c = translation - self.lower_translation;
            if c < speculative_distance {
                let row = s.limit(c, step.inv_h, use_bias);
                let delta =
                    solve_limit_row(&mut self.lower_impulse, axial_mass, axial_velocity(p), row);
                p.apply(&s, axis * delta, s_ax * delta, s_bx * delta);
            } else {
                self.lower_impulse = 0.0;
            }
            let c = self.upper_translation - translation;
            if c < speculative_distance {
                let row = s.limit(c, step.inv_h, use_bias);
                let delta =
                    solve_limit_row(&mut self.upper_impulse, axial_mass, -axial_velocity(p), row);
                p.apply(&s, axis * -delta, s_ax * -delta, s_bx * -delta);
            } else {
                self.upper_impulse = 0.0;
            }
        }

        // The bodies do not turn relative to each other.
        if rotating {
            let (bias, mass_scale, impulse_scale) = if use_bias {
                let quat_a = p.dq_a * s.frame_a.rotation;
                let quat_b = p.dq_b * s.frame_b.rotation;
                let rel = quat_a.conjugate() * quat_b;
                let c = -(quat_a * math::delta_quat_to_rotation(rel, Quat::IDENTITY));
                (
                    c * s.softness.bias_rate,
                    s.softness.mass_scale,
                    s.softness.impulse_scale,
                )
            } else {
                (Vec3::ZERO, 1.0, 0.0)
            };
            let cdot = p.w_b - p.w_a;
            let impulse = -(self.rotation_mass * (cdot + bias)) * mass_scale
                - self.angular_impulse * impulse_scale;
            self.angular_impulse += impulse;
            p.apply_angular(&s, impulse);
        }

        // Point to line: frame B's origin stays on frame A's x axis.
        let perp_y = p.dq_a * self.perp_axis_y;
        let perp_z = p.dq_a * self.perp_axis_z;
        let (bias, mass_scale, impulse_scale) = if use_bias {
            let rate = s.softness.bias_rate;
            (
                (rate * perp_y.dot(d), rate * perp_z.dot(d)),
                s.softness.mass_scale,
                s.softness.impulse_scale,
            )
        } else {
            ((0.0, 0.0), 1.0, 0.0)
        };
        let v_rel = p.v_b + p.w_b.cross(r_b) - p.v_a - p.w_a.cross(lever_a);
        let s_ay = lever_a.cross(perp_y);
        let s_by = r_b.cross(perp_y);
        let s_az = lever_a.cross(perp_z);
        let s_bz = r_b.cross(perp_z);
        let kyy = m_a + m_b + s_ay.dot(s.inv_i_a * s_ay) + s_by.dot(s.inv_i_b * s_by);
        let kyz = s_ay.dot(s.inv_i_a * s_az) + s_by.dot(s.inv_i_b * s_bz);
        let kzz = m_a + m_b + s_az.dot(s.inv_i_a * s_az) + s_bz.dot(s.inv_i_b * s_bz);
        let delta = solve_pair(
            (kyy, kyz, kzz),
            (perp_y.dot(v_rel), perp_z.dot(v_rel)),
            bias,
            (mass_scale, impulse_scale),
            &mut self.perp_impulse,
        );
        p.apply(
            &s,
            perp_y * delta.x + perp_z * delta.y,
            s_ay * delta.x + s_az * delta.y,
            s_by * delta.x + s_bz * delta.y,
        );
    }
}

// ---------------------------------------------------------------- distance

/// Distance joint: keeps the frame origins at `length`, rigidly, or as a spring with a force
/// range, with optional length limits and a motor (Box3D `distance_joint.c`). A spring of zero
/// hertz with limits is a rope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistanceJoint {
    /// The bodies and frames; only the frame origins matter.
    pub base: JointBase3d,
    /// Rest length, at least the linear slop.
    pub length: f32,
    /// Whether the joint is a spring instead of rigid.
    pub enable_spring: bool,
    /// Spring stiffness in Hz.
    pub hertz: f32,
    /// Spring damping ratio.
    pub damping_ratio: f32,
    /// Largest tension the spring sustains, N (negative).
    pub lower_spring_force: f32,
    /// Largest compression the spring sustains, N.
    pub upper_spring_force: f32,
    /// Whether the length limits are on.
    pub enable_limit: bool,
    /// Shortest length.
    pub min_length: f32,
    /// Longest length.
    pub max_length: f32,
    /// Whether the motor is on.
    pub enable_motor: bool,
    /// Largest motor force, N.
    pub max_motor_force: f32,
    /// Motor target speed along the joint, m/s.
    pub motor_speed: f32,
    /// Accumulated impulse along the joint.
    pub impulse: f32,
    /// Accumulated lower limit impulse.
    pub lower_impulse: f32,
    /// Accumulated upper limit impulse.
    pub upper_impulse: f32,
    /// Accumulated motor impulse.
    pub motor_impulse: f32,
    #[serde(skip)]
    axial_mass: f32,
    #[serde(skip)]
    axis: Vec3,
    #[serde(skip)]
    distance_softness: Softness3d,
}

joint_base_builders!(DistanceJoint, Distance);

impl DistanceJoint {
    /// Joins body A and body B at the given local anchors, rigidly at `length` (at least the
    /// linear slop).
    #[must_use]
    pub fn new(
        body_a: impl Into<BodyId3d>,
        body_b: impl Into<BodyId3d>,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        length: f32,
    ) -> Self {
        Self {
            base: JointBase3d::new(
                body_a,
                body_b,
                Frame3d::at(local_anchor_a),
                Frame3d::at(local_anchor_b),
            ),
            length: length.max(LINEAR_SLOP),
            enable_spring: false,
            hertz: 0.0,
            damping_ratio: 0.0,
            lower_spring_force: -f32::MAX,
            upper_spring_force: f32::MAX,
            enable_limit: false,
            min_length: LINEAR_SLOP,
            max_length: HUGE_LENGTH,
            enable_motor: false,
            max_motor_force: 0.0,
            motor_speed: 0.0,
            impulse: 0.0,
            lower_impulse: 0.0,
            upper_impulse: 0.0,
            motor_impulse: 0.0,
            axial_mass: 0.0,
            axis: Vec3::ZERO,
            distance_softness: Softness3d::default(),
        }
    }

    /// Makes the joint a spring of `hertz` and `damping_ratio`; zero hertz gives a rope.
    #[must_use]
    pub fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Limits the spring force to [lower, upper] newtons.
    #[must_use]
    pub fn with_spring_force_range(mut self, lower: f32, upper: f32) -> Self {
        self.lower_spring_force = lower.min(upper);
        self.upper_spring_force = lower.max(upper);
        self
    }

    /// Enables length limits, clamped as in Box3D (`min` at least the linear slop, `max` at
    /// least `min`).
    #[must_use]
    pub fn with_length_range(mut self, min: f32, max: f32) -> Self {
        self.enable_limit = true;
        self.min_length = min.clamp(LINEAR_SLOP, HUGE_LENGTH);
        self.max_length = max.clamp(self.min_length, HUGE_LENGTH);
        self
    }

    /// Enables the motor towards `speed` m/s along the joint with at most `max_force` N.
    #[must_use]
    pub fn with_motor(mut self, speed: f32, max_force: f32) -> Self {
        self.enable_motor = true;
        self.motor_speed = speed;
        self.max_motor_force = max_force;
        self
    }

    /// The current distance between the frame origins.
    #[must_use]
    pub fn current_length(&self, body_a: &RigidBody, body_b: &RigidBody) -> f32 {
        let (frame_a, frame_b) = self.base.world_frames(body_a, body_b);
        (frame_b.position - frame_a.position).length()
    }

    fn is_soft(&self) -> bool {
        self.enable_spring && (self.min_length < self.max_length || !self.enable_limit)
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        let (r_a, r_b) = (s.frame_a.position, s.frame_b.position);
        let axis = normalize(r_b - r_a + s.delta_center);
        let cr_a = r_a.cross(axis);
        let cr_b = r_b.cross(axis);
        let (m_a, m_b) = s.masses();
        self.axial_mass =
            mass_of(m_a + m_b + cr_a.dot(s.inv_i_a * cr_a) + cr_b.dot(s.inv_i_b * cr_b));
        self.axis = axis;
        self.distance_softness = Softness3d::new(self.hertz, self.damping_ratio, h);
        if !self.enable_limit {
            self.lower_impulse = 0.0;
            self.upper_impulse = 0.0;
        }
        if !self.enable_motor {
            self.motor_impulse = 0.0;
        }
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        let (r_a, r_b) = p.anchors(s);
        let axis = normalize(p.separation(s, r_a, r_b));
        let axial = self.impulse + self.lower_impulse - self.upper_impulse + self.motor_impulse;
        p.apply_linear(s, axis * axial, r_a, r_b);
    }

    fn solve(&mut self, p: &mut Pair, step: Substep, use_bias: bool) {
        let s = self.base.step;
        let (r_a, r_b) = p.anchors(&s);
        let separation = p.separation(&s, r_a, r_b);
        let length = separation.length();
        let axis = normalize(separation);
        self.axis = axis;
        let cdot = |p: &Pair| p.anchor_velocity(r_a, r_b).dot(axis);

        if self.is_soft() {
            if self.hertz > 0.0 {
                let c = length - self.length;
                let soft = self.distance_softness;
                let m = soft.mass_scale * self.axial_mass;
                let old = self.impulse;
                let delta = -m * (cdot(p) + soft.bias_rate * c) - soft.impulse_scale * old;
                self.impulse = (old + delta).clamp(
                    self.lower_spring_force * step.h,
                    self.upper_spring_force * step.h,
                );
                p.apply_linear(&s, axis * (self.impulse - old), r_a, r_b);
            }

            if self.enable_limit {
                let row = s.limit(length - self.min_length, step.inv_h, use_bias);
                let delta = solve_limit_row(&mut self.lower_impulse, self.axial_mass, cdot(p), row);
                p.apply_linear(&s, axis * delta, r_a, r_b);
                let row = s.limit(self.max_length - length, step.inv_h, use_bias);
                let delta =
                    solve_limit_row(&mut self.upper_impulse, self.axial_mass, -cdot(p), row);
                p.apply_linear(&s, axis * -delta, r_a, r_b);
            }

            if self.enable_motor {
                let delta = solve_motor_row(
                    &mut self.motor_impulse,
                    self.axial_mass,
                    cdot(p) - self.motor_speed,
                    step.h * self.max_motor_force,
                );
                p.apply_linear(&s, axis * delta, r_a, r_b);
            }
        } else {
            let (bias, mass_scale, impulse_scale) = s.rigid(length - self.length, use_bias);
            let delta =
                -mass_scale * self.axial_mass * (cdot(p) + bias) - impulse_scale * self.impulse;
            self.impulse += delta;
            p.apply_linear(&s, axis * delta, r_a, r_b);
        }
    }
}

// ---------------------------------------------------------------- weld

/// Weld joint: locks the frames together, rigidly or with linear and angular springs (Box3D
/// `weld_joint.c`). The bodies keep the relative rotation of their frames: with frames
/// aligned with the bodies they are welded at equal rotations, and `Frame3d::from_world` welds
/// them as they are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeldJoint {
    /// The bodies and frames.
    pub base: JointBase3d,
    /// Linear spring stiffness in Hz; zero is rigid.
    pub linear_hertz: f32,
    /// Linear spring damping ratio.
    pub linear_damping_ratio: f32,
    /// Angular spring stiffness in Hz; zero is rigid.
    pub angular_hertz: f32,
    /// Angular spring damping ratio.
    pub angular_damping_ratio: f32,
    /// Accumulated linear impulse.
    pub linear_impulse: Vec3,
    /// Accumulated angular impulse.
    pub angular_impulse: Vec3,
    #[serde(skip)]
    angular_mass: Mat3,
    #[serde(skip)]
    linear_spring: Softness3d,
    #[serde(skip)]
    angular_spring: Softness3d,
}

joint_base_builders!(WeldJoint, Weld);

impl WeldJoint {
    /// Welds body A and body B at the given local anchors, frames aligned with the bodies.
    #[must_use]
    pub fn new(
        body_a: impl Into<BodyId3d>,
        body_b: impl Into<BodyId3d>,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
    ) -> Self {
        Self {
            base: JointBase3d::new(
                body_a,
                body_b,
                Frame3d::at(local_anchor_a),
                Frame3d::at(local_anchor_b),
            ),
            linear_hertz: 0.0,
            linear_damping_ratio: 0.0,
            angular_hertz: 0.0,
            angular_damping_ratio: 0.0,
            linear_impulse: Vec3::ZERO,
            angular_impulse: Vec3::ZERO,
            angular_mass: Mat3::ZERO,
            linear_spring: Softness3d::default(),
            angular_spring: Softness3d::default(),
        }
    }

    /// Makes the linear lock a spring.
    #[must_use]
    pub fn with_linear_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.linear_hertz = hertz;
        self.linear_damping_ratio = damping_ratio;
        self
    }

    /// Makes the angular lock a spring.
    #[must_use]
    pub fn with_angular_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.angular_hertz = hertz;
        self.angular_damping_ratio = damping_ratio;
        self
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        self.angular_mass = math::invert(&s.inv_inertia_sum());
        self.linear_spring = if self.linear_hertz == 0.0 {
            s.softness
        } else {
            Softness3d::new(self.linear_hertz, self.linear_damping_ratio, h)
        };
        self.angular_spring = if self.angular_hertz == 0.0 {
            s.softness
        } else {
            Softness3d::new(self.angular_hertz, self.angular_damping_ratio, h)
        };
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        let (r_a, r_b) = p.anchors(s);
        p.apply(
            s,
            self.linear_impulse,
            r_a.cross(self.linear_impulse) + self.angular_impulse,
            r_b.cross(self.linear_impulse) + self.angular_impulse,
        );
    }

    fn solve(&mut self, p: &mut Pair, use_bias: bool) {
        let s = self.base.step;
        if !s.fixed_rotation {
            let (bias, mass_scale, impulse_scale) = if use_bias || self.angular_hertz > 0.0 {
                let (quat_a, quat_b) = p.frame_rotations(&s);
                let rel = quat_a.conjugate() * quat_b;
                let c = -(quat_a * math::delta_quat_to_rotation(rel, Quat::IDENTITY));
                let spring = self.angular_spring;
                (
                    c * spring.bias_rate,
                    spring.mass_scale,
                    spring.impulse_scale,
                )
            } else {
                (Vec3::ZERO, 1.0, 0.0)
            };
            let cdot = p.w_b - p.w_a;
            let impulse = -(self.angular_mass * (cdot + bias)) * mass_scale
                - self.angular_impulse * impulse_scale;
            self.angular_impulse += impulse;
            p.apply_angular(&s, impulse);
        }

        let (r_a, r_b) = p.anchors(&s);
        let cdot = p.anchor_velocity(r_a, r_b);
        let (bias, mass_scale, impulse_scale) = if use_bias || self.linear_hertz > 0.0 {
            let spring = self.linear_spring;
            (
                p.separation(&s, r_a, r_b) * spring.bias_rate,
                spring.mass_scale,
                spring.impulse_scale,
            )
        } else {
            (Vec3::ZERO, 1.0, 0.0)
        };
        let k = math::point_mass(s.masses(), &s.inv_i_a, &s.inv_i_b, r_a, r_b);
        let b = math::solve3(&k, cdot + bias);
        let impulse = b * -mass_scale - self.linear_impulse * impulse_scale;
        self.linear_impulse += impulse;
        p.apply_linear(&s, impulse, r_a, r_b);
    }
}

// ---------------------------------------------------------------- wheel

/// Wheel joint: body B (the wheel) moves along frame A's x axis on a suspension spring with
/// optional limits, spins about frame B's z axis with an optional motor, and can be steered
/// about the suspension axis by a spring with limits (Box3D `wheel_joint.c`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WheelJoint {
    /// The bodies and frames.
    pub base: JointBase3d,
    /// Whether the suspension spring is on (default true).
    pub enable_suspension_spring: bool,
    /// Suspension stiffness in Hz (default 1).
    pub suspension_hertz: f32,
    /// Suspension damping ratio (default 0.7).
    pub suspension_damping_ratio: f32,
    /// Whether the suspension limits are on.
    pub enable_suspension_limit: bool,
    /// Lower suspension travel, metres.
    pub lower_suspension_limit: f32,
    /// Upper suspension travel, metres.
    pub upper_suspension_limit: f32,
    /// Whether the spin motor is on.
    pub enable_spin_motor: bool,
    /// Largest spin motor torque, N·m.
    pub max_spin_torque: f32,
    /// Spin motor target speed, rad/s.
    pub spin_speed: f32,
    /// Whether steering is on; otherwise the wheel cannot turn about the suspension axis.
    pub enable_steering: bool,
    /// Steering spring stiffness in Hz (default 1).
    pub steering_hertz: f32,
    /// Steering spring damping ratio (default 0.7).
    pub steering_damping_ratio: f32,
    /// Steering target angle, radians.
    pub target_steering_angle: f32,
    /// Largest steering torque, N·m.
    pub max_steering_torque: f32,
    /// Whether the steering limits are on.
    pub enable_steering_limit: bool,
    /// Lower steering angle, radians.
    pub lower_steering_limit: f32,
    /// Upper steering angle, radians.
    pub upper_steering_limit: f32,
    /// Accumulated point-to-line impulse.
    pub linear_impulse: Vec2,
    /// Accumulated collinearity impulse.
    pub angular_impulse: Vec2,
    /// Accumulated spin motor impulse.
    pub spin_impulse: f32,
    /// Accumulated suspension spring impulse.
    pub suspension_spring_impulse: f32,
    /// Accumulated lower suspension limit impulse.
    pub lower_suspension_impulse: f32,
    /// Accumulated upper suspension limit impulse.
    pub upper_suspension_impulse: f32,
    /// Accumulated steering spring impulse.
    pub steering_spring_impulse: f32,
    /// Accumulated lower steering limit impulse.
    pub lower_steering_impulse: f32,
    /// Accumulated upper steering limit impulse.
    pub upper_steering_impulse: f32,
    #[serde(skip)]
    spin_mass: f32,
    #[serde(skip)]
    suspension_mass: f32,
    #[serde(skip)]
    steering_mass: f32,
    #[serde(skip)]
    suspension_softness: Softness3d,
    #[serde(skip)]
    steering_softness: Softness3d,
}

joint_base_builders!(WheelJoint, Wheel);

/// The steering angle's sine and cosine and the steering axis of a wheel with frame
/// rotations `matrix_a` and `matrix_b` (Box3D: steering about frame A's x axis).
fn steering(matrix_a: &Mat3, matrix_b: &Mat3) -> (f32, f32, Vec3) {
    let cs = matrix_b.z_axis.dot(matrix_a.z_axis);
    let ss = -matrix_b.z_axis.dot(matrix_a.y_axis);
    let den = cs * cs + ss * ss;
    let den = if den > 0.0 { 1.0 / den } else { 0.0 };
    let axis = matrix_b
        .z_axis
        .cross(matrix_a.y_axis * -cs - matrix_a.z_axis * ss)
        * den;
    (ss, cs, axis)
}

impl WheelJoint {
    /// A wheel joint between chassis `body_a` and wheel `body_b` at the given local frames,
    /// with Box3D's default 1 Hz suspension spring at damping ratio 0.7.
    #[must_use]
    pub fn new(
        body_a: impl Into<BodyId3d>,
        body_b: impl Into<BodyId3d>,
        local_frame_a: Frame3d,
        local_frame_b: Frame3d,
    ) -> Self {
        Self {
            base: JointBase3d::new(body_a, body_b, local_frame_a, local_frame_b),
            enable_suspension_spring: true,
            suspension_hertz: 1.0,
            suspension_damping_ratio: 0.7,
            enable_suspension_limit: false,
            lower_suspension_limit: 0.0,
            upper_suspension_limit: 0.0,
            enable_spin_motor: false,
            max_spin_torque: 0.0,
            spin_speed: 0.0,
            enable_steering: false,
            steering_hertz: 1.0,
            steering_damping_ratio: 0.7,
            target_steering_angle: 0.0,
            max_steering_torque: 0.0,
            enable_steering_limit: false,
            lower_steering_limit: 0.0,
            upper_steering_limit: 0.0,
            linear_impulse: Vec2::ZERO,
            angular_impulse: Vec2::ZERO,
            spin_impulse: 0.0,
            suspension_spring_impulse: 0.0,
            lower_suspension_impulse: 0.0,
            upper_suspension_impulse: 0.0,
            steering_spring_impulse: 0.0,
            lower_steering_impulse: 0.0,
            upper_steering_impulse: 0.0,
            spin_mass: 0.0,
            suspension_mass: 0.0,
            steering_mass: 0.0,
            suspension_softness: Softness3d::default(),
            steering_softness: Softness3d::default(),
        }
    }

    /// Sets the suspension spring.
    #[must_use]
    pub fn with_suspension(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_suspension_spring = true;
        self.suspension_hertz = hertz;
        self.suspension_damping_ratio = damping_ratio;
        self
    }

    /// Turns the suspension spring off.
    #[must_use]
    pub fn without_suspension_spring(mut self) -> Self {
        self.enable_suspension_spring = false;
        self
    }

    /// Enables suspension limits, ordered.
    #[must_use]
    pub fn with_suspension_limits(mut self, lower: f32, upper: f32) -> Self {
        self.enable_suspension_limit = true;
        self.lower_suspension_limit = lower.min(upper);
        self.upper_suspension_limit = lower.max(upper);
        self
    }

    /// Enables the spin motor towards `speed` rad/s with at most `max_torque`.
    #[must_use]
    pub fn with_spin_motor(mut self, speed: f32, max_torque: f32) -> Self {
        self.enable_spin_motor = true;
        self.spin_speed = speed;
        self.max_spin_torque = max_torque;
        self
    }

    /// Enables steering with a spring towards `target` radians, at most `max_torque`.
    #[must_use]
    pub fn with_steering(
        mut self,
        hertz: f32,
        damping_ratio: f32,
        max_torque: f32,
        target: f32,
    ) -> Self {
        self.enable_steering = true;
        self.steering_hertz = hertz;
        self.steering_damping_ratio = damping_ratio;
        self.max_steering_torque = max_torque;
        self.target_steering_angle = target;
        self
    }

    /// Enables steering limits, ordered.
    #[must_use]
    pub fn with_steering_limits(mut self, lower: f32, upper: f32) -> Self {
        self.enable_steering_limit = true;
        self.lower_steering_limit = lower.min(upper);
        self.upper_steering_limit = lower.max(upper);
        self
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        let (r_a, r_b) = (s.frame_a.position, s.frame_b.position);
        let matrix_a = Mat3::from_quat(s.frame_a.rotation);
        let matrix_b = Mat3::from_quat(s.frame_b.rotation);
        let i_sum = s.inv_inertia_sum();
        let suspension = matrix_a.x_axis;
        let r_an = r_a.cross(suspension);
        let r_bn = r_b.cross(suspension);
        self.suspension_mass = mass_of(
            s.inv_mass_a + s.inv_mass_b + r_an.dot(s.inv_i_a * r_an) + r_bn.dot(s.inv_i_b * r_bn),
        );
        self.suspension_softness =
            Softness3d::new(self.suspension_hertz, self.suspension_damping_ratio, h);
        self.steering_softness =
            Softness3d::new(self.steering_hertz, self.steering_damping_ratio, h);
        let spin = matrix_b.z_axis;
        self.spin_mass = mass_of(spin.dot(i_sum * spin));
        let (_, _, steering_axis) = steering(&matrix_a, &matrix_b);
        self.steering_mass = mass_of(steering_axis.dot(i_sum * steering_axis));
        if !self.enable_spin_motor {
            self.spin_impulse = 0.0;
        }
        if !self.enable_suspension_spring {
            self.suspension_spring_impulse = 0.0;
        }
        if !self.enable_suspension_limit {
            self.lower_suspension_impulse = 0.0;
            self.upper_suspension_impulse = 0.0;
        }
        if !self.enable_steering {
            self.steering_spring_impulse = 0.0;
        }
        if !self.enable_steering || !self.enable_steering_limit {
            self.lower_steering_impulse = 0.0;
            self.upper_steering_impulse = 0.0;
        }
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        let (r_a, r_b) = p.anchors(s);
        let d = p.separation(s, r_a, r_b);
        let (quat_a, quat_b) = p.frame_rotations(s);
        let matrix_a = Mat3::from_quat(quat_a);
        let matrix_b = Mat3::from_quat(quat_b);
        let (ax, ay, az) = (matrix_a.x_axis, matrix_a.y_axis, matrix_a.z_axis);
        let suspension = self.suspension_spring_impulse + self.lower_suspension_impulse
            - self.upper_suspension_impulse;
        let (ly, lz) = (self.linear_impulse.x, self.linear_impulse.y);
        let lever_a = d + r_a;
        let linear = ax * suspension + ay * ly + az * lz;
        let angular_a =
            lever_a.cross(ax) * suspension + lever_a.cross(ay) * ly + lever_a.cross(az) * lz;
        let angular_b = r_b.cross(ax) * suspension + r_b.cross(ay) * ly + r_b.cross(az) * lz;
        let spin_axis = matrix_b.z_axis;
        let angular = if self.enable_steering {
            let (_, _, steering_axis) = steering(&matrix_a, &matrix_b);
            let perp = spin_axis.cross(ax);
            let steer = self.steering_spring_impulse + self.lower_steering_impulse
                - self.upper_steering_impulse;
            perp * self.angular_impulse.x + spin_axis * self.spin_impulse + steering_axis * steer
        } else {
            // Box3D adds the spin impulse about frame A's z axis and again about the spin
            // axis here; the port keeps that.
            let rel = quat_a.conjugate() * quat_b;
            let (perp_x, perp_y) = perp_axes(quat_a, rel);
            az * self.spin_impulse
                + (perp_x * self.angular_impulse.x
                    + perp_y * self.angular_impulse.y
                    + spin_axis * self.spin_impulse)
        };
        p.apply(s, linear, angular_a + angular, angular_b + angular);
    }

    fn solve(&mut self, p: &mut Pair, step: Substep, use_bias: bool) {
        let s = self.base.step;
        let rotating = !s.fixed_rotation;
        let (r_a, r_b) = p.anchors(&s);
        let (quat_a, quat_b) = p.frame_rotations(&s);
        let matrix_a = Mat3::from_quat(quat_a);
        let matrix_b = Mat3::from_quat(quat_b);
        let ax = matrix_a.x_axis;
        let d = p.separation(&s, r_a, r_b);
        let s_ax = (d + r_a).cross(ax);
        let s_bx = r_b.cross(ax);
        let translation = ax.dot(d);
        let suspension_velocity =
            |p: &Pair| ax.dot(p.v_b - p.v_a) + s_bx.dot(p.w_b) - s_ax.dot(p.w_a);

        if self.enable_spin_motor && rotating {
            let spin_axis = matrix_b.z_axis;
            let cdot = (p.w_b - p.w_a).dot(spin_axis) - self.spin_speed;
            let max_impulse = step.h * self.max_spin_torque;
            let delta = solve_motor_row(&mut self.spin_impulse, self.spin_mass, cdot, max_impulse);
            p.apply_angular(&s, spin_axis * delta);
        }

        if self.enable_suspension_spring {
            // A real spring, applied in the relax pass too.
            let soft = self.suspension_softness;
            let delta = -soft.mass_scale
                * self.suspension_mass
                * (suspension_velocity(p) + soft.bias_rate * translation)
                - soft.impulse_scale * self.suspension_spring_impulse;
            self.suspension_spring_impulse += delta;
            p.apply(&s, ax * delta, s_ax * delta, s_bx * delta);
        }

        if self.enable_steering && rotating {
            self.solve_steering(p, (&matrix_a, &matrix_b), step, use_bias);
        }

        if self.enable_suspension_limit {
            let row = s.limit(
                translation - self.lower_suspension_limit,
                step.inv_h,
                use_bias,
            );
            let cdot = suspension_velocity(p);
            let mass = self.suspension_mass;
            let delta = solve_limit_row(&mut self.lower_suspension_impulse, mass, cdot, row);
            p.apply(&s, ax * delta, s_ax * delta, s_bx * delta);
            let row = s.limit(
                self.upper_suspension_limit - translation,
                step.inv_h,
                use_bias,
            );
            let cdot = -suspension_velocity(p);
            let delta = solve_limit_row(&mut self.upper_suspension_impulse, mass, cdot, row);
            p.apply(&s, ax * -delta, s_ax * -delta, s_bx * -delta);
        }

        if rotating {
            self.solve_collinearity(p, (quat_a, quat_b), (&matrix_a, &matrix_b), use_bias);
        }

        self.solve_point_to_line(p, &matrix_a, (r_a, r_b, d), use_bias);
    }

    /// The steering spring and steering limits about the steering axis.
    fn solve_steering(
        &mut self,
        p: &mut Pair,
        (matrix_a, matrix_b): (&Mat3, &Mat3),
        step: Substep,
        use_bias: bool,
    ) {
        let s = self.base.step;
        let (ss, cs, axis) = steering(matrix_a, matrix_b);
        let angle = math::atan2(ss, cs);
        {
            // A real spring, applied in the relax pass too.
            let soft = self.steering_softness;
            let cdot = axis.dot(p.w_b - p.w_a);
            let old = self.steering_spring_impulse;
            let c = angle - self.target_steering_angle;
            let delta = -soft.mass_scale * self.steering_mass * (cdot + soft.bias_rate * c)
                - soft.impulse_scale * old;
            let max_impulse = step.h * self.max_steering_torque;
            self.steering_spring_impulse = (old + delta).clamp(-max_impulse, max_impulse);
            p.apply_angular(&s, axis * (self.steering_spring_impulse - old));
        }
        if self.enable_steering_limit {
            let mass = self.steering_mass;
            let row = s.limit(angle - self.lower_steering_limit, step.inv_h, use_bias);
            let cdot = axis.dot(p.w_b - p.w_a);
            let delta = solve_limit_row(&mut self.lower_steering_impulse, mass, cdot, row);
            p.apply_angular(&s, axis * delta);
            let row = s.limit(self.upper_steering_limit - angle, step.inv_h, use_bias);
            let cdot = axis.dot(p.w_a - p.w_b);
            let delta = solve_limit_row(&mut self.upper_steering_impulse, mass, cdot, row);
            p.apply_angular(&s, -(axis * delta));
        }
    }

    /// Collinearity: with steering, the spin axis stays perpendicular to the suspension axis;
    /// without, it keeps its direction in frame A.
    fn solve_collinearity(
        &mut self,
        p: &mut Pair,
        (quat_a, quat_b): (Quat, Quat),
        (matrix_a, matrix_b): (&Mat3, &Mat3),
        use_bias: bool,
    ) {
        let s = self.base.step;
        let i_sum = s.inv_inertia_sum();
        if self.enable_steering {
            let (bias, mass_scale, impulse_scale) =
                s.rigid(matrix_a.x_axis.dot(matrix_b.z_axis), use_bias);
            let u = matrix_b.z_axis.cross(matrix_a.x_axis);
            let cdot = (p.w_b - p.w_a).dot(u);
            let perp_mass = mass_of(u.dot(i_sum * u));
            let delta =
                -mass_scale * perp_mass * (cdot + bias) - impulse_scale * self.angular_impulse.x;
            self.angular_impulse.x += delta;
            p.apply_angular(&s, u * delta);
        } else {
            let rel = quat_a.conjugate() * quat_b;
            let (bias, mass_scale, impulse_scale) = if use_bias {
                let rate = s.softness.bias_rate;
                (
                    (rate * rel.x, rate * rel.y),
                    s.softness.mass_scale,
                    s.softness.impulse_scale,
                )
            } else {
                ((0.0, 0.0), 1.0, 0.0)
            };
            let (perp_x, perp_y) = perp_axes(quat_a, rel);
            let k = axis_pair_mass(&i_sum, perp_x, perp_y);
            let w_rel = p.w_b - p.w_a;
            let delta = solve_pair(
                k,
                (w_rel.dot(perp_x), w_rel.dot(perp_y)),
                bias,
                (mass_scale, impulse_scale),
                &mut self.angular_impulse,
            );
            p.apply_angular(&s, perp_x * delta.x + perp_y * delta.y);
        }
    }

    /// Point to line: the wheel stays on the suspension axis.
    fn solve_point_to_line(
        &mut self,
        p: &mut Pair,
        matrix_a: &Mat3,
        (r_a, r_b, d): (Vec3, Vec3, Vec3),
        use_bias: bool,
    ) {
        let s = self.base.step;
        let (m_a, m_b) = s.masses();
        let (ay, az) = (matrix_a.y_axis, matrix_a.z_axis);
        let lever_a = d + r_a;
        let (s_ay, s_by) = (lever_a.cross(ay), r_b.cross(ay));
        let (s_az, s_bz) = (lever_a.cross(az), r_b.cross(az));
        let (bias, mass_scale, impulse_scale) = if use_bias {
            let rate = s.softness.bias_rate;
            (
                (rate * ay.dot(d), rate * az.dot(d)),
                s.softness.mass_scale,
                s.softness.impulse_scale,
            )
        } else {
            ((0.0, 0.0), 1.0, 0.0)
        };
        let v_rel = p.v_b + p.w_b.cross(r_b) - p.v_a - p.w_a.cross(lever_a);
        let kyy = m_a + m_b + s_ay.dot(s.inv_i_a * s_ay) + s_by.dot(s.inv_i_b * s_by);
        let kyz = s_ay.dot(s.inv_i_a * s_az) + s_by.dot(s.inv_i_b * s_bz);
        let kzz = m_a + m_b + s_az.dot(s.inv_i_a * s_az) + s_bz.dot(s.inv_i_b * s_bz);
        let delta = solve_pair(
            (kyy, kyz, kzz),
            (ay.dot(v_rel), az.dot(v_rel)),
            bias,
            (mass_scale, impulse_scale),
            &mut self.linear_impulse,
        );
        p.apply(
            &s,
            ay * delta.x + az * delta.y,
            s_ay * delta.x + s_az * delta.y,
            s_by * delta.x + s_bz * delta.y,
        );
    }
}

// ---------------------------------------------------------------- motor

/// Motor joint: drives body B's frame relative to body A's towards a linear and an angular
/// velocity with force and torque limits, and holds it with linear and angular springs
/// (Box3D `motor_joint.c`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotorJoint {
    /// The bodies and frames.
    pub base: JointBase3d,
    /// Target relative linear velocity, world space, m/s.
    pub linear_velocity: Vec3,
    /// Largest force of the velocity drive, N.
    pub max_velocity_force: f32,
    /// Target relative angular velocity, world space, rad/s.
    pub angular_velocity: Vec3,
    /// Largest torque of the velocity drive, N·m.
    pub max_velocity_torque: f32,
    /// Linear spring stiffness in Hz.
    pub linear_hertz: f32,
    /// Linear spring damping ratio.
    pub linear_damping_ratio: f32,
    /// Largest linear spring force, N.
    pub max_spring_force: f32,
    /// Angular spring stiffness in Hz.
    pub angular_hertz: f32,
    /// Angular spring damping ratio.
    pub angular_damping_ratio: f32,
    /// Largest angular spring torque, N·m.
    pub max_spring_torque: f32,
    /// Accumulated velocity drive impulse.
    pub linear_velocity_impulse: Vec3,
    /// Accumulated velocity drive angular impulse.
    pub angular_velocity_impulse: Vec3,
    /// Accumulated linear spring impulse.
    pub linear_spring_impulse: Vec3,
    /// Accumulated angular spring impulse.
    pub angular_spring_impulse: Vec3,
    #[serde(skip)]
    linear_spring: Softness3d,
    #[serde(skip)]
    angular_spring: Softness3d,
    #[serde(skip)]
    angular_mass: Mat3,
}

joint_base_builders!(MotorJoint, Motor);

impl MotorJoint {
    /// A motor joint between body A and body B with frames at the body origins and nothing
    /// driven yet.
    #[must_use]
    pub fn new(body_a: impl Into<BodyId3d>, body_b: impl Into<BodyId3d>) -> Self {
        Self {
            base: JointBase3d::new(body_a, body_b, Frame3d::IDENTITY, Frame3d::IDENTITY),
            linear_velocity: Vec3::ZERO,
            max_velocity_force: 0.0,
            angular_velocity: Vec3::ZERO,
            max_velocity_torque: 0.0,
            linear_hertz: 0.0,
            linear_damping_ratio: 0.0,
            max_spring_force: 0.0,
            angular_hertz: 0.0,
            angular_damping_ratio: 0.0,
            max_spring_torque: 0.0,
            linear_velocity_impulse: Vec3::ZERO,
            angular_velocity_impulse: Vec3::ZERO,
            linear_spring_impulse: Vec3::ZERO,
            angular_spring_impulse: Vec3::ZERO,
            linear_spring: Softness3d::default(),
            angular_spring: Softness3d::default(),
            angular_mass: Mat3::ZERO,
        }
    }

    /// Drives the relative velocities towards `linear` and `angular` with at most
    /// `max_force` and `max_torque`.
    #[must_use]
    pub fn with_velocity(
        mut self,
        linear: Vec3,
        max_force: f32,
        angular: Vec3,
        max_torque: f32,
    ) -> Self {
        self.linear_velocity = linear;
        self.max_velocity_force = max_force;
        self.angular_velocity = angular;
        self.max_velocity_torque = max_torque;
        self
    }

    /// Holds the frames' origins together with a spring of at most `max_force`.
    #[must_use]
    pub fn with_linear_spring(mut self, hertz: f32, damping_ratio: f32, max_force: f32) -> Self {
        self.linear_hertz = hertz;
        self.linear_damping_ratio = damping_ratio;
        self.max_spring_force = max_force;
        self
    }

    /// Holds the frames' rotations together with a spring of at most `max_torque`.
    #[must_use]
    pub fn with_angular_spring(mut self, hertz: f32, damping_ratio: f32, max_torque: f32) -> Self {
        self.angular_hertz = hertz;
        self.angular_damping_ratio = damping_ratio;
        self.max_spring_torque = max_torque;
        self
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        self.linear_spring = Softness3d::new(self.linear_hertz, self.linear_damping_ratio, h);
        self.angular_spring = Softness3d::new(self.angular_hertz, self.angular_damping_ratio, h);
        self.angular_mass = math::invert(&s.inv_inertia_sum());
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        let (r_a, r_b) = p.anchors(s);
        let linear = self.linear_velocity_impulse + self.linear_spring_impulse;
        let angular = self.angular_velocity_impulse + self.angular_spring_impulse;
        p.apply(
            s,
            linear,
            r_a.cross(linear) + angular,
            r_b.cross(linear) + angular,
        );
    }

    fn solve(&mut self, p: &mut Pair, step: Substep) {
        let s = self.base.step;
        if self.max_spring_torque > 0.0 && self.angular_hertz > 0.0 {
            let (quat_a, quat_b) = p.frame_rotations(&s);
            let rel = quat_a.conjugate() * quat_b;
            let c = -(quat_a * math::delta_quat_to_rotation(rel, Quat::IDENTITY));
            let spring = self.angular_spring;
            let cdot = p.w_b - p.w_a;
            let old = self.angular_spring_impulse;
            let impulse = -(self.angular_mass * (cdot + c * spring.bias_rate)) * spring.mass_scale
                - old * spring.impulse_scale;
            self.angular_spring_impulse =
                clamp_length(old + impulse, step.h * self.max_spring_torque);
            p.apply_angular(&s, self.angular_spring_impulse - old);
        }

        if self.max_velocity_torque > 0.0 {
            let cdot = (p.w_b - p.w_a) - self.angular_velocity;
            let old = self.angular_velocity_impulse;
            self.angular_velocity_impulse = clamp_length(
                old - self.angular_mass * cdot,
                step.h * self.max_velocity_torque,
            );
            p.apply_angular(&s, self.angular_velocity_impulse - old);
        }

        let (r_a, r_b) = p.anchors(&s);
        if self.max_spring_force > 0.0 && self.linear_hertz > 0.0 {
            let c = p.separation(&s, r_a, r_b);
            let spring = self.linear_spring;
            let cdot = p.anchor_velocity(r_a, r_b);
            let k = math::point_mass(s.masses(), &s.inv_i_a, &s.inv_i_b, r_a, r_b);
            let b = math::solve3(&k, cdot + c * spring.bias_rate);
            let old = self.linear_spring_impulse;
            let impulse = b * -spring.mass_scale - old * spring.impulse_scale;
            self.linear_spring_impulse =
                clamp_length(old + impulse, step.h * self.max_spring_force);
            p.apply_linear(&s, self.linear_spring_impulse - old, r_a, r_b);
        }

        if self.max_velocity_force > 0.0 {
            let cdot = p.anchor_velocity(r_a, r_b) - self.linear_velocity;
            let k = math::point_mass(s.masses(), &s.inv_i_a, &s.inv_i_b, r_a, r_b);
            let old = self.linear_velocity_impulse;
            self.linear_velocity_impulse = clamp_length(
                old - math::solve3(&k, cdot),
                step.h * self.max_velocity_force,
            );
            p.apply_linear(&s, self.linear_velocity_impulse - old, r_a, r_b);
        }
    }
}

// ---------------------------------------------------------------- parallel

/// Parallel joint: a spring keeping frame B's z axis parallel to frame A's, which leaves the
/// spin about it free; useful to keep a body upright (Box3D `parallel_joint.c`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelJoint {
    /// The bodies and frames.
    pub base: JointBase3d,
    /// Spring stiffness in Hz (default 1).
    pub hertz: f32,
    /// Spring damping ratio (default 1).
    pub damping_ratio: f32,
    /// Largest spring torque, N·m (default unlimited).
    pub max_torque: f32,
    /// Accumulated impulse about the two perpendicular axes.
    pub perp_impulse: Vec2,
    #[serde(skip)]
    perp_axis_x: Vec3,
    #[serde(skip)]
    perp_axis_y: Vec3,
    #[serde(skip)]
    softness: Softness3d,
}

joint_base_builders!(ParallelJoint, Parallel);

impl ParallelJoint {
    /// A parallel joint between body A and body B with frames aligned with the bodies and
    /// Box3D's default 1 Hz spring.
    #[must_use]
    pub fn new(body_a: impl Into<BodyId3d>, body_b: impl Into<BodyId3d>) -> Self {
        Self {
            base: JointBase3d::new(body_a, body_b, Frame3d::IDENTITY, Frame3d::IDENTITY),
            hertz: 1.0,
            damping_ratio: 1.0,
            max_torque: f32::MAX,
            perp_impulse: Vec2::ZERO,
            perp_axis_x: Vec3::ZERO,
            perp_axis_y: Vec3::ZERO,
            softness: Softness3d::default(),
        }
    }

    /// Sets the spring.
    #[must_use]
    pub fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Limits the spring torque.
    #[must_use]
    pub fn with_max_torque(mut self, max_torque: f32) -> Self {
        self.max_torque = max_torque;
        self
    }

    fn prepare(&mut self, h: f32) {
        let s = &self.base.step;
        let rel = s.frame_a.rotation.conjugate() * s.frame_b.rotation;
        (self.perp_axis_x, self.perp_axis_y) = perp_axes(s.frame_a.rotation, rel);
        self.softness = Softness3d::new(self.hertz, self.damping_ratio, h);
    }

    fn warm_start_pair(&self, p: &mut Pair) {
        let s = &self.base.step;
        p.apply_angular(
            s,
            self.perp_axis_x * self.perp_impulse.x + self.perp_axis_y * self.perp_impulse.y,
        );
    }

    fn solve(&mut self, p: &mut Pair, step: Substep) {
        let s = self.base.step;
        if s.fixed_rotation || self.max_torque <= 0.0 {
            return;
        }
        let (quat_a, quat_b) = p.frame_rotations(&s);
        let rel = quat_a.conjugate() * quat_b;
        let rate = self.softness.bias_rate;
        let (perp_x, perp_y) = perp_axes(quat_a, rel);
        self.perp_axis_x = perp_x;
        self.perp_axis_y = perp_y;
        let k = axis_pair_mass(&s.inv_inertia_sum(), perp_x, perp_y);
        let w_rel = p.w_b - p.w_a;
        let old = self.perp_impulse;
        solve_pair(
            k,
            (w_rel.dot(perp_x), w_rel.dot(perp_y)),
            (rate * rel.x, rate * rel.y),
            (self.softness.mass_scale, self.softness.impulse_scale),
            &mut self.perp_impulse,
        );
        let max_impulse = step.h * self.max_torque;
        if self.perp_impulse.length_squared() > max_impulse * max_impulse {
            self.perp_impulse *= max_impulse / self.perp_impulse.length();
        }
        let delta = self.perp_impulse - old;
        p.apply_angular(&s, perp_x * delta.x + perp_y * delta.y);
    }
}

// ---------------------------------------------------------------- filter

/// Filter joint: no constraint; its two bodies just do not collide with each other (Box3D
/// `b3CreateFilterJoint`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterJoint {
    /// The bodies; `collide_connected` is ignored.
    pub base: JointBase3d,
}

impl FilterJoint {
    /// Keeps body A and body B from colliding with each other.
    #[must_use]
    pub fn new(body_a: impl Into<BodyId3d>, body_b: impl Into<BodyId3d>) -> Self {
        Self {
            base: JointBase3d::new(body_a, body_b, Frame3d::IDENTITY, Frame3d::IDENTITY),
        }
    }
}

// ---------------------------------------------------------------- dispatch

/// Unified 3D joint enum dispatching to concrete joint constraint implementations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Joint {
    /// Spherical (ball-and-socket) joint.
    Spherical(SphericalJoint),
    /// Revolute (hinge) joint.
    Revolute(RevoluteJoint),
    /// Prismatic (slider) joint.
    Prismatic(PrismaticJoint),
    /// Distance joint.
    Distance(DistanceJoint),
    /// Weld joint.
    Weld(WeldJoint),
    /// Wheel joint.
    Wheel(WheelJoint),
    /// Motor joint.
    Motor(MotorJoint),
    /// Parallel joint.
    Parallel(ParallelJoint),
    /// Filter joint.
    Filter(FilterJoint),
}

impl Joint {
    /// The bodies, frames and tuning shared by every joint type.
    #[must_use]
    pub fn base(&self) -> &JointBase3d {
        match self {
            Self::Spherical(j) => &j.base,
            Self::Revolute(j) => &j.base,
            Self::Prismatic(j) => &j.base,
            Self::Distance(j) => &j.base,
            Self::Weld(j) => &j.base,
            Self::Wheel(j) => &j.base,
            Self::Motor(j) => &j.base,
            Self::Parallel(j) => &j.base,
            Self::Filter(j) => &j.base,
        }
    }

    /// The shared base, mutably.
    pub fn base_mut(&mut self) -> &mut JointBase3d {
        match self {
            Self::Spherical(j) => &mut j.base,
            Self::Revolute(j) => &mut j.base,
            Self::Prismatic(j) => &mut j.base,
            Self::Distance(j) => &mut j.base,
            Self::Weld(j) => &mut j.base,
            Self::Wheel(j) => &mut j.base,
            Self::Motor(j) => &mut j.base,
            Self::Parallel(j) => &mut j.base,
            Self::Filter(j) => &mut j.base,
        }
    }

    /// Sets the force threshold in Newtons for joint events.
    #[must_use]
    pub fn with_force_threshold(mut self, threshold: f32) -> Self {
        self.base_mut().force_threshold = threshold;
        self
    }

    /// Calls `mix` with each accumulated impulse, for the world's state hash.
    pub fn hash_impulses(&self, mix: &mut dyn FnMut(f32)) {
        match self {
            Self::Spherical(j) => {
                for v in [
                    j.linear_impulse.x,
                    j.linear_impulse.y,
                    j.linear_impulse.z,
                    j.spring_impulse.x,
                    j.spring_impulse.y,
                    j.spring_impulse.z,
                    j.motor_impulse.x,
                    j.motor_impulse.y,
                    j.motor_impulse.z,
                    j.lower_twist_impulse,
                    j.upper_twist_impulse,
                    j.swing_impulse,
                ] {
                    mix(v);
                }
            }
            Self::Revolute(j) => {
                for v in [
                    j.linear_impulse.x,
                    j.linear_impulse.y,
                    j.linear_impulse.z,
                    j.perp_impulse.x,
                    j.perp_impulse.y,
                    j.spring_impulse,
                    j.motor_impulse,
                    j.lower_impulse,
                    j.upper_impulse,
                ] {
                    mix(v);
                }
            }
            Self::Prismatic(j) => {
                for v in [
                    j.perp_impulse.x,
                    j.perp_impulse.y,
                    j.angular_impulse.x,
                    j.angular_impulse.y,
                    j.angular_impulse.z,
                    j.spring_impulse,
                    j.motor_impulse,
                    j.lower_impulse,
                    j.upper_impulse,
                ] {
                    mix(v);
                }
            }
            Self::Distance(j) => {
                for v in [j.impulse, j.lower_impulse, j.upper_impulse, j.motor_impulse] {
                    mix(v);
                }
            }
            Self::Weld(j) => {
                for v in [
                    j.linear_impulse.x,
                    j.linear_impulse.y,
                    j.linear_impulse.z,
                    j.angular_impulse.x,
                    j.angular_impulse.y,
                    j.angular_impulse.z,
                ] {
                    mix(v);
                }
            }
            Self::Wheel(j) => {
                for v in [
                    j.linear_impulse.x,
                    j.linear_impulse.y,
                    j.angular_impulse.x,
                    j.angular_impulse.y,
                    j.spin_impulse,
                    j.suspension_spring_impulse,
                    j.lower_suspension_impulse,
                    j.upper_suspension_impulse,
                    j.steering_spring_impulse,
                    j.lower_steering_impulse,
                    j.upper_steering_impulse,
                ] {
                    mix(v);
                }
            }
            Self::Motor(j) => {
                for v in [
                    j.linear_velocity_impulse.x,
                    j.linear_velocity_impulse.y,
                    j.linear_velocity_impulse.z,
                    j.angular_velocity_impulse.x,
                    j.angular_velocity_impulse.y,
                    j.angular_velocity_impulse.z,
                    j.linear_spring_impulse.x,
                    j.linear_spring_impulse.y,
                    j.linear_spring_impulse.z,
                    j.angular_spring_impulse.x,
                    j.angular_spring_impulse.y,
                    j.angular_spring_impulse.z,
                ] {
                    mix(v);
                }
            }
            Self::Parallel(j) => {
                for v in [j.perp_impulse.x, j.perp_impulse.y] {
                    mix(v);
                }
            }
            Self::Filter(_) => {}
        }
    }

    /// Sets the torque threshold in N·m for joint events.
    #[must_use]
    pub fn with_torque_threshold(mut self, threshold: f32) -> Self {
        self.base_mut().torque_threshold = threshold;
        self
    }

    /// Returns the index of body A connected by this joint.
    #[must_use]
    pub fn body_a(&self) -> u32 {
        self.base().body_a
    }

    /// Returns the index of body B connected by this joint.
    #[must_use]
    pub fn body_b(&self) -> u32 {
        self.base().body_b
    }

    /// The joint frame's origin on body A, in A's coordinates.
    #[must_use]
    pub fn local_anchor_a(&self) -> Vec3 {
        self.base().local_frame_a.position
    }

    /// The joint frame's origin on body B, in B's coordinates.
    #[must_use]
    pub fn local_anchor_b(&self) -> Vec3 {
        self.base().local_frame_b.position
    }

    /// Whether the joined bodies collide with each other; never for a filter joint.
    #[must_use]
    pub fn collide_connected(&self) -> bool {
        !matches!(self, Self::Filter(_)) && self.base().collide_connected
    }

    /// Sets whether the joined bodies collide with each other (Box3D `collideConnected`, off
    /// by default).
    #[must_use]
    pub fn with_collide_connected(mut self, collide: bool) -> Self {
        self.base_mut().collide_connected = collide;
        self
    }

    /// Prepares the joint for a step of substep `h`: frames, masses and softness from the
    /// bodies' poses and world inverse inertias `inv_i` at the start of the step (Box3D
    /// `b3PrepareJoint`).
    pub(crate) fn prepare(&mut self, bodies: &[RigidBody], inv_i: &[Mat3], h: f32) {
        let base = self.base_mut();
        let (a, b) = (base.body_a as usize, base.body_b as usize);
        let (body_a, body_b) = (&bodies[a], &bodies[b]);
        let hertz = base.constraint_hertz.min(0.25 / h);
        base.step = JointStep {
            inv_mass_a: body_a.inv_mass,
            inv_mass_b: body_b.inv_mass,
            inv_i_a: inv_i[a],
            inv_i_b: inv_i[b],
            frame_a: Frame3d::new(
                body_a.rotation * (base.local_frame_a.position - body_a.local_center),
                body_a.rotation * base.local_frame_a.rotation,
            ),
            frame_b: Frame3d::new(
                body_b.rotation * (base.local_frame_b.position - body_b.local_center),
                body_b.rotation * base.local_frame_b.rotation,
            ),
            delta_center: body_b.center_of_mass() - body_a.center_of_mass(),
            softness: Softness3d::new(hertz, base.constraint_damping_ratio, h),
            fixed_rotation: math::det(&(inv_i[a] + inv_i[b])) < MIN_DETERMINANT,
        };
        match self {
            Self::Spherical(j) => j.prepare(h),
            Self::Revolute(j) => j.prepare(h),
            Self::Prismatic(j) => j.prepare(h),
            Self::Distance(j) => j.prepare(h),
            Self::Weld(j) => j.prepare(h),
            Self::Wheel(j) => j.prepare(h),
            Self::Motor(j) => j.prepare(h),
            Self::Parallel(j) => j.prepare(h),
            Self::Filter(_) => {}
        }
    }

    /// Applies the accumulated impulses (Box3D `b3WarmStartJoint`).
    pub(crate) fn warm_start_step(&self, bodies: &mut [RigidBody], motions: &[BodyMotion3d]) {
        let mut p = Pair::load(bodies, motions, self.base());
        match self {
            Self::Spherical(j) => j.warm_start_pair(&mut p),
            Self::Revolute(j) => j.warm_start_pair(&mut p),
            Self::Prismatic(j) => j.warm_start_pair(&mut p),
            Self::Distance(j) => j.warm_start_pair(&mut p),
            Self::Weld(j) => j.warm_start_pair(&mut p),
            Self::Wheel(j) => j.warm_start_pair(&mut p),
            Self::Motor(j) => j.warm_start_pair(&mut p),
            Self::Parallel(j) => j.warm_start_pair(&mut p),
            Self::Filter(_) => return,
        }
        p.store(bodies, self.base());
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    pub(crate) unsafe fn warm_start_step_shared(
        &self,
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        motions: &crate::parallel::Shared<'_, BodyMotion3d>,
    ) {
        // SAFETY: forwarded from caller.
        let (mut p, dyn_a, dyn_b) = unsafe { Pair::load_shared(bodies, motions, self.base()) };
        match self {
            Self::Spherical(j) => j.warm_start_pair(&mut p),
            Self::Revolute(j) => j.warm_start_pair(&mut p),
            Self::Prismatic(j) => j.warm_start_pair(&mut p),
            Self::Distance(j) => j.warm_start_pair(&mut p),
            Self::Weld(j) => j.warm_start_pair(&mut p),
            Self::Wheel(j) => j.warm_start_pair(&mut p),
            Self::Motor(j) => j.warm_start_pair(&mut p),
            Self::Parallel(j) => j.warm_start_pair(&mut p),
            Self::Filter(_) => return,
        }
        // SAFETY: forwarded from caller.
        unsafe { p.store_shared(bodies, self.base(), dyn_a, dyn_b) };
    }

    /// Solves the joint for one substep, with position correction (`use_bias`) in the biased
    /// solve and without it in the relax pass (Box3D `b3SolveJoint`).
    pub(crate) fn solve_step(
        &mut self,
        bodies: &mut [RigidBody],
        motions: &[BodyMotion3d],
        step: Substep,
        use_bias: bool,
    ) {
        let mut p = Pair::load(bodies, motions, self.base());
        match self {
            Self::Spherical(j) => j.solve(&mut p, step, use_bias),
            Self::Revolute(j) => j.solve(&mut p, step, use_bias),
            Self::Prismatic(j) => j.solve(&mut p, step, use_bias),
            Self::Distance(j) => j.solve(&mut p, step, use_bias),
            Self::Weld(j) => j.solve(&mut p, use_bias),
            Self::Wheel(j) => j.solve(&mut p, step, use_bias),
            Self::Motor(j) => j.solve(&mut p, step),
            Self::Parallel(j) => j.solve(&mut p, step),
            Self::Filter(_) => return,
        }
        p.store(bodies, self.base());
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    pub(crate) unsafe fn solve_step_shared(
        &mut self,
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        motions: &crate::parallel::Shared<'_, BodyMotion3d>,
        step: Substep,
        use_bias: bool,
    ) {
        // SAFETY: forwarded from caller.
        let (mut p, dyn_a, dyn_b) = unsafe { Pair::load_shared(bodies, motions, self.base()) };
        match self {
            Self::Spherical(j) => j.solve(&mut p, step, use_bias),
            Self::Revolute(j) => j.solve(&mut p, step, use_bias),
            Self::Prismatic(j) => j.solve(&mut p, step, use_bias),
            Self::Distance(j) => j.solve(&mut p, step, use_bias),
            Self::Weld(j) => j.solve(&mut p, use_bias),
            Self::Wheel(j) => j.solve(&mut p, step, use_bias),
            Self::Motor(j) => j.solve(&mut p, step),
            Self::Parallel(j) => j.solve(&mut p, step),
            Self::Filter(_) => return,
        }
        // SAFETY: forwarded from caller.
        unsafe { p.store_shared(bodies, self.base(), dyn_a, dyn_b) };
    }

    /// Applies the accumulated impulses to two bodies at their current poses, as the solver's
    /// warm start does at the start of a step.
    pub fn warm_start(&self, body_a: &mut RigidBody, body_b: &mut RigidBody) {
        let mut joint = self.clone();
        let base = joint.base_mut();
        base.body_a = 0;
        base.body_b = 1;
        let mut pair = [body_a.clone(), body_b.clone()];
        let inv_i = [pair[0].world_inv_inertia(), pair[1].world_inv_inertia()];
        joint.prepare(&pair, &inv_i, 1.0 / 240.0);
        joint.warm_start_step(&mut pair, &[BodyMotion3d::NONE; 2]);
        let [a, b] = pair;
        body_a.linear_velocity = a.linear_velocity;
        body_a.angular_velocity = a.angular_velocity;
        body_b.linear_velocity = b.linear_velocity;
        body_b.angular_velocity = b.angular_velocity;
    }

    /// The force the joint applied to body B during the last substep, in newtons (Box3D
    /// `b3Joint_GetConstraintForce`), given that substep's inverse `inv_h`.
    #[must_use]
    pub fn constraint_force(&self, inv_h: f32) -> Vec3 {
        let impulse = match self {
            Self::Spherical(j) => j.linear_impulse,
            Self::Revolute(j) => j.linear_impulse,
            Self::Weld(j) => j.linear_impulse,
            Self::Prismatic(j) => {
                let axial = j.spring_impulse + j.motor_impulse + j.lower_impulse - j.upper_impulse;
                j.joint_axis * axial
                    + j.perp_axis_y * j.perp_impulse.x
                    + j.perp_axis_z * j.perp_impulse.y
            }
            Self::Distance(j) => {
                j.axis * (j.impulse + j.lower_impulse - j.upper_impulse + j.motor_impulse)
            }
            Self::Wheel(j) => {
                let m = Mat3::from_quat(j.base.step.frame_a.rotation);
                let suspension = j.suspension_spring_impulse + j.lower_suspension_impulse
                    - j.upper_suspension_impulse;
                m.x_axis * suspension
                    + m.y_axis * j.linear_impulse.x
                    + m.z_axis * j.linear_impulse.y
            }
            Self::Motor(j) => j.linear_velocity_impulse + j.linear_spring_impulse,
            Self::Parallel(_) | Self::Filter(_) => Vec3::ZERO,
        };
        impulse * inv_h
    }

    /// The torque the joint applied to body B during the last substep, in N·m (Box3D
    /// `b3Joint_GetConstraintTorque`), given that substep's inverse `inv_h`.
    #[must_use]
    pub fn constraint_torque(&self, inv_h: f32) -> Vec3 {
        let impulse = match self {
            Self::Spherical(j) => {
                let twist_axis = j.base.step.frame_b.rotation * Vec3::Z;
                j.spring_impulse
                    + j.motor_impulse
                    + twist_axis * (j.lower_twist_impulse - j.upper_twist_impulse)
                    + j.swing_axis * j.swing_impulse
            }
            Self::Revolute(j) => {
                j.perp_axis_x * j.perp_impulse.x
                    + j.perp_axis_y * j.perp_impulse.y
                    + j.rotation_axis_z * j.axial_impulse()
            }
            Self::Prismatic(j) => j.angular_impulse,
            Self::Weld(j) => j.angular_impulse,
            Self::Wheel(j) => j.base.step.frame_b.rotation * Vec3::Z * j.spin_impulse,
            Self::Motor(j) => j.angular_velocity_impulse + j.angular_spring_impulse,
            Self::Parallel(j) => {
                j.perp_axis_x * j.perp_impulse.x + j.perp_axis_y * j.perp_impulse.y
            }
            Self::Distance(_) | Self::Filter(_) => Vec3::ZERO,
        };
        impulse * inv_h
    }

    /// How far the joint's position constraint is from holding, in metres (Box3D
    /// `b3Joint_GetLinearSeparation`), for bodies at their current poses.
    #[must_use]
    pub fn linear_separation(&self, body_a: &RigidBody, body_b: &RigidBody) -> f32 {
        let (frame_a, frame_b) = self.base().world_frames(body_a, body_b);
        let dp = frame_b.position - frame_a.position;
        let along_axis = |limit: Option<(f32, f32)>| {
            let axis = frame_a.rotation * Vec3::X;
            let along = dp.dot(axis);
            let perpendicular = (dp - axis * along).length();
            let beyond = limit.map_or(0.0, |(lower, upper)| {
                (lower - along).max(along - upper).max(0.0)
            });
            (perpendicular * perpendicular + beyond * beyond).sqrt()
        };
        match self {
            Self::Distance(j) => {
                let length = dp.length();
                if !j.enable_spring {
                    (length - j.length).abs()
                } else if j.enable_limit {
                    (j.min_length - length).max(length - j.max_length).max(0.0)
                } else {
                    0.0
                }
            }
            Self::Prismatic(j) => along_axis(
                j.enable_limit
                    .then_some((j.lower_translation, j.upper_translation)),
            ),
            Self::Wheel(j) => along_axis(
                j.enable_suspension_limit
                    .then_some((j.lower_suspension_limit, j.upper_suspension_limit)),
            ),
            Self::Revolute(_) | Self::Spherical(_) => dp.length(),
            Self::Weld(j) if j.linear_hertz == 0.0 => dp.length(),
            Self::Weld(_) | Self::Motor(_) | Self::Parallel(_) | Self::Filter(_) => 0.0,
        }
    }

    /// How far the joint's rotation constraint is from holding, in radians (Box3D
    /// `b3Joint_GetAngularSeparation`), for bodies at their current poses.
    #[must_use]
    pub fn angular_separation(&self, body_a: &RigidBody, body_b: &RigidBody) -> f32 {
        let rel = relative_rotation(self.base(), body_a, body_b);
        let without_twist = Quat::from_xyzw(rel.x, rel.y, 0.0, rel.w);
        match self {
            Self::Parallel(_) => math::quat_angle(without_twist),
            Self::Prismatic(_) => math::quat_angle(rel),
            Self::Revolute(j) => {
                let angle = math::twist_angle(rel);
                if j.enable_limit && (angle < j.lower_angle || j.upper_angle < angle) {
                    math::quat_angle(rel)
                } else {
                    math::quat_angle(without_twist)
                }
            }
            Self::Spherical(j) => {
                let mut sum = 0.0;
                if j.enable_cone_limit {
                    sum += (math::swing_angle(rel) - j.cone_angle).max(0.0);
                }
                if j.enable_twist_limit {
                    let twist = math::twist_angle(rel);
                    sum += (j.lower_twist_angle - twist).max(0.0)
                        + (twist - j.upper_twist_angle).max(0.0);
                }
                sum
            }
            Self::Weld(j) if j.angular_hertz == 0.0 => math::quat_angle(rel),
            Self::Weld(_)
            | Self::Distance(_)
            | Self::Motor(_)
            | Self::Wheel(_)
            | Self::Filter(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wheel_and_base() -> Vec<RigidBody> {
        let base = RigidBody::new(BodyType::Static, Vec3::ZERO);
        let wheel = RigidBody::new(BodyType::Dynamic, Vec3::ZERO);
        vec![base, wheel]
    }

    fn step_once(joint: &mut Joint, bodies: &mut [RigidBody]) {
        let inv_i: Vec<Mat3> = bodies.iter().map(RigidBody::world_inv_inertia).collect();
        let motions = vec![BodyMotion3d::NONE; bodies.len()];
        joint.prepare(bodies, &inv_i, 0.25);
        joint.warm_start_step(bodies, &motions);
        joint.solve_step(
            bodies,
            &motions,
            Substep {
                h: 0.25,
                inv_h: 4.0,
            },
            true,
        );
    }

    #[test]
    fn switching_a_motor_off_drops_its_impulse() {
        let mut bodies = wheel_and_base();
        let mut joint = Joint::Revolute(
            RevoluteJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Z).with_motor(10.0, 1.0),
        );
        step_once(&mut joint, &mut bodies);
        let Joint::Revolute(hinge) = &mut joint else {
            unreachable!()
        };
        assert!(hinge.motor_impulse > 0.0);
        hinge.enable_motor = false;
        let spin = bodies[1].angular_velocity;
        step_once(&mut joint, &mut bodies);
        let Joint::Revolute(hinge) = &joint else {
            unreachable!()
        };
        assert_eq!(hinge.motor_impulse, 0.0);
        assert_eq!(
            bodies[1].angular_velocity.z, spin.z,
            "no stale motor impulse"
        );
    }

    #[test]
    fn switching_a_limit_off_drops_its_impulses() {
        let mut bodies = wheel_and_base();
        bodies[1].linear_velocity = Vec3::new(5.0, 0.0, 0.0);
        let mut joint = Joint::Prismatic(
            PrismaticJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::X).with_limits(-0.1, 0.0),
        );
        step_once(&mut joint, &mut bodies);
        let Joint::Prismatic(slider) = &mut joint else {
            unreachable!()
        };
        assert!(slider.upper_impulse > 0.0);
        slider.enable_limit = false;
        step_once(&mut joint, &mut bodies);
        let Joint::Prismatic(slider) = &joint else {
            unreachable!()
        };
        assert_eq!((slider.lower_impulse, slider.upper_impulse), (0.0, 0.0));
    }

    #[test]
    fn frames_from_world_coincide_with_the_world_frame() {
        let body = RigidBody::new(BodyType::Dynamic, Vec3::new(1.0, 2.0, 3.0))
            .with_rotation(Quat::from_rotation_y(0.4));
        let world = Frame3d::new(Vec3::new(0.0, 1.0, 0.0), Quat::from_rotation_x(0.3));
        let local = Frame3d::from_world(&body, world);
        let back = body.position + body.rotation * local.position;
        assert!((back - world.position).length() < 1e-6);
        assert!((body.rotation * local.rotation).dot(world.rotation).abs() > 0.999_999);
    }
}
