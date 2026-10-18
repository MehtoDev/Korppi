//! 2D joints, ported from Box2D v3.2 (`joint.c` and the joint files).
//!
//! A joint joins a local frame on body A to one on body B (Box2D `localFrameA`, `localFrameB`):
//! a position relative to the body origin and a rotation, which takes the place of a reference
//! angle or a slider axis (the frame's x-axis). The world prepares each awake joint once per
//! step, recording the frames relative to the bodies' centres of mass in world orientation, the
//! distance between the centres and the effective masses (Box2D `b2PrepareJointsTask`). Each
//! substep then warm starts it and solves it on the bodies' velocities and their motion since
//! the step began (Box2D `b2BodyState`), with position correction, and after the positions moved
//! relaxes it without, as the contacts.
//!
//! Every joint with rigid rows (the revolute pin, the weld locks, the slider guides, limits, a
//! distance rod) softens them with the joint's `constraint_hertz` and `constraint_damping_ratio`
//! (60 Hz and 2 by default, capped at a quarter of the substep rate); its own springs have their
//! own stiffness. Joints of the same pair of bodies do not make them collide unless their
//! [`JointInstance2d`](crate::world::JointInstance2d) says so; a [`Joint2d::Filter`] does nothing
//! else.

use crate::body::RigidBody2d;
pub(crate) use crate::body_state::BodyState2d;
use crate::math::{
    clamp, cross, cross_sv, inv_mul_rot, left_perp, normalize, relative_angle, rot_angle,
    spring_damper, unwind_angle, Mat22,
};
use korppi_ecs::Component;
use korppi_math::{Rot2, Transform2d, Vec2};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Largest revolute limit magnitude, in radians (Box2D clamps limits to 0.99π).
pub const MAX_REVOLUTE_LIMIT: f32 = 0.99 * PI;

/// Default stiffness of a joint's rigid rows, in Hertz (Box2D `b2JointDef::constraintHertz`).
pub const DEFAULT_CONSTRAINT_HERTZ: f32 = 60.0;

/// Default damping ratio of a joint's rigid rows (Box2D `b2JointDef::constraintDampingRatio`).
pub const DEFAULT_CONSTRAINT_DAMPING_RATIO: f32 = 2.0;

/// The longest length a distance joint takes, in metres (Box2D `B2_HUGE`).
pub const HUGE_LENGTH: f32 = 100_000.0;

/// Softness parameters of a soft constraint row (Box2D `b2Softness`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Softness2d {
    /// Velocity bias per unit of constraint error.
    pub bias_rate: f32,
    /// Multiplier of the effective mass.
    pub mass_scale: f32,
    /// Multiplier of the accumulated impulse.
    pub impulse_scale: f32,
}

impl Softness2d {
    /// The softness of a row at `hertz` and damping ratio `zeta` for substep `h` (Box2D
    /// `b2MakeSoft`); `hertz = 0` gives a row that does nothing.
    #[must_use]
    pub fn new(hertz: f32, zeta: f32, h: f32) -> Self {
        if hertz == 0.0 {
            return Self::default();
        }
        let omega = 2.0 * PI * hertz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }

    /// A rigid row without position correction.
    pub const RIGID: Self = Self {
        bias_rate: 0.0,
        mass_scale: 1.0,
        impulse_scale: 0.0,
    };

    /// The bias, mass scale and impulse scale of a rigid row: this softness with bias, rigid
    /// without (the relax pass).
    const fn rigid_row(self, use_bias: bool) -> Self {
        if use_bias {
            self
        } else {
            Self::RIGID
        }
    }

    /// The row of a one-sided limit whose error `c` is positive while the limit holds: a held
    /// limit only removes the speed that would cross it within the substep (speculative),
    /// one past its end is corrected with this softness, or rigidly without bias.
    fn limit_row(self, c: f32, inv_h: f32, use_bias: bool) -> (f32, Self) {
        if c > 0.0 {
            (c * inv_h, Self::RIGID)
        } else if use_bias {
            (self.bias_rate * c, self)
        } else {
            (0.0, Self::RIGID)
        }
    }
}

/// A joint's data for one step (the runtime part of Box2D's `b2JointSim`): the bodies' inverse
/// masses, the frames relative to their centres of mass in world orientation at the step's
/// start, the distance between the centres, the softness of the rows and effective masses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct JointStep2d {
    pub inv_mass_a: f32,
    pub inv_mass_b: f32,
    pub inv_inertia_a: f32,
    pub inv_inertia_b: f32,
    pub frame_a: Transform2d,
    pub frame_b: Transform2d,
    pub delta_center: Vec2,
    pub constraint: Softness2d,
    pub spring: Softness2d,
    pub angular_spring: Softness2d,
    pub axial_mass: f32,
    pub perp_mass: f32,
    pub motor_mass: f32,
    pub angular_mass: f32,
    pub linear_mass: Mat22,
    /// Body A's awake index plus one this step, zero if it is not awake: a still, unmoved
    /// body (see [`BodyState2d`]).
    pub index_a: u32,
    /// Body B's awake index plus one, zero if it is not awake.
    pub index_b: u32,
}

/// The substep a joint is solved over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JointSolveContext {
    /// Substep length.
    pub h: f32,
    /// Substeps per second (Box2D's `inv_h`, the substep count over the step).
    pub inv_h: f32,
    /// Whether rigid rows correct positions (the solve pass) or not (the relax pass).
    pub use_bias: bool,
    /// The world's length units per metre, for the prismatic limit's speculative bias.
    pub length_units: f32,
}

/// The common part of Box2D's joint prepare: inverse masses, frames relative to the centres of
/// mass in world orientation, the centre delta and the rigid rows' softness.
fn prepare_frames(
    (frame_a, frame_b): (&Transform2d, &Transform2d),
    (a, b): (&RigidBody2d, &RigidBody2d),
    (center_a, center_b): (Vec2, Vec2),
    (constraint_hertz, constraint_damping_ratio): (f32, f32),
    h: f32,
    inv_h: f32,
) -> JointStep2d {
    let hertz = constraint_hertz.min(0.25 * inv_h);
    JointStep2d {
        inv_mass_a: a.inv_mass,
        inv_mass_b: b.inv_mass,
        inv_inertia_a: a.inv_inertia,
        inv_inertia_b: b.inv_inertia,
        frame_a: Transform2d::from_translation_rotation(
            a.rotation.rotate(frame_a.translation - a.local_center),
            a.rotation * frame_a.rotation,
        ),
        frame_b: Transform2d::from_translation_rotation(
            b.rotation.rotate(frame_b.translation - b.local_center),
            b.rotation * frame_b.rotation,
        ),
        delta_center: center_b - center_a,
        constraint: Softness2d::new(hertz, constraint_damping_ratio, h),
        ..JointStep2d::default()
    }
}

/// `1 / k`, or zero for `k <= 0`.
fn inverse_mass(k: f32) -> f32 {
    if k > 0.0 {
        1.0 / k
    } else {
        0.0
    }
}

/// Body `body`'s frame `local` in world space.
fn world_frame(body: &RigidBody2d, local: &Transform2d) -> Transform2d {
    Transform2d::from_translation_rotation(
        body.position + body.rotation.rotate(local.translation),
        body.rotation * local.rotation,
    )
}

/// The current anchors and their separation (Box2D: anchors rotated by the bodies' turns
/// since the step began; separation `dcB - dcA + rB - rA + deltaCenter`).
fn anchors(step: &JointStep2d, a: &BodyState2d, b: &BodyState2d) -> (Vec2, Vec2, Vec2) {
    let r_a = a.delta_rotation.rotate(step.frame_a.translation);
    let r_b = b.delta_rotation.rotate(step.frame_b.translation);
    let separation = (b.delta_position - a.delta_position) + (r_b - r_a) + step.delta_center;
    (r_a, r_b, separation)
}

/// [`anchors`] for the prismatic and wheel joints, which add the centre delta before the
/// anchors (Box2D `d = (dcB - dcA + deltaCenter) + (rB - rA)`).
fn slider_anchors(step: &JointStep2d, a: &BodyState2d, b: &BodyState2d) -> (Vec2, Vec2, Vec2) {
    let r_a = a.delta_rotation.rotate(step.frame_a.translation);
    let r_b = b.delta_rotation.rotate(step.frame_b.translation);
    let d = ((b.delta_position - a.delta_position) + step.delta_center) + (r_b - r_a);
    (r_a, r_b, d)
}

/// The point-to-point effective mass matrix of anchors `r_a` and `r_b`.
fn point_mass(step: &JointStep2d, r_a: Vec2, r_b: Vec2) -> Mat22 {
    let (m_a, m_b, i_a, i_b) = (
        step.inv_mass_a,
        step.inv_mass_b,
        step.inv_inertia_a,
        step.inv_inertia_b,
    );
    let k12 = -r_a.y * r_a.x * i_a - r_b.y * r_b.x * i_b;
    Mat22 {
        cx: Vec2::new(m_a + m_b + r_a.y * r_a.y * i_a + r_b.y * r_b.y * i_b, k12),
        cy: Vec2::new(k12, m_a + m_b + r_a.x * r_a.x * i_a + r_b.x * r_b.x * i_b),
    }
}

/// Applies linear impulse `p` at anchors `r_a` and `r_b` plus angular impulse `angular`, B
/// receiving them and A their opposite.
fn apply_pair(
    step: &JointStep2d,
    (a, b): (&mut BodyState2d, &mut BodyState2d),
    (r_a, r_b): (Vec2, Vec2),
    p: Vec2,
    angular: f32,
) {
    a.apply(
        step.inv_mass_a,
        step.inv_inertia_a,
        -p,
        -(cross(r_a, p) + angular),
    );
    b.apply(
        step.inv_mass_b,
        step.inv_inertia_b,
        p,
        cross(r_b, p) + angular,
    );
}

/// Velocities that a solve updates as Box2D's locals `vA, wA, vB, wB`, written back once.
#[derive(Debug, Clone, Copy)]
struct Velocities {
    v_a: Vec2,
    w_a: f32,
    v_b: Vec2,
    w_b: f32,
}

impl Velocities {
    const fn of(a: &BodyState2d, b: &BodyState2d) -> Self {
        Self {
            v_a: a.linear_velocity,
            w_a: a.angular_velocity,
            v_b: b.linear_velocity,
            w_b: b.angular_velocity,
        }
    }

    /// Box2D's `vA = b2MulSub(vA, mA, P); wA -= iA * LA; vB = b2MulAdd(vB, mB, P); wB += iB * LB`.
    fn apply(&mut self, step: &JointStep2d, p: Vec2, l_a: f32, l_b: f32) {
        self.v_a = Vec2::new(
            self.v_a.x - step.inv_mass_a * p.x,
            self.v_a.y - step.inv_mass_a * p.y,
        );
        self.w_a -= step.inv_inertia_a * l_a;
        self.v_b = Vec2::new(
            self.v_b.x + step.inv_mass_b * p.x,
            self.v_b.y + step.inv_mass_b * p.y,
        );
        self.w_b += step.inv_inertia_b * l_b;
    }

    /// Angular impulse `impulse`, B receiving it and A its opposite.
    fn turn(&mut self, step: &JointStep2d, impulse: f32) {
        self.w_a -= step.inv_inertia_a * impulse;
        self.w_b += step.inv_inertia_b * impulse;
    }

    fn store(self, a: &mut BodyState2d, b: &mut BodyState2d) {
        if a.dynamic {
            a.linear_velocity = self.v_a;
            a.angular_velocity = self.w_a;
        }
        if b.dynamic {
            b.linear_velocity = self.v_b;
            b.angular_velocity = self.w_b;
        }
    }
}

/// The frame at `anchor` without rotation.
const fn anchor_frame(anchor: Vec2) -> Transform2d {
    Transform2d::from_translation(anchor)
}

/// A frame at `anchor` whose x-axis is `axis`.
fn axis_frame(anchor: Vec2, axis: Vec2) -> Transform2d {
    let axis = axis.normalize_or_zero();
    let rotation = if axis == Vec2::ZERO {
        Rot2::IDENTITY
    } else {
        Rot2::new(axis.x, axis.y)
    };
    Transform2d::from_translation_rotation(anchor, rotation)
}

/// Turns frame B so that the joint holds body B turned `angle` relative to body A: the joint
/// angle, frame B's turn relative to frame A, is then zero there.
fn reference_angle(frame_a: &Transform2d, frame_b: &mut Transform2d, angle: f32) {
    frame_b.rotation = frame_a.rotation * Rot2::from_angle(-angle);
}

/// A distance joint: a rod, or a spring, between two anchors (Box2D `b2DistanceJoint`).
///
/// Without a spring it holds the anchors `length` apart. With `enable_spring` it pulls them
/// toward `length` at `hertz` (no force at `hertz = 0`) with a force between
/// `lower_spring_force` and `upper_spring_force`, while `enable_limit` keeps the length between
/// `min_length` and `max_length` and `enable_motor` drives the length at `motor_speed`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistanceJoint2d {
    /// Frame on body A; its position is the anchor.
    pub local_frame_a: Transform2d,
    /// Frame on body B; its position is the anchor.
    pub local_frame_b: Transform2d,
    /// Rest length.
    pub length: f32,
    /// Whether the joint is a spring rather than a rod.
    pub enable_spring: bool,
    /// Spring stiffness in Hertz.
    pub hertz: f32,
    /// Spring damping ratio.
    pub damping_ratio: f32,
    /// Most compressive spring force, negative to push (Box2D `lowerSpringForce`).
    pub lower_spring_force: f32,
    /// Most tensile spring force (Box2D `upperSpringForce`).
    pub upper_spring_force: f32,
    /// Whether `min_length` and `max_length` bound a spring.
    pub enable_limit: bool,
    /// Shortest length.
    pub min_length: f32,
    /// Longest length.
    pub max_length: f32,
    /// Whether the motor drives a spring's length.
    pub enable_motor: bool,
    /// Largest motor force, in newtons.
    pub max_motor_force: f32,
    /// Speed the motor drives the length at, in m/s.
    pub motor_speed: f32,
    /// Stiffness of the rod and the limits, in Hertz.
    pub constraint_hertz: f32,
    /// Damping ratio of the rod and the limits.
    pub constraint_damping_ratio: f32,
    /// Accumulated rod or spring impulse.
    pub impulse: f32,
    /// Accumulated lower limit impulse.
    pub lower_impulse: f32,
    /// Accumulated upper limit impulse.
    pub upper_impulse: f32,
    /// Accumulated motor impulse.
    pub motor_impulse: f32,
}

impl Component for DistanceJoint2d {}

impl Default for DistanceJoint2d {
    /// Box2D `b2DefaultDistanceJointDef`: a 1 m rod at the bodies' origins.
    fn default() -> Self {
        Self {
            local_frame_a: Transform2d::IDENTITY,
            local_frame_b: Transform2d::IDENTITY,
            length: 1.0,
            enable_spring: false,
            hertz: 0.0,
            damping_ratio: 0.0,
            lower_spring_force: -f32::MAX,
            upper_spring_force: f32::MAX,
            enable_limit: false,
            min_length: 0.0,
            max_length: HUGE_LENGTH,
            enable_motor: false,
            max_motor_force: 0.0,
            motor_speed: 0.0,
            constraint_hertz: DEFAULT_CONSTRAINT_HERTZ,
            constraint_damping_ratio: DEFAULT_CONSTRAINT_DAMPING_RATIO,
            impulse: 0.0,
            lower_impulse: 0.0,
            upper_impulse: 0.0,
            motor_impulse: 0.0,
        }
    }
}

impl DistanceJoint2d {
    /// A joint between `local_anchor_a` and `local_anchor_b`, `length` apart: a rod, or a
    /// spring at `hertz` if that is above zero.
    #[must_use]
    pub fn new(
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        length: f32,
        hertz: f32,
        damping_ratio: f32,
    ) -> Self {
        Self {
            local_frame_a: anchor_frame(local_anchor_a),
            local_frame_b: anchor_frame(local_anchor_b),
            length,
            enable_spring: hertz > 0.0,
            hertz,
            damping_ratio,
            ..Self::default()
        }
    }

    /// Sets the frames.
    #[must_use]
    pub const fn with_local_frames(mut self, frame_a: Transform2d, frame_b: Transform2d) -> Self {
        self.local_frame_a = frame_a;
        self.local_frame_b = frame_b;
        self
    }

    /// Sets the rest length.
    #[must_use]
    pub const fn with_length(mut self, length: f32) -> Self {
        self.length = length;
        self
    }

    /// Makes the joint a spring at `hertz` (no force at zero).
    #[must_use]
    pub const fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Bounds the spring force.
    #[must_use]
    pub fn with_spring_force_range(mut self, lower: f32, upper: f32) -> Self {
        self.lower_spring_force = lower.min(upper);
        self.upper_spring_force = lower.max(upper);
        self
    }

    /// Keeps a spring's length between `min` and `max`.
    #[must_use]
    pub fn with_limits(mut self, min: f32, max: f32) -> Self {
        self.enable_limit = true;
        self.min_length = min.min(max);
        self.max_length = min.max(max);
        self
    }

    /// Drives a spring's length at `speed` with at most `max_force`.
    #[must_use]
    pub const fn with_motor(mut self, speed: f32, max_force: f32) -> Self {
        self.enable_motor = true;
        self.motor_speed = speed;
        self.max_motor_force = max_force;
        self
    }

    /// Sets the softness of the rod and the limits.
    #[must_use]
    pub const fn with_constraint_softness(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.constraint_hertz = hertz;
        self.constraint_damping_ratio = damping_ratio;
        self
    }

    /// The distance between the anchors (Box2D `b2DistanceJoint_GetCurrentLength`).
    #[must_use]
    pub fn current_length(&self, body_a: &RigidBody2d, body_b: &RigidBody2d) -> f32 {
        let p_a = world_frame(body_a, &self.local_frame_a).translation;
        let p_b = world_frame(body_b, &self.local_frame_b).translation;
        (p_b - p_a).length()
    }

    fn prepare(&self, step: &mut JointStep2d, h: f32) {
        let (r_a, r_b) = (step.frame_a.translation, step.frame_b.translation);
        let axis = normalize((r_b - r_a) + step.delta_center);
        let cr_a = cross(r_a, axis);
        let cr_b = cross(r_b, axis);
        step.axial_mass = inverse_mass(
            step.inv_mass_a
                + step.inv_mass_b
                + step.inv_inertia_a * cr_a * cr_a
                + step.inv_inertia_b * cr_b * cr_b,
        );
        step.spring = Softness2d::new(self.hertz, self.damping_ratio, h);
    }

    fn warm_start(&self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        let (r_a, r_b, separation) = anchors(step, a, b);
        let axis = normalize(separation);
        let axial = self.impulse + self.lower_impulse - self.upper_impulse + self.motor_impulse;
        apply_pair(step, (a, b), (r_a, r_b), axis * axial, 0.0);
    }

    fn solve_rows(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        let mut vel = Velocities::of(a, b);
        let (r_a, r_b, separation) = anchors(step, a, b);
        let length = separation.length();
        let axis = normalize(separation);
        let axial_speed = |vel: &Velocities| {
            axis.dot((vel.v_b - vel.v_a) + (cross_sv(vel.w_b, r_b) - cross_sv(vel.w_a, r_a)))
        };
        let push = |vel: &mut Velocities, impulse: f32| {
            let p = axis * impulse;
            vel.apply(step, p, cross(r_a, p), cross(r_b, p));
        };

        // Soft while the spring is on, unless equal limits hold it.
        if self.enable_spring && (self.min_length < self.max_length || !self.enable_limit) {
            if self.hertz > 0.0 {
                let cdot = axial_speed(&vel);
                let bias = step.spring.bias_rate * (length - self.length);
                let m = step.spring.mass_scale * step.axial_mass;
                let old = self.impulse;
                let impulse = -m * (cdot + bias) - step.spring.impulse_scale * old;
                self.impulse = clamp(
                    self.impulse + impulse,
                    self.lower_spring_force * ctx.h,
                    self.upper_spring_force * ctx.h,
                );
                push(&mut vel, self.impulse - old);
            }
            if self.enable_motor {
                let cdot = axial_speed(&vel);
                let impulse = step.axial_mass * (self.motor_speed - cdot);
                let old = self.motor_impulse;
                let max = ctx.h * self.max_motor_force;
                self.motor_impulse = clamp(self.motor_impulse + impulse, -max, max);
                push(&mut vel, self.motor_impulse - old);
            }
            if self.enable_limit {
                // Lower.
                let cdot = axial_speed(&vel);
                let (bias, soft) =
                    step.constraint
                        .limit_row(length - self.min_length, ctx.inv_h, ctx.use_bias);
                let impulse = -soft.mass_scale * step.axial_mass * (cdot + bias)
                    - soft.impulse_scale * self.lower_impulse;
                let new = (self.lower_impulse + impulse).max(0.0);
                let impulse = new - self.lower_impulse;
                self.lower_impulse = new;
                push(&mut vel, impulse);

                // Upper, signs flipped so that the impulse is positive while it holds.
                let cdot = axis
                    .dot((vel.v_a - vel.v_b) + (cross_sv(vel.w_a, r_a) - cross_sv(vel.w_b, r_b)));
                let (bias, soft) =
                    step.constraint
                        .limit_row(self.max_length - length, ctx.inv_h, ctx.use_bias);
                let impulse = -soft.mass_scale * step.axial_mass * (cdot + bias)
                    - soft.impulse_scale * self.upper_impulse;
                let new = (self.upper_impulse + impulse).max(0.0);
                let impulse = new - self.upper_impulse;
                self.upper_impulse = new;
                push(&mut vel, -impulse);
            }
        } else {
            // Rod.
            let cdot = axial_speed(&vel);
            let soft = step.constraint.rigid_row(ctx.use_bias);
            let bias = soft.bias_rate * (length - self.length);
            let impulse = -soft.mass_scale * step.axial_mass * (cdot + bias)
                - soft.impulse_scale * self.impulse;
            self.impulse += impulse;
            push(&mut vel, impulse);
        }
        vel.store(a, b);
    }
}

/// A revolute joint: a hinge that pins the frames' origins together (Box2D
/// `b2RevoluteJoint`). The joint angle is frame B's turn relative to frame A; a spring can pull
/// it toward `target_angle`, limits bound it and a motor drives it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RevoluteJoint2d {
    /// Frame on body A.
    pub local_frame_a: Transform2d,
    /// Frame on body B.
    pub local_frame_b: Transform2d,
    /// Whether the rotational spring pulls the joint angle toward `target_angle`.
    pub enable_spring: bool,
    /// Rotational spring stiffness in Hertz.
    pub hertz: f32,
    /// Rotational spring damping ratio.
    pub damping_ratio: f32,
    /// Joint angle the spring pulls toward, in radians.
    pub target_angle: f32,
    /// Whether the angle limits are on.
    pub enable_limit: bool,
    /// Lower angle limit, in radians.
    pub lower_angle: f32,
    /// Upper angle limit, in radians.
    pub upper_angle: f32,
    /// Whether the motor drives the joint angle.
    pub enable_motor: bool,
    /// Motor speed, in rad/s.
    pub motor_speed: f32,
    /// Largest motor torque, in N m.
    pub max_motor_torque: f32,
    /// Stiffness of the pin and the limits, in Hertz.
    pub constraint_hertz: f32,
    /// Damping ratio of the pin and the limits.
    pub constraint_damping_ratio: f32,
    /// Accumulated pin impulse.
    pub linear_impulse: Vec2,
    /// Accumulated spring impulse.
    pub spring_impulse: f32,
    /// Accumulated motor impulse.
    pub motor_impulse: f32,
    /// Accumulated lower limit impulse.
    pub lower_impulse: f32,
    /// Accumulated upper limit impulse.
    pub upper_impulse: f32,
}

impl Component for RevoluteJoint2d {}

impl Default for RevoluteJoint2d {
    /// Box2D `b2DefaultRevoluteJointDef`: a free hinge at the bodies' origins.
    fn default() -> Self {
        Self {
            local_frame_a: Transform2d::IDENTITY,
            local_frame_b: Transform2d::IDENTITY,
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
            constraint_hertz: DEFAULT_CONSTRAINT_HERTZ,
            constraint_damping_ratio: DEFAULT_CONSTRAINT_DAMPING_RATIO,
            linear_impulse: Vec2::ZERO,
            spring_impulse: 0.0,
            motor_impulse: 0.0,
            lower_impulse: 0.0,
            upper_impulse: 0.0,
        }
    }
}

impl RevoluteJoint2d {
    /// A hinge at `local_anchor_a` and `local_anchor_b`; `hertz` and `damping_ratio` configure
    /// the rotational spring, which stays off until enabled (see [`Self::with_spring`]).
    #[must_use]
    pub fn new(local_anchor_a: Vec2, local_anchor_b: Vec2, hertz: f32, damping_ratio: f32) -> Self {
        Self {
            local_frame_a: anchor_frame(local_anchor_a),
            local_frame_b: anchor_frame(local_anchor_b),
            hertz,
            damping_ratio,
            ..Self::default()
        }
    }

    /// Sets the frames.
    #[must_use]
    pub const fn with_local_frames(mut self, frame_a: Transform2d, frame_b: Transform2d) -> Self {
        self.local_frame_a = frame_a;
        self.local_frame_b = frame_b;
        self
    }

    /// Makes the bodies rest at joint angle zero when B is turned `angle` relative to A, by
    /// turning frame B.
    #[must_use]
    pub fn with_reference_angle(mut self, angle: f32) -> Self {
        reference_angle(&self.local_frame_a, &mut self.local_frame_b, angle);
        self
    }

    /// Turns on the rotational spring.
    #[must_use]
    pub const fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Sets the angle the spring pulls toward, within ±π.
    #[must_use]
    pub fn with_target_angle(mut self, angle: f32) -> Self {
        self.target_angle = clamp(angle, -PI, PI);
        self
    }

    /// Sets the softness of the pin and the limits.
    #[must_use]
    pub const fn with_constraint_softness(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.constraint_hertz = hertz;
        self.constraint_damping_ratio = damping_ratio;
        self
    }

    /// Turns on the angle limits (see [`Self::set_limits`]).
    #[must_use]
    pub fn with_limits(mut self, lower: f32, upper: f32) -> Self {
        self.set_limits(lower, upper);
        self.enable_limit = true;
        self
    }

    /// Turns on the motor.
    #[must_use]
    pub const fn with_motor(mut self, speed: f32, max_torque: f32) -> Self {
        self.enable_motor = true;
        self.motor_speed = speed;
        self.max_motor_torque = max_torque;
        self
    }

    /// Sets the angle limits, ordered and clamped to [`MAX_REVOLUTE_LIMIT`].
    pub fn set_limits(&mut self, lower: f32, upper: f32) {
        self.lower_angle = clamp(lower.min(upper), -MAX_REVOLUTE_LIMIT, MAX_REVOLUTE_LIMIT);
        self.upper_angle = clamp(lower.max(upper), -MAX_REVOLUTE_LIMIT, MAX_REVOLUTE_LIMIT);
    }

    /// Sets the motor speed and largest torque.
    pub fn set_motor(&mut self, speed: f32, max_torque: f32) {
        self.motor_speed = speed;
        self.max_motor_torque = max_torque;
    }

    /// The joint angle, frame B's turn relative to frame A (Box2D
    /// `b2RevoluteJoint_GetAngle`).
    #[must_use]
    pub fn angle(&self, body_a: &RigidBody2d, body_b: &RigidBody2d) -> f32 {
        relative_angle(
            body_a.rotation * self.local_frame_a.rotation,
            body_b.rotation * self.local_frame_b.rotation,
        )
    }

    fn prepare(&self, step: &mut JointStep2d, h: f32) {
        step.axial_mass = inverse_mass(step.inv_inertia_a + step.inv_inertia_b);
        step.spring = Softness2d::new(self.hertz, self.damping_ratio, h);
    }

    fn warm_start(&self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        let r_a = a.delta_rotation.rotate(step.frame_a.translation);
        let r_b = b.delta_rotation.rotate(step.frame_b.translation);
        let axial =
            self.spring_impulse + self.motor_impulse + self.lower_impulse - self.upper_impulse;
        apply_pair(step, (a, b), (r_a, r_b), self.linear_impulse, axial);
    }

    fn solve_rows(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        let mut vel = Velocities::of(a, b);
        let q_a = a.delta_rotation * step.frame_a.rotation;
        let q_b = b.delta_rotation * step.frame_b.rotation;
        let rel_q = inv_mul_rot(q_a, q_b);
        let fixed_rotation = step.inv_inertia_a + step.inv_inertia_b == 0.0;

        if self.enable_spring && !fixed_rotation {
            let c = unwind_angle(rot_angle(rel_q) - self.target_angle);
            let soft = step.spring;
            let cdot = vel.w_b - vel.w_a;
            let impulse = -soft.mass_scale * step.axial_mass * (cdot + soft.bias_rate * c)
                - soft.impulse_scale * self.spring_impulse;
            self.spring_impulse += impulse;
            vel.turn(step, impulse);
        }

        if self.enable_motor && !fixed_rotation {
            let cdot = vel.w_b - vel.w_a - self.motor_speed;
            let impulse = -step.axial_mass * cdot;
            let old = self.motor_impulse;
            let max = ctx.h * self.max_motor_torque;
            self.motor_impulse = clamp(self.motor_impulse + impulse, -max, max);
            vel.turn(step, self.motor_impulse - old);
        }

        if self.enable_limit && !fixed_rotation {
            let angle = rot_angle(rel_q);
            // Lower.
            let (bias, soft) =
                step.constraint
                    .limit_row(angle - self.lower_angle, ctx.inv_h, ctx.use_bias);
            let cdot = vel.w_b - vel.w_a;
            let old = self.lower_impulse;
            let impulse =
                -soft.mass_scale * step.axial_mass * (cdot + bias) - soft.impulse_scale * old;
            self.lower_impulse = (old + impulse).max(0.0);
            vel.turn(step, self.lower_impulse - old);

            // Upper, signs flipped so that the impulse is positive while it holds.
            let (bias, soft) =
                step.constraint
                    .limit_row(self.upper_angle - angle, ctx.inv_h, ctx.use_bias);
            let cdot = vel.w_a - vel.w_b;
            let old = self.upper_impulse;
            let impulse =
                -soft.mass_scale * step.axial_mass * (cdot + bias) - soft.impulse_scale * old;
            self.upper_impulse = (old + impulse).max(0.0);
            vel.turn(step, -(self.upper_impulse - old));
        }

        // Pin.
        let (r_a, r_b, separation) = anchors(step, a, b);
        let cdot = (vel.v_b + cross_sv(vel.w_b, r_b)) - (vel.v_a + cross_sv(vel.w_a, r_a));
        let soft = step.constraint.rigid_row(ctx.use_bias);
        let bias = if ctx.use_bias {
            separation * soft.bias_rate
        } else {
            Vec2::ZERO
        };
        let k = point_mass(step, r_a, r_b);
        let solved = k.solve(cdot + bias);
        let impulse = Vec2::new(
            -soft.mass_scale * solved.x - soft.impulse_scale * self.linear_impulse.x,
            -soft.mass_scale * solved.y - soft.impulse_scale * self.linear_impulse.y,
        );
        self.linear_impulse += impulse;
        vel.apply(step, impulse, cross(r_a, impulse), cross(r_b, impulse));
        vel.store(a, b);
    }
}

/// A prismatic joint: a slider along frame A's x-axis that keeps the frames' rotations equal
/// (Box2D `b2PrismaticJoint`). A spring can pull the translation toward `target_translation`,
/// limits bound it and a motor drives it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrismaticJoint2d {
    /// Frame on body A; its x-axis is the slider axis.
    pub local_frame_a: Transform2d,
    /// Frame on body B.
    pub local_frame_b: Transform2d,
    /// Whether the spring pulls the translation toward `target_translation`.
    pub enable_spring: bool,
    /// Spring stiffness in Hertz.
    pub hertz: f32,
    /// Spring damping ratio.
    pub damping_ratio: f32,
    /// Translation the spring pulls toward.
    pub target_translation: f32,
    /// Whether the translation limits are on.
    pub enable_limit: bool,
    /// Lower translation limit.
    pub lower_translation: f32,
    /// Upper translation limit.
    pub upper_translation: f32,
    /// Whether the motor drives the translation.
    pub enable_motor: bool,
    /// Motor speed, in m/s.
    pub motor_speed: f32,
    /// Largest motor force, in newtons.
    pub max_motor_force: f32,
    /// Stiffness of the guides and the limits, in Hertz.
    pub constraint_hertz: f32,
    /// Damping ratio of the guides and the limits.
    pub constraint_damping_ratio: f32,
    /// Accumulated guide impulses: perpendicular and angular.
    pub impulse: Vec2,
    /// Accumulated spring impulse.
    pub spring_impulse: f32,
    /// Accumulated motor impulse.
    pub motor_impulse: f32,
    /// Accumulated lower limit impulse.
    pub lower_impulse: f32,
    /// Accumulated upper limit impulse.
    pub upper_impulse: f32,
}

impl Component for PrismaticJoint2d {}

impl Default for PrismaticJoint2d {
    /// Box2D `b2DefaultPrismaticJointDef`: a free slider along the x-axis at the origins.
    fn default() -> Self {
        Self {
            local_frame_a: Transform2d::IDENTITY,
            local_frame_b: Transform2d::IDENTITY,
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
            constraint_hertz: DEFAULT_CONSTRAINT_HERTZ,
            constraint_damping_ratio: DEFAULT_CONSTRAINT_DAMPING_RATIO,
            impulse: Vec2::ZERO,
            spring_impulse: 0.0,
            motor_impulse: 0.0,
            lower_impulse: 0.0,
            upper_impulse: 0.0,
        }
    }
}

impl PrismaticJoint2d {
    /// A slider along `local_axis_a` (on body A) through the anchors; `hertz` and
    /// `damping_ratio` configure the spring, which stays off until enabled.
    #[must_use]
    pub fn new(
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        local_axis_a: Vec2,
        hertz: f32,
        damping_ratio: f32,
    ) -> Self {
        let frame_a = axis_frame(local_anchor_a, local_axis_a);
        Self {
            local_frame_a: frame_a,
            local_frame_b: Transform2d::from_translation_rotation(local_anchor_b, frame_a.rotation),
            hertz,
            damping_ratio,
            ..Self::default()
        }
    }

    /// Sets the frames.
    #[must_use]
    pub const fn with_local_frames(mut self, frame_a: Transform2d, frame_b: Transform2d) -> Self {
        self.local_frame_a = frame_a;
        self.local_frame_b = frame_b;
        self
    }

    /// Makes the bodies rest when B is turned `angle` relative to A, by turning frame B.
    #[must_use]
    pub fn with_reference_angle(mut self, angle: f32) -> Self {
        reference_angle(&self.local_frame_a, &mut self.local_frame_b, angle);
        self
    }

    /// Turns on the spring.
    #[must_use]
    pub const fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Sets the translation the spring pulls toward.
    #[must_use]
    pub const fn with_target_translation(mut self, translation: f32) -> Self {
        self.target_translation = translation;
        self
    }

    /// Sets the softness of the guides and the limits.
    #[must_use]
    pub const fn with_constraint_softness(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.constraint_hertz = hertz;
        self.constraint_damping_ratio = damping_ratio;
        self
    }

    /// Turns on the translation limits.
    #[must_use]
    pub fn with_limits(mut self, lower: f32, upper: f32) -> Self {
        self.set_limits(lower, upper);
        self.enable_limit = true;
        self
    }

    /// Turns on the motor.
    #[must_use]
    pub const fn with_motor(mut self, speed: f32, max_force: f32) -> Self {
        self.enable_motor = true;
        self.motor_speed = speed;
        self.max_motor_force = max_force;
        self
    }

    /// Sets the translation limits, ordered.
    pub fn set_limits(&mut self, lower: f32, upper: f32) {
        self.lower_translation = lower.min(upper);
        self.upper_translation = lower.max(upper);
    }

    /// Sets the motor speed and largest force.
    pub fn set_motor(&mut self, speed: f32, max_force: f32) {
        self.motor_speed = speed;
        self.max_motor_force = max_force;
    }

    /// Frame B's origin along frame A's axis (Box2D `b2PrismaticJoint_GetTranslation`).
    #[must_use]
    pub fn translation(&self, body_a: &RigidBody2d, body_b: &RigidBody2d) -> f32 {
        let axis = body_a.rotation.rotate(self.local_frame_a.rotation.x_axis());
        let p_a = body_a.position + body_a.rotation.rotate(self.local_frame_a.translation);
        let p_b = body_b.position + body_b.rotation.rotate(self.local_frame_b.translation);
        (p_b - p_a).dot(axis)
    }

    /// The rate the translation changes at (Box2D `b2PrismaticJoint_GetSpeed`).
    #[must_use]
    pub fn speed(&self, body_a: &RigidBody2d, body_b: &RigidBody2d) -> f32 {
        let (q_a, q_b) = (body_a.rotation, body_b.rotation);
        let axis = q_a.rotate(self.local_frame_a.rotation.x_axis());
        let r_a = q_a.rotate(self.local_frame_a.translation - body_a.local_center);
        let r_b = q_b.rotate(self.local_frame_b.translation - body_b.local_center);
        let d = (body_b.world_center() - body_a.world_center()) + (r_b - r_a);
        let (v_a, v_b) = (body_a.linear_velocity, body_b.linear_velocity);
        let (w_a, w_b) = (body_a.angular_velocity, body_b.angular_velocity);
        let v_rel = (v_b + cross_sv(w_b, r_b)) - (v_a + cross_sv(w_a, r_a));
        d.dot(cross_sv(w_a, axis)) + axis.dot(v_rel)
    }

    fn prepare(&self, step: &mut JointStep2d, h: f32) {
        step.spring = Softness2d::new(self.hertz, self.damping_ratio, h);
    }

    fn warm_start(&self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        let (r_a, r_b, d) = slider_anchors(step, a, b);
        let axis = a.delta_rotation.rotate(step.frame_a.rotation.x_axis());
        let a1 = cross(r_a + d, axis);
        let a2 = cross(r_b, axis);
        let axial =
            self.spring_impulse + self.motor_impulse + self.lower_impulse - self.upper_impulse;
        let perp = left_perp(axis);
        let s1 = cross(r_a + d, perp);
        let s2 = cross(r_b, perp);
        let (perp_impulse, angle_impulse) = (self.impulse.x, self.impulse.y);
        let p = axis * axial + perp * perp_impulse;
        let l_a = axial * a1 + perp_impulse * s1 + angle_impulse;
        let l_b = axial * a2 + perp_impulse * s2 + angle_impulse;
        a.apply(step.inv_mass_a, step.inv_inertia_a, -p, -l_a);
        b.apply(step.inv_mass_b, step.inv_inertia_b, p, l_b);
    }

    fn solve_rows(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        let mut vel = Velocities::of(a, b);
        let q_a = a.delta_rotation * step.frame_a.rotation;
        let q_b = b.delta_rotation * step.frame_b.rotation;
        let rel_q = inv_mul_rot(q_a, q_b);
        let (r_a, r_b, d) = slider_anchors(step, a, b);
        let axis = a.delta_rotation.rotate(step.frame_a.rotation.x_axis());
        let translation = axis.dot(d);
        let a1 = cross(r_a + d, axis);
        let a2 = cross(r_b, axis);
        let (m_a, m_b, i_a, i_b) = (
            step.inv_mass_a,
            step.inv_mass_b,
            step.inv_inertia_a,
            step.inv_inertia_b,
        );
        let axial_mass = inverse_mass(m_a + m_b + i_a * a1 * a1 + i_b * a2 * a2);
        let axial_speed =
            |vel: &Velocities| axis.dot(vel.v_b - vel.v_a) + a2 * vel.w_b - a1 * vel.w_a;
        let push = |vel: &mut Velocities, impulse: f32| {
            vel.apply(step, axis * impulse, impulse * a1, impulse * a2);
        };

        if self.enable_spring {
            let soft = step.spring;
            let c = translation - self.target_translation;
            let cdot = axial_speed(&vel);
            let impulse = -soft.mass_scale * axial_mass * (cdot + soft.bias_rate * c)
                - soft.impulse_scale * self.spring_impulse;
            self.spring_impulse += impulse;
            push(&mut vel, impulse);
        }

        if self.enable_motor {
            let cdot = axial_speed(&vel);
            let impulse = axial_mass * (self.motor_speed - cdot);
            let old = self.motor_impulse;
            let max = ctx.h * self.max_motor_force;
            self.motor_impulse = clamp(self.motor_impulse + impulse, -max, max);
            push(&mut vel, self.motor_impulse - old);
        }

        if self.enable_limit {
            // Only near a limit: within a quarter of the range. A held limit's speculative
            // distance is capped at a metre (Box2D's `safe`).
            let speculative_distance = 0.25 * (self.upper_translation - self.lower_translation);
            let limit_row = |c: f32| {
                if c > 0.0 {
                    (c.min(ctx.length_units) * ctx.inv_h, Softness2d::RIGID)
                } else if ctx.use_bias {
                    (step.constraint.bias_rate * c, step.constraint)
                } else {
                    (0.0, Softness2d::RIGID)
                }
            };

            // Lower.
            let c = translation - self.lower_translation;
            if c < speculative_distance {
                let (bias, soft) = limit_row(c);
                let old = self.lower_impulse;
                let cdot = axial_speed(&vel);
                let impulse =
                    -axial_mass * soft.mass_scale * (cdot + bias) - soft.impulse_scale * old;
                self.lower_impulse = (old + impulse).max(0.0);
                push(&mut vel, self.lower_impulse - old);
            } else {
                self.lower_impulse = 0.0;
            }

            // Upper, signs flipped so that the impulse is positive while it holds.
            let c = self.upper_translation - translation;
            if c < speculative_distance {
                let (bias, soft) = limit_row(c);
                let old = self.upper_impulse;
                let cdot = axis.dot(vel.v_a - vel.v_b) + a1 * vel.w_a - a2 * vel.w_b;
                let impulse =
                    -axial_mass * soft.mass_scale * (cdot + bias) - soft.impulse_scale * old;
                self.upper_impulse = (old + impulse).max(0.0);
                push(&mut vel, -(self.upper_impulse - old));
            } else {
                self.upper_impulse = 0.0;
            }
        }

        // Guides: perpendicular and angular.
        let perp = left_perp(axis);
        let s1 = cross(d + r_a, perp);
        let s2 = cross(r_b, perp);
        let cdot = Vec2::new(
            perp.dot(vel.v_b - vel.v_a) + s2 * vel.w_b - s1 * vel.w_a,
            vel.w_b - vel.w_a,
        );
        let soft = step.constraint.rigid_row(ctx.use_bias);
        let bias = if ctx.use_bias {
            Vec2::new(perp.dot(d), rot_angle(rel_q)) * soft.bias_rate
        } else {
            Vec2::ZERO
        };
        let k11 = m_a + m_b + i_a * s1 * s1 + i_b * s2 * s2;
        let k12 = i_a * s1 + i_b * s2;
        let mut k22 = i_a + i_b;
        if k22 == 0.0 {
            // Neither body can turn.
            k22 = 1.0;
        }
        let k = Mat22 {
            cx: Vec2::new(k11, k12),
            cy: Vec2::new(k12, k22),
        };
        let solved = k.solve(cdot + bias);
        let impulse = Vec2::new(
            -soft.mass_scale * solved.x - soft.impulse_scale * self.impulse.x,
            -soft.mass_scale * solved.y - soft.impulse_scale * self.impulse.y,
        );
        self.impulse += impulse;
        vel.apply(
            step,
            perp * impulse.x,
            impulse.x * s1 + impulse.y,
            impulse.x * s2 + impulse.y,
        );
        vel.store(a, b);
    }
}

/// A wheel joint: a suspension along frame A's x-axis, with the wheel free to turn (Box2D
/// `b2WheelJoint`). A spring holds the translation at zero, limits bound it, and a motor drives
/// the wheel's rotation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WheelJoint2d {
    /// Frame on body A (the chassis); its x-axis is the suspension axis.
    pub local_frame_a: Transform2d,
    /// Frame on body B (the wheel).
    pub local_frame_b: Transform2d,
    /// Whether the suspension spring is on.
    pub enable_spring: bool,
    /// Suspension stiffness in Hertz.
    pub hertz: f32,
    /// Suspension damping ratio.
    pub damping_ratio: f32,
    /// Whether the translation limits are on.
    pub enable_limit: bool,
    /// Lower translation limit.
    pub lower_translation: f32,
    /// Upper translation limit.
    pub upper_translation: f32,
    /// Whether the motor drives the wheel.
    pub enable_motor: bool,
    /// Motor speed, in rad/s.
    pub motor_speed: f32,
    /// Largest motor torque, in N m.
    pub max_motor_torque: f32,
    /// Stiffness of the guide and the limits, in Hertz.
    pub constraint_hertz: f32,
    /// Damping ratio of the guide and the limits.
    pub constraint_damping_ratio: f32,
    /// Accumulated guide impulse.
    pub perp_impulse: f32,
    /// Accumulated spring impulse.
    pub spring_impulse: f32,
    /// Accumulated motor impulse.
    pub motor_impulse: f32,
    /// Accumulated lower limit impulse.
    pub lower_impulse: f32,
    /// Accumulated upper limit impulse.
    pub upper_impulse: f32,
}

impl Component for WheelJoint2d {}

impl Default for WheelJoint2d {
    /// Box2D `b2DefaultWheelJointDef`: a 1 Hz suspension, damping ratio 0.7, along the x-axis.
    fn default() -> Self {
        Self {
            local_frame_a: Transform2d::IDENTITY,
            local_frame_b: Transform2d::IDENTITY,
            enable_spring: true,
            hertz: 1.0,
            damping_ratio: 0.7,
            enable_limit: false,
            lower_translation: 0.0,
            upper_translation: 0.0,
            enable_motor: false,
            motor_speed: 0.0,
            max_motor_torque: 0.0,
            constraint_hertz: DEFAULT_CONSTRAINT_HERTZ,
            constraint_damping_ratio: DEFAULT_CONSTRAINT_DAMPING_RATIO,
            perp_impulse: 0.0,
            spring_impulse: 0.0,
            motor_impulse: 0.0,
            lower_impulse: 0.0,
            upper_impulse: 0.0,
        }
    }
}

impl WheelJoint2d {
    /// A suspension along `local_axis_a` (on body A) through the anchors, at `hertz`.
    #[must_use]
    pub fn new(
        local_anchor_a: Vec2,
        local_anchor_b: Vec2,
        local_axis_a: Vec2,
        hertz: f32,
        damping_ratio: f32,
    ) -> Self {
        let frame_a = axis_frame(local_anchor_a, local_axis_a);
        Self {
            local_frame_a: frame_a,
            local_frame_b: Transform2d::from_translation_rotation(local_anchor_b, frame_a.rotation),
            hertz,
            damping_ratio,
            ..Self::default()
        }
    }

    /// Sets the frames.
    #[must_use]
    pub const fn with_local_frames(mut self, frame_a: Transform2d, frame_b: Transform2d) -> Self {
        self.local_frame_a = frame_a;
        self.local_frame_b = frame_b;
        self
    }

    /// Turns on the suspension spring.
    #[must_use]
    pub const fn with_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.enable_spring = true;
        self.hertz = hertz;
        self.damping_ratio = damping_ratio;
        self
    }

    /// Sets the softness of the guide and the limits.
    #[must_use]
    pub const fn with_constraint_softness(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.constraint_hertz = hertz;
        self.constraint_damping_ratio = damping_ratio;
        self
    }

    /// Turns on the translation limits.
    #[must_use]
    pub fn with_limits(mut self, lower: f32, upper: f32) -> Self {
        self.set_limits(lower, upper);
        self.enable_limit = true;
        self
    }

    /// Turns on the motor.
    #[must_use]
    pub const fn with_motor(mut self, speed: f32, max_torque: f32) -> Self {
        self.enable_motor = true;
        self.motor_speed = speed;
        self.max_motor_torque = max_torque;
        self
    }

    /// Sets the translation limits, ordered.
    pub fn set_limits(&mut self, lower: f32, upper: f32) {
        self.lower_translation = lower.min(upper);
        self.upper_translation = lower.max(upper);
    }

    /// Sets the motor speed and largest torque.
    pub fn set_motor(&mut self, speed: f32, max_torque: f32) {
        self.motor_speed = speed;
        self.max_motor_torque = max_torque;
    }

    fn prepare(&self, step: &mut JointStep2d, h: f32) {
        let (r_a, r_b) = (step.frame_a.translation, step.frame_b.translation);
        let d = step.delta_center + (r_b - r_a);
        let axis = step.frame_a.rotation.x_axis();
        let perp = left_perp(axis);
        let (m_a, m_b, i_a, i_b) = (
            step.inv_mass_a,
            step.inv_mass_b,
            step.inv_inertia_a,
            step.inv_inertia_b,
        );
        let s1 = cross(d + r_a, perp);
        let s2 = cross(r_b, perp);
        step.perp_mass = inverse_mass(m_a + m_b + i_a * s1 * s1 + i_b * s2 * s2);
        let a1 = cross(d + r_a, axis);
        let a2 = cross(r_b, axis);
        step.axial_mass = inverse_mass(m_a + m_b + i_a * a1 * a1 + i_b * a2 * a2);
        step.spring = Softness2d::new(self.hertz, self.damping_ratio, h);
        step.motor_mass = inverse_mass(i_a + i_b);
    }

    fn warm_start(&self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        let (r_a, r_b, d) = slider_anchors(step, a, b);
        let axis = a.delta_rotation.rotate(step.frame_a.rotation.x_axis());
        let perp = left_perp(axis);
        let a1 = cross(d + r_a, axis);
        let a2 = cross(r_b, axis);
        let s1 = cross(d + r_a, perp);
        let s2 = cross(r_b, perp);
        let axial = self.spring_impulse + self.lower_impulse - self.upper_impulse;
        let p = axis * axial + perp * self.perp_impulse;
        let l_a = axial * a1 + self.perp_impulse * s1 + self.motor_impulse;
        let l_b = axial * a2 + self.perp_impulse * s2 + self.motor_impulse;
        a.apply(step.inv_mass_a, step.inv_inertia_a, -p, -l_a);
        b.apply(step.inv_mass_b, step.inv_inertia_b, p, l_b);
    }

    fn solve_rows(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        let mut vel = Velocities::of(a, b);
        let fixed_rotation = step.inv_inertia_a + step.inv_inertia_b == 0.0;
        let (r_a, r_b, d) = slider_anchors(step, a, b);
        let axis = a.delta_rotation.rotate(step.frame_a.rotation.x_axis());
        let translation = axis.dot(d);
        let a1 = cross(d + r_a, axis);
        let a2 = cross(r_b, axis);
        let axial_speed =
            |vel: &Velocities| axis.dot(vel.v_b - vel.v_a) + a2 * vel.w_b - a1 * vel.w_a;
        let push = |vel: &mut Velocities, impulse: f32| {
            vel.apply(step, axis * impulse, impulse * a1, impulse * a2);
        };

        if self.enable_motor && !fixed_rotation {
            let cdot = vel.w_b - vel.w_a - self.motor_speed;
            let impulse = -step.motor_mass * cdot;
            let old = self.motor_impulse;
            let max = ctx.h * self.max_motor_torque;
            self.motor_impulse = clamp(self.motor_impulse + impulse, -max, max);
            vel.turn(step, self.motor_impulse - old);
        }

        if self.enable_spring {
            let soft = step.spring;
            let cdot = axial_speed(&vel);
            let impulse =
                -soft.mass_scale * step.axial_mass * (cdot + soft.bias_rate * translation)
                    - soft.impulse_scale * self.spring_impulse;
            self.spring_impulse += impulse;
            push(&mut vel, impulse);
        }

        if self.enable_limit {
            // Lower.
            let (bias, soft) = step.constraint.limit_row(
                translation - self.lower_translation,
                ctx.inv_h,
                ctx.use_bias,
            );
            let cdot = axial_speed(&vel);
            let impulse = -soft.mass_scale * step.axial_mass * (cdot + bias)
                - soft.impulse_scale * self.lower_impulse;
            let old = self.lower_impulse;
            self.lower_impulse = (old + impulse).max(0.0);
            push(&mut vel, self.lower_impulse - old);

            // Upper, signs flipped so that the impulse is positive while it holds.
            let (bias, soft) = step.constraint.limit_row(
                self.upper_translation - translation,
                ctx.inv_h,
                ctx.use_bias,
            );
            let cdot = axis.dot(vel.v_a - vel.v_b) + a1 * vel.w_a - a2 * vel.w_b;
            let impulse = -soft.mass_scale * step.axial_mass * (cdot + bias)
                - soft.impulse_scale * self.upper_impulse;
            let old = self.upper_impulse;
            self.upper_impulse = (old + impulse).max(0.0);
            push(&mut vel, -(self.upper_impulse - old));
        }

        // Guide.
        let perp = left_perp(axis);
        let soft = step.constraint.rigid_row(ctx.use_bias);
        let bias = if ctx.use_bias {
            soft.bias_rate * perp.dot(d)
        } else {
            0.0
        };
        let s1 = cross(d + r_a, perp);
        let s2 = cross(r_b, perp);
        let cdot = perp.dot(vel.v_b - vel.v_a) + s2 * vel.w_b - s1 * vel.w_a;
        let impulse = -soft.mass_scale * step.perp_mass * (cdot + bias)
            - soft.impulse_scale * self.perp_impulse;
        self.perp_impulse += impulse;
        vel.apply(step, perp * impulse, impulse * s1, impulse * s2);
        vel.store(a, b);
    }
}

/// A weld joint: holds the frames together, rigidly or through linear and angular springs
/// (Box2D `b2WeldJoint`). A zero `linear_hertz` or `angular_hertz` makes that part rigid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WeldJoint2d {
    /// Frame on body A.
    pub local_frame_a: Transform2d,
    /// Frame on body B.
    pub local_frame_b: Transform2d,
    /// Linear spring stiffness in Hertz; zero is rigid.
    pub linear_hertz: f32,
    /// Linear spring damping ratio.
    pub linear_damping_ratio: f32,
    /// Angular spring stiffness in Hertz; zero is rigid.
    pub angular_hertz: f32,
    /// Angular spring damping ratio.
    pub angular_damping_ratio: f32,
    /// Stiffness of the rigid parts, in Hertz.
    pub constraint_hertz: f32,
    /// Damping ratio of the rigid parts.
    pub constraint_damping_ratio: f32,
    /// Accumulated linear impulse.
    pub linear_impulse: Vec2,
    /// Accumulated angular impulse.
    pub angular_impulse: f32,
}

impl Component for WeldJoint2d {}

impl Default for WeldJoint2d {
    /// Box2D `b2DefaultWeldJointDef`: rigid, at the bodies' origins.
    fn default() -> Self {
        Self {
            local_frame_a: Transform2d::IDENTITY,
            local_frame_b: Transform2d::IDENTITY,
            linear_hertz: 0.0,
            linear_damping_ratio: 0.0,
            angular_hertz: 0.0,
            angular_damping_ratio: 0.0,
            constraint_hertz: DEFAULT_CONSTRAINT_HERTZ,
            constraint_damping_ratio: DEFAULT_CONSTRAINT_DAMPING_RATIO,
            linear_impulse: Vec2::ZERO,
            angular_impulse: 0.0,
        }
    }
}

impl WeldJoint2d {
    /// A weld at `local_anchor_a` and `local_anchor_b`, both parts at `hertz` (rigid at zero).
    #[must_use]
    pub fn new(local_anchor_a: Vec2, local_anchor_b: Vec2, hertz: f32, damping_ratio: f32) -> Self {
        Self {
            local_frame_a: anchor_frame(local_anchor_a),
            local_frame_b: anchor_frame(local_anchor_b),
            linear_hertz: hertz,
            linear_damping_ratio: damping_ratio,
            angular_hertz: hertz,
            angular_damping_ratio: damping_ratio,
            ..Self::default()
        }
    }

    /// Sets the frames.
    #[must_use]
    pub const fn with_local_frames(mut self, frame_a: Transform2d, frame_b: Transform2d) -> Self {
        self.local_frame_a = frame_a;
        self.local_frame_b = frame_b;
        self
    }

    /// Makes the bodies rest when B is turned `angle` relative to A, by turning frame B.
    #[must_use]
    pub fn with_reference_angle(mut self, angle: f32) -> Self {
        reference_angle(&self.local_frame_a, &mut self.local_frame_b, angle);
        self
    }

    /// Softens the linear part.
    #[must_use]
    pub const fn with_linear_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.linear_hertz = hertz;
        self.linear_damping_ratio = damping_ratio;
        self
    }

    /// Softens the angular part.
    #[must_use]
    pub const fn with_angular_spring(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.angular_hertz = hertz;
        self.angular_damping_ratio = damping_ratio;
        self
    }

    /// Sets the softness of the rigid parts.
    #[must_use]
    pub const fn with_constraint_softness(mut self, hertz: f32, damping_ratio: f32) -> Self {
        self.constraint_hertz = hertz;
        self.constraint_damping_ratio = damping_ratio;
        self
    }

    fn prepare(&self, step: &mut JointStep2d, h: f32) {
        step.axial_mass = inverse_mass(step.inv_inertia_a + step.inv_inertia_b);
        step.spring = if self.linear_hertz == 0.0 {
            step.constraint
        } else {
            Softness2d::new(self.linear_hertz, self.linear_damping_ratio, h)
        };
        step.angular_spring = if self.angular_hertz == 0.0 {
            step.constraint
        } else {
            Softness2d::new(self.angular_hertz, self.angular_damping_ratio, h)
        };
    }

    fn warm_start(&self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        let r_a = a.delta_rotation.rotate(step.frame_a.translation);
        let r_b = b.delta_rotation.rotate(step.frame_b.translation);
        apply_pair(
            step,
            (a, b),
            (r_a, r_b),
            self.linear_impulse,
            self.angular_impulse,
        );
    }

    fn solve_rows(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        let mut vel = Velocities::of(a, b);

        // Angular.
        {
            let q_a = a.delta_rotation * step.frame_a.rotation;
            let q_b = b.delta_rotation * step.frame_b.rotation;
            let angle = rot_angle(inv_mul_rot(q_a, q_b));
            let (bias, soft) = if ctx.use_bias || self.angular_hertz > 0.0 {
                (step.angular_spring.bias_rate * angle, step.angular_spring)
            } else {
                (0.0, Softness2d::RIGID)
            };
            let cdot = vel.w_b - vel.w_a;
            let impulse = -soft.mass_scale * step.axial_mass * (cdot + bias)
                - soft.impulse_scale * self.angular_impulse;
            self.angular_impulse += impulse;
            vel.turn(step, impulse);
        }

        // Linear.
        {
            let (r_a, r_b, separation) = anchors(step, a, b);
            let (bias, soft) = if ctx.use_bias || self.linear_hertz > 0.0 {
                (separation * step.spring.bias_rate, step.spring)
            } else {
                (Vec2::ZERO, Softness2d::RIGID)
            };
            let cdot = (vel.v_b + cross_sv(vel.w_b, r_b)) - (vel.v_a + cross_sv(vel.w_a, r_a));
            let solved = point_mass(step, r_a, r_b).solve(cdot + bias);
            let impulse = Vec2::new(
                -soft.mass_scale * solved.x - soft.impulse_scale * self.linear_impulse.x,
                -soft.mass_scale * solved.y - soft.impulse_scale * self.linear_impulse.y,
            );
            self.linear_impulse += impulse;
            vel.apply(step, impulse, cross(r_a, impulse), cross(r_b, impulse));
        }
        vel.store(a, b);
    }
}

/// A motor joint: drives body B's frame relative to body A's (Box2D v3.2 `b2MotorJoint`).
/// Springs pull the frames together with limited force and torque; velocity rows drive their
/// relative velocities at `linear_velocity` and `angular_velocity`, and with zero target act as
/// top-down friction. A row with a zero maximum is off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct MotorJoint2d {
    /// Frame on body A: where the springs pull frame B.
    pub local_frame_a: Transform2d,
    /// Frame on body B.
    pub local_frame_b: Transform2d,
    /// Relative linear velocity to drive.
    pub linear_velocity: Vec2,
    /// Largest velocity force, in newtons.
    pub max_velocity_force: f32,
    /// Relative angular velocity to drive.
    pub angular_velocity: f32,
    /// Largest velocity torque, in N m.
    pub max_velocity_torque: f32,
    /// Linear spring stiffness in Hertz.
    pub linear_hertz: f32,
    /// Linear spring damping ratio.
    pub linear_damping_ratio: f32,
    /// Largest spring force, in newtons.
    pub max_spring_force: f32,
    /// Angular spring stiffness in Hertz.
    pub angular_hertz: f32,
    /// Angular spring damping ratio.
    pub angular_damping_ratio: f32,
    /// Largest spring torque, in N m.
    pub max_spring_torque: f32,
    /// Accumulated linear velocity impulse.
    pub linear_velocity_impulse: Vec2,
    /// Accumulated angular velocity impulse.
    pub angular_velocity_impulse: f32,
    /// Accumulated linear spring impulse.
    pub linear_spring_impulse: Vec2,
    /// Accumulated angular spring impulse.
    pub angular_spring_impulse: f32,
}

impl Component for MotorJoint2d {}

impl MotorJoint2d {
    /// Sets the frames.
    #[must_use]
    pub const fn with_local_frames(mut self, frame_a: Transform2d, frame_b: Transform2d) -> Self {
        self.local_frame_a = frame_a;
        self.local_frame_b = frame_b;
        self
    }

    /// Pulls the frames' origins together at `hertz` with at most `max_force`.
    #[must_use]
    pub const fn with_linear_spring(
        mut self,
        hertz: f32,
        damping_ratio: f32,
        max_force: f32,
    ) -> Self {
        self.linear_hertz = hertz;
        self.linear_damping_ratio = damping_ratio;
        self.max_spring_force = max_force;
        self
    }

    /// Turns the frames into line at `hertz` with at most `max_torque`.
    #[must_use]
    pub const fn with_angular_spring(
        mut self,
        hertz: f32,
        damping_ratio: f32,
        max_torque: f32,
    ) -> Self {
        self.angular_hertz = hertz;
        self.angular_damping_ratio = damping_ratio;
        self.max_spring_torque = max_torque;
        self
    }

    /// Drives the relative linear velocity with at most `max_force`.
    #[must_use]
    pub const fn with_linear_velocity(mut self, velocity: Vec2, max_force: f32) -> Self {
        self.linear_velocity = velocity;
        self.max_velocity_force = max_force;
        self
    }

    /// Drives the relative angular velocity with at most `max_torque`.
    #[must_use]
    pub const fn with_angular_velocity(mut self, velocity: f32, max_torque: f32) -> Self {
        self.angular_velocity = velocity;
        self.max_velocity_torque = max_torque;
        self
    }

    fn prepare(&self, step: &mut JointStep2d, h: f32) {
        step.spring = Softness2d::new(self.linear_hertz, self.linear_damping_ratio, h);
        step.angular_spring = Softness2d::new(self.angular_hertz, self.angular_damping_ratio, h);
        step.linear_mass =
            point_mass(step, step.frame_a.translation, step.frame_b.translation).inverse();
        step.angular_mass = inverse_mass(step.inv_inertia_a + step.inv_inertia_b);
    }

    fn warm_start(&self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        let r_a = a.delta_rotation.rotate(step.frame_a.translation);
        let r_b = b.delta_rotation.rotate(step.frame_b.translation);
        let linear = self.linear_velocity_impulse + self.linear_spring_impulse;
        let angular = self.angular_velocity_impulse + self.angular_spring_impulse;
        apply_pair(step, (a, b), (r_a, r_b), linear, angular);
    }

    fn solve_rows(
        &mut self,
        step: &mut JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        let mut vel = Velocities::of(a, b);

        // Angular spring.
        if self.max_spring_torque > 0.0 && self.angular_hertz > 0.0 {
            let q_a = a.delta_rotation * step.frame_a.rotation;
            let q_b = b.delta_rotation * step.frame_b.rotation;
            let c = rot_angle(inv_mul_rot(q_a, q_b));
            let soft = step.angular_spring;
            let cdot = vel.w_b - vel.w_a;
            let max = ctx.h * self.max_spring_torque;
            let old = self.angular_spring_impulse;
            let impulse = -soft.mass_scale * step.angular_mass * (cdot + soft.bias_rate * c)
                - soft.impulse_scale * old;
            self.angular_spring_impulse = clamp(old + impulse, -max, max);
            vel.turn(step, self.angular_spring_impulse - old);
        }

        // Angular velocity.
        if self.max_velocity_torque > 0.0 {
            let cdot = vel.w_b - vel.w_a - self.angular_velocity;
            let impulse = -step.angular_mass * cdot;
            let max = ctx.h * self.max_velocity_torque;
            let old = self.angular_velocity_impulse;
            self.angular_velocity_impulse = clamp(old + impulse, -max, max);
            vel.turn(step, self.angular_velocity_impulse - old);
        }

        let r_a = a.delta_rotation.rotate(step.frame_a.translation);
        let r_b = b.delta_rotation.rotate(step.frame_b.translation);

        // Linear spring.
        if self.max_spring_force > 0.0 && self.linear_hertz > 0.0 {
            let c = (b.delta_position - a.delta_position) + (r_b - r_a) + step.delta_center;
            let soft = step.spring;
            let cdot = anchor_velocity_of(&vel, r_a, r_b) + c * soft.bias_rate;
            step.linear_mass = point_mass(step, r_a, r_b).inverse();
            let solved = step.linear_mass.mul_vec(cdot);
            let old = self.linear_spring_impulse;
            let impulse = Vec2::new(
                -soft.mass_scale * solved.x - soft.impulse_scale * old.x,
                -soft.mass_scale * solved.y - soft.impulse_scale * old.y,
            );
            let max = ctx.h * self.max_spring_force;
            self.linear_spring_impulse += impulse;
            if self.linear_spring_impulse.length_squared() > max * max {
                self.linear_spring_impulse = normalize(self.linear_spring_impulse) * max;
            }
            let impulse = self.linear_spring_impulse - old;
            vel.apply(step, impulse, cross(r_a, impulse), cross(r_b, impulse));
        }

        // Linear velocity.
        if self.max_velocity_force > 0.0 {
            let cdot = anchor_velocity_of(&vel, r_a, r_b) - self.linear_velocity;
            let solved = step.linear_mass.mul_vec(cdot);
            let impulse = Vec2::new(-solved.x, -solved.y);
            let old = self.linear_velocity_impulse;
            let max = ctx.h * self.max_velocity_force;
            self.linear_velocity_impulse += impulse;
            if self.linear_velocity_impulse.length_squared() > max * max {
                self.linear_velocity_impulse = normalize(self.linear_velocity_impulse) * max;
            }
            let impulse = self.linear_velocity_impulse - old;
            vel.apply(step, impulse, cross(r_a, impulse), cross(r_b, impulse));
        }
        vel.store(a, b);
    }
}

/// The relative velocity of the anchors under velocities `vel`.
fn anchor_velocity_of(vel: &Velocities, r_a: Vec2, r_b: Vec2) -> Vec2 {
    (vel.v_b + cross_sv(vel.w_b, r_b)) - (vel.v_a + cross_sv(vel.w_a, r_a))
}

/// A mover joint: drives body B's linear velocity relative to body A's with a force limited
/// per axis, without turning either (Box2D v3.2 `b2MoverJoint`, for character movers).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct MoverJoint2d {
    /// Relative linear velocity to drive.
    pub linear_velocity: Vec2,
    /// Largest force along each axis; an axis with zero is free.
    pub max_velocity_force: Vec2,
    /// Accumulated impulse.
    pub linear_velocity_impulse: Vec2,
}

impl Component for MoverJoint2d {}

impl MoverJoint2d {
    /// Drives the relative velocity at `linear_velocity` with at most `max_velocity_force`.
    #[must_use]
    pub const fn new(linear_velocity: Vec2, max_velocity_force: Vec2) -> Self {
        Self {
            linear_velocity,
            max_velocity_force,
            linear_velocity_impulse: Vec2::ZERO,
        }
    }

    fn prepare(step: &mut JointStep2d) {
        step.inv_inertia_a = 0.0;
        step.inv_inertia_b = 0.0;
        step.axial_mass = inverse_mass(step.inv_mass_a + step.inv_mass_b);
    }

    fn warm_start(&self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        a.apply(step.inv_mass_a, 0.0, -self.linear_velocity_impulse, 0.0);
        b.apply(step.inv_mass_b, 0.0, self.linear_velocity_impulse, 0.0);
    }

    fn solve_rows(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        let mut vel = Velocities::of(a, b);
        if self.max_velocity_force.x > 0.0 || self.max_velocity_force.y > 0.0 {
            let cdot = (vel.v_b - vel.v_a) - self.linear_velocity;
            let solved = cdot * step.axial_mass;
            let old = self.linear_velocity_impulse;
            let max = self.max_velocity_force * ctx.h;
            let total = old - solved;
            self.linear_velocity_impulse =
                Vec2::new(clamp(total.x, -max.x, max.x), clamp(total.y, -max.y, max.y));
            let impulse = self.linear_velocity_impulse - old;
            vel.apply(step, impulse, 0.0, 0.0);
        } else {
            self.linear_velocity_impulse = Vec2::ZERO;
        }
        vel.store(a, b);
    }
}

/// A pogo joint: a spring along `normal` that holds body B's frame `rest_length` from body A's
/// along frame B's y-axis, pushing with at most `max_compression_force` and pulling with at
/// most `max_tension_force` (Box2D v3.2 `b2PogoJoint`, a character mover's suspension). Its
/// position correction keeps its own spring velocity rather than feeding the mover's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PogoJoint2d {
    /// Frame on body A.
    pub local_frame_a: Transform2d,
    /// Frame on body B; its y-axis is the pogo axis.
    pub local_frame_b: Transform2d,
    /// Direction the pogo pushes B along (the ground normal).
    pub normal: Vec2,
    /// Spring length.
    pub rest_length: f32,
    /// Spring stiffness in Hertz; zero turns the pogo off.
    pub hertz: f32,
    /// Spring damping ratio.
    pub damping_ratio: f32,
    /// Largest pulling force, in newtons.
    pub max_tension_force: f32,
    /// Largest pushing force, in newtons.
    pub max_compression_force: f32,
    /// Accumulated impulse.
    pub impulse: f32,
    /// The spring's own velocity.
    pub velocity: f32,
}

impl Component for PogoJoint2d {}

impl PogoJoint2d {
    /// A pogo along `normal` of `rest_length` at `hertz`, with unlimited force.
    #[must_use]
    pub const fn new(normal: Vec2, rest_length: f32, hertz: f32, damping_ratio: f32) -> Self {
        Self {
            local_frame_a: Transform2d::IDENTITY,
            local_frame_b: Transform2d::IDENTITY,
            normal,
            rest_length,
            hertz,
            damping_ratio,
            max_tension_force: f32::MAX,
            max_compression_force: f32::MAX,
            impulse: 0.0,
            velocity: 0.0,
        }
    }

    /// Sets the frames.
    #[must_use]
    pub const fn with_local_frames(mut self, frame_a: Transform2d, frame_b: Transform2d) -> Self {
        self.local_frame_a = frame_a;
        self.local_frame_b = frame_b;
        self
    }

    /// Bounds the pulling and pushing forces.
    #[must_use]
    pub const fn with_force_limits(mut self, max_tension: f32, max_compression: f32) -> Self {
        self.max_tension_force = max_tension;
        self.max_compression_force = max_compression;
        self
    }

    /// The distance from frame A's origin to frame B's along frame B's y-axis (Box2D
    /// `b2PogoJoint_GetLength`).
    #[must_use]
    pub fn length(&self, body_a: &RigidBody2d, body_b: &RigidBody2d) -> f32 {
        let axis = body_b
            .rotation
            .rotate(self.local_frame_b.rotation.rotate(Vec2::Y));
        let p_a = world_frame(body_a, &self.local_frame_a).translation;
        let p_b = world_frame(body_b, &self.local_frame_b).translation;
        (p_b - p_a).dot(axis)
    }

    fn prepare(&self, step: &mut JointStep2d) {
        let (r_a, r_b) = (step.frame_a.translation, step.frame_b.translation);
        let cr_a = cross(r_a, self.normal);
        let cr_b = cross(r_b, self.normal);
        step.axial_mass = inverse_mass(
            step.inv_mass_a
                + step.inv_mass_b
                + step.inv_inertia_a * cr_a * cr_a
                + step.inv_inertia_b * cr_b * cr_b,
        );
    }

    fn warm_start(&mut self, step: &JointStep2d, a: &mut BodyState2d, b: &mut BodyState2d) {
        if self.hertz == 0.0 {
            self.impulse = 0.0;
            return;
        }
        let r_a = a.delta_rotation.rotate(step.frame_a.translation);
        let r_b = b.delta_rotation.rotate(step.frame_b.translation);
        apply_pair(step, (a, b), (r_a, r_b), self.normal * self.impulse, 0.0);
    }

    fn solve_rows(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        if self.hertz == 0.0 {
            self.impulse = 0.0;
            return;
        }
        let mut vel = Velocities::of(a, b);
        let (r_a, r_b, d) = anchors(step, a, b);
        let mut bias = 0.0;
        if ctx.use_bias {
            // The pogo's own position correction, so as not to feed velocity into the mover.
            let axis = step.frame_b.rotation.rotate(Vec2::Y);
            let c = axis.dot(d) - self.rest_length;
            self.velocity = spring_damper(self.hertz, self.damping_ratio, c, self.velocity, ctx.h);
            bias = -self.velocity;
        }
        let cdot = self.normal.dot(anchor_velocity_of(&vel, r_a, r_b));
        let max_tension = ctx.h * self.max_tension_force;
        let max_compression = ctx.h * self.max_compression_force;
        let old = self.impulse;
        let impulse = -step.axial_mass * (cdot + bias);
        self.impulse = clamp(self.impulse + impulse, -max_tension, max_compression);
        let p = self.normal * (self.impulse - old);
        vel.apply(step, p, cross(r_a, p), cross(r_b, p));
        vel.store(a, b);
    }
}

/// A 2D joint (Box2D `b2JointType`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Joint2d {
    /// Rod or spring between two anchors.
    Distance(DistanceJoint2d),
    /// Only keeps the bodies from colliding (Box2D `b2_filterJoint`).
    Filter,
    /// Drives the relative pose and velocity.
    Motor(MotorJoint2d),
    /// Drives the relative linear velocity of a character mover.
    Mover(MoverJoint2d),
    /// A character mover's suspension.
    Pogo(PogoJoint2d),
    /// Slider.
    Prismatic(PrismaticJoint2d),
    /// Hinge.
    Revolute(RevoluteJoint2d),
    /// Weld.
    Weld(WeldJoint2d),
    /// Wheel suspension.
    Wheel(WheelJoint2d),
}

impl Component for Joint2d {}

impl Joint2d {
    /// The joint's frames on body A and body B (Box2D `localFrameA`, `localFrameB`); the
    /// origins for the frameless filter and mover joints.
    #[must_use]
    pub const fn local_frames(&self) -> (Transform2d, Transform2d) {
        match self {
            Self::Distance(j) => (j.local_frame_a, j.local_frame_b),
            Self::Motor(j) => (j.local_frame_a, j.local_frame_b),
            Self::Pogo(j) => (j.local_frame_a, j.local_frame_b),
            Self::Prismatic(j) => (j.local_frame_a, j.local_frame_b),
            Self::Revolute(j) => (j.local_frame_a, j.local_frame_b),
            Self::Weld(j) => (j.local_frame_a, j.local_frame_b),
            Self::Wheel(j) => (j.local_frame_a, j.local_frame_b),
            Self::Filter | Self::Mover(_) => (Transform2d::IDENTITY, Transform2d::IDENTITY),
        }
    }

    /// The stiffness and damping ratio of the joint's rigid rows.
    #[must_use]
    pub const fn constraint_tuning(&self) -> (f32, f32) {
        match self {
            Self::Distance(j) => (j.constraint_hertz, j.constraint_damping_ratio),
            Self::Prismatic(j) => (j.constraint_hertz, j.constraint_damping_ratio),
            Self::Revolute(j) => (j.constraint_hertz, j.constraint_damping_ratio),
            Self::Weld(j) => (j.constraint_hertz, j.constraint_damping_ratio),
            Self::Wheel(j) => (j.constraint_hertz, j.constraint_damping_ratio),
            Self::Filter | Self::Motor(_) | Self::Mover(_) | Self::Pogo(_) => {
                (DEFAULT_CONSTRAINT_HERTZ, DEFAULT_CONSTRAINT_DAMPING_RATIO)
            }
        }
    }

    /// The joint as Box2D's `b2Create*Joint` stores it, for a world of linear slop
    /// `linear_slop`: lengths and limits ordered and clamped.
    #[must_use]
    pub fn sanitized(mut self, linear_slop: f32, length_units: f32) -> Self {
        match &mut self {
            Self::Distance(j) => {
                let huge = HUGE_LENGTH * length_units;
                j.length = clamp(j.length, linear_slop, huge);
                j.min_length = clamp(j.min_length, linear_slop, huge);
                j.max_length = clamp(j.max_length, linear_slop, huge);
                let (min, max) = (
                    j.min_length.min(j.max_length),
                    j.min_length.max(j.max_length),
                );
                (j.min_length, j.max_length) = (min, max);
                let (lower, upper) = (j.lower_spring_force, j.upper_spring_force);
                j.lower_spring_force = lower.min(upper);
                j.upper_spring_force = lower.max(upper);
            }
            Self::Revolute(j) => {
                j.target_angle = clamp(j.target_angle, -PI, PI);
                j.set_limits(j.lower_angle, j.upper_angle);
            }
            Self::Prismatic(j) => j.set_limits(j.lower_translation, j.upper_translation),
            Self::Wheel(j) => j.set_limits(j.lower_translation, j.upper_translation),
            Self::Filter | Self::Motor(_) | Self::Mover(_) | Self::Pogo(_) | Self::Weld(_) => {}
        }
        self
    }

    /// Clears the accumulated impulses, so the next step starts cold (Box2D's prepare with
    /// warm starting off).
    pub fn reset_impulses(&mut self) {
        match self {
            Self::Distance(j) => {
                (j.impulse, j.lower_impulse, j.upper_impulse, j.motor_impulse) =
                    (0.0, 0.0, 0.0, 0.0);
            }
            Self::Motor(j) => {
                j.linear_velocity_impulse = Vec2::ZERO;
                j.angular_velocity_impulse = 0.0;
                j.linear_spring_impulse = Vec2::ZERO;
                j.angular_spring_impulse = 0.0;
            }
            Self::Mover(j) => j.linear_velocity_impulse = Vec2::ZERO,
            Self::Pogo(j) => j.impulse = 0.0,
            Self::Prismatic(j) => {
                j.impulse = Vec2::ZERO;
                (j.spring_impulse, j.motor_impulse) = (0.0, 0.0);
                (j.lower_impulse, j.upper_impulse) = (0.0, 0.0);
            }
            Self::Revolute(j) => {
                j.linear_impulse = Vec2::ZERO;
                (j.spring_impulse, j.motor_impulse) = (0.0, 0.0);
                (j.lower_impulse, j.upper_impulse) = (0.0, 0.0);
            }
            Self::Weld(j) => {
                j.linear_impulse = Vec2::ZERO;
                j.angular_impulse = 0.0;
            }
            Self::Wheel(j) => {
                (j.perp_impulse, j.spring_impulse, j.motor_impulse) = (0.0, 0.0, 0.0);
                (j.lower_impulse, j.upper_impulse) = (0.0, 0.0);
            }
            Self::Filter => {}
        }
    }

    /// Calls `mix` with each accumulated impulse, for the world's state hash.
    pub(crate) fn hash_impulses(&self, mix: &mut dyn FnMut(f32)) {
        let values: &[f32] = match self {
            Self::Distance(j) => &[j.impulse, j.lower_impulse, j.upper_impulse, j.motor_impulse],
            Self::Motor(j) => &[
                j.linear_velocity_impulse.x,
                j.linear_velocity_impulse.y,
                j.angular_velocity_impulse,
                j.linear_spring_impulse.x,
                j.linear_spring_impulse.y,
                j.angular_spring_impulse,
            ],
            Self::Mover(j) => &[j.linear_velocity_impulse.x, j.linear_velocity_impulse.y],
            Self::Pogo(j) => &[j.impulse, j.velocity],
            Self::Prismatic(j) => &[
                j.impulse.x,
                j.impulse.y,
                j.spring_impulse,
                j.motor_impulse,
                j.lower_impulse,
                j.upper_impulse,
            ],
            Self::Revolute(j) => &[
                j.linear_impulse.x,
                j.linear_impulse.y,
                j.spring_impulse,
                j.motor_impulse,
                j.lower_impulse,
                j.upper_impulse,
            ],
            Self::Weld(j) => &[j.linear_impulse.x, j.linear_impulse.y, j.angular_impulse],
            Self::Wheel(j) => &[
                j.perp_impulse,
                j.motor_impulse,
                j.spring_impulse,
                j.lower_impulse,
                j.upper_impulse,
            ],
            Self::Filter => &[],
        };
        values.iter().for_each(|&v| mix(v));
    }

    /// The joint's linear and angular reaction impulses over the last substep (Box2D
    /// `b2GetJointReaction`); times the substep rate they give the force and torque the joint
    /// events compare with their thresholds.
    #[must_use]
    pub fn reaction_impulses(&self) -> (f32, f32) {
        match self {
            Self::Distance(j) => (
                (j.impulse + j.lower_impulse - j.upper_impulse + j.motor_impulse).abs(),
                0.0,
            ),
            Self::Motor(j) => (
                (j.linear_velocity_impulse + j.linear_spring_impulse).length(),
                (j.angular_velocity_impulse + j.angular_spring_impulse).abs(),
            ),
            Self::Mover(j) => (j.linear_velocity_impulse.length(), 0.0),
            Self::Pogo(j) => (j.impulse.abs(), 0.0),
            Self::Prismatic(j) => {
                let axial = j.motor_impulse + j.lower_impulse - j.upper_impulse;
                (
                    (j.impulse.x * j.impulse.x + axial * axial).sqrt(),
                    j.impulse.y.abs(),
                )
            }
            Self::Revolute(j) => (
                j.linear_impulse.length(),
                (j.motor_impulse + j.lower_impulse - j.upper_impulse).abs(),
            ),
            Self::Weld(j) => (j.linear_impulse.length(), j.angular_impulse.abs()),
            Self::Wheel(j) => {
                let axial = j.spring_impulse + j.lower_impulse - j.upper_impulse;
                (
                    (j.perp_impulse * j.perp_impulse + axial * axial).sqrt(),
                    j.motor_impulse.abs(),
                )
            }
            Self::Filter => (0.0, 0.0),
        }
    }

    /// The force the joint applies to body B, given the substep rate `inv_h` and body A
    /// (Box2D `b2Joint_GetConstraintForce`).
    #[must_use]
    pub fn constraint_force(&self, body_a: &RigidBody2d, body_b: &RigidBody2d, inv_h: f32) -> Vec2 {
        let axis_a = |frame: &Transform2d| body_a.rotation.rotate(frame.rotation.x_axis());
        match self {
            Self::Distance(j) => {
                let p_a = world_frame(body_a, &j.local_frame_a).translation;
                let p_b = world_frame(body_b, &j.local_frame_b).translation;
                let axis = normalize(p_b - p_a);
                axis * ((j.impulse + j.lower_impulse - j.upper_impulse + j.motor_impulse) * inv_h)
            }
            Self::Motor(j) => (j.linear_velocity_impulse + j.linear_spring_impulse) * inv_h,
            Self::Mover(j) => j.linear_velocity_impulse * inv_h,
            Self::Pogo(j) => j.normal * (inv_h * j.impulse),
            Self::Prismatic(j) => {
                let axis = axis_a(&j.local_frame_a);
                let perp_force = inv_h * j.impulse.x;
                let axial_force = inv_h * (j.motor_impulse + j.lower_impulse - j.upper_impulse);
                left_perp(axis) * perp_force + axis * axial_force
            }
            Self::Revolute(j) => j.linear_impulse * inv_h,
            Self::Weld(j) => j.linear_impulse * inv_h,
            Self::Wheel(j) => {
                let axis = axis_a(&j.local_frame_a);
                let perp_force = inv_h * j.perp_impulse;
                let axial_force = inv_h * (j.spring_impulse + j.lower_impulse - j.upper_impulse);
                left_perp(axis) * perp_force + axis * axial_force
            }
            Self::Filter => Vec2::ZERO,
        }
    }

    /// The torque the joint applies to body B, given the substep rate `inv_h` (Box2D
    /// `b2Joint_GetConstraintTorque`).
    #[must_use]
    pub fn constraint_torque(&self, inv_h: f32) -> f32 {
        match self {
            Self::Motor(j) => inv_h * (j.angular_velocity_impulse + j.angular_spring_impulse),
            Self::Prismatic(j) => inv_h * j.impulse.y,
            Self::Revolute(j) => inv_h * (j.motor_impulse + j.lower_impulse - j.upper_impulse),
            Self::Weld(j) => inv_h * j.angular_impulse,
            Self::Wheel(j) => inv_h * j.motor_impulse,
            Self::Distance(_) | Self::Filter | Self::Mover(_) | Self::Pogo(_) => 0.0,
        }
    }

    /// How far the frames are from where the joint holds them (Box2D
    /// `b2Joint_GetLinearSeparation`): zero for joints that hold no position.
    #[must_use]
    pub fn linear_separation(&self, body_a: &RigidBody2d, body_b: &RigidBody2d) -> f32 {
        let (frame_a, frame_b) = self.local_frames();
        let p_a = world_frame(body_a, &frame_a).translation;
        let p_b = world_frame(body_b, &frame_b).translation;
        let dp = p_b - p_a;
        let slider = |enable_limit: bool, lower: f32, upper: f32| {
            let axis = body_a.rotation.rotate(frame_a.rotation.x_axis());
            let perpendicular = left_perp(axis).dot(dp).abs();
            let mut limit = 0.0;
            if enable_limit {
                let translation = axis.dot(dp);
                if translation < lower {
                    limit = lower - translation;
                }
                if upper < translation {
                    limit = translation - upper;
                }
            }
            (perpendicular * perpendicular + limit * limit).sqrt()
        };
        match self {
            Self::Distance(j) => {
                let length = dp.length();
                if !j.enable_spring {
                    (length - j.length).abs()
                } else if j.enable_limit && length < j.min_length {
                    j.min_length - length
                } else if j.enable_limit && length > j.max_length {
                    length - j.max_length
                } else {
                    0.0
                }
            }
            Self::Prismatic(j) => slider(j.enable_limit, j.lower_translation, j.upper_translation),
            Self::Wheel(j) => slider(j.enable_limit, j.lower_translation, j.upper_translation),
            Self::Revolute(_) => dp.length(),
            Self::Weld(j) if j.linear_hertz == 0.0 => dp.length(),
            Self::Weld(_) | Self::Filter | Self::Motor(_) | Self::Mover(_) | Self::Pogo(_) => 0.0,
        }
    }

    /// How far the bodies are turned from where the joint holds them (Box2D
    /// `b2Joint_GetAngularSeparation`), from the bodies' relative angle.
    #[must_use]
    pub fn angular_separation(&self, body_a: &RigidBody2d, body_b: &RigidBody2d) -> f32 {
        let relative = relative_angle(body_a.rotation, body_b.rotation);
        match self {
            Self::Prismatic(_) => relative,
            Self::Revolute(j) if j.enable_limit => {
                if relative < j.lower_angle {
                    j.lower_angle - relative
                } else if j.upper_angle < relative {
                    relative - j.upper_angle
                } else {
                    0.0
                }
            }
            Self::Weld(j) if j.angular_hertz == 0.0 => relative,
            _ => 0.0,
        }
    }

    /// Prepares the joint for a step of substep `h` and substep rate `inv_h` from the bodies'
    /// poses and world centres of mass at its start (Box2D `b2PrepareJoint`).
    pub(crate) fn prepare(
        &self,
        (body_a, body_b): (&RigidBody2d, &RigidBody2d),
        centers: (Vec2, Vec2),
        h: f32,
        inv_h: f32,
    ) -> JointStep2d {
        let frames = self.local_frames();
        let mut step = prepare_frames(
            (&frames.0, &frames.1),
            (body_a, body_b),
            centers,
            self.constraint_tuning(),
            h,
            inv_h,
        );
        match self {
            Self::Distance(j) => j.prepare(&mut step, h),
            Self::Motor(j) => j.prepare(&mut step, h),
            Self::Mover(_) => MoverJoint2d::prepare(&mut step),
            Self::Pogo(j) => j.prepare(&mut step),
            Self::Prismatic(j) => j.prepare(&mut step, h),
            Self::Revolute(j) => j.prepare(&mut step, h),
            Self::Weld(j) => j.prepare(&mut step, h),
            Self::Wheel(j) => j.prepare(&mut step, h),
            Self::Filter => {}
        }
        step
    }

    /// Re-applies the accumulated impulses (Box2D `b2WarmStartJoint`).
    pub(crate) fn warm_start_step(
        &mut self,
        step: &JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
    ) {
        match self {
            Self::Distance(j) => j.warm_start(step, a, b),
            Self::Motor(j) => j.warm_start(step, a, b),
            Self::Mover(j) => j.warm_start(step, a, b),
            Self::Pogo(j) => j.warm_start(step, a, b),
            Self::Prismatic(j) => j.warm_start(step, a, b),
            Self::Revolute(j) => j.warm_start(step, a, b),
            Self::Weld(j) => j.warm_start(step, a, b),
            Self::Wheel(j) => j.warm_start(step, a, b),
            Self::Filter => {}
        }
    }

    /// Solves the joint over one substep (Box2D `b2SolveJoint`).
    pub(crate) fn solve_step(
        &mut self,
        step: &mut JointStep2d,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        ctx: &JointSolveContext,
    ) {
        match self {
            Self::Distance(j) => j.solve_rows(step, a, b, ctx),
            Self::Motor(j) => j.solve_rows(step, a, b, ctx),
            Self::Mover(j) => j.solve_rows(step, a, b, ctx),
            Self::Pogo(j) => j.solve_rows(step, a, b, ctx),
            Self::Prismatic(j) => j.solve_rows(step, a, b, ctx),
            Self::Revolute(j) => j.solve_rows(step, a, b, ctx),
            Self::Weld(j) => j.solve_rows(step, a, b, ctx),
            Self::Wheel(j) => j.solve_rows(step, a, b, ctx),
            Self::Filter => {}
        }
    }

    /// Prepares the joint from the bodies' current poses and solves it once over `h` with
    /// position correction, for two bodies driven by hand rather than by a world.
    pub fn solve(&mut self, body_a: &mut RigidBody2d, body_b: &mut RigidBody2d, h: f32) {
        self.solve_by_hand(body_a, body_b, h, true);
    }

    /// As [`Self::solve`], without position correction (Box2D's relax pass).
    pub fn relax(&mut self, body_a: &mut RigidBody2d, body_b: &mut RigidBody2d, h: f32) {
        self.solve_by_hand(body_a, body_b, h, false);
    }

    /// Prepares the joint from the bodies' current poses and re-applies its accumulated
    /// impulses to them.
    pub fn warm_start(&self, body_a: &mut RigidBody2d, body_b: &mut RigidBody2d) {
        let mut joint = *self;
        let centers = (body_a.world_center(), body_b.world_center());
        let step = joint.prepare((body_a, body_b), centers, 1.0, 1.0);
        let (mut a, mut b) = (BodyState2d::at_rest(body_a), BodyState2d::at_rest(body_b));
        joint.warm_start_step(&step, &mut a, &mut b);
        a.store(body_a);
        b.store(body_b);
    }

    fn solve_by_hand(
        &mut self,
        body_a: &mut RigidBody2d,
        body_b: &mut RigidBody2d,
        h: f32,
        use_bias: bool,
    ) {
        if h <= 0.0 {
            return;
        }
        let inv_h = 1.0 / h;
        let centers = (body_a.world_center(), body_b.world_center());
        let mut step = self.prepare((body_a, body_b), centers, h, inv_h);
        let (mut a, mut b) = (BodyState2d::at_rest(body_a), BodyState2d::at_rest(body_b));
        let ctx = JointSolveContext {
            h,
            inv_h,
            use_bias,
            length_units: 1.0,
        };
        self.solve_step(&mut step, &mut a, &mut b, &ctx);
        a.store(body_a);
        b.store(body_b);
    }
}

/// Hand-driven solving for each joint type, through [`Joint2d`].
macro_rules! solve_by_hand {
    ($($ty:ident => $variant:ident),* $(,)?) => {$(
        impl $ty {
            /// Prepares the joint from the bodies' current poses and solves it once over `h`
            /// with position correction (see [`Joint2d::solve`]).
            pub fn solve(&mut self, body_a: &mut RigidBody2d, body_b: &mut RigidBody2d, h: f32) {
                let mut joint = Joint2d::$variant(*self);
                joint.solve(body_a, body_b, h);
                if let Joint2d::$variant(solved) = joint {
                    *self = solved;
                }
            }
        }
    )*};
}

solve_by_hand!(
    DistanceJoint2d => Distance,
    MotorJoint2d => Motor,
    MoverJoint2d => Mover,
    PogoJoint2d => Pogo,
    PrismaticJoint2d => Prismatic,
    RevoluteJoint2d => Revolute,
    WeldJoint2d => Weld,
    WheelJoint2d => Wheel,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revolute_limits_are_clamped_inside_plus_minus_pi() {
        let joint = RevoluteJoint2d::default().with_limits(4.0, -4.0);
        assert_eq!(joint.lower_angle, -MAX_REVOLUTE_LIMIT);
        assert_eq!(joint.upper_angle, MAX_REVOLUTE_LIMIT);

        let joint = RevoluteJoint2d::default().with_limits(0.5, -0.25);
        assert_eq!((joint.lower_angle, joint.upper_angle), (-0.25, 0.5));
    }
}
