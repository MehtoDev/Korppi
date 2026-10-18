//! 2D Soft Step contact solver, ported from Box2D v3's `contact_solver.c`.
//!
//! A step prepares one [`ContactConstraint2d`] per touching manifold, then each substep warm
//! starts it, pushes the bodies apart with the soft bias ([`ContactConstraint2d::push`]) and,
//! after the positions are integrated, relaxes it without bias and with friction
//! ([`ContactConstraint2d::relax`]). After the substeps the restitution pass adds bounce, and
//! the accumulated impulses are stored back in the manifold to warm start the next step.
//!
//! Anchors are fixed relative to each body's centre of mass at prepare time; the current
//! separation is the separation at prepare plus how far the anchors have moved since, measured
//! from each body's [`BodyState2d`] motion. This module solves contacts one at a time (the
//! overflow); [`crate::solver_wide`] solves the colored ones four at a time.

use crate::body::{BodyType, RigidBody2d};
use crate::body_state::BodyState2d;
use crate::joint::Softness2d;
use crate::math::{cross, cross_sv, right_perp};
use crate::narrowphase::ContactManifold2d;
use korppi_math::{Rot2, Vec2};
use serde::{Deserialize, Serialize};

/// Largest speed at which overlapping bodies are pushed apart, in m/s (Box2D `contactSpeed`).
pub const CONTACT_PUSH_SPEED: f32 = 3.0;

/// Approach speed below which contacts do not bounce, in m/s (Box2D `restitutionThreshold`).
pub const RESTITUTION_THRESHOLD: f32 = 1.0;

/// Default passes of the restitution solver after the substeps (Box2D's default
/// `restitutionIterations`); see `WorldConfig2d::restitution_iterations`.
pub const RESTITUTION_ITERATIONS: usize = 2;

/// Configuration parameters for the 2D soft-step solver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig2d {
    /// Number of sub-steps per frame tick.
    pub sub_steps: usize,
    /// Contact stiffness in Hertz (Box2D `contactHertz`, default 30). Clamped to an eighth of
    /// the substep rate; contacts with static bodies are twice as stiff.
    pub hertz: f32,
    /// Contact damping ratio (Box2D `contactDampingRatio`, default 10).
    pub damping_ratio: f32,
    /// Coefficient of friction used by [`solve_contacts_2d`]; the world takes each contact's
    /// friction from its colliders.
    pub friction: f32,
    /// Coefficient of restitution used by [`solve_contacts_2d`]; the world takes each contact's
    /// restitution from its colliders.
    pub restitution: f32,
}

impl Default for SolverConfig2d {
    fn default() -> Self {
        Self {
            sub_steps: 4,
            hertz: 30.0,
            damping_ratio: 10.0,
            friction: 0.6,
            restitution: 0.0,
        }
    }
}

impl SolverConfig2d {
    /// Softness of contacts between movable bodies and of contacts with a static body, for
    /// substep `h` (Box2D `contactSoftness` and `staticSoftness`).
    #[must_use]
    pub fn contact_softness(&self, h: f32) -> (Softness2d, Softness2d) {
        contact_softness(self.hertz, self.damping_ratio, h)
    }
}

/// Contact softness at `hertz` and `damping_ratio` for substep `h`: between movable bodies,
/// and twice as stiff against a static body; the stiffness is capped at an eighth of the
/// substep rate (Box2D `contactSoftness`, `staticSoftness`).
#[must_use]
pub(crate) fn contact_softness(hertz: f32, damping_ratio: f32, h: f32) -> (Softness2d, Softness2d) {
    let hertz = hertz.min(0.125 / h);
    (
        Softness2d::new(hertz, damping_ratio, h),
        Softness2d::new(2.0 * hertz, damping_ratio, h),
    )
}

/// A body's centre of mass and rotation at the start of the step, and how far it moved and
/// turned during it (its [`BodyState2d`] motion): what continuous collision sweeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StepFrame2d {
    /// World centre of mass at the start of the step.
    pub center: Vec2,
    /// Rotation at the start of the step.
    pub rotation: Rot2,
    /// The centre's motion this step.
    pub delta_position: Vec2,
    /// The turn this step.
    pub delta_rotation: Rot2,
}

impl StepFrame2d {
    /// Records `body`'s current pose, not yet moved.
    #[must_use]
    pub fn of(body: &RigidBody2d) -> Self {
        Self {
            center: body.world_center(),
            rotation: body.rotation,
            delta_position: Vec2::ZERO,
            delta_rotation: Rot2::IDENTITY,
        }
    }
}

/// One manifold point of a [`ContactConstraint2d`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ContactConstraintPoint2d {
    /// Contact point relative to body A's centre of mass at prepare time.
    anchor_a: Vec2,
    /// Contact point relative to body B's centre of mass at prepare time.
    anchor_b: Vec2,
    /// Separation at prepare time minus the anchors' contribution.
    base_separation: f32,
    /// Normal velocity before the step, for restitution.
    relative_velocity: f32,
    normal_impulse: f32,
    tangent_impulse: f32,
    /// Normal impulse applied over the whole step, warm starts included.
    total_normal_impulse: f32,
    /// Part of the normal impulse the restitution pass added beyond stopping the approach.
    restitution_impulse: f32,
    normal_mass: f32,
    tangent_mass: f32,
}

/// The surface properties of one contact, mixed from its two colliders.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ContactMaterial2d {
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution coefficient.
    pub restitution: f32,
    /// Rolling resistance, already scaled by the larger shape radius (Box2D's contact
    /// `rollingResistance`).
    pub rolling_resistance: f32,
    /// Relative surface speed along the tangent (Box2D `tangentSpeed`).
    pub tangent_speed: f32,
}

/// The solver's view of one touching contact for the current step, solved one at a time: the
/// overflow contacts, which no graph color took (Box2D `b2ContactConstraint`,
/// `contact_solver.c`). Colored contacts go four at a time through
/// [`crate::solver_wide`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ContactConstraint2d {
    /// Body A's awake index plus one, zero if it is not awake (see [`BodyState2d`]).
    pub index_a: u32,
    /// Body B's awake index plus one.
    pub index_b: u32,
    inv_mass_a: f32,
    inv_inertia_a: f32,
    inv_mass_b: f32,
    inv_inertia_b: f32,
    normal: Vec2,
    friction: f32,
    restitution: f32,
    rolling_resistance: f32,
    tangent_speed: f32,
    /// Accumulated rolling impulse.
    rolling_impulse: f32,
    /// Effective mass of relative rotation, `1 / (iA + iB)`.
    rolling_mass: f32,
    softness: Softness2d,
    points: [ContactConstraintPoint2d; 2],
    point_count: usize,
}

/// Box2D's `vA = b2MulSub(vA, mA, P); wA -= iA * b2Cross(rA, P)` and the same added to B.
fn apply_impulse(
    (v_a, w_a): (&mut Vec2, &mut f32),
    (v_b, w_b): (&mut Vec2, &mut f32),
    c: &ContactConstraint2d,
    r_a: Vec2,
    r_b: Vec2,
    p: Vec2,
) {
    *v_a -= p * c.inv_mass_a;
    *w_a -= c.inv_inertia_a * cross(r_a, p);
    *v_b += p * c.inv_mass_b;
    *w_b += c.inv_inertia_b * cross(r_b, p);
}

/// The relative normal velocity of a point (Box2D `dot(vrB - vrA, normal)`).
fn normal_velocity(
    normal: Vec2,
    (v_a, w_a): (Vec2, f32),
    (v_b, w_b): (Vec2, f32),
    cp: &ContactConstraintPoint2d,
) -> f32 {
    let vr_a = v_a + cross_sv(w_a, cp.anchor_a);
    let vr_b = v_b + cross_sv(w_b, cp.anchor_b);
    (vr_b - vr_a).dot(normal)
}

impl ContactConstraint2d {
    /// Prepares the constraint for a step (Box2D `b2PrepareContacts_Overflow`). `indices` are
    /// the bodies' awake indices plus one; `manifold` holds the contact points as anchors
    /// relative to the bodies' centres of mass, and the impulses of the previous step, used to
    /// warm start. A contact with a body that is not awake (a static one) takes the stiffer
    /// `static_softness`.
    #[must_use]
    pub fn prepare(
        (index_a, index_b): (u32, u32),
        (body_a, body_b): (&RigidBody2d, &RigidBody2d),
        manifold: &ContactManifold2d,
        material: ContactMaterial2d,
        (contact_softness, static_softness): (Softness2d, Softness2d),
        warm_start_scale: f32,
    ) -> Self {
        let normal = manifold.normal;
        let tangent = right_perp(normal);
        let (m_a, i_a, m_b, i_b) = (
            body_a.inv_mass,
            body_a.inv_inertia,
            body_b.inv_mass,
            body_b.inv_inertia,
        );
        let mut points = [ContactConstraintPoint2d::default(); 2];
        for (cp, mp) in points
            .iter_mut()
            .zip(&manifold.points[..manifold.point_count])
        {
            let (r_a, r_b) = (mp.anchor_a, mp.anchor_b);
            let rn_a = cross(r_a, normal);
            let rn_b = cross(r_b, normal);
            let rt_a = cross(r_a, tangent);
            let rt_b = cross(r_b, tangent);
            let k_normal = m_a + m_b + i_a * rn_a * rn_a + i_b * rn_b * rn_b;
            let k_tangent = m_a + m_b + i_a * rt_a * rt_a + i_b * rt_b * rt_b;
            let v_a = body_a.linear_velocity + cross_sv(body_a.angular_velocity, r_a);
            let v_b = body_b.linear_velocity + cross_sv(body_b.angular_velocity, r_b);
            *cp = ContactConstraintPoint2d {
                anchor_a: r_a,
                anchor_b: r_b,
                base_separation: mp.separation - (r_b - r_a).dot(normal),
                relative_velocity: (v_b - v_a).dot(normal),
                normal_impulse: warm_start_scale * mp.normal_impulse,
                tangent_impulse: warm_start_scale * mp.tangent_impulse,
                total_normal_impulse: 0.0,
                restitution_impulse: 0.0,
                normal_mass: if k_normal > 0.0 { 1.0 / k_normal } else { 0.0 },
                tangent_mass: if k_tangent > 0.0 {
                    1.0 / k_tangent
                } else {
                    0.0
                },
            };
        }
        Self {
            index_a,
            index_b,
            inv_mass_a: m_a,
            inv_inertia_a: i_a,
            inv_mass_b: m_b,
            inv_inertia_b: i_b,
            normal,
            friction: material.friction,
            restitution: material.restitution,
            rolling_resistance: material.rolling_resistance,
            tangent_speed: material.tangent_speed,
            rolling_impulse: warm_start_scale * manifold.rolling_impulse,
            rolling_mass: if i_a + i_b > 0.0 {
                1.0 / (i_a + i_b)
            } else {
                0.0
            },
            softness: if index_a == 0 || index_b == 0 {
                static_softness
            } else {
                contact_softness
            },
            points,
            point_count: manifold.point_count,
        }
    }

    /// Whether this contact bounces (has restitution).
    #[must_use]
    pub fn has_restitution(&self) -> bool {
        self.restitution > 0.0
    }

    /// Current separation of a point: its separation at prepare time plus the anchors' motion.
    fn separation(&self, cp: &ContactConstraintPoint2d, a: &BodyState2d, b: &BodyState2d) -> f32 {
        let d = (b.delta_position - a.delta_position)
            + (b.delta_rotation.rotate(cp.anchor_b) - a.delta_rotation.rotate(cp.anchor_a));
        cp.base_separation + d.dot(self.normal)
    }

    /// Re-applies the impulses the constraint accumulated (Box2D `b2WarmStartContacts`).
    pub fn warm_start(&mut self, a: &mut BodyState2d, b: &mut BodyState2d) {
        let (mut v_a, mut w_a) = (a.linear_velocity, a.angular_velocity);
        let (mut v_b, mut w_b) = (b.linear_velocity, b.angular_velocity);
        let tangent = right_perp(self.normal);
        for i in 0..self.point_count {
            let cp = self.points[i];
            let p = self.normal * cp.normal_impulse + tangent * cp.tangent_impulse;
            self.points[i].total_normal_impulse += cp.normal_impulse;
            apply_impulse(
                (&mut v_a, &mut w_a),
                (&mut v_b, &mut w_b),
                self,
                cp.anchor_a,
                cp.anchor_b,
                p,
            );
        }
        w_a -= self.inv_inertia_a * self.rolling_impulse;
        w_b += self.inv_inertia_b * self.rolling_impulse;
        store(a, v_a, w_a);
        store(b, v_b, w_b);
    }

    /// Solves non-penetration with the soft bias that pushes overlapping bodies apart no faster
    /// than `push_speed`, without friction (Box2D `b2PushContacts`).
    pub fn push(&mut self, a: &mut BodyState2d, b: &mut BodyState2d, inv_h: f32, push_speed: f32) {
        let (mut v_a, mut w_a) = (a.linear_velocity, a.angular_velocity);
        let (mut v_b, mut w_b) = (b.linear_velocity, b.angular_velocity);
        for i in 0..self.point_count {
            let cp = self.points[i];
            let s = self.separation(&cp, a, b);
            let (bias, mass_scale, impulse_scale) = if s > 0.0 {
                // Speculative: allow the approach that just closes the gap.
                (s * inv_h, 1.0, 0.0)
            } else {
                (
                    (self.softness.mass_scale * self.softness.bias_rate * s).max(-push_speed),
                    self.softness.mass_scale,
                    self.softness.impulse_scale,
                )
            };
            let vn = normal_velocity(self.normal, (v_a, w_a), (v_b, w_b), &cp);
            let impulse =
                -cp.normal_mass * (mass_scale * vn + bias) - impulse_scale * cp.normal_impulse;
            let new_impulse = (cp.normal_impulse + impulse).max(0.0);
            let applied = new_impulse - cp.normal_impulse;
            let point = &mut self.points[i];
            point.normal_impulse = new_impulse;
            point.total_normal_impulse += applied;
            apply_impulse(
                (&mut v_a, &mut w_a),
                (&mut v_b, &mut w_b),
                self,
                cp.anchor_a,
                cp.anchor_b,
                self.normal * applied,
            );
        }
        store(a, v_a, w_a);
        store(b, v_b, w_b);
    }

    /// Solves non-penetration without the soft bias, then rolling resistance and friction
    /// (Box2D `b2SolveContacts`, the relax stage).
    pub fn relax(&mut self, a: &mut BodyState2d, b: &mut BodyState2d, inv_h: f32) {
        let (mut v_a, mut w_a) = (a.linear_velocity, a.angular_velocity);
        let (mut v_b, mut w_b) = (b.linear_velocity, b.angular_velocity);
        let mut total_normal_impulse = 0.0;
        for i in 0..self.point_count {
            let cp = self.points[i];
            let s = self.separation(&cp, a, b);
            // Speculative bias only; zero once overlapping.
            let bias = if s > 0.0 { s * inv_h } else { 0.0 };
            let vn = normal_velocity(self.normal, (v_a, w_a), (v_b, w_b), &cp);
            let impulse = -cp.normal_mass * (vn + bias);
            let new_impulse = (cp.normal_impulse + impulse).max(0.0);
            let applied = new_impulse - cp.normal_impulse;
            let point = &mut self.points[i];
            point.normal_impulse = new_impulse;
            point.total_normal_impulse += applied;
            total_normal_impulse += new_impulse;
            apply_impulse(
                (&mut v_a, &mut w_a),
                (&mut v_b, &mut w_b),
                self,
                cp.anchor_a,
                cp.anchor_b,
                self.normal * applied,
            );
        }

        // Rolling resistance: resist relative spin, up to the normal impulse times the
        // coefficient.
        let max_rolling = self.rolling_resistance * total_normal_impulse;
        let delta = -self.rolling_mass * (w_b - w_a);
        let new_rolling = (self.rolling_impulse + delta).clamp(-max_rolling, max_rolling);
        let applied = new_rolling - self.rolling_impulse;
        self.rolling_impulse = new_rolling;
        w_a -= self.inv_inertia_a * applied;
        w_b += self.inv_inertia_b * applied;

        self.solve_friction((&mut v_a, &mut w_a), (&mut v_b, &mut w_b));
        store(a, v_a, w_a);
        store(b, v_b, w_b);
    }

    /// Solves friction alone, for [`solve_contacts_2d`].
    fn apply_friction(&mut self, a: &mut BodyState2d, b: &mut BodyState2d) {
        let (mut v_a, mut w_a) = (a.linear_velocity, a.angular_velocity);
        let (mut v_b, mut w_b) = (b.linear_velocity, b.angular_velocity);
        self.solve_friction((&mut v_a, &mut w_a), (&mut v_b, &mut w_b));
        store(a, v_a, w_a);
        store(b, v_b, w_b);
    }

    /// Friction at each point, bounded by the point's normal impulse, toward the tangent speed.
    fn solve_friction(
        &mut self,
        (v_a, w_a): (&mut Vec2, &mut f32),
        (v_b, w_b): (&mut Vec2, &mut f32),
    ) {
        let tangent = right_perp(self.normal);
        for i in 0..self.point_count {
            let cp = self.points[i];
            let vr_a = *v_a + cross_sv(*w_a, cp.anchor_a);
            let vr_b = *v_b + cross_sv(*w_b, cp.anchor_b);
            let vt = (vr_b - vr_a).dot(tangent) - self.tangent_speed;
            let max_friction = self.friction * cp.normal_impulse;
            let new_impulse =
                (cp.tangent_impulse - cp.tangent_mass * vt).clamp(-max_friction, max_friction);
            let applied = new_impulse - cp.tangent_impulse;
            self.points[i].tangent_impulse = new_impulse;
            apply_impulse(
                (&mut *v_a, &mut *w_a),
                (&mut *v_b, &mut *w_b),
                self,
                cp.anchor_a,
                cp.anchor_b,
                tangent * applied,
            );
        }
    }

    /// Adds the bounce of points that hit faster than `threshold` (Box2D `b2ApplyRestitution`):
    /// Poisson restitution, which never adds energy.
    pub fn apply_restitution(
        &mut self,
        a: &mut BodyState2d,
        b: &mut BodyState2d,
        inv_h: f32,
        (threshold, propagate): (f32, bool),
    ) {
        if !propagate && self.restitution == 0.0 {
            return;
        }
        let (mut v_a, mut w_a) = (a.linear_velocity, a.angular_velocity);
        let (mut v_b, mut w_b) = (b.linear_velocity, b.angular_velocity);
        for i in 0..self.point_count {
            let cp = self.points[i];
            // The total normal impulse is zero for speculative points that never touched.
            let compression = cp.total_normal_impulse - cp.restitution_impulse;
            let armed =
                self.restitution > 0.0 && cp.relative_velocity < -threshold && compression > 0.0;
            let bias = if armed {
                self.restitution * cp.relative_velocity
            } else {
                let s = self.separation(&cp, a, b);
                if s > 0.0 {
                    s * inv_h
                } else {
                    0.0
                }
            };
            let vn = normal_velocity(self.normal, (v_a, w_a), (v_b, w_b), &cp);
            let mut impulse = -cp.normal_mass * (vn + bias);
            impulse = (cp.normal_impulse + impulse).max(0.0) - cp.normal_impulse;
            let approach = (-cp.normal_mass * vn).max(0.0).min(impulse.max(0.0));
            if armed {
                let allowance =
                    self.restitution * (compression + approach) - cp.restitution_impulse;
                impulse = impulse.min(approach + allowance.max(0.0));
            }
            let point = &mut self.points[i];
            point.normal_impulse += impulse;
            point.restitution_impulse += impulse - approach;
            point.total_normal_impulse += impulse;
            apply_impulse(
                (&mut v_a, &mut w_a),
                (&mut v_b, &mut w_b),
                self,
                cp.anchor_a,
                cp.anchor_b,
                self.normal * impulse,
            );
        }
        store(a, v_a, w_a);
        store(b, v_b, w_b);
    }

    /// Stores the accumulated impulses in the manifold, to warm start the next step.
    pub fn store(&self, manifold: &mut ContactManifold2d) {
        manifold.rolling_impulse = self.rolling_impulse;
        for (mp, cp) in manifold.points[..self.point_count]
            .iter_mut()
            .zip(&self.points)
        {
            mp.normal_impulse = cp.normal_impulse;
            mp.tangent_impulse = cp.tangent_impulse;
            mp.total_normal_impulse = cp.total_normal_impulse;
            mp.normal_velocity = cp.relative_velocity;
        }
    }
}

/// Writes solved velocities to a dynamic body's state (Box2D writes back only to dynamic
/// bodies, leaving the dummy and kinematic states untouched).
fn store(state: &mut BodyState2d, v: Vec2, w: f32) {
    if state.dynamic {
        state.linear_velocity = v;
        state.angular_velocity = w;
    }
}

/// Solves one manifold for one substep: prepare, warm start, push with the soft bias, then
/// friction, storing the impulses back in `manifold`.
///
/// The world solves all contacts in stages instead; this is for driving two bodies by hand. It
/// leaves out the rigid relax pass, which in the world follows the position integration: run
/// right after the push, it would cancel the push-out. `config.friction` and
/// `config.restitution` are the contact's material; restitution needs the world's restitution
/// pass and is not applied here.
pub fn solve_contacts_2d(
    body_a: &mut RigidBody2d,
    body_b: &mut RigidBody2d,
    manifold: &mut ContactManifold2d,
    sub_dt: f32,
    config: &SolverConfig2d,
) {
    if body_a.inv_mass + body_b.inv_mass == 0.0 || manifold.point_count == 0 {
        return;
    }
    let (soft, stiff) = config.contact_softness(sub_dt);
    let softness = if body_a.body_type == BodyType::Static || body_b.body_type == BodyType::Static {
        stiff
    } else {
        soft
    };
    // The solver takes anchors relative to the centres of mass.
    let mut centered = *manifold;
    let offset_a = body_a.rotation.rotate(body_a.local_center);
    let offset_b = body_b.rotation.rotate(body_b.local_center);
    for point in &mut centered.points[..centered.point_count] {
        point.anchor_a -= offset_a;
        point.anchor_b -= offset_b;
    }
    let mut constraint = ContactConstraint2d::prepare(
        (1, 2),
        (body_a, body_b),
        &centered,
        ContactMaterial2d {
            friction: config.friction,
            restitution: config.restitution,
            ..ContactMaterial2d::default()
        },
        (softness, softness),
        1.0,
    );
    let (mut a, mut b) = (BodyState2d::at_rest(body_a), BodyState2d::at_rest(body_b));
    let inv_h = 1.0 / sub_dt;
    constraint.warm_start(&mut a, &mut b);
    constraint.push(&mut a, &mut b, inv_h, CONTACT_PUSH_SPEED);
    constraint.apply_friction(&mut a, &mut b);
    a.store(body_a);
    b.store(body_b);
    constraint.store(manifold);
}
