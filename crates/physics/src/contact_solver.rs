//! 3D Soft Step contact solver, ported from Box3D's convex contact path
//! (`contact_solver_wide.inl`, one contact at a time as `contact_solver.c` does).
//!
//! A step prepares one [`ContactConstraint3d`] per touching manifold. Each substep warm starts
//! it, pushes the bodies apart with the soft bias ([`ContactConstraint3d::push`]) and, after the
//! positions are integrated, relaxes it without bias, with rolling resistance, twist friction
//! about the normal and central friction at the friction centre
//! ([`ContactConstraint3d::relax`]). After the substeps the restitution pass adds bounce, and
//! the accumulated impulses are stored back in the manifold to warm start the next step.
//!
//! Anchors are fixed relative to each body's centre of mass at prepare time and are the
//! Jacobians' lever arms; the current separation is the separation at prepare plus how far the
//! anchors have moved since, measured from each body's [`BodyMotion3d`]. Friction impulses are
//! kept as a world vector between steps, so they warm start in the plane of a new tangent basis.

use crate::body::RigidBody;
use crate::math;
use crate::narrowphase::ContactManifold;
use korppi_math::{Mat3, Quat, Vec3};

/// Box3D's linear slop: the allowed overlap, in metres (`B3_LINEAR_SLOP`).
pub const LINEAR_SLOP: f32 = 0.005;

/// Box3D's speculative distance, four linear slops (`B3_SPECULATIVE_DISTANCE`).
pub const SPECULATIVE_DISTANCE: f32 = 4.0 * LINEAR_SLOP;

/// The smallest friction-centre weight of a contact point (`B3_MIN_FRICTION_WEIGHT`).
const MIN_FRICTION_WEIGHT: f32 = 1e-10;

/// Box3D's smallest determinant treated as invertible, `1000 * FLT_MIN`.
const MIN_DETERMINANT: f32 = 1000.0 * f32::MIN_POSITIVE;

/// Soft constraint coefficients for one substep (Box3D `b3Softness`, `b3MakeSoft`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Softness3d {
    /// Velocity bias per unit of position error, 1/s.
    pub bias_rate: f32,
    /// Scale of the effective mass.
    pub mass_scale: f32,
    /// Scale of the accumulated impulse subtracted each iteration.
    pub impulse_scale: f32,
}

impl Softness3d {
    /// A spring of `hertz` and damping ratio `zeta` over a substep `h`; zero hertz gives no
    /// response at all, as in Box3D.
    #[must_use]
    pub fn new(hertz: f32, zeta: f32, h: f32) -> Self {
        if hertz == 0.0 {
            return Self::default();
        }
        let omega = 2.0 * std::f32::consts::PI * hertz;
        let a1 = 2.0 * zeta + h * omega;
        let a2 = h * omega * a1;
        let a3 = 1.0 / (1.0 + a2);
        Self {
            bias_rate: omega / a1,
            mass_scale: a2 * a3,
            impulse_scale: a3,
        }
    }
}

/// How far a body moved and turned since the start of the step (Box3D's body state
/// `deltaPosition` and `deltaRotation`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BodyMotion3d {
    pub delta_position: Vec3,
    pub delta_rotation: Quat,
}

impl BodyMotion3d {
    pub const NONE: Self = Self {
        delta_position: Vec3::ZERO,
        delta_rotation: Quat::IDENTITY,
    };
}

/// Box3D `b3Perp`: a unit vector perpendicular to the unit vector `a`.
pub(crate) fn perp(a: Vec3) -> Vec3 {
    let p = if a.x < -0.5 || 0.5 < a.x {
        Vec3::new(a.y, -a.x, 0.0)
    } else {
        Vec3::new(0.0, a.z, -a.y)
    };
    normalize(p)
}

/// Box3D `b3Normalize`: zero for vectors too short to normalize.
pub(crate) fn normalize(a: Vec3) -> Vec3 {
    let length_squared = a.x * a.x + a.y * a.y + a.z * a.z;
    if length_squared > MIN_DETERMINANT {
        let s = 1.0 / length_squared.sqrt();
        Vec3::new(s * a.x, s * a.y, s * a.z)
    } else {
        Vec3::ZERO
    }
}

/// One manifold point of a [`ContactConstraint3d`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ConstraintPoint3d {
    /// Contact point relative to body A's centre of mass at prepare time.
    anchor_a: Vec3,
    /// Contact point relative to body B's centre of mass at prepare time.
    anchor_b: Vec3,
    /// The point's normal, from A to B. Convex manifolds share one normal; a mesh manifold
    /// merges points of several triangles.
    normal: Vec3,
    /// Separation at prepare time minus the anchors' contribution.
    base_separation: f32,
    /// Normal velocity before the step, for restitution.
    relative_velocity: f32,
    normal_impulse: f32,
    /// Normal impulse applied over the relax passes and restitution of the step.
    total_normal_impulse: f32,
    /// Part of the normal impulse the restitution pass added beyond stopping the approach.
    restitution_impulse: f32,
    normal_mass: f32,
    /// Distance from the friction centre, scaling the point's share of twist friction.
    lever_arm: f32,
}

/// The solver's view of one touching manifold for the current step (Box3D
/// `b3ContactConstraint` with one `b3ManifoldConstraint`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ContactConstraint3d {
    pub body_a: usize,
    pub body_b: usize,
    dynamic_a: bool,
    dynamic_b: bool,
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_i_a: Mat3,
    inv_i_b: Mat3,
    softness: Softness3d,
    friction: f32,
    restitution: f32,
    rolling_resistance: f32,
    normal: Vec3,
    tangent1: Vec3,
    tangent2: Vec3,
    tangent_speed: (f32, f32),
    /// Friction centre relative to each body's centre of mass.
    center_a: Vec3,
    center_b: Vec3,
    /// Inverse of the 2×2 tangent effective mass at the friction centre (xx, xy, yy).
    tangent_mass: (f32, f32, f32),
    friction_impulse: (f32, f32),
    twist_mass: f32,
    twist_impulse: f32,
    rolling_mass: Mat3,
    rolling_impulse: Vec3,
    points: [ConstraintPoint3d; 4],
    point_count: usize,
    /// A mesh contact's manifold, solved in Box3D's scalar mesh order.
    mesh: bool,
}

/// Two distinct bodies of `bodies`, mutably.
pub(crate) fn body_pair(
    bodies: &mut [RigidBody],
    a: usize,
    b: usize,
) -> (&mut RigidBody, &mut RigidBody) {
    debug_assert_ne!(a, b);
    if a < b {
        let (left, right) = bodies.split_at_mut(b);
        (&mut left[a], &mut right[0])
    } else {
        let (left, right) = bodies.split_at_mut(a);
        (&mut right[0], &mut left[b])
    }
}

/// The relative velocity of two anchors (Box3D `vrB - vrA`).
fn relative_velocity(a: &RigidBody, b: &RigidBody, r_a: Vec3, r_b: Vec3) -> Vec3 {
    let vr_a = a.linear_velocity + a.angular_velocity.cross(r_a);
    let vr_b = b.linear_velocity + b.angular_velocity.cross(r_b);
    vr_b - vr_a
}

impl ContactConstraint3d {
    /// Prepares a manifold for the step (Box3D `b3PrepareContacts_Convex`), with the anchors its
    /// points had when it was built. `inv_i` holds each body's world inverse inertia; `soft` is
    /// the contact softness between movable bodies, `stiff` the one against a static body.
    /// `warm_start` keeps the manifold's impulses from the last step.
    pub fn prepare(
        manifold: &ContactManifold,
        bodies: &[RigidBody],
        inv_i: &[Mat3],
        (soft, stiff): (Softness3d, Softness3d),
        warm_start: bool,
    ) -> Self {
        let (a, b) = (manifold.body_a as usize, manifold.body_b as usize);
        let (body_a, body_b) = (&bodies[a], &bodies[b]);
        let (m_a, m_b) = (body_a.inv_mass, body_b.inv_mass);
        let (i_a, i_b) = (inv_i[a], inv_i[b]);
        let warm = if warm_start { 1.0 } else { 0.0 };
        let touches_static = body_a.body_type == crate::BodyType::Static
            || body_b.body_type == crate::BodyType::Static;

        let normal = manifold.normal;
        let tangent1 = perp(normal);
        let tangent2 = tangent1.cross(normal);

        let mut constraint = Self {
            body_a: a,
            body_b: b,
            dynamic_a: body_a.body_type == crate::BodyType::Dynamic,
            dynamic_b: body_b.body_type == crate::BodyType::Dynamic,
            inv_mass_a: m_a,
            inv_mass_b: m_b,
            inv_i_a: i_a,
            inv_i_b: i_b,
            softness: if touches_static { stiff } else { soft },
            friction: manifold.friction,
            restitution: manifold.restitution,
            rolling_resistance: manifold.rolling_resistance,
            normal,
            tangent1,
            tangent2,
            tangent_speed: (
                manifold.tangent_velocity.dot(tangent1),
                manifold.tangent_velocity.dot(tangent2),
            ),
            center_a: Vec3::ZERO,
            center_b: Vec3::ZERO,
            tangent_mass: (0.0, 0.0, 0.0),
            friction_impulse: (
                warm * manifold.friction_impulse.dot(tangent1),
                warm * manifold.friction_impulse.dot(tangent2),
            ),
            twist_mass: 0.0,
            twist_impulse: warm * manifold.twist_impulse,
            rolling_mass: Mat3::ZERO,
            rolling_impulse: if manifold.rolling_resistance > 0.0 {
                manifold.rolling_impulse * warm
            } else {
                Vec3::ZERO
            },
            points: [ConstraintPoint3d::default(); 4],
            point_count: manifold.points.len(),
            mesh: manifold.mesh,
        };

        let inv_tau = 1.0 / SPECULATIVE_DISTANCE;
        let (mut center_a, mut center_b) = (Vec3::ZERO, Vec3::ZERO);
        let mut total_weight = 0.0;
        for (cp, mp) in constraint.points.iter_mut().zip(manifold.points.iter()) {
            let (r_a, r_b) = (mp.anchor_a, mp.anchor_b);
            let n = mp.normal;
            let s = mp.separation;

            // Friction centre decay (Box3D: needed to keep spinning tops from drifting); points
            // far beyond the speculative distance hardly count.
            let weight = (2.0 - s * inv_tau).clamp(MIN_FRICTION_WEIGHT, 1.0);
            center_a += r_a * weight;
            center_b += r_b * weight;
            total_weight += weight;

            let rn_a = r_a.cross(n);
            let rn_b = r_b.cross(n);
            let k_normal = m_a + m_b + rn_a.dot(i_a * rn_a) + rn_b.dot(i_b * rn_b);
            let rel_vel = relative_velocity(body_a, body_b, r_a, r_b).dot(n);
            *cp = ConstraintPoint3d {
                anchor_a: r_a,
                anchor_b: r_b,
                normal: n,
                base_separation: s - (r_b - r_a).dot(n),
                relative_velocity: rel_vel,
                normal_impulse: warm * mp.normal_impulse,
                total_normal_impulse: 0.0,
                restitution_impulse: 0.0,
                normal_mass: if k_normal > 0.0 { 1.0 / k_normal } else { 0.0 },
                lever_arm: 0.0,
            };
        }
        let inv_weight = if total_weight > 0.0 {
            1.0 / total_weight
        } else {
            0.0
        };
        center_a *= inv_weight;
        center_b *= inv_weight;
        constraint.center_a = center_a;
        constraint.center_b = center_b;
        for cp in &mut constraint.points[..constraint.point_count] {
            cp.lever_arm = (cp.anchor_a - center_a).length();
        }

        let rt_a1 = center_a.cross(tangent1);
        let rt_a2 = center_a.cross(tangent2);
        let rt_b1 = center_b.cross(tangent1);
        let rt_b2 = center_b.cross(tangent2);
        let kxx = m_a + m_b + rt_a1.dot(i_a * rt_a1) + rt_b1.dot(i_b * rt_b1);
        let kyy = m_a + m_b + rt_a2.dot(i_a * rt_a2) + rt_b2.dot(i_b * rt_b2);
        let kxy = rt_a1.dot(i_a * rt_a2) + rt_b1.dot(i_b * rt_b2);
        let det = kxx * kyy - kxy * kxy;
        let inv_det = if det.abs() > MIN_DETERMINANT {
            1.0 / det
        } else {
            0.0
        };
        constraint.tangent_mass = (inv_det * kyy, -(inv_det * kxy), inv_det * kxx);

        let inv_i_ab = i_a + i_b;
        let k_twist = normal.dot(inv_i_ab * normal);
        constraint.twist_mass = if k_twist > 0.0 { 1.0 / k_twist } else { 0.0 };
        if constraint.rolling_resistance > 0.0 || constraint.mesh {
            constraint.rolling_mass = math::invert_symmetric(&inv_i_ab);
        }
        constraint
    }

    fn points(&self) -> &[ConstraintPoint3d] {
        &self.points[..self.point_count]
    }

    /// Applies a linear impulse `p` at the anchors and an angular impulse to each body.
    #[inline(always)]
    fn apply(&self, a: &mut RigidBody, b: &mut RigidBody, p: Vec3, (l_a, l_b): (Vec3, Vec3)) {
        if self.dynamic_a {
            a.angular_velocity -= self.inv_i_a * l_a;
            a.linear_velocity -= p * self.inv_mass_a;
        }
        if self.dynamic_b {
            b.angular_velocity += self.inv_i_b * l_b;
            b.linear_velocity += p * self.inv_mass_b;
        }
    }

    /// Applies the impulses accumulated so far (Box3D `b3WarmStartContacts_Convex`, or
    /// `b3WarmStartContacts_Mesh` impulse by impulse).
    pub fn warm_start(&self, bodies: &mut [RigidBody]) {
        let (a, b) = body_pair(bodies, self.body_a, self.body_b);
        if self.mesh {
            for cp in self.points() {
                let p = cp.normal * cp.normal_impulse;
                self.apply(a, b, p, (cp.anchor_a.cross(p), cp.anchor_b.cross(p)));
            }
            let p =
                self.tangent1 * self.friction_impulse.0 + self.tangent2 * self.friction_impulse.1;
            self.apply(a, b, p, (self.center_a.cross(p), self.center_b.cross(p)));
            let twist = self.normal * self.twist_impulse;
            a.angular_velocity -= self.inv_i_a * twist;
            b.angular_velocity += self.inv_i_b * twist;
            a.angular_velocity -= self.inv_i_a * self.rolling_impulse;
            b.angular_velocity += self.inv_i_b * self.rolling_impulse;
            return;
        }
        let friction =
            self.tangent1 * self.friction_impulse.0 + self.tangent2 * self.friction_impulse.1;
        let twist = self.normal * self.twist_impulse;
        let mut linear = friction;
        let mut angular_a = self.center_a.cross(friction) + twist;
        let mut angular_b = self.center_b.cross(friction) + twist;
        for cp in self.points() {
            let p = cp.normal * cp.normal_impulse;
            linear += p;
            angular_a += cp.anchor_a.cross(p);
            angular_b += cp.anchor_b.cross(p);
        }
        if self.rolling_resistance > 0.0 {
            angular_a += self.rolling_impulse;
            angular_b += self.rolling_impulse;
        }
        self.apply(a, b, linear, (angular_a, angular_b));
    }

    /// The current separation of a point: the separation at prepare plus how far its anchors
    /// moved along its normal since.
    #[inline(always)]
    fn separation(cp: &ConstraintPoint3d, motion_a: &BodyMotion3d, motion_b: &BodyMotion3d) -> f32 {
        let dp = motion_b.delta_position - motion_a.delta_position;
        let r_a = if motion_a.delta_rotation == Quat::IDENTITY {
            cp.anchor_a
        } else {
            motion_a.delta_rotation * cp.anchor_a
        };
        let r_b = if motion_b.delta_rotation == Quat::IDENTITY {
            cp.anchor_b
        } else {
            motion_b.delta_rotation * cp.anchor_b
        };
        let ds = dp + (r_b - r_a);
        ds.dot(cp.normal) + cp.base_separation
    }

    /// The biased normal pass: no friction, the soft bias for overlap capped at `contact_speed`,
    /// and the speculative bias for separated points (Box3D `b3PushContacts_Convex`).
    pub fn push(
        &mut self,
        bodies: &mut [RigidBody],
        motions: &[BodyMotion3d],
        inv_h: f32,
        contact_speed: f32,
    ) {
        let (motion_a, motion_b) = (&motions[self.body_a], &motions[self.body_b]);
        let (a, b) = body_pair(bodies, self.body_a, self.body_b);
        let softness = self.softness;
        for index in 0..self.point_count {
            let cp = self.points[index];
            let s = Self::separation(&cp, motion_a, motion_b);
            let (velocity_bias, mass_scale, impulse_scale) = if s > 0.0 {
                (s * inv_h, 1.0, 0.0)
            } else {
                (
                    (softness.mass_scale * softness.bias_rate * s).max(-contact_speed),
                    softness.mass_scale,
                    softness.impulse_scale,
                )
            };
            let vn = relative_velocity(a, b, cp.anchor_a, cp.anchor_b).dot(cp.normal);
            let impulse = -cp.normal_mass * (mass_scale * vn + velocity_bias)
                - impulse_scale * cp.normal_impulse;
            let new_impulse = (cp.normal_impulse + impulse).max(0.0);
            let delta = new_impulse - cp.normal_impulse;
            self.points[index].normal_impulse = new_impulse;
            let p = cp.normal * delta;
            self.apply(a, b, p, (cp.anchor_a.cross(p), cp.anchor_b.cross(p)));
        }
    }

    /// The relax pass: the normal rows without bias except the speculative one, then rolling
    /// resistance, twist friction and central friction (Box3D `b3SolveContacts_Convex`); a
    /// mesh manifold takes twist friction before rolling resistance (`b3SolveContacts_Mesh`).
    pub fn relax(&mut self, bodies: &mut [RigidBody], motions: &[BodyMotion3d], inv_h: f32) {
        if self.mesh {
            self.relax_mesh(bodies, motions, inv_h);
            return;
        }
        let (motion_a, motion_b) = (&motions[self.body_a], &motions[self.body_b]);
        let (a, b) = body_pair(bodies, self.body_a, self.body_b);
        let mut total_normal_impulse = 0.0;
        let mut total_twist_limit = 0.0;
        for index in 0..self.point_count {
            let cp = self.points[index];
            let s = Self::separation(&cp, motion_a, motion_b);
            let velocity_bias = (s * inv_h).max(0.0);
            let vn = relative_velocity(a, b, cp.anchor_a, cp.anchor_b).dot(cp.normal);
            let new_impulse = (cp.normal_impulse - cp.normal_mass * (vn + velocity_bias)).max(0.0);
            let delta = new_impulse - cp.normal_impulse;
            let point = &mut self.points[index];
            point.normal_impulse = new_impulse;
            point.total_normal_impulse += new_impulse;
            total_normal_impulse += new_impulse;
            total_twist_limit += cp.lever_arm * new_impulse;
            let p = cp.normal * delta;
            self.apply(a, b, p, (cp.anchor_a.cross(p), cp.anchor_b.cross(p)));
        }

        if self.rolling_resistance > 0.0 {
            let old = self.rolling_impulse;
            let mut impulse = old + self.rolling_mass * (a.angular_velocity - b.angular_velocity);
            let max_impulse = self.rolling_resistance * total_normal_impulse;
            let length_squared = impulse.dot(impulse);
            if length_squared > f32::EPSILON + max_impulse * max_impulse {
                // Box3D adds an epsilon to the length instead of an approximate square root.
                impulse *= max_impulse / (length_squared.sqrt() + f32::EPSILON);
            }
            self.rolling_impulse = impulse;
            let delta = impulse - old;
            a.angular_velocity -= self.inv_i_a * delta;
            b.angular_velocity += self.inv_i_b * delta;
        }

        // Twist friction about the normal, limited by each point's lever arm.
        if total_twist_limit > 0.0 || self.twist_impulse != 0.0 {
            let twist_speed = self.normal.dot(b.angular_velocity - a.angular_velocity);
            let max_impulse = self.friction * total_twist_limit;
            let old = self.twist_impulse;
            self.twist_impulse =
                (old - self.twist_mass * twist_speed).clamp(-max_impulse, max_impulse);
            let l = self.normal * (self.twist_impulse - old);
            a.angular_velocity -= self.inv_i_a * l;
            b.angular_velocity += self.inv_i_b * l;
        }

        // Central friction at the friction centre, within the Coulomb circle.
        {
            let vr = relative_velocity(a, b, self.center_a, self.center_b);
            let vt = (
                vr.dot(self.tangent1) - self.tangent_speed.0,
                vr.dot(self.tangent2) - self.tangent_speed.1,
            );
            let (mxx, mxy, myy) = self.tangent_mass;
            let delta = (-(mxx * vt.0 + mxy * vt.1), -(mxy * vt.0 + myy * vt.1));
            let mut new = (
                self.friction_impulse.0 + delta.0,
                self.friction_impulse.1 + delta.1,
            );
            let max_impulse = self.friction * total_normal_impulse;
            let length_squared = new.0 * new.0 + new.1 * new.1;
            if length_squared > max_impulse * max_impulse {
                let scale = max_impulse / (length_squared.sqrt() + f32::EPSILON);
                new = (scale * new.0, scale * new.1);
            }
            let delta = (
                new.0 - self.friction_impulse.0,
                new.1 - self.friction_impulse.1,
            );
            self.friction_impulse = new;
            let p = self.tangent1 * delta.0 + self.tangent2 * delta.1;
            self.apply(a, b, p, (self.center_a.cross(p), self.center_b.cross(p)));
        }
    }

    /// Box3D `b3SolveContacts_Mesh`: normal rows, twist friction, rolling resistance, then
    /// central friction, clamped without Box3D's convex epsilon.
    fn relax_mesh(&mut self, bodies: &mut [RigidBody], motions: &[BodyMotion3d], inv_h: f32) {
        let (motion_a, motion_b) = (&motions[self.body_a], &motions[self.body_b]);
        let (a, b) = body_pair(bodies, self.body_a, self.body_b);
        let mut total_normal_impulse = 0.0;
        let mut total_twist_limit = 0.0;
        for index in 0..self.point_count {
            let cp = self.points[index];
            let s = Self::separation(&cp, motion_a, motion_b);
            let velocity_bias = if s > 0.0 { s * inv_h } else { 0.0 };
            let vn = relative_velocity(a, b, cp.anchor_a, cp.anchor_b).dot(cp.normal);
            let new_impulse = (cp.normal_impulse - cp.normal_mass * (vn + velocity_bias)).max(0.0);
            let delta = new_impulse - cp.normal_impulse;
            let point = &mut self.points[index];
            point.normal_impulse = new_impulse;
            point.total_normal_impulse += new_impulse;
            total_normal_impulse += new_impulse;
            total_twist_limit += cp.lever_arm * new_impulse;
            let p = cp.normal * delta;
            self.apply(a, b, p, (cp.anchor_a.cross(p), cp.anchor_b.cross(p)));
        }

        {
            let twist_speed = self.normal.dot(b.angular_velocity - a.angular_velocity);
            let max_impulse = self.friction * total_twist_limit;
            let old = self.twist_impulse;
            self.twist_impulse =
                (old - self.twist_mass * twist_speed).clamp(-max_impulse, max_impulse);
            let l = self.normal * (self.twist_impulse - old);
            a.angular_velocity -= self.inv_i_a * l;
            b.angular_velocity += self.inv_i_b * l;
        }

        if self.rolling_resistance > 0.0 {
            let old = self.rolling_impulse;
            let mut impulse = old - self.rolling_mass * (b.angular_velocity - a.angular_velocity);
            let max_impulse = self.rolling_resistance * total_normal_impulse;
            let length_squared = impulse.dot(impulse);
            if length_squared > max_impulse * max_impulse + f32::EPSILON {
                impulse *= max_impulse / length_squared.sqrt();
            }
            self.rolling_impulse = impulse;
            let delta = impulse - old;
            a.angular_velocity -= self.inv_i_a * delta;
            b.angular_velocity += self.inv_i_b * delta;
        }

        {
            let vr = relative_velocity(a, b, self.center_a, self.center_b);
            let vt = (
                vr.dot(self.tangent1) - self.tangent_speed.0,
                vr.dot(self.tangent2) - self.tangent_speed.1,
            );
            let (mxx, mxy, myy) = self.tangent_mass;
            let delta = (-(mxx * vt.0 + mxy * vt.1), -(mxy * vt.0 + myy * vt.1));
            let mut new = (
                self.friction_impulse.0 + delta.0,
                self.friction_impulse.1 + delta.1,
            );
            let max_impulse = self.friction * total_normal_impulse;
            let length_squared = new.0 * new.0 + new.1 * new.1;
            if length_squared > max_impulse * max_impulse {
                let scale = max_impulse / length_squared.sqrt();
                new = (scale * new.0, scale * new.1);
            }
            let delta = (
                new.0 - self.friction_impulse.0,
                new.1 - self.friction_impulse.1,
            );
            self.friction_impulse = new;
            let p = self.tangent1 * delta.0 + self.tangent2 * delta.1;
            self.apply(a, b, p, (self.center_a.cross(p), self.center_b.cross(p)));
        }
    }

    /// Whether the restitution pass has anything to do for this contact.
    pub fn has_restitution(&self) -> bool {
        self.restitution > 0.0
    }

    /// One restitution pass (Box3D `b3ApplyRestitution_Convex`): points that approached faster
    /// than `threshold` and pushed rebound with Poisson's kinetic restitution, which never adds
    /// energy; the others keep only their speculative bias.
    pub fn restitute(
        &mut self,
        bodies: &mut [RigidBody],
        motions: &[BodyMotion3d],
        inv_h: f32,
        threshold: f32,
    ) {
        let (motion_a, motion_b) = (&motions[self.body_a], &motions[self.body_b]);
        let (a, b) = body_pair(bodies, self.body_a, self.body_b);
        let restitution = self.restitution;
        for index in 0..self.point_count {
            let cp = self.points[index];
            let compression = cp.total_normal_impulse - cp.restitution_impulse;
            let armed = cp.relative_velocity < -threshold && compression > 0.0;
            let velocity_bias = if armed {
                restitution * cp.relative_velocity
            } else {
                (Self::separation(&cp, motion_a, motion_b) * inv_h).max(0.0)
            };
            let vn = relative_velocity(a, b, cp.anchor_a, cp.anchor_b).dot(cp.normal);
            let new_impulse = (cp.normal_impulse - cp.normal_mass * (vn + velocity_bias)).max(0.0);
            let mut impulse = new_impulse - cp.normal_impulse;
            let approach = (-cp.normal_mass * vn).max(0.0).min(impulse.max(0.0));
            if armed {
                let allowance = restitution * (compression + approach) - cp.restitution_impulse;
                impulse = impulse.min(approach + allowance.max(0.0));
            }
            let point = &mut self.points[index];
            point.normal_impulse += impulse;
            point.restitution_impulse += impulse - approach;
            point.total_normal_impulse += impulse;
            let p = cp.normal * impulse;
            self.apply(a, b, p, (cp.anchor_a.cross(p), cp.anchor_b.cross(p)));
        }
    }

    /// Stores the accumulated impulses in the manifold for the next step's warm start (Box3D
    /// `b3StoreImpulses_Convex`).
    pub fn store(&self, manifold: &mut ContactManifold) {
        manifold.friction_impulse =
            self.tangent1 * self.friction_impulse.0 + self.tangent2 * self.friction_impulse.1;
        manifold.twist_impulse = self.twist_impulse;
        manifold.rolling_impulse = self.rolling_impulse;
        for (mp, cp) in manifold.points.iter_mut().zip(self.points()) {
            mp.normal_impulse = cp.normal_impulse;
            mp.total_normal_impulse = cp.total_normal_impulse;
            mp.relative_velocity = cp.relative_velocity;
        }
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    #[inline(always)]
    unsafe fn load_pair(&self, bodies: &crate::parallel::Shared<'_, RigidBody>) -> [RigidBody; 2] {
        // SAFETY: forwarded from caller.
        unsafe {
            [
                bodies.get(self.body_a).clone(),
                bodies.get(self.body_b).clone(),
            ]
        }
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    #[inline(always)]
    unsafe fn store_pair(
        &self,
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        pair: &[RigidBody; 2],
    ) {
        if self.dynamic_a {
            // SAFETY: caller guarantees exclusive access to this dynamic body.
            let a = unsafe { bodies.get_mut(self.body_a) };
            a.linear_velocity = pair[0].linear_velocity;
            a.angular_velocity = pair[0].angular_velocity;
        }
        if self.dynamic_b {
            // SAFETY: caller guarantees exclusive access to this dynamic body.
            let b = unsafe { bodies.get_mut(self.body_b) };
            b.linear_velocity = pair[1].linear_velocity;
            b.angular_velocity = pair[1].angular_velocity;
        }
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    pub unsafe fn warm_start_shared(&self, bodies: &crate::parallel::Shared<'_, RigidBody>) {
        // SAFETY: forwarded from caller.
        let mut pair = unsafe { self.load_pair(bodies) };
        let mut c = *self;
        c.body_a = 0;
        c.body_b = 1;
        c.warm_start(&mut pair);
        // SAFETY: forwarded from caller.
        unsafe { self.store_pair(bodies, &pair) };
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    pub unsafe fn push_shared(
        &mut self,
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        motions: &crate::parallel::Shared<'_, BodyMotion3d>,
        inv_h: f32,
        contact_speed: f32,
    ) {
        let (a_idx, b_idx) = (self.body_a, self.body_b);
        // SAFETY: forwarded from caller.
        let mut pair = unsafe { self.load_pair(bodies) };
        // SAFETY: motions are read-only during constraint solving.
        let motion_pair = unsafe { [motions.read(a_idx), motions.read(b_idx)] };
        self.body_a = 0;
        self.body_b = 1;
        self.push(&mut pair, &motion_pair, inv_h, contact_speed);
        self.body_a = a_idx;
        self.body_b = b_idx;
        // SAFETY: forwarded from caller.
        unsafe { self.store_pair(bodies, &pair) };
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    pub unsafe fn relax_shared(
        &mut self,
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        motions: &crate::parallel::Shared<'_, BodyMotion3d>,
        inv_h: f32,
    ) {
        let (a_idx, b_idx) = (self.body_a, self.body_b);
        // SAFETY: forwarded from caller.
        let mut pair = unsafe { self.load_pair(bodies) };
        // SAFETY: motions are read-only during constraint solving.
        let motion_pair = unsafe { [motions.read(a_idx), motions.read(b_idx)] };
        self.body_a = 0;
        self.body_b = 1;
        self.relax(&mut pair, &motion_pair, inv_h);
        self.body_a = a_idx;
        self.body_b = b_idx;
        // SAFETY: forwarded from caller.
        unsafe { self.store_pair(bodies, &pair) };
    }

    /// # Safety
    ///
    /// No other thread may access the dynamic bodies of `self` during the call.
    pub unsafe fn restitute_shared(
        &mut self,
        bodies: &crate::parallel::Shared<'_, RigidBody>,
        motions: &crate::parallel::Shared<'_, BodyMotion3d>,
        inv_h: f32,
        threshold: f32,
    ) {
        let (a_idx, b_idx) = (self.body_a, self.body_b);
        // SAFETY: forwarded from caller.
        let mut pair = unsafe { self.load_pair(bodies) };
        // SAFETY: motions are read-only during constraint solving.
        let motion_pair = unsafe { [motions.read(a_idx), motions.read(b_idx)] };
        self.body_a = 0;
        self.body_b = 1;
        self.restitute(&mut pair, &motion_pair, inv_h, threshold);
        self.body_a = a_idx;
        self.body_b = b_idx;
        // SAFETY: forwarded from caller.
        unsafe { self.store_pair(bodies, &pair) };
    }
}
