//! Wide 3D contact solver: four colored single-manifold convex contacts at a time in [`FloatW`]
//! SIMD lanes, ported from Box3D's `contact_solver_wide.inl` (`b3ContactConstraintWide`).
//!
//! Each graph color's single-manifold convex contacts are packed four to a
//! [`WideContactConstraint3d`], with trailing empty lanes zeroed out. Within any graph color
//! `0..OVERFLOW_INDEX`, no two constraints share a dynamic body, so lanes never write the same
//! dynamic body's velocities and wide constraints of the same color can be solved in parallel
//! across worker threads with bit-identical results on any worker count.

use crate::body::{BodyType, RigidBody};
use crate::contact::Contact3d;
use crate::contact_solver::{BodyMotion3d, Softness3d, SPECULATIVE_DISTANCE};
use crate::narrowphase::ContactManifold;
use crate::parallel::Shared;
use korppi_math::{FloatW, Mat3, Quat, Vec3};

/// Contacts packed into one [`WideContactConstraint3d`] (`LANES = 4` on SSE2, NEON, and portable).
pub(crate) const WIDE: usize = korppi_math::simd::LANES;

/// Sentinel for an unused lane in a [`WideContactConstraint3d`].
pub(crate) const EMPTY_LANE: u32 = u32::MAX;

/// Smallest friction-centre weight of a contact point (`B3_MIN_FRICTION_WEIGHT`).
const MIN_FRICTION_WEIGHT: f32 = 1e-10;

/// Smallest determinant treated as invertible (`1000 * FLT_MIN`).
const MIN_DETERMINANT: f32 = 1000.0 * f32::MIN_POSITIVE;

/// Four lanes of 3D vectors (`b3Vec3W`).
#[derive(Debug, Clone, Copy, Default)]
struct Vec3W {
    x: FloatW,
    y: FloatW,
    z: FloatW,
}

/// Four lanes of quaternions (`b3QuatW`).
#[derive(Debug, Clone, Copy, Default)]
struct QuatW {
    v: Vec3W,
    w: FloatW,
}

/// Four lanes of column-major 3×3 matrices (`b3Mat33W`).
#[derive(Debug, Clone, Copy, Default)]
struct Mat3W {
    cx: Vec3W,
    cy: Vec3W,
    cz: Vec3W,
}

/// Four lanes of contact softness coefficients (`b3SoftnessW`).
#[derive(Debug, Clone, Copy, Default)]
struct SoftnessW {
    bias_rate: FloatW,
    mass_scale: FloatW,
    impulse_scale: FloatW,
}

/// Four lanes of body velocities and step motion deltas (`b3BodyStateW`).
#[derive(Debug, Clone, Copy)]
struct BodyStateW {
    v: Vec3W,
    w: Vec3W,
    dp: Vec3W,
    dq: QuatW,
    dynamic: [bool; WIDE],
}

/// One manifold contact point across four wide lanes (`b3ConstraintPointW`).
#[derive(Debug, Clone, Copy, Default)]
struct WidePoint3d {
    anchor_a: Vec3W,
    anchor_b: Vec3W,
    base_separation: FloatW,
    relative_velocity: FloatW,
    normal_impulse: FloatW,
    total_normal_impulse: FloatW,
    restitution_impulse: FloatW,
    normal_mass: FloatW,
    lever_arm: FloatW,
    /// All-ones mask in lanes where this point index is active (`< point_count`).
    active: FloatW,
}

/// Four single-manifold convex contacts of one graph color, solved together in [`FloatW`] SIMD
/// lanes (`b3ContactConstraintWide`).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WideContactConstraint3d {
    pub body_a: [u32; WIDE],
    pub body_b: [u32; WIDE],
    inv_mass_a: FloatW,
    inv_mass_b: FloatW,
    inv_i_a: Mat3W,
    inv_i_b: Mat3W,
    softness: SoftnessW,
    friction: FloatW,
    restitution: FloatW,
    rolling_resistance: FloatW,
    normal: Vec3W,
    tangent1: Vec3W,
    tangent2: Vec3W,
    tangent_speed: (FloatW, FloatW),
    center_a: Vec3W,
    center_b: Vec3W,
    tangent_mass: (FloatW, FloatW, FloatW),
    friction_impulse: (FloatW, FloatW),
    twist_mass: FloatW,
    twist_impulse: FloatW,
    rolling_mass: Mat3W,
    rolling_impulse: Vec3W,
    points: [WidePoint3d; 4],
    point_count: [usize; WIDE],
    active: FloatW,
}

#[inline(always)]
fn lanes<T>(items: &[T; WIDE], f: impl Fn(&T) -> f32) -> FloatW {
    FloatW::new(f(&items[0]), f(&items[1]), f(&items[2]), f(&items[3]))
}

#[inline(always)]
fn vec3_w<T>(items: &[T; WIDE], f: impl Fn(&T) -> Vec3) -> Vec3W {
    let v = [f(&items[0]), f(&items[1]), f(&items[2]), f(&items[3])];
    Vec3W {
        x: FloatW::new(v[0].x, v[1].x, v[2].x, v[3].x),
        y: FloatW::new(v[0].y, v[1].y, v[2].y, v[3].y),
        z: FloatW::new(v[0].z, v[1].z, v[2].z, v[3].z),
    }
}

#[inline(always)]
fn mat3_w<T>(items: &[T; WIDE], f: impl Fn(&T) -> Mat3) -> Mat3W {
    let m = [f(&items[0]), f(&items[1]), f(&items[2]), f(&items[3])];
    Mat3W {
        cx: vec3_w(&m, |m| m.x_axis),
        cy: vec3_w(&m, |m| m.y_axis),
        cz: vec3_w(&m, |m| m.z_axis),
    }
}

#[inline(always)]
fn add(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}

#[inline(always)]
fn sub(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

#[inline(always)]
fn scale(v: Vec3W, s: FloatW) -> Vec3W {
    Vec3W {
        x: v.x * s,
        y: v.y * s,
        z: v.z * s,
    }
}

#[inline(always)]
fn dot(a: Vec3W, b: Vec3W) -> FloatW {
    a.x * b.x + a.y * b.y + a.z * b.z
}

#[inline(always)]
fn cross(a: Vec3W, b: Vec3W) -> Vec3W {
    Vec3W {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

#[inline(always)]
fn blend_vec3(a: Vec3W, b: Vec3W, mask: FloatW) -> Vec3W {
    Vec3W {
        x: a.x.blend(b.x, mask),
        y: a.y.blend(b.y, mask),
        z: a.z.blend(b.z, mask),
    }
}

#[inline(always)]
fn mul_mat3(m: Mat3W, v: Vec3W) -> Vec3W {
    Vec3W {
        x: m.cx.x * v.x + m.cy.x * v.y + m.cz.x * v.z,
        y: m.cx.y * v.x + m.cy.y * v.y + m.cz.y * v.z,
        z: m.cx.z * v.x + m.cy.z * v.y + m.cz.z * v.z,
    }
}

#[inline(always)]
fn add_mat3(a: Mat3W, b: Mat3W) -> Mat3W {
    Mat3W {
        cx: add(a.cx, b.cx),
        cy: add(a.cy, b.cy),
        cz: add(a.cz, b.cz),
    }
}

/// Box3D `b3InvertTW`: symmetric 3×3 matrix inverse in wide lanes, zero where singular or masked.
#[inline(always)]
fn invert_symmetric_w(m: &Mat3W, active: FloatW) -> Mat3W {
    let zero = FloatW::zero();
    let one = FloatW::splat(1.0);
    let c0 = cross(m.cy, m.cz);
    let c1 = cross(m.cz, m.cx);
    let c2 = cross(m.cx, m.cy);
    let det = dot(m.cx, c0);
    let valid = det
        .abs()
        .greater_than(FloatW::splat(MIN_DETERMINANT))
        .and(active);
    let inv_det = zero.blend(one / det, valid);
    Mat3W {
        cx: scale(c0, inv_det),
        cy: scale(c1, inv_det),
        cz: scale(c2, inv_det),
    }
}

/// Box3D `b3NormalizeW`: normalizes each lane or returns zero if squared length `<= MIN_DETERMINANT`.
#[inline(always)]
fn normalize_w(a: Vec3W) -> Vec3W {
    let zero = FloatW::zero();
    let one = FloatW::splat(1.0);
    let len_sq = dot(a, a);
    let valid = len_sq.greater_than(FloatW::splat(MIN_DETERMINANT));
    let s = zero.blend(one / len_sq.sqrt(), valid);
    scale(a, s)
}

/// Box3D `b3PerpW`: unit vector perpendicular to unit vector `a` in each lane.
#[inline(always)]
fn perp_w(a: Vec3W) -> Vec3W {
    let zero = FloatW::zero();
    let mask =
        a.x.less_than(FloatW::splat(-0.5))
            .or(FloatW::splat(0.5).less_than(a.x));
    let p = Vec3W {
        x: zero.blend(a.y, mask),
        y: a.z.blend(zero - a.x, mask),
        z: (zero - a.y).blend(zero, mask),
    };
    normalize_w(p)
}

/// Rotates `v` by quaternion `q` in each lane, matching `Quat * Vec3` when `q != IDENTITY` and
/// preserving `v` untouched when `q == IDENTITY`.
#[inline(always)]
fn rotate_quat_w(q: QuatW, v: Vec3W) -> Vec3W {
    let zero = FloatW::zero();
    let one = FloatW::splat(1.0);
    let two = FloatW::splat(2.0);
    let is_id =
        q.v.x
            .equals(zero)
            .and(q.v.y.equals(zero))
            .and(q.v.z.equals(zero))
            .and(q.w.equals(one));
    let b2 = dot(q.v, q.v);
    let s1 = q.w * q.w - b2;
    let s2 = dot(v, q.v) * two;
    let s3 = q.w * two;
    let c = cross(q.v, v);
    let rotated = add(add(scale(v, s1), scale(q.v, s2)), scale(c, s3));
    blend_vec3(rotated, v, is_id)
}

/// Relative velocity of anchor B with respect to anchor A (`vr_b - vr_a`).
#[inline(always)]
fn relative_velocity_w(a: &BodyStateW, b: &BodyStateW, r_a: Vec3W, r_b: Vec3W) -> Vec3W {
    let vr_a = add(a.v, cross(a.w, r_a));
    let vr_b = add(b.v, cross(b.w, r_b));
    sub(vr_b, vr_a)
}

/// Current point separation in each lane.
#[inline(always)]
fn separation_w(cp: &WidePoint3d, a: &BodyStateW, b: &BodyStateW, normal: Vec3W) -> FloatW {
    let dp = sub(b.dp, a.dp);
    let r_a = rotate_quat_w(a.dq, cp.anchor_a);
    let r_b = rotate_quat_w(b.dq, cp.anchor_b);
    let ds = add(dp, sub(r_b, r_a));
    dot(ds, normal) + cp.base_separation
}

/// Gathers four bodies' velocities and step motions into wide lanes (`b3GatherBodies`).
///
/// # Safety
///
/// No other thread may write to the dynamic bodies referenced by `indices` during the call.
#[inline(always)]
unsafe fn gather(
    bodies: &Shared<'_, RigidBody>,
    motions: &Shared<'_, BodyMotion3d>,
    indices: &[u32; WIDE],
) -> BodyStateW {
    let mut v = [Vec3::ZERO; WIDE];
    let mut w = [Vec3::ZERO; WIDE];
    let mut dp = [Vec3::ZERO; WIDE];
    let mut dq = [Quat::IDENTITY; WIDE];
    let mut dynamic = [false; WIDE];

    for lane in 0..WIDE {
        let idx = indices[lane];
        if idx == EMPTY_LANE {
            continue;
        }
        // SAFETY: caller guarantees no concurrent write to this body or its motion.
        let body = unsafe { bodies.get(idx as usize) };
        let motion = if body.is_moving() {
            // SAFETY: caller guarantees no concurrent write to this body's motion.
            unsafe { motions.read(idx as usize) }
        } else {
            BodyMotion3d::NONE
        };
        v[lane] = body.linear_velocity;
        w[lane] = body.angular_velocity;
        dp[lane] = motion.delta_position;
        dq[lane] = motion.delta_rotation;
        dynamic[lane] = body.body_type == BodyType::Dynamic;
    }

    BodyStateW {
        v: vec3_w(&v, |v| *v),
        w: vec3_w(&w, |w| *w),
        dp: vec3_w(&dp, |dp| *dp),
        dq: QuatW {
            v: Vec3W {
                x: FloatW::new(dq[0].x, dq[1].x, dq[2].x, dq[3].x),
                y: FloatW::new(dq[0].y, dq[1].y, dq[2].y, dq[3].y),
                z: FloatW::new(dq[0].z, dq[1].z, dq[2].z, dq[3].z),
            },
            w: FloatW::new(dq[0].w, dq[1].w, dq[2].w, dq[3].w),
        },
        dynamic,
    }
}

/// Scatters updated linear and angular velocities back to dynamic bodies (`b3ScatterBodies`).
///
/// # Safety
///
/// No other thread may access the dynamic bodies in `indices` during the call.
#[inline(always)]
unsafe fn scatter(bodies: &Shared<'_, RigidBody>, indices: &[u32; WIDE], state: &BodyStateW) {
    let vx = state.v.x.to_array();
    let vy = state.v.y.to_array();
    let vz = state.v.z.to_array();
    let wx = state.w.x.to_array();
    let wy = state.w.y.to_array();
    let wz = state.w.z.to_array();
    for lane in 0..WIDE {
        if state.dynamic[lane] {
            let idx = indices[lane] as usize;
            // SAFETY: within a graph color, all dynamic bodies are distinct; caller guarantees
            // exclusive access to this dynamic body.
            let body = unsafe { bodies.get_mut(idx) };
            body.linear_velocity = Vec3::new(vx[lane], vy[lane], vz[lane]);
            body.angular_velocity = Vec3::new(wx[lane], wy[lane], wz[lane]);
        }
    }
}

#[derive(Clone, Copy)]
struct LaneInput {
    body_a: u32,
    body_b: u32,
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_i_a: Mat3,
    inv_i_b: Mat3,
    softness: Softness3d,
    friction: f32,
    restitution: f32,
    rolling_resistance: f32,
    normal: Vec3,
    tangent_velocity: Vec3,
    friction_impulse: Vec3,
    twist_impulse: f32,
    rolling_impulse: Vec3,
    point_count: usize,
    anchor_a: [Vec3; 4],
    anchor_b: [Vec3; 4],
    separation: [f32; 4],
    normal_impulse: [f32; 4],
    active: bool,
}

impl Default for LaneInput {
    fn default() -> Self {
        Self {
            body_a: EMPTY_LANE,
            body_b: EMPTY_LANE,
            inv_mass_a: 0.0,
            inv_mass_b: 0.0,
            inv_i_a: Mat3::ZERO,
            inv_i_b: Mat3::ZERO,
            softness: Softness3d::default(),
            friction: 0.0,
            restitution: 0.0,
            rolling_resistance: 0.0,
            normal: Vec3::X,
            tangent_velocity: Vec3::ZERO,
            friction_impulse: Vec3::ZERO,
            twist_impulse: 0.0,
            rolling_impulse: Vec3::ZERO,
            point_count: 0,
            anchor_a: [Vec3::ZERO; 4],
            anchor_b: [Vec3::ZERO; 4],
            separation: [0.0; 4],
            normal_impulse: [0.0; 4],
            active: false,
        }
    }
}

impl WideContactConstraint3d {
    /// Prepares up to four single-manifold convex contacts in [`FloatW`] SIMD lanes
    /// (Box3D `b3PrepareContacts_Wide`), returning the wide constraint and whether any lane has
    /// positive restitution.
    ///
    /// # Safety
    ///
    /// No other thread may write to the contacts, bodies, or motions of `slots` during the call.
    pub unsafe fn prepare(
        slots: [u32; WIDE],
        contacts: &Shared<'_, Option<Contact3d>>,
        bodies: &Shared<'_, RigidBody>,
        motions: &Shared<'_, BodyMotion3d>,
        inv_inertia: &[Mat3],
        (soft, stiff): (Softness3d, Softness3d),
        warm_start: bool,
    ) -> (Self, bool) {
        let mut inputs = [LaneInput::default(); WIDE];
        for lane in 0..WIDE {
            let slot = slots[lane];
            if slot == EMPTY_LANE {
                continue;
            }
            // SAFETY: caller guarantees no concurrent write to this contact slot.
            let Some(contact) = (unsafe { contacts.get(slot as usize).as_ref() }) else {
                continue;
            };
            let Some(manifold) = contact.manifolds.first() else {
                continue;
            };
            let (a, b) = (manifold.body_a as usize, manifold.body_b as usize);
            // SAFETY: caller guarantees bodies are read-only during prepare.
            let (body_a, body_b) = unsafe { (bodies.get(a), bodies.get(b)) };
            let touches_static =
                body_a.body_type == BodyType::Static || body_b.body_type == BodyType::Static;
            let i_a = if body_a.is_moving() {
                inv_inertia[a]
            } else {
                Mat3::ZERO
            };
            let i_b = if body_b.is_moving() {
                inv_inertia[b]
            } else {
                Mat3::ZERO
            };
            let mut anchor_a = [Vec3::ZERO; 4];
            let mut anchor_b = [Vec3::ZERO; 4];
            let mut separation = [0.0; 4];
            let mut normal_impulse = [0.0; 4];
            let point_count = manifold.points.len().min(4);
            for (j, mp) in manifold.points.iter().enumerate().take(point_count) {
                anchor_a[j] = mp.anchor_a;
                anchor_b[j] = mp.anchor_b;
                separation[j] = mp.separation;
                normal_impulse[j] = mp.normal_impulse;
            }
            inputs[lane] = LaneInput {
                body_a: manifold.body_a,
                body_b: manifold.body_b,
                inv_mass_a: body_a.inv_mass,
                inv_mass_b: body_b.inv_mass,
                inv_i_a: i_a,
                inv_i_b: i_b,
                softness: if touches_static { stiff } else { soft },
                friction: manifold.friction,
                restitution: manifold.restitution,
                rolling_resistance: manifold.rolling_resistance,
                normal: manifold.normal,
                tangent_velocity: manifold.tangent_velocity,
                friction_impulse: manifold.friction_impulse,
                twist_impulse: manifold.twist_impulse,
                rolling_impulse: manifold.rolling_impulse,
                point_count,
                anchor_a,
                anchor_b,
                separation,
                normal_impulse,
                active: true,
            };
        }

        let zero = FloatW::zero();
        let one = FloatW::splat(1.0);
        let warm = FloatW::splat(if warm_start { 1.0 } else { 0.0 });
        let body_a = inputs.map(|d| d.body_a);
        let body_b = inputs.map(|d| d.body_b);
        let point_count = inputs.map(|d| d.point_count);
        let point_count_w = lanes(&inputs, |d| d.point_count as f32);
        let active_lane = lanes(&inputs, |d| if d.active { 1.0 } else { 0.0 }).greater_than(zero);

        let m_a = lanes(&inputs, |d| d.inv_mass_a);
        let m_b = lanes(&inputs, |d| d.inv_mass_b);
        let i_a = mat3_w(&inputs, |d| d.inv_i_a);
        let i_b = mat3_w(&inputs, |d| d.inv_i_b);
        let normal = vec3_w(&inputs, |d| d.normal);
        let tangent1 = perp_w(normal);
        let tangent2 = cross(tangent1, normal);
        let tangent_vel = vec3_w(&inputs, |d| d.tangent_velocity);
        let friction_imp_in = vec3_w(&inputs, |d| d.friction_impulse);
        let rolling_imp_in = vec3_w(&inputs, |d| d.rolling_impulse);
        let rolling_resistance = lanes(&inputs, |d| d.rolling_resistance);
        let restitution = lanes(&inputs, |d| d.restitution);
        let rr_mask = rolling_resistance.greater_than(zero);

        // SAFETY: caller guarantees no concurrent write to bodies or motions during prepare.
        let (state_a, state_b) = unsafe {
            (
                gather(bodies, motions, &body_a),
                gather(bodies, motions, &body_b),
            )
        };

        let inv_tau = FloatW::splat(1.0 / SPECULATIVE_DISTANCE);
        let min_weight = FloatW::splat(MIN_FRICTION_WEIGHT);
        let two = FloatW::splat(2.0);
        let mut center_a = Vec3W::default();
        let mut center_b = Vec3W::default();
        let mut total_weight = zero;
        let mut points = [WidePoint3d::default(); 4];

        for (j, wp) in points.iter_mut().enumerate() {
            let active = point_count_w.greater_than(FloatW::splat(j as f32));
            let r_a = vec3_w(&inputs, |d| d.anchor_a[j]);
            let r_b = vec3_w(&inputs, |d| d.anchor_b[j]);
            let s = lanes(&inputs, |d| d.separation[j]);
            let mp_imp = lanes(&inputs, |d| d.normal_impulse[j]);

            let weight = (two - s * inv_tau).clamp_w(min_weight, one);
            let weight = zero.blend(weight, active);
            center_a = blend_vec3(center_a, add(center_a, scale(r_a, weight)), active);
            center_b = blend_vec3(center_b, add(center_b, scale(r_b, weight)), active);
            total_weight = total_weight.blend(total_weight + weight, active);

            let rn_a = cross(r_a, normal);
            let rn_b = cross(r_b, normal);
            let k_normal =
                m_a + m_b + dot(rn_a, mul_mat3(i_a, rn_a)) + dot(rn_b, mul_mat3(i_b, rn_b));
            let normal_mass = zero.blend(one / k_normal, k_normal.greater_than(zero).and(active));
            let rel_vel = zero.blend(
                dot(relative_velocity_w(&state_a, &state_b, r_a, r_b), normal),
                active,
            );
            *wp = WidePoint3d {
                anchor_a: r_a,
                anchor_b: r_b,
                base_separation: zero.blend(s - dot(sub(r_b, r_a), normal), active),
                relative_velocity: rel_vel,
                normal_impulse: zero.blend(warm * mp_imp, active),
                total_normal_impulse: zero,
                restitution_impulse: zero,
                normal_mass,
                lever_arm: zero,
                active,
            };
        }

        let inv_weight = zero.blend(one / total_weight, total_weight.greater_than(zero));
        center_a = scale(center_a, inv_weight);
        center_b = scale(center_b, inv_weight);
        for wp in &mut points {
            let d = sub(wp.anchor_a, center_a);
            wp.lever_arm = zero.blend(dot(d, d).sqrt(), wp.active);
        }

        let rt_a1 = cross(center_a, tangent1);
        let rt_a2 = cross(center_a, tangent2);
        let rt_b1 = cross(center_b, tangent1);
        let rt_b2 = cross(center_b, tangent2);
        let kxx = m_a + m_b + dot(rt_a1, mul_mat3(i_a, rt_a1)) + dot(rt_b1, mul_mat3(i_b, rt_b1));
        let kyy = m_a + m_b + dot(rt_a2, mul_mat3(i_a, rt_a2)) + dot(rt_b2, mul_mat3(i_b, rt_b2));
        let kxy = dot(rt_a1, mul_mat3(i_a, rt_a2)) + dot(rt_b1, mul_mat3(i_b, rt_b2));
        let det = kxx * kyy - kxy * kxy;
        let inv_det = zero.blend(
            one / det,
            det.abs()
                .greater_than(FloatW::splat(MIN_DETERMINANT))
                .and(active_lane),
        );
        let tangent_mass = (inv_det * kyy, zero - (inv_det * kxy), inv_det * kxx);

        let inv_i_ab = add_mat3(i_a, i_b);
        let k_twist = dot(normal, mul_mat3(inv_i_ab, normal));
        let twist_mass = zero.blend(one / k_twist, k_twist.greater_than(zero).and(active_lane));
        let rolling_mass = if rr_mask.any_true() {
            invert_symmetric_w(&inv_i_ab, rr_mask)
        } else {
            Mat3W::default()
        };

        let any_restitution = restitution.greater_than(zero).and(active_lane).any_true();
        (
            Self {
                body_a,
                body_b,
                inv_mass_a: m_a,
                inv_mass_b: m_b,
                inv_i_a: i_a,
                inv_i_b: i_b,
                softness: SoftnessW {
                    bias_rate: lanes(&inputs, |d| d.softness.bias_rate),
                    mass_scale: lanes(&inputs, |d| d.softness.mass_scale),
                    impulse_scale: lanes(&inputs, |d| d.softness.impulse_scale),
                },
                friction: lanes(&inputs, |d| d.friction),
                restitution,
                rolling_resistance,
                normal,
                tangent1,
                tangent2,
                tangent_speed: (dot(tangent_vel, tangent1), dot(tangent_vel, tangent2)),
                center_a,
                center_b,
                tangent_mass,
                friction_impulse: (
                    warm * dot(friction_imp_in, tangent1),
                    warm * dot(friction_imp_in, tangent2),
                ),
                twist_mass,
                twist_impulse: warm * lanes(&inputs, |d| d.twist_impulse),
                rolling_mass,
                rolling_impulse: blend_vec3(Vec3W::default(), scale(rolling_imp_in, warm), rr_mask),
                points,
                point_count,
                active: active_lane,
            },
            any_restitution,
        )
    }

    /// Applies linear impulse `p` and angular impulses `(l_a, l_b)` on lanes selected by `mask`.
    #[inline(always)]
    fn apply_masked(
        &self,
        a: &mut BodyStateW,
        b: &mut BodyStateW,
        p: Vec3W,
        (l_a, l_b): (Vec3W, Vec3W),
        mask: FloatW,
    ) {
        a.w = blend_vec3(a.w, sub(a.w, mul_mat3(self.inv_i_a, l_a)), mask);
        a.v = blend_vec3(a.v, sub(a.v, scale(p, self.inv_mass_a)), mask);
        b.w = blend_vec3(b.w, add(b.w, mul_mat3(self.inv_i_b, l_b)), mask);
        b.v = blend_vec3(b.v, add(b.v, scale(p, self.inv_mass_b)), mask);
    }

    /// Warm-starts the four lanes' contacts (`b3WarmStartContacts_Wide`).
    ///
    /// # Safety
    ///
    /// No other thread may access the dynamic bodies in `self.body_a` or `self.body_b`.
    pub unsafe fn warm_start(
        &self,
        bodies: &Shared<'_, RigidBody>,
        motions: &Shared<'_, BodyMotion3d>,
    ) {
        // SAFETY: caller guarantees exclusive access to dynamic bodies in this constraint.
        let (mut a, mut b) = unsafe {
            (
                gather(bodies, motions, &self.body_a),
                gather(bodies, motions, &self.body_b),
            )
        };
        let zero = FloatW::zero();
        let friction = add(
            scale(self.tangent1, self.friction_impulse.0),
            scale(self.tangent2, self.friction_impulse.1),
        );
        let twist = scale(self.normal, self.twist_impulse);
        let mut linear = friction;
        let mut angular_a = add(cross(self.center_a, friction), twist);
        let mut angular_b = add(cross(self.center_b, friction), twist);
        for cp in &self.points {
            if cp.active.all_zero() {
                break;
            }
            let p = scale(self.normal, cp.normal_impulse);
            linear = blend_vec3(linear, add(linear, p), cp.active);
            angular_a = blend_vec3(angular_a, add(angular_a, cross(cp.anchor_a, p)), cp.active);
            angular_b = blend_vec3(angular_b, add(angular_b, cross(cp.anchor_b, p)), cp.active);
        }
        let rr_mask = self.rolling_resistance.greater_than(zero);
        angular_a = blend_vec3(angular_a, add(angular_a, self.rolling_impulse), rr_mask);
        angular_b = blend_vec3(angular_b, add(angular_b, self.rolling_impulse), rr_mask);
        self.apply_masked(&mut a, &mut b, linear, (angular_a, angular_b), self.active);
        // SAFETY: as above.
        unsafe {
            scatter(bodies, &self.body_a, &a);
            scatter(bodies, &self.body_b, &b);
        }
    }

    /// Solves the biased normal pass across the four lanes (`b3PushContacts_Wide`).
    ///
    /// # Safety
    ///
    /// No other thread may access the dynamic bodies in `self.body_a` or `self.body_b`.
    pub unsafe fn push(
        &mut self,
        bodies: &Shared<'_, RigidBody>,
        motions: &Shared<'_, BodyMotion3d>,
        inv_h: f32,
        contact_speed: f32,
    ) {
        // SAFETY: caller guarantees exclusive access to dynamic bodies in this constraint.
        let (mut a, mut b) = unsafe {
            (
                gather(bodies, motions, &self.body_a),
                gather(bodies, motions, &self.body_b),
            )
        };
        let zero = FloatW::zero();
        let one = FloatW::splat(1.0);
        let inv_h_w = FloatW::splat(inv_h);
        let neg_contact_speed = FloatW::splat(-contact_speed);
        let softness = self.softness;

        for index in 0..4 {
            let cp = self.points[index];
            if cp.active.all_zero() {
                break;
            }
            let s = separation_w(&cp, &a, &b, self.normal);
            let separated = s.greater_than(zero);
            let overlap_bias =
                (softness.mass_scale * softness.bias_rate * s).max(neg_contact_speed);
            let velocity_bias = overlap_bias.blend(s * inv_h_w, separated);
            let mass_scale = softness.mass_scale.blend(one, separated);
            let impulse_scale = softness.impulse_scale.blend(zero, separated);

            let vn = dot(
                relative_velocity_w(&a, &b, cp.anchor_a, cp.anchor_b),
                self.normal,
            );
            let impulse = (zero - cp.normal_mass) * (mass_scale * vn + velocity_bias)
                - impulse_scale * cp.normal_impulse;
            let new_impulse = cp
                .normal_impulse
                .blend((cp.normal_impulse + impulse).max(zero), cp.active);
            let delta = new_impulse - cp.normal_impulse;
            self.points[index].normal_impulse = new_impulse;
            let p = scale(self.normal, delta);
            self.apply_masked(
                &mut a,
                &mut b,
                p,
                (cross(cp.anchor_a, p), cross(cp.anchor_b, p)),
                cp.active,
            );
        }

        // SAFETY: as above.
        unsafe {
            scatter(bodies, &self.body_a, &a);
            scatter(bodies, &self.body_b, &b);
        }
    }

    /// Solves the relax pass (unbiased normal rows, rolling resistance, twist friction, and
    /// central friction) across the four lanes (`b3SolveContacts_Wide`).
    ///
    /// # Safety
    ///
    /// No other thread may access the dynamic bodies in `self.body_a` or `self.body_b`.
    pub unsafe fn relax(
        &mut self,
        bodies: &Shared<'_, RigidBody>,
        motions: &Shared<'_, BodyMotion3d>,
        inv_h: f32,
    ) {
        // SAFETY: caller guarantees exclusive access to dynamic bodies in this constraint.
        let (mut a, mut b) = unsafe {
            (
                gather(bodies, motions, &self.body_a),
                gather(bodies, motions, &self.body_b),
            )
        };
        let zero = FloatW::zero();
        let eps = FloatW::splat(f32::EPSILON);
        let inv_h_w = FloatW::splat(inv_h);
        let mut total_normal_impulse = zero;
        let mut total_twist_limit = zero;

        for index in 0..4 {
            let cp = self.points[index];
            if cp.active.all_zero() {
                break;
            }
            let s = separation_w(&cp, &a, &b, self.normal);
            let velocity_bias = (s * inv_h_w).max(zero);
            let vn = dot(
                relative_velocity_w(&a, &b, cp.anchor_a, cp.anchor_b),
                self.normal,
            );
            let new_impulse = cp.normal_impulse.blend(
                (cp.normal_impulse - cp.normal_mass * (vn + velocity_bias)).max(zero),
                cp.active,
            );
            let delta = new_impulse - cp.normal_impulse;
            let point = &mut self.points[index];
            point.normal_impulse = new_impulse;
            point.total_normal_impulse = cp
                .total_normal_impulse
                .blend(cp.total_normal_impulse + new_impulse, cp.active);
            total_normal_impulse =
                total_normal_impulse.blend(total_normal_impulse + new_impulse, cp.active);
            total_twist_limit =
                total_twist_limit.blend(total_twist_limit + cp.lever_arm * new_impulse, cp.active);
            let p = scale(self.normal, delta);
            self.apply_masked(
                &mut a,
                &mut b,
                p,
                (cross(cp.anchor_a, p), cross(cp.anchor_b, p)),
                cp.active,
            );
        }

        // Rolling resistance
        let rr_mask = self.rolling_resistance.greater_than(zero).and(self.active);
        if rr_mask.any_true() {
            let old = self.rolling_impulse;
            let mut impulse = add(old, mul_mat3(self.rolling_mass, sub(a.w, b.w)));
            let max_impulse = self.rolling_resistance * total_normal_impulse;
            let length_squared = dot(impulse, impulse);
            let clamp_mask = length_squared.greater_than(eps + max_impulse * max_impulse);
            let clamp_scale = max_impulse / (length_squared.sqrt() + eps);
            impulse = blend_vec3(impulse, scale(impulse, clamp_scale), clamp_mask);
            self.rolling_impulse = blend_vec3(old, impulse, rr_mask);
            let delta = sub(self.rolling_impulse, old);
            a.w = blend_vec3(a.w, sub(a.w, mul_mat3(self.inv_i_a, delta)), rr_mask);
            b.w = blend_vec3(b.w, add(b.w, mul_mat3(self.inv_i_b, delta)), rr_mask);
        }

        // Twist friction about the normal
        let twist_mask = total_twist_limit
            .greater_than(zero)
            .or(self.twist_impulse.abs().greater_than(zero))
            .and(self.active);
        if twist_mask.any_true() {
            let twist_speed = dot(self.normal, sub(b.w, a.w));
            let max_impulse = self.friction * total_twist_limit;
            let old = self.twist_impulse;
            let clamped = (old - self.twist_mass * twist_speed).sym_clamp(max_impulse);
            self.twist_impulse = old.blend(clamped, twist_mask);
            let l = scale(self.normal, self.twist_impulse - old);
            a.w = blend_vec3(a.w, sub(a.w, mul_mat3(self.inv_i_a, l)), twist_mask);
            b.w = blend_vec3(b.w, add(b.w, mul_mat3(self.inv_i_b, l)), twist_mask);
        }

        // Central friction at the friction centre
        {
            let vr = relative_velocity_w(&a, &b, self.center_a, self.center_b);
            let vt0 = dot(vr, self.tangent1) - self.tangent_speed.0;
            let vt1 = dot(vr, self.tangent2) - self.tangent_speed.1;
            let (mxx, mxy, myy) = self.tangent_mass;
            let d0 = zero - (mxx * vt0 + mxy * vt1);
            let d1 = zero - (mxy * vt0 + myy * vt1);
            let mut new0 = self.friction_impulse.0 + d0;
            let mut new1 = self.friction_impulse.1 + d1;
            let max_impulse = self.friction * total_normal_impulse;
            let length_squared = new0 * new0 + new1 * new1;
            let over = length_squared.greater_than(max_impulse * max_impulse);
            let clamp_scale = max_impulse / (length_squared.sqrt() + eps);
            new0 = new0.blend(clamp_scale * new0, over);
            new1 = new1.blend(clamp_scale * new1, over);
            new0 = self.friction_impulse.0.blend(new0, self.active);
            new1 = self.friction_impulse.1.blend(new1, self.active);
            let delta0 = new0 - self.friction_impulse.0;
            let delta1 = new1 - self.friction_impulse.1;
            self.friction_impulse = (new0, new1);
            let p = add(scale(self.tangent1, delta0), scale(self.tangent2, delta1));
            self.apply_masked(
                &mut a,
                &mut b,
                p,
                (cross(self.center_a, p), cross(self.center_b, p)),
                self.active,
            );
        }

        // SAFETY: as above.
        unsafe {
            scatter(bodies, &self.body_a, &a);
            scatter(bodies, &self.body_b, &b);
        }
    }

    /// Applies one restitution pass across the four lanes (`b3ApplyRestitution_Wide`).
    ///
    /// # Safety
    ///
    /// No other thread may access the dynamic bodies in `self.body_a` or `self.body_b`.
    pub unsafe fn restitute(
        &mut self,
        bodies: &Shared<'_, RigidBody>,
        motions: &Shared<'_, BodyMotion3d>,
        inv_h: f32,
        threshold: f32,
    ) {
        let zero = FloatW::zero();
        let rest_mask = self.restitution.greater_than(zero).and(self.active);
        if rest_mask.all_zero() {
            return;
        }
        // SAFETY: caller guarantees exclusive access to dynamic bodies in this constraint.
        let (mut a, mut b) = unsafe {
            (
                gather(bodies, motions, &self.body_a),
                gather(bodies, motions, &self.body_b),
            )
        };
        let inv_h_w = FloatW::splat(inv_h);
        let neg_threshold = FloatW::splat(-threshold);
        let restitution = self.restitution;

        for index in 0..4 {
            let cp = self.points[index];
            let mask = cp.active.and(rest_mask);
            if mask.all_zero() {
                continue;
            }
            let compression = cp.total_normal_impulse - cp.restitution_impulse;
            let armed = cp
                .relative_velocity
                .less_than(neg_threshold)
                .and(compression.greater_than(zero));
            let spec_bias = (separation_w(&cp, &a, &b, self.normal) * inv_h_w).max(zero);
            let velocity_bias = spec_bias.blend(restitution * cp.relative_velocity, armed);
            let vn = dot(
                relative_velocity_w(&a, &b, cp.anchor_a, cp.anchor_b),
                self.normal,
            );
            let new_impulse = (cp.normal_impulse - cp.normal_mass * (vn + velocity_bias)).max(zero);
            let mut impulse = new_impulse - cp.normal_impulse;
            let approach = ((zero - cp.normal_mass) * vn)
                .max(zero)
                .min(impulse.max(zero));
            let allowance = restitution * (compression + approach) - cp.restitution_impulse;
            let armed_impulse = impulse.min(approach + allowance.max(zero));
            impulse = impulse.blend(armed_impulse, armed);
            impulse = zero.blend(impulse, mask);

            let point = &mut self.points[index];
            point.normal_impulse = cp.normal_impulse.blend(cp.normal_impulse + impulse, mask);
            point.restitution_impulse = cp
                .restitution_impulse
                .blend(cp.restitution_impulse + (impulse - approach), mask);
            point.total_normal_impulse = cp
                .total_normal_impulse
                .blend(cp.total_normal_impulse + impulse, mask);
            let p = scale(self.normal, impulse);
            self.apply_masked(
                &mut a,
                &mut b,
                p,
                (cross(cp.anchor_a, p), cross(cp.anchor_b, p)),
                mask,
            );
        }

        // SAFETY: as above.
        unsafe {
            scatter(bodies, &self.body_a, &a);
            scatter(bodies, &self.body_b, &b);
        }
    }

    /// Stores the accumulated impulses of `lane` back into `manifold` (`b3StoreImpulses_Wide`).
    pub fn store(&self, lane: usize, manifold: &mut ContactManifold) {
        let fi = add(
            scale(self.tangent1, self.friction_impulse.0),
            scale(self.tangent2, self.friction_impulse.1),
        );
        let fi_x = fi.x.to_array();
        let fi_y = fi.y.to_array();
        let fi_z = fi.z.to_array();
        let ti = self.twist_impulse.to_array();
        let ri_x = self.rolling_impulse.x.to_array();
        let ri_y = self.rolling_impulse.y.to_array();
        let ri_z = self.rolling_impulse.z.to_array();

        manifold.friction_impulse = Vec3::new(fi_x[lane], fi_y[lane], fi_z[lane]);
        manifold.twist_impulse = ti[lane];
        manifold.rolling_impulse = Vec3::new(ri_x[lane], ri_y[lane], ri_z[lane]);

        let count = self.point_count[lane].min(manifold.points.len());
        for (j, mp) in manifold.points.iter_mut().enumerate().take(count) {
            let cp = &self.points[j];
            mp.normal_impulse = cp.normal_impulse.to_array()[lane];
            mp.total_normal_impulse = cp.total_normal_impulse.to_array()[lane];
            mp.relative_velocity = cp.relative_velocity.to_array()[lane];
        }
    }
}
