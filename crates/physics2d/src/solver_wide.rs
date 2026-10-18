//! The wide contact solver: four colored contacts at a time in [`FloatW`] lanes, ported from
//! Box2D v3.2's `contact_solver_wide.inl` (`b2ContactConstraintWide`).
//!
//! A graph color's contacts are packed four to a [`WideContactConstraint2d`], the last one
//! padded with empty lanes (Box2D's zero contact). Each stage gathers the eight bodies' states
//! into lanes, solves the four contacts together and scatters the velocities back to the
//! dynamic bodies. A color's contacts share no dynamic body, so the lanes never write the same
//! state, and wide constraints of one color may be solved on different threads. The arithmetic
//! follows Box2D's to the operation, with SSE2's `min` and `max` on every platform (see
//! [`korppi_math::simd`]), so a step gives Box2D's bits.

use crate::body::RigidBody2d;
use crate::body_state::BodyState2d;
use crate::joint::Softness2d;
use crate::narrowphase::{ContactManifold2d, ManifoldPoint2d};
use crate::parallel::Shared;
use crate::solver::ContactMaterial2d;
use korppi_math::{FloatW, Vec2};

/// Contacts in a [`WideContactConstraint2d`] (Box2D `B2_SIMD_WIDTH` on SSE2 and NEON).
pub(crate) const WIDE: usize = korppi_math::simd::LANES;

/// Two lanes of vectors (Box2D `b2Vec2W`).
#[derive(Debug, Clone, Copy)]
struct Vec2W {
    x: FloatW,
    y: FloatW,
}

/// Four lanes of rotations (Box2D `b2RotW`).
#[derive(Debug, Clone, Copy)]
struct RotW {
    c: FloatW,
    s: FloatW,
}

/// Four body states in lanes (Box2D `b2BodyStateW`), with which lanes are dynamic.
#[derive(Debug, Clone, Copy)]
struct BodyStateW {
    v: Vec2W,
    w: FloatW,
    dp: Vec2W,
    dq: RotW,
    dynamic: [bool; WIDE],
}

/// `a.x * b.y - a.y * b.x` (Box2D `b2CrossW`).
#[inline(always)]
fn cross(a: Vec2W, b: Vec2W) -> FloatW {
    a.x * b.y - a.y * b.x
}

/// `a.x * b.x + a.y * b.y` (Box2D `b2DotW`).
#[inline(always)]
fn dot(a: Vec2W, b: Vec2W) -> FloatW {
    a.x * b.x + a.y * b.y
}

/// `v` rotated by `q` (Box2D `b2RotateVectorW`).
#[inline(always)]
fn rotate(q: RotW, v: Vec2W) -> Vec2W {
    Vec2W {
        x: q.c * v.x - q.s * v.y,
        y: q.s * v.x + q.c * v.y,
    }
}

/// The lanes of `f` applied to four items.
#[inline(always)]
fn lanes<T>(items: &[T; WIDE], f: impl Fn(&T) -> f32) -> FloatW {
    FloatW::new(f(&items[0]), f(&items[1]), f(&items[2]), f(&items[3]))
}

/// The states of four bodies by awake index plus one, zero giving the identity (Box2D
/// `b2GatherBodies`).
///
/// # Safety
///
/// No other thread may write the states of the lanes' bodies during the call.
#[inline(always)]
unsafe fn gather(states: &Shared<'_, BodyState2d>, indices: &[u32; WIDE]) -> BodyStateW {
    let s = indices.map(|index| {
        if index == 0 {
            BodyState2d::IDENTITY
        } else {
            // SAFETY: the caller guarantees no concurrent write to this state.
            unsafe { states.read(index as usize - 1) }
        }
    });
    BodyStateW {
        v: Vec2W {
            x: lanes(&s, |s| s.linear_velocity.x),
            y: lanes(&s, |s| s.linear_velocity.y),
        },
        w: lanes(&s, |s| s.angular_velocity),
        dp: Vec2W {
            x: lanes(&s, |s| s.delta_position.x),
            y: lanes(&s, |s| s.delta_position.y),
        },
        dq: RotW {
            c: lanes(&s, |s| s.delta_rotation.c),
            s: lanes(&s, |s| s.delta_rotation.s),
        },
        dynamic: s.map(|s| s.dynamic),
    }
}

/// Writes the lanes' velocities back to the dynamic bodies (Box2D `b2ScatterBodies`).
///
/// # Safety
///
/// No other thread may access the states of the lanes' dynamic bodies during the call.
#[inline(always)]
unsafe fn scatter(states: &Shared<'_, BodyState2d>, indices: &[u32; WIDE], body: &BodyStateW) {
    let (vx, vy, w) = (body.v.x.to_array(), body.v.y.to_array(), body.w.to_array());
    for lane in 0..WIDE {
        if body.dynamic[lane] {
            // SAFETY: a dynamic lane has a non-zero index; the caller guarantees exclusive
            // access to its state.
            let state = unsafe { states.get_mut(indices[lane] as usize - 1) };
            state.linear_velocity = Vec2::new(vx[lane], vy[lane]);
            state.angular_velocity = w[lane];
        }
    }
}

/// `1 / k` where `k > 0`, else zero (Box2D's blended effective mass).
#[inline(always)]
fn effective_mass(k: FloatW) -> FloatW {
    let zero = FloatW::zero();
    zero.blend(FloatW::splat(1.0) / k, k.greater_than(zero))
}

/// The velocity of B's anchor relative to A's (Box2D's `dvx`, `dvy`).
#[inline(always)]
fn relative_velocity(a: &BodyStateW, b: &BodyStateW, r_a: Vec2W, r_b: Vec2W) -> Vec2W {
    Vec2W {
        x: (b.v.x - b.w * r_b.y) - (a.v.x - a.w * r_a.y),
        y: (b.v.y + b.w * r_b.x) - (a.v.y + a.w * r_a.x),
    }
}

/// The current separation: the base separation plus the anchors' motion along the normal.
#[inline(always)]
fn separation(
    a: &BodyStateW,
    b: &BodyStateW,
    dp: Vec2W,
    (r_a, r_b): (Vec2W, Vec2W),
    normal: Vec2W,
    base: FloatW,
) -> FloatW {
    let rs_a = rotate(a.dq, r_a);
    let rs_b = rotate(b.dq, r_b);
    let ds = Vec2W {
        x: dp.x + (rs_b.x - rs_a.x),
        y: dp.y + (rs_b.y - rs_a.y),
    };
    dot(normal, ds) + base
}

/// One point of a [`WideContactConstraint2d`], in lanes.
#[derive(Debug, Clone, Copy)]
struct WidePoint {
    anchor_a: Vec2W,
    anchor_b: Vec2W,
    base_separation: FloatW,
    normal_mass: FloatW,
    tangent_mass: FloatW,
    normal_impulse: FloatW,
    tangent_impulse: FloatW,
    total_normal_impulse: FloatW,
    relative_velocity: FloatW,
    restitution_impulse: FloatW,
}

/// Four contacts of one graph color, solved together (Box2D `b2ContactConstraintWide`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct WideContactConstraint2d {
    /// Body A's awake index plus one in each lane, zero for a body that is not awake.
    index_a: [u32; WIDE],
    /// Body B's awake index plus one in each lane.
    index_b: [u32; WIDE],
    inv_mass_a: FloatW,
    inv_mass_b: FloatW,
    inv_i_a: FloatW,
    inv_i_b: FloatW,
    normal: Vec2W,
    friction: FloatW,
    tangent_speed: FloatW,
    rolling_resistance: FloatW,
    rolling_impulse: FloatW,
    restitution: FloatW,
    points: [WidePoint; 2],
}

/// What a lane of a wide constraint is prepared from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WideLane<'a> {
    /// The bodies' awake indices plus one.
    pub indices: (u32, u32),
    pub bodies: (&'a RigidBody2d, &'a RigidBody2d),
    pub manifold: &'a ContactManifold2d,
    pub material: ContactMaterial2d,
}

/// A lane's scalar inputs; an empty lane is all zero (Box2D's `b2_zeroContactSim`).
#[derive(Debug, Clone, Copy, Default)]
struct LaneData {
    indices: (u32, u32),
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_i_a: f32,
    inv_i_b: f32,
    normal: Vec2,
    material: ContactMaterial2d,
    rolling_impulse: f32,
    point_count: f32,
    /// The manifold's points; a point beyond its count is zero, as in Box2D's manifolds.
    points: [ManifoldPoint2d; 2],
}

impl LaneData {
    fn of(lane: Option<&WideLane<'_>>) -> Self {
        let Some(lane) = lane else {
            return Self::default();
        };
        let (a, b) = lane.bodies;
        let manifold = lane.manifold;
        let mut points = [ManifoldPoint2d::default(); 2];
        points[..manifold.point_count].copy_from_slice(&manifold.points[..manifold.point_count]);
        Self {
            indices: lane.indices,
            inv_mass_a: a.inv_mass,
            inv_mass_b: b.inv_mass,
            inv_i_a: a.inv_inertia,
            inv_i_b: b.inv_inertia,
            normal: manifold.normal,
            material: lane.material,
            rolling_impulse: manifold.rolling_impulse,
            point_count: manifold.point_count as f32,
            points,
        }
    }
}

/// The step's parameters for pushing (Box2D's softness splats in `b2PushContacts_Wide`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PushParams {
    pub contact_softness: Softness2d,
    pub static_softness: Softness2d,
    pub inv_h: f32,
    pub push_speed: f32,
}

impl WideContactConstraint2d {
    /// Prepares four contacts, `None` lanes empty (Box2D `b2PrepareContacts_Wide`), and
    /// returns whether any lane has restitution. `warm_start_scale` is one with warm starting
    /// and zero without.
    ///
    /// # Safety
    ///
    /// No other thread may write the states of the lanes' bodies during the call.
    pub unsafe fn prepare(
        lanes_in: [Option<WideLane<'_>>; WIDE],
        states: &Shared<'_, BodyState2d>,
        warm_start_scale: f32,
    ) -> (Self, bool) {
        let data = [
            LaneData::of(lanes_in[0].as_ref()),
            LaneData::of(lanes_in[1].as_ref()),
            LaneData::of(lanes_in[2].as_ref()),
            LaneData::of(lanes_in[3].as_ref()),
        ];
        let zero = FloatW::zero();
        let one = FloatW::splat(1.0);
        let warm_start_scale = FloatW::splat(warm_start_scale);
        let normal = Vec2W {
            x: lanes(&data, |d| d.normal.x),
            y: lanes(&data, |d| d.normal.y),
        };
        // b2RightPerpW: (y, -x).
        let tangent = Vec2W {
            x: normal.y,
            y: -normal.x,
        };
        let inv_mass_a = lanes(&data, |d| d.inv_mass_a);
        let inv_mass_b = lanes(&data, |d| d.inv_mass_b);
        let inv_i_a = lanes(&data, |d| d.inv_i_a);
        let inv_i_b = lanes(&data, |d| d.inv_i_b);
        let restitution = lanes(&data, |d| d.material.restitution);
        let two_points = lanes(&data, |d| d.point_count).greater_than(one);

        let point = |p: usize| {
            let anchor_a = Vec2W {
                x: lanes(&data, |d| d.points[p].anchor_a.x),
                y: lanes(&data, |d| d.points[p].anchor_a.y),
            };
            let anchor_b = Vec2W {
                x: lanes(&data, |d| d.points[p].anchor_b.x),
                y: lanes(&data, |d| d.points[p].anchor_b.y),
            };
            let separation = lanes(&data, |d| d.points[p].separation);
            let offset = dot(
                Vec2W {
                    x: anchor_b.x - anchor_a.x,
                    y: anchor_b.y - anchor_a.y,
                },
                normal,
            );
            let mass = |axis: Vec2W| {
                let r_a = cross(anchor_a, axis);
                let r_b = cross(anchor_b, axis);
                let k = (inv_mass_a + inv_mass_b)
                    .mul_add(inv_i_a, r_a * r_a)
                    .mul_add(inv_i_b, r_b * r_b);
                effective_mass(k)
            };
            WidePoint {
                anchor_a,
                anchor_b,
                base_separation: separation - offset,
                normal_mass: mass(normal),
                tangent_mass: mass(tangent),
                normal_impulse: warm_start_scale * lanes(&data, |d| d.points[p].normal_impulse),
                tangent_impulse: warm_start_scale * lanes(&data, |d| d.points[p].tangent_impulse),
                total_normal_impulse: zero,
                relative_velocity: zero,
                restitution_impulse: zero,
            }
        };
        let mut points = [point(0), point(1)];
        // Zero effective mass on lanes that only have one point.
        points[1].normal_mass = zero.blend(points[1].normal_mass, two_points);
        points[1].tangent_mass = zero.blend(points[1].tangent_mass, two_points);

        let mut constraint = Self {
            index_a: data.map(|d| d.indices.0),
            index_b: data.map(|d| d.indices.1),
            inv_mass_a,
            inv_mass_b,
            inv_i_a,
            inv_i_b,
            normal,
            friction: lanes(&data, |d| d.material.friction),
            tangent_speed: lanes(&data, |d| d.material.tangent_speed),
            rolling_resistance: lanes(&data, |d| d.material.rolling_resistance),
            rolling_impulse: warm_start_scale * lanes(&data, |d| d.rolling_impulse),
            restitution,
            points,
        };

        // The normal velocity before the step, for restitution and hit events. Box2D samples it
        // only for those; the velocities are the same either way.
        // SAFETY: the caller guarantees no concurrent write to the lanes' states.
        let (a, b) = unsafe {
            (
                gather(states, &constraint.index_a),
                gather(states, &constraint.index_b),
            )
        };
        for point in &mut constraint.points {
            let dv = relative_velocity(&a, &b, point.anchor_a, point.anchor_b);
            point.relative_velocity = dv.x * normal.x + dv.y * normal.y;
        }
        let any_restitution = restitution.greater_than(zero).any_true();
        (constraint, any_restitution)
    }

    /// Applies an impulse `p` at the point anchors to both bodies.
    #[inline(always)]
    fn apply(&self, a: &mut BodyStateW, b: &mut BodyStateW, r_a: Vec2W, r_b: Vec2W, p: Vec2W) {
        a.v.x = a.v.x.mul_sub(self.inv_mass_a, p.x);
        a.v.y = a.v.y.mul_sub(self.inv_mass_a, p.y);
        a.w = a.w.mul_sub(self.inv_i_a, r_a.x * p.y - r_a.y * p.x);
        b.v.x = b.v.x.mul_add(self.inv_mass_b, p.x);
        b.v.y = b.v.y.mul_add(self.inv_mass_b, p.y);
        b.w = b.w.mul_add(self.inv_i_b, r_b.x * p.y - r_b.y * p.x);
    }

    /// Re-applies the accumulated impulses (Box2D `b2WarmStartContacts_Wide`).
    ///
    /// # Safety
    ///
    /// No other thread may access the states of the lanes' bodies during the call.
    pub unsafe fn warm_start(&mut self, states: &Shared<'_, BodyState2d>) {
        // SAFETY: the caller guarantees exclusive access to the lanes' states.
        let (mut a, mut b) =
            unsafe { (gather(states, &self.index_a), gather(states, &self.index_b)) };
        let tangent_x = self.normal.y;
        let tangent_y = FloatW::zero() - self.normal.x;
        for i in 0..2 {
            let point = self.points[i];
            let (r_a, r_b) = (point.anchor_a, point.anchor_b);
            let p = Vec2W {
                x: point.normal_impulse * self.normal.x + point.tangent_impulse * tangent_x,
                y: point.normal_impulse * self.normal.y + point.tangent_impulse * tangent_y,
            };
            a.w = a.w.mul_sub(self.inv_i_a, cross(r_a, p));
            a.v.x = a.v.x.mul_sub(self.inv_mass_a, p.x);
            a.v.y = a.v.y.mul_sub(self.inv_mass_a, p.y);
            b.w = b.w.mul_add(self.inv_i_b, cross(r_b, p));
            b.v.x = b.v.x.mul_add(self.inv_mass_b, p.x);
            b.v.y = b.v.y.mul_add(self.inv_mass_b, p.y);
            self.points[i].total_normal_impulse = point.total_normal_impulse + point.normal_impulse;
        }
        a.w = a.w.mul_sub(self.inv_i_a, self.rolling_impulse);
        b.w = b.w.mul_add(self.inv_i_b, self.rolling_impulse);
        // SAFETY: as above.
        unsafe {
            scatter(states, &self.index_a, &a);
            scatter(states, &self.index_b, &b);
        }
    }

    /// Solves non-penetration with the soft bias, without friction (Box2D
    /// `b2PushContacts_Wide`).
    ///
    /// # Safety
    ///
    /// No other thread may access the states of the lanes' bodies during the call.
    pub unsafe fn push(&mut self, states: &Shared<'_, BodyState2d>, params: &PushParams) {
        // SAFETY: the caller guarantees exclusive access to the lanes' states.
        let (mut a, mut b) =
            unsafe { (gather(states, &self.index_a), gather(states, &self.index_b)) };
        let (zero, one) = (FloatW::zero(), FloatW::splat(1.0));
        let inv_h = FloatW::splat(params.inv_h);
        let contact_speed = FloatW::splat(-params.push_speed);
        let (soft, stiff) = (params.contact_softness, params.static_softness);
        // Box2D `b2SoftMaskW`: the stiffer static softness where a body is not awake.
        let static_mask = FloatW::from_array(
            [0, 1, 2, 3].map(|l| mask_lane(self.index_a[l] == 0 || self.index_b[l] == 0)),
        );
        let bias_rate = FloatW::splat(soft.mass_scale * soft.bias_rate).blend(
            FloatW::splat(stiff.mass_scale * stiff.bias_rate),
            static_mask,
        );
        let mass_scale =
            FloatW::splat(soft.mass_scale).blend(FloatW::splat(stiff.mass_scale), static_mask);
        let impulse_scale = FloatW::splat(soft.impulse_scale)
            .blend(FloatW::splat(stiff.impulse_scale), static_mask);
        let dp = Vec2W {
            x: b.dp.x - a.dp.x,
            y: b.dp.y - a.dp.y,
        };
        for i in 0..2 {
            let point = self.points[i];
            let (r_a, r_b) = (point.anchor_a, point.anchor_b);
            let s = separation(&a, &b, dp, (r_a, r_b), self.normal, point.base_separation);
            let separated = s.greater_than(zero);
            let spec_bias = s * inv_h;
            let overlap_bias = (bias_rate * s).max(contact_speed);
            let velocity_bias = overlap_bias.blend(spec_bias, separated);
            let point_mass_scale = mass_scale.blend(one, separated);
            let point_impulse_scale = impulse_scale.blend(zero, separated);
            let dv = relative_velocity(&a, &b, r_a, r_b);
            let vn = dv.x * self.normal.x + dv.y * self.normal.y;
            let neg_impulse = point.normal_mass * (point_mass_scale * vn + velocity_bias)
                + point_impulse_scale * point.normal_impulse;
            let new_impulse = (point.normal_impulse - neg_impulse).max(zero);
            let impulse = new_impulse - point.normal_impulse;
            self.points[i].normal_impulse = new_impulse;
            self.points[i].total_normal_impulse = point.total_normal_impulse + impulse;
            let p = Vec2W {
                x: impulse * self.normal.x,
                y: impulse * self.normal.y,
            };
            self.apply(&mut a, &mut b, r_a, r_b, p);
        }
        // SAFETY: as above.
        unsafe {
            scatter(states, &self.index_a, &a);
            scatter(states, &self.index_b, &b);
        }
    }

    /// Solves non-penetration without the soft bias, then rolling resistance and friction
    /// (Box2D `b2SolveContacts_Wide`, the relax stage).
    ///
    /// # Safety
    ///
    /// No other thread may access the states of the lanes' bodies during the call.
    pub unsafe fn relax(&mut self, states: &Shared<'_, BodyState2d>, inv_h: f32) {
        // SAFETY: the caller guarantees exclusive access to the lanes' states.
        let (mut a, mut b) =
            unsafe { (gather(states, &self.index_a), gather(states, &self.index_b)) };
        let zero = FloatW::zero();
        let inv_h = FloatW::splat(inv_h);
        let mut total_normal_impulse = zero;
        let dp = Vec2W {
            x: b.dp.x - a.dp.x,
            y: b.dp.y - a.dp.y,
        };
        let biases = [0, 1].map(|i| {
            let point = &self.points[i];
            let s = separation(
                &a,
                &b,
                dp,
                (point.anchor_a, point.anchor_b),
                self.normal,
                point.base_separation,
            );
            // Speculative bias, positive if separated and zero if overlapped.
            zero.max(s * inv_h)
        });
        for (i, bias) in biases.into_iter().enumerate() {
            let point = self.points[i];
            let (r_a, r_b) = (point.anchor_a, point.anchor_b);
            let dv = relative_velocity(&a, &b, r_a, r_b);
            let vn = dv.x * self.normal.x + dv.y * self.normal.y;
            let neg_impulse = point.normal_mass * (vn + bias);
            let new_impulse = (point.normal_impulse - neg_impulse).max(zero);
            let impulse = new_impulse - point.normal_impulse;
            self.points[i].normal_impulse = new_impulse;
            self.points[i].total_normal_impulse = point.total_normal_impulse + impulse;
            total_normal_impulse = total_normal_impulse + new_impulse;
            let p = Vec2W {
                x: impulse * self.normal.x,
                y: impulse * self.normal.y,
            };
            self.apply(&mut a, &mut b, r_a, r_b, p);
        }

        // Rolling resistance.
        if !self.rolling_resistance.all_zero() {
            let k = self.inv_i_a + self.inv_i_b;
            let delta = zero.blend((a.w - b.w) / k, k.greater_than(zero));
            let lambda = self.rolling_impulse;
            let max_lambda = self.rolling_resistance * total_normal_impulse;
            self.rolling_impulse = (lambda + delta).sym_clamp(max_lambda);
            let delta = self.rolling_impulse - lambda;
            a.w = a.w.mul_sub(self.inv_i_a, delta);
            b.w = b.w.mul_add(self.inv_i_b, delta);
        }

        // Friction.
        let tangent = Vec2W {
            x: self.normal.y,
            y: zero - self.normal.x,
        };
        for i in 0..2 {
            let point = self.points[i];
            let (r_a, r_b) = (point.anchor_a, point.anchor_b);
            let dv = relative_velocity(&a, &b, r_a, r_b);
            let vt = (dv.x * tangent.x + dv.y * tangent.y) - self.tangent_speed;
            let neg_impulse = point.tangent_mass * vt;
            let max_friction = self.friction * point.normal_impulse;
            let new_impulse = point.tangent_impulse - neg_impulse;
            let new_impulse = (zero - max_friction).max(new_impulse.min(max_friction));
            let impulse = new_impulse - point.tangent_impulse;
            self.points[i].tangent_impulse = new_impulse;
            let p = Vec2W {
                x: impulse * tangent.x,
                y: impulse * tangent.y,
            };
            self.apply(&mut a, &mut b, r_a, r_b, p);
        }
        // SAFETY: as above.
        unsafe {
            scatter(states, &self.index_a, &a);
            scatter(states, &self.index_b, &b);
        }
    }

    /// Adds the bounce of points that hit faster than `threshold` (Box2D
    /// `b2ApplyRestitution_Wide`); with `propagate` the lanes without restitution are solved too.
    ///
    /// # Safety
    ///
    /// No other thread may access the states of the lanes' bodies during the call.
    pub unsafe fn apply_restitution(
        &mut self,
        states: &Shared<'_, BodyState2d>,
        inv_h: f32,
        (threshold, propagate): (f32, bool),
    ) {
        if !propagate && self.restitution.all_zero() {
            return;
        }
        // SAFETY: the caller guarantees exclusive access to the lanes' states.
        let (mut a, mut b) =
            unsafe { (gather(states, &self.index_a), gather(states, &self.index_b)) };
        let zero = FloatW::zero();
        let inv_h = FloatW::splat(inv_h);
        let neg_threshold = FloatW::splat(-threshold);
        let restitution_mask = self.restitution.greater_than(zero);
        let dp = Vec2W {
            x: b.dp.x - a.dp.x,
            y: b.dp.y - a.dp.y,
        };
        for i in 0..2 {
            let point = self.points[i];
            let normal_mass = if propagate {
                point.normal_mass
            } else {
                zero.blend(point.normal_mass, restitution_mask)
            };
            let (r_a, r_b) = (point.anchor_a, point.anchor_b);
            let compression = point.total_normal_impulse - point.restitution_impulse;
            let armed = restitution_mask
                .and(point.relative_velocity.less_than(neg_threshold))
                .and(compression.greater_than(zero));
            let s = separation(&a, &b, dp, (r_a, r_b), self.normal, point.base_separation);
            let spec_bias = zero.max(s * inv_h);
            let velocity_bias = spec_bias.blend(self.restitution * point.relative_velocity, armed);
            let dv = relative_velocity(&a, &b, r_a, r_b);
            let vn = dv.x * self.normal.x + dv.y * self.normal.y;
            let neg_impulse = normal_mass * (vn + velocity_bias);
            let new_impulse = (point.normal_impulse - neg_impulse).max(zero);
            let impulse = new_impulse - point.normal_impulse;
            let approach = (-(normal_mass * vn)).max(zero).min(impulse.max(zero));
            let allowance = self.restitution * (compression + approach) - point.restitution_impulse;
            let max_impulse = approach + allowance.max(zero);
            let impulse = impulse.blend(impulse.min(max_impulse), armed);
            let solved = &mut self.points[i];
            solved.normal_impulse = point.normal_impulse + impulse;
            solved.restitution_impulse = point.restitution_impulse + (impulse - approach);
            solved.total_normal_impulse = point.total_normal_impulse + impulse;
            let p = Vec2W {
                x: impulse * self.normal.x,
                y: impulse * self.normal.y,
            };
            self.apply(&mut a, &mut b, r_a, r_b, p);
        }
        // SAFETY: as above.
        unsafe {
            scatter(states, &self.index_a, &a);
            scatter(states, &self.index_b, &b);
        }
    }

    /// Stores lane `lane`'s accumulated impulses and pre-step normal velocities in its
    /// manifold, to warm start the next step (Box2D `b2StoreImpulses_Wide`).
    pub fn store(&self, lane: usize, manifold: &mut ContactManifold2d) {
        manifold.rolling_impulse = self.rolling_impulse.to_array()[lane];
        let count = manifold.point_count;
        for (mp, point) in manifold.points[..count].iter_mut().zip(&self.points) {
            mp.normal_impulse = point.normal_impulse.to_array()[lane];
            mp.tangent_impulse = point.tangent_impulse.to_array()[lane];
            mp.total_normal_impulse = point.total_normal_impulse.to_array()[lane];
            mp.normal_velocity = point.relative_velocity.to_array()[lane];
        }
    }
}

/// A mask lane: every bit set or clear.
const fn mask_lane(set: bool) -> f32 {
    f32::from_bits(if set { u32::MAX } else { 0 })
}

/// An empty wide constraint, to size buffers with.
impl Default for WideContactConstraint2d {
    fn default() -> Self {
        let zero = FloatW::zero();
        let v = Vec2W { x: zero, y: zero };
        let point = WidePoint {
            anchor_a: v,
            anchor_b: v,
            base_separation: zero,
            normal_mass: zero,
            tangent_mass: zero,
            normal_impulse: zero,
            tangent_impulse: zero,
            total_normal_impulse: zero,
            relative_velocity: zero,
            restitution_impulse: zero,
        };
        Self {
            index_a: [0; WIDE],
            index_b: [0; WIDE],
            inv_mass_a: zero,
            inv_mass_b: zero,
            inv_i_a: zero,
            inv_i_b: zero,
            normal: v,
            friction: zero,
            tangent_speed: zero,
            rolling_resistance: zero,
            rolling_impulse: zero,
            restitution: zero,
            points: [point; 2],
        }
    }
}
