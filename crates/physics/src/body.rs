//! Rigid body dynamic state, mass, velocities, and damping parameters.

use crate::math;
use korppi_ecs::Component;
use korppi_math::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Type of rigid body defining mobility and mass characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BodyType {
    /// Zero velocity and infinite mass; unaffected by forces and collisions.
    #[default]
    Static,
    /// Velocity driven directly by user/gameplay; unaffected by forces.
    Kinematic,
    /// Full Newtonian physics integration driven by forces, torques, and contacts.
    Dynamic,
}

impl BodyType {
    /// Index of the broadphase dynamic tree for this body type (0 = Static, 1 = Kinematic, 2 = Dynamic).
    #[must_use]
    pub const fn tree_index(self) -> usize {
        match self {
            Self::Static => 0,
            Self::Kinematic => 1,
            Self::Dynamic => 2,
        }
    }
}

/// Largest turn of a body in one step, in radians (Box3D `B3_MAX_ROTATION`), unless the body
/// allows fast rotation.
pub const MAX_ROTATION: f32 = 0.25 * std::f32::consts::PI;

/// Axes along and about which a body may not move (Box3D `b3MotionLocks`). The axes are the
/// world's. Locking all three rotations gives the body no rotational inertia.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotionLocks3d {
    /// No translation along the world x-axis.
    pub linear_x: bool,
    /// No translation along the world y-axis.
    pub linear_y: bool,
    /// No translation along the world z-axis.
    pub linear_z: bool,
    /// No rotation about the world x-axis.
    pub angular_x: bool,
    /// No rotation about the world y-axis.
    pub angular_y: bool,
    /// No rotation about the world z-axis.
    pub angular_z: bool,
}

impl MotionLocks3d {
    /// Whether every rotation is locked (Box3D `b3_fixedRotation`).
    #[must_use]
    pub const fn fixed_rotation(&self) -> bool {
        self.angular_x && self.angular_y && self.angular_z
    }

    /// Zeroes the locked components of a linear and an angular velocity.
    #[must_use]
    pub fn apply(&self, v: Vec3, w: Vec3) -> (Vec3, Vec3) {
        let pick = |locked: bool, value: f32| if locked { 0.0 } else { value };
        (
            Vec3::new(
                pick(self.linear_x, v.x),
                pick(self.linear_y, v.y),
                pick(self.linear_z, v.z),
            ),
            Vec3::new(
                pick(self.angular_x, w.x),
                pick(self.angular_y, w.y),
                pick(self.angular_z, w.z),
            ),
        )
    }
}

/// Dynamic rigid body state. Can be used as a standalone physics object or as an ECS component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidBody {
    /// Mobility type.
    pub body_type: BodyType,
    /// The body origin in world space, where its colliders and joint frames are placed (Box3D's
    /// body position). The centre of mass is [`Self::center_of_mass`].
    pub position: Vec3,
    /// Orientation quaternion.
    pub rotation: Quat,
    /// Velocity of the centre of mass (m/s).
    pub linear_velocity: Vec3,
    /// Angular velocity vector (rad/s).
    pub angular_velocity: Vec3,
    /// Linear damping coefficient, 1/s (Box3D `linearDamping`, default 0). Each substep scales
    /// the velocity by `1 / (1 + h c)`.
    pub linear_damping: f32,
    /// Angular damping coefficient, 1/s (Box3D `angularDamping`, default 0).
    pub angular_damping: f32,
    /// Scale of the world's gravity on this body (Box3D `gravityScale`, default 1).
    #[serde(default = "one")]
    pub gravity_scale: f32,
    /// Total mass in kilograms.
    pub mass: f32,
    /// Inverse mass (1/mass), or 0 for static/kinematic bodies.
    pub inv_mass: f32,
    /// Rotational inertia about the centre of mass in the body's frame, a symmetric tensor in
    /// kg·m².
    pub local_inertia: Mat3,
    /// The centre of mass in the body's frame (Box3D `localCenter`); the colliders' when the
    /// body has any, else the origin.
    #[serde(default)]
    pub local_center: Vec3,
    /// The radius of the largest sphere about the centre of mass inside every collider, for
    /// continuous collision (Box3D `minExtent`); unbounded without colliders.
    #[serde(default = "huge")]
    pub min_extent: f32,
    /// How far the colliders reach from the centre of mass along each body axis (Box3D
    /// `maxExtent`).
    #[serde(default)]
    pub max_extent: Vec3,
    /// Inverse of `local_inertia`, or zero for static and kinematic bodies, singular inertia
    /// and fixed rotation.
    pub inv_local_inertia: Mat3,
    /// Accumulated external force for the next step, applied every substep and then cleared.
    pub force: Vec3,
    /// Accumulated external torque for the next step, applied every substep and then cleared.
    pub torque: Vec3,
    /// Axes the body may not move along or about.
    #[serde(default)]
    pub motion_locks: MotionLocks3d,
    /// Whether the body may turn more than [`MAX_ROTATION`] per step (Box3D
    /// `allowFastRotation`), for wheels and other round bodies.
    #[serde(default)]
    pub allow_fast_rotation: bool,
    /// Indicates whether the body is currently sleeping in an inactive island.
    pub is_sleeping: bool,
    /// Accumulated duration in seconds that the body has remained below velocity sleep thresholds.
    pub sleep_time: f32,
    /// Whether this body is allowed to fall asleep.
    pub enable_sleep: bool,
    /// Sleep velocity threshold in m/s below which the body rests (Box3D `sleepThreshold`,
    /// default 0.05).
    #[serde(default = "default_sleep_threshold")]
    pub sleep_threshold: f32,
    /// Whether continuous collision also sweeps this body against kinematic and dynamic bodies
    /// that are not bullets, not only static ones (Box3D `isBullet`, off by default).
    #[serde(default)]
    pub is_bullet: bool,
    /// The body is swept for continuous collision when it moves more than this times its
    /// smallest extent in a step (Box3D `safetyFactor`, default 0.5).
    #[serde(default = "default_safety_factor")]
    pub safety_factor: f32,
    /// Whether this body's contacts may be recycled when it barely moves relative to the other
    /// body (Box3D `enableContactRecycling`, on by default); recycling can make ghost
    /// collisions, which characters may want to avoid.
    #[serde(default = "default_true")]
    pub enable_contact_recycling: bool,
    /// Whether the body is in the simulation (`false` puts it in the disabled set; Box3D
    /// `b3BodyDef::isEnabled`).
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// Box3D's default continuous collision safety factor.
pub const CONTINUOUS_SAFETY_FACTOR: f32 = 0.5;

/// Box3D's default sleep velocity threshold, in m/s (`0.05`).
pub const DEFAULT_SLEEP_THRESHOLD: f32 = 0.05;

fn one() -> f32 {
    1.0
}

fn huge() -> f32 {
    f32::MAX
}

fn default_safety_factor() -> f32 {
    CONTINUOUS_SAFETY_FACTOR
}

fn default_sleep_threshold() -> f32 {
    DEFAULT_SLEEP_THRESHOLD
}

fn default_true() -> bool {
    true
}

impl Component for RigidBody {}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: BodyType::Static,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            mass: 1.0,
            inv_mass: 0.0,
            local_inertia: Mat3::IDENTITY,
            inv_local_inertia: Mat3::ZERO,
            local_center: Vec3::ZERO,
            min_extent: f32::MAX,
            max_extent: Vec3::ZERO,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            motion_locks: MotionLocks3d::default(),
            allow_fast_rotation: false,
            is_sleeping: false,
            sleep_time: 0.0,
            enable_sleep: true,
            sleep_threshold: DEFAULT_SLEEP_THRESHOLD,
            is_bullet: false,
            safety_factor: CONTINUOUS_SAFETY_FACTOR,
            enable_contact_recycling: true,
            enabled: true,
        }
    }
}

impl RigidBody {
    /// Creates a new rigid body with the specified body type and position.
    #[must_use]
    pub fn new(body_type: BodyType, position: Vec3) -> Self {
        let mut body = Self {
            body_type,
            position,
            ..Default::default()
        };
        body.recompute_mass_properties();
        body
    }

    /// Sets whether the body starts enabled in the simulation (Box3D `b3BodyDef::isEnabled`).
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Sets the position translation.
    #[must_use]
    pub fn with_position(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    /// Sets the orientation quaternion.
    #[must_use]
    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Sets the initial linear velocity in m/s.
    #[must_use]
    pub fn with_linear_velocity(mut self, linear_velocity: Vec3) -> Self {
        self.linear_velocity = linear_velocity;
        self
    }

    /// Sets the initial angular velocity in rad/s.
    #[must_use]
    pub fn with_angular_velocity(mut self, angular_velocity: Vec3) -> Self {
        self.angular_velocity = angular_velocity;
        self
    }

    /// Sets whether sleep is enabled.
    #[must_use]
    pub fn with_enable_sleep(mut self, enable_sleep: bool) -> Self {
        self.enable_sleep = enable_sleep;
        self
    }

    /// Sets the sleep velocity threshold in m/s (Box3D `sleepThreshold`).
    #[must_use]
    pub fn with_sleep_threshold(mut self, sleep_threshold: f32) -> Self {
        self.sleep_threshold = if sleep_threshold.is_finite() {
            sleep_threshold.max(0.0)
        } else {
            DEFAULT_SLEEP_THRESHOLD
        };
        self
    }

    /// Makes the body a bullet (see [`Self::is_bullet`]).
    #[must_use]
    pub fn with_bullet(mut self, bullet: bool) -> Self {
        self.is_bullet = bullet;
        self
    }

    /// Sets whether the body's contacts may be recycled (see
    /// [`Self::enable_contact_recycling`]).
    #[must_use]
    pub fn with_contact_recycling(mut self, enable: bool) -> Self {
        self.enable_contact_recycling = enable;
        self
    }

    /// Sets the continuous collision safety factor (see [`Self::safety_factor`]).
    #[must_use]
    pub fn with_safety_factor(mut self, safety_factor: f32) -> Self {
        self.safety_factor = safety_factor;
        self
    }

    /// Sets the rotational inertia about the centre of mass, in the body's frame, and updates
    /// its inverse.
    #[must_use]
    pub fn with_inertia(mut self, inertia: Mat3) -> Self {
        self.local_inertia = inertia;
        self.recompute_mass_properties();
        self
    }

    /// Whether the solver moves this body: an awake dynamic body, or a kinematic body, which
    /// moves with its velocity and never sleeps.
    #[must_use]
    pub fn is_moving(&self) -> bool {
        match self.body_type {
            BodyType::Static => false,
            BodyType::Kinematic => true,
            BodyType::Dynamic => !self.is_sleeping,
        }
    }

    /// Wakes up the body from sleeping.
    pub fn wake_up(&mut self) {
        self.is_sleeping = false;
        self.sleep_time = 0.0;
    }

    /// Transitions this dynamic body into sleep state, zeroing velocities.
    pub fn put_to_sleep(&mut self) {
        if self.body_type == BodyType::Dynamic && self.enable_sleep {
            self.is_sleeping = true;
            self.linear_velocity = Vec3::ZERO;
            self.angular_velocity = Vec3::ZERO;
            self.force = Vec3::ZERO;
            self.torque = Vec3::ZERO;
        }
    }

    /// Recomputes the inverse mass and inverse inertia from `body_type`, `mass`,
    /// `local_inertia` and the motion locks (Box3D `b3Body_SetMassData`): a singular inertia or
    /// fixed rotation gives no inverse inertia.
    pub fn recompute_mass_properties(&mut self) {
        if self.body_type == BodyType::Dynamic && self.mass > 0.0 {
            self.inv_mass = 1.0 / self.mass;
            self.inv_local_inertia =
                if math::det(&self.local_inertia) > 0.0 && !self.motion_locks.fixed_rotation() {
                    math::invert_symmetric(&self.local_inertia)
                } else {
                    Mat3::ZERO
                };
        } else {
            self.inv_mass = 0.0;
            self.inv_local_inertia = Mat3::ZERO;
        }
    }

    /// The centre of mass in world space (Box3D `b3Body_GetWorldCenter`).
    #[must_use]
    pub fn center_of_mass(&self) -> Vec3 {
        self.position + self.rotation * self.local_center
    }

    /// Computes the inverse moment of inertia tensor in world coordinates: $R \cdot I^{-1} \cdot R^T$.
    #[must_use]
    pub fn world_inv_inertia(&self) -> Mat3 {
        if self.body_type != BodyType::Dynamic {
            return Mat3::ZERO;
        }
        let rot_mat = Mat3::from_quat(self.rotation);
        rot_mat * self.inv_local_inertia * rot_mat.transpose()
    }

    /// Integrates the velocities over one substep `h` (Box3D `b3IntegrateVelocitiesTask`):
    /// gravity times the gravity scale (none without mass), the force and torque, Padé damping,
    /// and the gyroscopic torque by one implicit Newton step on the inertia tensor.
    /// `inv_inertia_world` is the body's world inverse inertia at the start of the step.
    pub(crate) fn integrate_velocities(
        &mut self,
        gravity: Vec3,
        inv_inertia_world: &Mat3,
        delta_rotation: Quat,
        start_rotation: Quat,
        h: f32,
    ) {
        let linear_damping = 1.0 / (1.0 + h * self.linear_damping);
        let angular_damping = 1.0 / (1.0 + h * self.angular_damping);
        let gravity_scale = if self.inv_mass > 0.0 {
            self.gravity_scale
        } else {
            0.0
        };
        let linear_delta = self.force * (h * self.inv_mass) + gravity * (h * gravity_scale);
        self.linear_velocity = linear_delta + self.linear_velocity * linear_damping;
        let angular_delta = (*inv_inertia_world * self.torque) * h;
        let w = angular_delta + self.angular_velocity * angular_damping;
        self.angular_velocity = self.gyroscopic_step(w, h, delta_rotation, start_rotation);
    }

    /// Solves `I (w2 - w1) + h w2 x I w2 = 0` for `w2` by one Newton step in the body's frame,
    /// as Box3D does (`B3_GYROSCOPIC_ITERATIONS`). A body without rotational inertia or with
    /// isotropic inertia keeps `w`.
    fn gyroscopic_step(&self, w: Vec3, h: f32, delta_rotation: Quat, start_rotation: Quat) -> Vec3 {
        if w == Vec3::ZERO || self.inv_local_inertia == Mat3::ZERO {
            return w;
        }
        let inv = &self.inv_local_inertia;
        if inv.y_axis.x == 0.0
            && inv.z_axis.x == 0.0
            && inv.z_axis.y == 0.0
            && inv.x_axis.x == inv.y_axis.y
            && inv.y_axis.y == inv.z_axis.z
        {
            return w;
        }
        let inertia = math::invert(inv);
        let q = (delta_rotation * start_rotation).normalize();
        let omega1 = q.conjugate() * w;
        let mut omega2 = omega1;

        let (i00, i01, i02) = (inertia.x_axis.x, inertia.y_axis.x, inertia.z_axis.x);
        let (i11, i12, i22) = (inertia.y_axis.y, inertia.z_axis.y, inertia.z_axis.z);
        for _ in 0..GYROSCOPIC_ITERATIONS {
            let (w1, w2, w3) = (omega2.x, omega2.y, omega2.z);
            let iw1 = i00 * w1 + i01 * w2 + i02 * w3;
            let iw2 = i01 * w1 + i11 * w2 + i12 * w3;
            let iw3 = i02 * w1 + i12 * w2 + i22 * w3;

            let dw = omega2 - omega1;
            let b = Vec3::new(
                i00 * dw.x + i01 * dw.y + i02 * dw.z + h * (w2 * iw3 - w3 * iw2),
                i01 * dw.x + i11 * dw.y + i12 * dw.z + h * (w3 * iw1 - w1 * iw3),
                i02 * dw.x + i12 * dw.y + i22 * dw.z + h * (w1 * iw2 - w2 * iw1),
            );
            let jacobian = Mat3::from_cols(
                Vec3::new(
                    i00 + h * (w2 * i02 - w3 * i01),
                    i01 + h * (w3 * i00 - w1 * i02 - iw3),
                    i02 + h * (w1 * i01 - w2 * i00 + iw2),
                ),
                Vec3::new(
                    i01 + h * (w2 * i12 - w3 * i11 + iw3),
                    i11 + h * (w3 * i01 - w1 * i12),
                    i12 + h * (w1 * i11 - w2 * i01 - iw1),
                ),
                Vec3::new(
                    i02 + h * (w2 * i22 - w3 * i12 - iw2),
                    i12 + h * (w3 * i02 - w1 * i22 + iw1),
                    i22 + h * (w1 * i12 - w2 * i02),
                ),
            );
            omega2 -= math::solve3(&jacobian, b);
        }
        q * omega2
    }

    /// Applies an external force at the center of mass for the next step.
    pub fn apply_force(&mut self, force: Vec3) {
        if self.body_type == BodyType::Dynamic {
            self.wake_up();
            self.force += force;
        }
    }

    /// Applies an external force at a world point for the next step: the force, and its torque
    /// about the centre of mass (Box3D `b3Body_ApplyForce`).
    pub fn apply_force_at_point(&mut self, force: Vec3, point: Vec3) {
        if self.body_type == BodyType::Dynamic {
            self.wake_up();
            self.force += force;
            self.torque += (point - self.center_of_mass()).cross(force);
        }
    }

    /// Applies an external torque for the next step.
    pub fn apply_torque(&mut self, torque: Vec3) {
        if self.body_type == BodyType::Dynamic {
            self.wake_up();
            self.torque += torque;
        }
    }

    /// Applies an instantaneous linear impulse at the center of mass.
    pub fn apply_linear_impulse(&mut self, impulse: Vec3) {
        if self.body_type == BodyType::Dynamic {
            self.wake_up();
            self.linear_velocity += impulse * self.inv_mass;
        }
    }

    /// Applies an instantaneous impulse at a world-space point.
    pub fn apply_impulse_at_point(&mut self, impulse: Vec3, point: Vec3) {
        if self.body_type == BodyType::Dynamic {
            self.wake_up();
            self.linear_velocity += impulse * self.inv_mass;
            let r = point - self.center_of_mass();
            let torque_impulse = r.cross(impulse);
            self.angular_velocity += self.world_inv_inertia() * torque_impulse;
        }
    }

    /// Applies an instantaneous angular impulse, in N·m·s (Box3D `b3Body_ApplyAngularImpulse`).
    pub fn apply_angular_impulse(&mut self, impulse: Vec3) {
        if self.body_type == BodyType::Dynamic {
            self.wake_up();
            let local = self.rotation.conjugate() * impulse;
            self.angular_velocity += self.rotation * (self.inv_local_inertia * local);
        }
    }
}

/// Newton steps of the gyroscopic solve (Box3D `B3_GYROSCOPIC_ITERATIONS`).
const GYROSCOPIC_ITERATIONS: usize = 1;
