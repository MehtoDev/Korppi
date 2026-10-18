//! 2D rigid bodies: state, mass data, forces, and Box2D v3's velocity and position integration.

use crate::shape::MassData2d;
use korppi_ecs::Component;
use korppi_math::{Rot2, Transform2d, Vec2};
use serde::{Deserialize, Serialize};

/// The motion type of a rigid body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyType {
    /// Immovable body with infinite mass and zero velocity.
    Static,
    /// Kinematic body moved explicitly by velocity, unaffected by forces/contacts.
    Kinematic,
    /// Full dynamic rigid body simulated by forces, impulses, and constraints.
    Dynamic,
}

/// Largest linear speed of any body, in m/s (Box2D `maximumLinearSpeed`).
pub const MAX_LINEAR_SPEED: f32 = 400.0;

/// The default continuous collision safety factor of a body (Box2D's default
/// `b2BodyDef::safetyFactor`).
pub const CONTINUOUS_SAFETY_FACTOR: f32 = 0.5;

/// Largest turn of a body in one step, in radians (Box2D `B2_MAX_ROTATION`), unless the body
/// allows fast rotation.
pub const MAX_ROTATION: f32 = 0.25 * std::f32::consts::PI;

/// Axes along and about which a body may not move (Box2D `b2MotionLocks`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotionLocks2d {
    /// No translation along the world x-axis.
    pub linear_x: bool,
    /// No translation along the world y-axis.
    pub linear_y: bool,
    /// No rotation; the body also has no inverse inertia, so constraints cannot turn it.
    pub angular_z: bool,
}

/// A 2D rigid body representing a simulated physical entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidBody2d {
    /// Motion classification of the body.
    pub body_type: BodyType,
    /// Position of the body origin in world coordinates.
    pub position: Vec2,
    /// Orientation complex rotation in world space.
    pub rotation: Rot2,
    /// Linear velocity of the centre of mass in meters per second.
    pub linear_velocity: Vec2,
    /// Angular velocity in radians per second.
    pub angular_velocity: f32,
    /// Total mass in kilograms. Set from the collider's density when one is attached, or with
    /// [`Self::set_mass`]; a dynamic body without either is massless, as in Box2D v3.
    pub mass: f32,
    /// Inverse mass (0.0 for static/kinematic bodies).
    pub inv_mass: f32,
    /// Rotational inertia about local center of mass.
    pub rotational_inertia: f32,
    /// Inverse rotational inertia (0.0 for static/kinematic bodies).
    pub inv_inertia: f32,
    /// Centre of mass in body-local coordinates. The body rotates about it, and solver lever
    /// arms are measured from it.
    pub local_center: Vec2,
    /// Linear damping factor.
    pub linear_damping: f32,
    /// Angular damping factor.
    pub angular_damping: f32,
    /// Multiplier for global gravity.
    pub gravity_scale: f32,
    /// Whether the body sleeps. The world keeps it in step with its solver sets: read it, and
    /// change it with `PhysicsWorld2d::set_awake`. A body created with it set starts asleep.
    pub is_sleeping: bool,
    /// How long the body has rested, in seconds (Box2D `sleepTime`).
    pub sleep_time: f32,
    /// Whether this body is allowed to fall asleep.
    pub enable_sleep: bool,
    /// Whether this body uses Continuous Collision Detection (CCD) to prevent tunneling.
    pub is_bullet: bool,
    /// Whether this body's contacts may be recycled when the bodies barely moved (Box2D
    /// `enableContactRecycling`, default `true`; both bodies must allow it).
    pub enable_contact_recycling: bool,
    /// Axes the body may not move along or about. Change them with [`Self::set_motion_locks`].
    pub motion_locks: MotionLocks2d,
    /// Whether the body may turn more than [`MAX_ROTATION`] per step.
    pub allow_fast_rotation: bool,
    /// The body is fast, and gets continuous collision, when it moves more than this many times
    /// its smallest extent in a step (Box2D `b2BodyDef::safetyFactor`, default 0.5, recommended
    /// 0.01 to 0.5). Smaller sweeps more bodies; larger risks tunneling.
    pub safety_factor: f32,
    /// Force accumulated for the coming step, in newtons; cleared after each step.
    pub force: Vec2,
    /// Torque accumulated for the coming step, in N·m; cleared after each step.
    pub torque: f32,
    /// Whether the body takes part in the simulation (Box2D `b2BodyDef::isEnabled`). Set it with
    /// [`Self::with_enabled`] before creating the body, and afterwards through
    /// `PhysicsWorld2d::disable_body` and `enable_body`, which keep the broadphase in step.
    #[serde(default = "enabled_by_default")]
    pub(crate) enabled: bool,
}

const fn enabled_by_default() -> bool {
    true
}

impl Component for RigidBody2d {}

impl RigidBody2d {
    /// Creates a new rigid body. It has no mass until a collider gives it one (from the
    /// collider's density) or [`Self::set_mass`] sets it: until then a dynamic body ignores
    /// gravity, joints and contacts, as in Box2D v3.
    #[must_use]
    pub fn new(body_type: BodyType) -> Self {
        Self {
            body_type,
            position: Vec2::ZERO,
            rotation: Rot2::IDENTITY,
            linear_velocity: Vec2::ZERO,
            angular_velocity: 0.0,
            mass: 0.0,
            inv_mass: 0.0,
            rotational_inertia: 0.0,
            inv_inertia: 0.0,
            local_center: Vec2::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            is_sleeping: false,
            sleep_time: 0.0,
            enable_sleep: true,
            is_bullet: false,
            enable_contact_recycling: true,
            motion_locks: MotionLocks2d {
                linear_x: false,
                linear_y: false,
                angular_z: false,
            },
            allow_fast_rotation: false,
            safety_factor: CONTINUOUS_SAFETY_FACTOR,
            force: Vec2::ZERO,
            torque: 0.0,
            enabled: true,
        }
    }

    /// Creates the body enabled or disabled, in builder style. A disabled body has no
    /// broadphase proxy, contacts or active joints, and does not move.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Whether the body takes part in the simulation.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Sets the motion locks in builder style.
    #[must_use]
    pub fn with_motion_locks(mut self, locks: MotionLocks2d) -> Self {
        self.set_motion_locks(locks);
        self
    }

    /// Locks the body's motion along or about axes (Box2D `b2Body_SetMotionLocks`): the locked
    /// velocity components are zeroed, and a rotation lock removes the inverse inertia.
    pub fn set_motion_locks(&mut self, locks: MotionLocks2d) {
        self.motion_locks = locks;
        if locks.linear_x {
            self.linear_velocity.x = 0.0;
        }
        if locks.linear_y {
            self.linear_velocity.y = 0.0;
        }
        if locks.angular_z {
            self.angular_velocity = 0.0;
        }
        self.inv_inertia = self.inverse_inertia();
    }

    /// Lets the body turn faster than [`MAX_ROTATION`] per step, in builder style.
    #[must_use]
    pub const fn with_fast_rotation(mut self, allow: bool) -> Self {
        self.allow_fast_rotation = allow;
        self
    }

    /// Sets whether this body treats collisions as a fast-moving bullet with continuous collision detection.
    #[must_use]
    pub const fn with_bullet(mut self, bullet: bool) -> Self {
        self.is_bullet = bullet;
        self
    }

    /// Sets the continuous collision safety factor (see [`Self::safety_factor`]).
    #[must_use]
    pub const fn with_safety_factor(mut self, safety_factor: f32) -> Self {
        self.safety_factor = safety_factor;
        self
    }

    /// Sets whether this body's contacts may be recycled (Box2D `enableContactRecycling`).
    #[must_use]
    pub const fn with_contact_recycling(mut self, recycle: bool) -> Self {
        self.enable_contact_recycling = recycle;
        self
    }

    /// Sets the bullet status of this body.
    pub fn set_bullet(&mut self, bullet: bool) {
        self.is_bullet = bullet;
    }

    /// Returns `true` if this body is treated as a continuous collision bullet.
    #[must_use]
    pub const fn is_bullet(&self) -> bool {
        self.is_bullet
    }

    /// Sets the initial world-space position.
    #[must_use]
    pub const fn with_position(mut self, position: Vec2) -> Self {
        self.position = position;
        self
    }

    /// Sets the initial world-space rotation.
    #[must_use]
    pub const fn with_rotation(mut self, rotation: Rot2) -> Self {
        self.rotation = rotation;
        self
    }

    /// Sets the initial linear velocity.
    #[must_use]
    pub const fn with_linear_velocity(mut self, velocity: Vec2) -> Self {
        self.linear_velocity = velocity;
        self
    }

    /// Sets the initial angular velocity.
    #[must_use]
    pub const fn with_angular_velocity(mut self, angular_velocity: f32) -> Self {
        self.angular_velocity = angular_velocity;
        self
    }

    /// Sets the mass and rotational inertia.
    #[must_use]
    pub fn with_mass(mut self, mass: f32, rotational_inertia: f32) -> Self {
        self.set_mass(mass, rotational_inertia);
        self
    }

    /// Sets the mass and the rotational inertia about the centre of mass of a dynamic body,
    /// keeping its centre of mass (Box2D `b2Body_SetMassData`). Attaching a collider afterwards
    /// replaces them with the collider's. A zero inertia keeps the body from rotating.
    pub fn set_mass(&mut self, mass: f32, rotational_inertia: f32) {
        self.set_mass_data(MassData2d {
            mass,
            center: self.local_center,
            rotational_inertia,
        });
    }

    /// Returns the mass, local centre of mass and rotational inertia about it.
    #[must_use]
    pub const fn mass_data(&self) -> MassData2d {
        MassData2d {
            mass: self.mass,
            center: self.local_center,
            rotational_inertia: self.rotational_inertia,
        }
    }

    /// Sets the mass, local centre of mass and rotational inertia about it (Box2D
    /// `b2Body_SetMassData`). Static and kinematic bodies stay massless. Moving the centre of
    /// mass keeps the velocity of the body's material points, as Box2D does.
    pub fn set_mass_data(&mut self, mass_data: MassData2d) {
        let old_center = self.world_center();
        self.apply_mass_data(mass_data);
        let offset = self.world_center() - old_center;
        self.linear_velocity += Vec2::new(
            -self.angular_velocity * offset.y,
            self.angular_velocity * offset.x,
        );
    }

    /// Sets the mass data as [`Self::set_mass_data`] without moving the velocity, for the
    /// world, which moves it from the centre of mass it keeps.
    pub(crate) fn apply_mass_data(&mut self, mass_data: MassData2d) {
        if self.body_type == BodyType::Dynamic {
            self.mass = mass_data.mass.max(0.0);
            self.rotational_inertia = mass_data.rotational_inertia.max(0.0);
            self.local_center = mass_data.center;
        } else {
            self.mass = 0.0;
            self.rotational_inertia = 0.0;
            self.local_center = Vec2::ZERO;
        }
        self.inv_mass = if self.mass > 0.0 {
            1.0 / self.mass
        } else {
            0.0
        };
        self.inv_inertia = self.inverse_inertia();
    }

    /// The inverse inertia: zero without inertia or with rotation locked.
    fn inverse_inertia(&self) -> f32 {
        if self.rotational_inertia > 0.0 && !self.motion_locks.angular_z {
            1.0 / self.rotational_inertia
        } else {
            0.0
        }
    }

    /// Returns the centre of mass in world coordinates.
    #[must_use]
    pub fn world_center(&self) -> Vec2 {
        self.position + self.rotation.rotate(self.local_center)
    }

    /// Computes the 2D affine transform of the rigid body.
    #[must_use]
    pub fn transform(&self) -> Transform2d {
        Transform2d::from_translation_rotation(self.position, self.rotation)
    }

    /// Integrates the velocities of a dynamic body over `dt` (Box2D `b2IntegrateVelocities`):
    /// gravity (only with mass) and the accumulated force and torque, then Padé damping
    /// `v / (1 + c dt)`.
    pub fn integrate_velocities(&mut self, gravity: Vec2, dt: f32) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        (self.linear_velocity, self.angular_velocity) =
            self.integrated_velocities(self.linear_velocity, self.angular_velocity, gravity, dt);
    }

    /// Velocities `v` and `w` of this dynamic body after [`Self::integrate_velocities`] over
    /// `dt`, for the solver's body states.
    pub(crate) fn integrated_velocities(
        &self,
        v: Vec2,
        w: f32,
        gravity: Vec2,
        dt: f32,
    ) -> (Vec2, f32) {
        let gravity_scale = if self.inv_mass > 0.0 {
            self.gravity_scale
        } else {
            0.0
        };
        let linear_damping = 1.0 / (1.0 + dt * self.linear_damping);
        let angular_damping = 1.0 / (1.0 + dt * self.angular_damping);
        let delta_v = self.force * (dt * self.inv_mass) + gravity * (dt * gravity_scale);
        let delta_w = dt * self.inv_inertia * self.torque;
        (delta_v + v * linear_damping, delta_w + w * angular_damping)
    }

    /// Applies the motion locks and the speed caps to the velocities, as Box2D does before it
    /// integrates positions: at most `max_linear_speed`, and at most `max_angular_speed` unless
    /// the body allows fast rotation.
    pub fn limit_velocities(&mut self, max_linear_speed: f32, max_angular_speed: f32) {
        (self.linear_velocity, self.angular_velocity) = self.limited_velocities(
            self.linear_velocity,
            self.angular_velocity,
            max_linear_speed,
            max_angular_speed,
        );
    }

    /// Velocities `v` and `w` after this body's motion locks and the speed caps (see
    /// [`Self::limit_velocities`]), for the solver's body states.
    pub(crate) fn limited_velocities(
        &self,
        mut v: Vec2,
        mut w: f32,
        max_linear_speed: f32,
        max_angular_speed: f32,
    ) -> (Vec2, f32) {
        if self.motion_locks.linear_x {
            v.x = 0.0;
        }
        if self.motion_locks.linear_y {
            v.y = 0.0;
        }
        if self.motion_locks.angular_z {
            w = 0.0;
        }
        let speed_squared = v.length_squared();
        if speed_squared > max_linear_speed * max_linear_speed {
            v *= max_linear_speed / speed_squared.sqrt();
        }
        if !self.allow_fast_rotation && w * w > max_angular_speed * max_angular_speed {
            w *= max_angular_speed / w.abs();
        }
        (v, w)
    }

    /// Integrates position and rotation using current velocities. The body rotates about its
    /// centre of mass.
    pub fn integrate_positions(&mut self, dt: f32) {
        if self.body_type == BodyType::Static || self.is_sleeping {
            return;
        }

        let center = self.world_center() + self.linear_velocity * dt;
        self.rotation = self.rotation.integrate(self.angular_velocity * dt);
        self.position = center - self.rotation.rotate(self.local_center);
    }

    /// Adds a force at the centre of mass for the coming step
    /// (Box2D `b2Body_ApplyForceToCenter`).
    pub fn apply_force_to_center(&mut self, force: Vec2) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.force += force;
    }

    /// Adds a force at a world point for the coming step: off the centre
    /// of mass it also adds a torque (Box2D `b2Body_ApplyForce`).
    pub fn apply_force(&mut self, force: Vec2, point: Vec2) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.force += force;
        let r = point - self.world_center();
        self.torque += r.x * force.y - r.y * force.x;
    }

    /// Adds a torque for the coming step (Box2D `b2Body_ApplyTorque`).
    pub fn apply_torque(&mut self, torque: f32) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.torque += torque;
    }

    /// Clears the accumulated force and torque, as the world does after each step.
    pub fn clear_forces(&mut self) {
        self.force = Vec2::ZERO;
        self.torque = 0.0;
    }

    /// Applies a linear impulse at the centre of mass.
    pub fn apply_linear_impulse(&mut self, impulse: Vec2) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
    }

    /// Applies an angular impulse.
    pub fn apply_angular_impulse(&mut self, impulse: f32) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.angular_velocity += impulse * self.inv_inertia;
    }

    /// Applies an impulse at a world-space point.
    pub fn apply_impulse_at_point(&mut self, impulse: Vec2, point: Vec2) {
        if self.body_type != BodyType::Dynamic {
            return;
        }
        self.linear_velocity += impulse * self.inv_mass;
        let r = point - self.world_center();
        // 2D cross product: r.x * impulse.y - r.y * impulse.x
        let torque_impulse = r.x * impulse.y - r.y * impulse.x;
        self.angular_velocity += torque_impulse * self.inv_inertia;
    }
}
