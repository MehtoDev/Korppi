//! Central EngineApp coordinating ECS, simulation clock, dual physics, audio, input, rendering, and UI.

use crate::backend::GraphicsBackend;
use crate::config::EngineConfig;
use crate::metrics::DiagnosticMetrics;
use crate::presenter::{QuadBatchSource, WindowPresenter};
use korppi_asset::AssetServer;
use korppi_audio::output::{AudioBackend, AudioOutput};
use korppi_audio::player::update_audio_players;
use korppi_audio::raytrace::{AcousticGeometry, AcousticMaterial};
use korppi_audio::{AudioCoordinator, AudioOcclusion, AudioRoom, AudioSettings};
use korppi_core::arena::FrameArena;
use korppi_core::clock::FixedClock;
use korppi_ecs::{Entity, Schedule, Stage, World};
use korppi_input::InputManager;
use korppi_math::{Quat, Transform, Vec3};
use korppi_physics::joint::{
    DistanceJoint, Joint as Joint3d, PrismaticJoint, RevoluteJoint, SphericalJoint, WeldJoint,
};
use korppi_physics::{BodyId3d, BodyType, Collider, JointId3d, PhysicsWorld, RigidBody};
use korppi_physics2d::joint::Joint2d;
use korppi_physics2d::{collider::Collider2d, BodyId2d, JointId2d, PhysicsWorld2d, RigidBody2d};
use korppi_render::graph::{RenderGraph, TextureAccess};
use korppi_render::render2d::{QuadBatch2d, QuadBatchRenderer};
use korppi_rhi::{
    GraphicsDevice, PresentTarget, RhiError, SwapchainDescriptor, TextureFormat, TextureHandle,
};
use korppi_scene::{propagate_transforms, GlobalTransform, LocalTransform};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d};
use korppi_ui::render::{add_ui_clear_render_pass, UiDrawCommand};
use korppi_ui::{UiCanvasRenderer, UiInputDispatcher};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// 3D joint constraint connecting two ECS entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityJoint3d {
    /// Entity handle of the first connected body.
    pub entity_a: Entity,
    /// Entity handle of the second connected body.
    pub entity_b: Entity,
    /// Joint constraint configuration and geometric parameters.
    pub config: Joint3dConfig,
}

impl korppi_ecs::Component for EntityJoint3d {}

/// 3D joint constraint geometric configurations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Joint3dConfig {
    /// Spherical (ball-and-socket) joint: constrains anchor points to coincide in 3D.
    Spherical {
        /// Local-space anchor on body A.
        local_anchor_a: Vec3,
        /// Local-space anchor on body B.
        local_anchor_b: Vec3,
    },
    /// Revolute (hinge) joint: constrains motion to 1 rotational DOF along a hinge axis.
    Revolute {
        /// Local-space anchor on body A.
        local_anchor_a: Vec3,
        /// Local-space anchor on body B.
        local_anchor_b: Vec3,
        /// Local hinge axis on body A.
        local_axis_a: Vec3,
    },
    /// Prismatic (slider) joint: constrains motion to 1 linear translational DOF along a slider axis.
    Prismatic {
        /// Local-space anchor on body A.
        local_anchor_a: Vec3,
        /// Local-space anchor on body B.
        local_anchor_b: Vec3,
        /// Local slider axis on body A.
        local_axis_a: Vec3,
    },
    /// Distance joint: maintains a fixed scalar separation distance between two anchor points.
    Distance {
        /// Local-space anchor on body A.
        local_anchor_a: Vec3,
        /// Local-space anchor on body B.
        local_anchor_b: Vec3,
        /// Target distance in meters.
        distance: f32,
    },
    /// Weld joint: eliminates all linear and angular relative motion between two bodies.
    Weld {
        /// Local-space anchor on body A.
        local_anchor_a: Vec3,
        /// Local-space anchor on body B.
        local_anchor_b: Vec3,
    },
}

/// 2D joint constraint connecting two ECS entities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityJoint2d {
    /// Entity handle of the first connected body.
    pub entity_a: Entity,
    /// Entity handle of the second connected body.
    pub entity_b: Entity,
    /// 2D joint constraint specification.
    pub joint: Joint2d,
}

impl korppi_ecs::Component for EntityJoint2d {}

/// Central engine application orchestrator unifying all engine subsystems.
pub struct EngineApp {
    /// Active engine configuration.
    pub config: EngineConfig,
    /// Primary Entity Component System world.
    pub world: World,
    /// Stage-based system pipeline scheduler.
    pub schedule: Schedule,
    /// Fixed-timestep simulation clock.
    pub clock: FixedClock,
    /// 3D physics simulation world.
    pub physics_3d: PhysicsWorld,
    /// 2D physics simulation world.
    pub physics_2d: PhysicsWorld2d,
    /// Unified input coordinator.
    pub input: InputManager,
    /// Spatial and raytraced audio coordinator.
    pub audio: AudioCoordinator,
    /// The audio output the mixer plays into; the voice API is the world's
    /// [`AudioVoices`](korppi_audio::mixer::AudioVoices) resource.
    pub audio_output: AudioOutput,
    /// The audio systems, run after transforms propagate.
    audio_systems: Schedule,
    /// The acoustic scene: a snapshot of the static 3D colliders with their entities'
    /// `AcousticMaterial`s, rebuilt when static bodies come or go (FIX-7.5, ADR-051). The world's
    /// `AudioOcclusion` traces it.
    pub acoustic_geometry: Option<Arc<AcousticGeometry>>,
    /// Whether a static body came or went since the acoustic scene was built.
    acoustics_stale: bool,
    /// Real-time diagnostic metrics.
    pub metrics: DiagnosticMetrics,
    /// Per-frame scratch linear bump allocator.
    pub frame_arena: FrameArena,
    /// UI synthetic and physical event dispatcher.
    pub ui_dispatcher: UiInputDispatcher,
    /// UI batch canvas renderer.
    pub ui_renderer: UiCanvasRenderer,
    /// Intermediate UI draw commands extracted during the most recent `Stage::Render`.
    pub ui_draw_commands: Vec<UiDrawCommand>,
    /// Compiled 2D quad batch for UI controls from the most recent `Stage::Render`.
    pub ui_quad_batch: QuadBatch2d,
    /// Background clear color (authored sRGB) used when presenting or rendering UI passes.
    pub clear_color: [f32; 4],
    /// Reusable scratch buffer for root UI control entities (Rule 4).
    ui_roots_scratch: Vec<Entity>,
    /// Lazily initialized GPU quad renderer for offscreen `RenderGraph` UI passes.
    offscreen_ui_quads: Option<QuadBatchRenderer>,
    /// Currently active RHI graphics device.
    pub graphics_device: Option<Arc<dyn GraphicsDevice>>,
    /// Active window presenter when attached to an OS window.
    pub presenter: Option<WindowPresenter>,
    /// Flag indicating if the application loop should continue running.
    pub is_running: bool,
    /// Entity to 3D physics body index mapping.
    entity_to_body3d: HashMap<Entity, BodyId3d>,
    /// Entity to 2D physics body mapping.
    entity_to_body2d: HashMap<Entity, BodyId2d>,
    /// Entity to 3D joint index mapping.
    entity_to_joint3d: HashMap<Entity, JointId3d>,
    /// Entity to 2D joint mapping.
    entity_to_joint2d: HashMap<Entity, JointId2d>,
    /// Reusable scratch buffer for 3D body mutation and stale-body sweeps (Rule 4).
    scratch_bodies_3d: Vec<(Entity, BodyId3d)>,
    /// Reusable scratch buffer for stale 3D joint sweeps (Rule 4).
    scratch_joints_3d: Vec<(Entity, JointId3d)>,
    /// Reusable scratch buffer for 2D body mutation and stale-body sweeps (Rule 4).
    scratch_bodies_2d: Vec<(Entity, BodyId2d)>,
    /// Reusable scratch buffer for stale 2D joint sweeps (Rule 4).
    scratch_joints_2d: Vec<(Entity, JointId2d)>,
}

impl QuadBatchSource for EngineApp {
    fn quad_batch(&self) -> &QuadBatch2d {
        &self.ui_quad_batch
    }
}

impl EngineApp {
    /// Creates a new `EngineApp` configured with default parameters.
    pub fn new() -> Result<Self, RhiError> {
        Self::with_config(EngineConfig::default())
    }

    /// Creates a new `EngineApp` configured with custom parameters.
    pub fn with_config(config: EngineConfig) -> Result<Self, RhiError> {
        let clock = FixedClock::new(config.tick_rate);
        let physics_3d = PhysicsWorld::new();
        let physics_2d = PhysicsWorld2d::new();
        let input = InputManager::new();
        let audio = AudioCoordinator::new(48_000.0);
        let audio_settings = AudioSettings::default();
        let (audio_output, voices) = AudioOutput::open(config.audio, &audio_settings)
            .unwrap_or_else(|error| {
                tracing::error!("cannot open the audio device ({error}); audio runs headless");
                AudioOutput::open(AudioBackend::Headless, &audio_settings)
                    .expect("a headless audio output always opens")
            });
        let mut world = World::new();
        world.insert_resource(voices);
        world.insert_resource(audio_settings);
        world.insert_resource(AudioOcclusion::default());
        world.insert_resource(AudioRoom::default());
        let mut audio_systems = Schedule::new();
        audio_systems
            .add(Stage::PostUpdate, update_audio_players)
            .run_if(|world| world.contains_resource::<AssetServer>());
        let metrics = DiagnosticMetrics::default();
        let frame_arena = FrameArena::with_capacity(1024 * 1024); // 1 MiB frame arena
        let ui_dispatcher = UiInputDispatcher::new();
        let ui_renderer = UiCanvasRenderer::new();

        let mut app = Self {
            config,
            world,
            schedule: Schedule::new(),
            clock,
            physics_3d,
            physics_2d,
            input,
            audio,
            audio_output,
            audio_systems,
            acoustic_geometry: None,
            acoustics_stale: false,
            metrics,
            frame_arena,
            ui_dispatcher,
            ui_renderer,
            ui_draw_commands: Vec::with_capacity(64),
            ui_quad_batch: QuadBatch2d::new(1024),
            clear_color: [0.05, 0.06, 0.08, 1.0],
            ui_roots_scratch: Vec::with_capacity(16),
            offscreen_ui_quads: None,
            graphics_device: None,
            presenter: None,
            is_running: true,
            entity_to_body3d: HashMap::new(),
            entity_to_body2d: HashMap::new(),
            entity_to_joint3d: HashMap::new(),
            entity_to_joint2d: HashMap::new(),
            scratch_bodies_3d: Vec::with_capacity(64),
            scratch_joints_3d: Vec::with_capacity(16),
            scratch_bodies_2d: Vec::with_capacity(64),
            scratch_joints_2d: Vec::with_capacity(16),
        };

        app.set_graphics_backend(app.config.backend)?;
        Ok(app)
    }

    fn destroy_offscreen_ui_quads(&mut self) -> Result<(), RhiError> {
        if let Some(quads) = self.offscreen_ui_quads.take() {
            if let Some(ref device) = self.graphics_device {
                device.wait_idle()?;
                quads.destroy(device.as_ref())?;
            }
        }
        Ok(())
    }

    /// Attaches an OS window presentation target and creates a [`WindowPresenter`] on the active
    /// graphics backend, unifying [`Self::graphics_device`] with the presenter's device.
    ///
    /// # Errors
    /// Returns [`RhiError::Unsupported`] if [`Self::config`]'s backend is [`GraphicsBackend::Headless`].
    pub fn attach_window(
        &mut self,
        target: Arc<dyn PresentTarget>,
        desc: &SwapchainDescriptor,
    ) -> Result<(), RhiError> {
        self.detach_window()?;
        self.destroy_offscreen_ui_quads()?;
        if let Some(ref device) = self.graphics_device {
            device.wait_idle()?;
        }
        let presenter = WindowPresenter::new(self.config.backend, target, desc)?;
        self.graphics_device = Some(presenter.device().clone());
        self.presenter = Some(presenter);
        Ok(())
    }

    /// Detaches the active [`WindowPresenter`], if any, waiting for GPU completion and releasing
    /// the window swapchain.
    pub fn detach_window(&mut self) -> Result<(), RhiError> {
        if let Some(presenter) = self.presenter.take() {
            presenter.device().wait_idle()?;
            drop(presenter);
        }
        Ok(())
    }

    /// Sets or changes the active graphics backend at runtime.
    ///
    /// When a window is attached via [`Self::attach_window`], migrates the live [`WindowPresenter`]
    /// to `backend` and updates [`Self::graphics_device`] to match the presenter's device.
    pub fn set_graphics_backend(&mut self, backend: GraphicsBackend) -> Result<(), RhiError> {
        if let Some(ref mut presenter) = self.presenter {
            if backend == GraphicsBackend::Headless {
                return presenter.set_backend(backend);
            }
            if let Some(quads) = self.offscreen_ui_quads.take() {
                if let Some(ref device) = self.graphics_device {
                    device.wait_idle()?;
                    quads.destroy(device.as_ref())?;
                }
            }
            presenter.set_backend(backend)?;
            self.graphics_device = Some(presenter.device().clone());
            self.config.backend = backend;
            return Ok(());
        }

        self.destroy_offscreen_ui_quads()?;
        if let Some(ref device) = self.graphics_device {
            device.wait_idle()?;
        }

        match backend {
            GraphicsBackend::Headless => {
                self.graphics_device = None;
            }
            GraphicsBackend::OpenGl => {
                let device = korppi_rhi_opengl::OpenGlDevice::create_headless()?;
                self.graphics_device = Some(Arc::new(device));
            }
            GraphicsBackend::Vulkan => {
                let device = korppi_rhi_vulkan::VulkanDevice::create_headless()?;
                self.graphics_device = Some(Arc::new(device));
            }
        }

        self.config.backend = backend;
        Ok(())
    }

    /// Synchronizes 3D ECS rigid bodies and colliders into the 3D physics world prior to simulation.
    fn sync_physics_3d_before_step(&mut self) {
        // Synchronize mutations on already-registered 3D bodies and their colliders before
        // removing stale bodies/joints (which may wake neighboring bodies in physics_3d).
        self.scratch_bodies_3d.clear();
        {
            let world = &self.world;
            let physics_3d = &self.physics_3d;
            self.scratch_bodies_3d
                .extend(
                    self.entity_to_body3d
                        .iter()
                        .filter_map(|(&entity, &body_id)| {
                            let ecs_b = world.get::<RigidBody>(entity)?;
                            let phys_b = physics_3d.body(body_id)?;
                            let body_changed = ecs_b != phys_b;
                            let phys_col = physics_3d
                                .collider_of(body_id)
                                .and_then(|cid| physics_3d.collider(cid));
                            let ecs_col = world.get::<Collider>(entity);
                            let col_changed = phys_col != ecs_col;
                            (body_changed || col_changed).then_some((entity, body_id))
                        }),
                );
        }

        if !self.scratch_bodies_3d.is_empty() {
            self.scratch_bodies_3d
                .sort_unstable_by_key(|&(entity, _)| entity);
            let mut body_type_recreated = false;

            for i in 0..self.scratch_bodies_3d.len() {
                let (entity, mut body_id) = self.scratch_bodies_3d[i];
                if let (Some(ecs_body), Some(phys_body)) = (
                    self.world.get::<RigidBody>(entity),
                    self.physics_3d.body(body_id),
                ) {
                    if ecs_body != phys_body {
                        if ecs_body.body_type != phys_body.body_type {
                            self.acoustics_stale |= ecs_body.body_type == BodyType::Static
                                || phys_body.body_type == BodyType::Static;
                            let new_body = ecs_body.clone();
                            let new_col = self.world.get::<Collider>(entity).cloned();
                            self.physics_3d.destroy_body(body_id);
                            let new_id = self.physics_3d.create_body(new_body);
                            if let Some(col) = new_col {
                                self.physics_3d.create_collider(new_id, col);
                            }
                            self.entity_to_body3d.insert(entity, new_id);
                            body_type_recreated = true;
                            continue;
                        }

                        if ecs_body.body_type == BodyType::Static
                            && (ecs_body.position != phys_body.position
                                || ecs_body.rotation != phys_body.rotation)
                        {
                            self.acoustics_stale = true;
                        }

                        let sleep_changed = ecs_body.is_sleeping != phys_body.is_sleeping;
                        let target_sleep = ecs_body.is_sleeping;
                        let has_collider = self.physics_3d.collider_of(body_id).is_some();
                        let mass_props_changed = ecs_body.motion_locks != phys_body.motion_locks
                            || (!has_collider
                                && (ecs_body.mass != phys_body.mass
                                    || ecs_body.local_inertia != phys_body.local_inertia
                                    || ecs_body.local_center != phys_body.local_center));
                        let ecs_snapshot = ecs_body.clone();

                        if let Some(pb) = self.physics_3d.body_mut(body_id) {
                            pb.position = ecs_snapshot.position;
                            pb.rotation = ecs_snapshot.rotation;
                            pb.linear_velocity = ecs_snapshot.linear_velocity;
                            pb.angular_velocity = ecs_snapshot.angular_velocity;
                            pb.linear_damping = ecs_snapshot.linear_damping;
                            pb.angular_damping = ecs_snapshot.angular_damping;
                            pb.gravity_scale = ecs_snapshot.gravity_scale;
                            pb.force = ecs_snapshot.force;
                            pb.torque = ecs_snapshot.torque;
                            pb.motion_locks = ecs_snapshot.motion_locks;
                            pb.allow_fast_rotation = ecs_snapshot.allow_fast_rotation;
                            pb.enable_sleep = ecs_snapshot.enable_sleep;
                            pb.is_bullet = ecs_snapshot.is_bullet;
                            pb.safety_factor = ecs_snapshot.safety_factor;
                            pb.enable_contact_recycling = ecs_snapshot.enable_contact_recycling;
                            if mass_props_changed {
                                pb.mass = ecs_snapshot.mass;
                                pb.local_inertia = ecs_snapshot.local_inertia;
                                pb.local_center = ecs_snapshot.local_center;
                                pb.recompute_mass_properties();
                            }
                            if sleep_changed && target_sleep {
                                pb.is_sleeping = true;
                            }
                        }
                    }
                }

                body_id = self.entity_to_body3d[&entity];
                let is_static = self
                    .physics_3d
                    .body(body_id)
                    .is_some_and(|b| b.body_type == BodyType::Static);
                let existing_cid = self.physics_3d.collider_of(body_id);
                let ecs_col = self.world.get::<Collider>(entity);

                match (existing_cid, ecs_col) {
                    (None, Some(col)) => {
                        self.acoustics_stale |= is_static;
                        self.physics_3d.create_collider(body_id, col.clone());
                        self.physics_3d.wake_body(body_id);
                    }
                    (Some(cid), None) => {
                        self.acoustics_stale |= is_static;
                        self.physics_3d.destroy_collider(cid);
                    }
                    (Some(cid), Some(col)) => {
                        if self.physics_3d.collider(cid) != Some(col) {
                            self.acoustics_stale |= is_static;
                            let cloned = col.clone();
                            self.physics_3d.destroy_collider(cid);
                            self.physics_3d.create_collider(body_id, cloned);
                            self.physics_3d.wake_body(body_id);
                        }
                    }
                    (None, None) => {}
                }
            }

            if body_type_recreated {
                let physics = &self.physics_3d;
                self.entity_to_joint3d
                    .retain(|_, &mut jid| physics.joint(jid).is_some());
            }
        }

        self.remove_stale_physics_3d();

        // Collect new entities with (RigidBody, Collider)
        let mut new_bodies = Vec::new();
        {
            let mut query = self.world.query_mut::<(Entity, &RigidBody, &Collider)>();
            for (entity, body, collider) in &mut query {
                if !self.entity_to_body3d.contains_key(&entity) {
                    new_bodies.push((entity, body.clone(), collider.clone()));
                }
            }
        }

        for (entity, body, collider) in new_bodies {
            self.acoustics_stale |= body.body_type == BodyType::Static;
            let body_id = self.physics_3d.create_body(body);
            self.physics_3d.create_collider(body_id, collider);
            self.entity_to_body3d.insert(entity, body_id);
        }

        // Also check entities with only RigidBody (no collider)
        let mut new_bare_bodies = Vec::new();
        {
            let mut query = self.world.query_mut::<(Entity, &RigidBody)>();
            for (entity, body) in &mut query {
                if !self.entity_to_body3d.contains_key(&entity) {
                    new_bare_bodies.push((entity, body.clone()));
                }
            }
        }
        for (entity, body) in new_bare_bodies {
            let body_id = self.physics_3d.create_body(body);
            self.entity_to_body3d.insert(entity, body_id);
        }

        // Synchronize 3D joints
        let mut new_joints = Vec::new();
        {
            let mut query = self.world.query_mut::<(Entity, &EntityJoint3d)>();
            for (entity, joint) in &mut query {
                if !self.entity_to_joint3d.contains_key(&entity) {
                    if let (Some(&body_a), Some(&body_b)) = (
                        self.entity_to_body3d.get(&joint.entity_a),
                        self.entity_to_body3d.get(&joint.entity_b),
                    ) {
                        let phys_joint = match joint.config {
                            Joint3dConfig::Spherical {
                                local_anchor_a,
                                local_anchor_b,
                            } => Joint3d::Spherical(SphericalJoint::new(
                                body_a,
                                body_b,
                                local_anchor_a,
                                local_anchor_b,
                            )),
                            Joint3dConfig::Revolute {
                                local_anchor_a,
                                local_anchor_b,
                                local_axis_a,
                            } => Joint3d::Revolute(RevoluteJoint::new(
                                body_a,
                                body_b,
                                local_anchor_a,
                                local_anchor_b,
                                local_axis_a,
                            )),
                            Joint3dConfig::Prismatic {
                                local_anchor_a,
                                local_anchor_b,
                                local_axis_a,
                            } => Joint3d::Prismatic(PrismaticJoint::new(
                                body_a,
                                body_b,
                                local_anchor_a,
                                local_anchor_b,
                                local_axis_a,
                            )),
                            Joint3dConfig::Distance {
                                local_anchor_a,
                                local_anchor_b,
                                distance,
                            } => Joint3d::Distance(DistanceJoint::new(
                                body_a,
                                body_b,
                                local_anchor_a,
                                local_anchor_b,
                                distance,
                            )),
                            Joint3dConfig::Weld {
                                local_anchor_a,
                                local_anchor_b,
                            } => Joint3d::Weld(WeldJoint::new(
                                body_a,
                                body_b,
                                local_anchor_a,
                                local_anchor_b,
                            )),
                        };
                        new_joints.push((entity, phys_joint));
                    }
                }
            }
        }
        for (entity, phys_joint) in new_joints {
            let j_id = self.physics_3d.create_joint(phys_joint);
            self.entity_to_joint3d.insert(entity, j_id);
        }
    }

    /// Synchronizes 3D simulation results back into ECS components after physics stepping.
    fn sync_physics_3d_after_step(&mut self) {
        for (&entity, &body_id) in &self.entity_to_body3d {
            if let Some(phys_body) = self.physics_3d.body(body_id) {
                let pos = phys_body.position;
                let rot = phys_body.rotation;
                let phys_snapshot = phys_body.clone();

                if let Some(rb) = self.world.get_mut::<RigidBody>(entity) {
                    *rb = phys_snapshot;
                }

                let mut local_scale = None;
                if let Some(local) = self.world.get_mut::<LocalTransform>(entity) {
                    local.translation = pos;
                    local.rotation = rot;
                    local_scale = Some(local.scale);
                }

                if let Some(global) = self.world.get_mut::<GlobalTransform>(entity) {
                    let scale = local_scale.unwrap_or_else(|| global.scale());
                    *global = GlobalTransform::from_transform(Transform {
                        translation: pos,
                        rotation: rot,
                        scale,
                    });
                }
            }
        }
    }

    /// Destroys the bodies of entities that were despawned or lost their `RigidBody`, and the
    /// joints of joint entities that went away. Joints destroyed with their bodies are
    /// forgotten too. Entities are handled in sorted order, so that slot reuse, and with it the
    /// simulation, stays deterministic.
    pub fn remove_stale_physics_3d(&mut self) {
        self.scratch_bodies_3d.clear();
        {
            let world = &self.world;
            self.scratch_bodies_3d.extend(
                self.entity_to_body3d
                    .iter()
                    .filter(|&(&entity, _)| {
                        !world.is_alive(entity) || world.get::<RigidBody>(entity).is_none()
                    })
                    .map(|(&entity, &body)| (entity, body)),
            );
        }
        if !self.scratch_bodies_3d.is_empty() {
            self.scratch_bodies_3d
                .sort_unstable_by_key(|&(entity, _)| entity);
            for i in 0..self.scratch_bodies_3d.len() {
                let (entity, body) = self.scratch_bodies_3d[i];
                self.acoustics_stale |= self
                    .physics_3d
                    .body(body)
                    .is_some_and(|body| body.body_type == BodyType::Static);
                self.physics_3d.destroy_body(body);
                self.entity_to_body3d.remove(&entity);
            }
        }

        self.scratch_joints_3d.clear();
        {
            let world = &self.world;
            let physics = &self.physics_3d;
            self.scratch_joints_3d.extend(
                self.entity_to_joint3d
                    .iter()
                    .filter(|&(&entity, &joint)| {
                        !world.is_alive(entity)
                            || world.get::<EntityJoint3d>(entity).is_none()
                            || physics.joint(joint).is_none()
                    })
                    .map(|(&entity, &joint)| (entity, joint)),
            );
        }
        if !self.scratch_joints_3d.is_empty() {
            self.scratch_joints_3d
                .sort_unstable_by_key(|&(entity, _)| entity);
            for i in 0..self.scratch_joints_3d.len() {
                let (entity, joint) = self.scratch_joints_3d[i];
                self.physics_3d.destroy_joint(joint);
                self.entity_to_joint3d.remove(&entity);
                if self.world.is_alive(entity) && self.world.get::<EntityJoint3d>(entity).is_some()
                {
                    self.world.remove_component::<EntityJoint3d>(entity);
                    if self
                        .world
                        .entity_components(entity)
                        .is_some_and(|comps| comps.is_empty())
                    {
                        self.world.despawn(entity);
                    }
                }
            }
        }
    }

    /// Synchronizes 2D ECS rigid bodies and colliders into the 2D physics world prior to simulation.
    fn sync_physics_2d_before_step(&mut self) {
        // Synchronize mutations on already-registered 2D bodies and their colliders before
        // removing stale bodies/joints (which may wake neighboring bodies in physics_2d).
        self.scratch_bodies_2d.clear();
        {
            let world = &self.world;
            let physics_2d = &self.physics_2d;
            let body3d_map = &self.entity_to_body3d;
            self.scratch_bodies_2d
                .extend(
                    self.entity_to_body2d
                        .iter()
                        .filter_map(|(&entity, &body_id)| {
                            if body3d_map.contains_key(&entity) {
                                return None;
                            }
                            let ecs_b = world.get::<RigidBody2d>(entity)?;
                            let phys_b = physics_2d.body(body_id)?;
                            let body_changed = ecs_b != phys_b;
                            let phys_col = physics_2d
                                .collider_of(body_id)
                                .and_then(|cid| physics_2d.collider(cid));
                            let ecs_col = world.get::<Collider2d>(entity);
                            let col_changed = phys_col != ecs_col;
                            (body_changed || col_changed).then_some((entity, body_id))
                        }),
                );
        }

        if !self.scratch_bodies_2d.is_empty() {
            self.scratch_bodies_2d
                .sort_unstable_by_key(|&(entity, _)| entity);

            for i in 0..self.scratch_bodies_2d.len() {
                let (entity, body_id) = self.scratch_bodies_2d[i];
                if let (Some(ecs_body), Some(phys_body)) = (
                    self.world.get::<RigidBody2d>(entity),
                    self.physics_2d.body(body_id),
                ) {
                    if ecs_body != phys_body {
                        let ecs_snapshot = ecs_body.clone();
                        let old_type = phys_body.body_type;
                        let old_enabled = phys_body.is_enabled();
                        let sleep_changed = ecs_snapshot.is_sleeping != phys_body.is_sleeping;
                        let target_sleep = ecs_snapshot.is_sleeping;
                        let locks_changed = ecs_snapshot.motion_locks != phys_body.motion_locks;

                        if ecs_snapshot.body_type != old_type {
                            self.physics_2d
                                .set_body_type(body_id, ecs_snapshot.body_type);
                        }
                        if ecs_snapshot.is_enabled() != old_enabled {
                            if ecs_snapshot.is_enabled() {
                                self.physics_2d.enable_body(body_id);
                            } else {
                                self.physics_2d.disable_body(body_id);
                            }
                        }

                        if let Some(pb) = self.physics_2d.body_mut(body_id) {
                            pb.position = ecs_snapshot.position;
                            pb.rotation = ecs_snapshot.rotation;
                            pb.linear_velocity = ecs_snapshot.linear_velocity;
                            pb.angular_velocity = ecs_snapshot.angular_velocity;
                            pb.linear_damping = ecs_snapshot.linear_damping;
                            pb.angular_damping = ecs_snapshot.angular_damping;
                            pb.gravity_scale = ecs_snapshot.gravity_scale;
                            pb.enable_sleep = ecs_snapshot.enable_sleep;
                            pb.is_bullet = ecs_snapshot.is_bullet;
                            pb.enable_contact_recycling = ecs_snapshot.enable_contact_recycling;
                            pb.allow_fast_rotation = ecs_snapshot.allow_fast_rotation;
                            pb.safety_factor = ecs_snapshot.safety_factor;
                            pb.force = ecs_snapshot.force;
                            pb.torque = ecs_snapshot.torque;
                            if locks_changed {
                                pb.set_motion_locks(ecs_snapshot.motion_locks);
                            }
                        }

                        if sleep_changed && target_sleep {
                            self.physics_2d.set_awake(body_id, false);
                        }
                    }
                }

                let existing_cid = self.physics_2d.collider_of(body_id);
                let ecs_col = self.world.get::<Collider2d>(entity).copied();

                match (existing_cid, ecs_col) {
                    (None, Some(col)) => {
                        self.physics_2d.create_collider(body_id, col);
                        self.physics_2d.set_awake(body_id, true);
                    }
                    (Some(cid), None) => {
                        self.physics_2d.destroy_collider(cid);
                    }
                    (Some(cid), Some(col)) => {
                        if self.physics_2d.collider(cid).copied() != Some(col) {
                            self.physics_2d.destroy_collider(cid);
                            self.physics_2d.create_collider(body_id, col);
                            self.physics_2d.set_awake(body_id, true);
                        }
                    }
                    (None, None) => {}
                }
            }
        }

        self.remove_stale_physics_2d();

        let mut new_bodies = Vec::new();
        {
            let mut query = self
                .world
                .query_mut::<(Entity, &RigidBody2d, &Collider2d)>();
            for (entity, body, collider) in &mut query {
                if !self.entity_to_body2d.contains_key(&entity)
                    && !self.entity_to_body3d.contains_key(&entity)
                {
                    new_bodies.push((entity, body.clone(), *collider));
                }
            }
        }

        for (entity, body, collider) in new_bodies {
            let body_id = self.physics_2d.create_body(body);
            self.physics_2d.create_collider(body_id, collider);
            self.entity_to_body2d.insert(entity, body_id);
        }

        let mut new_bare_bodies = Vec::new();
        {
            let mut query = self.world.query_mut::<(Entity, &RigidBody2d)>();
            for (entity, body) in &mut query {
                if !self.entity_to_body2d.contains_key(&entity)
                    && !self.entity_to_body3d.contains_key(&entity)
                {
                    new_bare_bodies.push((entity, body.clone()));
                }
            }
        }
        for (entity, body) in new_bare_bodies {
            let body_id = self.physics_2d.create_body(body);
            self.entity_to_body2d.insert(entity, body_id);
        }

        let mut new_joints = Vec::new();
        {
            let mut query = self.world.query_mut::<(Entity, &EntityJoint2d)>();
            for (entity, joint) in &mut query {
                if !self.entity_to_joint2d.contains_key(&entity) {
                    if let (Some(&body_a), Some(&body_b)) = (
                        self.entity_to_body2d.get(&joint.entity_a),
                        self.entity_to_body2d.get(&joint.entity_b),
                    ) {
                        new_joints.push((entity, body_a, body_b, joint.joint));
                    }
                }
            }
        }
        for (entity, body_a, body_b, joint_spec) in new_joints {
            let j_id = self.physics_2d.create_joint(body_a, body_b, joint_spec);
            self.entity_to_joint2d.insert(entity, j_id);
        }
    }

    /// Destroys the bodies of entities that were despawned or lost their `RigidBody2d`, and the
    /// joints of joint entities that went away. Joints destroyed with their bodies are
    /// forgotten too. Entities are handled in sorted order, so that slot reuse, and with it the
    /// simulation, stays deterministic.
    fn remove_stale_physics_2d(&mut self) {
        self.scratch_bodies_2d.clear();
        {
            let world = &self.world;
            let body3d_map = &self.entity_to_body3d;
            self.scratch_bodies_2d.extend(
                self.entity_to_body2d
                    .iter()
                    .filter(|&(&entity, _)| {
                        !world.is_alive(entity)
                            || world.get::<RigidBody2d>(entity).is_none()
                            || body3d_map.contains_key(&entity)
                    })
                    .map(|(&entity, &body)| (entity, body)),
            );
        }
        if !self.scratch_bodies_2d.is_empty() {
            self.scratch_bodies_2d
                .sort_unstable_by_key(|&(entity, _)| entity);
            for i in 0..self.scratch_bodies_2d.len() {
                let (entity, body) = self.scratch_bodies_2d[i];
                self.physics_2d.destroy_body(body);
                self.entity_to_body2d.remove(&entity);
            }
        }

        self.scratch_joints_2d.clear();
        {
            let world = &self.world;
            let physics = &self.physics_2d;
            self.scratch_joints_2d.extend(
                self.entity_to_joint2d
                    .iter()
                    .filter(|&(&entity, &joint)| {
                        !world.is_alive(entity)
                            || world.get::<EntityJoint2d>(entity).is_none()
                            || physics.joint(joint).is_none()
                    })
                    .map(|(&entity, &joint)| (entity, joint)),
            );
        }
        if !self.scratch_joints_2d.is_empty() {
            self.scratch_joints_2d
                .sort_unstable_by_key(|&(entity, _)| entity);
            for i in 0..self.scratch_joints_2d.len() {
                let (entity, joint) = self.scratch_joints_2d[i];
                self.physics_2d.destroy_joint(joint);
                self.entity_to_joint2d.remove(&entity);
                if self.world.is_alive(entity) && self.world.get::<EntityJoint2d>(entity).is_some()
                {
                    self.world.remove_component::<EntityJoint2d>(entity);
                    if self
                        .world
                        .entity_components(entity)
                        .is_some_and(|comps| comps.is_empty())
                    {
                        self.world.despawn(entity);
                    }
                }
            }
        }
    }

    /// Synchronizes 2D simulation results back into ECS components after physics stepping.
    fn sync_physics_2d_after_step(&mut self) {
        for (&entity, &body_id) in &self.entity_to_body2d {
            if let Some(phys_body) = self.physics_2d.body(body_id) {
                let pos = phys_body.position;
                let rot = phys_body.rotation;
                let rot_quat = Quat::from_axis_angle(Vec3::Z, rot.angle());
                let phys_snapshot = phys_body.clone();

                if let Some(rb) = self.world.get_mut::<RigidBody2d>(entity) {
                    *rb = phys_snapshot;
                }

                let mut local_z_and_scale = None;
                if let Some(local) = self.world.get_mut::<LocalTransform>(entity) {
                    local.translation.x = pos.x;
                    local.translation.y = pos.y;
                    local.rotation = rot_quat;
                    local_z_and_scale = Some((local.translation.z, local.scale));
                }

                if let Some(global) = self.world.get_mut::<GlobalTransform>(entity) {
                    let (z, scale) = local_z_and_scale
                        .unwrap_or_else(|| (global.translation().z, global.scale()));
                    *global = GlobalTransform::from_transform(Transform {
                        translation: Vec3::new(pos.x, pos.y, z),
                        rotation: rot_quat,
                        scale,
                    });
                }
            }
        }
    }

    /// Executes one fixed simulation tick for deterministic systems and physics solvers.
    fn step_fixed_simulation(&mut self) {
        self.schedule.run_stage(Stage::FixedUpdate, &mut self.world);

        // Synchronize and step 3D physics
        self.sync_physics_3d_before_step();
        let dt = self.clock.timestep().as_secs_f32();
        self.physics_3d.step(dt);
        self.sync_physics_3d_after_step();

        // Synchronize and step 2D physics
        self.sync_physics_2d_before_step();
        self.physics_2d.step(dt);
        self.sync_physics_2d_after_step();

        self.clock.tick();
        self.metrics.total_fixed_ticks += 1;
        self.metrics.fixed_updates_this_frame += 1;
    }

    /// Advances the engine simulation and rendering pipeline by `delta` time.
    pub fn update(&mut self, delta: Duration) {
        self.metrics.fixed_updates_this_frame = 0;
        self.clock.advance(delta);

        // 1. PreUpdate Stage
        self.frame_arena.reset();
        self.input.tick();
        self.schedule.run_stage(Stage::PreUpdate, &mut self.world);

        // 2. FixedUpdate Stage(s)
        while self.clock.should_tick() {
            self.step_fixed_simulation();
        }

        // 3. Update Stage (Gameplay logic)
        self.schedule.run_stage(Stage::Update, &mut self.world);

        // 4. PostUpdate Stage (Hierarchy, transforms, audio, UI layout)
        self.schedule.run_stage(Stage::PostUpdate, &mut self.world);
        propagate_transforms(&mut self.world);
        if self.acoustics_stale {
            self.rebuild_acoustic_geometry();
        }
        self.audio.update_ecs(&mut self.world);
        self.audio_systems
            .run_stage(Stage::PostUpdate, &mut self.world);
        self.audio_output.advance(delta);
        resolve_ui_layout_system(
            &mut self.world,
            Rect2d::new(
                0.0,
                0.0,
                self.config.window_size.x,
                self.config.window_size.y,
            ),
        );

        // 5. Render Stage
        self.metrics.alpha = self.clock.alpha();
        self.ui_renderer.collect_draw_commands_into(
            &self.world,
            &mut self.ui_roots_scratch,
            &mut self.ui_draw_commands,
        );
        self.ui_renderer
            .compile_into_quad_batch(&self.ui_draw_commands, &mut self.ui_quad_batch);
        self.schedule.run_stage(Stage::Render, &mut self.world);
        if let Some(ref mut presenter) = self.presenter {
            let canvas = (self.config.window_size.x, self.config.window_size.y);
            if let Err(error) =
                presenter.present_quads(&self.ui_quad_batch, canvas, self.clear_color)
            {
                tracing::error!("presenting UI frame failed: {error}");
            }
        }
        self.metrics.allocated_scratch_bytes = self.frame_arena.allocated_bytes();

        self.metrics.record_frame(delta);
    }

    /// Executes the UI canvas pass for the current [`Self::ui_quad_batch`] through a
    /// [`RenderGraph`] into an offscreen `target` texture of `format` and `width` x `height`.
    pub fn render_ui_graph(
        &mut self,
        target: TextureHandle,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        let device = self
            .graphics_device
            .as_ref()
            .ok_or_else(|| {
                RhiError::Unsupported("headless backend has no GPU device for UI rendering".into())
            })?
            .clone();

        if self
            .offscreen_ui_quads
            .as_ref()
            .map(QuadBatchRenderer::color_format)
            != Some(format)
        {
            if let Some(old) = self.offscreen_ui_quads.take() {
                device.wait_idle()?;
                old.destroy(device.as_ref())?;
            }
            self.offscreen_ui_quads = Some(QuadBatchRenderer::new(device.as_ref(), format)?);
        }

        let mut graph = RenderGraph::new();
        let color_target = graph.import_texture(
            target,
            TextureAccess {
                stage: korppi_rhi::PipelineStageFlags::TOP_OF_PIPE,
                access: korppi_rhi::AccessFlags::empty(),
                layout: korppi_rhi::ImageLayout::Undefined,
            },
            width,
            height,
        );
        add_ui_clear_render_pass(&mut graph, color_target, self.clear_color);
        let mut compiled = graph
            .compile()
            .map_err(|e| RhiError::ValidationError(e.to_string()))?;

        let frame = device.begin_frame()?;
        let mut recorder = device.create_command_recorder()?;
        recorder.begin()?;

        let canvas_size = (self.config.window_size.x, self.config.window_size.y);
        let batch = &self.ui_quad_batch;
        let quads = self.offscreen_ui_quads.as_mut().expect("created above");
        let recorded = compiled.execute_with(recorder.as_mut(), |pass_name, _ctx, cmd| {
            if pass_name == "ui_canvas_pass" {
                quads.record_in_pass(
                    device.as_ref(),
                    cmd,
                    frame,
                    (width, height),
                    canvas_size,
                    batch,
                )?;
            }
            Ok(())
        });

        let submitted = recorded.and_then(|()| device.submit(recorder.finish()?));
        device.end_frame()?;
        let submission = submitted?;
        device.wait_for_submission(submission)
    }

    /// Snapshots the static 3D colliders for the acoustics, each with its entity's
    /// `AcousticMaterial` (or the default), and points the world's `AudioOcclusion` at it.
    fn rebuild_acoustic_geometry(&mut self) {
        let body_entities: HashMap<BodyId3d, Entity> = self
            .entity_to_body3d
            .iter()
            .map(|(&entity, &body)| (body, entity))
            .collect();
        let world = &self.world;
        let physics = &self.physics_3d;
        let geometry = Arc::new(AcousticGeometry::from_physics(physics, |collider| {
            let entity = body_entities.get(&physics.collider_body(collider)?)?;
            world.get::<AcousticMaterial>(*entity).copied()
        }));
        self.world.insert_resource(AudioOcclusion {
            raycaster: Some(geometry.clone()),
        });
        if let Some(room) = self.world.resource_mut::<AudioRoom>() {
            room.geometry = Some(Arc::clone(&geometry));
        }
        self.acoustic_geometry = Some(geometry);
        self.acoustics_stale = false;
    }

    /// Stops the engine application.
    pub fn quit(&mut self) {
        self.is_running = false;
    }
}

impl Drop for EngineApp {
    fn drop(&mut self) {
        if let Err(error) = self.destroy_offscreen_ui_quads() {
            tracing::error!("destroying offscreen UI quad renderer: {error}");
        }
    }
}
