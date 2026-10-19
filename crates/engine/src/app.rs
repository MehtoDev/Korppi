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
    DistanceJoint, FilterJoint, Joint as Joint3d, MotorJoint, ParallelJoint, PrismaticJoint,
    RevoluteJoint, SphericalJoint, WeldJoint, WheelJoint,
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

impl EntityJoint3d {
    /// Creates an [`EntityJoint3d`] connecting `entity_a` and `entity_b` from any [`Joint3dConfig`] or 3D joint definition.
    #[must_use]
    pub fn new(entity_a: Entity, entity_b: Entity, config: impl Into<Joint3dConfig>) -> Self {
        Self {
            entity_a,
            entity_b,
            config: config.into(),
        }
    }
}

impl korppi_ecs::Component for EntityJoint3d {}

/// 3D joint constraint geometric and solver configurations.
#[allow(clippy::large_enum_variant)]
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
    /// Full 3D physics joint specification preserving all limits, motors, springs, frames, and joint types (`ENG-12`).
    Full(Joint3d),
}

impl Joint3dConfig {
    /// Materializes a [`Joint3d`] bound to `body_a` and `body_b` with cold solver impulses.
    #[must_use]
    pub fn to_joint(&self, body_a: BodyId3d, body_b: BodyId3d) -> Joint3d {
        match *self {
            Self::Spherical {
                local_anchor_a,
                local_anchor_b,
            } => Joint3d::Spherical(SphericalJoint::new(
                body_a,
                body_b,
                local_anchor_a,
                local_anchor_b,
            )),
            Self::Revolute {
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
            Self::Prismatic {
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
            Self::Distance {
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
            Self::Weld {
                local_anchor_a,
                local_anchor_b,
            } => Joint3d::Weld(WeldJoint::new(
                body_a,
                body_b,
                local_anchor_a,
                local_anchor_b,
            )),
            Self::Full(ref joint) => {
                let mut cloned = joint.clone();
                let base = cloned.base_mut();
                base.body_a = body_a.index();
                base.body_b = body_b.index();
                cloned.reset_impulses();
                cloned
            }
        }
    }
}

impl From<Joint3d> for Joint3dConfig {
    fn from(joint: Joint3d) -> Self {
        Self::Full(joint)
    }
}

impl From<SphericalJoint> for Joint3dConfig {
    fn from(joint: SphericalJoint) -> Self {
        Self::Full(Joint3d::Spherical(joint))
    }
}

impl From<RevoluteJoint> for Joint3dConfig {
    fn from(joint: RevoluteJoint) -> Self {
        Self::Full(Joint3d::Revolute(joint))
    }
}

impl From<PrismaticJoint> for Joint3dConfig {
    fn from(joint: PrismaticJoint) -> Self {
        Self::Full(Joint3d::Prismatic(joint))
    }
}

impl From<DistanceJoint> for Joint3dConfig {
    fn from(joint: DistanceJoint) -> Self {
        Self::Full(Joint3d::Distance(joint))
    }
}

impl From<WeldJoint> for Joint3dConfig {
    fn from(joint: WeldJoint) -> Self {
        Self::Full(Joint3d::Weld(joint))
    }
}

impl From<WheelJoint> for Joint3dConfig {
    fn from(joint: WheelJoint) -> Self {
        Self::Full(Joint3d::Wheel(joint))
    }
}

impl From<MotorJoint> for Joint3dConfig {
    fn from(joint: MotorJoint) -> Self {
        Self::Full(Joint3d::Motor(joint))
    }
}

impl From<ParallelJoint> for Joint3dConfig {
    fn from(joint: ParallelJoint) -> Self {
        Self::Full(Joint3d::Parallel(joint))
    }
}

impl From<FilterJoint> for Joint3dConfig {
    fn from(joint: FilterJoint) -> Self {
        Self::Full(Joint3d::Filter(joint))
    }
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

impl korppi_scene::EntityRemap for EntityJoint3d {
    fn remap_entities(&mut self, map: &HashMap<Entity, Entity>) {
        if let Some(&new_a) = map.get(&self.entity_a) {
            self.entity_a = new_a;
        }
        if let Some(&new_b) = map.get(&self.entity_b) {
            self.entity_b = new_b;
        }
    }
}

impl korppi_scene::EntityRemap for EntityJoint2d {
    fn remap_entities(&mut self, map: &HashMap<Entity, Entity>) {
        if let Some(&new_a) = map.get(&self.entity_a) {
            self.entity_a = new_a;
        }
        if let Some(&new_b) = map.get(&self.entity_b) {
            self.entity_b = new_b;
        }
    }
}

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
    /// Persistent render graph reused across `render_ui_graph` invocations (`ENG-14`).
    ui_render_graph: RenderGraph,
    /// Last cursor position forwarded into `ui_dispatcher` (`ENG-10`).
    last_dispatched_mouse_pos: Option<korppi_math::Vec2>,
    /// Currently active RHI graphics device.
    pub graphics_device: Option<Arc<dyn GraphicsDevice>>,
    /// Active window presenter when attached to an OS window.
    pub presenter: Option<WindowPresenter>,
    /// Flag indicating if the application loop should continue running.
    pub is_running: bool,
    /// Entity to 3D physics body index mapping.
    entity_to_body3d: HashMap<Entity, BodyId3d>,
    /// Reverse 3D physics body to Entity mapping (`ENG-09`).
    body3d_to_entity: HashMap<BodyId3d, Entity>,
    /// Entity to 2D physics body mapping.
    entity_to_body2d: HashMap<Entity, BodyId2d>,
    /// Entity to 3D joint index mapping.
    entity_to_joint3d: HashMap<Entity, JointId3d>,
    /// Last synchronized 3D joint state per entity (`ENG-12`).
    synced_joints_3d: HashMap<Entity, EntityJoint3d>,
    /// Entity to 2D joint mapping.
    entity_to_joint2d: HashMap<Entity, JointId2d>,
    /// Last synchronized 2D joint state per entity (`ENG-12`).
    synced_joints_2d: HashMap<Entity, EntityJoint2d>,
    /// Reusable scratch buffer for 3D body mutation and stale-body sweeps (Rule 4).
    scratch_bodies_3d: Vec<(Entity, BodyId3d)>,
    /// Reusable scratch buffer for stale 3D joint sweeps (Rule 4).
    scratch_joints_3d: Vec<(Entity, JointId3d)>,
    /// Reusable scratch buffer for 2D body mutation and stale-body sweeps (Rule 4).
    scratch_bodies_2d: Vec<(Entity, BodyId2d)>,
    /// Reusable scratch buffer for stale 2D joint sweeps (Rule 4).
    scratch_joints_2d: Vec<(Entity, JointId2d)>,
    /// Reusable scratch buffer for newly spawned 3D `(RigidBody, Collider)` entities (`ENG-09`).
    new_bodies_3d: Vec<(Entity, RigidBody, Collider)>,
    /// Reusable scratch buffer for newly spawned bare 3D `RigidBody` entities (`ENG-09`).
    new_bare_bodies_3d: Vec<(Entity, RigidBody)>,
    /// Reusable scratch buffer for newly spawned 3D joints (`ENG-09`, `ENG-12`).
    new_joints_3d: Vec<(Entity, EntityJoint3d, Joint3d)>,
    /// Reusable scratch buffer for newly spawned 2D `(RigidBody2d, Collider2d)` entities (`ENG-09`).
    new_bodies_2d: Vec<(Entity, RigidBody2d, Collider2d)>,
    /// Reusable scratch buffer for newly spawned bare 2D `RigidBody2d` entities (`ENG-09`).
    new_bare_bodies_2d: Vec<(Entity, RigidBody2d)>,
    /// Reusable scratch buffer for newly spawned 2D joints (`ENG-09`, `ENG-12`).
    new_joints_2d: Vec<(Entity, EntityJoint2d, BodyId2d, BodyId2d)>,
}

impl QuadBatchSource for EngineApp {
    fn quad_batch(&self) -> &QuadBatch2d {
        &self.ui_quad_batch
    }
}

impl EngineApp {
    /// Builds a [`korppi_scene::SceneRegistry`] pre-registered with standard engine, scene,
    /// physics, and audio components.
    #[must_use]
    pub fn default_scene_registry() -> korppi_scene::SceneRegistry {
        let mut registry = korppi_scene::SceneRegistry::default();
        registry.register::<RigidBody>("RigidBody");
        registry.register::<Collider>("Collider");
        registry.register::<RigidBody2d>("RigidBody2d");
        registry.register::<Collider2d>("Collider2d");
        registry.register_remappable::<EntityJoint3d>("EntityJoint3d");
        registry.register_remappable::<EntityJoint2d>("EntityJoint2d");
        registry.register::<korppi_audio::AudioListener>("AudioListener");
        registry.register::<korppi_audio::AcousticMaterialComponent>("AcousticMaterialComponent");
        registry.register::<AcousticMaterial>("AcousticMaterial");
        registry
    }

    /// Creates a new `EngineApp` configured with default parameters.
    pub fn new() -> Result<Self, RhiError> {
        Self::with_config(EngineConfig::default())
    }

    /// Creates a new `EngineApp` configured with custom parameters.
    pub fn with_config(config: EngineConfig) -> Result<Self, RhiError> {
        if config.tick_rate == 0 {
            return Err(RhiError::ValidationError(
                "EngineConfig::tick_rate must be greater than zero".into(),
            ));
        }
        if config.max_sub_steps == 0 {
            return Err(RhiError::ValidationError(
                "EngineConfig::max_sub_steps must be greater than zero".into(),
            ));
        }
        if !config.window_size.x.is_finite()
            || !config.window_size.y.is_finite()
            || config.window_size.x <= 0.0
            || config.window_size.y <= 0.0
        {
            return Err(RhiError::ValidationError(
                "EngineConfig::window_size dimensions must be finite and positive".into(),
            ));
        }
        let mut clock = FixedClock::new(config.tick_rate);
        clock.set_max_sub_steps(config.max_sub_steps);
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
            ui_render_graph: RenderGraph::new(),
            last_dispatched_mouse_pos: None,
            graphics_device: None,
            presenter: None,
            is_running: true,
            entity_to_body3d: HashMap::with_capacity(64),
            body3d_to_entity: HashMap::with_capacity(64),
            entity_to_body2d: HashMap::with_capacity(64),
            entity_to_joint3d: HashMap::with_capacity(16),
            synced_joints_3d: HashMap::with_capacity(16),
            entity_to_joint2d: HashMap::with_capacity(16),
            synced_joints_2d: HashMap::with_capacity(16),
            scratch_bodies_3d: Vec::with_capacity(64),
            scratch_joints_3d: Vec::with_capacity(16),
            scratch_bodies_2d: Vec::with_capacity(64),
            scratch_joints_2d: Vec::with_capacity(16),
            new_bodies_3d: Vec::with_capacity(64),
            new_bare_bodies_3d: Vec::with_capacity(64),
            new_joints_3d: Vec::with_capacity(16),
            new_bodies_2d: Vec::with_capacity(64),
            new_bare_bodies_2d: Vec::with_capacity(64),
            new_joints_2d: Vec::with_capacity(16),
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
                            self.body3d_to_entity.remove(&body_id);
                            let new_id = self.physics_3d.create_body(new_body);
                            if let Some(col) = new_col {
                                self.physics_3d.create_collider(new_id, col);
                            }
                            self.entity_to_body3d.insert(entity, new_id);
                            self.body3d_to_entity.insert(new_id, entity);
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
                let joint_map = &self.entity_to_joint3d;
                self.synced_joints_3d
                    .retain(|entity, _| joint_map.contains_key(entity));
            }
        }

        self.remove_stale_physics_3d();

        // Collect new entities with (RigidBody, Collider)
        self.new_bodies_3d.clear();
        {
            let entity_to_body3d = &self.entity_to_body3d;
            let new_bodies = &mut self.new_bodies_3d;
            let mut query = self.world.query_mut::<(Entity, &RigidBody, &Collider)>();
            for (entity, body, collider) in &mut query {
                if !entity_to_body3d.contains_key(&entity) {
                    new_bodies.push((entity, body.clone(), collider.clone()));
                }
            }
        }
        self.new_bodies_3d
            .sort_unstable_by_key(|&(entity, _, _)| entity);

        for (entity, body, collider) in self.new_bodies_3d.drain(..) {
            self.acoustics_stale |= body.body_type == BodyType::Static;
            let body_id = self.physics_3d.create_body(body);
            self.physics_3d.create_collider(body_id, collider);
            self.entity_to_body3d.insert(entity, body_id);
            self.body3d_to_entity.insert(body_id, entity);
        }

        // Also check entities with only RigidBody (no collider)
        self.new_bare_bodies_3d.clear();
        {
            let entity_to_body3d = &self.entity_to_body3d;
            let new_bare_bodies = &mut self.new_bare_bodies_3d;
            let mut query = self.world.query_mut::<(Entity, &RigidBody)>();
            for (entity, body) in &mut query {
                if !entity_to_body3d.contains_key(&entity) {
                    new_bare_bodies.push((entity, body.clone()));
                }
            }
        }
        self.new_bare_bodies_3d
            .sort_unstable_by_key(|&(entity, _)| entity);
        for (entity, body) in self.new_bare_bodies_3d.drain(..) {
            let body_id = self.physics_3d.create_body(body);
            self.entity_to_body3d.insert(entity, body_id);
            self.body3d_to_entity.insert(body_id, entity);
        }

        // Synchronize mutations on existing 3D joints (`ENG-12`)
        self.scratch_joints_3d.clear();
        {
            let world = &self.world;
            let synced = &self.synced_joints_3d;
            self.scratch_joints_3d.extend(
                self.entity_to_joint3d
                    .iter()
                    .filter(|&(&entity, _)| {
                        world.get::<EntityJoint3d>(entity) != synced.get(&entity)
                    })
                    .map(|(&entity, &jid)| (entity, jid)),
            );
        }
        if !self.scratch_joints_3d.is_empty() {
            self.scratch_joints_3d
                .sort_unstable_by_key(|&(entity, _)| entity);
            for i in 0..self.scratch_joints_3d.len() {
                let (entity, jid) = self.scratch_joints_3d[i];
                let Some(ecs_joint) = self.world.get::<EntityJoint3d>(entity).cloned() else {
                    continue;
                };
                if let (Some(&body_a), Some(&body_b)) = (
                    self.entity_to_body3d.get(&ecs_joint.entity_a),
                    self.entity_to_body3d.get(&ecs_joint.entity_b),
                ) {
                    let phys_joint = ecs_joint.config.to_joint(body_a, body_b);
                    let same_bodies = self.physics_3d.joint(jid).is_some_and(|j| {
                        j.body_a() == body_a.index() && j.body_b() == body_b.index()
                    });
                    if same_bodies {
                        if let Some(j) = self.physics_3d.joint_mut(jid) {
                            *j = phys_joint;
                        }
                    } else {
                        self.physics_3d.destroy_joint(jid);
                        let new_jid = self.physics_3d.create_joint(phys_joint);
                        self.entity_to_joint3d.insert(entity, new_jid);
                    }
                    self.synced_joints_3d.insert(entity, ecs_joint);
                }
            }
        }

        // Synchronize newly spawned 3D joints
        self.new_joints_3d.clear();
        {
            let entity_to_joint3d = &self.entity_to_joint3d;
            let entity_to_body3d = &self.entity_to_body3d;
            let new_joints = &mut self.new_joints_3d;
            let mut query = self.world.query_mut::<(Entity, &EntityJoint3d)>();
            for (entity, joint) in &mut query {
                if !entity_to_joint3d.contains_key(&entity) {
                    if let (Some(&body_a), Some(&body_b)) = (
                        entity_to_body3d.get(&joint.entity_a),
                        entity_to_body3d.get(&joint.entity_b),
                    ) {
                        let phys_joint = joint.config.to_joint(body_a, body_b);
                        new_joints.push((entity, joint.clone(), phys_joint));
                    }
                }
            }
        }
        self.new_joints_3d
            .sort_unstable_by_key(|&(entity, _, _)| entity);
        for (entity, ecs_joint, phys_joint) in self.new_joints_3d.drain(..) {
            let j_id = self.physics_3d.create_joint(phys_joint);
            self.entity_to_joint3d.insert(entity, j_id);
            self.synced_joints_3d.insert(entity, ecs_joint);
        }
    }

    /// Computes the world-space transform matrix of `entity`'s parent hierarchy (`None` if root).
    fn parent_world_matrix(world: &World, entity: Entity) -> Option<korppi_math::Mat4> {
        let _ = world.parent_of(entity)?;
        let mut acc = korppi_math::Mat4::IDENTITY;
        for anc in world.ancestors(entity) {
            let anc_mat = if let Some(local) = world.get::<LocalTransform>(anc) {
                local.compute_matrix()
            } else if let Some(global) = world.get::<GlobalTransform>(anc) {
                global.matrix
            } else {
                korppi_math::Mat4::IDENTITY
            };
            acc = anc_mat * acc;
        }
        Some(acc)
    }

    /// Synchronizes 3D simulation results back into ECS components after physics stepping (`ENG-11`).
    fn sync_physics_3d_after_step(&mut self) {
        self.scratch_bodies_3d.clear();
        self.scratch_bodies_3d
            .extend(self.entity_to_body3d.iter().map(|(&e, &b)| (e, b)));
        {
            let world = &self.world;
            self.scratch_bodies_3d
                .sort_unstable_by_key(|&(entity, _)| (world.ancestors(entity).count(), entity));
        }

        for i in 0..self.scratch_bodies_3d.len() {
            let (entity, body_id) = self.scratch_bodies_3d[i];
            if let Some(phys_body) = self.physics_3d.body(body_id) {
                let pos = phys_body.position;
                let rot = phys_body.rotation;

                if self.world.get::<RigidBody>(entity) != Some(phys_body) {
                    let phys_snapshot = phys_body.clone();
                    if let Some(rb) = self.world.get_mut::<RigidBody>(entity) {
                        *rb = phys_snapshot;
                    }
                }

                let parent_mat = Self::parent_world_matrix(&self.world, entity);
                let mut local_scale = None;
                let mut target_world_mat = None;
                if let Some(&current_local) = self.world.get::<LocalTransform>(entity) {
                    local_scale = Some(current_local.scale);
                    let desired_local = if let Some(p_mat) = parent_mat {
                        let p_rot = GlobalTransform::from_matrix(p_mat).rotation();
                        LocalTransform {
                            translation: p_mat.inverse().transform_point3(pos),
                            rotation: (p_rot.inverse() * rot).normalize(),
                            scale: current_local.scale,
                        }
                    } else {
                        LocalTransform {
                            translation: pos,
                            rotation: rot,
                            scale: current_local.scale,
                        }
                    };
                    let current_world_mat = parent_mat.map_or_else(
                        || current_local.compute_matrix(),
                        |p| p * current_local.compute_matrix(),
                    );
                    let cur_global = GlobalTransform::from_matrix(current_world_mat);
                    let pose_changed = if parent_mat.is_some() {
                        (cur_global.translation() - pos).length_squared() > 1e-12
                            || cur_global.rotation().dot(rot).abs() < 1.0 - 1e-6
                    } else {
                        current_local != desired_local
                    };
                    let final_local = if pose_changed {
                        if let Some(local) = self.world.get_mut::<LocalTransform>(entity) {
                            *local = desired_local;
                        }
                        desired_local
                    } else {
                        current_local
                    };
                    target_world_mat = Some(parent_mat.map_or_else(
                        || final_local.compute_matrix(),
                        |p| p * final_local.compute_matrix(),
                    ));
                }

                if let Some(&current_global) = self.world.get::<GlobalTransform>(entity) {
                    let desired_global = match target_world_mat {
                        Some(mat) => GlobalTransform::from_matrix(mat),
                        None => {
                            let scale = local_scale.unwrap_or_else(|| current_global.scale());
                            GlobalTransform::from_transform(Transform {
                                translation: pos,
                                rotation: rot,
                                scale,
                            })
                        }
                    };
                    if current_global != desired_global {
                        if let Some(global) = self.world.get_mut::<GlobalTransform>(entity) {
                            *global = desired_global;
                        }
                    }
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
                self.body3d_to_entity.remove(&body);
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
                self.synced_joints_3d.remove(&entity);
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

        self.new_bodies_2d.clear();
        {
            let entity_to_body2d = &self.entity_to_body2d;
            let entity_to_body3d = &self.entity_to_body3d;
            let new_bodies = &mut self.new_bodies_2d;
            let mut query = self
                .world
                .query_mut::<(Entity, &RigidBody2d, &Collider2d)>();
            for (entity, body, collider) in &mut query {
                if !entity_to_body2d.contains_key(&entity)
                    && !entity_to_body3d.contains_key(&entity)
                {
                    new_bodies.push((entity, body.clone(), *collider));
                }
            }
        }
        self.new_bodies_2d
            .sort_unstable_by_key(|&(entity, _, _)| entity);

        for (entity, body, collider) in self.new_bodies_2d.drain(..) {
            let body_id = self.physics_2d.create_body(body);
            self.physics_2d.create_collider(body_id, collider);
            self.entity_to_body2d.insert(entity, body_id);
        }

        self.new_bare_bodies_2d.clear();
        {
            let entity_to_body2d = &self.entity_to_body2d;
            let entity_to_body3d = &self.entity_to_body3d;
            let new_bare_bodies = &mut self.new_bare_bodies_2d;
            let mut query = self.world.query_mut::<(Entity, &RigidBody2d)>();
            for (entity, body) in &mut query {
                if !entity_to_body2d.contains_key(&entity)
                    && !entity_to_body3d.contains_key(&entity)
                {
                    new_bare_bodies.push((entity, body.clone()));
                }
            }
        }
        self.new_bare_bodies_2d
            .sort_unstable_by_key(|&(entity, _)| entity);
        for (entity, body) in self.new_bare_bodies_2d.drain(..) {
            let body_id = self.physics_2d.create_body(body);
            self.entity_to_body2d.insert(entity, body_id);
        }

        // Synchronize mutations on existing 2D joints (`ENG-12`)
        self.scratch_joints_2d.clear();
        {
            let world = &self.world;
            let synced = &self.synced_joints_2d;
            self.scratch_joints_2d.extend(
                self.entity_to_joint2d
                    .iter()
                    .filter(|&(&entity, _)| {
                        world.get::<EntityJoint2d>(entity) != synced.get(&entity)
                    })
                    .map(|(&entity, &jid)| (entity, jid)),
            );
        }
        if !self.scratch_joints_2d.is_empty() {
            self.scratch_joints_2d
                .sort_unstable_by_key(|&(entity, _)| entity);
            for i in 0..self.scratch_joints_2d.len() {
                let (entity, jid) = self.scratch_joints_2d[i];
                let Some(ecs_joint) = self.world.get::<EntityJoint2d>(entity).cloned() else {
                    continue;
                };
                if let (Some(&body_a), Some(&body_b)) = (
                    self.entity_to_body2d.get(&ecs_joint.entity_a),
                    self.entity_to_body2d.get(&ecs_joint.entity_b),
                ) {
                    let same_bodies = self
                        .physics_2d
                        .joint(jid)
                        .is_some_and(|inst| inst.body_a == body_a && inst.body_b == body_b);
                    if same_bodies {
                        if let Some(inst) = self.physics_2d.joint_mut(jid) {
                            inst.joint = ecs_joint.joint;
                        }
                    } else {
                        self.physics_2d.destroy_joint(jid);
                        let new_jid = self
                            .physics_2d
                            .create_joint(body_a, body_b, ecs_joint.joint);
                        self.entity_to_joint2d.insert(entity, new_jid);
                    }
                    self.synced_joints_2d.insert(entity, ecs_joint);
                }
            }
        }

        // Synchronize newly spawned 2D joints
        self.new_joints_2d.clear();
        {
            let entity_to_joint2d = &self.entity_to_joint2d;
            let entity_to_body2d = &self.entity_to_body2d;
            let new_joints = &mut self.new_joints_2d;
            let mut query = self.world.query_mut::<(Entity, &EntityJoint2d)>();
            for (entity, joint) in &mut query {
                if !entity_to_joint2d.contains_key(&entity) {
                    if let (Some(&body_a), Some(&body_b)) = (
                        entity_to_body2d.get(&joint.entity_a),
                        entity_to_body2d.get(&joint.entity_b),
                    ) {
                        new_joints.push((entity, joint.clone(), body_a, body_b));
                    }
                }
            }
        }
        self.new_joints_2d
            .sort_unstable_by_key(|&(entity, _, _, _)| entity);
        for (entity, ecs_joint, body_a, body_b) in self.new_joints_2d.drain(..) {
            let j_id = self
                .physics_2d
                .create_joint(body_a, body_b, ecs_joint.joint);
            self.entity_to_joint2d.insert(entity, j_id);
            self.synced_joints_2d.insert(entity, ecs_joint);
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
                self.synced_joints_2d.remove(&entity);
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

    /// Synchronizes 2D simulation results back into ECS components after physics stepping (`ENG-11`).
    fn sync_physics_2d_after_step(&mut self) {
        self.scratch_bodies_2d.clear();
        self.scratch_bodies_2d
            .extend(self.entity_to_body2d.iter().map(|(&e, &b)| (e, b)));
        {
            let world = &self.world;
            self.scratch_bodies_2d
                .sort_unstable_by_key(|&(entity, _)| (world.ancestors(entity).count(), entity));
        }

        for i in 0..self.scratch_bodies_2d.len() {
            let (entity, body_id) = self.scratch_bodies_2d[i];
            if let Some(phys_body) = self.physics_2d.body(body_id) {
                let pos = phys_body.position;
                let rot = phys_body.rotation;
                let rot_quat = Quat::from_axis_angle(Vec3::Z, rot.angle());

                if self.world.get::<RigidBody2d>(entity) != Some(phys_body) {
                    let phys_snapshot = phys_body.clone();
                    if let Some(rb) = self.world.get_mut::<RigidBody2d>(entity) {
                        *rb = phys_snapshot;
                    }
                }

                let parent_mat = Self::parent_world_matrix(&self.world, entity);
                let mut local_z_and_scale = None;
                let mut target_world_mat = None;
                if let Some(&current_local) = self.world.get::<LocalTransform>(entity) {
                    local_z_and_scale = Some((current_local.translation.z, current_local.scale));
                    let desired_local = if let Some(p_mat) = parent_mat {
                        let current_world = p_mat * current_local.compute_matrix();
                        let world_z = current_world.col(3).z;
                        let world_pos = Vec3::new(pos.x, pos.y, world_z);
                        let p_rot = GlobalTransform::from_matrix(p_mat).rotation();
                        LocalTransform {
                            translation: p_mat.inverse().transform_point3(world_pos),
                            rotation: (p_rot.inverse() * rot_quat).normalize(),
                            scale: current_local.scale,
                        }
                    } else {
                        LocalTransform {
                            translation: Vec3::new(pos.x, pos.y, current_local.translation.z),
                            rotation: rot_quat,
                            scale: current_local.scale,
                        }
                    };
                    let current_world_mat = parent_mat.map_or_else(
                        || current_local.compute_matrix(),
                        |p| p * current_local.compute_matrix(),
                    );
                    let cur_global = GlobalTransform::from_matrix(current_world_mat);
                    let cur_t = cur_global.translation();
                    let pose_changed = if parent_mat.is_some() {
                        (cur_t.x - pos.x).powi(2) + (cur_t.y - pos.y).powi(2) > 1e-12
                            || cur_global.rotation().dot(rot_quat).abs() < 1.0 - 1e-6
                    } else {
                        current_local != desired_local
                    };
                    let final_local = if pose_changed {
                        if let Some(local) = self.world.get_mut::<LocalTransform>(entity) {
                            *local = desired_local;
                        }
                        desired_local
                    } else {
                        current_local
                    };
                    target_world_mat = Some(parent_mat.map_or_else(
                        || final_local.compute_matrix(),
                        |p| p * final_local.compute_matrix(),
                    ));
                }

                if let Some(&current_global) = self.world.get::<GlobalTransform>(entity) {
                    let desired_global = match target_world_mat {
                        Some(mat) => GlobalTransform::from_matrix(mat),
                        None => {
                            let (z, scale) = local_z_and_scale.unwrap_or_else(|| {
                                (current_global.translation().z, current_global.scale())
                            });
                            GlobalTransform::from_transform(Transform {
                                translation: Vec3::new(pos.x, pos.y, z),
                                rotation: rot_quat,
                                scale,
                            })
                        }
                    };
                    if current_global != desired_global {
                        if let Some(global) = self.world.get_mut::<GlobalTransform>(entity) {
                            *global = desired_global;
                        }
                    }
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

    /// Dispatches current frame [`InputManager`] transitions into [`UiInputDispatcher`] (`ENG-10`).
    fn dispatch_input_to_ui(&mut self) {
        use korppi_input::{KeyCode, MouseButton};

        let pos = self.input.mouse.position();
        let buttons = [MouseButton::Left, MouseButton::Right, MouseButton::Middle];
        let any_button_edge = buttons
            .iter()
            .any(|&b| self.input.mouse.is_just_pressed(b) || self.input.mouse.is_just_released(b));
        let wheel = self.input.mouse.wheel_delta();
        let moved = self.last_dispatched_mouse_pos != Some(pos)
            || self.input.mouse.delta() != korppi_math::Vec2::ZERO;

        if moved || any_button_edge || wheel != korppi_math::Vec2::ZERO {
            if moved || any_button_edge {
                self.ui_dispatcher.inject_mouse_move(&mut self.world, pos);
                self.last_dispatched_mouse_pos = Some(pos);
            }
            for &button in &buttons {
                if self.input.mouse.is_just_pressed(button) {
                    self.ui_dispatcher
                        .inject_mouse_button(&mut self.world, pos, button, true);
                }
                if self.input.mouse.is_just_released(button) {
                    self.ui_dispatcher
                        .inject_mouse_button(&mut self.world, pos, button, false);
                }
            }
            if wheel != korppi_math::Vec2::ZERO {
                self.ui_dispatcher
                    .inject_mouse_wheel(&mut self.world, pos, wheel);
            }
        }

        for &key in KeyCode::ALL {
            if self.input.keyboard.is_just_pressed(key) {
                self.ui_dispatcher.inject_key(&mut self.world, key, true);
            }
            if self.input.keyboard.is_just_released(key) {
                self.ui_dispatcher.inject_key(&mut self.world, key, false);
            }
        }
    }

    /// Advances the engine simulation and rendering pipeline by `delta` time.
    pub fn update(&mut self, delta: Duration) {
        self.metrics.fixed_updates_this_frame = 0;
        self.clock.advance(delta);

        // 1. PreUpdate Stage
        self.frame_arena.reset();
        self.schedule.run_stage(Stage::PreUpdate, &mut self.world);

        // 2. FixedUpdate Stage(s)
        while self.clock.should_tick() {
            self.step_fixed_simulation();
        }

        // 3. Update Stage (Gameplay logic)
        self.schedule.run_stage(Stage::Update, &mut self.world);

        // 4. PostUpdate Stage (Hierarchy, transforms, audio, UI layout & UI input dispatch)
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
        self.dispatch_input_to_ui();

        // 5. Render Stage
        self.metrics.alpha = self.clock.alpha();
        self.ui_renderer.collect_draw_commands_into(
            &self.world,
            &mut self.ui_roots_scratch,
            &mut self.ui_draw_commands,
        );
        self.ui_renderer
            .compile_into_quad_batch(&self.ui_draw_commands, &mut self.ui_quad_batch);
        self.metrics.ui_quad_count = self.ui_quad_batch.quad_count() as u32;
        self.metrics.draw_calls = self.ui_quad_batch.draws().len() as u32
            + u32::from(
                self.ui_quad_batch.draws().is_empty() && !self.ui_quad_batch.indices().is_empty(),
            );
        self.metrics.froxel_light_count = if let Some(culling) =
            self.world
                .resource::<korppi_render::clustered::CullingOutput>()
        {
            culling.global_light_indices.len() as u32
        } else if let Some(scene) = self.world.resource::<korppi_render::ForwardScene>() {
            (scene.point_lights.len() + scene.spot_lights.len()) as u32
        } else {
            0
        };
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
        self.input.tick();

        self.metrics.record_frame(delta);
    }

    /// Forwards an OS [`korppi_window::WindowEvent`] into the engine's input and UI subsystems (`ENG-10`).
    pub fn handle_window_event(&mut self, event: &korppi_window::WindowEvent) {
        match *event {
            korppi_window::WindowEvent::Resized { width, height } => {
                if width > 0 && height > 0 {
                    self.config.window_size = korppi_math::Vec2::new(width as f32, height as f32);
                }
            }
            korppi_window::WindowEvent::Focused(focused) => {
                self.input.on_focused(focused);
            }
            korppi_window::WindowEvent::CloseRequested => {
                self.is_running = false;
            }
            korppi_window::WindowEvent::CursorMoved { x, y } => {
                self.input.on_cursor_moved(x, y);
            }
            korppi_window::WindowEvent::CursorLeft => {
                self.input.on_cursor_left();
            }
            korppi_window::WindowEvent::MouseWheel { delta_x, delta_y } => {
                self.input.on_mouse_wheel(delta_x, delta_y);
            }
            korppi_window::WindowEvent::MouseInput { button, pressed } => {
                self.input.on_mouse_input(button, pressed);
            }
            korppi_window::WindowEvent::KeyboardInput { key_code, pressed } => {
                self.input.on_key_input(key_code, pressed);
            }
            korppi_window::WindowEvent::ScaleFactorChanged { .. }
            | korppi_window::WindowEvent::Moved { .. }
            | korppi_window::WindowEvent::RedrawRequested
            | korppi_window::WindowEvent::CursorEntered => {}
        }
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

        self.ui_render_graph.clear();
        let color_target = self.ui_render_graph.import_texture(
            target,
            TextureAccess {
                stage: korppi_rhi::PipelineStageFlags::TOP_OF_PIPE,
                access: korppi_rhi::AccessFlags::empty(),
                layout: korppi_rhi::ImageLayout::Undefined,
            },
            width,
            height,
        );
        add_ui_clear_render_pass(&mut self.ui_render_graph, color_target, self.clear_color);
        let mut compiled = self
            .ui_render_graph
            .compile_reusable()
            .map_err(|e| RhiError::ValidationError(e.to_string()))?;

        let frame = device.begin_frame()?;
        let mut recorder = device.create_command_recorder()?;
        recorder.begin()?;

        let canvas_size = (width as f32, height as f32);
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
        let body_entities = &self.body3d_to_entity;
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
