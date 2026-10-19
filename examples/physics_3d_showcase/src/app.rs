//! 3D Physics Showcase application and scene controller.

use crate::camera::Camera3d;
use crate::gallery::{
    FrictionRestitutionGallery, Gallery3d, HeavyStackingGallery, Ragdoll3dGallery,
    RollbackDeterminismGallery, SoftStepTuningGallery, SpatialQueriesGallery,
};
use korppi_core::clock::FixedClock;
use korppi_ecs::{Entity, World};
use korppi_input::mouse::MouseButton;
use korppi_input::KeyCode;
use korppi_math::{Aabb, Quat, Ray, Vec2, Vec3};
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::QueryFilter3d;
use korppi_physics::distance::ShapeProxy;
use korppi_physics::joint::{Frame3d, Joint, MotorJoint};
use korppi_physics::shape::Shape;
use korppi_physics::snapshot::WorldSnapshot3d;
use korppi_physics::{BodyId3d, JointId3d, PhysicsWorld, ShapeCastHit3d};
use korppi_render::debug3d::DebugDraw3d;
use korppi_render::render2d::batch::QuadBatch2d;
use korppi_ui::containers::{HBoxContainer, PanelContainer, VBoxContainer};
use korppi_ui::control::{Control, FocusMode, LayoutPreset, MouseFilter};
use korppi_ui::input::{UiEvent, UiInputDispatcher, UiReceiver};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect};
use korppi_ui::render::{UiCanvasRenderer, UiDrawCommand};
use korppi_ui::theme::{StyleBox, StyleBoxFlat, Theme, ThemeOverride};
use korppi_ui::widgets::{Button, Label};
use korppi_window::{KorppiApp, Window, WindowEvent};
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Application orchestrating 3D physics demonstration scenes, orbital/fly camera, picking, and debug drawing.
pub struct Physics3dShowcaseApp {
    /// Active 3D physics simulation world.
    pub world: PhysicsWorld,
    /// 3D Camera with perspective projection.
    pub camera: Camera3d,
    galleries: Vec<Box<dyn Gallery3d>>,
    /// Index of the active gallery (`0..6`).
    pub current_gallery: usize,
    /// Whether the simulation is paused.
    pub paused: bool,
    step_once: bool,
    clock: FixedClock,
    /// Whether 3D collision shape wireframes are drawn.
    pub draw_shapes: bool,
    /// Whether 3D wireframes are drawn (alias kept in sync with `draw_shapes`).
    pub draw_wireframes: bool,
    /// Whether 3D joint anchors and links are drawn.
    pub draw_joints: bool,
    /// Whether contact points and surface normals are drawn.
    pub draw_contacts: bool,
    /// Whether BVH broadphase AABBs are drawn.
    pub draw_bvh_aabbs: bool,
    /// Whether sleeping dynamic bodies are tinted blue-gray.
    pub draw_sleep_tint: bool,
    snapshots: VecDeque<WorldSnapshot3d>,
    debug_drawer: DebugDraw3d,
    /// Compiled 2D quad batch containing projected 3D debug lines and HUD overlay.
    pub quad_batch: QuadBatch2d,
    last_sim_time_ms: f32,
    smoothed_fps: f32,
    window_size: (f32, f32),
    has_mouse_pos: bool,
    last_mouse_pos: (f32, f32),
    is_left_down: bool,
    is_right_down: bool,
    ground_body_id: Option<BodyId3d>,
    drag_joint_id: Option<JointId3d>,
    dragged_body: Option<BodyId3d>,
    drag_distance: f32,
    last_overlap_count: usize,
    last_shape_cast_hit: Option<ShapeCastHit3d>,
    /// ECS world hosting the `korppi_ui` telemetry HUD.
    pub ui_world: World,
    /// Input dispatcher routing mouse/keyboard events into the HUD.
    pub ui_dispatcher: UiInputDispatcher,
    /// Canvas renderer compiling the HUD into `QuadBatch2d`.
    pub ui_renderer: UiCanvasRenderer,
    /// Reusable buffer of intermediate UI draw commands.
    pub ui_draw_commands: Vec<UiDrawCommand>,
    ui_roots_scratch: Vec<Entity>,
    hud_title_label: Entity,
    hud_desc_label: Entity,
    hud_timing_label: Entity,
    hud_counts_label: Entity,
    hud_controls_label: Entity,
    btn_prev: Entity,
    btn_next: Entity,
    btn_pause: Entity,
    btn_step: Entity,
    btn_reset: Entity,
}

impl Default for Physics3dShowcaseApp {
    fn default() -> Self {
        Self::new()
    }
}

impl Physics3dShowcaseApp {
    /// Creates a new 3D showcase application initialized to Gallery 1 (Heavy Stacking).
    #[must_use]
    pub fn new() -> Self {
        let galleries: Vec<Box<dyn Gallery3d>> = vec![
            Box::new(HeavyStackingGallery),
            Box::new(SoftStepTuningGallery),
            Box::new(FrictionRestitutionGallery),
            Box::new(Ragdoll3dGallery),
            Box::new(SpatialQueriesGallery),
            Box::new(RollbackDeterminismGallery),
        ];

        let mut ui_world = World::new();
        let (
            hud_title_label,
            hud_desc_label,
            hud_timing_label,
            hud_counts_label,
            hud_controls_label,
            btn_prev,
            btn_next,
            btn_pause,
            btn_step,
            btn_reset,
        ) = Self::build_hud(&mut ui_world);

        let mut clock = FixedClock::new(60);
        clock.set_max_sub_steps(8);

        let mut app = Self {
            world: PhysicsWorld::new(),
            camera: Camera3d::new(),
            galleries,
            current_gallery: 0,
            paused: false,
            step_once: false,
            clock,
            draw_shapes: true,
            draw_wireframes: true,
            draw_joints: true,
            draw_contacts: true,
            draw_bvh_aabbs: false,
            draw_sleep_tint: true,
            snapshots: VecDeque::with_capacity(128),
            debug_drawer: DebugDraw3d::new(),
            quad_batch: QuadBatch2d::new(32768),
            last_sim_time_ms: 0.0,
            smoothed_fps: 60.0,
            window_size: (1280.0, 720.0),
            has_mouse_pos: false,
            last_mouse_pos: (0.0, 0.0),
            is_left_down: false,
            is_right_down: false,
            ground_body_id: None,
            drag_joint_id: None,
            dragged_body: None,
            drag_distance: 10.0,
            last_overlap_count: 0,
            last_shape_cast_hit: None,
            ui_world,
            ui_dispatcher: UiInputDispatcher::new(),
            ui_renderer: UiCanvasRenderer::new(),
            ui_draw_commands: Vec::with_capacity(64),
            ui_roots_scratch: Vec::with_capacity(8),
            hud_title_label,
            hud_desc_label,
            hud_timing_label,
            hud_counts_label,
            hud_controls_label,
            btn_prev,
            btn_next,
            btn_pause,
            btn_step,
            btn_reset,
        };

        app.camera.update_orbit_eye();
        app.load_gallery(0);
        resolve_ui_layout_system(
            &mut app.ui_world,
            Rect2d::new(0.0, 0.0, app.window_size.0, app.window_size.1),
        );
        app
    }

    #[allow(clippy::type_complexity)]
    fn build_hud(
        ui_world: &mut World,
    ) -> (
        Entity,
        Entity,
        Entity,
        Entity,
        Entity,
        Entity,
        Entity,
        Entity,
        Entity,
        Entity,
    ) {
        let mut root_ctl = Control {
            mouse_filter: MouseFilter::Ignore,
            ..Default::default()
        };
        root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
        let root = ui_world.spawn((root_ctl, Theme::dark(), ResolvedRect::default()));

        let panel_ctl = Control {
            mouse_filter: MouseFilter::Pass,
            offset_left: 12.0,
            offset_top: 12.0,
            offset_right: 560.0,
            offset_bottom: 200.0,
            ..Default::default()
        };
        let mut panel_theme = ThemeOverride::default();
        panel_theme.set_stylebox(
            "panel",
            StyleBox::Flat(StyleBoxFlat {
                bg_color: [0.08, 0.10, 0.13, 0.90],
                border_width: [2.0, 2.0, 2.0, 2.0],
                border_color: [0.20, 0.85, 0.50, 1.0],
                ..StyleBoxFlat::default()
            }),
        );
        let panel = ui_world.spawn((
            panel_ctl,
            PanelContainer::all(10.0),
            panel_theme,
            ResolvedRect::default(),
        ));
        ui_world.add_pair::<korppi_ecs::ChildOf>(panel, root);

        let vbox_ctl = Control {
            mouse_filter: MouseFilter::Ignore,
            ..Default::default()
        };
        let vbox = ui_world.spawn((vbox_ctl, VBoxContainer::new(4.0), ResolvedRect::default()));
        ui_world.add_pair::<korppi_ecs::ChildOf>(vbox, panel);

        let spawn_label = |w: &mut World, parent: Entity, text: &str, color: [f32; 4]| -> Entity {
            let mut ov = ThemeOverride::default();
            ov.set_color("font_color", color);
            let ctl = Control {
                mouse_filter: MouseFilter::Ignore,
                custom_minimum_size: Vec2::new(520.0, 18.0),
                ..Default::default()
            };
            let e = w.spawn((ctl, Label::new(text), ov, ResolvedRect::default()));
            w.add_pair::<korppi_ecs::ChildOf>(e, parent);
            e
        };

        let title_label = spawn_label(
            ui_world,
            vbox,
            "Gallery 1/6: 1. Heavy Multi-Shape Stacking (100+ Bodies)",
            [0.25, 0.95, 0.60, 1.0],
        );
        let desc_label = spawn_label(
            ui_world,
            vbox,
            "Cross-hatch Jenga tower, 6-tier 3D pyramid, and mixed shape columns",
            [0.85, 0.88, 0.92, 1.0],
        );
        let timing_label = spawn_label(
            ui_world,
            vbox,
            "FPS: 60.0 | Step: 0.00 ms | State: RUNNING",
            [0.95, 0.95, 0.95, 1.0],
        );
        let counts_label = spawn_label(
            ui_world,
            vbox,
            "Bodies: 0 (Awake: 0, Sleep: 0) | Contacts: 0 | Joints: 0",
            [0.80, 0.90, 1.0, 1.0],
        );
        let controls_label = spawn_label(
            ui_world,
            vbox,
            "[1-6] Gallery [Space] Pause [T] Step [R] Reset [W/C/B/S] Overlays [Z] Rewind",
            [0.70, 0.75, 0.82, 1.0],
        );

        let btn_row_ctl = Control {
            mouse_filter: MouseFilter::Pass,
            ..Default::default()
        };
        let btn_row = ui_world.spawn((
            btn_row_ctl,
            HBoxContainer::new(6.0),
            ResolvedRect::default(),
        ));
        ui_world.add_pair::<korppi_ecs::ChildOf>(btn_row, vbox);

        let spawn_button = |w: &mut World, parent: Entity, text: &str| -> Entity {
            let ctl = Control {
                mouse_filter: MouseFilter::Stop,
                focus_mode: FocusMode::All,
                custom_minimum_size: Vec2::new(72.0, 26.0),
                ..Default::default()
            };
            let e = w.spawn((
                ctl,
                Button::new(text),
                UiReceiver::new(),
                ResolvedRect::default(),
            ));
            w.add_pair::<korppi_ecs::ChildOf>(e, parent);
            e
        };

        let btn_prev = spawn_button(ui_world, btn_row, "< Prev");
        let btn_next = spawn_button(ui_world, btn_row, "Next >");
        let btn_pause = spawn_button(ui_world, btn_row, "Pause");
        let btn_step = spawn_button(ui_world, btn_row, "Step");
        let btn_reset = spawn_button(ui_world, btn_row, "Reset");

        (
            title_label,
            desc_label,
            timing_label,
            counts_label,
            controls_label,
            btn_prev,
            btn_next,
            btn_pause,
            btn_step,
            btn_reset,
        )
    }

    /// Loads the gallery at `index` and initializes simulation bodies.
    pub fn load_gallery(&mut self, index: usize) {
        if index < self.galleries.len() {
            self.current_gallery = index;
            self.snapshots.clear();
            self.drag_joint_id = None;
            self.dragged_body = None;
            self.last_overlap_count = 0;
            self.last_shape_cast_hit = None;
            let mut clock = FixedClock::new(60);
            clock.set_max_sub_steps(8);
            self.clock = clock;
            self.world = PhysicsWorld::new();
            self.galleries[index].setup(&mut self.world);
            self.ground_body_id = self
                .world
                .bodies()
                .find(|(_, b)| b.body_type == BodyType::Static)
                .map(|(id, _)| id);
        }
    }

    /// Switches to the gallery at `index` (alias for [`Self::load_gallery`]).
    pub fn switch_gallery(&mut self, index: usize) {
        self.load_gallery(index);
    }

    /// Returns the human-readable title of the active gallery.
    #[must_use]
    pub fn current_gallery_name(&self) -> &'static str {
        self.galleries[self.current_gallery].name()
    }

    /// Returns the description of the active gallery.
    #[must_use]
    pub fn current_gallery_description(&self) -> &'static str {
        self.galleries[self.current_gallery].description()
    }

    /// Returns the active gallery index (0..5).
    #[must_use]
    pub fn current_gallery_index(&self) -> usize {
        self.current_gallery
    }

    /// Returns the count of registered galleries.
    #[must_use]
    pub fn gallery_count(&self) -> usize {
        self.galleries.len()
    }

    /// Returns an immutable reference to the physics world.
    #[must_use]
    pub fn world(&self) -> &PhysicsWorld {
        &self.world
    }

    /// Returns a mutable reference to the physics world.
    pub fn world_mut(&mut self) -> &mut PhysicsWorld {
        &mut self.world
    }

    /// Returns the rendered quad batch.
    #[must_use]
    pub fn quad_batch(&self) -> &QuadBatch2d {
        &self.quad_batch
    }

    /// Returns `true` if simulation is paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Sets simulation paused state.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Toggles simulation play/pause state.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Steps the physics simulation by exactly one tick when paused.
    pub fn step_single_tick(&mut self) {
        self.step_once = true;
    }

    /// Requests a single fixed-step tick while paused (alias for [`Self::step_single_tick`]).
    pub fn request_single_step(&mut self) {
        self.step_single_tick();
    }

    /// Resets the current gallery.
    pub fn reset_current_gallery(&mut self) {
        self.load_gallery(self.current_gallery);
    }

    /// Advances to the next gallery in sequence.
    pub fn next_gallery(&mut self) {
        let next = (self.current_gallery + 1) % self.galleries.len();
        self.load_gallery(next);
    }

    /// Navigates to the previous gallery in sequence.
    pub fn prev_gallery(&mut self) {
        let prev = if self.current_gallery == 0 {
            self.galleries.len() - 1
        } else {
            self.current_gallery - 1
        };
        self.load_gallery(prev);
    }

    /// Toggles wireframe rendering.
    pub fn toggle_wireframes(&mut self) {
        self.draw_wireframes = !self.draw_wireframes;
        self.draw_shapes = self.draw_wireframes;
    }

    /// Toggles contact point and normal visualization.
    pub fn toggle_contacts(&mut self) {
        self.draw_contacts = !self.draw_contacts;
    }

    /// Toggles BVH tree bounding box visualization.
    pub fn toggle_bvh_aabbs(&mut self) {
        self.draw_bvh_aabbs = !self.draw_bvh_aabbs;
    }

    /// Toggles sleep state tinting.
    pub fn toggle_sleep_tint(&mut self) {
        self.draw_sleep_tint = !self.draw_sleep_tint;
    }

    /// Adjusts the 3D soft-step solver contact frequency (`hertz`) and wakes dynamic bodies.
    pub fn adjust_contact_hertz(&mut self, delta: f32) {
        let cfg = self.world.solver_config_mut();
        cfg.hertz = (cfg.hertz + delta).clamp(5.0, 240.0);
        self.wake_all_dynamic_bodies();
    }

    /// Adjusts the 3D soft-step solver contact damping ratio (`damping_ratio`) and wakes dynamic bodies.
    pub fn adjust_contact_damping(&mut self, delta: f32) {
        let cfg = self.world.solver_config_mut();
        cfg.damping_ratio = (cfg.damping_ratio + delta).clamp(0.1, 20.0);
        self.wake_all_dynamic_bodies();
    }

    fn wake_all_dynamic_bodies(&mut self) {
        loop {
            let next_sleeping = self.world.bodies().find_map(|(id, b)| {
                (b.body_type == BodyType::Dynamic && b.is_sleeping).then_some(id)
            });
            let Some(id) = next_sleeping else {
                break;
            };
            if let Some(b) = self.world.body_mut(id) {
                b.is_sleeping = false;
            }
        }
    }

    /// Toggles camera mode between [`crate::camera::CameraMode::Orbit`] and [`crate::camera::CameraMode::Fly`].
    pub fn toggle_camera_mode(&mut self) {
        self.camera.toggle_mode();
    }

    /// Moves the camera in free-flight mode along its forward, right, and world-up axes.
    pub fn fly_camera(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        self.camera.fly_move(forward, right, up, dt);
    }

    /// Casts a world ray from `screen_pos` and attaches a 3D [`MotorJoint`] spring if a dynamic body is hit.
    pub fn pointer_down(&mut self, screen_pos: (f32, f32)) -> Option<BodyId3d> {
        let ray = self.camera.screen_to_ray(screen_pos, self.window_size);
        let hit = self.world.ray_cast(&ray, 200.0)?;
        let (local_anchor, mg) = self.world.body(hit.body_id).and_then(|body| {
            if body.body_type != BodyType::Dynamic {
                return None;
            }
            let local_anchor = body.rotation.conjugate() * (hit.point - body.position);
            let mg = (body.mass * 9.81).max(10.0);
            Some((local_anchor, mg))
        })?;

        let cached_ground = self.ground_body_id.filter(|&gid| {
            gid != hit.body_id
                && self
                    .world
                    .body(gid)
                    .is_some_and(|b| b.body_type == BodyType::Static)
        });
        let ground = cached_ground
            .or_else(|| {
                self.world
                    .bodies()
                    .find(|&(id, b)| id != hit.body_id && b.body_type == BodyType::Static)
                    .map(|(id, _)| id)
            })
            .unwrap_or_else(|| {
                self.world.create_body(RigidBody {
                    body_type: BodyType::Static,
                    ..Default::default()
                })
            });
        self.ground_body_id = Some(ground);

        let (ground_pos, ground_rot) = self
            .world
            .body(ground)
            .map(|g| (g.position, g.rotation))
            .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
        let target_local = ground_rot.conjugate() * (hit.point - ground_pos);

        let mut motor = MotorJoint::new(ground, hit.body_id)
            .with_local_frames(Frame3d::at(target_local), Frame3d::at(local_anchor))
            .with_linear_spring(15.0, 1.0, 200.0 * mg)
            .with_angular_spring(4.0, 1.0, 20.0 * mg);
        motor.base.collide_connected = true;

        let jid = self.world.create_joint(Joint::Motor(motor));
        self.drag_joint_id = Some(jid);
        self.dragged_body = Some(hit.body_id);
        self.drag_distance = (hit.point - ray.origin).length().max(1.0);
        Some(hit.body_id)
    }

    /// Updates the world-space target of the active 3D drag joint along the camera ray through `screen_pos`.
    pub fn pointer_move(&mut self, screen_pos: (f32, f32)) {
        let Some(jid) = self.drag_joint_id else {
            return;
        };
        let ray = self.camera.screen_to_ray(screen_pos, self.window_size);
        let target_world = ray.origin + ray.direction * self.drag_distance;

        let ground_pose = self
            .world
            .joint(jid)
            .and_then(|j| self.world.body(j.body_a()))
            .map(|g| (g.position, g.rotation));
        if let (Some((ground_pos, ground_rot)), Some(Joint::Motor(ref mut motor))) =
            (ground_pose, self.world.joint_mut(jid))
        {
            motor.base.local_frame_a.position =
                ground_rot.conjugate() * (target_world - ground_pos);
        }
    }

    /// Destroys the active 3D drag joint when the mouse button is released.
    pub fn pointer_up(&mut self) {
        if let Some(jid) = self.drag_joint_id.take() {
            self.world.destroy_joint(jid);
        }
        self.dragged_body = None;
    }

    /// Returns the active 3D drag joint handle, if a dynamic body is currently being dragged.
    #[must_use]
    pub const fn drag_joint(&self) -> Option<JointId3d> {
        self.drag_joint_id
    }

    /// Returns the number of dynamic colliders overlapping the spatial query volume in Gallery 5.
    #[must_use]
    pub const fn spatial_query_overlap_count(&self) -> usize {
        self.last_overlap_count
    }

    /// Returns the closest hit of the swept sphere shape cast in Gallery 5.
    #[must_use]
    pub const fn spatial_query_shape_cast_hit(&self) -> Option<ShapeCastHit3d> {
        self.last_shape_cast_hit
    }

    /// Restores the simulation state `ticks_back` ticks in the past from the snapshot ring buffer.
    pub fn rollback(&mut self, ticks_back: usize) -> bool {
        if self.snapshots.is_empty() {
            return false;
        }

        let target_idx = if ticks_back >= self.snapshots.len() {
            0
        } else {
            self.snapshots.len() - 1 - ticks_back
        };

        if let Some(snapshot) = self.snapshots.get(target_idx).cloned() {
            self.world.restore_snapshot(&snapshot);
            // Truncate newer snapshots after target
            self.snapshots.truncate(target_idx + 1);
            true
        } else {
            false
        }
    }

    /// Advances the simulation by `dt` seconds and records a state snapshot in the ring buffer
    /// when the Rollback Determinism gallery is active.
    pub fn step_simulation(&mut self, dt: f32) {
        // Record state snapshot before step only in Gallery 6 (Rollback Determinism)
        if self.current_gallery == 5 {
            if self.snapshots.len() >= 120 {
                self.snapshots.pop_front();
            }
            self.snapshots.push_back(self.world.create_snapshot());
        }

        let start = Instant::now();
        self.world.step(dt);
        self.last_sim_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    }

    /// Compiles 3D wireframe geometry into `DebugDraw3d` and projects onto `QuadBatch2d`.
    pub fn render_debug(&mut self) {
        self.debug_drawer.clear();
        self.quad_batch.clear();

        let draw_shapes = self.draw_shapes && self.draw_wireframes;

        // 1. Draw 3D Ground Grid on XZ plane and all bodies/collision shapes
        if draw_shapes {
            let grid_extent = 20.0f32;
            let grid_step = 2.0f32;
            let grid_color = [0.25, 0.28, 0.32, 0.5];

            let mut x = -grid_extent;
            while x <= grid_extent {
                self.debug_drawer.draw_line(
                    Vec3::new(x, 0.0, -grid_extent),
                    Vec3::new(x, 0.0, grid_extent),
                    grid_color,
                );
                x += grid_step;
            }

            let mut z = -grid_extent;
            while z <= grid_extent {
                self.debug_drawer.draw_line(
                    Vec3::new(-grid_extent, 0.0, z),
                    Vec3::new(grid_extent, 0.0, z),
                    grid_color,
                );
                z += grid_step;
            }

            for (id, body) in self.world.bodies() {
                let is_dragged = self.dragged_body == Some(id);
                let color = if is_dragged {
                    [1.0, 0.85, 0.2, 1.0]
                } else if body.is_sleeping && self.draw_sleep_tint {
                    [0.4, 0.5, 0.6, 0.8] // Sleeping blue-gray
                } else {
                    match body.body_type {
                        BodyType::Static => [0.45, 0.5, 0.55, 1.0],
                        BodyType::Kinematic => [0.85, 0.3, 0.75, 1.0],
                        BodyType::Dynamic => [0.2, 0.85, 0.45, 1.0],
                    }
                };

                for &collider_id in self.world.colliders_of(id) {
                    let Some(collider) = self.world.collider(collider_id) else {
                        continue;
                    };
                    // The shape's pose: the body's, moved by the collider's frame.
                    let (position, rotation) = collider.world_pose(body.position, body.rotation);
                    match collider.shape {
                        Shape::Box { half_extents } => {
                            self.debug_drawer
                                .draw_box(position, half_extents, rotation, color);
                        }
                        Shape::Sphere { radius } => {
                            self.debug_drawer.draw_sphere(position, radius, 16, color);
                        }
                        Shape::Capsule {
                            radius,
                            half_height,
                        } => {
                            let y_axis = rotation * Vec3::new(0.0, half_height, 0.0);
                            let p1 = position - y_axis;
                            let p2 = position + y_axis;
                            self.debug_drawer.draw_capsule(p1, p2, radius, 12, color);
                        }
                        Shape::ConvexHull(ref hull) => {
                            // Wireframe of the hull's edges
                            let points = hull.points();
                            for (i, j) in hull.edges() {
                                let p1 = position + (rotation * points[i]);
                                let p2 = position + (rotation * points[j]);
                                self.debug_drawer.draw_line(p1, p2, color);
                            }
                        }
                        Shape::TriangleMesh(ref mesh) => {
                            for tri in mesh.triangles() {
                                let p0 = position + (rotation * tri.vertices[0]);
                                let p1 = position + (rotation * tri.vertices[1]);
                                let p2 = position + (rotation * tri.vertices[2]);
                                self.debug_drawer.draw_line(p0, p1, color);
                                self.debug_drawer.draw_line(p1, p2, color);
                                self.debug_drawer.draw_line(p2, p0, color);
                            }
                        }
                        Shape::HeightField(ref hf) => {
                            for i in 0..hf.triangle_count() {
                                let tri = hf.triangle(i);
                                let p0 = position + (rotation * tri.vertices[0]);
                                let p1 = position + (rotation * tri.vertices[1]);
                                let p2 = position + (rotation * tri.vertices[2]);
                                self.debug_drawer.draw_line(p0, p1, color);
                                self.debug_drawer.draw_line(p1, p2, color);
                                self.debug_drawer.draw_line(p2, p0, color);
                            }
                        }
                        Shape::Compound(ref comp) => {
                            for child in comp.children() {
                                let child_pos = position + rotation * child.position;
                                let child_rot = rotation * child.rotation;
                                match child.shape {
                                    Shape::Box { half_extents } => {
                                        self.debug_drawer.draw_box(
                                            child_pos,
                                            half_extents,
                                            child_rot,
                                            color,
                                        );
                                    }
                                    Shape::Sphere { radius } => {
                                        self.debug_drawer.draw_sphere(child_pos, radius, 16, color);
                                    }
                                    Shape::Capsule {
                                        radius,
                                        half_height,
                                    } => {
                                        let y_axis = child_rot * Vec3::new(0.0, half_height, 0.0);
                                        let p1 = child_pos - y_axis;
                                        let p2 = child_pos + y_axis;
                                        self.debug_drawer.draw_capsule(p1, p2, radius, 12, color);
                                    }
                                    Shape::ConvexHull(ref hull) => {
                                        let points = hull.points();
                                        for (i, j) in hull.edges() {
                                            let p1 = child_pos + (child_rot * points[i]);
                                            let p2 = child_pos + (child_rot * points[j]);
                                            self.debug_drawer.draw_line(p1, p2, color);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Shape::Triangle { a, b, c } => {
                            let p0 = position + (rotation * a);
                            let p1 = position + (rotation * b);
                            let p2 = position + (rotation * c);
                            self.debug_drawer.draw_line(p0, p1, color);
                            self.debug_drawer.draw_line(p1, p2, color);
                            self.debug_drawer.draw_line(p2, p0, color);
                        }
                    }
                }
            }
        }

        // 1b. Draw broadphase collider AABBs when toggled (`KeyB`)
        if self.draw_bvh_aabbs {
            for (id, body) in self.world.bodies() {
                for &collider_id in self.world.colliders_of(id) {
                    let Some(collider) = self.world.collider(collider_id) else {
                        continue;
                    };
                    let (position, rotation) = collider.world_pose(body.position, body.rotation);
                    let aabb = collider.shape.compute_aabb(position, rotation);
                    self.debug_drawer
                        .draw_aabb(aabb.min, aabb.max, [0.95, 0.75, 0.15, 0.65]);
                }
            }
        }

        // 2. Draw 3D joint linkages
        if self.draw_joints {
            for (_, joint) in self.world.joints() {
                if let (Some(ba), Some(bb)) = (
                    self.world.body(joint.body_a()),
                    self.world.body(joint.body_b()),
                ) {
                    let pa = ba.position + (ba.rotation * joint.local_anchor_a());
                    let pb = bb.position + (bb.rotation * joint.local_anchor_b());
                    self.debug_drawer.draw_line(pa, pb, [1.0, 0.65, 0.15, 0.95]);
                    self.debug_drawer
                        .draw_cross(pa, 0.15, [0.1, 0.95, 1.0, 1.0]);
                }
            }
        }

        // 3. Draw contact normals and points (zero steady-state heap allocations)
        if self.draw_contacts {
            for contact in self
                .world
                .contact_manager()
                .iter()
                .filter(|c| c.is_touching)
            {
                for manifold in contact.manifolds() {
                    for pt in &manifold.points {
                        self.debug_drawer
                            .draw_cross(pt.point, 0.2, [1.0, 1.0, 0.0, 1.0]);
                        self.debug_drawer.draw_normal(
                            pt.point,
                            pt.normal,
                            0.5,
                            [1.0, 0.25, 0.25, 0.9],
                        );
                    }
                }
            }
        }

        // 4. Special Visualization for Gallery 5 (Spatial Queries) — zero heap allocation
        if self.current_gallery == 4 {
            // 4a. Volume query AABB and highlight of overlapping dynamic colliders
            let query_aabb = Aabb {
                min: Vec3::new(-4.0, 0.0, -4.0),
                max: Vec3::new(4.0, 4.0, 4.0),
            };
            let mut overlap_count = 0usize;
            let world = &self.world;
            let drawer = &mut self.debug_drawer;
            world.overlap_aabb(&query_aabb, &QueryFilter3d::DEFAULT, |cid| {
                if let (Some(body_id), Some(collider)) =
                    (world.collider_body(cid), world.collider(cid))
                {
                    if let Some(body) = world.body(body_id) {
                        if body.body_type == BodyType::Dynamic {
                            overlap_count += 1;
                            if draw_shapes {
                                let (pos, rot) = collider.world_pose(body.position, body.rotation);
                                let bounds = collider.shape.compute_aabb(pos, rot).expand(0.06);
                                drawer.draw_aabb(bounds.min, bounds.max, [1.0, 0.9, 0.1, 0.95]);
                            }
                        }
                    }
                }
                true
            });
            self.last_overlap_count = overlap_count;

            // 4b. Swept sphere shape cast along -Z through the target cluster
            let cast_origin = Vec3::new(0.0, 1.5, 9.0);
            let cast_points = [cast_origin];
            let cast_radius = 0.45f32;
            let cast_proxy = ShapeProxy {
                points: &cast_points,
                radius: cast_radius,
            };
            let cast_translation = Vec3::new(0.0, 0.0, -18.0);
            let shape_hit = self.world.cast_shape_closest(
                &cast_proxy,
                cast_translation,
                &QueryFilter3d::DEFAULT,
            );
            self.last_shape_cast_hit = shape_hit;

            if draw_shapes {
                // Continuous ray cast along camera look vector
                let ray_origin = self.camera.eye;
                let ray_dir = (self.camera.target - self.camera.eye).normalize_or_zero();
                let test_ray = Ray::new(ray_origin, ray_dir);

                if let Some(hit) = self.world.ray_cast(&test_ray, 100.0) {
                    self.debug_drawer
                        .draw_line(ray_origin, hit.point, [0.0, 1.0, 0.8, 1.0]);
                    self.debug_drawer
                        .draw_cross(hit.point, 0.4, [1.0, 0.1, 0.1, 1.0]);
                    self.debug_drawer
                        .draw_normal(hit.point, hit.normal, 1.0, [1.0, 0.8, 0.2, 1.0]);
                } else {
                    self.debug_drawer
                        .draw_ray(ray_origin, ray_dir, 50.0, [0.4, 0.4, 0.4, 0.6]);
                }

                // Draw query AABB wireframe
                self.debug_drawer
                    .draw_aabb(query_aabb.min, query_aabb.max, [0.2, 0.7, 1.0, 0.7]);

                // Draw swept sphere shape cast start, trajectory, and impact
                self.debug_drawer
                    .draw_sphere(cast_origin, cast_radius, 12, [0.9, 0.4, 1.0, 0.8]);
                if let Some(hit) = shape_hit {
                    let impact_center = cast_origin + cast_translation * hit.fraction;
                    self.debug_drawer
                        .draw_line(cast_origin, impact_center, [0.95, 0.3, 1.0, 0.95]);
                    self.debug_drawer.draw_sphere(
                        impact_center,
                        cast_radius,
                        12,
                        [1.0, 0.2, 0.8, 1.0],
                    );
                    self.debug_drawer
                        .draw_cross(hit.point, 0.3, [1.0, 0.95, 0.2, 1.0]);
                    self.debug_drawer
                        .draw_normal(hit.point, hit.normal, 0.8, [1.0, 0.5, 0.9, 1.0]);
                } else {
                    self.debug_drawer.draw_line(
                        cast_origin,
                        cast_origin + cast_translation,
                        [0.6, 0.3, 0.7, 0.6],
                    );
                }
            }
        }

        // 5. Clip and project all 3D line primitives into QuadBatch2d using Camera3d
        let screen_size = self.window_size;
        let line_verts = self.debug_drawer.vertices();
        let line_indices = self.debug_drawer.indices();

        for &[i1, i2] in line_indices.as_chunks::<2>().0 {
            let v1 = line_verts[i1 as usize];
            let v2 = line_verts[i2 as usize];

            let p1 = Vec3::new(v1.position[0], v1.position[1], v1.position[2]);
            let p2 = Vec3::new(v2.position[0], v2.position[1], v2.position[2]);

            if let Some((s1, s2)) = self.camera.clip_and_project_line(p1, p2, screen_size) {
                self.quad_batch.push_line(
                    Vec2::new(s1.0, s1.1),
                    Vec2::new(s2.0, s2.1),
                    1.5,
                    v1.color,
                );
            }
        }

        // 6. Update and compile korppi_ui Telemetry HUD (zero steady-state heap allocations)
        self.update_hud_labels();
        resolve_ui_layout_system(
            &mut self.ui_world,
            Rect2d::new(0.0, 0.0, screen_size.0, screen_size.1),
        );
        self.ui_renderer.collect_draw_commands_into(
            &self.ui_world,
            &mut self.ui_roots_scratch,
            &mut self.ui_draw_commands,
        );
        self.ui_renderer
            .append_into_quad_batch(&self.ui_draw_commands, &mut self.quad_batch);
    }

    fn update_hud_labels(&mut self) {
        let gallery_idx = self.current_gallery + 1;
        let gallery_total = self.galleries.len();
        let gallery_name = self.current_gallery_name();
        let gallery_desc = self.current_gallery_description();
        let fps = self.smoothed_fps;
        let sim_ms = self.last_sim_time_ms;
        let state_str = if self.paused { "PAUSED" } else { "RUNNING" };

        let mut total_bodies = 0usize;
        let mut awake_bodies = 0usize;
        let mut sleeping_bodies = 0usize;
        for (_, body) in self.world.bodies() {
            total_bodies += 1;
            if body.body_type == BodyType::Dynamic {
                if body.is_sleeping {
                    sleeping_bodies += 1;
                } else {
                    awake_bodies += 1;
                }
            }
        }
        let contact_count = self.world.contact_count();
        let joint_count = self.world.joint_count();
        let hertz = self.world.solver_config().hertz;
        let damping = self.world.solver_config().damping_ratio;
        let cam_mode_str = match self.camera.mode {
            crate::camera::CameraMode::Orbit => "ORBIT",
            crate::camera::CameraMode::Fly => "FLY",
        };

        if let Some(lbl) = self.ui_world.get_mut::<Label>(self.hud_title_label) {
            lbl.update_text_with(|s| {
                let _ = write!(s, "Gallery {gallery_idx}/{gallery_total}: {gallery_name}");
            });
        }
        if let Some(lbl) = self.ui_world.get_mut::<Label>(self.hud_desc_label) {
            lbl.update_text_with(|s| {
                s.push_str(gallery_desc);
            });
        }
        if let Some(lbl) = self.ui_world.get_mut::<Label>(self.hud_timing_label) {
            lbl.update_text_with(|s| {
                let _ = write!(
                    s,
                    "FPS: {fps:.1} | Step: {sim_ms:.2} ms | State: {state_str} | Cam: {cam_mode_str}"
                );
            });
        }
        if let Some(lbl) = self.ui_world.get_mut::<Label>(self.hud_counts_label) {
            lbl.update_text_with(|s| {
                let _ = write!(
                    s,
                    "Bodies: {total_bodies} (Awake: {awake_bodies}, Sleep: {sleeping_bodies}) | Contacts: {contact_count} | Joints: {joint_count} | Hz: {hertz:.0} Damp: {damping:.2}"
                );
            });
        }
        if let Some(lbl) = self.ui_world.get_mut::<Label>(self.hud_controls_label) {
            lbl.update_text_with(|s| {
                s.push_str(
                    "[1-6] Gallery [Space] Pause [T] Step [R] Reset [W/C/B/S] Overlays [F] Cam [Z] Rewind",
                );
            });
        }
    }

    fn consume_hud_button_clicks(&mut self, pos: Vec2) -> bool {
        let buttons = [
            (self.btn_prev, 0u8),
            (self.btn_next, 1u8),
            (self.btn_pause, 2u8),
            (self.btn_step, 3u8),
            (self.btn_reset, 4u8),
        ];
        let mut clicked_action = None;
        for (entity, action) in buttons {
            let inside = self
                .ui_world
                .get::<ResolvedRect>(entity)
                .is_some_and(|r| r.global.contains_point(pos));
            if let Some(rx) = self.ui_world.get_mut::<UiReceiver>(entity) {
                let had_press = rx.events.iter().any(|e| {
                    matches!(
                        e,
                        UiEvent::MouseButton {
                            button: MouseButton::Left,
                            pressed: true,
                            ..
                        }
                    )
                });
                let had_release = rx.events.iter().any(|e| {
                    matches!(
                        e,
                        UiEvent::MouseButton {
                            button: MouseButton::Left,
                            pressed: false,
                            ..
                        }
                    )
                });
                rx.clear();
                if inside && had_press && had_release && clicked_action.is_none() {
                    clicked_action = Some(action);
                }
            }
        }
        match clicked_action {
            Some(0) => {
                self.prev_gallery();
                true
            }
            Some(1) => {
                self.next_gallery();
                true
            }
            Some(2) => {
                self.toggle_pause();
                true
            }
            Some(3) => {
                self.step_single_tick();
                true
            }
            Some(4) => {
                self.reset_current_gallery();
                true
            }
            _ => false,
        }
    }

    fn is_over_hud_button(&self, pos: Vec2) -> bool {
        [
            self.btn_prev,
            self.btn_next,
            self.btn_pause,
            self.btn_step,
            self.btn_reset,
        ]
        .iter()
        .any(|&e| {
            self.ui_world
                .get::<ResolvedRect>(e)
                .is_some_and(|r| r.global.contains_point(pos))
        })
    }
}

impl korppi_engine::QuadBatchSource for Physics3dShowcaseApp {
    fn quad_batch(&self) -> &QuadBatch2d {
        &self.quad_batch
    }
}

impl KorppiApp for Physics3dShowcaseApp {
    fn on_start(&mut self, window: &mut Window) {
        self.window_size = (window.width() as f32, window.height() as f32);
        self.camera.aspect = self.window_size.0 / self.window_size.1.max(1.0);
        tracing::info!(
            "Physics3dShowcaseApp started with gallery: {}",
            self.current_gallery_name()
        );
    }

    fn on_update(&mut self, window: &mut Window, delta_time: f32) {
        self.window_size = (window.width() as f32, window.height() as f32);
        self.camera.aspect = self.window_size.0 / self.window_size.1.max(1.0);
        for _ in window.drain_events() {}

        if delta_time > 1e-5 && delta_time.is_finite() {
            let inst_fps = 1.0 / delta_time;
            self.smoothed_fps = self.smoothed_fps * 0.9 + inst_fps * 0.1;
        }

        if self.paused {
            if self.step_once {
                self.step_simulation(1.0 / 60.0);
                self.step_once = false;
            }
        } else if self.step_once {
            self.step_simulation(1.0 / 60.0);
            self.step_once = false;
        } else if delta_time > 0.0 && delta_time.is_finite() {
            let nanos = (f64::from(delta_time) * 1_000_000_000.0).round() as u64;
            self.clock.advance(Duration::from_nanos(nanos));
            while self.clock.should_tick() {
                self.clock.tick();
                self.step_simulation(1.0 / 60.0);
            }
        }

        self.render_debug();
    }

    fn on_event(&mut self, _window: &mut Window, event: &WindowEvent) {
        match event {
            WindowEvent::Resized { width, height } => {
                self.window_size = (*width as f32, *height as f32);
                self.camera.aspect = self.window_size.0 / self.window_size.1.max(1.0);
                resolve_ui_layout_system(
                    &mut self.ui_world,
                    Rect2d::new(0.0, 0.0, self.window_size.0, self.window_size.1),
                );
            }
            WindowEvent::CursorMoved { x, y } => {
                let current_mouse = (*x as f32, *y as f32);
                if !self.has_mouse_pos {
                    self.has_mouse_pos = true;
                    self.last_mouse_pos = current_mouse;
                }
                let delta_x = current_mouse.0 - self.last_mouse_pos.0;
                let delta_y = current_mouse.1 - self.last_mouse_pos.1;

                if self.drag_joint_id.is_some() {
                    self.pointer_move(current_mouse);
                } else if self.is_left_down {
                    // Orbit or rotate camera
                    self.camera.orbit_drag(delta_x, delta_y);
                } else if self.is_right_down {
                    // Pan camera
                    self.camera.pan(delta_x, delta_y);
                }

                self.last_mouse_pos = current_mouse;
                let current_vec = Vec2::new(current_mouse.0, current_mouse.1);
                self.ui_dispatcher
                    .inject_mouse_move(&mut self.ui_world, current_vec);
                for &btn in &[
                    self.btn_prev,
                    self.btn_next,
                    self.btn_pause,
                    self.btn_step,
                    self.btn_reset,
                ] {
                    if let Some(rx) = self.ui_world.get_mut::<UiReceiver>(btn) {
                        rx.events
                            .retain(|e| matches!(e, UiEvent::MouseButton { .. }));
                    }
                }
            }
            WindowEvent::CursorLeft | WindowEvent::Focused(false) => {
                self.has_mouse_pos = false;
                self.is_left_down = false;
                self.is_right_down = false;
                self.pointer_up();
            }
            WindowEvent::MouseInput { button, pressed } => {
                let pos = Vec2::new(self.last_mouse_pos.0, self.last_mouse_pos.1);
                if *button == 0 {
                    self.ui_dispatcher.inject_mouse_button(
                        &mut self.ui_world,
                        pos,
                        MouseButton::Left,
                        *pressed,
                    );
                    if *pressed {
                        if !self.is_over_hud_button(pos)
                            && self.pointer_down(self.last_mouse_pos).is_none()
                        {
                            self.is_left_down = true;
                        }
                    } else {
                        let _ = self.consume_hud_button_clicks(pos);
                        self.pointer_up();
                        self.is_left_down = false;
                    }
                } else if *button == 1 {
                    self.is_right_down = *pressed;
                }
            }
            WindowEvent::MouseWheel { delta_y, .. } => {
                let factor = if *delta_y > 0.0 { 0.9 } else { 1.1 };
                self.camera.zoom(factor);
            }
            WindowEvent::KeyboardInput {
                key_code: Some(code),
                pressed: true,
            } => {
                self.handle_raw_key_code(*code);
            }
            _ => {}
        }
    }
}

impl Physics3dShowcaseApp {
    fn handle_raw_key_code(&mut self, code: u32) {
        let Some(key) = KeyCode::from_u32(code) else {
            return;
        };

        match key {
            KeyCode::Digit1 => self.load_gallery(0),
            KeyCode::Digit2 => self.load_gallery(1),
            KeyCode::Digit3 => self.load_gallery(2),
            KeyCode::Digit4 => self.load_gallery(3),
            KeyCode::Digit5 => self.load_gallery(4),
            KeyCode::Digit6 => self.load_gallery(5),
            KeyCode::Space => self.toggle_pause(),
            KeyCode::KeyT => self.step_single_tick(),
            KeyCode::KeyR => self.reset_current_gallery(),
            KeyCode::KeyW => self.toggle_wireframes(),
            KeyCode::KeyC => self.toggle_contacts(),
            KeyCode::KeyB => self.toggle_bvh_aabbs(),
            KeyCode::KeyS => self.toggle_sleep_tint(),
            KeyCode::KeyF => self.toggle_camera_mode(),
            KeyCode::KeyZ => {
                let _ = self.rollback(60);
            }
            KeyCode::ArrowUp => {
                if self.camera.mode == crate::camera::CameraMode::Fly {
                    self.fly_camera(1.0, 0.0, 0.0, 1.0 / 60.0);
                } else {
                    self.adjust_contact_hertz(5.0);
                }
            }
            KeyCode::ArrowDown => {
                if self.camera.mode == crate::camera::CameraMode::Fly {
                    self.fly_camera(-1.0, 0.0, 0.0, 1.0 / 60.0);
                } else {
                    self.adjust_contact_hertz(-5.0);
                }
            }
            KeyCode::PageUp => {
                if self.camera.mode == crate::camera::CameraMode::Fly {
                    self.fly_camera(0.0, 0.0, 1.0, 1.0 / 60.0);
                } else {
                    self.adjust_contact_damping(0.25);
                }
            }
            KeyCode::PageDown => {
                if self.camera.mode == crate::camera::CameraMode::Fly {
                    self.fly_camera(0.0, 0.0, -1.0, 1.0 / 60.0);
                } else {
                    self.adjust_contact_damping(-0.25);
                }
            }
            KeyCode::KeyN => self.next_gallery(),
            KeyCode::KeyP => self.prev_gallery(),
            KeyCode::ArrowRight => {
                if self.camera.mode == crate::camera::CameraMode::Fly {
                    self.fly_camera(0.0, 1.0, 0.0, 1.0 / 60.0);
                } else {
                    self.next_gallery();
                }
            }
            KeyCode::ArrowLeft => {
                if self.camera.mode == crate::camera::CameraMode::Fly {
                    self.fly_camera(0.0, -1.0, 0.0, 1.0 / 60.0);
                } else {
                    self.prev_gallery();
                }
            }
            _ => {}
        }
    }
}
