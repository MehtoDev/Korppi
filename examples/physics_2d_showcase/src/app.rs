//! 2D Physics Showcase application and scene controller.

use crate::gallery::{
    ContactVisualizerGallery, Gallery2d, JointsGallery, KineticGallery, RagdollGallery,
    StackingGallery,
};
use korppi_core::clock::FixedClock;
use korppi_ecs::{Entity, World};
use korppi_input::mouse::MouseButton;
use korppi_input::KeyCode;
use korppi_math::{Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::joint::{Joint2d, MotorJoint2d};
use korppi_physics2d::shape::{ChainSegment2d, Shape2d};
use korppi_physics2d::{
    BodyId2d, JointId2d, JointInstance2d, PhysicsWorld2d, QueryFilter2d, MAX_POLYGON_VERTICES,
};
use korppi_render::render2d::batch::QuadBatch2d;
use korppi_render::render2d::debug::DebugDraw2d;
use korppi_ui::containers::{HBoxContainer, PanelContainer, VBoxContainer};
use korppi_ui::control::{Control, FocusMode, LayoutPreset, MouseFilter};
use korppi_ui::input::{UiEvent, UiInputDispatcher, UiReceiver};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect};
use korppi_ui::render::{UiCanvasRenderer, UiDrawCommand};
use korppi_ui::theme::{StyleBox, StyleBoxFlat, Theme, ThemeOverride};
use korppi_ui::widgets::{Button, Label};
use korppi_window::{KorppiApp, Window, WindowEvent};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

/// Application orchestrating 2D physics demonstration scenes, camera, picking, and debug telemetry.
pub struct Physics2dShowcaseApp {
    /// Active 2D physics simulation world.
    pub world: PhysicsWorld2d,
    galleries: Vec<Box<dyn Gallery2d>>,
    /// Index of the active gallery (`0..5`).
    pub current_gallery: usize,
    /// Whether the simulation is paused.
    pub paused: bool,
    step_once: bool,
    clock: FixedClock,
    /// Whether collision shape wireframes are drawn.
    pub draw_shapes: bool,
    /// Whether contact manifolds and impulses are drawn.
    pub draw_contacts: bool,
    /// Whether joint anchors and links are drawn.
    pub draw_joints: bool,
    /// Whether sensor colliders are highlighted.
    pub draw_sensors: bool,
    /// Whether sleeping dynamic bodies are tinted blue-gray.
    pub draw_sleep_tint: bool,
    camera_offset: Vec2,
    camera_zoom: f32,
    screen_mouse_pos: Vec2,
    ground_body_id: Option<BodyId2d>,
    drag_joint_idx: Option<JointId2d>,
    dragged_body: Option<BodyId2d>,
    is_right_dragging: bool,
    has_mouse_pos: bool,
    last_mouse_screen: Vec2,
    /// Compiled 2D quad batch containing physics debug geometry and HUD overlay.
    pub quad_batch: QuadBatch2d,
    last_sim_time_ms: f32,
    smoothed_fps: f32,
    window_size: (f32, f32),
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

impl Default for Physics2dShowcaseApp {
    fn default() -> Self {
        Self::new()
    }
}

impl Physics2dShowcaseApp {
    /// Creates a new showcase application initialized to Gallery 1 (Stacking & Stability).
    #[must_use]
    pub fn new() -> Self {
        let galleries: Vec<Box<dyn Gallery2d>> = vec![
            Box::new(StackingGallery),
            Box::new(JointsGallery),
            Box::new(RagdollGallery),
            Box::new(KineticGallery),
            Box::new(ContactVisualizerGallery),
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
            world: PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -9.81)),
            galleries,
            current_gallery: 0,
            paused: false,
            step_once: false,
            clock,
            draw_shapes: true,
            draw_contacts: true,
            draw_joints: true,
            draw_sensors: true,
            draw_sleep_tint: true,
            camera_offset: Vec2::new(0.0, 4.0),
            camera_zoom: 32.0,
            screen_mouse_pos: Vec2::ZERO,
            ground_body_id: None,
            drag_joint_idx: None,
            dragged_body: None,
            is_right_dragging: false,
            has_mouse_pos: false,
            last_mouse_screen: Vec2::ZERO,
            quad_batch: QuadBatch2d::new(16384),
            last_sim_time_ms: 0.0,
            smoothed_fps: 60.0,
            window_size: (1280.0, 720.0),
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
            offset_right: 540.0,
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
                custom_minimum_size: Vec2::new(500.0, 18.0),
                ..Default::default()
            };
            let e = w.spawn((ctl, Label::new(text), ov, ResolvedRect::default()));
            w.add_pair::<korppi_ecs::ChildOf>(e, parent);
            e
        };

        let title_label = spawn_label(
            ui_world,
            vbox,
            "Gallery 1/5: 1. Stacking & Stability",
            [0.25, 0.95, 0.60, 1.0],
        );
        let desc_label = spawn_label(
            ui_world,
            vbox,
            "10-box tall tower, 21-box pyramid, irregular convex polygon wall",
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
            "[1-5] Gallery [Space] Pause [T] Step [R] Reset [C/J/S] Overlays",
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

    /// Loads the gallery at `index` and populates the simulation world.
    pub fn load_gallery(&mut self, index: usize) {
        if index < self.galleries.len() {
            self.current_gallery = index;
            self.drag_joint_idx = None;
            self.dragged_body = None;
            let mut clock = FixedClock::new(60);
            clock.set_max_sub_steps(8);
            self.clock = clock;
            self.world = PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -9.81));
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

    /// Returns the active gallery index (0..4).
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
    pub fn world(&self) -> &PhysicsWorld2d {
        &self.world
    }

    /// Returns a mutable reference to the physics world.
    pub fn world_mut(&mut self) -> &mut PhysicsWorld2d {
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

    /// Resets the current gallery back to its initial state.
    pub fn reset_current_gallery(&mut self) {
        self.load_gallery(self.current_gallery);
    }

    /// Advances to the next gallery in the sequence.
    pub fn next_gallery(&mut self) {
        let next = (self.current_gallery + 1) % self.galleries.len();
        self.load_gallery(next);
    }

    /// Navigates to the previous gallery in the sequence.
    pub fn prev_gallery(&mut self) {
        let prev = if self.current_gallery == 0 {
            self.galleries.len() - 1
        } else {
            self.current_gallery - 1
        };
        self.load_gallery(prev);
    }

    /// Toggles contact point and impulse vector visualization.
    pub fn toggle_contacts(&mut self) {
        self.draw_contacts = !self.draw_contacts;
    }

    /// Toggles joint constraint visualization.
    pub fn toggle_joints(&mut self) {
        self.draw_joints = !self.draw_joints;
    }

    /// Toggles body sleep tinting.
    pub fn toggle_sleep_tint(&mut self) {
        self.draw_sleep_tint = !self.draw_sleep_tint;
    }

    /// Transforms screen coordinates (pixels, top-left origin) to physics world space (meters).
    #[must_use]
    pub fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        let center_x = self.window_size.0 * 0.5;
        let center_y = self.window_size.1 * 0.5;
        let x = self.camera_offset.x + (screen.x - center_x) / self.camera_zoom;
        let y = self.camera_offset.y - (screen.y - center_y) / self.camera_zoom;
        Vec2::new(x, y)
    }

    /// Transforms physics world coordinates (meters) to screen coordinates (pixels).
    #[must_use]
    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        let center_x = self.window_size.0 * 0.5;
        let center_y = self.window_size.1 * 0.5;
        let x = center_x + (world.x - self.camera_offset.x) * self.camera_zoom;
        let y = center_y - (world.y - self.camera_offset.y) * self.camera_zoom;
        Vec2::new(x, y)
    }

    /// Drags a dynamic body under `world_pos` with a motor joint, as Box2D v3.2's samples do: a
    /// 7.5 Hz, critically damped spring with a hundred times the body's weight pulls the grabbed
    /// point to the pointer, and an angular velocity row acts as angular friction.
    pub fn pointer_down(&mut self, world_pos: Vec2) {
        let Some(body_id) = self.world.query_point(world_pos, QueryFilter2d::default()) else {
            return;
        };
        let Some((local_anchor, mg, lever)) = self.world.body(body_id).and_then(|body| {
            if body.body_type != BodyType::Dynamic {
                return None;
            }
            let local_anchor = body.rotation.inverse().rotate(world_pos - body.position);
            let mg = body.mass * self.world.gravity().length();
            let lever = if body.mass > 0.0 {
                (body.rotational_inertia / body.mass).sqrt()
            } else {
                0.0
            };
            Some((local_anchor, mg, lever))
        }) else {
            return;
        };

        // Ensure the drag joint is always anchored to a live Static ground body (never a
        // dynamic body or body_id itself).
        let cached_ground = self.ground_body_id.filter(|&gid| {
            gid != body_id
                && self
                    .world
                    .body(gid)
                    .is_some_and(|b| b.body_type == BodyType::Static)
        });
        let ground = cached_ground
            .or_else(|| {
                self.world
                    .bodies()
                    .find(|&(id, b)| id != body_id && b.body_type == BodyType::Static)
                    .map(|(id, _)| id)
            })
            .unwrap_or_else(|| self.world.create_body(RigidBody2d::new(BodyType::Static)));
        self.ground_body_id = Some(ground);

        let Some((ground_pos, ground_rot)) =
            self.world.body(ground).map(|g| (g.position, g.rotation))
        else {
            return;
        };
        let target = Transform2d::from_translation(ground_rot.inv_rotate(world_pos - ground_pos));
        let drag = MotorJoint2d::default()
            .with_local_frames(target, Transform2d::from_translation(local_anchor))
            .with_linear_spring(7.5, 1.0, 100.0 * mg)
            .with_angular_velocity(0.0, 0.25 * lever * mg);
        let j_idx = self.world.add_joint(
            JointInstance2d::new(ground, body_id, Joint2d::Motor(drag))
                .with_collide_connected(true),
        );
        self.world.set_awake(body_id, true);
        self.drag_joint_idx = Some(j_idx);
        self.dragged_body = Some(body_id);
    }

    /// Moves the drag target of the active drag joint to `world_pos`.
    pub fn pointer_move(&mut self, world_pos: Vec2) {
        if let Some(j_idx) = self.drag_joint_idx {
            let ground = self
                .world
                .joint(j_idx)
                .and_then(|j| self.world.body(j.body_a))
                .map(|g| (g.position, g.rotation));
            if let (Some((position, rotation)), Some(j_inst)) =
                (ground, self.world.joint_mut(j_idx))
            {
                if let Joint2d::Motor(ref mut drag) = j_inst.joint {
                    drag.local_frame_a.translation = rotation.inv_rotate(world_pos - position);
                }
            }
            if let Some(body_id) = self.dragged_body {
                self.world.set_awake(body_id, true);
            }
        }
    }

    /// Releases the active drag joint and releases the dragged body.
    pub fn pointer_up(&mut self) {
        if let Some(j_idx) = self.drag_joint_idx.take() {
            self.world.destroy_joint(j_idx);
        }
        self.dragged_body = None;
    }

    /// Advances the simulation by `dt` seconds and records step duration metrics.
    pub fn step_simulation(&mut self, dt: f32) {
        let start = Instant::now();

        // Gallery 2 (Joints & Constraints): oscillate the motorized prismatic elevator every 60 ticks.
        if self.current_gallery == 1
            && self.world.step_index() > 0
            && self.world.step_index().is_multiple_of(60)
        {
            let prismatic_id = self.world.joints().find_map(|(id, j)| {
                matches!(&j.joint, Joint2d::Prismatic(pj) if pj.enable_motor).then_some(id)
            });
            if let Some(jid) = prismatic_id {
                let mut bodies_to_wake = None;
                if let Some(j_inst) = self.world.joint_mut(jid) {
                    if let Joint2d::Prismatic(ref mut pj) = j_inst.joint {
                        pj.motor_speed = -pj.motor_speed;
                    }
                    bodies_to_wake = Some((j_inst.body_a, j_inst.body_b));
                }
                if let Some((ba, bb)) = bodies_to_wake {
                    self.world.set_awake(ba, true);
                    self.world.set_awake(bb, true);
                }
            }
        }

        self.world.step(dt);

        // Destroy any breakable joints whose constraint force or torque exceeded their thresholds.
        let mut broken = [None; 8];
        let mut broken_count = 0usize;
        for ev in self.world.joint_events() {
            if broken_count < broken.len() {
                broken[broken_count] = Some(ev.joint);
                broken_count += 1;
            }
        }
        for jid in broken[..broken_count].iter().flatten() {
            self.world.destroy_joint(*jid);
        }

        self.last_sim_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    }

    /// Compiles all bodies, colliders, joints, contact manifolds, and HUD telemetry into `QuadBatch2d`.
    pub fn render_debug(&mut self) {
        self.quad_batch.clear();

        let (win_w, win_h) = self.window_size;
        let cam_offset = self.camera_offset;
        let cam_zoom = self.camera_zoom;
        let draw_shapes = self.draw_shapes;
        let draw_joints = self.draw_joints;
        let draw_contacts = self.draw_contacts;
        let draw_sensors = self.draw_sensors;
        let draw_sleep_tint = self.draw_sleep_tint;
        let dragged_body = self.dragged_body;

        let to_screen = move |world: Vec2| -> Vec2 {
            let center_x = win_w * 0.5;
            let center_y = win_h * 0.5;
            let x = center_x + (world.x - cam_offset.x) * cam_zoom;
            let y = center_y - (world.y - cam_offset.y) * cam_zoom;
            Vec2::new(x, y)
        };

        {
            let world = &self.world;
            let mut drawer = DebugDraw2d::new(&mut self.quad_batch);

            // 1. Render all bodies and attached colliders
            if draw_shapes {
                for (id, body) in world.bodies() {
                    let is_dragged = dragged_body == Some(id);
                    let body_color = if is_dragged {
                        [1.0, 0.85, 0.2, 1.0] // Highlight yellow
                    } else if body.is_sleeping && draw_sleep_tint {
                        [0.35, 0.45, 0.55, 0.85] // Sleeping blue-gray
                    } else {
                        match body.body_type {
                            BodyType::Static => [0.45, 0.5, 0.52, 1.0],
                            BodyType::Kinematic => [0.85, 0.35, 0.75, 1.0],
                            BodyType::Dynamic => [0.2, 0.85, 0.5, 1.0],
                        }
                    };

                    for collider in world.colliders_of(id).filter_map(|c| world.collider(c)) {
                        if collider.is_sensor && !draw_sensors {
                            continue;
                        }
                        let color = if collider.is_sensor {
                            [0.2, 0.85, 1.0, 0.75]
                        } else {
                            body_color
                        };
                        match &collider.shape {
                            Shape2d::Circle(c) => {
                                let center_world = body.position + body.rotation.rotate(c.center);
                                let center_screen = to_screen(center_world);
                                let radius_screen = c.radius * cam_zoom;
                                drawer.draw_circle(center_screen, radius_screen, 24, 2.0, color);

                                // Draw orientation radius spoke
                                let spoke_end_world =
                                    center_world + body.rotation.rotate(Vec2::new(c.radius, 0.0));
                                let spoke_end_screen = to_screen(spoke_end_world);
                                drawer.draw_line(center_screen, spoke_end_screen, 2.0, color);
                            }
                            Shape2d::Capsule(cap) => {
                                let p1_world = body.position + body.rotation.rotate(cap.point1);
                                let p2_world = body.position + body.rotation.rotate(cap.point2);
                                let p1_screen = to_screen(p1_world);
                                let p2_screen = to_screen(p2_world);
                                let radius_screen = cap.radius * cam_zoom;
                                drawer.draw_capsule(
                                    p1_screen,
                                    p2_screen,
                                    radius_screen,
                                    16,
                                    2.0,
                                    color,
                                );
                            }
                            Shape2d::Polygon(poly) => {
                                let mut verts_screen = [Vec2::ZERO; MAX_POLYGON_VERTICES];
                                let src = poly.vertices();
                                let count = src.len().min(verts_screen.len());
                                for (dst, &v) in verts_screen[..count].iter_mut().zip(src.iter()) {
                                    *dst = to_screen(body.position + body.rotation.rotate(v));
                                }
                                drawer.draw_polygon(&verts_screen[..count], 2.0, color);
                            }
                            Shape2d::Segment(seg)
                            | Shape2d::ChainSegment(ChainSegment2d { segment: seg, .. }) => {
                                let p1_world = body.position + body.rotation.rotate(seg.point1);
                                let p2_world = body.position + body.rotation.rotate(seg.point2);
                                let p1_screen = to_screen(p1_world);
                                let p2_screen = to_screen(p2_world);
                                drawer.draw_line(p1_screen, p2_screen, 2.5, color);
                            }
                        }
                    }
                }
            }

            // 2. Render joint linkages
            if draw_joints {
                for (_, j_inst) in world.joints() {
                    let body_a_opt = world.body(j_inst.body_a);
                    let body_b_opt = world.body(j_inst.body_b);

                    let (Some(ba), Some(bb)) = (body_a_opt, body_b_opt) else {
                        continue;
                    };
                    let (frame_a, frame_b) = j_inst.joint.local_frames();
                    let pa = to_screen(ba.position + ba.rotation.rotate(frame_a.translation));
                    let pb = to_screen(bb.position + bb.rotation.rotate(frame_b.translation));
                    match &j_inst.joint {
                        Joint2d::Distance(_) => {
                            drawer.draw_line(pa, pb, 2.0, [0.0, 0.9, 0.9, 0.8]);
                            drawer.draw_point(pa, 6.0, [0.0, 1.0, 1.0, 1.0]);
                            drawer.draw_point(pb, 6.0, [0.0, 1.0, 1.0, 1.0]);
                        }
                        Joint2d::Revolute(_) => {
                            drawer.draw_circle(pa, 6.0, 12, 2.0, [1.0, 0.6, 0.1, 1.0]);
                        }
                        Joint2d::Prismatic(_) => {
                            drawer.draw_line(pa, pb, 2.5, [0.8, 0.3, 1.0, 0.9]);
                        }
                        Joint2d::Wheel(_) => drawer.draw_line(pa, pb, 2.0, [0.3, 0.6, 1.0, 0.9]),
                        Joint2d::Weld(_) => drawer.draw_point(pa, 8.0, [1.0, 0.4, 0.1, 1.0]),
                        Joint2d::Motor(_) => {
                            // The drag joint: from the grabbed point to the pointer.
                            drawer.draw_line(pb, pa, 3.0, [1.0, 0.15, 0.15, 1.0]);
                            drawer.draw_point(pa, 8.0, [1.0, 0.9, 0.1, 1.0]);
                        }
                        Joint2d::Pogo(_) => drawer.draw_line(pa, pb, 2.0, [0.4, 1.0, 0.4, 0.9]),
                        Joint2d::Filter | Joint2d::Mover(_) => {}
                    }
                }
            }

            // 3. Render contact manifolds and impulse vectors (zero steady-state heap allocations)
            if draw_contacts {
                for contact in world.contact_manager().values().filter(|c| c.is_touching) {
                    let manifold = &contact.manifold;
                    for i in 0..manifold.point_count {
                        let pt = &manifold.points[i];
                        let screen_pt = to_screen(pt.point);

                        // Yellow contact point cross
                        drawer.draw_point(screen_pt, 6.0, [1.0, 1.0, 0.1, 1.0]);

                        // Red normal impulse vector proportional to lambda_n
                        let normal_len_screen = (15.0 + pt.normal_impulse * 2.0).min(60.0);
                        let normal_screen_dir = Vec2::new(manifold.normal.x, -manifold.normal.y);
                        let normal_end = screen_pt + normal_screen_dir * normal_len_screen;
                        drawer.draw_line(screen_pt, normal_end, 2.0, [1.0, 0.25, 0.25, 0.9]);

                        // Cyan friction tangent vector
                        if pt.tangent_impulse.abs() > 1e-4 {
                            let tangent_screen_dir =
                                Vec2::new(-normal_screen_dir.y, normal_screen_dir.x);
                            let tangent_len = (pt.tangent_impulse * 2.0).clamp(-40.0, 40.0);
                            let tangent_end = screen_pt + tangent_screen_dir * tangent_len;
                            drawer.draw_line(screen_pt, tangent_end, 1.5, [0.1, 0.9, 1.0, 0.8]);
                        }
                    }
                }
            }
        }

        // 4. Update and compile korppi_ui Telemetry HUD (zero steady-state heap allocations)
        self.update_hud_labels();
        resolve_ui_layout_system(&mut self.ui_world, Rect2d::new(0.0, 0.0, win_w, win_h));
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
        let contact_count = self
            .world
            .contact_manager()
            .values()
            .filter(|c| c.is_touching)
            .count();
        let joint_count = self.world.joint_count();

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
                    "FPS: {fps:.1} | Step: {sim_ms:.2} ms | State: {state_str}"
                );
            });
        }
        if let Some(lbl) = self.ui_world.get_mut::<Label>(self.hud_counts_label) {
            lbl.update_text_with(|s| {
                let _ = write!(
                    s,
                    "Bodies: {total_bodies} (Awake: {awake_bodies}, Sleep: {sleeping_bodies}) | Contacts: {contact_count} | Joints: {joint_count}"
                );
            });
        }
        if let Some(lbl) = self.ui_world.get_mut::<Label>(self.hud_controls_label) {
            lbl.update_text_with(|s| {
                s.push_str("[1-5] Gallery [Space] Pause [T] Step [R] Reset [C/J/S] Overlays");
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

impl korppi_engine::QuadBatchSource for Physics2dShowcaseApp {
    fn quad_batch(&self) -> &QuadBatch2d {
        &self.quad_batch
    }
}

impl KorppiApp for Physics2dShowcaseApp {
    fn on_start(&mut self, window: &mut Window) {
        self.window_size = (window.width() as f32, window.height() as f32);
        tracing::info!(
            "Physics2dShowcaseApp started with gallery: {}",
            self.current_gallery_name()
        );
    }

    fn on_update(&mut self, window: &mut Window, delta_time: f32) {
        self.window_size = (window.width() as f32, window.height() as f32);
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
                resolve_ui_layout_system(
                    &mut self.ui_world,
                    Rect2d::new(0.0, 0.0, self.window_size.0, self.window_size.1),
                );
            }
            WindowEvent::CursorMoved { x, y } => {
                let current_screen = Vec2::new(*x as f32, *y as f32);
                if !self.has_mouse_pos {
                    self.has_mouse_pos = true;
                    self.last_mouse_screen = current_screen;
                }
                let delta = current_screen - self.last_mouse_screen;

                if self.is_right_dragging {
                    // Pan camera in world space
                    self.camera_offset.x -= delta.x / self.camera_zoom;
                    self.camera_offset.y += delta.y / self.camera_zoom;
                }

                self.screen_mouse_pos = current_screen;
                self.last_mouse_screen = current_screen;

                self.ui_dispatcher
                    .inject_mouse_move(&mut self.ui_world, current_screen);
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

                let world_pos = self.screen_to_world(current_screen);
                self.pointer_move(world_pos);
            }
            WindowEvent::CursorLeft | WindowEvent::Focused(false) => {
                self.has_mouse_pos = false;
                self.is_right_dragging = false;
                self.pointer_up();
            }
            WindowEvent::MouseInput { button, pressed } => {
                let pos = self.screen_mouse_pos;
                let world_pos = self.screen_to_world(pos);
                if *button == 0 {
                    self.ui_dispatcher.inject_mouse_button(
                        &mut self.ui_world,
                        pos,
                        MouseButton::Left,
                        *pressed,
                    );
                    if *pressed {
                        if !self.is_over_hud_button(pos) {
                            self.pointer_down(world_pos);
                        }
                    } else {
                        let _ = self.consume_hud_button_clicks(pos);
                        self.pointer_up();
                    }
                } else if *button == 1 {
                    // Right click
                    self.is_right_dragging = *pressed;
                }
            }
            WindowEvent::MouseWheel { delta_y, .. } => {
                let factor = if *delta_y > 0.0 { 1.15 } else { 1.0 / 1.15 };
                self.camera_zoom = (self.camera_zoom * factor).clamp(8.0, 120.0);
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

impl Physics2dShowcaseApp {
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
            KeyCode::Space => self.toggle_pause(),
            KeyCode::KeyT => self.step_single_tick(),
            KeyCode::KeyR => self.reset_current_gallery(),
            KeyCode::KeyC => self.toggle_contacts(),
            KeyCode::KeyJ => self.toggle_joints(),
            KeyCode::KeyS => self.toggle_sleep_tint(),
            KeyCode::KeyN | KeyCode::ArrowRight => self.next_gallery(),
            KeyCode::KeyP | KeyCode::ArrowLeft => self.prev_gallery(),
            _ => {}
        }
    }
}
