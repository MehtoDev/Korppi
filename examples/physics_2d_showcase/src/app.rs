//! 2D Physics Showcase application and scene controller.

use crate::gallery::{
    ContactVisualizerGallery, Gallery2d, JointsGallery, KineticGallery, RagdollGallery,
    StackingGallery,
};
use korppi_math::{Transform2d, Vec2};
use korppi_physics2d::body::BodyType;
use korppi_physics2d::joint::{Joint2d, MotorJoint2d};
use korppi_physics2d::shape::{ChainSegment2d, Shape2d};
use korppi_physics2d::{BodyId2d, JointId2d, JointInstance2d, PhysicsWorld2d, QueryFilter2d};
use korppi_render::render2d::batch::QuadBatch2d;
use korppi_render::render2d::debug::DebugDraw2d;
use korppi_window::{KorppiApp, Window, WindowEvent};
use std::time::Instant;

/// Application orchestrating 2D physics demonstration scenes, camera, picking, and debug telemetry.
pub struct Physics2dShowcaseApp {
    /// Active 2D physics simulation world.
    pub world: PhysicsWorld2d,
    galleries: Vec<Box<dyn Gallery2d>>,
    current_gallery: usize,
    paused: bool,
    step_once: bool,
    draw_contacts: bool,
    draw_joints: bool,
    draw_sleep_tint: bool,
    camera_offset: Vec2,
    camera_zoom: f32,
    screen_mouse_pos: Vec2,
    drag_joint_idx: Option<JointId2d>,
    dragged_body: Option<BodyId2d>,
    is_right_dragging: bool,
    last_mouse_screen: Vec2,
    quad_batch: QuadBatch2d,
    last_sim_time_ms: f32,
    window_size: (f32, f32),
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

        let mut app = Self {
            world: PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -9.81)),
            galleries,
            current_gallery: 0,
            paused: false,
            step_once: false,
            draw_contacts: true,
            draw_joints: true,
            draw_sleep_tint: true,
            camera_offset: Vec2::new(0.0, 4.0),
            camera_zoom: 32.0,
            screen_mouse_pos: Vec2::ZERO,
            drag_joint_idx: None,
            dragged_body: None,
            is_right_dragging: false,
            last_mouse_screen: Vec2::ZERO,
            quad_batch: QuadBatch2d::new(16384),
            last_sim_time_ms: 0.0,
            window_size: (1280.0, 720.0),
        };

        app.load_gallery(0);
        app
    }

    /// Loads the gallery at `index` and populates the simulation world.
    pub fn load_gallery(&mut self, index: usize) {
        if index < self.galleries.len() {
            self.current_gallery = index;
            self.drag_joint_idx = None;
            self.dragged_body = None;
            self.world = PhysicsWorld2d::new().with_gravity(Vec2::new(0.0, -9.81));
            self.galleries[index].setup(&mut self.world);
        }
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
        if let Some(body_id) = self.world.query_point(world_pos, QueryFilter2d::default()) {
            if let Some(body) = self.world.body(body_id) {
                if body.body_type == BodyType::Dynamic {
                    let local_anchor = body.rotation.inverse().rotate(world_pos - body.position);
                    let mg = body.mass * self.world.gravity().length();
                    let lever = if body.mass > 0.0 {
                        (body.rotational_inertia / body.mass).sqrt()
                    } else {
                        0.0
                    };
                    // The first body is the gallery's ground, which holds the drag target; the
                    // dragged body must keep colliding with it.
                    let Some((ground, ground_body)) = self.world.bodies().next() else {
                        return;
                    };
                    let target = Transform2d::from_translation(
                        ground_body
                            .rotation
                            .inv_rotate(world_pos - ground_body.position),
                    );
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
            }
        }
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
        self.world.step(dt);
        self.last_sim_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    }

    /// Compiles all bodies, colliders, joints, contact manifolds, and HUD telemetry into `QuadBatch2d`.
    pub fn render_debug(&mut self) {
        self.quad_batch.clear();

        let (win_w, win_h) = self.window_size;
        let cam_offset = self.camera_offset;
        let cam_zoom = self.camera_zoom;
        let draw_joints = self.draw_joints;
        let draw_contacts = self.draw_contacts;
        let draw_sleep_tint = self.draw_sleep_tint;
        let dragged_body = self.dragged_body;

        let to_screen = move |world: Vec2| -> Vec2 {
            let center_x = win_w * 0.5;
            let center_y = win_h * 0.5;
            let x = center_x + (world.x - cam_offset.x) * cam_zoom;
            let y = center_y - (world.y - cam_offset.y) * cam_zoom;
            Vec2::new(x, y)
        };

        let world = &self.world;
        let mut drawer = DebugDraw2d::new(&mut self.quad_batch);

        // 1. Render all bodies and attached colliders
        for (id, body) in world.bodies() {
            let is_dragged = dragged_body == Some(id);
            let color = if is_dragged {
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
                        drawer.draw_capsule(p1_screen, p2_screen, radius_screen, 16, 2.0, color);
                    }
                    Shape2d::Polygon(poly) => {
                        let verts_screen: Vec<Vec2> = poly
                            .vertices()
                            .iter()
                            .map(|&v| {
                                let world_p = body.position + body.rotation.rotate(v);
                                to_screen(world_p)
                            })
                            .collect();
                        drawer.draw_polygon(&verts_screen, 2.0, color);
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
                    Joint2d::Prismatic(_) => drawer.draw_line(pa, pb, 2.5, [0.8, 0.3, 1.0, 0.9]),
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

        // 3. Render contact manifolds and impulse vectors
        if draw_contacts {
            for (_, _, manifold) in world.manifolds() {
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

        // 4. Render Telemetry HUD Cards (Screen space)
        // Background HUD Card at top-left
        let hud_min = Vec2::new(16.0, 16.0);
        let hud_max = Vec2::new(480.0, 140.0);
        drawer.batch.push_quad(
            hud_min,
            hud_max,
            Vec2::ZERO,
            Vec2::ONE,
            [0.08, 0.1, 0.13, 0.88],
        );

        // Accent top border
        drawer.draw_line(
            Vec2::new(16.0, 16.0),
            Vec2::new(480.0, 16.0),
            3.0,
            [0.2, 0.85, 0.5, 1.0],
        );

        // Legend backdrop at bottom-left
        let leg_min = Vec2::new(16.0, win_h - 48.0);
        let leg_max = Vec2::new(680.0, win_h - 16.0);
        drawer.batch.push_quad(
            leg_min,
            leg_max,
            Vec2::ZERO,
            Vec2::ONE,
            [0.08, 0.1, 0.13, 0.85],
        );
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

        let dt = if delta_time > 0.0 {
            delta_time.min(0.05)
        } else {
            1.0 / 60.0
        };

        if !self.paused || self.step_once {
            self.step_simulation(dt);
            self.step_once = false;
        }

        self.render_debug();
    }

    fn on_event(&mut self, _window: &mut Window, event: &WindowEvent) {
        match event {
            WindowEvent::Resized { width, height } => {
                self.window_size = (*width as f32, *height as f32);
            }
            WindowEvent::CursorMoved { x, y } => {
                let current_screen = Vec2::new(*x as f32, *y as f32);
                let delta = current_screen - self.last_mouse_screen;

                if self.is_right_dragging {
                    // Pan camera in world space
                    self.camera_offset.x -= delta.x / self.camera_zoom;
                    self.camera_offset.y += delta.y / self.camera_zoom;
                }

                self.screen_mouse_pos = current_screen;
                self.last_mouse_screen = current_screen;

                let world_pos = self.screen_to_world(current_screen);
                self.pointer_move(world_pos);
            }
            WindowEvent::MouseInput { button, pressed } => {
                let world_pos = self.screen_to_world(self.screen_mouse_pos);
                if *button == 0 {
                    // Left click
                    if *pressed {
                        self.pointer_down(world_pos);
                    } else {
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
        // Map common winit key codes
        use winit::keyboard::KeyCode;

        if code == KeyCode::Digit1 as u32 {
            self.load_gallery(0);
        } else if code == KeyCode::Digit2 as u32 {
            self.load_gallery(1);
        } else if code == KeyCode::Digit3 as u32 {
            self.load_gallery(2);
        } else if code == KeyCode::Digit4 as u32 {
            self.load_gallery(3);
        } else if code == KeyCode::Digit5 as u32 {
            self.load_gallery(4);
        } else if code == KeyCode::Space as u32 {
            self.toggle_pause();
        } else if code == KeyCode::KeyT as u32 {
            self.step_single_tick();
        } else if code == KeyCode::KeyR as u32 {
            self.reset_current_gallery();
        } else if code == KeyCode::KeyC as u32 {
            self.toggle_contacts();
        } else if code == KeyCode::KeyJ as u32 {
            self.toggle_joints();
        } else if code == KeyCode::KeyS as u32 {
            self.toggle_sleep_tint();
        } else if code == KeyCode::KeyN as u32 || code == KeyCode::ArrowRight as u32 {
            self.next_gallery();
        } else if code == KeyCode::KeyP as u32 || code == KeyCode::ArrowLeft as u32 {
            self.prev_gallery();
        }
    }
}
