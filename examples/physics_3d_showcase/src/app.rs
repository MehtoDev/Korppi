//! 3D Physics Showcase application and scene controller.

use crate::camera::Camera3d;
use crate::gallery::{
    FrictionRestitutionGallery, Gallery3d, HeavyStackingGallery, Ragdoll3dGallery,
    RollbackDeterminismGallery, SoftStepTuningGallery, SpatialQueriesGallery,
};
use korppi_math::{Aabb, Ray, Vec2, Vec3};
use korppi_physics::body::BodyType;
use korppi_physics::shape::Shape;
use korppi_physics::snapshot::WorldSnapshot3d;
use korppi_physics::PhysicsWorld;
use korppi_render::debug3d::DebugDraw3d;
use korppi_render::render2d::batch::QuadBatch2d;
use korppi_window::{KorppiApp, Window, WindowEvent};
use std::collections::VecDeque;
use std::time::Instant;

/// Application orchestrating 3D physics demonstration scenes, orbital camera, and debug drawing.
pub struct Physics3dShowcaseApp {
    /// Active 3D physics simulation world.
    pub world: PhysicsWorld,
    /// 3D Camera with perspective projection.
    pub camera: Camera3d,
    galleries: Vec<Box<dyn Gallery3d>>,
    current_gallery: usize,
    paused: bool,
    step_once: bool,
    draw_wireframes: bool,
    draw_contacts: bool,
    draw_bvh_aabbs: bool,
    draw_sleep_tint: bool,
    snapshots: VecDeque<WorldSnapshot3d>,
    debug_drawer: DebugDraw3d,
    quad_batch: QuadBatch2d,
    last_sim_time_ms: f32,
    window_size: (f32, f32),
    last_mouse_pos: (f32, f32),
    is_left_down: bool,
    is_right_down: bool,
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

        let mut app = Self {
            world: PhysicsWorld::new(),
            camera: Camera3d::new(),
            galleries,
            current_gallery: 0,
            paused: false,
            step_once: false,
            draw_wireframes: true,
            draw_contacts: true,
            draw_bvh_aabbs: false,
            draw_sleep_tint: true,
            snapshots: VecDeque::with_capacity(128),
            debug_drawer: DebugDraw3d::new(),
            quad_batch: QuadBatch2d::new(16384),
            last_sim_time_ms: 0.0,
            window_size: (1280.0, 720.0),
            last_mouse_pos: (0.0, 0.0),
            is_left_down: false,
            is_right_down: false,
        };

        app.camera.update_orbit_eye();
        app.load_gallery(0);
        app
    }

    /// Loads the gallery at `index` and initializes simulation bodies.
    pub fn load_gallery(&mut self, index: usize) {
        if index < self.galleries.len() {
            self.current_gallery = index;
            self.snapshots.clear();
            self.world = PhysicsWorld::new();
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

    /// Advances the simulation by `dt` seconds and records a state snapshot in the ring buffer.
    pub fn step_simulation(&mut self, dt: f32) {
        // Record state snapshot before step
        if self.snapshots.len() >= 120 {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(self.world.create_snapshot());

        let start = Instant::now();
        self.world.step(dt);
        self.last_sim_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    }

    /// Compiles 3D wireframe geometry into `DebugDraw3d` and projects onto `QuadBatch2d`.
    pub fn render_debug(&mut self) {
        self.debug_drawer.clear();
        self.quad_batch.clear();

        // 1. Draw 3D Ground Grid on XZ plane
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

        // 2. Draw all bodies and collision shapes
        if self.draw_wireframes {
            for id in 0..self.world.body_count() as u32 {
                let Some(body) = self.world.body(id) else {
                    continue;
                };

                let color = if body.is_sleeping && self.draw_sleep_tint {
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

        // 3. Draw contact normals and points
        if self.draw_contacts {
            for manifold in self.world.manifolds() {
                for pt in &manifold.points {
                    self.debug_drawer
                        .draw_cross(pt.point, 0.2, [1.0, 1.0, 0.0, 1.0]);
                    self.debug_drawer
                        .draw_normal(pt.point, pt.normal, 0.5, [1.0, 0.25, 0.25, 0.9]);
                }
            }
        }

        // 4. Special Visualization for Gallery 5 (Spatial Queries)
        if self.current_gallery == 4 {
            // Test continuous ray cast along camera look vector
            let ray_origin = self.camera.eye;
            let ray_dir = (self.camera.target - self.camera.eye).normalize_or_zero();
            let test_ray = Ray::new(ray_origin, ray_dir);

            if let Some(hit) = self.world.ray_cast(&test_ray, 100.0) {
                // Highlight hit ray
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

            // Volume query AABB visualization
            let query_aabb = Aabb {
                min: Vec3::new(-4.0, 0.0, -4.0),
                max: Vec3::new(4.0, 4.0, 4.0),
            };
            self.debug_drawer
                .draw_aabb(query_aabb.min, query_aabb.max, [0.2, 0.7, 1.0, 0.7]);
        }

        // 5. Project all 3D line primitives into QuadBatch2d using Camera3d perspective matrix
        let screen_size = self.window_size;
        let line_verts = self.debug_drawer.vertices();
        let line_indices = self.debug_drawer.indices();

        for &[i1, i2] in line_indices.as_chunks::<2>().0 {
            let v1 = line_verts[i1 as usize];
            let v2 = line_verts[i2 as usize];

            let p1 = Vec3::new(v1.position[0], v1.position[1], v1.position[2]);
            let p2 = Vec3::new(v2.position[0], v2.position[1], v2.position[2]);

            if let (Some(s1), Some(s2)) = (
                self.camera.project_to_screen(p1, screen_size),
                self.camera.project_to_screen(p2, screen_size),
            ) {
                self.quad_batch.push_line(
                    Vec2::new(s1.0, s1.1),
                    Vec2::new(s2.0, s2.1),
                    1.5,
                    v1.color,
                );
            }
        }

        // 6. Draw HUD Telemetry backdrop cards
        // Top-left diagnostic panel
        self.quad_batch.push_quad(
            Vec2::new(16.0, 16.0),
            Vec2::new(480.0, 140.0),
            Vec2::ZERO,
            Vec2::ONE,
            [0.08, 0.1, 0.13, 0.88],
        );
        self.quad_batch.push_line(
            Vec2::new(16.0, 16.0),
            Vec2::new(480.0, 16.0),
            3.0,
            [0.2, 0.85, 0.5, 1.0],
        );

        // Bottom-left controls legend
        self.quad_batch.push_quad(
            Vec2::new(16.0, screen_size.1 - 48.0),
            Vec2::new(720.0, screen_size.1 - 16.0),
            Vec2::ZERO,
            Vec2::ONE,
            [0.08, 0.1, 0.13, 0.85],
        );
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
                self.camera.aspect = self.window_size.0 / self.window_size.1.max(1.0);
            }
            WindowEvent::CursorMoved { x, y } => {
                let current_mouse = (*x as f32, *y as f32);
                let delta_x = current_mouse.0 - self.last_mouse_pos.0;
                let delta_y = current_mouse.1 - self.last_mouse_pos.1;

                if self.is_left_down {
                    // Orbit camera
                    self.camera.orbit_drag(delta_x, delta_y);
                } else if self.is_right_down {
                    // Pan camera
                    self.camera.pan(delta_x, delta_y);
                }

                self.last_mouse_pos = current_mouse;
            }
            WindowEvent::MouseInput { button, pressed } => {
                if *button == 0 {
                    self.is_left_down = *pressed;
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
        } else if code == KeyCode::Digit6 as u32 {
            self.load_gallery(5);
        } else if code == KeyCode::Space as u32 {
            self.toggle_pause();
        } else if code == KeyCode::KeyT as u32 {
            self.step_single_tick();
        } else if code == KeyCode::KeyR as u32 {
            self.reset_current_gallery();
        } else if code == KeyCode::KeyW as u32 {
            self.toggle_wireframes();
        } else if code == KeyCode::KeyC as u32 {
            self.toggle_contacts();
        } else if code == KeyCode::KeyB as u32 {
            self.toggle_bvh_aabbs();
        } else if code == KeyCode::KeyS as u32 {
            self.toggle_sleep_tint();
        } else if code == KeyCode::KeyZ as u32 {
            self.rollback(60);
        } else if code == KeyCode::KeyN as u32 || code == KeyCode::ArrowRight as u32 {
            self.next_gallery();
        } else if code == KeyCode::KeyP as u32 || code == KeyCode::ArrowLeft as u32 {
            self.prev_gallery();
        }
    }
}
