//! 3D camera with orbital rotation, panning, zoom, free-flight mode, and perspective projection.

use korppi_math::{Mat4, Ray, Vec3, Vec4};

/// Camera movement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMode {
    /// Rotates around a focus target point.
    Orbit,
    /// First-person free flight camera.
    Fly,
}

/// 3D View and Projection Camera.
pub struct Camera3d {
    /// Active mode.
    pub mode: CameraMode,
    /// Eye position in world space.
    pub eye: Vec3,
    /// Orbital focal target in world space.
    pub target: Vec3,
    /// Azimuth / Yaw angle in radians.
    pub yaw: f32,
    /// Elevation / Pitch angle in radians.
    pub pitch: f32,
    /// Distance from target in Orbit mode.
    pub distance: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near clip distance.
    pub near: f32,
    /// Far clip distance.
    pub far: f32,
    /// Fly mode movement speed.
    pub move_speed: f32,
}

impl Default for Camera3d {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera3d {
    /// Creates a camera positioned at an elevation overlooking origin.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: CameraMode::Orbit,
            eye: Vec3::new(0.0, 10.0, 22.0),
            target: Vec3::new(0.0, 2.0, 0.0),
            yaw: 0.0,
            pitch: 25.0f32.to_radians(),
            distance: 22.0,
            fov_y: 50.0f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 500.0,
            move_speed: 12.0,
        }
    }

    /// Returns the unit forward look direction implied by `yaw` and `pitch`.
    #[must_use]
    pub fn forward_vector(&self) -> Vec3 {
        let cos_p = self.pitch.cos();
        let sin_p = self.pitch.sin();
        let sin_y = self.yaw.sin();
        let cos_y = self.yaw.cos();
        Vec3::new(-cos_p * sin_y, -sin_p, -cos_p * cos_y).normalize_or_zero()
    }

    /// Computes the camera eye position based on orbit angles and distance.
    pub fn update_orbit_eye(&mut self) {
        if self.mode == CameraMode::Orbit {
            let cos_p = self.pitch.cos();
            let sin_p = self.pitch.sin();
            let sin_y = self.yaw.sin();
            let cos_y = self.yaw.cos();

            let offset = Vec3::new(
                self.distance * cos_p * sin_y,
                self.distance * sin_p,
                self.distance * cos_p * cos_y,
            );
            self.eye = self.target + offset;
        }
    }

    /// Toggles between [`CameraMode::Orbit`] and [`CameraMode::Fly`].
    pub fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            CameraMode::Orbit => CameraMode::Fly,
            CameraMode::Fly => {
                self.update_orbit_eye();
                CameraMode::Orbit
            }
        };
    }

    /// Translates the camera along its local forward, right, and world-up axes in [`CameraMode::Fly`].
    pub fn fly_move(&mut self, forward_input: f32, right_input: f32, up_input: f32, dt: f32) {
        let forward = (self.target - self.eye).normalize_or_zero();
        let (right, _) = self.right_and_up_vectors();
        let velocity = (forward * forward_input + right * right_input + Vec3::Y * up_input)
            * self.move_speed
            * dt;
        self.eye += velocity;
        self.target += velocity;
    }

    /// Computes the right and up unit vectors perpendicular to the camera look direction.
    pub fn right_and_up_vectors(&self) -> (Vec3, Vec3) {
        let forward = (self.target - self.eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        (right, up)
    }

    /// Returns the 4x4 view transform matrix.
    #[must_use]
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, Vec3::Y)
    }

    /// Returns the 4x4 perspective projection matrix.
    #[must_use]
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Transforms a 3D world space coordinate into 2D screen coordinates (pixels).
    ///
    /// Returns `None` if the point is behind the camera or outside clip boundaries.
    pub fn project_to_screen(&self, point: Vec3, screen_size: (f32, f32)) -> Option<(f32, f32)> {
        let view_proj = self.projection_matrix() * self.view_matrix();
        let clip = view_proj * Vec4::new(point.x, point.y, point.z, 1.0);

        if clip.w <= 0.05 {
            return None;
        }

        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let ndc_z = clip.z / clip.w;

        if !(0.0..=1.0).contains(&ndc_z) {
            return None;
        }

        let screen_x = (ndc_x + 1.0) * 0.5 * screen_size.0;
        let screen_y = (1.0 - ndc_y) * 0.5 * screen_size.1;
        Some((screen_x, screen_y))
    }

    /// Clips a 3D world-space line segment `[p1, p2]` against the camera near and far planes
    /// before projecting both endpoints to 2D screen coordinates.
    ///
    /// Returns `None` only when the entire segment lies behind the near plane or beyond the far plane.
    #[must_use]
    pub fn clip_and_project_line(
        &self,
        p1: Vec3,
        p2: Vec3,
        screen_size: (f32, f32),
    ) -> Option<((f32, f32), (f32, f32))> {
        let view = self.view_matrix();
        let v1_4 = view * Vec4::new(p1.x, p1.y, p1.z, 1.0);
        let v2_4 = view * Vec4::new(p2.x, p2.y, p2.z, 1.0);
        let mut c1 = Vec3::new(v1_4.x, v1_4.y, v1_4.z);
        let mut c2 = Vec3::new(v2_4.x, v2_4.y, v2_4.z);

        // Right-handed view space: visible points satisfy -far <= z <= -near.
        let near_z = -self.near.max(0.05);
        let far_z = -self.far;

        if c1.z > near_z && c2.z > near_z {
            return None;
        }
        if c1.z < far_z && c2.z < far_z {
            return None;
        }

        if c1.z > near_z {
            let dz = c2.z - c1.z;
            if dz.abs() < 1e-6 {
                return None;
            }
            let t = (near_z - c1.z) / dz;
            c1 += (c2 - c1) * t;
        } else if c2.z > near_z {
            let dz = c2.z - c1.z;
            if dz.abs() < 1e-6 {
                return None;
            }
            let t = (near_z - c1.z) / dz;
            c2 = c1 + (c2 - c1) * t;
        }

        if c1.z < far_z && c2.z < far_z {
            return None;
        }
        if c1.z < far_z {
            let dz = c2.z - c1.z;
            if dz.abs() < 1e-6 {
                return None;
            }
            let t = (far_z - c1.z) / dz;
            c1 += (c2 - c1) * t;
        } else if c2.z < far_z {
            let dz = c2.z - c1.z;
            if dz.abs() < 1e-6 {
                return None;
            }
            let t = (far_z - c1.z) / dz;
            c2 = c1 + (c2 - c1) * t;
        }

        let proj = self.projection_matrix();
        let clip1 = proj * Vec4::new(c1.x, c1.y, c1.z, 1.0);
        let clip2 = proj * Vec4::new(c2.x, c2.y, c2.z, 1.0);
        if clip1.w <= 1e-5 || clip2.w <= 1e-5 {
            return None;
        }

        let s1 = (
            ((clip1.x / clip1.w) + 1.0) * 0.5 * screen_size.0,
            (1.0 - (clip1.y / clip1.w)) * 0.5 * screen_size.1,
        );
        let s2 = (
            ((clip2.x / clip2.w) + 1.0) * 0.5 * screen_size.0,
            (1.0 - (clip2.y / clip2.w)) * 0.5 * screen_size.1,
        );
        if !s1.0.is_finite() || !s1.1.is_finite() || !s2.0.is_finite() || !s2.1.is_finite() {
            return None;
        }
        Some((s1, s2))
    }

    /// Unprojects a 2D screen coordinate (pixels, top-left origin) into a world-space [`Ray`].
    #[must_use]
    pub fn screen_to_ray(&self, screen_pos: (f32, f32), screen_size: (f32, f32)) -> Ray {
        let w = screen_size.0.max(1.0);
        let h = screen_size.1.max(1.0);
        let ndc_x = (screen_pos.0 / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_pos.1 / h) * 2.0;

        let tan_half_fov = (self.fov_y * 0.5).tan();
        let forward = (self.target - self.eye).normalize_or_zero();
        let (right, up) = self.right_and_up_vectors();
        let dir =
            (forward + right * (ndc_x * self.aspect * tan_half_fov) + up * (ndc_y * tan_half_fov))
                .normalize_or_zero();
        Ray::new(self.eye, dir)
    }

    /// Orbits or rotates camera view angles in response to mouse drag delta.
    pub fn orbit_drag(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.005;
        self.yaw -= delta_x * sensitivity;
        self.pitch = (self.pitch + delta_y * sensitivity).clamp(-1.45, 1.45);
        match self.mode {
            CameraMode::Orbit => self.update_orbit_eye(),
            CameraMode::Fly => {
                self.target = self.eye + self.forward_vector() * self.distance;
            }
        }
    }

    /// Pans camera focal target along view right and up vectors.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let (right, up) = self.right_and_up_vectors();
        let factor = self.distance * 0.0015;
        let shift = -right * (delta_x * factor) + up * (delta_y * factor);
        self.target += shift;
        self.eye += shift;
    }

    /// Zooms camera in or out by scaling distance.
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(2.0, 150.0);
        if self.mode == CameraMode::Orbit {
            self.update_orbit_eye();
        }
    }
}
