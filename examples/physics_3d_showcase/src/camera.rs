//! 3D camera with orbital rotation, panning, zoom, and perspective projection.

use korppi_math::{Mat4, Vec3, Vec4};

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

    /// Orbits camera view angles in response to mouse drag delta.
    pub fn orbit_drag(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.005;
        self.yaw -= delta_x * sensitivity;
        self.pitch = (self.pitch + delta_y * sensitivity).clamp(-1.45, 1.45);
        self.update_orbit_eye();
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
        self.update_orbit_eye();
    }
}
