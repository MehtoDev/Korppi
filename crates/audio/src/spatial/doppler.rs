//! Classical acoustic Doppler pitch shifting from listener and emitter velocities (AUD-12).

use korppi_math::Vec3;

/// Classical acoustic Doppler effect calculator (`(c + v_L) / (c + v_E)` along the line of sight).
#[derive(Debug, Clone, Copy)]
pub struct DopplerPitch {
    /// Speed of sound in meters per second (default 343.0 m/s).
    pub speed_of_sound: f32,
}

impl Default for DopplerPitch {
    fn default() -> Self {
        Self {
            speed_of_sound: 343.0,
        }
    }
}

impl DopplerPitch {
    /// Creates a new Doppler calculator with the specified speed of sound.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            speed_of_sound: 343.0,
        }
    }

    /// Calculates the observed Doppler frequency scale factor.
    #[must_use]
    pub fn calculate_pitch(
        &self,
        listener_pos: Vec3,
        listener_vel: Vec3,
        emitter_pos: Vec3,
        emitter_vel: Vec3,
    ) -> f32 {
        let delta = emitter_pos - listener_pos;
        let distance = delta.length();

        if distance < 1e-4 {
            return 1.0;
        }

        let direction = delta / distance;
        let c = self.speed_of_sound;
        let max_v = c * 0.9;

        // Relative radial velocities along the listener-to-emitter axis
        let v_l = listener_vel.dot(direction).clamp(-max_v, max_v);
        let v_e = emitter_vel.dot(direction).clamp(-max_v, max_v);

        let pitch = (c + v_l) / (c + v_e);
        pitch.clamp(0.2, 5.0)
    }
}
