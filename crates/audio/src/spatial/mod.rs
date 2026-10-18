//! 3D spatial audio interfaces, psychoacoustic models, and panning algorithms.

pub mod attenuation;
pub mod binaural;
pub mod directivity;
pub mod doppler;
pub mod panning;
pub mod raytraced;
pub mod standard;

pub use attenuation::{AttenuationModel, DistanceAttenuation};
pub use binaural::WoodworthBinaural;
pub use directivity::EmitterCone;
pub use doppler::DopplerPitch;
pub use panning::EqualPowerPanner;
pub use raytraced::RaytracedSpatializer;
pub use standard::StandardSpatializer;

use korppi_math::Vec3;
use serde::{Deserialize, Serialize};

/// Global spatialization algorithm mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SpatializerMode {
    /// Standard lightweight mathematical spatial audio (Default).
    #[default]
    Standard,
    /// Physically simulated geometric raytraced acoustics.
    Raytraced,
}

/// Dynamic spatial state of an audio listener (camera / player ears).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ListenerState {
    /// World position of the listener in meters.
    pub position: Vec3,
    /// Forward facing direction unit vector: a transform's −Z axis (the coordinate contract).
    pub forward: Vec3,
    /// Up direction unit vector.
    pub up: Vec3,
    /// Linear velocity vector in meters per second.
    pub velocity: Vec3,
}

impl Default for ListenerState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            velocity: Vec3::ZERO,
        }
    }
}

/// Dynamic spatial state of an audio emitter (sound source in the world).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmitterState {
    /// World position of the emitter in meters.
    pub position: Vec3,
    /// Forward facing direction unit vector (for directional cones): a transform's −Z axis.
    pub forward: Vec3,
    /// Linear velocity vector in meters per second.
    pub velocity: Vec3,
    /// Distance attenuation model.
    pub attenuation: AttenuationModel,
    /// Optional directional emission cone.
    pub cone: Option<EmitterCone>,
    /// Doppler effect intensity multiplier (0.0 = disabled, 1.0 = physical).
    pub doppler_factor: f32,
}

impl Default for EmitterState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            velocity: Vec3::ZERO,
            attenuation: AttenuationModel::default(),
            cone: None,
            doppler_factor: 1.0,
        }
    }
}

/// Output parameters computed by a spatializer for a specific voice emitter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceSpatialParams {
    /// Left channel linear gain multiplier (0.0 to 1.0+).
    pub gain_left: f32,
    /// Right channel linear gain multiplier (0.0 to 1.0+).
    pub gain_right: f32,
    /// Left ear propagation delay in seconds (for binaural ITD).
    pub delay_left_sec: f32,
    /// Right ear propagation delay in seconds (for binaural ITD).
    pub delay_right_sec: f32,
    /// Pitch scale multiplier (combining native pitch and Doppler shift).
    pub pitch_scale: f32,
    /// Low-pass filter cutoff in Hz (air absorption, cone, and occlusion muffling).
    pub low_pass_cutoff: f32,
    /// The head's shadow on the left ear: a low-pass cutoff in Hz, 22 kHz unless the source is
    /// on the right.
    pub head_shadow_left: f32,
    /// The head's shadow on the right ear, likewise.
    pub head_shadow_right: f32,
    /// Reverb auxiliary send amount (0.0 to 1.0).
    pub reverb_send: f32,
}

impl Default for VoiceSpatialParams {
    fn default() -> Self {
        Self {
            gain_left: 1.0,
            gain_right: 1.0,
            delay_left_sec: 0.0,
            delay_right_sec: 0.0,
            pitch_scale: 1.0,
            low_pass_cutoff: 22_000.0,
            head_shadow_left: 22_000.0,
            head_shadow_right: 22_000.0,
            reverb_send: 0.1,
        }
    }
}

/// Abstract line-of-sight and geometric occlusion query interface.
pub trait AcousticRaycaster: Send + Sync {
    /// Casts an acoustic occlusion ray between `from` and `to`.
    ///
    /// Returns `Some(occlusion_fraction)` if blocked (e.g. 1.0 for fully blocked),
    /// or `None` if unobstructed.
    fn raycast_occlusion(&self, from: Vec3, to: Vec3) -> Option<f32>;
}

/// Global environmental room reverberation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReverbEnvironmentParams {
    /// Room size / decay time parameter (0.0 to 1.0).
    pub room_size: f32,
    /// High-frequency absorption / damping factor (0.0 to 1.0).
    pub damping: f32,
    /// Wet signal level multiplier.
    pub wet_mix: f32,
}

impl Default for ReverbEnvironmentParams {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            wet_mix: 0.3,
        }
    }
}

/// Abstract spatialization interface implemented by both standard and raytraced audio engines.
pub trait SpatialAudioModel: Send + Sync {
    /// Computes spatial gains, delay offsets, and filtering parameters for an active emitter.
    fn spatialize_emitter(
        &mut self,
        listener: &ListenerState,
        emitter: &EmitterState,
        occlusion_query: Option<&dyn AcousticRaycaster>,
        output_params: &mut VoiceSpatialParams,
    );

    /// Updates global room acoustic reverberation parameters.
    fn update_environment_acoustics(
        &mut self,
        listener: &ListenerState,
        raycaster: Option<&dyn AcousticRaycaster>,
        reverb_params: &mut ReverbEnvironmentParams,
    );
}
