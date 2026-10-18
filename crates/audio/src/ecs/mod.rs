//! ECS components for spatial audio listeners and global audio configuration resources.

use crate::spatial::SpatializerMode;
use korppi_ecs::Component;
use serde::{Deserialize, Serialize};

/// Audio listener component (ears / camera microphone in the virtual world).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioListener {
    /// Whether this is the primary active listener in the scene.
    pub is_primary: bool,
    /// Listener master volume multiplier (0.0 to 1.0+).
    pub volume: f32,
    /// Whether this listener is currently active and processing audio.
    pub active: bool,
}

impl Component for AudioListener {}

impl Default for AudioListener {
    fn default() -> Self {
        Self {
            is_primary: true,
            volume: 1.0,
            active: true,
        }
    }
}

/// Global audio engine configuration resource.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioSettings {
    /// Global master volume multiplier.
    pub master_volume: f32,
    /// Spatialization algorithm mode (`Standard` by default, `Raytraced` opt-in).
    pub spatializer_mode: SpatializerMode,
    /// Maximum concurrent polyphonic voice playback limit.
    pub max_voices: usize,
    /// Number of stochastic rays cast in raytraced mode.
    pub ray_count: usize,
    /// Maximum reflection bounce depth for raytraced acoustics.
    pub max_bounces: usize,
    /// The room standard mode's spatialized players reverberate in (Freeverb on the "Reverb"
    /// bus; FIX-7.5).
    pub room_reverb: RoomReverbSettings,
}

/// The room reverb of standard mode: Freeverb's size and damping (0 to 1), and the level of the
/// "Reverb" bus it plays on: 0, off, until a game sets a room. (Raytraced mode's reverb plays at
/// full level: the spatializer's sends carry the room.)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoomReverbSettings {
    /// The room's size: longer decays as it grows.
    pub size: f32,
    /// High-frequency damping.
    pub damping: f32,
    /// The reverb's level (linear); 0 silences it.
    pub wet: f32,
}

impl Default for RoomReverbSettings {
    fn default() -> Self {
        Self {
            size: 0.5,
            damping: 0.5,
            wet: 0.0,
        }
    }
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            spatializer_mode: SpatializerMode::default(),
            max_voices: 32,
            ray_count: 128,
            max_bounces: 4,
            room_reverb: RoomReverbSettings::default(),
        }
    }
}

pub use crate::raytrace::material::AcousticMaterial as AcousticMaterialComponent;
