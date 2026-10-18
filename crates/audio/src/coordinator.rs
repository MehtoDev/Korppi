//! Audio engine coordinator tracking spatialization mode and ECS configuration.

use crate::ecs::AudioSettings;
use crate::spatial::SpatializerMode;
use korppi_ecs::World;

/// Audio subsystem coordinator tracking the active spatialization mode from [`AudioSettings`].
///
/// Voices, mixing, reverb, and raytraced room acoustics run through [`crate::player::AudioPlayer`],
/// [`crate::player::update_audio_players`], and [`crate::mixer::AudioMixer`] (ADR-050, ADR-051).
#[derive(Debug, Clone)]
pub struct AudioCoordinator {
    sample_rate: f32,
    mode: SpatializerMode,
}

impl AudioCoordinator {
    /// Creates a new audio coordinator initialized for the target sampling rate.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000.0),
            mode: SpatializerMode::Standard,
        }
    }

    /// Returns the configured sampling rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the currently active spatialization mode (`Standard` or `Raytraced`).
    #[must_use]
    pub fn mode(&self) -> SpatializerMode {
        self.mode
    }

    /// Sets the active spatialization mode at runtime.
    pub fn set_mode(&mut self, mode: SpatializerMode) {
        self.mode = mode;
    }

    /// Synchronizes the spatialization mode from the [`AudioSettings`] resource in `world`.
    pub fn update_ecs(&mut self, world: &mut World) {
        if let Some(settings) = world.resource::<AudioSettings>() {
            self.set_mode(settings.spatializer_mode);
        }
    }

    /// Resets the coordinator state to defaults.
    pub fn reset(&mut self) {
        self.mode = SpatializerMode::Standard;
    }
}
