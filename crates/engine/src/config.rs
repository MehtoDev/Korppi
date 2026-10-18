//! Engine configuration parameters.

use crate::backend::GraphicsBackend;
use korppi_audio::output::AudioBackend;
use korppi_math::Vec2;

/// Configuration options for initializing the Korppi engine coordinator.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Title of the game / application.
    pub app_name: String,
    /// Simulation fixed update rate in Hertz (default: 60 Hz).
    pub tick_rate: u32,
    /// Maximum fixed updates per frame before clamp (prevents spiral of death).
    pub max_sub_steps: u32,
    /// Initial graphics backend.
    pub backend: GraphicsBackend,
    /// Viewport width and height.
    pub window_size: Vec2,
    /// Target frame rate cap (0 for uncapped).
    pub target_fps: Option<u32>,
    /// Where audio goes: headless by default; [`AudioBackend::Device`] opens the default output
    /// device, falling back to headless output where there is none.
    pub audio: AudioBackend,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            app_name: "Korppi Engine Application".to_string(),
            tick_rate: 60,
            max_sub_steps: 8,
            backend: GraphicsBackend::Headless,
            window_size: Vec2::new(1280.0, 720.0),
            target_fps: None,
            audio: AudioBackend::Headless,
        }
    }
}
