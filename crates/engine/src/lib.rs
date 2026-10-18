//! Central Korppi Engine Coordinator.
//!
//! Unifies all subsystems (ECS, Dual Physics, Scene Graph, Audio, Input, RHI, Render, UI)
//! into a deterministic, high-performance simulation and rendering runtime.

#![warn(missing_docs)]

pub mod app;
pub mod backend;
pub mod config;
pub mod metrics;
pub mod presenter;

pub use app::EngineApp;
pub use backend::GraphicsBackend;
pub use config::EngineConfig;
pub use metrics::DiagnosticMetrics;
pub use presenter::{
    run_windowed, PresentOptions, QuadBatchPresentation, QuadBatchSource, WindowPresenter,
};
