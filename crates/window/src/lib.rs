//! Cross-platform windowing, event loop management, and OS surface integration.

pub mod app;
pub mod config;
pub mod event;
pub mod window;

pub use app::{AppRunner, EngineApp, KorppiApp};
pub use config::{WindowConfig, WindowMode};
pub use event::WindowEvent;
pub use window::{Window, WindowError};
