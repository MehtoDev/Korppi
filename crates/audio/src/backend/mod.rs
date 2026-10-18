//! Platform-specific hardware audio backends.

#[cfg(feature = "cpal")]
pub mod cpal_backend;

#[cfg(feature = "cpal")]
pub use cpal_backend::CpalAudioDevice;
