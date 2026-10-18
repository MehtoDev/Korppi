//! Audio engine error definitions.

use std::fmt;

/// Result type alias for audio operations.
pub type AudioResult<T> = Result<T, AudioError>;

/// Errors that can occur within the audio subsystem.
#[derive(Debug)]
pub enum AudioError {
    /// Audio device initialization failure.
    DeviceInitialization(String),

    /// Output stream creation failure.
    StreamCreation(String),

    /// Invalid audio buffer configuration.
    InvalidConfiguration(String),

    /// Ring buffer queue is full (would block real-time thread).
    QueueFull,

    /// Every voice slot is in use by a voice of higher priority than the one to start.
    VoicePoolFull,

    /// Ring buffer queue is empty.
    QueueEmpty,

    /// Missing or invalid audio asset.
    AssetError(String),

    /// General audio backend error.
    BackendError(String),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceInitialization(msg) => {
                write!(f, "Audio device initialization failed: {msg}")
            }
            Self::StreamCreation(msg) => write!(f, "Output stream creation failed: {msg}"),
            Self::InvalidConfiguration(msg) => {
                write!(f, "Invalid audio buffer configuration: {msg}")
            }
            Self::QueueFull => write!(f, "Audio ring buffer is full"),
            Self::VoicePoolFull => {
                write!(
                    f,
                    "every voice slot is in use by a voice of higher priority"
                )
            }
            Self::QueueEmpty => write!(f, "Audio ring buffer is empty"),
            Self::AssetError(msg) => write!(f, "Audio asset error: {msg}"),
            Self::BackendError(msg) => write!(f, "Audio backend error: {msg}"),
        }
    }
}

impl std::error::Error for AudioError {}
