//! Hardware graphics backend selector.

use serde::{Deserialize, Serialize};

/// Available Render Hardware Interface (RHI) backend implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GraphicsBackend {
    /// Headless backend for deterministic simulations, testing, and AI agent verification.
    #[default]
    Headless,
    /// Modern OpenGL 4.5 Core Direct State Access (DSA) backend.
    OpenGl,
    /// Vulkan 1.3 Core backend with dynamic rendering and synchronization2.
    Vulkan,
}
