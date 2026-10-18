//! Window configuration settings and display mode options.

use serde::{Deserialize, Serialize};

/// Display mode for the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WindowMode {
    /// Standard floating window with OS decorations.
    #[default]
    Windowed,
    /// Exclusive fullscreen mode occupying the entire monitor.
    Fullscreen,
    /// Borderless window covering the entire screen resolution.
    BorderlessFullscreen,
}

/// Configuration settings used to create or adjust a window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Window title displayed in the OS title bar.
    pub title: String,
    /// Logical width of the window client area in points.
    pub width: u32,
    /// Logical height of the window client area in points.
    pub height: u32,
    /// Whether the user can resize the window.
    pub resizable: bool,
    /// Whether vertical synchronization (VSync) is enabled.
    pub vsync: bool,
    /// The window display mode (Windowed, Fullscreen, etc.).
    pub mode: WindowMode,
    /// Whether the OS mouse cursor is visible inside the window.
    pub cursor_visible: bool,
    /// Whether the cursor is locked/grabbed within the window bounds.
    pub cursor_grabbed: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Rust Game Engine".to_string(),
            width: 1280,
            height: 720,
            resizable: true,
            vsync: true,
            mode: WindowMode::Windowed,
            cursor_visible: true,
            cursor_grabbed: false,
        }
    }
}

impl WindowConfig {
    /// Sets the window title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Sets the logical dimensions (width and height) in points.
    #[must_use]
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets whether the window is user-resizable.
    #[must_use]
    pub fn with_resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }

    /// Sets whether VSync is requested.
    #[must_use]
    pub fn with_vsync(mut self, vsync: bool) -> Self {
        self.vsync = vsync;
        self
    }

    /// Sets the window display mode.
    #[must_use]
    pub fn with_mode(mut self, mode: WindowMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets cursor visibility.
    #[must_use]
    pub fn with_cursor_visible(mut self, visible: bool) -> Self {
        self.cursor_visible = visible;
        self
    }

    /// Sets cursor grab/lock state.
    #[must_use]
    pub fn with_cursor_grabbed(mut self, grabbed: bool) -> Self {
        self.cursor_grabbed = grabbed;
        self
    }
}
