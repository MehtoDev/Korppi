//! Window event definitions decoupled from OS window backends.

use serde::{Deserialize, Serialize};

/// High-level window events dispatched from OS event loop to engine systems.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WindowEvent {
    /// Window resized to new logical dimensions.
    Resized {
        /// Logical width in points.
        width: u32,
        /// Logical height in points.
        height: u32,
    },

    /// The display scale factor (DPI) changed.
    ScaleFactorChanged {
        /// New monitor scale factor.
        scale_factor: f64,
    },

    /// Window gained or lost focus.
    Focused(bool),

    /// Window moved to a new screen coordinate.
    Moved {
        /// Screen X coordinate.
        x: i32,
        /// Screen Y coordinate.
        y: i32,
    },

    /// The user requested the window to close (e.g. clicked the X button or Alt+F4).
    CloseRequested,

    /// OS requests window surface redraw.
    RedrawRequested,

    /// Cursor moved within the window.
    CursorMoved {
        /// X position relative to top-left of window client area.
        x: f64,
        /// Y position relative to top-left of window client area.
        y: f64,
    },

    /// Cursor entered the window client area.
    CursorEntered,

    /// Cursor left the window client area.
    CursorLeft,

    /// Scroll wheel moved.
    MouseWheel {
        /// Horizontal scroll delta.
        delta_x: f32,
        /// Vertical scroll delta.
        delta_y: f32,
    },

    /// Mouse button pressed or released.
    MouseInput {
        /// Numeric button ID (0 = Left, 1 = Right, 2 = Middle).
        button: u32,
        /// `true` if pressed, `false` if released.
        pressed: bool,
    },

    /// Keyboard key pressed or released.
    KeyboardInput {
        /// Scan code or key identifier.
        key_code: Option<u32>,
        /// `true` if pressed, `false` if released.
        pressed: bool,
    },
}
