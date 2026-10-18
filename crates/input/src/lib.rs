//! Input subsystem providing keyboard, mouse, gamepad state, and action bindings.

pub mod action;
pub mod gamepad;
pub mod keyboard;
pub mod manager;
pub mod mouse;

pub use action::{ActionBinding, ActionMap, InputAction};
pub use gamepad::{GamepadAxis, GamepadButton, GamepadId, GamepadState};
pub use keyboard::{KeyCode, KeyboardState};
pub use manager::InputManager;
pub use mouse::{MouseButton, MouseState};

/// Common input types re-exported for convenience.
pub mod prelude {
    pub use crate::action::{ActionBinding, ActionMap, InputAction};
    pub use crate::gamepad::{GamepadAxis, GamepadButton, GamepadId, GamepadState};
    pub use crate::keyboard::{KeyCode, KeyboardState};
    pub use crate::manager::InputManager;
    pub use crate::mouse::{MouseButton, MouseState};
}
