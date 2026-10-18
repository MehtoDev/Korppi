//! Central InputManager coordinating keyboard, mouse, gamepad, and action maps.

use crate::action::ActionMap;
use crate::gamepad::GamepadState;
use crate::keyboard::KeyboardState;
use crate::mouse::MouseState;
use korppi_math::Vec2;

/// Unified input coordinator managing all hardware devices and action mappings.
#[derive(Debug, Clone, Default)]
pub struct InputManager {
    /// Keyboard input state.
    pub keyboard: KeyboardState,
    /// Mouse input state.
    pub mouse: MouseState,
    /// Gamepad input state.
    pub gamepad: GamepadState,
    /// Action and axis bindings.
    pub actions: ActionMap,
}

impl InputManager {
    /// Creates a new, initialized `InputManager`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queries whether the named action is triggered by any bound device.
    #[must_use]
    pub fn is_active(&self, action: &str) -> bool {
        self.actions
            .is_active(action, &self.keyboard, &self.mouse, &self.gamepad)
    }

    /// Queries a 1D scalar axis value for the named action.
    #[must_use]
    pub fn axis_1d(&self, action: &str) -> f32 {
        self.actions
            .axis_1d(action, &self.keyboard, &self.mouse, &self.gamepad)
    }

    /// Queries a 2D directional vector for the named action.
    #[must_use]
    pub fn axis_2d(&self, action: &str) -> Vec2 {
        self.actions
            .axis_2d(action, &self.keyboard, &self.mouse, &self.gamepad)
    }

    /// Advances frame tick across all hardware devices.
    pub fn tick(&mut self) {
        self.keyboard.tick();
        self.mouse.tick();
        self.gamepad.tick();
    }
}
