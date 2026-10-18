//! Mouse buttons, position, scroll wheel, and raw motion tracking.

use korppi_math::Vec2;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Identifiers for mouse buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MouseButton {
    /// Primary / Left mouse button.
    Left,
    /// Secondary / Right mouse button.
    Right,
    /// Auxiliary / Middle mouse button (wheel click).
    Middle,
    /// Extra buttons (e.g. thumb buttons).
    Other(u16),
}

/// Tracks cursor position, raw delta, wheel scrolling, and mouse button states.
#[derive(Debug, Clone, Default)]
pub struct MouseState {
    position: Vec2,
    delta: Vec2,
    wheel_delta: Vec2,
    pressed: HashSet<MouseButton>,
    just_pressed: HashSet<MouseButton>,
    just_released: HashSet<MouseButton>,
}

impl MouseState {
    /// Creates a new `MouseState`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the current client area cursor position in points.
    #[must_use]
    pub fn position(&self) -> Vec2 {
        self.position
    }

    /// Returns the cursor motion delta accumulated during the current frame.
    #[must_use]
    pub fn delta(&self) -> Vec2 {
        self.delta
    }

    /// Returns the scroll wheel delta accumulated during the current frame.
    #[must_use]
    pub fn wheel_delta(&self) -> Vec2 {
        self.wheel_delta
    }

    /// Returns `true` if the mouse button is currently held down.
    #[must_use]
    pub fn is_pressed(&self, button: MouseButton) -> bool {
        self.pressed.contains(&button)
    }

    /// Returns `true` if the mouse button was pressed during this frame.
    #[must_use]
    pub fn is_just_pressed(&self, button: MouseButton) -> bool {
        self.just_pressed.contains(&button)
    }

    /// Returns `true` if the mouse button was released during this frame.
    #[must_use]
    pub fn is_just_released(&self, button: MouseButton) -> bool {
        self.just_released.contains(&button)
    }

    /// Updates the cursor position and accumulates motion delta.
    pub fn on_moved(&mut self, new_pos: Vec2) {
        self.delta += new_pos - self.position;
        self.position = new_pos;
    }

    /// Directly accumulates raw relative mouse delta (useful for FPS cameras and locked cursors).
    pub fn on_raw_motion(&mut self, delta: Vec2) {
        self.delta += delta;
    }

    /// Accumulates scroll wheel movement.
    pub fn on_wheel(&mut self, wheel: Vec2) {
        self.wheel_delta += wheel;
    }

    /// Registers a mouse button press event.
    pub fn on_button_pressed(&mut self, button: MouseButton) {
        if self.pressed.insert(button) {
            self.just_pressed.insert(button);
        }
    }

    /// Registers a mouse button release event.
    pub fn on_button_released(&mut self, button: MouseButton) {
        if self.pressed.remove(&button) {
            self.just_released.insert(button);
        }
    }

    /// Resets per-frame motion and button transitions at frame boundaries.
    pub fn tick(&mut self) {
        self.delta = Vec2::ZERO;
        self.wheel_delta = Vec2::ZERO;
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Resets all mouse state.
    pub fn reset(&mut self) {
        self.position = Vec2::ZERO;
        self.delta = Vec2::ZERO;
        self.wheel_delta = Vec2::ZERO;
        self.pressed.clear();
        self.just_pressed.clear();
        self.just_released.clear();
    }
}
