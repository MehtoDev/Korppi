//! Mouse buttons, position, scroll wheel, and raw motion tracking.

use korppi_math::Vec2;
use serde::{Deserialize, Serialize};

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

impl MouseButton {
    /// Converts a raw `u32` button identifier (from `WindowEvent::MouseInput`) into a [`MouseButton`].
    #[inline]
    #[must_use]
    pub const fn from_u32(button: u32) -> Self {
        match button {
            0 => Self::Left,
            1 => Self::Right,
            2 => Self::Middle,
            n => Self::Other(if n > u16::MAX as u32 {
                u16::MAX
            } else {
                n as u16
            }),
        }
    }

    /// Converts this [`MouseButton`] into the numeric identifier used by `WindowEvent::MouseInput`.
    #[inline]
    #[must_use]
    pub const fn to_u32(self) -> u32 {
        match self {
            Self::Left => 0,
            Self::Right => 1,
            Self::Middle => 2,
            Self::Other(o) => o as u32,
        }
    }

    #[inline]
    const fn bit_mask(self) -> u64 {
        let bit = match self {
            Self::Left => 0,
            Self::Right => 1,
            Self::Middle => 2,
            Self::Other(o) => {
                let idx = if o < 3 { o + 3 } else { o };
                if idx >= 64 {
                    63
                } else {
                    idx
                }
            }
        };
        1u64 << bit
    }
}

/// Tracks cursor position, raw delta, wheel scrolling, and mouse button states
/// using fixed-size stack bitmasks (`0` heap allocations).
#[derive(Debug, Clone, Default)]
pub struct MouseState {
    position: Vec2,
    delta: Vec2,
    wheel_delta: Vec2,
    has_position: bool,
    pressed: u64,
    just_pressed: u64,
    just_released: u64,
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
    #[inline]
    #[must_use]
    pub fn is_pressed(&self, button: MouseButton) -> bool {
        (self.pressed & button.bit_mask()) != 0
    }

    /// Returns `true` if the mouse button was pressed during this frame.
    #[inline]
    #[must_use]
    pub fn is_just_pressed(&self, button: MouseButton) -> bool {
        (self.just_pressed & button.bit_mask()) != 0
    }

    /// Returns `true` if the mouse button was released during this frame.
    #[inline]
    #[must_use]
    pub fn is_just_released(&self, button: MouseButton) -> bool {
        (self.just_released & button.bit_mask()) != 0
    }

    /// Updates the cursor position and accumulates motion delta.
    ///
    /// On the first cursor observation (or the first observation after [`Self::on_cursor_left`]),
    /// records `new_pos` without adding `new_pos - (0, 0)` to `delta` (`INP-03`).
    pub fn on_moved(&mut self, new_pos: Vec2) {
        if !new_pos.x.is_finite() || !new_pos.y.is_finite() {
            return;
        }
        if self.has_position {
            self.delta += new_pos - self.position;
        } else {
            self.has_position = true;
        }
        self.position = new_pos;
    }

    /// Marks the cursor as having left the window so the next [`Self::on_moved`] event
    /// re-establishes position without a delta spike.
    pub fn on_cursor_left(&mut self) {
        self.has_position = false;
    }

    /// Directly accumulates raw relative mouse delta (useful for FPS cameras and locked cursors).
    pub fn on_raw_motion(&mut self, delta: Vec2) {
        if delta.x.is_finite() && delta.y.is_finite() {
            self.delta += delta;
        }
    }

    /// Accumulates scroll wheel movement.
    pub fn on_wheel(&mut self, wheel: Vec2) {
        if wheel.x.is_finite() && wheel.y.is_finite() {
            self.wheel_delta += wheel;
        }
    }

    /// Registers a mouse button press event.
    #[inline]
    pub fn on_button_pressed(&mut self, button: MouseButton) {
        let mask = button.bit_mask();
        if (self.pressed & mask) == 0 {
            self.pressed |= mask;
            self.just_pressed |= mask;
        }
    }

    /// Registers a mouse button release event.
    #[inline]
    pub fn on_button_released(&mut self, button: MouseButton) {
        let mask = button.bit_mask();
        if (self.pressed & mask) != 0 {
            self.pressed &= !mask;
            self.just_released |= mask;
        }
    }

    /// Releases all currently held mouse buttons and marks them as `just_released` (e.g. on focus loss).
    pub fn on_focus_lost(&mut self) {
        self.just_released |= self.pressed;
        self.pressed = 0;
    }

    /// Resets per-frame motion and button transitions at frame boundaries.
    pub fn tick(&mut self) {
        self.delta = Vec2::ZERO;
        self.wheel_delta = Vec2::ZERO;
        self.just_pressed = 0;
        self.just_released = 0;
    }

    /// Resets all mouse state.
    pub fn reset(&mut self) {
        self.position = Vec2::ZERO;
        self.delta = Vec2::ZERO;
        self.wheel_delta = Vec2::ZERO;
        self.has_position = false;
        self.pressed = 0;
        self.just_pressed = 0;
        self.just_released = 0;
    }
}
