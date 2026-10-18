//! Keyboard key representations and state tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Cross-platform scan codes and keyboard keys.
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyCode {
    // Letters
    KeyA,
    KeyB,
    KeyC,
    KeyD,
    KeyE,
    KeyF,
    KeyG,
    KeyH,
    KeyI,
    KeyJ,
    KeyK,
    KeyL,
    KeyM,
    KeyN,
    KeyO,
    KeyP,
    KeyQ,
    KeyR,
    KeyS,
    KeyT,
    KeyU,
    KeyV,
    KeyW,
    KeyX,
    KeyY,
    KeyZ,

    // Digits
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,

    // Modifiers
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    SuperLeft,
    SuperRight,

    // Controls & Navigation
    Space,
    Enter,
    Escape,
    Tab,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,

    // Function keys
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
}

/// Tracks current, just-pressed, and just-released keyboard state across frame boundaries.
#[derive(Debug, Clone, Default)]
pub struct KeyboardState {
    pressed: HashSet<KeyCode>,
    just_pressed: HashSet<KeyCode>,
    just_released: HashSet<KeyCode>,
}

impl KeyboardState {
    /// Creates a new, cleared `KeyboardState`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a key press event.
    pub fn press(&mut self, key: KeyCode) {
        if self.pressed.insert(key) {
            self.just_pressed.insert(key);
        }
    }

    /// Registers a key release event.
    pub fn release(&mut self, key: KeyCode) {
        if self.pressed.remove(&key) {
            self.just_released.insert(key);
        }
    }

    /// Returns `true` if the key is currently held down.
    #[must_use]
    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    /// Returns `true` if the key was pressed during the current frame.
    #[must_use]
    pub fn is_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed.contains(&key)
    }

    /// Returns `true` if the key was released during the current frame.
    #[must_use]
    pub fn is_just_released(&self, key: KeyCode) -> bool {
        self.just_released.contains(&key)
    }

    /// Returns `true` if any of the provided keys are currently held down.
    #[must_use]
    pub fn is_any_pressed(&self, keys: &[KeyCode]) -> bool {
        keys.iter().any(|&k| self.is_pressed(k))
    }

    /// Returns `true` if either Shift key is currently held down.
    #[must_use]
    pub fn is_shift_pressed(&self) -> bool {
        self.is_pressed(KeyCode::ShiftLeft) || self.is_pressed(KeyCode::ShiftRight)
    }

    /// Returns `true` if either Control key is currently held down.
    #[must_use]
    pub fn is_ctrl_pressed(&self) -> bool {
        self.is_pressed(KeyCode::ControlLeft) || self.is_pressed(KeyCode::ControlRight)
    }

    /// Returns `true` if either Alt key is currently held down.
    #[must_use]
    pub fn is_alt_pressed(&self) -> bool {
        self.is_pressed(KeyCode::AltLeft) || self.is_pressed(KeyCode::AltRight)
    }

    /// Clears frame-transient states (`just_pressed` and `just_released`).
    pub fn tick(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Clears all keyboard state.
    pub fn reset(&mut self) {
        self.pressed.clear();
        self.just_pressed.clear();
        self.just_released.clear();
    }
}
