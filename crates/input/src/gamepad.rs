//! Gamepad buttons, analog sticks, triggers, and deadzone processing.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a connected gamepad device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GamepadId(pub usize);

/// Standardized gamepad buttons (matching modern controller layouts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GamepadButton {
    /// Bottom face button (A on Xbox, Cross on PlayStation, B on Nintendo).
    South,
    /// Right face button (B on Xbox, Circle on PlayStation, A on Nintendo).
    East,
    /// Left face button (X on Xbox, Square on PlayStation, Y on Nintendo).
    West,
    /// Top face button (Y on Xbox, Triangle on PlayStation, X on Nintendo).
    North,
    /// D-pad Up.
    DPadUp,
    /// D-pad Down.
    DPadDown,
    /// D-pad Left.
    DPadLeft,
    /// D-pad Right.
    DPadRight,
    /// Left shoulder bumper button (LB / L1).
    LeftBumper,
    /// Right shoulder bumper button (RB / R1).
    RightBumper,
    /// Left trigger digital click or threshold (LT / L2).
    LeftTrigger,
    /// Right trigger digital click or threshold (RT / R2).
    RightTrigger,
    /// Left analog stick click (L3).
    LeftThumb,
    /// Right analog stick click (R3).
    RightThumb,
    /// Select / Back / View button.
    Select,
    /// Start / Menu button.
    Start,
    /// Guide / Home / PS button.
    Mode,
}

/// Standardized analog axes on gamepads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GamepadAxis {
    /// Left analog stick horizontal axis (-1.0 Left to +1.0 Right).
    LeftStickX,
    /// Left analog stick vertical axis (-1.0 Down to +1.0 Up).
    LeftStickY,
    /// Right analog stick horizontal axis (-1.0 Left to +1.0 Right).
    RightStickX,
    /// Right analog stick vertical axis (-1.0 Down to +1.0 Up).
    RightStickY,
    /// Left analog trigger (0.0 Released to +1.0 Fully Pressed).
    LeftTrigger,
    /// Right analog trigger (0.0 Released to +1.0 Fully Pressed).
    RightTrigger,
}

/// Internal state tracking for a single connected gamepad.
#[derive(Debug, Clone)]
struct GamepadDevice {
    pressed: HashSet<GamepadButton>,
    just_pressed: HashSet<GamepadButton>,
    just_released: HashSet<GamepadButton>,
    axes: HashMap<GamepadAxis, f32>,
    deadzone: f32,
}

impl Default for GamepadDevice {
    fn default() -> Self {
        Self {
            pressed: HashSet::new(),
            just_pressed: HashSet::new(),
            just_released: HashSet::new(),
            axes: HashMap::new(),
            deadzone: 0.15,
        }
    }
}

/// Manages multiple gamepads, button transitions, and analog axis deadzone calibration.
#[derive(Debug, Clone, Default)]
pub struct GamepadState {
    devices: HashMap<GamepadId, GamepadDevice>,
}

impl GamepadState {
    /// Creates an empty `GamepadState`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a gamepad connection.
    pub fn connect(&mut self, id: GamepadId) {
        self.devices.entry(id).or_default();
    }

    /// Unregisters a disconnected gamepad.
    pub fn disconnect(&mut self, id: GamepadId) {
        self.devices.remove(&id);
    }

    /// Returns `true` if the gamepad is currently connected.
    #[must_use]
    pub fn is_connected(&self, id: GamepadId) -> bool {
        self.devices.contains_key(&id)
    }

    /// Sets the axis deadzone for the gamepad (default is 0.15).
    pub fn set_deadzone(&mut self, id: GamepadId, deadzone: f32) {
        if let Some(dev) = self.devices.get_mut(&id) {
            dev.deadzone = deadzone.clamp(0.0, 1.0);
        }
    }

    /// Sets raw axis value, applying radial deadzone calibration.
    pub fn set_axis(&mut self, id: GamepadId, axis: GamepadAxis, raw_val: f32) {
        if let Some(dev) = self.devices.get_mut(&id) {
            let filtered = if raw_val.abs() < dev.deadzone {
                0.0
            } else {
                let sign = raw_val.signum();
                let scaled = (raw_val.abs() - dev.deadzone) / (1.0 - dev.deadzone);
                sign * scaled.clamp(0.0, 1.0)
            };
            dev.axes.insert(axis, filtered);
        }
    }

    /// Reads current calibrated value for `axis`.
    #[must_use]
    pub fn axis(&self, id: GamepadId, axis: GamepadAxis) -> f32 {
        self.devices
            .get(&id)
            .and_then(|dev| dev.axes.get(&axis).copied())
            .unwrap_or(0.0)
    }

    /// Registers a gamepad button press.
    pub fn press(&mut self, id: GamepadId, button: GamepadButton) {
        if let Some(dev) = self.devices.get_mut(&id) {
            if dev.pressed.insert(button) {
                dev.just_pressed.insert(button);
            }
        }
    }

    /// Registers a gamepad button release.
    pub fn release(&mut self, id: GamepadId, button: GamepadButton) {
        if let Some(dev) = self.devices.get_mut(&id) {
            if dev.pressed.remove(&button) {
                dev.just_released.insert(button);
            }
        }
    }

    /// Returns `true` if the button is held down.
    #[must_use]
    pub fn is_pressed(&self, id: GamepadId, button: GamepadButton) -> bool {
        self.devices
            .get(&id)
            .is_some_and(|dev| dev.pressed.contains(&button))
    }

    /// Returns `true` if the button was pressed in the current frame.
    #[must_use]
    pub fn is_just_pressed(&self, id: GamepadId, button: GamepadButton) -> bool {
        self.devices
            .get(&id)
            .is_some_and(|dev| dev.just_pressed.contains(&button))
    }

    /// Returns `true` if the button was released in the current frame.
    #[must_use]
    pub fn is_just_released(&self, id: GamepadId, button: GamepadButton) -> bool {
        self.devices
            .get(&id)
            .is_some_and(|dev| dev.just_released.contains(&button))
    }

    /// Advances frame tick, clearing frame-transient transitions.
    pub fn tick(&mut self) {
        for dev in self.devices.values_mut() {
            dev.just_pressed.clear();
            dev.just_released.clear();
        }
    }
}
