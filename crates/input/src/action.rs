//! Abstract action mapping, chords, and multi-input aggregation.

use crate::gamepad::{GamepadAxis, GamepadButton, GamepadId, GamepadState};
use crate::keyboard::{KeyCode, KeyboardState};
use crate::mouse::{MouseButton, MouseState};
use korppi_math::Vec2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An input event binding linked to a virtual game action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionBinding {
    /// Digital keyboard key press.
    Key(KeyCode),

    /// Digital mouse button press.
    MouseButton(MouseButton),

    /// Digital gamepad button press on any connected controller.
    GamepadButton(GamepadButton),

    /// Analog gamepad axis with scale factor.
    GamepadAxis {
        /// The target analog axis.
        axis: GamepadAxis,
        /// Scale multiplier.
        scale: f32,
    },

    /// Key chord requiring all modifier keys plus the trigger key to be held simultaneously.
    Chord {
        /// Required modifier keys (e.g. `ControlLeft`, `ShiftLeft`).
        modifiers: Vec<KeyCode>,
        /// Main trigger key (e.g. `KeyS`).
        key: KeyCode,
    },

    /// 1D virtual axis composed of positive and negative keys (e.g. D = +1, A = -1).
    Axis1D {
        /// Key producing positive deflection.
        positive: KeyCode,
        /// Key producing negative deflection.
        negative: KeyCode,
        /// Output multiplier.
        scale: f32,
    },

    /// 2D virtual directional axis (e.g. WASD or Arrow Keys).
    Axis2D {
        /// Upward direction key (+Y).
        up: KeyCode,
        /// Downward direction key (-Y).
        down: KeyCode,
        /// Leftward direction key (-X).
        left: KeyCode,
        /// Rightward direction key (+X).
        right: KeyCode,
    },
}

/// A logical action with one or more assigned bindings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InputAction {
    /// Human-readable action identifier (e.g. `"Jump"`, `"Fire"`, `"Move"`).
    pub name: String,
    /// Collection of alternative bindings that can trigger this action.
    pub bindings: Vec<ActionBinding>,
}

/// Central registry mapping game action names to physical input bindings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionMap {
    actions: HashMap<String, InputAction>,
}

impl ActionMap {
    /// Creates a new, empty `ActionMap`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Associates an input binding with a named action.
    pub fn bind(&mut self, action_name: impl Into<String>, binding: ActionBinding) {
        let name = action_name.into();
        let entry = self
            .actions
            .entry(name.clone())
            .or_insert_with(|| InputAction {
                name,
                bindings: Vec::new(),
            });
        entry.bindings.push(binding);
    }

    /// Returns `true` if any binding associated with `action_name` is currently active.
    #[must_use]
    pub fn is_active(
        &self,
        action_name: &str,
        keyboard: &KeyboardState,
        mouse: &MouseState,
        gamepad: &GamepadState,
    ) -> bool {
        let Some(action) = self.actions.get(action_name) else {
            return false;
        };

        for binding in &action.bindings {
            match binding {
                ActionBinding::Key(key) => {
                    if keyboard.is_pressed(*key) {
                        return true;
                    }
                }
                ActionBinding::MouseButton(btn) => {
                    if mouse.is_pressed(*btn) {
                        return true;
                    }
                }
                ActionBinding::GamepadButton(btn) => {
                    // Check default gamepad 0 or any connected gamepad
                    if gamepad.is_pressed(GamepadId(0), *btn) {
                        return true;
                    }
                }
                ActionBinding::Chord { modifiers, key } => {
                    let all_mods = modifiers.iter().all(|&m| keyboard.is_pressed(m));
                    if all_mods && keyboard.is_pressed(*key) {
                        return true;
                    }
                }
                ActionBinding::Axis1D {
                    positive, negative, ..
                } => {
                    if keyboard.is_pressed(*positive) || keyboard.is_pressed(*negative) {
                        return true;
                    }
                }
                ActionBinding::Axis2D {
                    up,
                    down,
                    left,
                    right,
                } => {
                    if keyboard.is_pressed(*up)
                        || keyboard.is_pressed(*down)
                        || keyboard.is_pressed(*left)
                        || keyboard.is_pressed(*right)
                    {
                        return true;
                    }
                }
                ActionBinding::GamepadAxis { axis, .. } => {
                    if gamepad.axis(GamepadId(0), *axis).abs() > 0.1 {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Computes aggregated 1D scalar value for `action_name` (e.g. throttle, steering).
    #[must_use]
    pub fn axis_1d(
        &self,
        action_name: &str,
        keyboard: &KeyboardState,
        _mouse: &MouseState,
        gamepad: &GamepadState,
    ) -> f32 {
        let Some(action) = self.actions.get(action_name) else {
            return 0.0;
        };

        let mut val = 0.0;

        for binding in &action.bindings {
            match binding {
                ActionBinding::Axis1D {
                    positive,
                    negative,
                    scale,
                } => {
                    let mut delta = 0.0;
                    if keyboard.is_pressed(*positive) {
                        delta += 1.0;
                    }
                    if keyboard.is_pressed(*negative) {
                        delta -= 1.0;
                    }
                    val += delta * scale;
                }
                ActionBinding::GamepadAxis { axis, scale } => {
                    val += gamepad.axis(GamepadId(0), *axis) * scale;
                }
                _ => {}
            }
        }

        val.clamp(-1.0, 1.0)
    }

    /// Computes aggregated 2D directional vector for `action_name` (e.g. movement, look).
    #[must_use]
    pub fn axis_2d(
        &self,
        action_name: &str,
        keyboard: &KeyboardState,
        mouse: &MouseState,
        gamepad: &GamepadState,
    ) -> Vec2 {
        let Some(action) = self.actions.get(action_name) else {
            return Vec2::ZERO;
        };

        let mut dir = Vec2::ZERO;

        for binding in &action.bindings {
            match binding {
                ActionBinding::Axis2D {
                    up,
                    down,
                    left,
                    right,
                } => {
                    if keyboard.is_pressed(*up) {
                        dir.y += 1.0;
                    }
                    if keyboard.is_pressed(*down) {
                        dir.y -= 1.0;
                    }
                    if keyboard.is_pressed(*left) {
                        dir.x -= 1.0;
                    }
                    if keyboard.is_pressed(*right) {
                        dir.x += 1.0;
                    }
                }
                ActionBinding::GamepadAxis { axis, scale } => match axis {
                    GamepadAxis::LeftStickX | GamepadAxis::RightStickX => {
                        dir.x += gamepad.axis(GamepadId(0), *axis) * scale;
                    }
                    GamepadAxis::LeftStickY | GamepadAxis::RightStickY => {
                        dir.y += gamepad.axis(GamepadId(0), *axis) * scale;
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        let _ = mouse; // Available for look/aim bindings
        dir
    }
}
