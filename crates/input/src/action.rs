//! Abstract action mapping, chords, and multi-input aggregation.

use crate::gamepad::{GamepadAxis, GamepadButton, GamepadState};
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

    /// Alias for [`Self::bind`].
    pub fn bind_action(&mut self, action_name: impl Into<String>, binding: ActionBinding) {
        self.bind(action_name, binding);
    }

    /// Removes a specific binding from `action_name`, returning `true` if a matching binding was removed.
    pub fn unbind(&mut self, action_name: &str, binding: &ActionBinding) -> bool {
        let Some(action) = self.actions.get_mut(action_name) else {
            return false;
        };
        let before = action.bindings.len();
        action.bindings.retain(|b| b != binding);
        action.bindings.len() != before
    }

    /// Removes all bindings for `action_name`, returning `true` if the action existed.
    pub fn clear_action(&mut self, action_name: &str) -> bool {
        self.actions.remove(action_name).is_some()
    }

    /// Clears all registered actions and bindings.
    pub fn clear(&mut self) {
        self.actions.clear();
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
                    if gamepad
                        .connected_ids()
                        .any(|id| gamepad.is_pressed(id, *btn))
                    {
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
                    if gamepad
                        .connected_ids()
                        .any(|id| gamepad.axis(id, *axis).abs() > 0.1)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Returns `true` if any binding associated with `action_name` was just pressed this frame.
    #[must_use]
    pub fn is_just_pressed(
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
                    if keyboard.is_just_pressed(*key) {
                        return true;
                    }
                }
                ActionBinding::MouseButton(btn) => {
                    if mouse.is_just_pressed(*btn) {
                        return true;
                    }
                }
                ActionBinding::GamepadButton(btn) => {
                    if gamepad
                        .connected_ids()
                        .any(|id| gamepad.is_just_pressed(id, *btn))
                    {
                        return true;
                    }
                }
                ActionBinding::Chord { modifiers, key } => {
                    let all_held = modifiers.iter().all(|&m| keyboard.is_pressed(m))
                        && keyboard.is_pressed(*key);
                    let any_just = keyboard.is_just_pressed(*key)
                        || modifiers.iter().any(|&m| keyboard.is_just_pressed(m));
                    if all_held && any_just {
                        return true;
                    }
                }
                ActionBinding::Axis1D {
                    positive, negative, ..
                } => {
                    if keyboard.is_just_pressed(*positive) || keyboard.is_just_pressed(*negative) {
                        return true;
                    }
                }
                ActionBinding::Axis2D {
                    up,
                    down,
                    left,
                    right,
                } => {
                    if keyboard.is_just_pressed(*up)
                        || keyboard.is_just_pressed(*down)
                        || keyboard.is_just_pressed(*left)
                        || keyboard.is_just_pressed(*right)
                    {
                        return true;
                    }
                }
                ActionBinding::GamepadAxis { .. } => {}
            }
        }

        false
    }

    /// Returns `true` if any binding associated with `action_name` was just released this frame.
    #[must_use]
    pub fn is_just_released(
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
                    if keyboard.is_just_released(*key) {
                        return true;
                    }
                }
                ActionBinding::MouseButton(btn) => {
                    if mouse.is_just_released(*btn) {
                        return true;
                    }
                }
                ActionBinding::GamepadButton(btn) => {
                    if gamepad
                        .connected_ids()
                        .any(|id| gamepad.is_just_released(id, *btn))
                    {
                        return true;
                    }
                }
                ActionBinding::Chord { modifiers, key } => {
                    let key_was_active =
                        keyboard.is_pressed(*key) || keyboard.is_just_released(*key);
                    let mods_were_active = modifiers
                        .iter()
                        .all(|&m| keyboard.is_pressed(m) || keyboard.is_just_released(m));
                    let any_released = keyboard.is_just_released(*key)
                        || modifiers.iter().any(|&m| keyboard.is_just_released(m));
                    if key_was_active && mods_were_active && any_released {
                        return true;
                    }
                }
                ActionBinding::Axis1D {
                    positive, negative, ..
                } => {
                    if keyboard.is_just_released(*positive) || keyboard.is_just_released(*negative)
                    {
                        return true;
                    }
                }
                ActionBinding::Axis2D {
                    up,
                    down,
                    left,
                    right,
                } => {
                    if keyboard.is_just_released(*up)
                        || keyboard.is_just_released(*down)
                        || keyboard.is_just_released(*left)
                        || keyboard.is_just_released(*right)
                    {
                        return true;
                    }
                }
                ActionBinding::GamepadAxis { .. } => {}
            }
        }

        false
    }

    /// Computes aggregated 1D scalar value for `action_name` (e.g. throttle, steering), clamped to `[-1.0, 1.0]`.
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

        let mut val = 0.0f32;

        for binding in &action.bindings {
            match binding {
                ActionBinding::Axis1D {
                    positive,
                    negative,
                    scale,
                } => {
                    if !scale.is_finite() {
                        continue;
                    }
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
                    if !scale.is_finite() {
                        continue;
                    }
                    for id in gamepad.connected_ids() {
                        let a = gamepad.axis(id, *axis);
                        if a.is_finite() {
                            val += a * scale;
                        }
                    }
                }
                _ => {}
            }
        }

        if !val.is_finite() {
            0.0
        } else {
            val.clamp(-1.0, 1.0)
        }
    }

    /// Computes aggregated 2D directional vector for `action_name` (e.g. movement, look),
    /// clamped to a maximum Euclidean magnitude of `1.0` (`INP-05`).
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
                ActionBinding::GamepadAxis { axis, scale } => {
                    if !scale.is_finite() {
                        continue;
                    }
                    for id in gamepad.connected_ids() {
                        let a = gamepad.axis(id, *axis);
                        if !a.is_finite() {
                            continue;
                        }
                        match axis {
                            GamepadAxis::LeftStickX | GamepadAxis::RightStickX => {
                                dir.x += a * scale;
                            }
                            GamepadAxis::LeftStickY | GamepadAxis::RightStickY => {
                                dir.y += a * scale;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        let _ = mouse;
        let len_sq = dir.length_squared();
        if !len_sq.is_finite() {
            Vec2::ZERO
        } else if len_sq > 1.0 {
            dir / len_sq.sqrt()
        } else {
            dir
        }
    }
}
