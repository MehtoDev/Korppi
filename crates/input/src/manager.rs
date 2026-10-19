//! Central InputManager coordinating keyboard, mouse, gamepad, and action maps.

use crate::action::{ActionBinding, ActionMap};
use crate::gamepad::{GamepadAxis, GamepadId, GamepadState};
use crate::keyboard::{KeyCode, KeyboardState};
use crate::mouse::{MouseButton, MouseState};
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

    // -----------------------------------------------------------------------
    // WindowEvent Ingestion (INP-01 / INP-03)
    // -----------------------------------------------------------------------

    /// Processes a keyboard input event from `WindowEvent::KeyboardInput`.
    pub fn on_key_input(&mut self, key_code: Option<u32>, pressed: bool) {
        if let Some(code) = key_code.and_then(KeyCode::from_u32) {
            if pressed {
                self.keyboard.press(code);
            } else {
                self.keyboard.release(code);
            }
        }
    }

    /// Processes a mouse button event from `WindowEvent::MouseInput`.
    pub fn on_mouse_input(&mut self, button: u32, pressed: bool) {
        let btn = MouseButton::from_u32(button);
        if pressed {
            self.mouse.on_button_pressed(btn);
        } else {
            self.mouse.on_button_released(btn);
        }
    }

    /// Processes a cursor movement event from `WindowEvent::CursorMoved`.
    pub fn on_cursor_moved(&mut self, x: f64, y: f64) {
        self.mouse.on_moved(Vec2::new(x as f32, y as f32));
    }

    /// Processes a `WindowEvent::CursorLeft` event so the next cursor entry does not spike delta.
    pub fn on_cursor_left(&mut self) {
        self.mouse.on_cursor_left();
    }

    /// Processes a scroll wheel event from `WindowEvent::MouseWheel`.
    pub fn on_mouse_wheel(&mut self, delta_x: f32, delta_y: f32) {
        self.mouse.on_wheel(Vec2::new(delta_x, delta_y));
    }

    /// Processes a `WindowEvent::Focused` event, releasing all held keys and buttons on focus loss.
    pub fn on_focused(&mut self, focused: bool) {
        if !focused {
            self.on_focus_lost();
        }
    }

    /// Releases all currently held keyboard keys and mouse buttons, marking them `just_released`.
    pub fn on_focus_lost(&mut self) {
        self.keyboard.on_focus_lost();
        self.mouse.on_focus_lost();
    }

    // -----------------------------------------------------------------------
    // Direct Device Convenience Queries (INP-06 / docs/crates/input.md)
    // -----------------------------------------------------------------------

    /// Returns `true` if `key` is currently held down.
    #[inline]
    #[must_use]
    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.keyboard.is_pressed(key)
    }

    /// Returns `true` if `key` was pressed during the current frame.
    #[inline]
    #[must_use]
    pub fn is_just_pressed(&self, key: KeyCode) -> bool {
        self.keyboard.is_just_pressed(key)
    }

    /// Returns `true` if `key` was released during the current frame.
    #[inline]
    #[must_use]
    pub fn is_just_released(&self, key: KeyCode) -> bool {
        self.keyboard.is_just_released(key)
    }

    /// Returns the current cursor position in logical client coordinates.
    #[inline]
    #[must_use]
    pub fn mouse_position(&self) -> Vec2 {
        self.mouse.position()
    }

    /// Returns the cursor motion delta accumulated during the current frame.
    #[inline]
    #[must_use]
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse.delta()
    }

    /// Returns `true` if the specified mouse button is currently held down.
    #[inline]
    #[must_use]
    pub fn is_mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.mouse.is_pressed(button)
    }

    /// Reads the calibrated analog axis value for `gamepad_id`.
    #[inline]
    #[must_use]
    pub fn gamepad_axis(&self, gamepad_id: u32, axis: GamepadAxis) -> f32 {
        self.gamepad.axis(GamepadId(gamepad_id as usize), axis)
    }

    // -----------------------------------------------------------------------
    // Action Map Queries (INP-05 / INP-06)
    // -----------------------------------------------------------------------

    /// Associates an input binding with a named action.
    pub fn bind_action(&mut self, action: impl Into<String>, binding: ActionBinding) {
        self.actions.bind(action, binding);
    }

    /// Queries whether the named action is currently active on any bound device.
    #[must_use]
    pub fn is_active(&self, action: &str) -> bool {
        self.actions
            .is_active(action, &self.keyboard, &self.mouse, &self.gamepad)
    }

    /// Alias for [`Self::is_active`].
    #[must_use]
    pub fn is_action_active(&self, action: &str) -> bool {
        self.is_active(action)
    }

    /// Queries whether the named action was just triggered during the current frame.
    #[must_use]
    pub fn is_action_just_pressed(&self, action: &str) -> bool {
        self.actions
            .is_just_pressed(action, &self.keyboard, &self.mouse, &self.gamepad)
    }

    /// Queries whether the named action was just released during the current frame.
    #[must_use]
    pub fn is_action_just_released(&self, action: &str) -> bool {
        self.actions
            .is_just_released(action, &self.keyboard, &self.mouse, &self.gamepad)
    }

    /// Queries a 1D scalar axis value for the named action.
    #[must_use]
    pub fn axis_1d(&self, action: &str) -> f32 {
        self.actions
            .axis_1d(action, &self.keyboard, &self.mouse, &self.gamepad)
    }

    /// Alias for [`Self::axis_1d`].
    #[must_use]
    pub fn action_axis(&self, action: &str) -> f32 {
        self.axis_1d(action)
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

    /// Advances input frame state and clears transient edge detectors (alias for [`Self::tick`]).
    pub fn update(&mut self) {
        self.tick();
    }
}
