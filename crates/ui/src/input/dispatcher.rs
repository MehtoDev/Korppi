//! Synthetic programmatic input dispatcher for automated tests and AI agent interaction.

use crate::control::{Control, FocusMode};
use crate::input::event::{UiEvent, UiReceiver};
use crate::input::focus::FocusManager;
use crate::input::routing::hit_test_controls;
use korppi_ecs::{Entity, World};
use korppi_input::keyboard::KeyCode;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;

/// Programmatic input dispatcher simulating mouse, keyboard, and focus events.
#[derive(Clone, Debug, Default)]
pub struct UiInputDispatcher {
    /// Focus manager tracking the currently active control.
    pub focus_manager: FocusManager,
    /// Currently hovered control entities.
    hovered_controls: Vec<Entity>,
}

impl UiInputDispatcher {
    /// Creates a new `UiInputDispatcher`.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Injects a mouse move event, automatically generating `MouseEnter` and `MouseExit` events.
    pub fn inject_mouse_move(&mut self, world: &mut World, position: Vec2) {
        let new_hovered = hit_test_controls(world, position);

        // 1. Controls exited
        for &entity in &self.hovered_controls {
            if !new_hovered.contains(&entity) {
                if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                    receiver.push(UiEvent::MouseExit);
                }
            }
        }

        // 2. Controls entered
        for &entity in &new_hovered {
            if !self.hovered_controls.contains(&entity) {
                if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                    receiver.push(UiEvent::MouseEnter);
                }
            }
        }

        // 3. Mouse move event on all hovered controls
        for &entity in &new_hovered {
            if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                receiver.push(UiEvent::MouseMove { position });
            }
        }

        self.hovered_controls = new_hovered;
    }

    /// Injects a mouse button press or release event.
    ///
    /// If `pressed` is true and left button is clicked, assigns focus to the front-most focusable control hit.
    pub fn inject_mouse_button(
        &mut self,
        world: &mut World,
        position: Vec2,
        button: MouseButton,
        pressed: bool,
    ) {
        let hits = hit_test_controls(world, position);

        // Update focus on left click press
        if pressed && button == MouseButton::Left {
            let focus_target = hits
                .iter()
                .find(|&&e| {
                    if let Some(ctl) = world.get::<Control>(e) {
                        ctl.focus_mode != FocusMode::None
                    } else {
                        false
                    }
                })
                .copied();

            if let Some(target) = focus_target {
                self.focus_manager.set_focus(world, Some(target));
            }
        }

        // Dispatch mouse button event to all hit controls
        for &entity in &hits {
            if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                receiver.push(UiEvent::MouseButton {
                    position,
                    button,
                    pressed,
                });
            }
        }
    }

    /// Convenience helper injecting a full click (press followed by release).
    pub fn inject_mouse_click(&mut self, world: &mut World, position: Vec2, button: MouseButton) {
        self.inject_mouse_button(world, position, button, true);
        self.inject_mouse_button(world, position, button, false);
    }

    /// Injects a mouse wheel scroll event.
    pub fn inject_mouse_wheel(&mut self, world: &mut World, position: Vec2, delta: Vec2) {
        let hits = hit_test_controls(world, position);
        for &entity in &hits {
            if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                receiver.push(UiEvent::MouseWheel { position, delta });
            }
        }
    }

    /// Injects a Tab navigation event.
    pub fn inject_tab(&mut self, world: &mut World, reverse: bool) {
        self.focus_manager.navigate_tab(world, reverse);
    }

    /// Injects a keyboard key press or release targeted to the currently focused control.
    pub fn inject_key(&mut self, world: &mut World, key: KeyCode, pressed: bool) {
        if let Some(focused) = self.focus_manager.focused_control() {
            if let Some(receiver) = world.get_mut::<UiReceiver>(focused) {
                if pressed {
                    receiver.push(UiEvent::KeyDown { key });
                } else {
                    receiver.push(UiEvent::KeyUp { key });
                }
            }
        }
    }

    /// Injects printable character input targeted to the currently focused control.
    pub fn inject_char(&mut self, world: &mut World, character: char) {
        if let Some(focused) = self.focus_manager.focused_control() {
            if let Some(receiver) = world.get_mut::<UiReceiver>(focused) {
                receiver.push(UiEvent::CharInput { character });
            }
        }
    }
}
