//! Synthetic programmatic input dispatcher for automated tests and AI agent interaction.

use crate::containers::{ScrollContainer, ScrollMode};
use crate::control::{Control, FocusMode};
use crate::input::event::{UiEvent, UiReceiver};
use crate::input::focus::FocusManager;
use crate::input::routing::hit_test_controls;
use crate::layout::hierarchy::ResolvedRect;
use crate::widgets::{
    Button, CheckBox, CheckButton, LineEdit, Slider, SliderOrientation, TextEdit,
};
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
    /// Control that captured the active left mouse button press.
    captured_control: Option<Entity>,
}

impl UiInputDispatcher {
    /// Creates a new `UiInputDispatcher`.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    fn sync_hover_state(&mut self, world: &mut World, new_hovered: &[Entity]) {
        // 1. Controls exited
        for &entity in &self.hovered_controls {
            if !new_hovered.contains(&entity) && world.is_alive(entity) {
                if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                    receiver.push(UiEvent::MouseExit);
                }
                if let Some(btn) = world.get_mut::<Button>(entity) {
                    btn.handle_event(&UiEvent::MouseExit);
                }
            }
        }

        // 2. Controls entered
        for &entity in new_hovered {
            if !self.hovered_controls.contains(&entity) && world.is_alive(entity) {
                if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                    receiver.push(UiEvent::MouseEnter);
                }
                if let Some(btn) = world.get_mut::<Button>(entity) {
                    btn.handle_event(&UiEvent::MouseEnter);
                }
            }
        }

        self.hovered_controls.clear();
        self.hovered_controls.extend_from_slice(new_hovered);
    }

    fn update_slider_at_position(world: &mut World, entity: Entity, position: Vec2) {
        let Some(rect) = world.get::<ResolvedRect>(entity).map(|r| r.global) else {
            return;
        };
        if let Some(slider) = world.get_mut::<Slider>(entity) {
            match slider.orientation {
                SliderOrientation::Horizontal => {
                    slider.set_value_from_offset(position.x - rect.x, rect.width);
                }
                SliderOrientation::Vertical => {
                    slider.set_value_from_offset(rect.y + rect.height - position.y, rect.height);
                }
            }
        }
    }

    /// Injects a mouse move event, automatically generating `MouseEnter` and `MouseExit` events.
    pub fn inject_mouse_move(&mut self, world: &mut World, position: Vec2) {
        self.focus_manager.validate_focus(world);
        let new_hovered = hit_test_controls(world, position);
        self.sync_hover_state(world, &new_hovered);

        // Mouse move event on all hovered controls
        for &entity in &new_hovered {
            if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                receiver.push(UiEvent::MouseMove { position });
            }
        }

        if let Some(captured) = self.captured_control {
            if world.is_alive(captured) {
                Self::update_slider_at_position(world, captured, position);
            } else {
                self.captured_control = None;
            }
        }
    }

    /// Injects a mouse button press or release event.
    ///
    /// If `pressed` is true and left button is clicked, assigns focus to the front-most focusable
    /// control hit, or clears focus if no focusable control is hit.
    pub fn inject_mouse_button(
        &mut self,
        world: &mut World,
        position: Vec2,
        button: MouseButton,
        pressed: bool,
    ) {
        self.focus_manager.validate_focus(world);
        let hits = hit_test_controls(world, position);
        self.sync_hover_state(world, &hits);

        // Update focus on left click press
        if pressed && button == MouseButton::Left {
            let focus_target = hits
                .iter()
                .find(|&&e| {
                    world
                        .get::<Control>(e)
                        .is_some_and(|ctl| ctl.focus_mode != FocusMode::None)
                })
                .copied();

            self.focus_manager.set_focus(world, focus_target);
            self.captured_control = hits.first().copied();
        }

        let event = UiEvent::MouseButton {
            position,
            button,
            pressed,
        };

        // Dispatch mouse button event to all hit controls
        for &entity in &hits {
            if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                receiver.push(event.clone());
            }
            if let Some(btn) = world.get_mut::<Button>(entity) {
                btn.handle_event(&event);
            }
            if button == MouseButton::Left {
                if pressed {
                    Self::update_slider_at_position(world, entity, position);
                } else {
                    if let Some(cb) = world.get_mut::<CheckBox>(entity) {
                        cb.toggle();
                    }
                    if let Some(cb) = world.get_mut::<CheckButton>(entity) {
                        cb.toggle();
                    }
                }
            }
        }

        // If releasing left button outside the captured control, deliver release to captured control
        if !pressed && button == MouseButton::Left {
            if let Some(captured) = self.captured_control.take() {
                if !hits.contains(&captured) && world.is_alive(captured) {
                    if let Some(receiver) = world.get_mut::<UiReceiver>(captured) {
                        receiver.push(event.clone());
                    }
                    if let Some(btn) = world.get_mut::<Button>(captured) {
                        btn.handle_event(&event);
                    }
                }
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
        self.focus_manager.validate_focus(world);
        let hits = hit_test_controls(world, position);
        for &entity in &hits {
            if let Some(receiver) = world.get_mut::<UiReceiver>(entity) {
                receiver.push(UiEvent::MouseWheel { position, delta });
            }
            if let Some(scroll) = world.get_mut::<ScrollContainer>(entity) {
                if scroll.horizontal_scroll_mode != ScrollMode::Disabled {
                    scroll.scroll_horizontal = (scroll.scroll_horizontal - delta.x).max(0.0);
                }
                if scroll.vertical_scroll_mode != ScrollMode::Disabled {
                    scroll.scroll_vertical = (scroll.scroll_vertical - delta.y).max(0.0);
                }
            }
            if let Some(slider) = world.get_mut::<Slider>(entity) {
                if slider.scrollable && slider.editable {
                    let step = if slider.range.step > 0.0 {
                        slider.range.step
                    } else {
                        (slider.range.max_value - slider.range.min_value) * 0.05
                    };
                    let dir = if delta.y > 0.0 || delta.x > 0.0 {
                        1.0
                    } else if delta.y < 0.0 || delta.x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let next = slider.range.value() + dir * step;
                    slider.range.set_value(next);
                }
            }
        }
    }

    /// Injects a Tab navigation event.
    pub fn inject_tab(&mut self, world: &mut World, reverse: bool) {
        self.focus_manager.navigate_tab(world, reverse);
    }

    /// Injects a keyboard key press or release targeted to the currently focused control.
    pub fn inject_key(&mut self, world: &mut World, key: KeyCode, pressed: bool) {
        self.focus_manager.validate_focus(world);
        if let Some(focused) = self.focus_manager.focused_control() {
            let event = if pressed {
                UiEvent::KeyDown { key }
            } else {
                UiEvent::KeyUp { key }
            };
            if let Some(receiver) = world.get_mut::<UiReceiver>(focused) {
                receiver.push(event.clone());
            }
            if let Some(btn) = world.get_mut::<Button>(focused) {
                btn.handle_event(&event);
            }
            if pressed {
                if matches!(key, KeyCode::Enter | KeyCode::Space) {
                    if let Some(cb) = world.get_mut::<CheckBox>(focused) {
                        cb.toggle();
                    }
                    if let Some(cb) = world.get_mut::<CheckButton>(focused) {
                        cb.toggle();
                    }
                }
                if let Some(le) = world.get_mut::<LineEdit>(focused) {
                    match key {
                        KeyCode::Backspace => le.delete_char_before_caret(),
                        KeyCode::Delete => le.delete_char_at_caret(),
                        KeyCode::ArrowLeft => {
                            le.set_caret_column(le.caret_column().saturating_sub(1));
                        }
                        KeyCode::ArrowRight => {
                            le.set_caret_column(le.caret_column() + 1);
                        }
                        KeyCode::Home => le.set_caret_column(0),
                        KeyCode::End => le.set_caret_column(usize::MAX),
                        _ => {}
                    }
                }
                if let Some(te) = world.get_mut::<TextEdit>(focused) {
                    match key {
                        KeyCode::Backspace => te.delete_char_before_caret(),
                        KeyCode::Delete => te.delete_char_at_caret(),
                        KeyCode::Enter => te.insert_text("\n"),
                        _ => {}
                    }
                }
            }
        }
    }

    /// Injects printable character input targeted to the currently focused control.
    pub fn inject_char(&mut self, world: &mut World, character: char) {
        self.focus_manager.validate_focus(world);
        if let Some(focused) = self.focus_manager.focused_control() {
            if let Some(receiver) = world.get_mut::<UiReceiver>(focused) {
                receiver.push(UiEvent::CharInput { character });
            }
            if !character.is_control() {
                let mut buf = [0u8; 4];
                let s = character.encode_utf8(&mut buf);
                if let Some(le) = world.get_mut::<LineEdit>(focused) {
                    le.insert_text(s);
                }
                if let Some(te) = world.get_mut::<TextEdit>(focused) {
                    te.insert_text(s);
                }
            }
        }
    }
}
