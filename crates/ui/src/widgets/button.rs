//! Push button and toggle button widget with state management.

use crate::input::UiEvent;
use crate::widgets::alignment::HorizontalAlignment;
use korppi_ecs::Component;
use korppi_input::mouse::MouseButton;

/// Interactive visual state of a Button.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ButtonState {
    /// Default idle state.
    #[default]
    Normal,
    /// Mouse cursor is hovering over the button.
    Hover,
    /// Mouse button or activate key is currently held down.
    Pressed,
    /// Button is inactive and non-interactive.
    Disabled,
    /// Button currently possesses keyboard or gamepad focus.
    Focused,
}

/// Push button and toggle button widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Button {
    /// Text label displayed inside the button.
    text: String,
    /// Optional icon texture path or identifier.
    pub icon: Option<String>,
    /// Alignment of the icon relative to button text.
    pub icon_alignment: HorizontalAlignment,
    /// Current interactive state.
    pub state: ButtonState,
    /// Whether the button is disabled.
    pub disabled: bool,
    /// If true, button toggles between pressed and released states like a latch.
    pub toggle_mode: bool,
    /// Whether the button is currently in the toggled pressed state.
    button_pressed: bool,
}

impl Component for Button {}

impl Button {
    /// Creates a new `Button` with the given text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            icon: None,
            icon_alignment: HorizontalAlignment::Left,
            state: ButtonState::Normal,
            disabled: false,
            toggle_mode: false,
            button_pressed: false,
        }
    }

    /// Returns the button text.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Updates the button text.
    #[inline]
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    /// Returns whether the button is in the pressed/toggled state.
    #[inline]
    pub fn is_pressed(&self) -> bool {
        if self.toggle_mode {
            self.button_pressed
        } else {
            self.state == ButtonState::Pressed
        }
    }

    /// Explicitly sets the pressed state (relevant for toggle mode and programmatic triggers).
    #[inline]
    pub fn set_pressed(&mut self, pressed: bool) {
        self.button_pressed = pressed;
        if !self.toggle_mode {
            if pressed {
                self.state = ButtonState::Pressed;
            } else if self.state == ButtonState::Pressed {
                self.state = ButtonState::Normal;
            }
        }
    }

    /// Sets whether the button is disabled and updates button state accordingly.
    pub fn set_disabled(&mut self, disabled: bool) {
        self.disabled = disabled;
        if disabled {
            self.state = ButtonState::Disabled;
        } else if self.state == ButtonState::Disabled {
            self.state = ButtonState::Normal;
        }
    }

    /// Processes an incoming UI event and updates button state.
    ///
    /// Returns `true` if a full click activation occurred.
    pub fn handle_event(&mut self, event: &UiEvent) -> bool {
        if self.disabled {
            self.state = ButtonState::Disabled;
            return false;
        }

        match event {
            UiEvent::MouseEnter => {
                if self.state != ButtonState::Pressed {
                    self.state = ButtonState::Hover;
                }
                false
            }
            UiEvent::MouseExit => {
                if self.state != ButtonState::Pressed {
                    self.state = ButtonState::Normal;
                }
                false
            }
            UiEvent::MouseButton {
                button: MouseButton::Left,
                pressed: true,
                ..
            } => {
                self.state = ButtonState::Pressed;
                if self.toggle_mode {
                    self.button_pressed = !self.button_pressed;
                }
                false
            }
            UiEvent::MouseButton {
                button: MouseButton::Left,
                pressed: false,
                ..
            } => {
                let was_pressed = self.state == ButtonState::Pressed;
                self.state = ButtonState::Hover;
                was_pressed
            }
            UiEvent::FocusGained => {
                if self.state == ButtonState::Normal {
                    self.state = ButtonState::Focused;
                }
                false
            }
            UiEvent::FocusLost => {
                if self.state == ButtonState::Focused {
                    self.state = ButtonState::Normal;
                }
                false
            }
            _ => false,
        }
    }
}
