//! Binary toggle widgets: `CheckBox` and `CheckButton`.

use korppi_ecs::Component;

/// Binary checkbox widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CheckBox {
    /// Label text displayed adjacent to the checkbox box.
    text: String,
    /// Checked toggle state.
    checked: bool,
    /// Whether user interaction is disabled.
    pub disabled: bool,
}

impl Component for CheckBox {}

impl CheckBox {
    /// Creates a new checkbox with the given label text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            checked: false,
            disabled: false,
        }
    }

    /// Returns label text.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Updates label text.
    #[inline]
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    /// Returns whether the checkbox is checked.
    #[inline]
    pub fn is_checked(&self) -> bool {
        self.checked
    }

    /// Sets the checked state.
    #[inline]
    pub fn set_checked(&mut self, checked: bool) {
        if !self.disabled {
            self.checked = checked;
        }
    }

    /// Toggles the checked state.
    #[inline]
    pub fn toggle(&mut self) {
        if !self.disabled {
            self.checked = !self.checked;
        }
    }
}

/// Binary toggle switch widget component (switch button presentation).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CheckButton {
    /// Label text displayed adjacent to the switch.
    text: String,
    /// Checked toggle state.
    checked: bool,
    /// Whether user interaction is disabled.
    pub disabled: bool,
}

impl Component for CheckButton {}

impl CheckButton {
    /// Creates a new check button with the given label text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            checked: false,
            disabled: false,
        }
    }

    /// Returns label text.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Updates label text.
    #[inline]
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
    }

    /// Returns whether the switch is in the active checked state.
    #[inline]
    pub fn is_checked(&self) -> bool {
        self.checked
    }

    /// Sets the checked state.
    #[inline]
    pub fn set_checked(&mut self, checked: bool) {
        if !self.disabled {
            self.checked = checked;
        }
    }

    /// Toggles the checked state.
    #[inline]
    pub fn toggle(&mut self) {
        if !self.disabled {
            self.checked = !self.checked;
        }
    }
}
