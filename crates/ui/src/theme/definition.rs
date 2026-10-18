//! Central UI theme repository and per-control style overrides.

use crate::theme::stylebox::{StyleBox, StyleBoxFlat};
use korppi_ecs::Component;
use std::collections::HashMap;

/// Central theme mapping UI control types and item keys to visual styling attributes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Theme {
    /// Color map: `(theme_type, item_name)` -> RGBA.
    pub colors: HashMap<(String, String), [f32; 4]>,
    /// Integer constant map: `(theme_type, item_name)` -> value.
    pub constants: HashMap<(String, String), i32>,
    /// Font size map: `(theme_type, item_name)` -> size.
    pub font_sizes: HashMap<(String, String), u32>,
    /// Font asset key map: `(theme_type, item_name)` -> asset identifier.
    pub fonts: HashMap<(String, String), String>,
    /// StyleBox map: `(theme_type, item_name)` -> StyleBox.
    pub styleboxes: HashMap<(String, String), StyleBox>,
    /// Default fallback font size.
    pub default_font_size: u32,
    /// Default fallback font asset path.
    pub default_font: String,
}

impl Component for Theme {}

impl Theme {
    /// Registers a color attribute for a specific control type.
    pub fn set_color(&mut self, theme_type: &str, item: &str, color: [f32; 4]) {
        self.colors
            .insert((theme_type.to_string(), item.to_string()), color);
    }

    /// Queries a color attribute for a specific control type.
    #[must_use]
    pub fn get_color(&self, theme_type: &str, item: &str) -> Option<[f32; 4]> {
        self.colors
            .get(&(theme_type.to_string(), item.to_string()))
            .copied()
    }

    /// Registers a numeric constant for a specific control type.
    pub fn set_constant(&mut self, theme_type: &str, item: &str, value: i32) {
        self.constants
            .insert((theme_type.to_string(), item.to_string()), value);
    }

    /// Queries a numeric constant for a specific control type.
    #[must_use]
    pub fn get_constant(&self, theme_type: &str, item: &str) -> Option<i32> {
        self.constants
            .get(&(theme_type.to_string(), item.to_string()))
            .copied()
    }

    /// Registers a font size for a specific control type.
    pub fn set_font_size(&mut self, theme_type: &str, item: &str, size: u32) {
        self.font_sizes
            .insert((theme_type.to_string(), item.to_string()), size);
    }

    /// Queries a font size for a specific control type.
    #[must_use]
    pub fn get_font_size(&self, theme_type: &str, item: &str) -> Option<u32> {
        self.font_sizes
            .get(&(theme_type.to_string(), item.to_string()))
            .copied()
    }

    /// Registers a stylebox for a specific control type and state.
    pub fn set_stylebox(&mut self, theme_type: &str, item: &str, stylebox: StyleBox) {
        self.styleboxes
            .insert((theme_type.to_string(), item.to_string()), stylebox);
    }

    /// Queries a stylebox for a specific control type and state.
    #[must_use]
    pub fn get_stylebox(&self, theme_type: &str, item: &str) -> Option<&StyleBox> {
        self.styleboxes
            .get(&(theme_type.to_string(), item.to_string()))
    }

    /// Creates a modern dark theme preset with standard controls styled.
    #[must_use]
    pub fn dark() -> Self {
        let mut theme = Self {
            default_font_size: 14,
            default_font: "fonts/default.ttf".to_string(),
            ..Default::default()
        };

        // Button styling
        theme.set_color("Button", "font_color", [0.87, 0.87, 0.87, 1.0]);
        theme.set_color("Button", "font_hover_color", [1.0, 1.0, 1.0, 1.0]);
        theme.set_color("Button", "font_pressed_color", [1.0, 1.0, 1.0, 1.0]);
        theme.set_color("Button", "font_disabled_color", [0.4, 0.4, 0.4, 1.0]);
        theme.set_font_size("Button", "font_size", 14);

        let btn_normal = StyleBoxFlat {
            bg_color: [0.18, 0.18, 0.20, 1.0],
            border_color: [0.28, 0.28, 0.32, 1.0],
            border_width: [1.0; 4],
            corner_radius: [4.0; 4],
            content_margin: [8.0, 4.0, 8.0, 4.0],
            ..Default::default()
        };
        theme.set_stylebox("Button", "normal", StyleBox::Flat(btn_normal.clone()));

        let mut btn_hover = btn_normal.clone();
        btn_hover.bg_color = [0.24, 0.24, 0.28, 1.0];
        btn_hover.border_color = [0.38, 0.38, 0.44, 1.0];
        theme.set_stylebox("Button", "hover", StyleBox::Flat(btn_hover));

        let mut btn_pressed = btn_normal.clone();
        btn_pressed.bg_color = [0.12, 0.12, 0.14, 1.0];
        btn_pressed.border_color = [0.45, 0.45, 0.55, 1.0];
        theme.set_stylebox("Button", "pressed", StyleBox::Flat(btn_pressed));

        let mut btn_disabled = btn_normal;
        btn_disabled.bg_color = [0.14, 0.14, 0.15, 0.6];
        btn_disabled.border_color = [0.2, 0.2, 0.22, 0.6];
        theme.set_stylebox("Button", "disabled", StyleBox::Flat(btn_disabled));

        // Panel styling
        let panel = StyleBoxFlat {
            bg_color: [0.13, 0.13, 0.15, 1.0],
            border_color: [0.22, 0.22, 0.25, 1.0],
            border_width: [1.0; 4],
            corner_radius: [6.0; 4],
            content_margin: [8.0; 4],
            ..Default::default()
        };
        theme.set_stylebox("Panel", "panel", StyleBox::Flat(panel));

        // LineEdit styling
        let line_edit = StyleBoxFlat {
            bg_color: [0.10, 0.10, 0.12, 1.0],
            border_color: [0.25, 0.25, 0.28, 1.0],
            border_width: [1.0; 4],
            corner_radius: [4.0; 4],
            content_margin: [6.0, 4.0, 6.0, 4.0],
            ..Default::default()
        };
        theme.set_stylebox("LineEdit", "normal", StyleBox::Flat(line_edit.clone()));

        let mut line_edit_focus = line_edit;
        line_edit_focus.border_color = [0.35, 0.55, 0.90, 1.0];
        theme.set_stylebox("LineEdit", "focus", StyleBox::Flat(line_edit_focus));

        theme
    }
}

/// Component storing local, per-control visual attribute overrides.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ThemeOverride {
    /// Local color overrides: `item_name` -> RGBA.
    pub colors: HashMap<String, [f32; 4]>,
    /// Local constant overrides: `item_name` -> value.
    pub constants: HashMap<String, i32>,
    /// Local font size overrides: `item_name` -> size.
    pub font_sizes: HashMap<String, u32>,
    /// Local StyleBox overrides: `item_name` -> StyleBox.
    pub styleboxes: HashMap<String, StyleBox>,
}

impl Component for ThemeOverride {}

impl ThemeOverride {
    /// Sets a local color override.
    pub fn set_color(&mut self, item: &str, color: [f32; 4]) {
        self.colors.insert(item.to_string(), color);
    }

    /// Queries a local color override.
    #[must_use]
    pub fn get_color(&self, item: &str) -> Option<[f32; 4]> {
        self.colors.get(item).copied()
    }

    /// Sets a local constant override.
    pub fn set_constant(&mut self, item: &str, value: i32) {
        self.constants.insert(item.to_string(), value);
    }

    /// Queries a local constant override.
    #[must_use]
    pub fn get_constant(&self, item: &str) -> Option<i32> {
        self.constants.get(item).copied()
    }

    /// Sets a local font size override.
    pub fn set_font_size(&mut self, item: &str, size: u32) {
        self.font_sizes.insert(item.to_string(), size);
    }

    /// Queries a local font size override.
    #[must_use]
    pub fn get_font_size(&self, item: &str) -> Option<u32> {
        self.font_sizes.get(item).copied()
    }

    /// Sets a local stylebox override.
    pub fn set_stylebox(&mut self, item: &str, stylebox: StyleBox) {
        self.styleboxes.insert(item.to_string(), stylebox);
    }

    /// Queries a local stylebox override.
    #[must_use]
    pub fn get_stylebox(&self, item: &str) -> Option<&StyleBox> {
        self.styleboxes.get(item)
    }
}
