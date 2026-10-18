//! Unit tests for Godot-inspired Theming & Skinning Engine:
//! StyleBoxFlat, StyleBoxTexture (9-patch), StyleBoxEmpty, Theme hierarchy resolution,
//! and per-control Theme overrides.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::control::Control;
use korppi_ui::layout::Rect2d;
use korppi_ui::theme::{
    get_theme_color, get_theme_constant, get_theme_stylebox, AxisStretchMode, StyleBox,
    StyleBoxEmpty, StyleBoxFlat, StyleBoxTexture, Theme, ThemeOverride,
};

#[test]
fn test_stylebox_flat_metrics_margins_and_draw_rect() {
    let mut flat = StyleBoxFlat::default();
    flat.bg_color = [0.2, 0.3, 0.4, 1.0];
    flat.border_color = [1.0, 1.0, 1.0, 1.0];
    flat.border_width = [2.0, 4.0, 2.0, 4.0]; // left, top, right, bottom
    flat.corner_radius = [8.0, 8.0, 8.0, 8.0];
    flat.expand_margin = [5.0, 10.0, 5.0, 10.0];
    flat.content_margin = [12.0, 8.0, 12.0, 8.0];

    let base_rect = Rect2d {
        x: 100.0,
        y: 100.0,
        width: 200.0,
        height: 50.0,
    };

    // Draw rect expanded by expand margins:
    // x = 100 - 5 = 95
    // y = 100 - 10 = 90
    // w = 200 + 5 + 5 = 210
    // h = 50 + 10 + 10 = 70
    let draw_rect = flat.get_draw_rect(base_rect);
    assert_eq!(draw_rect.x, 95.0);
    assert_eq!(draw_rect.y, 90.0);
    assert_eq!(draw_rect.width, 210.0);
    assert_eq!(draw_rect.height, 70.0);

    // Minimum size: content margins (12+12, 8+8) = (24, 16)
    assert_eq!(flat.get_minimum_size(), Vec2::new(24.0, 16.0));
}

#[test]
fn test_stylebox_texture_9patch_slicing_and_stretch() {
    let mut tex_box = StyleBoxTexture::default();
    tex_box.texture_margin = [8.0, 8.0, 8.0, 8.0]; // 8px corners
    tex_box.region_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
    };
    tex_box.axis_h = AxisStretchMode::Stretch;
    tex_box.axis_v = AxisStretchMode::Tile;
    tex_box.content_margin = [10.0, 10.0, 10.0, 10.0];

    assert_eq!(tex_box.get_minimum_size(), Vec2::new(20.0, 20.0));
    assert_eq!(tex_box.axis_h, AxisStretchMode::Stretch);
    assert_eq!(tex_box.axis_v, AxisStretchMode::Tile);
}

#[test]
fn test_stylebox_empty_metrics() {
    let mut empty = StyleBoxEmpty::default();
    empty.content_margin = [5.0, 5.0, 5.0, 5.0];

    let base_rect = Rect2d {
        x: 10.0,
        y: 20.0,
        width: 100.0,
        height: 50.0,
    };
    assert_eq!(empty.get_draw_rect(base_rect), base_rect);
    assert_eq!(empty.get_minimum_size(), Vec2::new(10.0, 10.0));
}

#[test]
fn test_theme_definition_and_value_queries() {
    let mut theme = Theme::default();

    // Button colors
    theme.set_color("Button", "font_color", [1.0, 1.0, 1.0, 1.0]);
    theme.set_color("Button", "font_hover_color", [1.0, 1.0, 0.0, 1.0]);

    // BoxContainer constant
    theme.set_constant("BoxContainer", "separation", 8);

    // Button StyleBox
    let mut btn_normal = StyleBoxFlat::default();
    btn_normal.bg_color = [0.1, 0.1, 0.1, 1.0];
    theme.set_stylebox("Button", "normal", StyleBox::Flat(btn_normal.clone()));

    assert_eq!(
        theme.get_color("Button", "font_color"),
        Some([1.0, 1.0, 1.0, 1.0])
    );
    assert_eq!(
        theme.get_color("Button", "font_hover_color"),
        Some([1.0, 1.0, 0.0, 1.0])
    );
    assert_eq!(theme.get_constant("BoxContainer", "separation"), Some(8));
    assert_eq!(
        theme.get_stylebox("Button", "normal"),
        Some(&StyleBox::Flat(btn_normal))
    );
}

#[test]
fn test_hierarchical_theme_cascade_and_local_overrides() {
    let mut world = World::new();

    // 1. Root Entity with Global Theme
    let mut root_theme = Theme::default();
    root_theme.set_color("Button", "font_color", [0.8, 0.8, 0.8, 1.0]);
    root_theme.set_constant("Panel", "padding", 16);
    let root = world.spawn((Control::default(), root_theme));

    // 2. Middle Panel with Sub-Theme override
    let mut sub_theme = Theme::default();
    sub_theme.set_color("Button", "font_color", [0.0, 1.0, 0.0, 1.0]); // green buttons in this panel
    let panel = world.spawn((Control::default(), sub_theme));
    attach_child(&mut world, root, panel).expect("Attach panel");

    // 3. Button A inside panel (inherits green font_color from sub_theme)
    let btn_a = world.spawn((Control::default(),));
    attach_child(&mut world, panel, btn_a).expect("Attach btn_a");

    // 4. Button B inside panel with local ThemeOverride (overrides font_color to red)
    let mut btn_b_override = ThemeOverride::default();
    btn_b_override.set_color("font_color", [1.0, 0.0, 0.0, 1.0]);
    let btn_b = world.spawn((Control::default(), btn_b_override));
    attach_child(&mut world, panel, btn_b).expect("Attach btn_b");

    // Query btn_a: font_color is green from panel's sub_theme
    let color_a = get_theme_color(&world, btn_a, "Button", "font_color");
    assert_eq!(color_a, Some([0.0, 1.0, 0.0, 1.0]));

    // Query btn_b: font_color is red from local ThemeOverride
    let color_b = get_theme_color(&world, btn_b, "Button", "font_color");
    assert_eq!(color_b, Some([1.0, 0.0, 0.0, 1.0]));

    // Query btn_a: padding is 16 from root_theme (ancestor cascade)
    let padding_a = get_theme_constant(&world, btn_a, "Panel", "padding");
    assert_eq!(padding_a, Some(16));

    // Query btn_a: button normal stylebox falls back to default dark theme
    let sb_a = get_theme_stylebox(&world, btn_a, "Button", "normal");
    assert!(sb_a.is_some());
}

#[test]
fn test_default_dark_theme_presets() {
    let dark = Theme::dark();
    assert!(dark.get_color("Button", "font_color").is_some());
    assert!(dark.get_stylebox("Button", "normal").is_some());
    assert!(dark.get_stylebox("Button", "hover").is_some());
    assert!(dark.get_stylebox("Button", "pressed").is_some());
    assert!(dark.get_stylebox("Panel", "panel").is_some());
}
