//! Unit tests for Godot-inspired Control Node layout foundation, anchors, offsets, presets, and ECS hierarchy resolution.

use korppi_ecs::World;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::control::{Control, FocusMode, LayoutPreset, MouseFilter};
use korppi_ui::layout::{
    resolve_control_rect, resolve_ui_layout_system, Rect2d, ResolvedRect, SizeFlags,
};

#[test]
fn test_control_default_state_and_presets() {
    let mut control = Control::default();
    assert_eq!(control.anchor_left, 0.0);
    assert_eq!(control.anchor_top, 0.0);
    assert_eq!(control.anchor_right, 0.0);
    assert_eq!(control.anchor_bottom, 0.0);
    assert_eq!(control.mouse_filter, MouseFilter::Stop);
    assert_eq!(control.focus_mode, FocusMode::None);

    // Preset: Full Rect
    control.set_anchors_preset(LayoutPreset::FullRect, false);
    assert_eq!(control.anchor_left, 0.0);
    assert_eq!(control.anchor_top, 0.0);
    assert_eq!(control.anchor_right, 1.0);
    assert_eq!(control.anchor_bottom, 1.0);

    // Preset: Center
    control.set_anchors_preset(LayoutPreset::Center, false);
    assert_eq!(control.anchor_left, 0.5);
    assert_eq!(control.anchor_top, 0.5);
    assert_eq!(control.anchor_right, 0.5);
    assert_eq!(control.anchor_bottom, 0.5);

    // Preset: Top Wide
    control.set_anchors_preset(LayoutPreset::TopWide, false);
    assert_eq!(control.anchor_left, 0.0);
    assert_eq!(control.anchor_top, 0.0);
    assert_eq!(control.anchor_right, 1.0);
    assert_eq!(control.anchor_bottom, 0.0);

    // Preset: Bottom Wide
    control.set_anchors_preset(LayoutPreset::BottomWide, false);
    assert_eq!(control.anchor_left, 0.0);
    assert_eq!(control.anchor_top, 1.0);
    assert_eq!(control.anchor_right, 1.0);
    assert_eq!(control.anchor_bottom, 1.0);
}

#[test]
fn test_resolve_control_rect_anchors_and_offsets() {
    let parent = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1920.0,
        height: 1080.0,
    };

    // 1. Full Rect with 10px margin padding
    let mut control = Control::default();
    control.set_anchors_preset(LayoutPreset::FullRect, false);
    control.offset_left = 10.0;
    control.offset_top = 20.0;
    control.offset_right = -10.0;
    control.offset_bottom = -20.0;

    let rect = resolve_control_rect(&control, parent);
    assert_eq!(rect.x, 10.0);
    assert_eq!(rect.y, 20.0);
    assert_eq!(rect.width, 1900.0);
    assert_eq!(rect.height, 1040.0);

    // 2. Centered dialog of size 400x300
    let mut center_dialog = Control::default();
    center_dialog.set_anchors_preset(LayoutPreset::Center, false);
    center_dialog.offset_left = -200.0;
    center_dialog.offset_top = -150.0;
    center_dialog.offset_right = 200.0;
    center_dialog.offset_bottom = 150.0;

    let rect_center = resolve_control_rect(&center_dialog, parent);
    assert_eq!(rect_center.x, 960.0 - 200.0);
    assert_eq!(rect_center.y, 540.0 - 150.0);
    assert_eq!(rect_center.width, 400.0);
    assert_eq!(rect_center.height, 300.0);

    // 3. Minimum size clamping
    let mut min_clamped = Control::default();
    min_clamped.set_anchors_preset(LayoutPreset::TopLeft, false);
    min_clamped.offset_right = 20.0; // requests width 20
    min_clamped.offset_bottom = 10.0; // requests height 10
    min_clamped.custom_minimum_size = Vec2::new(100.0, 50.0); // enforces min 100x50

    let rect_clamped = resolve_control_rect(&min_clamped, parent);
    assert_eq!(rect_clamped.width, 100.0);
    assert_eq!(rect_clamped.height, 50.0);
}

#[test]
fn test_rect2d_geometric_operations() {
    let r1 = Rect2d {
        x: 10.0,
        y: 10.0,
        width: 100.0,
        height: 50.0,
    };

    assert!(r1.contains_point(Vec2::new(15.0, 20.0)));
    assert!(r1.contains_point(Vec2::new(10.0, 10.0)));
    assert!(!r1.contains_point(Vec2::new(5.0, 20.0)));
    assert!(!r1.contains_point(Vec2::new(15.0, 65.0)));

    let r2 = Rect2d {
        x: 50.0,
        y: 20.0,
        width: 100.0,
        height: 100.0,
    };
    assert!(r1.intersects(&r2));

    let intersection = r1.intersection(&r2).expect("Intersection exists");
    assert_eq!(intersection.x, 50.0);
    assert_eq!(intersection.y, 20.0);
    assert_eq!(intersection.width, 60.0);
    assert_eq!(intersection.height, 40.0);
}

#[test]
fn test_size_flags_combinations() {
    let flags = SizeFlags::EXPAND | SizeFlags::FILL;
    assert!(flags.contains(SizeFlags::EXPAND));
    assert!(flags.contains(SizeFlags::FILL));
    assert!(!flags.contains(SizeFlags::SHRINK_CENTER));
}

#[test]
fn test_ecs_hierarchy_layout_resolution_system() {
    let mut world = World::new();

    // Root UI viewport / canvas
    let mut root_control = Control::default();
    root_control.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world.spawn((root_control,));

    // Child Panel: Center Top, size 600x100
    let mut panel_control = Control::default();
    panel_control.set_anchors_preset(LayoutPreset::CenterTop, false);
    panel_control.offset_left = -300.0;
    panel_control.offset_right = 300.0;
    panel_control.offset_top = 0.0;
    panel_control.offset_bottom = 100.0;
    let panel = world.spawn((panel_control,));
    attach_child(&mut world, root, panel).expect("Attach panel to root");

    // Child Button inside Panel: Centered 200x40
    let mut button_control = Control::default();
    button_control.set_anchors_preset(LayoutPreset::Center, false);
    button_control.offset_left = -100.0;
    button_control.offset_right = 100.0;
    button_control.offset_top = -20.0;
    button_control.offset_bottom = 20.0;
    let button = world.spawn((button_control,));
    attach_child(&mut world, panel, button).expect("Attach button to panel");

    // Execute layout resolution system for viewport 1920x1080
    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1920.0,
        height: 1080.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    // Verify root rect
    let root_resolved = world.get::<ResolvedRect>(root).expect("Root resolved");
    assert_eq!(root_resolved.global, screen_rect);

    // Verify panel rect: centered horizontally at x = 960 - 300 = 660, y = 0, size 600x100
    let panel_resolved = world.get::<ResolvedRect>(panel).expect("Panel resolved");
    assert_eq!(panel_resolved.global.x, 660.0);
    assert_eq!(panel_resolved.global.y, 0.0);
    assert_eq!(panel_resolved.global.width, 600.0);
    assert_eq!(panel_resolved.global.height, 100.0);

    // Verify button rect: centered inside panel (panel center is (960, 50))
    // button x = 960 - 100 = 860, y = 50 - 20 = 30, size 200x40
    let button_resolved = world.get::<ResolvedRect>(button).expect("Button resolved");
    assert_eq!(button_resolved.global.x, 860.0);
    assert_eq!(button_resolved.global.y, 30.0);
    assert_eq!(button_resolved.global.width, 200.0);
    assert_eq!(button_resolved.global.height, 40.0);
}
