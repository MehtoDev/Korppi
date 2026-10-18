//! Unit tests for Godot-inspired Container suite:
//! BoxContainer (VBox/HBox), GridContainer, MarginContainer, PanelContainer,
//! ScrollContainer, CenterContainer, AspectRatioContainer, and SplitContainer.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::containers::{
    AspectRatioContainer, AspectStretchMode, BoxAlignment, CenterContainer, GridContainer,
    HBoxContainer, HSplitContainer, MarginContainer, PanelContainer, ScrollContainer,
    VBoxContainer,
};
use korppi_ui::control::{Control, LayoutPreset};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect, SizeFlags};

#[test]
fn test_hbox_container_linear_layout_and_expansion() {
    let mut world = World::new();

    // HBox root of size 800x100
    let mut hbox_ctl = Control::default();
    hbox_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let hbox = world.spawn((hbox_ctl, HBoxContainer::new(10.0))); // 10px separation

    // Child 1: Fixed 100px width (no EXPAND)
    let mut c1_ctl = Control::default();
    c1_ctl.custom_minimum_size = Vec2::new(100.0, 50.0);
    c1_ctl.size_flags_horizontal = SizeFlags::FILL; // no expand
    let c1 = world.spawn((c1_ctl,));
    attach_child(&mut world, hbox, c1).expect("Attach c1");

    // Child 2: Expanding (stretch ratio 1.0)
    let mut c2_ctl = Control::default();
    c2_ctl.custom_minimum_size = Vec2::new(100.0, 50.0);
    c2_ctl.size_flags_horizontal = SizeFlags::EXPAND_FILL;
    c2_ctl.stretch_ratio = 1.0;
    let c2 = world.spawn((c2_ctl,));
    attach_child(&mut world, hbox, c2).expect("Attach c2");

    // Child 3: Expanding (stretch ratio 2.0)
    let mut c3_ctl = Control::default();
    c3_ctl.custom_minimum_size = Vec2::new(100.0, 50.0);
    c3_ctl.size_flags_horizontal = SizeFlags::EXPAND_FILL;
    c3_ctl.stretch_ratio = 2.0;
    let c3 = world.spawn((c3_ctl,));
    attach_child(&mut world, hbox, c3).expect("Attach c3");

    // Total width = 800. Fixed = 100. Separations = 2 * 10 = 20.
    // Remaining space for expanding children = 800 - 100 - 20 = 680.
    // Child 2 gets 1/3 of 680 = 226.666...
    // Child 3 gets 2/3 of 680 = 453.333...

    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 100.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    let r1 = world.get::<ResolvedRect>(c1).unwrap().global;
    let r2 = world.get::<ResolvedRect>(c2).unwrap().global;
    let r3 = world.get::<ResolvedRect>(c3).unwrap().global;

    assert_eq!(r1.x, 0.0);
    assert_eq!(r1.width, 100.0);
    assert_eq!(r1.height, 100.0); // SizeFlags::FILL fills cross axis height

    assert_eq!(r2.x, 110.0); // 100 + 10 sep
    assert!((r2.width - 226.666).abs() < 1.0);
    assert_eq!(r2.height, 100.0);

    assert!((r3.x - (110.0 + r2.width + 10.0)).abs() < 1.0);
    assert!((r3.width - 453.333).abs() < 1.0);
    assert_eq!(r3.height, 100.0);
}

#[test]
fn test_vbox_container_linear_layout_and_alignment() {
    let mut world = World::new();

    // VBox root of size 200x500 with Center alignment
    let mut vbox_ctl = Control::default();
    vbox_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let mut vbox_comp = VBoxContainer::new(20.0);
    vbox_comp.alignment = BoxAlignment::Center;
    let vbox = world.spawn((vbox_ctl, vbox_comp));

    // Two children of height 50 each
    let mut c1_ctl = Control::default();
    c1_ctl.custom_minimum_size = Vec2::new(150.0, 50.0);
    c1_ctl.size_flags_vertical = SizeFlags::SHRINK_CENTER;
    let c1 = world.spawn((c1_ctl,));
    attach_child(&mut world, vbox, c1).expect("Attach c1");

    let mut c2_ctl = Control::default();
    c2_ctl.custom_minimum_size = Vec2::new(150.0, 50.0);
    c2_ctl.size_flags_vertical = SizeFlags::SHRINK_CENTER;
    let c2 = world.spawn((c2_ctl,));
    attach_child(&mut world, vbox, c2).expect("Attach c2");

    // Total content height = 50 + 20 + 50 = 120. Total container = 500.
    // Centered start y = (500 - 120) / 2 = 190.
    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 200.0,
        height: 500.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    let r1 = world.get::<ResolvedRect>(c1).unwrap().global;
    let r2 = world.get::<ResolvedRect>(c2).unwrap().global;

    assert_eq!(r1.y, 190.0);
    assert_eq!(r1.height, 50.0);
    assert_eq!(r2.y, 190.0 + 50.0 + 20.0);
    assert_eq!(r2.height, 50.0);
}

#[test]
fn test_grid_container_multi_column_flow() {
    let mut world = World::new();

    // 2-column GridContainer inside 400x400
    let mut grid_ctl = Control::default();
    grid_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let grid_comp = GridContainer::new(2, 10.0, 20.0); // 2 cols, 10px h_sep, 20px v_sep
    let grid = world.spawn((grid_ctl, grid_comp));

    // 4 items:
    // Item 0: col 0, row 0 (size 100x40)
    // Item 1: col 1, row 0 (size 150x60)
    // Item 2: col 0, row 1 (size 120x50)
    // Item 3: col 1, row 1 (size 80x30)
    let mut c0_ctl = Control::default();
    c0_ctl.custom_minimum_size = Vec2::new(100.0, 40.0);
    let c0 = world.spawn((c0_ctl,));
    attach_child(&mut world, grid, c0).expect("Attach c0");

    let mut c1_ctl = Control::default();
    c1_ctl.custom_minimum_size = Vec2::new(150.0, 60.0);
    let c1 = world.spawn((c1_ctl,));
    attach_child(&mut world, grid, c1).expect("Attach c1");

    let mut c2_ctl = Control::default();
    c2_ctl.custom_minimum_size = Vec2::new(120.0, 50.0);
    let c2 = world.spawn((c2_ctl,));
    attach_child(&mut world, grid, c2).expect("Attach c2");

    let mut c3_ctl = Control::default();
    c3_ctl.custom_minimum_size = Vec2::new(80.0, 30.0);
    let c3 = world.spawn((c3_ctl,));
    attach_child(&mut world, grid, c3).expect("Attach c3");

    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 400.0,
        height: 400.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    // Col 0 width = max(100, 120) = 120.
    // Col 1 width = max(150, 80) = 150.
    // Row 0 height = max(40, 60) = 60.
    // Row 1 height = max(50, 30) = 50.

    let r0 = world.get::<ResolvedRect>(c0).unwrap().global;
    let r1 = world.get::<ResolvedRect>(c1).unwrap().global;
    let r2 = world.get::<ResolvedRect>(c2).unwrap().global;
    let r3 = world.get::<ResolvedRect>(c3).unwrap().global;

    // Col 0, Row 0:
    assert_eq!(r0.x, 0.0);
    assert_eq!(r0.y, 0.0);
    assert_eq!(r0.width, 120.0);
    assert_eq!(r0.height, 60.0);

    // Col 1, Row 0:
    assert_eq!(r1.x, 120.0 + 10.0);
    assert_eq!(r1.y, 0.0);
    assert_eq!(r1.width, 150.0);
    assert_eq!(r1.height, 60.0);

    // Col 0, Row 1:
    assert_eq!(r2.x, 0.0);
    assert_eq!(r2.y, 60.0 + 20.0);
    assert_eq!(r2.width, 120.0);
    assert_eq!(r2.height, 50.0);

    // Col 1, Row 1:
    assert_eq!(r3.x, 120.0 + 10.0);
    assert_eq!(r3.y, 60.0 + 20.0);
    assert_eq!(r3.width, 150.0);
    assert_eq!(r3.height, 50.0);
}

#[test]
fn test_margin_and_panel_containers() {
    let mut world = World::new();

    // MarginContainer with (left: 15, top: 25, right: 35, bottom: 45)
    let mut margin_ctl = Control::default();
    margin_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let margin_comp = MarginContainer::new(15.0, 25.0, 35.0, 45.0);
    let margin = world.spawn((margin_ctl, margin_comp));

    // Child filling the margin container
    let mut child_ctl = Control::default();
    child_ctl.size_flags_horizontal = SizeFlags::FILL;
    child_ctl.size_flags_vertical = SizeFlags::FILL;
    let child = world.spawn((child_ctl,));
    attach_child(&mut world, margin, child).expect("Attach child to margin");

    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 500.0,
        height: 300.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    let child_rect = world.get::<ResolvedRect>(child).unwrap().global;
    assert_eq!(child_rect.x, 15.0);
    assert_eq!(child_rect.y, 25.0);
    assert_eq!(child_rect.width, 500.0 - 15.0 - 35.0);
    assert_eq!(child_rect.height, 300.0 - 25.0 - 45.0);

    // PanelContainer with uniform 8px padding
    let mut panel_ctl = Control::default();
    panel_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let panel_comp = PanelContainer::all(8.0);
    let panel = world.spawn((panel_ctl, panel_comp));

    let panel_child_ctl = Control::default();
    let panel_child = world.spawn((panel_child_ctl,));
    attach_child(&mut world, panel, panel_child).expect("Attach panel child");

    resolve_ui_layout_system(&mut world, screen_rect);
    let p_rect = world.get::<ResolvedRect>(panel_child).unwrap().global;
    assert_eq!(p_rect.x, 8.0);
    assert_eq!(p_rect.y, 8.0);
    assert_eq!(p_rect.width, 500.0 - 16.0);
    assert_eq!(p_rect.height, 300.0 - 16.0);
}

#[test]
fn test_scroll_container_viewport_and_clamping() {
    let mut world = World::new();

    // ScrollContainer of size 300x200
    let mut scroll_ctl = Control::default();
    scroll_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let mut scroll_comp = ScrollContainer::default();
    scroll_comp.scroll_horizontal = 150.0;
    scroll_comp.scroll_vertical = 100.0;
    let scroll = world.spawn((scroll_ctl, scroll_comp));

    // Inner content of size 1000x800
    let mut content_ctl = Control::default();
    content_ctl.custom_minimum_size = Vec2::new(1000.0, 800.0);
    let content = world.spawn((content_ctl,));
    attach_child(&mut world, scroll, content).expect("Attach content to scroll");

    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 300.0,
        height: 200.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    let content_rect = world.get::<ResolvedRect>(content).unwrap().global;
    assert_eq!(content_rect.x, -150.0);
    assert_eq!(content_rect.y, -100.0);
    assert_eq!(content_rect.width, 1000.0);
    assert_eq!(content_rect.height, 800.0);

    // Verify container enforces clip_contents
    let updated_scroll_ctl = world.get::<Control>(scroll).unwrap();
    assert!(updated_scroll_ctl.clip_contents);
}

#[test]
fn test_center_and_aspect_ratio_containers() {
    let mut world = World::new();

    // CenterContainer inside 600x400
    let mut center_ctl = Control::default();
    center_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let center = world.spawn((center_ctl, CenterContainer::default()));

    let mut child_ctl = Control::default();
    child_ctl.custom_minimum_size = Vec2::new(200.0, 100.0);
    let child = world.spawn((child_ctl,));
    attach_child(&mut world, center, child).expect("Attach child");

    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 600.0,
        height: 400.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    let child_rect = world.get::<ResolvedRect>(child).unwrap().global;
    assert_eq!(child_rect.x, (600.0 - 200.0) / 2.0);
    assert_eq!(child_rect.y, (400.0 - 100.0) / 2.0);
    assert_eq!(child_rect.width, 200.0);
    assert_eq!(child_rect.height, 100.0);

    // AspectRatioContainer 16:9 Fit inside 800x600
    let mut arc_ctl = Control::default();
    arc_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let arc_comp = AspectRatioContainer::new(16.0 / 9.0, AspectStretchMode::Fit);
    let arc = world.spawn((arc_ctl, arc_comp));

    let mut arc_child_ctl = Control::default();
    arc_child_ctl.size_flags_horizontal = SizeFlags::FILL;
    arc_child_ctl.size_flags_vertical = SizeFlags::FILL;
    let arc_child = world.spawn((arc_child_ctl,));
    attach_child(&mut world, arc, arc_child).expect("Attach arc child");

    let screen_800x600 = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen_800x600);

    // With 16:9 in 800x600:
    // Scale by width: 800 / 16 * 9 = 450 height <= 600. Fits!
    // Centered vertically: (600 - 450) / 2 = 75.
    let arc_child_rect = world.get::<ResolvedRect>(arc_child).unwrap().global;
    assert_eq!(arc_child_rect.x, 0.0);
    assert_eq!(arc_child_rect.y, 75.0);
    assert_eq!(arc_child_rect.width, 800.0);
    assert_eq!(arc_child_rect.height, 450.0);
}

#[test]
fn test_split_container_dragger_partitioning() {
    let mut world = World::new();

    // HSplitContainer inside 1000x500 with 10px dragger
    let mut split_ctl = Control::default();
    split_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let mut split_comp = HSplitContainer::new(10.0);
    split_comp.split_offset = 100.0; // shifts split by +100px to the right
    let split = world.spawn((split_ctl, split_comp));

    // Left child
    let mut left_ctl = Control::default();
    left_ctl.custom_minimum_size = Vec2::new(100.0, 100.0);
    let left = world.spawn((left_ctl,));
    attach_child(&mut world, split, left).expect("Attach left");

    // Right child
    let mut right_ctl = Control::default();
    right_ctl.custom_minimum_size = Vec2::new(100.0, 100.0);
    let right = world.spawn((right_ctl,));
    attach_child(&mut world, split, right).expect("Attach right");

    // Total width = 1000. Dragger = 10. Available = 990.
    // Default 50% split = 495.
    // With split_offset +100: Left gets 495 + 100 = 595.
    // Right gets 495 - 100 = 395.
    let screen_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1000.0,
        height: 500.0,
    };
    resolve_ui_layout_system(&mut world, screen_rect);

    let left_rect = world.get::<ResolvedRect>(left).unwrap().global;
    let right_rect = world.get::<ResolvedRect>(right).unwrap().global;

    assert_eq!(left_rect.x, 0.0);
    assert_eq!(left_rect.width, 595.0);
    assert_eq!(left_rect.height, 500.0);

    assert_eq!(right_rect.x, 595.0 + 10.0);
    assert_eq!(right_rect.width, 395.0);
    assert_eq!(right_rect.height, 500.0);
}
