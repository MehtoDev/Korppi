//! Regression tests for Milestone FIX-14.8 (`crates/ui` remediation: UI-01 through UI-08).

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_input::keyboard::KeyCode;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::containers::{
    AspectRatioContainer, AspectStretchMode, CenterContainer, DraggerVisibility, GridContainer,
    HBoxContainer, HSplitContainer, MarginContainer, PanelContainer, ScrollContainer, ScrollMode,
};
use korppi_ui::control::{Control, FocusMode, GrowDirection, LayoutPreset};
use korppi_ui::input::{
    collect_front_to_back_order, hit_test_controls, FocusDirection, UiEvent, UiInputDispatcher,
    UiReceiver,
};
use korppi_ui::layout::{
    resolve_control_rect, resolve_ui_layout_system, Rect2d, ResolvedRect, SizeFlags,
};
use korppi_ui::render::{UiCanvasRenderer, UiDrawCommand};
use korppi_ui::theme::{
    get_theme_color, get_theme_constant, get_theme_font_size, get_theme_stylebox, StyleBox,
    StyleBoxFlat, StyleBoxTexture, Theme, ThemeOverride,
};
use korppi_ui::widgets::{
    Button, ButtonState, CheckBox, CheckButton, ColorPicker, ItemList, Label, LineEdit, MenuBar,
    OptionButton, PopupMenu, Range, TextEdit, Tree,
};

#[test]
fn ui_01_bottom_up_minimum_size_propagation() {
    // 1. CenterContainer -> HBoxContainer -> [Button(100x40), Button(120x40)]
    let mut world = World::new();

    let mut center_ctl = Control::default();
    center_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let center = world.spawn((center_ctl, CenterContainer::default()));

    let hbox = world.spawn((Control::default(), HBoxContainer::new(10.0)));
    attach_child(&mut world, center, hbox).unwrap();

    let btn_a = world.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 40.0),
            ..Default::default()
        },
        Button::new("Save"),
    ));
    let btn_b = world.spawn((
        Control {
            custom_minimum_size: Vec2::new(120.0, 40.0),
            ..Default::default()
        },
        Button::new("Cancel"),
    ));
    attach_child(&mut world, hbox, btn_a).unwrap();
    attach_child(&mut world, hbox, btn_b).unwrap();

    resolve_ui_layout_system(&mut world, Rect2d::new(0.0, 0.0, 800.0, 600.0));

    let hbox_rect = world.get::<ResolvedRect>(hbox).unwrap().global;
    let a_rect = world.get::<ResolvedRect>(btn_a).unwrap().global;
    let b_rect = world.get::<ResolvedRect>(btn_b).unwrap().global;

    assert_eq!(hbox_rect, Rect2d::new(285.0, 280.0, 230.0, 40.0));
    assert_eq!(a_rect, Rect2d::new(285.0, 280.0, 100.0, 40.0));
    assert_eq!(b_rect, Rect2d::new(395.0, 280.0, 120.0, 40.0));

    // 2. MarginContainer -> HBoxContainer -> Child
    let mut world2 = World::new();
    let mut root_center = Control::default();
    root_center.set_anchors_preset(LayoutPreset::FullRect, false);
    let rc = world2.spawn((root_center, CenterContainer::default()));
    let margin = world2.spawn((
        Control::default(),
        MarginContainer::new(10.0, 20.0, 10.0, 20.0),
    ));
    let inner = world2.spawn((Control {
        custom_minimum_size: Vec2::new(80.0, 30.0),
        ..Default::default()
    },));
    attach_child(&mut world2, rc, margin).unwrap();
    attach_child(&mut world2, margin, inner).unwrap();
    resolve_ui_layout_system(&mut world2, Rect2d::new(0.0, 0.0, 400.0, 200.0));
    let margin_rect = world2.get::<ResolvedRect>(margin).unwrap().global;
    assert_eq!(margin_rect.width, 100.0);
    assert_eq!(margin_rect.height, 70.0);

    // 3. Label with StyleBoxFlat content_margin has non-zero intrinsic minimum size
    let mut world3 = World::new();
    let mut ov = ThemeOverride::default();
    ov.set_stylebox(
        "panel",
        StyleBox::Flat(StyleBoxFlat {
            content_margin: [16.0, 12.0, 16.0, 12.0],
            ..Default::default()
        }),
    );
    let styled_label = world3.spawn((Control::default(), Label::new("Hello World"), ov));
    resolve_ui_layout_system(&mut world3, Rect2d::new(0.0, 0.0, 800.0, 600.0));
    let lbl_rect = world3.get::<ResolvedRect>(styled_label).unwrap().global;
    assert!(lbl_rect.width >= 32.0);
    assert!(lbl_rect.height >= 24.0);
}

#[test]
fn ui_02_grow_direction_pixel_snapping_and_z_index() {
    let parent = Rect2d::new(0.0, 0.0, 500.0, 400.0);

    // GrowDirection::Begin
    let mut c_begin = Control::default();
    c_begin.set_anchors_preset(LayoutPreset::TopRight, false);
    c_begin.offset_left = -20.0;
    c_begin.offset_right = 0.0;
    c_begin.custom_minimum_size = Vec2::new(120.0, 40.0);
    c_begin.grow_horizontal = GrowDirection::Begin;
    let r_begin = resolve_control_rect(&c_begin, parent);
    assert_eq!(r_begin.x, 380.0);
    assert_eq!(r_begin.width, 120.0);

    // GrowDirection::Both
    let mut c_both = Control::default();
    c_both.set_anchors_preset(LayoutPreset::Center, false);
    c_both.custom_minimum_size = Vec2::new(200.0, 80.0);
    c_both.grow_horizontal = GrowDirection::Both;
    c_both.grow_vertical = GrowDirection::Both;
    let r_both = resolve_control_rect(&c_both, parent);
    assert_eq!(r_both.x, 150.0);
    assert_eq!(r_both.y, 160.0);
    assert_eq!(r_both.width, 200.0);
    assert_eq!(r_both.height, 80.0);

    // HBox 100px / 3 children -> integer pixel snapping without sub-pixel seams
    let mut world = World::new();
    let mut hbox_ctl = Control::default();
    hbox_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let hbox = world.spawn((hbox_ctl, HBoxContainer::new(0.0)));
    let mut children = Vec::new();
    for _ in 0..3 {
        let e = world.spawn((Control {
            size_flags_horizontal: SizeFlags::EXPAND_FILL,
            ..Default::default()
        },));
        attach_child(&mut world, hbox, e).unwrap();
        children.push(e);
    }
    resolve_ui_layout_system(&mut world, Rect2d::new(0.0, 0.0, 100.0, 30.0));
    let r0 = world.get::<ResolvedRect>(children[0]).unwrap().global;
    let r1 = world.get::<ResolvedRect>(children[1]).unwrap().global;
    let r2 = world.get::<ResolvedRect>(children[2]).unwrap().global;
    assert_eq!(r0.width.fract(), 0.0);
    assert_eq!(r1.x.fract(), 0.0);
    assert_eq!(r1.width.fract(), 0.0);
    assert_eq!(r2.x.fract(), 0.0);
    assert_eq!(r2.width.fract(), 0.0);
    assert_eq!(r0.width + r1.width + r2.width, 100.0);

    // Deterministic root ordering independent of archetype & z_index support
    let mut world_z = World::new();
    let mut a_ctl = Control::default();
    a_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let modal_a = world_z.spawn((a_ctl, Button::new("Modal A"), ThemeOverride::default()));

    let mut b_ctl = Control::default();
    b_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let backdrop_b = world_z.spawn((b_ctl,));

    resolve_ui_layout_system(&mut world_z, Rect2d::new(0.0, 0.0, 400.0, 300.0));

    // With equal z_index=0, backdrop_b (spawned second) is in front of modal_a (spawned first)
    let ftb = collect_front_to_back_order(&world_z);
    assert_eq!(ftb, vec![backdrop_b, modal_a]);
    let hits = hit_test_controls(&world_z, Vec2::new(100.0, 100.0));
    assert_eq!(hits, vec![backdrop_b]);

    // Raise modal_a z_index above backdrop_b
    world_z.get_mut::<Control>(modal_a).unwrap().z_index = 10;
    let ftb_elevated = collect_front_to_back_order(&world_z);
    assert_eq!(ftb_elevated, vec![modal_a, backdrop_b]);
    let hits_elevated = hit_test_controls(&world_z, Vec2::new(100.0, 100.0));
    assert_eq!(hits_elevated, vec![modal_a]);
}

#[test]
fn ui_03_visibility_propagation_and_focus_cleanup() {
    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::new();

    let mut parent_ctl = Control::default();
    parent_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let parent = world.spawn((parent_ctl,));

    let mut btn_ctl = Control::default();
    btn_ctl.offset_right = 200.0;
    btn_ctl.offset_bottom = 50.0;
    btn_ctl.focus_mode = FocusMode::All;
    let btn = world.spawn((btn_ctl, Button::new("Secret"), UiReceiver::default()));
    attach_child(&mut world, parent, btn).unwrap();

    let mut btn2_ctl = Control::default();
    btn2_ctl.offset_left = 300.0;
    btn2_ctl.offset_right = 500.0;
    btn2_ctl.offset_bottom = 50.0;
    btn2_ctl.focus_mode = FocusMode::All;
    let btn2 = world.spawn((btn2_ctl, Button::new("Visible"), UiReceiver::default()));

    let screen = Rect2d::new(0.0, 0.0, 800.0, 600.0);
    resolve_ui_layout_system(&mut world, screen);

    dispatcher.focus_manager.set_focus(&mut world, Some(btn));
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(btn));

    // Hide parent
    world.get_mut::<Control>(parent).unwrap().visible = false;
    resolve_ui_layout_system(&mut world, screen);
    world.get_mut::<UiReceiver>(btn).unwrap().clear();

    // Hidden child must not be hit-tested
    assert!(hit_test_controls(&world, Vec2::new(50.0, 25.0)).is_empty());

    // Hidden child must drop focus and not receive key events
    dispatcher.inject_key(&mut world, KeyCode::Enter, true);
    assert_eq!(dispatcher.focus_manager.focused_control(), None);
    assert!(world.get::<UiReceiver>(btn).unwrap().events.is_empty());

    // Tab navigation must skip hidden child and focus btn2
    dispatcher.inject_tab(&mut world, false);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(btn2));

    // Directional navigation from btn2 must not land on hidden btn
    dispatcher
        .focus_manager
        .navigate_direction(&mut world, FocusDirection::Left);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(btn2));

    // Despawning focused control must clear focus on next interaction
    world.despawn(btn2);
    dispatcher.inject_key(&mut world, KeyCode::Space, true);
    assert_eq!(dispatcher.focus_manager.focused_control(), None);
}

#[test]
fn ui_04_stylebox_edges_draw_center_modulate_clipping_and_widgets() {
    let renderer = UiCanvasRenderer::new();

    // 1. StyleBoxFlat left border only + draw_center = false
    let mut flat = StyleBoxFlat::default();
    flat.border_width = [10.0, 0.0, 0.0, 0.0]; // [left, top, right, bottom]
    flat.draw_center = false;
    let flat_batch = renderer.compile_to_quad_batch(
        &[UiDrawCommand::StyleBoxFlat {
            rect: Rect2d::new(0.0, 0.0, 100.0, 50.0),
            style: flat,
            scissor: None,
        }],
        64,
    );
    assert_eq!(flat_batch.quad_count(), 1);
    let v = flat_batch.vertices();
    assert_eq!(
        (
            v[2].position[0] - v[0].position[0],
            v[2].position[1] - v[0].position[1]
        ),
        (10.0, 50.0)
    );

    // 2. StyleBoxTexture left margin + draw_center = false + modulate
    let mut tex = StyleBoxTexture::default();
    tex.region_rect = Rect2d::new(0.0, 0.0, 64.0, 64.0);
    tex.texture_margin = [16.0, 0.0, 0.0, 0.0]; // [left, top, right, bottom]
    tex.draw_center = false;
    tex.modulate = [1.0, 0.2, 0.4, 0.5];
    let tex_batch = renderer.compile_to_quad_batch(
        &[UiDrawCommand::StyleBoxTexture {
            rect: Rect2d::new(0.0, 0.0, 100.0, 50.0),
            style: tex,
            scissor: None,
        }],
        64,
    );
    assert_eq!(tex_batch.quad_count(), 8);
    let tv = tex_batch.vertices();
    let tl_size = Vec2::new(
        tv[2].position[0] - tv[0].position[0],
        tv[2].position[1] - tv[0].position[1],
    );
    assert_eq!(tl_size, Vec2::new(16.0, 0.0));
    assert_eq!(tv[0].color, [1.0, 0.2, 0.4, 0.5]);

    // 3. Disjoint nested scissor culling and quad clipping
    let mut world = World::new();
    let outer = world.spawn((Control {
        offset_right: 200.0,
        offset_bottom: 200.0,
        clip_contents: true,
        ..Default::default()
    },));
    let inner = world.spawn((Control {
        offset_left: 500.0,
        offset_top: 500.0,
        offset_right: 600.0,
        offset_bottom: 600.0,
        clip_contents: true,
        ..Default::default()
    },));
    let btn = world.spawn((
        Control {
            offset_right: 80.0,
            offset_bottom: 30.0,
            ..Default::default()
        },
        Button::new("ClippedOut"),
    ));
    attach_child(&mut world, outer, inner).unwrap();
    attach_child(&mut world, inner, btn).unwrap();
    resolve_ui_layout_system(&mut world, Rect2d::new(0.0, 0.0, 1024.0, 768.0));
    let cmds = renderer.collect_draw_commands(&world);
    assert!(cmds.is_empty());

    // 4. Theme cascade for PanelContainer & Label + 8 widgets draw commands
    let mut world2 = World::new();
    let mut theme = Theme::dark();
    theme.set_color("Label", "font_color", [0.1, 0.8, 0.2, 1.0]);
    let mut custom_panel = StyleBoxFlat::default();
    custom_panel.bg_color = [0.9, 0.1, 0.1, 1.0];
    theme.set_stylebox("Panel", "panel", StyleBox::Flat(custom_panel));

    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world2.spawn((root_ctl, theme));

    let mut panel_ctl = Control::default();
    panel_ctl.offset_right = 300.0;
    panel_ctl.offset_bottom = 200.0;
    let panel = world2.spawn((panel_ctl, PanelContainer::all(8.0)));
    attach_child(&mut world2, root, panel).unwrap();

    let lbl = world2.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        Label::new("Themed Text"),
    ));
    attach_child(&mut world2, panel, lbl).unwrap();

    resolve_ui_layout_system(&mut world2, Rect2d::new(0.0, 0.0, 800.0, 600.0));
    let themed_cmds = renderer.collect_draw_commands(&world2);
    assert!(themed_cmds
        .iter()
        .any(|c| matches!(c, UiDrawCommand::StyleBoxFlat { style, .. } if style.bg_color == [0.9, 0.1, 0.1, 1.0])));
    assert!(themed_cmds
        .iter()
        .any(|c| matches!(c, UiDrawCommand::Text { color, .. } if *color == [0.1, 0.8, 0.2, 1.0])));

    // 5. All 8 previously missing widgets emit draw commands
    let mut world3 = World::new();
    let mut r_ctl = Control::default();
    r_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let r = world3.spawn((r_ctl, korppi_ui::containers::VBoxContainer::new(4.0)));

    let c1 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        CheckButton::new("Switch"),
    ));
    let mut te = TextEdit::new();
    te.set_text("Code line");
    let c2 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        te,
    ));
    let mut tree = Tree::new();
    tree.create_item(None, "RootNode");
    let c3 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        tree,
    ));
    let mut il = ItemList::new();
    il.add_item("Item 0");
    let c4 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        il,
    ));
    let mut pm = PopupMenu::new();
    pm.visible = true;
    pm.add_item(1, "Open");
    let c5 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        pm.clone(),
    ));
    let mut mb = MenuBar::new();
    mb.add_menu("File", pm);
    let c6 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        mb,
    ));
    let mut ob = OptionButton::new();
    ob.add_item(0, "Vulkan");
    let c7 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        ob,
    ));
    let c8 = world3.spawn((
        Control {
            custom_minimum_size: Vec2::new(100.0, 30.0),
            ..Default::default()
        },
        ColorPicker::new(),
    ));
    for c in [c1, c2, c3, c4, c5, c6, c7, c8] {
        attach_child(&mut world3, r, c).unwrap();
    }
    resolve_ui_layout_system(&mut world3, Rect2d::new(0.0, 0.0, 800.0, 600.0));
    let widget_cmds = renderer.collect_draw_commands(&world3);
    assert!(widget_cmds.len() >= 8);
}

#[test]
fn ui_05_button_state_machine_and_dispatcher_widget_forwarding() {
    // 1. Press-drag-out-release aborts click
    let mut btn = Button::new("Submit");
    btn.handle_event(&UiEvent::MouseEnter);
    btn.handle_event(&UiEvent::MouseButton {
        position: Vec2::new(10.0, 10.0),
        button: MouseButton::Left,
        pressed: true,
    });
    btn.handle_event(&UiEvent::MouseExit);
    assert_eq!(btn.state, ButtonState::Normal);
    let clicked_after_abort = btn.handle_event(&UiEvent::MouseButton {
        position: Vec2::new(500.0, 500.0),
        button: MouseButton::Left,
        pressed: false,
    });
    assert!(!clicked_after_abort);

    // 2. Toggle button visual state and keyboard Enter/Space activation
    let mut toggle_btn = Button::new("Latch");
    toggle_btn.toggle_mode = true;
    toggle_btn.set_pressed(true);
    assert_eq!(toggle_btn.state, ButtonState::Pressed);
    let key_activated = toggle_btn.handle_event(&UiEvent::KeyDown {
        key: KeyCode::Enter,
    });
    assert!(key_activated);
    assert!(!toggle_btn.is_pressed());

    // 3. UiInputDispatcher forwards events to widgets and clears focus on empty click
    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::new();

    let btn_e = world.spawn((
        Control {
            offset_right: 100.0,
            offset_bottom: 40.0,
            focus_mode: FocusMode::All,
            ..Default::default()
        },
        Button::new("OK"),
        UiReceiver::default(),
    ));
    let le_e = world.spawn((
        Control {
            offset_left: 120.0,
            offset_right: 320.0,
            offset_bottom: 40.0,
            focus_mode: FocusMode::All,
            ..Default::default()
        },
        LineEdit::new(),
        UiReceiver::default(),
    ));
    let cb_e = world.spawn((
        Control {
            offset_left: 340.0,
            offset_right: 440.0,
            offset_bottom: 40.0,
            focus_mode: FocusMode::All,
            ..Default::default()
        },
        CheckBox::new("Check"),
        UiReceiver::default(),
    ));
    resolve_ui_layout_system(&mut world, Rect2d::new(0.0, 0.0, 800.0, 600.0));

    dispatcher.inject_mouse_button(&mut world, Vec2::new(50.0, 20.0), MouseButton::Left, true);
    assert_eq!(
        world.get::<Button>(btn_e).unwrap().state,
        ButtonState::Pressed
    );
    dispatcher.inject_mouse_button(&mut world, Vec2::new(50.0, 20.0), MouseButton::Left, false);

    dispatcher.inject_mouse_click(&mut world, Vec2::new(150.0, 20.0), MouseButton::Left);
    for ch in ['a', 'b', 'c'] {
        dispatcher.inject_char(&mut world, ch);
    }
    assert_eq!(world.get::<LineEdit>(le_e).unwrap().text(), "abc");

    dispatcher.inject_mouse_click(&mut world, Vec2::new(360.0, 20.0), MouseButton::Left);
    assert!(world.get::<CheckBox>(cb_e).unwrap().is_checked());

    // Click empty space -> clears focus
    dispatcher.inject_mouse_click(&mut world, Vec2::new(700.0, 500.0), MouseButton::Left);
    assert_eq!(dispatcher.focus_manager.focused_control(), None);
}

#[test]
fn ui_06_container_scroll_split_grid_and_aspect_ratio_fixes() {
    // 1. ScrollContainer respects ScrollMode::Disabled and clamps scroll offsets
    let mut world = World::new();
    let mut sc_ctl = Control::default();
    sc_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let sc = world.spawn((
        sc_ctl,
        ScrollContainer {
            scroll_horizontal: 100.0,
            scroll_vertical: 5000.0,
            horizontal_scroll_mode: ScrollMode::Disabled,
            vertical_scroll_mode: ScrollMode::Auto,
            scrollbar_thickness: 12.0,
        },
    ));
    let sc_child = world.spawn((Control {
        custom_minimum_size: Vec2::new(400.0, 500.0),
        ..Default::default()
    },));
    attach_child(&mut world, sc, sc_child).unwrap();
    resolve_ui_layout_system(&mut world, Rect2d::new(0.0, 0.0, 300.0, 200.0));
    let sc_child_rect = world.get::<ResolvedRect>(sc_child).unwrap().global;
    assert_eq!(sc_child_rect.x, 0.0);
    assert_eq!(sc_child_rect.y, -300.0);

    // 2. SplitContainer respects DraggerVisibility::Hidden and hides collapsed pane subtree
    let mut world2 = World::new();
    let mut sp_ctl = Control::default();
    sp_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let mut split = HSplitContainer::new(10.0);
    split.dragger_visibility = DraggerVisibility::Hidden;
    let sp = world2.spawn((sp_ctl, split));
    let p0 = world2.spawn((Control::default(),));
    let p1 = world2.spawn((Control::default(),));
    attach_child(&mut world2, sp, p0).unwrap();
    attach_child(&mut world2, sp, p1).unwrap();
    resolve_ui_layout_system(&mut world2, Rect2d::new(0.0, 0.0, 1000.0, 500.0));
    let p0_r = world2.get::<ResolvedRect>(p0).unwrap().global;
    let p1_r = world2.get::<ResolvedRect>(p1).unwrap().global;
    assert_eq!((p0_r.width, p1_r.x, p1_r.width), (500.0, 500.0, 500.0));

    world2
        .get_mut::<korppi_ui::containers::SplitContainer>(sp)
        .unwrap()
        .collapsed = true;
    world2
        .get_mut::<korppi_ui::containers::SplitContainer>(sp)
        .unwrap()
        .first_collapsed = true;
    let p0_child = world2.spawn((
        Control {
            custom_minimum_size: Vec2::new(150.0, 40.0),
            ..Default::default()
        },
        Button::new("CollapsedChild"),
    ));
    attach_child(&mut world2, p0, p0_child).unwrap();
    resolve_ui_layout_system(&mut world2, Rect2d::new(0.0, 0.0, 1000.0, 500.0));
    let p0_child_r = world2.get::<ResolvedRect>(p0_child).unwrap().global;
    assert_eq!((p0_child_r.width, p0_child_r.height), (0.0, 0.0));
    assert!(UiCanvasRenderer::new()
        .collect_draw_commands(&world2)
        .is_empty());

    // 3. GridContainer weights expansion by stretch_ratio
    let mut world3 = World::new();
    let mut g_ctl = Control::default();
    g_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let g = world3.spawn((g_ctl, GridContainer::new(2, 0.0, 0.0)));
    let gc0 = world3.spawn((Control {
        size_flags_horizontal: SizeFlags::EXPAND_FILL,
        stretch_ratio: 1.0,
        ..Default::default()
    },));
    let gc1 = world3.spawn((Control {
        size_flags_horizontal: SizeFlags::EXPAND_FILL,
        stretch_ratio: 3.0,
        ..Default::default()
    },));
    attach_child(&mut world3, g, gc0).unwrap();
    attach_child(&mut world3, g, gc1).unwrap();
    resolve_ui_layout_system(&mut world3, Rect2d::new(0.0, 0.0, 400.0, 100.0));
    let gc0_w = world3.get::<ResolvedRect>(gc0).unwrap().global.width;
    let gc1_w = world3.get::<ResolvedRect>(gc1).unwrap().global.width;
    assert_eq!((gc0_w, gc1_w), (100.0, 300.0));

    // 4. AspectRatioContainer sanitizes NaN ratio
    let mut world4 = World::new();
    let mut arc_ctl = Control::default();
    arc_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let arc = world4.spawn((
        arc_ctl,
        AspectRatioContainer::new(f32::NAN, AspectStretchMode::Fit),
    ));
    let arc_child = world4.spawn((Control::default(),));
    attach_child(&mut world4, arc, arc_child).unwrap();
    resolve_ui_layout_system(&mut world4, Rect2d::new(0.0, 0.0, 400.0, 200.0));
    let arc_r = world4.get::<ResolvedRect>(arc_child).unwrap().global;
    assert!(arc_r.x.is_finite() && arc_r.width > 0.0);
}

#[test]
fn ui_07_widget_robustness_against_panics_and_nan() {
    // 1. ColorPicker UTF-8 hex safety & NaN sanitization
    let mut picker = ColorPicker::new();
    assert!(!picker.set_from_hex("#aädef"));
    picker.set_rgba([f32::NAN, 0.2, 0.3, 1.0]);
    assert!(picker.color().iter().all(|v| v.is_finite()));
    picker.set_hsva(f32::NAN, 0.5, 0.5, 1.0);
    assert!(picker.color().iter().all(|v| v.is_finite()));

    // 2. Range inverted bounds and NaN sanitization
    let r_inv = Range::new(100.0, 0.0, 1.0);
    assert!(r_inv.value().is_finite());
    let mut r_nan = Range::new(f64::NAN, 100.0, 1.0);
    r_nan.set_value(f64::NAN);
    assert!(r_nan.value().is_finite() && r_nan.ratio().is_finite());

    // 3. LineEdit max_length on set_text and selection replacement on insert/delete
    let mut le = LineEdit::new();
    le.max_length = 5;
    le.set_text("0123456789");
    assert_eq!(le.text(), "01234");
    le.select_all();
    le.set_text("Hi");
    assert_eq!(le.selection, None);
    le.select_all();
    le.insert_text("!");
    assert_eq!(le.text(), "!");

    // 4. Tree::create_item with out-of-bounds parent does not create unreachable orphan
    let mut tree = Tree::new();
    let id = tree.create_item(Some(999), "FallbackRoot");
    assert_eq!(tree.root_items(), &[id]);
}

#[test]
fn ui_08_theme_queries_zero_heap_allocation() {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    struct CountingAlloc;
    thread_local! {
        static THREAD_ALLOCS: Cell<usize> = const { Cell::new(0) };
    }

    // SAFETY: Delegates directly to std::alloc::System while incrementing a thread-local counter.
    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get().saturating_add(1)));
            // SAFETY: Forwards valid layout to System allocator.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: Forwards valid pointer and layout to System allocator.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    let mut world = World::new();
    let e = world.spawn((Control::default(),));
    let dark = Theme::dark();

    let count_allocs = |f: &dyn Fn()| -> usize {
        let before = THREAD_ALLOCS.with(|c| c.get());
        f();
        THREAD_ALLOCS.with(|c| c.get().saturating_sub(before))
    };

    assert_eq!(
        count_allocs(&|| {
            let _ = dark.get_color("Button", "font_color");
        }),
        0
    );
    assert_eq!(
        count_allocs(&|| {
            let _ = get_theme_color(&world, e, "Button", "font_color");
        }),
        0
    );
    assert_eq!(
        count_allocs(&|| {
            let _ = get_theme_constant(&world, e, "BoxContainer", "separation");
        }),
        0
    );
    assert_eq!(
        count_allocs(&|| {
            let _ = get_theme_font_size(&world, e, "Button", "font_size");
        }),
        0
    );
    assert_eq!(
        count_allocs(&|| {
            let _ = get_theme_stylebox(&world, e, "Button", "normal");
        }),
        0
    );
}
