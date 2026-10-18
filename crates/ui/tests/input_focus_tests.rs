//! Unit tests for UI Input Event Routing, Mouse Filtering, Focus Navigation,
//! and Synthetic Input Injection.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_input::keyboard::KeyCode;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::control::{Control, FocusMode, LayoutPreset, MouseFilter};
use korppi_ui::input::{FocusDirection, FocusManager, UiEvent, UiInputDispatcher, UiReceiver};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d};

#[test]
fn test_mouse_filter_stop_pass_ignore() {
    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    // Parent background: (0, 0, 400, 400) - MouseFilter::Stop
    let mut bg_ctl = Control::default();
    bg_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    bg_ctl.mouse_filter = MouseFilter::Stop;
    let bg = world.spawn((bg_ctl, UiReceiver::default()));

    // Child A: (50, 50, 100, 100) - MouseFilter::Pass
    let mut a_ctl = Control::default();
    a_ctl.offset_left = 50.0;
    a_ctl.offset_top = 50.0;
    a_ctl.offset_right = 150.0;
    a_ctl.offset_bottom = 150.0;
    a_ctl.mouse_filter = MouseFilter::Pass;
    let child_a = world.spawn((a_ctl, UiReceiver::default()));
    attach_child(&mut world, bg, child_a).expect("Attach A");

    // Child B: (200, 200, 100, 100) - MouseFilter::Ignore
    let mut b_ctl = Control::default();
    b_ctl.offset_left = 200.0;
    b_ctl.offset_top = 200.0;
    b_ctl.offset_right = 300.0;
    b_ctl.offset_bottom = 300.0;
    b_ctl.mouse_filter = MouseFilter::Ignore;
    let child_b = world.spawn((b_ctl, UiReceiver::default()));
    attach_child(&mut world, bg, child_b).expect("Attach B");

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 400.0,
        height: 400.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Click on Child A (at 75, 75):
    // Child A has MouseFilter::Pass -> Child A receives event, AND bg also receives it!
    dispatcher.inject_mouse_button(&mut world, Vec2::new(75.0, 75.0), MouseButton::Left, true);

    let a_events = world.get::<UiReceiver>(child_a).unwrap().events.clone();
    let bg_events = world.get::<UiReceiver>(bg).unwrap().events.clone();
    assert!(a_events
        .iter()
        .any(|e| matches!(e, UiEvent::MouseButton { pressed: true, .. })));
    assert!(bg_events
        .iter()
        .any(|e| matches!(e, UiEvent::MouseButton { pressed: true, .. })));

    // Clear event buffers
    world.get_mut::<UiReceiver>(child_a).unwrap().clear();
    world.get_mut::<UiReceiver>(bg).unwrap().clear();

    // Click on Child B (at 250, 250):
    // Child B has MouseFilter::Ignore -> Child B receives NOTHING; bg receives it!
    dispatcher.inject_mouse_button(&mut world, Vec2::new(250.0, 250.0), MouseButton::Left, true);

    let b_events = world.get::<UiReceiver>(child_b).unwrap().events.clone();
    let bg_events2 = world.get::<UiReceiver>(bg).unwrap().events.clone();
    assert!(b_events.is_empty());
    assert!(bg_events2
        .iter()
        .any(|e| matches!(e, UiEvent::MouseButton { pressed: true, .. })));
}

#[test]
fn test_mouse_enter_exit_tracking() {
    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    // Root canvas
    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world.spawn((root_ctl,));

    // Button at (100, 100, 100, 50)
    let mut btn_ctl = Control::default();
    btn_ctl.offset_left = 100.0;
    btn_ctl.offset_top = 100.0;
    btn_ctl.offset_right = 200.0;
    btn_ctl.offset_bottom = 150.0;
    let btn = world.spawn((btn_ctl, UiReceiver::default()));
    attach_child(&mut world, root, btn).expect("Attach btn");

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // 1. Cursor moves to (50, 50) - outside button
    dispatcher.inject_mouse_move(&mut world, Vec2::new(50.0, 50.0));
    assert!(world.get::<UiReceiver>(btn).unwrap().events.is_empty());

    // 2. Cursor moves to (150, 125) - inside button -> MouseEnter
    dispatcher.inject_mouse_move(&mut world, Vec2::new(150.0, 125.0));
    let events = world.get::<UiReceiver>(btn).unwrap().events.clone();
    assert!(events.contains(&UiEvent::MouseEnter));

    // 3. Cursor moves to (300, 300) - outside button -> MouseExit
    dispatcher.inject_mouse_move(&mut world, Vec2::new(300.0, 300.0));
    let events2 = world.get::<UiReceiver>(btn).unwrap().events.clone();
    assert!(events2.contains(&UiEvent::MouseExit));
}

#[test]
fn test_focus_click_and_tab_navigation() {
    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    // Root canvas
    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world.spawn((root_ctl,));

    // Control 1: FocusMode::All at (0, 0, 100, 40)
    let mut c1_ctl = Control::default();
    c1_ctl.offset_right = 100.0;
    c1_ctl.offset_bottom = 40.0;
    c1_ctl.focus_mode = FocusMode::All;
    let c1 = world.spawn((c1_ctl, UiReceiver::default()));
    attach_child(&mut world, root, c1).expect("Attach c1");

    // Control 2: FocusMode::None at (120, 0, 220, 40)
    let mut c2_ctl = Control::default();
    c2_ctl.offset_left = 120.0;
    c2_ctl.offset_right = 220.0;
    c2_ctl.offset_bottom = 40.0;
    c2_ctl.focus_mode = FocusMode::None;
    let c2 = world.spawn((c2_ctl, UiReceiver::default()));
    attach_child(&mut world, root, c2).expect("Attach c2");

    // Control 3: FocusMode::All at (240, 0, 340, 40)
    let mut c3_ctl = Control::default();
    c3_ctl.offset_left = 240.0;
    c3_ctl.offset_right = 340.0;
    c3_ctl.offset_bottom = 40.0;
    c3_ctl.focus_mode = FocusMode::All;
    let c3 = world.spawn((c3_ctl, UiReceiver::default()));
    attach_child(&mut world, root, c3).expect("Attach c3");

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Click on c2 (FocusMode::None) -> does NOT receive focus
    dispatcher.inject_mouse_button(&mut world, Vec2::new(150.0, 20.0), MouseButton::Left, true);
    assert_eq!(dispatcher.focus_manager.focused_control(), None);

    // Click on c1 (FocusMode::All) -> receives focus!
    dispatcher.inject_mouse_button(&mut world, Vec2::new(50.0, 20.0), MouseButton::Left, true);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(c1));
    assert!(world
        .get::<UiReceiver>(c1)
        .unwrap()
        .events
        .contains(&UiEvent::FocusGained));

    // Inject Tab key -> skips c2 (FocusMode::None) and moves focus to c3!
    dispatcher.inject_tab(&mut world, false);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(c3));
    assert!(world
        .get::<UiReceiver>(c1)
        .unwrap()
        .events
        .contains(&UiEvent::FocusLost));
    assert!(world
        .get::<UiReceiver>(c3)
        .unwrap()
        .events
        .contains(&UiEvent::FocusGained));

    // Inject Shift+Tab key -> cycles back to c1
    dispatcher.inject_tab(&mut world, true);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(c1));
}

#[test]
fn test_directional_focus_navigation() {
    let mut world = World::new();
    let mut focus_mgr = FocusManager::default();

    // 2x2 grid of buttons:
    // Top-Left (0, 0), Top-Right (100, 0)
    // Bottom-Left (0, 100), Bottom-Right (100, 100)
    let mut tl_ctl = Control::default();
    tl_ctl.offset_right = 80.0;
    tl_ctl.offset_bottom = 40.0;
    tl_ctl.focus_mode = FocusMode::All;
    let tl = world.spawn((tl_ctl,));

    let mut tr_ctl = Control::default();
    tr_ctl.offset_left = 100.0;
    tr_ctl.offset_right = 180.0;
    tr_ctl.offset_bottom = 40.0;
    tr_ctl.focus_mode = FocusMode::All;
    let tr = world.spawn((tr_ctl,));

    let mut bl_ctl = Control::default();
    bl_ctl.offset_top = 100.0;
    bl_ctl.offset_right = 80.0;
    bl_ctl.offset_bottom = 140.0;
    bl_ctl.focus_mode = FocusMode::All;
    let bl = world.spawn((bl_ctl,));

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Start at Top-Left
    focus_mgr.set_focus(&mut world, Some(tl));
    assert_eq!(focus_mgr.focused_control(), Some(tl));

    // Move Right -> Top-Right
    focus_mgr.navigate_direction(&mut world, FocusDirection::Right);
    assert_eq!(focus_mgr.focused_control(), Some(tr));

    // Move Down -> Bottom-Left is closest downwards
    focus_mgr.navigate_direction(&mut world, FocusDirection::Down);
    assert_eq!(focus_mgr.focused_control(), Some(bl));
}

#[test]
fn test_synthetic_keyboard_and_wheel() {
    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    let mut ctl = Control::default();
    ctl.offset_right = 200.0;
    ctl.offset_bottom = 100.0;
    ctl.focus_mode = FocusMode::All;
    let target = world.spawn((ctl, UiReceiver::default()));

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Focus target
    dispatcher.focus_manager.set_focus(&mut world, Some(target));

    // Inject keyboard KeyDown / KeyUp
    dispatcher.inject_key(&mut world, KeyCode::KeyA, true);
    dispatcher.inject_key(&mut world, KeyCode::KeyA, false);
    dispatcher.inject_char(&mut world, 'A');

    // Inject mouse wheel
    dispatcher.inject_mouse_wheel(&mut world, Vec2::new(50.0, 50.0), Vec2::new(0.0, 120.0));

    let events = world.get::<UiReceiver>(target).unwrap().events.clone();
    assert!(events.contains(&UiEvent::KeyDown { key: KeyCode::KeyA }));
    assert!(events.contains(&UiEvent::KeyUp { key: KeyCode::KeyA }));
    assert!(events.contains(&UiEvent::CharInput { character: 'A' }));
    assert!(events.contains(&UiEvent::MouseWheel {
        position: Vec2::new(50.0, 50.0),
        delta: Vec2::new(0.0, 120.0),
    }));
}

#[test]
fn test_explicit_focus_neighbors_and_tab_cycle() {
    let mut world = World::new();
    let mut focus_mgr = FocusManager::default();

    // Node 1
    let mut c1 = Control::default();
    c1.offset_right = 100.0;
    c1.offset_bottom = 50.0;
    c1.focus_mode = FocusMode::All;
    let e1 = world.spawn((c1,));

    // Node 2 (placed far away to the right)
    let mut c2 = Control::default();
    c2.offset_left = 500.0;
    c2.offset_right = 600.0;
    c2.offset_bottom = 50.0;
    c2.focus_mode = FocusMode::All;
    let e2 = world.spawn((c2,));

    // Node 3 (placed close to e1 on right)
    let mut c3 = Control::default();
    c3.offset_left = 120.0;
    c3.offset_right = 220.0;
    c3.offset_bottom = 50.0;
    c3.focus_mode = FocusMode::All;
    let _e3 = world.spawn((c3,));

    // Set explicit focus neighbor on e1: Right points explicitly to e2 (skipping e3!)
    world.get_mut::<Control>(e1).unwrap().focus_neighbor_right = Some(e2);
    // Set explicit focus_next on e1: points explicitly to e2
    world.get_mut::<Control>(e1).unwrap().focus_next = Some(e2);
    // Set explicit focus_previous on e2: points explicitly to e1
    world.get_mut::<Control>(e2).unwrap().focus_previous = Some(e1);

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    focus_mgr.set_focus(&mut world, Some(e1));
    // Explicit neighbor overrides geometric proximity (e2 is chosen instead of e3)
    focus_mgr.navigate_direction(&mut world, FocusDirection::Right);
    assert_eq!(focus_mgr.focused_control(), Some(e2));

    // Explicit focus_previous on e2 leads back to e1
    focus_mgr.navigate_tab(&mut world, true);
    assert_eq!(focus_mgr.focused_control(), Some(e1));

    // Explicit focus_next on e1 leads to e2
    focus_mgr.navigate_tab(&mut world, false);
    assert_eq!(focus_mgr.focused_control(), Some(e2));
}
