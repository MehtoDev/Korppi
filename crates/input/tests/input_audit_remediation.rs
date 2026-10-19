//! Regression tests for `FIX-12.3` / `FIX-13.3` input audit findings (`INP-01`–`INP-06`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use korppi_input::{
    ActionBinding, ActionMap, GamepadAxis, GamepadButton, GamepadId, GamepadState, InputManager,
    KeyCode, KeyboardState, MouseButton, MouseState,
};
use korppi_math::Vec2;

struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: delegates all allocation operations directly to the system allocator while counting calls.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.set_or_add(1)));
        // SAFETY: caller upholds `GlobalAlloc::alloc` layout invariants.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller upholds `GlobalAlloc::dealloc` pointer and layout invariants.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.set_or_add(1)));
        // SAFETY: caller upholds `GlobalAlloc::realloc` pointer, layout, and size invariants.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

trait CellAdd {
    fn set_or_add(&self, delta: usize) -> usize;
}

impl CellAdd for Cell<usize> {
    fn set_or_add(&self, delta: usize) -> usize {
        self.get() + delta
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocations_in(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

/// INP-01 (I1): `KeyCode` discriminants and `from_u32` / `from_winit_code` must round-trip all 69
/// keys and match `winit::keyboard::KeyCode` discriminants emitted by `WindowEvent::KeyboardInput`,
/// and `InputManager` must ingest window key/mouse/cursor/wheel/focus events directly.
#[test]
fn i1_keycode_winit_discriminants_and_input_manager_event_ingestion() {
    // Spot-check the winit discriminants highlighted by probe I1:
    // In winit::keyboard::KeyCode: Digit1 = 6, KeyA = 19, Space = 62.
    assert_eq!(KeyCode::Digit1.to_u32(), 6);
    assert_eq!(KeyCode::KeyA.to_u32(), 19);
    assert_eq!(KeyCode::Space.to_u32(), 62);

    for &key in KeyCode::ALL {
        let code = key.to_u32();
        assert_eq!(KeyCode::from_u32(code), Some(key));
        assert_eq!(KeyCode::from_winit_code(code), Some(key));
    }
    assert_eq!(KeyCode::from_u32(u32::MAX), None);

    // MouseButton u32 round-trip matching WindowEvent::MouseInput:
    assert_eq!(MouseButton::from_u32(0), MouseButton::Left);
    assert_eq!(MouseButton::from_u32(1), MouseButton::Right);
    assert_eq!(MouseButton::from_u32(2), MouseButton::Middle);
    assert_eq!(MouseButton::from_u32(5), MouseButton::Other(5));
    assert_eq!(MouseButton::Left.to_u32(), 0);
    assert_eq!(MouseButton::Right.to_u32(), 1);
    assert_eq!(MouseButton::Middle.to_u32(), 2);
    assert_eq!(MouseButton::Other(5).to_u32(), 5);

    // End-to-end event ingestion through InputManager:
    let mut input = InputManager::new();
    input.bind_action("Jump", ActionBinding::Key(KeyCode::Space));
    input.bind_action("Fire", ActionBinding::MouseButton(MouseButton::Left));

    input.on_key_input(Some(KeyCode::Space.to_u32()), true);
    input.on_mouse_input(0, true);
    input.on_cursor_moved(120.0, 240.0);
    input.on_cursor_moved(130.0, 255.0);
    input.on_mouse_wheel(0.0, 1.5);

    assert!(input.is_pressed(KeyCode::Space));
    assert!(input.is_just_pressed(KeyCode::Space));
    assert!(input.is_active("Jump"));
    assert!(input.is_action_active("Jump"));
    assert!(input.is_action_just_pressed("Jump"));
    assert!(input.is_mouse_button_pressed(MouseButton::Left));
    assert!(input.is_active("Fire"));
    assert_eq!(input.mouse_position(), Vec2::new(130.0, 255.0));
    assert_eq!(input.mouse_delta(), Vec2::new(10.0, 15.0));
    assert_eq!(input.mouse.wheel_delta(), Vec2::new(0.0, 1.5));
}

/// INP-02 (I2): `GamepadState::set_axis` must never divide by zero when `deadzone == 1.0` or pass
/// `NaN` through to `ActionMap`, must clamp triggers to `[0.0, 1.0]`, and must support radial
/// 2D stick deadzone calibration.
#[test]
fn i2_gamepad_deadzone_nan_guard_trigger_clamping_and_radial_stick() {
    let mut gp = GamepadState::new();
    let pad = GamepadId(0);
    gp.connect(pad);

    // 1. deadzone == 1.0 + full deflection 1.0 must not produce 0.0 / 0.0 = NaN:
    gp.set_deadzone(pad, 1.0);
    gp.set_axis(pad, GamepadAxis::LeftStickX, 1.0);
    let val_at_1 = gp.axis(pad, GamepadAxis::LeftStickX);
    assert!(!val_at_1.is_nan(), "axis() returned NaN at deadzone = 1.0");
    assert_eq!(val_at_1, 0.0);

    // Non-finite deadzone or axis input must be ignored/zeroed safely:
    gp.set_deadzone(pad, f32::NAN);
    gp.set_axis(pad, GamepadAxis::LeftStickX, f32::NAN);
    assert_eq!(gp.axis(pad, GamepadAxis::LeftStickX), 0.0);
    gp.set_axis(pad, GamepadAxis::LeftStickX, f32::INFINITY);
    assert_eq!(gp.axis(pad, GamepadAxis::LeftStickX), 0.0);

    let mut actions = ActionMap::new();
    actions.bind(
        "Steer",
        ActionBinding::GamepadAxis {
            axis: GamepadAxis::LeftStickX,
            scale: 1.0,
        },
    );
    let steer = actions.axis_1d("Steer", &KeyboardState::new(), &MouseState::new(), &gp);
    assert!(!steer.is_nan());
    assert_eq!(steer, 0.0);

    // 2. Triggers must be clamped to [0.0, 1.0]:
    gp.set_deadzone(pad, 0.15);
    gp.set_axis(pad, GamepadAxis::LeftTrigger, -0.8);
    assert_eq!(
        gp.axis(pad, GamepadAxis::LeftTrigger),
        0.0,
        "LeftTrigger must clamp negative values to 0.0"
    );
    gp.set_axis(pad, GamepadAxis::RightTrigger, -0.5);
    assert_eq!(
        gp.axis(pad, GamepadAxis::RightTrigger),
        0.0,
        "RightTrigger must clamp negative values to 0.0"
    );
    gp.set_axis(pad, GamepadAxis::LeftTrigger, 0.575);
    assert!((gp.axis(pad, GamepadAxis::LeftTrigger) - 0.5).abs() < 1e-5);

    // 3. Radial 2D stick calibration:
    // A diagonal deflection (0.12, 0.12) has axial components < 0.15, but radial magnitude
    // sqrt(0.12^2 + 0.12^2) = 0.1697 > 0.15, so radial stick query must preserve direction!
    gp.set_stick(
        pad,
        GamepadAxis::LeftStickX,
        GamepadAxis::LeftStickY,
        0.12,
        0.12,
    );
    let stick = gp.left_stick(pad);
    assert!(
        stick.length() > 0.0 && stick.x > 0.0 && stick.y > 0.0,
        "Radial stick deadzone should pass diagonal (0.12, 0.12) with length 0.1697 > 0.15, got {stick:?}"
    );
    assert!(
        (stick.x - stick.y).abs() < 1e-5,
        "Radial deadzone must preserve 45-degree diagonal angle, got {stick:?}"
    );
}

/// INP-03 (I3): First `MouseState::on_moved` must not produce a delta spike from `(0, 0)`,
/// `on_cursor_left` must re-arm initial placement, and `on_focus_lost` must release held inputs.
#[test]
fn i3_mouse_initial_move_and_focus_loss_release() {
    let mut mouse = MouseState::new();
    mouse.on_moved(Vec2::new(640.0, 360.0));
    assert_eq!(mouse.position(), Vec2::new(640.0, 360.0));
    assert_eq!(
        mouse.delta(),
        Vec2::ZERO,
        "First on_moved must not spike delta from (0, 0)"
    );

    mouse.on_moved(Vec2::new(652.0, 350.0));
    assert_eq!(mouse.position(), Vec2::new(652.0, 350.0));
    assert_eq!(mouse.delta(), Vec2::new(12.0, -10.0));

    mouse.tick();
    mouse.on_cursor_left();
    mouse.on_moved(Vec2::new(100.0, 100.0));
    assert_eq!(mouse.position(), Vec2::new(100.0, 100.0));
    assert_eq!(
        mouse.delta(),
        Vec2::ZERO,
        "First on_moved after CursorLeft must not spike delta"
    );

    // Focus loss must release held keys and mouse buttons and mark them just_released:
    let mut input = InputManager::new();
    input.keyboard.press(KeyCode::KeyW);
    input.keyboard.press(KeyCode::ShiftLeft);
    input.mouse.on_button_pressed(MouseButton::Left);
    input.tick();

    assert!(input.is_pressed(KeyCode::KeyW));
    assert!(input.is_mouse_button_pressed(MouseButton::Left));

    input.on_focused(false);
    assert!(!input.is_pressed(KeyCode::KeyW));
    assert!(!input.is_pressed(KeyCode::ShiftLeft));
    assert!(!input.is_mouse_button_pressed(MouseButton::Left));
    assert!(input.is_just_released(KeyCode::KeyW));
    assert!(input.is_just_released(KeyCode::ShiftLeft));
    assert!(input.mouse.is_just_released(MouseButton::Left));

    input.update();
    assert!(!input.is_just_released(KeyCode::KeyW));
    assert!(!input.mouse.is_just_released(MouseButton::Left));
}

/// INP-04 (I4): `KeyboardState`, `MouseState`, and `GamepadState` must use fixed-size inline
/// bitsets/arrays and perform 0 heap allocations during input transitions and queries.
#[test]
fn i4_input_hotpath_performs_zero_heap_allocations() {
    let mut input = InputManager::new();
    let pad = GamepadId(0);

    let allocs = allocations_in(|| {
        input.keyboard.press(KeyCode::KeyW);
        input.keyboard.press(KeyCode::Space);
        input.keyboard.press(KeyCode::F12);
        input.keyboard.release(KeyCode::KeyW);
        let _ = input.keyboard.is_pressed(KeyCode::Space);
        let _ = input.keyboard.is_just_pressed(KeyCode::Space);
        let _ = input.keyboard.is_just_released(KeyCode::KeyW);

        input.mouse.on_button_pressed(MouseButton::Left);
        input.mouse.on_button_pressed(MouseButton::Other(4));
        input.mouse.on_button_released(MouseButton::Left);
        input.mouse.on_moved(Vec2::new(320.0, 240.0));
        input.mouse.on_moved(Vec2::new(330.0, 250.0));
        input.mouse.on_wheel(Vec2::new(0.0, 1.0));

        input.gamepad.connect(pad);
        input.gamepad.press(pad, GamepadButton::South);
        input.gamepad.release(pad, GamepadButton::South);
        input.gamepad.set_axis(pad, GamepadAxis::LeftStickX, 0.5);
        let _ = input.gamepad.axis(pad, GamepadAxis::LeftStickX);

        input.tick();
    });

    assert_eq!(
        allocs, 0,
        "Input transitions and queries allocated {allocs} times on the heap (expected 0)"
    );
}

/// INP-05 (I5): `ActionMap` must query all connected gamepads (`GamepadId(1+)`), clamp `axis_2d`
/// vector length to `<= 1.0`, and support `is_just_pressed`, `is_just_released`, `unbind`, and `clear`.
#[test]
fn i5_action_map_multi_gamepad_clamped_axis2d_edge_queries_and_unbind() {
    let mut input = InputManager::new();
    input
        .actions
        .bind("Jump", ActionBinding::GamepadButton(GamepadButton::South));
    input
        .actions
        .bind("Jump", ActionBinding::Key(KeyCode::Space));
    input.actions.bind(
        "Move",
        ActionBinding::Axis2D {
            up: KeyCode::KeyW,
            down: KeyCode::KeyS,
            left: KeyCode::KeyA,
            right: KeyCode::KeyD,
        },
    );
    input.actions.bind(
        "Move",
        ActionBinding::GamepadAxis {
            axis: GamepadAxis::LeftStickX,
            scale: 1.0,
        },
    );
    input.actions.bind(
        "Move",
        ActionBinding::GamepadAxis {
            axis: GamepadAxis::LeftStickY,
            scale: 1.0,
        },
    );

    // 1. Secondary controller GamepadId(1) must trigger GamepadButton and GamepadAxis bindings:
    let pad1 = GamepadId(1);
    input.gamepad.connect(pad1);
    input.gamepad.press(pad1, GamepadButton::South);
    assert!(
        input.is_active("Jump"),
        "GamepadId(1) button press must activate ActionBinding::GamepadButton"
    );
    assert!(input.is_action_just_pressed("Jump"));
    assert!(!input.is_action_just_released("Jump"));

    input.tick();
    assert!(input.is_active("Jump"));
    assert!(!input.is_action_just_pressed("Jump"));

    input.gamepad.release(pad1, GamepadButton::South);
    assert!(!input.is_active("Jump"));
    assert!(input.is_action_just_released("Jump"));
    input.tick();

    // 2. Diagonal keyboard + gamepad stick deflection must clamp axis_2d magnitude to <= 1.0:
    let pad0 = GamepadId(0);
    input.gamepad.connect(pad0);
    input.gamepad.set_axis(pad0, GamepadAxis::LeftStickX, 1.0);
    input.gamepad.set_axis(pad0, GamepadAxis::LeftStickY, 1.0);
    input.keyboard.press(KeyCode::KeyW);
    input.keyboard.press(KeyCode::KeyD);

    let move_vec = input.axis_2d("Move");
    assert!(
        move_vec.length() <= 1.0 + 1e-5,
        "axis_2d length must be clamped to <= 1.0, got {} ({move_vec:?})",
        move_vec.length()
    );
    assert!((move_vec.length() - 1.0).abs() < 1e-5);

    // 3. Unbind a specific binding and clear an action:
    assert!(input
        .actions
        .unbind("Jump", &ActionBinding::Key(KeyCode::Space)));
    input.keyboard.press(KeyCode::Space);
    assert!(
        !input.is_active("Jump"),
        "Unbound Space key should no longer activate Jump"
    );

    input.actions.clear_action("Move");
    assert_eq!(input.axis_2d("Move"), Vec2::ZERO);
}
