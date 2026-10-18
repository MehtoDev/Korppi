//! Integration tests for keyboard, mouse, gamepad, chords, and action mapping.

use korppi_input::action::{ActionBinding, ActionMap};
use korppi_input::gamepad::{GamepadAxis, GamepadButton, GamepadId, GamepadState};
use korppi_input::keyboard::{KeyCode, KeyboardState};
use korppi_input::mouse::{MouseButton, MouseState};
use korppi_math::Vec2;

#[test]
fn keyboard_state_tracks_press_just_pressed_and_release() {
    let mut keyboard = KeyboardState::new();

    assert!(!keyboard.is_pressed(KeyCode::Space));
    assert!(!keyboard.is_just_pressed(KeyCode::Space));

    // Press Space
    keyboard.press(KeyCode::Space);
    assert!(keyboard.is_pressed(KeyCode::Space));
    assert!(keyboard.is_just_pressed(KeyCode::Space));
    assert!(!keyboard.is_just_released(KeyCode::Space));

    // Advancing frame tick: should stay pressed, but no longer just_pressed
    keyboard.tick();
    assert!(keyboard.is_pressed(KeyCode::Space));
    assert!(!keyboard.is_just_pressed(KeyCode::Space));

    // Release Space
    keyboard.release(KeyCode::Space);
    assert!(!keyboard.is_pressed(KeyCode::Space));
    assert!(keyboard.is_just_released(KeyCode::Space));

    // Next tick: just_released is cleared
    keyboard.tick();
    assert!(!keyboard.is_just_released(KeyCode::Space));
}

#[test]
fn keyboard_modifier_helpers() {
    let mut keyboard = KeyboardState::new();
    assert!(!keyboard.is_shift_pressed());
    assert!(!keyboard.is_ctrl_pressed());

    keyboard.press(KeyCode::ShiftLeft);
    keyboard.press(KeyCode::ControlRight);
    assert!(keyboard.is_shift_pressed());
    assert!(keyboard.is_ctrl_pressed());

    keyboard.release(KeyCode::ShiftLeft);
    assert!(!keyboard.is_shift_pressed());
    assert!(keyboard.is_ctrl_pressed());
}

#[test]
fn mouse_state_tracks_position_delta_and_buttons() {
    let mut mouse = MouseState::new();

    mouse.on_moved(Vec2::new(100.0, 200.0));
    assert_eq!(mouse.position(), Vec2::new(100.0, 200.0));
    assert_eq!(mouse.delta(), Vec2::new(100.0, 200.0));
    mouse.tick();

    // Raw delta accumulation (e.g. for FPS camera)
    mouse.on_raw_motion(Vec2::new(5.0, -3.0));
    assert_eq!(mouse.delta(), Vec2::new(5.0, -3.0));

    // Button handling
    mouse.on_button_pressed(MouseButton::Left);
    assert!(mouse.is_pressed(MouseButton::Left));
    assert!(mouse.is_just_pressed(MouseButton::Left));

    mouse.tick();
    assert_eq!(mouse.delta(), Vec2::ZERO);
    assert!(mouse.is_pressed(MouseButton::Left));
    assert!(!mouse.is_just_pressed(MouseButton::Left));
}

#[test]
fn gamepad_state_tracks_buttons_and_axes_with_deadzone() {
    let mut gamepad = GamepadState::new();
    let pad = GamepadId(0);
    gamepad.connect(pad);

    assert!(gamepad.is_connected(pad));

    // Sub-deadzone movement (deadzone default is 0.15)
    gamepad.set_axis(pad, GamepadAxis::LeftStickX, 0.05);
    assert_eq!(gamepad.axis(pad, GamepadAxis::LeftStickX), 0.0);

    // Movement above deadzone
    gamepad.set_axis(pad, GamepadAxis::LeftStickX, 0.8);
    assert!(gamepad.axis(pad, GamepadAxis::LeftStickX) > 0.7);

    // Button states
    gamepad.press(pad, GamepadButton::South);
    assert!(gamepad.is_pressed(pad, GamepadButton::South));
    assert!(gamepad.is_just_pressed(pad, GamepadButton::South));

    gamepad.tick();
    assert!(gamepad.is_pressed(pad, GamepadButton::South));
    assert!(!gamepad.is_just_pressed(pad, GamepadButton::South));
}

#[test]
fn action_map_resolves_chords_keys_and_axes() {
    let mut action_map = ActionMap::new();

    // Map "Jump" to Space or Gamepad South
    action_map.bind("Jump", ActionBinding::Key(KeyCode::Space));
    action_map.bind("Jump", ActionBinding::GamepadButton(GamepadButton::South));

    // Map "Save" chord to Ctrl + S
    action_map.bind(
        "Save",
        ActionBinding::Chord {
            modifiers: vec![KeyCode::ControlLeft],
            key: KeyCode::KeyS,
        },
    );

    // Map "MoveX" 1D axis to A/D
    action_map.bind(
        "MoveX",
        ActionBinding::Axis1D {
            positive: KeyCode::KeyD,
            negative: KeyCode::KeyA,
            scale: 1.0,
        },
    );

    let mut keyboard = KeyboardState::new();
    let mouse = MouseState::new();
    let gamepad = GamepadState::new();

    assert!(!action_map.is_active("Jump", &keyboard, &mouse, &gamepad));

    // Press Space
    keyboard.press(KeyCode::Space);
    assert!(action_map.is_active("Jump", &keyboard, &mouse, &gamepad));

    // Chord test: pressing S alone does NOT activate "Save"
    keyboard.press(KeyCode::KeyS);
    assert!(!action_map.is_active("Save", &keyboard, &mouse, &gamepad));

    // Pressing Ctrl activates "Save"
    keyboard.press(KeyCode::ControlLeft);
    assert!(action_map.is_active("Save", &keyboard, &mouse, &gamepad));

    // Axis 1D test: D pressed gives +1.0
    keyboard.press(KeyCode::KeyD);
    assert_eq!(
        action_map.axis_1d("MoveX", &keyboard, &mouse, &gamepad),
        1.0
    );

    // A and D pressed cancels out to 0.0
    keyboard.press(KeyCode::KeyA);
    assert_eq!(
        action_map.axis_1d("MoveX", &keyboard, &mouse, &gamepad),
        0.0
    );
}
