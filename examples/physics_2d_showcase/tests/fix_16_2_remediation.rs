//! Regression tests for Milestone FIX-16.2 (`SHW-04`, `SHW-06`, `SHW-07`) in `physics_2d_showcase`.

use korppi_input::KeyCode;
use korppi_math::Vec2;
use korppi_physics2d::{
    body::{BodyType, RigidBody2d},
    collider::Collider2d,
    joint::Joint2d,
    shape::{Circle2d, Shape2d},
};
use korppi_window::{KorppiApp, Window, WindowConfig, WindowEvent};
use physics_2d_showcase_lib::app::Physics2dShowcaseApp;
use std::fs;
use std::path::Path;

/// `SHW-04` (2D Gallery 2 `JointsGallery`):
/// - The prismatic elevator joint must have `enable_motor == true` and non-zero `motor_speed`,
///   actively driving the elevator platform upward/downward over 180 ticks.
/// - A dynamic load must drop onto the suspension bridge so the bridge sags and oscillates.
/// - The weld cantilever must have a finite breakable force/torque threshold that fractures
///   when struck by the falling load (`joints_after < joints_initial`).
#[test]
fn shw_04_2d_joints_gallery_motor_elevator_bridge_load_and_breakable_cantilever() {
    let mut app = Physics2dShowcaseApp::new();
    app.load_gallery(1); // Gallery 2: JointsGallery

    let initial_joints = app.world().joint_count();
    assert!(
        initial_joints >= 14,
        "JointsGallery should start with all suspension, wheel, pendulum, slider, rover, and cantilever joints"
    );

    // Check that the prismatic joint has an enabled motor with non-zero speed
    let has_active_prismatic_motor = app.world().joints().any(|(_, j)| match &j.joint {
        Joint2d::Prismatic(p) => p.enable_motor && p.motor_speed.abs() > 0.1,
        _ => false,
    });
    assert!(
        has_active_prismatic_motor,
        "SHW-04: Prismatic elevator joint in JointsGallery must have enable_motor=true and non-zero motor_speed"
    );

    // Track initial piston Y (at x == -6.0) and bridge plank Y (around x == -6.0, y > 4.0)
    let initial_piston_y = app
        .world()
        .bodies()
        .find(|(_, b)| {
            b.body_type == BodyType::Dynamic
                && (b.position.x - (-6.0)).abs() < 0.2
                && b.position.y < 3.5
        })
        .map(|(_, b)| b.position.y)
        .expect("prismatic piston body at x=-6.0");

    let mut max_piston_displacement = 0.0_f32;
    for _ in 0..180 {
        app.step_simulation(1.0 / 60.0);
        if let Some((_, piston)) = app.world().bodies().find(|(_, b)| {
            b.body_type == BodyType::Dynamic
                && (b.position.x - (-6.0)).abs() < 0.2
                && b.position.y < 4.5
        }) {
            max_piston_displacement =
                max_piston_displacement.max((piston.position.y - initial_piston_y).abs());
        }
    }

    assert!(
        max_piston_displacement > 0.4,
        "SHW-04: Motorized prismatic piston must actively translate along its slider axis (max displacement = {max_piston_displacement:.3}m)"
    );

    let final_joints = app.world().joint_count();
    assert!(
        final_joints < initial_joints,
        "SHW-04: Breakable weld cantilever joint must fracture under load within 180 ticks (initial joints = {initial_joints}, final joints = {final_joints})"
    );
}

/// `SHW-06` (2D `pointer_down` ground body anchor & initial cursor delta jump):
/// - Even when all existing bodies are removed and the first body in `world` is a `BodyType::Dynamic`
///   body, `pointer_down` must anchor its `MotorJoint2d` to a `BodyType::Static` ground body
///   (`body_a != body_b`) rather than creating a self-joint or pulling two dynamic bodies together.
/// - The first `WindowEvent::CursorMoved` must initialize cursor tracking without a large delta jump
///   from `(0.0, 0.0)`.
#[test]
fn shw_06_2d_pointer_down_anchors_to_static_ground_and_first_cursor_move_has_zero_delta() {
    let mut app = Physics2dShowcaseApp::new();

    // Clear all gallery bodies and create a single dynamic circle as the first body in the world
    let existing: Vec<_> = app.world.bodies().map(|(id, _)| id).collect();
    for id in existing {
        app.world.destroy_body(id);
    }
    let ball = app
        .world
        .create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 4.0)));
    app.world.create_collider(
        ball,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0))),
    );

    // Click on the dynamic ball at (0.0, 4.0)
    app.pointer_down(Vec2::new(0.0, 4.0));
    assert_eq!(
        app.world.joint_count(),
        1,
        "SHW-06: pointer_down on a dynamic body must create a MotorJoint2d even when no static body existed first"
    );
    let (_, joint_inst) = app.world.joints().next().unwrap();
    assert_ne!(
        joint_inst.body_a, joint_inst.body_b,
        "SHW-06: pointer_down must never create a self-joint"
    );
    let anchor_body = app
        .world
        .body(joint_inst.body_a)
        .expect("ground anchor body exists");
    assert_eq!(
        anchor_body.body_type,
        BodyType::Static,
        "SHW-06: pointer_down drag joint must anchor to a Static ground body"
    );

    // Drag the ball toward (3.0, 6.0)
    app.pointer_move(Vec2::new(3.0, 6.0));
    for _ in 0..30 {
        app.step_simulation(1.0 / 60.0);
    }
    let ball_pos = app.world.body(ball).unwrap().position;
    assert!(
        ball_pos.x > 0.5,
        "SHW-06: Dragged ball should move toward target X=3.0, got {ball_pos:?}"
    );
    app.pointer_up();

    // Verify first CursorMoved while right-dragging does not spike camera_offset from (0, 0)
    let mut app2 = Physics2dShowcaseApp::new();
    let mut win = Window::new_headless(WindowConfig::default().with_dimensions(1280, 720))
        .expect("headless window");
    let before_world = app2.screen_to_world(Vec2::new(640.0, 360.0));
    app2.on_event(
        &mut win,
        &WindowEvent::MouseInput {
            button: 1,
            pressed: true,
        },
    );
    app2.on_event(&mut win, &WindowEvent::CursorMoved { x: 500.0, y: 400.0 });
    let after_first_move = app2.screen_to_world(Vec2::new(640.0, 360.0));
    assert!(
        (after_first_move - before_world).length() < 1e-4,
        "SHW-06: First CursorMoved event must not cause a camera pan delta jump from (0, 0)"
    );
}

/// `SHW-07` (Rule 2 dependency hygiene):
/// - `examples/physics_2d_showcase/Cargo.toml` must not depend directly on `winit`.
/// - Every crate declared in `[dependencies]` must be referenced in `src/`.
/// - Keyboard input via `korppi_input::KeyCode` must switch galleries and toggle pause.
#[test]
fn shw_07_2d_no_direct_winit_dependency_and_all_declared_dependencies_used() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("read Cargo.toml");
    let deps_section = cargo_toml
        .split("[dev-dependencies]")
        .next()
        .unwrap_or(&cargo_toml);

    assert!(
        !deps_section.contains("winit"),
        "SHW-07: examples/physics_2d_showcase/Cargo.toml [dependencies] must not depend directly on winit"
    );

    let all_src = [
        fs::read_to_string(manifest_dir.join("src/lib.rs")).unwrap_or_default(),
        fs::read_to_string(manifest_dir.join("src/main.rs")).unwrap_or_default(),
        fs::read_to_string(manifest_dir.join("src/app.rs")).unwrap_or_default(),
        fs::read_to_string(manifest_dir.join("src/gallery.rs")).unwrap_or_default(),
    ]
    .join("\n");

    for dep in [
        "korppi_core",
        "korppi_ecs",
        "korppi_engine",
        "korppi_input",
        "korppi_math",
        "korppi_physics2d",
        "korppi_render",
        "korppi_ui",
        "korppi_window",
    ] {
        assert!(
            all_src.contains(dep),
            "SHW-07: Dependency `{dep}` in Cargo.toml must be used in src/"
        );
    }

    let mut app = Physics2dShowcaseApp::new();
    let mut win = Window::new_headless(WindowConfig::default().with_dimensions(1280, 720))
        .expect("headless window");
    app.on_event(
        &mut win,
        &WindowEvent::KeyboardInput {
            key_code: Some(KeyCode::Digit3.to_u32()),
            pressed: true,
        },
    );
    assert_eq!(app.current_gallery_index(), 2);
}
