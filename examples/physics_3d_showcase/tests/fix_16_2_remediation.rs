//! Regression tests for Milestone FIX-16.2 (`SHW-04`, `SHW-05`, `SHW-06`, `SHW-07`) in `physics_3d_showcase`.

use korppi_input::KeyCode;
use korppi_math::Vec3;
use korppi_physics::body::BodyType;
use korppi_window::{KorppiApp, Window, WindowConfig, WindowEvent};
use physics_3d_showcase_lib::app::Physics3dShowcaseApp;
use physics_3d_showcase_lib::camera::{Camera3d, CameraMode};
use std::fs;
use std::path::Path;

/// `SHW-04` Part 1 (3D Gallery 2 `SoftStepTuningGallery`):
/// - The left tower (`x < -1.0`) and right tower (`x > 1.0`) must have distinct mass/compliance
///   profiles so their settling heights differ significantly after 120 ticks (`max_tower_diff > 0.05`).
/// - Interactive Soft Step tuning methods (`adjust_contact_hertz`, `adjust_contact_damping`) and
///   keyboard bindings (`ArrowUp`/`ArrowDown`, `PageUp`/`PageDown`) must update `SolverConfig3d`
///   and wake the simulation bodies.
#[test]
fn shw_04_3d_soft_step_gallery_differentiation_and_live_tuning_controls() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(1); // Gallery 2: SoftStepTuningGallery

    for _ in 0..120 {
        app.step_simulation(1.0 / 60.0);
    }

    let mut left_ys = Vec::new();
    let mut right_ys = Vec::new();
    for (_, b) in app.world.bodies() {
        if b.position.x < -1.0 {
            left_ys.push(b.position.y);
        } else if b.position.x > 1.0 {
            right_ys.push(b.position.y);
        }
    }
    left_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    right_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        !left_ys.is_empty() && !right_ys.is_empty(),
        "Both left and right towers must exist in SoftStepTuningGallery"
    );

    let max_tower_diff = left_ys
        .iter()
        .zip(right_ys.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_tower_diff > 0.05,
        "SHW-04: Left and right stress towers in SoftStepTuningGallery must exhibit distinct vertical compression/height profiles (max_tower_diff = {max_tower_diff:.6}m)"
    );

    // Verify live Soft Step solver tuning API and keyboard controls
    let initial_hz = app.world.solver_config().hertz;
    let initial_damping = app.world.solver_config().damping_ratio;

    app.adjust_contact_hertz(10.0);
    assert!(
        (app.world.solver_config().hertz - (initial_hz + 10.0)).abs() < 1e-4,
        "SHW-04: adjust_contact_hertz must update SolverConfig3d::hertz"
    );

    app.adjust_contact_damping(2.0);
    assert!(
        (app.world.solver_config().damping_ratio - (initial_damping + 2.0)).abs() < 1e-4,
        "SHW-04: adjust_contact_damping must update SolverConfig3d::damping_ratio"
    );

    let mut win = Window::new_headless(WindowConfig::default().with_dimensions(1280, 720))
        .expect("headless window");
    app.on_event(
        &mut win,
        &WindowEvent::KeyboardInput {
            key_code: Some(KeyCode::ArrowDown.to_u32()),
            pressed: true,
        },
    );
    assert!(
        app.world.solver_config().hertz < initial_hz + 10.0,
        "SHW-04: ArrowDown key must decrease contact_hertz"
    );
}

/// `SHW-04` Part 2 (3D Gallery 5 `SpatialQueriesGallery`):
/// - Toggling `draw_bvh_aabbs` (`KeyB` / `toggle_bvh_aabbs`) must render broadphase collider AABBs
///   into `quad_batch` (`quads_bvh_on > quads_bvh_off + 100`).
/// - Gallery 5 must execute AABB overlap queries (`spatial_query_overlap_count() > 0`) and
///   a 3D convex shape-cast query (`spatial_query_shape_cast_hit().is_some()`).
#[test]
fn shw_04_3d_spatial_queries_bvh_aabbs_overlap_highlight_and_shape_cast() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(4); // Gallery 5: SpatialQueriesGallery
    app.step_simulation(1.0 / 60.0);

    app.render_debug();
    let quads_bvh_off = app.quad_batch().quad_count();

    app.toggle_bvh_aabbs();
    assert!(app.draw_bvh_aabbs);
    app.render_debug();
    let quads_bvh_on = app.quad_batch().quad_count();

    assert!(
        quads_bvh_on > quads_bvh_off + 100,
        "SHW-04: Enabling draw_bvh_aabbs must emit collider AABB wireframe lines (off={quads_bvh_off}, on={quads_bvh_on})"
    );

    assert!(
        app.spatial_query_overlap_count() > 0,
        "SHW-04: SpatialQueriesGallery must find overlapping dynamic colliders inside query_aabb"
    );
    assert!(
        app.spatial_query_shape_cast_hit().is_some(),
        "SHW-04: SpatialQueriesGallery must perform a 3D shape-cast sweep that hits a target shape"
    );
}

/// `SHW-05` (`Camera3d::clip_and_project_line` & `CameraMode::Fly`):
/// - A 3D line segment crossing the camera near plane must be clipped to the near plane and
///   projected to `Some((s1, s2))` rather than dropped (`None`).
/// - A 3D line segment completely behind the near plane must return `None`.
/// - `toggle_camera_mode()` (`KeyF`) and `fly_camera()` must switch to `CameraMode::Fly` and
///   translate the camera along its view axes.
#[test]
fn shw_05_3d_camera_near_plane_line_clipping_and_fly_mode() {
    let mut cam = Camera3d::new();
    cam.target = Vec3::new(0.0, 1.0, 0.0);
    cam.distance = 5.0;
    cam.pitch = 0.0;
    cam.yaw = 0.0;
    cam.update_orbit_eye();

    // Camera eye is at (0, 1, 5) looking toward (0, 1, 0) along -Z.
    // Floor line from z = -10.0 (in front) to z = 10.0 (behind eye at z = 5.0):
    let p_front = Vec3::new(0.0, 0.0, -10.0);
    let p_behind = Vec3::new(0.0, 0.0, 10.0);

    let clipped = cam.clip_and_project_line(p_front, p_behind, (1280.0, 720.0));
    assert!(
        clipped.is_some(),
        "SHW-05: 3D line crossing the near plane must be clipped to the near plane and projected, not dropped"
    );
    let (s1, s2) = clipped.unwrap();
    assert!(
        s1.0.is_finite() && s1.1.is_finite() && s2.0.is_finite() && s2.1.is_finite(),
        "SHW-05: Clipped screen endpoints must be finite: ({s1:?}, {s2:?})"
    );

    // Line completely behind the camera eye (z = 6.0 to z = 10.0) must be culled
    let both_behind = cam.clip_and_project_line(
        Vec3::new(0.0, 0.0, 6.0),
        Vec3::new(0.0, 0.0, 10.0),
        (1280.0, 720.0),
    );
    assert!(
        both_behind.is_none(),
        "SHW-05: Line with both endpoints behind the near plane must return None"
    );

    // Verify CameraMode::Fly toggle and movement
    let mut app = Physics3dShowcaseApp::new();
    assert_eq!(app.camera.mode, CameraMode::Orbit);
    let eye_before = app.camera.eye;
    app.toggle_camera_mode();
    assert_eq!(app.camera.mode, CameraMode::Fly);
    app.fly_camera(1.0, 0.0, 0.0, 0.5);
    assert!(
        (app.camera.eye - eye_before).length() > 1.0,
        "SHW-05: fly_camera must translate Camera3d::eye in Fly mode"
    );
}

/// `SHW-06` (3D mouse-raycast picking, sparse body iteration & initial cursor delta):
/// - `Camera3d::screen_to_ray` must round-trip with `project_to_screen`.
/// - `Physics3dShowcaseApp::pointer_down` must raycast from screen coordinates, pick a dynamic
///   3D body, attach a 3D `MotorJoint` anchored to a static ground body, drag it via
///   `pointer_move`, and clean up the joint on `pointer_up`.
/// - First `WindowEvent::CursorMoved` while `is_left_down` is true must not jump camera yaw/pitch.
#[test]
fn shw_06_3d_mouse_raycast_picking_sparse_bodies_and_initial_cursor_delta() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(4); // SpatialQueriesGallery has a sphere at (0.0, 1.5, 0.0)

    // Find the target dynamic body near origin (0.0, 1.5, 0.0)
    let (target_id, target_pos) = app
        .world
        .bodies()
        .find(|(_, b)| {
            b.body_type == BodyType::Dynamic && b.position.x.abs() < 0.5 && b.position.z.abs() < 0.5
        })
        .map(|(id, b)| (id, b.position))
        .expect("dynamic body at origin in Gallery 5");

    let screen_pt = app
        .camera
        .project_to_screen(target_pos, (1280.0, 720.0))
        .expect("target body projects onto screen");

    // Verify screen_to_ray points directly at target_pos
    let ray = app.camera.screen_to_ray(screen_pt, (1280.0, 720.0));
    let to_target = (target_pos - app.camera.eye).normalize_or_zero();
    assert!(
        ray.direction.dot(to_target) > 0.999,
        "SHW-06: screen_to_ray must align with vector from eye to projected world point"
    );

    // Pick the body via pointer_down
    let picked = app.pointer_down(screen_pt);
    assert_eq!(
        picked,
        Some(target_id),
        "SHW-06: pointer_down at projected screen coordinates must pick the dynamic body"
    );
    assert!(
        app.drag_joint().is_some(),
        "SHW-06: pointer_down must create a 3D drag MotorJoint"
    );

    // Drag the body to the right on screen and step 30 ticks
    app.pointer_move((screen_pt.0 + 180.0, screen_pt.1 - 60.0));
    for _ in 0..30 {
        app.step_simulation(1.0 / 60.0);
    }

    let moved_pos = app.world.body(target_id).unwrap().position;
    assert!(
        moved_pos.x > target_pos.x + 0.3,
        "SHW-06: Dragged 3D body must move toward the dragged ray target (before={target_pos:?}, after={moved_pos:?})"
    );

    app.pointer_up();
    assert!(
        app.drag_joint().is_none(),
        "SHW-06: pointer_up must destroy the 3D drag joint"
    );

    // Verify first CursorMoved while left mouse button is down does not spike camera yaw/pitch
    let mut app2 = Physics3dShowcaseApp::new();
    let mut win = Window::new_headless(WindowConfig::default().with_dimensions(1280, 720))
        .expect("headless window");
    let yaw_before = app2.camera.yaw;
    let pitch_before = app2.camera.pitch;
    app2.on_event(
        &mut win,
        &WindowEvent::MouseInput {
            button: 0,
            pressed: true,
        },
    );
    app2.on_event(&mut win, &WindowEvent::CursorMoved { x: 700.0, y: 450.0 });
    assert!(
        (app2.camera.yaw - yaw_before).abs() < 1e-5
            && (app2.camera.pitch - pitch_before).abs() < 1e-5,
        "SHW-06: First CursorMoved event must initialize cursor position without spiking camera orbit angles"
    );
}

/// `SHW-07` (Rule 2 dependency hygiene):
/// - `examples/physics_3d_showcase/Cargo.toml` must not depend directly on `winit`.
/// - Every crate declared in `[dependencies]` must be referenced in `src/`.
#[test]
fn shw_07_3d_no_direct_winit_dependency_and_all_declared_dependencies_used() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("read Cargo.toml");
    let deps_section = cargo_toml
        .split("[dev-dependencies]")
        .next()
        .unwrap_or(&cargo_toml);

    assert!(
        !deps_section.contains("winit"),
        "SHW-07: examples/physics_3d_showcase/Cargo.toml [dependencies] must not depend directly on winit"
    );

    let all_src = [
        fs::read_to_string(manifest_dir.join("src/lib.rs")).unwrap_or_default(),
        fs::read_to_string(manifest_dir.join("src/main.rs")).unwrap_or_default(),
        fs::read_to_string(manifest_dir.join("src/app.rs")).unwrap_or_default(),
        fs::read_to_string(manifest_dir.join("src/camera.rs")).unwrap_or_default(),
        fs::read_to_string(manifest_dir.join("src/gallery.rs")).unwrap_or_default(),
    ]
    .join("\n");

    for dep in [
        "korppi_core",
        "korppi_ecs",
        "korppi_engine",
        "korppi_input",
        "korppi_math",
        "korppi_physics",
        "korppi_render",
        "korppi_ui",
        "korppi_window",
    ] {
        assert!(
            all_src.contains(dep),
            "SHW-07: Dependency `{dep}` in Cargo.toml must be used in src/"
        );
    }
}
