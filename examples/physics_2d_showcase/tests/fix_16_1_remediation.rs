//! Regression tests for Milestone FIX-16.1 (`examples/physics_2d_showcase` remediation:
//! `SHW-01`, `SHW-02`, and `SHW-03`).

use korppi_ui::render::UiDrawCommand;
use korppi_ui::widgets::Label;
use korppi_window::{KorppiApp, Window, WindowConfig, WindowEvent};
use physics_2d_showcase_lib::app::Physics2dShowcaseApp;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct TrackingAllocator;

thread_local! {
    static TRACKING_ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[global_allocator]
static GLOBAL_ALLOC: TrackingAllocator = TrackingAllocator;

// SAFETY: Delegates directly to the OS `System` allocator while incrementing thread-local counters.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TRACKING_ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: Forwarded to `System::alloc` with the caller's layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TRACKING_ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: Forwarded to `System::realloc` with the caller's pointer, layout, and size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Forwarded to `System::dealloc` with the caller's pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn measure_allocs<F: FnOnce()>(f: F) -> usize {
    ALLOC_COUNT.with(|c| c.set(0));
    TRACKING_ENABLED.with(|e| e.set(true));
    f();
    TRACKING_ENABLED.with(|e| e.set(false));
    ALLOC_COUNT.with(|c| c.get())
}

#[test]
fn shw_01_2d_fixed_timestep_loop_and_single_step_invariance() {
    // 1. 1.0 second of simulated wall-clock time at 30 FPS vs 120 FPS must execute the exact
    //    same 60 fixed simulation ticks (dt = 1/60 s) and produce identical world state.
    let mut app_30fps = Physics2dShowcaseApp::new();
    let mut win_30 = Window::new_headless(WindowConfig::default()).unwrap();
    for _ in 0..30 {
        app_30fps.on_update(&mut win_30, 1.0 / 30.0);
    }

    let mut app_120fps = Physics2dShowcaseApp::new();
    let mut win_120 = Window::new_headless(WindowConfig::default()).unwrap();
    for _ in 0..120 {
        app_120fps.on_update(&mut win_120, 1.0 / 120.0);
    }

    assert_eq!(
        app_30fps.world.step_index(),
        60,
        "30 frames at 1/30 s must execute 60 fixed 60 Hz ticks"
    );
    assert_eq!(
        app_120fps.world.step_index(),
        60,
        "120 frames at 1/120 s must execute 60 fixed 60 Hz ticks"
    );
    assert_eq!(
        app_30fps.world.state_hash(),
        app_120fps.world.state_hash(),
        "30 FPS and 120 FPS runs must produce bit-identical 2D physics state after 1.0 s"
    );

    // 2. Single-stepping while paused must advance by a fixed 1/60 s tick regardless of the
    //    frame's wall-clock delta_time.
    let mut app_step_30 = Physics2dShowcaseApp::new();
    app_step_30.toggle_pause();
    app_step_30.request_single_step();
    app_step_30.on_update(&mut win_30, 1.0 / 30.0);
    // Subsequent paused frame without single-step must not advance the simulation
    app_step_30.on_update(&mut win_30, 1.0 / 30.0);

    let mut app_step_120 = Physics2dShowcaseApp::new();
    app_step_120.toggle_pause();
    app_step_120.request_single_step();
    app_step_120.on_update(&mut win_120, 1.0 / 120.0);
    app_step_120.on_update(&mut win_120, 1.0 / 120.0);

    assert_eq!(app_step_30.world.step_index(), 1);
    assert_eq!(app_step_120.world.step_index(), 1);
    assert_eq!(
        app_step_30.world.state_hash(),
        app_step_120.world.state_hash(),
        "Paused single-step must advance by fixed 1/60 s regardless of wall-clock delta_time"
    );
}

#[test]
fn shw_02_2d_telemetry_hud_uses_korppi_ui_and_updates_metrics() {
    let mut app = Physics2dShowcaseApp::new();
    app.step_simulation(1.0 / 60.0);

    // Disable all physics debug geometry so only the korppi_ui HUD is compiled into quad_batch
    app.draw_shapes = false;
    app.draw_joints = false;
    app.draw_contacts = false;
    app.draw_sensors = false;
    app.render_debug();

    // Verify UiCanvasRenderer generated real stylebox, text, and button commands
    assert!(
        app.ui_draw_commands
            .iter()
            .any(|c| matches!(c, UiDrawCommand::StyleBoxFlat { .. })),
        "HUD must emit a StyleBoxFlat panel background command"
    );
    let text_cmds: Vec<&str> = app
        .ui_draw_commands
        .iter()
        .filter_map(|c| match c {
            UiDrawCommand::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        text_cmds.len() >= 4,
        "HUD must emit Label text commands for gallery title, description, timing, and counts"
    );
    assert!(
        text_cmds.iter().any(|t| t.contains("Stacking")),
        "HUD must display the active gallery name ('Stacking & Friction'), got: {text_cmds:?}"
    );
    assert!(
        text_cmds.iter().any(|t| t.contains("Bodies:")),
        "HUD must display body count telemetry, got: {text_cmds:?}"
    );
    assert!(
        app.ui_draw_commands
            .iter()
            .any(|c| matches!(c, UiDrawCommand::Button { .. })),
        "HUD must emit interactive Button draw commands"
    );
    assert!(
        app.quad_batch.quad_count() >= 10,
        "HUD must compile panel + borders + labels + buttons into at least 10 quads (got {})",
        app.quad_batch.quad_count()
    );

    // Switching gallery must update HUD labels
    app.switch_gallery(1);
    app.render_debug();
    let mut found_joints_title = false;
    {
        let mut query = app.ui_world.query_mut::<&Label>();
        for label in &mut query {
            if label.text().contains("Joints") {
                found_joints_title = true;
            }
        }
    }
    assert!(
        found_joints_title,
        "Switching to Gallery 2 must update the korppi_ui Label text to mention 'Joints'"
    );

    // Interactive HUD Next button click via WindowEvent must switch gallery
    let mut win = Window::new_headless(WindowConfig::default()).unwrap();
    let next_btn_rect = app
        .ui_draw_commands
        .iter()
        .find_map(|c: &UiDrawCommand| match c {
            UiDrawCommand::Button { rect, text, .. } if text.contains("Next") => Some(*rect),
            _ => None,
        })
        .expect("HUD must include a 'Next' button");
    let click_x = next_btn_rect.x + next_btn_rect.width * 0.5;
    let click_y = next_btn_rect.y + next_btn_rect.height * 0.5;
    app.on_event(
        &mut win,
        &WindowEvent::CursorMoved {
            x: f64::from(click_x),
            y: f64::from(click_y),
        },
    );
    app.on_event(
        &mut win,
        &WindowEvent::MouseInput {
            button: 0,
            pressed: true,
        },
    );
    app.on_event(
        &mut win,
        &WindowEvent::MouseInput {
            button: 0,
            pressed: false,
        },
    );
    assert_eq!(
        app.current_gallery, 2,
        "Clicking the HUD 'Next' button must advance current_gallery from 1 to 2"
    );
}

#[test]
fn shw_03_2d_zero_steady_state_heap_allocations() {
    let mut app = Physics2dShowcaseApp::new();
    // Warm up caches and scratch pools
    for _ in 0..5 {
        app.step_simulation(1.0 / 60.0);
        app.render_debug();
    }

    let step_allocs = measure_allocs(|| {
        app.step_simulation(1.0 / 60.0);
    });
    assert_eq!(
        step_allocs, 0,
        "2D step_simulation must perform 0 steady-state heap allocations (got {step_allocs})"
    );

    let render_allocs = measure_allocs(|| {
        app.render_debug();
    });
    assert_eq!(
        render_allocs, 0,
        "2D render_debug must perform 0 steady-state heap allocations (got {render_allocs})"
    );
}
