//! Regression tests for Milestone FIX-15.1 (`crates/engine`):
//! - `ENG-09` (Probe N9): Zero steady-state heap allocations on `EngineApp::update` with UI
//!   widgets (`Button`, `Label`, `LineEdit`, containers), and preallocated physics/acoustics
//!   scratch buffers + persistent `body3d_to_entity` map on `EngineApp`.
//! - `ENG-10` (Probe N10): `WindowEvent` -> `InputManager` -> `UiInputDispatcher` wiring in
//!   `EngineApp` and `FixedClock::with_max_steps(config.tick_rate, config.max_sub_steps)`.
//! - `ENG-13` (Probe N13): `EngineConfig` validation (`tick_rate == 0` returns `Err` instead of
//!   panicking), `DiagnosticMetrics` (`ui_quad_count`, `draw_calls`, `froxel_light_count`)
//!   population, and rolling `fps` spike guard.
//! - `ENG-14` (Probe N14): Zero steady-state heap allocations in `EngineApp::render_ui_graph`
//!   and target `(width, height)` `canvas_size` scaling.

use korppi_ecs::{ChildOf, Stage};
use korppi_engine::{EngineApp, EngineConfig, GraphicsBackend};
use korppi_input::MouseButton;
use korppi_math::{Vec2, Vec3};
use korppi_physics::{BodyType, Collider, RigidBody, Shape};
use korppi_rhi::{
    BufferDescriptor, BufferTextureCopy, BufferUsage, ImageLayout, ImageMemoryBarrier, MemoryUsage,
    PipelineBarrier, PipelineStageFlags, SampleCount, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSubresourceRange, TextureUsage,
};
use korppi_ui::{
    Button, ButtonState, Control, FocusMode, Label, LineEdit, StyleBox, StyleBoxFlat,
    ThemeOverride, VBoxContainer,
};
use korppi_window::WindowEvent;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct TrackingAllocator;

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

// SAFETY: delegates directly to the standard `System` allocator while updating atomic counters.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: caller upholds `GlobalAlloc::alloc` invariants.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller upholds `GlobalAlloc::dealloc` invariants.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: caller upholds `GlobalAlloc::realloc` invariants.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

fn measure_allocs<F: FnOnce()>(f: F) -> (usize, usize) {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    TRACKING.store(true, Ordering::SeqCst);
    f();
    TRACKING.store(false, Ordering::SeqCst);
    (
        ALLOC_COUNT.load(Ordering::SeqCst),
        ALLOC_BYTES.load(Ordering::SeqCst),
    )
}

const DT: Duration = Duration::from_nanos(16_666_667);

#[test]
fn test_eng_09_zero_alloc_steady_state_ui_and_physics() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = EngineApp::new().expect("engine creation");

    let root = app.world.spawn((
        Control {
            custom_minimum_size: Vec2::new(300.0, 200.0),
            ..Control::default()
        },
        VBoxContainer::new(4.0),
    ));
    let btn = app.world.spawn((
        Control {
            custom_minimum_size: Vec2::new(120.0, 40.0),
            focus_mode: FocusMode::All,
            ..Control::default()
        },
        Button::new("Play"),
    ));
    let lbl = app.world.spawn((
        Control {
            custom_minimum_size: Vec2::new(120.0, 24.0),
            ..Control::default()
        },
        Label::new("Korppi Engine"),
    ));
    let mut line_edit = LineEdit::new();
    line_edit.set_text("Player 1");
    line_edit.placeholder_text = "Enter name".to_string();
    let edit = app.world.spawn((
        Control {
            custom_minimum_size: Vec2::new(120.0, 28.0),
            ..Control::default()
        },
        line_edit,
    ));
    app.world.add_pair::<ChildOf>(btn, root);
    app.world.add_pair::<ChildOf>(lbl, root);
    app.world.add_pair::<ChildOf>(edit, root);

    for i in 0..8 {
        app.world.spawn((
            RigidBody::new(BodyType::Static, Vec3::new(i as f32 * 2.0, 0.0, 0.0)),
            Collider::new(Shape::Box {
                half_extents: Vec3::ONE,
            }),
        ));
    }

    // Warm up 5 frames
    for _ in 0..5 {
        app.update(DT);
    }
    assert!(!app.ui_draw_commands.is_empty());

    let (allocs, bytes) = measure_allocs(|| {
        for _ in 0..60 {
            app.update(DT);
        }
    });
    assert_eq!(
        allocs, 0,
        "ENG-09: steady-state 60 ticks with UI Button/Label/LineEdit/VBoxContainer must perform 0 heap allocations, got {allocs} allocs ({bytes} bytes)"
    );
}

#[test]
fn test_eng_10_input_to_ui_dispatch_and_max_sub_steps() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let config = EngineConfig {
        tick_rate: 60,
        max_sub_steps: 2,
        ..EngineConfig::default()
    };
    let mut app = EngineApp::with_config(config).expect("engine creation");

    let fixed_counter = Arc::new(AtomicUsize::new(0));
    let fc = Arc::clone(&fixed_counter);
    app.schedule
        .add_system(Stage::FixedUpdate, move |_world: &mut korppi_ecs::World| {
            fc.fetch_add(1, Ordering::SeqCst);
        });

    let btn_entity = app.world.spawn((
        Control {
            custom_minimum_size: Vec2::new(200.0, 60.0),
            focus_mode: FocusMode::All,
            ..Control::default()
        },
        Button::new("ClickMe"),
    ));

    // Feed mouse move and left button press via InputManager directly (Probe N10 pattern)
    app.input.mouse.on_moved(Vec2::new(50.0, 20.0));
    app.input.mouse.on_button_pressed(MouseButton::Left);

    // 200ms delta = 12 ticks at 60 Hz, must be clamped to config.max_sub_steps = 2
    app.update(Duration::from_millis(200));

    assert_eq!(
        fixed_counter.load(Ordering::SeqCst),
        2,
        "ENG-10: 200ms delta with max_sub_steps=2 must clamp fixed updates to 2"
    );
    assert_eq!(
        app.ui_dispatcher.focus_manager.focused_control(),
        Some(btn_entity),
        "ENG-10: clicking inside Button via app.input must focus the button in ui_dispatcher"
    );
    assert_eq!(
        app.world.get::<Button>(btn_entity).unwrap().state,
        ButtonState::Pressed,
        "ENG-10: pressing Left mouse button over Button must transition it to ButtonState::Pressed"
    );

    // Also verify WindowEvent ingestion via handle_window_event releases the button
    app.handle_window_event(&WindowEvent::MouseInput {
        button: 0,
        pressed: false,
    });
    app.update(DT);
    assert_ne!(
        app.world.get::<Button>(btn_entity).unwrap().state,
        ButtonState::Pressed,
        "ENG-10: releasing Left mouse button via handle_window_event must release the Button"
    );
}

#[test]
fn test_eng_13_config_validation_and_diagnostic_metrics() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // 1. tick_rate == 0 must return Err without panicking
    let zero_tick_res = std::panic::catch_unwind(|| {
        EngineApp::with_config(EngineConfig {
            tick_rate: 0,
            ..EngineConfig::default()
        })
    });
    assert!(
        zero_tick_res.is_ok(),
        "ENG-13: EngineApp::with_config(tick_rate = 0) must not panic"
    );
    assert!(
        zero_tick_res.unwrap().is_err(),
        "ENG-13: EngineApp::with_config(tick_rate = 0) must return Err"
    );

    // 2. max_sub_steps == 0 must also return Err
    assert!(
        EngineApp::with_config(EngineConfig {
            max_sub_steps: 0,
            ..EngineConfig::default()
        })
        .is_err(),
        "ENG-13: EngineApp::with_config(max_sub_steps = 0) must return Err"
    );

    // 3. DiagnosticMetrics ui_quad_count, draw_calls, and froxel_light_count
    let mut app = EngineApp::new().expect("engine creation");
    let ctrl = Control {
        custom_minimum_size: Vec2::new(100.0, 50.0),
        ..Control::default()
    };
    let mut theme = ThemeOverride::default();
    theme.set_stylebox(
        "panel",
        StyleBox::Flat(StyleBoxFlat {
            bg_color: [1.0, 1.0, 1.0, 1.0],
            ..StyleBoxFlat::default()
        }),
    );
    app.world.spawn((ctrl, theme));
    let mut culling = korppi_render::clustered::CullingOutput::default();
    culling.global_light_indices.push(0);
    app.world.insert_resource(culling);

    app.update(DT);
    assert_eq!(app.ui_quad_batch.quad_count(), 1);
    assert_eq!(
        app.metrics.ui_quad_count, 1,
        "ENG-13: metrics.ui_quad_count must reflect compiled UI quads"
    );
    assert!(
        app.metrics.draw_calls >= 1,
        "ENG-13: metrics.draw_calls must be populated when UI quads are drawn"
    );
    assert_eq!(
        app.metrics.froxel_light_count, 1,
        "ENG-13: metrics.froxel_light_count must be populated from active world lights"
    );

    // 4. 1ns frame spike followed by 60 FPS frame must not poison metrics.fps to 90M FPS
    app.update(Duration::from_nanos(1));
    app.update(DT);
    assert!(
        app.metrics.fps.is_finite() && app.metrics.fps <= 10_000.0,
        "ENG-13: metrics.fps must not be poisoned by a 1ns frame spike, got {}",
        app.metrics.fps
    );
}

#[test]
fn test_eng_14_render_ui_graph_zero_alloc_and_target_canvas_scaling() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let config = EngineConfig {
        backend: GraphicsBackend::Vulkan,
        window_size: Vec2::new(1280.0, 720.0),
        ..EngineConfig::default()
    };
    let mut app = EngineApp::with_config(config).expect("engine creation");
    app.clear_color = [0.0, 0.0, 0.0, 1.0];

    // Spawn a 64x64 solid red panel at (0, 0)
    let ctrl = Control {
        custom_minimum_size: Vec2::new(64.0, 64.0),
        ..Control::default()
    };
    let mut theme = ThemeOverride::default();
    theme.set_stylebox(
        "panel",
        StyleBox::Flat(StyleBoxFlat {
            bg_color: [1.0, 0.0, 0.0, 1.0],
            border_width: [0.0; 4],
            ..StyleBoxFlat::default()
        }),
    );
    app.world.spawn((ctrl, theme));
    app.update(DT);

    let device = app.graphics_device.as_ref().unwrap().clone();
    let target = device
        .create_texture(
            &TextureDescriptor {
                width: 64,
                height: 64,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
                dimension: TextureDimension::D2,
                label: Some("eng14_target"),
            },
            None,
        )
        .expect("create_texture");

    // Single warmup call (matches Probe N14)
    app.render_ui_graph(target, TextureFormat::Rgba8Unorm, 64, 64)
        .expect("warmup render_ui_graph");

    let (allocs, bytes) = measure_allocs(|| {
        app.render_ui_graph(target, TextureFormat::Rgba8Unorm, 64, 64)
            .expect("steady render_ui_graph");
    });
    assert_eq!(
        allocs, 0,
        "ENG-14: steady-state render_ui_graph must perform 0 heap allocations, got {allocs} ({bytes} bytes)"
    );

    // Read back pixels and verify the 64x64 panel covers pixel (60, 60) even though window_size is 1280x720
    let byte_len = 64 * 64 * 4;
    let readback = device
        .create_buffer(
            &BufferDescriptor {
                size: byte_len,
                usage: BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuToCpu,
                label: Some("eng14_readback"),
            },
            None,
        )
        .expect("readback buffer");
    let mut rec = device.create_command_recorder().expect("recorder");
    rec.begin().expect("begin");
    rec.pipeline_barrier(
        &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture: target,
            range: TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access: korppi_rhi::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stages: PipelineStageFlags::TRANSFER,
            dst_access: korppi_rhi::AccessFlags::TRANSFER_READ,
            old_layout: ImageLayout::ColorAttachment,
            new_layout: ImageLayout::TransferSrc,
        }),
    );
    rec.copy_texture_to_buffer(
        target,
        readback,
        &BufferTextureCopy {
            buffer_offset: 0,
            buffer_row_pitch: 0,
            texture_mip_level: 0,
            texture_layer: 0,
            origin: [0, 0, 0],
            extent: [64, 64, 1],
        },
    );
    let sub = device
        .submit(rec.finish().expect("finish"))
        .expect("submit");
    device.wait_for_submission(sub).expect("wait");

    let pixels = device
        .read_buffer(readback, 0, byte_len)
        .expect("read_buffer");

    let idx = (60 * 64 + 60) * 4;
    let br_pixel = &pixels[idx..idx + 4];
    assert!(
        br_pixel[0] >= 240 && br_pixel[1] <= 15 && br_pixel[2] <= 15,
        "ENG-14: bottom-right pixel (60, 60) of 64x64 offscreen target must be red, got {br_pixel:?}"
    );

    device.destroy_buffer(readback).expect("destroy readback");
    device.destroy_texture(target).expect("destroy target");
}
