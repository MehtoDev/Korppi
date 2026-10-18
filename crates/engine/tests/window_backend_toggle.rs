//! Integration tests for `FIX-10.3` (`ENG-03`): runtime graphics backend switching
//! across live window swapchains in `WindowPresenter` and `EngineApp`.

use korppi_engine::{EngineApp, EngineConfig, GraphicsBackend, WindowPresenter};
use korppi_math::Vec2;
use korppi_render::render2d::QuadBatch2d;
use korppi_rhi::{PresentMode, RhiError, SwapchainDescriptor};
use korppi_rhi_conformance::display::{display_or_skip, TestWindow};
use std::sync::Arc;
use std::time::{Duration, Instant};

const W: u32 = 64;
const H: u32 = 48;

fn full_window_quad(width: u32, height: u32, color: [f32; 4]) -> QuadBatch2d {
    let mut batch = QuadBatch2d::new(1);
    batch.push_quad(
        Vec2::ZERO,
        Vec2::new(width as f32, height as f32),
        Vec2::ZERO,
        Vec2::ONE,
        color,
    );
    batch
}

fn present_and_expect_color(
    presenter: &mut WindowPresenter,
    window: &TestWindow,
    width: u32,
    height: u32,
    color_f32: [f32; 4],
    expected_rgba8: [u8; 4],
) {
    let batch = full_window_quad(width, height, color_f32);
    let canvas = (width as f32, height as f32);
    let clear = [0.0, 0.0, 0.0, 1.0];

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let presented = presenter
            .present_quads(&batch, canvas, clear)
            .expect("present_quads should succeed");
        assert!(
            presented,
            "swapchain image should be acquired and presented"
        );
        presenter
            .device()
            .wait_idle()
            .expect("wait_idle should succeed");

        let img = window.capture(width, height);
        let center = img.pixel(width / 2, height / 2);
        if center == expected_rgba8 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "window pixel at ({}, {}) did not reach {expected_rgba8:?} on {:?} (last seen: {center:?})",
            width / 2,
            height / 2,
            presenter.backend(),
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn window_presenter_toggles_between_vulkan_and_opengl_on_same_window() {
    let Some(display) = display_or_skip("window_presenter_backend_toggle") else {
        return;
    };
    let window = Arc::new(TestWindow::create(&display, W, H));
    let desc = SwapchainDescriptor {
        present_mode: PresentMode::Immediate,
        ..SwapchainDescriptor::new(W, H)
    };

    let mut presenter = WindowPresenter::new(GraphicsBackend::Vulkan, window.clone(), &desc)
        .expect("Vulkan WindowPresenter creation should succeed");
    assert_eq!(presenter.backend(), GraphicsBackend::Vulkan);

    // 1. Render and verify red frame on Vulkan.
    present_and_expect_color(
        &mut presenter,
        &window,
        W,
        H,
        [1.0, 0.0, 0.0, 1.0],
        [255, 0, 0, 255],
    );

    // 2. Toggle live presenter to OpenGL on the same window.
    presenter
        .set_backend(GraphicsBackend::OpenGl)
        .expect("toggling WindowPresenter from Vulkan to OpenGL should succeed");
    assert_eq!(presenter.backend(), GraphicsBackend::OpenGl);
    let info_gl = presenter.info().expect("OpenGL swapchain info");
    assert_eq!((info_gl.width, info_gl.height), (W, H));

    present_and_expect_color(
        &mut presenter,
        &window,
        W,
        H,
        [0.0, 1.0, 0.0, 1.0],
        [0, 255, 0, 255],
    );

    // 3. Resize while on OpenGL, then toggle back to Vulkan and verify the resized dimensions hold.
    let (new_w, new_h) = (96, 72);
    window.resize(new_w, new_h);
    presenter
        .resize(new_w, new_h)
        .expect("resizing OpenGL swapchain should succeed");

    presenter
        .set_backend(GraphicsBackend::Vulkan)
        .expect("toggling WindowPresenter from OpenGL back to Vulkan should succeed");
    assert_eq!(presenter.backend(), GraphicsBackend::Vulkan);
    let info_vk = presenter
        .info()
        .expect("Vulkan swapchain info after resize");
    assert_eq!((info_vk.width, info_vk.height), (new_w, new_h));

    present_and_expect_color(
        &mut presenter,
        &window,
        new_w,
        new_h,
        [0.0, 0.0, 1.0, 1.0],
        [0, 0, 255, 255],
    );
}

#[test]
fn engine_app_migrates_attached_window_presenter_and_rejects_headless_while_attached() {
    let Some(display) = display_or_skip("engine_app_window_backend_toggle") else {
        return;
    };
    let window = Arc::new(TestWindow::create(&display, W, H));
    let desc = SwapchainDescriptor {
        present_mode: PresentMode::Immediate,
        ..SwapchainDescriptor::new(W, H)
    };

    let config = EngineConfig {
        backend: GraphicsBackend::Vulkan,
        ..EngineConfig::default()
    };
    let mut app = EngineApp::with_config(config).expect("EngineApp creation should succeed");

    app.attach_window(window.clone(), &desc)
        .expect("attaching window to EngineApp should succeed");
    assert_eq!(
        app.presenter.as_ref().map(WindowPresenter::backend),
        Some(GraphicsBackend::Vulkan)
    );
    assert!(Arc::ptr_eq(
        app.graphics_device.as_ref().expect("graphics_device"),
        app.presenter.as_ref().expect("presenter").device()
    ));

    present_and_expect_color(
        app.presenter.as_mut().expect("presenter"),
        &window,
        W,
        H,
        [1.0, 0.0, 0.0, 1.0],
        [255, 0, 0, 255],
    );

    // Toggle EngineApp from Vulkan to OpenGL while window is attached.
    app.set_graphics_backend(GraphicsBackend::OpenGl)
        .expect("toggling EngineApp with attached window to OpenGL should succeed");
    assert_eq!(app.config.backend, GraphicsBackend::OpenGl);
    assert_eq!(
        app.presenter.as_ref().map(WindowPresenter::backend),
        Some(GraphicsBackend::OpenGl)
    );
    assert!(Arc::ptr_eq(
        app.graphics_device.as_ref().expect("graphics_device"),
        app.presenter.as_ref().expect("presenter").device()
    ));

    present_and_expect_color(
        app.presenter.as_mut().expect("presenter"),
        &window,
        W,
        H,
        [0.0, 1.0, 0.0, 1.0],
        [0, 255, 0, 255],
    );

    // Attempting to switch to Headless while a window is attached must be rejected without
    // breaking the active presenter.
    let err = app
        .set_graphics_backend(GraphicsBackend::Headless)
        .expect_err("Headless backend must be rejected while a window presenter is attached");
    assert!(
        matches!(err, RhiError::Unsupported(_)),
        "expected RhiError::Unsupported, got {err:?}"
    );
    assert_eq!(app.config.backend, GraphicsBackend::OpenGl);
    present_and_expect_color(
        app.presenter.as_mut().expect("presenter"),
        &window,
        W,
        H,
        [1.0, 1.0, 0.0, 1.0],
        [255, 255, 0, 255],
    );

    // Detaching the window releases the swapchain and allows switching to Headless.
    app.detach_window()
        .expect("detaching window should succeed");
    assert!(app.presenter.is_none());
    app.set_graphics_backend(GraphicsBackend::Headless)
        .expect("switching to Headless after detaching window should succeed");
    assert!(app.graphics_device.is_none());
}
