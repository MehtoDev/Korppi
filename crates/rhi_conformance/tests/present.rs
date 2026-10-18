//! Presentation to X11 windows (FIX-1.3; findings RHI-02, RVK-02, RGL-07).
//!
//! Every test presents into a real window, on `$DISPLAY` or on an `Xvfb` server it starts, and
//! reads back what the window shows. Without a display the tests are skipped (see
//! `korppi_rhi_conformance::display`).
#![cfg(target_os = "linux")]

use korppi_rhi::{
    ColorTargetState, ColorWriteMask, CullMode, FrontFace, GraphicsPipelineDescriptor, LoadOp,
    PipelineHandle, PolygonMode, PresentMode, PrimitiveTopology, RasterizerState, RhiError,
    ShaderStage, SubmissionId, SwapchainDescriptor, SwapchainHandle, SwapchainImage, TextureFormat,
    TextureUsage, VertexLayoutDescriptor,
};
use korppi_rhi_conformance::display::{display_or_skip, TestWindow, VirtualDisplay};
use korppi_rhi_conformance::{
    approx_rgba, color_pass, undefined_to_color_attachment, Backend, Image, TestDevice,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

const W: u32 = 64;
const H: u32 = 48;
/// Linear 0.5 blue: an sRGB swapchain stores and displays it as 188.
const CLEAR: [f32; 4] = [0.0, 0.0, 0.5, 1.0];
const RED: [u8; 4] = [255, 0, 0, 255];
const SRGB_HALF_BLUE: [u8; 4] = [0, 0, 188, 255];

/// CCW triangle in +Y-up NDC that covers exactly the top half of the viewport.
const TOP_HALF_TRIANGLE: &str = r#"#version 450
const vec2 P[3] = vec2[3](vec2(-1.0, 0.0), vec2(3.0, 0.0), vec2(-1.0, 4.0));
void main() { gl_Position = vec4(P[gl_VertexIndex], 0.5, 1.0); }
"#;

const OPAQUE_RED: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.0, 0.0, 1.0); }
"#;

/// A window, a device that can present to it, and a swapchain.
struct Presentation {
    window: Arc<TestWindow>,
    device: TestDevice,
    swapchain: SwapchainHandle,
}

impl Presentation {
    fn open(backend: Backend, display: &VirtualDisplay, desc: &SwapchainDescriptor) -> Self {
        let window = Arc::new(TestWindow::create(display, desc.width, desc.height));
        let device = TestDevice::open_for_display(backend, window.clone());
        let swapchain = device
            .create_swapchain(window.clone(), desc)
            .unwrap_or_else(|e| panic!("{backend:?}: create_swapchain failed: {e}"));
        Self {
            window,
            device,
            swapchain,
        }
    }

    fn acquire(&self) -> SwapchainImage {
        self.device
            .acquire_next_image(self.swapchain)
            .expect("acquire_next_image")
            .expect("the window has an area, so an image is available")
    }

    fn pipeline(&self) -> PipelineHandle {
        let format = self
            .device
            .swapchain_info(self.swapchain)
            .expect("swapchain_info")
            .format;
        let vertex_shader = self.device.shader(ShaderStage::Vertex, TOP_HALF_TRIANGLE);
        let fragment_shader = self.device.shader(ShaderStage::Fragment, OPAQUE_RED);
        self.device
            .create_graphics_pipeline(&GraphicsPipelineDescriptor {
                vertex_shader,
                fragment_shader: Some(fragment_shader),
                vertex_layout: VertexLayoutDescriptor::new(),
                topology: PrimitiveTopology::TriangleList,
                rasterizer: RasterizerState {
                    cull_mode: CullMode::None,
                    front_face: FrontFace::CounterClockwise,
                    polygon_mode: PolygonMode::Fill,
                    depth_bias: 0.0,
                    slope_scaled_depth_bias: 0.0,
                    depth_bias_clamp: 0.0,
                    depth_clamp: false,
                },
                multisample: Default::default(),
                depth_stencil: None,
                color_targets: vec![ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWriteMask::ALL,
                }],
                bind_group_layouts: vec![],
                push_constant_ranges: vec![],
                label: Some("present_top_half"),
            })
            .expect("create_graphics_pipeline")
    }

    /// Clears `image` to [`CLEAR`] and draws its top half red.
    fn render(&self, image: &SwapchainImage, pipeline: PipelineHandle) -> SubmissionId {
        let mut recorder = self.device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.pipeline_barrier(&undefined_to_color_attachment(image.texture));
        recorder
            .begin_rendering(&color_pass(
                image.texture,
                image.width,
                image.height,
                LoadOp::Clear,
                CLEAR,
            ))
            .expect("begin_rendering");
        recorder.set_viewport(0.0, 0.0, image.width as f32, image.height as f32, 0.0, 1.0);
        recorder.set_scissor(0, 0, image.width, image.height);
        recorder.bind_pipeline(pipeline);
        recorder.draw(3, 1, 0, 0);
        recorder.end_rendering().expect("end_rendering");
        self.device
            .submit(recorder.finish().expect("finish"))
            .expect("submit")
    }

    /// Renders and presents one frame.
    fn frame(&self, pipeline: PipelineHandle) -> SwapchainImage {
        self.device.begin_frame().expect("begin_frame");
        let image = self.acquire();
        self.render(&image, pipeline);
        self.device.present(&image).expect("present");
        image
    }

    /// Captures the window until `expected` holds (presentation may complete asynchronously),
    /// for at most two seconds, and returns the last capture.
    fn wait_for_window(&self, width: u32, height: u32, expected: impl Fn(&Image) -> bool) -> Image {
        self.device.wait_idle().expect("wait_idle");
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let image = self.window.capture(width, height);
            if expected(&image) || Instant::now() > deadline {
                return image;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Presentation {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.device
                .destroy_swapchain(self.swapchain)
                .expect("destroy_swapchain");
        }
    }
}

/// True if the top half of `image` is red and the bottom half is the sRGB-encoded clear colour.
fn shows_red_over_blue(image: &Image) -> bool {
    let half = image.height / 2;
    (0..image.height).all(|y| {
        let expected = if y < half { RED } else { SRGB_HALF_BLUE };
        (0..image.width).all(|x| approx_rgba(image.pixel(x, y), expected, 3))
    })
}

fn describe(image: &Image) -> String {
    let rows: Vec<String> = [0, image.height / 2 - 1, image.height / 2, image.height - 1]
        .iter()
        .map(|&y| format!("row {y}: {:?}", image.pixel(image.width / 2, y)))
        .collect();
    rows.join(", ")
}

#[test]
fn presented_frames_reach_the_window_top_row_first_and_srgb_encoded() {
    let Some(display) = display_or_skip("present") else {
        return;
    };
    for backend in Backend::ALL {
        let presentation = Presentation::open(backend, &display, &SwapchainDescriptor::new(W, H));
        let info = presentation
            .device
            .swapchain_info(presentation.swapchain)
            .expect("swapchain_info");
        assert!(info.format.is_srgb(), "{backend:?}: {info:?}");
        assert_eq!((info.width, info.height), (W, H), "{backend:?}");
        assert!(info.usage.contains(TextureUsage::COLOR_ATTACHMENT));
        assert!(info.image_count >= 2, "{backend:?}: {info:?}");

        let pipeline = presentation.pipeline();
        for _ in 0..3 {
            let image = presentation.frame(pipeline);
            assert_eq!((image.width, image.height), (W, H));
        }
        let shown = presentation.wait_for_window(W, H, shows_red_over_blue);
        assert!(
            shows_red_over_blue(&shown),
            "{backend:?}: window does not show red over sRGB blue: {}",
            describe(&shown)
        );
    }
}

#[test]
fn resizing_recreates_the_swapchain_at_the_new_size() {
    let Some(display) = display_or_skip("present_resize") else {
        return;
    };
    for backend in Backend::ALL {
        let presentation = Presentation::open(backend, &display, &SwapchainDescriptor::new(W, H));
        let pipeline = presentation.pipeline();
        presentation.frame(pipeline);

        let (w, h) = (96, 72);
        presentation.window.resize(w, h);
        presentation
            .device
            .resize_swapchain(presentation.swapchain, w, h)
            .expect("resize_swapchain");
        let image = presentation.frame(pipeline);
        assert_eq!((image.width, image.height), (w, h), "{backend:?}");
        let info = presentation
            .device
            .swapchain_info(presentation.swapchain)
            .expect("swapchain_info");
        assert_eq!((info.width, info.height), (w, h), "{backend:?}");

        let shown = presentation.wait_for_window(w, h, shows_red_over_blue);
        assert!(
            shows_red_over_blue(&shown),
            "{backend:?}: resized window: {}",
            describe(&shown)
        );
    }
}

#[test]
fn acquisition_pauses_while_the_swapchain_has_no_area() {
    let Some(display) = display_or_skip("present_zero_area") else {
        return;
    };
    for backend in Backend::ALL {
        let presentation = Presentation::open(backend, &display, &SwapchainDescriptor::new(W, H));
        let device = &presentation.device;
        device
            .resize_swapchain(presentation.swapchain, 0, 0)
            .expect("resize to zero");
        assert_eq!(
            device.acquire_next_image(presentation.swapchain),
            Ok(None),
            "{backend:?}: a zero-area swapchain has no image to acquire"
        );
        let info = device.swapchain_info(presentation.swapchain).expect("info");
        assert_eq!((info.width, info.height), (0, 0), "{backend:?}");

        device
            .resize_swapchain(presentation.swapchain, W, H)
            .expect("resize back");
        let pipeline = presentation.pipeline();
        let image = presentation.frame(pipeline);
        assert_eq!((image.width, image.height), (W, H), "{backend:?}");
    }
}

#[test]
fn many_frames_reuse_synchronization_objects_without_api_errors() {
    let Some(display) = display_or_skip("present_many_frames") else {
        return;
    };
    for backend in Backend::ALL {
        let presentation = Presentation::open(backend, &display, &SwapchainDescriptor::new(W, H));
        let pipeline = presentation.pipeline();
        for _ in 0..30 {
            presentation.frame(pipeline);
        }
        // Presenting an image without rendering into it is valid too (its contents are
        // undefined): the backend still synchronizes and transitions it.
        let image = presentation.acquire();
        presentation.device.present(&image).expect("present");
        presentation.device.wait_idle().expect("wait_idle");
        presentation.device.assert_no_api_errors();
    }
}

#[test]
fn swapchain_misuse_is_rejected() {
    let Some(display) = display_or_skip("present_misuse") else {
        return;
    };
    for backend in Backend::ALL {
        let presentation = Presentation::open(backend, &display, &SwapchainDescriptor::new(W, H));
        let device = &presentation.device;
        let image = presentation.acquire();
        assert!(
            matches!(
                device.acquire_next_image(presentation.swapchain),
                Err(RhiError::ValidationError(_))
            ),
            "{backend:?}: acquiring again before presenting must fail"
        );
        assert!(
            device.destroy_texture(image.texture).is_err(),
            "{backend:?}: swapchain textures belong to the swapchain"
        );
        device.present(&image).expect("present");
        assert!(
            matches!(device.present(&image), Err(RhiError::ValidationError(_))),
            "{backend:?}: an image cannot be presented twice"
        );
    }
}

#[test]
fn destroying_a_swapchain_releases_its_window_and_textures() {
    let Some(display) = display_or_skip("present_destroy") else {
        return;
    };
    for backend in Backend::ALL {
        let window = Arc::new(TestWindow::create(&display, W, H));
        let device = TestDevice::open_for_display(backend, window.clone());
        let references = Arc::strong_count(&window);
        let swapchain = device
            .create_swapchain(window.clone(), &SwapchainDescriptor::new(W, H))
            .expect("create_swapchain");
        assert!(Arc::strong_count(&window) > references);
        let image = device
            .acquire_next_image(swapchain)
            .expect("acquire")
            .expect("image");
        device.present(&image).expect("present");

        device
            .destroy_swapchain(swapchain)
            .expect("destroy_swapchain");
        assert_eq!(
            Arc::strong_count(&window),
            references,
            "{backend:?}: the swapchain must release the window"
        );
        assert!(matches!(
            device.acquire_next_image(swapchain),
            Err(RhiError::InvalidHandle(_))
        ));
        assert!(matches!(
            device.destroy_swapchain(swapchain),
            Err(RhiError::InvalidHandle(_))
        ));
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        // The texture of a destroyed swapchain is no longer a valid render target.
        let rendered =
            recorder.begin_rendering(&color_pass(image.texture, W, H, LoadOp::Clear, CLEAR));
        if rendered.is_ok() {
            let _ = recorder.end_rendering();
            assert!(
                recorder.finish().is_err(),
                "{backend:?}: recording into a destroyed swapchain texture must fail"
            );
        }
    }
}

#[test]
fn present_mode_and_format_requests_are_honoured_or_fall_back() {
    let Some(display) = display_or_skip("present_modes") else {
        return;
    };
    for backend in Backend::ALL {
        for mode in [
            PresentMode::Fifo,
            PresentMode::FifoRelaxed,
            PresentMode::Mailbox,
            PresentMode::Immediate,
        ] {
            let desc = SwapchainDescriptor {
                present_mode: mode,
                srgb: false,
                ..SwapchainDescriptor::new(W, H)
            };
            let presentation = Presentation::open(backend, &display, &desc);
            let info = presentation
                .device
                .swapchain_info(presentation.swapchain)
                .expect("swapchain_info");
            assert!(
                info.present_mode == mode || info.present_mode == PresentMode::Fifo,
                "{backend:?}: requested {mode:?}, got {:?}",
                info.present_mode
            );
            assert!(
                matches!(
                    info.format,
                    TextureFormat::Bgra8Unorm | TextureFormat::Rgba8Unorm
                ),
                "{backend:?}: srgb = false must select a UNORM format, got {:?}",
                info.format
            );
            let pipeline = presentation.pipeline();
            presentation.frame(pipeline);
        }
    }
}

/// Runs every other test of this file again in a child process with API validation off, the
/// default of release builds: rejecting a destroyed swapchain's texture must not depend on GL
/// debug output (ISSUE-019, RHI-16).
#[test]
fn checks_hold_without_api_validation() {
    const NAME: &str = "checks_hold_without_api_validation";
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args(["--skip", NAME])
        .env(korppi_rhi::VALIDATION_ENV_VAR, "0")
        .output()
        .expect("run the tests again");
    assert!(
        output.status.success(),
        "with {}=0: {}\nstdout:\n{}\nstderr:\n{}",
        korppi_rhi::VALIDATION_ENV_VAR,
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
