//! GPU rendering and presentation of `QuadBatch2d` into real windows on both backends (FIX-1.3, FIX-6.3).

use korppi_math::Vec2;
use korppi_render::render2d::{QuadBatch2d, QuadBatchRenderer};
use korppi_rhi::{
    AccessFlags, FrameIndex, ImageLayout, ImageMemoryBarrier, PipelineBarrier, PipelineStageFlags,
    PresentMode, SampleCount, SwapchainDescriptor, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsage,
};
use korppi_rhi_conformance::display::{display_or_skip, TestWindow, VirtualDisplay};
use korppi_rhi_conformance::{Backend, Image, TestDevice};
use std::sync::Arc;
use std::time::{Duration, Instant};

const W: u32 = 64;
const H: u32 = 48;
const CLEAR: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

struct WindowSession {
    window: Arc<TestWindow>,
    device: TestDevice,
}

fn open_session(display: &VirtualDisplay, backend: Backend) -> WindowSession {
    let window = Arc::new(TestWindow::create(display, W, H));
    let device = TestDevice::open_for_display(backend, window.clone());
    WindowSession { window, device }
}

/// Renders `batch` (canvas = `canvas` units) into a real window swapchain and captures the window.
fn render(session: &WindowSession, srgb: bool, canvas: (f32, f32), batch: &QuadBatch2d) -> Image {
    render_with_clear(session, srgb, canvas, batch, CLEAR)
}

fn render_with_clear(
    session: &WindowSession,
    srgb: bool,
    canvas: (f32, f32),
    batch: &QuadBatch2d,
    clear: [f32; 4],
) -> Image {
    let swapchain = session
        .device
        .create_swapchain(
            session.window.clone(),
            &SwapchainDescriptor {
                present_mode: PresentMode::Immediate,
                srgb,
                ..SwapchainDescriptor::new(W, H)
            },
        )
        .expect("create_swapchain");
    let format = session
        .device
        .swapchain_info(swapchain)
        .expect("swapchain_info")
        .format;
    assert_eq!(
        format.is_srgb(),
        srgb,
        "swapchain sRGB mismatch: {format:?}"
    );

    let mut renderer = QuadBatchRenderer::new(&*session.device, format).expect("renderer");

    for frame in 0..3 {
        session.device.begin_frame().expect("begin_frame");
        let image = session
            .device
            .acquire_next_image(swapchain)
            .expect("acquire_next_image")
            .expect("swapchain image");
        let mut recorder = session.device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.pipeline_barrier(
            &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
                texture: image.texture,
                range: korppi_rhi::TextureSubresourceRange::ALL,
                src_stages: PipelineStageFlags::TOP_OF_PIPE,
                src_access: AccessFlags::empty(),
                dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                old_layout: ImageLayout::Undefined,
                new_layout: ImageLayout::ColorAttachment,
            }),
        );
        renderer
            .record(
                &*session.device,
                recorder.as_mut(),
                FrameIndex::new(frame % 2),
                image.texture,
                (W, H),
                canvas,
                clear,
                batch,
            )
            .expect("record");
        let sub = session
            .device
            .submit(recorder.finish().expect("finish"))
            .expect("submit");
        session.device.present(&image).expect("present");
        session
            .device
            .wait_for_submission(sub)
            .expect("wait_for_submission");
    }

    session.device.wait_idle().expect("wait_idle");
    let deadline = Instant::now() + Duration::from_secs(2);
    let captured = loop {
        let img = session.window.capture(W, H);
        let non_black = (0..H).any(|y| (0..W).any(|x| img.pixel(x, y) != [0, 0, 0, 255]));
        if non_black || Instant::now() > deadline {
            break img;
        }
        std::thread::sleep(Duration::from_millis(10));
    };

    renderer
        .destroy(&*session.device)
        .expect("destroy renderer");
    session
        .device
        .destroy_swapchain(swapchain)
        .expect("destroy swapchain");
    session.device.assert_no_api_errors();
    captured
}

fn close(actual: [u8; 4], expected: [u8; 4]) -> bool {
    actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 2)
}

#[test]
fn quads_land_at_their_canvas_position_with_a_top_left_origin() {
    let Some(display) = display_or_skip("quad_batch_top_left") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let mut batch = QuadBatch2d::new(16);
        // Red quad in the top-left corner, green half-transparent quad over the bottom right.
        batch.push_quad(
            Vec2::new(0.0, 0.0),
            Vec2::new(16.0, 8.0),
            Vec2::ZERO,
            Vec2::ONE,
            [1.0, 0.0, 0.0, 1.0],
        );
        batch.push_quad(
            Vec2::new(32.0, 24.0),
            Vec2::new(64.0, 48.0),
            Vec2::ZERO,
            Vec2::ONE,
            [0.0, 1.0, 0.0, 0.5],
        );
        let img = render(&session, false, (W as f32, H as f32), &batch);
        assert!(
            close(img.pixel(4, 2), [255, 0, 0, 255]),
            "{backend:?}: top-left quad, got {:?}",
            img.pixel(4, 2)
        );
        assert!(
            close(img.pixel(4, 40), [0, 0, 255, 255]),
            "{backend:?}: bottom-left is clear colour, got {:?}",
            img.pixel(4, 40)
        );
        // Blended 50 % green over blue.
        assert!(
            close(img.pixel(48, 40), [0, 128, 128, 255]),
            "{backend:?}: blended quad, got {:?}",
            img.pixel(48, 40)
        );
    }
}

#[test]
fn canvas_units_scale_to_the_target_size() {
    let Some(display) = display_or_skip("quad_batch_scale") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let mut batch = QuadBatch2d::new(4);
        // On a 32 x 24 canvas (a 2x HiDPI window), the left half of the canvas covers the left
        // half of the 64 x 48 target.
        batch.push_quad(
            Vec2::new(0.0, 0.0),
            Vec2::new(16.0, 24.0),
            Vec2::ZERO,
            Vec2::ONE,
            [1.0, 1.0, 1.0, 1.0],
        );
        let img = render(&session, false, (32.0, 24.0), &batch);
        assert!(
            close(img.pixel(30, 40), [255, 255, 255, 255]),
            "{backend:?}"
        );
        assert!(close(img.pixel(34, 40), [0, 0, 255, 255]), "{backend:?}");
    }
}

#[test]
fn srgb_targets_show_the_authored_colours() {
    let Some(display) = display_or_skip("quad_batch_srgb") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let mut batch = QuadBatch2d::new(4);
        batch.push_quad(
            Vec2::new(0.0, 0.0),
            Vec2::new(64.0, 48.0),
            Vec2::ZERO,
            Vec2::ONE,
            [0.5, 0.25, 0.75, 1.0],
        );
        for srgb in [false, true] {
            let img = render(&session, srgb, (W as f32, H as f32), &batch);
            assert!(
                close(img.pixel(10, 10), [128, 64, 191, 255]),
                "{backend:?} srgb={srgb}: got {:?}",
                img.pixel(10, 10)
            );
        }
    }
}

#[test]
fn the_clear_colour_is_an_authored_srgb_colour_too() {
    let Some(display) = display_or_skip("quad_batch_clear_srgb") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let batch = QuadBatch2d::new(1);
        for srgb in [false, true] {
            let img = render_with_clear(
                &session,
                srgb,
                (W as f32, H as f32),
                &batch,
                [0.5, 0.25, 0.75, 1.0],
            );
            assert!(
                close(img.pixel(3, 3), [128, 64, 191, 255]),
                "{backend:?} srgb={srgb}: got {:?}",
                img.pixel(3, 3)
            );
        }
    }
}

#[test]
fn empty_batches_only_clear() {
    let Some(display) = display_or_skip("quad_batch_empty") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let batch = QuadBatch2d::new(1);
        let img = render(&session, false, (W as f32, H as f32), &batch);
        assert!(close(img.pixel(0, 0), [0, 0, 255, 255]), "{backend:?}");
    }
}

#[test]
fn textured_quads_sample_bound_textures_and_switch_per_batch() {
    let Some(display) = display_or_skip("quad_batch_textured") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let device = &*session.device;
        // tex_a: 2x2 RGBA8 (Top-Left Red, Top-Right Green, Bottom-Left Blue, Bottom-Right Yellow)
        let tex_a_data: [u8; 16] = [
            255, 0, 0, 255, 0, 255, 0, 255, // row 0
            0, 0, 255, 255, 255, 255, 0, 255, // row 1
        ];
        let tex_a = device
            .create_texture(
                &TextureDescriptor {
                    width: 2,
                    height: 2,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::SAMPLED,
                    dimension: TextureDimension::D2,
                    label: Some("tex_a"),
                },
                Some(&tex_a_data),
            )
            .expect("create tex_a");

        // tex_b: 2x1 RGBA8 (Left Cyan, Right Magenta)
        let tex_b_data: [u8; 8] = [0, 255, 255, 255, 255, 0, 255, 255];
        let tex_b = device
            .create_texture(
                &TextureDescriptor {
                    width: 2,
                    height: 1,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::SAMPLED,
                    dimension: TextureDimension::D2,
                    label: Some("tex_b"),
                },
                Some(&tex_b_data),
            )
            .expect("create tex_b");

        let mut batch = QuadBatch2d::new(16);
        // 1. Quad [0..20, 0..20] with tex_a
        batch.push_textured_quad(
            tex_a,
            Vec2::new(0.0, 0.0),
            Vec2::new(20.0, 20.0),
            Vec2::ZERO,
            Vec2::ONE,
            [1.0, 1.0, 1.0, 1.0],
        );
        // 2. Untextured quad [24..36, 0..20] solid white
        batch.push_quad(
            Vec2::new(24.0, 0.0),
            Vec2::new(36.0, 20.0),
            Vec2::ZERO,
            Vec2::ONE,
            [1.0, 1.0, 1.0, 1.0],
        );
        // 3. Quad [40..60, 0..20] with tex_b tinted by [0.5, 0.5, 1.0, 1.0]
        batch.push_textured_quad(
            tex_b,
            Vec2::new(40.0, 0.0),
            Vec2::new(60.0, 20.0),
            Vec2::ZERO,
            Vec2::ONE,
            [0.5, 0.5, 1.0, 1.0],
        );

        let img = render(&session, false, (W as f32, H as f32), &batch);

        // Verify tex_a 4 quadrants
        assert!(
            close(img.pixel(5, 5), [255, 0, 0, 255]),
            "{backend:?}: tex_a TL got {:?}",
            img.pixel(5, 5)
        );
        assert!(
            close(img.pixel(15, 5), [0, 255, 0, 255]),
            "{backend:?}: tex_a TR got {:?}",
            img.pixel(15, 5)
        );
        assert!(
            close(img.pixel(5, 15), [0, 0, 255, 255]),
            "{backend:?}: tex_a BL got {:?}",
            img.pixel(5, 15)
        );
        assert!(
            close(img.pixel(15, 15), [255, 255, 0, 255]),
            "{backend:?}: tex_a BR got {:?}",
            img.pixel(15, 15)
        );

        // Verify untextured quad between them uses default 1x1 white texture
        assert!(
            close(img.pixel(30, 10), [255, 255, 255, 255]),
            "{backend:?}: untextured quad got {:?}",
            img.pixel(30, 10)
        );

        // Verify tex_b tinted halves
        assert!(
            close(img.pixel(45, 10), [0, 128, 255, 255]),
            "{backend:?}: tex_b left half got {:?}",
            img.pixel(45, 10)
        );
        assert!(
            close(img.pixel(55, 10), [128, 0, 255, 255]),
            "{backend:?}: tex_b right half got {:?}",
            img.pixel(55, 10)
        );

        device.destroy_texture(tex_a).expect("destroy tex_a");
        device.destroy_texture(tex_b).expect("destroy tex_b");
    }
}

#[test]
fn atlas_uvs_and_9slice_render_correctly_on_the_gpu() {
    use korppi_render::render2d::{NinePatchMargins, Rect2d, TextureAtlas2d};

    let Some(display) = display_or_skip("quad_batch_9slice") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let device = &*session.device;
        // 8x4 RGBA8 atlas:
        // Left 4x4 = 9-slice panel:
        //   corners = Red [255, 0, 0, 255]
        //   top/bottom edges = Green [0, 255, 0, 255]
        //   left/right edges = Yellow [255, 255, 0, 255]
        //   center 2x2 = White [255, 255, 255, 255]
        // Right 4x4 = Sprite region: Magenta [255, 0, 255, 255]
        let mut atlas_bytes = vec![0u8; 8 * 4 * 4];
        for y in 0..4u32 {
            for x in 0..8u32 {
                let rgba = if x >= 4 {
                    [255, 0, 255, 255]
                } else {
                    let is_x_edge = x == 0 || x == 3;
                    let is_y_edge = y == 0 || y == 3;
                    match (is_x_edge, is_y_edge) {
                        (true, true) => [255, 0, 0, 255],
                        (false, true) => [0, 255, 0, 255],
                        (true, false) => [255, 255, 0, 255],
                        (false, false) => [255, 255, 255, 255],
                    }
                };
                let idx = ((y * 8 + x) * 4) as usize;
                atlas_bytes[idx..idx + 4].copy_from_slice(&rgba);
            }
        }

        let atlas_tex = device
            .create_texture(
                &TextureDescriptor {
                    width: 8,
                    height: 4,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::SAMPLED,
                    dimension: TextureDimension::D2,
                    label: Some("atlas_8x4"),
                },
                Some(&atlas_bytes),
            )
            .expect("create atlas_tex");

        let mut atlas = TextureAtlas2d::new(8, 4);
        let panel_region = atlas.add_region(
            "panel",
            Rect2d {
                x: 0.0,
                y: 0.0,
                width: 4.0,
                height: 4.0,
            },
        );
        let sprite_region = atlas.add_region(
            "gem",
            Rect2d {
                x: 4.0,
                y: 0.0,
                width: 4.0,
                height: 4.0,
            },
        );

        let mut batch = QuadBatch2d::new(32);
        // Stretch 4x4 panel (with 4px borders on canvas) into [0..32, 0..24]
        let panel_uv = atlas.get_uv(panel_region).expect("panel uv");
        batch.push_9patch(
            Some(atlas_tex),
            Rect2d {
                x: 0.0,
                y: 0.0,
                width: 16.0,
                height: 16.0,
            },
            panel_uv,
            NinePatchMargins {
                left: 4.0,
                right: 4.0,
                top: 4.0,
                bottom: 4.0,
            },
            Rect2d {
                x: 0.0,
                y: 0.0,
                width: 32.0,
                height: 24.0,
            },
            [1.0, 1.0, 1.0, 1.0],
        );

        // Draw the atlas sprite at [40..56, 4..20]
        assert!(batch.push_atlas_region(
            atlas_tex,
            &atlas,
            sprite_region,
            Vec2::new(40.0, 4.0),
            Vec2::new(56.0, 20.0),
            [1.0, 1.0, 1.0, 1.0],
        ));

        let img = render(&session, false, (W as f32, H as f32), &batch);

        // 9-patch corners: Red
        assert!(
            close(img.pixel(2, 2), [255, 0, 0, 255]),
            "{backend:?}: 9patch TL corner got {:?}",
            img.pixel(2, 2)
        );
        assert!(
            close(img.pixel(30, 22), [255, 0, 0, 255]),
            "{backend:?}: 9patch BR corner got {:?}",
            img.pixel(30, 22)
        );
        // 9-patch top border: Green
        assert!(
            close(img.pixel(16, 2), [0, 255, 0, 255]),
            "{backend:?}: 9patch top edge got {:?}",
            img.pixel(16, 2)
        );
        // 9-patch left border: Yellow
        assert!(
            close(img.pixel(2, 12), [255, 255, 0, 255]),
            "{backend:?}: 9patch left edge got {:?}",
            img.pixel(2, 12)
        );
        // 9-patch center stretched area: White
        assert!(
            close(img.pixel(16, 12), [255, 255, 255, 255]),
            "{backend:?}: 9patch center got {:?}",
            img.pixel(16, 12)
        );
        // Atlas sprite region: Magenta
        assert!(
            close(img.pixel(48, 12), [255, 0, 255, 255]),
            "{backend:?}: atlas sprite got {:?}",
            img.pixel(48, 12)
        );

        device.destroy_texture(atlas_tex).expect("destroy atlas");
    }
}

#[test]
fn bitmap_font_glyph_atlas_renders_tinted_text_on_the_gpu() {
    use korppi_render::render2d::BitmapFont2d;

    let Some(display) = display_or_skip("quad_batch_font") else {
        return;
    };
    for backend in Backend::ALL {
        let session = open_session(&display, backend);
        let device = &*session.device;
        let font = BitmapFont2d::create_default(device).expect("create font");
        let mut batch = QuadBatch2d::new(32);
        // Draw "HI" at (4, 4) with scale 2.0 (each 8x8 glyph becomes 16x16 pixels on canvas)
        batch.push_text(&font, "HI", Vec2::new(4.0, 4.0), 2.0, [0.0, 1.0, 0.5, 1.0]);

        let img = render(&session, false, (W as f32, H as f32), &batch);

        // Count rendered ink pixels of the tinted green-cyan color [0, 255, 128, 255] inside the text region
        let mut ink_pixels = 0usize;
        let mut bg_pixels = 0usize;
        for y in 4..20 {
            for x in 4..36 {
                let p = img.pixel(x, y);
                if close(p, [0, 255, 128, 255]) {
                    ink_pixels += 1;
                } else if close(p, [0, 0, 255, 255]) {
                    bg_pixels += 1;
                }
            }
        }
        assert!(
            ink_pixels >= 80 && bg_pixels >= 80,
            "{backend:?}: expected both glyph ink pixels ({ink_pixels}) and transparent background pixels ({bg_pixels}) in rendered 'HI'"
        );

        font.destroy(device).expect("destroy font");
    }
}
