//! Integration tests for `FIX-10.4` (`ENG-04`): UI canvas extraction, quad batch
//! compilation, `RenderGraph` execution, and windowed presentation during `EngineApp::update`.

use korppi_engine::{EngineApp, EngineConfig, GraphicsBackend};
use korppi_math::Vec2;
use korppi_rhi::{
    BufferDescriptor, BufferTextureCopy, BufferUsage, CommandBufferRecorder, ImageLayout,
    ImageMemoryBarrier, MemoryUsage, PipelineBarrier, PipelineStageFlags, PresentMode, SampleCount,
    SwapchainDescriptor, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle,
    TextureSubresourceRange, TextureUsage,
};
use korppi_rhi_conformance::display::{display_or_skip, TestWindow};
use korppi_ui::{Control, LayoutPreset, StyleBox, StyleBoxFlat, ThemeOverride};
use std::sync::Arc;
use std::time::{Duration, Instant};

const W: u32 = 64;
const H: u32 = 48;

fn spawn_colored_panel(
    app: &mut EngineApp,
    color: [f32; 4],
    preset: LayoutPreset,
    size: Vec2,
) -> korppi_ecs::Entity {
    let mut control = Control::default();
    control.set_anchors_preset(preset, false);
    control.custom_minimum_size = size;

    let flat = StyleBoxFlat {
        bg_color: color,
        border_width: [0.0; 4],
        ..StyleBoxFlat::default()
    };
    let mut theme_override = ThemeOverride::default();
    theme_override.set_stylebox("panel", StyleBox::Flat(flat));

    app.world.spawn((control, theme_override))
}

#[test]
fn engine_app_update_renders_and_presents_ui_controls_to_attached_window() {
    let Some(display) = display_or_skip("engine_app_ui_window_present") else {
        return;
    };
    let window = Arc::new(TestWindow::create(&display, W, H));
    let desc = SwapchainDescriptor {
        present_mode: PresentMode::Immediate,
        ..SwapchainDescriptor::new(W, H)
    };

    let config = EngineConfig {
        backend: GraphicsBackend::Vulkan,
        window_size: Vec2::new(W as f32, H as f32),
        ..EngineConfig::default()
    };
    let mut app = EngineApp::with_config(config).expect("EngineApp creation should succeed");
    app.clear_color = [0.0, 0.0, 0.0, 1.0];
    app.attach_window(window.clone(), &desc)
        .expect("attach_window should succeed");

    let panel = spawn_colored_panel(
        &mut app,
        [0.0, 1.0, 0.0, 1.0],
        LayoutPreset::FullRect,
        Vec2::new(W as f32, H as f32),
    );

    // Stepping `app.update` resolves UI layout in PostUpdate and extracts/compiles/presents
    // the UI canvas in Stage::Render without manual `present_quads` calls.
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        app.update(Duration::from_secs_f32(1.0 / 60.0));
        assert_eq!(app.ui_draw_commands.len(), 1);
        assert_eq!(app.ui_quad_batch.quad_count(), 1);

        app.graphics_device
            .as_ref()
            .expect("device")
            .wait_idle()
            .expect("wait_idle");
        let img = window.capture(W, H);
        if img.pixel(W / 2, H / 2) == [0, 255, 0, 255] {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "UI panel did not present green [0, 255, 0, 255] to window (last seen: {:?})",
            img.pixel(W / 2, H / 2)
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    // Mutate the UI panel's theme color in ECS and verify the next `app.update` frames present red.
    {
        let theme = app
            .world
            .get_mut::<ThemeOverride>(panel)
            .expect("ThemeOverride on panel");
        let red_flat = StyleBoxFlat {
            bg_color: [1.0, 0.0, 0.0, 1.0],
            border_width: [0.0; 4],
            ..StyleBoxFlat::default()
        };
        theme.set_stylebox("panel", StyleBox::Flat(red_flat));
    }

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        app.update(Duration::from_secs_f32(1.0 / 60.0));
        app.graphics_device
            .as_ref()
            .expect("device")
            .wait_idle()
            .expect("wait_idle");
        let img = window.capture(W, H);
        if img.pixel(W / 2, H / 2) == [255, 0, 0, 255] {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "mutated UI panel did not present red [255, 0, 0, 255] to window (last seen: {:?})",
            img.pixel(W / 2, H / 2)
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn readback_rgba8(
    device: &dyn korppi_rhi::GraphicsDevice,
    texture: TextureHandle,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let byte_len = (width * height * 4) as u64;
    let staging = device
        .create_buffer(
            &BufferDescriptor {
                size: byte_len,
                usage: BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuToCpu,
                label: Some("ui_readback_staging"),
            },
            None,
        )
        .expect("create staging buffer");
    let mut rec: Box<dyn CommandBufferRecorder> =
        device.create_command_recorder().expect("recorder");
    rec.begin().expect("begin");
    rec.pipeline_barrier(
        &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture,
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
        texture,
        staging,
        &BufferTextureCopy {
            buffer_offset: 0,
            buffer_row_pitch: 0,
            texture_mip_level: 0,
            texture_layer: 0,
            origin: [0, 0, 0],
            extent: [width, height, 1],
        },
    );
    let sub = device
        .submit(rec.finish().expect("finish"))
        .expect("submit");
    device.wait_for_submission(sub).expect("wait");
    let bytes = device
        .read_buffer(staging, 0, byte_len)
        .expect("read_buffer");
    device
        .destroy_buffer(staging)
        .expect("destroy staging buffer");
    bytes
}

#[test]
fn engine_app_renders_ui_through_render_graph_on_both_backends() {
    for backend in [GraphicsBackend::Vulkan, GraphicsBackend::OpenGl] {
        let config = EngineConfig {
            backend,
            window_size: Vec2::new(W as f32, H as f32),
            ..EngineConfig::default()
        };
        let mut app = EngineApp::with_config(config).expect("EngineApp creation should succeed");
        app.clear_color = [0.0, 0.0, 1.0, 1.0];

        // Spawn a top-left 32x24 red UI panel.
        spawn_colored_panel(
            &mut app,
            [1.0, 0.0, 0.0, 1.0],
            LayoutPreset::TopLeft,
            Vec2::new(32.0, 24.0),
        );

        app.update(Duration::from_secs_f32(1.0 / 60.0));
        assert_eq!(app.ui_draw_commands.len(), 1);
        assert_eq!(app.ui_quad_batch.quad_count(), 1);

        let device = app.graphics_device.as_ref().expect("device").clone();
        let target = device
            .create_texture(
                &TextureDescriptor {
                    width: W,
                    height: H,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
                    dimension: TextureDimension::D2,
                    label: Some("ui_graph_target"),
                },
                None,
            )
            .expect("create target texture");

        app.render_ui_graph(target, TextureFormat::Rgba8Unorm, W, H)
            .expect("render_ui_graph should succeed");

        let pixels = readback_rgba8(device.as_ref(), target, W, H);
        let pixel_at = |x: u32, y: u32| -> [u8; 4] {
            let idx = ((y * W + x) * 4) as usize;
            [
                pixels[idx],
                pixels[idx + 1],
                pixels[idx + 2],
                pixels[idx + 3],
            ]
        };

        // Inside the top-left 32x24 UI panel -> red.
        assert_eq!(
            pixel_at(16, 12),
            [255, 0, 0, 255],
            "inside UI panel should be red on {backend:?}"
        );
        // Outside the UI panel -> blue clear color.
        assert_eq!(
            pixel_at(48, 36),
            [0, 0, 255, 255],
            "outside UI panel should be blue clear color on {backend:?}"
        );

        device.destroy_texture(target).expect("destroy target");
    }
}
