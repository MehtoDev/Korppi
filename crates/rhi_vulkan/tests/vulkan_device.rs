//! Integration test verifying Vulkan 1.3 device initialization and buffer/texture operations.

use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_vulkan_headless_device_creation_and_memory_roundtrip() {
    let device =
        VulkanDevice::create_headless().expect("Headless Vulkan 1.3 device creation failed");

    let desc = BufferDescriptor {
        size: 1024,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("test_buffer"),
    };
    let buffer = device
        .create_buffer(&desc, None)
        .expect("Create buffer failed");

    let write_bytes: Vec<u8> = (0..256).map(|i| i as u8).collect();
    device
        .write_buffer(buffer, 0, &write_bytes)
        .expect("Write buffer failed");

    let read_back = device
        .read_buffer(buffer, 0, 256)
        .expect("Read buffer failed");
    assert_eq!(read_back, write_bytes);

    device
        .destroy_buffer(buffer)
        .expect("Destroy buffer failed");

    let tex_desc = TextureDescriptor {
        width: 128,
        height: 128,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: korppi_rhi::SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        dimension: TextureDimension::D2,
        label: Some("test_texture"),
    };
    let texture = device
        .create_texture(&tex_desc, None)
        .expect("Create texture failed");
    device
        .destroy_texture(texture)
        .expect("Destroy texture failed");

    device.wait_idle().expect("Wait idle failed");
    assert_no_api_errors(&device);
}

#[test]
fn test_vulkan_dynamic_rendering_clear_pass() {
    use korppi_rhi::{
        AccessFlags, ImageLayout, ImageMemoryBarrier, LoadOp, PipelineBarrier, PipelineStageFlags,
        RenderingAttachment, RenderingDescriptor, StoreOp,
    };

    let device =
        VulkanDevice::create_headless().expect("Headless Vulkan 1.3 device creation failed");

    let tex_desc = TextureDescriptor {
        width: 64,
        height: 64,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: korppi_rhi::SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("offscreen_color_target"),
    };
    let texture = device
        .create_texture(&tex_desc, None)
        .expect("Create texture failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create command recorder failed");
    recorder.begin().expect("Begin recording failed");

    // Barrier: UNDEFINED -> COLOR_ATTACHMENT
    let barrier = PipelineBarrier {
        memory_barriers: korppi_rhi::smallvec![],
        buffer_barriers: korppi_rhi::smallvec![],
        image_barriers: korppi_rhi::smallvec![ImageMemoryBarrier {
            texture,
            range: korppi_rhi::TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::ColorAttachment,
        }],
    };
    recorder.pipeline_barrier(&barrier);

    let render_desc = RenderingDescriptor {
        width: 64,
        height: 64,
        offset_x: 0,
        offset_y: 0,
        color_attachments: korppi_rhi::smallvec![RenderingAttachment {
            texture,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color: Some([0.1, 0.2, 0.3, 1.0]),
            resolve_target: None,
        }],
        depth_attachment: None,
    };

    recorder
        .begin_rendering(&render_desc)
        .expect("Begin rendering failed");
    // Viewport with negative height (ADR-013)
    recorder.set_viewport(0.0, 64.0, 64.0, -64.0, 0.0, 1.0);
    recorder.set_scissor(0, 0, 64, 64);
    recorder.end_rendering().expect("End rendering failed");

    let cmd_buf = recorder.finish().expect("Finish command recording failed");
    device
        .submit(cmd_buf)
        .expect("Submit command buffer failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(texture, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 64 * 64 * 4);

    let r = pixels[0];
    let g = pixels[1];
    let b = pixels[2];
    let a = pixels[3];

    let expected_r = (0.1f32 * 255.0).round() as u8;
    let expected_g = (0.2f32 * 255.0).round() as u8;
    let expected_b = (0.3f32 * 255.0).round() as u8;
    let expected_a = 255u8;

    assert!((r as i32 - expected_r as i32).abs() <= 1);
    assert!((g as i32 - expected_g as i32).abs() <= 1);
    assert!((b as i32 - expected_b as i32).abs() <= 1);
    assert_eq!(a, expected_a);

    device
        .destroy_texture(texture)
        .expect("Destroy texture failed");
    assert_no_api_errors(&device);
}

/// Fails the test if the backend's API validation (Vulkan validation layers, OpenGL debug
/// output) reported errors (FIX-1.0).
fn assert_no_api_errors(device: &dyn korppi_rhi::GraphicsDevice) {
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == korppi_rhi::DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "API errors reported: {errors:#?}");
}
