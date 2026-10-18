//! Headless offscreen hardware integration tests for 2D Batched Rendering, VBO/IBO streaming, and Canvas passes.

use korppi_math::Vec2;
use korppi_render::render2d::{QuadBatch2d, Render2dPipeline};
use korppi_render::{RenderGraph, TextureAccess, TransientResourcePool};
use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferUsage, GraphicsDevice, ImageLayout, MemoryUsage,
    PipelineStageFlags, SampleCount, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_2d_batch_vbo_ibo_upload_and_roundtrip_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut batch = QuadBatch2d::new(64);
    for i in 0..16 {
        batch.push_quad(
            Vec2::new(i as f32 * 10.0, 0.0),
            Vec2::new((i + 1) as f32 * 10.0, 10.0),
            Vec2::ZERO,
            Vec2::ONE,
            [0.2, 0.4, 0.8, 1.0],
        );
    }
    assert_eq!(batch.quad_count(), 16);

    // 1. Upload Vertex Buffer
    let vtx_slice = batch.vertices();
    let vtx_bytes: &[u8] = korppi_core::pod::cast_slice(vtx_slice);

    let vbo_desc = BufferDescriptor {
        size: vtx_bytes.len() as u64,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("gl_quad_vbo"),
    };
    let vbo = device
        .create_buffer(&vbo_desc, None)
        .expect("Create VBO failed");

    device
        .write_buffer(vbo, 0, vtx_bytes)
        .expect("Write VBO failed");

    let vbo_readback = device
        .read_buffer(vbo, 0, vtx_bytes.len() as u64)
        .expect("Read VBO failed");
    assert_eq!(vbo_readback, vtx_bytes);

    // 2. Upload Index Buffer
    let idx_slice = batch.indices();
    let idx_bytes: &[u8] = korppi_core::pod::cast_slice(idx_slice);

    let ibo_desc = BufferDescriptor {
        size: idx_bytes.len() as u64,
        usage: BufferUsage::INDEX | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("gl_quad_ibo"),
    };
    let ibo = device
        .create_buffer(&ibo_desc, None)
        .expect("Create IBO failed");

    device
        .write_buffer(ibo, 0, idx_bytes)
        .expect("Write IBO failed");

    let ibo_readback = device
        .read_buffer(ibo, 0, idx_bytes.len() as u64)
        .expect("Read IBO failed");
    assert_eq!(ibo_readback, idx_bytes);

    device.destroy_buffer(vbo).expect("Destroy VBO failed");
    device.destroy_buffer(ibo).expect("Destroy IBO failed");
}

#[test]
fn test_2d_canvas_rendering_headless_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let canvas_desc = TextureDescriptor {
        width: 32,
        height: 32,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("gl_canvas2d_target"),
    };
    let canvas_rhi = device
        .create_texture(&canvas_desc, None)
        .expect("Failed to create canvas texture");

    let mut pool = TransientResourcePool::new();
    let mut graph = RenderGraph::new();

    let canvas_v = graph.import_texture(
        canvas_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        32,
        32,
    );

    Render2dPipeline::build(&mut graph, canvas_v, 32, 32, Some([0.15, 0.45, 0.85, 1.0]));

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile 2D graph failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create recorder failed");
    recorder.begin().expect("Begin recorder failed");
    compiled.execute(&mut *recorder).expect("Execute failed");
    let cmd_buf = recorder.finish().expect("Finish recorder failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(canvas_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 32 * 32 * 4);
    // [0.15*255=38, 0.45*255=115, 0.85*255=216, 255]
    assert!((pixels[0] as i32 - 38).abs() <= 2);
    assert!((pixels[1] as i32 - 115).abs() <= 2);
    assert!((pixels[2] as i32 - 216).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(canvas_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}

#[test]
fn test_2d_canvas_rendering_headless_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        println!("Headless Vulkan 1.3 device unavailable, skipping Vulkan 2D test");
        return;
    };

    let canvas_desc = TextureDescriptor {
        width: 32,
        height: 32,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("vk_canvas2d_target"),
    };
    let canvas_rhi = device
        .create_texture(&canvas_desc, None)
        .expect("Failed to create canvas texture");

    let mut pool = TransientResourcePool::new();
    let mut graph = RenderGraph::new();

    let canvas_v = graph.import_texture(
        canvas_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        32,
        32,
    );

    Render2dPipeline::build(&mut graph, canvas_v, 32, 32, Some([0.75, 0.25, 0.55, 1.0]));

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile 2D graph failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create recorder failed");
    recorder.begin().expect("Begin recorder failed");
    compiled.execute(&mut *recorder).expect("Execute failed");
    let cmd_buf = recorder.finish().expect("Finish recorder failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(canvas_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 32 * 32 * 4);
    // [0.75*255=191, 0.25*255=64, 0.55*255=140, 255]
    assert!((pixels[0] as i32 - 191).abs() <= 2);
    assert!((pixels[1] as i32 - 64).abs() <= 2);
    assert!((pixels[2] as i32 - 140).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(canvas_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}
