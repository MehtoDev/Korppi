//! Headless offscreen hardware integration tests for HDR Post-Processing, Bloom, and Tonemapping.

use korppi_render::postprocess::{
    generate_ssao_kernel, BloomConfig, PostProcessPipeline, TonemapConfig, TonemapOperator,
};
use korppi_render::{
    PassType, RenderGraph, TextureAccess, TransientResourcePool, TransientTextureDesc,
};
use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferUsage, GraphicsDevice, ImageLayout, LoadOp, MemoryUsage,
    PipelineStageFlags, SampleCount, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_ssao_and_tonemap_ubo_roundtrip_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let tonemap_cfg = TonemapConfig {
        operator: TonemapOperator::Aces,
        exposure: 1.25,
        gamma: 2.2,
        white_point: 12.0,
    };
    let bloom_cfg = BloomConfig {
        threshold: 1.5,
        soft_knee: 0.5,
        intensity: 0.08,
        filter_radius: 1.2,
        mip_levels: 5,
    };
    let kernel = generate_ssao_kernel(16);

    let mut ubo_bytes = Vec::new();
    // vec4 tonemap_params: [exposure, gamma, white_point, operator_id]
    ubo_bytes.extend_from_slice(&tonemap_cfg.exposure.to_ne_bytes());
    ubo_bytes.extend_from_slice(&tonemap_cfg.gamma.to_ne_bytes());
    ubo_bytes.extend_from_slice(&tonemap_cfg.white_point.to_ne_bytes());
    ubo_bytes.extend_from_slice(&(1.0f32).to_ne_bytes()); // ACES = 1

    // vec4 bloom_params: [threshold, soft_knee, intensity, filter_radius]
    ubo_bytes.extend_from_slice(&bloom_cfg.threshold.to_ne_bytes());
    ubo_bytes.extend_from_slice(&bloom_cfg.soft_knee.to_ne_bytes());
    ubo_bytes.extend_from_slice(&bloom_cfg.intensity.to_ne_bytes());
    ubo_bytes.extend_from_slice(&bloom_cfg.filter_radius.to_ne_bytes());

    // 16 vec4 ssao_samples
    for sample in &kernel {
        ubo_bytes.extend_from_slice(&sample.x.to_ne_bytes());
        ubo_bytes.extend_from_slice(&sample.y.to_ne_bytes());
        ubo_bytes.extend_from_slice(&sample.z.to_ne_bytes());
        ubo_bytes.extend_from_slice(&0.0f32.to_ne_bytes()); // 16-byte alignment padding
    }

    let buf_desc = BufferDescriptor {
        size: ubo_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("postprocess_params_ubo"),
    };
    let gpu_buf = device
        .create_buffer(&buf_desc, None)
        .expect("Create buffer failed");

    device
        .write_buffer(gpu_buf, 0, &ubo_bytes)
        .expect("Write buffer failed");

    let readback = device
        .read_buffer(gpu_buf, 0, ubo_bytes.len() as u64)
        .expect("Read buffer failed");
    assert_eq!(readback, ubo_bytes);

    device
        .destroy_buffer(gpu_buf)
        .expect("Destroy buffer failed");
}

#[test]
fn test_postprocess_tonemap_and_bloom_headless_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let ldr_desc = TextureDescriptor {
        width: 32,
        height: 32,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("gl_postprocess_ldr_target"),
    };
    let ldr_rhi = device
        .create_texture(&ldr_desc, None)
        .expect("Failed to create LDR texture");

    let mut pool = TransientResourcePool::new();
    let mut graph = RenderGraph::new();

    let ldr_v = graph.import_texture(
        ldr_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        32,
        32,
    );

    let hdr_scene = graph.create_transient_texture(TransientTextureDesc::new(
        "TransientHdrScene",
        32,
        32,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    let depth_v = graph.create_transient_texture(TransientTextureDesc::new(
        "TransientDepth",
        32,
        32,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    // Pass 1: Forward HDR Shading Pass
    graph
        .add_pass("HdrForwardPass", PassType::Graphics)
        .write_color(
            hdr_scene,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [2.5, 1.2, 0.4, 1.0],
        )
        .write_depth(depth_v, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 32, 32);
        });

    // Pass 2 & 3: PostProcessPipeline build
    let tonemap_cfg = TonemapConfig {
        operator: TonemapOperator::Aces,
        exposure: 1.0,
        gamma: 2.2,
        white_point: 10.0,
    };
    PostProcessPipeline::build(
        &mut graph,
        hdr_scene,
        Some(depth_v),
        ldr_v,
        tonemap_cfg,
        Some(BloomConfig::default()),
        None,
    );

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile post-processing graph failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create recorder failed");
    recorder.begin().expect("Begin recorder failed");
    compiled.execute(&mut *recorder).expect("Execute failed");
    let cmd_buf = recorder.finish().expect("Finish recorder failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(ldr_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 32 * 32 * 4);
    assert_eq!(pixels[3], 255); // Alpha is 255

    device
        .destroy_texture(ldr_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}

#[test]
fn test_postprocess_headless_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        println!("Headless Vulkan 1.3 device unavailable, skipping Vulkan post-process test");
        return;
    };

    let ldr_desc = TextureDescriptor {
        width: 32,
        height: 32,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("vk_postprocess_ldr_target"),
    };
    let ldr_rhi = device
        .create_texture(&ldr_desc, None)
        .expect("Failed to create LDR texture");

    let mut pool = TransientResourcePool::new();
    let mut graph = RenderGraph::new();

    let ldr_v = graph.import_texture(
        ldr_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        32,
        32,
    );

    let hdr_scene = graph.create_transient_texture(TransientTextureDesc::new(
        "VkTransientHdrScene",
        32,
        32,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    graph
        .add_pass("VkHdrPass", PassType::Graphics)
        .write_color(
            hdr_scene,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [1.5, 0.8, 0.2, 1.0],
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 32, 32);
        });

    PostProcessPipeline::build(
        &mut graph,
        hdr_scene,
        None,
        ldr_v,
        TonemapConfig::default(),
        None,
        None,
    );

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile post-processing graph failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create recorder failed");
    recorder.begin().expect("Begin recorder failed");
    compiled.execute(&mut *recorder).expect("Execute failed");
    let cmd_buf = recorder.finish().expect("Finish recorder failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(ldr_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 32 * 32 * 4);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(ldr_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}
