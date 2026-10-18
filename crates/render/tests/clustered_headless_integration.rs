//! Headless offscreen hardware integration tests for Depth Prepass, Reverse-Z, and Clustered Light buffers.

use korppi_math::Vec3;
use korppi_render::clustered::{
    add_depth_prepass, ClusterGridConfig, ClusteredLightCuller, DepthPrepassConfig, PointLight,
    ReverseZ, SpotLight,
};
use korppi_render::{PassType, RenderGraph, TextureAccess, TransientResourcePool};
use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferUsage, GraphicsDevice, ImageLayout, LoadOp, MemoryUsage,
    PipelineStageFlags, SampleCount, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_depth_prepass_headless_execution_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let color_desc = TextureDescriptor {
        width: 64,
        height: 64,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("gl_forward_color_target"),
    };
    let color_rhi = device
        .create_texture(&color_desc, None)
        .expect("Failed to create color texture");

    let mut pool = TransientResourcePool::new();
    let mut graph = RenderGraph::new();

    let color_v = graph.import_texture(
        color_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );

    // 1. Depth Prepass using helper
    let prepass_config = DepthPrepassConfig {
        width: 64,
        height: 64,
        format: TextureFormat::Depth32Float,
        sample_count: SampleCount::Count1,
    };
    let depth_v = add_depth_prepass(&mut graph, &prepass_config);

    // 2. Forward Shading Pass (loads prepass depth, writes color)
    graph
        .add_pass("ForwardShadingPass", PassType::Graphics)
        .write_color(
            color_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.2, 0.7, 0.4, 1.0],
        )
        .write_depth(depth_v, LoadOp::Load, StoreOp::Store, 0.0)
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Graph compilation failed");

    assert_eq!(compiled.passes().len(), 2);

    let mut recorder = device
        .create_command_recorder()
        .expect("Create command recorder failed");
    recorder.begin().expect("Begin recording failed");
    compiled.execute(&mut *recorder).expect("Execution failed");
    let cmd_buf = recorder.finish().expect("Finish recording failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    // Read back color pixels (expect clear color [0.2*255=51, 0.7*255=178, 0.4*255=102, 255])
    let pixels = device
        .read_texture(color_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 64 * 64 * 4);
    assert!((pixels[0] as i32 - 51).abs() <= 2);
    assert!((pixels[1] as i32 - 178).abs() <= 2);
    assert!((pixels[2] as i32 - 102).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(color_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}

#[test]
fn test_clustered_culling_buffer_upload_and_roundtrip() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let config = ClusterGridConfig::new(128, 128, 64, 4, 1.0, 20.0);
    let proj = ReverseZ::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 1.0, 20.0);
    let froxels = config.build_froxel_aabbs(proj.inverse());

    let point_lights = vec![PointLight {
        position_vs: Vec3::new(0.0, 0.0, -5.0),
        radius: 4.0,
        color: Vec3::new(1.0, 0.5, 0.2),
        intensity: 50.0,
    }];
    let spot_lights = vec![SpotLight {
        position_vs: Vec3::new(0.0, 0.0, -2.0),
        direction_vs: Vec3::new(0.0, 0.0, -1.0),
        range: 8.0,
        half_angle: std::f32::consts::FRAC_PI_4,
        color: Vec3::new(0.2, 0.8, 1.0),
        intensity: 80.0,
    }];

    let mut culler = ClusteredLightCuller::new(&config);
    let output = culler.cull_lights(&froxels, &point_lights, &spot_lights);

    assert_eq!(
        output.cluster_records.len(),
        config.total_clusters() as usize
    );
    assert!(!output.global_light_indices.is_empty());

    // Pack records into byte slice
    let records_size =
        std::mem::size_of_val(&output.cluster_records[0]) * output.cluster_records.len();
    let records_raw: &[u8] = korppi_core::pod::cast_slice(&output.cluster_records);

    // Create GPU buffer and roundtrip test
    let buf_desc = BufferDescriptor {
        size: records_size as u64,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("cluster_records_ssbo"),
    };
    let gpu_buf = device
        .create_buffer(&buf_desc, None)
        .expect("Create buffer failed");

    device
        .write_buffer(gpu_buf, 0, records_raw)
        .expect("Write buffer failed");

    let readback_bytes = device
        .read_buffer(gpu_buf, 0, records_size as u64)
        .expect("Read buffer failed");
    assert_eq!(readback_bytes, records_raw);

    device
        .destroy_buffer(gpu_buf)
        .expect("Destroy buffer failed");
}

#[test]
fn test_depth_prepass_headless_execution_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        println!("Headless Vulkan 1.3 device unavailable, skipping Vulkan depth prepass test");
        return;
    };

    let color_desc = TextureDescriptor {
        width: 32,
        height: 32,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("vk_forward_color_target"),
    };
    let color_rhi = device
        .create_texture(&color_desc, None)
        .expect("Failed to create color texture");

    let mut pool = TransientResourcePool::new();
    let mut graph = RenderGraph::new();

    let color_v = graph.import_texture(
        color_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        32,
        32,
    );

    let prepass_config = DepthPrepassConfig {
        width: 32,
        height: 32,
        format: TextureFormat::Depth32Float,
        sample_count: SampleCount::Count1,
    };
    let depth_v = add_depth_prepass(&mut graph, &prepass_config);

    graph
        .add_pass("VkForwardShadingPass", PassType::Graphics)
        .write_color(
            color_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.8, 0.4, 0.1, 1.0],
        )
        .write_depth(depth_v, LoadOp::Load, StoreOp::Store, 0.0)
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 32, 32);
        });

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Graph compilation failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create command recorder failed");
    recorder.begin().expect("Begin recording failed");
    compiled.execute(&mut *recorder).expect("Execution failed");
    let cmd_buf = recorder.finish().expect("Finish recording failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(color_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 32 * 32 * 4);
    // [0.8*255=204, 0.4*255=102, 0.1*255=25, 255]
    assert!((pixels[0] as i32 - 204).abs() <= 2);
    assert!((pixels[1] as i32 - 102).abs() <= 2);
    assert!((pixels[2] as i32 - 25).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(color_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}
