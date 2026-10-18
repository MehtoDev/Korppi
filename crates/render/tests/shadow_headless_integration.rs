//! Headless offscreen hardware integration tests for Shadow Atlas rendering and CSM buffer upload.

use korppi_math::{Mat4, Vec3};
use korppi_render::shadows::{compute_cascades, CascadeConfig, ShadowAtlas};
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
fn test_shadow_atlas_headless_execution_opengl() {
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
        label: Some("gl_shadow_forward_color"),
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

    let shadow_atlas_v = graph.create_transient_texture(TransientTextureDesc::new(
        "TransientShadowAtlas",
        512,
        512,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    // Pass 1: Render Shadow Atlas depth
    graph
        .add_pass("ShadowAtlasPass", PassType::Graphics)
        .write_depth(shadow_atlas_v, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 512.0, 512.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 512, 512);
        });

    // Pass 2: Forward Shading reading shadow atlas
    graph
        .add_pass("ForwardShadingPass", PassType::Graphics)
        .read_texture(
            shadow_atlas_v,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(
            color_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.4, 0.2, 0.6, 1.0],
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile shadow graph failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create recorder failed");
    recorder.begin().expect("Begin recorder failed");
    compiled.execute(&mut *recorder).expect("Execute failed");
    let cmd_buf = recorder.finish().expect("Finish recorder failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(color_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 64 * 64 * 4);
    // [0.4*255=102, 0.2*255=51, 0.6*255=153, 255]
    assert!((pixels[0] as i32 - 102).abs() <= 2);
    assert!((pixels[1] as i32 - 51).abs() <= 2);
    assert!((pixels[2] as i32 - 153).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(color_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}

#[test]
fn test_csm_uniform_buffer_upload_and_roundtrip() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let config = CascadeConfig {
        cascade_count: 4,
        split_lambda: 0.85,
        shadow_map_resolution: 2048,
        near_plane: 0.5,
        far_plane: 150.0,
    };
    let cam_view = Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
    let cam_proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.5, 150.0);
    let light_dir = Vec3::new(0.2, -1.0, 0.5).normalize();

    let cascades = compute_cascades(cam_view, cam_proj, light_dir, &config);
    assert_eq!(cascades.len(), 4);

    // Build binary UBO data containing 4 cascade matrices and split depths
    let mut ubo_bytes = Vec::new();
    for cascade in &cascades {
        for col in cascade.view_proj_matrix.to_cols_array() {
            ubo_bytes.extend_from_slice(&col.to_ne_bytes());
        }
        ubo_bytes.extend_from_slice(&cascade.split_depth.to_ne_bytes());
        ubo_bytes.extend_from_slice(&cascade.texel_size_world.to_ne_bytes());
        ubo_bytes.extend_from_slice(&0.0f32.to_ne_bytes()); // padding
        ubo_bytes.extend_from_slice(&0.0f32.to_ne_bytes()); // padding
    }

    let buf_desc = BufferDescriptor {
        size: ubo_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("csm_cascade_ubo"),
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
fn test_shadow_atlas_headless_execution_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        println!("Headless Vulkan 1.3 device unavailable, skipping Vulkan shadow test");
        return;
    };

    let mut atlas = ShadowAtlas::new(1024);
    let slot0 = atlas.allocate(512).expect("Slot 0 allocation");
    let slot1 = atlas.allocate(512).expect("Slot 1 allocation");
    assert_ne!(slot0.uv_min, slot1.uv_min);

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
        label: Some("vk_shadow_color_target"),
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

    let shadow_atlas_v = graph.create_transient_texture(TransientTextureDesc::new(
        "VkShadowAtlas",
        256,
        256,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    graph
        .add_pass("VkShadowPass", PassType::Graphics)
        .write_depth(shadow_atlas_v, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 256.0, 256.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 256, 256);
        });

    graph
        .add_pass("VkForwardPass", PassType::Graphics)
        .read_texture(
            shadow_atlas_v,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(
            color_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.9, 0.1, 0.3, 1.0],
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 32, 32);
        });

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile shadow graph failed");

    let mut recorder = device
        .create_command_recorder()
        .expect("Create recorder failed");
    recorder.begin().expect("Begin recorder failed");
    compiled.execute(&mut *recorder).expect("Execute failed");
    let cmd_buf = recorder.finish().expect("Finish recorder failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(color_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 32 * 32 * 4);
    // [0.9*255=230, 0.1*255=25, 0.3*255=76, 255]
    assert!((pixels[0] as i32 - 230).abs() <= 2);
    assert!((pixels[1] as i32 - 25).abs() <= 2);
    assert!((pixels[2] as i32 - 76).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(color_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}
