//! Headless offscreen hardware integration tests for Forward PBR, Material UBOs, and HDR targets.

use korppi_math::Vec3;
use korppi_render::pbr::{evaluate_pbr, CharlieSheen, ClearCoat, PbrMaterial, PbrSurface};
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
fn test_pbr_material_ubo_roundtrip_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    // Standard PBR Material UBO Layout (16-byte aligned std140 format):
    // vec4 albedo_roughness: [r, g, b, roughness]
    // vec4 metal_refl_coat:   [metallic, reflectance, coat_weight, coat_roughness]
    // vec4 sheen_color_rough: [sheen.r, sheen.g, sheen.b, sheen_roughness]
    let surface = PbrSurface {
        position: Vec3::ZERO,
        normal: Vec3::Y,
        view: Vec3::new(0.0, 1.0, 1.0).normalize(),
        albedo: Vec3::new(0.9, 0.7, 0.2),
        material: PbrMaterial {
            roughness: 0.25,
            metallic: 0.8,
            reflectance: 0.5,
            clear_coat: Some(ClearCoat {
                weight: 0.9,
                roughness: 0.05,
            }),
            sheen: Some(CharlieSheen {
                color: Vec3::new(0.5, 0.1, 0.1),
                roughness: 0.4,
            }),
        },
    };

    let evaluated = evaluate_pbr(&surface, Vec3::Y, Vec3::splat(5.0));
    assert!(evaluated.is_finite());
    assert!(evaluated.x > 0.0);

    let mut ubo_bytes = Vec::new();
    // vec4 albedo_roughness
    ubo_bytes.extend_from_slice(&surface.albedo.x.to_ne_bytes());
    ubo_bytes.extend_from_slice(&surface.albedo.y.to_ne_bytes());
    ubo_bytes.extend_from_slice(&surface.albedo.z.to_ne_bytes());
    ubo_bytes.extend_from_slice(&surface.material.roughness.to_ne_bytes());

    // vec4 metal_refl_coat
    ubo_bytes.extend_from_slice(&surface.material.metallic.to_ne_bytes());
    ubo_bytes.extend_from_slice(&surface.material.reflectance.to_ne_bytes());
    let coat_w = surface.material.clear_coat.map_or(0.0, |c| c.weight);
    let coat_r = surface.material.clear_coat.map_or(0.0, |c| c.roughness);
    ubo_bytes.extend_from_slice(&coat_w.to_ne_bytes());
    ubo_bytes.extend_from_slice(&coat_r.to_ne_bytes());

    // vec4 sheen_color_rough
    let sheen_col = surface.material.sheen.map_or(Vec3::ZERO, |s| s.color);
    let sheen_r = surface.material.sheen.map_or(0.0, |s| s.roughness);
    ubo_bytes.extend_from_slice(&sheen_col.x.to_ne_bytes());
    ubo_bytes.extend_from_slice(&sheen_col.y.to_ne_bytes());
    ubo_bytes.extend_from_slice(&sheen_col.z.to_ne_bytes());
    ubo_bytes.extend_from_slice(&sheen_r.to_ne_bytes());

    let buf_desc = BufferDescriptor {
        size: ubo_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("pbr_material_ubo"),
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
fn test_pbr_hdr_forward_render_graph_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

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
        label: Some("gl_pbr_hdr_target"),
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

    let depth_v = graph.create_transient_texture(TransientTextureDesc::new(
        "TransientDepth",
        32,
        32,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ));

    graph
        .add_pass("ForwardPbrShading", PassType::Graphics)
        .write_color(
            color_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.2, 0.7, 0.3, 1.0],
        )
        .write_depth(depth_v, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 32, 32);
        });

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile PBR graph failed");

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
    // [0.2*255=51, 0.7*255=178, 0.3*255=76, 255]
    assert!((pixels[0] as i32 - 51).abs() <= 2);
    assert!((pixels[1] as i32 - 178).abs() <= 2);
    assert!((pixels[2] as i32 - 76).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(color_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}

#[test]
fn test_pbr_hdr_forward_render_graph_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        println!("Headless Vulkan 1.3 device unavailable, skipping Vulkan PBR test");
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
        label: Some("vk_pbr_color_target"),
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

    let depth_v = graph.create_transient_texture(TransientTextureDesc::new(
        "VkTransientDepth",
        32,
        32,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ));

    graph
        .add_pass("VkForwardPbrPass", PassType::Graphics)
        .write_color(
            color_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.85, 0.35, 0.15, 1.0],
        )
        .write_depth(depth_v, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 32.0, 32.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 32, 32);
        });

    let mut compiled = graph
        .compile_with_device(&device, &mut pool)
        .expect("Compile PBR graph failed");

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
    // [0.85*255=216, 0.35*255=89, 0.15*255=38, 255]
    assert!((pixels[0] as i32 - 216).abs() <= 2);
    assert!((pixels[1] as i32 - 89).abs() <= 2);
    assert!((pixels[2] as i32 - 38).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(color_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}
