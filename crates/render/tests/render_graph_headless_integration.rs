//! Offscreen headless hardware integration tests for Render Graph execution and transient resource pooling.

use korppi_render::{
    PassType, RenderGraph, TextureAccess, TransientResourcePool, TransientTextureDesc,
};
use korppi_rhi::{
    AccessFlags, GraphicsDevice, ImageLayout, LoadOp, PipelineStageFlags, SampleCount, StoreOp,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_render_graph_headless_dynamic_rendering_execution_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let target_desc = TextureDescriptor {
        width: 64,
        height: 64,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("output_render_target"),
    };
    let target_rhi = device
        .create_texture(&target_desc, None)
        .expect("Failed to create target texture");

    let mut pool = TransientResourcePool::new();

    let mut graph = RenderGraph::new();
    let target_v = graph.import_texture(
        target_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );

    let transient_v = graph.create_transient_texture(TransientTextureDesc::new(
        "transient_tex",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    // Pass 1: Render red to transient texture
    graph
        .add_pass("Pass1_ClearTransient", PassType::Graphics)
        .write_color(
            transient_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [1.0, 0.0, 0.0, 1.0],
        )
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });

    // Pass 2: Read transient texture and render green to target
    graph
        .add_pass("Pass2_RenderToOutput", PassType::Graphics)
        .read_texture(
            transient_v,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(
            target_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.0, 1.0, 0.0, 1.0],
        )
        .record(move |ctx, cmd| {
            let real_transient = ctx.get_texture(transient_v);
            let real_target = ctx.get_texture(target_v);
            assert_ne!(real_transient, real_target);
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
    compiled
        .execute(&mut *recorder)
        .expect("Compiled graph execution failed");
    let cmd_buf = recorder.finish().expect("Finish command recording failed");

    device
        .submit(cmd_buf)
        .expect("Submit command buffer failed");
    device.wait_idle().expect("Wait idle failed");

    // Read back output pixels and verify clear color (green: [0, 255, 0, 255])
    let pixels = device
        .read_texture(target_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 64 * 64 * 4);
    assert_eq!(pixels[0], 0);
    assert_eq!(pixels[1], 255);
    assert_eq!(pixels[2], 0);
    assert_eq!(pixels[3], 255);

    // Clean up
    device
        .destroy_texture(target_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}

#[test]
fn test_render_graph_transient_pool_multi_frame_reuse() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut pool = TransientResourcePool::new();

    // Frame 1
    let tex_h1 = {
        let mut graph = RenderGraph::new();
        let transient = graph.create_transient_texture(TransientTextureDesc::new(
            "frame_tex",
            128,
            128,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
        ));
        graph
            .add_pass("Frame1Pass", PassType::Graphics)
            .write_color(
                transient,
                0,
                LoadOp::Clear,
                StoreOp::Store,
                [0.0, 0.0, 0.0, 1.0],
            )
            .mark_as_root();

        let compiled = graph
            .compile_with_device(&device, &mut pool)
            .expect("Frame 1 compile failed");
        compiled.resolve_texture(transient)
    };

    // Frame 2: Graph with identical texture requirement
    let tex_h2 = {
        let mut graph = RenderGraph::new();
        let transient = graph.create_transient_texture(TransientTextureDesc::new(
            "frame_tex",
            128,
            128,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
        ));
        graph
            .add_pass("Frame2Pass", PassType::Graphics)
            .write_color(
                transient,
                0,
                LoadOp::Clear,
                StoreOp::Store,
                [1.0, 1.0, 1.0, 1.0],
            )
            .mark_as_root();

        let compiled = graph
            .compile_with_device(&device, &mut pool)
            .expect("Frame 2 compile failed");
        compiled.resolve_texture(transient)
    };

    // The pool should reuse the physical texture allocated in Frame 1
    assert_eq!(tex_h1, tex_h2);

    pool.destroy(&device).expect("Destroy pool failed");
}

#[test]
fn test_render_graph_headless_dynamic_rendering_execution_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        println!("Headless Vulkan 1.3 device unavailable, skipping Vulkan graph test");
        return;
    };

    let target_desc = TextureDescriptor {
        width: 32,
        height: 32,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("vk_output_target"),
    };
    let target_rhi = device
        .create_texture(&target_desc, None)
        .expect("Failed to create target texture");

    let mut pool = TransientResourcePool::new();

    let mut graph = RenderGraph::new();
    let target_v = graph.import_texture(
        target_rhi,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::Undefined,
        },
        32,
        32,
    );

    graph
        .add_pass("VkPass", PassType::Graphics)
        .write_color(
            target_v,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.2, 0.4, 0.8, 1.0],
        )
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
    let cmd_buf = recorder.finish().expect("Finish command recorder failed");

    device.submit(cmd_buf).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    let pixels = device
        .read_texture(target_rhi, 0)
        .expect("Read texture failed");
    assert_eq!(pixels.len(), 32 * 32 * 4);
    // [0.2 * 255 ~= 51, 0.4 * 255 ~= 102, 0.8 * 255 ~= 204, 255]
    assert!((pixels[0] as i32 - 51).abs() <= 2);
    assert!((pixels[1] as i32 - 102).abs() <= 2);
    assert!((pixels[2] as i32 - 204).abs() <= 2);
    assert_eq!(pixels[3], 255);

    device
        .destroy_texture(target_rhi)
        .expect("Destroy texture failed");
    pool.destroy(&device).expect("Destroy pool failed");
}

fn check_g4_and_g5_and_msaa_on_device(device: &dyn GraphicsDevice) {
    let _ = device.take_diagnostics();

    // 1. G4: Overlapping transient textures with identical descriptors must get distinct GPU handles
    let color_desc = TextureDescriptor {
        width: 64,
        height: 64,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::SAMPLED,
        dimension: TextureDimension::D2,
        label: Some("g4_out"),
    };
    let out_tex = device
        .create_texture(&color_desc, None)
        .expect("create out_tex");
    let mut pool = TransientResourcePool::new();
    let mut g = RenderGraph::new();
    let t_desc = TransientTextureDesc::new(
        "GBuffer",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let t1 = g.create_transient_texture(t_desc.clone());
    let t2 = g.create_transient_texture(t_desc);
    let out = g.import_texture(
        out_tex,
        TextureAccess {
            stage: PipelineStageFlags::TOP_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );
    g.add_pass("WriteBoth", PassType::Graphics)
        .write_color(t1, 0, LoadOp::Clear, StoreOp::Store, [1.0, 0.0, 0.0, 1.0])
        .write_color(t2, 1, LoadOp::Clear, StoreOp::Store, [0.0, 1.0, 0.0, 1.0])
        .record(|_, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });
    g.add_pass("ReadBoth", PassType::Graphics)
        .read_color(t1)
        .read_color(t2)
        .write_color(out, 0, LoadOp::Clear, StoreOp::Store, [0.0, 0.0, 1.0, 1.0])
        .record(|_, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });

    let mut compiled = g
        .compile_with_device(device, &mut pool)
        .expect("G4 compile");
    let a = compiled.resolve_texture(t1);
    let b = compiled.resolve_texture(t2);
    assert_ne!(
        a, b,
        "G4: Simultaneously live transients must resolve to distinct GPU textures"
    );

    let mut rec = device.create_command_recorder().expect("recorder");
    rec.begin().expect("begin");
    compiled.execute(&mut *rec).expect("execute G4");
    device
        .submit_and_wait(rec.finish().expect("finish"))
        .expect("submit G4");

    // 2. Per-frame slot pooling: slot 1 must not reuse slot 0's textures while slot 0 is in flight,
    //    and returning to slot 0 reuses slot 0's textures.
    let slot0 = device.begin_frame().expect("begin_frame 0");
    pool.begin_frame(slot0);
    let mut g_slot0 = RenderGraph::new();
    let t_s0 = g_slot0.create_transient_texture(TransientTextureDesc::new(
        "GBuffer",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    g_slot0
        .add_pass("Slot0Pass", PassType::Graphics)
        .write_color(t_s0, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .mark_as_root();
    let c_s0 = g_slot0
        .compile_with_device(device, &mut pool)
        .expect("compile slot 0");
    let h_s0 = c_s0.resolve_texture(t_s0);
    device.end_frame().expect("end_frame 0");

    let slot1 = device.begin_frame().expect("begin_frame 1");
    pool.begin_frame(slot1);
    let mut g_slot1 = RenderGraph::new();
    let t_s1 = g_slot1.create_transient_texture(TransientTextureDesc::new(
        "GBuffer",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    g_slot1
        .add_pass("Slot1Pass", PassType::Graphics)
        .write_color(t_s1, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .mark_as_root();
    let c_s1 = g_slot1
        .compile_with_device(device, &mut pool)
        .expect("compile slot 1");
    let h_s1 = c_s1.resolve_texture(t_s1);
    device.end_frame().expect("end_frame 1");
    assert_ne!(
        h_s0, h_s1,
        "Different frame-in-flight slots must not share live pooled textures"
    );

    // 3. G5: Allocate >= 100 textures so imported TextureHandle has id >= 100, and verify no collision
    let mut dummy_textures = Vec::new();
    let high_id_target = loop {
        let t = device
            .create_texture(&color_desc, None)
            .expect("create dummy texture");
        dummy_textures.push(t);
        if t.id() >= 100 {
            break t;
        }
    };
    let mut g5 = RenderGraph::new();
    let scratch = g5.create_transient_texture(TransientTextureDesc::new(
        "Scratch",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let out5 = g5.import_texture(
        high_id_target,
        TextureAccess {
            stage: PipelineStageFlags::TOP_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );
    g5.add_pass("DrawScratch", PassType::Graphics)
        .write_color(scratch, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });
    g5.add_pass("CompositeToHighId", PassType::Graphics)
        .read_color(scratch)
        .write_color(out5, 0, LoadOp::Clear, StoreOp::Store, [1.0, 0.0, 1.0, 1.0])
        .record(|_, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });
    let mut c5 = g5
        .compile_with_device(device, &mut pool)
        .expect("G5 compile");
    assert_eq!(
        c5.resolve_texture(out5),
        high_id_target,
        "G5: Imported texture with id >= 100 must not collide with transient handles"
    );
    assert_eq!(
        c5.passes()[1]
            .rendering_descriptor()
            .unwrap()
            .color_attachments[0]
            .texture,
        high_id_target
    );

    // 4. 4x MSAA color attachment resolved into single-sample imported target through RenderGraph
    let mut g_msaa = RenderGraph::new();
    let mut msaa_desc = TransientTextureDesc::new(
        "MsaaColor",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    );
    msaa_desc.sample_count = SampleCount::Count4;
    let msaa_tex = g_msaa.create_transient_texture(msaa_desc);
    let resolve_out = g_msaa.import_texture_with_final_access(
        out_tex,
        TextureAccess {
            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            layout: ImageLayout::ColorAttachment,
        },
        TextureAccess {
            stage: PipelineStageFlags::TRANSFER,
            access: AccessFlags::TRANSFER_READ,
            layout: ImageLayout::TransferSrc,
        },
        64,
        64,
    );
    g_msaa
        .add_pass("MsaaClearAndResolve", PassType::Graphics)
        .write_color_msaa(
            msaa_tex,
            resolve_out,
            0,
            LoadOp::Clear,
            StoreOp::DontCare,
            [0.0, 1.0, 0.0, 1.0],
        )
        .record(|_, cmd| {
            cmd.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 64, 64);
        });
    let mut c_msaa = g_msaa
        .compile_with_device(device, &mut pool)
        .expect("compile MSAA graph");
    let mut rec2 = device.create_command_recorder().expect("recorder 2");
    rec2.begin().expect("begin 2");
    c5.execute(&mut *rec2).expect("execute G5");
    c_msaa.execute(&mut *rec2).expect("execute MSAA");
    device
        .submit_and_wait(rec2.finish().expect("finish 2"))
        .expect("submit 2");

    let resolved_pixels = device
        .read_texture(out_tex, 0)
        .expect("read resolved out_tex");
    assert_eq!(resolved_pixels[0], 0);
    assert_eq!(resolved_pixels[1], 255);
    assert_eq!(resolved_pixels[2], 0);
    assert_eq!(resolved_pixels[3], 255);

    let g5_pixels = device
        .read_texture(high_id_target, 0)
        .expect("read high_id_target");
    assert_eq!(g5_pixels[0], 255);
    assert_eq!(g5_pixels[1], 0);
    assert_eq!(g5_pixels[2], 255);
    assert_eq!(g5_pixels[3], 255);

    // Ensure zero API/synchronization validation errors or warnings
    let diags = device.take_diagnostics();
    assert!(
        diags.is_empty(),
        "Expected zero backend validation diagnostics, got: {diags:#?}"
    );

    pool.destroy(device).expect("destroy pool");
    device.destroy_texture(out_tex).expect("destroy out_tex");
    for t in dummy_textures {
        device.destroy_texture(t).expect("destroy dummy");
    }
}

#[test]
fn test_g4_g5_msaa_and_slot_pooling_opengl() {
    let device = OpenGlDevice::create_headless().expect("OpenGL headless");
    check_g4_and_g5_and_msaa_on_device(&device);
}

#[test]
fn test_g4_g5_msaa_and_slot_pooling_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        return;
    };
    check_g4_and_g5_and_msaa_on_device(&device);
}
