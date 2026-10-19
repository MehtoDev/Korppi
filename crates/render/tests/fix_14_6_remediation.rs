//! Remediation regression tests for `FIX-14.6` (`RND-17` through `RND-26`, probes `G15`–`G25`).

use korppi_math::{Mat4, Vec2, Vec3};
use korppi_render::clustered::{ClusterGridConfig, FroxelAabb, GpuLightCuller, PointLight};
use korppi_render::graph::{
    PassType, RenderGraph, RenderGraphTextureHandle, TextureAccess, TransientResourcePool,
    TransientTextureDesc,
};
use korppi_render::pbr::{
    compute_effective_roughness_toksvig, compute_geometric_roughness, compute_toksvig_roughness,
    rectify_silhouette_normal,
};
use korppi_render::pipeline::{
    Camera3d, ForwardMaterial, ForwardPipeline, ForwardPipelineConfig, ForwardScene, GpuMesh,
    MeshData, MeshDraw,
};
use korppi_render::postprocess::{
    BloomConfig, PostProcessPipeline, SsaoConfig, TonemapConfig, TonemapOperator,
};
use korppi_render::render2d::{CanvasScaleMode, CanvasTransform, QuadBatch2d, QuadBatchRenderer};
use korppi_render::shadows::{encode_octahedral, CachedShadowAtlas, ShadowAtlas};
use korppi_render::RenderError;
use korppi_rhi::{
    AccessFlags, ColorAttachments, FrameIndex, ImageLayout, ImageMemoryBarrier, LoadOp,
    PipelineBarrier, PipelineStageFlags, PresentMode, RenderingAttachment, RenderingDescriptor,
    SampleCount, StoreOp, SwapchainDescriptor, TextureDescriptor, TextureDimension, TextureFormat,
    TextureHandle, TextureSubresourceRange, TextureUsage,
};
use korppi_rhi_conformance::display::{display_or_skip, TestWindow};
use korppi_rhi_conformance::{Backend, Image, TestDevice};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn rnd_17_quad_batch_renderer_multi_batch_and_growth_and_texture_eviction() {
    let Some(display) =
        display_or_skip("rnd_17_quad_batch_renderer_multi_batch_and_growth_and_texture_eviction")
    else {
        return;
    };
    const W: u32 = 128;
    const H: u32 = 128;

    for backend in Backend::ALL {
        let window = Arc::new(TestWindow::create(&display, W, H));
        let device = TestDevice::open_for_display(backend, window.clone());
        let swapchain = device
            .create_swapchain(
                window.clone(),
                &SwapchainDescriptor {
                    present_mode: PresentMode::Immediate,
                    srgb: false,
                    ..SwapchainDescriptor::new(W, H)
                },
            )
            .expect("create_swapchain");
        let format = device
            .swapchain_info(swapchain)
            .expect("swapchain_info")
            .format;

        let mut renderer = QuadBatchRenderer::new(&*device, format).expect("QuadBatchRenderer");

        // Batch A: left red quad (1 quad <= initial 64 capacity)
        let mut batch_a = QuadBatch2d::new(8);
        batch_a.push_quad(
            Vec2::new(8.0, 8.0),
            Vec2::new(56.0, 120.0),
            Vec2::ZERO,
            Vec2::ONE,
            [1.0, 0.0, 0.0, 1.0],
        );

        // Batch B: right green quads (70 quads > initial 64 capacity -> forces in-frame buffer growth)
        let mut batch_b = QuadBatch2d::new(80);
        for i in 0..70 {
            let y0 = 8.0 + (i % 10) as f32 * 8.0;
            batch_b.push_quad(
                Vec2::new(72.0, y0),
                Vec2::new(120.0, y0 + 16.0),
                Vec2::ZERO,
                Vec2::ONE,
                [0.0, 1.0, 0.0, 1.0],
            );
        }

        // Also test texture cache eviction
        let custom_tex = device
            .create_texture(
                &TextureDescriptor {
                    width: 1,
                    height: 1,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                    dimension: TextureDimension::D2,
                    label: Some("rnd17_custom_tex"),
                },
                Some(&[0, 0, 255, 255]),
            )
            .expect("create custom_tex");
        renderer
            .evict_texture(&*device, custom_tex)
            .expect("evict uncached texture");
        device
            .destroy_texture(custom_tex)
            .expect("destroy custom_tex");

        for _ in 0..3 {
            let fidx = device.begin_frame().expect("begin_frame");
            renderer
                .begin_frame(&*device, fidx)
                .expect("renderer.begin_frame");
            let image = device
                .acquire_next_image(swapchain)
                .expect("acquire")
                .expect("image");
            let mut rec = device.create_command_recorder().expect("recorder");
            rec.begin().expect("begin");

            rec.pipeline_barrier(
                &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
                    texture: image.texture,
                    range: TextureSubresourceRange::ALL,
                    src_stages: PipelineStageFlags::TOP_OF_PIPE,
                    src_access: AccessFlags::empty(),
                    dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    old_layout: ImageLayout::Undefined,
                    new_layout: ImageLayout::ColorAttachment,
                }),
            );

            let mut color_attachments = ColorAttachments::new();
            color_attachments.push(RenderingAttachment {
                texture: image.texture,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: Some([0.0, 0.0, 0.0, 1.0]),
                resolve_target: None,
            });
            rec.begin_rendering(&RenderingDescriptor {
                offset_x: 0,
                offset_y: 0,
                width: W,
                height: H,
                color_attachments,
                depth_attachment: None,
            })
            .expect("begin_rendering");

            // First batch in pass draws left red quad
            renderer
                .record_in_pass(
                    &*device,
                    rec.as_mut(),
                    fidx,
                    (W, H),
                    (W as f32, H as f32),
                    &batch_a,
                )
                .expect("record_in_pass batch_a");
            // Second batch in the SAME pass draws 70 right green quads (triggers buffer growth without overwriting or destroying batch_a's vertices)
            renderer
                .record_in_pass(
                    &*device,
                    rec.as_mut(),
                    fidx,
                    (W, H),
                    (W as f32, H as f32),
                    &batch_b,
                )
                .expect("record_in_pass batch_b");

            rec.end_rendering().expect("end_rendering");
            rec.pipeline_barrier(
                &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
                    texture: image.texture,
                    range: TextureSubresourceRange::ALL,
                    src_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_stages: PipelineStageFlags::BOTTOM_OF_PIPE,
                    dst_access: AccessFlags::empty(),
                    old_layout: ImageLayout::ColorAttachment,
                    new_layout: ImageLayout::Present,
                }),
            );

            let sub = device
                .submit(rec.finish().expect("finish"))
                .expect("submit");
            device.present(&image).expect("present");
            device.wait_for_submission(sub).expect("wait");
        }
        device.wait_idle().expect("wait_idle");

        let deadline = Instant::now() + Duration::from_secs(2);
        let img = loop {
            let im = window.capture(W, H);
            if (im.pixel(32, 32)[0] > 200 && im.pixel(96, 32)[1] > 200) || Instant::now() > deadline
            {
                break im;
            }
            std::thread::sleep(Duration::from_millis(10));
        };

        let left_px = img.pixel(32, 32);
        let right_px = img.pixel(96, 32);
        assert!(
            left_px[0] > 200 && left_px[1] < 20,
            "{backend:?} RND-17: left red quad from batch A must remain intact after batch B in-frame growth, got {left_px:?}"
        );
        assert!(
            right_px[1] > 200 && right_px[0] < 20,
            "{backend:?} RND-17: right green quad from batch B must be drawn, got {right_px:?}"
        );

        renderer.destroy(&*device).expect("destroy renderer");
        device
            .destroy_swapchain(swapchain)
            .expect("destroy_swapchain");
        device.assert_no_api_errors();
    }
}

#[test]
fn rnd_18_postprocess_pipeline_wires_ssao_and_bloom_and_target_dimensions() {
    let mut graph = RenderGraph::new();
    let hdr = graph.create_transient_texture(TransientTextureDesc::new(
        "hdr",
        640,
        360,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let depth = graph.create_transient_texture(TransientTextureDesc::new(
        "depth",
        640,
        360,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let ldr = graph.create_transient_texture(TransientTextureDesc::new(
        "ldr",
        640,
        360,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    graph
        .add_pass("scene", PassType::Graphics)
        .write_color(hdr, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .write_depth(depth, LoadOp::Clear, StoreOp::Store, 0.0);

    let pp = PostProcessPipeline::new(
        TonemapConfig::default(),
        Some(BloomConfig {
            mip_levels: 4,
            ..BloomConfig::default()
        }),
        Some(SsaoConfig::default()),
    );
    pp.setup_graph(&mut graph, hdr, Some(depth), ldr);

    let device = TestDevice::open(Backend::Vulkan);
    let mut pool = TransientResourcePool::new();
    let mut compiled = graph
        .compile_with_device(&*device, &mut pool)
        .expect("compile postprocess graph");
    let pass_names: Vec<&str> = compiled.passes().iter().map(|p| p.name()).collect();

    assert!(
        pass_names.contains(&"SsaoGenerationPass"),
        "RND-18: SsaoGenerationPass must not be dead-pass-culled, got {pass_names:?}"
    );
    assert!(
        pass_names.contains(&"SsaoBlurPass"),
        "RND-18: SsaoBlurPass must be wired into composite pass, got {pass_names:?}"
    );
    for mip in 0..4 {
        let down_name = format!("BloomDownsamplePass_{mip}");
        assert!(
            pass_names.contains(&down_name.as_str()),
            "RND-18: missing {down_name} in {pass_names:?}"
        );
    }
    for mip in 0..3 {
        let up_name = format!("BloomUpsamplePass_{mip}");
        assert!(
            pass_names.contains(&up_name.as_str()),
            "RND-18: missing {up_name} in {pass_names:?}"
        );
    }
    assert!(
        pass_names.contains(&"PostProcessCompositePass"),
        "RND-18: missing PostProcessCompositePass in {pass_names:?}"
    );

    let mut rec = device.create_command_recorder().expect("recorder");
    rec.begin().expect("begin");
    compiled.execute(rec.as_mut()).expect("execute");
    let _cmd = rec.finish().expect("finish");
    pool.destroy(&*device).unwrap();
    device.assert_no_api_errors();
}

#[test]
fn rnd_19_and_20_forward_pipeline_bloom_mip_levels_and_msaa_depth_prepass() {
    let Some(display) =
        display_or_skip("rnd_19_and_20_forward_pipeline_bloom_mip_levels_and_msaa_depth_prepass")
    else {
        return;
    };
    const W: u32 = 64;
    const H: u32 = 64;

    let window = Arc::new(TestWindow::create(&display, W, H));
    let device = TestDevice::open_for_display(Backend::Vulkan, window.clone());
    let swapchain = device
        .create_swapchain(
            window.clone(),
            &SwapchainDescriptor {
                present_mode: PresentMode::Immediate,
                srgb: false,
                ..SwapchainDescriptor::new(W, H)
            },
        )
        .expect("create_swapchain");
    let format = device
        .swapchain_info(swapchain)
        .expect("swapchain_info")
        .format;

    let quad_mesh = GpuMesh::upload(&*device, &MeshData::facing_quad(1.0, 1.0), Some("quad"))
        .expect("upload quad");

    let camera = Camera3d {
        position: Vec3::new(0.0, 0.0, 2.0),
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y_radians: std::f32::consts::FRAC_PI_3,
        aspect_ratio: 1.0,
        z_near: 0.2,
        z_far: 10.0,
    };

    let render_with_mips = |mip_levels: u32| -> Image {
        let config = ForwardPipelineConfig {
            width: W,
            height: H,
            ldr_format: format,
            msaa_samples: SampleCount::Count4,
            shadow_atlas_size: 512,
            tonemap: TonemapConfig {
                operator: TonemapOperator::AgX,
                exposure: 1.0,
                gamma: 2.2,
                white_point: 10.0,
            },
            bloom: Some(BloomConfig {
                threshold: 1.0,
                soft_knee: 0.5,
                intensity: 0.8,
                filter_radius: 1.5,
                mip_levels,
            }),
            ssao: None,
        };
        let mut pipeline = ForwardPipeline::new(&*device, config).expect("ForwardPipeline");
        let scene = ForwardScene {
            camera,
            directional_light: None,
            ambient_light: Vec3::ZERO,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            draws: vec![MeshDraw {
                mesh: quad_mesh,
                transform: Mat4::from_scale(Vec3::new(0.25, 0.25, 1.0)),
                material: ForwardMaterial {
                    albedo: Vec3::ONE,
                    emissive: Vec3::new(20.0, 20.0, 20.0),
                    ..Default::default()
                },
                casts_shadows: false,
            }],
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
        };

        for frame in 0..3 {
            device.begin_frame().expect("begin_frame");
            let image = device
                .acquire_next_image(swapchain)
                .expect("acquire")
                .expect("image");
            let mut rec = device.create_command_recorder().expect("recorder");
            rec.begin().expect("begin");
            pipeline
                .record_scene(
                    &*device,
                    rec.as_mut(),
                    FrameIndex::new(frame % 2),
                    &scene,
                    image.texture,
                )
                .expect("record_scene");
            let sub = device
                .submit(rec.finish().expect("finish"))
                .expect("submit");
            device.present(&image).expect("present");
            device.wait_for_submission(sub).expect("wait");
        }
        device.wait_idle().expect("wait_idle");
        let deadline = Instant::now() + Duration::from_secs(2);
        let img = loop {
            let im = window.capture(W, H);
            let ready = if mip_levels > 1 {
                im.pixel(32, 32)[0] > 180 && im.pixel(16, 32)[0] > 10
            } else {
                im.pixel(32, 32)[0] > 180 && im.pixel(16, 32)[0] < 5
            };
            if ready || Instant::now() > deadline {
                break im;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        pipeline.destroy(&*device).expect("destroy pipeline");
        img
    };

    let img_1mip = render_with_mips(1);
    let img_5mips = render_with_mips(5);

    let px_1 = img_1mip.pixel(16, 32);
    let px_5 = img_5mips.pixel(16, 32);
    assert!(
        px_5[0] > px_1[0] + 10,
        "RND-20: 5-mip bloom pyramid must spread wide halo farther than 1-mip (mip1={px_1:?}, mip5={px_5:?})"
    );

    quad_mesh.destroy(&*device).expect("destroy mesh");
    device
        .destroy_swapchain(swapchain)
        .expect("destroy_swapchain");
    device.assert_no_api_errors();
}

#[test]
fn rnd_21_render_graph_validates_foreign_and_out_of_bounds_handles() {
    let mut graph_a = RenderGraph::new();
    let tex_a0 = graph_a.create_transient_texture(TransientTextureDesc::new(
        "a0",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let tex_a1_unused = graph_a.create_transient_texture(TransientTextureDesc::new(
        "a1_unused",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));
    let _ = tex_a0;

    let mut graph_b = RenderGraph::new();
    let _tex_b0 = graph_b.create_transient_texture(TransientTextureDesc::new(
        "b0",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    // Using foreign handle `tex_a1_unused` from `graph_a` in `graph_b` must fail compilation
    graph_b
        .add_pass("foreign_pass", PassType::Graphics)
        .write_color(tex_a1_unused, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .mark_as_root();

    let res = graph_b.compile();
    assert!(
        matches!(res, Err(RenderError::InvalidHandle(_))),
        "RND-21: foreign handle from another RenderGraph must be rejected with InvalidHandle, got {res:?}"
    );

    // Out-of-bounds forged handle must fail cleanly without panicking
    let mut graph_c = RenderGraph::new();
    graph_c
        .add_pass("oob_pass", PassType::Graphics)
        .write_color(
            RenderGraphTextureHandle::from_raw(999),
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.0; 4],
        )
        .mark_as_root();
    let res_oob = graph_c.compile();
    assert!(
        matches!(res_oob, Err(RenderError::InvalidHandle(_))),
        "RND-21: out-of-bounds handle must return Err(RenderError::InvalidHandle), got {res_oob:?}"
    );

    // Resolving a culled transient texture handle must not panic
    let mut graph_d = RenderGraph::new();
    let live_tex = graph_d.create_transient_texture(TransientTextureDesc::new(
        "live",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));
    let culled_tex = graph_d.create_transient_texture(TransientTextureDesc::new(
        "culled",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));
    graph_d
        .add_pass("live_pass", PassType::Graphics)
        .write_color(live_tex, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .mark_as_root();

    let device = TestDevice::open(Backend::Vulkan);
    let mut pool = TransientResourcePool::new();
    let compiled_d = graph_d
        .compile_with_device(&*device, &mut pool)
        .expect("compile graph_d");
    assert_eq!(compiled_d.try_resolve_texture(culled_tex), None);
    let fallback = compiled_d.resolve_texture(culled_tex);
    assert!(!fallback.is_valid());
    pool.destroy(&*device).unwrap();
    device.assert_no_api_errors();
}

#[test]
fn rnd_22_render_graph_subresource_barriers_track_all_mips_and_layers() {
    // G20: Imported 4-mip texture initialized in ShaderReadOnly; pass 0 writes mip 0; pass 1 reads ALL.
    {
        let mut graph = RenderGraph::new();
        let tex = graph.import_texture(
            TextureHandle::new(1, 1),
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
            128,
            128,
        );

        graph
            .add_pass("write_mip0", PassType::Graphics)
            .write_color_subresource(tex, 0, 0, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4]);

        graph
            .add_pass("read_all", PassType::Graphics)
            .read_texture_subresource(
                tex,
                TextureAccess {
                    stage: PipelineStageFlags::FRAGMENT_SHADER,
                    access: AccessFlags::SHADER_READ,
                    layout: ImageLayout::ShaderReadOnly,
                },
                TextureSubresourceRange {
                    base_mip_level: 0,
                    mip_level_count: 4,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    ..TextureSubresourceRange::ALL
                },
            )
            .mark_as_root();

        let compiled = graph.compile().expect("compile G20");
        let pass1_barriers = &compiled.passes()[1].pre_pass_barrier().image_barriers;
        let has_mip0_transition = pass1_barriers.iter().any(|b| {
            b.range.base_mip_level == 0
                && b.range.mip_level_count == 1
                && b.old_layout == ImageLayout::ColorAttachment
                && b.new_layout == ImageLayout::ShaderReadOnly
        });
        assert!(
            has_mip0_transition,
            "RND-22 (G20): mip 0 must transition from ColorAttachment to ShaderReadOnly, got {pass1_barriers:?}"
        );
        for b in pass1_barriers {
            if b.range.base_mip_level > 0 {
                assert_ne!(
                    b.old_layout,
                    ImageLayout::ColorAttachment,
                    "RND-22 (G20): untouched mips 1..3 must not claim old_layout == ColorAttachment"
                );
            }
        }
    }

    // G25: Imported 4-mip texture in General; pass 0 reads mip 1 as ShaderReadOnly; pass 1 reads ALL as TransferSrc.
    {
        let mut graph = RenderGraph::new();
        let tex = graph.import_texture(
            TextureHandle::new(2, 1),
            TextureAccess {
                stage: PipelineStageFlags::COMPUTE_SHADER,
                access: AccessFlags::SHADER_WRITE,
                layout: ImageLayout::General,
            },
            128,
            128,
        );

        let mip1 = TextureSubresourceRange {
            base_mip_level: 1,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            ..TextureSubresourceRange::ALL
        };
        let all_4_mips = TextureSubresourceRange {
            base_mip_level: 0,
            mip_level_count: 4,
            base_array_layer: 0,
            array_layer_count: 1,
            ..TextureSubresourceRange::ALL
        };

        graph
            .add_pass("read_mip1", PassType::Graphics)
            .read_texture_subresource(
                tex,
                TextureAccess {
                    stage: PipelineStageFlags::FRAGMENT_SHADER,
                    access: AccessFlags::SHADER_READ,
                    layout: ImageLayout::ShaderReadOnly,
                },
                mip1,
            )
            .mark_as_root();

        graph
            .add_pass("read_all_4_mips", PassType::Transfer)
            .read_texture_subresource(
                tex,
                TextureAccess {
                    stage: PipelineStageFlags::TRANSFER,
                    access: AccessFlags::TRANSFER_READ,
                    layout: ImageLayout::TransferSrc,
                },
                all_4_mips,
            )
            .mark_as_root();

        let compiled = graph.compile().expect("compile G25");
        let p1_barriers = &compiled.passes()[1].pre_pass_barrier().image_barriers;
        let covers_mip0 = p1_barriers.iter().any(|b| {
            b.range.base_mip_level == 0
                && b.old_layout == ImageLayout::General
                && b.new_layout == ImageLayout::TransferSrc
        });
        let covers_mip1 = p1_barriers.iter().any(|b| {
            b.range.base_mip_level <= 1
                && b.range.base_mip_level + b.range.mip_level_count > 1
                && b.old_layout == ImageLayout::ShaderReadOnly
                && b.new_layout == ImageLayout::TransferSrc
        });
        let covers_mips_2_3 = p1_barriers.iter().any(|b| {
            b.range.base_mip_level <= 2
                && b.range.base_mip_level + b.range.mip_level_count >= 4
                && b.old_layout == ImageLayout::General
                && b.new_layout == ImageLayout::TransferSrc
        });
        assert!(
            covers_mip0 && covers_mip1 && covers_mips_2_3,
            "RND-22 (G25): expected disjoint barriers covering mip 0 (General), mip 1 (ShaderReadOnly), and mips 2..4 (General), got {p1_barriers:?}"
        );
    }
}

#[test]
fn rnd_23_gpu_light_culler_per_frame_output_buffers_and_capacity() {
    let device = TestDevice::open(Backend::Vulkan);
    let mut culler = GpuLightCuller::new();
    culler.init(&*device).expect("init culler");

    let froxels = vec![
        FroxelAabb {
            aabb_min: Vec3::new(-10.0, -10.0, -10.0),
            aabb_max: Vec3::new(10.0, 10.0, 10.0),
            sphere_center: Vec3::ZERO,
            sphere_radius: 17.32,
        };
        4
    ];
    let points = vec![PointLight {
        position_vs: Vec3::ZERO,
        radius: 5.0,
        color: Vec3::ONE,
        intensity: 1.0,
    }];

    let mut rec0 = device.create_command_recorder().expect("rec0");
    rec0.begin().unwrap();
    let out0 = culler
        .dispatch_culling_for_frame(
            &*device,
            rec0.as_mut(),
            FrameIndex::new(0),
            &froxels,
            &points,
            &[],
            &[],
            &[],
        )
        .expect("dispatch frame 0");
    let _cmd0 = rec0.finish().unwrap();

    let mut rec1 = device.create_command_recorder().expect("rec1");
    rec1.begin().unwrap();
    let out1 = culler
        .dispatch_culling_for_frame(
            &*device,
            rec1.as_mut(),
            FrameIndex::new(1),
            &froxels,
            &points,
            &[],
            &[],
            &[],
        )
        .expect("dispatch frame 1");
    let _cmd1 = rec1.finish().unwrap();

    assert_ne!(
        out0.records_buffer, out1.records_buffer,
        "RND-23: records_buffer must be double-buffered across FRAMES_IN_FLIGHT"
    );
    assert_ne!(
        out0.light_indices_buffer, out1.light_indices_buffer,
        "RND-23: light_indices_buffer must be double-buffered across FRAMES_IN_FLIGHT"
    );
    assert_ne!(
        out0.decal_indices_buffer, out1.decal_indices_buffer,
        "RND-23: decal_indices_buffer must be double-buffered across FRAMES_IN_FLIGHT"
    );
    assert_ne!(
        out0.probe_indices_buffer, out1.probe_indices_buffer,
        "RND-23: probe_indices_buffer must be double-buffered across FRAMES_IN_FLIGHT"
    );

    // 4 clusters * (256 point + 256 spot) * 4 bytes = 8192 bytes minimum
    assert!(
        out0.light_indices_size >= (4 * 512 * 4) as u64,
        "RND-23: light_indices_size ({}) must accommodate 512 light indices per cluster",
        out0.light_indices_size
    );

    culler.destroy(&*device).expect("destroy culler");
    device.assert_no_api_errors();
}

#[test]
fn rnd_24_and_26_shadow_atlas_dedup_subdivision_and_caching() {
    let mut atlas = ShadowAtlas::new(512);
    let tile_512 = atlas.allocate(512).expect("allocate 512");

    // Duplicate deallocate must not push duplicate free slots
    atlas.deallocate(tile_512);
    atlas.deallocate(tile_512);

    let re1 = atlas.allocate(512).expect("re-allocate 512");
    let re2 = atlas.allocate(512);
    assert_eq!(re1, tile_512);
    assert!(
        re2.is_none(),
        "RND-24: duplicate deallocate must not allow double-allocating the same 512x512 tile"
    );

    // Freeing the 512x512 tile and requesting four 256x256 tiles must subdivide the freed 512x512 slot
    atlas.deallocate(re1);
    let sub0 = atlas.allocate(256).expect("subdivide 256 #0");
    let sub1 = atlas.allocate(256).expect("subdivide 256 #1");
    let sub2 = atlas.allocate(256).expect("subdivide 256 #2");
    let sub3 = atlas.allocate(256).expect("subdivide 256 #3");
    assert_ne!(sub0, sub1);
    assert_ne!(sub1, sub2);
    assert_ne!(sub2, sub3);
    assert!(atlas.allocate(256).is_none());

    // CachedShadowAtlas steady-state test and mixed-resolution LRU eviction (`G22`)
    let mut cached = CachedShadowAtlas::new(1024);
    cached
        .register_or_update_light(1, 1024)
        .expect("register 1024 light 1");

    // Advance frame so light 1 becomes inactive, then request a 512x512 light 2 -> must evict light 1 and subdivide its 1024x1024 slot
    let _plan = cached.prepare_frame(&[], &[]);
    cached
        .register_or_update_light(2, 512)
        .expect("RND-24 (G22): evicting 1024x1024 inactive light for 512x512 request must succeed");
}

#[test]
fn rnd_25_nan_guards_and_canvas_stretch_transform() {
    // 1. encode_octahedral(Vec3::ZERO) must be finite
    let uv = encode_octahedral(Vec3::ZERO);
    assert!(
        uv.is_finite(),
        "RND-25: encode_octahedral(Vec3::ZERO) must be finite, got {uv:?}"
    );

    // 2. rectify_silhouette_normal on zero and opposite vectors must be finite
    let n_zero = rectify_silhouette_normal(Vec3::ZERO, Vec3::ZERO, 0.1);
    let n_opp = rectify_silhouette_normal(Vec3::NEG_Z, Vec3::Z, 0.1);
    assert!(
        n_zero.is_finite() && n_opp.is_finite(),
        "RND-25: rectify_silhouette_normal must be finite, got zero={n_zero:?}, opp={n_opp:?}"
    );

    // 3. JHFAA roughness > 1.0 must not panic
    let r1 = compute_geometric_roughness(Vec3::ZERO, Vec3::ZERO, 1.5, 1.0);
    let r2 = compute_toksvig_roughness(1.5, 0.8);
    let r3 = compute_effective_roughness_toksvig(1.5, Vec3::ZERO, Vec3::ZERO, 1.0, 0.8);
    assert!(
        (r1 - 1.0).abs() < 1e-6 && (r2 - 1.0).abs() < 1e-6 && (r3 - 1.0).abs() < 1e-6,
        "RND-25: roughness > 1.0 must clamp cleanly to 1.0, got ({r1}, {r2}, {r3})"
    );

    // 4. ClusterGridConfig with 0x0 viewport must have >= 1 cluster per axis
    let cfg = ClusterGridConfig::new(0, 0, 64, 16, 0.1, 100.0);
    assert!(
        cfg.dim_x() >= 1 && cfg.dim_y() >= 1 && cfg.total_clusters() >= 16,
        "RND-25: 0x0 ClusterGridConfig must clamp dim_x/dim_y to >= 1"
    );

    // 5. CanvasScaleMode::Stretch must map virtual_size to window_size independently on X and Y
    let virtual_size = Vec2::new(320.0, 180.0);
    let window_size = Vec2::new(640.0, 720.0);
    let ct = CanvasTransform::compute(virtual_size, window_size, CanvasScaleMode::Stretch);
    let br = ct.virtual_to_screen(virtual_size);
    assert!(
        (br.x - 640.0).abs() < 1e-3 && (br.y - 720.0).abs() < 1e-3,
        "RND-25: CanvasScaleMode::Stretch virtual_to_screen({virtual_size:?}) must map to {window_size:?}, got {br:?}"
    );
    let round_trip = ct.screen_to_virtual(window_size);
    assert!(
        (round_trip.x - 320.0).abs() < 1e-3 && (round_trip.y - 180.0).abs() < 1e-3,
        "RND-25: CanvasScaleMode::Stretch screen_to_virtual({window_size:?}) must map back to {virtual_size:?}, got {round_trip:?}"
    );
}
