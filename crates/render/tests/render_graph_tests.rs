//! Unit tests for render graph DAG scheduling, topological sort, dead-pass culling, and barrier deduction.

use korppi_render::{
    BufferAccess, PassType, RenderError, RenderGraph, TextureAccess, TransientBufferDesc,
    TransientTextureDesc,
};
use korppi_rhi::{
    AccessFlags, BufferUsage, ImageLayout, LoadOp, PipelineStageFlags, StoreOp, TextureFormat,
    TextureUsage,
};

#[test]
fn test_render_graph_topological_sort_linear() {
    let mut graph = RenderGraph::new();

    let t1_desc = TransientTextureDesc::new(
        "T1",
        800,
        600,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let t1 = graph.create_transient_texture(t1_desc);

    let t2_desc = TransientTextureDesc::new(
        "T2",
        800,
        600,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let t2 = graph.create_transient_texture(t2_desc);

    // Pass C: Reads T2, marked root so it won't be culled
    graph
        .add_pass("PassC", PassType::Graphics)
        .read_texture(
            t2,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    // Pass A: Writes T1
    graph
        .add_pass("PassA", PassType::Graphics)
        .write_color(t1, 0, LoadOp::Clear, StoreOp::Store, [1.0, 0.0, 0.0, 1.0])
        .record(|_ctx, _cmd| {});

    // Pass B: Reads T1, writes T2
    graph
        .add_pass("PassB", PassType::Graphics)
        .read_texture(
            t1,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(t2, 0, LoadOp::Clear, StoreOp::Store, [0.0, 1.0, 0.0, 1.0])
        .record(|_ctx, _cmd| {});

    let compiled = graph.compile().expect("Graph should compile successfully");
    let pass_names: Vec<&str> = compiled.passes().iter().map(|p| p.name()).collect();

    assert_eq!(
        pass_names,
        vec!["PassA", "PassB", "PassC"],
        "Topological sort must order dependencies: PassA -> PassB -> PassC"
    );
}

#[test]
fn test_render_graph_detects_dependency_cycle() {
    let mut graph = RenderGraph::new();

    let t1_desc = TransientTextureDesc::new(
        "T1",
        800,
        600,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let t1 = graph.create_transient_texture(t1_desc);

    let t2_desc = TransientTextureDesc::new(
        "T2",
        800,
        600,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let t2 = graph.create_transient_texture(t2_desc);

    // Pass A reads T2 and writes T1
    graph
        .add_pass("PassA", PassType::Graphics)
        .read_texture(
            t2,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(t1, 0, LoadOp::Load, StoreOp::Store, [0.0, 0.0, 0.0, 1.0])
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    // Pass B reads T1 and writes T2
    graph
        .add_pass("PassB", PassType::Graphics)
        .read_texture(
            t1,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .write_color(t2, 0, LoadOp::Load, StoreOp::Store, [0.0, 0.0, 0.0, 1.0])
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    let result = graph.compile();
    assert!(
        result.is_err(),
        "Graph compilation must detect cycle and fail"
    );
    match result.unwrap_err() {
        RenderError::CyclicDependency { .. } => {}
        other => panic!("Expected CyclicDependency error, got: {:?}", other),
    }
}

#[test]
fn test_render_graph_dead_pass_culling() {
    let mut graph = RenderGraph::new();

    let root_tex = graph.create_transient_texture(TransientTextureDesc::new(
        "RootTex",
        800,
        600,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    let dead_tex = graph.create_transient_texture(TransientTextureDesc::new(
        "DeadTex",
        800,
        600,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    // Active pass: writes RootTex, marked root
    graph
        .add_pass("ActivePass", PassType::Graphics)
        .write_color(
            root_tex,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [1.0, 1.0, 1.0, 1.0],
        )
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    // Dead pass: writes DeadTex, never read anywhere and not marked root
    graph
        .add_pass("DeadPass", PassType::Graphics)
        .write_color(
            dead_tex,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.0, 0.0, 0.0, 1.0],
        )
        .record(|_ctx, _cmd| {});

    let compiled = graph
        .compile()
        .expect("Graph should compile and cull dead pass");
    let pass_names: Vec<&str> = compiled.passes().iter().map(|p| p.name()).collect();

    assert_eq!(
        pass_names,
        vec!["ActivePass"],
        "DeadPass must be pruned by dead-pass culling"
    );
}

#[test]
fn test_render_graph_barrier_synthesis() {
    let mut graph = RenderGraph::new();

    let tex = graph.create_transient_texture(TransientTextureDesc::new(
        "SceneColor",
        800,
        600,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    // Pass 1: Writes to tex as color attachment
    graph
        .add_pass("ColorPass", PassType::Graphics)
        .write_color(tex, 0, LoadOp::Clear, StoreOp::Store, [0.0, 0.0, 0.0, 1.0])
        .record(|_ctx, _cmd| {});

    // Pass 2: Reads tex in fragment shader
    graph
        .add_pass("PostPass", PassType::Graphics)
        .read_texture(
            tex,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    let compiled = graph
        .compile()
        .expect("Graph should compile and deduce barriers");
    let passes = compiled.passes();
    assert_eq!(passes.len(), 2);

    let post_pass = &passes[1];
    let barrier = post_pass.pre_pass_barrier();

    assert_eq!(
        barrier.image_barriers.len(),
        1,
        "PostPass must have 1 image barrier transitioning SceneColor"
    );
    let img_barrier = &barrier.image_barriers[0];
    assert_eq!(
        img_barrier.old_layout,
        ImageLayout::ColorAttachment,
        "Old layout must be ColorAttachment from Pass 1"
    );
    assert_eq!(
        img_barrier.new_layout,
        ImageLayout::ShaderReadOnly,
        "New layout must be ShaderReadOnly for Pass 2"
    );
    assert_eq!(
        img_barrier.src_stages,
        PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(img_barrier.dst_stages, PipelineStageFlags::FRAGMENT_SHADER);
    assert_eq!(img_barrier.src_access, AccessFlags::COLOR_ATTACHMENT_WRITE);
    assert_eq!(img_barrier.dst_access, AccessFlags::SHADER_READ);
}

#[test]
fn test_render_graph_transient_resource_aliasing() {
    let mut graph = RenderGraph::new();

    let desc = TransientTextureDesc::new(
        "Target",
        1920,
        1080,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    );

    let t1 = graph.create_transient_texture(desc.clone());
    let t2 = graph.create_transient_texture(desc);

    // Pass 1 uses T1 exclusively
    graph
        .add_pass("Pass1", PassType::Graphics)
        .write_color(t1, 0, LoadOp::Clear, StoreOp::Store, [1.0, 0.0, 0.0, 1.0])
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    // Pass 2 uses T2 exclusively (disjoint lifetime from T1)
    graph
        .add_pass("Pass2", PassType::Graphics)
        .write_color(t2, 0, LoadOp::Clear, StoreOp::Store, [0.0, 1.0, 0.0, 1.0])
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    let compiled = graph
        .compile()
        .expect("Graph should compile and alias disjoint textures");

    let t1_physical = compiled.resolve_texture(t1);
    let t2_physical = compiled.resolve_texture(t2);

    assert_eq!(
        t1_physical, t2_physical,
        "T1 and T2 must alias to the exact same physical texture handle because their lifetimes do not overlap"
    );
}

#[test]
fn test_render_graph_buffer_dependency_and_barrier() {
    let mut graph = RenderGraph::new();

    let buf_desc = TransientBufferDesc::new(
        "LightSSBO",
        4096,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
    );
    let buf = graph.create_transient_buffer(buf_desc);

    // Pass 1: Compute pass writes to buffer
    graph
        .add_pass("ComputeCull", PassType::Compute)
        .write_buffer(
            buf,
            BufferAccess {
                stage: PipelineStageFlags::COMPUTE_SHADER,
                access: AccessFlags::SHADER_WRITE,
            },
        )
        .record(|_ctx, _cmd| {});

    // Pass 2: Fragment shader reads buffer
    graph
        .add_pass("ForwardShade", PassType::Graphics)
        .read_buffer(
            buf,
            BufferAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
            },
        )
        .mark_as_root()
        .record(|_ctx, _cmd| {});

    let compiled = graph
        .compile()
        .expect("Graph should compile buffer dependency");
    assert_eq!(compiled.passes().len(), 2);

    let shade_pass = &compiled.passes()[1];
    let barrier = shade_pass.pre_pass_barrier();

    assert_eq!(
        barrier.buffer_barriers.len(),
        1,
        "ForwardShade must have 1 buffer barrier for LightSSBO"
    );
    let buf_barrier = &barrier.buffer_barriers[0];
    assert_eq!(buf_barrier.src_stages, PipelineStageFlags::COMPUTE_SHADER);
    assert_eq!(buf_barrier.dst_stages, PipelineStageFlags::FRAGMENT_SHADER);
    assert_eq!(buf_barrier.src_access, AccessFlags::SHADER_WRITE);
    assert_eq!(buf_barrier.dst_access, AccessFlags::SHADER_READ);
}

#[test]
fn test_g2_ping_pong_passes_compile_in_declaration_order() {
    let mut graph = RenderGraph::new();
    let desc = TransientTextureDesc::new(
        "PingPong",
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    );
    let t1 = graph.create_transient_texture(desc.clone());
    let t2 = graph.create_transient_texture(desc);
    let out = graph.import_texture(
        korppi_rhi::TextureHandle::new(8, 1),
        TextureAccess {
            stage: PipelineStageFlags::TOP_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );

    graph
        .add_pass("A", PassType::Graphics)
        .write_color(t1, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    graph
        .add_pass("B", PassType::Graphics)
        .read_color(t1)
        .write_color(t2, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    graph
        .add_pass("C", PassType::Graphics)
        .read_color(t2)
        .write_color(t1, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    graph
        .add_pass("D", PassType::Graphics)
        .read_color(t1)
        .write_color(out, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});

    let compiled = graph
        .compile()
        .expect("Ping-pong graph A -> B -> C -> D must compile without false cycle");
    let order: Vec<&str> = compiled.passes().iter().map(|p| p.name()).collect();
    assert_eq!(order, vec!["A", "B", "C", "D"]);
}

#[test]
fn test_g3_write_after_read_preserves_declaration_order() {
    let mut graph = RenderGraph::new();
    let src = graph.import_texture(
        korppi_rhi::TextureHandle::new(7, 1),
        TextureAccess {
            stage: PipelineStageFlags::FRAGMENT_SHADER,
            access: AccessFlags::SHADER_READ,
            layout: ImageLayout::ShaderReadOnly,
        },
        64,
        64,
    );
    let out = graph.import_texture(
        korppi_rhi::TextureHandle::new(8, 1),
        TextureAccess {
            stage: PipelineStageFlags::TOP_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );

    graph
        .add_pass("ReadPrevious", PassType::Graphics)
        .read_color(src)
        .write_color(out, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    graph
        .add_pass("Overwrite", PassType::Graphics)
        .write_color(src, 0, LoadOp::Clear, StoreOp::Store, [1.0; 4])
        .record(|_, _| {});

    let compiled = graph.compile().expect("WAR graph must compile");
    let order: Vec<&str> = compiled.passes().iter().map(|p| p.name()).collect();
    assert_eq!(
        order,
        vec!["ReadPrevious", "Overwrite"],
        "ReadPrevious must execute before Overwrite clobbers src"
    );
}

#[test]
fn test_g6_write_after_write_emits_barriers_for_color_and_buffers() {
    let mut graph = RenderGraph::new();
    let out = graph.import_texture(
        korppi_rhi::TextureHandle::new(8, 1),
        TextureAccess {
            stage: PipelineStageFlags::TOP_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );
    graph
        .add_pass("Opaque", PassType::Graphics)
        .write_color(out, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    graph
        .add_pass("Transparent", PassType::Graphics)
        .write_color(out, 0, LoadOp::Load, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});

    let compiled = graph.compile().expect("WAW color graph must compile");
    assert_eq!(compiled.passes().len(), 2);
    let transparent_barriers = &compiled.passes()[1].pre_pass_barrier().image_barriers;
    assert_eq!(
        transparent_barriers.len(),
        1,
        "Transparent pass loading color written by Opaque must have a barrier"
    );
    assert_eq!(
        transparent_barriers[0].src_access,
        AccessFlags::COLOR_ATTACHMENT_WRITE
    );
    assert_eq!(
        transparent_barriers[0].dst_access,
        AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE
    );

    // Compute buffer WAW + initial transient buffer barrier
    let mut g2 = RenderGraph::new();
    let buf =
        g2.create_transient_buffer(TransientBufferDesc::new("ssbo", 256, BufferUsage::STORAGE));
    let write = BufferAccess {
        stage: PipelineStageFlags::COMPUTE_SHADER,
        access: AccessFlags::SHADER_WRITE,
    };
    g2.add_pass("ComputeA", PassType::Compute)
        .write_buffer(buf, write)
        .record(|_, _| {});
    g2.add_pass("ComputeB", PassType::Compute)
        .write_buffer(buf, write)
        .mark_as_root()
        .record(|_, _| {});

    let compiled2 = g2.compile().expect("WAW buffer graph must compile");
    assert_eq!(compiled2.passes().len(), 2);
    assert_eq!(
        compiled2.passes()[0]
            .pre_pass_barrier()
            .buffer_barriers
            .len(),
        1,
        "First use of transient buffer must emit initial TOP_OF_PIPE barrier"
    );
    let b_wa_w = &compiled2.passes()[1].pre_pass_barrier().buffer_barriers;
    assert_eq!(
        b_wa_w.len(),
        1,
        "Second compute write to same buffer must emit WAW barrier"
    );
    assert_eq!(b_wa_w[0].src_access, AccessFlags::SHADER_WRITE);
    assert_eq!(b_wa_w[0].dst_access, AccessFlags::SHADER_WRITE);
}

#[test]
fn test_g7_depth_barrier_includes_early_and_late_fragment_tests_and_read_access() {
    let mut graph = RenderGraph::new();
    let depth = graph.create_transient_texture(TransientTextureDesc::new(
        "Depth",
        64,
        64,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let out = graph.import_texture(
        korppi_rhi::TextureHandle::new(8, 1),
        TextureAccess {
            stage: PipelineStageFlags::TOP_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Undefined,
        },
        64,
        64,
    );
    graph
        .add_pass("Prepass", PassType::Graphics)
        .write_depth(depth, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_, _| {});
    graph
        .add_pass("Forward", PassType::Graphics)
        .write_depth(depth, LoadOp::Load, StoreOp::Store, 0.0)
        .write_color(out, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});

    let compiled = graph.compile().expect("Depth graph must compile");
    let prepass_barrier = &compiled.passes()[0].pre_pass_barrier().image_barriers[0];
    assert_eq!(
        prepass_barrier.dst_stages,
        PipelineStageFlags::EARLY_FRAGMENT_TESTS | PipelineStageFlags::LATE_FRAGMENT_TESTS
    );
    assert_eq!(
        prepass_barrier.dst_access,
        AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
    );

    let forward_depth_barrier = compiled.passes()[1]
        .pre_pass_barrier()
        .image_barriers
        .iter()
        .find(|b| b.new_layout == ImageLayout::DepthStencilAttachment)
        .expect("Forward pass must have a depth barrier after Prepass");
    assert_eq!(
        forward_depth_barrier.src_stages,
        PipelineStageFlags::EARLY_FRAGMENT_TESTS | PipelineStageFlags::LATE_FRAGMENT_TESTS
    );
    assert_eq!(
        forward_depth_barrier.dst_stages,
        PipelineStageFlags::EARLY_FRAGMENT_TESTS | PipelineStageFlags::LATE_FRAGMENT_TESTS
    );
}

#[test]
fn test_rnd10_slot_ordering_msaa_resolve_subresources_and_read_only_depth() {
    let mut graph = RenderGraph::new();
    let mut msaa_desc = TransientTextureDesc::new(
        "MsaaHdr",
        128,
        128,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT,
    );
    msaa_desc.sample_count = korppi_rhi::SampleCount::Count4;
    let msaa_hdr = graph.create_transient_texture(msaa_desc);
    let resolved_hdr = graph.create_transient_texture(TransientTextureDesc::new(
        "ResolvedHdr",
        128,
        128,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let aux_target = graph.create_transient_texture(TransientTextureDesc::new(
        "AuxTarget",
        128,
        128,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let depth = graph.create_transient_texture(TransientTextureDesc::new(
        "Depth",
        128,
        128,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let bloom_mips = graph.create_transient_texture_ext(
        TransientTextureDesc::new(
            "BloomMips",
            64,
            64,
            TextureFormat::Rgba16Float,
            TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        ),
        3,
        1,
    );
    let final_out = graph.import_texture_with_final_access(
        korppi_rhi::TextureHandle::new(50, 1),
        TextureAccess {
            stage: PipelineStageFlags::TOP_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Undefined,
        },
        TextureAccess {
            stage: PipelineStageFlags::BOTTOM_OF_PIPE,
            access: AccessFlags::empty(),
            layout: ImageLayout::Present,
        },
        128,
        128,
    );

    // 1. Depth prepass
    graph
        .add_pass("DepthPrepass", PassType::Graphics)
        .write_depth(depth, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_, _| {});

    // 2. MSAA forward pass reading depth (read-only) and resolving msaa_hdr -> resolved_hdr
    graph
        .add_pass("MsaaForward", PassType::Graphics)
        .read_depth(depth)
        .write_color_msaa(
            msaa_hdr,
            resolved_hdr,
            0,
            LoadOp::Clear,
            StoreOp::DontCare,
            [0.0; 4],
        )
        .record(|_, _| {});

    // 3. Bloom mip 0 -> mip 1 using subresource attachments & ranges
    graph
        .add_pass("BloomMip0", PassType::Graphics)
        .read_color(resolved_hdr)
        .write_color_subresource(bloom_mips, 0, 0, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    graph
        .add_pass("BloomMip1", PassType::Graphics)
        .read_texture_subresource(
            bloom_mips,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
            korppi_rhi::TextureSubresourceRange::single(0, 0),
        )
        .write_color_subresource(bloom_mips, 0, 1, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});

    // 4. Composite pass with out-of-order attachment slots (slot 1 declared before slot 0)
    graph
        .add_pass("Composite", PassType::Graphics)
        .read_texture_subresource(
            bloom_mips,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
            korppi_rhi::TextureSubresourceRange::single(1, 0),
        )
        .write_color(aux_target, 1, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .write_color(final_out, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});

    let compiled = graph.compile().expect("Complex RND-10 graph must compile");
    assert_eq!(compiled.passes().len(), 5);

    // Verify MsaaForward has both read-only depth attachment and resolve_target
    let msaa_desc_out = compiled.passes()[1]
        .rendering_descriptor()
        .expect("MsaaForward must have rendering descriptor");
    assert_eq!(
        msaa_desc_out.color_attachments[0].resolve_target,
        Some(compiled.resolve_texture(resolved_hdr))
    );
    let depth_att = msaa_desc_out
        .depth_attachment
        .as_ref()
        .expect("MsaaForward must have read-only depth attachment");
    assert_eq!(depth_att.load_op, LoadOp::Load);
    assert_eq!(depth_att.store_op, StoreOp::Store);

    // Verify BloomMip1 render area was scaled to mip 1 (32x32)
    let mip1_desc = compiled.passes()[3]
        .rendering_descriptor()
        .expect("BloomMip1 rendering desc");
    assert_eq!((mip1_desc.width, mip1_desc.height), (32, 32));
    assert_eq!(mip1_desc.color_attachments[0].mip_level, 1);

    // Verify Composite attachments were sorted by slot (slot 0 = final_out, slot 1 = aux_target)
    let comp_desc = compiled.passes()[4]
        .rendering_descriptor()
        .expect("Composite rendering desc");
    assert_eq!(
        comp_desc.color_attachments[0].texture,
        compiled.resolve_texture(final_out)
    );
    assert_eq!(
        comp_desc.color_attachments[1].texture,
        compiled.resolve_texture(aux_target)
    );

    // Verify post-graph barrier transitions final_out to Present
    let post_barriers = &compiled.post_graph_barrier().image_barriers;
    assert_eq!(post_barriers.len(), 1);
    assert_eq!(post_barriers[0].old_layout, ImageLayout::ColorAttachment);
    assert_eq!(post_barriers[0].new_layout, ImageLayout::Present);

    // Verify RND-14 honest physical pool reporting
    assert_eq!(compiled.aliasing_report.physical_texture_count, 5);
}
