//! Verification test for Physical GPU Memory Aliasing in Render Graph (AUDIT-4.6 / Finding 3.6).
//!
//! Research reference: `research/render/render_graph_and_frame_scheduling.md:88-140`.
//! Resources with non-overlapping lifetimes must alias overlapping physical memory offsets
//! in shared memory heaps, even when formats, dimensions, and usages differ!

use korppi_render::graph::{PassType, RenderGraph, TransientBufferDesc, TransientTextureDesc};
use korppi_rhi::{BufferUsage, LoadOp, StoreOp, TextureFormat, TextureUsage};

#[test]
fn test_physical_memory_aliasing_different_formats_and_dimensions() {
    let mut graph = RenderGraph::new();

    // Texture A: 1920x1080 RGBA16F (8 bytes/px ~ 16.59 MB), used in passes 0 and 1
    let tex_a = graph.create_transient_texture(TransientTextureDesc::new(
        "HDR_Color_A",
        1920,
        1080,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));

    // Texture B: 1280x720 RGBA8 (4 bytes/px ~ 3.69 MB), used in pass 1
    let tex_b = graph.create_transient_texture(TransientTextureDesc::new(
        "LDR_Color_B",
        1280,
        720,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    // Texture C: 1024x1024 Depth32F (4 bytes/px ~ 4.19 MB), used in pass 2
    let tex_c = graph.create_transient_texture(TransientTextureDesc::new(
        "Shadow_Depth_C",
        1024,
        1024,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ));

    // Pass 0: Generates A
    graph
        .add_pass("Pass0_Generate_A", PassType::Graphics)
        .write_color(tex_a, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .build();

    // Pass 1: Samples A, writes B
    graph
        .add_pass("Pass1_ToneMap_A_to_B", PassType::Graphics)
        .read_color(tex_a)
        .write_color(tex_b, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .build();

    // Pass 2: Texture A is now dead. Texture C is written.
    graph
        .add_pass("Pass2_Shadow_C", PassType::Graphics)
        .read_color(tex_b)
        .write_depth(tex_c, LoadOp::Clear, StoreOp::Store, 0.0)
        .mark_as_root()
        .build();

    let compiled = graph.compile().expect("Graph compilation failed");

    let alloc_a = compiled
        .texture_allocations
        .get(&tex_a)
        .expect("Texture A must have physical memory allocation");
    let alloc_c = compiled
        .texture_allocations
        .get(&tex_c)
        .expect("Texture C must have physical memory allocation");

    // Texture A and Texture C have disjoint lifetimes (Passes 0..1 vs Pass 2)!
    // Despite having different dimensions (1920x1080 vs 2048x2048) and formats (RGBA16F vs Depth32F),
    // they MUST share overlapping physical memory offsets!
    assert_eq!(
        alloc_a.offset, alloc_c.offset,
        "Texture A and C have disjoint lifetimes and must alias the same starting offset (0)"
    );

    let report = &compiled.aliasing_report;
    assert!(
        report.unaliased_bytes > report.aliased_bytes,
        "Aliased memory ({}) must be significantly less than unaliased memory ({})",
        report.aliased_bytes,
        report.unaliased_bytes
    );
    assert!(
        report.bytes_saved > 0,
        "Memory aliasing must save positive bytes"
    );
    assert!(
        report.savings_ratio > 0.0,
        "Savings ratio must be positive, got {}",
        report.savings_ratio
    );
}

#[test]
fn test_concurrent_transient_resources_allocate_disjoint_offsets() {
    let mut graph = RenderGraph::new();

    let tex_1 = graph.create_transient_texture(TransientTextureDesc::new(
        "Color1",
        1024,
        1024,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    let tex_2 = graph.create_transient_texture(TransientTextureDesc::new(
        "Color2",
        1024,
        1024,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT,
    ));

    // Both are written concurrently in Pass 0
    graph
        .add_pass("Pass0_MRT", PassType::Graphics)
        .write_color(tex_1, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .write_color(tex_2, 1, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .mark_as_root()
        .build();

    let compiled = graph.compile().expect("Compilation failed");

    let alloc_1 = compiled.texture_allocations.get(&tex_1).unwrap();
    let alloc_2 = compiled.texture_allocations.get(&tex_2).unwrap();

    // Since both are active at the same time, their byte intervals must NOT overlap!
    let overlap = alloc_1.offset < (alloc_2.offset + alloc_2.size)
        && alloc_2.offset < (alloc_1.offset + alloc_1.size);
    assert!(
        !overlap,
        "Concurrent resources must have disjoint physical memory ranges"
    );
}

#[test]
fn test_transient_buffer_physical_aliasing() {
    let mut graph = RenderGraph::new();

    let buf_a = graph.create_transient_buffer(TransientBufferDesc::new(
        "BufferA",
        1024 * 1024, // 1 MB
        BufferUsage::STORAGE,
    ));

    let buf_b = graph.create_transient_buffer(TransientBufferDesc::new(
        "BufferB",
        512 * 1024, // 512 KB
        BufferUsage::STORAGE,
    ));

    graph
        .add_pass("Pass0_Compute_A", PassType::Compute)
        .write_storage_buffer(buf_a)
        .build();

    graph
        .add_pass("Pass1_Read_A", PassType::Compute)
        .read_storage_buffer(buf_a)
        .mark_as_root()
        .build();

    graph
        .add_pass("Pass2_Compute_B", PassType::Compute)
        .write_storage_buffer(buf_b)
        .mark_as_root()
        .build();

    let compiled = graph.compile().expect("Compilation failed");

    let alloc_a = compiled.buffer_allocations.get(&buf_a).unwrap();
    let alloc_b = compiled.buffer_allocations.get(&buf_b).unwrap();

    // Buffer A and Buffer B have disjoint lifetimes (Passes 0..1 vs Pass 2)
    assert_eq!(
        alloc_a.offset, alloc_b.offset,
        "Buffers with disjoint lifetimes must share overlapping physical memory offset"
    );
    assert!(compiled.aliasing_report.bytes_saved >= 512 * 1024);
}
