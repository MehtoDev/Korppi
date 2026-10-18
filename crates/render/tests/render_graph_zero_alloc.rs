//! Regression test for RND-11 / Probe G8 (Rule 4 / ISSUE-005 item 1):
//! Zero heap allocations when building, compiling, and executing a steady-state 5-pass render graph.

use korppi_render::{
    BufferAccess, CompiledRenderGraph, PassType, RenderGraph, TextureAccess, TransientBufferDesc,
    TransientResourcePool, TransientTextureDesc,
};
use korppi_rhi::{
    AccessFlags, BufferUsage, GraphicsDevice, ImageLayout, LoadOp, PipelineStageFlags, SampleCount,
    StoreOp, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAlloc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards directly to the standard `System` allocator while incrementing an atomic counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: caller upholds `GlobalAlloc::alloc` preconditions.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller upholds `GlobalAlloc::dealloc` preconditions.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

fn allocations_in<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let before = ALLOC_COUNT.load(Ordering::SeqCst);
    let out = f();
    let after = ALLOC_COUNT.load(Ordering::SeqCst);
    (after.saturating_sub(before), out)
}

fn build_five_pass_frame(
    device: &dyn GraphicsDevice,
    pool: &mut TransientResourcePool,
    out_tex: korppi_rhi::TextureHandle,
) -> CompiledRenderGraph {
    let mut g = RenderGraph::new();
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
    let depth = g.create_transient_texture(TransientTextureDesc::new(
        "Depth",
        64,
        64,
        TextureFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let hdr = g.create_transient_texture(TransientTextureDesc::new(
        "Hdr",
        64,
        64,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let bloom = g.create_transient_texture(TransientTextureDesc::new(
        "Bloom",
        32,
        32,
        TextureFormat::Rgba16Float,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
    ));
    let lights = g.create_transient_buffer(TransientBufferDesc::new(
        "LightGrid",
        4096,
        BufferUsage::STORAGE,
    ));
    let compute_write = BufferAccess {
        stage: PipelineStageFlags::COMPUTE_SHADER,
        access: AccessFlags::SHADER_WRITE,
    };
    let fragment_read = BufferAccess {
        stage: PipelineStageFlags::FRAGMENT_SHADER,
        access: AccessFlags::SHADER_READ,
    };
    let shader_read = TextureAccess {
        stage: PipelineStageFlags::FRAGMENT_SHADER,
        access: AccessFlags::SHADER_READ,
        layout: ImageLayout::ShaderReadOnly,
    };
    g.add_pass("DepthPrepass", PassType::Graphics)
        .write_depth(depth, LoadOp::Clear, StoreOp::Store, 0.0)
        .record(|_, _| {});
    g.add_pass("LightCull", PassType::Compute)
        .read_texture(depth, shader_read)
        .write_buffer(lights, compute_write)
        .record(|_, _| {});
    g.add_pass("Forward", PassType::Graphics)
        .read_depth(depth)
        .read_buffer(lights, fragment_read)
        .write_color(hdr, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    g.add_pass("BloomDown", PassType::Graphics)
        .read_color(hdr)
        .write_color(bloom, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    g.add_pass("Tonemap", PassType::Graphics)
        .read_color(hdr)
        .read_color(bloom)
        .write_color(out, 0, LoadOp::Clear, StoreOp::Store, [0.0; 4])
        .record(|_, _| {});
    g.compile_with_device(device, pool)
        .expect("5-pass graph must compile")
}

fn check_zero_alloc_on_device(device: &dyn GraphicsDevice, backend_name: &str) {
    let out_desc = TextureDescriptor {
        width: 64,
        height: 64,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("g8_out"),
    };
    let out_tex = device.create_texture(&out_desc, None).expect("out_tex");
    let mut pool = TransientResourcePool::new();

    // Warm up 3 frames (including command recorder reuse)
    for _ in 0..3 {
        let mut compiled = build_five_pass_frame(device, &mut pool, out_tex);
        let mut rec = device.create_command_recorder().expect("recorder");
        rec.begin().expect("begin");
        compiled.execute(&mut *rec).expect("execute warmup");
        let cmd = rec.finish().expect("finish warmup");
        device.submit_and_wait(cmd).expect("submit_and_wait warmup");
    }
    let _ = device.take_diagnostics();

    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");

    let (build_compile_allocs, mut compiled) =
        allocations_in(|| build_five_pass_frame(device, &mut pool, out_tex));
    let (execute_allocs, exec_res) = allocations_in(|| compiled.execute(&mut *recorder));
    exec_res.expect("execute measured frame");
    drop(compiled);

    let cmd = recorder.finish().expect("finish");
    device.submit_and_wait(cmd).expect("submit_and_wait");

    let diags = device.take_diagnostics();
    assert!(
        diags.is_empty(),
        "{backend_name}: expected zero validation diagnostics on 5-pass graph, got: {diags:#?}"
    );
    assert_eq!(
        build_compile_allocs, 0,
        "{backend_name}: steady-state RenderGraph build + compile_with_device must perform 0 heap allocations"
    );
    assert_eq!(
        execute_allocs, 0,
        "{backend_name}: steady-state CompiledRenderGraph::execute must perform 0 heap allocations"
    );

    pool.destroy(device).expect("destroy pool");
    device.destroy_texture(out_tex).expect("destroy out_tex");
}

#[test]
fn test_g8_zero_allocations_per_frame_opengl_and_vulkan() {
    let gl_device = OpenGlDevice::create_headless().expect("OpenGL headless");
    check_zero_alloc_on_device(&gl_device, "OpenGL");

    if let Ok(vk_device) = VulkanDevice::create_headless() {
        check_zero_alloc_on_device(&vk_device, "Vulkan");
    }
}
