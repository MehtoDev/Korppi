//! Zero-allocation recording regression test from audit probe R12.
//!
//! `AGENTS.md` forbids heap allocation in frame loops. This binary installs a counting global
//! allocator that only counts allocations made by the measuring thread while counting is on, and
//! runs representative frames: a recorder, a barrier, a pass with 1 000 vertex-buffer draws, and
//! the submission. After a warm-up frame, frames must not allocate (RHI-11, RVK-15, RGL-18).

use korppi_rhi::{
    BufferDescriptor, BufferUsage, ColorTargetState, ColorWriteMask, CullMode, FrontFace,
    GraphicsPipelineDescriptor, LoadOp, MemoryUsage, PolygonMode, PrimitiveTopology,
    RasterizerState, ShaderStage, TextureFormat, TextureUsage, VertexAttribute, VertexBinding,
    VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
use korppi_rhi_conformance::{
    color_pass, color_target, undefined_to_color_attachment, Backend, TestDevice,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

/// Forwards to [`System`] and counts allocations made on threads that enabled counting.
struct CountingAllocator;

fn record_allocation() {
    // `try_with` because allocations can happen while thread-locals are being torn down.
    let counting = COUNTING.try_with(Cell::get).unwrap_or(false);
    if counting {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    }
}

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract. The counter only touches const-initialized `Cell` thread-locals, which
// never allocate, so counting cannot recurse into the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: forwarded unchanged; the caller upholds `alloc_zeroed`'s contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged; the caller upholds `dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Counts allocations made by the current thread while `f` runs.
fn count_allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    f();
    COUNTING.with(|c| c.set(false));
    ALLOCATIONS.with(Cell::get)
}

const POSITIONS: &str = r#"#version 450
layout(location = 0) in vec2 pos;
void main() { gl_Position = vec4(pos, 0.5, 1.0); }
"#;

const RED: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.0, 0.0, 1.0); }
"#;

fn r12_frame_recording_does_not_allocate(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, 64, 64, TextureUsage::empty());
    let vertex_shader = device.shader(ShaderStage::Vertex, POSITIONS);
    let fragment_shader = device.shader(ShaderStage::Fragment, RED);
    let pipeline = device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout: VertexLayoutDescriptor::new()
                .with_binding(VertexBinding {
                    binding: 0,
                    stride: 8,
                    step_mode: VertexStepMode::Vertex,
                })
                .with_attribute(VertexAttribute {
                    location: 0,
                    binding: 0,
                    format: VertexFormat::Float32x2,
                    offset: 0,
                }),
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::None,
                front_face: FrontFace::CounterClockwise,
                polygon_mode: PolygonMode::Fill,
                depth_bias: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_bias_clamp: 0.0,
                depth_clamp: false,
            },
            multisample: Default::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create_graphics_pipeline");
    let positions: Vec<u8> = [-1.0_f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();
    let vertex_buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: positions.len() as u64,
                usage: BufferUsage::VERTEX,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            Some(&positions),
        )
        .expect("create vertex buffer");
    let barrier = undefined_to_color_attachment(target);
    let pass = color_pass(target, 64, 64, LoadOp::Clear, [0.0, 0.0, 0.0, 1.0]);

    // Frame 0 warms up the recycled storage (recorders, command buffers, recording contexts,
    // command streams); later frames are steady state and allocate nothing, from creating the
    // recorder through submitting its commands.
    let mut allocations = Vec::new();
    for _ in 0..4 {
        let counted = count_allocations(|| {
            let mut recorder = device.create_command_recorder().expect("recorder");
            recorder.begin().expect("begin");
            recorder.pipeline_barrier(&barrier);
            recorder.begin_rendering(&pass).expect("begin_rendering");
            recorder.set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
            recorder.set_scissor(0, 0, 64, 64);
            recorder.bind_pipeline(pipeline);
            for _ in 0..1_000 {
                recorder.bind_vertex_buffers(0, &[vertex_buffer], &[0]);
                recorder.draw(3, 1, 0, 0);
            }
            recorder.end_rendering().expect("end_rendering");
            device
                .submit(recorder.finish().expect("finish"))
                .expect("submit");
        });
        device.wait_idle().expect("wait_idle");
        allocations.push(counted);
    }
    let allocations = allocations[1..].iter().max().copied().unwrap_or(0);
    assert_eq!(
        allocations, 0,
        "{backend:?}: heap allocations in a steady-state frame"
    );
}

#[test]
fn r12_frame_recording_does_not_allocate_vulkan() {
    r12_frame_recording_does_not_allocate(Backend::Vulkan);
}

#[test]
fn r12_frame_recording_does_not_allocate_opengl() {
    r12_frame_recording_does_not_allocate(Backend::OpenGl);
}
