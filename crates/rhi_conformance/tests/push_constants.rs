//! Push constants must reach the shader stages that declared them (findings RVK-10, RGL-04).
//!
//! Vulkan shaders read a `push_constant` block. OpenGL has no push constants; the backend
//! emulates them with a 128-byte `std140` uniform block at binding
//! `korppi_rhi_opengl::PUSH_CONSTANT_BINDING` (64).

use korppi_rhi::{
    ColorTargetState, ColorWriteMask, CullMode, GraphicsPipelineDescriptor, LoadOp,
    PrimitiveTopology, PushConstantRange, RasterizerState, ShaderStage, TextureFormat,
    TextureUsage, VertexLayoutDescriptor,
};
use korppi_rhi_conformance::{
    approx_rgba, color_pass, color_target, undefined_to_color_attachment, Backend, Image,
    TestDevice,
};

const SIZE: u32 = 64;

/// Full-screen triangle, no push constants.
const FULLSCREEN: &str = r#"#version 450
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(P[gl_VertexIndex], 0.5, 1.0); }
"#;

/// Fragment colour from push-constant bytes 0..16.
const PUSHED_COLOR_AT_0: &str = r#"#version 450
layout(push_constant) uniform PC { vec4 color; } pc;
layout(location = 0) out vec4 o;
void main() { o = pc.color; }
"#;

/// Triangle confined to the left half, shifted by push-constant bytes 0..16 (`offset.xy`).
const PUSHED_OFFSET_AT_0: &str = r#"#version 450
layout(push_constant) uniform PC { vec4 offset; } pc;
const vec2 P[3] = vec2[3](vec2(-0.9, -0.5), vec2(-0.1, -0.5), vec2(-0.5, 0.5));
void main() { gl_Position = vec4(P[gl_VertexIndex] + pc.offset.xy, 0.5, 1.0); }
"#;

/// Vertex stage of the disjoint-ranges case: Vulkan declares only its own range (bytes 0..16);
/// OpenGL must declare the whole block identically in every stage of the program.
const PUSHED_OFFSET_DISJOINT: &str = r#"#version 450
layout(push_constant) uniform PC { vec4 offset; } pc;
const vec2 P[3] = vec2[3](vec2(-0.9, -0.5), vec2(-0.1, -0.5), vec2(-0.5, 0.5));
void main() { gl_Position = vec4(P[gl_VertexIndex] + pc.offset.xy, 0.5, 1.0); }
"#;

/// Fragment colour from push-constant bytes 16..32.
const PUSHED_COLOR_AT_16: &str = r#"#version 450
layout(push_constant) uniform PC { layout(offset = 16) vec4 color; } pc;
layout(location = 0) out vec4 o;
void main() { o = pc.color; }
"#;

/// Fragment colour from bytes 16..32 of a block the vertex stage also reads (`offset` at 0).
const SHARED_BLOCK_COLOR: &str = r#"#version 450
layout(push_constant) uniform PC { vec4 offset; vec4 color; } pc;
layout(location = 0) out vec4 o;
void main() { o = pc.color; }
"#;

/// Vertex stage of the shared block: reads `offset` at 0 and declares `color` at 16.
const SHARED_BLOCK_OFFSET: &str = r#"#version 450
layout(push_constant) uniform PC { vec4 offset; vec4 color; } pc;
const vec2 P[3] = vec2[3](vec2(-0.9, -0.5), vec2(-0.1, -0.5), vec2(-0.5, 0.5));
void main() { gl_Position = vec4(P[gl_VertexIndex] + pc.offset.xy, 0.5, 1.0); }
"#;

/// Fixed colour, used where only the vertex stage reads push constants.
const SOLID_WHITE: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0); }
"#;

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

/// Draws one triangle with the given stages and push-constant ranges after pushing `bytes` at
/// offset 0, over a target cleared to opaque black.
fn render(
    backend: Backend,
    vertex: &'static str,
    fragment: &'static str,
    ranges: Vec<PushConstantRange>,
    bytes: &[u8],
) -> Image {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let vertex_shader = device.shader(ShaderStage::Vertex, vertex);
    let fragment_shader = device.shader(ShaderStage::Fragment, fragment);
    let pipeline = device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout: VertexLayoutDescriptor::new(),
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::None,
                ..RasterizerState::default()
            },
            multisample: Default::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: ranges,
            label: None,
        })
        .expect("create_graphics_pipeline");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    recorder
        .begin_rendering(&color_pass(
            target,
            SIZE,
            SIZE,
            LoadOp::Clear,
            [0.0, 0.0, 0.0, 1.0],
        ))
        .expect("begin_rendering");
    recorder.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
    recorder.set_scissor(0, 0, SIZE, SIZE);
    recorder.bind_pipeline(pipeline);
    recorder.push_constants(0, bytes).expect("push_constants");
    recorder.draw(3, 1, 0, 0);
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    Image::read(&*device, target, SIZE, SIZE)
}

fn range(stage: ShaderStage, offset: u32, size: u32) -> PushConstantRange {
    PushConstantRange {
        stages: match stage {
            ShaderStage::Vertex => korppi_rhi::ShaderStages::VERTEX,
            ShaderStage::Fragment => korppi_rhi::ShaderStages::FRAGMENT,
            ShaderStage::Compute => korppi_rhi::ShaderStages::COMPUTE,
        },
        offset,
        size,
    }
}

/// Texels drawn (not black) in the left and right halves.
fn drawn_per_half(image: &Image) -> (usize, usize) {
    let (mut left, mut right) = (0, 0);
    for y in 0..SIZE {
        for x in 0..SIZE {
            if image.pixel(x, y)[..3] != [0, 0, 0] {
                if x < SIZE / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }
    (left, right)
}

fn fragment_push_constants_reach_the_shader(backend: Backend) {
    let image = render(
        backend,
        FULLSCREEN,
        PUSHED_COLOR_AT_0,
        vec![range(ShaderStage::Fragment, 0, 16)],
        &floats(&[0.25, 0.5, 0.75, 1.0]),
    );
    let texel = image.pixel(SIZE / 2, SIZE / 2);
    assert!(
        approx_rgba(texel, [64, 128, 191, 255], 1),
        "{backend:?}: texel {texel:?}, expected ~[64, 128, 191, 255]"
    );
}

#[test]
fn fragment_push_constants_reach_the_shader_vulkan() {
    fragment_push_constants_reach_the_shader(Backend::Vulkan);
}

#[test]
fn fragment_push_constants_reach_the_shader_opengl() {
    fragment_push_constants_reach_the_shader(Backend::OpenGl);
}

fn vertex_push_constants_reach_the_shader(backend: Backend) {
    let image = render(
        backend,
        PUSHED_OFFSET_AT_0,
        SOLID_WHITE,
        vec![range(ShaderStage::Vertex, 0, 16)],
        &floats(&[1.0, 0.0, 0.0, 0.0]),
    );
    let (left, right) = drawn_per_half(&image);
    assert!(
        left == 0 && right > 100,
        "{backend:?}: drawn left {left}, right {right}"
    );
}

#[test]
fn vertex_push_constants_reach_the_shader_vulkan() {
    vertex_push_constants_reach_the_shader(Backend::Vulkan);
}

#[test]
fn vertex_push_constants_reach_the_shader_opengl() {
    vertex_push_constants_reach_the_shader(Backend::OpenGl);
}

fn shared_range_reaches_both_stages(backend: Backend) {
    let image = render(
        backend,
        SHARED_BLOCK_OFFSET,
        SHARED_BLOCK_COLOR,
        vec![
            range(ShaderStage::Vertex, 0, 32),
            range(ShaderStage::Fragment, 0, 32),
        ],
        &floats(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
    );
    let (left, right) = drawn_per_half(&image);
    assert!(
        left == 0 && right > 100,
        "{backend:?}: drawn left {left}, right {right}"
    );
    let green = image.count(|p| p == [0, 255, 0, 255]);
    assert_eq!(green, right, "{backend:?}: every drawn texel must be green");
}

#[test]
fn shared_range_reaches_both_stages_vulkan() {
    shared_range_reaches_both_stages(Backend::Vulkan);
}

#[test]
fn shared_range_reaches_both_stages_opengl() {
    shared_range_reaches_both_stages(Backend::OpenGl);
}

fn disjoint_stage_ranges_in_one_push(backend: Backend) {
    let image = render(
        backend,
        PUSHED_OFFSET_DISJOINT,
        PUSHED_COLOR_AT_16,
        vec![
            range(ShaderStage::Vertex, 0, 16),
            range(ShaderStage::Fragment, 16, 16),
        ],
        &floats(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
    );
    let (left, right) = drawn_per_half(&image);
    assert!(
        left == 0 && right > 100,
        "{backend:?}: drawn left {left}, right {right}"
    );
    let blue = image.count(|p| p == [0, 0, 255, 255]);
    assert_eq!(blue, right, "{backend:?}: every drawn texel must be blue");
}

#[test]
fn disjoint_stage_ranges_in_one_push_vulkan() {
    disjoint_stage_ranges_in_one_push(Backend::Vulkan);
}

#[test]
fn disjoint_stage_ranges_in_one_push_opengl() {
    disjoint_stage_ranges_in_one_push(Backend::OpenGl);
}

/// Push constants and a uniform buffer in one shader both arrive: on OpenGL the uniform buffer
/// gets a binding point of its own, below the one push constants are emulated at.
#[test]
fn push_constants_and_uniform_buffers_do_not_collide() {
    use korppi_rhi::{
        AccessFlags, BufferDescriptor, BufferMemoryBarrier, BufferUsage, ComputePipelineDescriptor,
        MemoryUsage, PipelineBarrier, PipelineStageFlags, ShaderStages,
    };
    use korppi_rhi_conformance::{buffer_range, STORAGE, UNIFORM};
    const SHADER: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(push_constant) uniform Push { vec4 p; };
layout(set = 0, binding = 0) uniform U { vec4 u; };
layout(set = 0, binding = 1) buffer O { vec4 o; };
void main() { o = p + u * 10.0; }
"#;
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let bytes = |v: [f32; 4]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
        let ubo = device
            .create_buffer(
                &BufferDescriptor {
                    size: 16,
                    usage: BufferUsage::UNIFORM,
                    memory: MemoryUsage::CpuToGpu,
                    label: None,
                },
                Some(&bytes([1.0, 2.0, 3.0, 4.0])),
            )
            .expect("uniform buffer");
        let output = device
            .create_buffer(
                &BufferDescriptor {
                    size: 16,
                    usage: BufferUsage::STORAGE,
                    memory: MemoryUsage::GpuToCpu,
                    label: None,
                },
                None,
            )
            .expect("output buffer");
        let (layout, group) = device.bind_group(&[
            (0, UNIFORM, ShaderStages::COMPUTE, buffer_range(ubo, 0, 16)),
            (
                1,
                STORAGE,
                ShaderStages::COMPUTE,
                buffer_range(output, 0, 16),
            ),
        ]);
        let pipeline = device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                compute_shader: device.shader(ShaderStage::Compute, SHADER),
                bind_group_layouts: vec![layout],
                push_constant_ranges: vec![PushConstantRange {
                    stages: korppi_rhi::ShaderStages::COMPUTE,
                    offset: 0,
                    size: 16,
                }],
                label: None,
            })
            .expect("compute pipeline");
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.bind_pipeline(pipeline);
        recorder.set_bind_group(0, group, &[]);
        recorder
            .push_constants(0, &bytes([0.5, 0.25, 0.125, 0.0]))
            .expect("push_constants");
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&PipelineBarrier::new().with_buffer_barrier(
            BufferMemoryBarrier {
                buffer: output,
                src_stages: PipelineStageFlags::COMPUTE_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_stages: PipelineStageFlags::HOST,
                dst_access: AccessFlags::HOST_READ,
                offset: 0,
                size: 16,
            },
        ));
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");
        let result = device.read_buffer(output, 0, 16).expect("read_buffer");
        assert_eq!(
            result,
            bytes([10.5, 20.25, 30.125, 40.0]),
            "{backend:?}: push constants + 10 * uniform"
        );
    }
}
