//! Initial data passed to `create_buffer` / `create_texture` (findings RGL-10, RVK-08).
//!
//! Contract: buffer `initial_data` may be shorter than the buffer (the rest is unspecified) but
//! not longer; texture `initial_data` must be exactly one tightly packed mip-0 level. Device-local
//! (`GpuOnly`) buffers receive their initial data on every backend, mapped or not.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferMemoryBarrier, BufferUsage, ColorTargetState,
    ColorWriteMask, CullMode, GraphicsPipelineDescriptor, LoadOp, MemoryUsage, PipelineBarrier,
    PipelineStageFlags, PrimitiveTopology, RasterizerState, RhiError, SampleCount, ShaderStage,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsage, VertexAttribute,
    VertexBinding, VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
use korppi_rhi_conformance::{
    color_pass, color_target, undefined_to_color_attachment, Backend, Image, TestDevice,
};

fn buffer(size: u64, memory: MemoryUsage) -> BufferDescriptor {
    BufferDescriptor {
        size,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_SRC,
        memory,
        label: None,
    }
}

fn rgba8_texture(size: u32) -> TextureDescriptor {
    TextureDescriptor {
        width: size,
        height: size,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        dimension: TextureDimension::D2,
        label: None,
    }
}

fn buffer_initial_data_longer_than_the_buffer_is_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let result = device.create_buffer(&buffer(16, MemoryUsage::CpuToGpu), Some(&[1_u8; 32]));
    assert!(
        matches!(result, Err(RhiError::ValidationError(_))),
        "{backend:?}: expected ValidationError, got {result:?}"
    );
}

#[test]
fn buffer_initial_data_longer_than_the_buffer_is_rejected_vulkan() {
    buffer_initial_data_longer_than_the_buffer_is_rejected(Backend::Vulkan);
}

#[test]
fn buffer_initial_data_longer_than_the_buffer_is_rejected_opengl() {
    buffer_initial_data_longer_than_the_buffer_is_rejected(Backend::OpenGl);
}

fn shorter_buffer_initial_data_fills_the_start(backend: Backend) {
    let device = TestDevice::open(backend);
    let data: Vec<u8> = (1..=16).collect();
    let handle = device
        .create_buffer(&buffer(64, MemoryUsage::CpuToGpu), Some(&data))
        .expect("create buffer");
    assert_eq!(
        device.read_buffer(handle, 0, 16).expect("read"),
        data,
        "{backend:?}"
    );
}

#[test]
fn shorter_buffer_initial_data_fills_the_start_vulkan() {
    shorter_buffer_initial_data_fills_the_start(Backend::Vulkan);
}

#[test]
fn shorter_buffer_initial_data_fills_the_start_opengl() {
    shorter_buffer_initial_data_fills_the_start(Backend::OpenGl);
}

fn texture_initial_data_must_be_one_full_level(backend: Backend) {
    let device = TestDevice::open(backend);
    for len in [8_usize, 128] {
        let result = device.create_texture(&rgba8_texture(4), Some(&vec![0_u8; len]));
        assert!(
            matches!(result, Err(RhiError::ValidationError(_))),
            "{backend:?}: {len} bytes for a 4x4 RGBA8 level (64 bytes): got {result:?}"
        );
    }
    device
        .create_texture(&rgba8_texture(4), Some(&[0_u8; 64]))
        .expect("exactly one level is accepted");
}

#[test]
fn texture_initial_data_must_be_one_full_level_vulkan() {
    texture_initial_data_must_be_one_full_level(Backend::Vulkan);
}

#[test]
fn texture_initial_data_must_be_one_full_level_opengl() {
    texture_initial_data_must_be_one_full_level(Backend::OpenGl);
}

/// OpenGL sizes and offsets are 32-bit signed in the backend; larger buffers must be refused
/// instead of being truncated (no memory is allocated by this test).
#[test]
fn opengl_rejects_buffers_of_2_gib_or_more() {
    let device = TestDevice::open(Backend::OpenGl);
    let result = device.create_buffer(&buffer(3 << 30, MemoryUsage::GpuOnly), None);
    assert!(
        matches!(result, Err(RhiError::Unsupported(_))),
        "expected Unsupported, got {result:?}"
    );
}

const POSITIONS: &str = r#"#version 450
layout(location = 0) in vec2 pos;
void main() { gl_Position = vec4(pos, 0.5, 1.0); }
"#;

const WHITE: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0); }
"#;

fn gpu_only_vertex_buffers_receive_initial_data(backend: Backend) {
    const SIZE: u32 = 32;
    let device = TestDevice::open(backend);
    let positions: Vec<u8> = [-1.0_f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();
    let vertices = device
        .create_buffer(
            &buffer(positions.len() as u64, MemoryUsage::GpuOnly),
            Some(&positions),
        )
        .expect("create GpuOnly vertex buffer");
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let vertex_shader = device.shader(ShaderStage::Vertex, POSITIONS);
    let fragment_shader = device.shader(ShaderStage::Fragment, WHITE);
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
            push_constant_ranges: vec![],
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
    recorder.bind_vertex_buffers(0, &[vertices], &[0]);
    recorder.draw(3, 1, 0, 0);
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let image = Image::read(&*device, target, SIZE, SIZE);
    assert_eq!(
        image.count(|p| p == [255, 255, 255, 255]),
        (SIZE * SIZE) as usize,
        "{backend:?}: a full-screen triangle from a GpuOnly buffer must cover the target"
    );
}

#[test]
fn gpu_only_vertex_buffers_receive_initial_data_vulkan() {
    gpu_only_vertex_buffers_receive_initial_data(Backend::Vulkan);
}

#[test]
fn gpu_only_vertex_buffers_receive_initial_data_opengl() {
    gpu_only_vertex_buffers_receive_initial_data(Backend::OpenGl);
}

/// `write_buffer` into device-local memory arrives, checked through a GPU copy so the test does
/// not depend on `GpuOnly` memory being mappable.
fn gpu_only_buffers_accept_write_buffer(backend: Backend) {
    let device = TestDevice::open(backend);
    let handle = device
        .create_buffer(&buffer(16, MemoryUsage::GpuOnly), Some(&[0_u8; 16]))
        .expect("create buffer");
    let written: Vec<u8> = (100..108).collect();
    device
        .write_buffer(handle, 4, &written)
        .unwrap_or_else(|e| panic!("{backend:?}: write_buffer to GpuOnly: {e:?}"));
    let readback = device
        .create_buffer(
            &BufferDescriptor {
                size: 16,
                usage: BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            None,
        )
        .expect("create readback buffer");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.copy_buffer_to_buffer(handle, 0, readback, 0, 16);
    recorder.pipeline_barrier(
        &PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
            buffer: readback,
            src_stages: PipelineStageFlags::TRANSFER,
            src_access: AccessFlags::TRANSFER_WRITE,
            dst_stages: PipelineStageFlags::HOST,
            dst_access: AccessFlags::HOST_READ,
            offset: 0,
            size: 16,
        }),
    );
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let bytes = device.read_buffer(readback, 0, 16).expect("read_buffer");
    assert_eq!(&bytes[4..12], &written[..], "{backend:?}: written range");
    assert_eq!(&bytes[..4], &[0; 4], "{backend:?}: bytes before the write");
    assert_eq!(&bytes[12..], &[0; 4], "{backend:?}: bytes after the write");
}

#[test]
fn gpu_only_buffers_accept_write_buffer_vulkan() {
    gpu_only_buffers_accept_write_buffer(Backend::Vulkan);
}

#[test]
fn gpu_only_buffers_accept_write_buffer_opengl() {
    gpu_only_buffers_accept_write_buffer(Backend::OpenGl);
}

/// Zero-sized buffers are rejected: Vulkan forbids them (VUID-VkBufferCreateInfo-size-00912).
fn zero_sized_buffers_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    for memory in [MemoryUsage::GpuOnly, MemoryUsage::CpuToGpu] {
        let result = device.create_buffer(&buffer(0, memory), None);
        assert!(
            matches!(result, Err(RhiError::ValidationError(_))),
            "{backend:?} {memory:?}: {result:?}"
        );
    }
}

#[test]
fn zero_sized_buffers_are_rejected_vulkan() {
    zero_sized_buffers_are_rejected(Backend::Vulkan);
}

#[test]
fn zero_sized_buffers_are_rejected_opengl() {
    zero_sized_buffers_are_rejected(Backend::OpenGl);
}
