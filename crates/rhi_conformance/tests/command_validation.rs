//! Command-recording validation (findings RHI-12, RHI-16; FIX-1.0, FIX-9.4).
//!
//! Recording misuse must make `finish()` return `RhiError::ValidationError` naming the problem,
//! on every backend and in every build: before FIX-9.4 the checks ran only when API validation
//! was requested, so a release build recorded copies past a buffer's end or without copy usage
//! and reported success (probes R18 to R20). Valid recording must still succeed.
//! `checks_hold_without_api_validation` runs this file's tests again with
//! `KORPPI_RHI_VALIDATION=0`, so that no check depends on the Khronos layer or GL debug output.

use korppi_rhi::{
    BufferDescriptor, BufferHandle, BufferUsage, ColorTargetState, ColorWriteMask, CommandBuffer,
    CommandBufferRecorder, CullMode, DepthStencilState, FrontFace, GraphicsPipelineDescriptor,
    LoadOp, MemoryUsage, PipelineHandle, PolygonMode, PrimitiveTopology, PushConstantRange,
    RasterizerState, RhiError, ShaderStage, TextureFormat, TextureUsage, VertexLayoutDescriptor,
    VALIDATION_ENV_VAR,
};
use korppi_rhi_conformance::{
    color_pass, color_target, undefined_to_color_attachment, Backend, TestDevice,
};

const SIZE: u32 = 16;

const FULLSCREEN: &str = r#"#version 450
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(P[gl_VertexIndex], 0.5, 1.0); }
"#;

const WHITE: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0); }
"#;

struct PipelineSetup {
    color_format: TextureFormat,
    depth: bool,
    push_constant_ranges: Vec<PushConstantRange>,
}

impl Default for PipelineSetup {
    fn default() -> Self {
        Self {
            color_format: TextureFormat::Rgba8Unorm,
            depth: false,
            push_constant_ranges: vec![],
        }
    }
}

fn pipeline(device: &TestDevice, setup: PipelineSetup) -> PipelineHandle {
    let vertex_shader = device.shader(ShaderStage::Vertex, FULLSCREEN);
    let fragment_shader = device.shader(ShaderStage::Fragment, WHITE);
    device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout: VertexLayoutDescriptor::new(),
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
            depth_stencil: setup.depth.then(DepthStencilState::default),
            color_targets: vec![ColorTargetState {
                format: setup.color_format,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: setup.push_constant_ranges,
            label: None,
        })
        .expect("create_graphics_pipeline")
}

/// Records `body` between `begin` and `finish`. Returns the command buffer, or the validation
/// message if `finish` rejected the recording. Misuse scenarios never submit what they recorded,
/// so a missed validation cannot make the driver execute invalid commands.
fn record(
    device: &TestDevice,
    body: impl FnOnce(&mut dyn CommandBufferRecorder),
) -> Result<Box<dyn CommandBuffer>, String> {
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    body(recorder.as_mut());
    match recorder.finish() {
        Ok(command_buffer) => Ok(command_buffer),
        Err(RhiError::ValidationError(message)) => Err(message),
        Err(other) => panic!("finish failed with a non-validation error: {other:?}"),
    }
}

/// Runs a misuse scenario and checks that validation rejects it with a message containing
/// `expected`.
fn expect_rejected(
    result: Result<Box<dyn CommandBuffer>, String>,
    expected: &str,
    backend: Backend,
) {
    match result {
        Ok(_) => {
            panic!("{backend:?}: misuse was accepted; expected an error mentioning {expected:?}")
        }
        Err(message) => assert!(
            message.contains(expected),
            "{backend:?}: validation message {message:?} does not mention {expected:?}"
        ),
    }
}

fn valid_frame_is_accepted(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let pipeline = pipeline(&device, PipelineSetup::default());
    let result = record(&device, |r| {
        r.pipeline_barrier(&undefined_to_color_attachment(target));
        r.begin_rendering(&color_pass(target, SIZE, SIZE, LoadOp::Clear, [0.0; 4]))
            .expect("begin_rendering");
        r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        r.set_scissor(0, 0, SIZE, SIZE);
        r.bind_pipeline(pipeline);
        r.draw(3, 1, 0, 0);
        r.end_rendering().expect("end_rendering");
    });
    let command_buffer = match result {
        Ok(command_buffer) => command_buffer,
        Err(message) => panic!("{backend:?}: a valid frame was rejected: {message}"),
    };
    device.submit(command_buffer).expect("submit");
    device.wait_idle().expect("wait_idle");
}

fn draw_outside_pass_is_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let pipeline = pipeline(&device, PipelineSetup::default());
    let result = record(&device, |r| {
        r.bind_pipeline(pipeline);
        r.draw(3, 1, 0, 0);
    });
    expect_rejected(result, "outside a render pass", backend);
}

fn pipeline_format_mismatch_is_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let pipeline = pipeline(
        &device,
        PipelineSetup {
            color_format: TextureFormat::Rgba16Float,
            ..PipelineSetup::default()
        },
    );
    let result = record(&device, |r| {
        r.pipeline_barrier(&undefined_to_color_attachment(target));
        r.begin_rendering(&color_pass(target, SIZE, SIZE, LoadOp::Clear, [0.0; 4]))
            .expect("begin_rendering");
        r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        r.set_scissor(0, 0, SIZE, SIZE);
        r.bind_pipeline(pipeline);
        r.draw(3, 1, 0, 0);
        r.end_rendering().expect("end_rendering");
    });
    expect_rejected(result, "do not match the pass attachments", backend);
}

fn depth_presence_mismatch_is_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let pipeline = pipeline(
        &device,
        PipelineSetup {
            depth: true,
            ..PipelineSetup::default()
        },
    );
    let result = record(&device, |r| {
        r.pipeline_barrier(&undefined_to_color_attachment(target));
        r.begin_rendering(&color_pass(target, SIZE, SIZE, LoadOp::Clear, [0.0; 4]))
            .expect("begin_rendering");
        r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        r.set_scissor(0, 0, SIZE, SIZE);
        r.bind_pipeline(pipeline);
        r.draw(3, 1, 0, 0);
        r.end_rendering().expect("end_rendering");
    });
    expect_rejected(result, "depth", backend);
}

fn push_constants_outside_declared_ranges_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let pipeline = pipeline(
        &device,
        PipelineSetup {
            push_constant_ranges: vec![PushConstantRange {
                stages: korppi_rhi::ShaderStages::VERTEX,
                offset: 0,
                size: 16,
            }],
            ..PipelineSetup::default()
        },
    );
    let result = record(&device, |r| {
        r.bind_pipeline(pipeline);
        r.push_constants(8, &[0_u8; 16])
            .expect("push_constants within the 128-byte budget");
    });
    expect_rejected(result, "push constant", backend);
}

fn unfinished_pass_is_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let result = record(&device, |r| {
        r.pipeline_barrier(&undefined_to_color_attachment(target));
        r.begin_rendering(&color_pass(target, SIZE, SIZE, LoadOp::Clear, [0.0; 4]))
            .expect("begin_rendering");
    });
    expect_rejected(result, "still active", backend);
}

fn buffer(device: &TestDevice, size: u64, usage: BufferUsage) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size,
                usage,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            None,
        )
        .expect("create_buffer")
}

/// R18: copies that reach past either buffer's end, by size or by overflowing offsets, and empty
/// copies are rejected (Vulkan leaves them undefined: VUID-vkCmdCopyBuffer-size-00115).
fn buffer_copies_outside_the_buffers_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let src = buffer(&device, 16, BufferUsage::COPY_SRC);
    let dst = buffer(&device, 16, BufferUsage::COPY_DST);
    let copy = |src_offset, dst_offset, size| {
        record(&device, |r| {
            r.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size)
        })
    };
    expect_rejected(
        copy(8, 0, 16),
        "past the end of the 16-byte source",
        backend,
    );
    expect_rejected(
        copy(0, 8, 16),
        "past the end of the 16-byte destination",
        backend,
    );
    expect_rejected(
        copy(u64::MAX, 0, 2),
        "past the end of the 16-byte source",
        backend,
    );
    expect_rejected(copy(0, 0, 0), "0 bytes", backend);
    assert!(
        copy(8, 8, 8).is_ok(),
        "{backend:?}: an in-bounds copy was rejected"
    );
}

/// A copy within one buffer whose source and destination overlap (VUID-vkCmdCopyBuffer-pRegions-00117;
/// GL_INVALID_VALUE in OpenGL).
fn overlapping_copies_within_a_buffer_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let both = buffer(&device, 64, BufferUsage::COPY_SRC | BufferUsage::COPY_DST);
    let result = record(&device, |r| r.copy_buffer_to_buffer(both, 0, both, 8, 16));
    expect_rejected(result, "overlap", backend);
    assert!(
        record(&device, |r| r.copy_buffer_to_buffer(both, 0, both, 16, 16)).is_ok(),
        "{backend:?}: adjacent ranges of one buffer were rejected"
    );
}

/// R20: a copy from a buffer without `COPY_SRC`, or into one without `COPY_DST`
/// (VUID-vkCmdCopyBuffer-srcBuffer-00118, -dstBuffer-00120).
fn buffer_copies_without_copy_usage_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let src = buffer(&device, 16, BufferUsage::COPY_SRC);
    let dst = buffer(&device, 16, BufferUsage::COPY_DST);
    let storage = buffer(&device, 16, BufferUsage::STORAGE);
    let result = record(&device, |r| r.copy_buffer_to_buffer(storage, 0, dst, 0, 16));
    expect_rejected(result, "lacks COPY_SRC", backend);
    let result = record(&device, |r| r.copy_buffer_to_buffer(src, 0, storage, 0, 16));
    expect_rejected(result, "lacks COPY_DST", backend);
}

/// R19: a copy into a destroyed buffer, whose slot a new buffer reuses, is rejected instead of
/// being dropped silently; so is a copy naming another device's buffer (RHI-15).
fn copies_naming_dead_or_foreign_buffers_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let src = buffer(&device, 16, BufferUsage::COPY_SRC);
    let dead = buffer(&device, 16, BufferUsage::COPY_DST);
    device.destroy_buffer(dead).expect("destroy_buffer");
    let reused = buffer(&device, 16, BufferUsage::COPY_DST);
    assert_eq!(
        reused.id(),
        dead.id(),
        "{backend:?}: precondition: the slot is reused"
    );
    let result = record(&device, |r| r.copy_buffer_to_buffer(src, 0, dead, 0, 16));
    expect_rejected(result, "unknown buffer handle", backend);

    let other = TestDevice::open(backend);
    let foreign = buffer(&other, 16, BufferUsage::COPY_DST);
    let result = record(&device, |r| r.copy_buffer_to_buffer(src, 0, foreign, 0, 16));
    expect_rejected(result, "unknown buffer handle", backend);
}

/// Runs every other test of this file again in a child process with API validation off, the
/// default of release builds.
#[test]
fn checks_hold_without_api_validation() {
    const NAME: &str = "checks_hold_without_api_validation";
    let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args(["--skip", NAME])
        .env(VALIDATION_ENV_VAR, "0")
        .output()
        .expect("run the tests again");
    assert!(
        output.status.success(),
        "with {VALIDATION_ENV_VAR}=0: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

macro_rules! per_backend {
    ($scenario:ident) => {
        mod $scenario {
            use super::*;

            #[test]
            fn vulkan() {
                $scenario(Backend::Vulkan);
            }

            #[test]
            fn opengl() {
                $scenario(Backend::OpenGl);
            }
        }
    };
}

per_backend!(valid_frame_is_accepted);
per_backend!(draw_outside_pass_is_rejected);
per_backend!(pipeline_format_mismatch_is_rejected);
per_backend!(depth_presence_mismatch_is_rejected);
per_backend!(push_constants_outside_declared_ranges_are_rejected);
per_backend!(unfinished_pass_is_rejected);
per_backend!(buffer_copies_outside_the_buffers_are_rejected);
per_backend!(overlapping_copies_within_a_buffer_are_rejected);
per_backend!(buffer_copies_without_copy_usage_are_rejected);
per_backend!(copies_naming_dead_or_foreign_buffers_are_rejected);
