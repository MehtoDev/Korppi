//! Memory barriers between dependent GPU work (finding RGL-16: OpenGL derives its
//! `glMemoryBarrier` bits from the RHI access masks instead of issuing `GL_ALL_BARRIER_BITS`).

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferHandle, BufferMemoryBarrier, BufferUsage,
    ComputePipelineDescriptor, MemoryUsage, PipelineBarrier, PipelineHandle, PipelineStageFlags,
    ShaderStage,
};
use korppi_rhi::{BindGroupLayoutHandle, ShaderStages};
use korppi_rhi_conformance::{buffer_range, STORAGE};
use korppi_rhi_conformance::{Backend, TestDevice};

const COUNT: u32 = 256;

/// Writes `index * 3 + 1` to every element of the storage buffer at binding 0.
const PRODUCE: &str = r#"#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer Out { uint values[]; };
void main() { values[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * 3u + 1u; }
"#;

/// Reads the storage buffer at binding 0, writes twice each value to the one at binding 1.
const CONSUME: &str = r#"#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer In { uint inputs[]; };
layout(set = 0, binding = 1) buffer Out { uint outputs[]; };
void main() { outputs[gl_GlobalInvocationID.x] = inputs[gl_GlobalInvocationID.x] * 2u; }
"#;

fn compute_pipeline(
    device: &TestDevice,
    sources: &str,
    layout: BindGroupLayoutHandle,
) -> PipelineHandle {
    let compute_shader = device.shader(ShaderStage::Compute, sources);
    device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create compute pipeline")
}

fn storage_buffer(device: &TestDevice) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size: u64::from(COUNT) * 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            None,
        )
        .expect("create storage buffer")
}

fn barrier(
    buffer: BufferHandle,
    dst_stages: PipelineStageFlags,
    dst_access: AccessFlags,
) -> PipelineBarrier {
    PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
        buffer,
        src_stages: PipelineStageFlags::COMPUTE_SHADER,
        src_access: AccessFlags::SHADER_WRITE,
        dst_stages,
        dst_access,
        offset: 0,
        size: u64::from(COUNT) * 4,
    })
}

fn dependent_dispatches_see_earlier_writes(backend: Backend) {
    let device = TestDevice::open(backend);
    let intermediate = storage_buffer(&device);
    let output = storage_buffer(&device);
    let size = u64::from(COUNT) * 4;
    // One layout for both pipelines; the producer only uses binding 0.
    let (layout, group) = device.bind_group(&[
        (
            0,
            STORAGE,
            ShaderStages::COMPUTE,
            buffer_range(intermediate, 0, size),
        ),
        (
            1,
            STORAGE,
            ShaderStages::COMPUTE,
            buffer_range(output, 0, size),
        ),
    ]);
    let produce = compute_pipeline(&device, PRODUCE, layout);
    let consume = compute_pipeline(&device, CONSUME, layout);

    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.bind_pipeline(produce);
    recorder.set_bind_group(0, group, &[]);
    recorder.dispatch(COUNT / 64, 1, 1);
    recorder.pipeline_barrier(&barrier(
        intermediate,
        PipelineStageFlags::COMPUTE_SHADER,
        AccessFlags::SHADER_READ,
    ));
    recorder.bind_pipeline(consume);
    recorder.dispatch(COUNT / 64, 1, 1);
    recorder.pipeline_barrier(&barrier(
        output,
        PipelineStageFlags::HOST,
        AccessFlags::HOST_READ,
    ));
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");

    let bytes = device.read_buffer(output, 0, size).expect("read_buffer");
    for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let index = index as u32;
        assert_eq!(
            u32::from_ne_bytes(*chunk),
            (index * 3 + 1) * 2,
            "{backend:?}: element {index}"
        );
    }
}

#[test]
fn dependent_dispatches_see_earlier_writes_vulkan() {
    dependent_dispatches_see_earlier_writes(Backend::Vulkan);
}

#[test]
fn dependent_dispatches_see_earlier_writes_opengl() {
    dependent_dispatches_see_earlier_writes(Backend::OpenGl);
}
