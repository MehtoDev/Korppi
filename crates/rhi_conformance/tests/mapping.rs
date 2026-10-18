//! Buffer mapping (FIX-1.4; finding RHI-05): persistent host access to `CpuToGpu` and
//! `GpuToCpu` buffers, exclusive per buffer, streaming per frame in flight, and `GpuOnly`
//! buffers read and written through staging.

use korppi_rhi::{
    AccessFlags, BindingType, BufferDescriptor, BufferHandle, BufferMemoryBarrier, BufferUsage,
    ComputePipelineDescriptor, MemoryUsage, PipelineBarrier, PipelineStageFlags, RhiError,
    ShaderStage, ShaderStages, FRAMES_IN_FLIGHT,
};
use korppi_rhi_conformance::{buffer_range, Backend, TestDevice};

/// Copies `value` (uniform, dynamic offset) into `result` (storage, dynamic offset), doubled.
const DOUBLE: &str = r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform Params { vec4 value; };
layout(set = 0, binding = 1) buffer Out { vec4 result; };
void main() { result = value * 2.0; }
";

fn buffer(device: &TestDevice, size: u64, usage: BufferUsage, memory: MemoryUsage) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size,
                usage,
                memory,
                label: None,
            },
            None,
        )
        .expect("create_buffer")
}

fn vec4(values: [f32; 4]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

fn to_host(buffer: BufferHandle, size: u64) -> PipelineBarrier {
    PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
        buffer,
        src_stages: PipelineStageFlags::COMPUTE_SHADER,
        src_access: AccessFlags::SHADER_WRITE,
        dst_stages: PipelineStageFlags::HOST,
        dst_access: AccessFlags::HOST_READ,
        offset: 0,
        size,
    })
}

/// A pipeline running [`DOUBLE`] and its bind group over `input` and `output`, both with
/// dynamic offsets.
fn doubling(
    device: &TestDevice,
    input: BufferHandle,
    output: BufferHandle,
) -> (korppi_rhi::PipelineHandle, korppi_rhi::BindGroupHandle) {
    let (layout, group) = device.bind_group(&[
        (
            0,
            BindingType::UniformBuffer {
                dynamic_offset: true,
            },
            ShaderStages::COMPUTE,
            buffer_range(input, 0, 16),
        ),
        (
            1,
            BindingType::StorageBuffer {
                read_only: false,
                dynamic_offset: true,
            },
            ShaderStages::COMPUTE,
            buffer_range(output, 0, 16),
        ),
    ]);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: device.shader(ShaderStage::Compute, DOUBLE),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("pipeline");
    (pipeline, group)
}

#[test]
fn mapped_buffers_feed_and_read_back_gpu_work() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let input = buffer(&device, 512, BufferUsage::UNIFORM, MemoryUsage::CpuToGpu);
        let output = buffer(&device, 512, BufferUsage::STORAGE, MemoryUsage::GpuToCpu);
        let (pipeline, group) = doubling(&device, input, output);

        let mut inputs = device.map_buffer(input).expect("map input");
        assert_eq!(inputs.len(), 512);
        inputs.write(0, &vec4([1.0, 2.0, 3.0, 4.0])).unwrap();
        inputs.write(256, &vec4([5.0, 6.0, 7.0, 8.0])).unwrap();
        // Writes are bounds-checked.
        assert!(inputs.write(500, &vec4([0.0; 4])).is_err());

        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.bind_pipeline(pipeline);
        for offset in [0, 256] {
            recorder.set_bind_group(0, group, &[offset, offset]);
            recorder.dispatch(1, 1, 1);
        }
        recorder.pipeline_barrier(&to_host(output, 512));
        // The input stays mapped while the GPU reads it.
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");
        drop(inputs);

        let outputs = device.map_buffer(output).expect("map output");
        let mut bytes = [0; 16];
        outputs.read(0, &mut bytes).unwrap();
        assert_eq!(floats(&bytes), [2.0, 4.0, 6.0, 8.0], "{backend:?}");
        outputs.read(256, &mut bytes).unwrap();
        assert_eq!(floats(&bytes), [10.0, 12.0, 14.0, 16.0], "{backend:?}");
    }
}

#[test]
fn host_access_to_a_mapped_buffer_is_exclusive() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let staging = buffer(&device, 64, BufferUsage::COPY_SRC, MemoryUsage::CpuToGpu);
        let mapped = device.map_buffer(staging).expect("map");
        let busy = |result: Result<(), RhiError>, what: &str| {
            assert!(
                matches!(&result, Err(RhiError::ValidationError(m)) if m.contains("mapped")),
                "{backend:?}: {what} while mapped: {result:?}"
            );
        };
        busy(device.map_buffer(staging).map(|_| ()), "map_buffer");
        busy(device.write_buffer(staging, 0, &[1; 4]), "write_buffer");
        busy(device.read_buffer(staging, 0, 4).map(|_| ()), "read_buffer");
        busy(device.destroy_buffer(staging), "destroy_buffer");
        drop(mapped);
        device
            .write_buffer(staging, 0, &[7; 4])
            .expect("write after unmap");
        assert_eq!(device.read_buffer(staging, 0, 4).unwrap(), [7; 4]);
        let again = device.map_buffer(staging).expect("map again");
        let mut out = [0; 4];
        again.read(0, &mut out).unwrap();
        assert_eq!(
            out, [7; 4],
            "{backend:?}: the mapping sees write_buffer's data"
        );
        drop(again);
        device.destroy_buffer(staging).expect("destroy after unmap");
    }
}

#[test]
fn gpu_only_buffers_are_not_mapped_but_still_read_and_written() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let local = buffer(
            &device,
            64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
            MemoryUsage::GpuOnly,
        );
        assert!(
            matches!(device.map_buffer(local), Err(RhiError::ValidationError(_))),
            "{backend:?}"
        );
        device.write_buffer(local, 8, &[1, 2, 3, 4]).expect("write");
        assert_eq!(
            device.read_buffer(local, 8, 4).expect("read"),
            [1, 2, 3, 4],
            "{backend:?}"
        );
    }
}

/// One mapped uniform ring, one region per frame in flight: each frame writes its region while
/// the GPU may still read the other one, and `begin_frame` keeps the CPU from overwriting a
/// region in use.
#[test]
fn a_persistently_mapped_ring_streams_data_across_frames() {
    const FRAMES: u32 = 6;
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let ring = buffer(
            &device,
            256 * FRAMES_IN_FLIGHT as u64,
            BufferUsage::UNIFORM,
            MemoryUsage::CpuToGpu,
        );
        let output = buffer(
            &device,
            256 * u64::from(FRAMES),
            BufferUsage::STORAGE,
            MemoryUsage::GpuToCpu,
        );
        let (pipeline, group) = doubling(&device, ring, output);
        let mut mapped = device.map_buffer(ring).expect("map ring");
        for frame in 0..FRAMES {
            let slot = device.begin_frame().expect("begin_frame").index() as u32;
            mapped
                .write(u64::from(slot) * 256, &vec4([frame as f32; 4]))
                .unwrap();
            let mut recorder = device.create_command_recorder().expect("recorder");
            recorder.begin().expect("begin");
            recorder.bind_pipeline(pipeline);
            recorder.set_bind_group(0, group, &[slot * 256, frame * 256]);
            recorder.dispatch(1, 1, 1);
            recorder.pipeline_barrier(&to_host(output, 256 * u64::from(FRAMES)));
            device
                .submit(recorder.finish().expect("finish"))
                .expect("submit");
            device.end_frame().expect("end_frame");
        }
        device.wait_idle().expect("wait_idle");
        drop(mapped);
        let results = device
            .read_buffer(output, 0, 256 * u64::from(FRAMES))
            .expect("read");
        for frame in 0..FRAMES as usize {
            assert_eq!(
                floats(&results[frame * 256..frame * 256 + 16]),
                [frame as f32 * 2.0; 4],
                "{backend:?}: frame {frame}"
            );
        }
    }
}

/// RVK-19, RGL-21: write_buffer and read_buffer reject overflowing offsets and sizes with
/// ValidationError instead of panicking, wrapping, or performing out-of-bounds memory accesses.
#[test]
fn buffer_write_and_read_reject_overflowing_and_out_of_bounds_offsets() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let buf = buffer(
            &device,
            16,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            MemoryUsage::CpuToGpu,
        );

        // Overflowing offset arithmetic: offset + len wraps around
        let res_write = device.write_buffer(buf, u64::MAX - 3, &[0xAA; 8]);
        assert!(
            matches!(res_write, Err(RhiError::ValidationError(_))),
            "{backend:?}: write_buffer with overflowing offset must fail with ValidationError, got {res_write:?}"
        );

        let res_read = device.read_buffer(buf, u64::MAX - 3, 8);
        assert!(
            matches!(res_read, Err(RhiError::ValidationError(_))),
            "{backend:?}: read_buffer with overflowing offset must fail with ValidationError, got {res_read:?}"
        );

        // Ordinary out of bounds
        let res_write_oob = device.write_buffer(buf, 12, &[0xAA; 8]);
        assert!(
            matches!(res_write_oob, Err(RhiError::ValidationError(_))),
            "{backend:?}: write_buffer out of bounds must fail with ValidationError, got {res_write_oob:?}"
        );

        let res_read_oob = device.read_buffer(buf, 12, 8);
        assert!(
            matches!(res_read_oob, Err(RhiError::ValidationError(_))),
            "{backend:?}: read_buffer out of bounds must fail with ValidationError, got {res_read_oob:?}"
        );
    }
}
