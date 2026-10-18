//! SPIR-V modules (FIX-1.5; finding RVK-18): a module given as bytes may be in either byte
//! order, which its magic number tells; data without the magic number is rejected.

use korppi_rhi::{
    BufferDescriptor, BufferUsage, ComputePipelineDescriptor, MemoryUsage, RhiError,
    ShaderDescriptor, ShaderSource, ShaderStage, ShaderStages,
};
use korppi_rhi_conformance::{buffer_range, compile_spirv, Backend, TestDevice, STORAGE};

const WRITE_42: &str = r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer Out { uint value; };
void main() { value = 42u; }
";

fn module(device: &TestDevice, source: ShaderSource) -> Result<korppi_rhi::ShaderHandle, RhiError> {
    device.create_shader(&ShaderDescriptor {
        stage: ShaderStage::Compute,
        entry_point: "main",
        source,
        label: None,
    })
}

/// Runs `shader` and returns the value it wrote.
fn run(device: &TestDevice, shader: korppi_rhi::ShaderHandle) -> u32 {
    let output = device
        .create_buffer(
            &BufferDescriptor {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            Some(&[0; 4]),
        )
        .expect("buffer");
    let (layout, group) = device.bind_group(&[(
        0,
        STORAGE,
        ShaderStages::COMPUTE,
        buffer_range(output, 0, 4),
    )]);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("pipeline");
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    r.bind_pipeline(pipeline);
    r.set_bind_group(0, group, &[]);
    r.dispatch(1, 1, 1);
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");
    let bytes = device.read_buffer(output, 0, 4).expect("read");
    u32::from_ne_bytes(bytes.try_into().expect("4 bytes"))
}

fn spirv_in_either_byte_order_is_accepted(backend: Backend) {
    let device = TestDevice::open(backend);
    let native = compile_spirv(WRITE_42, ShaderStage::Compute);
    let swapped: Vec<u8> = native
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|w| [w[3], w[2], w[1], w[0]])
        .collect();
    for (what, bytes) in [("native", &native), ("byte-swapped", &swapped)] {
        let shader = module(&device, ShaderSource::SpirvBytes(bytes))
            .unwrap_or_else(|e| panic!("{backend:?} {what}: {e:?}"));
        assert_eq!(run(&device, shader), 42, "{backend:?} {what}");
    }
    // Words built from the other byte order are swapped back too.
    let swapped_words: Vec<u32> = swapped
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_ne_bytes(*w))
        .collect();
    let shader = module(&device, ShaderSource::Spirv(&swapped_words))
        .unwrap_or_else(|e| panic!("{backend:?} swapped words: {e:?}"));
    assert_eq!(run(&device, shader), 42, "{backend:?} swapped words");
}

#[test]
fn spirv_in_either_byte_order_is_accepted_vulkan() {
    spirv_in_either_byte_order_is_accepted(Backend::Vulkan);
}

#[test]
fn spirv_in_either_byte_order_is_accepted_opengl() {
    spirv_in_either_byte_order_is_accepted(Backend::OpenGl);
}

fn data_that_is_not_spirv_is_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let native = compile_spirv(WRITE_42, ShaderStage::Compute);
    let mut no_magic = native.clone();
    no_magic[..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    let header_only = native[..20].to_vec();
    for (what, bytes) in [
        ("no magic number", no_magic.as_slice()),
        ("odd length", &native[..native.len() - 1]),
        ("empty", &[][..]),
        ("header only", header_only.as_slice()),
    ] {
        let result = module(&device, ShaderSource::SpirvBytes(bytes));
        assert!(
            matches!(result, Err(RhiError::ShaderCompilationFailed(_))),
            "{backend:?} {what}: {result:?}"
        );
    }
    // A module without the requested entry point, or with it for another stage.
    for (stage, entry_point) in [
        (ShaderStage::Compute, "not_main"),
        (ShaderStage::Fragment, "main"),
    ] {
        let result = device.create_shader(&ShaderDescriptor {
            stage,
            entry_point,
            source: ShaderSource::SpirvBytes(&native),
            label: None,
        });
        assert!(
            matches!(result, Err(RhiError::ShaderCompilationFailed(_))),
            "{backend:?} {stage:?} {entry_point}: {result:?}"
        );
    }
}

#[test]
fn data_that_is_not_spirv_is_rejected_vulkan() {
    data_that_is_not_spirv_is_rejected(Backend::Vulkan);
}

#[test]
fn data_that_is_not_spirv_is_rejected_opengl() {
    data_that_is_not_spirv_is_rejected(Backend::OpenGl);
}
