//! Handles belong to the device that created them (FIX-9.3; finding RHI-15, probe R16).
//!
//! Every device numbers its resources from slot 0. Before FIX-9.3 a handle carried only a slot
//! and a generation, so device B resolved device A's handle to B's own resource in the same slot:
//! B read its own bytes through A's buffer handle, and destroying A's handle on B destroyed B's
//! resource. Command buffers recorded by another device are covered in `tests/submission.rs`.

use korppi_rhi::{
    BufferDescriptor, BufferUsage, MemoryUsage, RhiError, SampleCount, SamplerDescriptor,
    ShaderStage, ShaderStages, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_conformance::{buffer_range, Backend, TestDevice, STORAGE};

const COMPUTE: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer B { uint v; };
void main() { v = 1u; }
"#;

fn buffer(device: &TestDevice, fill: u8) -> korppi_rhi::BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size: 4,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            Some(&[fill; 4]),
        )
        .expect("create_buffer")
}

fn texture(device: &TestDevice, fill: u8) -> korppi_rhi::TextureHandle {
    device
        .create_texture(
            &TextureDescriptor {
                width: 1,
                height: 1,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                dimension: TextureDimension::D2,
                label: None,
            },
            Some(&[fill; 4]),
        )
        .expect("create_texture")
}

fn assert_rejected<T: std::fmt::Debug>(backend: Backend, result: Result<T, RhiError>, what: &str) {
    assert!(
        matches!(result, Err(RhiError::InvalidHandle(_))),
        "{backend:?}: {what} with another device's handle: expected InvalidHandle, got {result:?}"
    );
}

/// R16: device B rejects every kind of handle device A created, even where B holds a resource
/// in the same slot, and B's own resources are untouched.
fn handles_from_another_device_are_rejected(backend: Backend) {
    let a = TestDevice::open(backend);
    let b = TestDevice::open(backend);

    let a_buffer = buffer(&a, 7);
    let b_buffer = buffer(&b, 2);
    assert_eq!(
        a_buffer.id(),
        b_buffer.id(),
        "{backend:?}: precondition: both devices put their first buffer in the same slot"
    );
    let a_texture = texture(&a, 7);
    let b_texture = texture(&b, 2);
    let a_sampler = a
        .create_sampler(&SamplerDescriptor::default())
        .expect("sampler");
    let b_sampler = b
        .create_sampler(&SamplerDescriptor::default())
        .expect("sampler");
    let a_shader = a.shader(ShaderStage::Compute, COMPUTE);
    let b_shader = b.shader(ShaderStage::Compute, COMPUTE);

    assert_rejected(backend, b.read_buffer(a_buffer, 0, 4), "read_buffer");
    assert_rejected(
        backend,
        b.write_buffer(a_buffer, 0, &[9; 4]),
        "write_buffer",
    );
    assert_rejected(backend, b.read_texture(a_texture, 0), "read_texture");
    let b_layout = b.layout(&[(0, STORAGE, ShaderStages::COMPUTE)]);
    assert_rejected(
        backend,
        b.create_bind_group(&korppi_rhi::BindGroupDescriptor {
            layout: b_layout,
            entries: vec![korppi_rhi::BindGroupEntry {
                binding: 0,
                resource: buffer_range(a_buffer, 0, 4),
            }],
            label: None,
        }),
        "create_bind_group",
    );
    assert_rejected(backend, b.destroy_buffer(a_buffer), "destroy_buffer");
    assert_rejected(backend, b.destroy_texture(a_texture), "destroy_texture");
    assert_rejected(backend, b.destroy_sampler(a_sampler), "destroy_sampler");
    assert_rejected(backend, b.destroy_shader(a_shader), "destroy_shader");

    // B's resources in the same slots are untouched, and so are A's.
    assert_eq!(
        b.read_buffer(b_buffer, 0, 4).expect("read"),
        [2; 4],
        "{backend:?}"
    );
    assert_eq!(
        b.read_texture(b_texture, 0).expect("read"),
        [2; 4],
        "{backend:?}"
    );
    assert_eq!(
        a.read_buffer(a_buffer, 0, 4).expect("read"),
        [7; 4],
        "{backend:?}"
    );
    b.destroy_sampler(b_sampler)
        .expect("B's sampler is still B's");
    b.destroy_shader(b_shader).expect("B's shader is still B's");
}

#[test]
fn handles_from_another_device_are_rejected_vulkan() {
    handles_from_another_device_are_rejected(Backend::Vulkan);
}

#[test]
fn handles_from_another_device_are_rejected_opengl() {
    handles_from_another_device_are_rejected(Backend::OpenGl);
}

/// A device of one backend rejects the handles of the other backend's devices too.
#[test]
fn handles_from_the_other_backend_are_rejected() {
    let vulkan = TestDevice::open(Backend::Vulkan);
    let opengl = TestDevice::open(Backend::OpenGl);
    let vulkan_buffer = buffer(&vulkan, 1);
    let opengl_buffer = buffer(&opengl, 2);
    assert_rejected(
        Backend::OpenGl,
        opengl.read_buffer(vulkan_buffer, 0, 4),
        "read_buffer",
    );
    assert_rejected(
        Backend::Vulkan,
        vulkan.read_buffer(opengl_buffer, 0, 4),
        "read_buffer",
    );
}
