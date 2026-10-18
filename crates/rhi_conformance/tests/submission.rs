//! `GraphicsDevice::submit` must reject command buffers it did not record (findings RHI-03 and,
//! for OpenGL devices among themselves, RGL-22).
//!
//! Before FIX-1.1 both backends reinterpreted any `Box<dyn CommandBuffer>` as their own type,
//! which is undefined behaviour reachable from safe code.

use korppi_rhi::{CommandBuffer, GraphicsDevice, RhiError};
use korppi_rhi_conformance::{Backend, TestDevice};

fn empty_command_buffer(device: &dyn GraphicsDevice) -> Box<dyn CommandBuffer> {
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.finish().expect("finish")
}

fn assert_rejected<T: std::fmt::Debug>(result: Result<T, RhiError>, what: &str) {
    assert!(
        matches!(result, Err(RhiError::InvalidHandle(_))),
        "{what}: expected RhiError::InvalidHandle, got {result:?}"
    );
}

#[test]
fn submit_rejects_command_buffers_from_the_other_backend() {
    let vulkan = TestDevice::open(Backend::Vulkan);
    let opengl = TestDevice::open(Backend::OpenGl);
    assert_rejected(
        vulkan.submit(empty_command_buffer(&*opengl)),
        "OpenGL command buffer submitted to Vulkan",
    );
    assert_rejected(
        opengl.submit(empty_command_buffer(&*vulkan)),
        "Vulkan command buffer submitted to OpenGL",
    );
}

#[test]
fn vulkan_submit_rejects_command_buffers_from_another_vulkan_device() {
    let first = TestDevice::open(Backend::Vulkan);
    let second = TestDevice::open(Backend::Vulkan);
    assert_rejected(
        first.submit(empty_command_buffer(&*second)),
        "command buffer from a different VulkanDevice",
    );
}

/// R15b (RGL-22): OpenGL replayed a command buffer recorded by another OpenGL device against the
/// submitting device's resources.
#[test]
fn opengl_submit_rejects_command_buffers_from_another_opengl_device() {
    let first = TestDevice::open(Backend::OpenGl);
    let second = TestDevice::open(Backend::OpenGl);
    assert_rejected(
        first.submit(empty_command_buffer(&*second)),
        "command buffer from a different OpenGlDevice",
    );
}

/// R15: a finished command buffer outlives the device that recorded it; another device of the
/// same backend must still reject it.
#[test]
fn submit_rejects_command_buffers_whose_device_is_gone() {
    for backend in Backend::ALL {
        let recording = TestDevice::open(backend);
        let command_buffer = empty_command_buffer(&*recording);
        drop(recording);
        let submitting = TestDevice::open(backend);
        assert_rejected(
            submitting.submit(command_buffer),
            &format!("{backend:?}: command buffer of a dropped device"),
        );
    }
}

#[test]
fn submit_accepts_its_own_command_buffers() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        device
            .submit(empty_command_buffer(&*device))
            .unwrap_or_else(|e| panic!("{backend:?}: {e:?}"));
        device.wait_idle().expect("wait_idle");
    }
}
