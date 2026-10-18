//! OpenGL and Vulkan devices in one process (ISSUE-016).
//!
//! Unloading a Vulkan driver while OpenGL is in use in the same process corrupted the OpenGL
//! driver's heap on the owner's machine (NVIDIA's Vulkan driver shares its core library with its
//! OpenGL driver). The Vulkan backend must keep the drivers it loaded until the process exits, so
//! Vulkan devices can come and go next to a working OpenGL device.
//!
//! Heap corruption usually goes unnoticed until much later, so the test runs its scenario again
//! in a child process with glibc's checking allocator (`libc_malloc_debug.so.0`), which aborts at
//! the first corrupted heap operation. Without that library the scenario runs unchecked.
#![cfg(target_os = "linux")]

use korppi_rhi::{BufferDescriptor, BufferUsage, MemoryUsage};
use korppi_rhi_conformance::{Backend, TestDevice};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set in the child process that runs the scenario under glibc's heap checks.
const CHILD_ENV: &str = "KORPPI_MIXED_BACKENDS_CHILD";
const MALLOC_DEBUG_LIBRARY: &str = "libc_malloc_debug.so.0";
const VULKAN_DEVICES: usize = 20;

fn buffer_round_trip(device: &TestDevice) {
    let bytes = [7u8; 256];
    let buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: bytes.len() as u64,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            Some(&bytes),
        )
        .expect("create_buffer");
    assert_eq!(
        device
            .read_buffer(buffer, 0, bytes.len() as u64)
            .expect("read_buffer"),
        bytes
    );
    device.destroy_buffer(buffer).expect("destroy_buffer");
}

/// An OpenGL device works on one thread while Vulkan devices are created and destroyed, one at
/// a time, on another.
fn scenario() {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let opengl = scope.spawn(|| {
            let device = TestDevice::open(Backend::OpenGl);
            while !done.load(Ordering::Relaxed) {
                buffer_round_trip(&device);
            }
            buffer_round_trip(&device);
        });
        for _ in 0..VULKAN_DEVICES {
            buffer_round_trip(&TestDevice::open(Backend::Vulkan));
        }
        done.store(true, Ordering::Relaxed);
        opengl.join().expect("OpenGL thread");
    });
}

#[test]
fn opengl_keeps_working_while_vulkan_devices_come_and_go() {
    if std::env::var_os(CHILD_ENV).is_some() {
        scenario();
        return;
    }
    let status = Command::new(std::env::current_exe().expect("test binary path"))
        .args([
            "--exact",
            "opengl_keeps_working_while_vulkan_devices_come_and_go",
            "--test-threads=1",
            "--nocapture",
        ])
        .env(CHILD_ENV, "1")
        .env("LD_PRELOAD", MALLOC_DEBUG_LIBRARY)
        .env(
            "GLIBC_TUNABLES",
            "glibc.malloc.check=3:glibc.malloc.perturb=165",
        )
        .status()
        .expect("run the scenario in a child process");
    assert!(
        status.success(),
        "OpenGL and Vulkan in one process: the scenario failed under glibc's heap checks \
         ({status}); a signal or abort points to heap corruption"
    );
}
