//! Threading and device-lifetime regression tests from audit probes R9, R10 and R13.
//!
//! `GraphicsDevice` is `Send + Sync`, so every backend must work when called from any thread,
//! must allow several devices to coexist, and must allow concurrent command recording.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferMemoryBarrier, BufferTextureCopy, BufferUsage,
    GraphicsDevice, ImageLayout, ImageMemoryBarrier, LoadOp, MemoryUsage, PipelineBarrier,
    PipelineStageFlags, TextureUsage,
};
use korppi_rhi_conformance::{color_pass, color_target, undefined_to_color_attachment, Image};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;
use std::sync::{mpsc, Arc};
use std::time::Duration;

fn small_uniform_buffer() -> BufferDescriptor {
    BufferDescriptor {
        size: 64,
        usage: BufferUsage::UNIFORM,
        memory: MemoryUsage::GpuOnly,
        label: None,
    }
}

/// R9: a device created on one thread must be usable from another.
#[test]
fn r9_opengl_device_is_usable_from_another_thread() {
    let device = Arc::new(OpenGlDevice::create_headless().expect("OpenGL device"));
    let shared = Arc::clone(&device);
    let result = std::thread::spawn(move || shared.create_buffer(&small_uniform_buffer(), None))
        .join()
        .expect("worker thread panicked");
    assert!(
        result.is_ok(),
        "create_buffer from a second thread failed: {result:?}"
    );
}

/// Clears a 16x16 target to red on `device` and checks the readback.
fn clear_and_read_back(device: &dyn GraphicsDevice) {
    let target = color_target(device, 16, 16, TextureUsage::empty());
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    recorder
        .begin_rendering(&color_pass(
            target,
            16,
            16,
            LoadOp::Clear,
            [1.0, 0.0, 0.0, 1.0],
        ))
        .expect("begin_rendering");
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let image = Image::read(device, target, 16, 16);
    assert_eq!(image.count(|p| p[0] > 200), 16 * 16);
}

/// R10: dropping one device must not disable another that is still alive.
#[test]
fn r10_dropping_one_opengl_device_leaves_others_working() {
    let first = OpenGlDevice::create_headless().expect("first OpenGL device");
    let second = OpenGlDevice::create_headless().expect("second OpenGL device");
    drop(first);
    clear_and_read_back(&second);
}

/// R10 under concurrency: devices created, used and dropped on parallel threads (as parallel
/// tests do) must not terminate the shared EGL display under each other.
#[test]
fn opengl_devices_can_be_created_and_dropped_concurrently() {
    let workers: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..25 {
                    let device = OpenGlDevice::create_headless().expect("OpenGL device");
                    clear_and_read_back(&device);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().expect("worker thread panicked");
    }
}

/// R13: two threads recording on their own recorders must both make progress.
#[test]
fn r13_vulkan_concurrent_recording_does_not_deadlock() {
    const ITERATIONS: usize = 50_000;
    let device = Arc::new(VulkanDevice::create_headless().expect("Vulkan device"));
    let texture = color_target(&*device, 4, 4, TextureUsage::COPY_DST);
    let buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: 64,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            None,
        )
        .expect("create buffer");

    let (done, finished) = mpsc::channel();
    for worker in 0..2 {
        let device = Arc::clone(&device);
        let done = done.clone();
        std::thread::spawn(move || {
            let mut recorder = device.create_command_recorder().expect("recorder");
            recorder.begin().expect("begin");
            let barrier = PipelineBarrier::new()
                .with_image_barrier(ImageMemoryBarrier {
                    texture,
                    range: korppi_rhi::TextureSubresourceRange::ALL,
                    src_stages: PipelineStageFlags::TRANSFER,
                    src_access: AccessFlags::TRANSFER_WRITE,
                    dst_stages: PipelineStageFlags::TRANSFER,
                    dst_access: AccessFlags::TRANSFER_WRITE,
                    old_layout: ImageLayout::General,
                    new_layout: ImageLayout::General,
                })
                .with_buffer_barrier(BufferMemoryBarrier {
                    buffer,
                    src_stages: PipelineStageFlags::TRANSFER,
                    src_access: AccessFlags::TRANSFER_READ,
                    dst_stages: PipelineStageFlags::TRANSFER,
                    dst_access: AccessFlags::TRANSFER_READ,
                    offset: 0,
                    size: 64,
                });
            let copy = BufferTextureCopy {
                buffer_offset: 0,
                buffer_row_pitch: 0,
                texture_mip_level: 0,
                texture_layer: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
            };
            // Copies into one image need a write-after-write barrier between them
            // (synchronization validation reports the hazard otherwise).
            let between_copies = |old_layout| {
                PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
                    texture,
                    range: korppi_rhi::TextureSubresourceRange::ALL,
                    src_stages: PipelineStageFlags::TRANSFER,
                    src_access: AccessFlags::TRANSFER_WRITE,
                    dst_stages: PipelineStageFlags::TRANSFER,
                    dst_access: AccessFlags::TRANSFER_WRITE,
                    old_layout,
                    new_layout: ImageLayout::TransferDst,
                })
            };
            if worker == 0 {
                recorder.pipeline_barrier(&between_copies(ImageLayout::Undefined));
            }
            for _ in 0..ITERATIONS {
                if worker == 0 {
                    recorder.copy_buffer_to_texture(buffer, texture, &copy);
                    recorder.pipeline_barrier(&between_copies(ImageLayout::TransferDst));
                } else {
                    recorder.pipeline_barrier(&barrier);
                }
            }
            let _ = done.send(worker);
        });
    }
    drop(done);

    // Single-threaded, both loops together take well under a second on lavapipe.
    for completed in 0..2 {
        assert!(
            finished.recv_timeout(Duration::from_secs(10)).is_ok(),
            "only {completed} of 2 recording threads finished within 10 s (deadlock)"
        );
    }
    // The validation layer's thread-safety checks report unsynchronized use of a shared
    // command pool.
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == korppi_rhi::DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "API errors reported: {errors:#?}");
}
