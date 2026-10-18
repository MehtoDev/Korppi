//! Vulkan recording contexts and deferred destruction (findings RVK-04, RVK-11, RVK-17).
//!
//! Every recorder owns a command pool and descriptor pools that are reused once its submission
//! completes, or at once if it is dropped unsubmitted; destroyed resources are freed only after
//! the submissions issued before their destruction have completed.

use korppi_rhi::{BufferDescriptor, BufferUsage, DiagnosticSeverity, GraphicsDevice, MemoryUsage};
use korppi_rhi_vulkan::VulkanDevice;

fn assert_no_api_errors(device: &dyn GraphicsDevice) {
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "API errors reported: {errors:#?}");
}

/// RVK-17: a recorder dropped before `finish` (or a command buffer dropped unsubmitted) used to
/// leak its command buffer. Now its recording context is reused.
#[test]
fn dropped_recorders_and_command_buffers_are_reused() {
    let device = VulkanDevice::create_headless().expect("Vulkan device");
    for _ in 0..100 {
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        drop(recorder);

        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        drop(recorder.finish().expect("finish"));
    }
    let stats = device.recording_stats();
    assert_eq!(stats.contexts_created, 1, "{stats:?}");
    assert_eq!(stats.idle, 1, "{stats:?}");
    assert_eq!(stats.in_flight, 0, "{stats:?}");
    assert_no_api_errors(&device);
}

/// Two frames in flight need no more than three recording contexts however many frames run.
#[test]
fn recording_contexts_are_recycled_across_frames() {
    let device = VulkanDevice::create_headless().expect("Vulkan device");
    for _ in 0..50 {
        device.begin_frame().expect("begin_frame");
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        device
            .submit(recorder.finish().expect("finish"))
            .expect("submit");
        device.end_frame().expect("end_frame");
    }
    device.wait_idle().expect("wait_idle");
    let stats = device.recording_stats();
    assert!(stats.contexts_created <= 3, "{stats:?}");
    assert_eq!(stats.in_flight, 0, "{stats:?}");
    assert_no_api_errors(&device);
}

/// Destruction waits for the submissions issued before it, and happens once they complete.
#[test]
fn destruction_is_deferred_until_earlier_submissions_complete() {
    let device = VulkanDevice::create_headless().expect("Vulkan device");
    let buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: 256,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuOnly,
                label: None,
            },
            None,
        )
        .expect("create buffer");

    // Nothing is in flight: destruction is immediate.
    device.destroy_buffer(buffer).expect("destroy");
    assert_eq!(device.pending_destruction_count(), 0);

    let buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: 256,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuOnly,
                label: None,
            },
            None,
        )
        .expect("create buffer");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    let submission = device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.destroy_buffer(buffer).expect("destroy");
    // The buffer waits for the submission (or is already gone if the device collected it). A
    // completed submission does not collect it by itself: `is_submission_complete` only polls,
    // and the waits and later frames collect (ISSUE-007).
    assert!(device.pending_destruction_count() <= 1);
    device.wait_for_submission(submission).expect("wait");
    assert_eq!(device.pending_destruction_count(), 0);
    assert_no_api_errors(&device);
}
