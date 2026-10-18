//! Non-blocking submission, frames in flight and resource lifetime (findings RHI-02, RVK-04,
//! RVK-17, RGL-16).
//!
//! `submit` returns a `SubmissionId` without waiting for the GPU; `wait_for_submission` and
//! `is_submission_complete` observe completion; `begin_frame` cycles through
//! `FRAMES_IN_FLIGHT` slots and waits for the work of the frame that last used the slot.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferHandle, BufferMemoryBarrier, BufferUsage, CommandBuffer,
    ComputePipelineDescriptor, MemoryUsage, PipelineBarrier, PipelineHandle, PipelineStageFlags,
    RhiError, ShaderStage, SubmissionId, FRAMES_IN_FLIGHT,
};
use korppi_rhi::{BindGroupHandle, ShaderStages};
use korppi_rhi_conformance::{buffer_range, STORAGE};
use korppi_rhi_conformance::{Backend, TestDevice};
use std::time::{Duration, Instant};

/// Each invocation runs `iterations` steps of a linear congruential generator seeded with its
/// invocation index and stores the result. The count is read from the buffer, so the loop cannot
/// be folded at compile time.
const BUSY: &str = r#"#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer B { uint iterations; uint pad[3]; uint values[]; };
void main() {
    uint x = gl_GlobalInvocationID.x;
    for (uint i = 0u; i < iterations; ++i) { x = x * 1664525u + 1013904223u; }
    values[gl_GlobalInvocationID.x] = x;
}
"#;

const GROUPS: u32 = 64;
const INVOCATIONS: u32 = GROUPS * 64;
/// Iterations that take well over 100 ms on a software rasterizer (lavapipe), and a few
/// milliseconds on a desktop GPU.
const SLOW_ITERATIONS: u32 = 250_000;
/// How long slow work lasts at least, on any device: the iteration count doubles from
/// [`SLOW_ITERATIONS`] until one submission takes this long (ISSUE-010).
const SLOW_WORK: Duration = Duration::from_millis(100);
/// The doubling stops here: a device that runs this many iterations within [`SLOW_WORK`] is not
/// running the shader.
const MAX_SLOW_ITERATIONS: u32 = 1 << 28;

fn expected_value(invocation: u32, iterations: u32) -> u32 {
    (0..iterations).fold(invocation, |x, _| {
        x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    })
}

/// A compute pipeline running [`BUSY`] and the storage buffer it writes.
struct BusyWork {
    pipeline: PipelineHandle,
    group: BindGroupHandle,
    buffer: BufferHandle,
    size: u64,
}

impl BusyWork {
    fn new(device: &TestDevice, iterations: u32) -> Self {
        let size = 16 + u64::from(INVOCATIONS) * 4;
        let mut header = [0_u8; 16];
        header[..4].copy_from_slice(&iterations.to_ne_bytes());
        let buffer = device
            .create_buffer(
                &BufferDescriptor {
                    size,
                    usage: BufferUsage::STORAGE,
                    memory: MemoryUsage::GpuToCpu,
                    label: None,
                },
                Some(&header),
            )
            .expect("create storage buffer");
        let (layout, group) = device.bind_group(&[(
            0,
            STORAGE,
            ShaderStages::COMPUTE,
            buffer_range(buffer, 0, size),
        )]);
        let compute_shader = device.shader(ShaderStage::Compute, BUSY);
        let pipeline = device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                compute_shader,
                bind_group_layouts: vec![layout],
                push_constant_ranges: vec![],
                label: None,
            })
            .expect("create compute pipeline");
        Self {
            pipeline,
            group,
            buffer,
            size,
        }
    }

    /// Sets the iteration count of the next submissions; the GPU must be done with the buffer.
    fn set_iterations(&self, device: &TestDevice, iterations: u32) {
        device
            .write_buffer(self.buffer, 0, &iterations.to_ne_bytes())
            .expect("write_buffer");
    }

    fn record(&self, device: &TestDevice) -> Box<dyn CommandBuffer> {
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.bind_pipeline(self.pipeline);
        recorder.set_bind_group(0, self.group, &[]);
        recorder.dispatch(GROUPS, 1, 1);
        recorder.pipeline_barrier(&PipelineBarrier::new().with_buffer_barrier(
            BufferMemoryBarrier {
                buffer: self.buffer,
                src_stages: PipelineStageFlags::COMPUTE_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_stages: PipelineStageFlags::HOST,
                dst_access: AccessFlags::HOST_READ,
                offset: 0,
                size: self.size,
            },
        ));
        recorder.finish().expect("finish")
    }

    fn values(&self, device: &TestDevice) -> Vec<u32> {
        let bytes = device
            .read_buffer(self.buffer, 16, self.size - 16)
            .expect("read_buffer");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_ne_bytes(*c))
            .collect()
    }
}

fn submission_ids_increase(backend: Backend) {
    let device = TestDevice::open(backend);
    let work = BusyWork::new(&device, 1);
    let first = device.submit(work.record(&device)).expect("submit");
    let second = device.submit(work.record(&device)).expect("submit");
    assert!(first < second, "{backend:?}: {first:?} then {second:?}");
    device.wait_for_submission(second).expect("wait");
    assert!(device.is_submission_complete(first).expect("query"));
    assert!(device.is_submission_complete(second).expect("query"));
}

#[test]
fn submission_ids_increase_vulkan() {
    submission_ids_increase(Backend::Vulkan);
}

#[test]
fn submission_ids_increase_opengl() {
    submission_ids_increase(Backend::OpenGl);
}

/// After `wait_for_submission` the GPU's writes are visible to `read_buffer`, without
/// `wait_idle`.
fn wait_for_submission_makes_gpu_writes_visible(backend: Backend) {
    let device = TestDevice::open(backend);
    let iterations = 1000;
    let work = BusyWork::new(&device, iterations);
    let id = device.submit(work.record(&device)).expect("submit");
    device.wait_for_submission(id).expect("wait");
    let values = work.values(&device);
    for invocation in [0, 1, 63, 64, INVOCATIONS - 1] {
        assert_eq!(
            values[invocation as usize],
            expected_value(invocation, iterations),
            "{backend:?}: invocation {invocation}"
        );
    }
}

#[test]
fn wait_for_submission_makes_gpu_writes_visible_vulkan() {
    wait_for_submission_makes_gpu_writes_visible(Backend::Vulkan);
}

#[test]
fn wait_for_submission_makes_gpu_writes_visible_opengl() {
    wait_for_submission_makes_gpu_writes_visible(Backend::OpenGl);
}

/// `submit` must return long before slow GPU work finishes (RVK-04: Vulkan waited for the queue
/// to drain inside every submit). The work doubles until a submission lasts [`SLOW_WORK`] on the
/// device at hand, and that submission is the one checked; one tiny submission first moves
/// pipeline compilation and the GPU's clock ramp-up out of the timing.
fn submit_returns_before_the_gpu_finishes(backend: Backend) {
    let device = TestDevice::open(backend);
    let work = BusyWork::new(&device, 1);
    let warm_up = device.submit(work.record(&device)).expect("submit");
    device.wait_for_submission(warm_up).expect("wait");

    let mut iterations = SLOW_ITERATIONS;
    let (id, submitted, finished) = loop {
        work.set_iterations(&device, iterations);
        let command_buffer = work.record(&device);
        let start = Instant::now();
        let id = device.submit(command_buffer).expect("submit");
        let submitted = start.elapsed();
        device.wait_for_submission(id).expect("wait");
        let finished = start.elapsed();
        if finished >= SLOW_WORK {
            break (id, submitted, finished);
        }
        assert!(
            iterations <= MAX_SLOW_ITERATIONS / 2,
            "{backend:?}: {iterations} iterations took only {finished:?}"
        );
        iterations *= 2;
    };
    eprintln!(
        "{backend:?}: {iterations} iterations; submit {submitted:?}, complete after {finished:?}"
    );
    assert!(
        submitted * 4 < finished,
        "{backend:?}: submit took {submitted:?} of {finished:?} (it waited for the GPU)"
    );
    assert!(device.is_submission_complete(id).expect("query"));
    assert_eq!(
        work.values(&device)[5],
        expected_value(5, iterations),
        "{backend:?}"
    );
}

#[test]
fn submit_returns_before_the_gpu_finishes_vulkan() {
    submit_returns_before_the_gpu_finishes(Backend::Vulkan);
}

#[test]
fn submit_returns_before_the_gpu_finishes_opengl() {
    submit_returns_before_the_gpu_finishes(Backend::OpenGl);
}

/// Frame slots cycle, and reusing a slot waits for the work of the frame that used it before.
fn begin_frame_waits_for_the_previous_use_of_its_slot(backend: Backend) {
    assert_eq!(FRAMES_IN_FLIGHT, 2, "the test assumes two frames in flight");
    let device = TestDevice::open(backend);
    let work = BusyWork::new(&device, SLOW_ITERATIONS / 4);
    assert_eq!(device.begin_frame().expect("frame 0").index(), 0);
    let id = device.submit(work.record(&device)).expect("submit");
    assert_eq!(device.begin_frame().expect("frame 1").index(), 1);
    assert_eq!(device.begin_frame().expect("frame 2").index(), 0);
    assert!(
        device.is_submission_complete(id).expect("query"),
        "{backend:?}: frame 2 reuses frame 0's slot, so frame 0's work must be complete"
    );
    device.end_frame().expect("end_frame");
}

#[test]
fn begin_frame_waits_for_the_previous_use_of_its_slot_vulkan() {
    begin_frame_waits_for_the_previous_use_of_its_slot(Backend::Vulkan);
}

#[test]
fn begin_frame_waits_for_the_previous_use_of_its_slot_opengl() {
    begin_frame_waits_for_the_previous_use_of_its_slot(Backend::OpenGl);
}

/// Waiting for an ID the device never returned is an error, not a hang.
fn unknown_submissions_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let work = BusyWork::new(&device, 1);
    let id = device.submit(work.record(&device)).expect("submit");
    let unknown = SubmissionId::new(id.get() + 1000);
    assert!(
        matches!(
            device.wait_for_submission(unknown),
            Err(RhiError::ValidationError(_))
        ),
        "{backend:?}: wait_for_submission"
    );
    assert!(
        matches!(
            device.is_submission_complete(unknown),
            Err(RhiError::ValidationError(_))
        ),
        "{backend:?}: is_submission_complete"
    );
    device.wait_idle().expect("wait_idle");
}

#[test]
fn unknown_submissions_are_rejected_vulkan() {
    unknown_submissions_are_rejected(Backend::Vulkan);
}

#[test]
fn unknown_submissions_are_rejected_opengl() {
    unknown_submissions_are_rejected(Backend::OpenGl);
}

/// Records `works` into one command buffer.
fn record_all(device: &TestDevice, works: &[&BusyWork]) -> Box<dyn CommandBuffer> {
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    for work in works {
        recorder.bind_pipeline(work.pipeline);
        recorder.set_bind_group(0, work.group, &[]);
        recorder.dispatch(GROUPS, 1, 1);
    }
    recorder.pipeline_barrier(
        &PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
            buffer: works[0].buffer,
            src_stages: PipelineStageFlags::COMPUTE_SHADER,
            src_access: AccessFlags::SHADER_WRITE,
            dst_stages: PipelineStageFlags::HOST,
            dst_access: AccessFlags::HOST_READ,
            offset: 0,
            size: works[0].size,
        }),
    );
    recorder.finish().expect("finish")
}

/// Destroying resources that an in-flight submission uses is allowed: the backend keeps them
/// alive until the submission completes (RVK-17). The validation layer would report
/// VUID-vkDestroyPipeline-pipeline-00765 / VUID-vkDestroyBuffer-buffer-00922 /
/// VUID-vkFreeDescriptorSets-pDescriptorSets-00309 otherwise, and
/// `TestDevice` fails the test on API errors.
fn resources_destroyed_while_in_flight_outlive_the_submission(backend: Backend) {
    let device = TestDevice::open(backend);
    let kept = BusyWork::new(&device, SLOW_ITERATIONS / 4);
    let doomed = BusyWork::new(&device, SLOW_ITERATIONS / 4);
    let id = device
        .submit(record_all(&device, &[&kept, &doomed]))
        .expect("submit");
    device
        .destroy_pipeline(kept.pipeline)
        .expect("destroy pipeline");
    device
        .destroy_pipeline(doomed.pipeline)
        .expect("destroy pipeline");
    device
        .destroy_buffer(doomed.buffer)
        .expect("destroy buffer");
    device
        .destroy_bind_group(doomed.group)
        .expect("destroy bind group");
    device.wait_for_submission(id).expect("wait");
    assert_eq!(
        kept.values(&device)[INVOCATIONS as usize - 1],
        expected_value(INVOCATIONS - 1, SLOW_ITERATIONS / 4),
        "{backend:?}"
    );
    // The handles are invalid at once, even though the objects live on.
    assert!(matches!(
        device.destroy_buffer(doomed.buffer),
        Err(RhiError::InvalidHandle(_))
    ));
}

#[test]
fn resources_destroyed_while_in_flight_outlive_the_submission_vulkan() {
    resources_destroyed_while_in_flight_outlive_the_submission(Backend::Vulkan);
}

#[test]
fn resources_destroyed_while_in_flight_outlive_the_submission_opengl() {
    resources_destroyed_while_in_flight_outlive_the_submission(Backend::OpenGl);
}

/// Recorders and command buffers dropped without being submitted are fine on every backend.
fn unsubmitted_work_can_be_dropped(backend: Backend) {
    let device = TestDevice::open(backend);
    let work = BusyWork::new(&device, 1);
    for _ in 0..50 {
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.bind_pipeline(work.pipeline);
        recorder.set_bind_group(0, work.group, &[]);
        drop(recorder);
        drop(work.record(&device));
    }
    device
        .submit_and_wait(work.record(&device))
        .expect("submit after drops");
    assert_eq!(work.values(&device)[3], expected_value(3, 1), "{backend:?}");
}

#[test]
fn unsubmitted_work_can_be_dropped_vulkan() {
    unsubmitted_work_can_be_dropped(Backend::Vulkan);
}

#[test]
fn unsubmitted_work_can_be_dropped_opengl() {
    unsubmitted_work_can_be_dropped(Backend::OpenGl);
}
