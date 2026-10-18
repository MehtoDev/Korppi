//! Graphics-queue submission tracked by a timeline semaphore (findings RHI-02, RVK-04).
//!
//! Every submission signals the device's timeline semaphore with the next value, which becomes
//! its [`SubmissionId`]. Submissions on one queue complete in order, so the semaphore's counter
//! is the ID of the last completed submission.

use crate::context::VulkanContext;
use ash::vk;
use korppi_rhi::SmallVec;
use korppi_rhi::{RhiError, SubmissionId};
use std::sync::{Arc, Mutex};

/// The device's graphics queue plus its completion timeline.
pub(crate) struct SubmissionQueue {
    context: Arc<VulkanContext>,
    timeline: vk::Semaphore,
    /// Last value a submission will signal. The lock also provides the external synchronization
    /// Vulkan requires for host access to the queue (`vkQueueSubmit2`, `vkQueueWaitIdle`, and
    /// `vkDeviceWaitIdle`, which accesses every queue).
    last_issued: Mutex<u64>,
}

impl SubmissionQueue {
    /// Creates the timeline semaphore (Vulkan 1.2 `timelineSemaphore`, enabled by the context).
    pub(crate) fn new(context: Arc<VulkanContext>) -> Result<Self, RhiError> {
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        // SAFETY: `info` is a valid timeline-semaphore create info; the semaphore is destroyed
        // in `Drop` with the same device.
        let timeline = unsafe { context.device.create_semaphore(&info, None) }.map_err(|e| {
            RhiError::InitializationFailed(format!("timeline semaphore creation failed: {e}"))
        })?;
        Ok(Self {
            context,
            timeline,
            last_issued: Mutex::new(0),
        })
    }

    /// Submits `command_buffer` (executable, not pending) and returns the timeline value it
    /// signals when done.
    pub(crate) fn submit(
        &self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<SubmissionId, RhiError> {
        self.submit_batch(&[command_buffer], &[], &[])
    }

    /// Submits one batch: `command_buffers` (executable, not pending; possibly none) run after
    /// the binary semaphores in `waits` are signalled (each blocking the given stages), and the
    /// binary semaphores in `signals` are signalled with the timeline when the batch, and every
    /// earlier submission, has completed.
    pub(crate) fn submit_batch(
        &self,
        command_buffers: &[vk::CommandBuffer],
        waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        signals: &[vk::Semaphore],
    ) -> Result<SubmissionId, RhiError> {
        let mut last_issued = self.lock();
        let value = *last_issued + 1;
        // Inline storage: submitting does not allocate (RHI-11).
        let command_buffer_infos: SmallVec<[vk::CommandBufferSubmitInfo; 4]> = command_buffers
            .iter()
            .map(|&cmd| vk::CommandBufferSubmitInfo::default().command_buffer(cmd))
            .collect();
        let wait_infos: SmallVec<[vk::SemaphoreSubmitInfo; 4]> = waits
            .iter()
            .map(|&(semaphore, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .stage_mask(stage)
            })
            .collect();
        // Signal operations of a queue submission include all commands earlier in submission
        // order, so the binary semaphores are signalled after every earlier submission too.
        let signal_infos: SmallVec<[vk::SemaphoreSubmitInfo; 4]> = std::iter::once(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        )
        .chain(signals.iter().map(|&semaphore| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        }))
        .collect();
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(&signal_infos);
        // SAFETY: the queue is externally synchronized by `last_issued`'s lock; `value` is greater
        // than every value signalled or pending on the timeline (VUID-VkSemaphoreSubmitInfo-
        // semaphore-03882); callers pass executable command buffers of this device, binary
        // semaphores to wait for that have a signal operation pending (swapchain acquisitions),
        // and unsignalled binary semaphores with no pending operation to signal.
        unsafe {
            self.context.device.queue_submit2(
                self.context.graphics_queue,
                &[submit],
                vk::Fence::null(),
            )
        }
        .map_err(|e| crate::error::vk_error("vkQueueSubmit2", e))?;
        *last_issued = value;
        Ok(SubmissionId::new(value))
    }

    /// Queues image `index` of `swapchain` for presentation once `wait` is signalled. Returns
    /// whether the swapchain is suboptimal; `ERROR_OUT_OF_DATE_KHR` is returned as an error.
    pub(crate) fn present(
        &self,
        loader: &ash::khr::swapchain::Device,
        swapchain: vk::SwapchainKHR,
        index: u32,
        wait: vk::Semaphore,
    ) -> Result<bool, vk::Result> {
        let _queue = self.lock();
        let swapchains = [swapchain];
        let indices = [index];
        let waits = [wait];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: the queue is externally synchronized by the lock held here; the caller owns
        // the swapchain's external synchronization, acquired image `index` from it, and has
        // submitted a signal operation for `wait`.
        unsafe { loader.queue_present(self.context.graphics_queue, &info) }
    }

    /// ID of the last successful submission (0 before the first).
    pub(crate) fn last_issued(&self) -> SubmissionId {
        SubmissionId::new(*self.lock())
    }

    /// Rejects IDs this queue never returned (waiting for them would never finish).
    fn check(&self, submission: SubmissionId) -> Result<(), RhiError> {
        let last_issued = *self.lock();
        if submission.get() > last_issued {
            return Err(RhiError::ValidationError(format!(
                "{submission:?} was not returned by this device (last submission: {last_issued})"
            )));
        }
        Ok(())
    }

    /// Value of the last completed submission.
    pub(crate) fn completed(&self) -> Result<u64, RhiError> {
        // SAFETY: the semaphore is a live timeline semaphore of this device.
        unsafe {
            self.context
                .device
                .get_semaphore_counter_value(self.timeline)
        }
        .map_err(|e| crate::error::vk_error("vkGetSemaphoreCounterValue", e))
    }

    /// Whether `submission` has completed.
    pub(crate) fn is_complete(&self, submission: SubmissionId) -> Result<bool, RhiError> {
        self.check(submission)?;
        Ok(self.completed()? >= submission.get())
    }

    /// Blocks until `submission` has completed. Does not block other submitters.
    pub(crate) fn wait(&self, submission: SubmissionId) -> Result<(), RhiError> {
        self.check(submission)?;
        let semaphores = [self.timeline];
        let values = [submission.get()];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore is a live timeline semaphore; `values` was issued by a successful
        // submit (checked above), so the wait terminates.
        unsafe { self.context.device.wait_semaphores(&info, u64::MAX) }
            .map_err(|e| crate::error::vk_error("vkWaitSemaphores", e))
    }

    /// Blocks until the device is idle.
    pub(crate) fn wait_idle(&self) -> Result<(), RhiError> {
        let _queue = self.lock();
        // SAFETY: `vkDeviceWaitIdle` needs every queue externally synchronized; this device has
        // one queue, guarded by the lock held here.
        unsafe { self.context.device.device_wait_idle() }
            .map_err(|e| crate::error::vk_error("vkDeviceWaitIdle", e))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, u64> {
        self.last_issued
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for SubmissionQueue {
    fn drop(&mut self) {
        // SAFETY: the device is idle (waited here) so no pending submission signals the
        // semaphore any more.
        unsafe {
            let _ = self.context.device.device_wait_idle();
            self.context.device.destroy_semaphore(self.timeline, None);
        }
    }
}
