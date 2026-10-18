//! Per-recorder command pools, recycled after their submission completes (findings RVK-04,
//! RVK-11, RVK-17).
//!
//! Every command recorder owns a [`RecordingContext`]: a command pool with one primary command
//! buffer. Nothing in it is shared, so
//! recorders on different threads never touch the same pool (Vulkan requires host access to a
//! command pool, including recording into its command buffers, to be externally synchronized).
//! Once submitted, a context stays in flight until its submission completes and is then reset
//! and reused; a context that is never submitted (a dropped recorder or command buffer) is reset
//! and reused at once.

use crate::context::VulkanContext;
use ash::vk;
use korppi_rhi::{RhiError, SubmissionId};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A command pool with one primary command buffer, owned by one recorder (or command buffer, or
/// submission in flight) at a time.
pub(crate) struct RecordingContext {
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    /// Swapchain images the commands use (cleared, capacity kept, on reuse: RHI-11).
    pub(crate) swapchain_uses: Vec<crate::command::SwapchainUse>,
    /// Layout transitions of the commands' barriers, in order (texture ID, range, layout).
    pub(crate) layout_changes: Vec<(
        korppi_rhi::TextureHandle,
        korppi_rhi::TextureSubresourceRange,
        vk::ImageLayout,
    )>,
}

impl RecordingContext {
    fn create(context: &VulkanContext) -> Result<Self, RhiError> {
        let device = &context.device;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(context.graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: `pool_info` is valid; the pool is destroyed by `RecordingContext::destroy`.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| RhiError::OutOfMemory(format!("vkCreateCommandPool failed: {e}")))?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `pool` was just created and is not used by any other thread.
        match unsafe { device.allocate_command_buffers(&alloc_info) } {
            Ok(buffers) => {
                context.register_command_buffer(buffers[0]);
                Ok(Self {
                    pool,
                    command_buffer: buffers[0],
                    swapchain_uses: Vec::new(),
                    layout_changes: Vec::new(),
                })
            }
            Err(e) => {
                // SAFETY: the pool is unused.
                unsafe { device.destroy_command_pool(pool, None) };
                Err(RhiError::OutOfMemory(format!(
                    "vkAllocateCommandBuffers failed: {e}"
                )))
            }
        }
    }

    /// The context's primary command buffer.
    pub(crate) fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Returns the command buffer to the initial state. The command buffer must not be
    /// pending.
    fn reset(&mut self, device: &ash::Device) -> Result<(), RhiError> {
        self.swapchain_uses.clear();
        self.layout_changes.clear();
        // SAFETY: the caller guarantees the command buffer is not pending; the pool is owned by
        // this context alone.
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }
            .map_err(|e| crate::error::vk_error("vkResetCommandPool", e))
    }

    fn destroy(self, context: &VulkanContext) {
        // SAFETY: the caller guarantees the command buffer is not pending; destroying the pool
        // frees it.
        unsafe { context.device.destroy_command_pool(self.pool, None) };
        context.unregister_command_buffer(self.command_buffer);
    }
}

/// Counts of recording contexts, for tests and diagnostics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RecordingStats {
    /// Contexts created so far (each holds one command pool).
    pub contexts_created: usize,
    /// Contexts ready for reuse.
    pub idle: usize,
    /// Contexts whose submission has not been observed complete yet.
    pub in_flight: usize,
}

/// The device's recording contexts. The idle and in-flight lists have separate locks and no
/// function holds both.
pub(crate) struct RecordingContexts {
    context: Arc<VulkanContext>,
    idle: Mutex<Vec<RecordingContext>>,
    in_flight: Mutex<VecDeque<(SubmissionId, RecordingContext)>>,
    created: AtomicUsize,
}

impl RecordingContexts {
    pub(crate) fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            idle: Mutex::new(Vec::new()),
            in_flight: Mutex::new(VecDeque::new()),
            created: AtomicUsize::new(0),
        }
    }

    /// Takes an idle context, or creates one.
    pub(crate) fn acquire(&self) -> Result<RecordingContext, RhiError> {
        if let Some(recording) = lock(&self.idle).pop() {
            return Ok(recording);
        }
        let recording = RecordingContext::create(&self.context)?;
        self.created.fetch_add(1, Ordering::Relaxed);
        Ok(recording)
    }

    /// Returns a context that is not pending (never submitted, or its submission completed).
    pub(crate) fn release(&self, mut recording: RecordingContext) {
        match recording.reset(&self.context.device) {
            Ok(()) => lock(&self.idle).push(recording),
            Err(error) => {
                tracing::warn!("discarding a recording context that failed to reset: {error}");
                recording.destroy(&self.context);
            }
        }
    }

    /// Keeps a submitted context until `submission` completes.
    pub(crate) fn retire(&self, recording: RecordingContext, submission: SubmissionId) {
        lock(&self.in_flight).push_back((submission, recording));
    }

    /// Recycles the contexts of every submission up to `completed`.
    pub(crate) fn recycle(&self, completed: u64) {
        loop {
            let recording = {
                let mut in_flight = lock(&self.in_flight);
                match in_flight.front() {
                    Some((submission, _)) if submission.get() <= completed => {
                        in_flight.pop_front().map(|(_, recording)| recording)
                    }
                    _ => None,
                }
            };
            match recording {
                Some(recording) => self.release(recording),
                None => break,
            }
        }
    }

    pub(crate) fn stats(&self) -> RecordingStats {
        RecordingStats {
            contexts_created: self.created.load(Ordering::Relaxed),
            idle: lock(&self.idle).len(),
            in_flight: lock(&self.in_flight).len(),
        }
    }
}

impl Drop for RecordingContexts {
    /// Destroys every context. The owning device waits for idle before this runs, and recorders
    /// and command buffers keep this alive (through an `Arc`) while they hold a context.
    fn drop(&mut self) {
        let context = &self.context;
        // SAFETY: nothing can be pending once the device is idle and the device is gone.
        let _ = unsafe { context.device.device_wait_idle() };
        for recording in lock(&self.idle).drain(..) {
            recording.destroy(context);
        }
        for (_, recording) in lock(&self.in_flight).drain(..) {
            recording.destroy(context);
        }
    }
}

use korppi_rhi::lock;
