use crate::binding::DescriptorPools;
use crate::command::{RecorderPools, VulkanCommandBuffer, VulkanCommandBufferRecorder};
use crate::context::VulkanContext;
use crate::memory::{AllocationRequest, DedicatedResource, ResourceKind};
use crate::memory::{Suballocation, VulkanMemoryManager};
use crate::queue::SubmissionQueue;
use crate::recording::{RecordingContexts, RecordingStats};
use crate::resource::{
    TextureStorage, VulkanBuffer, VulkanResourceRegistry, VulkanSampler, VulkanShader,
};
use crate::swapchain::{RetiredSwapchain, SwapchainMap};
use crate::types::{sampler_create_info, to_vk_buffer_usage};
use ash::vk;
use korppi_rhi::validation::{validate_recorder_pooled, ValidatingRecorder};
use korppi_rhi::{
    check_pipeline_layouts, BindGroupDescriptor, BindGroupHandle, BindGroupLayoutDescriptor,
    BindGroupLayoutHandle, BindGroupLayoutInfo, BoundResource, BoxPool, MappedBuffer, ShaderStages,
};
use korppi_rhi::{
    AdapterInfo, DeviceFeatures, DeviceLimits, FormatCapabilities, ImageLayout, TextureFormat,
    TextureSubresourceRange, TextureUsage, TextureViewDescriptor, TextureViewHandle,
};
use korppi_rhi::{
    BackendDiagnostic, BufferDescriptor, BufferHandle, BufferUsage, CommandBuffer,
    CommandBufferRecorder, ComputePipelineDescriptor, FrameIndex, FrameTracker, GraphicsDevice,
    GraphicsPipelineDescriptor, MemoryUsage, PipelineHandle, PresentTarget, RhiError,
    SamplerDescriptor, SamplerHandle, ShaderDescriptor, ShaderHandle, SubmissionId,
    SwapchainDescriptor, SwapchainHandle, SwapchainImage, SwapchainInfo, TextureDescriptor,
    TextureHandle,
};
use std::collections::VecDeque;
use std::ffi::CString;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

/// Concrete Vulkan 1.3 graphics device implementing `korppi_rhi::GraphicsDevice`.
///
/// `submit` does not wait for the GPU (RVK-04). Command buffers and their descriptor sets are
/// recycled, and destroyed resources freed, only once the submissions that may use them have
/// completed (RVK-17); completed work is collected whenever the device is called into
/// (`create_command_recorder`, `submit`, waits, `begin_frame`).
pub struct VulkanDevice {
    pub(crate) context: Arc<VulkanContext>,
    pub(crate) memory: Arc<VulkanMemoryManager>,
    pub(crate) resources: Arc<VulkanResourceRegistry>,
    /// Pools bind groups' descriptor sets come from.
    pub(crate) descriptor_pools: DescriptorPools,
    pub(crate) recording: Arc<RecordingContexts>,
    pub(crate) queue: SubmissionQueue,
    pub(crate) frames: FrameTracker,
    /// Destroyed resources waiting for the last submission issued before their destruction.
    pub(crate) garbage: Mutex<VecDeque<(SubmissionId, Garbage)>>,
    /// Window surfaces and their swapchains (FIX-1.3).
    pub(crate) swapchains: SwapchainMap,
    pub(crate) next_id: AtomicU32,
    /// Finished recorders, reused by `create_command_recorder` (RHI-11).
    recorder_pool: Arc<BoxPool<VulkanCommandBufferRecorder>>,
    /// Submitted command buffer boxes, reused by `finish`.
    command_buffer_pool: Arc<BoxPool<VulkanCommandBuffer>>,
    /// Validating recorders (debug builds), reused likewise.
    validator_pool: Arc<BoxPool<ValidatingRecorder>>,
    /// The pipeline cache every pipeline is created through (RVK-16).
    pub(crate) pipeline_cache: crate::pipeline_cache::PipelineCache,
    /// Shared pipeline layouts and pipelines (RVK-16).
    pub(crate) pipeline_objects: crate::pipeline::PipelineObjects,
}

/// Native objects of a destroyed resource whose destruction waits for in-flight submissions.
pub(crate) enum Garbage {
    Buffer(vk::Buffer, Suballocation),
    /// An image with its views (default, attachment and texture views).
    Texture(vk::Image, Vec<vk::ImageView>, Suballocation),
    /// A texture view destroyed on its own.
    View(vk::ImageView),
    Sampler(vk::Sampler),
    Pipeline(vk::Pipeline),
    /// A shared pipeline layout whose last pipeline was destroyed.
    PipelineLayout(vk::PipelineLayout),
    /// A bind group layout.
    SetLayout(vk::DescriptorSetLayout),
    /// A query set's pool.
    QueryPool(vk::QueryPool),
    /// A bind group's descriptor set and the pool it came from.
    DescriptorSet(vk::DescriptorPool, vk::DescriptorSet),
    /// A swapchain retired by recreation.
    Swapchain(RetiredSwapchain),
}

impl VulkanDevice {
    /// Creates a headless Vulkan 1.3 device for offscreen rendering, testing, and compute.
    pub fn create_headless() -> Result<Self, RhiError> {
        let context = VulkanContext::create_headless()?;
        let memory = Arc::new(VulkanMemoryManager::new(context.clone()));
        let descriptor_pools = DescriptorPools::new(context.clone());
        let pipeline_cache = crate::pipeline_cache::PipelineCache::new(context.clone())?;
        let resources = VulkanResourceRegistry::for_device(korppi_rhi::DeviceId::next());
        let _ = resources.format_capabilities.set(
            TextureFormat::ALL
                .iter()
                .map(|&format| context.capabilities.format(format))
                .collect(),
        );
        let _ = resources.features.set(context.capabilities.features);
        let resources = Arc::new(resources);
        let recording = Arc::new(RecordingContexts::new(context.clone()));
        let queue = SubmissionQueue::new(context.clone())?;
        Ok(Self {
            context,
            memory,
            resources,
            descriptor_pools,
            recording,
            queue,
            frames: FrameTracker::new(),
            garbage: Mutex::new(VecDeque::new()),
            swapchains: Mutex::default(),
            next_id: AtomicU32::new(1),
            recorder_pool: BoxPool::new(),
            command_buffer_pool: BoxPool::new(),
            validator_pool: BoxPool::new(),
            pipeline_cache,
            pipeline_objects: Default::default(),
        })
    }

    /// Returns the underlying Vulkan context.
    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    /// Budget and usage of every memory heap (RVK-08).
    pub fn memory_budgets(&self) -> Vec<crate::memory::HeapBudget> {
        self.memory.heap_budgets()
    }

    /// Returns the number of allocated VkDeviceMemory chunks.
    pub fn memory_chunk_count(&self) -> usize {
        self.memory.chunk_count()
    }

    /// Returns total allocated GPU memory bytes across all chunks.
    pub fn memory_allocated_bytes(&self) -> u64 {
        self.memory.allocated_bytes()
    }

    /// Returns total active suballocated bytes currently in use.
    pub fn memory_used_bytes(&self) -> u64 {
        self.memory.used_bytes()
    }

    /// The descriptor set layouts of a pipeline's bind group layouts, checked against the
    /// device's per-stage limits, with their validation summaries.
    pub(crate) fn pipeline_set_layouts(
        &self,
        handles: &[BindGroupLayoutHandle],
        stages: ShaderStages,
    ) -> Result<(Vec<vk::DescriptorSetLayout>, Vec<BindGroupLayoutInfo>), RhiError> {
        let layouts = lock(&self.resources.bind_group_layouts);
        let mut found = Vec::with_capacity(handles.len());
        for handle in handles {
            let (layout, _) = layouts.get(handle).ok_or_else(|| {
                RhiError::InvalidHandle(format!("bind group layout {handle:?} not found"))
            })?;
            found.push(layout);
        }
        let descs: Vec<&BindGroupLayoutDescriptor> = found.iter().map(|l| &l.desc).collect();
        check_pipeline_layouts(&descs, stages, &self.context.capabilities.limits)?;
        Ok((
            found.iter().map(|l| l.layout).collect(),
            found.iter().map(|l| l.info).collect(),
        ))
    }

    /// Counts of recording contexts (one command pool each).
    pub fn recording_stats(&self) -> RecordingStats {
        self.recording.stats()
    }

    /// Number of destroyed resources still waiting for in-flight submissions.
    pub fn pending_destruction_count(&self) -> usize {
        lock(&self.garbage).len()
    }

    /// Records the layout submitted or internal work left texture `id` in.
    pub(crate) fn set_texture_layout(&self, texture: TextureHandle, layout: vk::ImageLayout) {
        self.set_texture_range_layout(texture, &TextureSubresourceRange::ALL, layout);
    }

    /// Records the layout submitted or internal work left a range of texture `id` in.
    pub(crate) fn set_texture_range_layout(
        &self,
        id: TextureHandle,
        range: &TextureSubresourceRange,
        layout: vk::ImageLayout,
    ) {
        if let Some((texture, _)) = lock(&self.resources.textures).get_mut(id) {
            texture.set_layout(range, layout);
        }
    }

    /// Recycles the recording contexts and frees the destroyed resources of every completed
    /// submission.
    pub(crate) fn collect_completed(&self) -> Result<(), RhiError> {
        let completed = self.queue.completed()?;
        self.recording.recycle(completed);
        loop {
            let garbage = {
                let mut queue = lock(&self.garbage);
                match queue.front() {
                    Some((after, _)) if after.get() <= completed => {
                        queue.pop_front().map(|(_, garbage)| garbage)
                    }
                    _ => None,
                }
            };
            match garbage {
                Some(garbage) => self.destroy_now(garbage),
                None => return Ok(()),
            }
        }
    }

    /// Destroys `garbage` once every submission issued so far has completed.
    pub(crate) fn defer_destruction(&self, garbage: Garbage) -> Result<(), RhiError> {
        let after = self.queue.last_issued();
        if self.queue.completed()? >= after.get() {
            self.destroy_now(garbage);
        } else {
            lock(&self.garbage).push_back((after, garbage));
        }
        Ok(())
    }

    fn destroy_now(&self, garbage: Garbage) {
        let device = &self.context.device;
        // SAFETY: every submission that could use these objects has completed (or the device is
        // idle), and the registry no longer hands them out.
        unsafe {
            match garbage {
                Garbage::Buffer(buffer, suballoc) => {
                    device.destroy_buffer(buffer, None);
                    self.memory.deallocate(&suballoc);
                }
                Garbage::Texture(image, views, suballoc) => {
                    for view in views {
                        device.destroy_image_view(view, None);
                    }
                    device.destroy_image(image, None);
                    self.memory.deallocate(&suballoc);
                }
                Garbage::View(view) => device.destroy_image_view(view, None),
                Garbage::Sampler(sampler) => device.destroy_sampler(sampler, None),
                Garbage::Pipeline(pipeline) => device.destroy_pipeline(pipeline, None),
                Garbage::PipelineLayout(layout) => device.destroy_pipeline_layout(layout, None),
                Garbage::SetLayout(layout) => device.destroy_descriptor_set_layout(layout, None),
                Garbage::QueryPool(pool) => device.destroy_query_pool(pool, None),
                Garbage::DescriptorSet(pool, set) => self.descriptor_pools.free(pool, set),
                Garbage::Swapchain(retired) => self.destroy_retired(retired),
            }
        }
    }
}

/// Layout a texture created with initial data is left in, by usage: the layout of its most
/// specific use (documented on `GraphicsDevice::create_texture`).
pub(crate) fn initial_data_layout(usage: TextureUsage) -> ImageLayout {
    if usage.contains(TextureUsage::SAMPLED) {
        ImageLayout::ShaderReadOnly
    } else if usage.contains(TextureUsage::COLOR_ATTACHMENT) {
        ImageLayout::ColorAttachment
    } else if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        ImageLayout::DepthStencilAttachment
    } else if usage.contains(TextureUsage::STORAGE) {
        ImageLayout::General
    } else if usage.contains(TextureUsage::COPY_SRC) {
        ImageLayout::TransferSrc
    } else {
        ImageLayout::TransferDst
    }
}

pub(crate) use korppi_rhi::lock;

impl VulkanDevice {
    /// Copies `data` into `dst` at `dst_offset` through a temporary host-visible buffer and a
    /// one-time transfer submission (RVK-08). Synchronous until upload rings arrive (FIX-1.2).
    fn upload_via_staging(
        &self,
        dst: vk::Buffer,
        dst_offset: u64,
        data: &[u8],
    ) -> Result<(), RhiError> {
        let staging = self.create_buffer(
            &BufferDescriptor {
                size: data.len() as u64,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryUsage::CpuToGpu,
                label: Some("buffer_upload_staging"),
            },
            Some(data),
        )?;
        let staging_vk = self.vk_buffer(staging)?;
        let result = self.submit_one_time(|device, cmd| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset,
                size: data.len() as u64,
            };
            // SAFETY: `cmd` is recording; both buffers are live, the staging buffer has
            // TRANSFER_SRC and `dst` has TRANSFER_DST usage, and the region is in bounds.
            unsafe { device.cmd_copy_buffer(cmd, staging_vk, dst, &[region]) };
        });
        self.destroy_buffer(staging)?;
        result
    }

    /// Reads `size` bytes at `offset` of `src` through a temporary host-visible buffer and a
    /// one-time transfer submission (device-local memory that is not mappable).
    fn download_via_staging(
        &self,
        src: vk::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        let staging = self.create_buffer(
            &BufferDescriptor {
                size,
                usage: BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuToCpu,
                label: Some("buffer_readback_staging"),
            },
            None,
        )?;
        let staging_vk = self.vk_buffer(staging)?;
        let copied = self.submit_one_time(|device, cmd| {
            let region = vk::BufferCopy {
                src_offset: offset,
                dst_offset: 0,
                size,
            };
            let to_host = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ);
            // SAFETY: `cmd` is recording; `src` has TRANSFER_SRC usage (GpuOnly buffers get it)
            // and the staging buffer TRANSFER_DST; the region is in bounds.
            unsafe {
                device.cmd_copy_buffer(cmd, src, staging_vk, &[region]);
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&to_host)),
                );
            }
        });
        let result = copied.and_then(|()| self.read_buffer(staging, 0, size));
        self.destroy_buffer(staging)?;
        result
    }

    /// The `VkBuffer` behind a live handle.
    pub(crate) fn vk_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, RhiError> {
        lock(&self.resources.buffers)
            .get(handle)
            .map(|(b, _)| b.buffer)
            .ok_or_else(|| RhiError::InvalidHandle(format!("Buffer handle {handle:?} not found")))
    }

    /// Records `record` into a one-time command buffer, submits it and waits for completion.
    pub(crate) fn submit_one_time(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), RhiError> {
        let recording = self.recording.acquire()?;
        let cmd = recording.command_buffer();
        let device = &self.context.device;
        // Submissions no longer drain the queue, so order this work after everything submitted
        // before it and before everything submitted after it.
        let full_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
        let dependency = vk::DependencyInfo::default().memory_barriers(&full_barrier);
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the recording context is owned here; its command buffer is in the initial
        // state and is only reset (by `release`) after its submission completed or failed.
        let result = unsafe {
            (|| {
                device
                    .begin_command_buffer(cmd, &begin_info)
                    .map_err(|e| crate::error::vk_error("vkBeginCommandBuffer", e))?;
                device.cmd_pipeline_barrier2(cmd, &dependency);
                record(device, cmd);
                device.cmd_pipeline_barrier2(cmd, &dependency);
                device
                    .end_command_buffer(cmd)
                    .map_err(|e| crate::error::vk_error("vkEndCommandBuffer", e))?;
                let submission = self.queue.submit(cmd)?;
                self.queue.wait(submission)
            })()
        };
        self.recording.release(recording);
        result
    }
}

impl GraphicsDevice for VulkanDevice {
    fn adapter_info(&self) -> AdapterInfo {
        self.context.capabilities.adapter.clone()
    }

    fn limits(&self) -> DeviceLimits {
        self.context.capabilities.limits.clone()
    }

    fn features(&self) -> DeviceFeatures {
        self.context.capabilities.features
    }

    fn format_capabilities(&self, format: TextureFormat) -> FormatCapabilities {
        self.context.capabilities.format(format)
    }

    fn create_buffer(
        &self,
        desc: &BufferDescriptor,
        initial_data: Option<&[u8]>,
    ) -> Result<BufferHandle, RhiError> {
        desc.validate()?;
        if let Some(data) = initial_data {
            desc.validate_initial_data(data)?;
        }
        let mut usage = to_vk_buffer_usage(desc.usage);
        if desc.memory == MemoryUsage::GpuOnly {
            // Device-local buffers are filled and read through staging copies (RVK-08).
            usage |= vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
        }
        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: `buffer_info` is a valid create info (`validate` rejected a zero size).
        let buffer = unsafe {
            self.context
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| RhiError::OutOfMemory(format!("vkCreateBuffer failed: {e}")))?
        };

        // Memory (RVK-08): by usage, in the buffer pool, dedicated where the driver wants it.
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        // SAFETY: `buffer` was just created.
        unsafe {
            self.context.device.get_buffer_memory_requirements2(
                &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                &mut requirements2,
            )
        };
        let request = AllocationRequest {
            requirements: requirements2.memory_requirements,
            usage: desc.memory,
            kind: ResourceKind::Buffer,
            dedicated: (dedicated.prefers_dedicated_allocation == vk::TRUE
                || dedicated.requires_dedicated_allocation == vk::TRUE)
                .then_some(DedicatedResource::Buffer(buffer)),
        };
        let suballoc = match self.memory.allocate(&request) {
            Ok(suballoc) => suballoc,
            Err(error) => {
                // SAFETY: the buffer is unused and has no memory.
                unsafe { self.context.device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
        // SAFETY: the allocation satisfies the buffer's requirements and is unused.
        if let Err(error) = unsafe {
            self.context
                .device
                .bind_buffer_memory(buffer, suballoc.device_memory, suballoc.offset)
        } {
            // SAFETY: as above.
            unsafe { self.context.device.destroy_buffer(buffer, None) };
            self.memory.deallocate(&suballoc);
            return Err(RhiError::OutOfMemory(format!(
                "vkBindBufferMemory failed: {error}"
            )));
        }

        let vk_buf = VulkanBuffer {
            buffer,
            suballoc,
            size: desc.size,
            usage: desc.usage,
            memory: desc.memory,
            host_access: korppi_rhi::HostAccessLock::new(),
        };
        self.set_object_name(buffer, desc.label);
        let (id, generation) = lock(&self.resources.buffers).insert(vk_buf);
        let handle = BufferHandle::from_parts(self.resources.device, id, generation);

        if let Some(data) = initial_data.filter(|d| !d.is_empty()) {
            // GpuOnly memory is usually not host-visible on discrete GPUs, where the old mapped
            // write silently dropped the data (RVK-08). Always stage it, so this path is also
            // exercised where device-local memory happens to be mappable.
            let upload = if desc.memory == MemoryUsage::GpuOnly || suballoc.mapped_ptr.is_null() {
                self.upload_via_staging(buffer, 0, data)
            } else {
                // SAFETY: the mapping covers the whole suballocation, which is at least
                // `desc.size` bytes, and `data.len() <= desc.size` was validated above.
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), suballoc.mapped_ptr, data.len());
                }
                Ok(())
            };
            if let Err(error) = upload {
                let _ = self.destroy_buffer(handle);
                return Err(error);
            }
        }
        Ok(handle)
    }

    fn destroy_buffer(&self, handle: BufferHandle) -> Result<(), RhiError> {
        let removed = {
            let mut buffers = lock(&self.resources.buffers);
            if let Some((buf, _)) = buffers.get(handle) {
                // Not while a mapping guard exists: its memory must stay valid.
                buf.host_access.acquire_for("destroy_buffer")?;
            }
            buffers.remove(handle)
        };
        if let Some((buf, _gen)) = removed {
            self.invalidate_bind_groups(&[BoundResource::Buffer(handle)]);
            self.defer_destruction(Garbage::Buffer(buf.buffer, buf.suballoc))
        } else {
            Err(RhiError::InvalidHandle(format!(
                "Buffer handle {:?} not found",
                handle
            )))
        }
    }

    fn write_buffer(&self, handle: BufferHandle, offset: u64, data: &[u8]) -> Result<(), RhiError> {
        // Copy what is needed out of the registry: a staged write creates a buffer itself.
        let (buffer, size, memory, mapped_ptr, host_access) = {
            let buffers = lock(&self.resources.buffers);
            let (buf, _) = buffers.get(handle).ok_or_else(|| {
                RhiError::InvalidHandle(format!("Buffer handle {:?} not found", handle))
            })?;
            (
                buf.buffer,
                buf.size,
                buf.memory,
                buf.suballoc.mapped_ptr,
                buf.host_access.clone(),
            )
        };
        if offset
            .checked_add(data.len() as u64)
            .is_none_or(|end| end > size)
        {
            return Err(RhiError::ValidationError(
                "Buffer write out of bounds".into(),
            ));
        }
        if data.is_empty() {
            return Ok(());
        }
        if memory == MemoryUsage::GpuOnly || mapped_ptr.is_null() {
            // Device-local memory is usually not host-visible: stage and copy (RVK-08). GpuOnly
            // always stages, as in `create_buffer`, so the path does not depend on the driver.
            return self.upload_via_staging(buffer, offset, data);
        }
        let _access = host_access.acquire_for("write_buffer")?;
        // SAFETY: the mapping covers the whole suballocation (at least `size` bytes), and
        // `offset + data.len() <= size` was checked with overflow protection above (RVK-19);
        // the host-access lock excludes other host access, and the buffer cannot be destroyed
        // while it is held. Ordering against GPU use of the range is the caller's (as documented
        // on `write_buffer`).
        unsafe {
            let dest = mapped_ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dest, data.len());
        }
        Ok(())
    }

    fn read_buffer(
        &self,
        handle: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        let (buffer, buffer_size, mapped_ptr, host_access) = {
            let buffers = lock(&self.resources.buffers);
            let (buf, _) = buffers.get(handle).ok_or_else(|| {
                RhiError::InvalidHandle(format!("Buffer handle {handle:?} not found"))
            })?;
            (
                buf.buffer,
                buf.size,
                buf.suballoc.mapped_ptr,
                buf.host_access.clone(),
            )
        };
        if offset.checked_add(size).is_none_or(|end| end > buffer_size) {
            return Err(RhiError::ValidationError(
                "Buffer read out of bounds".into(),
            ));
        }
        if mapped_ptr.is_null() {
            // Device-local memory that is not host-visible (discrete GPUs): copy it into a
            // readback buffer first (RVK-08).
            return self.download_via_staging(buffer, offset, size);
        }
        let _access = host_access.acquire_for("read_buffer")?;
        let mut output = vec![0u8; size as usize];
        // SAFETY: the mapping covers the whole buffer and the range was checked; the lock
        // excludes other host access and destruction.
        unsafe {
            let src = mapped_ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(src, output.as_mut_ptr(), size as usize);
        }
        Ok(output)
    }

    fn map_buffer(&self, handle: BufferHandle) -> Result<MappedBuffer<'_>, RhiError> {
        let buffers = lock(&self.resources.buffers);
        let (buf, _) = buffers.get(handle).ok_or_else(|| {
            RhiError::InvalidHandle(format!("Buffer handle {handle:?} not found"))
        })?;
        let ptr = std::ptr::NonNull::new(buf.suballoc.mapped_ptr)
            .filter(|_| buf.memory != MemoryUsage::GpuOnly)
            .ok_or_else(|| {
                RhiError::ValidationError(format!(
                    "{handle:?} is GpuOnly; only CpuToGpu and GpuToCpu buffers can be mapped"
                ))
            })?;
        let guard = buf.host_access.acquire_for("map_buffer")?;
        // SAFETY: the chunk holding the suballocation stays mapped until the memory manager
        // drops, which the device (borrowed for the guard's lifetime) outlives; the buffer
        // cannot be destroyed while the guard's lock is held. The mapping covers `size` bytes.
        Ok(unsafe { MappedBuffer::new(ptr, buf.size, guard) })
    }

    fn create_texture(
        &self,
        desc: &TextureDescriptor,
        init: Option<&[u8]>,
    ) -> Result<TextureHandle, RhiError> {
        self.create_texture_impl(desc, init)
    }

    fn destroy_texture(&self, handle: TextureHandle) -> Result<(), RhiError> {
        // The texture's views die with it; groups using either become unusable.
        let mut destroyed = vec![BoundResource::Texture(handle)];
        if let Some((texture, _)) = lock(&self.resources.textures).get(handle) {
            destroyed.extend(texture.views.iter().map(|&view| BoundResource::View(view)));
        }
        self.destroy_texture_impl(handle)?;
        self.invalidate_bind_groups(&destroyed);
        Ok(())
    }

    fn create_texture_view(
        &self,
        desc: &TextureViewDescriptor,
    ) -> Result<TextureViewHandle, RhiError> {
        self.create_texture_view_impl(desc)
    }

    fn destroy_texture_view(&self, view: TextureViewHandle) -> Result<(), RhiError> {
        self.destroy_texture_view_impl(view)?;
        self.invalidate_bind_groups(&[BoundResource::View(view)]);
        Ok(())
    }

    fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayoutHandle, RhiError> {
        self.create_bind_group_layout_impl(desc)
    }

    fn create_query_set(
        &self,
        desc: &korppi_rhi::QuerySetDescriptor,
    ) -> Result<korppi_rhi::QuerySetHandle, RhiError> {
        self.create_query_set_impl(desc)
    }

    fn destroy_query_set(&self, query_set: korppi_rhi::QuerySetHandle) -> Result<(), RhiError> {
        self.destroy_query_set_impl(query_set)
    }

    fn destroy_bind_group_layout(&self, layout: BindGroupLayoutHandle) -> Result<(), RhiError> {
        self.destroy_bind_group_layout_impl(layout)
    }

    fn create_bind_group(&self, desc: &BindGroupDescriptor) -> Result<BindGroupHandle, RhiError> {
        self.create_bind_group_impl(desc)
    }

    fn destroy_bind_group(&self, group: BindGroupHandle) -> Result<(), RhiError> {
        self.destroy_bind_group_impl(group)
    }

    fn read_texture(&self, handle: TextureHandle, mip: u32) -> Result<Vec<u8>, RhiError> {
        self.read_texture_impl(handle, mip)
    }

    fn create_sampler(&self, desc: &SamplerDescriptor) -> Result<SamplerHandle, RhiError> {
        desc.validate()?;
        let sampler_info = sampler_create_info(
            desc,
            self.context.sampler_anisotropy,
            self.context.limits.max_sampler_anisotropy,
            self.context.limits.max_sampler_lod_bias,
        );

        // SAFETY: `sampler_info` is valid: the LOD range was validated, the bias and anisotropy
        // are clamped to the device limits, and anisotropy is only enabled with the feature
        // (`sampler_create_info`).
        let sampler = unsafe {
            self.context
                .device
                .create_sampler(&sampler_info, None)
                .map_err(|e| RhiError::OutOfMemory(format!("vkCreateSampler failed: {e}")))?
        };
        self.set_object_name(sampler, desc.label);
        let (id, generation) = lock(&self.resources.samplers).insert(VulkanSampler { sampler });
        Ok(SamplerHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn destroy_sampler(&self, handle: SamplerHandle) -> Result<(), RhiError> {
        let removed = lock(&self.resources.samplers).remove(handle);
        if let Some((samp, _)) = removed {
            self.invalidate_bind_groups(&[BoundResource::Sampler(handle)]);
            self.defer_destruction(Garbage::Sampler(samp.sampler))
        } else {
            Err(RhiError::InvalidHandle(format!(
                "Sampler handle {:?} not found",
                handle
            )))
        }
    }

    fn create_shader(&self, desc: &ShaderDescriptor) -> Result<ShaderHandle, RhiError> {
        // Native byte order, framing and entry point checked: drivers crash on malformed
        // modules (RVK-18).
        let code_words = desc.spirv()?.ok_or_else(|| {
            RhiError::ShaderCompilationFailed(
                "the Vulkan backend takes SPIR-V; compile GLSL to SPIR-V first".into(),
            )
        })?;

        let create_info = vk::ShaderModuleCreateInfo::default().code(&code_words);
        // SAFETY: `code_words` is the SPIR-V as 32-bit words and outlives the call.
        let module = unsafe {
            self.context
                .device
                .create_shader_module(&create_info, None)
                .map_err(|e| {
                    RhiError::ShaderCompilationFailed(format!("vkCreateShaderModule failed: {e}"))
                })?
        };
        self.set_object_name(module, desc.label);

        let entry_point = CString::new(desc.entry_point).map_err(|_| {
            RhiError::ShaderCompilationFailed(
                "Invalid entry point string (contains null byte)".into(),
            )
        })?;
        let (id, generation) = lock(&self.resources.shaders).insert(VulkanShader {
            module,
            stage: desc.stage,
            entry_point,
        });
        Ok(ShaderHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn destroy_shader(&self, handle: ShaderHandle) -> Result<(), RhiError> {
        let removed = lock(&self.resources.shaders).remove(handle);
        if let Some((shader, _)) = removed {
            // SAFETY: pipelines do not reference shader modules after creation, so a module may
            // be destroyed while pipelines built from it are in use.
            unsafe {
                self.context
                    .device
                    .destroy_shader_module(shader.module, None);
            }
            Ok(())
        } else {
            Err(RhiError::InvalidHandle(format!(
                "Shader handle {:?} not found",
                handle
            )))
        }
    }

    fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        self.create_graphics_pipeline_impl(desc)
    }

    fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        self.create_compute_pipeline_impl(desc)
    }

    fn destroy_pipeline(&self, handle: PipelineHandle) -> Result<(), RhiError> {
        self.destroy_pipeline_impl(handle)
    }

    fn pipeline_cache_data(&self) -> Result<Vec<u8>, RhiError> {
        self.pipeline_cache.data()
    }

    fn load_pipeline_cache(&self, data: &[u8]) -> Result<bool, RhiError> {
        self.pipeline_cache.load(data)
    }

    fn create_command_recorder(&self) -> Result<Box<dyn CommandBufferRecorder>, RhiError> {
        self.collect_completed()?;
        // Recorders, command buffers and validators are pooled: steady-state frames allocate
        // none of them (RHI-11).
        let recorder = VulkanCommandBufferRecorder::new(
            self.context.clone(),
            self.resources.clone(),
            self.recording.acquire()?,
            Arc::clone(&self.recording),
            RecorderPools {
                recorders: Arc::downgrade(&self.recorder_pool),
                command_buffers: Arc::downgrade(&self.command_buffer_pool),
            },
        );
        let recorder: Box<dyn CommandBufferRecorder> = match self.recorder_pool.take() {
            Some(mut boxed) => {
                *boxed = recorder;
                boxed
            }
            None => Box::new(recorder),
        };
        // Every recorder is validated, in every build (RHI-16).
        Ok(validate_recorder_pooled(
            recorder,
            self.resources.clone(),
            &self.validator_pool,
        ))
    }

    fn submit(&self, command_buffer: Box<dyn CommandBuffer>) -> Result<SubmissionId, RhiError> {
        let command_buffer: Box<dyn std::any::Any> = command_buffer;
        let mut cmd_box = command_buffer
            .downcast::<VulkanCommandBuffer>()
            .map_err(|_| {
                RhiError::InvalidHandle("command buffer was not recorded by a Vulkan device".into())
            })?;
        if !Arc::ptr_eq(&cmd_box.context, &self.context) {
            return Err(RhiError::InvalidHandle(
                "command buffer was recorded by a different Vulkan device".into(),
            ));
        }

        let recording = cmd_box.take_recording().ok_or_else(|| {
            RhiError::InvalidHandle("command buffer has no recording context".into())
        })?;
        // Swapchain images the commands use: wait for their acquisition (FIX-1.3, RVK-02).
        let waits = match self.swapchain_waits(&recording.swapchain_uses) {
            Ok(waits) => waits,
            Err(error) => {
                self.recording.release(recording);
                return Err(error);
            }
        };
        match self
            .queue
            .submit_batch(&[recording.command_buffer()], &waits, &[])
        {
            Ok(submission) => {
                self.swapchains_submitted(&recording.swapchain_uses, submission);
                for (texture, range, layout) in &recording.layout_changes {
                    self.set_texture_range_layout(*texture, range, *layout);
                }
                // Recycled by `collect_completed` once the submission has completed.
                self.recording.retire(recording, submission);
                self.command_buffer_pool.give(cmd_box);
                self.frames.record(submission);
                self.collect_completed()?;
                Ok(submission)
            }
            Err(error) => {
                // Not pending: the submission failed.
                self.recording.release(recording);
                Err(error)
            }
        }
    }

    fn is_submission_complete(&self, submission: SubmissionId) -> Result<bool, RhiError> {
        self.queue.is_complete(submission)
    }

    fn wait_for_submission(&self, submission: SubmissionId) -> Result<(), RhiError> {
        self.queue.wait(submission)?;
        self.collect_completed()
    }

    fn wait_idle(&self) -> Result<(), RhiError> {
        self.queue.wait_idle()?;
        self.collect_completed()
    }

    fn begin_frame(&self) -> Result<FrameIndex, RhiError> {
        let (frame, wait_for) = self.frames.begin_frame();
        if let Some(submission) = wait_for {
            self.queue.wait(submission)?;
        }
        self.collect_completed()?;
        Ok(frame)
    }

    fn create_swapchain(
        &self,
        target: Arc<dyn PresentTarget>,
        desc: &SwapchainDescriptor,
    ) -> Result<SwapchainHandle, RhiError> {
        self.create_swapchain_impl(target, desc)
    }

    fn destroy_swapchain(&self, swapchain: SwapchainHandle) -> Result<(), RhiError> {
        self.destroy_swapchain_impl(swapchain)
    }

    fn resize_swapchain(
        &self,
        swapchain: SwapchainHandle,
        width: u32,
        height: u32,
    ) -> Result<(), RhiError> {
        self.resize_swapchain_impl(swapchain, width, height)
    }

    fn swapchain_info(&self, swapchain: SwapchainHandle) -> Result<SwapchainInfo, RhiError> {
        self.swapchain_info_impl(swapchain)
    }

    fn acquire_next_image(
        &self,
        swapchain: SwapchainHandle,
    ) -> Result<Option<SwapchainImage>, RhiError> {
        self.acquire_impl(swapchain)
    }

    fn present(&self, image: &SwapchainImage) -> Result<(), RhiError> {
        self.present_impl(image)
    }

    fn api_validation_enabled(&self) -> bool {
        self.context.validation_enabled
    }

    fn take_diagnostics(&self) -> Vec<BackendDiagnostic> {
        self.context.diagnostics.take()
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        for (_, garbage) in lock(&self.garbage).drain(..).collect::<Vec<_>>() {
            self.destroy_now(garbage);
        }
        self.destroy_all_swapchains();

        for (_, (set, _)) in lock(&self.resources.query_sets).drain() {
            // SAFETY: the device is idle.
            unsafe { self.context.device.destroy_query_pool(set.pool, None) };
        }
        // Bind groups' sets are freed with their pools.
        lock(&self.resources.bind_groups).clear();
        for (_, (layout, _)) in lock(&self.resources.bind_group_layouts).drain() {
            // SAFETY: the device is idle.
            unsafe {
                self.context
                    .device
                    .destroy_descriptor_set_layout(layout.layout, None)
            };
        }

        // Pipelines share their Vulkan objects: destroy each once.
        lock(&self.resources.pipelines).clear();
        let (pipelines, layouts) = self.pipeline_objects.drain();
        // SAFETY: the device is idle.
        unsafe {
            for pipeline in pipelines {
                self.context.device.destroy_pipeline(pipeline, None);
            }
            for layout in layouts {
                self.context.device.destroy_pipeline_layout(layout, None);
            }
        }

        let mut shaders = lock(&self.resources.shaders);
        for (_, (s, _)) in shaders.drain() {
            // SAFETY: the device is idle and the registry held the only handle.
            unsafe {
                self.context.device.destroy_shader_module(s.module, None);
            }
        }

        let mut samplers = lock(&self.resources.samplers);
        for (_, (s, _)) in samplers.drain() {
            // SAFETY: the device is idle and the registry held the only handle.
            unsafe {
                self.context.device.destroy_sampler(s.sampler, None);
            }
        }

        for (_, (view, _)) in lock(&self.resources.texture_views).drain() {
            // SAFETY: the device is idle.
            unsafe { self.context.device.destroy_image_view(view.view, None) };
        }
        let mut textures = lock(&self.resources.textures);
        for (_, (t, _)) in textures.drain() {
            // Swapchain images and their views were destroyed with their swapchains above.
            if let TextureStorage::Device(_) = t.storage {
                // SAFETY: the device is idle; the views belong to the image.
                unsafe {
                    for view in t.attachment_views.into_iter().flatten() {
                        self.context.device.destroy_image_view(view, None);
                    }
                    if t.view != vk::ImageView::null() {
                        self.context.device.destroy_image_view(t.view, None);
                    }
                    self.context.device.destroy_image(t.image, None);
                }
            }
        }

        let mut buffers = lock(&self.resources.buffers);
        for (_, (b, _)) in buffers.drain() {
            // SAFETY: the device is idle and the registry held the only handle; the memory
            // chunks are freed with the allocator afterwards.
            unsafe {
                self.context.device.destroy_buffer(b.buffer, None);
            }
        }
    }
}
