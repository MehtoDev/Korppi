use crate::context::VulkanContext;
use crate::recording::{RecordingContext, RecordingContexts};
use crate::resource::{TextureStorage, VulkanResourceRegistry};
use crate::types::{
    barrier_aspect, buffer_row_length_texels, copy_aspect, push_constant_segments,
    to_vk_access_flags, to_vk_image_layout, to_vk_load_op, to_vk_pipeline_stage_flags,
    to_vk_store_op, to_vk_subresource_range,
};
use ash::vk;
use korppi_rhi::{
    BindGroupHandle, BufferHandle, BufferTextureCopy, CommandBuffer, CommandBufferRecorder,
    FilterMode, IndexType, PipelineBarrier, PipelineHandle, QuerySetHandle, RenderingDescriptor,
    RhiError, TextureBlit, TextureClear, TextureCopy, TextureDimension, TextureFormat,
    TextureHandle, TextureInfo, TextureSubresourceRange, MAX_BIND_GROUPS,
    MAX_DYNAMIC_OFFSETS_PER_GROUP, MAX_PUSH_CONSTANTS_SIZE,
};
use korppi_rhi::{BoxPool, SmallVec};
use std::sync::Arc;

/// Concrete Vulkan executable command buffer.
///
/// Owns the recording context (command pool and descriptor pools) it was recorded with until it
/// is submitted; dropped unsubmitted, it returns the context for reuse (RVK-17).
pub struct VulkanCommandBuffer {
    pub(crate) context: Arc<VulkanContext>,
    recording: Option<RecordingContext>,
    contexts: Arc<RecordingContexts>,
}

/// A swapchain image used by a command buffer (FIX-1.3, RVK-02).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SwapchainUse {
    /// ID of the swapchain (`SwapchainHandle::id`).
    pub(crate) swapchain: u32,
    /// Index of the image in the swapchain.
    pub(crate) index: u32,
    /// Texture ID the image was used through.
    pub(crate) texture: u32,
    /// Layout the command buffer's last barrier on the image left it in, if it has one.
    pub(crate) final_layout: Option<vk::ImageLayout>,
}

impl VulkanCommandBuffer {
    /// Takes the recording context for submission; the submitter becomes responsible for it.
    pub(crate) fn take_recording(&mut self) -> Option<RecordingContext> {
        self.recording.take()
    }
}

impl CommandBuffer for VulkanCommandBuffer {}

impl Drop for VulkanCommandBuffer {
    fn drop(&mut self) {
        if let Some(recording) = self.recording.take() {
            // Never submitted, so not pending.
            self.contexts.release(recording);
        }
    }
}

/// Concrete Vulkan command buffer recorder.
///
/// Records into a recording context of its own (RVK-11): no command or descriptor pool is shared
/// with another recorder, so recorders on different threads need no synchronization.
pub struct VulkanCommandBufferRecorder {
    /// Where `finish` returns this box, and where command buffer boxes come from (RHI-11).
    pools: RecorderPools,
    context: Arc<VulkanContext>,
    resources: Arc<VulkanResourceRegistry>,
    /// `Some` until `finish` hands the context to the command buffer.
    recording: Option<RecordingContext>,
    command_buffer: vk::CommandBuffer,
    contexts: Arc<RecordingContexts>,
    current_pipeline_layout: vk::PipelineLayout,
    /// Descriptor sets in the current pipeline layout.
    current_set_count: u32,
    current_bind_point: vk::PipelineBindPoint,
    current_push_ranges: Option<Arc<[vk::PushConstantRange]>>,
    /// Bind groups set per index: descriptor set and dynamic offsets (RHI-07).
    bound_groups: [Option<BoundGroup>; MAX_BIND_GROUPS],
    /// Indices whose group must be (re)bound before the next draw or dispatch.
    dirty_groups: [bool; MAX_BIND_GROUPS],
}

impl Drop for VulkanCommandBufferRecorder {
    fn drop(&mut self) {
        if let Some(recording) = self.recording.take() {
            // Dropped before `finish`: the command buffer was never submitted.
            self.contexts.release(recording);
        }
    }
}

impl VulkanCommandBufferRecorder {
    /// Creates a recorder that records into `recording`, which it returns to `contexts` if it
    /// is dropped before `finish`.
    pub(crate) fn new(
        context: Arc<VulkanContext>,
        resources: Arc<VulkanResourceRegistry>,
        recording: RecordingContext,
        contexts: Arc<RecordingContexts>,
        pools: RecorderPools,
    ) -> Self {
        Self {
            pools,
            context,
            resources,
            command_buffer: recording.command_buffer(),
            recording: Some(recording),
            contexts,
            current_pipeline_layout: vk::PipelineLayout::null(),
            current_set_count: 0,
            current_bind_point: vk::PipelineBindPoint::GRAPHICS,
            current_push_ranges: None,
            bound_groups: [None; MAX_BIND_GROUPS],
            dirty_groups: [false; MAX_BIND_GROUPS],
        }
    }

    /// The recording context; present from creation until `finish`.
    fn context_mut(&mut self) -> &mut RecordingContext {
        self.recording
            .as_mut()
            .expect("the recording context lives until finish")
    }

    // Registry lookups. Each takes exactly one registry lock and releases it before returning,
    // so a recorder never holds two locks at once or a lock across a Vulkan call (RVK-11).

    fn buffer(&self, handle: BufferHandle) -> Option<vk::Buffer> {
        lock(&self.resources.buffers)
            .get(handle)
            .map(|(buffer, _)| buffer.buffer)
    }

    /// The single-mip, single-layer attachment view of `handle` at (`mip`, `layer`); records
    /// swapchain image uses like [`Self::texture`].
    fn attachment_view(
        &mut self,
        handle: TextureHandle,
        mip: u32,
        layer: u32,
    ) -> Option<vk::ImageView> {
        self.texture(handle, None)?;
        lock(&self.resources.textures)
            .get(handle)
            .and_then(|(texture, _)| texture.attachment_view(mip, layer))
    }

    /// Looks up a texture and, if it is a swapchain image, records the use (with the layout a
    /// barrier leaves it in, if `final_layout` is given).
    fn texture(
        &mut self,
        handle: TextureHandle,
        final_layout: Option<vk::ImageLayout>,
    ) -> Option<(vk::Image, vk::ImageView, TextureFormat, bool)> {
        let (image, view, format, swapchain) = {
            let textures = lock(&self.resources.textures);
            let (texture, _) = textures.get(handle)?;
            let swapchain = match texture.storage {
                TextureStorage::Swapchain { swapchain, index } => Some((swapchain, index)),
                TextureStorage::Device(_) => None,
            };
            (texture.image, texture.view, texture.format, swapchain)
        };
        if let Some((swapchain, index)) = swapchain {
            let uses = &mut self.context_mut().swapchain_uses;
            match uses.iter_mut().find(|u| u.texture == handle.id()) {
                Some(used) => used.final_layout = final_layout.or(used.final_layout),
                None => uses.push(SwapchainUse {
                    swapchain,
                    index,
                    texture: handle.id(),
                    final_layout,
                }),
            }
        }
        Some((image, view, format, swapchain.is_some()))
    }

    fn texture_info(&self, handle: TextureHandle) -> Option<TextureInfo> {
        lock(&self.resources.textures)
            .get(handle)
            .map(|(texture, _)| texture.info)
    }

    /// Binds the groups set since the last draw or dispatch (or since the pipeline changed) for
    /// the sets the current pipeline layout has.
    fn flush_bind_groups(&mut self) {
        if self.current_pipeline_layout == vk::PipelineLayout::null() {
            return;
        }
        for index in 0..self.current_set_count.min(MAX_BIND_GROUPS as u32) {
            let slot = index as usize;
            if !std::mem::take(&mut self.dirty_groups[slot]) {
                continue;
            }
            let Some(group) = self.bound_groups[slot] else {
                continue;
            };
            // SAFETY: the command buffer is recording and owned by this recorder; the set is
            // live (bind groups are destroyed only after the submissions using them complete).
            unsafe {
                self.context.device.cmd_bind_descriptor_sets(
                    self.command_buffer,
                    self.current_bind_point,
                    self.current_pipeline_layout,
                    index,
                    &[group.set],
                    &group.dynamic_offsets[..group.dynamic_count],
                );
            }
        }
    }
}

/// A bind group set on a recorder.
#[derive(Copy, Clone)]
struct BoundGroup {
    set: vk::DescriptorSet,
    dynamic_offsets: [u32; MAX_DYNAMIC_OFFSETS_PER_GROUP],
    dynamic_count: usize,
}

impl CommandBufferRecorder for VulkanCommandBufferRecorder {
    fn begin(&mut self) -> Result<(), RhiError> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: the command buffer belongs to this recorder and is not recording or pending
        // (recorders are reset only after their submission completed).
        unsafe {
            self.context
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| crate::error::vk_error("vkBeginCommandBuffer", e))?;
            // Dynamic state every graphics pipeline declares, at its RHI defaults.
            self.context.device.cmd_set_stencil_reference(
                self.command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                0,
            );
            self.context
                .device
                .cmd_set_blend_constants(self.command_buffer, &[0.0; 4]);
        }
        Ok(())
    }

    fn begin_rendering(&mut self, desc: &RenderingDescriptor) -> Result<(), RhiError> {
        // Vulkan 1.3 Core Dynamic Rendering
        let render_area = vk::Rect2D {
            offset: vk::Offset2D {
                x: desc.offset_x,
                y: desc.offset_y,
            },
            extent: vk::Extent2D {
                width: desc.width,
                height: desc.height,
            },
        };

        // Inline storage: recording allocates nothing (RHI-11, RVK-15).
        let mut color_attachments: SmallVec<
            [vk::RenderingAttachmentInfo; korppi_rhi::MAX_COLOR_ATTACHMENTS],
        > = SmallVec::new();
        for att in &desc.color_attachments {
            let clear_val = att.clear_color.unwrap_or([0.0, 0.0, 0.0, 1.0]);
            // Attachments render through single-mip, single-layer views (RVK-13).
            let image_view = self
                .attachment_view(att.texture, att.mip_level, att.layer)
                .ok_or_else(|| {
                    RhiError::InvalidHandle(format!(
                        "Color attachment {:?} mip {} layer {} is not an attachment view",
                        att.texture, att.mip_level, att.layer
                    ))
                })?;

            let mut vk_att = vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(to_vk_load_op(att.load_op))
                .store_op(to_vk_store_op(att.store_op))
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: clear_val },
                });
            if let Some(resolve) = att.resolve_target {
                let (resolve_view, format) = self
                    .attachment_view(resolve, 0, 0)
                    .zip(self.texture(resolve, None).map(|(_, _, format, _)| format))
                    .ok_or_else(|| {
                        RhiError::InvalidHandle(format!(
                            "resolve target {resolve:?} is not an attachment"
                        ))
                    })?;
                // Integer formats cannot be averaged.
                let mode = if format.is_integer() {
                    vk::ResolveModeFlags::SAMPLE_ZERO
                } else {
                    vk::ResolveModeFlags::AVERAGE
                };
                vk_att = vk_att
                    .resolve_mode(mode)
                    .resolve_image_view(resolve_view)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            }
            color_attachments.push(vk_att);
        }

        let mut depth_attachment = None;
        let mut stencil_attachment = None;
        if let Some(ref depth) = desc.depth_attachment {
            let clear_d = depth.clear_depth.unwrap_or(0.0);
            let clear_s = depth.clear_stencil.unwrap_or(0);
            let image_view = self
                .attachment_view(depth.texture, depth.mip_level, depth.layer)
                .ok_or_else(|| {
                    RhiError::InvalidHandle(format!(
                        "Depth attachment {:?} mip {} layer {} is not an attachment view",
                        depth.texture, depth.mip_level, depth.layer
                    ))
                })?;
            let has_stencil = self
                .texture(depth.texture, None)
                .is_some_and(|(_, _, format, _)| format.is_stencil());
            let clear_value = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: clear_d,
                    stencil: clear_s,
                },
            };
            depth_attachment = Some(
                vk::RenderingAttachmentInfo::default()
                    .image_view(image_view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(to_vk_load_op(depth.load_op))
                    .store_op(to_vk_store_op(depth.store_op))
                    .clear_value(clear_value),
            );
            if has_stencil {
                stencil_attachment = Some(
                    vk::RenderingAttachmentInfo::default()
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .load_op(to_vk_load_op(depth.stencil_load_op))
                        .store_op(to_vk_store_op(depth.stencil_store_op))
                        .clear_value(clear_value),
                );
            }
        }

        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);

        if let Some(ref d_att) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(d_att);
        }
        if let Some(ref s_att) = stencil_attachment {
            rendering_info = rendering_info.stencil_attachment(s_att);
        }

        // SAFETY: the command buffer is recording and owned by this recorder; every attachment view
        // was looked up live above, and `rendering_info` and its attachments outlive the call.
        unsafe {
            self.context
                .device
                .cmd_begin_rendering(self.command_buffer, &rendering_info);
        }
        Ok(())
    }

    fn end_rendering(&mut self) -> Result<(), RhiError> {
        // SAFETY: the command buffer is recording and owned by this recorder.
        unsafe {
            self.context.device.cmd_end_rendering(self.command_buffer);
        }
        Ok(())
    }

    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        // Enforce ADR-013: Negative viewport height transforms NDC Y up without shader branching.
        // If caller passes positive height (standard cross-backend coordinates), internally map
        // to Vulkan flipped coordinates: y_vk = y + height, h_vk = -height.
        // If caller passes negative height (legacy explicit Vulkan coordinates), preserve it directly.
        let (vk_y, vk_height) = if height > 0.0 {
            (y + height, -height)
        } else {
            (y, height)
        };
        let viewport = vk::Viewport {
            x,
            y: vk_y,
            width,
            height: vk_height,
            min_depth,
            max_depth,
        };
        // SAFETY: the command buffer is recording and owned by this recorder; viewports are
        // dynamic state of every graphics pipeline.
        unsafe {
            self.context
                .device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
        }
    }

    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        };
        // SAFETY: the command buffer is recording and owned by this recorder; scissors are
        // dynamic state of every graphics pipeline.
        unsafe {
            self.context
                .device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        // SAFETY: the command buffer is recording and owned by this recorder.
        unsafe {
            self.context.device.cmd_set_stencil_reference(
                self.command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                reference,
            );
        }
    }

    fn set_blend_constant(&mut self, color: [f32; 4]) {
        // SAFETY: the command buffer is recording and owned by this recorder.
        unsafe {
            self.context
                .device
                .cmd_set_blend_constants(self.command_buffer, &color);
        }
    }

    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        let found = lock(&self.resources.pipelines)
            .get(pipeline)
            .map(|(pipe, _)| {
                (
                    pipe.pipeline,
                    pipe.layout,
                    pipe.bind_point,
                    Arc::clone(&pipe.push_constant_ranges),
                    pipe.set_count,
                    pipe.dynamic,
                )
            });
        if let Some((vk_pipeline, layout, bind_point, push_ranges, set_count, dynamic)) = found {
            self.current_pipeline_layout = layout;
            self.current_bind_point = bind_point;
            self.current_push_ranges = Some(push_ranges);
            // SAFETY: the command buffer is recording and owned by this recorder.
            unsafe {
                self.context
                    .device
                    .cmd_bind_pipeline(self.command_buffer, bind_point, vk_pipeline);
                // Graphics pipelines share PSOs that differ only in dynamic state (RVK-16).
                if let Some(dynamic) = dynamic {
                    dynamic.record(&self.context.device, self.command_buffer);
                }
            }
            self.current_set_count = set_count;
            // A new layout disturbs bound sets; bind them again for it.
            self.dirty_groups = [true; MAX_BIND_GROUPS];
        }
    }

    fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[BufferHandle],
        offsets: &[u64],
    ) {
        if buffers.len() != offsets.len() {
            tracing::error!(
                "bind_vertex_buffers: {} buffers but {} offsets; not bound",
                buffers.len(),
                offsets.len()
            );
            return;
        }
        let vk_buffers: Option<SmallVec<[vk::Buffer; 16]>> = {
            let reg = lock(&self.resources.buffers);
            buffers
                .iter()
                .map(|b| reg.get(b).map(|(buf, _)| buf.buffer))
                .collect()
        };
        let Some(vk_buffers) = vk_buffers else {
            tracing::error!("bind_vertex_buffers: unknown buffer handle in {buffers:?}; not bound");
            return;
        };

        // SAFETY: the command buffer is recording and owned by this recorder; every buffer is
        // live, and there is one offset per buffer (ash passes one count for both).
        unsafe {
            self.context.device.cmd_bind_vertex_buffers(
                self.command_buffer,
                first_binding,
                &vk_buffers,
                offsets,
            );
        }
    }

    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64, index_type: IndexType) {
        let vk_index_type = match index_type {
            IndexType::Uint16 => vk::IndexType::UINT16,
            IndexType::Uint32 => vk::IndexType::UINT32,
        };
        if let Some(vk_buffer) = self.buffer(buffer) {
            // SAFETY: the command buffer is recording and owned by this recorder.
            unsafe {
                self.context.device.cmd_bind_index_buffer(
                    self.command_buffer,
                    vk_buffer,
                    offset,
                    vk_index_type,
                );
            }
        }
    }

    fn set_bind_group(&mut self, index: u32, group: BindGroupHandle, dynamic_offsets: &[u32]) {
        let slot = index as usize;
        if slot >= MAX_BIND_GROUPS || dynamic_offsets.len() > MAX_DYNAMIC_OFFSETS_PER_GROUP {
            return;
        }
        let set = lock(&self.resources.bind_groups)
            .get(group)
            .filter(|(g, _)| g.info.live)
            .map(|(g, _)| g.set);
        let Some(set) = set else {
            tracing::error!("set_bind_group: {group:?} is unknown or uses destroyed resources");
            return;
        };
        let mut offsets = [0; MAX_DYNAMIC_OFFSETS_PER_GROUP];
        offsets[..dynamic_offsets.len()].copy_from_slice(dynamic_offsets);
        self.bound_groups[slot] = Some(BoundGroup {
            set,
            dynamic_offsets: offsets,
            dynamic_count: dynamic_offsets.len(),
        });
        self.dirty_groups[slot] = true;
    }

    fn push_constants(&mut self, offset: u32, data: &[u8]) -> Result<(), RhiError> {
        if (offset as usize)
            .checked_add(data.len())
            .is_none_or(|end| end > MAX_PUSH_CONSTANTS_SIZE)
        {
            return Err(RhiError::ValidationError(format!(
                "Push constants size {} + offset {} exceeds maximum 128 bytes",
                data.len(),
                offset
            )));
        }

        if let Some(ranges) = &self.current_push_ranges {
            // Stage flags must match the layout's ranges exactly (RVK-10); a push spanning
            // ranges of different stages is split into one call per segment.
            let (device, command_buffer, layout) = (
                &self.context.device,
                self.command_buffer,
                self.current_pipeline_layout,
            );
            push_constant_segments(ranges, offset, data.len() as u32, |stages, start, size| {
                let bytes = &data[(start - offset) as usize..(start - offset + size) as usize];
                // SAFETY: the command buffer is recording and `layout` is the bound pipeline's
                // layout, whose ranges cover `start..start + size` for exactly `stages`.
                unsafe {
                    device.cmd_push_constants(command_buffer, layout, stages, start, bytes);
                }
            });
        }
        Ok(())
    }

    fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.flush_bind_groups();
        // SAFETY: the command buffer is recording and owned by this recorder.
        unsafe {
            self.context.device.cmd_draw(
                self.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.flush_bind_groups();
        // SAFETY: the command buffer is recording and owned by this recorder.
        unsafe {
            self.context.device.cmd_draw_indexed(
                self.command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.flush_bind_groups();
        // SAFETY: the command buffer is recording and owned by this recorder.
        unsafe {
            self.context.device.cmd_dispatch(
                self.command_buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        self.flush_bind_groups();
        if let Some(buffer) = self.buffer(buffer) {
            // SAFETY: the command buffer is recording inside a pass with a graphics pipeline;
            // the argument range and features were validated (debug builds).
            unsafe {
                self.context.device.cmd_draw_indirect(
                    self.command_buffer,
                    buffer,
                    offset,
                    draw_count,
                    stride,
                );
            }
        }
    }

    fn draw_indexed_indirect(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        self.flush_bind_groups();
        if let Some(buffer) = self.buffer(buffer) {
            // SAFETY: as in `draw_indirect`; an index buffer is bound.
            unsafe {
                self.context.device.cmd_draw_indexed_indirect(
                    self.command_buffer,
                    buffer,
                    offset,
                    draw_count,
                    stride,
                );
            }
        }
    }

    fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        self.flush_bind_groups();
        if let Some(buffer) = self.buffer(buffer) {
            // SAFETY: the command buffer is recording outside a pass with a compute pipeline.
            unsafe {
                self.context
                    .device
                    .cmd_dispatch_indirect(self.command_buffer, buffer, offset);
            }
        }
    }

    fn write_timestamp(&mut self, query_set: QuerySetHandle, index: u32) {
        let pool = lock(&self.resources.query_sets)
            .get(query_set)
            .map(|(q, _)| q.pool);
        if let Some(pool) = pool {
            // SAFETY: the command buffer is recording; the query is in range and was reset by
            // the set's creation or last resolve.
            unsafe {
                self.context.device.cmd_write_timestamp2(
                    self.command_buffer,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    pool,
                    index,
                );
            }
        }
    }

    fn resolve_query_set(
        &mut self,
        query_set: QuerySetHandle,
        first: u32,
        count: u32,
        dst: BufferHandle,
        dst_offset: u64,
    ) {
        let pool = lock(&self.resources.query_sets)
            .get(query_set)
            .map(|(q, _)| q.pool);
        if let (Some(pool), Some(dst)) = (pool, self.buffer(dst)) {
            // SAFETY: the command buffer is recording outside a pass; the range, the
            // destination's COPY_DST usage and its size were validated (debug builds). The
            // copy waits for the results; the reset makes the queries writable again.
            unsafe {
                let device = &self.context.device;
                device.cmd_copy_query_pool_results(
                    self.command_buffer,
                    pool,
                    first,
                    count,
                    dst,
                    dst_offset,
                    8,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                );
                device.cmd_reset_query_pool(self.command_buffer, pool, first, count);
            }
        }
    }

    fn push_debug_group(&mut self, label: &str) {
        if let (Some(debug_utils), Ok(name)) =
            (&self.context.debug_utils, std::ffi::CString::new(label))
        {
            let info = vk::DebugUtilsLabelEXT::default().label_name(&name);
            // SAFETY: the command buffer is recording; `info` outlives the call.
            unsafe { debug_utils.cmd_begin_debug_utils_label(self.command_buffer, &info) };
        }
    }

    fn pop_debug_group(&mut self) {
        if let Some(debug_utils) = &self.context.debug_utils {
            // SAFETY: the command buffer is recording; validation checks a group is open.
            unsafe { debug_utils.cmd_end_debug_utils_label(self.command_buffer) };
        }
    }

    fn insert_debug_marker(&mut self, label: &str) {
        if let (Some(debug_utils), Ok(name)) =
            (&self.context.debug_utils, std::ffi::CString::new(label))
        {
            let info = vk::DebugUtilsLabelEXT::default().label_name(&name);
            // SAFETY: the command buffer is recording; `info` outlives the call.
            unsafe { debug_utils.cmd_insert_debug_utils_label(self.command_buffer, &info) };
        }
    }

    fn pipeline_barrier(&mut self, barrier: &PipelineBarrier) {
        // Vulkan 1.3 Synchronization2: vkCmdPipelineBarrier2
        let mut image_barriers: SmallVec<[vk::ImageMemoryBarrier2; 8]> = SmallVec::new();
        for ib in &barrier.image_barriers {
            let new_layout = to_vk_image_layout(ib.new_layout);
            let Some((img, _, format, swapchain_image)) =
                self.texture(ib.texture, Some(new_layout))
            else {
                continue;
            };
            self.context_mut()
                .layout_changes
                .push((ib.texture, ib.range, new_layout));
            let mut src_stages = to_vk_pipeline_stage_flags(ib.src_stages);
            if swapchain_image {
                // The batch that first uses an acquired image waits for its acquire semaphore at
                // COLOR_ATTACHMENT_OUTPUT. Including that stage chains the layout transition
                // after the wait even when the caller passed TOP_OF_PIPE, as is usual for a
                // transition out of UNDEFINED.
                src_stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
            }
            image_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stages)
                    .src_access_mask(to_vk_access_flags(ib.src_access))
                    .dst_stage_mask(to_vk_pipeline_stage_flags(ib.dst_stages))
                    .dst_access_mask(to_vk_access_flags(ib.dst_access))
                    .old_layout(to_vk_image_layout(ib.old_layout))
                    .new_layout(new_layout)
                    .image(img)
                    .subresource_range(to_vk_subresource_range(&ib.range, format)),
            );
        }
        // Global memory barriers (RVK-07: they were dropped).
        let memory_barriers: SmallVec<[vk::MemoryBarrier2; 4]> = barrier
            .memory_barriers
            .iter()
            .map(|mb| {
                vk::MemoryBarrier2::default()
                    .src_stage_mask(to_vk_pipeline_stage_flags(mb.src_stages))
                    .src_access_mask(to_vk_access_flags(mb.src_access))
                    .dst_stage_mask(to_vk_pipeline_stage_flags(mb.dst_stages))
                    .dst_access_mask(to_vk_access_flags(mb.dst_access))
            })
            .collect();

        let buffer_barriers: SmallVec<[vk::BufferMemoryBarrier2; 8]> = barrier
            .buffer_barriers
            .iter()
            .filter_map(|bb| {
                let buf = self.buffer(bb.buffer)?;
                Some(
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(to_vk_pipeline_stage_flags(bb.src_stages))
                        .src_access_mask(to_vk_access_flags(bb.src_access))
                        .dst_stage_mask(to_vk_pipeline_stage_flags(bb.dst_stages))
                        .dst_access_mask(to_vk_access_flags(bb.dst_access))
                        .buffer(buf)
                        .offset(bb.offset)
                        .size(bb.size),
                )
            })
            .collect();

        let dependency_info = vk::DependencyInfo::default()
            .memory_barriers(&memory_barriers)
            .image_memory_barriers(&image_barriers)
            .buffer_memory_barriers(&buffer_barriers);

        // SAFETY: the command buffer is recording and owned by this recorder; the barriers name
        // live buffers and images (unknown handles were filtered out above).
        unsafe {
            self.context
                .device
                .cmd_pipeline_barrier2(self.command_buffer, &dependency_info);
        }
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) {
        if let (Some(src_buffer), Some(dst_buffer)) = (self.buffer(src), self.buffer(dst)) {
            let region = vk::BufferCopy {
                src_offset,
                dst_offset,
                size,
            };
            // SAFETY: the command buffer is recording and owned by this recorder.
            unsafe {
                self.context.device.cmd_copy_buffer(
                    self.command_buffer,
                    src_buffer,
                    dst_buffer,
                    &[region],
                );
            }
        }
    }

    fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32) {
        let found = lock(&self.resources.buffers)
            .get(buffer)
            .map(|(b, _)| (b.buffer, b.size));
        let Some((vk_buffer, buffer_size)) = found else {
            tracing::error!("fill_buffer: unknown buffer {buffer:?}; skipped");
            return;
        };
        // Checked here too: release builds do not wrap the recorder in the validator.
        if let Err(problem) = korppi_rhi::transfer::check_fill_buffer(buffer_size, offset, size) {
            tracing::error!("fill_buffer: {problem}; skipped");
            return;
        }
        // SAFETY: the command buffer is recording and owned by this recorder; the buffer is
        // live, and the range is word-aligned and inside it (checked above). Its TRANSFER_DST
        // usage is checked by the validator (debug builds).
        unsafe {
            self.context.device.cmd_fill_buffer(
                self.command_buffer,
                vk_buffer,
                offset,
                size,
                value,
            );
        }
    }

    fn clear_texture(
        &mut self,
        texture: TextureHandle,
        range: &TextureSubresourceRange,
        value: TextureClear,
    ) {
        let Some(info) = self.texture_info(texture) else {
            tracing::error!("clear_texture: unknown texture {texture:?}; skipped");
            return;
        };
        if let Err(problem) = korppi_rhi::transfer::check_clear_texture(&info, range, &value) {
            tracing::error!("clear_texture: {problem}; skipped");
            return;
        }
        let Some((image, _, format, _)) = self.texture(texture, None) else {
            return;
        };
        let ranges = [to_vk_subresource_range(range, format)];
        let (device, command_buffer) = (&self.context.device, self.command_buffer);
        // SAFETY: the command buffer is recording and owned by this recorder; the image is
        // live, the range lies within it and the value matches its format (checked above); the
        // subresources are in TRANSFER_DST_OPTIMAL by contract.
        unsafe {
            match value {
                TextureClear::Color(color) => {
                    let color = if format.is_integer() {
                        vk::ClearColorValue {
                            uint32: color.map(|c| c as u32),
                        }
                    } else {
                        vk::ClearColorValue { float32: color }
                    };
                    device.cmd_clear_color_image(
                        command_buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &color,
                        &ranges,
                    );
                }
                TextureClear::DepthStencil { depth, stencil } => {
                    device.cmd_clear_depth_stencil_image(
                        command_buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &vk::ClearDepthStencilValue { depth, stencil },
                        &ranges,
                    );
                }
            }
        }
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: BufferHandle,
        dst: TextureHandle,
        copy: &BufferTextureCopy,
    ) {
        if let (Some(src_buffer), Some((dst_image, _, format, _))) =
            (self.buffer(src), self.texture(dst, None))
        {
            let region = buffer_image_region(copy, format);
            // SAFETY: the command buffer is recording; the region was validated against the
            // texture (debug builds) and the texture is in TRANSFER_DST_OPTIMAL by contract.
            unsafe {
                self.context.device.cmd_copy_buffer_to_image(
                    self.command_buffer,
                    src_buffer,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
        }
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: TextureHandle,
        dst: BufferHandle,
        copy: &BufferTextureCopy,
    ) {
        if let (Some((src_image, _, format, _)), Some(dst_buffer)) =
            (self.texture(src, None), self.buffer(dst))
        {
            let region = buffer_image_region(copy, format);
            // SAFETY: the command buffer is recording; the texture is in TRANSFER_SRC_OPTIMAL by
            // contract.
            unsafe {
                self.context.device.cmd_copy_image_to_buffer(
                    self.command_buffer,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_buffer,
                    &[region],
                );
            }
        }
    }

    fn copy_texture_to_texture(
        &mut self,
        src: TextureHandle,
        dst: TextureHandle,
        copy: &TextureCopy,
    ) {
        if let (Some((src_image, _, format, _)), Some((dst_image, _, _, _))) =
            (self.texture(src, None), self.texture(dst, None))
        {
            let layers = |mip, layer| vk::ImageSubresourceLayers {
                aspect_mask: barrier_aspect(format),
                mip_level: mip,
                base_array_layer: layer,
                layer_count: copy.layer_count,
            };
            let region = vk::ImageCopy {
                src_subresource: layers(copy.src_mip_level, copy.src_layer),
                src_offset: offset3(copy.src_origin),
                dst_subresource: layers(copy.dst_mip_level, copy.dst_layer),
                dst_offset: offset3(copy.dst_origin),
                extent: vk::Extent3D {
                    width: copy.extent[0],
                    height: copy.extent[1],
                    depth: copy.extent[2],
                },
            };
            // SAFETY: the command buffer is recording; source and destination are in the
            // transfer layouts by contract.
            unsafe {
                self.context.device.cmd_copy_image(
                    self.command_buffer,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
        }
    }

    fn blit_texture(&mut self, src: TextureHandle, dst: TextureHandle, blit: &TextureBlit) {
        let (Some((src_image, _, format, _)), Some((dst_image, _, _, _))) =
            (self.texture(src, None), self.texture(dst, None))
        else {
            return;
        };
        let (Some(src_info), Some(dst_info)) = (self.texture_info(src), self.texture_info(dst))
        else {
            return;
        };
        // Layers of 3D textures are depth slices.
        let side = |info: &TextureInfo, mip, layer: u32, r: [u32; 4]| {
            let (subresource_layer, z0, z1, count) = if info.dimension == TextureDimension::D3 {
                (0, layer, layer + blit.layer_count, 1)
            } else {
                (layer, 0, 1, blit.layer_count)
            };
            (
                vk::ImageSubresourceLayers {
                    aspect_mask: barrier_aspect(format),
                    mip_level: mip,
                    base_array_layer: subresource_layer,
                    layer_count: count,
                },
                [
                    vk::Offset3D {
                        x: r[0] as i32,
                        y: r[1] as i32,
                        z: z0 as i32,
                    },
                    vk::Offset3D {
                        x: r[2] as i32,
                        y: r[3] as i32,
                        z: z1 as i32,
                    },
                ],
            )
        };
        let (src_subresource, src_offsets) = side(
            &src_info,
            blit.src_mip_level,
            blit.src_layer,
            blit.src_region,
        );
        let (dst_subresource, dst_offsets) = side(
            &dst_info,
            blit.dst_mip_level,
            blit.dst_layer,
            blit.dst_region,
        );
        let region = vk::ImageBlit {
            src_subresource,
            src_offsets,
            dst_subresource,
            dst_offsets,
        };
        let filter = match blit.filter {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        };
        // SAFETY: the command buffer is recording; source and destination are in the transfer
        // layouts by contract and the formats were checked for blitting (debug builds).
        unsafe {
            self.context.device.cmd_blit_image(
                self.command_buffer,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                filter,
            );
        }
    }

    fn generate_mipmaps(&mut self, texture: TextureHandle) {
        let (Some((image, _, format, _)), Some(info)) =
            (self.texture(texture, None), self.texture_info(texture))
        else {
            return;
        };
        let three_d = info.dimension == TextureDimension::D3;
        let layers = if three_d { 1 } else { info.array_layers };
        let to_transfer_src = |mip: u32, first: bool| {
            // Level 0 was last written by the caller's commands; later levels by the blit
            // before.
            let (src_stage, src_access) = if first {
                (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_WRITE,
                )
            } else {
                (
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_WRITE,
                )
            };
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: barrier_aspect(format),
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: layers,
                })
        };
        let device = &self.context.device;
        let extent = |mip: u32| {
            let size = korppi_rhi::transfer::mip_extent(&info, mip);
            vk::Offset3D {
                x: size[0] as i32,
                y: size[1] as i32,
                z: size[2] as i32,
            }
        };
        let subresource = |mip| vk::ImageSubresourceLayers {
            aspect_mask: barrier_aspect(format),
            mip_level: mip,
            base_array_layer: 0,
            layer_count: layers,
        };
        // SAFETY: the command buffer is recording; every level is in TRANSFER_DST_OPTIMAL by
        // contract, and each level is turned into a blit source before it is read.
        unsafe {
            for mip in 1..info.mip_levels {
                let barrier = to_transfer_src(mip - 1, mip == 1);
                device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&barrier)),
                );
                let region = vk::ImageBlit {
                    src_subresource: subresource(mip - 1),
                    src_offsets: [vk::Offset3D::default(), extent(mip - 1)],
                    dst_subresource: subresource(mip),
                    dst_offsets: [vk::Offset3D::default(), extent(mip)],
                };
                device.cmd_blit_image(
                    self.command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                    vk::Filter::LINEAR,
                );
            }
            let last = to_transfer_src(info.mip_levels - 1, info.mip_levels == 1);
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&last)),
            );
        }
        self.context_mut().layout_changes.push((
            texture,
            TextureSubresourceRange::ALL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ));
    }

    fn finish(mut self: Box<Self>) -> Result<Box<dyn CommandBuffer>, RhiError> {
        // SAFETY: the command buffer is in the recording state and owned by this recorder.
        unsafe {
            self.context
                .device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| crate::error::vk_error("vkEndCommandBuffer", e))?;
        }

        let command_buffer = VulkanCommandBuffer {
            context: Arc::clone(&self.context),
            recording: self.recording.take(),
            contexts: Arc::clone(&self.contexts),
        };
        let boxed = match self.pools.command_buffers.upgrade().and_then(|p| p.take()) {
            Some(mut boxed) => {
                *boxed = command_buffer;
                boxed
            }
            None => Box::new(command_buffer),
        };
        let pool = self.pools.recorders.clone();
        BoxPool::give_back(&pool, self);
        Ok(boxed)
    }
}

/// The pools a recorder returns itself to and takes command buffer boxes from.
#[derive(Clone, Default)]
pub(crate) struct RecorderPools {
    pub(crate) recorders: std::sync::Weak<BoxPool<VulkanCommandBufferRecorder>>,
    pub(crate) command_buffers: std::sync::Weak<BoxPool<VulkanCommandBuffer>>,
}

fn offset3(origin: [u32; 3]) -> vk::Offset3D {
    vk::Offset3D {
        x: origin[0] as i32,
        y: origin[1] as i32,
        z: origin[2] as i32,
    }
}

/// The Vulkan region of a buffer/texture copy (`korppi_rhi::transfer` describes the layout).
fn buffer_image_region(copy: &BufferTextureCopy, format: TextureFormat) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: copy.buffer_offset,
        buffer_row_length: buffer_row_length_texels(copy.buffer_row_pitch, format),
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: copy_aspect(format),
            mip_level: copy.texture_mip_level,
            base_array_layer: copy.texture_layer,
            layer_count: 1,
        },
        image_offset: offset3(copy.origin),
        image_extent: vk::Extent3D {
            width: copy.extent[0],
            height: copy.extent[1],
            depth: copy.extent[2],
        },
    }
}

use korppi_rhi::lock;
