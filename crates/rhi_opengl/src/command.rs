//! Emulated OpenGL command buffer packet stream and recorder.
//!
//! Commands hold no heap data (RHI-11, RGL-18): pass descriptors, vertex-buffer lists and bytes
//! (push constants, debug labels) go into side arenas of the [`GlCommandStream`] and commands
//! refer to them by index or range; barriers are reduced to their `glMemoryBarrier` bits when
//! recorded. After the render thread has executed a stream it is cleared and returned to the
//! device's [`CommandStreamPool`], from which the next recorder takes it, so a steady-state frame
//! records without allocating.

use korppi_rhi::{
    BindGroupHandle, BoxPool, BufferHandle, BufferTextureCopy, CommandBuffer,
    CommandBufferRecorder, IndexType, PipelineBarrier, PipelineHandle, QuerySetHandle,
    RenderingDescriptor, RhiError, TextureBlit, TextureCopy, TextureHandle,
    MAX_DYNAMIC_OFFSETS_PER_GROUP, MAX_PUSH_CONSTANTS_SIZE,
};

/// Emulated OpenGL command packet.
#[derive(Clone, Debug)]
pub enum GlCommand {
    /// Begins dynamic rendering pass (index into `GlCommandStream::passes`).
    BeginRendering(u32),
    /// Ends dynamic rendering pass.
    EndRendering,
    /// Sets dynamic viewport dimensions.
    SetViewport {
        /// X coordinate.
        x: f32,
        /// Y coordinate.
        y: f32,
        /// Width.
        width: f32,
        /// Height.
        height: f32,
        /// Min depth.
        min_depth: f32,
        /// Max depth.
        max_depth: f32,
    },
    /// Sets dynamic scissor rectangle.
    SetScissor {
        /// X offset.
        x: i32,
        /// Y offset.
        y: i32,
        /// Width.
        width: u32,
        /// Height.
        height: u32,
    },
    /// Binds graphics or compute pipeline.
    BindPipeline(PipelineHandle),
    /// Binds vertex buffer slices: `count` (buffer, offset) pairs from `start` in
    /// `GlCommandStream::vertex_buffers`.
    BindVertexBuffers {
        /// First binding index.
        first_binding: u32,
        /// First pair.
        start: u32,
        /// Number of pairs.
        count: u32,
    },
    /// Binds index buffer.
    BindIndexBuffer {
        /// Buffer handle.
        buffer: BufferHandle,
        /// Byte offset.
        offset: u64,
        /// Index format.
        index_type: IndexType,
    },
    /// Sets the stencil reference.
    SetStencilReference(u32),
    /// Sets the blend constant.
    SetBlendConstant([f32; 4]),
    /// Sets a bind group (RHI-07).
    SetBindGroup {
        /// Descriptor set index.
        index: u32,
        /// The group.
        group: BindGroupHandle,
        /// Dynamic offsets, `count` of them.
        offsets: [u32; MAX_DYNAMIC_OFFSETS_PER_GROUP],
        /// Number of dynamic offsets.
        count: u8,
    },
    /// Push constant update: `len` bytes from `start` in `GlCommandStream::bytes`.
    PushConstants {
        /// Byte offset in the push-constant block.
        offset: u32,
        /// First byte.
        start: u32,
        /// Number of bytes.
        len: u32,
    },
    /// Non-indexed draw.
    Draw {
        /// Vertex count.
        vertex_count: u32,
        /// Instance count.
        instance_count: u32,
        /// First vertex.
        first_vertex: u32,
        /// First instance.
        first_instance: u32,
    },
    /// Indexed draw.
    DrawIndexed {
        /// Index count.
        index_count: u32,
        /// Instance count.
        instance_count: u32,
        /// First index.
        first_index: u32,
        /// Vertex offset.
        vertex_offset: i32,
        /// First instance.
        first_instance: u32,
    },
    /// Compute dispatch.
    Dispatch {
        /// Group count X.
        group_count_x: u32,
        /// Group count Y.
        group_count_y: u32,
        /// Group count Z.
        group_count_z: u32,
    },
    /// Explicit synchronization memory barrier, as `glMemoryBarrier` bits (0: none needed).
    PipelineBarrier(u32),
    /// Fills a buffer range with a repeated word.
    FillBuffer {
        /// Buffer.
        buffer: BufferHandle,
        /// Byte offset.
        offset: u64,
        /// Byte size.
        size: u64,
        /// Word repeated over the range.
        value: u32,
    },
    /// Clears subresources of a texture.
    ClearTexture {
        /// Texture.
        texture: TextureHandle,
        /// Subresources.
        range: korppi_rhi::TextureSubresourceRange,
        /// Value.
        value: korppi_rhi::TextureClear,
    },
    /// Buffer-to-buffer copy.
    CopyBufferToBuffer {
        /// Source buffer.
        src: BufferHandle,
        /// Source byte offset.
        src_offset: u64,
        /// Destination buffer.
        dst: BufferHandle,
        /// Destination byte offset.
        dst_offset: u64,
        /// Byte size.
        size: u64,
    },
    /// Buffer-to-texture copy.
    CopyBufferToTexture {
        /// Source buffer.
        src: BufferHandle,
        /// Destination texture.
        dst: TextureHandle,
        /// Copy parameters.
        copy: BufferTextureCopy,
    },
    /// Texture-to-buffer copy.
    CopyTextureToBuffer {
        /// Source texture.
        src: TextureHandle,
        /// Destination buffer.
        dst: BufferHandle,
        /// Copy parameters.
        copy: BufferTextureCopy,
    },
    /// Texture-to-texture copy.
    CopyTextureToTexture {
        /// Source texture.
        src: TextureHandle,
        /// Destination texture.
        dst: TextureHandle,
        /// Copy parameters.
        copy: TextureCopy,
    },
    /// Scaled copy.
    BlitTexture {
        /// Source texture.
        src: TextureHandle,
        /// Destination texture.
        dst: TextureHandle,
        /// Blit parameters.
        blit: TextureBlit,
    },
    /// Mip generation.
    GenerateMipmaps(TextureHandle),
    /// Indirect draws.
    DrawIndirect {
        /// Argument buffer.
        buffer: BufferHandle,
        /// Offset of the first record.
        offset: u64,
        /// Number of draws.
        draw_count: u32,
        /// Bytes between records.
        stride: u32,
        /// Indexed draws.
        indexed: bool,
    },
    /// Indirect dispatch.
    DispatchIndirect {
        /// Argument buffer.
        buffer: BufferHandle,
        /// Offset of the record.
        offset: u64,
    },
    /// Timestamp write.
    WriteTimestamp {
        /// Query set.
        query_set: QuerySetHandle,
        /// Query index.
        index: u32,
    },
    /// Query results copied into a buffer.
    ResolveQuerySet {
        /// Query set.
        query_set: QuerySetHandle,
        /// First query.
        first: u32,
        /// Number of queries.
        count: u32,
        /// Destination buffer.
        dst: BufferHandle,
        /// Destination offset.
        dst_offset: u64,
    },
    /// Opens a debug group labelled with `len` bytes from `start` in `GlCommandStream::bytes`.
    PushDebugGroup {
        /// First byte.
        start: u32,
        /// Number of bytes.
        len: u32,
    },
    /// Closes a debug group.
    PopDebugGroup,
    /// Inserts a debug marker labelled like `PushDebugGroup`.
    InsertDebugMarker {
        /// First byte.
        start: u32,
        /// Number of bytes.
        len: u32,
    },
}

/// Recorded commands and the data they refer to.
#[derive(Debug, Default)]
pub struct GlCommandStream {
    /// Commands in order.
    pub commands: Vec<GlCommand>,
    /// Pass descriptors of `BeginRendering`.
    pub passes: Vec<RenderingDescriptor>,
    /// (buffer, offset) pairs of `BindVertexBuffers`.
    pub vertex_buffers: Vec<(BufferHandle, u64)>,
    /// Push-constant data and debug labels.
    pub bytes: Vec<u8>,
}

impl GlCommandStream {
    /// Empties the stream, keeping its capacity.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.passes.clear();
        self.vertex_buffers.clear();
        self.bytes.clear();
    }

    /// `len` bytes from `start` of the byte arena.
    pub fn bytes(&self, start: u32, len: u32) -> &[u8] {
        &self.bytes[start as usize..(start + len) as usize]
    }

    /// Appends `data` to the byte arena and returns its range.
    fn push_bytes(&mut self, data: &[u8]) -> (u32, u32) {
        let start = self.bytes.len() as u32;
        self.bytes.extend_from_slice(data);
        (start, data.len() as u32)
    }
}

/// Streams executed by the render thread, kept for new recorders.
#[derive(Clone, Debug, Default)]
pub struct CommandStreamPool(std::sync::Arc<std::sync::Mutex<Vec<GlCommandStream>>>);

impl CommandStreamPool {
    /// Streams kept; further returned streams are dropped.
    const MAX_KEPT: usize = 16;

    /// A cleared stream, recycled if one is available.
    pub fn take(&self) -> GlCommandStream {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .unwrap_or_default()
    }

    /// Returns a stream for reuse.
    pub fn give(&self, mut stream: GlCommandStream) {
        stream.clear();
        let mut kept = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if kept.len() < Self::MAX_KEPT {
            kept.push(stream);
        }
    }
}

/// Concrete executable OpenGL command buffer holding recorded command packets.
#[derive(Debug)]
pub struct GlCommandBuffer {
    /// The recorded stream; `None` once submitted.
    pub stream: Option<GlCommandStream>,
    /// Where the stream goes back if the buffer is dropped unsubmitted.
    pub pool: CommandStreamPool,
    /// The device that recorded it, the only one that may submit it (RGL-22).
    pub device: korppi_rhi::DeviceId,
}

impl CommandBuffer for GlCommandBuffer {}

impl Drop for GlCommandBuffer {
    fn drop(&mut self) {
        if let Some(stream) = self.stream.take() {
            self.pool.give(stream);
        }
    }
}

/// Concrete OpenGL command buffer recorder.
pub struct GlCommandBufferRecorder {
    stream: GlCommandStream,
    pool: CommandStreamPool,
    /// The device recording; its command buffers carry it.
    device: korppi_rhi::DeviceId,
    is_recording: bool,
    /// Where `finish` returns this box (RHI-11).
    recorders: std::sync::Weak<BoxPool<GlCommandBufferRecorder>>,
    /// Where command buffer boxes come from.
    command_buffers: std::sync::Weak<BoxPool<GlCommandBuffer>>,
}

impl GlCommandBufferRecorder {
    /// Creates a recorder of `device` recording into a stream from `pool`.
    pub fn new(pool: CommandStreamPool, device: korppi_rhi::DeviceId) -> Self {
        Self {
            stream: pool.take(),
            pool,
            device,
            is_recording: false,
            recorders: std::sync::Weak::new(),
            command_buffers: std::sync::Weak::new(),
        }
    }

    /// A recorder whose box returns to `recorders` and whose command buffers come from
    /// `command_buffers`.
    pub fn pooled(
        pool: CommandStreamPool,
        device: korppi_rhi::DeviceId,
        recorders: &std::sync::Arc<BoxPool<GlCommandBufferRecorder>>,
        command_buffers: &std::sync::Arc<BoxPool<GlCommandBuffer>>,
    ) -> Box<Self> {
        let mut recorder = Self::new(pool, device);
        recorder.recorders = std::sync::Arc::downgrade(recorders);
        recorder.command_buffers = std::sync::Arc::downgrade(command_buffers);
        match recorders.take() {
            Some(mut boxed) => {
                *boxed = recorder;
                boxed
            }
            None => Box::new(recorder),
        }
    }
}

impl Drop for GlCommandBufferRecorder {
    fn drop(&mut self) {
        let stream = std::mem::take(&mut self.stream);
        if stream.commands.capacity() > 0 {
            self.pool.give(stream);
        }
    }
}

impl CommandBufferRecorder for GlCommandBufferRecorder {
    fn begin(&mut self) -> Result<(), RhiError> {
        self.stream.clear();
        self.is_recording = true;
        Ok(())
    }

    fn begin_rendering(&mut self, desc: &RenderingDescriptor) -> Result<(), RhiError> {
        let index = self.stream.passes.len() as u32;
        self.stream.passes.push(desc.clone());
        self.stream.commands.push(GlCommand::BeginRendering(index));
        Ok(())
    }

    fn end_rendering(&mut self) -> Result<(), RhiError> {
        self.stream.commands.push(GlCommand::EndRendering);
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
        self.stream.commands.push(GlCommand::SetViewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        });
    }

    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        self.stream.commands.push(GlCommand::SetScissor {
            x,
            y,
            width,
            height,
        });
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        self.stream
            .commands
            .push(GlCommand::SetStencilReference(reference));
    }

    fn set_blend_constant(&mut self, color: [f32; 4]) {
        self.stream
            .commands
            .push(GlCommand::SetBlendConstant(color));
    }

    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        self.stream.commands.push(GlCommand::BindPipeline(pipeline));
    }

    fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[BufferHandle],
        offsets: &[u64],
    ) {
        let start = self.stream.vertex_buffers.len() as u32;
        self.stream
            .vertex_buffers
            .extend(buffers.iter().copied().zip(offsets.iter().copied()));
        self.stream.commands.push(GlCommand::BindVertexBuffers {
            first_binding,
            start,
            count: buffers.len().min(offsets.len()) as u32,
        });
    }

    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64, index_type: IndexType) {
        self.stream.commands.push(GlCommand::BindIndexBuffer {
            buffer,
            offset,
            index_type,
        });
    }

    fn set_bind_group(&mut self, index: u32, group: BindGroupHandle, dynamic_offsets: &[u32]) {
        let count = dynamic_offsets.len().min(MAX_DYNAMIC_OFFSETS_PER_GROUP);
        let mut offsets = [0; MAX_DYNAMIC_OFFSETS_PER_GROUP];
        offsets[..count].copy_from_slice(&dynamic_offsets[..count]);
        self.stream.commands.push(GlCommand::SetBindGroup {
            index,
            group,
            offsets,
            count: count as u8,
        });
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
        let (start, len) = self.stream.push_bytes(data);
        self.stream
            .commands
            .push(GlCommand::PushConstants { offset, start, len });
        Ok(())
    }

    fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.stream.commands.push(GlCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }

    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.stream.commands.push(GlCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        });
    }

    fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.stream.commands.push(GlCommand::Dispatch {
            group_count_x,
            group_count_y,
            group_count_z,
        });
    }

    fn pipeline_barrier(&mut self, barrier: &PipelineBarrier) {
        let bits = crate::types::memory_barrier_bits(barrier);
        self.stream.commands.push(GlCommand::PipelineBarrier(bits));
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) {
        self.stream.commands.push(GlCommand::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        });
    }

    fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32) {
        self.stream.commands.push(GlCommand::FillBuffer {
            buffer,
            offset,
            size,
            value,
        });
    }

    fn clear_texture(
        &mut self,
        texture: TextureHandle,
        range: &korppi_rhi::TextureSubresourceRange,
        value: korppi_rhi::TextureClear,
    ) {
        self.stream.commands.push(GlCommand::ClearTexture {
            texture,
            range: *range,
            value,
        });
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: BufferHandle,
        dst: TextureHandle,
        copy: &BufferTextureCopy,
    ) {
        self.stream.commands.push(GlCommand::CopyBufferToTexture {
            src,
            dst,
            copy: copy.clone(),
        });
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: TextureHandle,
        dst: BufferHandle,
        copy: &BufferTextureCopy,
    ) {
        self.stream.commands.push(GlCommand::CopyTextureToBuffer {
            src,
            dst,
            copy: copy.clone(),
        });
    }

    fn copy_texture_to_texture(
        &mut self,
        src: TextureHandle,
        dst: TextureHandle,
        copy: &TextureCopy,
    ) {
        self.stream.commands.push(GlCommand::CopyTextureToTexture {
            src,
            dst,
            copy: *copy,
        });
    }

    fn blit_texture(&mut self, src: TextureHandle, dst: TextureHandle, blit: &TextureBlit) {
        self.stream.commands.push(GlCommand::BlitTexture {
            src,
            dst,
            blit: *blit,
        });
    }

    fn generate_mipmaps(&mut self, texture: TextureHandle) {
        self.stream
            .commands
            .push(GlCommand::GenerateMipmaps(texture));
    }

    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        self.stream.commands.push(GlCommand::DrawIndirect {
            buffer,
            offset,
            draw_count,
            stride,
            indexed: false,
        });
    }

    fn draw_indexed_indirect(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        self.stream.commands.push(GlCommand::DrawIndirect {
            buffer,
            offset,
            draw_count,
            stride,
            indexed: true,
        });
    }

    fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        self.stream
            .commands
            .push(GlCommand::DispatchIndirect { buffer, offset });
    }

    fn write_timestamp(&mut self, query_set: QuerySetHandle, index: u32) {
        self.stream
            .commands
            .push(GlCommand::WriteTimestamp { query_set, index });
    }

    fn resolve_query_set(
        &mut self,
        query_set: QuerySetHandle,
        first: u32,
        count: u32,
        dst: BufferHandle,
        dst_offset: u64,
    ) {
        self.stream.commands.push(GlCommand::ResolveQuerySet {
            query_set,
            first,
            count,
            dst,
            dst_offset,
        });
    }

    fn push_debug_group(&mut self, label: &str) {
        let (start, len) = self.stream.push_bytes(label.as_bytes());
        self.stream
            .commands
            .push(GlCommand::PushDebugGroup { start, len });
    }

    fn pop_debug_group(&mut self) {
        self.stream.commands.push(GlCommand::PopDebugGroup);
    }

    fn insert_debug_marker(&mut self, label: &str) {
        let (start, len) = self.stream.push_bytes(label.as_bytes());
        self.stream
            .commands
            .push(GlCommand::InsertDebugMarker { start, len });
    }

    fn finish(mut self: Box<Self>) -> Result<Box<dyn CommandBuffer>, RhiError> {
        self.is_recording = false;
        let command_buffer = GlCommandBuffer {
            stream: Some(std::mem::take(&mut self.stream)),
            pool: self.pool.clone(),
            device: self.device,
        };
        let boxed = match self.command_buffers.upgrade().and_then(|p| p.take()) {
            Some(mut boxed) => {
                *boxed = command_buffer;
                boxed
            }
            None => Box::new(command_buffer),
        };
        let recorders = self.recorders.clone();
        BoxPool::give_back(&recorders, self);
        Ok(boxed)
    }
}
