use crate::descriptor::{BufferTextureCopy, RenderingDescriptor};
use crate::error::RhiError;
use crate::handle::{BindGroupHandle, BufferHandle, PipelineHandle, TextureHandle};
use crate::sync::PipelineBarrier;
use crate::types::IndexType;
use std::any::Any;

/// Trait representing an executable command buffer ready for queue submission.
///
/// The `Any` supertrait lets backends recover their concrete type with a checked downcast
/// (`Box<dyn CommandBuffer>` upcasts to `Box<dyn Any>`), so submitting another backend's command
/// buffer is an error instead of undefined behaviour (finding RHI-03).
pub trait CommandBuffer: Any + Send + Sync {}

/// Trait for recording graphics, compute, and transfer commands.
///
/// Recorders are `Send`: a recorder created on one thread may record on another (parallel
/// recording), and devices can keep finished recorders for reuse.
pub trait CommandBufferRecorder: Send {
    /// Begins command recording.
    fn begin(&mut self) -> Result<(), RhiError>;

    /// Begins dynamic rendering into the specified attachments.
    fn begin_rendering(&mut self, desc: &RenderingDescriptor) -> Result<(), RhiError>;

    /// Ends dynamic rendering.
    fn end_rendering(&mut self) -> Result<(), RhiError>;

    /// Sets the dynamic viewport dimensions.
    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    );

    /// Sets the dynamic scissor rectangle.
    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32);

    /// Sets the stencil reference value of the following draws (default 0 at `begin`).
    fn set_stencil_reference(&mut self, reference: u32);

    /// Sets the blend constant `BlendFactor::Constant` refers to (default 0 at `begin`).
    fn set_blend_constant(&mut self, color: [f32; 4]);

    /// Binds a graphics or compute pipeline.
    fn bind_pipeline(&mut self, pipeline: PipelineHandle);

    /// Binds vertex buffer slices to consecutive binding slots.
    fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[BufferHandle],
        offsets: &[u64],
    );

    /// Binds an index buffer for indexed drawing.
    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64, index_type: IndexType);

    /// Binds `group` as descriptor set `index` (`layout(set = index)`) for the following draws
    /// and dispatches (RHI-07). `dynamic_offsets` has one offset per dynamic-offset binding of
    /// the group's layout, in binding order. Groups stay bound across pipeline changes; each
    /// draw or dispatch needs a group compatible with every layout its pipeline declares.
    fn set_bind_group(&mut self, index: u32, group: BindGroupHandle, dynamic_offsets: &[u32]);

    /// Updates push constant memory (maximum 128 bytes total across all stages).
    fn push_constants(&mut self, offset: u32, data: &[u8]) -> Result<(), RhiError>;

    /// Draws `draw_count` times with `DrawIndirectArgs` records read from `buffer` at `offset`,
    /// `stride` bytes apart (see `crate::query`).
    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32);

    /// Indexed `draw_indirect` with `DrawIndexedIndirectArgs` records; the index buffer must be
    /// bound at offset 0.
    fn draw_indexed_indirect(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    );

    /// Dispatches with a `DispatchIndirectArgs` record read from `buffer` at `offset`.
    fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64);

    /// Writes the GPU time into query `index` of `query_set` once earlier commands complete.
    fn write_timestamp(&mut self, query_set: crate::handle::QuerySetHandle, index: u32);

    /// Copies queries `first..first + count` of `query_set` into `dst` at `dst_offset` as `u64`
    /// values and makes them writable again (outside render passes).
    fn resolve_query_set(
        &mut self,
        query_set: crate::handle::QuerySetHandle,
        first: u32,
        count: u32,
        dst: BufferHandle,
        dst_offset: u64,
    );

    /// Opens a labelled group of commands for debuggers and profilers (RenderDoc, Nsight).
    fn push_debug_group(&mut self, label: &str);

    /// Closes the innermost open debug group.
    fn pop_debug_group(&mut self);

    /// Inserts a labelled marker.
    fn insert_debug_marker(&mut self, label: &str);

    /// Non-indexed draw call.
    fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    );

    /// Indexed draw call.
    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    );

    /// Dispatches compute workgroups.
    fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32);

    /// Inserts explicit synchronization memory and execution barriers.
    fn pipeline_barrier(&mut self, barrier: &PipelineBarrier);

    /// Copies a byte range between GPU buffers.
    fn copy_buffer_to_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    );

    /// Fills `size` bytes of `buffer` from `offset` with the 32-bit word `value` repeated
    /// (native byte order). `offset` and `size` are non-zero multiples of 4; the buffer needs
    /// `COPY_DST` usage.
    fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32);

    /// Clears the subresources in `range` (aspect `All`) of `texture` to `value`. The texture
    /// has `COPY_DST` usage and those subresources are in `TransferDst`; it is neither
    /// compressed nor multisampled. Attachments are cleared more cheaply with `LoadOp::Clear`.
    fn clear_texture(
        &mut self,
        texture: TextureHandle,
        range: &crate::sync::TextureSubresourceRange,
        value: crate::transfer::TextureClear,
    );

    /// Copies buffer data into a region of one layer (or slices) of a texture mip level, which
    /// is in `TransferDst` (see `crate::transfer` for the data layout).
    fn copy_buffer_to_texture(
        &mut self,
        src: BufferHandle,
        dst: TextureHandle,
        copy: &BufferTextureCopy,
    );

    /// Copies a region of a texture mip level, which is in `TransferSrc`, into a buffer, in the
    /// layout `copy_buffer_to_texture` reads.
    fn copy_texture_to_buffer(
        &mut self,
        src: TextureHandle,
        dst: BufferHandle,
        copy: &BufferTextureCopy,
    );

    /// Copies a region between textures of the same format: source in `TransferSrc`,
    /// destination in `TransferDst`.
    fn copy_texture_to_texture(
        &mut self,
        src: TextureHandle,
        dst: TextureHandle,
        copy: &crate::transfer::TextureCopy,
    );

    /// Copies a rectangle into another, scaling with the blit's filter and converting formats:
    /// source in `TransferSrc`, destination in `TransferDst`.
    fn blit_texture(
        &mut self,
        src: TextureHandle,
        dst: TextureHandle,
        blit: &crate::transfer::TextureBlit,
    );

    /// Fills mip levels 1.. of every layer from level 0 by linear downsampling. Every level is
    /// in `TransferDst` before and in `TransferSrc` after.
    fn generate_mipmaps(&mut self, texture: TextureHandle);

    /// Finalizes recording and returns an executable command buffer.
    fn finish(self: Box<Self>) -> Result<Box<dyn CommandBuffer>, RhiError>;
}
