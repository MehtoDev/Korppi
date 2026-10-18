use crate::graph::handle::{RenderGraphBufferHandle, RenderGraphTextureHandle};
use crate::graph::resource::{BufferAccess, TextureAccess};
use korppi_rhi::{
    AccessFlags, BufferHandle, CommandBufferRecorder, ImageLayout, LoadOp, PipelineBarrier,
    PipelineStageFlags, RenderingDescriptor, SmallVec, StoreOp, TextureHandle,
    TextureSubresourceRange,
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::mem::MaybeUninit;

/// Type of workload performed by a render graph pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassType {
    /// Rasterization pass utilizing dynamic rendering attachments.
    Graphics,
    /// Compute dispatch pass.
    Compute,
    /// Memory transfer, copy, or clear pass.
    Transfer,
}

/// Description of a color attachment written by a graphics pass (`RND-10`).
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachmentDesc {
    /// Virtual texture handle.
    pub handle: RenderGraphTextureHandle,
    /// Target attachment index (`layout(location = slot)`).
    pub slot: u32,
    /// Mipmap level to render into.
    pub mip_level: u32,
    /// Array layer (or cube face / 3D slice) to render into.
    pub layer: u32,
    /// Load operation.
    pub load_op: LoadOp,
    /// Store operation.
    pub store_op: StoreOp,
    /// Clear color value if `load_op == LoadOp::Clear`.
    pub clear_color: [f32; 4],
    /// Optional single-sample resolve target for multisampled attachments.
    pub resolve_target: Option<RenderGraphTextureHandle>,
}

/// Description of a depth attachment used by a graphics pass (`RND-10`).
#[derive(Clone, Copy, Debug)]
pub struct DepthAttachmentDesc {
    /// Virtual texture handle.
    pub handle: RenderGraphTextureHandle,
    /// Mipmap level to render into.
    pub mip_level: u32,
    /// Array layer (shadow cascade, cube face) to render into.
    pub layer: u32,
    /// Load operation.
    pub load_op: LoadOp,
    /// Store operation.
    pub store_op: StoreOp,
    /// Clear depth value if `load_op == LoadOp::Clear`.
    pub clear_depth: f32,
    /// Whether this attachment is bound for read-only depth testing without writes.
    pub read_only: bool,
}

/// Execution context provided to pass recording callbacks.
pub struct PassContext<'a> {
    pub(crate) texture_slice: &'a [Option<TextureHandle>],
    pub(crate) buffer_slice: &'a [Option<BufferHandle>],
    pub(crate) texture_map: &'a HashMap<RenderGraphTextureHandle, TextureHandle>,
    pub(crate) buffer_map: &'a HashMap<RenderGraphBufferHandle, BufferHandle>,
}

impl<'a> PassContext<'a> {
    /// Resolves a virtual texture handle to its concrete physical RHI handle.
    #[inline]
    pub fn get_texture(&self, handle: RenderGraphTextureHandle) -> TextureHandle {
        if let Some(Some(phys)) = self.texture_slice.get(handle.id() as usize) {
            return *phys;
        }
        *self.texture_map.get(&handle).unwrap_or_else(|| {
            panic!(
                "PassContext: texture handle {:?} not resolved in compiled graph",
                handle
            )
        })
    }

    /// Resolves a virtual buffer handle to its concrete physical RHI handle.
    #[inline]
    pub fn get_buffer(&self, handle: RenderGraphBufferHandle) -> BufferHandle {
        if let Some(Some(phys)) = self.buffer_slice.get(handle.id() as usize) {
            return *phys;
        }
        *self.buffer_map.get(&handle).unwrap_or_else(|| {
            panic!(
                "PassContext: buffer handle {:?} not resolved in compiled graph",
                handle
            )
        })
    }
}

const INLINE_CALLBACK_WORDS: usize = 8;
type BoxedPassFn = Box<dyn FnOnce(&PassContext, &mut dyn CommandBufferRecorder) + 'static>;

/// Pass recording callback stored inline for closures up to 64 bytes (zero heap allocation; `RND-11`).
pub struct PassCallback {
    storage: [MaybeUninit<u64>; INLINE_CALLBACK_WORDS],
    call_fn: unsafe fn(*mut u8, &PassContext, &mut dyn CommandBufferRecorder),
    drop_fn: Option<unsafe fn(*mut u8)>,
    boxed: Option<BoxedPassFn>,
}

impl PassCallback {
    /// Wraps `f` inline when it fits in 64 bytes with `<= 8`-byte alignment, or boxes it otherwise.
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(&PassContext, &mut dyn CommandBufferRecorder) + 'static,
    {
        if std::mem::size_of::<F>()
            <= std::mem::size_of::<[MaybeUninit<u64>; INLINE_CALLBACK_WORDS]>()
            && std::mem::align_of::<F>() <= std::mem::align_of::<u64>()
        {
            unsafe fn call_inline<F>(
                ptr: *mut u8,
                ctx: &PassContext,
                cmd: &mut dyn CommandBufferRecorder,
            ) where
                F: FnOnce(&PassContext, &mut dyn CommandBufferRecorder) + 'static,
            {
                // SAFETY: `ptr` points to an initialized `F` written in `PassCallback::new`,
                // and `drop_fn` was cleared by `invoke` before calling `call_fn`.
                let closure = unsafe { std::ptr::read(ptr.cast::<F>()) };
                closure(ctx, cmd);
            }

            unsafe fn drop_inline<F>(ptr: *mut u8) {
                // SAFETY: `ptr` points to an initialized `F` that was never invoked.
                unsafe { std::ptr::drop_in_place(ptr.cast::<F>()) };
            }

            let mut storage = [MaybeUninit::<u64>::uninit(); INLINE_CALLBACK_WORDS];
            // SAFETY: `size_of::<F>()` and `align_of::<F>()` were verified to fit inside `storage`.
            unsafe {
                std::ptr::write(storage.as_mut_ptr().cast::<F>(), f);
            }
            Self {
                storage,
                call_fn: call_inline::<F>,
                drop_fn: Some(drop_inline::<F>),
                boxed: None,
            }
        } else {
            unsafe fn noop_call(
                _ptr: *mut u8,
                _ctx: &PassContext,
                _cmd: &mut dyn CommandBufferRecorder,
            ) {
            }
            Self {
                storage: [MaybeUninit::<u64>::uninit(); INLINE_CALLBACK_WORDS],
                call_fn: noop_call,
                drop_fn: None,
                boxed: Some(Box::new(f)),
            }
        }
    }

    /// Consumes and invokes the callback.
    #[inline]
    pub fn invoke(mut self, ctx: &PassContext, cmd: &mut dyn CommandBufferRecorder) {
        if let Some(boxed) = self.boxed.take() {
            boxed(ctx, cmd);
        } else if self.drop_fn.take().is_some() {
            let call_fn = self.call_fn;
            // SAFETY: `self.drop_fn` was `Some` and is now `None`, so `self.storage` holds a
            // live `F` that will be moved out exactly once by `call_fn`.
            unsafe {
                call_fn(self.storage.as_mut_ptr().cast::<u8>(), ctx, cmd);
            }
        }
    }
}

impl Drop for PassCallback {
    fn drop(&mut self) {
        if let Some(drop_fn) = self.drop_fn.take() {
            // SAFETY: `self.drop_fn` was still `Some`, meaning `invoke` was not called and
            // `self.storage` contains a live `F`.
            unsafe {
                drop_fn(self.storage.as_mut_ptr().cast::<u8>());
            }
        }
    }
}

/// Declarative builder for specifying pass dependencies and workload.
pub struct PassBuilder<'a> {
    pub(crate) graph_passes: &'a mut Vec<PassNode>,
    pub(crate) pass: Option<PassNode>,
}

/// Uncompiled pass node stored within the Render Graph.
pub(crate) struct PassNode {
    pub name: Cow<'static, str>,
    pub pass_type: PassType,
    pub read_textures: SmallVec<
        [(
            RenderGraphTextureHandle,
            TextureAccess,
            TextureSubresourceRange,
        ); 4],
    >,
    pub write_textures: SmallVec<
        [(
            RenderGraphTextureHandle,
            TextureAccess,
            TextureSubresourceRange,
        ); 2],
    >,
    pub write_colors: SmallVec<[ColorAttachmentDesc; 4]>,
    pub write_depth: Option<DepthAttachmentDesc>,
    pub read_buffers: SmallVec<[(RenderGraphBufferHandle, BufferAccess); 4]>,
    pub write_buffers: SmallVec<[(RenderGraphBufferHandle, BufferAccess); 4]>,
    pub is_root: bool,
    pub callback: Option<PassCallback>,
}

impl<'a> PassBuilder<'a> {
    pub(crate) fn new(
        graph_passes: &'a mut Vec<PassNode>,
        name: impl Into<Cow<'static, str>>,
        pass_type: PassType,
    ) -> Self {
        Self {
            graph_passes,
            pass: Some(PassNode {
                name: name.into(),
                pass_type,
                read_textures: SmallVec::new(),
                write_textures: SmallVec::new(),
                write_colors: SmallVec::new(),
                write_depth: None,
                read_buffers: SmallVec::new(),
                write_buffers: SmallVec::new(),
                is_root: false,
                callback: None,
            }),
        }
    }

    /// Declares an input texture read by this pass across all subresources.
    pub fn read_texture(self, handle: RenderGraphTextureHandle, access: TextureAccess) -> Self {
        self.read_texture_subresource(handle, access, TextureSubresourceRange::ALL)
    }

    /// Declares an input texture subresource range read by this pass (`RND-10`).
    pub fn read_texture_subresource(
        mut self,
        handle: RenderGraphTextureHandle,
        access: TextureAccess,
        range: TextureSubresourceRange,
    ) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.read_textures.push((handle, access, range));
        }
        self
    }

    /// Declares a non-attachment texture write (e.g. storage image or transfer destination).
    pub fn write_texture(self, handle: RenderGraphTextureHandle, access: TextureAccess) -> Self {
        self.write_texture_subresource(handle, access, TextureSubresourceRange::ALL)
    }

    /// Declares a non-attachment texture subresource write (`RND-10`).
    pub fn write_texture_subresource(
        mut self,
        handle: RenderGraphTextureHandle,
        access: TextureAccess,
        range: TextureSubresourceRange,
    ) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.write_textures.push((handle, access, range));
        }
        self
    }

    /// Declares a color attachment write for a graphics pass (mip 0, layer 0, no resolve target).
    pub fn write_color(
        self,
        handle: RenderGraphTextureHandle,
        slot: u32,
        load_op: LoadOp,
        store_op: StoreOp,
        clear_color: [f32; 4],
    ) -> Self {
        self.write_color_attachment(ColorAttachmentDesc {
            handle,
            slot,
            mip_level: 0,
            layer: 0,
            load_op,
            store_op,
            clear_color,
            resolve_target: None,
        })
    }

    /// Declares a multisampled color attachment write that resolves into `resolve_target` (`RND-10`).
    pub fn write_color_msaa(
        self,
        handle: RenderGraphTextureHandle,
        resolve_target: RenderGraphTextureHandle,
        slot: u32,
        load_op: LoadOp,
        store_op: StoreOp,
        clear_color: [f32; 4],
    ) -> Self {
        self.write_color_attachment(ColorAttachmentDesc {
            handle,
            slot,
            mip_level: 0,
            layer: 0,
            load_op,
            store_op,
            clear_color,
            resolve_target: Some(resolve_target),
        })
    }

    /// Declares a color attachment write targeting a specific `mip_level` and `layer` (`RND-10`).
    #[allow(clippy::too_many_arguments)]
    pub fn write_color_subresource(
        self,
        handle: RenderGraphTextureHandle,
        slot: u32,
        mip_level: u32,
        layer: u32,
        load_op: LoadOp,
        store_op: StoreOp,
        clear_color: [f32; 4],
    ) -> Self {
        self.write_color_attachment(ColorAttachmentDesc {
            handle,
            slot,
            mip_level,
            layer,
            load_op,
            store_op,
            clear_color,
            resolve_target: None,
        })
    }

    /// Declares a full color attachment descriptor for a graphics pass (`RND-10`).
    pub fn write_color_attachment(mut self, desc: ColorAttachmentDesc) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.write_colors.push(desc);
        }
        self
    }

    /// Declares a depth attachment write for a graphics pass (mip 0, layer 0).
    pub fn write_depth(
        self,
        handle: RenderGraphTextureHandle,
        load_op: LoadOp,
        store_op: StoreOp,
        clear_depth: f32,
    ) -> Self {
        self.write_depth_subresource(handle, 0, 0, load_op, store_op, clear_depth)
    }

    /// Declares a depth attachment write targeting a specific `mip_level` and `layer` (`RND-10`).
    pub fn write_depth_subresource(
        mut self,
        handle: RenderGraphTextureHandle,
        mip_level: u32,
        layer: u32,
        load_op: LoadOp,
        store_op: StoreOp,
        clear_depth: f32,
    ) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.write_depth = Some(DepthAttachmentDesc {
                handle,
                mip_level,
                layer,
                load_op,
                store_op,
                clear_depth,
                read_only: false,
            });
        }
        self
    }

    /// Declares a read-only depth attachment (mip 0, layer 0) for depth testing without depth writes (`RND-10`).
    pub fn read_depth(self, handle: RenderGraphTextureHandle) -> Self {
        self.read_depth_subresource(handle, 0, 0)
    }

    /// Declares a read-only depth attachment targeting a specific `mip_level` and `layer` (`RND-10`).
    pub fn read_depth_subresource(
        mut self,
        handle: RenderGraphTextureHandle,
        mip_level: u32,
        layer: u32,
    ) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.write_depth = Some(DepthAttachmentDesc {
                handle,
                mip_level,
                layer,
                load_op: LoadOp::Load,
                store_op: StoreOp::Store,
                clear_depth: 0.0,
                read_only: true,
            });
        }
        self
    }

    /// Declares an input buffer read by this pass.
    pub fn read_buffer(mut self, handle: RenderGraphBufferHandle, access: BufferAccess) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.read_buffers.push((handle, access));
        }
        self
    }

    /// Declares a buffer write by this pass.
    pub fn write_buffer(mut self, handle: RenderGraphBufferHandle, access: BufferAccess) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.write_buffers.push((handle, access));
        }
        self
    }

    /// Marks this pass as an execution root, preventing dead-pass culling.
    pub fn mark_as_root(mut self) -> Self {
        if let Some(pass) = &mut self.pass {
            pass.is_root = true;
        }
        self
    }

    /// Declares an input texture read by this pass as a shader sample in fragment shader.
    pub fn read_color(self, handle: RenderGraphTextureHandle) -> Self {
        self.read_texture(
            handle,
            TextureAccess {
                stage: PipelineStageFlags::FRAGMENT_SHADER,
                access: AccessFlags::SHADER_READ,
                layout: ImageLayout::ShaderReadOnly,
            },
        )
    }

    /// Declares an input storage buffer read by this pass in compute shader.
    pub fn read_storage_buffer(self, handle: RenderGraphBufferHandle) -> Self {
        self.read_buffer(
            handle,
            BufferAccess {
                stage: PipelineStageFlags::COMPUTE_SHADER,
                access: AccessFlags::SHADER_READ,
            },
        )
    }

    /// Declares an output storage buffer write by this pass in compute shader.
    pub fn write_storage_buffer(self, handle: RenderGraphBufferHandle) -> Self {
        self.write_buffer(
            handle,
            BufferAccess {
                stage: PipelineStageFlags::COMPUTE_SHADER,
                access: AccessFlags::SHADER_WRITE,
            },
        )
    }

    /// Records the execution callback and commits the pass to the render graph.
    pub fn record<F>(mut self, callback: F)
    where
        F: FnOnce(&PassContext, &mut dyn CommandBufferRecorder) + 'static,
    {
        if let Some(pass) = &mut self.pass {
            pass.callback = Some(PassCallback::new(callback));
        }
    }

    /// Finishes building the pass and commits it to the render graph.
    pub fn build(self) {}
}

impl<'a> Drop for PassBuilder<'a> {
    fn drop(&mut self) {
        if let Some(pass) = self.pass.take() {
            self.graph_passes.push(pass);
        }
    }
}

/// A pass compiled and scheduled ready for GPU execution.
pub struct CompiledPass {
    pub(crate) name: Cow<'static, str>,
    pub(crate) pass_type: PassType,
    pub(crate) pre_pass_barrier: PipelineBarrier,
    pub(crate) rendering_desc: Option<RenderingDescriptor>,
    pub(crate) callback: Option<PassCallback>,
}

impl CompiledPass {
    /// Returns the debug name of the pass.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the pass type (Graphics, Compute, Transfer).
    #[inline]
    pub fn pass_type(&self) -> PassType {
        self.pass_type
    }

    /// Returns the pre-pass synchronization barrier to be emitted before pass execution.
    #[inline]
    pub fn pre_pass_barrier(&self) -> &PipelineBarrier {
        &self.pre_pass_barrier
    }

    /// Returns the concrete dynamic rendering descriptor if this is a graphics pass.
    #[inline]
    pub fn rendering_descriptor(&self) -> Option<&RenderingDescriptor> {
        self.rendering_desc.as_ref()
    }
}

impl std::fmt::Debug for CompiledPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledPass")
            .field("name", &self.name)
            .field("pass_type", &self.pass_type)
            .field("pre_pass_barrier", &self.pre_pass_barrier)
            .field("rendering_desc", &self.rendering_desc)
            .finish()
    }
}
