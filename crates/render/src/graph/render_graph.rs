use crate::error::RenderError;
use crate::graph::aliasing::{MemoryAliasingReport, PhysicalResourceAllocation};
use crate::graph::compiler::{GraphCompiler, OutputScratch};
use crate::graph::handle::{RenderGraphBufferHandle, RenderGraphTextureHandle};
use crate::graph::pass::{CompiledPass, PassBuilder, PassContext, PassNode, PassType};
use crate::graph::resource::{
    BufferAccess, TextureAccess, TransientBufferDesc, TransientResourcePool, TransientTextureDesc,
    VirtualBuffer, VirtualBufferKind, VirtualTexture, VirtualTextureKind,
};
use korppi_rhi::{BufferHandle, CommandBufferRecorder, PipelineBarrier, TextureHandle};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

const MAX_POOLED_SCRATCH: usize = 4;

struct InputScratch {
    textures: Vec<VirtualTexture>,
    buffers: Vec<VirtualBuffer>,
    passes: Vec<PassNode>,
}

impl InputScratch {
    fn new() -> Self {
        Self {
            textures: Vec::with_capacity(16),
            buffers: Vec::with_capacity(16),
            passes: Vec::with_capacity(16),
        }
    }

    fn clear(&mut self) {
        self.textures.clear();
        self.buffers.clear();
        self.passes.clear();
    }
}

thread_local! {
    static INPUT_SCRATCH_POOL: RefCell<Vec<InputScratch>> = const { RefCell::new(Vec::new()) };
    static OUTPUT_SCRATCH_POOL: RefCell<Vec<OutputScratch>> = const { RefCell::new(Vec::new()) };
}

fn take_input_scratch() -> InputScratch {
    INPUT_SCRATCH_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_else(InputScratch::new)
}

fn return_input_scratch(mut scratch: InputScratch) {
    scratch.clear();
    INPUT_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < MAX_POOLED_SCRATCH {
            pool.push(scratch);
        }
    });
}

fn take_output_scratch() -> OutputScratch {
    OUTPUT_SCRATCH_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_else(OutputScratch::new)
}

fn return_output_scratch(mut scratch: OutputScratch) {
    scratch.clear();
    OUTPUT_SCRATCH_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < MAX_POOLED_SCRATCH {
            pool.push(scratch);
        }
    });
}

/// Directed Acyclic Graph (DAG) pass scheduler and frame orchestrator.
pub struct RenderGraph {
    textures: Vec<VirtualTexture>,
    buffers: Vec<VirtualBuffer>,
    passes: Vec<PassNode>,
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RenderGraph {
    fn drop(&mut self) {
        return_input_scratch(InputScratch {
            textures: std::mem::take(&mut self.textures),
            buffers: std::mem::take(&mut self.buffers),
            passes: std::mem::take(&mut self.passes),
        });
    }
}

impl RenderGraph {
    /// Creates a new empty Render Graph, reusing thread-local vector capacities when available (`RND-11`).
    pub fn new() -> Self {
        let scratch = take_input_scratch();
        Self {
            textures: scratch.textures,
            buffers: scratch.buffers,
            passes: scratch.passes,
        }
    }

    /// Registers a transient texture (1 mip level, 1 array layer) to be allocated and managed by the graph.
    pub fn create_transient_texture(
        &mut self,
        desc: TransientTextureDesc,
    ) -> RenderGraphTextureHandle {
        self.create_transient_texture_ext(desc, 1, 1)
    }

    /// Registers a transient texture with explicit mip level and array layer counts (`RND-10`).
    pub fn create_transient_texture_ext(
        &mut self,
        desc: TransientTextureDesc,
        mip_levels: u32,
        array_layers: u32,
    ) -> RenderGraphTextureHandle {
        let id = self.textures.len() as u32;
        self.textures.push(VirtualTexture {
            kind: VirtualTextureKind::Transient {
                desc,
                mip_levels: mip_levels.max(1),
                array_layers: array_layers.max(1),
            },
        });
        RenderGraphTextureHandle(id)
    }

    /// Imports an existing external physical texture (e.g. swapchain image) into the graph.
    pub fn import_texture(
        &mut self,
        handle: TextureHandle,
        initial_access: TextureAccess,
        width: u32,
        height: u32,
    ) -> RenderGraphTextureHandle {
        let id = self.textures.len() as u32;
        self.textures.push(VirtualTexture {
            kind: VirtualTextureKind::Imported {
                handle,
                initial_access,
                final_access: None,
                width,
                height,
                mip_levels: 1,
                array_layers: 1,
            },
        });
        RenderGraphTextureHandle(id)
    }

    /// Imports an external texture and specifies the final access state/layout to transition it into
    /// after all passes complete (e.g. `ImageLayout::Present` for a swapchain target; `RND-05`).
    pub fn import_texture_with_final_access(
        &mut self,
        handle: TextureHandle,
        initial_access: TextureAccess,
        final_access: TextureAccess,
        width: u32,
        height: u32,
    ) -> RenderGraphTextureHandle {
        let id = self.textures.len() as u32;
        self.textures.push(VirtualTexture {
            kind: VirtualTextureKind::Imported {
                handle,
                initial_access,
                final_access: Some(final_access),
                width,
                height,
                mip_levels: 1,
                array_layers: 1,
            },
        });
        RenderGraphTextureHandle(id)
    }

    /// Imports an external texture with explicit mip level and array layer counts (`RND-10`).
    #[allow(clippy::too_many_arguments)]
    pub fn import_texture_ext(
        &mut self,
        handle: TextureHandle,
        initial_access: TextureAccess,
        final_access: Option<TextureAccess>,
        width: u32,
        height: u32,
        mip_levels: u32,
        array_layers: u32,
    ) -> RenderGraphTextureHandle {
        let id = self.textures.len() as u32;
        self.textures.push(VirtualTexture {
            kind: VirtualTextureKind::Imported {
                handle,
                initial_access,
                final_access,
                width,
                height,
                mip_levels: mip_levels.max(1),
                array_layers: array_layers.max(1),
            },
        });
        RenderGraphTextureHandle(id)
    }

    /// Registers a transient buffer to be allocated and managed by the graph.
    pub fn create_transient_buffer(
        &mut self,
        desc: TransientBufferDesc,
    ) -> RenderGraphBufferHandle {
        let id = self.buffers.len() as u32;
        self.buffers.push(VirtualBuffer {
            kind: VirtualBufferKind::Transient(desc),
        });
        RenderGraphBufferHandle(id)
    }

    /// Imports an existing external physical buffer into the graph.
    pub fn import_buffer(
        &mut self,
        handle: BufferHandle,
        initial_access: BufferAccess,
    ) -> RenderGraphBufferHandle {
        let id = self.buffers.len() as u32;
        self.buffers.push(VirtualBuffer {
            kind: VirtualBufferKind::Imported {
                handle,
                initial_access,
                final_access: None,
            },
        });
        RenderGraphBufferHandle(id)
    }

    /// Imports an external buffer and specifies the final access state to synchronize after the graph (`RND-05`).
    pub fn import_buffer_with_final_access(
        &mut self,
        handle: BufferHandle,
        initial_access: BufferAccess,
        final_access: BufferAccess,
    ) -> RenderGraphBufferHandle {
        let id = self.buffers.len() as u32;
        self.buffers.push(VirtualBuffer {
            kind: VirtualBufferKind::Imported {
                handle,
                initial_access,
                final_access: Some(final_access),
            },
        });
        RenderGraphBufferHandle(id)
    }

    /// Adds a new pass to the graph and returns a fluent builder.
    pub fn add_pass(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        pass_type: PassType,
    ) -> PassBuilder<'_> {
        PassBuilder::new(&mut self.passes, name, pass_type)
    }

    /// Compiles the render graph: performs topological sorting, dead-pass culling,
    /// transient memory aliasing, and barrier deduction.
    pub fn compile(mut self) -> Result<CompiledRenderGraph, RenderError> {
        let out_scratch = take_output_scratch();
        let compiler = GraphCompiler::new(&mut self.passes, &self.textures, &self.buffers);
        let output = compiler.compile(None, out_scratch)?;
        Ok(CompiledRenderGraph::from_output(output))
    }

    /// Compiles the render graph with a real graphics device and transient resource pool,
    /// materializing actual GPU textures and buffers for all scheduled passes (`RND-02`, `RND-03`).
    pub fn compile_with_device(
        mut self,
        device: &dyn korppi_rhi::GraphicsDevice,
        pool: &mut TransientResourcePool,
    ) -> Result<CompiledRenderGraph, RenderError> {
        let out_scratch = take_output_scratch();
        let compiler = GraphCompiler::new(&mut self.passes, &self.textures, &self.buffers);
        let output = compiler.compile(Some((device, pool)), out_scratch)?;
        Ok(CompiledRenderGraph::from_output(output))
    }
}

/// A compiled, fully scheduled render graph ready for execution on a command buffer.
#[derive(Debug)]
pub struct CompiledRenderGraph {
    passes: Vec<CompiledPass>,
    barrier_pool: Vec<PipelineBarrier>,
    texture_slice: Vec<Option<TextureHandle>>,
    buffer_slice: Vec<Option<BufferHandle>>,
    texture_map: HashMap<RenderGraphTextureHandle, TextureHandle>,
    buffer_map: HashMap<RenderGraphBufferHandle, BufferHandle>,
    /// Physical GPU memory allocations for transient textures within shared heaps.
    pub texture_allocations: HashMap<RenderGraphTextureHandle, PhysicalResourceAllocation>,
    /// Physical GPU memory allocations for transient buffers within shared heaps.
    pub buffer_allocations: HashMap<RenderGraphBufferHandle, PhysicalResourceAllocation>,
    /// Summary report detailing memory footprints and aliasing savings.
    pub aliasing_report: MemoryAliasingReport,
    post_graph_barrier: PipelineBarrier,
}

impl Drop for CompiledRenderGraph {
    fn drop(&mut self) {
        return_output_scratch(OutputScratch {
            passes: std::mem::take(&mut self.passes),
            barrier_pool: std::mem::take(&mut self.barrier_pool),
            texture_slice: std::mem::take(&mut self.texture_slice),
            buffer_slice: std::mem::take(&mut self.buffer_slice),
            texture_map: std::mem::take(&mut self.texture_map),
            buffer_map: std::mem::take(&mut self.buffer_map),
            texture_allocations: std::mem::take(&mut self.texture_allocations),
            buffer_allocations: std::mem::take(&mut self.buffer_allocations),
        });
    }
}

impl CompiledRenderGraph {
    fn from_output(output: crate::graph::compiler::CompiledGraphOutput) -> Self {
        Self {
            passes: output.passes,
            barrier_pool: output.barrier_pool,
            texture_slice: output.texture_slice,
            buffer_slice: output.buffer_slice,
            texture_map: output.texture_map,
            buffer_map: output.buffer_map,
            texture_allocations: output.texture_allocations,
            buffer_allocations: output.buffer_allocations,
            aliasing_report: output.aliasing_report,
            post_graph_barrier: output.post_graph_barrier,
        }
    }

    /// Returns the scheduled list of compiled passes.
    #[inline]
    pub fn passes(&self) -> &[CompiledPass] {
        &self.passes
    }

    /// Returns the post-graph synchronization barrier emitted after all passes finish.
    #[inline]
    pub fn post_graph_barrier(&self) -> &PipelineBarrier {
        &self.post_graph_barrier
    }

    /// Resolves a virtual texture handle to its assigned physical handle.
    #[inline]
    pub fn resolve_texture(&self, handle: RenderGraphTextureHandle) -> TextureHandle {
        if let Some(Some(phys)) = self.texture_slice.get(handle.id() as usize) {
            return *phys;
        }
        self.texture_map[&handle]
    }

    /// Resolves a virtual buffer handle to its assigned physical handle.
    #[inline]
    pub fn resolve_buffer(&self, handle: RenderGraphBufferHandle) -> BufferHandle {
        if let Some(Some(phys)) = self.buffer_slice.get(handle.id() as usize) {
            return *phys;
        }
        self.buffer_map[&handle]
    }

    /// Executes all scheduled passes sequentially on the provided RHI command buffer recorder.
    pub fn execute(
        &mut self,
        cmd: &mut dyn CommandBufferRecorder,
    ) -> Result<(), korppi_rhi::RhiError> {
        self.execute_with(cmd, |_, _, _| Ok(()))
    }

    /// Executes all scheduled passes sequentially on `cmd`, invoking `runner(pass_name, &context, cmd)`
    /// for each pass without heap-allocating capturing closures.
    pub fn execute_with<F>(
        &mut self,
        cmd: &mut dyn CommandBufferRecorder,
        mut runner: F,
    ) -> Result<(), korppi_rhi::RhiError>
    where
        F: FnMut(
            &str,
            &PassContext<'_>,
            &mut dyn CommandBufferRecorder,
        ) -> Result<(), korppi_rhi::RhiError>,
    {
        let context = PassContext {
            texture_slice: &self.texture_slice,
            buffer_slice: &self.buffer_slice,
            texture_map: &self.texture_map,
            buffer_map: &self.buffer_map,
        };

        for pass in &mut self.passes {
            // 1. Emit pre-pass synchronization barrier if any
            let barrier = &pass.pre_pass_barrier;
            if !barrier.image_barriers.is_empty()
                || !barrier.buffer_barriers.is_empty()
                || !barrier.memory_barriers.is_empty()
            {
                cmd.pipeline_barrier(barrier);
            }

            // 2. Begin dynamic rendering if this is a graphics pass with attachments
            if let Some(rendering_desc) = &pass.rendering_desc {
                cmd.begin_rendering(rendering_desc)?;
            }

            // 3. Record pass commands via callback and/or stack runner
            if let Some(callback) = pass.callback.take() {
                callback.invoke(&context, cmd);
            }
            runner(&pass.name, &context, cmd)?;

            // 4. End dynamic rendering
            if pass.rendering_desc.is_some() {
                cmd.end_rendering()?;
            }
        }

        // 5. Emit post-graph synchronization barrier if any imported resource requested a final state
        if !self.post_graph_barrier.image_barriers.is_empty()
            || !self.post_graph_barrier.buffer_barriers.is_empty()
            || !self.post_graph_barrier.memory_barriers.is_empty()
        {
            cmd.pipeline_barrier(&self.post_graph_barrier);
        }

        Ok(())
    }
}
