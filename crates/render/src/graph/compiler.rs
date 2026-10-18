use crate::error::RenderError;
use crate::graph::aliasing::{
    compute_transient_texture_total_bytes, MemoryAliasingReport, PhysicalResourceAllocation,
    TransientMemoryHeapAllocator,
};
use crate::graph::handle::{RenderGraphBufferHandle, RenderGraphTextureHandle};
use crate::graph::pass::{CompiledPass, PassNode, PassType};
use crate::graph::resource::{
    BufferAccess, TextureAccess, TransientBufferDesc, TransientResourcePool, TransientTextureDesc,
    VirtualBuffer, VirtualBufferKind, VirtualTexture, VirtualTextureKind,
};
use korppi_rhi::{
    AccessFlags, BufferHandle, BufferMemoryBarrier, GraphicsDevice, ImageLayout,
    ImageMemoryBarrier, LoadOp, PipelineBarrier, PipelineStageFlags, RenderingAttachment,
    RenderingDescriptor, SmallVec, StoreOp, TextureHandle, TextureSubresourceRange,
};
use std::collections::HashMap;

const WRITE_ACCESS_MASK: AccessFlags = AccessFlags::from_bits_truncate(
    AccessFlags::SHADER_WRITE.bits()
        | AccessFlags::COLOR_ATTACHMENT_WRITE.bits()
        | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE.bits()
        | AccessFlags::TRANSFER_WRITE.bits()
        | AccessFlags::HOST_WRITE.bits()
        | AccessFlags::MEMORY_WRITE.bits(),
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResourceState {
    stage: PipelineStageFlags,
    access: AccessFlags,
    layout: ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferState {
    stage: PipelineStageFlags,
    access: AccessFlags,
}

/// Output produced by graph compilation containing scheduled passes and resource mappings.
#[derive(Debug)]
pub struct CompiledGraphOutput {
    /// Scheduled compiled passes in execution order.
    pub passes: Vec<CompiledPass>,
    /// Fast indexed lookup for virtual texture handles (`0..textures.len()`).
    pub(crate) texture_slice: Vec<Option<TextureHandle>>,
    /// Fast indexed lookup for virtual buffer handles (`0..buffers.len()`).
    pub(crate) buffer_slice: Vec<Option<BufferHandle>>,
    /// Mapping from virtual texture handle to physical texture handle.
    pub texture_map: HashMap<RenderGraphTextureHandle, TextureHandle>,
    /// Mapping from virtual buffer handle to physical buffer handle.
    pub buffer_map: HashMap<RenderGraphBufferHandle, BufferHandle>,
    /// Physical memory allocations for transient textures within shared memory heaps.
    pub texture_allocations: HashMap<RenderGraphTextureHandle, PhysicalResourceAllocation>,
    /// Physical memory allocations for transient buffers within shared memory heaps.
    pub buffer_allocations: HashMap<RenderGraphBufferHandle, PhysicalResourceAllocation>,
    /// Memory aliasing report detailing memory savings.
    pub aliasing_report: MemoryAliasingReport,
    /// Synchronization barrier emitted after all passes complete (e.g. transitioning imported targets to `Present`).
    pub post_graph_barrier: PipelineBarrier,
    /// Recycled `PipelineBarrier` allocations across frames (`RND-11`).
    pub(crate) barrier_pool: Vec<PipelineBarrier>,
}

/// Pre-allocated output containers that `GraphCompiler::compile` populates in place (`RND-11`).
pub(crate) struct OutputScratch {
    pub passes: Vec<CompiledPass>,
    pub barrier_pool: Vec<PipelineBarrier>,
    pub texture_slice: Vec<Option<TextureHandle>>,
    pub buffer_slice: Vec<Option<BufferHandle>>,
    pub texture_map: HashMap<RenderGraphTextureHandle, TextureHandle>,
    pub buffer_map: HashMap<RenderGraphBufferHandle, BufferHandle>,
    pub texture_allocations: HashMap<RenderGraphTextureHandle, PhysicalResourceAllocation>,
    pub buffer_allocations: HashMap<RenderGraphBufferHandle, PhysicalResourceAllocation>,
}

impl OutputScratch {
    pub fn new() -> Self {
        Self {
            passes: Vec::with_capacity(16),
            barrier_pool: Vec::with_capacity(16),
            texture_slice: Vec::with_capacity(16),
            buffer_slice: Vec::with_capacity(16),
            texture_map: HashMap::with_capacity(16),
            buffer_map: HashMap::with_capacity(16),
            texture_allocations: HashMap::with_capacity(16),
            buffer_allocations: HashMap::with_capacity(16),
        }
    }

    pub fn clear(&mut self) {
        for mut pass in self.passes.drain(..).rev() {
            pass.pre_pass_barrier.image_barriers.clear();
            pass.pre_pass_barrier.buffer_barriers.clear();
            pass.pre_pass_barrier.memory_barriers.clear();
            self.barrier_pool.push(pass.pre_pass_barrier);
        }
        self.texture_slice.clear();
        self.buffer_slice.clear();
        self.texture_map.clear();
        self.buffer_map.clear();
        self.texture_allocations.clear();
        self.buffer_allocations.clear();
    }
}

#[derive(Clone)]
struct VersionTracker {
    is_imported: bool,
    last_writer: Option<usize>,
    current_readers: SmallVec<[usize; 8]>,
    unwritten_transient_readers: SmallVec<[usize; 4]>,
}

impl VersionTracker {
    fn new(is_imported: bool) -> Self {
        Self {
            is_imported,
            last_writer: None,
            current_readers: SmallVec::new(),
            unwritten_transient_readers: SmallVec::new(),
        }
    }
}

#[derive(Clone)]
struct PhysicalTexSlot {
    desc: TransientTextureDesc,
    mip_levels: u32,
    array_layers: u32,
    physical_handle: TextureHandle,
    prior_access: Option<TextureAccess>,
    last_used_pass: usize,
    is_pooled: bool,
}

#[derive(Clone)]
struct PhysicalBufSlot {
    desc: TransientBufferDesc,
    physical_handle: BufferHandle,
    prior_access: Option<BufferAccess>,
    last_used_pass: usize,
    is_pooled: bool,
}

#[derive(Clone)]
struct TrackedTextureState {
    handle: TextureHandle,
    mip_levels: u32,
    array_layers: u32,
    prior_frame_access: Option<TextureAccess>,
    subresources: SmallVec<[(TextureSubresourceRange, ResourceState); 2]>,
}

impl TrackedTextureState {
    fn new_transient(
        handle: TextureHandle,
        mip_levels: u32,
        array_layers: u32,
        prior_frame_access: Option<TextureAccess>,
    ) -> Self {
        Self {
            handle,
            mip_levels,
            array_layers,
            prior_frame_access,
            subresources: SmallVec::new(),
        }
    }

    fn new_imported(
        handle: TextureHandle,
        mip_levels: u32,
        array_layers: u32,
        initial: TextureAccess,
    ) -> Self {
        let mut subresources = SmallVec::new();
        subresources.push((
            TextureSubresourceRange::ALL,
            ResourceState {
                stage: initial.stage,
                access: initial.access,
                layout: initial.layout,
            },
        ));
        Self {
            handle,
            mip_levels,
            array_layers,
            prior_frame_access: None,
            subresources,
        }
    }

    fn effective_range(&self, requested: TextureSubresourceRange) -> TextureSubresourceRange {
        if self.mip_levels <= 1 && self.array_layers <= 1 {
            TextureSubresourceRange::ALL
        } else {
            requested
        }
    }

    fn transition(
        &mut self,
        requested_range: TextureSubresourceRange,
        desired: ResourceState,
        barriers: &mut SmallVec<[ImageMemoryBarrier; 4]>,
    ) {
        let range = self.effective_range(requested_range);
        if self.subresources.is_empty() {
            let (src_stages, src_access) = match self.prior_frame_access.take() {
                Some(prior) => (prior.stage, prior.access),
                None => (PipelineStageFlags::TOP_OF_PIPE, AccessFlags::empty()),
            };
            barriers.push(ImageMemoryBarrier {
                texture: self.handle,
                range,
                old_layout: ImageLayout::Undefined,
                new_layout: desired.layout,
                src_stages,
                dst_stages: desired.stage,
                src_access,
                dst_access: desired.access,
            });
            self.subresources.push((range, desired));
            return;
        }

        if range == TextureSubresourceRange::ALL {
            for (sub_range, prev) in &self.subresources {
                if needs_image_barrier(prev, &desired) {
                    barriers.push(ImageMemoryBarrier {
                        texture: self.handle,
                        range: *sub_range,
                        old_layout: prev.layout,
                        new_layout: desired.layout,
                        src_stages: prev.stage,
                        dst_stages: desired.stage,
                        src_access: prev.access,
                        dst_access: desired.access,
                    });
                }
            }
            self.subresources.clear();
            self.subresources
                .push((TextureSubresourceRange::ALL, desired));
            return;
        }

        if let Some((_, prev)) = self.subresources.iter_mut().find(|(r, _)| *r == range) {
            if needs_image_barrier(prev, &desired) {
                barriers.push(ImageMemoryBarrier {
                    texture: self.handle,
                    range,
                    old_layout: prev.layout,
                    new_layout: desired.layout,
                    src_stages: prev.stage,
                    dst_stages: desired.stage,
                    src_access: prev.access,
                    dst_access: desired.access,
                });
                *prev = desired;
            } else {
                prev.stage |= desired.stage;
                prev.access |= desired.access;
            }
        } else if let Some((_, all_prev)) = self
            .subresources
            .iter()
            .find(|(r, _)| *r == TextureSubresourceRange::ALL)
            .copied()
        {
            if needs_image_barrier(&all_prev, &desired) {
                barriers.push(ImageMemoryBarrier {
                    texture: self.handle,
                    range,
                    old_layout: all_prev.layout,
                    new_layout: desired.layout,
                    src_stages: all_prev.stage,
                    dst_stages: desired.stage,
                    src_access: all_prev.access,
                    dst_access: desired.access,
                });
            }
            self.subresources.push((range, desired));
        } else {
            barriers.push(ImageMemoryBarrier {
                texture: self.handle,
                range,
                old_layout: ImageLayout::Undefined,
                new_layout: desired.layout,
                src_stages: PipelineStageFlags::TOP_OF_PIPE,
                dst_stages: desired.stage,
                src_access: AccessFlags::empty(),
                dst_access: desired.access,
            });
            self.subresources.push((range, desired));
        }
    }

    fn summary_access(&self) -> Option<TextureAccess> {
        let mut stages = PipelineStageFlags::empty();
        let mut access = AccessFlags::empty();
        let mut layout = ImageLayout::Undefined;
        for (_, state) in &self.subresources {
            stages |= state.stage;
            access |= state.access;
            layout = state.layout;
        }
        if stages.is_empty() && self.subresources.is_empty() {
            None
        } else {
            Some(TextureAccess {
                stage: stages,
                access,
                layout,
            })
        }
    }
}

#[derive(Clone)]
struct TrackedBufferState {
    handle: BufferHandle,
    prior_frame_access: Option<BufferAccess>,
    current: Option<BufferState>,
}

impl TrackedBufferState {
    fn transition(
        &mut self,
        desired: BufferState,
        barriers: &mut SmallVec<[BufferMemoryBarrier; 4]>,
    ) {
        match &mut self.current {
            Some(prev) => {
                if needs_buffer_barrier(prev, &desired) {
                    barriers.push(BufferMemoryBarrier {
                        buffer: self.handle,
                        src_stages: prev.stage,
                        dst_stages: desired.stage,
                        src_access: prev.access,
                        dst_access: desired.access,
                        offset: 0,
                        size: u64::MAX,
                    });
                    *prev = desired;
                } else {
                    prev.stage |= desired.stage;
                    prev.access |= desired.access;
                }
            }
            None => {
                let (src_stages, src_access) = match self.prior_frame_access.take() {
                    Some(prior) => (prior.stage, prior.access),
                    None => (PipelineStageFlags::TOP_OF_PIPE, AccessFlags::empty()),
                };
                barriers.push(BufferMemoryBarrier {
                    buffer: self.handle,
                    src_stages,
                    dst_stages: desired.stage,
                    src_access,
                    dst_access: desired.access,
                    offset: 0,
                    size: u64::MAX,
                });
                self.current = Some(desired);
            }
        }
    }
}

#[inline]
fn needs_image_barrier(prev: &ResourceState, desired: &ResourceState) -> bool {
    prev.layout != desired.layout
        || prev.access.intersects(WRITE_ACCESS_MASK)
        || desired.access.intersects(WRITE_ACCESS_MASK)
        || !prev.stage.contains(desired.stage)
        || !prev.access.contains(desired.access)
}

#[inline]
fn needs_buffer_barrier(prev: &BufferState, desired: &BufferState) -> bool {
    prev.access.intersects(WRITE_ACCESS_MASK)
        || desired.access.intersects(WRITE_ACCESS_MASK)
        || !prev.stage.contains(desired.stage)
        || !prev.access.contains(desired.access)
}

/// Internal compiler that converts declarative PassNodes into a scheduled execution graph.
pub(crate) struct GraphCompiler<'a> {
    passes: &'a mut [PassNode],
    textures: &'a [VirtualTexture],
    buffers: &'a [VirtualBuffer],
}

impl<'a> GraphCompiler<'a> {
    pub fn new(
        passes: &'a mut [PassNode],
        textures: &'a [VirtualTexture],
        buffers: &'a [VirtualBuffer],
    ) -> Self {
        Self {
            passes,
            textures,
            buffers,
        }
    }

    pub fn compile(
        self,
        mut device_and_pool: Option<(&dyn GraphicsDevice, &mut TransientResourcePool)>,
        mut out: OutputScratch,
    ) -> Result<CompiledGraphOutput, RenderError> {
        out.clear();
        let pass_count = self.passes.len();
        if pass_count == 0 {
            return Ok(CompiledGraphOutput {
                passes: out.passes,
                barrier_pool: out.barrier_pool,
                texture_slice: out.texture_slice,
                buffer_slice: out.buffer_slice,
                texture_map: out.texture_map,
                buffer_map: out.buffer_map,
                texture_allocations: out.texture_allocations,
                buffer_allocations: out.buffer_allocations,
                aliasing_report: MemoryAliasingReport::default(),
                post_graph_barrier: PipelineBarrier::default(),
            });
        }

        // 1. Versioned Dependency Graph Construction (RND-04: RAW, WAR, WAW, Ping-Pong)
        let mut tex_versions: SmallVec<[VersionTracker; 16]> = self
            .textures
            .iter()
            .map(|t| VersionTracker::new(matches!(t.kind, VirtualTextureKind::Imported { .. })))
            .collect();
        let mut buf_versions: SmallVec<[VersionTracker; 16]> = self
            .buffers
            .iter()
            .map(|b| VersionTracker::new(matches!(b.kind, VirtualBufferKind::Imported { .. })))
            .collect();

        let mut adj: SmallVec<[SmallVec<[usize; 8]>; 16]> =
            (0..pass_count).map(|_| SmallVec::new()).collect();
        let mut data_rev_adj: SmallVec<[SmallVec<[usize; 8]>; 16]> =
            (0..pass_count).map(|_| SmallVec::new()).collect();

        let add_edge =
            |u: usize,
             v: usize,
             is_data_dep: bool,
             adj: &mut SmallVec<[SmallVec<[usize; 8]>; 16]>,
             data_rev_adj: &mut SmallVec<[SmallVec<[usize; 8]>; 16]>| {
                if u != v {
                    if !adj[u].contains(&v) {
                        adj[u].push(v);
                    }
                    if is_data_dep && !data_rev_adj[v].contains(&u) {
                        data_rev_adj[v].push(u);
                    }
                }
            };

        let record_read =
            |tracker: &mut VersionTracker,
             pass_idx: usize,
             adj: &mut SmallVec<[SmallVec<[usize; 8]>; 16]>,
             data_rev_adj: &mut SmallVec<[SmallVec<[usize; 8]>; 16]>| {
                if let Some(w) = tracker.last_writer {
                    add_edge(w, pass_idx, true, adj, data_rev_adj);
                    if !tracker.current_readers.contains(&pass_idx) {
                        tracker.current_readers.push(pass_idx);
                    }
                } else if tracker.is_imported {
                    if !tracker.current_readers.contains(&pass_idx) {
                        tracker.current_readers.push(pass_idx);
                    }
                } else if !tracker.unwritten_transient_readers.contains(&pass_idx) {
                    tracker.unwritten_transient_readers.push(pass_idx);
                }
            };

        let record_write =
            |tracker: &mut VersionTracker,
             pass_idx: usize,
             adj: &mut SmallVec<[SmallVec<[usize; 8]>; 16]>,
             data_rev_adj: &mut SmallVec<[SmallVec<[usize; 8]>; 16]>| {
                if !tracker.current_readers.is_empty() {
                    for &r in &tracker.current_readers {
                        add_edge(r, pass_idx, false, adj, data_rev_adj);
                    }
                    tracker.current_readers.clear();
                }
                if let Some(w) = tracker.last_writer {
                    add_edge(w, pass_idx, true, adj, data_rev_adj);
                } else if !tracker.unwritten_transient_readers.is_empty() {
                    for &r in &tracker.unwritten_transient_readers {
                        add_edge(pass_idx, r, true, adj, data_rev_adj);
                        if !tracker.current_readers.contains(&r) {
                            tracker.current_readers.push(r);
                        }
                    }
                    tracker.unwritten_transient_readers.clear();
                }
                tracker.last_writer = Some(pass_idx);
            };

        for (pass_idx, pass) in self.passes.iter().enumerate() {
            // Process reads first so a pass that reads mip 0 and writes mip 1 of the same texture
            // depends on the prior writer of mip 0 and becomes the new writer for subsequent passes.
            for (tex_h, _, _) in &pass.read_textures {
                record_read(
                    &mut tex_versions[tex_h.id() as usize],
                    pass_idx,
                    &mut adj,
                    &mut data_rev_adj,
                );
            }
            if let Some(depth_desc) = &pass.write_depth {
                if depth_desc.read_only {
                    record_read(
                        &mut tex_versions[depth_desc.handle.id() as usize],
                        pass_idx,
                        &mut adj,
                        &mut data_rev_adj,
                    );
                }
            }
            for (buf_h, _) in &pass.read_buffers {
                record_read(
                    &mut buf_versions[buf_h.id() as usize],
                    pass_idx,
                    &mut adj,
                    &mut data_rev_adj,
                );
            }

            // Process writes
            for color_desc in &pass.write_colors {
                record_write(
                    &mut tex_versions[color_desc.handle.id() as usize],
                    pass_idx,
                    &mut adj,
                    &mut data_rev_adj,
                );
                if let Some(resolve_h) = color_desc.resolve_target {
                    record_write(
                        &mut tex_versions[resolve_h.id() as usize],
                        pass_idx,
                        &mut adj,
                        &mut data_rev_adj,
                    );
                }
            }
            if let Some(depth_desc) = &pass.write_depth {
                if !depth_desc.read_only {
                    record_write(
                        &mut tex_versions[depth_desc.handle.id() as usize],
                        pass_idx,
                        &mut adj,
                        &mut data_rev_adj,
                    );
                }
            }
            for (tex_h, _, _) in &pass.write_textures {
                record_write(
                    &mut tex_versions[tex_h.id() as usize],
                    pass_idx,
                    &mut adj,
                    &mut data_rev_adj,
                );
            }
            for (buf_h, _) in &pass.write_buffers {
                record_write(
                    &mut buf_versions[buf_h.id() as usize],
                    pass_idx,
                    &mut adj,
                    &mut data_rev_adj,
                );
            }
        }

        // 2. Dead-Pass Culling
        let mut reachable: SmallVec<[bool; 16]> = (0..pass_count).map(|_| false).collect();
        let mut queue: SmallVec<[usize; 16]> = SmallVec::new();

        for (idx, pass) in self.passes.iter().enumerate() {
            let mut rooted = pass.is_root;
            if !rooted {
                for color_desc in &pass.write_colors {
                    if matches!(
                        self.textures[color_desc.handle.id() as usize].kind,
                        VirtualTextureKind::Imported { .. }
                    ) {
                        rooted = true;
                        break;
                    }
                    if let Some(resolve_h) = color_desc.resolve_target {
                        if matches!(
                            self.textures[resolve_h.id() as usize].kind,
                            VirtualTextureKind::Imported { .. }
                        ) {
                            rooted = true;
                            break;
                        }
                    }
                }
            }
            if !rooted {
                if let Some(depth_desc) = &pass.write_depth {
                    if !depth_desc.read_only
                        && matches!(
                            self.textures[depth_desc.handle.id() as usize].kind,
                            VirtualTextureKind::Imported { .. }
                        )
                    {
                        rooted = true;
                    }
                }
            }
            if !rooted {
                for (tex_h, _, _) in &pass.write_textures {
                    if matches!(
                        self.textures[tex_h.id() as usize].kind,
                        VirtualTextureKind::Imported { .. }
                    ) {
                        rooted = true;
                        break;
                    }
                }
            }
            if !rooted {
                for (buf_h, _) in &pass.write_buffers {
                    if matches!(
                        self.buffers[buf_h.id() as usize].kind,
                        VirtualBufferKind::Imported { .. }
                    ) {
                        rooted = true;
                        break;
                    }
                }
            }
            if rooted {
                reachable[idx] = true;
                queue.push(idx);
            }
        }

        let mut head = 0;
        while head < queue.len() {
            let curr = queue[head];
            head += 1;
            for &prev in &data_rev_adj[curr] {
                if !reachable[prev] {
                    reachable[prev] = true;
                    queue.push(prev);
                }
            }
        }

        let mut active_indices: SmallVec<[usize; 16]> = SmallVec::new();
        for (i, &is_reach) in reachable.iter().enumerate() {
            if is_reach {
                active_indices.push(i);
            }
        }

        // 3. Topological Sort with Cycle Detection on Active Passes
        let mut in_degrees: SmallVec<[usize; 16]> = (0..pass_count).map(|_| 0).collect();
        for &u in &active_indices {
            for &v in &adj[u] {
                if reachable[v] {
                    in_degrees[v] += 1;
                }
            }
        }

        let mut ready: SmallVec<[usize; 16]> = SmallVec::new();
        for &idx in &active_indices {
            if in_degrees[idx] == 0 {
                ready.push(idx);
            }
        }

        let mut sorted_pass_indices: SmallVec<[usize; 16]> = SmallVec::new();
        while !ready.is_empty() {
            // Pop smallest pass index to preserve registration order among independent passes
            let mut min_pos = 0;
            for i in 1..ready.len() {
                if ready[i] < ready[min_pos] {
                    min_pos = i;
                }
            }
            let u = ready.remove(min_pos);
            sorted_pass_indices.push(u);
            for &v in &adj[u] {
                if reachable[v] {
                    in_degrees[v] -= 1;
                    if in_degrees[v] == 0 {
                        ready.push(v);
                    }
                }
            }
        }

        if sorted_pass_indices.len() < active_indices.len() {
            let cycle_pass_name = active_indices
                .iter()
                .find(|&&idx| in_degrees[idx] > 0)
                .map(|&idx| self.passes[idx].name.to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            return Err(RenderError::CyclicDependency {
                pass_name: cycle_pass_name,
            });
        }

        // 4. Lifetime Analysis and Physical Resource Aliasing (RND-02, RND-03, RND-14)
        let mut tex_lifetimes: SmallVec<[Option<(usize, usize)>; 16]> =
            (0..self.textures.len()).map(|_| None).collect();
        let mut buf_lifetimes: SmallVec<[Option<(usize, usize)>; 16]> =
            (0..self.buffers.len()).map(|_| None).collect();

        let touch_tex =
            |h: RenderGraphTextureHandle,
             sched_idx: usize,
             lifetimes: &mut SmallVec<[Option<(usize, usize)>; 16]>| {
                let slot = &mut lifetimes[h.id() as usize];
                match slot {
                    Some((_, last)) => *last = sched_idx,
                    None => *slot = Some((sched_idx, sched_idx)),
                }
            };
        let touch_buf =
            |h: RenderGraphBufferHandle,
             sched_idx: usize,
             lifetimes: &mut SmallVec<[Option<(usize, usize)>; 16]>| {
                let slot = &mut lifetimes[h.id() as usize];
                match slot {
                    Some((_, last)) => *last = sched_idx,
                    None => *slot = Some((sched_idx, sched_idx)),
                }
            };

        for (sched_idx, &orig_idx) in sorted_pass_indices.iter().enumerate() {
            let pass = &self.passes[orig_idx];
            for (t, _, _) in &pass.read_textures {
                touch_tex(*t, sched_idx, &mut tex_lifetimes);
            }
            for (t, _, _) in &pass.write_textures {
                touch_tex(*t, sched_idx, &mut tex_lifetimes);
            }
            for c in &pass.write_colors {
                touch_tex(c.handle, sched_idx, &mut tex_lifetimes);
                if let Some(resolve_h) = c.resolve_target {
                    touch_tex(resolve_h, sched_idx, &mut tex_lifetimes);
                }
            }
            if let Some(d) = &pass.write_depth {
                touch_tex(d.handle, sched_idx, &mut tex_lifetimes);
            }
            for (b, _) in &pass.read_buffers {
                touch_buf(*b, sched_idx, &mut buf_lifetimes);
            }
            for (b, _) in &pass.write_buffers {
                touch_buf(*b, sched_idx, &mut buf_lifetimes);
            }
        }

        if let Some((_, ref mut pool)) = device_and_pool {
            pool.begin_compilation();
        }

        // Compute offline simulated ID base strictly above any imported handle ID so offline
        // `compile()` never collides with imported handles (RND-03).
        let mut max_imported_tex_id = 99u32;
        for v_tex in self.textures {
            if let VirtualTextureKind::Imported { handle, .. } = &v_tex.kind {
                max_imported_tex_id = max_imported_tex_id.max(handle.id());
            }
        }
        let mut next_offline_tex_id = max_imported_tex_id.saturating_add(1);

        let mut max_imported_buf_id = 199u32;
        for v_buf in self.buffers {
            if let VirtualBufferKind::Imported { handle, .. } = &v_buf.kind {
                max_imported_buf_id = max_imported_buf_id.max(handle.id());
            }
        }
        let mut next_offline_buf_id = max_imported_buf_id.saturating_add(1);

        out.texture_slice.resize(self.textures.len(), None);
        out.buffer_slice.resize(self.buffers.len(), None);

        let mut sorted_v_textures: SmallVec<[(RenderGraphTextureHandle, usize, usize); 16]> =
            SmallVec::new();
        for (idx, lifetime) in tex_lifetimes.iter().enumerate() {
            if let Some((first, last)) = *lifetime {
                sorted_v_textures.push((RenderGraphTextureHandle(idx as u32), first, last));
            }
        }
        sorted_v_textures.sort_unstable_by_key(|&(_, first, _)| first);

        let mut tex_slots: SmallVec<[PhysicalTexSlot; 16]> = SmallVec::new();
        let mut v_tex_to_slot: SmallVec<[Option<usize>; 16]> =
            (0..self.textures.len()).map(|_| None).collect();

        for &(v_handle, first_pass, last_pass) in &sorted_v_textures {
            let v_tex = &self.textures[v_handle.id() as usize];
            match &v_tex.kind {
                VirtualTextureKind::Imported { handle, .. } => {
                    out.texture_slice[v_handle.id() as usize] = Some(*handle);
                    out.texture_map.insert(v_handle, *handle);
                }
                VirtualTextureKind::Transient {
                    desc,
                    mip_levels,
                    array_layers,
                } => {
                    let mut reused_slot = None;
                    for (slot_idx, entry) in tex_slots.iter_mut().enumerate() {
                        if entry.desc.format == desc.format
                            && entry.desc.width == desc.width
                            && entry.desc.height == desc.height
                            && entry.mip_levels == *mip_levels
                            && entry.array_layers == *array_layers
                            && entry.desc.usage == desc.usage
                            && entry.desc.sample_count == desc.sample_count
                            && entry.desc.dimension == desc.dimension
                            && entry.last_used_pass < first_pass
                        {
                            entry.last_used_pass = last_pass;
                            reused_slot = Some((slot_idx, entry.physical_handle));
                            break;
                        }
                    }

                    if let Some((slot_idx, phys_handle)) = reused_slot {
                        v_tex_to_slot[v_handle.id() as usize] = Some(slot_idx);
                        out.texture_slice[v_handle.id() as usize] = Some(phys_handle);
                        out.texture_map.insert(v_handle, phys_handle);
                    } else {
                        let (phys_handle, prior_access, is_pooled) =
                            if let Some((device, ref mut pool)) = device_and_pool {
                                let (h, prior) = pool.acquire_texture_with_state(
                                    device,
                                    desc,
                                    *mip_levels,
                                    *array_layers,
                                )?;
                                (h, prior, true)
                            } else {
                                let h = TextureHandle::new(next_offline_tex_id, 1);
                                next_offline_tex_id += 1;
                                (h, None, false)
                            };
                        let slot_idx = tex_slots.len();
                        tex_slots.push(PhysicalTexSlot {
                            desc: desc.clone(),
                            mip_levels: *mip_levels,
                            array_layers: *array_layers,
                            physical_handle: phys_handle,
                            prior_access,
                            last_used_pass: last_pass,
                            is_pooled,
                        });
                        v_tex_to_slot[v_handle.id() as usize] = Some(slot_idx);
                        out.texture_slice[v_handle.id() as usize] = Some(phys_handle);
                        out.texture_map.insert(v_handle, phys_handle);
                    }
                }
            }
        }

        let mut sorted_v_buffers: SmallVec<[(RenderGraphBufferHandle, usize, usize); 16]> =
            SmallVec::new();
        for (idx, lifetime) in buf_lifetimes.iter().enumerate() {
            if let Some((first, last)) = *lifetime {
                sorted_v_buffers.push((RenderGraphBufferHandle(idx as u32), first, last));
            }
        }
        sorted_v_buffers.sort_unstable_by_key(|&(_, first, _)| first);

        let mut buf_slots: SmallVec<[PhysicalBufSlot; 16]> = SmallVec::new();
        let mut v_buf_to_slot: SmallVec<[Option<usize>; 16]> =
            (0..self.buffers.len()).map(|_| None).collect();

        for &(v_handle, first_pass, last_pass) in &sorted_v_buffers {
            let v_buf = &self.buffers[v_handle.id() as usize];
            match &v_buf.kind {
                VirtualBufferKind::Imported { handle, .. } => {
                    out.buffer_slice[v_handle.id() as usize] = Some(*handle);
                    out.buffer_map.insert(v_handle, *handle);
                }
                VirtualBufferKind::Transient(desc) => {
                    let mut reused_slot = None;
                    for (slot_idx, entry) in buf_slots.iter_mut().enumerate() {
                        if entry.desc.size >= desc.size
                            && entry.desc.usage == desc.usage
                            && entry.last_used_pass < first_pass
                        {
                            entry.last_used_pass = last_pass;
                            reused_slot = Some((slot_idx, entry.physical_handle));
                            break;
                        }
                    }

                    if let Some((slot_idx, phys_handle)) = reused_slot {
                        v_buf_to_slot[v_handle.id() as usize] = Some(slot_idx);
                        out.buffer_slice[v_handle.id() as usize] = Some(phys_handle);
                        out.buffer_map.insert(v_handle, phys_handle);
                    } else {
                        let (phys_handle, prior_access, is_pooled) =
                            if let Some((device, ref mut pool)) = device_and_pool {
                                let (h, prior) = pool.acquire_buffer_with_state(device, desc)?;
                                (h, prior, true)
                            } else {
                                let h = BufferHandle::new(next_offline_buf_id, 1);
                                next_offline_buf_id += 1;
                                (h, None, false)
                            };
                        let slot_idx = buf_slots.len();
                        buf_slots.push(PhysicalBufSlot {
                            desc: desc.clone(),
                            physical_handle: phys_handle,
                            prior_access,
                            last_used_pass: last_pass,
                            is_pooled,
                        });
                        v_buf_to_slot[v_handle.id() as usize] = Some(slot_idx);
                        out.buffer_slice[v_handle.id() as usize] = Some(phys_handle);
                        out.buffer_map.insert(v_handle, phys_handle);
                    }
                }
            }
        }

        // 4b. Physical GPU Memory Aliasing & Honest Pool Report (RND-14)
        let mut tex_allocator = TransientMemoryHeapAllocator::new(256);
        let mut buf_allocator = TransientMemoryHeapAllocator::new(256);

        let mut unaliased_bytes = 0u64;
        let mut transient_count = 0usize;

        for &(v_handle, first_pass, last_pass) in &sorted_v_textures {
            let v_tex = &self.textures[v_handle.id() as usize];
            if let VirtualTextureKind::Transient {
                desc,
                mip_levels,
                array_layers,
            } = &v_tex.kind
            {
                let size = compute_transient_texture_total_bytes(desc, *mip_levels, *array_layers);
                unaliased_bytes += size;
                transient_count += 1;
                let alloc = tex_allocator.allocate(size, first_pass, last_pass, 0);
                out.texture_allocations.insert(v_handle, alloc);
            }
        }

        for &(v_handle, first_pass, last_pass) in &sorted_v_buffers {
            let v_buf = &self.buffers[v_handle.id() as usize];
            if let VirtualBufferKind::Transient(desc) = &v_buf.kind {
                let size = desc.size;
                unaliased_bytes += size;
                transient_count += 1;
                let alloc = buf_allocator.allocate(size, first_pass, last_pass, 1);
                out.buffer_allocations.insert(v_handle, alloc);
            }
        }

        let mut physical_bytes = 0u64;
        for slot in &tex_slots {
            physical_bytes += compute_transient_texture_total_bytes(
                &slot.desc,
                slot.mip_levels,
                slot.array_layers,
            );
        }
        for slot in &buf_slots {
            physical_bytes += slot.desc.size;
        }
        let physical_bytes_saved = unaliased_bytes.saturating_sub(physical_bytes);

        let aliased_bytes = tex_allocator.peak_bytes() + buf_allocator.peak_bytes();
        let bytes_saved = unaliased_bytes.saturating_sub(aliased_bytes);
        let savings_ratio = if unaliased_bytes > 0 {
            bytes_saved as f32 / unaliased_bytes as f32
        } else {
            0.0
        };

        let aliasing_report = MemoryAliasingReport {
            unaliased_bytes,
            aliased_bytes,
            bytes_saved,
            savings_ratio,
            transient_resource_count: transient_count,
            physical_texture_count: tex_slots.len(),
            physical_buffer_count: buf_slots.len(),
            physical_bytes,
            physical_bytes_saved,
        };

        // 5. Automated Barrier Deduction & CompiledPass Generation (RND-05, RND-10)
        // Track states per physical slot (for transients) and per virtual handle (for imported resources).
        let mut transient_tex_states: SmallVec<[TrackedTextureState; 16]> = tex_slots
            .iter()
            .map(|slot| {
                TrackedTextureState::new_transient(
                    slot.physical_handle,
                    slot.mip_levels,
                    slot.array_layers,
                    slot.prior_access,
                )
            })
            .collect();
        let mut imported_tex_states: SmallVec<[Option<TrackedTextureState>; 16]> = self
            .textures
            .iter()
            .map(|t| match &t.kind {
                VirtualTextureKind::Imported {
                    handle,
                    initial_access,
                    mip_levels,
                    array_layers,
                    ..
                } => Some(TrackedTextureState::new_imported(
                    *handle,
                    *mip_levels,
                    *array_layers,
                    *initial_access,
                )),
                VirtualTextureKind::Transient { .. } => None,
            })
            .collect();

        let mut transient_buf_states: SmallVec<[TrackedBufferState; 16]> = buf_slots
            .iter()
            .map(|slot| TrackedBufferState {
                handle: slot.physical_handle,
                prior_frame_access: slot.prior_access,
                current: None,
            })
            .collect();
        let mut imported_buf_states: SmallVec<[Option<TrackedBufferState>; 16]> = self
            .buffers
            .iter()
            .map(|b| match &b.kind {
                VirtualBufferKind::Imported {
                    handle,
                    initial_access,
                    ..
                } => Some(TrackedBufferState {
                    handle: *handle,
                    prior_frame_access: None,
                    current: Some(BufferState {
                        stage: initial_access.stage,
                        access: initial_access.access,
                    }),
                }),
                VirtualBufferKind::Transient(_) => None,
            })
            .collect();

        let transition_tex =
            |v_handle: RenderGraphTextureHandle,
             range: TextureSubresourceRange,
             desired: ResourceState,
             barriers: &mut SmallVec<[ImageMemoryBarrier; 4]>,
             transient_tex_states: &mut SmallVec<[TrackedTextureState; 16]>,
             imported_tex_states: &mut SmallVec<[Option<TrackedTextureState>; 16]>| {
                if let Some(slot_idx) = v_tex_to_slot[v_handle.id() as usize] {
                    transient_tex_states[slot_idx].transition(range, desired, barriers);
                } else if let Some(state) = &mut imported_tex_states[v_handle.id() as usize] {
                    state.transition(range, desired, barriers);
                }
            };

        let transition_buf =
            |v_handle: RenderGraphBufferHandle,
             desired: BufferState,
             barriers: &mut SmallVec<[BufferMemoryBarrier; 4]>,
             transient_buf_states: &mut SmallVec<[TrackedBufferState; 16]>,
             imported_buf_states: &mut SmallVec<[Option<TrackedBufferState>; 16]>| {
                if let Some(slot_idx) = v_buf_to_slot[v_handle.id() as usize] {
                    transient_buf_states[slot_idx].transition(desired, barriers);
                } else if let Some(state) = &mut imported_buf_states[v_handle.id() as usize] {
                    state.transition(desired, barriers);
                }
            };

        for orig_idx in sorted_pass_indices {
            let pass = &mut self.passes[orig_idx];
            let mut pre_pass_barrier = out.barrier_pool.pop().unwrap_or_default();
            pre_pass_barrier.image_barriers.clear();
            pre_pass_barrier.buffer_barriers.clear();
            pre_pass_barrier.memory_barriers.clear();

            // Sort color attachments by slot index (RND-10)
            pass.write_colors.sort_unstable_by_key(|c| c.slot);

            // Texture reads first (so a pass that reads mip N and writes mip N+1 transitions both subresources cleanly)
            for (tex_h, access, range) in &pass.read_textures {
                transition_tex(
                    *tex_h,
                    *range,
                    ResourceState {
                        stage: access.stage,
                        access: access.access,
                        layout: access.layout,
                    },
                    &mut pre_pass_barrier.image_barriers,
                    &mut transient_tex_states,
                    &mut imported_tex_states,
                );
            }

            // Non-attachment texture writes
            for (tex_h, access, range) in &pass.write_textures {
                transition_tex(
                    *tex_h,
                    *range,
                    ResourceState {
                        stage: access.stage,
                        access: access.access,
                        layout: access.layout,
                    },
                    &mut pre_pass_barrier.image_barriers,
                    &mut transient_tex_states,
                    &mut imported_tex_states,
                );
            }

            // Color attachments and MSAA resolve targets
            for color_desc in &pass.write_colors {
                let access = if color_desc.load_op == LoadOp::Load {
                    AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE
                } else {
                    AccessFlags::COLOR_ATTACHMENT_WRITE
                };
                transition_tex(
                    color_desc.handle,
                    TextureSubresourceRange::single(color_desc.mip_level, color_desc.layer),
                    ResourceState {
                        stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        access,
                        layout: ImageLayout::ColorAttachment,
                    },
                    &mut pre_pass_barrier.image_barriers,
                    &mut transient_tex_states,
                    &mut imported_tex_states,
                );
                if let Some(resolve_h) = color_desc.resolve_target {
                    transition_tex(
                        resolve_h,
                        TextureSubresourceRange::single(0, 0),
                        ResourceState {
                            stage: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                            layout: ImageLayout::ColorAttachment,
                        },
                        &mut pre_pass_barrier.image_barriers,
                        &mut transient_tex_states,
                        &mut imported_tex_states,
                    );
                }
            }

            // Depth attachment (write or read-only: RND-05 / G7, RND-10)
            if let Some(depth_desc) = &pass.write_depth {
                let access = if depth_desc.read_only {
                    AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                } else {
                    AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                };
                transition_tex(
                    depth_desc.handle,
                    TextureSubresourceRange::single(depth_desc.mip_level, depth_desc.layer),
                    ResourceState {
                        stage: PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        access,
                        layout: ImageLayout::DepthStencilAttachment,
                    },
                    &mut pre_pass_barrier.image_barriers,
                    &mut transient_tex_states,
                    &mut imported_tex_states,
                );
            }

            // Buffer reads & writes
            for (buf_h, access) in &pass.read_buffers {
                transition_buf(
                    *buf_h,
                    BufferState {
                        stage: access.stage,
                        access: access.access,
                    },
                    &mut pre_pass_barrier.buffer_barriers,
                    &mut transient_buf_states,
                    &mut imported_buf_states,
                );
            }
            for (buf_h, access) in &pass.write_buffers {
                transition_buf(
                    *buf_h,
                    BufferState {
                        stage: access.stage,
                        access: access.access,
                    },
                    &mut pre_pass_barrier.buffer_barriers,
                    &mut transient_buf_states,
                    &mut imported_buf_states,
                );
            }

            // RenderingDescriptor for Graphics pass
            let rendering_desc = if pass.pass_type == PassType::Graphics
                && (!pass.write_colors.is_empty() || pass.write_depth.is_some())
            {
                let color_attachments: korppi_rhi::ColorAttachments = pass
                    .write_colors
                    .iter()
                    .map(|c| RenderingAttachment {
                        texture: out.texture_slice[c.handle.id() as usize]
                            .expect("resolved color attachment"),
                        mip_level: c.mip_level,
                        layer: c.layer,
                        load_op: c.load_op,
                        store_op: c.store_op,
                        clear_color: Some(c.clear_color),
                        resolve_target: c.resolve_target.map(|rh| {
                            out.texture_slice[rh.id() as usize].expect("resolved resolve_target")
                        }),
                    })
                    .collect();

                let depth_attachment =
                    pass.write_depth
                        .as_ref()
                        .map(|d| korppi_rhi::descriptor::DepthAttachment {
                            texture: out.texture_slice[d.handle.id() as usize]
                                .expect("resolved depth attachment"),
                            mip_level: d.mip_level,
                            layer: d.layer,
                            load_op: d.load_op,
                            store_op: d.store_op,
                            clear_depth: if d.load_op == LoadOp::Clear {
                                Some(d.clear_depth)
                            } else {
                                None
                            },
                            clear_stencil: None,
                            stencil_load_op: LoadOp::DontCare,
                            stencil_store_op: StoreOp::DontCare,
                        });

                let (width, height) = if let Some(first_color) = pass.write_colors.first() {
                    let (w, h, _, _) =
                        self.textures[first_color.handle.id() as usize].extent_and_subresources();
                    (
                        (w >> first_color.mip_level).max(1),
                        (h >> first_color.mip_level).max(1),
                    )
                } else if let Some(d) = &pass.write_depth {
                    let (w, h, _, _) =
                        self.textures[d.handle.id() as usize].extent_and_subresources();
                    ((w >> d.mip_level).max(1), (h >> d.mip_level).max(1))
                } else {
                    (0, 0)
                };

                Some(RenderingDescriptor {
                    offset_x: 0,
                    offset_y: 0,
                    width,
                    height,
                    color_attachments,
                    depth_attachment,
                })
            } else {
                None
            };

            out.passes.push(CompiledPass {
                name: pass.name.clone(),
                pass_type: pass.pass_type,
                pre_pass_barrier,
                rendering_desc,
                callback: pass.callback.take(),
            });
        }

        // 6. Post-Graph Final State Transitions for Imported Resources & Pool State Persistence (RND-05)
        let mut post_image_barriers: SmallVec<[ImageMemoryBarrier; 4]> = SmallVec::new();
        let mut post_buffer_barriers: SmallVec<[BufferMemoryBarrier; 4]> = SmallVec::new();

        for (idx, v_tex) in self.textures.iter().enumerate() {
            if let VirtualTextureKind::Imported {
                final_access: Some(final_access),
                ..
            } = &v_tex.kind
            {
                if let Some(state) = &mut imported_tex_states[idx] {
                    state.transition(
                        TextureSubresourceRange::ALL,
                        ResourceState {
                            stage: final_access.stage,
                            access: final_access.access,
                            layout: final_access.layout,
                        },
                        &mut post_image_barriers,
                    );
                }
            }
        }

        for (idx, v_buf) in self.buffers.iter().enumerate() {
            if let VirtualBufferKind::Imported {
                final_access: Some(final_access),
                ..
            } = &v_buf.kind
            {
                if let Some(state) = &mut imported_buf_states[idx] {
                    state.transition(
                        BufferState {
                            stage: final_access.stage,
                            access: final_access.access,
                        },
                        &mut post_buffer_barriers,
                    );
                }
            }
        }

        if let Some((_, ref mut pool)) = device_and_pool {
            for (slot, state) in tex_slots.iter().zip(transient_tex_states.iter()) {
                if slot.is_pooled {
                    if let Some(access) = state.summary_access() {
                        pool.set_texture_last_access(slot.physical_handle, access);
                    }
                }
            }
            for (slot, state) in buf_slots.iter().zip(transient_buf_states.iter()) {
                if slot.is_pooled {
                    if let Some(curr) = state.current {
                        pool.set_buffer_last_access(
                            slot.physical_handle,
                            BufferAccess {
                                stage: curr.stage,
                                access: curr.access,
                            },
                        );
                    }
                }
            }
        }

        let post_graph_barrier = PipelineBarrier {
            image_barriers: post_image_barriers,
            buffer_barriers: post_buffer_barriers,
            memory_barriers: Default::default(),
        };

        Ok(CompiledGraphOutput {
            passes: out.passes,
            barrier_pool: out.barrier_pool,
            texture_slice: out.texture_slice,
            buffer_slice: out.buffer_slice,
            texture_map: out.texture_map,
            buffer_map: out.buffer_map,
            texture_allocations: out.texture_allocations,
            buffer_allocations: out.buffer_allocations,
            aliasing_report,
            post_graph_barrier,
        })
    }
}
