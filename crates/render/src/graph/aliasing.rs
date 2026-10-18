//! Transient resource memory footprint analysis, lifetime interval bin-packing, and pool reporting.
//!
//! `TransientMemoryHeapAllocator` computes theoretical byte-offset placements over `[first_pass, last_pass]`
//! intervals, while `MemoryAliasingReport` records both theoretical heap-offset aliasing and the actual
//! descriptor-matched physical resource pooling performed by `TransientResourcePool` (`RND-14`).

use crate::graph::resource::TransientTextureDesc;
use korppi_rhi::{SampleCount, SmallVec};

/// Physical memory allocation record for an aliased transient resource within a shared memory heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhysicalResourceAllocation {
    /// Byte offset within the shared transient memory heap block.
    pub offset: u64,
    /// Allocated byte size including hardware alignment padding.
    pub size: u64,
    /// Physical memory heap block index.
    pub heap_index: usize,
}

/// Statistics and memory footprints calculated during transient resource aliasing and pooling (`RND-14`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemoryAliasingReport {
    /// Total bytes that would be required if every transient resource had its own dedicated allocation.
    pub unaliased_bytes: u64,
    /// Peak bytes required by 1D interval bin-packing across pass lifetimes.
    pub aliased_bytes: u64,
    /// Theoretical bytes saved by lifetime interval bin-packing.
    pub bytes_saved: u64,
    /// Theoretical memory savings ratio in $[0.0, 1.0]$.
    pub savings_ratio: f32,
    /// Total number of virtual transient resources declared in the graph.
    pub transient_resource_count: usize,
    /// Actual number of distinct physical textures required after descriptor-matched lifetime pooling.
    pub physical_texture_count: usize,
    /// Actual number of distinct physical buffers required after descriptor-matched lifetime pooling.
    pub physical_buffer_count: usize,
    /// Actual bytes allocated across distinct descriptor-pooled physical resources.
    pub physical_bytes: u64,
    /// Actual bytes saved by reusing descriptor-matched physical handles across non-overlapping passes.
    pub physical_bytes_saved: u64,
}

/// Computes the exact byte footprint of a transient texture based on dimensions, format, and MSAA samples.
pub fn compute_transient_texture_byte_size(desc: &TransientTextureDesc) -> u64 {
    let sample_count = match desc.sample_count {
        SampleCount::Count1 => 1,
        SampleCount::Count2 => 2,
        SampleCount::Count4 => 4,
        SampleCount::Count8 => 8,
    };
    desc.format.image_bytes(desc.width, desc.height) * sample_count
}

/// Computes the byte footprint of a transient texture across all mip levels and array layers.
pub fn compute_transient_texture_total_bytes(
    desc: &TransientTextureDesc,
    mip_levels: u32,
    array_layers: u32,
) -> u64 {
    let sample_count = match desc.sample_count {
        SampleCount::Count1 => 1,
        SampleCount::Count2 => 2,
        SampleCount::Count4 => 4,
        SampleCount::Count8 => 8,
    };
    let mut mip_sum = 0u64;
    for mip in 0..mip_levels.max(1) {
        let w = (desc.width >> mip).max(1);
        let h = (desc.height >> mip).max(1);
        mip_sum += desc.format.image_bytes(w, h);
    }
    mip_sum * sample_count * u64::from(array_layers.max(1))
}

/// Allocated byte range record with pass execution lifetimes.
#[derive(Clone, Copy, Debug)]
pub struct AllocatedMemoryChunk {
    /// Byte offset in the heap.
    pub offset: u64,
    /// Allocated byte size including alignment.
    pub size: u64,
    /// Index of first pass accessing this chunk.
    pub first_pass: usize,
    /// Index of last pass accessing this chunk.
    pub last_pass: usize,
}

/// 1D interval bin-packing allocator packing transient resources into non-overlapping lifetime intervals.
///
/// Uses inline `SmallVec` storage so steady-state frame compilation performs zero heap allocations (`RND-11`).
pub struct TransientMemoryHeapAllocator {
    alignment: u64,
    chunks: SmallVec<[AllocatedMemoryChunk; 16]>,
    total_peak_bytes: u64,
}

impl TransientMemoryHeapAllocator {
    /// Creates a new heap allocator with the specified byte alignment requirement.
    pub fn new(alignment: u64) -> Self {
        Self {
            alignment: alignment.max(1),
            chunks: SmallVec::new(),
            total_peak_bytes: 0,
        }
    }

    /// Allocates an aligned memory slice in the heap for a resource active during `[first_pass, last_pass]`.
    pub fn allocate(
        &mut self,
        size: u64,
        first_pass: usize,
        last_pass: usize,
        heap_index: usize,
    ) -> PhysicalResourceAllocation {
        let aligned_size = (size + self.alignment - 1) & !(self.alignment - 1);

        // 1. Gather all candidate offsets inline without heap allocation
        let mut candidates: SmallVec<[u64; 16]> = SmallVec::new();
        candidates.push(0);
        for c in &self.chunks {
            let cand = (c.offset + c.size + self.alignment - 1) & !(self.alignment - 1);
            candidates.push(cand);
        }
        candidates.sort_unstable();
        candidates.dedup();

        // 2. Select the lowest candidate offset that has zero lifetime collisions
        let mut chosen_offset = 0u64;
        for cand in candidates {
            let cand_end = cand + aligned_size;
            let collides = self.chunks.iter().any(|c| {
                let c_end = c.offset + c.size;
                let bytes_overlap = cand < c_end && cand_end > c.offset;
                let passes_overlap = !(c.last_pass < first_pass || last_pass < c.first_pass);
                bytes_overlap && passes_overlap
            });

            if !collides {
                chosen_offset = cand;
                break;
            }
        }

        self.chunks.push(AllocatedMemoryChunk {
            offset: chosen_offset,
            size: aligned_size,
            first_pass,
            last_pass,
        });

        self.total_peak_bytes = self.total_peak_bytes.max(chosen_offset + aligned_size);

        PhysicalResourceAllocation {
            offset: chosen_offset,
            size: aligned_size,
            heap_index,
        }
    }

    /// Total peak byte size required by the heap.
    #[inline]
    pub fn peak_bytes(&self) -> u64 {
        self.total_peak_bytes
    }
}
