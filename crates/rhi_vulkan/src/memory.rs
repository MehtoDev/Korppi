//! Device memory for buffers and images (finding RVK-08, ADR-014).
//!
//! - **Memory types** are chosen VMA-style by usage: required flags, then the most preferred
//!   ones (`GpuOnly`: device-local, preferably not host-visible; `CpuToGpu`: host-visible and
//!   coherent, preferably device-local; `GpuToCpu`: host-visible and coherent, preferably
//!   cached, since reading write-combined memory is slow).
//! - **Pools** are per memory type and resource kind: buffers and optimal-tiling images never
//!   share a chunk, so `bufferImageGranularity` cannot make them alias.
//! - **Chunks** are 256 MiB (64 MiB for host-only memory, at most an eighth of a small heap),
//!   starting at an eighth of that and doubling, so small apps stay small. Allocations of more
//!   than half a chunk, and resources the driver wants a dedicated allocation for, get their
//!   own `VkDeviceMemory`. Nothing exceeds `maxMemoryAllocationSize`.
//! - **Budget**: with `VK_EXT_memory_budget` the driver's per-heap budget and usage, otherwise
//!   80 % of the heap and this manager's own allocations. A new allocation that would exceed
//!   the budget goes to the next acceptable memory type.
//! - Host-visible memory is mapped persistently. An emptied chunk is kept for reuse while it is
//!   the pool's only empty one.

use crate::context::VulkanContext;
use ash::vk;
use korppi_rhi::{MemoryUsage, RhiError};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Chunk size for device-local memory (ADR-014).
pub const DEFAULT_CHUNK_SIZE: u64 = 256 * 1024 * 1024;

/// Chunk size for host-visible memory that is not device-local.
pub const HOST_CHUNK_SIZE: u64 = 64 * 1024 * 1024;

/// Fraction of a heap assumed usable without `VK_EXT_memory_budget`.
const ASSUMED_BUDGET_PERCENT: u64 = 80;

/// Whether a resource is linear (buffers) or optimal-tiling (images); they use separate pools.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// Buffers.
    Buffer,
    /// Optimal-tiling images.
    Image,
}

/// The resource a dedicated allocation is made for.
#[derive(Copy, Clone, Debug)]
pub enum DedicatedResource {
    /// A buffer.
    Buffer(vk::Buffer),
    /// An image.
    Image(vk::Image),
}

/// What to allocate memory for.
#[derive(Copy, Clone, Debug)]
pub struct AllocationRequest {
    /// The resource's memory requirements.
    pub requirements: vk::MemoryRequirements,
    /// Intended access.
    pub usage: MemoryUsage,
    /// Pool kind.
    pub kind: ResourceKind,
    /// The resource, if the driver prefers or requires a dedicated allocation for it.
    pub dedicated: Option<DedicatedResource>,
}

/// Suballocated memory block handle.
#[derive(Clone, Copy, Debug)]
pub struct Suballocation {
    /// Unique allocation identifier.
    pub id: u32,
    /// Parent VkDeviceMemory (a chunk, or the allocation's own memory if `dedicated`).
    pub device_memory: vk::DeviceMemory,
    /// Byte offset within the memory.
    pub offset: u64,
    /// Size of this suballocation in bytes.
    pub size: u64,
    /// Mapped host CPU pointer, or null if the memory is not host-visible.
    pub mapped_ptr: *mut u8,
    /// Memory type of the allocation.
    pub memory_type_index: u32,
    /// Whether the allocation owns its `VkDeviceMemory`.
    pub dedicated: bool,
}

// SAFETY: `mapped_ptr` points into persistently mapped device memory, valid on any thread
// until the allocation is freed; access to it is synchronized by its owners.
unsafe impl Send for Suballocation {}
// SAFETY: as above.
unsafe impl Sync for Suballocation {}

/// Free intervals of a chunk, indexed by offset (to coalesce neighbours) and by size (to pick
/// the best fit), so allocating and freeing are O(log n) in the number of intervals (RVK-08).
#[derive(Debug, Default)]
struct FreeList {
    by_offset: BTreeMap<u64, u64>,
    by_size: BTreeSet<(u64, u64)>,
}

/// Smallest fitting candidates examined before taking a block that fits at any alignment.
const BEST_FIT_CANDIDATES: usize = 4;

impl FreeList {
    /// A list with one free interval of `size` bytes at offset 0.
    fn new(size: u64) -> Self {
        let mut list = Self::default();
        list.insert(0, size);
        list
    }

    fn insert(&mut self, offset: u64, size: u64) {
        self.by_offset.insert(offset, size);
        self.by_size.insert((size, offset));
    }

    fn remove(&mut self, offset: u64, size: u64) {
        self.by_offset.remove(&offset);
        self.by_size.remove(&(size, offset));
    }

    /// Takes `size` bytes at `alignment` (a power of two) from the smallest interval that holds
    /// them; alignment padding stays free.
    fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let alignment = alignment.max(1);
        let fits = |&(block_size, offset): &(u64, u64)| {
            offset.next_multiple_of(alignment) - offset + size <= block_size
        };
        // The smallest intervals that might fit, then the smallest that fits at any offset.
        let (block_size, offset) = self
            .by_size
            .range((size, 0)..)
            .take(BEST_FIT_CANDIDATES)
            .find(|block| fits(block))
            .or_else(|| {
                self.by_size
                    .range((size.saturating_add(alignment - 1), 0)..)
                    .next()
            })
            .copied()?;
        self.remove(offset, block_size);
        let aligned = offset.next_multiple_of(alignment);
        if aligned > offset {
            self.insert(offset, aligned - offset);
        }
        let end = aligned + size;
        if end < offset + block_size {
            self.insert(end, offset + block_size - end);
        }
        Some(aligned)
    }

    /// Returns `size` bytes at `offset`, merging them with free neighbours.
    fn free(&mut self, mut offset: u64, mut size: u64) {
        if let Some((&before, &before_size)) = self.by_offset.range(..offset).next_back() {
            if before + before_size == offset {
                self.remove(before, before_size);
                offset = before;
                size += before_size;
            }
        }
        if let Some(&after_size) = self.by_offset.get(&(offset + size)) {
            self.remove(offset + size, after_size);
            size += after_size;
        }
        self.insert(offset, size);
    }

    /// Free bytes in all intervals.
    #[cfg(test)]
    fn free_bytes(&self) -> u64 {
        self.by_offset.values().sum()
    }

    /// Number of free intervals.
    #[cfg(test)]
    fn blocks(&self) -> usize {
        self.by_offset.len()
    }
}

struct Chunk {
    device_memory: vk::DeviceMemory,
    memory_type_index: u32,
    kind: ResourceKind,
    size: u64,
    free: FreeList,
    used_bytes: u64,
    mapped_ptr: *mut u8,
}

impl Chunk {
    fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let offset = self.free.allocate(size, alignment)?;
        self.used_bytes += size;
        Some(offset)
    }

    fn deallocate(&mut self, offset: u64, size: u64) {
        self.free.free(offset, size);
        self.used_bytes = self.used_bytes.saturating_sub(size);
    }
}

/// Budget and usage of one memory heap.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HeapBudget {
    /// Heap size in bytes.
    pub size: u64,
    /// Bytes the process may use (driver budget, or 80 % of the heap without
    /// `VK_EXT_memory_budget`).
    pub budget: u64,
    /// Bytes the process uses (driver-reported, or this manager's allocations).
    pub usage: u64,
    /// Bytes this manager has allocated from the heap (chunks and dedicated allocations).
    pub allocated: u64,
    /// Whether the heap is device-local.
    pub device_local: bool,
}

#[derive(Default)]
struct State {
    chunks: Vec<Chunk>,
    /// Dedicated allocations: memory, type, size, mapping.
    dedicated: Vec<(vk::DeviceMemory, u32, u64, *mut u8)>,
}

// SAFETY: the raw pointers are persistent mappings, valid on any thread; the state is only
// accessed under the manager's mutex.
unsafe impl Send for State {}

/// Chunk-based memory allocator (ADR-014; see the module documentation).
pub struct VulkanMemoryManager {
    context: Arc<VulkanContext>,
    state: Mutex<State>,
    alloc_id_counter: AtomicU32,
}

/// Required and preferred memory property flags for `usage`, and flags to avoid.
fn usage_flags(
    usage: MemoryUsage,
) -> (
    vk::MemoryPropertyFlags,
    vk::MemoryPropertyFlags,
    vk::MemoryPropertyFlags,
) {
    use vk::MemoryPropertyFlags as F;
    match usage {
        MemoryUsage::GpuOnly => (F::empty(), F::DEVICE_LOCAL, F::HOST_VISIBLE),
        MemoryUsage::CpuToGpu => (
            F::HOST_VISIBLE | F::HOST_COHERENT,
            F::DEVICE_LOCAL,
            F::HOST_CACHED,
        ),
        MemoryUsage::GpuToCpu => (
            F::HOST_VISIBLE | F::HOST_COHERENT,
            F::HOST_CACHED,
            F::empty(),
        ),
    }
}

/// Memory types in `type_bits` acceptable for `usage`, best first.
pub fn memory_type_candidates(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    usage: MemoryUsage,
) -> Vec<u32> {
    let (required, preferred, avoided) = usage_flags(usage);
    // Lazily allocated and protected memory are never wanted here.
    let excluded = vk::MemoryPropertyFlags::LAZILY_ALLOCATED | vk::MemoryPropertyFlags::PROTECTED;
    let mut candidates: Vec<(i32, u32)> = (0..properties.memory_type_count)
        .filter(|&i| type_bits & (1 << i) != 0)
        .filter_map(|i| {
            let flags = properties.memory_types[i as usize].property_flags;
            if !flags.contains(required) || flags.intersects(excluded) {
                return None;
            }
            let score = (flags & preferred).as_raw().count_ones() as i32 * 2
                - (flags & avoided).as_raw().count_ones() as i32;
            Some((score, i))
        })
        .collect();
    // Best score first; equal scores keep the driver's order.
    candidates.sort_by_key(|&(score, index)| (-score, index));
    candidates.into_iter().map(|(_, index)| index).collect()
}

impl VulkanMemoryManager {
    /// Creates a new chunk memory manager for the given Vulkan context.
    pub fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            state: Mutex::new(State::default()),
            alloc_id_counter: AtomicU32::new(1),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Number of `VkDeviceMemory` objects (chunks and dedicated allocations).
    pub fn chunk_count(&self) -> usize {
        let state = self.lock();
        state.chunks.len() + state.dedicated.len()
    }

    /// Bytes of `VkDeviceMemory` allocated (chunks and dedicated allocations).
    pub fn allocated_bytes(&self) -> u64 {
        let state = self.lock();
        state.chunks.iter().map(|c| c.size).sum::<u64>()
            + state.dedicated.iter().map(|d| d.2).sum::<u64>()
    }

    /// Bytes in use by live allocations.
    pub fn used_bytes(&self) -> u64 {
        let state = self.lock();
        state.chunks.iter().map(|c| c.used_bytes).sum::<u64>()
            + state.dedicated.iter().map(|d| d.2).sum::<u64>()
    }

    fn heap_of(&self, memory_type: u32) -> u32 {
        self.context.memory_properties.memory_types[memory_type as usize].heap_index
    }

    /// Bytes this manager has allocated per heap.
    fn allocated_per_heap(&self, state: &State) -> [u64; vk::MAX_MEMORY_HEAPS] {
        let mut per_heap = [0; vk::MAX_MEMORY_HEAPS];
        for chunk in &state.chunks {
            per_heap[self.heap_of(chunk.memory_type_index) as usize] += chunk.size;
        }
        for &(_, memory_type, size, _) in &state.dedicated {
            per_heap[self.heap_of(memory_type) as usize] += size;
        }
        per_heap
    }

    fn budgets(&self, state: &State) -> Vec<HeapBudget> {
        let properties = &self.context.memory_properties;
        let allocated = self.allocated_per_heap(state);
        let mut driver = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        if self.context.memory_budget_enabled {
            let mut properties2 =
                vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut driver);
            // SAFETY: the physical device belongs to the instance; VK_EXT_memory_budget is
            // enabled, so the chained structure is filled in.
            unsafe {
                self.context
                    .instance
                    .get_physical_device_memory_properties2(
                        self.context.physical_device,
                        &mut properties2,
                    )
            };
        }
        (0..properties.memory_heap_count as usize)
            .map(|heap| {
                let size = properties.memory_heaps[heap].size;
                let (budget, usage) = if self.context.memory_budget_enabled {
                    (driver.heap_budget[heap], driver.heap_usage[heap])
                } else {
                    (size / 100 * ASSUMED_BUDGET_PERCENT, allocated[heap])
                };
                HeapBudget {
                    size,
                    budget,
                    usage,
                    allocated: allocated[heap],
                    device_local: properties.memory_heaps[heap]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                }
            })
            .collect()
    }

    /// Budget and usage of every memory heap.
    pub fn heap_budgets(&self) -> Vec<HeapBudget> {
        let state = self.lock();
        self.budgets(&state)
    }

    /// Full chunk size for a memory type: 256 MiB, 64 MiB for host-only memory, at most an
    /// eighth of the heap.
    fn preferred_chunk_size(&self, memory_type: u32) -> u64 {
        let memory = &self.context.memory_properties;
        let flags = memory.memory_types[memory_type as usize].property_flags;
        let heap = memory.memory_heaps[self.heap_of(memory_type) as usize].size;
        let base = if flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            DEFAULT_CHUNK_SIZE
        } else {
            HOST_CHUNK_SIZE
        };
        base.min(heap / 8)
            .min(self.context.max_memory_allocation_size)
            .max(1 << 20)
    }

    fn map(&self, memory: vk::DeviceMemory, memory_type: u32) -> Result<*mut u8, RhiError> {
        let flags =
            self.context.memory_properties.memory_types[memory_type as usize].property_flags;
        if !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            return Ok(std::ptr::null_mut());
        }
        // SAFETY: `memory` was just allocated from a host-visible type and is not mapped.
        unsafe {
            self.context
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map(|ptr| ptr.cast::<u8>())
        .map_err(|e| RhiError::BufferMappingFailed(format!("vkMapMemory failed: {e}")))
    }

    fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        dedicated: Option<DedicatedResource>,
    ) -> Result<vk::DeviceMemory, vk::Result> {
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default();
        match dedicated {
            Some(DedicatedResource::Buffer(buffer)) => {
                dedicated_info = dedicated_info.buffer(buffer)
            }
            Some(DedicatedResource::Image(image)) => dedicated_info = dedicated_info.image(image),
            None => {}
        }
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type);
        if dedicated.is_some() {
            info = info.push_next(&mut dedicated_info);
        }
        // SAFETY: `info` is valid; the memory is freed by `deallocate` or on drop.
        unsafe { self.context.device.allocate_memory(&info, None) }
    }

    /// Allocates memory satisfying `request` (see the module documentation).
    ///
    /// # Errors
    /// `RhiError::OutOfMemory` if no acceptable memory type can hold the allocation.
    pub fn allocate(&self, request: &AllocationRequest) -> Result<Suballocation, RhiError> {
        let requirements = request.requirements;
        if requirements.size > self.context.max_memory_allocation_size {
            return Err(RhiError::OutOfMemory(format!(
                "{} bytes exceed maxMemoryAllocationSize ({})",
                requirements.size, self.context.max_memory_allocation_size
            )));
        }
        let candidates = memory_type_candidates(
            &self.context.memory_properties,
            requirements.memory_type_bits,
            request.usage,
        );
        if candidates.is_empty() {
            return Err(RhiError::OutOfMemory(format!(
                "no memory type in {:#x} suits {:?}",
                requirements.memory_type_bits, request.usage
            )));
        }
        let mut state = self.lock();

        // 1. Room in an existing chunk of an acceptable type, best type first.
        for &memory_type in &candidates {
            if request.dedicated.is_some() {
                break;
            }
            for chunk in state
                .chunks
                .iter_mut()
                .filter(|c| c.memory_type_index == memory_type && c.kind == request.kind)
            {
                if let Some(offset) = chunk.allocate(requirements.size, requirements.alignment) {
                    let mapped_ptr = if chunk.mapped_ptr.is_null() {
                        std::ptr::null_mut()
                    } else {
                        // SAFETY: the offset lies within the chunk's mapping.
                        unsafe { chunk.mapped_ptr.add(offset as usize) }
                    };
                    return Ok(Suballocation {
                        id: self.alloc_id_counter.fetch_add(1, Ordering::Relaxed),
                        device_memory: chunk.device_memory,
                        offset,
                        size: requirements.size,
                        mapped_ptr,
                        memory_type_index: memory_type,
                        dedicated: false,
                    });
                }
            }
        }

        // 2. New memory from the first type whose heap has budget left (or, failing that, the
        // best type anyway: the budget is advisory).
        let budgets = self.budgets(&state);
        let mut failure = None;
        let mut fallback = true;
        for (attempt, &memory_type) in candidates.iter().chain(candidates.first()).enumerate() {
            let last_resort = attempt == candidates.len();
            if last_resort && !fallback {
                break;
            }
            let preferred = self.preferred_chunk_size(memory_type);
            let dedicated = request.dedicated.is_some() || requirements.size > preferred / 2;
            let pool_chunks = state
                .chunks
                .iter()
                .filter(|c| c.memory_type_index == memory_type && c.kind == request.kind)
                .count() as u32;
            // Start at an eighth of the full size and double per chunk.
            let size = if dedicated {
                requirements.size
            } else {
                (preferred >> 3u32.saturating_sub(pool_chunks)).max(requirements.size)
            };
            let heap = &budgets[self.heap_of(memory_type) as usize];
            if !last_resort && heap.usage + size > heap.budget {
                continue;
            }
            fallback = false;
            match self.allocate_memory(size, memory_type, request.dedicated.filter(|_| dedicated)) {
                Ok(memory) => {
                    let mapped = match self.map(memory, memory_type) {
                        Ok(mapped) => mapped,
                        Err(error) => {
                            // SAFETY: the memory was just allocated and is unused.
                            unsafe { self.context.device.free_memory(memory, None) };
                            return Err(error);
                        }
                    };
                    let id = self.alloc_id_counter.fetch_add(1, Ordering::Relaxed);
                    if dedicated {
                        state.dedicated.push((memory, memory_type, size, mapped));
                        return Ok(Suballocation {
                            id,
                            device_memory: memory,
                            offset: 0,
                            size,
                            mapped_ptr: mapped,
                            memory_type_index: memory_type,
                            dedicated: true,
                        });
                    }
                    let mut chunk = Chunk {
                        device_memory: memory,
                        memory_type_index: memory_type,
                        kind: request.kind,
                        size,
                        free: FreeList::new(size),
                        used_bytes: 0,
                        mapped_ptr: mapped,
                    };
                    let offset = chunk
                        .allocate(requirements.size, requirements.alignment)
                        .expect("a new chunk holds the allocation");
                    let mapped_ptr = if mapped.is_null() {
                        mapped
                    } else {
                        // SAFETY: the offset lies within the chunk's mapping.
                        unsafe { mapped.add(offset as usize) }
                    };
                    state.chunks.push(chunk);
                    return Ok(Suballocation {
                        id,
                        device_memory: memory,
                        offset,
                        size: requirements.size,
                        mapped_ptr,
                        memory_type_index: memory_type,
                        dedicated: false,
                    });
                }
                Err(error) => failure = Some(error),
            }
        }
        Err(RhiError::OutOfMemory(format!(
            "vkAllocateMemory failed for {} bytes ({:?}): {}",
            requirements.size,
            request.usage,
            failure.map_or("every heap is over budget".to_string(), |e| e.to_string())
        )))
    }

    /// Returns memory; frees dedicated allocations, and chunks that are empty while another
    /// chunk of their pool is too. The memory must no longer be in use by the GPU.
    pub fn deallocate(&self, suballoc: &Suballocation) {
        let mut state = self.lock();
        if suballoc.dedicated {
            if let Some(index) = state
                .dedicated
                .iter()
                .position(|d| d.0 == suballoc.device_memory)
            {
                let (memory, _, _, mapped) = state.dedicated.swap_remove(index);
                // SAFETY: the caller guarantees the GPU no longer uses the memory; freeing
                // mapped memory unmaps it.
                unsafe {
                    if !mapped.is_null() {
                        self.context.device.unmap_memory(memory);
                    }
                    self.context.device.free_memory(memory, None);
                }
            }
            return;
        }
        let Some(index) = state
            .chunks
            .iter()
            .position(|c| c.device_memory == suballoc.device_memory)
        else {
            return;
        };
        state.chunks[index].deallocate(suballoc.offset, suballoc.size);
        let chunk = &state.chunks[index];
        if chunk.used_bytes != 0 {
            return;
        }
        let other_empty = state.chunks.iter().enumerate().any(|(i, c)| {
            i != index
                && c.used_bytes == 0
                && c.memory_type_index == chunk.memory_type_index
                && c.kind == chunk.kind
        });
        if other_empty {
            let chunk = state.chunks.swap_remove(index);
            // SAFETY: the chunk is empty, so no live allocation uses it.
            unsafe {
                if !chunk.mapped_ptr.is_null() {
                    self.context.device.unmap_memory(chunk.device_memory);
                }
                self.context.device.free_memory(chunk.device_memory, None);
            }
        }
    }
}

impl Drop for VulkanMemoryManager {
    fn drop(&mut self) {
        let mut state = self.lock();
        let chunks = std::mem::take(&mut state.chunks);
        let dedicated = std::mem::take(&mut state.dedicated);
        let device = &self.context.device;
        // SAFETY: the owning device waits for idle before the manager drops; nothing uses the
        // memory any more.
        unsafe {
            for chunk in chunks {
                if !chunk.mapped_ptr.is_null() {
                    device.unmap_memory(chunk.device_memory);
                }
                device.free_memory(chunk.device_memory, None);
            }
            for (memory, _, _, mapped) in dedicated {
                if !mapped.is_null() {
                    device.unmap_memory(memory);
                }
                device.free_memory(memory, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn properties(types: &[vk::MemoryPropertyFlags]) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: types.len() as u32,
            memory_heap_count: 1,
            ..Default::default()
        };
        for (i, &flags) in types.iter().enumerate() {
            props.memory_types[i] = vk::MemoryType {
                property_flags: flags,
                heap_index: 0,
            };
        }
        props
    }

    #[test]
    fn memory_types_follow_usage_preferences() {
        use vk::MemoryPropertyFlags as F;
        // A discrete GPU: VRAM, BAR, host write-combined, host cached.
        let props = properties(&[
            F::DEVICE_LOCAL,
            F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT,
            F::HOST_VISIBLE | F::HOST_COHERENT,
            F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
        ]);
        let all = 0b1111;
        assert_eq!(
            memory_type_candidates(&props, all, MemoryUsage::GpuOnly)[0],
            0
        );
        assert_eq!(
            memory_type_candidates(&props, all, MemoryUsage::CpuToGpu),
            [1, 2, 3]
        );
        assert_eq!(
            memory_type_candidates(&props, all, MemoryUsage::GpuToCpu)[0],
            3
        );
        // The resource's type bits restrict the choice.
        assert_eq!(
            memory_type_candidates(&props, 0b0100, MemoryUsage::GpuOnly),
            [2]
        );
        assert!(memory_type_candidates(&props, 0b0001, MemoryUsage::GpuToCpu).is_empty());
    }

    #[test]
    fn chunks_coalesce_freed_neighbours() {
        let mut chunk = Chunk {
            device_memory: vk::DeviceMemory::null(),
            memory_type_index: 0,
            kind: ResourceKind::Buffer,
            size: 1024,
            free: FreeList::new(1024),
            used_bytes: 0,
            mapped_ptr: std::ptr::null_mut(),
        };
        let a = chunk.allocate(100, 64).unwrap();
        let b = chunk.allocate(100, 64).unwrap();
        let c = chunk.allocate(100, 64).unwrap();
        assert_eq!((a, b, c), (0, 128, 256));
        chunk.deallocate(a, 100);
        chunk.deallocate(c, 100);
        chunk.deallocate(b, 100);
        assert_eq!((chunk.free.blocks(), chunk.free.free_bytes()), (1, 1024));
        assert_eq!(chunk.used_bytes, 0);
    }

    #[test]
    fn free_lists_pick_the_best_fit() {
        let mut list = FreeList::new(4096);
        let a = list.allocate(1000, 1).unwrap();
        let _b = list.allocate(100, 1).unwrap();
        let c = list.allocate(300, 1).unwrap();
        let _d = list.allocate(100, 1).unwrap();
        // Holes of 1000 (at 0) and 300 (at 1100), plus the tail.
        list.free(a, 1000);
        list.free(c, 300);
        assert_eq!(
            list.allocate(250, 1),
            Some(c),
            "the 300-byte hole fits best"
        );
        assert_eq!(list.allocate(900, 1), Some(a), "then the 1000-byte one");
        // Alignment padding stays free and is reused.
        let mut list = FreeList::new(1024);
        assert_eq!(list.allocate(10, 1), Some(0));
        assert_eq!(list.allocate(100, 256), Some(256));
        assert_eq!(list.allocate(200, 1), Some(10), "the padding before 256");
        assert_eq!(list.free_bytes(), 1024 - 310);
    }

    #[test]
    fn free_lists_stay_consistent_under_random_use() {
        const SIZE: u64 = 1 << 20;
        let mut list = FreeList::new(SIZE);
        let mut live: Vec<(u64, u64)> = Vec::new();
        // A fixed-seed xorshift: reproducible without a dependency.
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            if live.is_empty() || next() % 3 != 0 {
                let size = 1 + next() % 4096;
                let alignment = 1 << (next() % 9);
                if let Some(offset) = list.allocate(size, alignment) {
                    assert_eq!(offset % alignment, 0);
                    assert!(offset + size <= SIZE);
                    live.push((offset, size));
                }
            } else {
                let (offset, size) = live.swap_remove((next() % live.len() as u64) as usize);
                list.free(offset, size);
            }
        }
        live.sort_unstable();
        for pair in live.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0, "overlap: {pair:?}");
        }
        let used: u64 = live.iter().map(|(_, size)| size).sum();
        assert_eq!(list.free_bytes() + used, SIZE);
        for (offset, size) in live {
            list.free(offset, size);
        }
        assert_eq!(list.blocks(), 1, "everything coalesces back into one block");
        assert_eq!(list.free_bytes(), SIZE);
    }
}
