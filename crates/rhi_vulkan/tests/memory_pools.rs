//! Memory manager policy (FIX-1.4; finding RVK-08): separate buffer and image pools, chunks
//! that start small and grow, dedicated allocations for large resources, one empty chunk kept
//! per pool, and heap budgets.

use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, SampleCount, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_vulkan::{VulkanDevice, DEFAULT_CHUNK_SIZE};

const MIB: u64 = 1024 * 1024;

fn buffer(device: &VulkanDevice, size: u64) -> korppi_rhi::BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuOnly,
                label: None,
            },
            None,
        )
        .expect("create_buffer")
}

#[test]
fn buffers_and_images_never_share_a_chunk() {
    let device = VulkanDevice::create_headless().expect("device");
    let b = buffer(&device, 64 * 1024);
    assert_eq!(device.memory_chunk_count(), 1);
    let t = device
        .create_texture(
            &TextureDescriptor {
                width: 64,
                height: 64,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::SAMPLED,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("create_texture");
    // `bufferImageGranularity`: the image gets a chunk of its own even though the buffer's
    // chunk has room and (on lavapipe) the same memory type.
    assert_eq!(device.memory_chunk_count(), 2);
    device.destroy_texture(t).expect("destroy");
    device.destroy_buffer(b).expect("destroy");
    device.wait_idle().expect("wait_idle");
    assert_no_api_errors(&device);
}

#[test]
fn chunks_start_small_grow_and_large_resources_get_their_own_memory() {
    let device = VulkanDevice::create_headless().expect("device");
    let first = buffer(&device, MIB);
    let first_chunk = device.memory_allocated_bytes();
    assert!(
        first_chunk < DEFAULT_CHUNK_SIZE,
        "the first chunk is a fraction of the full size: {first_chunk}"
    );
    // Fill the first chunk: the next one is twice as large.
    let mut buffers = vec![first];
    while device.memory_chunk_count() == 1 {
        buffers.push(buffer(&device, MIB));
    }
    let second_chunk = device.memory_allocated_bytes() - first_chunk;
    assert_eq!(second_chunk, first_chunk * 2, "chunks double");

    // More than half a full chunk: a dedicated allocation of exactly its size.
    let before = (device.memory_chunk_count(), device.memory_allocated_bytes());
    let large_size = DEFAULT_CHUNK_SIZE / 2 + MIB;
    let large = buffer(&device, large_size);
    assert_eq!(device.memory_chunk_count(), before.0 + 1);
    assert!(device.memory_allocated_bytes() - before.1 >= large_size);
    device.destroy_buffer(large).expect("destroy");
    device.wait_idle().expect("wait_idle");
    assert_eq!(
        (device.memory_chunk_count(), device.memory_allocated_bytes()),
        before,
        "dedicated memory is freed at once"
    );

    // Emptying both chunks frees one of them and keeps the other for reuse.
    for b in buffers {
        device.destroy_buffer(b).expect("destroy");
    }
    device.wait_idle().expect("wait_idle");
    assert_eq!(device.memory_chunk_count(), 1);
    assert_eq!(device.memory_used_bytes(), 0);
    assert_no_api_errors(&device);
}

#[test]
fn heap_budgets_account_for_allocations() {
    let device = VulkanDevice::create_headless().expect("device");
    let before: u64 = device.memory_budgets().iter().map(|h| h.allocated).sum();
    let b = buffer(&device, MIB);
    let budgets = device.memory_budgets();
    assert!(!budgets.is_empty());
    let after: u64 = budgets.iter().map(|h| h.allocated).sum();
    assert_eq!(after - before, device.memory_allocated_bytes());
    for heap in &budgets {
        assert!(heap.budget > 0 && heap.budget <= heap.size, "{heap:?}");
    }
    device.destroy_buffer(b).expect("destroy");
    assert_no_api_errors(&device);
}

fn assert_no_api_errors(device: &dyn GraphicsDevice) {
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == korppi_rhi::DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "API errors reported: {errors:#?}");
}
