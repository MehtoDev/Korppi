//! Integration test verifying Vulkan memory suballocation reuse, coalescing, and zero leaks under allocation churn.

use korppi_rhi::{BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage};
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_vulkan_memory_churn_and_coalescing() {
    let device = VulkanDevice::create_headless().expect("Headless Vulkan device creation failed");

    let initial_chunk_count = device.memory_chunk_count();
    let initial_used_bytes = device.memory_used_bytes();
    assert_eq!(initial_chunk_count, 0);
    assert_eq!(initial_used_bytes, 0);

    // 1. Repeated allocate and destroy churn
    let desc = BufferDescriptor {
        size: 1024 * 1024, // 1 MiB
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("churn_buffer"),
    };

    for _ in 0..50 {
        let buf = device
            .create_buffer(&desc, None)
            .expect("Create buffer failed");
        assert!(device.memory_used_bytes() >= 1024 * 1024);
        assert_eq!(
            device.memory_chunk_count(),
            1,
            "Must not allocate new chunks when reusing memory"
        );
        device.destroy_buffer(buf).expect("Destroy buffer failed");
        assert_eq!(
            device.memory_used_bytes(),
            0,
            "Used bytes must drop back to 0 after destroy"
        );
    }

    // 2. Fragmented allocation and coalescing test
    let small_desc = BufferDescriptor {
        size: 256 * 1024, // 256 KiB
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory: MemoryUsage::CpuToGpu,
        label: Some("fragment_buffer"),
    };

    let mut buffers = Vec::new();
    for _ in 0..8 {
        buffers.push(
            device
                .create_buffer(&small_desc, None)
                .expect("Create buffer failed"),
        );
    }
    assert_eq!(device.memory_chunk_count(), 1);
    assert!(device.memory_used_bytes() >= 8 * 256 * 1024);

    // Destroy in interleaved order: even indices first, then odd
    for i in (0..8).step_by(2) {
        device
            .destroy_buffer(buffers[i])
            .expect("Destroy buffer failed");
    }
    for i in (1..8).step_by(2) {
        device
            .destroy_buffer(buffers[i])
            .expect("Destroy buffer failed");
    }

    // After all 8 are destroyed, they must coalesce completely into a single free block of chunk size
    assert_eq!(
        device.memory_used_bytes(),
        0,
        "All fragmented allocations must coalesce back to 0 used bytes"
    );

    // Can allocate a large contiguous buffer in the coalesced space
    let large_desc = BufferDescriptor {
        size: 2 * 1024 * 1024, // 2 MiB contiguous
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory: MemoryUsage::CpuToGpu,
        label: Some("large_coalesced_buffer"),
    };
    let large_buf = device
        .create_buffer(&large_desc, None)
        .expect("Must succeed in coalesced memory");
    assert_eq!(
        device.memory_chunk_count(),
        1,
        "Must fit inside the single coalesced chunk"
    );
    device
        .destroy_buffer(large_buf)
        .expect("Destroy buffer failed");
    assert_eq!(device.memory_used_bytes(), 0);
    assert_no_api_errors(&device);
}

/// Fails the test if the backend's API validation (Vulkan validation layers, OpenGL debug
/// output) reported errors (FIX-1.0).
fn assert_no_api_errors(device: &dyn korppi_rhi::GraphicsDevice) {
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == korppi_rhi::DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "API errors reported: {errors:#?}");
}
