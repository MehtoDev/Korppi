//! Synchronization validation is on whenever the validation layer is (FIX-1.4): a missing
//! barrier between two writes is reported as an API error.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferTextureCopy, BufferUsage, DiagnosticSeverity,
    GraphicsDevice, ImageLayout, ImageMemoryBarrier, MemoryUsage, PipelineBarrier,
    PipelineStageFlags, SampleCount, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSubresourceRange, TextureUsage,
};
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn unsynchronized_writes_are_reported() {
    let device = VulkanDevice::create_headless().expect("Vulkan device");
    let disabled =
        std::env::var(korppi_rhi_vulkan::context::SYNC_VALIDATION_ENV_VAR).is_ok_and(|v| v == "0");
    if !device.api_validation_enabled() || disabled {
        eprintln!("validation layer not installed or synchronization validation off; skipping");
        return;
    }
    let texture = device
        .create_texture(
            &TextureDescriptor {
                width: 4,
                height: 4,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::COPY_DST,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("texture");
    let buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: 64,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            None,
        )
        .expect("buffer");
    let copy = BufferTextureCopy {
        buffer_offset: 0,
        buffer_row_pitch: 0,
        texture_mip_level: 0,
        texture_layer: 0,
        origin: [0, 0, 0],
        extent: [4, 4, 1],
    };
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(
        &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture,
            range: TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::TRANSFER,
            dst_access: AccessFlags::TRANSFER_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::TransferDst,
        }),
    );
    // Two writes to the same texels with no barrier between them.
    recorder.copy_buffer_to_texture(buffer, texture, &copy);
    recorder.copy_buffer_to_texture(buffer, texture, &copy);
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");
    let hazards = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == DiagnosticSeverity::Error && d.message.contains("hazard"))
        .count();
    assert!(hazards > 0, "the write-after-write hazard was not reported");
}
