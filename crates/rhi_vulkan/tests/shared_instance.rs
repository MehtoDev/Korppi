//! One Vulkan instance per process (ISSUE-015): every device shares it, it outlives every device
//! that uses it, and validation messages still reach only the device that caused them.
//!
//! Destroying an instance that presented to a Wayland surface while another instance in the
//! process presents crashed the other instance's presentation on the owner's machine; with one
//! instance, no instance is destroyed while a device exists.

use ash::vk;
use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferTextureCopy, BufferUsage, DiagnosticSeverity,
    GraphicsDevice, ImageLayout, ImageMemoryBarrier, MemoryUsage, PipelineBarrier,
    PipelineStageFlags, SampleCount, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSubresourceRange, TextureUsage,
};
use korppi_rhi_vulkan::context::SYNC_VALIDATION_ENV_VAR;
use korppi_rhi_vulkan::VulkanDevice;

fn buffer_round_trip(device: &VulkanDevice) {
    let buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: 256,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            None,
        )
        .expect("buffer");
    let bytes: Vec<u8> = (0..=255).collect();
    device.write_buffer(buffer, 0, &bytes).expect("write");
    assert_eq!(device.read_buffer(buffer, 0, 256).expect("read"), bytes);
    device.destroy_buffer(buffer).expect("destroy");
}

#[test]
fn devices_share_one_instance_that_outlives_each_of_them() {
    let first = VulkanDevice::create_headless().expect("first device");
    let second = VulkanDevice::create_headless().expect("second device");
    assert_eq!(
        first.context().instance.handle(),
        second.context().instance.handle(),
        "two devices in one process have different Vulkan instances"
    );
    drop(first);
    buffer_round_trip(&second);
    drop(second);
    // A device created after every other one is gone works on its own instance.
    buffer_round_trip(&VulkanDevice::create_headless().expect("third device"));
}

/// Records two copies into the same texels without a barrier between them, on `device`.
fn write_after_write_hazard(device: &VulkanDevice) {
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
    recorder.copy_buffer_to_texture(buffer, texture, &copy);
    recorder.copy_buffer_to_texture(buffer, texture, &copy);
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");
}

#[test]
fn validation_messages_reach_only_the_device_that_caused_them() {
    let culprit = VulkanDevice::create_headless().expect("first device");
    let bystander = VulkanDevice::create_headless().expect("second device");
    if !culprit.api_validation_enabled() {
        eprintln!("skipped: the validation layer is not installed, or validation is disabled");
        return;
    }
    let _ = culprit.take_diagnostics();
    let _ = bystander.take_diagnostics();

    // A message naming the device: VUID-VkBufferCreateInfo-size-00912 (size must not be 0).
    let info = vk::BufferCreateInfo::default()
        .size(0)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
    let raw = &culprit.context().device;
    // SAFETY: deliberately invalid usage that the validation layer intercepts and reports; the
    // buffer, if the driver creates one anyway, is destroyed immediately and never used.
    unsafe {
        if let Ok(buffer) = raw.create_buffer(&info, None) {
            raw.destroy_buffer(buffer, None);
        }
    }
    // Messages naming a command buffer or the queue: a synchronization hazard.
    let sync_validation = std::env::var(SYNC_VALIDATION_ENV_VAR).map_or(true, |v| v != "0");
    if sync_validation {
        write_after_write_hazard(&culprit);
    }

    let reported = culprit.take_diagnostics();
    assert!(
        reported
            .iter()
            .any(|d| d.message.contains("VUID-VkBufferCreateInfo-size-00912")),
        "the device that created the invalid buffer did not report it: {reported:?}"
    );
    if sync_validation {
        assert!(
            reported
                .iter()
                .any(|d| d.severity == DiagnosticSeverity::Error && d.message.contains("hazard")),
            "the device that recorded the hazard did not report it: {reported:?}"
        );
    }
    let leaked: Vec<_> = bystander
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == DiagnosticSeverity::Error)
        .collect();
    assert!(
        leaked.is_empty(),
        "another device's errors reached this device: {leaked:#?}"
    );
}
