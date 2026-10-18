//! Device capabilities reported through the RHI (finding RHI-09): adapter information, limits,
//! enabled optional features and per-format support, computed once at device creation.

use crate::types::to_vk_format;
use ash::vk;
use ash::Instance;
use korppi_rhi::{
    AdapterInfo, DeviceFeatures, DeviceLimits, DeviceType, FormatCapabilities, GraphicsApi,
    TextureFormat, MAX_PUSH_CONSTANTS_SIZE,
};

/// Everything the device reports through `GraphicsDevice::{adapter_info, limits, features,
/// format_capabilities}`, plus the format the backend uses for `Depth24PlusStencil8`.
#[derive(Clone, Debug)]
pub struct VulkanCapabilities {
    /// Device and driver identification.
    pub adapter: AdapterInfo,
    /// Numeric limits.
    pub limits: DeviceLimits,
    /// Enabled optional features.
    pub features: DeviceFeatures,
    /// Whether the `imageCubeArray` feature is enabled (cube-array views).
    pub image_cube_array: bool,
    /// `D24_UNORM_S8_UINT` where it is a depth/stencil attachment format, else
    /// `D32_SFLOAT_S8_UINT` (Vulkan guarantees one of them; RVK-13).
    pub depth24_stencil8: vk::Format,
    formats: Vec<FormatCapabilities>,
}

impl VulkanCapabilities {
    /// Capabilities of `format`.
    pub fn format(&self, format: TextureFormat) -> FormatCapabilities {
        self.formats[format.index()]
    }

    /// The Vulkan format used for `format` on this device.
    pub fn vk_format(&self, format: TextureFormat) -> vk::Format {
        match format {
            TextureFormat::Depth24PlusStencil8 => self.depth24_stencil8,
            other => to_vk_format(other),
        }
    }
}

/// Optional core features the device enabled, as RHI feature bits.
pub(crate) fn device_features(
    enabled: &vk::PhysicalDeviceFeatures,
    timestamps: bool,
    presentation: bool,
) -> DeviceFeatures {
    let mut features = DeviceFeatures::empty();
    let pairs = [
        (
            enabled.sampler_anisotropy,
            DeviceFeatures::SAMPLER_ANISOTROPY,
        ),
        (
            enabled.texture_compression_bc,
            DeviceFeatures::TEXTURE_COMPRESSION_BC,
        ),
        (enabled.depth_clamp, DeviceFeatures::DEPTH_CLAMP),
        (enabled.depth_bias_clamp, DeviceFeatures::DEPTH_BIAS_CLAMP),
        (
            enabled.fill_mode_non_solid,
            DeviceFeatures::POLYGON_MODE_LINE,
        ),
        (
            enabled.multi_draw_indirect,
            DeviceFeatures::MULTI_DRAW_INDIRECT,
        ),
        (
            enabled.draw_indirect_first_instance,
            DeviceFeatures::INDIRECT_FIRST_INSTANCE,
        ),
    ];
    for (flag, feature) in pairs {
        if flag == vk::TRUE {
            features |= feature;
        }
    }
    if timestamps {
        features |= DeviceFeatures::TIMESTAMP_QUERY;
    }
    if presentation {
        features |= DeviceFeatures::PRESENTATION;
    }
    features
}

/// The subset of supported optional features the backend enables (all of those it uses).
pub(crate) fn features_to_enable(
    supported: &vk::PhysicalDeviceFeatures,
) -> vk::PhysicalDeviceFeatures {
    vk::PhysicalDeviceFeatures {
        sampler_anisotropy: supported.sampler_anisotropy,
        texture_compression_bc: supported.texture_compression_bc,
        depth_clamp: supported.depth_clamp,
        depth_bias_clamp: supported.depth_bias_clamp,
        fill_mode_non_solid: supported.fill_mode_non_solid,
        multi_draw_indirect: supported.multi_draw_indirect,
        draw_indirect_first_instance: supported.draw_indirect_first_instance,
        independent_blend: supported.independent_blend,
        image_cube_array: supported.image_cube_array,
        ..Default::default()
    }
}

fn device_type(kind: vk::PhysicalDeviceType) -> DeviceType {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::DiscreteGpu,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::IntegratedGpu,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::VirtualGpu,
        vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
        _ => DeviceType::Other,
    }
}

/// Builds the capabilities of `physical_device` with `enabled` features.
///
/// # Safety
/// `physical_device` must belong to `instance`.
pub(crate) unsafe fn query(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    enabled: &vk::PhysicalDeviceFeatures,
    timestamp_valid_bits: u32,
    presentation: bool,
) -> VulkanCapabilities {
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    // SAFETY: forwarded from the caller; the `pNext` chain outlives the call (Vulkan 1.2 core).
    unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
    let properties = properties2.properties;
    let limits = properties.limits;
    let driver_name = driver
        .driver_name_as_c_str()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let driver_info = driver
        .driver_info_as_c_str()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let adapter = AdapterInfo {
        api: GraphicsApi::Vulkan,
        name: properties
            .device_name_as_c_str()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        device_type: device_type(properties.device_type),
        driver: format!("{driver_name} {driver_info}").trim().to_owned(),
        api_version: (
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version),
            vk::api_version_patch(properties.api_version),
        ),
    };

    let timestamps = limits.timestamp_compute_and_graphics == vk::TRUE && timestamp_valid_bits > 0;
    let features = device_features(enabled, timestamps, presentation);
    let rhi_limits = DeviceLimits {
        max_texture_dimension_1d: limits.max_image_dimension1_d,
        max_texture_dimension_2d: limits.max_image_dimension2_d,
        max_texture_dimension_3d: limits.max_image_dimension3_d,
        max_texture_dimension_cube: limits.max_image_dimension_cube,
        max_texture_array_layers: limits.max_image_array_layers,
        max_color_attachments: limits.max_color_attachments,
        max_vertex_buffers: limits.max_vertex_input_bindings,
        max_vertex_attributes: limits.max_vertex_input_attributes,
        max_vertex_buffer_stride: limits.max_vertex_input_binding_stride,
        max_push_constants_size: limits
            .max_push_constants_size
            .min(MAX_PUSH_CONSTANTS_SIZE as u32),
        min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
        max_uniform_buffer_binding_size: u64::from(limits.max_uniform_buffer_range),
        max_storage_buffer_binding_size: u64::from(limits.max_storage_buffer_range),
        max_per_stage_uniform_buffers: limits.max_per_stage_descriptor_uniform_buffers,
        max_per_stage_storage_buffers: limits.max_per_stage_descriptor_storage_buffers,
        max_per_stage_sampled_textures: limits.max_per_stage_descriptor_sampled_images,
        max_per_stage_storage_textures: limits.max_per_stage_descriptor_storage_images,
        max_sampler_anisotropy: if features.contains(DeviceFeatures::SAMPLER_ANISOTROPY) {
            limits.max_sampler_anisotropy
        } else {
            1.0
        },
        max_compute_workgroup_size: limits.max_compute_work_group_size,
        max_compute_invocations_per_workgroup: limits.max_compute_work_group_invocations,
        max_compute_workgroup_count: limits.max_compute_work_group_count,
        max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
        timestamp_period_ns: if timestamps {
            limits.timestamp_period
        } else {
            0.0
        },
    };

    // SAFETY: forwarded from the caller.
    let format_props = |format: vk::Format| unsafe {
        instance.get_physical_device_format_properties(physical_device, format)
    };
    let d24s8_renderable = format_props(vk::Format::D24_UNORM_S8_UINT)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
    let depth24_stencil8 = if d24s8_renderable {
        vk::Format::D24_UNORM_S8_UINT
    } else {
        vk::Format::D32_SFLOAT_S8_UINT
    };

    let formats = TextureFormat::ALL
        .iter()
        .map(|&format| {
            let vk_format = match format {
                TextureFormat::Depth24PlusStencil8 => depth24_stencil8,
                other => to_vk_format(other),
            };
            let optimal = format_props(vk_format).optimal_tiling_features;
            let mut caps = FormatCapabilities::empty();
            let mapping = [
                (
                    vk::FormatFeatureFlags::SAMPLED_IMAGE,
                    FormatCapabilities::SAMPLED,
                ),
                (
                    vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
                    FormatCapabilities::FILTERABLE,
                ),
                (
                    vk::FormatFeatureFlags::STORAGE_IMAGE,
                    FormatCapabilities::STORAGE,
                ),
                (
                    vk::FormatFeatureFlags::COLOR_ATTACHMENT,
                    FormatCapabilities::COLOR_ATTACHMENT,
                ),
                (
                    vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
                    FormatCapabilities::BLENDABLE,
                ),
                (
                    vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
                    FormatCapabilities::DEPTH_STENCIL_ATTACHMENT,
                ),
                (
                    vk::FormatFeatureFlags::TRANSFER_SRC,
                    FormatCapabilities::COPY_SRC,
                ),
                (
                    vk::FormatFeatureFlags::TRANSFER_DST,
                    FormatCapabilities::COPY_DST,
                ),
            ];
            for (flag, cap) in mapping {
                if optimal.contains(flag) {
                    caps |= cap;
                }
            }
            if optimal.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST)
                && !format.is_compressed()
            {
                caps |= FormatCapabilities::BLIT;
            }
            if !caps.contains(FormatCapabilities::SAMPLED) {
                caps -= FormatCapabilities::FILTERABLE;
            }
            // Some drivers (NVIDIA) allow storage images of depth formats, but SPIR-V has no
            // storage format for them and OpenGL cannot bind them to image units, so the RHI
            // contract excludes them, as it does block-compressed formats.
            if format.is_depth() || format.is_compressed() {
                caps -= FormatCapabilities::STORAGE;
            }
            if format.is_compressed() && !features.contains(DeviceFeatures::TEXTURE_COMPRESSION_BC)
            {
                caps = FormatCapabilities::empty();
            }
            // Its depth layout differs between the two backing formats, so copies are not
            // portable.
            if format == TextureFormat::Depth24PlusStencil8 {
                caps -= FormatCapabilities::COPY_SRC | FormatCapabilities::COPY_DST;
            }
            let attachment_usage = if caps.contains(FormatCapabilities::COLOR_ATTACHMENT) {
                Some(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            } else if caps.contains(FormatCapabilities::DEPTH_STENCIL_ATTACHMENT) {
                Some(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            } else {
                None
            };
            if let Some(usage) = attachment_usage {
                // SAFETY: forwarded from the caller.
                let image_props = unsafe {
                    instance.get_physical_device_image_format_properties(
                        physical_device,
                        vk_format,
                        vk::ImageType::TYPE_2D,
                        vk::ImageTiling::OPTIMAL,
                        usage,
                        vk::ImageCreateFlags::empty(),
                    )
                };
                if let Ok(image_props) = image_props {
                    for (count, cap) in [
                        (
                            vk::SampleCountFlags::TYPE_2,
                            FormatCapabilities::MULTISAMPLE_X2,
                        ),
                        (
                            vk::SampleCountFlags::TYPE_4,
                            FormatCapabilities::MULTISAMPLE_X4,
                        ),
                        (
                            vk::SampleCountFlags::TYPE_8,
                            FormatCapabilities::MULTISAMPLE_X8,
                        ),
                    ] {
                        if image_props.sample_counts.contains(count) {
                            caps |= cap;
                        }
                    }
                }
            }
            caps
        })
        .collect();

    VulkanCapabilities {
        image_cube_array: enabled.image_cube_array == vk::TRUE,
        adapter,
        limits: rhi_limits,
        features,
        depth24_stencil8,
        formats,
    }
}
