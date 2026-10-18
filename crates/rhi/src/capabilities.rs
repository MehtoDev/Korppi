//! Device capabilities: adapter information, limits, optional features and per-format support
//! (finding RHI-09).
//!
//! Backends fill these from the graphics API at device creation, so the queries are cheap.
//! Engine code should check them instead of assuming: for example the storage-buffer offset
//! alignment, whether BC textures can be sampled, or whether a format can be rendered to with
//! 4x MSAA.

use crate::types::{SampleCount, TextureFormat, TextureUsage};
use korppi_core::bitflags;

/// The graphics API behind a device.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GraphicsApi {
    /// Vulkan 1.3.
    Vulkan,
    /// OpenGL 4.5 Core.
    OpenGl,
}

/// Kind of physical device.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// A separate GPU with its own memory.
    DiscreteGpu,
    /// A GPU sharing memory with the CPU.
    IntegratedGpu,
    /// A virtualized GPU.
    VirtualGpu,
    /// A software rasterizer (Mesa lavapipe/llvmpipe, SwiftShader).
    Cpu,
    /// Anything else, or unknown.
    Other,
}

/// Identification of the device and driver, for logs, bug reports and tools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    /// The API the device uses.
    pub api: GraphicsApi,
    /// Device name as the driver reports it.
    pub name: String,
    /// PCI vendor ID, or 0 where the API does not report it (OpenGL).
    pub vendor_id: u32,
    /// PCI device ID, or 0 where the API does not report it (OpenGL).
    pub device_id: u32,
    /// Kind of device.
    pub device_type: DeviceType,
    /// Driver name and version.
    pub driver: String,
    /// API version the device provides, as `(major, minor, patch)`.
    pub api_version: (u32, u32, u32),
}

/// Numeric limits of a device.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceLimits {
    /// Maximum width of a 1D texture.
    pub max_texture_dimension_1d: u32,
    /// Maximum width/height of a 2D texture.
    pub max_texture_dimension_2d: u32,
    /// Maximum width/height/depth of a 3D texture.
    pub max_texture_dimension_3d: u32,
    /// Maximum width/height of a cube texture face.
    pub max_texture_dimension_cube: u32,
    /// Maximum array layers of a texture.
    pub max_texture_array_layers: u32,
    /// Maximum colour attachments of a pass.
    pub max_color_attachments: u32,
    /// Maximum vertex buffer bindings.
    pub max_vertex_buffers: u32,
    /// Maximum vertex attributes.
    pub max_vertex_attributes: u32,
    /// Maximum vertex buffer stride in bytes.
    pub max_vertex_buffer_stride: u32,
    /// Push-constant bytes available through the RHI (at most `MAX_PUSH_CONSTANTS_SIZE`).
    pub max_push_constants_size: u32,
    /// Required alignment of uniform buffer binding offsets.
    pub min_uniform_buffer_offset_alignment: u64,
    /// Required alignment of storage buffer binding offsets.
    pub min_storage_buffer_offset_alignment: u64,
    /// Maximum size of a uniform buffer binding in bytes.
    pub max_uniform_buffer_binding_size: u64,
    /// Maximum size of a storage buffer binding in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum uniform buffers one shader stage can access.
    pub max_per_stage_uniform_buffers: u32,
    /// Maximum storage buffers one shader stage can access.
    pub max_per_stage_storage_buffers: u32,
    /// Maximum sampled textures one shader stage can access.
    pub max_per_stage_sampled_textures: u32,
    /// Maximum storage textures (images) one shader stage can access.
    pub max_per_stage_storage_textures: u32,
    /// Maximum sampler anisotropy (1.0 when anisotropic filtering is unsupported).
    pub max_sampler_anisotropy: f32,
    /// Maximum compute workgroup size per dimension.
    pub max_compute_workgroup_size: [u32; 3],
    /// Maximum invocations in one compute workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Maximum workgroups per dispatch dimension.
    pub max_compute_workgroup_count: [u32; 3],
    /// Maximum workgroup shared memory in bytes.
    pub max_compute_shared_memory_size: u32,
    /// Nanoseconds per timestamp tick; 0 if timestamps are unsupported.
    pub timestamp_period_ns: f32,
}

bitflags! {
    /// Optional device features.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct DeviceFeatures: u32 {
        /// Anisotropic sampler filtering.
        const SAMPLER_ANISOTROPY = 1 << 0;
        /// BC1-BC7 compressed textures can be sampled.
        const TEXTURE_COMPRESSION_BC = 1 << 1;
        /// Depth clamping instead of clipping (`RasterizerState::depth_clamp`).
        const DEPTH_CLAMP = 1 << 2;
        /// Depth-bias clamping (`RasterizerState::depth_bias_clamp`).
        const DEPTH_BIAS_CLAMP = 1 << 3;
        /// `PolygonMode::Line` and `PolygonMode::Point`.
        const POLYGON_MODE_LINE = 1 << 4;
        /// Indirect draws with a draw count above 1.
        const MULTI_DRAW_INDIRECT = 1 << 5;
        /// Indirect draws with a non-zero first instance.
        const INDIRECT_FIRST_INSTANCE = 1 << 6;
        /// GPU timestamp queries.
        const TIMESTAMP_QUERY = 1 << 7;
        /// Presentation to windows (`create_swapchain`).
        const PRESENTATION = 1 << 8;
    }
}

bitflags! {
    /// What a texture format supports on a device (optimal tiling, 2D).
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct FormatCapabilities: u32 {
        /// Can be sampled in shaders.
        const SAMPLED = 1 << 0;
        /// Can be sampled with linear filtering.
        const FILTERABLE = 1 << 1;
        /// Can be read and written as a storage image.
        const STORAGE = 1 << 2;
        /// Can be a colour attachment.
        const COLOR_ATTACHMENT = 1 << 3;
        /// Can be blended into as a colour attachment.
        const BLENDABLE = 1 << 4;
        /// Can be a depth/stencil attachment.
        const DEPTH_STENCIL_ATTACHMENT = 1 << 5;
        /// Can be a copy source.
        const COPY_SRC = 1 << 6;
        /// Can be a copy destination.
        const COPY_DST = 1 << 7;
        /// Can be created with 2 samples.
        const MULTISAMPLE_X2 = 1 << 8;
        /// Can be created with 4 samples.
        const MULTISAMPLE_X4 = 1 << 9;
        /// Can be created with 8 samples.
        const MULTISAMPLE_X8 = 1 << 10;
        /// Can be the source and destination of `blit_texture`.
        const BLIT = 1 << 11;
    }
}

impl FormatCapabilities {
    /// The capabilities a texture with `usage` and `samples` needs.
    pub fn required_for(usage: TextureUsage, samples: SampleCount) -> Self {
        let mut required = Self::empty();
        if usage.contains(TextureUsage::SAMPLED) {
            required |= Self::SAMPLED;
        }
        if usage.contains(TextureUsage::STORAGE) {
            required |= Self::STORAGE;
        }
        if usage.contains(TextureUsage::COLOR_ATTACHMENT) {
            required |= Self::COLOR_ATTACHMENT;
        }
        if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
            required |= Self::DEPTH_STENCIL_ATTACHMENT;
        }
        if usage.contains(TextureUsage::COPY_SRC) {
            required |= Self::COPY_SRC;
        }
        if usage.contains(TextureUsage::COPY_DST) {
            required |= Self::COPY_DST;
        }
        required |= match samples {
            SampleCount::Count1 => Self::empty(),
            SampleCount::Count2 => Self::MULTISAMPLE_X2,
            SampleCount::Count4 => Self::MULTISAMPLE_X4,
            SampleCount::Count8 => Self::MULTISAMPLE_X8,
        };
        required
    }

    /// Checks that `format`, with these capabilities, supports a texture with `usage` and
    /// `samples`.
    ///
    /// # Errors
    /// `RhiError::Unsupported` naming the missing capabilities.
    pub fn check(
        self,
        format: TextureFormat,
        usage: TextureUsage,
        samples: SampleCount,
    ) -> Result<(), crate::RhiError> {
        let missing = Self::required_for(usage, samples) - self;
        if missing.is_empty() {
            Ok(())
        } else {
            Err(crate::RhiError::Unsupported(format!(
                "{format:?} does not support {missing:?} on this device (usage {usage:?}, \
                 {samples:?})"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_capabilities_follow_usage_and_samples() {
        assert_eq!(
            FormatCapabilities::required_for(
                TextureUsage::SAMPLED | TextureUsage::COLOR_ATTACHMENT,
                SampleCount::Count4
            ),
            FormatCapabilities::SAMPLED
                | FormatCapabilities::COLOR_ATTACHMENT
                | FormatCapabilities::MULTISAMPLE_X4
        );
    }

    #[test]
    fn missing_capabilities_are_reported() {
        let caps = FormatCapabilities::SAMPLED | FormatCapabilities::COPY_DST;
        assert!(caps
            .check(
                TextureFormat::Bc1RgbaUnorm,
                TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                SampleCount::Count1
            )
            .is_ok());
        let error = caps
            .check(
                TextureFormat::Bc1RgbaUnorm,
                TextureUsage::COLOR_ATTACHMENT,
                SampleCount::Count1,
            )
            .unwrap_err();
        assert!(matches!(error, crate::RhiError::Unsupported(m) if m.contains("COLOR_ATTACHMENT")));
    }
}
