use crate::memory::Suballocation;
use ash::vk;
use korppi_rhi::{
    BufferHandle, BufferUsage, MemoryUsage, PipelineHandle, PipelineSignature, ResourceOracle,
    SamplerHandle, ShaderStage, TextureDimension, TextureFormat, TextureHandle, TextureInfo,
    TextureSubresourceRange, TextureUsage,
};
use std::ffi::CString;

/// Internal Vulkan buffer resource.
pub struct VulkanBuffer {
    /// Vulkan buffer handle.
    pub buffer: vk::Buffer,
    /// Memory suballocation block.
    pub suballoc: Suballocation,
    /// Total buffer size in bytes.
    pub size: u64,
    /// Buffer usage bitflags.
    pub usage: BufferUsage,
    /// CPU/GPU memory placement.
    pub memory: MemoryUsage,
    /// Makes host access to the mapped memory exclusive (`map_buffer`, `write_buffer`,
    /// `read_buffer`, `destroy_buffer`).
    pub host_access: korppi_rhi::HostAccessLock,
}

/// Who owns a texture's image and its memory.
pub enum TextureStorage {
    /// Created by the device in a memory suballocation; destroyed with the texture.
    Device(Suballocation),
    /// Image `index` of the swapchain with ID `swapchain`, which owns and destroys it (and the
    /// view).
    Swapchain {
        /// ID of the owning swapchain (`SwapchainHandle::id`).
        swapchain: u32,
        /// Index of the image in the swapchain.
        index: u32,
    },
}

/// Internal Vulkan texture and image view resource.
pub struct VulkanTexture {
    /// Vulkan image handle.
    pub image: vk::Image,
    /// Primary image view handle.
    pub view: vk::ImageView,
    /// Owner of the image and its memory.
    pub storage: TextureStorage,
    /// Pixel format.
    pub format: TextureFormat,
    /// Usage the texture was created with.
    pub usage: TextureUsage,
    /// Layout the last submitted (or internal) work left each subresource in, indexed
    /// `mip * array_layers + layer`, for internal helpers such as `read_texture` (RVK-07).
    pub layouts: Vec<vk::ImageLayout>,
    /// Dimensions in pixels.
    pub extent: vk::Extent3D,
    /// Dimension, mips, layers and samples.
    pub info: TextureInfo,
    /// Single-mip, single-layer views for attachments: `[mip][layer or slice]` (empty unless
    /// the texture is an attachment).
    pub attachment_views: Vec<Vec<vk::ImageView>>,
    /// Views of this texture, destroyed with it.
    pub views: Vec<korppi_rhi::TextureViewHandle>,
}

impl VulkanTexture {
    /// Tracked layout of subresource (`mip`, `layer`).
    pub fn layout_of(&self, mip: u32, layer: u32) -> vk::ImageLayout {
        self.layouts
            .get((mip * self.info.array_layers + layer) as usize)
            .copied()
            .unwrap_or(vk::ImageLayout::UNDEFINED)
    }

    /// Records that the subresources in `range` are now in `layout`.
    pub fn set_layout(&mut self, range: &TextureSubresourceRange, layout: vk::ImageLayout) {
        let layers = self.info.array_layers;
        for mip in range.mips(self.info.mip_levels) {
            for layer in range.layers(layers) {
                if let Some(slot) = self.layouts.get_mut((mip * layers + layer) as usize) {
                    *slot = layout;
                }
            }
        }
    }

    /// The attachment view of (`mip`, `layer`), if the texture is an attachment and has it.
    pub fn attachment_view(&self, mip: u32, layer: u32) -> Option<vk::ImageView> {
        self.attachment_views
            .get(mip as usize)?
            .get(layer as usize)
            .copied()
    }
}

/// A view of a range of a texture.
pub struct VulkanTextureView {
    /// The image view.
    pub view: vk::ImageView,
    /// The viewed texture.
    pub texture: TextureHandle,
    /// The viewed texture's description.
    pub info: TextureInfo,
    /// Dimension the view sees the texture as.
    pub dimension: TextureDimension,
    /// The view's description.
    pub desc: korppi_rhi::TextureViewDescriptor,
}

/// Internal Vulkan sampler resource.
pub struct VulkanSampler {
    /// Vulkan sampler handle.
    pub sampler: vk::Sampler,
}

/// Internal Vulkan shader module resource.
pub struct VulkanShader {
    /// Vulkan shader module handle.
    pub module: vk::ShaderModule,
    /// Shader stage.
    pub stage: ShaderStage,
    /// Entry point function name.
    pub entry_point: CString,
}

/// Internal Vulkan pipeline resource.
pub struct VulkanPipeline {
    /// Vulkan pipeline handle.
    pub pipeline: vk::Pipeline,
    /// Pipeline layout handle.
    pub layout: vk::PipelineLayout,
    /// Pipeline bind point (Graphics or Compute).
    pub bind_point: vk::PipelineBindPoint,
    /// Descriptor sets in `layout`.
    pub set_count: u32,
    /// Summary used by command validation.
    pub signature: PipelineSignature,
    /// Push-constant ranges of `layout`; `vkCmdPushConstants` stage flags are derived from them.
    pub push_constant_ranges: std::sync::Arc<[vk::PushConstantRange]>,
    /// State set dynamically when a graphics pipeline is bound (RVK-16).
    pub(crate) dynamic: Option<crate::pipeline::GraphicsDynamicState>,
    /// Key of the shared `VkPipeline`.
    pub(crate) key: crate::pipeline::PipelineKey,
    /// Key of the shared pipeline layout.
    pub(crate) layout_key: crate::pipeline::LayoutKey,
}

/// Thread-safe registry mapping generational IDs to native Vulkan resources, of one device:
/// its maps find no handle of another device (RHI-15).
pub struct VulkanResourceRegistry {
    /// The device whose resources these are; every handle the registry hands out carries it.
    pub device: korppi_rhi::DeviceId,
    /// Active buffers.
    pub buffers: std::sync::Mutex<korppi_rhi::SlotMap<VulkanBuffer>>,
    /// Active textures.
    pub textures: std::sync::Mutex<korppi_rhi::SlotMap<VulkanTexture>>,
    /// Active texture views.
    pub texture_views: std::sync::Mutex<korppi_rhi::SlotMap<VulkanTextureView>>,
    /// Active samplers.
    pub samplers: std::sync::Mutex<korppi_rhi::SlotMap<VulkanSampler>>,
    /// Active shader modules.
    pub shaders: std::sync::Mutex<korppi_rhi::SlotMap<VulkanShader>>,
    /// Active graphics/compute pipelines.
    pub pipelines: std::sync::Mutex<korppi_rhi::SlotMap<VulkanPipeline>>,
    /// Active bind group layouts.
    pub bind_group_layouts:
        std::sync::Mutex<korppi_rhi::SlotMap<crate::binding::VulkanBindGroupLayout>>,
    /// Active bind groups.
    pub bind_groups: std::sync::Mutex<korppi_rhi::SlotMap<crate::binding::VulkanBindGroup>>,
    /// Active query sets.
    pub query_sets: std::sync::Mutex<korppi_rhi::SlotMap<crate::query::VulkanQuerySet>>,
    /// The device's optional features, for recording validation; set at device creation.
    pub features: std::sync::OnceLock<korppi_rhi::DeviceFeatures>,
    /// Per-format capabilities of the device (indexed by `TextureFormat::index`), for recording
    /// validation; set once at device creation.
    pub format_capabilities: std::sync::OnceLock<Vec<korppi_rhi::FormatCapabilities>>,
}

impl VulkanResourceRegistry {
    /// An empty registry of `device`'s resources.
    pub fn for_device(device: korppi_rhi::DeviceId) -> Self {
        fn map<T>(device: korppi_rhi::DeviceId) -> std::sync::Mutex<korppi_rhi::SlotMap<T>> {
            std::sync::Mutex::new(korppi_rhi::SlotMap::for_device(device))
        }
        Self {
            device,
            buffers: map(device),
            textures: map(device),
            texture_views: map(device),
            samplers: map(device),
            shaders: map(device),
            pipelines: map(device),
            bind_group_layouts: map(device),
            bind_groups: map(device),
            query_sets: map(device),
            features: std::sync::OnceLock::new(),
            format_capabilities: std::sync::OnceLock::new(),
        }
    }
}

/// Locks a registry map (recovering from poisoning).
fn read<T>(
    map: &std::sync::Mutex<korppi_rhi::SlotMap<T>>,
) -> std::sync::MutexGuard<'_, korppi_rhi::SlotMap<T>> {
    map.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl ResourceOracle for VulkanResourceRegistry {
    fn buffer_exists(&self, handle: BufferHandle) -> bool {
        read(&self.buffers).contains(handle)
    }

    fn buffer_summary(&self, handle: BufferHandle) -> Option<korppi_rhi::BufferSummary> {
        read(&self.buffers)
            .get(handle)
            .map(|(b, _)| korppi_rhi::BufferSummary {
                size: Some(b.size),
                usage: Some(b.usage),
            })
    }

    fn texture_info(&self, handle: TextureHandle) -> Option<TextureInfo> {
        read(&self.textures).get(handle).map(|(t, _)| t.info)
    }

    fn sampler_exists(&self, handle: SamplerHandle) -> bool {
        read(&self.samplers).contains(handle)
    }

    fn pipeline_signature(&self, handle: PipelineHandle) -> Option<PipelineSignature> {
        read(&self.pipelines).get(handle).map(|(p, _)| p.signature)
    }

    fn bind_group_info(
        &self,
        handle: korppi_rhi::BindGroupHandle,
    ) -> Option<korppi_rhi::BindGroupInfo> {
        read(&self.bind_groups).get(handle).map(|(g, _)| g.info)
    }

    fn query_set_count(&self, handle: korppi_rhi::QuerySetHandle) -> Option<u32> {
        read(&self.query_sets).get(handle).map(|(q, _)| q.count)
    }

    fn buffer_usage(&self, handle: BufferHandle) -> Option<BufferUsage> {
        read(&self.buffers).get(handle).map(|(b, _)| b.usage)
    }

    fn features(&self) -> korppi_rhi::DeviceFeatures {
        self.features
            .get()
            .copied()
            .unwrap_or(korppi_rhi::DeviceFeatures::all())
    }

    fn buffer_size(&self, handle: BufferHandle) -> Option<u64> {
        read(&self.buffers).get(handle).map(|(b, _)| b.size)
    }

    fn format_capabilities(
        &self,
        format: korppi_rhi::TextureFormat,
    ) -> korppi_rhi::FormatCapabilities {
        self.format_capabilities
            .get()
            .and_then(|caps| caps.get(format.index()).copied())
            .unwrap_or(korppi_rhi::FormatCapabilities::all())
    }
}
