use crate::handle::{BufferHandle, TextureHandle};
use crate::types::TextureAspect;
use korppi_core::{bitflags, SmallVec};

bitflags! {
    /// Pipeline execution stages for synchronization barriers (modeled after Vulkan 1.3 `VkPipelineStageFlags2`).
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct PipelineStageFlags: u64 {
        /// Top of pipeline (before any stage begins).
        const TOP_OF_PIPE = 1 << 0;
        /// Indirect draw parameter consumption stage.
        const DRAW_INDIRECT = 1 << 1;
        /// Vertex and index buffer consumption stage.
        const VERTEX_INPUT = 1 << 2;
        /// Vertex shader execution stage.
        const VERTEX_SHADER = 1 << 3;
        /// Fragment / pixel shader execution stage.
        const FRAGMENT_SHADER = 1 << 4;
        /// Early fragment tests (early depth/stencil before fragment shader).
        const EARLY_FRAGMENT_TESTS = 1 << 5;
        /// Late fragment tests (late depth/stencil after fragment shader).
        const LATE_FRAGMENT_TESTS = 1 << 6;
        /// Color attachment output stage (writing to render targets).
        const COLOR_ATTACHMENT_OUTPUT = 1 << 7;
        /// Compute shader execution stage.
        const COMPUTE_SHADER = 1 << 8;
        /// Transfer / copy engine stage (buffer/texture copies).
        const TRANSFER = 1 << 9;
        /// Bottom of pipeline (after all stages complete).
        const BOTTOM_OF_PIPE = 1 << 10;
        /// All graphics pipeline stages combined.
        const ALL_GRAPHICS = 1 << 11;
        /// All commands and stages combined.
        const ALL_COMMANDS = 1 << 12;
        /// Host (CPU) reads and writes of mapped memory; pair with `AccessFlags::HOST_READ` /
        /// `HOST_WRITE` for device-to-host readback dependencies.
        const HOST = 1 << 13;
    }
}

bitflags! {
    /// Memory access flags for cache flushes and invalidations (modeled after Vulkan 1.3 `VkAccessFlags2`).
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct AccessFlags: u64 {
        /// Read access by indirect draw commands.
        const INDIRECT_COMMAND_READ = 1 << 0;
        /// Read access to index buffers.
        const INDEX_READ = 1 << 1;
        /// Read access to vertex attributes.
        const VERTEX_ATTRIBUTE_READ = 1 << 2;
        /// Read access to uniform buffers.
        const UNIFORM_READ = 1 << 3;
        /// Read access in shader (texture sampling, SSBO read).
        const SHADER_READ = 1 << 4;
        /// Write access in shader (storage image write, SSBO write).
        const SHADER_WRITE = 1 << 5;
        /// Read access to color attachments (blending).
        const COLOR_ATTACHMENT_READ = 1 << 6;
        /// Write access to color attachments.
        const COLOR_ATTACHMENT_WRITE = 1 << 7;
        /// Read access to depth/stencil attachments.
        const DEPTH_STENCIL_ATTACHMENT_READ = 1 << 8;
        /// Write access to depth/stencil attachments.
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 1 << 9;
        /// Read access in transfer operations.
        const TRANSFER_READ = 1 << 10;
        /// Write access in transfer operations.
        const TRANSFER_WRITE = 1 << 11;
        /// Read access by the host CPU.
        const HOST_READ = 1 << 12;
        /// Write access by the host CPU.
        const HOST_WRITE = 1 << 13;
        /// Read access across all engine memory types.
        const MEMORY_READ = 1 << 14;
        /// Write access across all engine memory types.
        const MEMORY_WRITE = 1 << 15;
    }
}

/// Image layout states for hardware textures.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageLayout {
    /// Undefined layout (initial layout; memory contents can be discarded on transition).
    Undefined,
    /// General layout (supports concurrent read/write in compute).
    General,
    /// Color attachment layout (optimal for rendering output).
    ColorAttachment,
    /// Depth/stencil attachment layout (optimal for depth testing and writing).
    DepthStencilAttachment,
    /// Read-only depth/stencil: depth testing without writes while the texture is also
    /// sampled (shadow maps, soft particles).
    DepthStencilReadOnly,
    /// Shader read-only layout (optimal for sampling in fragment/compute shaders).
    ShaderReadOnly,
    /// Copy source layout (optimal for copying from texture to buffer/texture).
    TransferSrc,
    /// Copy destination layout (optimal for copying into texture).
    TransferDst,
    /// Swapchain presentation layout (handed over to display surface).
    Present,
}

/// Global memory execution barrier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryBarrier {
    /// Source pipeline stages.
    pub src_stages: PipelineStageFlags,
    /// Source access flags.
    pub src_access: AccessFlags,
    /// Destination pipeline stages.
    pub dst_stages: PipelineStageFlags,
    /// Destination access flags.
    pub dst_access: AccessFlags,
}

/// Buffer memory barrier for synchronizing buffer ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferMemoryBarrier {
    /// Target buffer handle.
    pub buffer: BufferHandle,
    /// Source pipeline stages.
    pub src_stages: PipelineStageFlags,
    /// Source access flags.
    pub src_access: AccessFlags,
    /// Destination pipeline stages.
    pub dst_stages: PipelineStageFlags,
    /// Destination access flags.
    pub dst_access: AccessFlags,
    /// Byte offset within the buffer.
    pub offset: u64,
    /// Size of the synchronized byte range.
    pub size: u64,
}

/// A range of a texture's mips, layers and aspects (finding RHI-06).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TextureSubresourceRange {
    /// Aspects covered (`All` for colour formats and for both depth and stencil).
    pub aspect: TextureAspect,
    /// First mip level.
    pub base_mip_level: u32,
    /// Number of mip levels, or [`TextureSubresourceRange::REMAINING`].
    pub mip_level_count: u32,
    /// First array layer.
    pub base_array_layer: u32,
    /// Number of array layers, or [`TextureSubresourceRange::REMAINING`].
    pub array_layer_count: u32,
}

impl TextureSubresourceRange {
    /// Every level or layer from the base to the end of the texture.
    pub const REMAINING: u32 = u32::MAX;

    /// The whole texture.
    pub const ALL: Self = Self {
        aspect: TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Self::REMAINING,
        base_array_layer: 0,
        array_layer_count: Self::REMAINING,
    };

    /// One mip level of one layer.
    pub const fn single(mip_level: u32, array_layer: u32) -> Self {
        Self {
            aspect: TextureAspect::All,
            base_mip_level: mip_level,
            mip_level_count: 1,
            base_array_layer: array_layer,
            array_layer_count: 1,
        }
    }

    /// The covered mip levels of a texture with `mip_levels` levels, clamped to it.
    pub fn mips(&self, mip_levels: u32) -> std::ops::Range<u32> {
        let end = if self.mip_level_count == Self::REMAINING {
            mip_levels
        } else {
            self.base_mip_level.saturating_add(self.mip_level_count)
        };
        self.base_mip_level.min(mip_levels)..end.min(mip_levels)
    }

    /// The covered layers of a texture with `array_layers` layers, clamped to it.
    pub fn layers(&self, array_layers: u32) -> std::ops::Range<u32> {
        let end = if self.array_layer_count == Self::REMAINING {
            array_layers
        } else {
            self.base_array_layer.saturating_add(self.array_layer_count)
        };
        self.base_array_layer.min(array_layers)..end.min(array_layers)
    }

    /// True if the range lies within a texture with `mip_levels` levels and `array_layers`
    /// layers (and covers at least one subresource).
    pub fn fits(&self, mip_levels: u32, array_layers: u32) -> bool {
        let mips_ok = self.mip_level_count == Self::REMAINING
            || self
                .base_mip_level
                .checked_add(self.mip_level_count)
                .is_some_and(|end| end <= mip_levels && self.mip_level_count > 0);
        let layers_ok = self.array_layer_count == Self::REMAINING
            || self
                .base_array_layer
                .checked_add(self.array_layer_count)
                .is_some_and(|end| end <= array_layers && self.array_layer_count > 0);
        mips_ok
            && layers_ok
            && self.base_mip_level < mip_levels
            && self.base_array_layer < array_layers
    }
}

impl Default for TextureSubresourceRange {
    fn default() -> Self {
        Self::ALL
    }
}

/// Image memory barrier for synchronizing textures and executing layout transitions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryBarrier {
    /// Target texture handle.
    pub texture: TextureHandle,
    /// Mips, layers and aspects the barrier covers.
    pub range: TextureSubresourceRange,
    /// Source pipeline stages.
    pub src_stages: PipelineStageFlags,
    /// Source access flags.
    pub src_access: AccessFlags,
    /// Destination pipeline stages.
    pub dst_stages: PipelineStageFlags,
    /// Destination access flags.
    pub dst_access: AccessFlags,
    /// Current image layout prior to barrier.
    pub old_layout: ImageLayout,
    /// New image layout following barrier.
    pub new_layout: ImageLayout,
}

/// Comprehensive pipeline barrier containing memory, buffer, and image transitions.
///
/// The lists keep a few barriers inline (2 memory, 4 buffer, 4 image), so building a barrier
/// per frame does not allocate unless it holds more (RHI-11).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipelineBarrier {
    /// Global memory barriers.
    pub memory_barriers: SmallVec<[MemoryBarrier; 2]>,
    /// Buffer-specific memory barriers.
    pub buffer_barriers: SmallVec<[BufferMemoryBarrier; 4]>,
    /// Image memory barriers and layout transitions.
    pub image_barriers: SmallVec<[ImageMemoryBarrier; 4]>,
}

impl PipelineBarrier {
    /// Creates a new empty pipeline barrier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a global memory barrier.
    pub fn with_memory_barrier(mut self, barrier: MemoryBarrier) -> Self {
        self.memory_barriers.push(barrier);
        self
    }

    /// Adds a buffer memory barrier.
    pub fn with_buffer_barrier(mut self, barrier: BufferMemoryBarrier) -> Self {
        self.buffer_barriers.push(barrier);
        self
    }

    /// Adds an image memory barrier.
    pub fn with_image_barrier(mut self, barrier: ImageMemoryBarrier) -> Self {
        self.image_barriers.push(barrier);
        self
    }
}
