use korppi_core::bitflags;

/// Guaranteed minimum push constant capacity in bytes across all compliant hardware (Vulkan 1.3 spec Chapter 46).
pub const MAX_PUSH_CONSTANTS_SIZE: usize = 128;

/// Standard alignment for dynamic uniform buffer offsets (256 bytes per Vulkan & OpenGL specs).
pub const STANDARD_UBO_ALIGNMENT: u64 = 256;

/// Storage buffer offset alignment that every device accepts: `minStorageBufferOffsetAlignment`
/// is at most 256 in Vulkan, and OpenGL's `GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT` likewise
/// (RHI-09). Query `DeviceLimits::min_storage_buffer_offset_alignment` for the device's value.
pub const STANDARD_SSBO_ALIGNMENT: u64 = 256;

/// Calculates the next aligned offset for a uniform buffer sub-slice.
#[inline]
pub const fn align_to_ubo_offset(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    (offset + alignment - 1) & !(alignment - 1)
}

/// GPU texture formats (finding RHI-13 extended the set).
///
/// Not every format supports every use on every device: query
/// [`GraphicsDevice::format_capabilities`](crate::GraphicsDevice::format_capabilities).
/// `create_texture` rejects a usage the format does not support on the device.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureFormat {
    /// 8-bit unsigned normalized red (1 byte): masks, glyph atlases.
    R8Unorm,
    /// 8-bit unsigned normalized red/green (2 bytes).
    Rg8Unorm,
    /// 8-bit unsigned normalized RGBA (4 bytes).
    Rgba8Unorm,
    /// 8-bit unsigned normalized sRGB RGBA (4 bytes).
    Rgba8Srgb,
    /// 8-bit unsigned normalized BGRA (4 bytes, common for desktop swapchains).
    Bgra8Unorm,
    /// 8-bit unsigned normalized sRGB BGRA (4 bytes).
    Bgra8Srgb,
    /// 8-bit signed normalized RGBA (4 bytes): packed normals.
    Rgba8Snorm,
    /// 10-bit RGB + 2-bit alpha unsigned normalized (4 bytes): high-precision LDR targets.
    Rgb10A2Unorm,
    /// Packed 11/11/10-bit unsigned float RGB (4 bytes): compact HDR targets.
    Rg11B10Float,

    /// 16-bit floating-point red channel (2 bytes).
    R16Float,
    /// 16-bit floating-point RG channels (4 bytes).
    Rg16Float,
    /// 16-bit floating-point RGBA channels (8 bytes, standard HDR render target).
    Rgba16Float,
    /// 16-bit unsigned normalized RGBA (8 bytes).
    Rgba16Unorm,

    /// 32-bit floating-point red channel (4 bytes).
    R32Float,
    /// 32-bit floating-point RG channels (8 bytes).
    Rg32Float,
    /// 32-bit floating-point RGB channels (12 bytes). Optional for sampling and rendering on
    /// Vulkan: check `format_capabilities`.
    Rgb32Float,
    /// 32-bit floating-point RGBA channels (16 bytes).
    Rgba32Float,
    /// 32-bit unsigned integer red (4 bytes): entity picking, visibility buffers.
    R32Uint,
    /// 32-bit unsigned integer RG (8 bytes).
    Rg32Uint,
    /// 32-bit unsigned integer RGBA (16 bytes).
    Rgba32Uint,

    /// 16-bit unsigned normalized depth (2 bytes).
    Depth16Unorm,
    /// 32-bit floating-point depth buffer (4 bytes). Ideal for Reverse-Z.
    Depth32Float,
    /// Depth with at least 24 bits plus 8-bit stencil: `D24_UNORM_S8_UINT` where supported,
    /// otherwise `D32_SFLOAT_S8_UINT` (Vulkan guarantees one of them). The depth layout is
    /// device-specific, so it cannot be read back or copied to buffers.
    Depth24PlusStencil8,
    /// 32-bit floating-point depth plus 8-bit stencil.
    Depth32FloatStencil8,

    /// BC1 (DXT1) RGBA, 8 bytes per 4x4 block.
    Bc1RgbaUnorm,
    /// BC1 (DXT1) RGBA, sRGB, 8 bytes per 4x4 block.
    Bc1RgbaSrgb,
    /// BC2 (DXT3) RGBA, 16 bytes per 4x4 block.
    Bc2RgbaUnorm,
    /// BC2 (DXT3) RGBA, sRGB, 16 bytes per 4x4 block.
    Bc2RgbaSrgb,
    /// BC3 (DXT5) RGBA, 16 bytes per 4x4 block.
    Bc3RgbaUnorm,
    /// BC3 (DXT5) RGBA, sRGB, 16 bytes per 4x4 block.
    Bc3RgbaSrgb,
    /// BC4 red, 8 bytes per 4x4 block.
    Bc4RUnorm,
    /// BC4 red, signed, 8 bytes per 4x4 block.
    Bc4RSnorm,
    /// BC5 red/green (normal maps), 16 bytes per 4x4 block.
    Bc5RgUnorm,
    /// BC5 red/green, signed, 16 bytes per 4x4 block.
    Bc5RgSnorm,
    /// BC6H unsigned float HDR RGB, 16 bytes per 4x4 block.
    Bc6hRgbUfloat,
    /// BC6H signed float HDR RGB, 16 bytes per 4x4 block.
    Bc6hRgbSfloat,
    /// BC7 RGBA, 16 bytes per 4x4 block.
    Bc7RgbaUnorm,
    /// BC7 RGBA, sRGB, 16 bytes per 4x4 block.
    Bc7RgbaSrgb,
}

impl TextureFormat {
    /// Every format, in declaration order.
    pub const ALL: [TextureFormat; 38] = [
        Self::R8Unorm,
        Self::Rg8Unorm,
        Self::Rgba8Unorm,
        Self::Rgba8Srgb,
        Self::Bgra8Unorm,
        Self::Bgra8Srgb,
        Self::Rgba8Snorm,
        Self::Rgb10A2Unorm,
        Self::Rg11B10Float,
        Self::R16Float,
        Self::Rg16Float,
        Self::Rgba16Float,
        Self::Rgba16Unorm,
        Self::R32Float,
        Self::Rg32Float,
        Self::Rgb32Float,
        Self::Rgba32Float,
        Self::R32Uint,
        Self::Rg32Uint,
        Self::Rgba32Uint,
        Self::Depth16Unorm,
        Self::Depth32Float,
        Self::Depth24PlusStencil8,
        Self::Depth32FloatStencil8,
        Self::Bc1RgbaUnorm,
        Self::Bc1RgbaSrgb,
        Self::Bc2RgbaUnorm,
        Self::Bc2RgbaSrgb,
        Self::Bc3RgbaUnorm,
        Self::Bc3RgbaSrgb,
        Self::Bc4RUnorm,
        Self::Bc4RSnorm,
        Self::Bc5RgUnorm,
        Self::Bc5RgSnorm,
        Self::Bc6hRgbUfloat,
        Self::Bc6hRgbSfloat,
        Self::Bc7RgbaUnorm,
        Self::Bc7RgbaSrgb,
    ];

    /// Position of the format in [`TextureFormat::ALL`] (for per-format tables).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Bytes of one texel block: one texel for uncompressed formats, one 4x4 block for
    /// block-compressed formats. For the depth/stencil formats, the depth aspect as copied to
    /// buffers (`Depth24PlusStencil8` cannot be copied; 4 is nominal).
    #[inline]
    pub const fn block_bytes(&self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm | Self::R16Float | Self::Depth16Unorm => 2,
            Self::Rgba8Unorm
            | Self::Rgba8Srgb
            | Self::Bgra8Unorm
            | Self::Bgra8Srgb
            | Self::Rgba8Snorm
            | Self::Rgb10A2Unorm
            | Self::Rg11B10Float
            | Self::Rg16Float
            | Self::R32Float
            | Self::R32Uint
            | Self::Depth32Float
            | Self::Depth24PlusStencil8
            | Self::Depth32FloatStencil8 => 4,
            Self::Rgba16Float | Self::Rgba16Unorm | Self::Rg32Float | Self::Rg32Uint => 8,
            Self::Rgb32Float => 12,
            Self::Rgba32Float | Self::Rgba32Uint => 16,
            Self::Bc1RgbaUnorm | Self::Bc1RgbaSrgb | Self::Bc4RUnorm | Self::Bc4RSnorm => 8,
            Self::Bc2RgbaUnorm
            | Self::Bc2RgbaSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaSrgb
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbSfloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaSrgb => 16,
        }
    }

    /// Width and height in texels of one block: 4 x 4 for block-compressed formats, 1 x 1
    /// otherwise.
    #[inline]
    pub const fn block_dimensions(&self) -> (u32, u32) {
        if self.is_compressed() {
            (4, 4)
        } else {
            (1, 1)
        }
    }

    /// Returns the byte size of a single texel (uncompressed formats) or block (compressed
    /// formats); see [`TextureFormat::block_bytes`].
    #[inline]
    pub const fn bytes_per_pixel(&self) -> u32 {
        self.block_bytes()
    }

    /// Bytes of one tightly packed row of blocks spanning `width` texels.
    #[inline]
    pub const fn row_bytes(&self, width: u32) -> u64 {
        let (block_width, _) = self.block_dimensions();
        width.div_ceil(block_width) as u64 * self.block_bytes() as u64
    }

    /// Bytes of one tightly packed `width` x `height` image (one mip level of one layer).
    #[inline]
    pub const fn image_bytes(&self, width: u32, height: u32) -> u64 {
        let (_, block_height) = self.block_dimensions();
        self.row_bytes(width) * height.div_ceil(block_height) as u64
    }

    /// Returns true for the block-compressed (BC) formats.
    #[inline]
    pub const fn is_compressed(&self) -> bool {
        matches!(
            self,
            Self::Bc1RgbaUnorm
                | Self::Bc1RgbaSrgb
                | Self::Bc2RgbaUnorm
                | Self::Bc2RgbaSrgb
                | Self::Bc3RgbaUnorm
                | Self::Bc3RgbaSrgb
                | Self::Bc4RUnorm
                | Self::Bc4RSnorm
                | Self::Bc5RgUnorm
                | Self::Bc5RgSnorm
                | Self::Bc6hRgbUfloat
                | Self::Bc6hRgbSfloat
                | Self::Bc7RgbaUnorm
                | Self::Bc7RgbaSrgb
        )
    }

    /// Returns true if this format is a depth or depth/stencil format.
    #[inline]
    pub const fn is_depth(&self) -> bool {
        matches!(
            self,
            Self::Depth16Unorm
                | Self::Depth32Float
                | Self::Depth24PlusStencil8
                | Self::Depth32FloatStencil8
        )
    }

    /// Returns true if this format contains a stencil component.
    #[inline]
    pub const fn is_stencil(&self) -> bool {
        matches!(self, Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8)
    }

    /// Returns true if this format is sRGB encoded.
    #[inline]
    pub const fn is_srgb(&self) -> bool {
        matches!(
            self,
            Self::Rgba8Srgb
                | Self::Bgra8Srgb
                | Self::Bc1RgbaSrgb
                | Self::Bc2RgbaSrgb
                | Self::Bc3RgbaSrgb
                | Self::Bc7RgbaSrgb
        )
    }

    /// Returns true for unsigned-integer formats, which shaders read as `uvec4` and which
    /// cannot be filtered or blended.
    #[inline]
    pub const fn is_integer(&self) -> bool {
        matches!(self, Self::R32Uint | Self::Rg32Uint | Self::Rgba32Uint)
    }
}

bitflags! {
    /// Buffer usage flags.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct BufferUsage: u32 {
        /// Buffer can be used as a vertex buffer.
        const VERTEX = 1 << 0;
        /// Buffer can be used as an index buffer.
        const INDEX = 1 << 1;
        /// Buffer can be used as a uniform buffer (UBO).
        const UNIFORM = 1 << 2;
        /// Buffer can be used as a storage buffer (SSBO).
        const STORAGE = 1 << 3;
        /// Buffer can be used for indirect draw calls.
        const INDIRECT = 1 << 4;
        /// Buffer can be copied from (source for transfer).
        const COPY_SRC = 1 << 5;
        /// Buffer can be copied to (destination for transfer).
        const COPY_DST = 1 << 6;
    }
}

/// Intended CPU/GPU memory placement for buffers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryUsage {
    /// Fast GPU VRAM (`DEVICE_LOCAL`). CPU cannot access directly.
    GpuOnly,
    /// CPU accessible memory (`HOST_VISIBLE`), used for staging or dynamic dynamic updates.
    CpuToGpu,
    /// GPU written memory readable by CPU for readbacks and screenshots.
    GpuToCpu,
}

bitflags! {
    /// Texture usage flags.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct TextureUsage: u32 {
        /// Texture can be used as a color attachment in dynamic rendering.
        const COLOR_ATTACHMENT = 1 << 0;
        /// Texture can be used as a depth/stencil attachment in dynamic rendering.
        const DEPTH_STENCIL_ATTACHMENT = 1 << 1;
        /// Texture can be sampled by a shader via a sampler.
        const SAMPLED = 1 << 2;
        /// Texture can be used as a storage image (read/write in compute).
        const STORAGE = 1 << 3;
        /// Texture can be used as a copy source.
        const COPY_SRC = 1 << 4;
        /// Texture can be used as a copy destination.
        const COPY_DST = 1 << 5;
    }
}

/// Dimensionality of a texture, which is also the type of its default view (RVK-13, RGL-13).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureDimension {
    /// 1D texture (`height` and `depth` 1, one layer).
    D1,
    /// 2D texture (one layer).
    D2,
    /// 3D volume texture (`depth` slices, one layer).
    D3,
    /// Cube map: 6 square layers (+X, -X, +Y, -Y, +Z, -Z).
    Cube,
    /// Array of 2D layers.
    D2Array,
    /// Array of cube maps: a multiple of 6 square layers.
    CubeArray,
}

impl TextureDimension {
    /// True for the dimensions whose layers are cube faces.
    #[inline]
    pub const fn is_cube(&self) -> bool {
        matches!(self, Self::Cube | Self::CubeArray)
    }
}

/// Multisampling sample count.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SampleCount {
    /// 1 sample (no MSAA).
    #[default]
    Count1 = 1,
    /// 2 samples.
    Count2 = 2,
    /// 4 samples.
    Count4 = 4,
    /// 8 samples.
    Count8 = 8,
}

impl SampleCount {
    /// The number of samples.
    #[inline]
    pub const fn count(&self) -> u32 {
        *self as u32
    }
}

/// Which aspects of a depth/stencil texture a view or barrier covers.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureAspect {
    /// Every aspect (colour, or depth and stencil).
    #[default]
    All,
    /// The depth aspect of a depth or depth/stencil texture.
    DepthOnly,
    /// The stencil aspect of a depth/stencil texture.
    StencilOnly,
}

/// Texture filtering modes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FilterMode {
    /// Nearest neighbor filtering.
    Nearest,
    /// Linear filtering.
    Linear,
}

/// Texture coordinate wrap/addressing modes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerAddressMode {
    /// Tile the texture at every integer boundary.
    Repeat,
    /// Mirror the texture at every integer boundary.
    MirroredRepeat,
    /// Clamp texture coordinates to [0.0, 1.0].
    ClampToEdge,
    /// Clamp texture coordinates to border color.
    ClampToBorder,
}

/// Colour returned by `ClampToBorder` addressing outside [0, 1].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BorderColor {
    /// (0, 0, 0, 0): the default on both Vulkan and OpenGL.
    #[default]
    TransparentBlack,
    /// (0, 0, 0, 1).
    OpaqueBlack,
    /// (1, 1, 1, 1): typical for shadow-map comparisons outside the map.
    OpaqueWhite,
}

/// Comparison function for depth tests and shadow samplers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompareFunction {
    /// Never passes.
    Never,
    /// Passes if fragment value is less than reference.
    Less,
    /// Passes if fragment value equals reference.
    Equal,
    /// Passes if fragment value is less than or equal to reference.
    LessEqual,
    /// Passes if fragment value is greater than reference (used in standard Reverse-Z).
    Greater,
    /// Passes if fragment value does not equal reference.
    NotEqual,
    /// Passes if fragment value is greater than or equal to reference (Reverse-Z standard).
    GreaterEqual,
    /// Always passes.
    Always,
}

/// Index buffer element type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndexType {
    /// 16-bit unsigned integers (`u16`).
    Uint16,
    /// 32-bit unsigned integers (`u32`).
    Uint32,
}

impl IndexType {
    /// Returns the byte size of each index.
    #[inline]
    pub const fn stride(&self) -> u64 {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// Geometric primitive assembly topology.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PrimitiveTopology {
    /// Separate triangles (every 3 vertices form a triangle).
    TriangleList,
    /// Triangle strip.
    TriangleStrip,
    /// Separate lines (every 2 vertices form a line).
    LineList,
    /// Line strip.
    LineStrip,
    /// Individual point primitives.
    PointList,
}

/// Face culling mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CullMode {
    /// No triangles are culled.
    None,
    /// Front-facing triangles are culled.
    Front,
    /// Back-facing triangles are culled.
    Back,
}

/// Front-facing polygon winding orientation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrontFace {
    /// Counter-clockwise winding in screen/window space is front-facing.
    CounterClockwise,
    /// Clockwise winding in screen/window space is front-facing.
    Clockwise,
}

/// Polygon rasterization mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PolygonMode {
    /// Fill polygon interior.
    Fill,
    /// Render polygon boundary as wireframe lines.
    Line,
    /// Render vertices as isolated points.
    Point,
}

/// Load operation when starting a dynamic rendering attachment pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoadOp {
    /// Load existing content from attachment memory.
    Load,
    /// Clear attachment to the specified clear value at start of pass.
    Clear,
    /// Discard previous content (fastest when entire area will be overwritten).
    DontCare,
}

/// Store operation when ending a dynamic rendering attachment pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StoreOp {
    /// Store rendered contents to memory for later sampling or presentation.
    Store,
    /// Discard contents (useful for transient depth buffers not sampled later).
    DontCare,
}

bitflags! {
    /// Color channel write mask.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ColorWriteMask: u8 {
        /// Write red channel.
        const RED = 1 << 0;
        /// Write green channel.
        const GREEN = 1 << 1;
        /// Write blue channel.
        const BLUE = 1 << 2;
        /// Write alpha channel.
        const ALPHA = 1 << 3;
        /// Write all color channels.
        const ALL = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits() | Self::ALPHA.bits();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_indices_match_all() {
        for (i, format) in TextureFormat::ALL.iter().enumerate() {
            assert_eq!(format.index(), i, "{format:?}");
        }
    }

    #[test]
    fn image_sizes_count_whole_blocks() {
        assert_eq!(TextureFormat::Rgba8Unorm.image_bytes(3, 2), 24);
        assert_eq!(TextureFormat::R8Unorm.row_bytes(5), 5);
        // A 5x5 BC1 image is 2x2 blocks of 8 bytes.
        assert_eq!(TextureFormat::Bc1RgbaUnorm.image_bytes(5, 5), 32);
        assert_eq!(TextureFormat::Bc7RgbaUnorm.row_bytes(4), 16);
        assert_eq!(TextureFormat::Bc7RgbaUnorm.block_dimensions(), (4, 4));
    }
}
