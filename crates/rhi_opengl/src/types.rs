//! Type conversions from abstract RHI descriptors to OpenGL 4.5 Core constants.

use korppi_rhi::{
    AccessFlags, BlendFactor, BlendOp, BorderColor, CompareFunction, CullMode, FilterMode,
    FrontFace, IndexType, PolygonMode, PrimitiveTopology, SamplerAddressMode, ShaderStage,
    StencilOperation, TextureFormat, VertexFormat,
};

/// OpenGL texture pixel format mapping.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlTextureFormat {
    /// Sized internal format (e.g., `GL_RGBA8`).
    pub internal_format: u32,
    /// Pixel transfer format (e.g., `GL_RGBA`).
    pub format: u32,
    /// Pixel transfer data type (e.g., `GL_UNSIGNED_BYTE`).
    pub data_type: u32,
    /// Whether format is depth/stencil.
    pub is_depth_stencil: bool,
}

/// Converts `TextureFormat` to OpenGL internal format, transfer format, and data type.
///
/// Block-compressed formats use `internal_format` for transfers and have no `format`/`data_type`
/// (0). Depth/stencil formats transfer their depth aspect only (`DEPTH_COMPONENT`), matching the
/// RHI's copy contract (`Depth24PlusStencil8` is not copyable).
pub fn to_gl_texture_format(format: TextureFormat) -> GlTextureFormat {
    const COMPRESSED_RGBA_S3TC_DXT1_EXT: u32 = 0x83F1;
    const COMPRESSED_RGBA_S3TC_DXT3_EXT: u32 = 0x83F2;
    const COMPRESSED_RGBA_S3TC_DXT5_EXT: u32 = 0x83F3;
    const COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT: u32 = 0x8C4D;
    const COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT: u32 = 0x8C4E;
    const COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT: u32 = 0x8C4F;
    const COMPRESSED_RED_RGTC1: u32 = 0x8DBB;
    const COMPRESSED_SIGNED_RED_RGTC1: u32 = 0x8DBC;
    const COMPRESSED_RG_RGTC2: u32 = 0x8DBD;
    const COMPRESSED_SIGNED_RG_RGTC2: u32 = 0x8DBE;
    const COMPRESSED_RGBA_BPTC_UNORM: u32 = 0x8E8C;
    const COMPRESSED_SRGB_ALPHA_BPTC_UNORM: u32 = 0x8E8D;
    const COMPRESSED_RGB_BPTC_SIGNED_FLOAT: u32 = 0x8E8E;
    const COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT: u32 = 0x8E8F;

    let (internal_format, transfer_format, data_type) = match format {
        TextureFormat::R8Unorm => (glow::R8, glow::RED, glow::UNSIGNED_BYTE),
        TextureFormat::Rg8Unorm => (glow::RG8, glow::RG, glow::UNSIGNED_BYTE),
        TextureFormat::Rgba8Unorm => (glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE),
        TextureFormat::Rgba8Srgb => (glow::SRGB8_ALPHA8, glow::RGBA, glow::UNSIGNED_BYTE),
        TextureFormat::Bgra8Unorm => (glow::RGBA8, glow::BGRA, glow::UNSIGNED_BYTE),
        TextureFormat::Bgra8Srgb => (glow::SRGB8_ALPHA8, glow::BGRA, glow::UNSIGNED_BYTE),
        TextureFormat::Rgba8Snorm => (glow::RGBA8_SNORM, glow::RGBA, glow::BYTE),
        // Red in the low bits, as Vulkan's A2B10G10R10_UNORM_PACK32.
        TextureFormat::Rgb10A2Unorm => (
            glow::RGB10_A2,
            glow::RGBA,
            glow::UNSIGNED_INT_2_10_10_10_REV,
        ),
        // Red in the low bits, as Vulkan's B10G11R11_UFLOAT_PACK32.
        TextureFormat::Rg11B10Float => (
            glow::R11F_G11F_B10F,
            glow::RGB,
            glow::UNSIGNED_INT_10F_11F_11F_REV,
        ),
        TextureFormat::R16Float => (glow::R16F, glow::RED, glow::HALF_FLOAT),
        TextureFormat::Rg16Float => (glow::RG16F, glow::RG, glow::HALF_FLOAT),
        TextureFormat::Rgba16Float => (glow::RGBA16F, glow::RGBA, glow::HALF_FLOAT),
        TextureFormat::Rgba16Unorm => (glow::RGBA16, glow::RGBA, glow::UNSIGNED_SHORT),
        TextureFormat::R32Float => (glow::R32F, glow::RED, glow::FLOAT),
        TextureFormat::Rg32Float => (glow::RG32F, glow::RG, glow::FLOAT),
        TextureFormat::Rgb32Float => (glow::RGB32F, glow::RGB, glow::FLOAT),
        TextureFormat::Rgba32Float => (glow::RGBA32F, glow::RGBA, glow::FLOAT),
        TextureFormat::R32Uint => (glow::R32UI, glow::RED_INTEGER, glow::UNSIGNED_INT),
        TextureFormat::Rg32Uint => (glow::RG32UI, glow::RG_INTEGER, glow::UNSIGNED_INT),
        TextureFormat::Rgba32Uint => (glow::RGBA32UI, glow::RGBA_INTEGER, glow::UNSIGNED_INT),
        TextureFormat::Depth16Unorm => (
            glow::DEPTH_COMPONENT16,
            glow::DEPTH_COMPONENT,
            glow::UNSIGNED_SHORT,
        ),
        TextureFormat::Depth32Float => {
            (glow::DEPTH_COMPONENT32F, glow::DEPTH_COMPONENT, glow::FLOAT)
        }
        TextureFormat::Depth24PlusStencil8 => (
            glow::DEPTH24_STENCIL8,
            glow::DEPTH_STENCIL,
            glow::UNSIGNED_INT_24_8,
        ),
        TextureFormat::Depth32FloatStencil8 => {
            (glow::DEPTH32F_STENCIL8, glow::DEPTH_COMPONENT, glow::FLOAT)
        }
        TextureFormat::Bc1RgbaUnorm => (COMPRESSED_RGBA_S3TC_DXT1_EXT, 0, 0),
        TextureFormat::Bc1RgbaSrgb => (COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, 0, 0),
        TextureFormat::Bc2RgbaUnorm => (COMPRESSED_RGBA_S3TC_DXT3_EXT, 0, 0),
        TextureFormat::Bc2RgbaSrgb => (COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, 0, 0),
        TextureFormat::Bc3RgbaUnorm => (COMPRESSED_RGBA_S3TC_DXT5_EXT, 0, 0),
        TextureFormat::Bc3RgbaSrgb => (COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, 0, 0),
        TextureFormat::Bc4RUnorm => (COMPRESSED_RED_RGTC1, 0, 0),
        TextureFormat::Bc4RSnorm => (COMPRESSED_SIGNED_RED_RGTC1, 0, 0),
        TextureFormat::Bc5RgUnorm => (COMPRESSED_RG_RGTC2, 0, 0),
        TextureFormat::Bc5RgSnorm => (COMPRESSED_SIGNED_RG_RGTC2, 0, 0),
        TextureFormat::Bc6hRgbUfloat => (COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, 0, 0),
        TextureFormat::Bc6hRgbSfloat => (COMPRESSED_RGB_BPTC_SIGNED_FLOAT, 0, 0),
        TextureFormat::Bc7RgbaUnorm => (COMPRESSED_RGBA_BPTC_UNORM, 0, 0),
        TextureFormat::Bc7RgbaSrgb => (COMPRESSED_SRGB_ALPHA_BPTC_UNORM, 0, 0),
    };
    GlTextureFormat {
        internal_format,
        format: transfer_format,
        data_type,
        is_depth_stencil: format.is_depth(),
    }
}

/// Converts `FilterMode` to OpenGL filter constant.
pub fn to_gl_filter(filter: FilterMode) -> i32 {
    match filter {
        FilterMode::Nearest => glow::NEAREST as i32,
        FilterMode::Linear => glow::LINEAR as i32,
    }
}

/// Converts a minification filter plus mipmap filter to the OpenGL minification filter.
///
/// Plain `GL_NEAREST`/`GL_LINEAR` never sample mip levels > 0 (RGL-13), so the mipmap filter is
/// always folded in; textures with a single level are still complete with these filters.
pub fn to_gl_min_filter(min: FilterMode, mipmap: FilterMode) -> i32 {
    (match (min, mipmap) {
        (FilterMode::Nearest, FilterMode::Nearest) => glow::NEAREST_MIPMAP_NEAREST,
        (FilterMode::Linear, FilterMode::Nearest) => glow::LINEAR_MIPMAP_NEAREST,
        (FilterMode::Nearest, FilterMode::Linear) => glow::NEAREST_MIPMAP_LINEAR,
        (FilterMode::Linear, FilterMode::Linear) => glow::LINEAR_MIPMAP_LINEAR,
    }) as i32
}

/// RGBA value of a border colour.
pub fn border_color_rgba(color: BorderColor) -> [f32; 4] {
    match color {
        BorderColor::TransparentBlack => [0.0, 0.0, 0.0, 0.0],
        BorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
        BorderColor::OpaqueWhite => [1.0, 1.0, 1.0, 1.0],
    }
}

/// Converts `SamplerAddressMode` to OpenGL address mode constant.
pub fn to_gl_address_mode(mode: SamplerAddressMode) -> i32 {
    match mode {
        SamplerAddressMode::Repeat => glow::REPEAT as i32,
        SamplerAddressMode::MirroredRepeat => glow::MIRRORED_REPEAT as i32,
        SamplerAddressMode::ClampToEdge => glow::CLAMP_TO_EDGE as i32,
        SamplerAddressMode::ClampToBorder => glow::CLAMP_TO_BORDER as i32,
    }
}

/// Converts `CompareFunction` to OpenGL depth comparison constant.
pub fn to_gl_compare_func(func: CompareFunction) -> i32 {
    match func {
        CompareFunction::Never => glow::NEVER as i32,
        CompareFunction::Less => glow::LESS as i32,
        CompareFunction::Equal => glow::EQUAL as i32,
        CompareFunction::LessEqual => glow::LEQUAL as i32,
        CompareFunction::Greater => glow::GREATER as i32,
        CompareFunction::NotEqual => glow::NOTEQUAL as i32,
        CompareFunction::GreaterEqual => glow::GEQUAL as i32,
        CompareFunction::Always => glow::ALWAYS as i32,
    }
}

/// Converts `PolygonMode` to OpenGL polygon rasterization mode.
pub fn to_gl_polygon_mode(mode: PolygonMode) -> u32 {
    match mode {
        PolygonMode::Fill => glow::FILL,
        PolygonMode::Line => glow::LINE,
        PolygonMode::Point => glow::POINT,
    }
}

/// Converts `PrimitiveTopology` to OpenGL primitive drawing mode.
pub fn to_gl_topology(topo: PrimitiveTopology) -> u32 {
    match topo {
        PrimitiveTopology::PointList => glow::POINTS,
        PrimitiveTopology::LineList => glow::LINES,
        PrimitiveTopology::LineStrip => glow::LINE_STRIP,
        PrimitiveTopology::TriangleList => glow::TRIANGLES,
        PrimitiveTopology::TriangleStrip => glow::TRIANGLE_STRIP,
    }
}

/// Converts `CullMode` to optional OpenGL culling constant.
pub fn to_gl_cull_mode(mode: CullMode) -> Option<u32> {
    match mode {
        CullMode::None => None,
        CullMode::Front => Some(glow::FRONT),
        CullMode::Back => Some(glow::BACK),
    }
}

/// Converts `FrontFace` to OpenGL front face constant.
pub fn to_gl_front_face(face: FrontFace) -> u32 {
    match face {
        FrontFace::CounterClockwise => glow::CCW,
        FrontFace::Clockwise => glow::CW,
    }
}

/// Converts `BlendFactor` to OpenGL blend factor constant.
pub fn to_gl_blend_factor(factor: BlendFactor) -> u32 {
    match factor {
        BlendFactor::Zero => glow::ZERO,
        BlendFactor::One => glow::ONE,
        BlendFactor::SrcAlpha => glow::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => glow::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstAlpha => glow::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => glow::ONE_MINUS_DST_ALPHA,
        BlendFactor::SrcColor => glow::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => glow::ONE_MINUS_SRC_COLOR,
        BlendFactor::DstColor => glow::DST_COLOR,
        BlendFactor::OneMinusDstColor => glow::ONE_MINUS_DST_COLOR,
        BlendFactor::Constant => glow::CONSTANT_COLOR,
        BlendFactor::OneMinusConstant => glow::ONE_MINUS_CONSTANT_COLOR,
    }
}

/// Converts an RHI stencil operation to OpenGL.
pub fn to_gl_stencil_op(op: StencilOperation) -> u32 {
    match op {
        StencilOperation::Keep => glow::KEEP,
        StencilOperation::Zero => glow::ZERO,
        StencilOperation::Replace => glow::REPLACE,
        StencilOperation::IncrementClamp => glow::INCR,
        StencilOperation::DecrementClamp => glow::DECR,
        StencilOperation::Invert => glow::INVERT,
        StencilOperation::IncrementWrap => glow::INCR_WRAP,
        StencilOperation::DecrementWrap => glow::DECR_WRAP,
    }
}

/// Converts `BlendOp` to OpenGL blend equation constant.
pub fn to_gl_blend_op(op: BlendOp) -> u32 {
    match op {
        BlendOp::Add => glow::FUNC_ADD,
        BlendOp::Subtract => glow::FUNC_SUBTRACT,
        BlendOp::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
        BlendOp::Min => glow::MIN,
        BlendOp::Max => glow::MAX,
    }
}

/// Converts `IndexType` to OpenGL index type constant and byte stride.
pub fn to_gl_index_type(index_type: IndexType) -> (u32, usize) {
    match index_type {
        IndexType::Uint16 => (glow::UNSIGNED_SHORT, 2),
        IndexType::Uint32 => (glow::UNSIGNED_INT, 4),
    }
}

/// Converts `ShaderStage` to OpenGL shader stage constant.
pub fn to_gl_shader_stage(stage: ShaderStage) -> u32 {
    match stage {
        ShaderStage::Vertex => glow::VERTEX_SHADER,
        ShaderStage::Fragment => glow::FRAGMENT_SHADER,
        ShaderStage::Compute => glow::COMPUTE_SHADER,
    }
}

/// How a vertex attribute is fetched in OpenGL.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlVertexFormat {
    /// Component count (1-4).
    pub size: i32,
    /// Component type (e.g. `GL_FLOAT`).
    pub data_type: u32,
    /// Integer attribute (`glVertexArrayAttribIFormat`) rather than float.
    pub integer: bool,
    /// Normalize fixed-point data to [0, 1] (float attributes only).
    pub normalized: bool,
}

/// Converts an RHI vertex format to its OpenGL attribute format.
pub fn to_gl_vertex_format(format: VertexFormat) -> GlVertexFormat {
    let (size, data_type, normalized) = match format {
        VertexFormat::Float32 => (1, glow::FLOAT, false),
        VertexFormat::Float32x2 => (2, glow::FLOAT, false),
        VertexFormat::Float32x3 => (3, glow::FLOAT, false),
        VertexFormat::Float32x4 => (4, glow::FLOAT, false),
        VertexFormat::Float16x2 => (2, glow::HALF_FLOAT, false),
        VertexFormat::Float16x4 => (4, glow::HALF_FLOAT, false),
        VertexFormat::Uint32 => (1, glow::UNSIGNED_INT, false),
        VertexFormat::Uint32x2 => (2, glow::UNSIGNED_INT, false),
        VertexFormat::Uint32x3 => (3, glow::UNSIGNED_INT, false),
        VertexFormat::Uint32x4 => (4, glow::UNSIGNED_INT, false),
        VertexFormat::Sint32 => (1, glow::INT, false),
        VertexFormat::Sint32x2 => (2, glow::INT, false),
        VertexFormat::Sint32x4 => (4, glow::INT, false),
        VertexFormat::Uint8x4Norm => (4, glow::UNSIGNED_BYTE, true),
        VertexFormat::Sint8x4Norm => (4, glow::BYTE, true),
        VertexFormat::Uint8x4 => (4, glow::UNSIGNED_BYTE, false),
        VertexFormat::Uint16x2Norm => (2, glow::UNSIGNED_SHORT, true),
        VertexFormat::Uint16x4Norm => (4, glow::UNSIGNED_SHORT, true),
        VertexFormat::Sint16x2Norm => (2, glow::SHORT, true),
        VertexFormat::Sint16x4Norm => (4, glow::SHORT, true),
        VertexFormat::Uint16x2 => (2, glow::UNSIGNED_SHORT, false),
        VertexFormat::Uint16x4 => (4, glow::UNSIGNED_SHORT, false),
    };
    GlVertexFormat {
        size,
        data_type,
        integer: format.is_integer(),
        normalized,
    }
}

/// `glMemoryBarrier` bits for an RHI barrier (RGL-16), or 0 if none is needed.
///
/// Only incoherent shader writes (storage buffers, image stores) need `glMemoryBarrier`; GL
/// orders and makes visible every other write by itself (GL 4.6 §7.13.2). The bits name how the
/// written data is used afterwards, so they follow the destination access of each barrier whose
/// source includes shader writes.
pub fn memory_barrier_bits(barrier: &korppi_rhi::PipelineBarrier) -> u32 {
    let accesses = barrier
        .memory_barriers
        .iter()
        .map(|b| (b.src_access, b.dst_access))
        .chain(
            barrier
                .buffer_barriers
                .iter()
                .map(|b| (b.src_access, b.dst_access)),
        )
        .chain(
            barrier
                .image_barriers
                .iter()
                .map(|b| (b.src_access, b.dst_access)),
        );
    accesses
        .filter(|(src, _)| src.intersects(AccessFlags::SHADER_WRITE | AccessFlags::MEMORY_WRITE))
        .fold(0, |bits, (_, dst)| bits | barrier_bits_for_access(dst))
}

/// `glMemoryBarrier` bits for data that shaders wrote and that is then accessed as `dst`.
fn barrier_bits_for_access(dst: AccessFlags) -> u32 {
    if dst.intersects(AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE) {
        return glow::ALL_BARRIER_BITS;
    }
    let mapping = [
        (
            AccessFlags::INDIRECT_COMMAND_READ,
            glow::COMMAND_BARRIER_BIT,
        ),
        (AccessFlags::INDEX_READ, glow::ELEMENT_ARRAY_BARRIER_BIT),
        (
            AccessFlags::VERTEX_ATTRIBUTE_READ,
            glow::VERTEX_ATTRIB_ARRAY_BARRIER_BIT,
        ),
        (AccessFlags::UNIFORM_READ, glow::UNIFORM_BARRIER_BIT),
        (
            AccessFlags::SHADER_READ,
            glow::SHADER_STORAGE_BARRIER_BIT
                | glow::TEXTURE_FETCH_BARRIER_BIT
                | glow::SHADER_IMAGE_ACCESS_BARRIER_BIT,
        ),
        (
            AccessFlags::SHADER_WRITE,
            glow::SHADER_STORAGE_BARRIER_BIT | glow::SHADER_IMAGE_ACCESS_BARRIER_BIT,
        ),
        (
            AccessFlags::COLOR_ATTACHMENT_READ
                | AccessFlags::COLOR_ATTACHMENT_WRITE
                | AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            glow::FRAMEBUFFER_BARRIER_BIT,
        ),
        (
            AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE,
            glow::BUFFER_UPDATE_BARRIER_BIT
                | glow::PIXEL_BUFFER_BARRIER_BIT
                | glow::TEXTURE_UPDATE_BARRIER_BIT,
        ),
        (
            AccessFlags::HOST_READ | AccessFlags::HOST_WRITE,
            glow::CLIENT_MAPPED_BUFFER_BARRIER_BIT
                | glow::BUFFER_UPDATE_BARRIER_BIT
                | glow::TEXTURE_UPDATE_BARRIER_BIT,
        ),
    ];
    mapping
        .iter()
        .filter(|(access, _)| dst.intersects(*access))
        .fold(0, |bits, (_, gl_bits)| bits | gl_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use korppi_rhi::{
        BufferHandle, BufferMemoryBarrier, MemoryBarrier, PipelineBarrier, PipelineStageFlags,
    };

    fn buffer_barrier(src: AccessFlags, dst: AccessFlags) -> PipelineBarrier {
        PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
            buffer: BufferHandle::new(1, 1),
            src_stages: PipelineStageFlags::COMPUTE_SHADER,
            src_access: src,
            dst_stages: PipelineStageFlags::ALL_COMMANDS,
            dst_access: dst,
            offset: 0,
            size: 16,
        })
    }

    #[test]
    fn barrier_bits_follow_the_destination_access_of_shader_writes() {
        let cases = [
            (AccessFlags::SHADER_READ, glow::SHADER_STORAGE_BARRIER_BIT),
            (AccessFlags::UNIFORM_READ, glow::UNIFORM_BARRIER_BIT),
            (
                AccessFlags::VERTEX_ATTRIBUTE_READ,
                glow::VERTEX_ATTRIB_ARRAY_BARRIER_BIT,
            ),
            (AccessFlags::INDEX_READ, glow::ELEMENT_ARRAY_BARRIER_BIT),
            (
                AccessFlags::INDIRECT_COMMAND_READ,
                glow::COMMAND_BARRIER_BIT,
            ),
            (AccessFlags::TRANSFER_READ, glow::BUFFER_UPDATE_BARRIER_BIT),
            (
                AccessFlags::HOST_READ,
                glow::CLIENT_MAPPED_BUFFER_BARRIER_BIT,
            ),
            (
                AccessFlags::COLOR_ATTACHMENT_WRITE,
                glow::FRAMEBUFFER_BARRIER_BIT,
            ),
        ];
        for (dst, expected) in cases {
            let bits = memory_barrier_bits(&buffer_barrier(AccessFlags::SHADER_WRITE, dst));
            assert_eq!(bits & expected, expected, "{dst:?} -> {bits:#x}");
            assert_ne!(
                bits,
                glow::ALL_BARRIER_BITS,
                "{dst:?} must not use every bit"
            );
        }
        assert_eq!(
            memory_barrier_bits(&buffer_barrier(
                AccessFlags::SHADER_WRITE,
                AccessFlags::MEMORY_READ
            )),
            glow::ALL_BARRIER_BITS
        );
    }

    #[test]
    fn barriers_without_shader_writes_need_no_gl_barrier() {
        // GL orders and makes visible every write except incoherent shader stores (GL 4.6
        // §7.13.2), so transfer and attachment writes need no glMemoryBarrier.
        for src in [
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            AccessFlags::SHADER_READ,
            AccessFlags::empty(),
        ] {
            assert_eq!(
                memory_barrier_bits(&buffer_barrier(src, AccessFlags::SHADER_READ)),
                0,
                "{src:?}"
            );
        }
        let global = PipelineBarrier::new().with_memory_barrier(MemoryBarrier {
            src_stages: PipelineStageFlags::ALL_COMMANDS,
            src_access: AccessFlags::MEMORY_WRITE,
            dst_stages: PipelineStageFlags::ALL_COMMANDS,
            dst_access: AccessFlags::SHADER_READ,
        });
        assert_ne!(
            memory_barrier_bits(&global),
            0,
            "MEMORY_WRITE includes shader writes"
        );
    }

    #[test]
    fn vertex_formats_map_to_gl_attribute_formats() {
        assert_eq!(
            to_gl_vertex_format(VertexFormat::Float32x3),
            GlVertexFormat {
                size: 3,
                data_type: glow::FLOAT,
                integer: false,
                normalized: false
            }
        );
        let packed = to_gl_vertex_format(VertexFormat::Uint8x4Norm);
        assert!(packed.normalized && !packed.integer && packed.data_type == glow::UNSIGNED_BYTE);
        let uint = to_gl_vertex_format(VertexFormat::Uint32x2);
        assert!(uint.integer && uint.size == 2 && uint.data_type == glow::UNSIGNED_INT);
    }

    #[test]
    fn min_filters_always_include_the_mipmap_filter() {
        assert_eq!(
            to_gl_min_filter(FilterMode::Linear, FilterMode::Linear),
            glow::LINEAR_MIPMAP_LINEAR as i32
        );
        assert_eq!(
            to_gl_min_filter(FilterMode::Nearest, FilterMode::Linear),
            glow::NEAREST_MIPMAP_LINEAR as i32
        );
        assert_eq!(
            to_gl_min_filter(FilterMode::Nearest, FilterMode::Nearest),
            glow::NEAREST_MIPMAP_NEAREST as i32
        );
    }

    #[test]
    fn index_types_map_to_gl_types_and_strides() {
        assert_eq!(
            to_gl_index_type(IndexType::Uint16),
            (glow::UNSIGNED_SHORT, 2)
        );
        assert_eq!(to_gl_index_type(IndexType::Uint32), (glow::UNSIGNED_INT, 4));
    }
}
