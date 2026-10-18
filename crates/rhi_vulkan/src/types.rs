use ash::vk;
use korppi_rhi::{
    AccessFlags, BlendFactor, BlendOp, BorderColor, BufferUsage, ColorTargetState, ColorWriteMask,
    CompareFunction, CullMode, FilterMode, FrontFace, ImageLayout, LoadOp, PipelineStageFlags,
    PolygonMode, PrimitiveTopology, PushConstantRange, RasterizerState, SamplerAddressMode,
    SamplerDescriptor, ShaderStages, StencilFaceState, StencilOperation, StoreOp, TextureFormat,
    TextureSubresourceRange, TextureUsage, VertexFormat, VertexStepMode,
};

/// Converts RHI TextureFormat to Vulkan vk::Format.
///
/// `Depth24PlusStencil8` maps to `D24_UNORM_S8_UINT` here; devices without it use
/// `D32_SFLOAT_S8_UINT` instead (`VulkanCapabilities::vk_format`, RVK-13).
pub fn to_vk_format(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::Rgba8Snorm => vk::Format::R8G8B8A8_SNORM,
        TextureFormat::Rgb10A2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
        TextureFormat::Rg11B10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
        TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,
        TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        TextureFormat::Rgb32Float => vk::Format::R32G32B32_SFLOAT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        TextureFormat::R32Uint => vk::Format::R32_UINT,
        TextureFormat::Rg32Uint => vk::Format::R32G32_UINT,
        TextureFormat::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
        TextureFormat::Depth16Unorm => vk::Format::D16_UNORM,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        TextureFormat::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
        TextureFormat::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
        TextureFormat::Bc1RgbaUnorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TextureFormat::Bc1RgbaSrgb => vk::Format::BC1_RGBA_SRGB_BLOCK,
        TextureFormat::Bc2RgbaUnorm => vk::Format::BC2_UNORM_BLOCK,
        TextureFormat::Bc2RgbaSrgb => vk::Format::BC2_SRGB_BLOCK,
        TextureFormat::Bc3RgbaUnorm => vk::Format::BC3_UNORM_BLOCK,
        TextureFormat::Bc3RgbaSrgb => vk::Format::BC3_SRGB_BLOCK,
        TextureFormat::Bc4RUnorm => vk::Format::BC4_UNORM_BLOCK,
        TextureFormat::Bc4RSnorm => vk::Format::BC4_SNORM_BLOCK,
        TextureFormat::Bc5RgUnorm => vk::Format::BC5_UNORM_BLOCK,
        TextureFormat::Bc5RgSnorm => vk::Format::BC5_SNORM_BLOCK,
        TextureFormat::Bc6hRgbUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
        TextureFormat::Bc6hRgbSfloat => vk::Format::BC6H_SFLOAT_BLOCK,
        TextureFormat::Bc7RgbaUnorm => vk::Format::BC7_UNORM_BLOCK,
        TextureFormat::Bc7RgbaSrgb => vk::Format::BC7_SRGB_BLOCK,
    }
}

/// Aspects a layout transition of a texture with `format` must cover: both depth and stencil
/// for combined formats (VUID-VkImageMemoryBarrier2-image-03320, as `separateDepthStencilLayouts`
/// is not enabled), depth for depth-only formats (-10749), colour otherwise (RVK-07).
pub fn barrier_aspect(format: TextureFormat) -> vk::ImageAspectFlags {
    if format.is_stencil() {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else if format.is_depth() {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

/// The Vulkan subresource range of an RHI range over a texture of `format` (RHI-06). Without
/// `separateDepthStencilLayouts`, depth/stencil images transition both aspects together, so the
/// range's aspect only matters for colour/depth-only formats.
pub fn to_vk_subresource_range(
    range: &TextureSubresourceRange,
    format: TextureFormat,
) -> vk::ImageSubresourceRange {
    let count = |n: u32| {
        if n == TextureSubresourceRange::REMAINING {
            vk::REMAINING_ARRAY_LAYERS
        } else {
            n
        }
    };
    vk::ImageSubresourceRange {
        aspect_mask: barrier_aspect(format),
        base_mip_level: range.base_mip_level,
        level_count: if range.mip_level_count == TextureSubresourceRange::REMAINING {
            vk::REMAINING_MIP_LEVELS
        } else {
            range.mip_level_count
        },
        base_array_layer: range.base_array_layer,
        layer_count: count(range.array_layer_count),
    }
}

/// Aspect a buffer/image copy of a texture's main data uses: depth for depth formats, colour
/// otherwise. (Copies address exactly one aspect.)
pub fn copy_aspect(format: TextureFormat) -> vk::ImageAspectFlags {
    if format.is_depth() {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

/// Converts the RHI's byte row pitch to `VkBufferImageCopy::bufferRowLength`, which is in texels
/// (0 means tightly packed) (RVK-06). For block-compressed formats a row of blocks covers
/// `block width` texels.
pub fn buffer_row_length_texels(row_pitch_bytes: u32, format: TextureFormat) -> u32 {
    let (block_width, _) = format.block_dimensions();
    row_pitch_bytes / format.block_bytes() * block_width
}

/// Maximum push-constant ranges per pipeline layout supported by [`push_constant_segments`].
pub const MAX_PUSH_CONSTANT_RANGES: usize = 16;

/// Splits a push of `size` bytes at `offset` into segments that a single
/// `vkCmdPushConstants` call each can update validly: within a segment, every byte is covered
/// by the same set of `ranges`, and the segment's stage flags are the union of those ranges'
/// stages (VUID-vkCmdPushConstants-offset-01795 and -01796). Bytes no range covers are skipped
/// (command validation reports them). Calls `emit(stage_flags, offset, size)` per segment.
pub fn push_constant_segments(
    ranges: &[vk::PushConstantRange],
    offset: u32,
    size: u32,
    mut emit: impl FnMut(vk::ShaderStageFlags, u32, u32),
) {
    let end = offset.saturating_add(size);
    let mut cuts = [0_u32; 2 + 2 * MAX_PUSH_CONSTANT_RANGES];
    let mut count = 0;
    let mut push_cut = |cut: u32| {
        if cut >= offset && cut <= end && count < cuts.len() {
            cuts[count] = cut;
            count += 1;
        }
    };
    push_cut(offset);
    push_cut(end);
    for range in ranges.iter().take(MAX_PUSH_CONSTANT_RANGES) {
        push_cut(range.offset);
        push_cut(range.offset.saturating_add(range.size));
    }
    let cuts = &mut cuts[..count];
    cuts.sort_unstable();
    for pair in cuts.windows(2) {
        let (start, stop) = (pair[0], pair[1]);
        if start == stop {
            continue;
        }
        let stages = ranges
            .iter()
            .filter(|r| r.offset <= start && stop <= r.offset.saturating_add(r.size))
            .fold(vk::ShaderStageFlags::empty(), |acc, r| acc | r.stage_flags);
        if !stages.is_empty() {
            emit(stages, start, stop - start);
        }
    }
}

/// Converts an RHI border colour to Vulkan.
pub fn to_vk_border_color(color: BorderColor) -> vk::BorderColor {
    match color {
        BorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        BorderColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        BorderColor::OpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
    }
}

/// Builds `VkSamplerCreateInfo` for `desc` (RVK-05).
///
/// - `mipmapMode` follows `mipmap_filter`, and `maxLod` is `VK_LOD_CLAMP_NONE`. The default of 0
///   clamped every lookup to mip 0 and made `magFilter` apply everywhere.
/// - Anisotropy is enabled only when the device feature is enabled (`anisotropy_supported`),
///   clamped to `max_anisotropy_limit` (`VkPhysicalDeviceLimits::maxSamplerAnisotropy`).
pub fn sampler_create_info(
    desc: &SamplerDescriptor,
    anisotropy_supported: bool,
    max_anisotropy_limit: f32,
    max_lod_bias: f32,
) -> vk::SamplerCreateInfo<'static> {
    let anisotropy = anisotropy_supported && desc.max_anisotropy > 1;
    let mipmap_mode = match desc.mipmap_filter {
        FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    };
    vk::SamplerCreateInfo::default()
        .mag_filter(to_vk_filter(desc.mag_filter))
        .min_filter(to_vk_filter(desc.min_filter))
        .mipmap_mode(mipmap_mode)
        .address_mode_u(to_vk_sampler_address_mode(desc.address_mode_u))
        .address_mode_v(to_vk_sampler_address_mode(desc.address_mode_v))
        .address_mode_w(to_vk_sampler_address_mode(desc.address_mode_w))
        .min_lod(desc.lod_min_clamp)
        .max_lod(desc.lod_max_clamp)
        // VUID-VkSamplerCreateInfo-mipLodBias-01069: at most `maxSamplerLodBias`.
        .mip_lod_bias(desc.mip_lod_bias.clamp(-max_lod_bias, max_lod_bias))
        .anisotropy_enable(anisotropy)
        .max_anisotropy(if anisotropy {
            f32::from(desc.max_anisotropy).min(max_anisotropy_limit)
        } else {
            1.0
        })
        .compare_enable(desc.compare.is_some())
        .compare_op(
            desc.compare
                .map(to_vk_compare_op)
                .unwrap_or(vk::CompareOp::ALWAYS),
        )
        .border_color(to_vk_border_color(desc.border_color))
}

/// Converts RHI BufferUsage to Vulkan vk::BufferUsageFlags.
pub fn to_vk_buffer_usage(usage: BufferUsage) -> vk::BufferUsageFlags {
    let mut flags = vk::BufferUsageFlags::empty();
    if usage.contains(BufferUsage::VERTEX) {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.contains(BufferUsage::INDEX) {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.contains(BufferUsage::STORAGE) {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if usage.contains(BufferUsage::COPY_SRC) {
        flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(BufferUsage::COPY_DST) {
        flags |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    flags
}

/// Converts RHI TextureUsage to Vulkan vk::ImageUsageFlags.
pub fn to_vk_image_usage(usage: TextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    if usage.contains(TextureUsage::COLOR_ATTACHMENT) {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
        flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if usage.contains(TextureUsage::SAMPLED) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(TextureUsage::STORAGE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.contains(TextureUsage::COPY_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(TextureUsage::COPY_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    flags
}

/// Converts RHI ImageLayout to Vulkan vk::ImageLayout.
pub fn to_vk_image_layout(layout: ImageLayout) -> vk::ImageLayout {
    match layout {
        ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
        ImageLayout::General => vk::ImageLayout::GENERAL,
        ImageLayout::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ImageLayout::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ImageLayout::DepthStencilReadOnly => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ImageLayout::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ImageLayout::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageLayout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageLayout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

/// Converts RHI PipelineStageFlags to Vulkan vk::PipelineStageFlags2.
pub fn to_vk_pipeline_stage_flags(stages: PipelineStageFlags) -> vk::PipelineStageFlags2 {
    let mut flags = vk::PipelineStageFlags2::empty();
    if stages.contains(PipelineStageFlags::TOP_OF_PIPE) {
        flags |= vk::PipelineStageFlags2::TOP_OF_PIPE;
    }
    if stages.contains(PipelineStageFlags::DRAW_INDIRECT) {
        flags |= vk::PipelineStageFlags2::DRAW_INDIRECT;
    }
    if stages.contains(PipelineStageFlags::VERTEX_INPUT) {
        flags |= vk::PipelineStageFlags2::VERTEX_INPUT;
    }
    if stages.contains(PipelineStageFlags::VERTEX_SHADER) {
        flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
    }
    if stages.contains(PipelineStageFlags::FRAGMENT_SHADER) {
        flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
    }
    if stages.contains(PipelineStageFlags::EARLY_FRAGMENT_TESTS) {
        flags |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
    }
    if stages.contains(PipelineStageFlags::LATE_FRAGMENT_TESTS) {
        flags |= vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
    }
    if stages.contains(PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) {
        flags |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
    }
    if stages.contains(PipelineStageFlags::COMPUTE_SHADER) {
        flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
    }
    if stages.contains(PipelineStageFlags::TRANSFER) {
        flags |= vk::PipelineStageFlags2::TRANSFER;
    }
    if stages.contains(PipelineStageFlags::BOTTOM_OF_PIPE) {
        flags |= vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
    }
    if stages.contains(PipelineStageFlags::ALL_GRAPHICS) {
        flags |= vk::PipelineStageFlags2::ALL_GRAPHICS;
    }
    if stages.contains(PipelineStageFlags::ALL_COMMANDS) {
        flags |= vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    if stages.contains(PipelineStageFlags::HOST) {
        flags |= vk::PipelineStageFlags2::HOST;
    }
    flags
}

/// Converts RHI AccessFlags to Vulkan vk::AccessFlags2.
pub fn to_vk_access_flags(access: AccessFlags) -> vk::AccessFlags2 {
    let mut flags = vk::AccessFlags2::empty();
    if access.contains(AccessFlags::INDIRECT_COMMAND_READ) {
        flags |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }
    if access.contains(AccessFlags::INDEX_READ) {
        flags |= vk::AccessFlags2::INDEX_READ;
    }
    if access.contains(AccessFlags::VERTEX_ATTRIBUTE_READ) {
        flags |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
    }
    if access.contains(AccessFlags::UNIFORM_READ) {
        flags |= vk::AccessFlags2::UNIFORM_READ;
    }
    if access.contains(AccessFlags::SHADER_READ) {
        flags |= vk::AccessFlags2::SHADER_READ;
    }
    if access.contains(AccessFlags::SHADER_WRITE) {
        flags |= vk::AccessFlags2::SHADER_WRITE;
    }
    if access.contains(AccessFlags::COLOR_ATTACHMENT_READ) {
        flags |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
    }
    if access.contains(AccessFlags::COLOR_ATTACHMENT_WRITE) {
        flags |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
    }
    if access.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ) {
        flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    if access.contains(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
        flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if access.contains(AccessFlags::TRANSFER_READ) {
        flags |= vk::AccessFlags2::TRANSFER_READ;
    }
    if access.contains(AccessFlags::TRANSFER_WRITE) {
        flags |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if access.contains(AccessFlags::HOST_READ) {
        flags |= vk::AccessFlags2::HOST_READ;
    }
    if access.contains(AccessFlags::HOST_WRITE) {
        flags |= vk::AccessFlags2::HOST_WRITE;
    }
    if access.contains(AccessFlags::MEMORY_READ) {
        flags |= vk::AccessFlags2::MEMORY_READ;
    }
    if access.contains(AccessFlags::MEMORY_WRITE) {
        flags |= vk::AccessFlags2::MEMORY_WRITE;
    }
    flags
}

/// Converts RHI CullMode to Vulkan vk::CullModeFlags.
pub fn to_vk_cull_mode(cull: CullMode) -> vk::CullModeFlags {
    match cull {
        CullMode::None => vk::CullModeFlags::NONE,
        CullMode::Front => vk::CullModeFlags::FRONT,
        CullMode::Back => vk::CullModeFlags::BACK,
    }
}

/// Converts RHI FrontFace to Vulkan vk::FrontFace.
///
/// The mapping is the identity, even though the backend flips the viewport (negative height).
/// The Vulkan polygon-area formula carries a leading minus sign because framebuffer y points
/// down: `a = -1/2 * sum(x_f[i] * y_f[i+1] - x_f[i+1] * y_f[i])` ("Basic Polygon Rasterization").
/// The negative-height viewport reflects y, which negates the sum, so a triangle that is
/// counter-clockwise in +Y-up NDC has `a > 0` and is front-facing under
/// `VK_FRONT_FACE_COUNTER_CLOCKWISE`, exactly as in OpenGL. (Finding RVK-01: mapping CCW to
/// `CLOCKWISE` made back-face culling remove every correctly wound triangle.)
pub fn to_vk_front_face(face: FrontFace) -> vk::FrontFace {
    match face {
        FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
        FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
    }
}

/// Converts RHI PolygonMode to Vulkan vk::PolygonMode.
pub fn to_vk_polygon_mode(mode: PolygonMode) -> vk::PolygonMode {
    match mode {
        PolygonMode::Fill => vk::PolygonMode::FILL,
        PolygonMode::Line => vk::PolygonMode::LINE,
        PolygonMode::Point => vk::PolygonMode::POINT,
    }
}

/// Converts RHI LoadOp to Vulkan vk::AttachmentLoadOp.
pub fn to_vk_load_op(op: LoadOp) -> vk::AttachmentLoadOp {
    match op {
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
        LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

/// Converts RHI StoreOp to Vulkan vk::AttachmentStoreOp.
pub fn to_vk_store_op(op: StoreOp) -> vk::AttachmentStoreOp {
    match op {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
    }
}

/// Converts RHI FilterMode to Vulkan vk::Filter.
pub fn to_vk_filter(filter: FilterMode) -> vk::Filter {
    match filter {
        FilterMode::Nearest => vk::Filter::NEAREST,
        FilterMode::Linear => vk::Filter::LINEAR,
    }
}

/// Converts RHI SamplerAddressMode to Vulkan vk::SamplerAddressMode.
pub fn to_vk_sampler_address_mode(mode: SamplerAddressMode) -> vk::SamplerAddressMode {
    match mode {
        SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        SamplerAddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        SamplerAddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

/// Converts RHI CompareFunction to Vulkan vk::CompareOp.
pub fn to_vk_compare_op(func: CompareFunction) -> vk::CompareOp {
    match func {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

/// Converts RHI VertexFormat to Vulkan vk::Format.
pub fn to_vk_vertex_format(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Float32 => vk::Format::R32_SFLOAT,
        VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Float16x2 => vk::Format::R16G16_SFLOAT,
        VertexFormat::Float16x4 => vk::Format::R16G16B16A16_SFLOAT,
        VertexFormat::Uint32 => vk::Format::R32_UINT,
        VertexFormat::Uint32x2 => vk::Format::R32G32_UINT,
        VertexFormat::Uint32x3 => vk::Format::R32G32B32_UINT,
        VertexFormat::Uint32x4 => vk::Format::R32G32B32A32_UINT,
        VertexFormat::Sint32 => vk::Format::R32_SINT,
        VertexFormat::Sint32x2 => vk::Format::R32G32_SINT,
        VertexFormat::Sint32x4 => vk::Format::R32G32B32A32_SINT,
        VertexFormat::Uint8x4Norm => vk::Format::R8G8B8A8_UNORM,
        VertexFormat::Sint8x4Norm => vk::Format::R8G8B8A8_SNORM,
        VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
        VertexFormat::Uint16x2Norm => vk::Format::R16G16_UNORM,
        VertexFormat::Uint16x4Norm => vk::Format::R16G16B16A16_UNORM,
        VertexFormat::Sint16x2Norm => vk::Format::R16G16_SNORM,
        VertexFormat::Sint16x4Norm => vk::Format::R16G16B16A16_SNORM,
        VertexFormat::Uint16x2 => vk::Format::R16G16_UINT,
        VertexFormat::Uint16x4 => vk::Format::R16G16B16A16_UINT,
    }
}

/// Converts an RHI blend factor to Vulkan.
pub fn to_vk_blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::Constant => vk::BlendFactor::CONSTANT_COLOR,
        BlendFactor::OneMinusConstant => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
    }
}

/// Converts an RHI stencil operation to Vulkan.
pub fn to_vk_stencil_op(op: StencilOperation) -> vk::StencilOp {
    match op {
        StencilOperation::Keep => vk::StencilOp::KEEP,
        StencilOperation::Zero => vk::StencilOp::ZERO,
        StencilOperation::Replace => vk::StencilOp::REPLACE,
        StencilOperation::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOperation::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOperation::Invert => vk::StencilOp::INVERT,
        StencilOperation::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOperation::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

/// The stencil state of one face; the reference is dynamic.
pub fn to_vk_stencil_face(
    face: &StencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> vk::StencilOpState {
    vk::StencilOpState {
        fail_op: to_vk_stencil_op(face.fail_op),
        pass_op: to_vk_stencil_op(face.pass_op),
        depth_fail_op: to_vk_stencil_op(face.depth_fail_op),
        compare_op: to_vk_compare_op(face.compare),
        compare_mask: read_mask,
        write_mask,
        reference: 0,
    }
}

/// Vulkan push-constant ranges for RHI ranges (RHI-04, RVK-10). Vulkan allows each stage in one
/// range only (VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292), so each stage gets
/// the hull of the RHI ranges that include it; stages with equal hulls share a range.
pub fn vk_push_constant_ranges(ranges: &[PushConstantRange]) -> Vec<vk::PushConstantRange> {
    let mut out: Vec<vk::PushConstantRange> = Vec::new();
    for (stage, flags) in [
        (ShaderStages::VERTEX, vk::ShaderStageFlags::VERTEX),
        (ShaderStages::FRAGMENT, vk::ShaderStageFlags::FRAGMENT),
        (ShaderStages::COMPUTE, vk::ShaderStageFlags::COMPUTE),
    ] {
        let hull = ranges
            .iter()
            .filter(|r| r.stages.contains(stage) && r.size > 0)
            .map(|r| (r.offset, r.offset.saturating_add(r.size)))
            .reduce(|(a0, a1), (b0, b1)| (a0.min(b0), a1.max(b1)));
        let Some((start, end)) = hull else {
            continue;
        };
        match out
            .iter_mut()
            .find(|r| r.offset == start && r.size == end - start)
        {
            Some(range) => range.stage_flags |= flags,
            None => out.push(vk::PushConstantRange {
                stage_flags: flags,
                offset: start,
                size: end - start,
            }),
        }
    }
    out
}

/// Converts an RHI blend operation to Vulkan.
pub fn to_vk_blend_op(op: BlendOp) -> vk::BlendOp {
    match op {
        BlendOp::Add => vk::BlendOp::ADD,
        BlendOp::Subtract => vk::BlendOp::SUBTRACT,
        BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOp::Min => vk::BlendOp::MIN,
        BlendOp::Max => vk::BlendOp::MAX,
    }
}

/// Converts an RHI colour write mask to Vulkan.
pub fn to_vk_color_write_mask(mask: ColorWriteMask) -> vk::ColorComponentFlags {
    let mut flags = vk::ColorComponentFlags::empty();
    if mask.contains(ColorWriteMask::RED) {
        flags |= vk::ColorComponentFlags::R;
    }
    if mask.contains(ColorWriteMask::GREEN) {
        flags |= vk::ColorComponentFlags::G;
    }
    if mask.contains(ColorWriteMask::BLUE) {
        flags |= vk::ColorComponentFlags::B;
    }
    if mask.contains(ColorWriteMask::ALPHA) {
        flags |= vk::ColorComponentFlags::A;
    }
    flags
}

/// Builds the Vulkan blend state for one colour target (blending and write mask).
pub fn to_vk_color_blend_attachment(
    target: &ColorTargetState,
) -> vk::PipelineColorBlendAttachmentState {
    let state = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(to_vk_color_write_mask(target.write_mask));
    match target.blend {
        None => state.blend_enable(false),
        Some(blend) => state
            .blend_enable(true)
            .src_color_blend_factor(to_vk_blend_factor(blend.color.src_factor))
            .dst_color_blend_factor(to_vk_blend_factor(blend.color.dst_factor))
            .color_blend_op(to_vk_blend_op(blend.color.operation))
            .src_alpha_blend_factor(to_vk_blend_factor(blend.alpha.src_factor))
            .dst_alpha_blend_factor(to_vk_blend_factor(blend.alpha.dst_factor))
            .alpha_blend_op(to_vk_blend_op(blend.alpha.operation)),
    }
}

/// Converts an RHI vertex step mode to a Vulkan vertex input rate.
pub fn to_vk_input_rate(step: VertexStepMode) -> vk::VertexInputRate {
    match step {
        VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
        VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
    }
}

/// Whether depth bias must be enabled: either bias term is non-zero.
pub fn depth_bias_enabled(rasterizer: &RasterizerState) -> bool {
    rasterizer.depth_bias != 0.0 || rasterizer.slope_scaled_depth_bias != 0.0
}

/// Converts RHI PrimitiveTopology to Vulkan vk::PrimitiveTopology.
pub fn to_vk_topology(topo: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topo {
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_state_maps_factors_ops_and_mask() {
        let state = to_vk_color_blend_attachment(&ColorTargetState {
            format: korppi_rhi::TextureFormat::Rgba8Unorm,
            blend: Some(korppi_rhi::BlendState::ALPHA_BLENDING),
            write_mask: ColorWriteMask::RED | ColorWriteMask::ALPHA,
        });
        assert_eq!(state.blend_enable, vk::TRUE);
        assert_eq!(state.src_color_blend_factor, vk::BlendFactor::SRC_ALPHA);
        assert_eq!(
            state.dst_color_blend_factor,
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA
        );
        assert_eq!(state.src_alpha_blend_factor, vk::BlendFactor::ONE);
        assert_eq!(state.alpha_blend_op, vk::BlendOp::ADD);
        assert_eq!(
            state.color_write_mask,
            vk::ColorComponentFlags::R | vk::ColorComponentFlags::A
        );
    }

    #[test]
    fn slope_only_depth_bias_enables_bias() {
        let rasterizer = RasterizerState {
            slope_scaled_depth_bias: 1.5,
            ..RasterizerState::default()
        };
        assert!(depth_bias_enabled(&rasterizer));
        assert!(!depth_bias_enabled(&RasterizerState::default()));
    }

    #[test]
    fn aspects_follow_the_texture_format() {
        assert_eq!(
            barrier_aspect(TextureFormat::Rgba8Unorm),
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(
            barrier_aspect(TextureFormat::Depth32Float),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            barrier_aspect(TextureFormat::Depth24PlusStencil8),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            copy_aspect(TextureFormat::Depth24PlusStencil8),
            vk::ImageAspectFlags::DEPTH
        );
    }

    #[test]
    fn row_pitch_converts_bytes_to_texels() {
        assert_eq!(buffer_row_length_texels(0, TextureFormat::Rgba8Unorm), 0);
        assert_eq!(buffer_row_length_texels(16, TextureFormat::Rgba8Unorm), 4);
        assert_eq!(buffer_row_length_texels(64, TextureFormat::Rgba16Float), 8);
    }

    fn segments(
        ranges: &[vk::PushConstantRange],
        offset: u32,
        size: u32,
    ) -> Vec<(vk::ShaderStageFlags, u32, u32)> {
        let mut out = Vec::new();
        push_constant_segments(ranges, offset, size, |s, o, n| out.push((s, o, n)));
        out
    }

    fn range(stage_flags: vk::ShaderStageFlags, offset: u32, size: u32) -> vk::PushConstantRange {
        vk::PushConstantRange {
            stage_flags,
            offset,
            size,
        }
    }

    #[test]
    fn push_constant_segments_use_exact_stage_unions() {
        use vk::ShaderStageFlags as S;
        // One fragment range: a single segment with exactly FRAGMENT.
        assert_eq!(
            segments(&[range(S::FRAGMENT, 0, 16)], 0, 16),
            vec![(S::FRAGMENT, 0, 16)]
        );
        // Overlapping vertex and fragment ranges: both stages for the shared bytes.
        assert_eq!(
            segments(&[range(S::VERTEX, 0, 32), range(S::FRAGMENT, 0, 32)], 0, 32),
            vec![(S::VERTEX | S::FRAGMENT, 0, 32)]
        );
        // Disjoint ranges pushed together: split per stage.
        assert_eq!(
            segments(
                &[range(S::VERTEX, 0, 16), range(S::FRAGMENT, 16, 16)],
                0,
                32
            ),
            vec![(S::VERTEX, 0, 16), (S::FRAGMENT, 16, 16)]
        );
        // Partial overlap: VERTEX 0..24, FRAGMENT 8..32, push 4..28.
        assert_eq!(
            segments(&[range(S::VERTEX, 0, 24), range(S::FRAGMENT, 8, 24)], 4, 24),
            vec![
                (S::VERTEX, 4, 4),
                (S::VERTEX | S::FRAGMENT, 8, 16),
                (S::FRAGMENT, 24, 4)
            ]
        );
        // Uncovered bytes are skipped.
        assert_eq!(
            segments(&[range(S::VERTEX, 0, 8)], 0, 16),
            vec![(S::VERTEX, 0, 8)]
        );
    }

    #[test]
    fn sampler_info_enables_mipmapping_and_clamps_anisotropy() {
        let desc = SamplerDescriptor {
            mipmap_filter: FilterMode::Linear,
            max_anisotropy: 16,
            border_color: BorderColor::OpaqueWhite,
            ..SamplerDescriptor::default()
        };
        let info = sampler_create_info(&desc, true, 8.0, 2.0);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::LINEAR);
        assert_eq!(info.max_lod, vk::LOD_CLAMP_NONE);
        assert_eq!(info.anisotropy_enable, vk::TRUE);
        assert_eq!(info.max_anisotropy, 8.0);
        assert_eq!(info.border_color, vk::BorderColor::FLOAT_OPAQUE_WHITE);

        let unsupported = sampler_create_info(&desc, false, 16.0, 2.0);
        assert_eq!(unsupported.anisotropy_enable, vk::FALSE);
        assert_eq!(unsupported.max_anisotropy, 1.0);
    }

    #[test]
    fn host_stage_maps_to_vulkan_host_stage() {
        assert_eq!(
            to_vk_pipeline_stage_flags(PipelineStageFlags::HOST),
            vk::PipelineStageFlags2::HOST
        );
    }

    #[test]
    fn front_face_maps_one_to_one_under_negative_viewport_height() {
        assert_eq!(
            to_vk_front_face(FrontFace::CounterClockwise),
            vk::FrontFace::COUNTER_CLOCKWISE
        );
        assert_eq!(
            to_vk_front_face(FrontFace::Clockwise),
            vk::FrontFace::CLOCKWISE
        );
    }

    /// The spec's signed area of the R1 conformance triangle after the backend's viewport flip
    /// (64x64 target) is positive, i.e. front-facing under `COUNTER_CLOCKWISE`.
    #[test]
    fn flipped_ccw_triangle_has_positive_vulkan_area() {
        let ndc = [(-0.5_f32, -0.5_f32), (0.5, -0.5), (0.0, 0.5)];
        let (w, h) = (64.0_f32, 64.0_f32);
        // y_vk = height, h_vk = -height: y_f = h/2 * (1 - y_ndc).
        let fb: Vec<(f32, f32)> = ndc
            .iter()
            .map(|&(x, y)| (w / 2.0 * (x + 1.0), h / 2.0 * (1.0 - y)))
            .collect();
        let sum: f32 = (0..3)
            .map(|i| {
                let (x0, y0) = fb[i];
                let (x1, y1) = fb[(i + 1) % 3];
                x0 * y1 - x1 * y0
            })
            .sum();
        let area = -0.5 * sum;
        assert!(area > 0.0, "area {area}");
    }

    #[test]
    fn push_constant_ranges_give_each_stage_one_hull() {
        let range = |stages, offset, size| PushConstantRange {
            stages,
            offset,
            size,
        };
        let ranges = vk_push_constant_ranges(&[
            range(ShaderStages::VERTEX, 0, 16),
            range(ShaderStages::VERTEX_FRAGMENT, 16, 16),
            range(ShaderStages::VERTEX, 48, 16),
        ]);
        assert_eq!(ranges.len(), 2, "{ranges:?}");
        let vertex = ranges
            .iter()
            .find(|r| r.stage_flags == vk::ShaderStageFlags::VERTEX)
            .expect("vertex range");
        assert_eq!((vertex.offset, vertex.size), (0, 64));
        let fragment = ranges
            .iter()
            .find(|r| r.stage_flags == vk::ShaderStageFlags::FRAGMENT)
            .expect("fragment range");
        assert_eq!((fragment.offset, fragment.size), (16, 16));
        let shared = vk_push_constant_ranges(&[range(ShaderStages::VERTEX_FRAGMENT, 0, 32)]);
        assert_eq!(
            shared[0].stage_flags,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
        );
        assert_eq!(shared.len(), 1);
    }
}
