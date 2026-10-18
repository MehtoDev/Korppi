//! Recorded transfers between buffers and textures: copies, blits, mip generation and clears
//! (finding RHI-05).
//!
//! Layouts: a texture read by a transfer is in `ImageLayout::TransferSrc`, a texture written in
//! `ImageLayout::TransferDst` (for the subresources involved). `generate_mipmaps` expects every
//! level in `TransferDst`, level 0 holding the image, and leaves every level in `TransferSrc`.
//!
//! Regions address one mip level. For 1D/2D textures, arrays and cube maps, `origin[2]` must be
//! 0 and `extent[2]` 1, and layers are selected separately; for 3D textures, the z coordinates
//! select depth slices and the layer must be 0. Block-compressed regions start on block
//! boundaries and cover whole blocks, except where they end at the edge of the mip level.
//!
//! Buffer data is laid out row by row (rows of blocks for compressed formats), `buffer_row_pitch`
//! bytes apart (0: tightly packed), and slice by slice for 3D copies, each slice `extent[1]` rows
//! (rounded up to whole blocks).

use crate::descriptor::BufferTextureCopy;
use crate::sync::TextureSubresourceRange;
use crate::types::{FilterMode, TextureAspect, TextureDimension};
use crate::validation::TextureInfo;

/// The value `CommandBufferRecorder::clear_texture` writes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureClear {
    /// A colour for colour formats; integer formats take the values converted to integers, as
    /// `LoadOp::Clear` does.
    Color([f32; 4]),
    /// Depth, and stencil for formats that have it.
    DepthStencil {
        /// Depth value.
        depth: f32,
        /// Stencil value (ignored without a stencil component).
        stencil: u32,
    },
}

/// Copies a region of one texture into another of the same format and sample count.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureCopy {
    /// Source mip level.
    pub src_mip_level: u32,
    /// First source layer (0 for 3D textures).
    pub src_layer: u32,
    /// Source origin in texels (z: depth slice of 3D textures).
    pub src_origin: [u32; 3],
    /// Destination mip level.
    pub dst_mip_level: u32,
    /// First destination layer (0 for 3D textures).
    pub dst_layer: u32,
    /// Destination origin in texels.
    pub dst_origin: [u32; 3],
    /// Size of the region in texels.
    pub extent: [u32; 3],
    /// Layers copied.
    pub layer_count: u32,
}

/// Copies a 2D region of one texture into a region of another, scaling and converting
/// formats. Regions are `[x0, y0, x1, y1]`; a destination with `x1 < x0` or `y1 < y0` mirrors.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureBlit {
    /// Source mip level.
    pub src_mip_level: u32,
    /// First source layer, or depth slice of a 3D texture.
    pub src_layer: u32,
    /// Source rectangle.
    pub src_region: [u32; 4],
    /// Destination mip level.
    pub dst_mip_level: u32,
    /// First destination layer, or depth slice of a 3D texture.
    pub dst_layer: u32,
    /// Destination rectangle.
    pub dst_region: [u32; 4],
    /// Layers (or slices) blitted.
    pub layer_count: u32,
    /// Filter for scaling: `Linear` needs a filterable, non-integer, colour format.
    pub filter: FilterMode,
}

/// Width, height and depth of mip `mip` of `info` (depth 1 unless 3D).
pub fn mip_extent(info: &TextureInfo, mip: u32) -> [u32; 3] {
    let depth = if info.dimension == TextureDimension::D3 {
        (info.depth >> mip).max(1)
    } else {
        1
    };
    [
        (info.width >> mip).max(1),
        (info.height >> mip).max(1),
        depth,
    ]
}

/// Checks that `layer_count` layers from `layer`, and the box `origin`..`origin + extent` of
/// mip `mip`, lie within `info`, and that compressed regions are block aligned.
pub fn check_region(
    info: &TextureInfo,
    mip: u32,
    layer: u32,
    layer_count: u32,
    origin: [u32; 3],
    extent: [u32; 3],
) -> Result<(), String> {
    if mip >= info.mip_levels {
        return Err(format!(
            "mip level {mip} of a texture with {} levels",
            info.mip_levels
        ));
    }
    if layer_count == 0 || extent.contains(&0) {
        return Err("empty region".into());
    }
    let three_d = info.dimension == TextureDimension::D3;
    if three_d && (layer != 0 || layer_count != 1) {
        return Err("3D textures have one layer; select slices with z".into());
    }
    if !three_d {
        if layer
            .checked_add(layer_count)
            .is_none_or(|end| end > info.array_layers)
        {
            return Err(format!(
                "layers {layer}..{} of a texture with {} layers",
                layer.saturating_add(layer_count),
                info.array_layers
            ));
        }
        if origin[2] != 0 || extent[2] != 1 {
            return Err("only 3D textures have depth: origin z must be 0 and extent z 1".into());
        }
    }
    let size = mip_extent(info, mip);
    for axis in 0..3 {
        if origin[axis]
            .checked_add(extent[axis])
            .is_none_or(|end| end > size[axis])
        {
            return Err(format!(
                "region {origin:?} + {extent:?} exceeds mip {mip} ({size:?})"
            ));
        }
    }
    let (bw, bh) = info.format.block_dimensions();
    for (axis, block) in [(0, bw), (1, bh)] {
        let ends_at_edge = origin[axis] + extent[axis] == size[axis];
        if !origin[axis].is_multiple_of(block)
            || (!extent[axis].is_multiple_of(block) && !ends_at_edge)
        {
            return Err(format!(
                "{:?} regions must be aligned to {bw}x{bh} blocks",
                info.format
            ));
        }
    }
    Ok(())
}

/// Bytes of buffer a buffer/texture copy of `info` touches, from `copy.buffer_offset`.
pub fn buffer_copy_bytes(info: &TextureInfo, copy: &BufferTextureCopy) -> u64 {
    let format = info.format;
    let (_, bh) = format.block_dimensions();
    let row = format.row_bytes(copy.extent[0]);
    let pitch = if copy.buffer_row_pitch == 0 {
        row
    } else {
        u64::from(copy.buffer_row_pitch)
    };
    let rows = u64::from(copy.extent[1].div_ceil(bh)) * u64::from(copy.extent[2]);
    pitch * rows.saturating_sub(1) + row
}

/// Checks a buffer/texture copy against the texture: region, row pitch (whole blocks, at least
/// one row) and buffer offset alignment (a whole block; 4 bytes for depth/stencil formats).
/// `buffer_size`, if known, must hold the copied bytes.
pub fn check_buffer_copy(
    info: &TextureInfo,
    copy: &BufferTextureCopy,
    buffer_size: Option<u64>,
) -> Result<(), String> {
    check_region(
        info,
        copy.texture_mip_level,
        copy.texture_layer,
        1,
        copy.origin,
        copy.extent,
    )?;
    if info.sample_count.count() != 1 {
        return Err("multisampled textures cannot be copied to or from buffers".into());
    }
    let format = info.format;
    let block = format.block_bytes();
    let row = format.row_bytes(copy.extent[0]);
    let pitch = copy.buffer_row_pitch;
    if pitch != 0 && (!pitch.is_multiple_of(block) || u64::from(pitch) < row) {
        return Err(format!(
            "buffer_row_pitch {pitch} must be 0 or a multiple of the {block}-byte texel block \
             that is at least the copied row ({row} bytes)"
        ));
    }
    let alignment = if format.is_depth() || format.is_stencil() {
        4
    } else {
        u64::from(block)
    };
    if !copy.buffer_offset.is_multiple_of(alignment) {
        return Err(format!(
            "buffer_offset {} must be a multiple of {alignment}",
            copy.buffer_offset
        ));
    }
    if let Some(size) = buffer_size {
        let copy_bytes = buffer_copy_bytes(info, copy);
        match copy.buffer_offset.checked_add(copy_bytes) {
            Some(end) if end > size => {
                return Err(format!(
                    "the copy reaches byte {end} of a {size}-byte buffer"
                ));
            }
            None => {
                return Err(format!(
                    "the copy reaches past the end of a {size}-byte buffer"
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Checks a texture-to-texture copy: same format and sample count, both regions in bounds, and
/// no overlap when source and destination are the same texture.
pub fn check_texture_copy(
    src: &TextureInfo,
    dst: &TextureInfo,
    same_texture: bool,
    copy: &TextureCopy,
) -> Result<(), String> {
    if src.format != dst.format {
        return Err(format!(
            "formats differ ({:?} -> {:?}); copies need equal formats",
            src.format, dst.format
        ));
    }
    if src.sample_count != dst.sample_count {
        return Err("sample counts differ".into());
    }
    check_region(
        src,
        copy.src_mip_level,
        copy.src_layer,
        copy.layer_count,
        copy.src_origin,
        copy.extent,
    )
    .map_err(|e| format!("source: {e}"))?;
    check_region(
        dst,
        copy.dst_mip_level,
        copy.dst_layer,
        copy.layer_count,
        copy.dst_origin,
        copy.extent,
    )
    .map_err(|e| format!("destination: {e}"))?;
    if same_texture && copy.src_mip_level == copy.dst_mip_level {
        let layers_overlap = copy.src_layer < copy.dst_layer + copy.layer_count
            && copy.dst_layer < copy.src_layer + copy.layer_count;
        if layers_overlap {
            return Err("source and destination subresources overlap".into());
        }
    }
    Ok(())
}

/// Checks a blit: both rectangles in bounds; no compressed or multisampled textures; depth,
/// stencil and integer formats only between equal formats with `Nearest`; `Linear` needs a
/// filterable source (`src_filterable`).
pub fn check_blit(
    src: &TextureInfo,
    dst: &TextureInfo,
    same_texture: bool,
    blit: &TextureBlit,
    src_filterable: bool,
) -> Result<(), String> {
    for (what, info) in [("source", src), ("destination", dst)] {
        if info.format.is_compressed() {
            return Err(format!(
                "{what} is block-compressed; blits cannot read or write it"
            ));
        }
        if info.sample_count.count() != 1 {
            return Err(format!(
                "{what} is multisampled; resolve it in a pass instead"
            ));
        }
    }
    let special = |info: &TextureInfo| {
        info.format.is_depth() || info.format.is_stencil() || info.format.is_integer()
    };
    if (special(src) || special(dst))
        && (src.format != dst.format || blit.filter != FilterMode::Nearest)
    {
        return Err(
            "depth, stencil and integer formats blit only to the same format, with Nearest".into(),
        );
    }
    if blit.filter == FilterMode::Linear && !src_filterable {
        return Err(format!("{:?} cannot be filtered linearly", src.format));
    }
    let rect = |info: &TextureInfo, mip, layer, r: [u32; 4]| {
        let (x0, x1) = (r[0].min(r[2]), r[0].max(r[2]));
        let (y0, y1) = (r[1].min(r[3]), r[1].max(r[3]));
        // 3D slices are addressed like layers.
        let (layer, origin_z) = if info.dimension == TextureDimension::D3 {
            (0, layer)
        } else {
            (layer, 0)
        };
        let depth = if info.dimension == TextureDimension::D3 {
            blit.layer_count
        } else {
            1
        };
        let layer_count = if info.dimension == TextureDimension::D3 {
            1
        } else {
            blit.layer_count
        };
        check_region(
            info,
            mip,
            layer,
            layer_count,
            [x0, y0, origin_z],
            [x1 - x0, y1 - y0, depth],
        )
    };
    rect(src, blit.src_mip_level, blit.src_layer, blit.src_region)
        .map_err(|e| format!("source: {e}"))?;
    rect(dst, blit.dst_mip_level, blit.dst_layer, blit.dst_region)
        .map_err(|e| format!("destination: {e}"))?;
    if same_texture
        && blit.src_mip_level == blit.dst_mip_level
        && blit.src_layer < blit.dst_layer + blit.layer_count
        && blit.dst_layer < blit.src_layer + blit.layer_count
    {
        return Err("source and destination subresources overlap".into());
    }
    Ok(())
}

/// Checks that mipmaps of `info` can be generated: a filterable colour format the device can
/// blit (`blit_filterable`), single-sampled, not compressed or integer.
pub fn check_generate_mipmaps(info: &TextureInfo, blit_filterable: bool) -> Result<(), String> {
    let format = info.format;
    if format.is_compressed() || format.is_depth() || format.is_stencil() || format.is_integer() {
        return Err(format!(
            "{format:?} mipmaps cannot be generated (compressed, depth, stencil or integer)"
        ));
    }
    if info.sample_count.count() != 1 {
        return Err("multisampled textures have one mip level".into());
    }
    if !blit_filterable {
        return Err(format!(
            "{format:?} cannot be blitted with linear filtering on this device"
        ));
    }
    Ok(())
}

/// Checks a `clear_texture` of `range` with `value`: the range covers whole aspects of existing
/// subresources, the value matches the format, and the texture is neither compressed nor
/// multisampled (those are cleared as attachments).
pub fn check_clear_texture(
    info: &TextureInfo,
    range: &TextureSubresourceRange,
    value: &TextureClear,
) -> Result<(), String> {
    let format = info.format;
    if format.is_compressed() {
        return Err(format!("{format:?} textures cannot be cleared"));
    }
    if info.sample_count.count() != 1 {
        return Err("multisampled textures are cleared as attachments (LoadOp::Clear)".into());
    }
    if range.aspect != TextureAspect::All {
        return Err(format!(
            "clears cover every aspect (TextureAspect::All), not {:?}",
            range.aspect
        ));
    }
    if !range.fits(info.mip_levels, info.array_layers) {
        return Err(format!(
            "{range:?} is outside the texture's {} mips and {} layers",
            info.mip_levels, info.array_layers
        ));
    }
    match (value, format.is_depth()) {
        (TextureClear::Color(_), true) => Err(format!(
            "{format:?} is a depth format: clear it with TextureClear::DepthStencil"
        )),
        (TextureClear::DepthStencil { .. }, false) => Err(format!(
            "{format:?} is a colour format: clear it with TextureClear::Color"
        )),
        _ => Ok(()),
    }
}

/// Checks a buffer-to-buffer copy of `size` bytes from `src_offset` of a `src_size`-byte buffer
/// to `dst_offset` of a `dst_size`-byte buffer: a non-zero size, both ranges inside their
/// buffers, and no overlap when both are the same buffer (`same_buffer`). Vulkan leaves each of
/// these undefined (VUID-VkBufferCopy-size-01988, VUID-vkCmdCopyBuffer-srcOffset-00113 to
/// -size-00116, VUID-vkCmdCopyBuffer-pRegions-00117); OpenGL reports them as errors.
pub fn check_buffer_to_buffer_copy(
    src_size: u64,
    src_offset: u64,
    dst_size: u64,
    dst_offset: u64,
    size: u64,
    same_buffer: bool,
) -> Result<(), String> {
    if size == 0 {
        return Err("a copy of 0 bytes".into());
    }
    let range = |offset: u64, buffer_size: u64, role: &str| match offset.checked_add(size) {
        Some(end) if end <= buffer_size => Ok(end),
        _ => Err(format!(
            "{size} bytes at {offset} run past the end of the {buffer_size}-byte {role}"
        )),
    };
    let src_end = range(src_offset, src_size, "source")?;
    let dst_end = range(dst_offset, dst_size, "destination")?;
    if same_buffer && src_offset < dst_end && dst_offset < src_end {
        return Err(format!(
            "the source ({src_offset}..{src_end}) and destination ({dst_offset}..{dst_end}) \
             ranges of one buffer overlap"
        ));
    }
    Ok(())
}

/// Checks a `fill_buffer` of `size` bytes at `offset` into a buffer of `buffer_size` bytes:
/// both are non-zero multiples of 4 (the word size) within the buffer.
pub fn check_fill_buffer(buffer_size: u64, offset: u64, size: u64) -> Result<(), String> {
    if !offset.is_multiple_of(4) || !size.is_multiple_of(4) || size == 0 {
        return Err(format!(
            "offset {offset} and size {size} must be multiples of 4, and the size non-zero"
        ));
    }
    if offset.checked_add(size).is_none_or(|end| end > buffer_size) {
        return Err(format!(
            "{size} bytes at {offset} run past the end of the {buffer_size}-byte buffer"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SampleCount, TextureFormat, TextureUsage};

    #[test]
    fn buffer_to_buffer_copies_stay_inside_and_apart() {
        assert!(check_buffer_to_buffer_copy(16, 8, 16, 8, 8, false).is_ok());
        assert!(check_buffer_to_buffer_copy(16, 8, 16, 0, 16, false).is_err());
        assert!(check_buffer_to_buffer_copy(16, 0, 16, 8, 16, false).is_err());
        assert!(check_buffer_to_buffer_copy(16, u64::MAX, 16, 0, 2, false).is_err());
        assert!(check_buffer_to_buffer_copy(16, 0, 16, 0, 0, false).is_err());
        assert!(check_buffer_to_buffer_copy(64, 0, 64, 8, 16, true).is_err());
        assert!(check_buffer_to_buffer_copy(64, 8, 64, 0, 16, true).is_err());
        assert!(check_buffer_to_buffer_copy(64, 0, 64, 16, 16, true).is_ok());
    }

    fn info(
        dimension: TextureDimension,
        size: (u32, u32, u32),
        layers: u32,
        mips: u32,
        format: TextureFormat,
    ) -> TextureInfo {
        TextureInfo {
            width: size.0,
            height: size.1,
            format,
            dimension,
            mip_levels: mips,
            array_layers: layers,
            depth: size.2,
            sample_count: SampleCount::Count1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        }
    }

    #[test]
    fn regions_must_lie_within_the_mip_and_layers() {
        let array = info(
            TextureDimension::D2Array,
            (16, 8, 1),
            4,
            3,
            TextureFormat::Rgba8Unorm,
        );
        assert!(check_region(&array, 1, 3, 1, [0, 0, 0], [8, 4, 1]).is_ok());
        assert!(check_region(&array, 1, 0, 1, [1, 0, 0], [8, 4, 1]).is_err());
        assert!(check_region(&array, 3, 0, 1, [0, 0, 0], [1, 1, 1]).is_err());
        assert!(check_region(&array, 0, 3, 2, [0, 0, 0], [1, 1, 1]).is_err());
        assert!(check_region(&array, 0, 0, 1, [0, 0, 1], [1, 1, 1]).is_err());
        let volume = info(
            TextureDimension::D3,
            (8, 8, 8),
            1,
            2,
            TextureFormat::R8Unorm,
        );
        assert!(check_region(&volume, 1, 0, 1, [0, 0, 2], [4, 4, 2]).is_ok());
        assert!(check_region(&volume, 1, 0, 1, [0, 0, 3], [4, 4, 2]).is_err());
        assert!(check_region(&volume, 0, 1, 1, [0, 0, 0], [1, 1, 1]).is_err());
    }

    #[test]
    fn compressed_regions_cover_whole_blocks_except_at_the_edge() {
        let bc = info(
            TextureDimension::D2,
            (10, 10, 1),
            1,
            1,
            TextureFormat::Bc1RgbaUnorm,
        );
        assert!(check_region(&bc, 0, 0, 1, [4, 4, 0], [4, 4, 1]).is_ok());
        assert!(check_region(&bc, 0, 0, 1, [8, 8, 0], [2, 2, 1]).is_ok());
        assert!(check_region(&bc, 0, 0, 1, [2, 0, 0], [4, 4, 1]).is_err());
        assert!(check_region(&bc, 0, 0, 1, [0, 0, 0], [3, 4, 1]).is_err());
    }

    #[test]
    fn buffer_copies_check_pitch_offset_and_size() {
        let rgba = info(
            TextureDimension::D2,
            (8, 8, 1),
            1,
            1,
            TextureFormat::Rgba8Unorm,
        );
        let copy = |offset, pitch, extent: [u32; 3]| BufferTextureCopy {
            buffer_offset: offset,
            buffer_row_pitch: pitch,
            texture_mip_level: 0,
            texture_layer: 0,
            origin: [0, 0, 0],
            extent,
        };
        // 4 rows of 256 bytes, the last one 16 bytes long.
        assert_eq!(
            buffer_copy_bytes(&rgba, &copy(0, 256, [4, 4, 1])),
            3 * 256 + 16
        );
        assert!(check_buffer_copy(&rgba, &copy(0, 256, [4, 4, 1]), Some(784)).is_ok());
        assert!(check_buffer_copy(&rgba, &copy(0, 256, [4, 4, 1]), Some(783)).is_err());
        assert!(check_buffer_copy(&rgba, &copy(2, 0, [4, 4, 1]), None).is_err());
        assert!(check_buffer_copy(&rgba, &copy(0, 10, [2, 1, 1]), None).is_err());
        assert!(check_buffer_copy(&rgba, &copy(0, 8, [4, 1, 1]), None).is_err());
    }

    #[test]
    fn blits_reject_compressed_depth_mixes_and_unfilterable_linear() {
        let rgba = info(
            TextureDimension::D2,
            (8, 8, 1),
            1,
            2,
            TextureFormat::Rgba8Unorm,
        );
        let depth = info(
            TextureDimension::D2,
            (8, 8, 1),
            1,
            1,
            TextureFormat::Depth32Float,
        );
        let blit = |filter| TextureBlit {
            src_mip_level: 0,
            src_layer: 0,
            src_region: [0, 0, 8, 8],
            dst_mip_level: 1,
            dst_layer: 0,
            dst_region: [4, 0, 0, 4],
            layer_count: 1,
            filter,
        };
        assert!(check_blit(&rgba, &rgba, true, &blit(FilterMode::Linear), true).is_ok());
        assert!(check_blit(&rgba, &rgba, true, &blit(FilterMode::Linear), false).is_err());
        let to_depth = TextureBlit {
            dst_mip_level: 0,
            dst_region: [0, 0, 8, 8],
            ..blit(FilterMode::Nearest)
        };
        assert!(check_blit(&rgba, &depth, false, &to_depth, true).is_err());
        assert!(check_blit(&depth, &depth, false, &to_depth, true).is_ok());
        let overlapping = TextureBlit {
            dst_mip_level: 0,
            ..blit(FilterMode::Nearest)
        };
        assert!(check_blit(&rgba, &rgba, true, &overlapping, true).is_err());
    }
}
