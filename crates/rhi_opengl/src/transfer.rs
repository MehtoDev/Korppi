//! Recorded transfers on OpenGL (finding RHI-05, RGL-05): buffer/texture copies through pixel
//! pack/unpack buffers, texture copies with `glCopyImageSubData`, blits through framebuffers
//! and mip generation with `glGenerateTextureMipmap`.
//!
//! Buffer copies follow the RHI layout (`korppi_rhi::transfer`): `GL_[UN]PACK_ROW_LENGTH` is
//! the row pitch in texels; block-compressed formats also set the compressed block parameters
//! so that pitched rows of blocks work. Pack and unpack alignment are 1 (set at context
//! creation). Layers of arrays and cube maps, and slices of 3D textures, are the z coordinate.

use crate::gl_fns::GlExtraFns;
use crate::types::to_gl_texture_format;
use glow::HasContext;
use korppi_rhi::{BufferTextureCopy, TextureFormat};

const UNPACK_COMPRESSED_BLOCK_WIDTH: u32 = 0x9127;
const UNPACK_COMPRESSED_BLOCK_HEIGHT: u32 = 0x9128;
const UNPACK_COMPRESSED_BLOCK_SIZE: u32 = 0x912A;
const PACK_COMPRESSED_BLOCK_WIDTH: u32 = 0x912B;
const PACK_COMPRESSED_BLOCK_HEIGHT: u32 = 0x912C;
const PACK_COMPRESSED_BLOCK_SIZE: u32 = 0x912E;

/// A texture taking part in a transfer.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TransferTexture {
    /// The GL texture.
    pub texture: glow::Texture,
    /// Its target (`TEXTURE_2D`, `TEXTURE_2D_ARRAY`, `TEXTURE_3D`, ...).
    pub target: u32,
    /// Its format.
    pub format: TextureFormat,
}

/// The z offset and depth of a buffer copy's region: the layer of arrays and cube maps, the
/// slices of 3D textures.
fn region_z(target: u32, copy: &BufferTextureCopy) -> (i32, i32) {
    if target == glow::TEXTURE_3D {
        (copy.origin[2] as i32, copy.extent[2] as i32)
    } else {
        (copy.texture_layer as i32, 1)
    }
}

/// Bytes a buffer copy's region holds when tightly packed (the compressed image size GL
/// expects).
fn tight_bytes(format: TextureFormat, copy: &BufferTextureCopy) -> i32 {
    let (_, bh) = format.block_dimensions();
    let rows = u64::from(copy.extent[1].div_ceil(bh)) * u64::from(copy.extent[2]);
    (format.row_bytes(copy.extent[0]) * rows) as i32
}

/// Sets the pack (`pack`) or unpack pixel storage for a copy with `row_pitch` bytes per row.
///
/// # Safety
/// The context must be current.
unsafe fn set_pixel_store(gl: &glow::Context, pack: bool, format: TextureFormat, row_pitch: u32) {
    let (bw, bh) = format.block_dimensions();
    let row_length = row_pitch / format.block_bytes() * bw;
    let (row, block_w, block_h, block_size) = if pack {
        (
            glow::PACK_ROW_LENGTH,
            PACK_COMPRESSED_BLOCK_WIDTH,
            PACK_COMPRESSED_BLOCK_HEIGHT,
            PACK_COMPRESSED_BLOCK_SIZE,
        )
    } else {
        (
            glow::UNPACK_ROW_LENGTH,
            UNPACK_COMPRESSED_BLOCK_WIDTH,
            UNPACK_COMPRESSED_BLOCK_HEIGHT,
            UNPACK_COMPRESSED_BLOCK_SIZE,
        )
    };
    // SAFETY: forwarded from the caller.
    unsafe {
        gl.pixel_store_i32(row, row_length as i32);
        if format.is_compressed() {
            gl.pixel_store_i32(block_w, bw as i32);
            gl.pixel_store_i32(block_h, bh as i32);
            gl.pixel_store_i32(block_size, format.block_bytes() as i32);
        }
    }
}

/// Restores the default pixel storage after [`set_pixel_store`].
///
/// # Safety
/// The context must be current.
unsafe fn reset_pixel_store(gl: &glow::Context, pack: bool) {
    let names: [u32; 4] = if pack {
        [
            glow::PACK_ROW_LENGTH,
            PACK_COMPRESSED_BLOCK_WIDTH,
            PACK_COMPRESSED_BLOCK_HEIGHT,
            PACK_COMPRESSED_BLOCK_SIZE,
        ]
    } else {
        [
            glow::UNPACK_ROW_LENGTH,
            UNPACK_COMPRESSED_BLOCK_WIDTH,
            UNPACK_COMPRESSED_BLOCK_HEIGHT,
            UNPACK_COMPRESSED_BLOCK_SIZE,
        ]
    };
    for name in names {
        // SAFETY: forwarded from the caller.
        unsafe { gl.pixel_store_i32(name, 0) };
    }
}

/// Copies buffer data into a texture region.
///
/// # Safety
/// The context must be current; `texture` and `buffer` must be live objects of it.
pub(crate) unsafe fn buffer_to_texture(
    gl: &glow::Context,
    extra: &GlExtraFns,
    texture: TransferTexture,
    buffer: glow::Buffer,
    copy: &BufferTextureCopy,
) -> Result<(), &'static str> {
    let format = texture.format;
    let gl_fmt = to_gl_texture_format(format);
    let name = texture.texture.0.get();
    let (z, depth) = region_z(texture.target, copy);
    let [x, y, _] = copy.origin.map(|v| v as i32);
    let [w, h, _] = copy.extent.map(|v| v as i32);
    let level = copy.texture_mip_level as i32;
    // The source is the pixel-unpack buffer: `offset` is a byte offset into it.
    let offset = copy.buffer_offset as usize as *const std::ffi::c_void;
    // SAFETY: forwarded from the caller; the unpack buffer is bound, so every pointer below is
    // an offset into it, and GL reads only within the validated region.
    unsafe {
        gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(buffer));
        set_pixel_store(gl, false, format, copy.buffer_row_pitch);
        let result = if format.is_compressed() {
            let size = tight_bytes(format, copy);
            if texture.target == glow::TEXTURE_2D {
                extra
                    .compressed_texture_sub_image_2d
                    .map(|f| {
                        f(
                            name,
                            level,
                            x,
                            y,
                            w,
                            h,
                            gl_fmt.internal_format,
                            size,
                            offset,
                        )
                    })
                    .ok_or("glCompressedTextureSubImage2D")
            } else {
                extra
                    .compressed_texture_sub_image_3d
                    .map(|f| {
                        f(
                            name,
                            level,
                            x,
                            y,
                            z,
                            w,
                            h,
                            depth,
                            gl_fmt.internal_format,
                            size,
                            offset,
                        )
                    })
                    .ok_or("glCompressedTextureSubImage3D")
            }
        } else {
            match texture.target {
                glow::TEXTURE_1D => extra
                    .texture_sub_image_1d
                    .map(|f| f(name, level, x, w, gl_fmt.format, gl_fmt.data_type, offset))
                    .ok_or("glTextureSubImage1D"),
                glow::TEXTURE_2D => {
                    gl.texture_sub_image_2d(
                        texture.texture,
                        level,
                        x,
                        y,
                        w,
                        h,
                        gl_fmt.format,
                        gl_fmt.data_type,
                        glow::PixelUnpackData::BufferOffset(copy.buffer_offset as u32),
                    );
                    Ok(())
                }
                _ => {
                    gl.texture_sub_image_3d(
                        texture.texture,
                        level,
                        x,
                        y,
                        z,
                        w,
                        h,
                        depth,
                        gl_fmt.format,
                        gl_fmt.data_type,
                        glow::PixelUnpackData::BufferOffset(copy.buffer_offset as u32),
                    );
                    Ok(())
                }
            }
        };
        reset_pixel_store(gl, false);
        gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
        result
    }
}

/// Copies a texture region into buffer memory from `copy.buffer_offset` on.
///
/// # Safety
/// The context must be current; `texture` and `buffer` must be live objects of it and
/// `buffer_size` the buffer's size.
pub(crate) unsafe fn texture_to_buffer(
    gl: &glow::Context,
    extra: &GlExtraFns,
    texture: TransferTexture,
    buffer: glow::Buffer,
    buffer_size: u64,
    copy: &BufferTextureCopy,
) -> Result<(), &'static str> {
    let format = texture.format;
    let gl_fmt = to_gl_texture_format(format);
    let name = texture.texture.0.get();
    let (z, depth) = region_z(texture.target, copy);
    let [x, y, _] = copy.origin.map(|v| v as i32);
    let [w, h, _] = copy.extent.map(|v| v as i32);
    let level = copy.texture_mip_level as i32;
    let available =
        i32::try_from(buffer_size.saturating_sub(copy.buffer_offset)).unwrap_or(i32::MAX);
    // The destination is the pixel-pack buffer: `offset` is a byte offset into it.
    let offset = copy.buffer_offset as usize as *mut std::ffi::c_void;
    // SAFETY: forwarded from the caller; the pack buffer is bound, so the pointer is an offset
    // into it and GL writes at most `available` bytes from there.
    unsafe {
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(buffer));
        set_pixel_store(gl, true, format, copy.buffer_row_pitch);
        let result = if format.is_compressed() {
            extra
                .get_compressed_texture_sub_image
                .map(|f| f(name, level, x, y, z, w, h, depth, available, offset))
                .ok_or("glGetCompressedTextureSubImage")
        } else {
            extra
                .get_texture_sub_image
                .map(|f| {
                    f(
                        name,
                        level,
                        x,
                        y,
                        z,
                        w,
                        h,
                        depth,
                        gl_fmt.format,
                        gl_fmt.data_type,
                        available,
                        offset,
                    )
                })
                .ok_or("glGetTextureSubImage")
        };
        reset_pixel_store(gl, true);
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        result
    }
}

/// Fills `size` bytes of `buffer` at `offset` with the word `value` (`glClearNamedBufferSubData`
/// with `R32UI` data). Returns the missing entry point's name if the driver lacks it.
///
/// # Safety
/// The context must be current on this thread; `buffer` is live, and the range is word-aligned
/// and within it.
pub(crate) unsafe fn fill_buffer(
    extra: &GlExtraFns,
    buffer: glow::Buffer,
    offset: u64,
    size: u64,
    value: u32,
) -> Result<(), &'static str> {
    let clear = extra
        .clear_named_buffer_sub_data
        .ok_or("glClearNamedBufferSubData")?;
    // SAFETY: forwarded from the caller; `value` outlives the call and is the one R32UI
    // texel GL reads.
    unsafe {
        clear(
            buffer.0.get(),
            glow::R32UI,
            offset as isize,
            size as isize,
            glow::RED_INTEGER,
            glow::UNSIGNED_INT,
            (&value as *const u32).cast(),
        );
    }
    Ok(())
}

/// Clears the subresources in `range` of `texture` to `value` (`glClearTexSubImage` per mip
/// level; layers of arrays and cube maps are z offsets).
///
/// # Safety
/// The context must be current on this thread; `texture` is live and described by `info`, and
/// the range and value were checked with `korppi_rhi::transfer::check_clear_texture`.
pub(crate) unsafe fn clear_texture(
    extra: &GlExtraFns,
    texture: glow::Texture,
    info: &korppi_rhi::TextureInfo,
    range: &korppi_rhi::TextureSubresourceRange,
    value: korppi_rhi::TextureClear,
) -> Result<(), &'static str> {
    use korppi_rhi::{TextureClear, TextureDimension};
    let clear = extra.clear_tex_sub_image.ok_or("glClearTexSubImage")?;
    // Client data in a format GL converts to the texture's.
    let words: [u32; 4];
    let (format, ty, data): (u32, u32, *const std::ffi::c_void) = match value {
        TextureClear::Color(color) if info.format.is_integer() => {
            words = color.map(|c| c as u32);
            (
                glow::RGBA_INTEGER,
                glow::UNSIGNED_INT,
                words.as_ptr().cast(),
            )
        }
        TextureClear::Color(color) => {
            words = color.map(f32::to_bits);
            (glow::RGBA, glow::FLOAT, words.as_ptr().cast())
        }
        // FLOAT_32_UNSIGNED_INT_24_8_REV: the depth as a float, then the stencil in the low
        // 8 bits of a word.
        TextureClear::DepthStencil { depth, stencil } if info.format.is_stencil() => {
            words = [depth.to_bits(), stencil & 0xFF, 0, 0];
            (
                glow::DEPTH_STENCIL,
                glow::FLOAT_32_UNSIGNED_INT_24_8_REV,
                words.as_ptr().cast(),
            )
        }
        TextureClear::DepthStencil { depth, .. } => {
            words = [depth.to_bits(), 0, 0, 0];
            (glow::DEPTH_COMPONENT, glow::FLOAT, words.as_ptr().cast())
        }
    };
    let layers = range.layers(info.array_layers);
    let (first, count) = (layers.start as i32, layers.len() as i32);
    for mip in range.mips(info.mip_levels) {
        let [w, h, d] = korppi_rhi::transfer::mip_extent(info, mip).map(|n| n as i32);
        let (origin, extent) = match info.dimension {
            TextureDimension::D1 => ([0, 0, 0], [w, 1, 1]),
            TextureDimension::D2 => ([0, 0, 0], [w, h, 1]),
            TextureDimension::D3 => ([0, 0, 0], [w, h, d]),
            TextureDimension::D2Array | TextureDimension::Cube | TextureDimension::CubeArray => {
                ([0, 0, first], [w, h, count])
            }
        };
        // SAFETY: forwarded from the caller; the region lies within mip `mip`, and `data`
        // points at one texel of `format`/`ty`, alive for the call.
        unsafe {
            clear(
                texture.0.get(),
                mip as i32,
                origin[0],
                origin[1],
                origin[2],
                extent[0],
                extent[1],
                extent[2],
                format,
                ty,
                data,
            );
        }
    }
    Ok(())
}
