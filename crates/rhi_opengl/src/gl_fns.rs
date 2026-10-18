//! OpenGL entry points the backend needs that `glow` does not wrap (DSA texture transfers and
//! query creation, texture views, multisample storage, query parameters, polygon-offset
//! clamping).
//!
//! They are loaded once per context through the platform's `GetProcAddress`; a missing function
//! is `None` and its user reports `RhiError::Unsupported` (every one is core in OpenGL 4.5,
//! except `glPolygonOffsetClamp`, which is 4.6 / `ARB_polygon_offset_clamp`).

use std::ffi::c_void;

type CompressedTextureSubImage2d = unsafe extern "system" fn(
    texture: u32,
    level: i32,
    xoffset: i32,
    yoffset: i32,
    width: i32,
    height: i32,
    format: u32,
    image_size: i32,
    data: *const c_void,
);
type CompressedTextureSubImage3d = unsafe extern "system" fn(
    texture: u32,
    level: i32,
    xoffset: i32,
    yoffset: i32,
    zoffset: i32,
    width: i32,
    height: i32,
    depth: i32,
    format: u32,
    image_size: i32,
    data: *const c_void,
);
type GetTextureSubImage = unsafe extern "system" fn(
    texture: u32,
    level: i32,
    xoffset: i32,
    yoffset: i32,
    zoffset: i32,
    width: i32,
    height: i32,
    depth: i32,
    format: u32,
    ty: u32,
    buf_size: i32,
    pixels: *mut c_void,
);
type GetCompressedTextureSubImage = unsafe extern "system" fn(
    texture: u32,
    level: i32,
    xoffset: i32,
    yoffset: i32,
    zoffset: i32,
    width: i32,
    height: i32,
    depth: i32,
    buf_size: i32,
    pixels: *mut c_void,
);
type GetQueryiv = unsafe extern "system" fn(target: u32, pname: u32, params: *mut i32);
type CreateQueries = unsafe extern "system" fn(target: u32, n: i32, ids: *mut u32);
type TextureView = unsafe extern "system" fn(
    texture: u32,
    target: u32,
    orig_texture: u32,
    internal_format: u32,
    min_level: u32,
    num_levels: u32,
    min_layer: u32,
    num_layers: u32,
);
type TextureStorage2dMultisample = unsafe extern "system" fn(
    texture: u32,
    samples: i32,
    internal_format: u32,
    width: i32,
    height: i32,
    fixed_sample_locations: u8,
);
type TextureStorage3dMultisample = unsafe extern "system" fn(
    texture: u32,
    samples: i32,
    internal_format: u32,
    width: i32,
    height: i32,
    depth: i32,
    fixed_sample_locations: u8,
);
type PolygonOffsetClamp = unsafe extern "system" fn(factor: f32, units: f32, clamp: f32);
type ClipControl = unsafe extern "system" fn(origin: u32, depth: u32);
type ClearTexSubImage = unsafe extern "system" fn(
    texture: u32,
    level: i32,
    xoffset: i32,
    yoffset: i32,
    zoffset: i32,
    width: i32,
    height: i32,
    depth: i32,
    format: u32,
    ty: u32,
    data: *const c_void,
);
type ClearNamedBufferSubData = unsafe extern "system" fn(
    buffer: u32,
    internal_format: u32,
    offset: isize,
    size: isize,
    format: u32,
    ty: u32,
    data: *const c_void,
);
type MultiDrawArraysIndirect =
    unsafe extern "system" fn(mode: u32, indirect: *const c_void, draw_count: i32, stride: i32);
type MultiDrawElementsIndirect = unsafe extern "system" fn(
    mode: u32,
    ty: u32,
    indirect: *const c_void,
    draw_count: i32,
    stride: i32,
);
type TextureStorage1d =
    unsafe extern "system" fn(texture: u32, levels: i32, internal_format: u32, width: i32);
type TextureSubImage1d = unsafe extern "system" fn(
    texture: u32,
    level: i32,
    xoffset: i32,
    width: i32,
    format: u32,
    ty: u32,
    pixels: *const c_void,
);

/// Function pointers of the current context (valid while it lives, on its thread).
#[derive(Clone, Copy, Default)]
pub struct GlExtraFns {
    /// `glCompressedTextureSubImage2D`.
    pub compressed_texture_sub_image_2d: Option<CompressedTextureSubImage2d>,
    /// `glCompressedTextureSubImage3D`.
    pub compressed_texture_sub_image_3d: Option<CompressedTextureSubImage3d>,
    /// `glGetTextureSubImage`.
    pub get_texture_sub_image: Option<GetTextureSubImage>,
    /// `glGetCompressedTextureSubImage`.
    pub get_compressed_texture_sub_image: Option<GetCompressedTextureSubImage>,
    /// `glGetQueryiv`.
    pub get_queryiv: Option<GetQueryiv>,
    /// `glCreateQueries`.
    pub create_queries: Option<CreateQueries>,
    /// `glTextureView`.
    pub texture_view: Option<TextureView>,
    /// `glTextureStorage2DMultisample`.
    pub texture_storage_2d_multisample: Option<TextureStorage2dMultisample>,
    /// `glTextureStorage3DMultisample`.
    pub texture_storage_3d_multisample: Option<TextureStorage3dMultisample>,
    /// `glPolygonOffsetClamp` (or the `EXT` variant).
    pub polygon_offset_clamp: Option<PolygonOffsetClamp>,
    /// `glClipControl`.
    pub clip_control: Option<ClipControl>,
    /// `glClearTexSubImage`.
    pub clear_tex_sub_image: Option<ClearTexSubImage>,
    /// `glClearNamedBufferSubData`.
    pub clear_named_buffer_sub_data: Option<ClearNamedBufferSubData>,
    /// `glTextureStorage1D`.
    pub texture_storage_1d: Option<TextureStorage1d>,
    /// `glTextureSubImage1D`.
    pub texture_sub_image_1d: Option<TextureSubImage1d>,
    /// `glMultiDrawArraysIndirect`.
    pub multi_draw_arrays_indirect: Option<MultiDrawArraysIndirect>,
    /// `glMultiDrawElementsIndirect`.
    pub multi_draw_elements_indirect: Option<MultiDrawElementsIndirect>,
}

impl GlExtraFns {
    /// Loads the functions with `load`, which returns a function's address or null.
    ///
    /// # Safety
    /// `load` must return addresses of the current context's functions with the C signatures
    /// declared above (as the platform's `GetProcAddress` does for these names).
    pub unsafe fn load(load: impl Fn(&str) -> *const c_void) -> Self {
        // SAFETY: the caller guarantees each non-null address is the named function, whose C
        // signature matches the declared type; `transmute_copy` reinterprets the pointer.
        unsafe fn get<T: Copy>(load: &impl Fn(&str) -> *const c_void, names: &[&str]) -> Option<T> {
            names.iter().find_map(|name| {
                let address = load(name);
                // SAFETY: see above.
                (!address.is_null()).then(|| unsafe { std::mem::transmute_copy(&address) })
            })
        }
        // SAFETY: forwarded from the caller.
        unsafe {
            Self {
                compressed_texture_sub_image_2d: get(&load, &["glCompressedTextureSubImage2D"]),
                compressed_texture_sub_image_3d: get(&load, &["glCompressedTextureSubImage3D"]),
                get_texture_sub_image: get(&load, &["glGetTextureSubImage"]),
                get_compressed_texture_sub_image: get(&load, &["glGetCompressedTextureSubImage"]),
                get_queryiv: get(&load, &["glGetQueryiv"]),
                create_queries: get(&load, &["glCreateQueries"]),
                texture_view: get(&load, &["glTextureView"]),
                texture_storage_2d_multisample: get(&load, &["glTextureStorage2DMultisample"]),
                texture_storage_3d_multisample: get(&load, &["glTextureStorage3DMultisample"]),
                polygon_offset_clamp: get(
                    &load,
                    &[
                        "glPolygonOffsetClamp",
                        "glPolygonOffsetClampARB",
                        "glPolygonOffsetClampEXT",
                    ],
                ),
                clip_control: get(&load, &["glClipControl"]),
                clear_tex_sub_image: get(&load, &["glClearTexSubImage"]),
                clear_named_buffer_sub_data: get(&load, &["glClearNamedBufferSubData"]),
                texture_storage_1d: get(&load, &["glTextureStorage1D"]),
                texture_sub_image_1d: get(&load, &["glTextureSubImage1D"]),
                multi_draw_arrays_indirect: get(&load, &["glMultiDrawArraysIndirect"]),
                multi_draw_elements_indirect: get(&load, &["glMultiDrawElementsIndirect"]),
            }
        }
    }
}
