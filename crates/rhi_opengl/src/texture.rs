//! Textures and texture views on the render thread (FIX-1.4; findings RHI-05, RGL-13).
//!
//! Each dimension maps to its texture target with immutable storage; multisampled textures use
//! the multisample targets. Uploads and readbacks go through the DSA calls that take a layer
//! range (`glTextureSubImage3D`, `glGetTextureSubImage`), which treat cube maps as six layers.
//! Views are separate textures created with `glTextureView`.

use crate::device::GlThread;
use crate::resource::{GlTexture, GlTextureView};
use crate::types::to_gl_texture_format;
use glow::{HasContext, PixelUnpackData};
use korppi_rhi::lock;
use korppi_rhi::{
    BoundResource, RhiError, SampleCount, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureHandle, TextureInfo, TextureViewDescriptor, TextureViewHandle,
};

/// The texture target of a texture (or view) dimension with `samples` samples.
pub(crate) fn texture_target(dimension: TextureDimension, samples: SampleCount) -> u32 {
    let multisampled = samples != SampleCount::Count1;
    match dimension {
        TextureDimension::D1 => glow::TEXTURE_1D,
        TextureDimension::D2 if multisampled => glow::TEXTURE_2D_MULTISAMPLE,
        TextureDimension::D2 => glow::TEXTURE_2D,
        TextureDimension::D2Array if multisampled => glow::TEXTURE_2D_MULTISAMPLE_ARRAY,
        TextureDimension::D2Array => glow::TEXTURE_2D_ARRAY,
        TextureDimension::Cube => glow::TEXTURE_CUBE_MAP,
        TextureDimension::CubeArray => glow::TEXTURE_CUBE_MAP_ARRAY,
        TextureDimension::D3 => glow::TEXTURE_3D,
    }
}

fn unavailable(name: &str) -> RhiError {
    RhiError::Unsupported(format!("{name} is unavailable"))
}

impl GlThread {
    /// Creates a texture; `levels` mip levels of `initial_data` (validated by
    /// `TextureDescriptor::validate_initial_data`) are uploaded.
    pub(crate) fn create_texture(
        &mut self,
        desc: &TextureDescriptor,
        initial_data: Option<(&[u8], u32)>,
    ) -> Result<TextureHandle, RhiError> {
        let gl = &self.context.gl;
        let extra = self.context.extra;
        let gl_fmt = to_gl_texture_format(desc.format);
        let target = texture_target(desc.dimension, desc.sample_count);
        let (width, height) = (desc.width as i32, desc.height as i32);
        let levels = desc.mip_levels as i32;
        let internal = gl_fmt.internal_format;
        // SAFETY: the context is current on the render thread; the storage matches the
        // validated descriptor, and every upload covers exactly one validated level of
        // `initial_data` (GL_UNPACK_ALIGNMENT is 1).
        let texture = unsafe {
            let texture = gl
                .create_named_texture(target)
                .map_err(|e| RhiError::OutOfMemory(format!("glCreateTextures failed: {e}")))?;
            let name = texture.0.get();
            let samples = desc.sample_count.count() as i32;
            let storage = match target {
                glow::TEXTURE_1D => extra
                    .texture_storage_1d
                    .map(|f| f(name, levels, internal, width))
                    .ok_or_else(|| unavailable("glTextureStorage1D")),
                glow::TEXTURE_2D | glow::TEXTURE_CUBE_MAP => {
                    gl.texture_storage_2d(texture, levels, internal, width, height);
                    Ok(())
                }
                glow::TEXTURE_2D_ARRAY | glow::TEXTURE_CUBE_MAP_ARRAY => {
                    gl.texture_storage_3d(
                        texture,
                        levels,
                        internal,
                        width,
                        height,
                        desc.array_layers as i32,
                    );
                    Ok(())
                }
                glow::TEXTURE_3D => {
                    gl.texture_storage_3d(
                        texture,
                        levels,
                        internal,
                        width,
                        height,
                        desc.depth as i32,
                    );
                    Ok(())
                }
                glow::TEXTURE_2D_MULTISAMPLE => extra
                    .texture_storage_2d_multisample
                    .map(|f| f(name, samples, internal, width, height, glow::TRUE))
                    .ok_or_else(|| unavailable("glTextureStorage2DMultisample")),
                _ => extra
                    .texture_storage_3d_multisample
                    .map(|f| {
                        f(
                            name,
                            samples,
                            internal,
                            width,
                            height,
                            desc.array_layers as i32,
                            glow::TRUE,
                        )
                    })
                    .ok_or_else(|| unavailable("glTextureStorage3DMultisample")),
            };
            if let Err(error) = storage {
                gl.delete_texture(texture);
                return Err(error);
            }
            if let Some((data, data_levels)) = initial_data {
                let mut offset = 0usize;
                for mip in 0..data_levels {
                    let (w, h, d) = desc.mip_extent(mip);
                    let size = desc.level_size_bytes(mip) as usize;
                    let level = &data[offset..offset + size];
                    offset += size;
                    // Layers (cube faces) or slices of the level.
                    let layers = if target == glow::TEXTURE_3D {
                        d
                    } else {
                        desc.array_layers
                    } as i32;
                    let uploaded = if desc.format.is_compressed() {
                        if target == glow::TEXTURE_2D {
                            extra.compressed_texture_sub_image_2d.map(|f| {
                                f(
                                    name,
                                    mip as i32,
                                    0,
                                    0,
                                    w as i32,
                                    h as i32,
                                    internal,
                                    size as i32,
                                    level.as_ptr().cast(),
                                )
                            })
                        } else {
                            extra.compressed_texture_sub_image_3d.map(|f| {
                                f(
                                    name,
                                    mip as i32,
                                    0,
                                    0,
                                    0,
                                    w as i32,
                                    h as i32,
                                    layers,
                                    internal,
                                    size as i32,
                                    level.as_ptr().cast(),
                                )
                            })
                        }
                        .ok_or_else(|| unavailable("glCompressedTextureSubImage"))
                    } else {
                        match target {
                            glow::TEXTURE_1D => extra
                                .texture_sub_image_1d
                                .map(|f| {
                                    f(
                                        name,
                                        mip as i32,
                                        0,
                                        w as i32,
                                        gl_fmt.format,
                                        gl_fmt.data_type,
                                        level.as_ptr().cast(),
                                    )
                                })
                                .ok_or_else(|| unavailable("glTextureSubImage1D")),
                            glow::TEXTURE_2D => {
                                gl.texture_sub_image_2d(
                                    texture,
                                    mip as i32,
                                    0,
                                    0,
                                    w as i32,
                                    h as i32,
                                    gl_fmt.format,
                                    gl_fmt.data_type,
                                    PixelUnpackData::Slice(Some(level)),
                                );
                                Ok(())
                            }
                            _ => {
                                gl.texture_sub_image_3d(
                                    texture,
                                    mip as i32,
                                    0,
                                    0,
                                    0,
                                    w as i32,
                                    h as i32,
                                    layers,
                                    gl_fmt.format,
                                    gl_fmt.data_type,
                                    PixelUnpackData::Slice(Some(level)),
                                );
                                Ok(())
                            }
                        }
                    };
                    if let Err(error) = uploaded {
                        gl.delete_texture(texture);
                        return Err(error);
                    }
                }
            }
            texture
        };
        self.label_object(glow::TEXTURE, texture.0.get(), desc.label);
        let (id, generation) = lock(&self.resources.textures).insert(GlTexture {
            texture,
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
            mip_levels: desc.mip_levels,
            format: desc.format,
            swapchain: None,
            target,
            info: TextureInfo::from_descriptor(desc),
            views: Vec::new(),
        });
        Ok(TextureHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    pub(crate) fn destroy_texture(&mut self, handle: TextureHandle) -> Result<(), RhiError> {
        let removed = {
            let mut textures = lock(&self.resources.textures);
            match textures.get(handle) {
                Some((texture, _)) if texture.swapchain.is_some() => {
                    return Err(RhiError::ValidationError(format!(
                        "texture {handle:?} belongs to a swapchain and is destroyed with it"
                    )))
                }
                _ => textures.remove(handle),
            }
        };
        let (texture, _) = removed.ok_or_else(|| {
            RhiError::InvalidHandle(format!("Texture handle {handle:?} not found"))
        })?;
        // Views of the texture die with it, and so do bind groups using either.
        let mut destroyed = vec![BoundResource::Texture(handle)];
        destroyed.extend(texture.views.iter().map(|&view| BoundResource::View(view)));
        self.resources.invalidate_bind_groups(&destroyed);
        for &view_handle in &texture.views {
            let view = lock(&self.resources.texture_views).remove(view_handle);
            if let Some((view, _)) = view {
                self.delete_texture_object(None, view.texture);
            }
        }
        self.delete_texture_object(Some(handle.id()), texture.texture);
        Ok(())
    }

    pub(crate) fn read_texture(
        &mut self,
        handle: TextureHandle,
        mip: u32,
    ) -> Result<Vec<u8>, RhiError> {
        let (texture, info, target) = {
            let textures = lock(&self.resources.textures);
            let (tex, _) = textures.get(handle).ok_or_else(|| {
                RhiError::InvalidHandle(format!("Texture handle {handle:?} not found"))
            })?;
            (tex.texture, tex.info, tex.target)
        };
        if info.format == TextureFormat::Depth24PlusStencil8 {
            return Err(RhiError::Unsupported(
                "Depth24PlusStencil8 has a device-specific depth layout and cannot be read back"
                    .into(),
            ));
        }
        if info.sample_count != SampleCount::Count1 {
            return Err(RhiError::Unsupported(
                "multisampled textures cannot be read back; resolve them first".into(),
            ));
        }
        if mip >= info.mip_levels {
            return Err(RhiError::ValidationError(format!(
                "mip {mip} of a texture with {} mips",
                info.mip_levels
            )));
        }
        let gl_fmt = to_gl_texture_format(info.format);
        let width = (info.width >> mip).max(1);
        let height = (info.height >> mip).max(1);
        let depth = if target == glow::TEXTURE_3D {
            (info.depth >> mip).max(1)
        } else {
            info.array_layers
        };
        // Whole blocks for compressed formats; the depth aspect for depth/stencil formats; every
        // layer or slice.
        let mut output =
            vec![0u8; (info.format.image_bytes(width, height) * u64::from(depth)) as usize];
        let size = i32::try_from(output.len())
            .map_err(|_| RhiError::Unsupported("texture level larger than 2 GiB".into()))?;
        let name = texture.0.get();
        let extra = self.context.extra;
        // `glGetTextureSubImage` reads any format and layer range without a framebuffer and
        // leaves no binding changed.
        // SAFETY: the context is current on the render thread; `output` holds exactly the
        // requested range, tightly packed (GL_PACK_ALIGNMENT is 1), and the call is told its
        // size so it cannot write past it.
        unsafe {
            if info.format.is_compressed() {
                let get = extra
                    .get_compressed_texture_sub_image
                    .ok_or_else(|| unavailable("glGetCompressedTextureSubImage"))?;
                get(
                    name,
                    mip as i32,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    depth as i32,
                    size,
                    output.as_mut_ptr().cast(),
                );
            } else {
                let get = extra
                    .get_texture_sub_image
                    .ok_or_else(|| unavailable("glGetTextureSubImage"))?;
                get(
                    name,
                    mip as i32,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    depth as i32,
                    gl_fmt.format,
                    gl_fmt.data_type,
                    size,
                    output.as_mut_ptr().cast(),
                );
            }
        }
        Ok(output)
    }

    pub(crate) fn create_texture_view(
        &mut self,
        desc: &TextureViewDescriptor,
    ) -> Result<TextureViewHandle, RhiError> {
        let (parent, info) = {
            let textures = lock(&self.resources.textures);
            let (texture, _) = textures.get(desc.texture).ok_or_else(|| {
                RhiError::InvalidHandle(format!("texture {:?} not found", desc.texture))
            })?;
            (texture.texture, texture.info)
        };
        desc.validate(&info)?;
        let texture_view = self
            .context
            .extra
            .texture_view
            .ok_or_else(|| unavailable("glTextureView"))?;
        let target = texture_target(desc.dimension, info.sample_count);
        let gl = &self.context.gl;
        // SAFETY: the context is current on the render thread. `glTextureView` needs a name
        // that was generated but never bound (`glGenTextures`, which `create_texture` calls);
        // the parent has immutable storage, and the range and target were validated.
        let view = unsafe {
            let view = gl
                .create_texture()
                .map_err(|e| RhiError::OutOfMemory(format!("glGenTextures failed: {e}")))?;
            texture_view(
                view.0.get(),
                target,
                parent.0.get(),
                to_gl_texture_format(info.format).internal_format,
                desc.base_mip_level,
                desc.mip_level_count,
                desc.base_array_layer,
                desc.array_layer_count,
            );
            if desc.aspect == TextureAspect::StencilOnly {
                gl.texture_parameter_i32(
                    view,
                    glow::DEPTH_STENCIL_TEXTURE_MODE,
                    glow::STENCIL_INDEX as i32,
                );
            }
            view
        };
        self.label_object(glow::TEXTURE, view.0.get(), desc.label);
        let (id, generation) = lock(&self.resources.texture_views).insert(GlTextureView {
            texture: view,
            parent: desc.texture,
            target,
            desc: desc.clone(),
        });
        let handle = TextureViewHandle::from_parts(self.resources.device, id, generation);
        if let Some((texture, _)) = lock(&self.resources.textures).get_mut(desc.texture) {
            texture.views.push(handle);
        }
        Ok(handle)
    }

    pub(crate) fn destroy_texture_view(
        &mut self,
        handle: TextureViewHandle,
    ) -> Result<(), RhiError> {
        let (view, _) = lock(&self.resources.texture_views)
            .remove(handle)
            .ok_or_else(|| RhiError::InvalidHandle(format!("texture view {handle:?} not found")))?;
        self.resources
            .invalidate_bind_groups(&[BoundResource::View(handle)]);
        if let Some((texture, _)) = lock(&self.resources.textures).get_mut(view.parent) {
            texture.views.retain(|&view| view != handle);
        }
        self.delete_texture_object(None, view.texture);
        Ok(())
    }
}
