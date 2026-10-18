//! Textures and texture views (FIX-1.4; findings RHI-05, RVK-13).
//!
//! Every dimension the RHI describes maps to an image type and default view type; multisampled
//! images use the descriptor's sample count. Textures usable as attachments get one view per
//! (mip, layer) up front, because Vulkan requires single-mip attachment views
//! (VUID-VkRenderingAttachmentInfo-imageView-12486); 3D attachments are created
//! `2D_ARRAY_COMPATIBLE` so a slice can be rendered. Square layered textures with at least six
//! layers are `CUBE_COMPATIBLE`, so they can be viewed as cubes.

use crate::device::{initial_data_layout, lock, Garbage, VulkanDevice};
use crate::resource::{TextureStorage, VulkanTexture, VulkanTextureView};
use crate::types::{barrier_aspect, copy_aspect, to_vk_image_layout, to_vk_image_usage};
use ash::vk;
use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, RhiError, SampleCount,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle, TextureInfo,
    TextureSubresourceRange, TextureUsage, TextureViewDescriptor, TextureViewHandle,
};

/// The Vulkan image type of a texture dimension.
pub(crate) fn image_type(dimension: TextureDimension) -> vk::ImageType {
    match dimension {
        TextureDimension::D1 => vk::ImageType::TYPE_1D,
        TextureDimension::D3 => vk::ImageType::TYPE_3D,
        _ => vk::ImageType::TYPE_2D,
    }
}

/// The Vulkan view type of a texture (or view) dimension.
pub(crate) fn view_type(dimension: TextureDimension) -> vk::ImageViewType {
    match dimension {
        TextureDimension::D1 => vk::ImageViewType::TYPE_1D,
        TextureDimension::D2 => vk::ImageViewType::TYPE_2D,
        TextureDimension::D3 => vk::ImageViewType::TYPE_3D,
        TextureDimension::Cube => vk::ImageViewType::CUBE,
        TextureDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
        TextureDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
    }
}

/// The Vulkan sample count flag of an RHI sample count.
pub(crate) fn sample_flags(samples: SampleCount) -> vk::SampleCountFlags {
    vk::SampleCountFlags::from_raw(samples.count())
}

/// Aspect of a view used for sampling: one aspect only (Vulkan forbids sampling both of a
/// depth/stencil view), depth by default.
fn sampling_aspect(format: TextureFormat, aspect: TextureAspect) -> vk::ImageAspectFlags {
    match aspect {
        TextureAspect::StencilOnly => vk::ImageAspectFlags::STENCIL,
        TextureAspect::DepthOnly => vk::ImageAspectFlags::DEPTH,
        TextureAspect::All if format.is_depth() => vk::ImageAspectFlags::DEPTH,
        TextureAspect::All => vk::ImageAspectFlags::COLOR,
    }
}

/// Usages that read or render through an image view.
const VIEW_USAGES: TextureUsage = TextureUsage::SAMPLED
    .union(TextureUsage::STORAGE)
    .union(TextureUsage::COLOR_ATTACHMENT)
    .union(TextureUsage::DEPTH_STENCIL_ATTACHMENT);

impl VulkanDevice {
    fn create_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        view_type: vk::ImageViewType,
        range: vk::ImageSubresourceRange,
    ) -> Result<vk::ImageView, RhiError> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .subresource_range(range);
        // SAFETY: `image` is live and was created with a usage and flags that allow `view_type`
        // over `range` (checked by the callers' descriptor validation).
        unsafe { self.context.device.create_image_view(&info, None) }
            .map_err(|e| RhiError::OutOfMemory(format!("vkCreateImageView failed: {e}")))
    }

    pub(crate) fn create_texture_impl(
        &self,
        desc: &TextureDescriptor,
        init: Option<&[u8]>,
    ) -> Result<TextureHandle, RhiError> {
        desc.validate()?;
        let levels_in_data = init
            .map(|data| desc.validate_initial_data(data))
            .transpose()?;
        // Usage the format must support on this device (RHI-09, RVK-13); initial data is copied
        // in, so it needs COPY_DST too.
        let mut required_usage = desc.usage;
        if init.is_some() {
            required_usage |= TextureUsage::COPY_DST;
        }
        let capabilities = &self.context.capabilities;
        capabilities
            .format(desc.format)
            .check(desc.format, required_usage, desc.sample_count)?;
        if desc.dimension == TextureDimension::CubeArray && !capabilities.image_cube_array {
            return Err(RhiError::Unsupported(
                "cube-array textures need the imageCubeArray feature".into(),
            ));
        }
        let vk_format = capabilities.vk_format(desc.format);
        let info = TextureInfo::from_descriptor(desc);

        let mut usage = to_vk_image_usage(required_usage);
        if usage.is_empty() {
            usage = vk::ImageUsageFlags::TRANSFER_DST;
        }
        let attachment = desc
            .usage
            .intersects(TextureUsage::COLOR_ATTACHMENT | TextureUsage::DEPTH_STENCIL_ATTACHMENT);
        let mut flags = vk::ImageCreateFlags::empty();
        if info.cube_compatible() {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        if desc.dimension == TextureDimension::D3 && attachment {
            flags |= vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE;
        }
        let extent = vk::Extent3D {
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
        };
        let image_info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type(desc.dimension))
            .format(vk_format)
            .extent(extent)
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(sample_flags(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let device = &self.context.device;
        // SAFETY: the create info was validated against the descriptor rules and the format's
        // capabilities; the image is destroyed through deferred destruction.
        let image = unsafe { device.create_image(&image_info, None) }
            .map_err(|e| RhiError::OutOfMemory(format!("vkCreateImage failed: {e}")))?;
        self.set_object_name(image, desc.label);
        // Memory (RVK-08): device-local, in the image pool, dedicated where the driver wants it
        // (typically large render targets).
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        // SAFETY: `image` was just created.
        unsafe {
            device.get_image_memory_requirements2(
                &vk::ImageMemoryRequirementsInfo2::default().image(image),
                &mut requirements2,
            )
        };
        let request = crate::memory::AllocationRequest {
            requirements: requirements2.memory_requirements,
            usage: korppi_rhi::MemoryUsage::GpuOnly,
            kind: crate::memory::ResourceKind::Image,
            dedicated: (dedicated.prefers_dedicated_allocation == vk::TRUE
                || dedicated.requires_dedicated_allocation == vk::TRUE)
                .then_some(crate::memory::DedicatedResource::Image(image)),
        };
        let suballoc = match self.memory.allocate(&request) {
            Ok(suballoc) => suballoc,
            Err(error) => {
                // SAFETY: the image is unused and has no memory.
                unsafe { device.destroy_image(image, None) };
                return Err(error);
            }
        };
        if let Err(error) =
            // SAFETY: the suballocation satisfies the image's requirements and is unused.
            unsafe {
                device.bind_image_memory(image, suballoc.device_memory, suballoc.offset)
            }
        {
            // SAFETY: as above.
            unsafe { device.destroy_image(image, None) };
            self.memory.deallocate(&suballoc);
            return Err(RhiError::OutOfMemory(format!(
                "vkBindImageMemory failed: {error}"
            )));
        }

        // Every view is registered with the texture as soon as it exists, so an error below
        // releases it with the texture.
        let (id, generation) = lock(&self.resources.textures).insert(VulkanTexture {
            image,
            view: vk::ImageView::null(),
            storage: TextureStorage::Device(suballoc),
            format: desc.format,
            usage: desc.usage,
            layouts: vec![
                vk::ImageLayout::UNDEFINED;
                (desc.mip_levels * desc.array_layers) as usize
            ],
            extent,
            info,
            attachment_views: Vec::new(),
            views: Vec::new(),
        });
        let handle = TextureHandle::from_parts(self.resources.device, id, generation);
        let created = self.create_texture_views(handle, image, vk_format, desc, attachment);
        if let Err(error) = created {
            let _ = self.destroy_texture_impl(handle);
            return Err(error);
        }
        if let (Some(data), Some(levels)) = (init, levels_in_data) {
            if let Err(error) = self.upload_initial_data(image, desc, data, levels) {
                let _ = self.destroy_texture_impl(handle);
                return Err(error);
            }
            self.set_texture_layout(handle, to_vk_image_layout(initial_data_layout(desc.usage)));
        }
        Ok(handle)
    }

    /// Creates the default view and the per-(mip, layer) attachment views of texture `id`.
    fn create_texture_views(
        &self,
        id: TextureHandle,
        image: vk::Image,
        vk_format: vk::Format,
        desc: &TextureDescriptor,
        attachment: bool,
    ) -> Result<(), RhiError> {
        if desc.usage.intersects(VIEW_USAGES) {
            let view = self.create_view(
                image,
                vk_format,
                view_type(desc.dimension),
                vk::ImageSubresourceRange {
                    aspect_mask: sampling_aspect(desc.format, TextureAspect::All),
                    base_mip_level: 0,
                    level_count: desc.mip_levels,
                    base_array_layer: 0,
                    layer_count: desc.array_layers,
                },
            )?;
            if let Some((texture, _)) = lock(&self.resources.textures).get_mut(id) {
                texture.view = view;
            }
        }
        if attachment {
            let info = TextureInfo::from_descriptor(desc);
            for mip in 0..desc.mip_levels {
                let mut per_layer = Vec::new();
                for layer in 0..info.attachment_layers(mip) {
                    let (view_type, base_array_layer) = match desc.dimension {
                        TextureDimension::D1 => (vk::ImageViewType::TYPE_1D, 0),
                        // A slice of a 2D_ARRAY_COMPATIBLE 3D image is a 2D view of one layer.
                        _ => (vk::ImageViewType::TYPE_2D, layer),
                    };
                    let view = self.create_view(
                        image,
                        vk_format,
                        view_type,
                        vk::ImageSubresourceRange {
                            aspect_mask: barrier_aspect(desc.format),
                            base_mip_level: mip,
                            level_count: 1,
                            base_array_layer,
                            layer_count: 1,
                        },
                    );
                    match view {
                        Ok(view) => per_layer.push(view),
                        Err(error) => {
                            self.keep_attachment_views(id, mip, per_layer);
                            return Err(error);
                        }
                    }
                }
                self.keep_attachment_views(id, mip, per_layer);
            }
        }
        Ok(())
    }

    fn keep_attachment_views(&self, id: TextureHandle, mip: u32, views: Vec<vk::ImageView>) {
        if let Some((texture, _)) = lock(&self.resources.textures).get_mut(id) {
            debug_assert_eq!(texture.attachment_views.len(), mip as usize);
            texture.attachment_views.push(views);
        }
    }

    /// Uploads `levels` mip levels of every layer from `data` (validated by
    /// `TextureDescriptor::validate_initial_data`) to a new image and moves every subresource to
    /// the layout of the texture's main use.
    fn upload_initial_data(
        &self,
        image: vk::Image,
        desc: &TextureDescriptor,
        data: &[u8],
        levels: u32,
    ) -> Result<(), RhiError> {
        let staging = self.create_buffer(
            &BufferDescriptor {
                size: data.len() as u64,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryUsage::CpuToGpu,
                label: Some("texture_init_staging"),
            },
            Some(data),
        )?;
        let staging_vk = self.vk_buffer(staging)?;
        let everything = vk::ImageSubresourceRange {
            aspect_mask: barrier_aspect(desc.format),
            base_mip_level: 0,
            level_count: desc.mip_levels,
            base_array_layer: 0,
            layer_count: desc.array_layers,
        };
        let to_transfer = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(everything);
        // One region per level: all layers (and slices) of a level are consecutive in `data`.
        let mut offset = 0;
        let regions: Vec<vk::BufferImageCopy> = (0..levels)
            .map(|mip| {
                let (width, height, depth) = desc.mip_extent(mip);
                let region = vk::BufferImageCopy {
                    buffer_offset: offset,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: copy_aspect(desc.format),
                        mip_level: mip,
                        base_array_layer: 0,
                        layer_count: desc.array_layers,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth,
                    },
                };
                offset += desc.level_size_bytes(mip);
                region
            })
            .collect();
        let settled = to_vk_image_layout(initial_data_layout(desc.usage));
        let to_settled = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(settled)
            .image(image)
            .subresource_range(everything);
        let result = self.submit_one_time(|device, cmd| {
            // SAFETY: `cmd` is recording; `image` was created with TRANSFER_DST usage and is not
            // yet visible to callers; the regions cover exactly the validated data, so the copy
            // reads only within the staging buffer.
            unsafe {
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&to_transfer)),
                );
                device.cmd_copy_buffer_to_image(
                    cmd,
                    staging_vk,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&to_settled)),
                );
            }
        });
        self.destroy_buffer(staging)?;
        result
    }

    pub(crate) fn read_texture_impl(
        &self,
        handle: TextureHandle,
        mip: u32,
    ) -> Result<Vec<u8>, RhiError> {
        let (image, info, layouts) = {
            let textures = lock(&self.resources.textures);
            let (tex, _) = textures.get(handle).ok_or_else(|| {
                RhiError::InvalidHandle(format!("Texture handle {handle:?} not found"))
            })?;
            let layouts: Vec<vk::ImageLayout> = (0..tex.info.array_layers)
                .map(|layer| tex.layout_of(mip.min(tex.info.mip_levels.saturating_sub(1)), layer))
                .collect();
            (tex.image, tex.info, layouts)
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
        let width = (info.width >> mip).max(1);
        let height = (info.height >> mip).max(1);
        let depth = (info.depth >> mip).max(1);
        // Whole blocks for compressed formats, the depth aspect for depth/stencil formats, every
        // layer and slice.
        let size = info.format.image_bytes(width, height)
            * u64::from(depth)
            * u64::from(info.array_layers);

        let staging = self.create_buffer(
            &BufferDescriptor {
                size,
                usage: BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuToCpu,
                label: Some("texture_readback_staging"),
            },
            None,
        )?;
        let staging_vk = self.vk_buffer(staging)?;
        // One transition per layer: layers of a mip may be in different layouts (RVK-07).
        let layer_range = |layer: u32| vk::ImageSubresourceRange {
            aspect_mask: barrier_aspect(info.format),
            base_mip_level: mip,
            level_count: 1,
            base_array_layer: layer,
            layer_count: 1,
        };
        let to_transfer: Vec<vk::ImageMemoryBarrier2> = layouts
            .iter()
            .enumerate()
            .map(|(layer, &layout)| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    // The layout submitted work left the subresource in (RVK-07).
                    .old_layout(layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(image)
                    .subresource_range(layer_range(layer as u32))
            })
            .collect();
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: copy_aspect(info.format),
                mip_level: mip,
                base_array_layer: 0,
                layer_count: info.array_layers,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth,
            },
        };
        let to_host = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .buffer(staging_vk)
            .offset(0)
            .size(size);
        // Back to the layouts the subresources had, so the caller's layout tracking stays valid
        // (undefined subresources stay in the transfer layout).
        let restored: Vec<vk::ImageLayout> = layouts
            .iter()
            .map(|&layout| {
                if layout == vk::ImageLayout::UNDEFINED {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                } else {
                    layout
                }
            })
            .collect();
        let restore: Vec<vk::ImageMemoryBarrier2> = restored
            .iter()
            .enumerate()
            .filter(|(_, &layout)| layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .map(|(layer, &layout)| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(layout)
                    .image(image)
                    .subresource_range(layer_range(layer as u32))
            })
            .collect();
        let copied = self.submit_one_time(|device, cmd| {
            // SAFETY: `cmd` is recording; the image (with TRANSFER_SRC usage) and the staging
            // buffer (TRANSFER_DST, `size` bytes, exactly one level of every layer) are live.
            unsafe {
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_transfer),
                );
                device.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    staging_vk,
                    &[region],
                );
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(&to_host))
                        .image_memory_barriers(&restore),
                );
            }
        });
        if copied.is_ok() {
            for (layer, layout) in restored.into_iter().enumerate() {
                self.set_texture_range_layout(
                    handle,
                    &TextureSubresourceRange::single(mip, layer as u32),
                    layout,
                );
            }
        }
        let output = copied.and_then(|()| GraphicsDevice::read_buffer(self, staging, 0, size));
        self.destroy_buffer(staging)?;
        output
    }

    pub(crate) fn destroy_texture_impl(&self, handle: TextureHandle) -> Result<(), RhiError> {
        let mut textures = lock(&self.resources.textures);
        match textures.get(handle) {
            None => {
                return Err(RhiError::InvalidHandle(format!(
                    "Texture handle {handle:?} not found"
                )))
            }
            Some((
                VulkanTexture {
                    storage: TextureStorage::Swapchain { .. },
                    ..
                },
                _,
            )) => {
                return Err(RhiError::ValidationError(format!(
                    "texture {handle:?} belongs to a swapchain and is destroyed with it"
                )))
            }
            Some(_) => {}
        }
        let (texture, _) = textures.remove(handle).expect("checked above");
        drop(textures);
        let TextureStorage::Device(suballoc) = texture.storage else {
            unreachable!("swapchain textures were rejected above")
        };
        // Views of the texture die with it.
        let mut views: Vec<vk::ImageView> =
            texture.attachment_views.into_iter().flatten().collect();
        {
            let mut registry = lock(&self.resources.texture_views);
            for view_handle in texture.views {
                if let Some((view, _)) = registry.remove(view_handle) {
                    views.push(view.view);
                }
            }
        }
        if texture.view != vk::ImageView::null() {
            views.push(texture.view);
        }
        self.defer_destruction(Garbage::Texture(texture.image, views, suballoc))
    }

    pub(crate) fn create_texture_view_impl(
        &self,
        desc: &TextureViewDescriptor,
    ) -> Result<TextureViewHandle, RhiError> {
        let (image, info) = {
            let textures = lock(&self.resources.textures);
            let (texture, _) = textures.get(desc.texture).ok_or_else(|| {
                RhiError::InvalidHandle(format!("texture {:?} not found", desc.texture))
            })?;
            (texture.image, texture.info)
        };
        desc.validate(&info)?;
        if desc.dimension == TextureDimension::CubeArray
            && !self.context.capabilities.image_cube_array
        {
            return Err(RhiError::Unsupported(
                "cube-array views need the imageCubeArray feature".into(),
            ));
        }
        let view = self.create_view(
            image,
            self.context.capabilities.vk_format(info.format),
            view_type(desc.dimension),
            vk::ImageSubresourceRange {
                aspect_mask: sampling_aspect(info.format, desc.aspect),
                base_mip_level: desc.base_mip_level,
                level_count: desc.mip_level_count,
                base_array_layer: desc.base_array_layer,
                layer_count: desc.array_layer_count,
            },
        )?;
        self.set_object_name(view, desc.label);
        let (id, generation) = lock(&self.resources.texture_views).insert(VulkanTextureView {
            view,
            texture: desc.texture,
            info,
            dimension: desc.dimension,
            desc: desc.clone(),
        });
        let handle = TextureViewHandle::from_parts(self.resources.device, id, generation);
        let registered = match lock(&self.resources.textures).get_mut(desc.texture) {
            Some((texture, _)) => {
                texture.views.push(handle);
                true
            }
            None => false,
        };
        if !registered {
            // The texture was destroyed meanwhile; nothing can use the view yet.
            lock(&self.resources.texture_views).remove(handle);
            // SAFETY: the view was just created and is unused.
            unsafe { self.context.device.destroy_image_view(view, None) };
            return Err(RhiError::InvalidHandle(format!(
                "texture {:?} not found",
                desc.texture
            )));
        }
        Ok(handle)
    }

    pub(crate) fn destroy_texture_view_impl(
        &self,
        handle: TextureViewHandle,
    ) -> Result<(), RhiError> {
        let (view, _) = lock(&self.resources.texture_views)
            .remove(handle)
            .ok_or_else(|| RhiError::InvalidHandle(format!("texture view {handle:?} not found")))?;
        if let Some((texture, _)) = lock(&self.resources.textures).get_mut(view.texture) {
            texture.views.retain(|&view| view != handle);
        }
        self.defer_destruction(Garbage::View(view.view))
    }
}
