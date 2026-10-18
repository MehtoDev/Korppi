//! Bind group layouts and bind groups (findings RHI-07, RVK-09).
//!
//! A bind group layout is a `VkDescriptorSetLayout` declared by the application, so pipeline
//! layouts hold only the sets and bindings their shaders use, within the device's per-stage
//! limits. A bind group is a descriptor set allocated and written once, when the group is
//! created, from long-lived pools; recording only binds it. Groups are freed individually
//! (`FREE_DESCRIPTOR_SET`) once the submissions that may use them have completed.

use crate::context::VulkanContext;
use crate::device::{lock, Garbage, VulkanDevice};
use crate::resource::{VulkanResourceRegistry, VulkanTextureView};
use crate::types::to_vk_image_layout;
use ash::vk;
use korppi_rhi::{
    validate_bind_group_resources, BindGroupDescriptor, BindGroupHandle, BindGroupInfo,
    BindGroupLayoutDescriptor, BindGroupLayoutHandle, BindGroupLayoutInfo, BindingLookup,
    BindingResource, BindingType, BoundResource, BufferHandle, BufferUsage, GraphicsDevice,
    ImageLayout, RhiError, SamplerHandle, ShaderStages, TextureBinding, TextureHandle, TextureInfo,
    TextureViewDescriptor, TextureViewHandle,
};
use std::sync::{Arc, Mutex};

/// Descriptor sets per pool. Pools are added when the existing ones are full.
const SETS_PER_POOL: u32 = 256;

/// Descriptors of each type per pool.
const DESCRIPTORS_PER_TYPE: u32 = SETS_PER_POOL * 4;

/// Internal Vulkan bind group layout.
pub struct VulkanBindGroupLayout {
    /// The descriptor set layout.
    pub layout: vk::DescriptorSetLayout,
    /// The declaration, entries sorted by binding.
    pub desc: BindGroupLayoutDescriptor,
    /// Summary for validation.
    pub info: BindGroupLayoutInfo,
}

/// Internal Vulkan bind group.
pub struct VulkanBindGroup {
    /// The descriptor set.
    pub set: vk::DescriptorSet,
    /// The pool it was allocated from.
    pub pool: vk::DescriptorPool,
    /// Summary for validation; `info.live` turns false when a resource it uses is destroyed.
    pub info: BindGroupInfo,
    /// Resources the group refers to.
    pub resources: Vec<BoundResource>,
}

/// The descriptor type of a binding.
fn descriptor_type(ty: BindingType) -> vk::DescriptorType {
    match ty {
        BindingType::UniformBuffer {
            dynamic_offset: false,
        } => vk::DescriptorType::UNIFORM_BUFFER,
        BindingType::UniformBuffer {
            dynamic_offset: true,
        } => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        BindingType::StorageBuffer {
            dynamic_offset: false,
            ..
        } => vk::DescriptorType::STORAGE_BUFFER,
        BindingType::StorageBuffer {
            dynamic_offset: true,
            ..
        } => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        BindingType::SampledTexture => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        BindingType::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
    }
}

/// Vulkan stage flags of `stages`.
pub(crate) fn stage_flags(stages: ShaderStages) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if stages.contains(ShaderStages::VERTEX) {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if stages.contains(ShaderStages::FRAGMENT) {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if stages.contains(ShaderStages::COMPUTE) {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    flags
}

/// Descriptor pools bind groups are allocated from, with the number of live sets in each.
/// Pools are not filled beyond `maxSets`, even where a driver would allow it.
pub(crate) struct DescriptorPools {
    context: Arc<VulkanContext>,
    pools: Mutex<Vec<(vk::DescriptorPool, u32)>>,
}

impl DescriptorPools {
    pub(crate) fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            pools: Mutex::new(Vec::new()),
        }
    }

    fn create_pool(&self) -> Result<vk::DescriptorPool, RhiError> {
        let sizes = [
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::DescriptorType::STORAGE_IMAGE,
        ]
        .map(|ty| vk::DescriptorPoolSize {
            ty,
            descriptor_count: DESCRIPTORS_PER_TYPE,
        });
        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(SETS_PER_POOL)
            .pool_sizes(&sizes);
        // SAFETY: `info` is valid; the pool is destroyed when the device drops.
        unsafe { self.context.device.create_descriptor_pool(&info, None) }
            .map_err(|e| RhiError::OutOfMemory(format!("vkCreateDescriptorPool failed: {e}")))
    }

    /// Allocates a set of `layout`, adding a pool when every existing one is full.
    fn allocate(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<(vk::DescriptorSet, vk::DescriptorPool), RhiError> {
        let layouts = [layout];
        let mut pools = lock(&self.pools);
        // The newest pool is the most likely to have room.
        for (pool, live) in pools.iter_mut().rev() {
            if *live >= SETS_PER_POOL {
                continue;
            }
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(*pool)
                .set_layouts(&layouts);
            // SAFETY: the pool is externally synchronized by the `pools` lock.
            match unsafe { self.context.device.allocate_descriptor_sets(&info) } {
                Ok(sets) => {
                    *live += 1;
                    return Ok((sets[0], *pool));
                }
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {}
                Err(e) => {
                    return Err(RhiError::OutOfMemory(format!(
                        "vkAllocateDescriptorSets failed: {e}"
                    )))
                }
            }
        }
        let pool = self.create_pool()?;
        pools.push((pool, 1));
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: as above; the pool is new and empty.
        let sets = unsafe { self.context.device.allocate_descriptor_sets(&info) }
            .map_err(|e| RhiError::OutOfMemory(format!("vkAllocateDescriptorSets failed: {e}")))?;
        Ok((sets[0], pool))
    }

    /// Frees `set`. No pending submission may use it.
    pub(crate) fn free(&self, pool: vk::DescriptorPool, set: vk::DescriptorSet) {
        let mut pools = lock(&self.pools);
        if let Some((_, live)) = pools.iter_mut().find(|(p, _)| *p == pool) {
            *live = live.saturating_sub(1);
        }
        // SAFETY: the pool was created with FREE_DESCRIPTOR_SET and is externally synchronized
        // by the `pools` lock; the caller guarantees no pending submission uses the set.
        if let Err(error) = unsafe { self.context.device.free_descriptor_sets(pool, &[set]) } {
            tracing::warn!("vkFreeDescriptorSets failed: {error}");
        }
    }

    /// Number of pools, for tests and diagnostics.
    pub(crate) fn len(&self) -> usize {
        lock(&self.pools).len()
    }
}

impl Drop for DescriptorPools {
    fn drop(&mut self) {
        for (pool, _) in lock(&self.pools).drain(..) {
            // SAFETY: the device is idle when it drops; destroying a pool frees its sets.
            unsafe { self.context.device.destroy_descriptor_pool(pool, None) };
        }
    }
}

impl BindingLookup for VulkanResourceRegistry {
    fn buffer(&self, buffer: BufferHandle) -> Option<(u64, BufferUsage)> {
        lock(&self.buffers)
            .get(buffer)
            .map(|(b, _)| (b.size, b.usage))
    }

    fn texture(&self, texture: TextureHandle) -> Option<TextureInfo> {
        lock(&self.textures).get(texture).map(|(t, _)| t.info)
    }

    fn texture_view(
        &self,
        view: TextureViewHandle,
    ) -> Option<(TextureViewDescriptor, TextureInfo)> {
        lock(&self.texture_views)
            .get(view)
            .map(|(v, _)| (v.desc.clone(), v.info))
    }

    fn sampler_exists(&self, sampler: SamplerHandle) -> bool {
        lock(&self.samplers).contains(sampler)
    }
}

/// A descriptor write's payload, resolved to native objects.
enum Payload {
    Buffer(vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
}

impl VulkanDevice {
    pub(crate) fn create_bind_group_layout_impl(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayoutHandle, RhiError> {
        desc.validate()?;
        let entries = desc.sorted_entries();
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = entries
            .iter()
            .map(|entry| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(entry.binding)
                    .descriptor_type(descriptor_type(entry.ty))
                    .descriptor_count(1)
                    .stage_flags(stage_flags(entry.stages))
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `info` and `bindings` are valid for the call.
        let layout = unsafe {
            self.context
                .device
                .create_descriptor_set_layout(&info, None)
        }
        .map_err(|e| RhiError::OutOfMemory(format!("vkCreateDescriptorSetLayout failed: {e}")))?;
        self.set_object_name(layout, desc.label);
        let desc = BindGroupLayoutDescriptor {
            entries,
            label: desc.label,
        };
        let (id, generation) =
            lock(&self.resources.bind_group_layouts).insert(VulkanBindGroupLayout {
                layout,
                info: desc.info(),
                desc,
            });
        Ok(BindGroupLayoutHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    pub(crate) fn destroy_bind_group_layout_impl(
        &self,
        handle: BindGroupLayoutHandle,
    ) -> Result<(), RhiError> {
        let (layout, _) = lock(&self.resources.bind_group_layouts)
            .remove(handle)
            .ok_or_else(|| {
                RhiError::InvalidHandle(format!("bind group layout {handle:?} not found"))
            })?;
        self.defer_destruction(Garbage::SetLayout(layout.layout))
    }

    pub(crate) fn create_bind_group_impl(
        &self,
        desc: &BindGroupDescriptor,
    ) -> Result<BindGroupHandle, RhiError> {
        let (set_layout, layout) = {
            let layouts = lock(&self.resources.bind_group_layouts);
            let (layout, _) = layouts.get(desc.layout).ok_or_else(|| {
                RhiError::InvalidHandle(format!("bind group layout {:?} not found", desc.layout))
            })?;
            (layout.layout, layout.desc.clone())
        };
        let limits = self.limits();
        validate_bind_group_resources(&layout, desc, &*self.resources, &limits)?;

        // Resolve every entry to native objects (one registry lock at a time).
        let mut writes: Vec<(u32, vk::DescriptorType, Payload)> =
            Vec::with_capacity(desc.entries.len());
        for entry in &desc.entries {
            let ty = layout
                .entries
                .iter()
                .find(|e| e.binding == entry.binding)
                .map(|e| descriptor_type(e.ty))
                .expect("validated against the layout");
            let payload = match entry.resource {
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                } => Payload::Buffer(vk::DescriptorBufferInfo {
                    buffer: lock(&self.resources.buffers)
                        .get(buffer)
                        .map(|(b, _)| b.buffer)
                        .ok_or_else(|| {
                            RhiError::InvalidHandle(format!("buffer {buffer:?} not found"))
                        })?,
                    offset,
                    range: size,
                }),
                BindingResource::SampledTexture {
                    texture,
                    sampler,
                    layout,
                } => Payload::Image(vk::DescriptorImageInfo {
                    sampler: lock(&self.resources.samplers)
                        .get(sampler)
                        .map(|(s, _)| s.sampler)
                        .ok_or_else(|| {
                            RhiError::InvalidHandle(format!("sampler {sampler:?} not found"))
                        })?,
                    image_view: self.binding_view(texture)?,
                    image_layout: to_vk_image_layout(layout),
                }),
                BindingResource::StorageTexture { texture } => {
                    Payload::Image(vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: self.binding_view(texture)?,
                        image_layout: to_vk_image_layout(ImageLayout::General),
                    })
                }
            };
            writes.push((entry.binding, ty, payload));
        }

        let (set, pool) = self.descriptor_pools.allocate(set_layout)?;
        self.set_object_name(set, desc.label);
        let vk_writes: Vec<vk::WriteDescriptorSet> = writes
            .iter()
            .map(|(binding, ty, payload)| {
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(*ty);
                match payload {
                    Payload::Buffer(info) => write.buffer_info(std::slice::from_ref(info)),
                    Payload::Image(info) => write.image_info(std::slice::from_ref(info)),
                }
            })
            .collect();
        // SAFETY: the set was just allocated and is not in use; every written object is live.
        unsafe { self.context.device.update_descriptor_sets(&vk_writes, &[]) };

        let buffer_size = |buffer: BufferHandle| {
            lock(&self.resources.buffers)
                .get(buffer)
                .map_or(0, |(b, _)| b.size)
        };
        let info = BindGroupInfo::new(&layout, desc, buffer_size, &limits);
        let (id, generation) = lock(&self.resources.bind_groups).insert(VulkanBindGroup {
            set,
            pool,
            info,
            resources: desc.resources().collect(),
        });
        Ok(BindGroupHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    pub(crate) fn destroy_bind_group_impl(&self, handle: BindGroupHandle) -> Result<(), RhiError> {
        let (group, _) = lock(&self.resources.bind_groups)
            .remove(handle)
            .ok_or_else(|| RhiError::InvalidHandle(format!("bind group {handle:?} not found")))?;
        self.defer_destruction(Garbage::DescriptorSet(group.pool, group.set))
    }

    /// The image view a texture binding reads: the texture's default view, or the view.
    fn binding_view(&self, texture: TextureBinding) -> Result<vk::ImageView, RhiError> {
        match texture {
            TextureBinding::Texture(handle) => lock(&self.resources.textures)
                .get(handle)
                .map(|(t, _)| t.view)
                .filter(|view| *view != vk::ImageView::null())
                .ok_or_else(|| RhiError::InvalidHandle(format!("texture {handle:?} has no view"))),
            TextureBinding::View(handle) => lock(&self.resources.texture_views)
                .get(handle)
                .map(|(v, _): &(VulkanTextureView, u32)| v.view)
                .ok_or_else(|| {
                    RhiError::InvalidHandle(format!("texture view {handle:?} not found"))
                }),
        }
    }

    /// Marks every bind group that refers to `destroyed` as no longer usable.
    pub(crate) fn invalidate_bind_groups(&self, destroyed: &[BoundResource]) {
        for (group, _) in lock(&self.resources.bind_groups).values_mut() {
            if group
                .resources
                .iter()
                .any(|bound| destroyed.contains(bound))
            {
                group.info.live = false;
            }
        }
    }

    /// Number of descriptor pools bind groups have been allocated from.
    pub fn descriptor_pool_count(&self) -> usize {
        self.descriptor_pools.len()
    }
}
