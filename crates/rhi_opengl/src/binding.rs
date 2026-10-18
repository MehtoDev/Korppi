//! Bind groups on OpenGL (findings RHI-07, RGL-12).
//!
//! OpenGL has one flat binding namespace per resource kind: uniform-block bindings, shader
//! storage bindings, texture units and image units. A pipeline numbers the bindings of its bind
//! group layouts set by set and, within a set, in binding order, separately per kind
//! ([`GlBindingTable`]); SPIR-V shaders are translated with those numbers when the pipeline is
//! created. Pipelines whose layouts are compatible therefore agree on every binding point, as
//! Vulkan pipeline layouts do on descriptor sets.
//!
//! Bind groups resolve their resources to GL objects once, at creation. Destroying a resource
//! marks the groups that use it as no longer usable (GL may reuse the object's name).

use crate::device::{GlThread, PUSH_CONSTANT_BINDING};
use crate::resource::{GlResourceRegistry, GlTextureView};
use crate::spirv::GlResourceKind;
use crate::state_cache::{GlImageBinding, GlStateCache};
use crate::types::to_gl_texture_format;
use korppi_rhi::lock;
use korppi_rhi::{
    validate_bind_group_resources, BindGroupDescriptor, BindGroupHandle, BindGroupInfo,
    BindGroupLayoutDescriptor, BindGroupLayoutHandle, BindGroupLayoutInfo, BindingLookup,
    BindingResource, BindingType, BoundResource, BufferHandle, BufferUsage, DeviceLimits, RhiError,
    SamplerHandle, StorageTextureAccess, TextureBinding, TextureHandle, TextureInfo,
    TextureViewDescriptor, TextureViewHandle, MAX_BINDINGS_PER_GROUP, MAX_BIND_GROUPS,
    MAX_DYNAMIC_OFFSETS_PER_GROUP,
};
use std::collections::HashMap;

/// Internal OpenGL bind group layout.
pub struct GlBindGroupLayout {
    /// The declaration, entries sorted by binding.
    pub desc: BindGroupLayoutDescriptor,
    /// Summary for validation.
    pub info: BindGroupLayoutInfo,
}

/// One resource of a bind group, resolved to GL objects.
#[derive(Copy, Clone, Debug)]
pub enum GlBinding {
    /// A uniform or storage buffer range.
    Buffer {
        /// `UNIFORM_BUFFER` or `SHADER_STORAGE_BUFFER`.
        target: u32,
        /// The buffer.
        buffer: glow::Buffer,
        /// Byte offset (the dynamic offset is added at bind time).
        offset: u64,
        /// Byte size.
        size: u64,
        /// Index of the binding's dynamic offset, if it has one.
        dynamic: Option<u8>,
    },
    /// A texture (or texture view) and sampler on a texture unit.
    SampledTexture {
        /// The texture.
        texture: glow::Texture,
        /// The sampler.
        sampler: glow::Sampler,
    },
    /// One mip level of a texture on an image unit.
    StorageTexture {
        /// The texture (level 0 is the bound level).
        texture: glow::Texture,
        /// Whether every layer is bound (arrays, cube maps, 3D textures).
        layered: bool,
        /// `READ_ONLY`, `WRITE_ONLY` or `READ_WRITE`.
        access: u32,
        /// Internal format the shader declares.
        format: u32,
    },
}

/// Internal OpenGL bind group.
pub struct GlBindGroup {
    /// Summary for validation; `info.live` turns false when a resource it uses is destroyed.
    pub info: BindGroupInfo,
    /// Resources the group refers to.
    pub resources: Vec<BoundResource>,
    /// (binding number, resource), in binding order.
    pub entries: Vec<(u32, GlBinding)>,
}

/// The GL binding point of every binding of a pipeline's bind group layouts.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct GlBindingTable {
    /// Binding point and kind per (set, binding number).
    points: [[Option<(GlResourceKind, u8)>; MAX_BINDINGS_PER_GROUP as usize]; MAX_BIND_GROUPS],
    /// Number of sets the pipeline declares.
    pub set_count: usize,
}

fn binding_kind(ty: BindingType) -> GlResourceKind {
    match ty {
        BindingType::UniformBuffer { .. } => GlResourceKind::UniformBuffer,
        BindingType::StorageBuffer { .. } => GlResourceKind::StorageBuffer,
        BindingType::SampledTexture => GlResourceKind::SampledTexture,
        BindingType::StorageTexture { .. } => GlResourceKind::StorageTexture,
    }
}

impl GlBindingTable {
    /// Numbers the bindings of `layouts` (sets 0, 1, ...) per kind, set by set, in binding
    /// order.
    ///
    /// # Errors
    /// `RhiError::Unsupported` if uniform blocks would reach the push-constant binding.
    pub fn new(layouts: &[&BindGroupLayoutDescriptor]) -> Result<Self, RhiError> {
        let mut table = Self {
            set_count: layouts.len().min(MAX_BIND_GROUPS),
            ..Self::default()
        };
        let mut next: HashMap<GlResourceKind, u32> = HashMap::new();
        for (set, layout) in layouts.iter().enumerate().take(MAX_BIND_GROUPS) {
            for entry in layout.sorted_entries() {
                let kind = binding_kind(entry.ty);
                let counter = next.entry(kind).or_insert(0);
                let point = *counter;
                *counter += 1;
                if kind == GlResourceKind::UniformBuffer && point >= PUSH_CONSTANT_BINDING {
                    return Err(RhiError::Unsupported(format!(
                        "more than {PUSH_CONSTANT_BINDING} uniform buffers in one pipeline"
                    )));
                }
                let point = u8::try_from(point).map_err(|_| {
                    RhiError::Unsupported("more than 256 bindings of one kind".into())
                })?;
                table.points[set][entry.binding as usize] = Some((kind, point));
            }
        }
        Ok(table)
    }

    /// The binding point of (`set`, `binding`), which a shader declares as `kind`.
    pub fn point(&self, set: u32, binding: u32, kind: GlResourceKind) -> Result<u8, String> {
        let declared = self
            .points
            .get(set as usize)
            .filter(|_| (set as usize) < self.set_count)
            .and_then(|points| points.get(binding as usize).copied().flatten());
        match declared {
            Some((declared, point)) if declared == kind => Ok(point),
            Some((declared, _)) => Err(format!(
                "set {set} binding {binding} is a {kind:?} in the shader but a {declared:?} in \
                 the pipeline's bind group layout"
            )),
            None => Err(format!(
                "set {set} binding {binding} ({kind:?}) is not in the pipeline's bind group \
                 layouts"
            )),
        }
    }

    /// The binding point of (`set`, `binding`) whatever its kind.
    fn point_of(&self, set: usize, binding: u32) -> Option<u32> {
        self.points[set]
            .get(binding as usize)
            .copied()
            .flatten()
            .map(|(_, point)| u32::from(point))
    }
}

/// A bind group set on a recorder: handle and dynamic offsets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BoundGroup {
    /// The group.
    pub group: BindGroupHandle,
    /// Dynamic offsets, `count` of them.
    pub offsets: [u32; MAX_DYNAMIC_OFFSETS_PER_GROUP],
    /// Number of dynamic offsets.
    pub count: u8,
}

/// Binds every group in `bound` at the binding points `table` gives them. Groups whose
/// resources were destroyed are skipped and reported through `report`.
///
/// # Safety
/// The context must be current; `state` must be its state cache.
pub unsafe fn apply_bind_groups(
    gl: &glow::Context,
    state: &mut GlStateCache,
    table: &GlBindingTable,
    bound: &[Option<BoundGroup>; MAX_BIND_GROUPS],
    groups: &korppi_rhi::SlotMap<GlBindGroup>,
    report: &mut dyn FnMut(String),
) {
    for (set, bound) in bound.iter().enumerate().take(table.set_count) {
        let Some(bound) = bound else {
            continue;
        };
        let Some((group, _)) = groups.get(bound.group) else {
            report(format!("set_bind_group: {:?} was destroyed", bound.group));
            continue;
        };
        if !group.info.live {
            report(format!(
                "set_bind_group: {:?} uses a destroyed resource; it was not bound",
                bound.group
            ));
            continue;
        }
        for &(binding, resource) in &group.entries {
            let Some(point) = table.point_of(set, binding) else {
                continue;
            };
            // SAFETY: forwarded from the caller; every object is live (the group is live).
            unsafe {
                match resource {
                    GlBinding::Buffer {
                        target,
                        buffer,
                        offset,
                        size,
                        dynamic,
                    } => {
                        let dynamic_offset = dynamic
                            .filter(|&i| i < bound.count)
                            .map_or(0, |i| u64::from(bound.offsets[i as usize]));
                        state.bind_buffer_range(
                            gl,
                            target,
                            point,
                            buffer,
                            (offset + dynamic_offset) as i32,
                            size as i32,
                        );
                    }
                    GlBinding::SampledTexture { texture, sampler } => {
                        state.bind_texture_unit(gl, point, Some(texture));
                        state.bind_sampler(gl, point, Some(sampler));
                    }
                    GlBinding::StorageTexture {
                        texture,
                        layered,
                        access,
                        format,
                    } => {
                        state.bind_image(
                            gl,
                            point,
                            Some(GlImageBinding {
                                texture,
                                layered,
                                access,
                                format,
                            }),
                        );
                    }
                }
            }
        }
    }
}

impl BindingLookup for GlResourceRegistry {
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
        let (desc, parent) = lock(&self.texture_views)
            .get(view)
            .map(|(v, _)| (v.desc.clone(), v.parent))?;
        let info = lock(&self.textures).get(parent).map(|(t, _)| t.info)?;
        Some((desc, info))
    }

    fn sampler_exists(&self, sampler: SamplerHandle) -> bool {
        lock(&self.samplers).contains(sampler)
    }
}

impl GlResourceRegistry {
    /// Marks every bind group that refers to `destroyed` as no longer usable.
    pub(crate) fn invalidate_bind_groups(&self, destroyed: &[BoundResource]) {
        for (group, _) in lock(&self.bind_groups).values_mut() {
            if group
                .resources
                .iter()
                .any(|bound| destroyed.contains(bound))
            {
                group.info.live = false;
            }
        }
    }

    /// The GL texture a binding reads and whether it has layers.
    fn binding_texture(&self, texture: TextureBinding) -> Result<(glow::Texture, bool), RhiError> {
        match texture {
            TextureBinding::Texture(handle) => lock(&self.textures)
                .get(handle)
                .map(|(t, _)| (t.texture, t.is_layered()))
                .ok_or_else(|| RhiError::InvalidHandle(format!("texture {handle:?} not found"))),
            TextureBinding::View(handle) => lock(&self.texture_views)
                .get(handle)
                .map(|(v, _): &(GlTextureView, u32)| {
                    let layered = !matches!(v.target, glow::TEXTURE_1D | glow::TEXTURE_2D);
                    (v.texture, layered)
                })
                .ok_or_else(|| {
                    RhiError::InvalidHandle(format!("texture view {handle:?} not found"))
                }),
        }
    }
}

impl GlThread {
    pub(crate) fn create_bind_group_layout(
        &mut self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<BindGroupLayoutHandle, RhiError> {
        desc.validate()?;
        let desc = BindGroupLayoutDescriptor {
            entries: desc.sorted_entries(),
            label: desc.label,
        };
        let (id, generation) = lock(&self.resources.bind_group_layouts).insert(GlBindGroupLayout {
            info: desc.info(),
            desc,
        });
        Ok(BindGroupLayoutHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    pub(crate) fn destroy_bind_group_layout(
        &mut self,
        handle: BindGroupLayoutHandle,
    ) -> Result<(), RhiError> {
        lock(&self.resources.bind_group_layouts)
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| {
                RhiError::InvalidHandle(format!("bind group layout {handle:?} not found"))
            })
    }

    pub(crate) fn create_bind_group(
        &mut self,
        desc: &BindGroupDescriptor,
        limits: &DeviceLimits,
    ) -> Result<BindGroupHandle, RhiError> {
        let resources = &self.resources;
        let layout = lock(&resources.bind_group_layouts)
            .get(desc.layout)
            .map(|(layout, _)| layout.desc.clone())
            .ok_or_else(|| {
                RhiError::InvalidHandle(format!("bind group layout {:?} not found", desc.layout))
            })?;
        validate_bind_group_resources(&layout, desc, &**resources, limits)?;

        let mut entries = Vec::with_capacity(desc.entries.len());
        let mut dynamic_index = 0u8;
        for declared in &layout.entries {
            let resource = desc
                .entries
                .iter()
                .find(|e| e.binding == declared.binding)
                .map(|e| e.resource)
                .expect("validated against the layout");
            let binding = match (declared.ty, resource) {
                (
                    ty,
                    BindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    },
                ) => {
                    let dynamic = ty.has_dynamic_offset().then(|| {
                        dynamic_index += 1;
                        dynamic_index - 1
                    });
                    GlBinding::Buffer {
                        target: if matches!(ty, BindingType::StorageBuffer { .. }) {
                            glow::SHADER_STORAGE_BUFFER
                        } else {
                            glow::UNIFORM_BUFFER
                        },
                        buffer: lock(&resources.buffers)
                            .get(buffer)
                            .map(|(b, _)| b.buffer)
                            .ok_or_else(|| {
                                RhiError::InvalidHandle(format!("buffer {buffer:?} not found"))
                            })?,
                        offset,
                        size,
                        dynamic,
                    }
                }
                (
                    _,
                    BindingResource::SampledTexture {
                        texture, sampler, ..
                    },
                ) => GlBinding::SampledTexture {
                    texture: resources.binding_texture(texture)?.0,
                    sampler: lock(&resources.samplers)
                        .get(sampler)
                        .map(|(s, _)| s.sampler)
                        .ok_or_else(|| {
                            RhiError::InvalidHandle(format!("sampler {sampler:?} not found"))
                        })?,
                },
                (
                    BindingType::StorageTexture { format, access },
                    BindingResource::StorageTexture { texture },
                ) => {
                    let (texture, layered) = resources.binding_texture(texture)?;
                    GlBinding::StorageTexture {
                        texture,
                        layered,
                        access: match access {
                            StorageTextureAccess::ReadOnly => glow::READ_ONLY,
                            StorageTextureAccess::WriteOnly => glow::WRITE_ONLY,
                            StorageTextureAccess::ReadWrite => glow::READ_WRITE,
                        },
                        format: to_gl_texture_format(format).internal_format,
                    }
                }
                _ => unreachable!("validated against the layout"),
            };
            entries.push((declared.binding, binding));
        }

        let buffer_size = |buffer: BufferHandle| {
            lock(&resources.buffers)
                .get(buffer)
                .map_or(0, |(b, _)| b.size)
        };
        let info = BindGroupInfo::new(&layout, desc, buffer_size, limits);
        let (id, generation) = lock(&self.resources.bind_groups).insert(GlBindGroup {
            info,
            resources: desc.resources().collect(),
            entries,
        });
        Ok(BindGroupHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    pub(crate) fn destroy_bind_group(&mut self, handle: BindGroupHandle) -> Result<(), RhiError> {
        // Submitted work is played back in order on this thread, so nothing still needs it.
        lock(&self.resources.bind_groups)
            .remove(handle)
            .map(|_| ())
            .ok_or_else(|| RhiError::InvalidHandle(format!("bind group {handle:?} not found")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use korppi_rhi::{BindGroupLayoutEntry, ShaderStages};

    fn layout(entries: &[(u32, BindingType)]) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor {
            entries: entries
                .iter()
                .map(|&(binding, ty)| BindGroupLayoutEntry {
                    binding,
                    ty,
                    stages: ShaderStages::FRAGMENT,
                })
                .collect(),
            label: None,
        }
    }

    const UBO: BindingType = BindingType::UniformBuffer {
        dynamic_offset: false,
    };
    const SSBO: BindingType = BindingType::StorageBuffer {
        read_only: true,
        dynamic_offset: false,
    };

    #[test]
    fn bindings_are_numbered_per_kind_across_sets_in_binding_order() {
        let set0 = layout(&[(3, UBO), (1, BindingType::SampledTexture), (0, UBO)]);
        let set1 = layout(&[(0, SSBO), (2, UBO), (1, BindingType::SampledTexture)]);
        let table = GlBindingTable::new(&[&set0, &set1]).expect("table");
        let ubo = GlResourceKind::UniformBuffer;
        let tex = GlResourceKind::SampledTexture;
        assert_eq!(table.point(0, 0, ubo), Ok(0));
        assert_eq!(table.point(0, 3, ubo), Ok(1));
        assert_eq!(table.point(1, 2, ubo), Ok(2));
        assert_eq!(table.point(0, 1, tex), Ok(0));
        assert_eq!(table.point(1, 1, tex), Ok(1));
        assert_eq!(table.point(1, 0, GlResourceKind::StorageBuffer), Ok(0));
    }

    #[test]
    fn shaders_must_match_the_layout() {
        let set0 = layout(&[(0, UBO)]);
        let table = GlBindingTable::new(&[&set0]).expect("table");
        assert!(table.point(0, 0, GlResourceKind::StorageBuffer).is_err());
        assert!(table.point(0, 1, GlResourceKind::UniformBuffer).is_err());
        assert!(table.point(1, 0, GlResourceKind::UniformBuffer).is_err());
    }

    #[test]
    fn uniform_buffers_stop_before_the_push_constant_binding() {
        // Four sets of sixteen uniform buffers is the most a pipeline can declare.
        let entries: Vec<_> = (0..MAX_BINDINGS_PER_GROUP).map(|b| (b, UBO)).collect();
        let full = layout(&entries);
        let table = GlBindingTable::new(&[&full, &full, &full, &full]).expect("table");
        let last = table.point(3, MAX_BINDINGS_PER_GROUP - 1, GlResourceKind::UniformBuffer);
        assert_eq!(last, Ok(PUSH_CONSTANT_BINDING as u8 - 1));
    }
}
