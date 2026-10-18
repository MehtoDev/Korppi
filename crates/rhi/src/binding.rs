//! Explicit resource binding (findings RHI-07, RVK-09).
//!
//! Shaders read resources through descriptor sets: `layout(set = S, binding = B)`. A
//! [`BindGroupLayoutDescriptor`] declares what one set contains: each binding's type and the
//! shader stages that see it. Pipelines list the layouts of their sets in order
//! (`bind_group_layouts[S]`); a [`BindGroupDescriptor`] fills a layout with concrete buffers,
//! textures and samplers once, up front; recording then only selects groups
//! (`CommandBufferRecorder::set_bind_group`), with dynamic offsets for bindings declared
//! `dynamic_offset` (per-draw uniform data in one buffer).
//!
//! On OpenGL, which has one flat binding space per resource kind, the backend numbers each
//! pipeline's bindings set by set and binding by binding within each kind (uniform buffers,
//! storage buffers, sampled textures, storage textures) and translates SPIR-V shaders with that
//! numbering. Hand-written GLSL must use the same numbers.

use crate::capabilities::DeviceLimits;
use crate::descriptor::TextureViewDescriptor;
use crate::error::RhiError;
use crate::handle::{
    BindGroupLayoutHandle, BufferHandle, SamplerHandle, TextureHandle, TextureViewHandle,
};
use crate::sync::ImageLayout;
use crate::types::{BufferUsage, SampleCount, TextureFormat, TextureUsage};
use crate::validation::TextureInfo;
use korppi_core::bitflags;
use std::hash::{Hash, Hasher};

/// Descriptor sets a pipeline can have.
pub const MAX_BIND_GROUPS: usize = 4;

/// Binding numbers in a set are below this.
pub const MAX_BINDINGS_PER_GROUP: u32 = 16;

/// Dynamic offsets one bind group can have.
pub const MAX_DYNAMIC_OFFSETS_PER_GROUP: usize = 8;

bitflags! {
    /// Shader stages that can access a binding or push-constant range.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ShaderStages: u32 {
        /// Vertex stage.
        const VERTEX = 1 << 0;
        /// Fragment stage.
        const FRAGMENT = 1 << 1;
        /// Compute stage.
        const COMPUTE = 1 << 2;
        /// Vertex and fragment stages.
        const VERTEX_FRAGMENT = Self::VERTEX.bits() | Self::FRAGMENT.bits();
    }
}

/// How a shader accesses a storage texture.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StorageTextureAccess {
    /// `readonly` image.
    ReadOnly,
    /// `writeonly` image.
    WriteOnly,
    /// Read and write.
    ReadWrite,
}

/// The kind of resource a binding holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BindingType {
    /// Uniform buffer (`uniform Block { ... }`).
    UniformBuffer {
        /// The offset is given at `set_bind_group` time (added to the bound offset).
        dynamic_offset: bool,
    },
    /// Storage buffer (`buffer Block { ... }`).
    StorageBuffer {
        /// The shader only reads it.
        read_only: bool,
        /// The offset is given at `set_bind_group` time (added to the bound offset).
        dynamic_offset: bool,
    },
    /// A texture with a sampler (`sampler2D`, `samplerCube`, ...).
    SampledTexture,
    /// A storage texture (`image2D`, ...) of `format`, one mip level.
    StorageTexture {
        /// Format the shader declares.
        format: TextureFormat,
        /// Access the shader declares.
        access: StorageTextureAccess,
    },
}

impl BindingType {
    /// True for buffer bindings with a dynamic offset.
    pub const fn has_dynamic_offset(&self) -> bool {
        matches!(
            self,
            Self::UniformBuffer {
                dynamic_offset: true
            } | Self::StorageBuffer {
                dynamic_offset: true,
                ..
            }
        )
    }
}

/// One binding of a bind group layout.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BindGroupLayoutEntry {
    /// `binding = B` in the shader.
    pub binding: u32,
    /// Kind of resource.
    pub ty: BindingType,
    /// Stages that access the binding.
    pub stages: ShaderStages,
}

/// Declaration of one descriptor set.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BindGroupLayoutDescriptor {
    /// Bindings, each `binding` number at most once.
    pub entries: Vec<BindGroupLayoutEntry>,
    /// Optional debug label; not serialized (RHI-14).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub label: Option<&'static str>,
}

impl BindGroupLayoutDescriptor {
    /// Checks for binding numbers out of range or declared twice, empty stage sets and too many
    /// dynamic offsets.
    ///
    /// # Errors
    /// `RhiError::ValidationError` describing the problem.
    pub fn validate(&self) -> Result<(), RhiError> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.binding >= MAX_BINDINGS_PER_GROUP {
                return Err(RhiError::ValidationError(format!(
                    "binding {} is not below {MAX_BINDINGS_PER_GROUP}",
                    entry.binding
                )));
            }
            if entry.stages.is_empty() {
                return Err(RhiError::ValidationError(format!(
                    "binding {} has no shader stages",
                    entry.binding
                )));
            }
            if self.entries[..i].iter().any(|e| e.binding == entry.binding) {
                return Err(RhiError::ValidationError(format!(
                    "binding {} is declared twice",
                    entry.binding
                )));
            }
        }
        let dynamic = self.dynamic_offset_count();
        if dynamic > MAX_DYNAMIC_OFFSETS_PER_GROUP {
            return Err(RhiError::ValidationError(format!(
                "{dynamic} dynamic-offset bindings; at most {MAX_DYNAMIC_OFFSETS_PER_GROUP} per group"
            )));
        }
        Ok(())
    }

    /// Number of bindings with a dynamic offset.
    pub fn dynamic_offset_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.ty.has_dynamic_offset())
            .count()
    }

    /// Entries ordered by binding number (the order of dynamic offsets at `set_bind_group`).
    pub fn sorted_entries(&self) -> Vec<BindGroupLayoutEntry> {
        let mut entries = self.entries.clone();
        entries.sort_by_key(|e| e.binding);
        entries
    }

    /// Summary for validation: layouts with the same entries (in any order) have the same
    /// fingerprint and are compatible.
    pub fn info(&self) -> BindGroupLayoutInfo {
        let mut hasher = std::hash::DefaultHasher::new();
        self.sorted_entries().hash(&mut hasher);
        BindGroupLayoutInfo {
            fingerprint: hasher.finish(),
            dynamic_offsets: self.dynamic_offset_count() as u32,
        }
    }
}

/// Summary of a bind group layout: a fingerprint of its entries, and its dynamic-offset count.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutInfo {
    /// Equal for layouts with equal entries (compatible layouts).
    pub fingerprint: u64,
    /// Dynamic offsets `set_bind_group` takes for groups of this layout.
    pub dynamic_offsets: u32,
}

/// The texture a sampled or storage binding reads: a whole texture (its default view) or a
/// view.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TextureBinding {
    /// The texture's default view (all mips and layers).
    Texture(TextureHandle),
    /// A texture view.
    View(TextureViewHandle),
}

impl From<TextureHandle> for TextureBinding {
    fn from(texture: TextureHandle) -> Self {
        Self::Texture(texture)
    }
}

impl From<TextureViewHandle> for TextureBinding {
    fn from(view: TextureViewHandle) -> Self {
        Self::View(view)
    }
}

/// The resource bound to one binding of a group.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BindingResource {
    /// A buffer range, for uniform and storage buffer bindings. With a dynamic offset, the
    /// offset given at `set_bind_group` is added to `offset`.
    Buffer {
        /// The buffer.
        buffer: BufferHandle,
        /// Byte offset (aligned to the device's minimum offset alignment).
        offset: u64,
        /// Byte size of the range.
        size: u64,
    },
    /// A texture and sampler, for sampled-texture bindings. `layout` is the layout the texture
    /// is in while the group is used: `ShaderReadOnly`, `DepthStencilReadOnly` or `General`.
    SampledTexture {
        /// The texture or view.
        texture: TextureBinding,
        /// The sampler.
        sampler: SamplerHandle,
        /// Layout of the texture while the group is used.
        layout: ImageLayout,
    },
    /// One mip level of a texture, for storage-texture bindings (in the `General` layout).
    StorageTexture {
        /// The texture or view (a single mip level).
        texture: TextureBinding,
    },
}

/// One entry of a bind group.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindGroupEntry {
    /// Binding number in the layout.
    pub binding: u32,
    /// The resource.
    pub resource: BindingResource,
}

/// A layout filled with resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindGroupDescriptor {
    /// The layout this group fills.
    pub layout: BindGroupLayoutHandle,
    /// One entry per binding of the layout.
    pub entries: Vec<BindGroupEntry>,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

/// Checks `group` against its `layout`: one entry per binding, each of the right kind.
///
/// # Errors
/// `RhiError::ValidationError` describing the first mismatch.
pub fn validate_bind_group(
    layout: &BindGroupLayoutDescriptor,
    group: &BindGroupDescriptor,
) -> Result<(), RhiError> {
    if group.entries.len() != layout.entries.len() {
        return Err(RhiError::ValidationError(format!(
            "the group has {} entries; its layout has {} bindings",
            group.entries.len(),
            layout.entries.len()
        )));
    }
    for entry in &group.entries {
        let declared = layout
            .entries
            .iter()
            .find(|e| e.binding == entry.binding)
            .ok_or_else(|| {
                RhiError::ValidationError(format!("binding {} is not in the layout", entry.binding))
            })?;
        let ok = matches!(
            (declared.ty, entry.resource),
            (
                BindingType::UniformBuffer { .. } | BindingType::StorageBuffer { .. },
                BindingResource::Buffer { .. }
            ) | (
                BindingType::SampledTexture,
                BindingResource::SampledTexture { .. }
            ) | (
                BindingType::StorageTexture { .. },
                BindingResource::StorageTexture { .. }
            )
        );
        if !ok {
            return Err(RhiError::ValidationError(format!(
                "binding {} is declared {:?} but given {:?}",
                entry.binding, declared.ty, entry.resource
            )));
        }
        if let BindingResource::SampledTexture { layout, .. } = entry.resource {
            if !matches!(
                layout,
                ImageLayout::ShaderReadOnly
                    | ImageLayout::DepthStencilReadOnly
                    | ImageLayout::General
            ) {
                return Err(RhiError::ValidationError(format!(
                    "binding {}: textures are sampled in ShaderReadOnly, DepthStencilReadOnly or \
                     General, not {layout:?}",
                    entry.binding
                )));
            }
        }
        if let BindingResource::Buffer { size: 0, .. } = entry.resource {
            return Err(RhiError::ValidationError(format!(
                "binding {}: empty buffer range",
                entry.binding
            )));
        }
    }
    Ok(())
}

/// Access to a device's resources for [`validate_bind_group_resources`].
pub trait BindingLookup {
    /// Size and usage of a live buffer.
    fn buffer(&self, buffer: BufferHandle) -> Option<(u64, BufferUsage)>;
    /// Description of a live texture.
    fn texture(&self, texture: TextureHandle) -> Option<TextureInfo>;
    /// A live view and the description of its texture.
    fn texture_view(&self, view: TextureViewHandle)
        -> Option<(TextureViewDescriptor, TextureInfo)>;
    /// True if `sampler` is live.
    fn sampler_exists(&self, sampler: SamplerHandle) -> bool;
}

/// The texture a binding reads and the mip levels it sees.
fn bound_texture(
    lookup: &dyn BindingLookup,
    texture: TextureBinding,
) -> Result<(TextureInfo, u32), RhiError> {
    match texture {
        TextureBinding::Texture(handle) => lookup
            .texture(handle)
            .map(|info| (info, info.mip_levels))
            .ok_or_else(|| RhiError::InvalidHandle(format!("unknown texture {handle:?}"))),
        TextureBinding::View(handle) => lookup
            .texture_view(handle)
            .map(|(view, info)| (info, view.mip_level_count))
            .ok_or_else(|| RhiError::InvalidHandle(format!("unknown texture view {handle:?}"))),
    }
}

/// Checks a bind group against its layout and the resources it names: buffer usage, offset
/// alignment and range; texture usage; storage textures with one mip level, one sample and the
/// declared format.
///
/// # Errors
/// `RhiError::InvalidHandle` for unknown resources, `RhiError::ValidationError` otherwise.
pub fn validate_bind_group_resources(
    layout: &BindGroupLayoutDescriptor,
    group: &BindGroupDescriptor,
    lookup: &dyn BindingLookup,
    limits: &DeviceLimits,
) -> Result<(), RhiError> {
    validate_bind_group(layout, group)?;
    let invalid = |binding: u32, what: String| {
        Err(RhiError::ValidationError(format!(
            "binding {binding}: {what}"
        )))
    };
    for entry in &group.entries {
        let declared = layout
            .entries
            .iter()
            .find(|e| e.binding == entry.binding)
            .map(|e| e.ty);
        match (declared, entry.resource) {
            (
                Some(ty),
                BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                },
            ) => {
                let (buffer_size, usage) = lookup
                    .buffer(buffer)
                    .ok_or_else(|| RhiError::InvalidHandle(format!("unknown buffer {buffer:?}")))?;
                let storage = matches!(ty, BindingType::StorageBuffer { .. });
                let (needed, alignment, max_size) = if storage {
                    (
                        BufferUsage::STORAGE,
                        limits.min_storage_buffer_offset_alignment,
                        limits.max_storage_buffer_binding_size,
                    )
                } else {
                    (
                        BufferUsage::UNIFORM,
                        limits.min_uniform_buffer_offset_alignment,
                        limits.max_uniform_buffer_binding_size,
                    )
                };
                if !usage.contains(needed) {
                    return invalid(entry.binding, format!("{buffer:?} lacks {needed:?} usage"));
                }
                if offset % alignment.max(1) != 0 {
                    return invalid(
                        entry.binding,
                        format!("offset {offset} is not a multiple of {alignment}"),
                    );
                }
                if offset.checked_add(size).is_none_or(|end| end > buffer_size) {
                    return invalid(
                        entry.binding,
                        format!("range {offset}+{size} exceeds the buffer's {buffer_size} bytes"),
                    );
                }
                if size > max_size {
                    return invalid(
                        entry.binding,
                        format!("{size} bytes exceed the device's {max_size}-byte binding limit"),
                    );
                }
            }
            (
                _,
                BindingResource::SampledTexture {
                    texture, sampler, ..
                },
            ) => {
                if !lookup.sampler_exists(sampler) {
                    return Err(RhiError::InvalidHandle(format!(
                        "unknown sampler {sampler:?}"
                    )));
                }
                let (info, _) = bound_texture(lookup, texture)?;
                if !info.usage.contains(TextureUsage::SAMPLED) {
                    return invalid(entry.binding, "the texture lacks SAMPLED usage".into());
                }
            }
            (
                Some(BindingType::StorageTexture { format, .. }),
                BindingResource::StorageTexture { texture },
            ) => {
                let (info, mips) = bound_texture(lookup, texture)?;
                if !info.usage.contains(TextureUsage::STORAGE) {
                    return invalid(entry.binding, "the texture lacks STORAGE usage".into());
                }
                if mips != 1 {
                    return invalid(
                        entry.binding,
                        format!("storage textures bind one mip level, not {mips} (use a view)"),
                    );
                }
                if info.sample_count != SampleCount::Count1 {
                    return invalid(
                        entry.binding,
                        "storage textures cannot be multisampled".into(),
                    );
                }
                if info.format != format {
                    return invalid(
                        entry.binding,
                        format!(
                            "the texture is {:?}; the layout declares {format:?}",
                            info.format
                        ),
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// A resource a bind group refers to, for invalidating groups when it is destroyed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundResource {
    /// A buffer.
    Buffer(BufferHandle),
    /// A texture (bound whole).
    Texture(TextureHandle),
    /// A texture view.
    View(TextureViewHandle),
    /// A sampler.
    Sampler(SamplerHandle),
}

impl BindGroupDescriptor {
    /// Every resource the group refers to.
    pub fn resources(&self) -> impl Iterator<Item = BoundResource> + '_ {
        self.entries.iter().flat_map(|entry| {
            let texture = |t: TextureBinding| match t {
                TextureBinding::Texture(t) => BoundResource::Texture(t),
                TextureBinding::View(v) => BoundResource::View(v),
            };
            let (first, second) = match entry.resource {
                BindingResource::Buffer { buffer, .. } => (BoundResource::Buffer(buffer), None),
                BindingResource::SampledTexture {
                    texture: t,
                    sampler,
                    ..
                } => (texture(t), Some(BoundResource::Sampler(sampler))),
                BindingResource::StorageTexture { texture: t } => (texture(t), None),
            };
            std::iter::once(first).chain(second)
        })
    }
}

/// A dynamic-offset binding of a bind group, for checking offsets at `set_bind_group`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DynamicBinding {
    /// End of the bound range (offset + size) before the dynamic offset is added.
    pub end: u64,
    /// Size of the buffer.
    pub buffer_size: u64,
    /// Required alignment of dynamic offsets.
    pub alignment: u64,
}

/// Summary of a bind group for recording validation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BindGroupInfo {
    /// Its layout.
    pub layout: BindGroupLayoutInfo,
    /// Dynamic-offset bindings in binding order (`layout.dynamic_offsets` of them).
    pub dynamic: [DynamicBinding; MAX_DYNAMIC_OFFSETS_PER_GROUP],
    /// False once a resource it refers to has been destroyed.
    pub live: bool,
}

impl BindGroupInfo {
    /// Summary of a group created from `group` (validated against `layout`).
    pub fn new(
        layout: &BindGroupLayoutDescriptor,
        group: &BindGroupDescriptor,
        buffer_size: impl Fn(BufferHandle) -> u64,
        limits: &DeviceLimits,
    ) -> Self {
        let mut dynamic = [DynamicBinding::default(); MAX_DYNAMIC_OFFSETS_PER_GROUP];
        let dynamic_entries = layout
            .sorted_entries()
            .into_iter()
            .filter(|e| e.ty.has_dynamic_offset());
        for (slot, declared) in dynamic.iter_mut().zip(dynamic_entries) {
            let resource = group
                .entries
                .iter()
                .find(|e| e.binding == declared.binding)
                .map(|e| e.resource);
            if let Some(BindingResource::Buffer {
                buffer,
                offset,
                size,
            }) = resource
            {
                *slot = DynamicBinding {
                    end: offset + size,
                    buffer_size: buffer_size(buffer),
                    alignment: if matches!(declared.ty, BindingType::StorageBuffer { .. }) {
                        limits.min_storage_buffer_offset_alignment
                    } else {
                        limits.min_uniform_buffer_offset_alignment
                    },
                };
            }
        }
        Self {
            layout: layout.info(),
            dynamic,
            live: true,
        }
    }
}

/// Checks the bind group layouts of a pipeline using `stages` against the device's per-stage
/// limits.
///
/// # Errors
/// `RhiError::ValidationError` for more than [`MAX_BIND_GROUPS`] layouts, `RhiError::Unsupported`
/// for a limit exceeded.
pub fn check_pipeline_layouts(
    layouts: &[&BindGroupLayoutDescriptor],
    stages: ShaderStages,
    limits: &DeviceLimits,
) -> Result<(), RhiError> {
    if layouts.len() > MAX_BIND_GROUPS {
        return Err(RhiError::ValidationError(format!(
            "{} bind group layouts; at most {MAX_BIND_GROUPS}",
            layouts.len()
        )));
    }
    for stage in [
        ShaderStages::VERTEX,
        ShaderStages::FRAGMENT,
        ShaderStages::COMPUTE,
    ] {
        if stages.contains(stage) {
            StageBindingCounts::of(layouts.iter().copied(), stage).check(limits, stage)?;
        }
    }
    Ok(())
}

/// Per-kind binding counts of one shader stage, for checking pipelines against device limits.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StageBindingCounts {
    /// Uniform buffers.
    pub uniform_buffers: u32,
    /// Storage buffers.
    pub storage_buffers: u32,
    /// Sampled textures.
    pub sampled_textures: u32,
    /// Storage textures.
    pub storage_textures: u32,
}

impl StageBindingCounts {
    /// Counts the bindings of `layouts` that `stage` sees.
    pub fn of<'a>(
        layouts: impl IntoIterator<Item = &'a BindGroupLayoutDescriptor>,
        stage: ShaderStages,
    ) -> Self {
        let mut counts = Self::default();
        for entry in layouts
            .into_iter()
            .flat_map(|l| &l.entries)
            .filter(|e| e.stages.intersects(stage))
        {
            match entry.ty {
                BindingType::UniformBuffer { .. } => counts.uniform_buffers += 1,
                BindingType::StorageBuffer { .. } => counts.storage_buffers += 1,
                BindingType::SampledTexture => counts.sampled_textures += 1,
                BindingType::StorageTexture { .. } => counts.storage_textures += 1,
            }
        }
        counts
    }

    /// Checks the counts against `limits`.
    ///
    /// # Errors
    /// `RhiError::Unsupported` naming the exceeded limit.
    pub fn check(&self, limits: &DeviceLimits, stage: ShaderStages) -> Result<(), RhiError> {
        let checks = [
            (
                self.uniform_buffers,
                limits.max_per_stage_uniform_buffers,
                "uniform buffers",
            ),
            (
                self.storage_buffers,
                limits.max_per_stage_storage_buffers,
                "storage buffers",
            ),
            (
                self.sampled_textures,
                limits.max_per_stage_sampled_textures,
                "sampled textures",
            ),
            (
                self.storage_textures,
                limits.max_per_stage_storage_textures,
                "storage textures",
            ),
        ];
        for (count, limit, kind) in checks {
            if count > limit {
                return Err(RhiError::Unsupported(format!(
                    "{stage:?} uses {count} {kind}; the device allows {limit} per stage"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(entries: Vec<BindGroupLayoutEntry>) -> BindGroupLayoutDescriptor {
        BindGroupLayoutDescriptor {
            entries,
            label: None,
        }
    }

    fn entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            ty,
            stages: ShaderStages::FRAGMENT,
        }
    }

    #[test]
    fn layouts_reject_duplicate_bindings_and_too_many_dynamic_offsets() {
        let ubo = BindingType::UniformBuffer {
            dynamic_offset: true,
        };
        assert!(layout(vec![entry(0, ubo), entry(1, ubo)])
            .validate()
            .is_ok());
        assert!(layout(vec![entry(0, ubo), entry(0, ubo)])
            .validate()
            .is_err());
        let many = (0..9).map(|b| entry(b, ubo)).collect();
        assert!(layout(many).validate().is_err());
    }

    #[test]
    fn groups_must_match_their_layout() {
        let declared = layout(vec![
            entry(
                0,
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                },
            ),
            entry(1, BindingType::SampledTexture),
        ]);
        let buffer = BindingResource::Buffer {
            buffer: BufferHandle::new(1, 1),
            offset: 0,
            size: 16,
        };
        let texture = BindingResource::SampledTexture {
            texture: TextureHandle::new(2, 1).into(),
            sampler: SamplerHandle::new(3, 1),
            layout: ImageLayout::ShaderReadOnly,
        };
        let group = |entries| BindGroupDescriptor {
            layout: BindGroupLayoutHandle::new(1, 1),
            entries,
            label: None,
        };
        let ok = group(vec![
            BindGroupEntry {
                binding: 0,
                resource: buffer,
            },
            BindGroupEntry {
                binding: 1,
                resource: texture,
            },
        ]);
        assert!(validate_bind_group(&declared, &ok).is_ok());
        let swapped = group(vec![
            BindGroupEntry {
                binding: 0,
                resource: texture,
            },
            BindGroupEntry {
                binding: 1,
                resource: buffer,
            },
        ]);
        assert!(validate_bind_group(&declared, &swapped).is_err());
        let missing = group(vec![BindGroupEntry {
            binding: 0,
            resource: buffer,
        }]);
        assert!(validate_bind_group(&declared, &missing).is_err());
    }

    #[test]
    fn stage_counts_follow_stage_visibility() {
        let declared = layout(vec![
            BindGroupLayoutEntry {
                binding: 0,
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                },
                stages: ShaderStages::VERTEX_FRAGMENT,
            },
            BindGroupLayoutEntry {
                binding: 1,
                ty: BindingType::SampledTexture,
                stages: ShaderStages::FRAGMENT,
            },
        ]);
        let vertex = StageBindingCounts::of([&declared], ShaderStages::VERTEX);
        assert_eq!((vertex.uniform_buffers, vertex.sampled_textures), (1, 0));
        let fragment = StageBindingCounts::of([&declared], ShaderStages::FRAGMENT);
        assert_eq!(
            (fragment.uniform_buffers, fragment.sampled_textures),
            (1, 1)
        );
    }
}
