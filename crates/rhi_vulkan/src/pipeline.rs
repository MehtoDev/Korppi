//! Pipelines with shared Vulkan objects and extended dynamic state (finding RVK-16).
//!
//! - **Pipeline layouts** are shared by every pipeline with the same set-layout structure and
//!   push-constant ranges (identically defined layouts are compatible).
//! - **`VkPipeline`s** are shared by RHI pipelines that differ only in state Vulkan 1.3 makes
//!   dynamic: cull mode, front face, depth test/write/compare, depth bias, stencil test, ops and
//!   masks. Binding a pipeline sets that state ([`GraphicsDynamicState`]), so render graphs that
//!   vary it do not multiply PSOs.
//!
//! Both are reference-counted by the RHI pipelines using them and destroyed (after the GPU is
//! done) with the last one. Shaders are keyed by handle, whose generation changes when a shader
//! is destroyed, never by raw module handles, which drivers may reuse.

use crate::device::{lock, Garbage, VulkanDevice};
use crate::resource::VulkanPipeline;
use crate::types::{
    to_vk_color_blend_attachment, to_vk_compare_op, to_vk_cull_mode, to_vk_front_face,
    to_vk_input_rate, to_vk_polygon_mode, to_vk_stencil_face, to_vk_topology, to_vk_vertex_format,
    vk_push_constant_ranges, MAX_PUSH_CONSTANT_RANGES,
};
use ash::vk;
use korppi_rhi::{
    BindGroupLayoutHandle, ColorTargetState, ComputePipelineDescriptor, GraphicsPipelineDescriptor,
    MultisampleState, PipelineHandle, PipelineSignature, PolygonMode, PrimitiveTopology,
    PushConstantRange, RhiError, ShaderHandle, ShaderStages, TextureFormat, VertexLayoutDescriptor,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Structure of a pipeline layout: set-layout fingerprints and push-constant ranges.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayoutKey {
    sets: Vec<u64>,
    push_constants: Vec<(u32, u32, u32)>,
}

/// The state baked into a graphics `VkPipeline`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GraphicsKey {
    vertex: ShaderHandle,
    fragment: Option<ShaderHandle>,
    layout: LayoutKey,
    vertex_layout: VertexLayoutDescriptor,
    topology: PrimitiveTopology,
    polygon_mode: PolygonMode,
    depth_clamp: bool,
    multisample: MultisampleState,
    color_targets: Vec<ColorTargetState>,
    depth_format: Option<TextureFormat>,
}

/// What a `VkPipeline` is shared by.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PipelineKey {
    Graphics(GraphicsKey),
    Compute(ShaderHandle, LayoutKey),
}

/// Reference-counted shared layouts and pipelines.
#[derive(Default)]
pub(crate) struct PipelineObjects {
    layouts: Mutex<HashMap<LayoutKey, (vk::PipelineLayout, u32)>>,
    pipelines: Mutex<HashMap<PipelineKey, (vk::Pipeline, u32)>>,
}

impl PipelineObjects {
    /// Distinct `VkPipeline`s.
    pub(crate) fn pipeline_count(&self) -> usize {
        lock(&self.pipelines).len()
    }

    /// Distinct `VkPipelineLayout`s.
    pub(crate) fn layout_count(&self) -> usize {
        lock(&self.layouts).len()
    }

    /// Every object, for device teardown.
    pub(crate) fn drain(&self) -> (Vec<vk::Pipeline>, Vec<vk::PipelineLayout>) {
        (
            lock(&self.pipelines).drain().map(|(_, (p, _))| p).collect(),
            lock(&self.layouts).drain().map(|(_, (l, _))| l).collect(),
        )
    }
}

/// A shared pipeline layout with what pipelines record about it.
struct AcquiredLayout {
    layout: vk::PipelineLayout,
    key: LayoutKey,
    set_count: u32,
    infos: Vec<korppi_rhi::BindGroupLayoutInfo>,
    push_ranges: Arc<[vk::PushConstantRange]>,
}

/// State of a graphics pipeline that is dynamic in Vulkan 1.3, set when the pipeline is bound.
#[derive(Copy, Clone, Debug)]
pub(crate) struct GraphicsDynamicState {
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    depth_test: bool,
    depth_write: bool,
    depth_compare: vk::CompareOp,
    depth_bias: bool,
    bias_constant: f32,
    bias_clamp: f32,
    bias_slope: f32,
    stencil_test: bool,
    front: vk::StencilOpState,
    back: vk::StencilOpState,
}

/// Graphics state that is dynamic on every pipeline.
pub(crate) const DYNAMIC_STATES: [vk::DynamicState; 15] = [
    vk::DynamicState::VIEWPORT,
    vk::DynamicState::SCISSOR,
    vk::DynamicState::STENCIL_REFERENCE,
    vk::DynamicState::BLEND_CONSTANTS,
    vk::DynamicState::CULL_MODE,
    vk::DynamicState::FRONT_FACE,
    vk::DynamicState::DEPTH_TEST_ENABLE,
    vk::DynamicState::DEPTH_WRITE_ENABLE,
    vk::DynamicState::DEPTH_COMPARE_OP,
    vk::DynamicState::DEPTH_BIAS_ENABLE,
    vk::DynamicState::DEPTH_BIAS,
    vk::DynamicState::STENCIL_TEST_ENABLE,
    vk::DynamicState::STENCIL_OP,
    vk::DynamicState::STENCIL_COMPARE_MASK,
    vk::DynamicState::STENCIL_WRITE_MASK,
];

impl GraphicsDynamicState {
    fn of(desc: &GraphicsPipelineDescriptor) -> Self {
        let r = &desc.rasterizer;
        let (depth_test, depth_write, depth_compare, stencil_test, front, back) =
            match &desc.depth_stencil {
                Some(ds) => (
                    ds.depth_test_enabled,
                    ds.depth_write_enabled,
                    to_vk_compare_op(ds.depth_compare),
                    ds.stencil_enabled(),
                    to_vk_stencil_face(
                        &ds.stencil_front,
                        ds.stencil_read_mask,
                        ds.stencil_write_mask,
                    ),
                    to_vk_stencil_face(
                        &ds.stencil_back,
                        ds.stencil_read_mask,
                        ds.stencil_write_mask,
                    ),
                ),
                None => (
                    false,
                    false,
                    vk::CompareOp::ALWAYS,
                    false,
                    vk::StencilOpState::default(),
                    vk::StencilOpState::default(),
                ),
            };
        Self {
            cull_mode: to_vk_cull_mode(r.cull_mode),
            front_face: to_vk_front_face(r.front_face),
            depth_test,
            depth_write,
            depth_compare,
            depth_bias: r.depth_bias != 0.0 || r.slope_scaled_depth_bias != 0.0,
            bias_constant: r.depth_bias,
            bias_clamp: r.depth_bias_clamp,
            bias_slope: r.slope_scaled_depth_bias,
            stencil_test,
            front,
            back,
        }
    }

    /// Records the state.
    ///
    /// # Safety
    /// `cmd` must be recording, with a graphics pipeline of this device bound.
    pub(crate) unsafe fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        // SAFETY: forwarded from the caller; every state is dynamic on every graphics pipeline
        // (DYNAMIC_STATES), and a non-zero bias clamp only exists where the feature is enabled.
        unsafe {
            device.cmd_set_cull_mode(cmd, self.cull_mode);
            device.cmd_set_front_face(cmd, self.front_face);
            device.cmd_set_depth_test_enable(cmd, self.depth_test);
            device.cmd_set_depth_write_enable(cmd, self.depth_write);
            device.cmd_set_depth_compare_op(cmd, self.depth_compare);
            device.cmd_set_depth_bias_enable(cmd, self.depth_bias);
            device.cmd_set_depth_bias(cmd, self.bias_constant, self.bias_clamp, self.bias_slope);
            device.cmd_set_stencil_test_enable(cmd, self.stencil_test);
            for (face, s) in [
                (vk::StencilFaceFlags::FRONT, &self.front),
                (vk::StencilFaceFlags::BACK, &self.back),
            ] {
                device.cmd_set_stencil_op(
                    cmd,
                    face,
                    s.fail_op,
                    s.pass_op,
                    s.depth_fail_op,
                    s.compare_op,
                );
                device.cmd_set_stencil_compare_mask(cmd, face, s.compare_mask);
                device.cmd_set_stencil_write_mask(cmd, face, s.write_mask);
            }
        }
    }
}

fn layout_key(
    sets: &[korppi_rhi::BindGroupLayoutInfo],
    push: &[vk::PushConstantRange],
) -> LayoutKey {
    LayoutKey {
        sets: sets.iter().map(|s| s.fingerprint).collect(),
        push_constants: push
            .iter()
            .map(|r| (r.stage_flags.as_raw(), r.offset, r.size))
            .collect(),
    }
}

impl VulkanDevice {
    /// The shared pipeline layout for `handles` and `push`, created if needed; takes a
    /// reference. Also returns the set count and layout summaries.
    fn acquire_layout(
        &self,
        handles: &[BindGroupLayoutHandle],
        stages: ShaderStages,
        push: &[PushConstantRange],
    ) -> Result<AcquiredLayout, RhiError> {
        let push_ranges = vk_push_constant_ranges(push);
        if push_ranges.len() > MAX_PUSH_CONSTANT_RANGES {
            return Err(RhiError::ValidationError(format!(
                "{} push constant ranges exceed the supported {MAX_PUSH_CONSTANT_RANGES}",
                push_ranges.len()
            )));
        }
        let (set_layouts, infos) = self.pipeline_set_layouts(handles, stages)?;
        let key = layout_key(&infos, &push_ranges);
        let mut layouts = lock(&self.pipeline_objects.layouts);
        let layout = match layouts.get_mut(&key) {
            Some((layout, users)) => {
                *users += 1;
                *layout
            }
            None => {
                let info = vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges);
                // SAFETY: the set layouts are live (looked up above) and `info` is valid.
                let layout = unsafe { self.context.device.create_pipeline_layout(&info, None) }
                    .map_err(|e| crate::error::vk_error("vkCreatePipelineLayout", e))?;
                layouts.insert(key.clone(), (layout, 1));
                layout
            }
        };
        Ok(AcquiredLayout {
            layout,
            key,
            set_count: set_layouts.len() as u32,
            infos,
            push_ranges: push_ranges.into(),
        })
    }

    /// Drops a reference to a shared layout, destroying it with the last.
    fn release_layout(&self, key: &LayoutKey) -> Result<(), RhiError> {
        let destroyed = {
            let mut layouts = lock(&self.pipeline_objects.layouts);
            match layouts.get_mut(key) {
                Some((_, users)) if *users > 1 => {
                    *users -= 1;
                    None
                }
                Some(_) => layouts.remove(key).map(|(layout, _)| layout),
                None => None,
            }
        };
        match destroyed {
            Some(layout) => self.defer_destruction(Garbage::PipelineLayout(layout)),
            None => Ok(()),
        }
    }

    /// The shared pipeline for `key`, created with `create` if needed; takes a reference.
    fn acquire_pipeline(
        &self,
        key: &PipelineKey,
        create: impl FnOnce() -> Result<vk::Pipeline, RhiError>,
    ) -> Result<vk::Pipeline, RhiError> {
        if let Some((pipeline, users)) = lock(&self.pipeline_objects.pipelines).get_mut(key) {
            *users += 1;
            return Ok(*pipeline);
        }
        // Created without the lock (compilation is slow); a concurrent creation of the same
        // key keeps the first and destroys its own.
        let pipeline = create()?;
        let mut pipelines = lock(&self.pipeline_objects.pipelines);
        match pipelines.get_mut(key) {
            Some((existing, users)) => {
                *users += 1;
                let existing = *existing;
                drop(pipelines);
                // SAFETY: the duplicate was just created and is unused.
                unsafe { self.context.device.destroy_pipeline(pipeline, None) };
                Ok(existing)
            }
            None => {
                pipelines.insert(key.clone(), (pipeline, 1));
                Ok(pipeline)
            }
        }
    }

    /// Drops a reference to a shared pipeline, destroying it with the last.
    fn release_pipeline(&self, key: &PipelineKey) -> Result<(), RhiError> {
        let destroyed = {
            let mut pipelines = lock(&self.pipeline_objects.pipelines);
            match pipelines.get_mut(key) {
                Some((_, users)) if *users > 1 => {
                    *users -= 1;
                    None
                }
                Some(_) => pipelines.remove(key).map(|(pipeline, _)| pipeline),
                None => None,
            }
        };
        match destroyed {
            Some(pipeline) => self.defer_destruction(Garbage::Pipeline(pipeline)),
            None => Ok(()),
        }
    }

    /// The module and entry point of a live `stage` shader.
    fn shader_stage(
        &self,
        handle: ShaderHandle,
        what: &str,
    ) -> Result<(vk::ShaderModule, std::ffi::CString), RhiError> {
        lock(&self.resources.shaders)
            .get(handle)
            .map(|(s, _)| (s.module, s.entry_point.clone()))
            .ok_or_else(|| RhiError::InvalidHandle(format!("{what} shader {handle:?} not found")))
    }

    pub(crate) fn create_graphics_pipeline_impl(
        &self,
        desc: &GraphicsPipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        desc.validate()?;
        let missing = desc.rasterizer.required_features() - self.context.capabilities.features;
        if !missing.is_empty() {
            return Err(RhiError::Unsupported(format!(
                "the rasterizer state needs {missing:?}, which this device lacks"
            )));
        }
        let depth_format = match &desc.depth_stencil {
            Some(ds) if !ds.format.is_depth() => {
                return Err(RhiError::ValidationError(format!(
                    "DepthStencilState::format {:?} is not a depth format",
                    ds.format
                )))
            }
            Some(ds) => Some(ds.format),
            None => None,
        };
        let stages = if desc.fragment_shader.is_some() {
            ShaderStages::VERTEX_FRAGMENT
        } else {
            ShaderStages::VERTEX
        };
        let vertex = self.shader_stage(desc.vertex_shader, "vertex")?;
        let fragment = desc
            .fragment_shader
            .map(|handle| self.shader_stage(handle, "fragment"))
            .transpose()?;
        let AcquiredLayout {
            layout,
            key: layout_key,
            set_count,
            infos,
            push_ranges,
        } = self.acquire_layout(&desc.bind_group_layouts, stages, &desc.push_constant_ranges)?;
        let key = PipelineKey::Graphics(GraphicsKey {
            vertex: desc.vertex_shader,
            fragment: desc.fragment_shader,
            layout: layout_key.clone(),
            vertex_layout: desc.vertex_layout.clone(),
            topology: desc.topology,
            polygon_mode: desc.rasterizer.polygon_mode,
            depth_clamp: desc.rasterizer.depth_clamp,
            multisample: desc.multisample,
            color_targets: desc.color_targets.clone(),
            depth_format,
        });
        let pipeline = self.acquire_pipeline(&key, || {
            self.build_graphics_pipeline(desc, layout, &vertex, fragment.as_ref(), depth_format)
        });
        let pipeline = match pipeline {
            Ok(pipeline) => pipeline,
            Err(error) => {
                self.release_layout(&layout_key)?;
                return Err(error);
            }
        };
        self.set_object_name(pipeline, desc.label);
        let (id, generation) = lock(&self.resources.pipelines).insert(VulkanPipeline {
            pipeline,
            layout,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            set_count,
            signature: PipelineSignature::graphics(desc).with_bind_group_layouts(&infos),
            push_constant_ranges: push_ranges,
            dynamic: Some(GraphicsDynamicState::of(desc)),
            key,
            layout_key,
        });
        Ok(PipelineHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    fn build_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDescriptor,
        layout: vk::PipelineLayout,
        vertex: &(vk::ShaderModule, std::ffi::CString),
        fragment: Option<&(vk::ShaderModule, std::ffi::CString)>,
        depth_format: Option<TextureFormat>,
    ) -> Result<vk::Pipeline, RhiError> {
        let bindings: Vec<vk::VertexInputBindingDescription> = desc
            .vertex_layout
            .bindings
            .iter()
            .map(|b| vk::VertexInputBindingDescription {
                binding: b.binding,
                stride: b.stride as u32,
                input_rate: to_vk_input_rate(b.step_mode),
            })
            .collect();
        let attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .vertex_layout
            .attributes
            .iter()
            .map(|a| vk::VertexInputAttributeDescription {
                location: a.location,
                binding: a.binding,
                format: to_vk_vertex_format(a.format),
                offset: a.offset as u32,
            })
            .collect();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(to_vk_topology(desc.topology))
            .primitive_restart_enable(false);
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&DYNAMIC_STATES);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // Culling, front face and depth bias are dynamic; only what stays baked in is set.
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(desc.rasterizer.depth_clamp)
            .rasterizer_discard_enable(false)
            .polygon_mode(to_vk_polygon_mode(desc.rasterizer.polygon_mode))
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(crate::texture::sample_flags(desc.multisample.count))
            .alpha_to_coverage_enable(desc.multisample.alpha_to_coverage);
        // Depth and stencil tests are dynamic.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = desc
            .color_targets
            .iter()
            .map(to_vk_color_blend_attachment)
            .collect();
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
        let mut stages = vec![vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex.0)
            .name(&vertex.1)];
        if let Some((module, entry)) = fragment {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(*module)
                    .name(entry),
            );
        }
        let color_formats: Vec<vk::Format> = desc
            .color_targets
            .iter()
            .map(|ct| self.context.capabilities.vk_format(ct.format))
            .collect();
        let vk_depth = depth_format.map_or(vk::Format::UNDEFINED, |f| {
            self.context.capabilities.vk_format(f)
        });
        // Combined depth/stencil formats render to the stencil aspect too.
        let vk_stencil = match depth_format {
            Some(f) if f.is_stencil() => vk_depth,
            _ => vk::Format::UNDEFINED,
        };
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(vk_depth)
            .stencil_attachment_format(vk_stencil);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rendering)
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(vk::RenderPass::null());
        // SAFETY: every referenced object is live and the create info and its chain outlive the
        // call; the cache is read-locked against merges.
        let pipelines = unsafe {
            self.context.device.create_graphics_pipelines(
                *self.pipeline_cache.read(),
                &[info],
                None,
            )
        }
        .map_err(|(_, e)| crate::error::vk_error("vkCreateGraphicsPipelines", e))?;
        Ok(pipelines[0])
    }

    pub(crate) fn create_compute_pipeline_impl(
        &self,
        desc: &ComputePipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        desc.validate()?;
        let (module, entry) = self.shader_stage(desc.compute_shader, "compute")?;
        let AcquiredLayout {
            layout,
            key: layout_key,
            set_count,
            infos,
            push_ranges,
        } = self.acquire_layout(
            &desc.bind_group_layouts,
            ShaderStages::COMPUTE,
            &desc.push_constant_ranges,
        )?;
        let key = PipelineKey::Compute(desc.compute_shader, layout_key.clone());
        let pipeline = self.acquire_pipeline(&key, || {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            // SAFETY: the module and layout are live; the cache is read-locked against merges.
            unsafe {
                self.context.device.create_compute_pipelines(
                    *self.pipeline_cache.read(),
                    &[info],
                    None,
                )
            }
            .map(|pipelines| pipelines[0])
            .map_err(|(_, e)| crate::error::vk_error("vkCreateComputePipelines", e))
        });
        let pipeline = match pipeline {
            Ok(pipeline) => pipeline,
            Err(error) => {
                self.release_layout(&layout_key)?;
                return Err(error);
            }
        };
        self.set_object_name(pipeline, desc.label);
        let (id, generation) = lock(&self.resources.pipelines).insert(VulkanPipeline {
            pipeline,
            layout,
            bind_point: vk::PipelineBindPoint::COMPUTE,
            set_count,
            signature: PipelineSignature::compute(desc).with_bind_group_layouts(&infos),
            push_constant_ranges: push_ranges,
            dynamic: None,
            key,
            layout_key,
        });
        Ok(PipelineHandle::from_parts(
            self.resources.device,
            id,
            generation,
        ))
    }

    pub(crate) fn destroy_pipeline_impl(&self, handle: PipelineHandle) -> Result<(), RhiError> {
        let (pipeline, _) = lock(&self.resources.pipelines)
            .remove(handle)
            .ok_or_else(|| {
                RhiError::InvalidHandle(format!("Pipeline handle {handle:?} not found"))
            })?;
        self.release_pipeline(&pipeline.key)?;
        self.release_layout(&pipeline.layout_key)
    }

    /// Number of distinct `VkPipeline`s, shared by RHI pipelines that differ only in dynamic
    /// state (RVK-16).
    pub fn pipeline_object_count(&self) -> usize {
        self.pipeline_objects.pipeline_count()
    }

    /// Number of distinct `VkPipelineLayout`s.
    pub fn pipeline_layout_count(&self) -> usize {
        self.pipeline_objects.layout_count()
    }
}
