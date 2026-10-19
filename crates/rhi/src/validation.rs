//! Validation of command recording, in every build (findings RHI-12, RHI-16).
//!
//! [`ValidatingRecorder`] wraps a backend's [`CommandBufferRecorder`], forwards every call, and
//! records misuse that the backends would otherwise ignore silently or turn into undefined
//! behaviour. [`CommandBufferRecorder::finish`] then returns [`RhiError::ValidationError`]
//! listing every problem. Backends wrap every recorder with [`validate_recorder`] or
//! [`validate_recorder_pooled`], whatever [`crate::api_validation_requested`] says: safe Rust
//! must not reach undefined behaviour in release builds (RHI-16). The costly layer-backed
//! validation (the Khronos layer, GL debug output) stays optional.
//!
//! Checks performed:
//! - commands recorded before `begin`, or `begin` while already recording;
//! - `begin_rendering` inside a pass, `end_rendering` outside one, `finish` with a pass active;
//! - draws outside a pass or without a graphics pipeline; dispatches inside a pass or without a
//!   compute pipeline; `draw_indexed` without an index buffer;
//! - pipeline colour-target formats, depth presence or sample count that differ from the active
//!   pass;
//! - attachment mip levels and layers outside the texture, attachments with different sample
//!   counts, and resolve targets that are multisampled, of another format, or of a
//!   single-sample attachment;
//! - push constants outside the bound pipeline's declared ranges;
//! - bind groups: set indices beyond [`crate::MAX_BIND_GROUPS`], dynamic offsets of the wrong
//!   count, misaligned or past the buffer's end, groups whose resources were destroyed, and
//!   draws/dispatches without a compatible group for every set the pipeline declares;
//! - image barrier ranges outside the texture or naming an aspect the format lacks;
//! - copies, blits, mip generation and barriers inside a pass (Vulkan forbids them without
//!   extra features); transfer regions outside the texture, misaligned or past the buffer's end,
//!   row pitches that are not whole texel blocks or shorter than a row, missing `COPY_SRC` /
//!   `COPY_DST` usage, copies between different formats, blits the format cannot do; buffer to
//!   buffer copies that are empty, reach past either buffer or overlap within one buffer
//!   (`crate::transfer`);
//! - indirect commands: argument ranges (alignment, stride, buffer end), `INDIRECT` usage,
//!   multi-draw and first-instance features, indexed indirect draws without an index buffer at
//!   offset 0; timestamps and query resolves outside the query set; unbalanced debug groups;
//! - unknown buffer, texture, sampler and pipeline handles.
//!
//! The valid path performs no heap allocation; messages are only built when a check fails.

use crate::binding::{BindGroupInfo, BindGroupLayoutInfo, MAX_BIND_GROUPS};
use crate::command::{CommandBuffer, CommandBufferRecorder};
use crate::descriptor::{
    BufferTextureCopy, ComputePipelineDescriptor, GraphicsPipelineDescriptor, PushConstantRange,
    RenderingDescriptor,
};
use crate::error::RhiError;
use crate::handle::{BindGroupHandle, BufferHandle, PipelineHandle, SamplerHandle, TextureHandle};
use crate::sync::PipelineBarrier;
use crate::types::{IndexType, SampleCount, TextureDimension, TextureFormat, TextureUsage};
use std::sync::Arc;

/// Colour targets tracked per pipeline and pass. Larger pipelines/passes are only checked for
/// their attachment count and first targets.
pub const MAX_VALIDATED_COLOR_TARGETS: usize = 8;
/// Push-constant ranges tracked per pipeline. Push constants of pipelines with more ranges are
/// not validated.
pub const MAX_VALIDATED_PUSH_CONSTANT_RANGES: usize = 8;
/// Validation messages kept per command buffer; further failures are only counted.
const MAX_MESSAGES: usize = 32;

/// Whether a pipeline is used for drawing or dispatching.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PipelineKind {
    /// Graphics pipeline.
    Graphics,
    /// Compute pipeline.
    Compute,
}

/// Fixed-size summary of a pipeline that recording validation checks against.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PipelineSignature {
    /// Graphics or compute.
    pub kind: PipelineKind,
    color_formats: [Option<TextureFormat>; MAX_VALIDATED_COLOR_TARGETS],
    color_count: usize,
    depth_format: Option<TextureFormat>,
    samples: SampleCount,
    push_ranges: [Option<PushConstantRange>; MAX_VALIDATED_PUSH_CONSTANT_RANGES],
    push_ranges_complete: bool,
    bind_group_layouts: [Option<BindGroupLayoutInfo>; MAX_BIND_GROUPS],
}

impl PipelineSignature {
    /// Signature of a graphics pipeline.
    pub fn graphics(desc: &GraphicsPipelineDescriptor) -> Self {
        let mut color_formats = [None; MAX_VALIDATED_COLOR_TARGETS];
        for (slot, target) in color_formats.iter_mut().zip(&desc.color_targets) {
            *slot = Some(target.format);
        }
        let (push_ranges, push_ranges_complete) = push_ranges(&desc.push_constant_ranges);
        Self {
            kind: PipelineKind::Graphics,
            color_formats,
            color_count: desc.color_targets.len(),
            depth_format: desc.depth_stencil.as_ref().map(|ds| ds.format),
            samples: desc.multisample.count,
            push_ranges,
            push_ranges_complete,
            bind_group_layouts: [None; MAX_BIND_GROUPS],
        }
    }

    /// Signature of a compute pipeline.
    pub fn compute(desc: &ComputePipelineDescriptor) -> Self {
        let (push_ranges, push_ranges_complete) = push_ranges(&desc.push_constant_ranges);
        Self {
            kind: PipelineKind::Compute,
            color_formats: [None; MAX_VALIDATED_COLOR_TARGETS],
            color_count: 0,
            depth_format: None,
            samples: SampleCount::Count1,
            push_ranges,
            push_ranges_complete,
            bind_group_layouts: [None; MAX_BIND_GROUPS],
        }
    }

    /// The signature with the pipeline's bind group layouts (sets 0, 1, ...); layouts beyond
    /// [`MAX_BIND_GROUPS`] are ignored (pipeline creation rejects them).
    pub fn with_bind_group_layouts(mut self, layouts: &[BindGroupLayoutInfo]) -> Self {
        for (slot, layout) in self.bind_group_layouts.iter_mut().zip(layouts) {
            *slot = Some(*layout);
        }
        self
    }

    /// True if every byte in `offset..offset + size` lies in a declared push-constant range.
    /// Always true when the pipeline declared more ranges than are tracked.
    pub fn covers_push_constants(&self, offset: u32, size: u32) -> bool {
        if !self.push_ranges_complete {
            return true;
        }
        (offset..offset.saturating_add(size)).all(|byte| {
            self.push_ranges
                .iter()
                .flatten()
                .any(|r| byte >= r.offset && byte < r.offset.saturating_add(r.size))
        })
    }
}

fn push_ranges(
    ranges: &[PushConstantRange],
) -> (
    [Option<PushConstantRange>; MAX_VALIDATED_PUSH_CONSTANT_RANGES],
    bool,
) {
    let mut out = [None; MAX_VALIDATED_PUSH_CONSTANT_RANGES];
    for (slot, range) in out.iter_mut().zip(ranges) {
        *slot = Some(*range);
    }
    (out, ranges.len() <= MAX_VALIDATED_PUSH_CONSTANT_RANGES)
}

/// What recording validation needs to know about a texture.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureInfo {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Texel format.
    pub format: TextureFormat,
    /// Dimensionality.
    pub dimension: TextureDimension,
    /// Mip levels.
    pub mip_levels: u32,
    /// Array layers.
    pub array_layers: u32,
    /// Depth (3D textures).
    pub depth: u32,
    /// Samples per texel.
    pub sample_count: SampleCount,
    /// Allowed usage.
    pub usage: TextureUsage,
}

impl TextureInfo {
    /// Summary of a texture created from `desc`.
    pub fn from_descriptor(desc: &crate::descriptor::TextureDescriptor) -> Self {
        Self {
            width: desc.width,
            height: desc.height,
            format: desc.format,
            dimension: desc.dimension,
            mip_levels: desc.mip_levels,
            array_layers: desc.array_layers,
            depth: desc.depth,
            sample_count: desc.sample_count,
            usage: desc.usage,
        }
    }

    /// True if layers of this texture can be viewed as cube faces: square 2D layers, at least 6.
    /// Backends create such textures cube-compatible.
    pub fn cube_compatible(&self) -> bool {
        matches!(
            self.dimension,
            TextureDimension::D2Array | TextureDimension::Cube | TextureDimension::CubeArray
        ) && self.array_layers >= 6
            && self.width == self.height
            && self.sample_count == SampleCount::Count1
    }

    /// Layers an attachment of mip `mip` can select: array layers, or slices of a 3D texture.
    pub fn attachment_layers(&self, mip: u32) -> u32 {
        if self.dimension == TextureDimension::D3 {
            (self.depth >> mip).max(1)
        } else {
            self.array_layers
        }
    }
}

/// What recording validation knows of a live buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BufferSummary {
    /// Size in bytes, if known (buffer ranges are then checked).
    pub size: Option<u64>,
    /// Usage, if known (usage checks).
    pub usage: Option<crate::types::BufferUsage>,
}

/// Read-only view of a backend's resource registry used by recording validation.
pub trait ResourceOracle: Send + Sync {
    /// True if `handle` names a live buffer.
    fn buffer_exists(&self, handle: BufferHandle) -> bool;
    /// Size and usage of a live buffer, or `None` if `handle` is unknown. Validation asks this
    /// once per buffer a command names, so backends answer it with one registry lookup.
    fn buffer_summary(&self, handle: BufferHandle) -> Option<BufferSummary> {
        self.buffer_exists(handle).then(|| BufferSummary {
            size: self.buffer_size(handle),
            usage: self.buffer_usage(handle),
        })
    }
    /// Description of a live texture, or `None` if `handle` is unknown.
    fn texture_info(&self, handle: TextureHandle) -> Option<TextureInfo>;
    /// Format of a live texture, or `None` if `handle` is unknown.
    fn texture_format(&self, handle: TextureHandle) -> Option<TextureFormat> {
        self.texture_info(handle).map(|info| info.format)
    }
    /// True if `handle` names a live sampler.
    fn sampler_exists(&self, handle: SamplerHandle) -> bool;
    /// Signature of a live pipeline, or `None` if `handle` is unknown.
    fn pipeline_signature(&self, handle: PipelineHandle) -> Option<PipelineSignature>;
    /// Summary of a live bind group, or `None` if `handle` is unknown.
    fn bind_group_info(&self, handle: BindGroupHandle) -> Option<BindGroupInfo>;
    /// Size in bytes of a live buffer, if known (buffer ranges are then checked).
    fn buffer_size(&self, _handle: BufferHandle) -> Option<u64> {
        None
    }
    /// Usage of a live buffer, if known (usage checks of indirect and query commands).
    fn buffer_usage(&self, _handle: BufferHandle) -> Option<crate::types::BufferUsage> {
        None
    }
    /// The device's optional features; all by default.
    fn features(&self) -> crate::capabilities::DeviceFeatures {
        crate::capabilities::DeviceFeatures::all()
    }
    /// Number of queries in a live query set, or `None` if `handle` is unknown.
    fn query_set_count(&self, handle: crate::handle::QuerySetHandle) -> Option<u32>;
    /// What `format` supports on the device (blit and mip-generation checks); all by default.
    fn format_capabilities(
        &self,
        _format: TextureFormat,
    ) -> crate::capabilities::FormatCapabilities {
        crate::capabilities::FormatCapabilities::all()
    }
}

/// Wraps `inner` in a [`ValidatingRecorder`] that checks recording against `oracle`.
pub fn validate_recorder(
    inner: Box<dyn CommandBufferRecorder>,
    oracle: Arc<dyn ResourceOracle>,
) -> Box<dyn CommandBufferRecorder> {
    Box::new(ValidatingRecorder::new(inner, oracle))
}

/// Like [`validate_recorder`], reusing a validator from `pool`; `finish` returns it there, so
/// steady-state frames allocate no validators (RHI-11).
pub fn validate_recorder_pooled(
    inner: Box<dyn CommandBufferRecorder>,
    oracle: Arc<dyn ResourceOracle>,
    pool: &Arc<crate::pool::BoxPool<ValidatingRecorder>>,
) -> Box<dyn CommandBufferRecorder> {
    let mut validator = match pool.take() {
        Some(mut validator) => {
            let mut messages = std::mem::take(&mut validator.messages);
            messages.clear();
            *validator = ValidatingRecorder::new(inner, oracle);
            validator.messages = messages;
            validator
        }
        None => Box::new(ValidatingRecorder::new(inner, oracle)),
    };
    validator.pool = Arc::downgrade(pool);
    validator
}

#[derive(Copy, Clone)]
struct PassState {
    color_formats: [Option<TextureFormat>; MAX_VALIDATED_COLOR_TARGETS],
    color_count: usize,
    depth_format: Option<TextureFormat>,
    samples: Option<SampleCount>,
}

/// Command recorder wrapper that validates recording (see the module documentation).
///
/// Commands that fail validation are **not** forwarded to the backend, so the driver never
/// records invalid state; `finish()` reports them all.
pub struct ValidatingRecorder {
    /// The backend's recorder; `None` while the validator waits in its pool.
    inner: Option<Box<dyn CommandBufferRecorder>>,
    /// Where `finish` returns this box (RHI-11).
    pool: std::sync::Weak<crate::pool::BoxPool<ValidatingRecorder>>,
    oracle: Arc<dyn ResourceOracle>,
    recording: bool,
    command_index: u32,
    pass: Option<PassState>,
    pipeline: Option<PipelineSignature>,
    bind_groups: [Option<BindGroupInfo>; MAX_BIND_GROUPS],
    /// Offset of the bound index buffer.
    index_buffer: Option<u64>,
    debug_groups: u32,
    viewport_set: bool,
    scissor_set: bool,
    messages: Vec<String>,
    suppressed: usize,
}

impl ValidatingRecorder {
    /// Wraps `inner`; see [`validate_recorder`].
    pub fn new(inner: Box<dyn CommandBufferRecorder>, oracle: Arc<dyn ResourceOracle>) -> Self {
        Self {
            inner: Some(inner),
            pool: std::sync::Weak::new(),
            oracle,
            recording: false,
            command_index: 0,
            pass: None,
            pipeline: None,
            bind_groups: [None; MAX_BIND_GROUPS],
            index_buffer: None,
            debug_groups: 0,
            viewport_set: false,
            scissor_set: false,
            messages: Vec::new(),
            suppressed: 0,
        }
    }

    /// The backend's recorder.
    fn inner(&mut self) -> &mut dyn CommandBufferRecorder {
        self.inner
            .as_deref_mut()
            .expect("a validator in use wraps a recorder")
    }

    /// Total failures recorded so far (kept and suppressed).
    fn failures(&self) -> usize {
        self.messages.len() + self.suppressed
    }

    fn fail(&mut self, command: &str, problem: impl FnOnce() -> String) {
        if self.messages.len() < MAX_MESSAGES {
            self.messages.push(format!(
                "command #{} `{command}`: {}",
                self.command_index,
                problem()
            ));
        } else {
            self.suppressed += 1;
        }
    }

    /// Bookkeeping shared by every recorded command. Returns the failure count before the
    /// command's own checks, for [`ValidatingRecorder::passed`].
    fn command(&mut self, name: &str) -> usize {
        let before = self.failures();
        self.command_index += 1;
        if !self.recording {
            self.fail(name, || "recorded before `begin`".into());
        }
        before
    }

    /// True if no failure was recorded since `before` (the command may be forwarded).
    fn passed(&self, before: usize) -> bool {
        self.failures() == before
    }

    fn require_outside_pass(&mut self, name: &str) {
        if self.pass.is_some() {
            self.fail(name, || "not allowed inside a render pass".into());
        }
    }

    /// Every set the pipeline declares needs a bound group of a compatible layout.
    fn check_bind_groups(&mut self, name: &str, pipeline: &PipelineSignature) {
        for (set, declared) in pipeline.bind_group_layouts.iter().enumerate() {
            let Some(declared) = declared else {
                continue;
            };
            match self.bind_groups[set] {
                None => self.fail(name, || format!("no bind group is bound at set {set}")),
                Some(bound) if bound.layout.fingerprint != declared.fingerprint => {
                    self.fail(name, || {
                        format!(
                            "the bind group at set {set} has a layout incompatible with the \
                             pipeline's"
                        )
                    })
                }
                Some(_) => {}
            }
        }
    }

    /// An indirect argument buffer: known, with `INDIRECT` usage, and the records in range.
    fn check_indirect(
        &mut self,
        name: &str,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
        record_size: u64,
    ) {
        use crate::capabilities::DeviceFeatures;
        let size = self.check_buffer_usage(
            name,
            buffer,
            crate::types::BufferUsage::INDIRECT,
            "INDIRECT",
        );
        if let Err(problem) =
            crate::query::check_indirect_range(offset, draw_count, stride, record_size, size)
        {
            self.fail(name, || problem);
        }
        if draw_count > 1
            && !self
                .oracle
                .features()
                .contains(DeviceFeatures::MULTI_DRAW_INDIRECT)
        {
            self.fail(name, || {
                "several draws per call need DeviceFeatures::MULTI_DRAW_INDIRECT".into()
            });
        }
    }

    fn check_query_range(
        &mut self,
        name: &str,
        query_set: crate::handle::QuerySetHandle,
        first: u32,
        count: u32,
    ) {
        match self.oracle.query_set_count(query_set) {
            None => self.fail(name, || format!("unknown query set {query_set:?}")),
            Some(size) if first.checked_add(count).is_none_or(|end| end > size) || count == 0 => {
                self.fail(name, || {
                    format!(
                        "queries {first}..{} of a set of {size}",
                        first.saturating_add(count)
                    )
                })
            }
            Some(_) => {}
        }
    }

    /// A buffer a command names must be live; returns its size and usage.
    fn check_buffer(&mut self, name: &str, buffer: BufferHandle) -> Option<BufferSummary> {
        let summary = self.oracle.buffer_summary(buffer);
        if summary.is_none() {
            self.fail(name, || format!("unknown buffer handle {buffer:?}"));
        }
        summary
    }

    /// A buffer a command names must be live and have `needed` usage; returns its size.
    fn check_buffer_usage(
        &mut self,
        name: &str,
        buffer: BufferHandle,
        needed: crate::types::BufferUsage,
        needed_name: &str,
    ) -> Option<u64> {
        let summary = self.check_buffer(name, buffer)?;
        if summary.usage.is_some_and(|usage| !usage.contains(needed)) {
            self.fail(name, || format!("{buffer:?} lacks {needed_name} usage"));
        }
        summary.size
    }

    /// A texture a transfer reads or writes: it must exist and have `usage`.
    fn transfer_texture(
        &mut self,
        name: &str,
        texture: TextureHandle,
        usage: TextureUsage,
    ) -> Option<TextureInfo> {
        let info = self.texture_info(name, texture)?;
        if !info.usage.contains(usage) {
            self.fail(name, || format!("{texture:?} lacks {usage:?} usage"));
        }
        Some(info)
    }

    fn texture_info(&mut self, name: &str, texture: TextureHandle) -> Option<TextureInfo> {
        let info = self.oracle.texture_info(texture);
        if info.is_none() {
            self.fail(name, || format!("unknown texture handle {texture:?}"));
        }
        info
    }

    /// Checks one attachment's usage, subresource, render-area bounds, and that its sample count
    /// matches the pass so far.
    #[allow(clippy::too_many_arguments)]
    fn check_attachment(
        &mut self,
        texture: TextureHandle,
        mip: u32,
        layer: u32,
        needed_usage: TextureUsage,
        offset_x: i32,
        offset_y: i32,
        width: u32,
        height: u32,
        samples: &mut Option<SampleCount>,
    ) -> Option<TextureInfo> {
        let info = self.texture_info("begin_rendering", texture)?;
        if !info.usage.contains(needed_usage) {
            self.fail("begin_rendering", || {
                format!("attachment {texture:?} lacks {needed_usage:?} usage")
            });
        }
        if mip >= info.mip_levels || layer >= info.attachment_layers(mip) {
            self.fail("begin_rendering", || {
                format!(
                    "attachment {texture:?} mip {mip} layer {layer} is outside the texture \
                     ({} mips, {} layers at that mip)",
                    info.mip_levels,
                    info.attachment_layers(mip.min(info.mip_levels.saturating_sub(1)))
                )
            });
        } else if offset_x >= 0 && offset_y >= 0 {
            let mip_width = (info.width >> mip).max(1);
            let mip_height = (info.height >> mip).max(1);
            if (offset_x as u32)
                .checked_add(width)
                .is_none_or(|end_x| end_x > mip_width)
                || (offset_y as u32)
                    .checked_add(height)
                    .is_none_or(|end_y| end_y > mip_height)
            {
                self.fail("begin_rendering", || {
                    format!(
                        "render area ({offset_x}, {offset_y}, {width}x{height}) exceeds attachment \
                         {texture:?} mip {mip} extent ({mip_width}x{mip_height})"
                    )
                });
            }
        }
        match *samples {
            None => *samples = Some(info.sample_count),
            Some(pass) if pass != info.sample_count => self.fail("begin_rendering", || {
                format!(
                    "attachment {texture:?} has {:?}, other attachments {pass:?}",
                    info.sample_count
                )
            }),
            Some(_) => {}
        }
        Some(info)
    }

    fn check_draw(&mut self, name: &str) {
        let Some(pass) = self.pass else {
            self.fail(name, || "recorded outside a render pass".into());
            return;
        };
        if !self.viewport_set {
            self.fail(name, || {
                "no viewport has been set in this command buffer (call `set_viewport`)".into()
            });
        }
        if !self.scissor_set {
            self.fail(name, || {
                "no scissor has been set in this command buffer (call `set_scissor`)".into()
            });
        }
        let Some(pipeline) = self.pipeline else {
            self.fail(name, || "no pipeline is bound".into());
            return;
        };
        if pipeline.kind != PipelineKind::Graphics {
            self.fail(name, || "the bound pipeline is a compute pipeline".into());
            return;
        }
        self.check_bind_groups(name, &pipeline);
        let tracked = pass.color_count.min(MAX_VALIDATED_COLOR_TARGETS);
        if pipeline.color_count != pass.color_count
            || pipeline.color_formats[..tracked] != pass.color_formats[..tracked]
        {
            self.fail(name, || {
                format!(
                    "pipeline colour targets {:?} do not match the pass attachments {:?}",
                    &pipeline.color_formats
                        [..pipeline.color_count.min(MAX_VALIDATED_COLOR_TARGETS)],
                    &pass.color_formats[..tracked]
                )
            });
        }
        if let Some(samples) = pass.samples {
            if pipeline.samples != samples {
                self.fail(name, || {
                    format!(
                        "pipeline multisample count {:?} does not match the pass attachments \
                         ({samples:?})",
                        pipeline.samples
                    )
                });
            }
        }
        if pipeline.depth_format != pass.depth_format {
            let describe = |format: Option<TextureFormat>| match format {
                Some(format) => format!("{format:?}"),
                None => "absent".to_string(),
            };
            self.fail(name, || {
                format!(
                    "pipeline depth state ({}) does not match the pass depth attachment ({})",
                    describe(pipeline.depth_format),
                    describe(pass.depth_format)
                )
            });
        }
    }

    fn report(&mut self) -> Result<(), RhiError> {
        if self.messages.is_empty() && self.suppressed == 0 {
            return Ok(());
        }
        let mut report = std::mem::take(&mut self.messages).join("\n");
        if self.suppressed > 0 {
            report.push_str(&format!("\n... and {} more", self.suppressed));
        }
        Err(RhiError::ValidationError(report))
    }
}

impl CommandBufferRecorder for ValidatingRecorder {
    fn begin(&mut self) -> Result<(), RhiError> {
        self.command_index += 1;
        if self.recording {
            self.fail("begin", || "called while already recording".into());
            return Ok(());
        }
        self.recording = true;
        self.inner().begin()
    }

    fn begin_rendering(&mut self, desc: &RenderingDescriptor) -> Result<(), RhiError> {
        let before = self.command("begin_rendering");
        if self.pass.is_some() {
            self.fail("begin_rendering", || {
                "a render pass is already active".into()
            });
        }
        if desc.color_attachments.len() > crate::descriptor::MAX_COLOR_ATTACHMENTS {
            self.fail("begin_rendering", || {
                format!(
                    "{} colour attachments exceed the maximum of {}",
                    desc.color_attachments.len(),
                    crate::descriptor::MAX_COLOR_ATTACHMENTS
                )
            });
        }
        if desc.offset_x < 0 || desc.offset_y < 0 || desc.width == 0 || desc.height == 0 {
            self.fail("begin_rendering", || {
                format!(
                    "render area ({}, {}, {}x{}) must have non-negative offsets and non-zero extent",
                    desc.offset_x, desc.offset_y, desc.width, desc.height
                )
            });
        }
        let mut color_formats = [None; MAX_VALIDATED_COLOR_TARGETS];
        let mut samples = None;
        for (index, attachment) in desc.color_attachments.iter().enumerate() {
            let info = self.check_attachment(
                attachment.texture,
                attachment.mip_level,
                attachment.layer,
                TextureUsage::COLOR_ATTACHMENT,
                desc.offset_x,
                desc.offset_y,
                desc.width,
                desc.height,
                &mut samples,
            );
            if let Some(slot) = color_formats.get_mut(index) {
                *slot = info.map(|i| i.format);
            }
            if let (Some(info), Some(resolve)) = (info, attachment.resolve_target) {
                if let Some(target) = self.texture_info("begin_rendering", resolve) {
                    let mip = attachment.mip_level.min(info.mip_levels.saturating_sub(1));
                    let mip_width = (info.width >> mip).max(1);
                    let mip_height = (info.height >> mip).max(1);
                    if !target.usage.contains(TextureUsage::COLOR_ATTACHMENT) {
                        self.fail("begin_rendering", || {
                            format!("resolve target {resolve:?} lacks COLOR_ATTACHMENT usage")
                        });
                    }
                    if info.sample_count == SampleCount::Count1
                        || target.sample_count != SampleCount::Count1
                        || target.format != info.format
                        || target.width != mip_width
                        || target.height != mip_height
                    {
                        self.fail("begin_rendering", || {
                            format!(
                                "resolve target {resolve:?} ({:?}, {:?}, {}x{}) must be single-sample \
                                 with the format and extent of the multisampled attachment ({:?}, {:?}, {}x{})",
                                target.format,
                                target.sample_count,
                                target.width,
                                target.height,
                                info.format,
                                info.sample_count,
                                mip_width,
                                mip_height
                            )
                        });
                    }
                }
            }
        }
        let depth_format = desc.depth_attachment.as_ref().and_then(|depth| {
            self.check_attachment(
                depth.texture,
                depth.mip_level,
                depth.layer,
                TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                desc.offset_x,
                desc.offset_y,
                desc.width,
                desc.height,
                &mut samples,
            )
            .map(|info| info.format)
        });
        if !self.passed(before) {
            return Ok(());
        }
        let result = self.inner().begin_rendering(desc);
        match &result {
            Ok(()) => {
                self.pass = Some(PassState {
                    color_formats,
                    color_count: desc.color_attachments.len(),
                    depth_format,
                    samples,
                });
            }
            Err(error) => {
                self.fail("begin_rendering", || error.to_string());
            }
        }
        result
    }

    fn end_rendering(&mut self) -> Result<(), RhiError> {
        let before = self.command("end_rendering");
        if self.pass.is_none() {
            self.fail("end_rendering", || "no render pass is active".into());
        }
        if !self.passed(before) {
            return Ok(());
        }
        self.pass = None;
        self.inner().end_rendering()
    }

    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        let before = self.command("set_viewport");
        if self.passed(before) {
            self.viewport_set = true;
            self.inner()
                .set_viewport(x, y, width, height, min_depth, max_depth);
        }
    }

    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let before = self.command("set_scissor");
        if self.passed(before) {
            self.scissor_set = true;
            self.inner().set_scissor(x, y, width, height);
        }
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        let before = self.command("set_stencil_reference");
        if self.passed(before) {
            self.inner().set_stencil_reference(reference);
        }
    }

    fn set_blend_constant(&mut self, color: [f32; 4]) {
        let before = self.command("set_blend_constant");
        if self.passed(before) {
            self.inner().set_blend_constant(color);
        }
    }

    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        let before = self.command("bind_pipeline");
        let signature = self.oracle.pipeline_signature(pipeline);
        if signature.is_none() {
            self.fail("bind_pipeline", || {
                format!("unknown pipeline handle {pipeline:?}")
            });
        }
        if self.passed(before) {
            self.pipeline = signature;
            self.inner().bind_pipeline(pipeline);
        }
    }

    fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[BufferHandle],
        offsets: &[u64],
    ) {
        let before = self.command("bind_vertex_buffers");
        if buffers.len() != offsets.len() {
            self.fail("bind_vertex_buffers", || {
                format!("{} buffers but {} offsets", buffers.len(), offsets.len())
            });
        }
        for (i, &buffer) in buffers.iter().enumerate() {
            let size = self.check_buffer_usage(
                "bind_vertex_buffers",
                buffer,
                crate::types::BufferUsage::VERTEX,
                "VERTEX",
            );
            if let (Some(size), Some(&offset)) = (size, offsets.get(i)) {
                if offset > size {
                    self.fail("bind_vertex_buffers", || {
                        format!(
                            "vertex buffer {i} ({buffer:?}) offset {offset} exceeds buffer size {size}"
                        )
                    });
                }
            }
        }
        if self.passed(before) {
            self.inner()
                .bind_vertex_buffers(first_binding, buffers, offsets);
        }
    }

    fn bind_index_buffer(&mut self, buffer: BufferHandle, offset: u64, index_type: IndexType) {
        let before = self.command("bind_index_buffer");
        let size = self.check_buffer_usage(
            "bind_index_buffer",
            buffer,
            crate::types::BufferUsage::INDEX,
            "INDEX",
        );
        let align = match index_type {
            IndexType::Uint16 => 2,
            IndexType::Uint32 => 4,
        };
        if !offset.is_multiple_of(align) {
            self.fail("bind_index_buffer", || {
                format!(
                    "index buffer offset {offset} is not a multiple of {align} for {index_type:?}"
                )
            });
        }
        if let Some(size) = size {
            if offset > size {
                self.fail("bind_index_buffer", || {
                    format!("index buffer ({buffer:?}) offset {offset} exceeds buffer size {size}")
                });
            }
        }
        if self.passed(before) {
            self.index_buffer = Some(offset);
            self.inner().bind_index_buffer(buffer, offset, index_type);
        }
    }

    fn set_bind_group(&mut self, index: u32, group: BindGroupHandle, dynamic_offsets: &[u32]) {
        const NAME: &str = "set_bind_group";
        let before = self.command(NAME);
        let info = self.oracle.bind_group_info(group);
        match info {
            None => self.fail(NAME, || format!("unknown bind group handle {group:?}")),
            Some(info) => {
                if !info.live {
                    self.fail(NAME, || {
                        format!("{group:?} refers to a resource that has been destroyed")
                    });
                }
                if dynamic_offsets.len() != info.layout.dynamic_offsets as usize {
                    self.fail(NAME, || {
                        format!(
                            "{} dynamic offsets given; the group's layout has {}",
                            dynamic_offsets.len(),
                            info.layout.dynamic_offsets
                        )
                    });
                }
                for (i, (&offset, binding)) in dynamic_offsets.iter().zip(&info.dynamic).enumerate()
                {
                    let offset = u64::from(offset);
                    if offset % binding.alignment.max(1) != 0 {
                        self.fail(NAME, || {
                            format!(
                                "dynamic offset {i} ({offset}) is not a multiple of {}",
                                binding.alignment
                            )
                        });
                    }
                    if binding
                        .end
                        .checked_add(offset)
                        .is_none_or(|end| end > binding.buffer_size)
                    {
                        self.fail(NAME, || {
                            format!(
                                "dynamic offset {i} ({offset}) moves the range past the buffer's \
                                 {} bytes",
                                binding.buffer_size
                            )
                        });
                    }
                }
            }
        }
        if index as usize >= MAX_BIND_GROUPS {
            self.fail(NAME, || {
                format!("set {index} is not below {MAX_BIND_GROUPS}")
            });
        }
        if self.passed(before) {
            self.bind_groups[index as usize] = info;
            self.inner().set_bind_group(index, group, dynamic_offsets);
        }
    }

    fn push_constants(&mut self, offset: u32, data: &[u8]) -> Result<(), RhiError> {
        let before = self.command("push_constants");
        match self.pipeline {
            None => self.fail("push_constants", || "no pipeline is bound".into()),
            Some(pipeline) => {
                let size = u32::try_from(data.len()).unwrap_or(u32::MAX);
                if !pipeline.covers_push_constants(offset, size) {
                    self.fail("push_constants", || {
                        format!(
                            "bytes {offset}..{} are outside the push constant ranges declared by \
                             the bound pipeline",
                            offset.saturating_add(size)
                        )
                    });
                }
            }
        }
        if !self.passed(before) {
            return Ok(());
        }
        self.inner().push_constants(offset, data)
    }

    fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let before = self.command("draw");
        self.check_draw("draw");
        if self.passed(before) {
            self.inner()
                .draw(vertex_count, instance_count, first_vertex, first_instance);
        }
    }

    fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let before = self.command("draw_indexed");
        self.check_draw("draw_indexed");
        if self.index_buffer.is_none() {
            self.fail("draw_indexed", || "no index buffer is bound".into());
        }
        if self.passed(before) {
            self.inner().draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    fn draw_indirect(&mut self, buffer: BufferHandle, offset: u64, draw_count: u32, stride: u32) {
        const NAME: &str = "draw_indirect";
        let before = self.command(NAME);
        self.check_draw(NAME);
        self.check_indirect(
            NAME,
            buffer,
            offset,
            draw_count,
            stride,
            crate::query::DrawIndirectArgs::SIZE,
        );
        if self.passed(before) {
            self.inner()
                .draw_indirect(buffer, offset, draw_count, stride);
        }
    }

    fn draw_indexed_indirect(
        &mut self,
        buffer: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) {
        const NAME: &str = "draw_indexed_indirect";
        let before = self.command(NAME);
        self.check_draw(NAME);
        match self.index_buffer {
            None => self.fail(NAME, || "no index buffer is bound".into()),
            Some(0) => {}
            Some(offset) => self.fail(NAME, || {
                format!(
                    "the index buffer is bound at offset {offset}; indexed indirect draws need \
                     offset 0 (OpenGL cannot offset it)"
                )
            }),
        }
        self.check_indirect(
            NAME,
            buffer,
            offset,
            draw_count,
            stride,
            crate::query::DrawIndexedIndirectArgs::SIZE,
        );
        if self.passed(before) {
            self.inner()
                .draw_indexed_indirect(buffer, offset, draw_count, stride);
        }
    }

    fn dispatch_indirect(&mut self, buffer: BufferHandle, offset: u64) {
        const NAME: &str = "dispatch_indirect";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        match self.pipeline {
            None => self.fail(NAME, || "no pipeline is bound".into()),
            Some(p) if p.kind != PipelineKind::Compute => {
                self.fail(NAME, || "the bound pipeline is a graphics pipeline".into())
            }
            Some(p) => self.check_bind_groups(NAME, &p),
        }
        self.check_indirect(
            NAME,
            buffer,
            offset,
            1,
            0,
            crate::query::DispatchIndirectArgs::SIZE,
        );
        if self.passed(before) {
            self.inner().dispatch_indirect(buffer, offset);
        }
    }

    fn write_timestamp(&mut self, query_set: crate::handle::QuerySetHandle, index: u32) {
        const NAME: &str = "write_timestamp";
        let before = self.command(NAME);
        self.check_query_range(NAME, query_set, index, 1);
        if self.passed(before) {
            self.inner().write_timestamp(query_set, index);
        }
    }

    fn resolve_query_set(
        &mut self,
        query_set: crate::handle::QuerySetHandle,
        first: u32,
        count: u32,
        dst: BufferHandle,
        dst_offset: u64,
    ) {
        const NAME: &str = "resolve_query_set";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        self.check_query_range(NAME, query_set, first, count);
        let dst_size =
            self.check_buffer_usage(NAME, dst, crate::types::BufferUsage::COPY_DST, "COPY_DST");
        if !dst_offset.is_multiple_of(8) {
            self.fail(NAME, || {
                format!("dst_offset {dst_offset} is not a multiple of 8")
            });
        }
        if let Some(size) = dst_size {
            let count_bytes = 8 * u64::from(count);
            match dst_offset.checked_add(count_bytes) {
                Some(end) if end > size => {
                    self.fail(NAME, || {
                        format!("the results end at byte {end} of a {size}-byte buffer")
                    });
                }
                None => {
                    self.fail(NAME, || {
                        format!("the results end past the end of the {size}-byte buffer")
                    });
                }
                _ => {}
            }
        }
        if self.passed(before) {
            self.inner()
                .resolve_query_set(query_set, first, count, dst, dst_offset);
        }
    }

    fn push_debug_group(&mut self, label: &str) {
        let before = self.command("push_debug_group");
        if self.passed(before) {
            self.debug_groups += 1;
            self.inner().push_debug_group(label);
        }
    }

    fn pop_debug_group(&mut self) {
        let before = self.command("pop_debug_group");
        if self.debug_groups == 0 {
            self.fail("pop_debug_group", || "no debug group is open".into());
        }
        if self.passed(before) {
            self.debug_groups -= 1;
            self.inner().pop_debug_group();
        }
    }

    fn insert_debug_marker(&mut self, label: &str) {
        let before = self.command("insert_debug_marker");
        if self.passed(before) {
            self.inner().insert_debug_marker(label);
        }
    }

    fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        let before = self.command("dispatch");
        self.require_outside_pass("dispatch");
        match self.pipeline {
            None => self.fail("dispatch", || "no pipeline is bound".into()),
            Some(p) if p.kind != PipelineKind::Compute => self.fail("dispatch", || {
                "the bound pipeline is a graphics pipeline".into()
            }),
            Some(p) => self.check_bind_groups("dispatch", &p),
        }
        if self.passed(before) {
            self.inner()
                .dispatch(group_count_x, group_count_y, group_count_z);
        }
    }

    fn pipeline_barrier(&mut self, barrier: &PipelineBarrier) {
        let before = self.command("pipeline_barrier");
        // VUID-vkCmdPipelineBarrier2-None-09553 without dynamic-rendering local read.
        self.require_outside_pass("pipeline_barrier");
        for b in &barrier.buffer_barriers {
            self.check_buffer("pipeline_barrier", b.buffer);
        }
        for b in &barrier.image_barriers {
            if let Some(info) = self.texture_info("pipeline_barrier", b.texture) {
                if !b.range.fits(info.mip_levels, info.array_layers) {
                    self.fail("pipeline_barrier", || {
                        format!(
                            "range {:?} of {:?} is outside the texture ({} mips, {} layers)",
                            b.range, b.texture, info.mip_levels, info.array_layers
                        )
                    });
                }
                let aspect_ok = match b.range.aspect {
                    crate::types::TextureAspect::All => true,
                    crate::types::TextureAspect::DepthOnly => info.format.is_depth(),
                    crate::types::TextureAspect::StencilOnly => info.format.is_stencil(),
                };
                if !aspect_ok {
                    self.fail("pipeline_barrier", || {
                        format!("{:?} has no {:?} aspect", info.format, b.range.aspect)
                    });
                }
            }
        }
        if self.passed(before) {
            self.inner().pipeline_barrier(barrier);
        }
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: BufferHandle,
        src_offset: u64,
        dst: BufferHandle,
        dst_offset: u64,
        size: u64,
    ) {
        const NAME: &str = "copy_buffer_to_buffer";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        let src_size =
            self.check_buffer_usage(NAME, src, crate::types::BufferUsage::COPY_SRC, "COPY_SRC");
        let dst_size =
            self.check_buffer_usage(NAME, dst, crate::types::BufferUsage::COPY_DST, "COPY_DST");
        if let (Some(src_size), Some(dst_size)) = (src_size, dst_size) {
            if let Err(problem) = crate::transfer::check_buffer_to_buffer_copy(
                src_size,
                src_offset,
                dst_size,
                dst_offset,
                size,
                src == dst,
            ) {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner()
                .copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
        }
    }

    fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32) {
        const NAME: &str = "fill_buffer";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        let buffer_size = self.check_buffer_usage(
            NAME,
            buffer,
            crate::types::BufferUsage::COPY_DST,
            "COPY_DST",
        );
        if let Some(buffer_size) = buffer_size {
            if let Err(problem) = crate::transfer::check_fill_buffer(buffer_size, offset, size) {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner().fill_buffer(buffer, offset, size, value);
        }
    }

    fn clear_texture(
        &mut self,
        texture: TextureHandle,
        range: &crate::sync::TextureSubresourceRange,
        value: crate::transfer::TextureClear,
    ) {
        const NAME: &str = "clear_texture";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        if let Some(info) = self.transfer_texture(NAME, texture, TextureUsage::COPY_DST) {
            if let Err(problem) = crate::transfer::check_clear_texture(&info, range, &value) {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner().clear_texture(texture, range, value);
        }
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: BufferHandle,
        dst: TextureHandle,
        copy: &BufferTextureCopy,
    ) {
        const NAME: &str = "copy_buffer_to_texture";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        let size =
            self.check_buffer_usage(NAME, src, crate::types::BufferUsage::COPY_SRC, "COPY_SRC");
        if let Some(info) = self.transfer_texture(NAME, dst, TextureUsage::COPY_DST) {
            if let Err(problem) = crate::transfer::check_buffer_copy(&info, copy, size) {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner().copy_buffer_to_texture(src, dst, copy);
        }
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: TextureHandle,
        dst: BufferHandle,
        copy: &BufferTextureCopy,
    ) {
        const NAME: &str = "copy_texture_to_buffer";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        let size =
            self.check_buffer_usage(NAME, dst, crate::types::BufferUsage::COPY_DST, "COPY_DST");
        if let Some(info) = self.transfer_texture(NAME, src, TextureUsage::COPY_SRC) {
            if let Err(problem) = crate::transfer::check_buffer_copy(&info, copy, size) {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner().copy_texture_to_buffer(src, dst, copy);
        }
    }

    fn copy_texture_to_texture(
        &mut self,
        src: TextureHandle,
        dst: TextureHandle,
        copy: &crate::transfer::TextureCopy,
    ) {
        const NAME: &str = "copy_texture_to_texture";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        let src_info = self.transfer_texture(NAME, src, TextureUsage::COPY_SRC);
        let dst_info = self.transfer_texture(NAME, dst, TextureUsage::COPY_DST);
        if let (Some(s), Some(d)) = (src_info, dst_info) {
            if let Err(problem) = crate::transfer::check_texture_copy(&s, &d, src == dst, copy) {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner().copy_texture_to_texture(src, dst, copy);
        }
    }

    fn blit_texture(
        &mut self,
        src: TextureHandle,
        dst: TextureHandle,
        blit: &crate::transfer::TextureBlit,
    ) {
        use crate::capabilities::FormatCapabilities;
        const NAME: &str = "blit_texture";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        let src_info = self.transfer_texture(NAME, src, TextureUsage::COPY_SRC);
        let dst_info = self.transfer_texture(NAME, dst, TextureUsage::COPY_DST);
        if let (Some(s), Some(d)) = (src_info, dst_info) {
            let src_caps = self.oracle.format_capabilities(s.format);
            let dst_caps = self.oracle.format_capabilities(d.format);
            if !src_caps.contains(FormatCapabilities::BLIT)
                || !dst_caps.contains(FormatCapabilities::BLIT)
            {
                self.fail(NAME, || {
                    format!(
                        "{:?} -> {:?} cannot be blitted on this device",
                        s.format, d.format
                    )
                });
            }
            let filterable = src_caps.contains(FormatCapabilities::FILTERABLE);
            if let Err(problem) = crate::transfer::check_blit(&s, &d, src == dst, blit, filterable)
            {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner().blit_texture(src, dst, blit);
        }
    }

    fn generate_mipmaps(&mut self, texture: TextureHandle) {
        use crate::capabilities::FormatCapabilities;
        const NAME: &str = "generate_mipmaps";
        let before = self.command(NAME);
        self.require_outside_pass(NAME);
        let usage = TextureUsage::COPY_SRC | TextureUsage::COPY_DST;
        if let Some(info) = self.transfer_texture(NAME, texture, usage) {
            let caps = self.oracle.format_capabilities(info.format);
            let able = caps.contains(FormatCapabilities::BLIT | FormatCapabilities::FILTERABLE);
            if let Err(problem) = crate::transfer::check_generate_mipmaps(&info, able) {
                self.fail(NAME, || problem);
            }
        }
        if self.passed(before) {
            self.inner().generate_mipmaps(texture);
        }
    }

    fn finish(mut self: Box<Self>) -> Result<Box<dyn CommandBuffer>, RhiError> {
        self.command_index += 1;
        if !self.recording {
            self.fail("finish", || "called before `begin`".into());
        }
        if self.pass.is_some() {
            self.fail("finish", || {
                "a render pass is still active (missing `end_rendering`)".into()
            });
        }
        if self.debug_groups > 0 {
            let open = self.debug_groups;
            self.fail("finish", || {
                format!("{open} debug group(s) still open (missing `pop_debug_group`)")
            });
        }
        let report = self.report();
        let pass_active = self.pass.is_some();
        let recording = self.recording;
        let mut inner = self
            .inner
            .take()
            .expect("a validator in use wraps a recorder");
        let pool = self.pool.clone();
        crate::pool::BoxPool::give_back(&pool, self);
        if let Err(error) = report {
            // Close and finish the backend recording anyway so it releases its command buffer,
            // then discard it: the caller only sees the validation error.
            if pass_active {
                let _ = inner.end_rendering();
            }
            if recording {
                let _ = inner.finish();
            }
            return Err(error);
        }
        inner.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{
        ColorTargetState, DepthAttachment, DepthStencilState, RasterizerState, RenderingAttachment,
        VertexLayoutDescriptor,
    };
    use crate::handle::ShaderHandle;
    use crate::types::{ColorWriteMask, LoadOp, PrimitiveTopology, StoreOp};
    use std::sync::Mutex;

    struct NullCommandBuffer;
    impl CommandBuffer for NullCommandBuffer {}

    /// Records the names of forwarded calls.
    struct SpyRecorder(Arc<Mutex<Vec<&'static str>>>);

    impl SpyRecorder {
        fn log(&self, name: &'static str) {
            self.0.lock().unwrap().push(name);
        }
    }

    impl CommandBufferRecorder for SpyRecorder {
        fn begin(&mut self) -> Result<(), RhiError> {
            self.log("begin");
            Ok(())
        }
        fn begin_rendering(&mut self, _: &RenderingDescriptor) -> Result<(), RhiError> {
            self.log("begin_rendering");
            Ok(())
        }
        fn end_rendering(&mut self) -> Result<(), RhiError> {
            self.log("end_rendering");
            Ok(())
        }
        fn set_viewport(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {
            self.log("set_viewport");
        }
        fn set_scissor(&mut self, _: i32, _: i32, _: u32, _: u32) {
            self.log("set_scissor");
        }
        fn set_stencil_reference(&mut self, _: u32) {
            self.log("set_stencil_reference");
        }
        fn set_blend_constant(&mut self, _: [f32; 4]) {
            self.log("set_blend_constant");
        }
        fn bind_pipeline(&mut self, _: PipelineHandle) {
            self.log("bind_pipeline");
        }
        fn bind_vertex_buffers(&mut self, _: u32, _: &[BufferHandle], _: &[u64]) {
            self.log("bind_vertex_buffers");
        }
        fn bind_index_buffer(&mut self, _: BufferHandle, _: u64, _: IndexType) {
            self.log("bind_index_buffer");
        }
        fn set_bind_group(&mut self, _: u32, _: BindGroupHandle, _: &[u32]) {
            self.log("set_bind_group");
        }
        fn push_constants(&mut self, _: u32, _: &[u8]) -> Result<(), RhiError> {
            self.log("push_constants");
            Ok(())
        }
        fn draw(&mut self, _: u32, _: u32, _: u32, _: u32) {
            self.log("draw");
        }
        fn draw_indexed(&mut self, _: u32, _: u32, _: u32, _: i32, _: u32) {
            self.log("draw_indexed");
        }
        fn dispatch(&mut self, _: u32, _: u32, _: u32) {
            self.log("dispatch");
        }
        fn draw_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
            self.log("draw_indirect");
        }
        fn draw_indexed_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
            self.log("draw_indexed_indirect");
        }
        fn dispatch_indirect(&mut self, _: BufferHandle, _: u64) {
            self.log("dispatch_indirect");
        }
        fn write_timestamp(&mut self, _: crate::handle::QuerySetHandle, _: u32) {
            self.log("write_timestamp");
        }
        fn resolve_query_set(
            &mut self,
            _: crate::handle::QuerySetHandle,
            _: u32,
            _: u32,
            _: BufferHandle,
            _: u64,
        ) {
            self.log("resolve_query_set");
        }
        fn push_debug_group(&mut self, _: &str) {
            self.log("push_debug_group");
        }
        fn pop_debug_group(&mut self) {
            self.log("pop_debug_group");
        }
        fn insert_debug_marker(&mut self, _: &str) {
            self.log("insert_debug_marker");
        }
        fn pipeline_barrier(&mut self, _: &PipelineBarrier) {
            self.log("pipeline_barrier");
        }
        fn copy_buffer_to_buffer(
            &mut self,
            _: BufferHandle,
            _: u64,
            _: BufferHandle,
            _: u64,
            _: u64,
        ) {
            self.log("copy_buffer_to_buffer");
        }
        fn fill_buffer(&mut self, _: BufferHandle, _: u64, _: u64, _: u32) {
            self.log("fill_buffer");
        }
        fn clear_texture(
            &mut self,
            _: TextureHandle,
            _: &crate::sync::TextureSubresourceRange,
            _: crate::transfer::TextureClear,
        ) {
            self.log("clear_texture");
        }
        fn copy_buffer_to_texture(
            &mut self,
            _: BufferHandle,
            _: TextureHandle,
            _: &BufferTextureCopy,
        ) {
            self.log("copy_buffer_to_texture");
        }
        fn copy_texture_to_buffer(
            &mut self,
            _: TextureHandle,
            _: BufferHandle,
            _: &BufferTextureCopy,
        ) {
            self.log("copy_texture_to_buffer");
        }
        fn copy_texture_to_texture(
            &mut self,
            _: TextureHandle,
            _: TextureHandle,
            _: &crate::transfer::TextureCopy,
        ) {
            self.log("copy_texture_to_texture");
        }
        fn blit_texture(
            &mut self,
            _: TextureHandle,
            _: TextureHandle,
            _: &crate::transfer::TextureBlit,
        ) {
            self.log("blit_texture");
        }
        fn generate_mipmaps(&mut self, _: TextureHandle) {
            self.log("generate_mipmaps");
        }
        fn finish(self: Box<Self>) -> Result<Box<dyn CommandBuffer>, RhiError> {
            self.log("finish");
            Ok(Box::new(NullCommandBuffer))
        }
    }

    const BUFFER: BufferHandle = BufferHandle::new(1, 1);
    const RGBA8_TARGET: TextureHandle = TextureHandle::new(2, 1);
    const DEPTH_TARGET: TextureHandle = TextureHandle::new(3, 1);
    const GRAPHICS: PipelineHandle = PipelineHandle::new(4, 1);
    const GRAPHICS_WITH_DEPTH: PipelineHandle = PipelineHandle::new(5, 1);
    const COMPUTE: PipelineHandle = PipelineHandle::new(6, 1);
    const SAMPLER: SamplerHandle = SamplerHandle::new(7, 1);
    const D24S8_TARGET: TextureHandle = TextureHandle::new(8, 1);
    const GROUP: BindGroupHandle = BindGroupHandle::new(9, 1);
    const QUERIES: crate::handle::QuerySetHandle = crate::handle::QuerySetHandle::new(12, 1);
    const OTHER_LAYOUT_GROUP: BindGroupHandle = BindGroupHandle::new(10, 1);
    const DEAD_GROUP: BindGroupHandle = BindGroupHandle::new(11, 1);
    const LAYOUT: BindGroupLayoutInfo = BindGroupLayoutInfo {
        fingerprint: 0xA,
        dynamic_offsets: 1,
    };

    /// A group of `LAYOUT` whose one dynamic binding covers bytes 0..16 of a 512-byte buffer.
    fn group_info(fingerprint: u64, live: bool) -> BindGroupInfo {
        let mut dynamic = [crate::binding::DynamicBinding::default();
            crate::binding::MAX_DYNAMIC_OFFSETS_PER_GROUP];
        dynamic[0] = crate::binding::DynamicBinding {
            end: 16,
            buffer_size: 512,
            alignment: 256,
        };
        BindGroupInfo {
            layout: BindGroupLayoutInfo {
                fingerprint,
                ..LAYOUT
            },
            dynamic,
            live,
        }
    }

    struct Oracle;

    fn graphics_desc(depth: bool) -> GraphicsPipelineDescriptor {
        GraphicsPipelineDescriptor {
            vertex_shader: ShaderHandle::new(0, 1),
            fragment_shader: None,
            vertex_layout: VertexLayoutDescriptor::new(),
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState::default(),
            multisample: Default::default(),
            depth_stencil: depth.then(DepthStencilState::default),
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![PushConstantRange {
                stages: crate::binding::ShaderStages::VERTEX,
                offset: 0,
                size: 16,
            }],
            label: None,
        }
    }

    impl ResourceOracle for Oracle {
        fn buffer_exists(&self, handle: BufferHandle) -> bool {
            handle == BUFFER
        }
        fn texture_info(&self, handle: TextureHandle) -> Option<TextureInfo> {
            let (format, attachment_usage) = match handle {
                RGBA8_TARGET => (TextureFormat::Rgba8Unorm, TextureUsage::COLOR_ATTACHMENT),
                DEPTH_TARGET => (
                    TextureFormat::Depth32Float,
                    TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                ),
                D24S8_TARGET => (
                    TextureFormat::Depth24PlusStencil8,
                    TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                ),
                _ => return None,
            };
            Some(TextureInfo {
                width: 4,
                height: 4,
                format,
                dimension: TextureDimension::D2,
                mip_levels: 1,
                array_layers: 1,
                depth: 1,
                sample_count: SampleCount::Count1,
                usage: attachment_usage | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            })
        }
        fn sampler_exists(&self, handle: SamplerHandle) -> bool {
            handle == SAMPLER
        }
        fn pipeline_signature(&self, handle: PipelineHandle) -> Option<PipelineSignature> {
            match handle {
                GRAPHICS => Some(PipelineSignature::graphics(&graphics_desc(false))),
                GRAPHICS_WITH_DEPTH => Some(PipelineSignature::graphics(&graphics_desc(true))),
                COMPUTE => Some(
                    PipelineSignature::compute(&ComputePipelineDescriptor {
                        compute_shader: ShaderHandle::new(0, 1),
                        bind_group_layouts: vec![],
                        push_constant_ranges: vec![],
                        label: None,
                    })
                    .with_bind_group_layouts(&[LAYOUT]),
                ),
                _ => None,
            }
        }
        fn query_set_count(&self, handle: crate::handle::QuerySetHandle) -> Option<u32> {
            (handle == QUERIES).then_some(4)
        }
        fn buffer_size(&self, handle: BufferHandle) -> Option<u64> {
            (handle == BUFFER).then_some(4096)
        }
        fn bind_group_info(&self, handle: BindGroupHandle) -> Option<BindGroupInfo> {
            match handle {
                GROUP => Some(group_info(LAYOUT.fingerprint, true)),
                OTHER_LAYOUT_GROUP => Some(group_info(0xB, true)),
                DEAD_GROUP => Some(group_info(LAYOUT.fingerprint, false)),
                _ => None,
            }
        }
    }

    fn recorder() -> (ValidatingRecorder, Arc<Mutex<Vec<&'static str>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let inner = Box::new(SpyRecorder(Arc::clone(&calls)));
        (ValidatingRecorder::new(inner, Arc::new(Oracle)), calls)
    }

    fn pass(depth: bool) -> RenderingDescriptor {
        RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width: 4,
            height: 4,
            color_attachments: korppi_core::smallvec![RenderingAttachment {
                texture: RGBA8_TARGET,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: None,
                resolve_target: None,
            }],
            depth_attachment: depth.then_some(DepthAttachment {
                texture: DEPTH_TARGET,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::DontCare,
                clear_depth: Some(0.0),
                clear_stencil: None,
                stencil_load_op: LoadOp::DontCare,
                stencil_store_op: StoreOp::DontCare,
            }),
        }
    }

    fn finish_error(r: ValidatingRecorder) -> String {
        match Box::new(r).finish() {
            Err(RhiError::ValidationError(message)) => message,
            Err(other) => panic!("unexpected error {other:?}"),
            Ok(_) => panic!("validation accepted misuse"),
        }
    }

    #[test]
    fn valid_recording_is_forwarded_and_accepted() {
        let (mut r, calls) = recorder();
        r.begin().unwrap();
        r.begin_rendering(&pass(true)).unwrap();
        r.set_viewport(0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
        r.set_scissor(0, 0, 4, 4);
        r.bind_pipeline(GRAPHICS_WITH_DEPTH);
        r.bind_vertex_buffers(0, &[BUFFER], &[0]);
        r.bind_index_buffer(BUFFER, 0, IndexType::Uint16);
        r.push_constants(4, &[0; 12]).unwrap();
        r.draw_indexed(3, 1, 0, 0, 0);
        r.end_rendering().unwrap();
        r.set_bind_group(0, GROUP, &[256]);
        r.bind_pipeline(COMPUTE);
        r.dispatch(1, 1, 1);
        assert!(Box::new(r).finish().is_ok());
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                "begin",
                "begin_rendering",
                "set_viewport",
                "set_scissor",
                "bind_pipeline",
                "bind_vertex_buffers",
                "bind_index_buffer",
                "push_constants",
                "draw_indexed",
                "end_rendering",
                "set_bind_group",
                "bind_pipeline",
                "dispatch",
                "finish",
            ]
        );
    }

    #[test]
    fn bind_group_misuse_is_reported() {
        let (mut r, calls) = recorder();
        r.begin().unwrap();
        r.bind_pipeline(COMPUTE);
        r.dispatch(1, 1, 1); // nothing bound at set 0
        r.set_bind_group(0, OTHER_LAYOUT_GROUP, &[0]);
        r.dispatch(1, 1, 1); // incompatible layout
        r.set_bind_group(0, GROUP, &[]); // offset count
        r.set_bind_group(0, GROUP, &[4]); // alignment
        r.set_bind_group(0, GROUP, &[512]); // past the end
        r.set_bind_group(0, DEAD_GROUP, &[0]); // destroyed resource
        r.set_bind_group(4, GROUP, &[0]); // set index
        let message = finish_error(r);
        for expected in [
            "command #3 `dispatch`: no bind group is bound at set 0",
            "command #5 `dispatch`: the bind group at set 0 has a layout incompatible",
            "command #6 `set_bind_group`: 0 dynamic offsets given; the group's layout has 1",
            "command #7 `set_bind_group`: dynamic offset 0 (4) is not a multiple of 256",
            "command #8 `set_bind_group`: dynamic offset 0 (512) moves the range past",
            "command #9 `set_bind_group`: BindGroupHandle { id: 11, generation: 1, device: \
             DeviceId(0) } refers to a resource that has been destroyed",
            "command #10 `set_bind_group`: set 4 is not below 4",
        ] {
            assert!(
                message.contains(expected),
                "missing {expected:?} in {message}"
            );
        }
        // Only the incompatible group (valid by itself) reached the backend.
        let forwarded = calls.lock().unwrap();
        assert_eq!(
            forwarded.iter().filter(|c| **c == "set_bind_group").count(),
            1
        );
    }

    #[test]
    fn indirect_query_and_debug_group_misuse_is_reported() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        r.begin_rendering(&pass(false)).unwrap();
        r.set_viewport(0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
        r.set_scissor(0, 0, 4, 4);
        r.bind_pipeline(GRAPHICS);
        r.draw_indirect(BUFFER, 2, 1, 16); // misaligned
        r.draw_indirect(BUFFER, 0, 300, 16); // 4800 bytes of a 4096-byte buffer
        r.bind_index_buffer(BUFFER, 4, IndexType::Uint16);
        r.draw_indexed_indirect(BUFFER, 0, 1, 20); // index buffer offset
        r.end_rendering().unwrap();
        r.write_timestamp(QUERIES, 4); // past the set
        r.resolve_query_set(QUERIES, 2, 2, BUFFER, 4); // misaligned offset
        r.pop_debug_group(); // nothing open
        r.push_debug_group("frame");
        let message = finish_error(r);
        for expected in [
            "`draw_indirect`: offset 2 is not a multiple of 4",
            "`draw_indirect`: the arguments end at byte 4800 of a 4096-byte buffer",
            "`draw_indexed_indirect`: the index buffer is bound at offset 4",
            "`write_timestamp`: queries 4..5 of a set of 4",
            "`resolve_query_set`: dst_offset 4 is not a multiple of 8",
            "`pop_debug_group`: no debug group is open",
            "`finish`: 1 debug group(s) still open",
        ] {
            assert!(
                message.contains(expected),
                "missing {expected:?} in {message}"
            );
        }
    }

    #[test]
    fn draw_outside_pass_is_reported() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        r.bind_pipeline(GRAPHICS);
        r.draw(3, 1, 0, 0);
        let message = finish_error(r);
        assert!(
            message.contains("command #3 `draw`: recorded outside a render pass"),
            "{message}"
        );
    }

    #[test]
    fn depth_presence_mismatch_is_reported() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        r.begin_rendering(&pass(true)).unwrap();
        r.set_viewport(0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
        r.set_scissor(0, 0, 4, 4);
        r.bind_pipeline(GRAPHICS);
        r.draw(3, 1, 0, 0);
        r.end_rendering().unwrap();
        let message = finish_error(r);
        assert!(
            message.contains("pipeline depth state (absent) does not match"),
            "{message}"
        );
    }

    #[test]
    fn depth_format_mismatch_is_reported() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        let mut desc = pass(true);
        desc.depth_attachment.as_mut().unwrap().texture = D24S8_TARGET;
        r.begin_rendering(&desc).unwrap();
        r.set_viewport(0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
        r.set_scissor(0, 0, 4, 4);
        r.bind_pipeline(GRAPHICS_WITH_DEPTH);
        r.draw(3, 1, 0, 0);
        r.end_rendering().unwrap();
        let message = finish_error(r);
        assert!(
            message.contains(
                "pipeline depth state (Depth32Float) does not match the pass depth attachment \
                 (Depth24PlusStencil8)"
            ),
            "{message}"
        );
    }

    #[test]
    fn push_constants_outside_ranges_are_reported() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        r.bind_pipeline(GRAPHICS);
        r.push_constants(8, &[0; 16]).unwrap();
        let message = finish_error(r);
        assert!(
            message.contains("bytes 8..24 are outside the push constant ranges"),
            "{message}"
        );
    }

    #[test]
    fn pass_structure_errors_are_reported() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        r.end_rendering().unwrap();
        r.begin_rendering(&pass(false)).unwrap();
        r.begin_rendering(&pass(false)).unwrap();
        r.dispatch(1, 1, 1);
        r.copy_buffer_to_buffer(BUFFER, 0, BUFFER, 0, 4);
        let message = finish_error(r);
        for expected in [
            "`end_rendering`: no render pass is active",
            "`begin_rendering`: a render pass is already active",
            "`dispatch`: not allowed inside a render pass",
            "`copy_buffer_to_buffer`: not allowed inside a render pass",
            "`finish`: a render pass is still active",
        ] {
            assert!(
                message.contains(expected),
                "missing {expected:?} in {message}"
            );
        }
    }

    #[test]
    fn unknown_handles_and_missing_state_are_reported() {
        let (mut r, _) = recorder();
        r.draw(3, 1, 0, 0);
        r.begin().unwrap();
        r.bind_pipeline(PipelineHandle::new(99, 1));
        r.set_bind_group(0, BindGroupHandle::new(98, 1), &[]);
        r.begin_rendering(&pass(false)).unwrap();
        r.bind_pipeline(GRAPHICS);
        r.draw_indexed(3, 1, 0, 0, 0);
        r.end_rendering().unwrap();
        let message = finish_error(r);
        for expected in [
            "`draw`: recorded before `begin`",
            "unknown pipeline handle",
            "unknown bind group handle",
            "`draw_indexed`: no index buffer is bound",
        ] {
            assert!(
                message.contains(expected),
                "missing {expected:?} in {message}"
            );
        }
    }

    #[test]
    fn invalid_commands_are_not_forwarded_to_the_backend() {
        let (mut r, calls) = recorder();
        r.draw(3, 1, 0, 0); // before begin
        r.begin().unwrap();
        r.begin().unwrap(); // nested begin
        r.end_rendering().unwrap(); // no pass
        r.bind_pipeline(GRAPHICS);
        r.draw(3, 1, 0, 0); // outside a pass
        r.push_constants(64, &[0; 4]).unwrap(); // outside declared ranges
        r.bind_vertex_buffers(0, &[BufferHandle::new(42, 1)], &[0]); // unknown buffer
        let _ = finish_error(r);
        assert_eq!(
            *calls.lock().unwrap(),
            vec!["begin", "bind_pipeline", "finish"]
        );
    }

    #[test]
    fn draw_requires_viewport_and_scissor() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        r.begin_rendering(&pass(false)).unwrap();
        r.bind_pipeline(GRAPHICS);
        r.draw(3, 1, 0, 0);
        r.end_rendering().unwrap();
        let message = finish_error(r);
        assert!(message.contains("no viewport has been set"), "{message}");
        assert!(message.contains("no scissor has been set"), "{message}");
    }

    #[test]
    fn rejected_recording_still_releases_the_backend_command_buffer() {
        let (mut r, calls) = recorder();
        r.begin().unwrap();
        r.begin_rendering(&pass(false)).unwrap();
        let _ = finish_error(r);
        assert_eq!(
            *calls.lock().unwrap(),
            vec!["begin", "begin_rendering", "end_rendering", "finish"]
        );
    }

    #[test]
    fn invalid_copy_row_pitch_is_reported() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        let copy = |pitch: u32| BufferTextureCopy {
            buffer_offset: 0,
            buffer_row_pitch: pitch,
            texture_mip_level: 0,
            texture_layer: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
        };
        r.copy_buffer_to_texture(BUFFER, RGBA8_TARGET, &copy(0)); // tightly packed: valid
        r.copy_buffer_to_texture(BUFFER, RGBA8_TARGET, &copy(32)); // padded rows: valid
        r.copy_buffer_to_texture(BUFFER, RGBA8_TARGET, &copy(18)); // not a texel multiple
        r.copy_buffer_to_texture(BUFFER, RGBA8_TARGET, &copy(8)); // shorter than a row
        let message = finish_error(r);
        assert!(
            message.contains("command #4 `copy_buffer_to_texture`: buffer_row_pitch 18"),
            "{message}"
        );
        assert!(
            message.contains("command #5 `copy_buffer_to_texture`: buffer_row_pitch 8"),
            "{message}"
        );
        assert_eq!(message.lines().count(), 2, "{message}");
    }

    #[test]
    fn message_count_is_bounded() {
        let (mut r, _) = recorder();
        r.begin().unwrap();
        for _ in 0..(MAX_MESSAGES + 5) {
            r.draw(3, 1, 0, 0);
        }
        let message = finish_error(r);
        assert_eq!(message.lines().count(), MAX_MESSAGES + 1);
        assert!(message.ends_with("... and 5 more"), "{message}");
    }
}
