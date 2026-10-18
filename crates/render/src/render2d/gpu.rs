//! GPU rendering of [`QuadBatch2d`] through the RHI (FIX-1.3, FIX-6.3: textured and solid-colour
//! batched 2D rendering).
//!
//! The renderer owns one pipeline, a `SampledTexture` bind group layout, a default 1x1 white
//! texture and sampler, and, per frame in flight, a vertex and an index buffer that grow when a
//! batch outgrows them and are otherwise reused, so steady-state frames allocate nothing. Shaders
//! are SPIR-V (`shaders/quad_batch.*`), which both backends accept (RHI-08).

use crate::render2d::batch::{QuadBatch2d, QuadBatchDraw, QuadVertex2d};
use korppi_rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupHandle, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutHandle, BindingResource, BindingType, BlendState,
    BufferDescriptor, BufferHandle, BufferUsage, ColorTargetState, ColorWriteMask,
    CommandBufferRecorder, CullMode, FilterMode, FrameIndex, FrontFace, GraphicsDevice,
    GraphicsPipelineDescriptor, ImageLayout, IndexType, LoadOp, MemoryUsage, PipelineHandle,
    PolygonMode, PrimitiveTopology, PushConstantRange, RasterizerState, RenderingAttachment,
    RenderingDescriptor, RhiError, SampleCount, SamplerAddressMode, SamplerDescriptor,
    SamplerHandle, ShaderDescriptor, ShaderHandle, ShaderSource, ShaderStage, ShaderStages,
    StoreOp, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle, TextureUsage,
    VertexAttribute, VertexBinding, VertexFormat, VertexLayoutDescriptor, VertexStepMode,
    FRAMES_IN_FLIGHT,
};
use std::collections::HashMap;

/// SPIR-V of the quad vertex stage (source: `shaders/quad_batch.vert`).
pub const QUAD_BATCH_VERT_SPIRV: &[u8] = include_bytes!("shaders/quad_batch.vert.spv");
/// SPIR-V of the quad fragment stage (source: `shaders/quad_batch.frag`).
pub const QUAD_BATCH_FRAG_SPIRV: &[u8] = include_bytes!("shaders/quad_batch.frag.spv");

/// Bytes of the `Canvas` push-constant block of `quad_batch.vert`.
const CANVAS_PUSH_CONSTANTS_SIZE: u32 = 16;

/// A growable buffer of one frame slot.
#[derive(Copy, Clone, Default)]
struct FrameBuffer {
    buffer: Option<BufferHandle>,
    capacity: u64,
}

impl FrameBuffer {
    /// Writes `data`, replacing the buffer by one of twice the needed size if it is too small.
    fn upload(
        &mut self,
        device: &dyn GraphicsDevice,
        usage: BufferUsage,
        label: &'static str,
        data: &[u8],
    ) -> Result<BufferHandle, RhiError> {
        let needed = data.len() as u64;
        let buffer = match self.buffer {
            Some(buffer) if self.capacity >= needed => buffer,
            old => {
                if let Some(old) = old {
                    device.destroy_buffer(old)?;
                }
                let capacity = needed.next_power_of_two().max(4096);
                let buffer = device.create_buffer(
                    &BufferDescriptor {
                        size: capacity,
                        usage,
                        memory: MemoryUsage::CpuToGpu,
                        label: Some(label),
                    },
                    None,
                )?;
                self.buffer = Some(buffer);
                self.capacity = capacity;
                buffer
            }
        };
        device.write_buffer(buffer, 0, data)?;
        Ok(buffer)
    }
}

/// Draws [`QuadBatch2d`]s (textured and solid-colour, alpha-blended) into a colour target.
pub struct QuadBatchRenderer {
    vertex_shader: ShaderHandle,
    fragment_shader: ShaderHandle,
    bind_group_layout: BindGroupLayoutHandle,
    sampler: SamplerHandle,
    white_texture: TextureHandle,
    white_bind_group: BindGroupHandle,
    texture_bind_groups: HashMap<TextureHandle, BindGroupHandle>,
    pipeline: PipelineHandle,
    color_format: TextureFormat,
    vertices: [FrameBuffer; FRAMES_IN_FLIGHT],
    indices: [FrameBuffer; FRAMES_IN_FLIGHT],
}

impl QuadBatchRenderer {
    /// Creates the pipeline and default resources for targets of `color_format`.
    pub fn new(device: &dyn GraphicsDevice, color_format: TextureFormat) -> Result<Self, RhiError> {
        let vertex_shader = device.create_shader(&ShaderDescriptor {
            stage: ShaderStage::Vertex,
            entry_point: "main",
            source: ShaderSource::SpirvBytes(QUAD_BATCH_VERT_SPIRV),
            label: Some("quad_batch_vert"),
        })?;
        let fragment_shader = match device.create_shader(&ShaderDescriptor {
            stage: ShaderStage::Fragment,
            entry_point: "main",
            source: ShaderSource::SpirvBytes(QUAD_BATCH_FRAG_SPIRV),
            label: Some("quad_batch_frag"),
        }) {
            Ok(shader) => shader,
            Err(error) => {
                let _ = device.destroy_shader(vertex_shader);
                return Err(error);
            }
        };
        let bind_group_layout = match device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                ty: BindingType::SampledTexture,
                stages: ShaderStages::FRAGMENT,
            }],
            label: Some("quad_batch_bgl"),
        }) {
            Ok(bgl) => bgl,
            Err(error) => {
                let _ = device.destroy_shader(vertex_shader);
                let _ = device.destroy_shader(fragment_shader);
                return Err(error);
            }
        };
        let sampler = match device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            label: Some("quad_batch_sampler"),
            ..Default::default()
        }) {
            Ok(s) => s,
            Err(error) => {
                let _ = device.destroy_bind_group_layout(bind_group_layout);
                let _ = device.destroy_shader(vertex_shader);
                let _ = device.destroy_shader(fragment_shader);
                return Err(error);
            }
        };
        let white_texture = match device.create_texture(
            &TextureDescriptor {
                width: 1,
                height: 1,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::SAMPLED,
                dimension: TextureDimension::D2,
                label: Some("quad_batch_white_1x1"),
            },
            Some(&[255, 255, 255, 255]),
        ) {
            Ok(tex) => tex,
            Err(error) => {
                let _ = device.destroy_sampler(sampler);
                let _ = device.destroy_bind_group_layout(bind_group_layout);
                let _ = device.destroy_shader(vertex_shader);
                let _ = device.destroy_shader(fragment_shader);
                return Err(error);
            }
        };
        let white_bind_group = match device.create_bind_group(&BindGroupDescriptor {
            layout: bind_group_layout,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    texture: white_texture.into(),
                    sampler,
                    layout: ImageLayout::ShaderReadOnly,
                },
            }],
            label: Some("quad_batch_white_bg"),
        }) {
            Ok(bg) => bg,
            Err(error) => {
                let _ = device.destroy_texture(white_texture);
                let _ = device.destroy_sampler(sampler);
                let _ = device.destroy_bind_group_layout(bind_group_layout);
                let _ = device.destroy_shader(vertex_shader);
                let _ = device.destroy_shader(fragment_shader);
                return Err(error);
            }
        };
        let float_size = std::mem::size_of::<f32>() as u64;
        let vertex_layout = VertexLayoutDescriptor::new()
            .with_binding(VertexBinding {
                binding: 0,
                stride: std::mem::size_of::<QuadVertex2d>() as u64,
                step_mode: VertexStepMode::Vertex,
            })
            .with_attribute(VertexAttribute {
                location: 0,
                binding: 0,
                format: VertexFormat::Float32x2,
                offset: 0,
            })
            .with_attribute(VertexAttribute {
                location: 1,
                binding: 0,
                format: VertexFormat::Float32x2,
                offset: 2 * float_size,
            })
            .with_attribute(VertexAttribute {
                location: 2,
                binding: 0,
                format: VertexFormat::Float32x4,
                offset: 4 * float_size,
            });
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout,
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::None,
                front_face: FrontFace::CounterClockwise,
                polygon_mode: PolygonMode::Fill,
                depth_bias: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_bias_clamp: 0.0,
                depth_clamp: false,
            },
            multisample: Default::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: color_format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::VERTEX,
                offset: 0,
                size: CANVAS_PUSH_CONSTANTS_SIZE,
            }],
            label: Some("quad_batch_pipeline"),
        });
        let pipeline = match pipeline {
            Ok(pipeline) => pipeline,
            Err(error) => {
                let _ = device.destroy_bind_group(white_bind_group);
                let _ = device.destroy_texture(white_texture);
                let _ = device.destroy_sampler(sampler);
                let _ = device.destroy_bind_group_layout(bind_group_layout);
                let _ = device.destroy_shader(vertex_shader);
                let _ = device.destroy_shader(fragment_shader);
                return Err(error);
            }
        };
        Ok(Self {
            vertex_shader,
            fragment_shader,
            bind_group_layout,
            sampler,
            white_texture,
            white_bind_group,
            texture_bind_groups: HashMap::new(),
            pipeline,
            color_format,
            vertices: [FrameBuffer::default(); FRAMES_IN_FLIGHT],
            indices: [FrameBuffer::default(); FRAMES_IN_FLIGHT],
        })
    }

    /// Format of the targets this renderer draws into.
    pub fn color_format(&self) -> TextureFormat {
        self.color_format
    }

    fn bind_group_for_texture(
        &mut self,
        device: &dyn GraphicsDevice,
        texture: Option<TextureHandle>,
    ) -> Result<BindGroupHandle, RhiError> {
        let Some(tex) = texture else {
            return Ok(self.white_bind_group);
        };
        if let Some(&bg) = self.texture_bind_groups.get(&tex) {
            return Ok(bg);
        }
        let bg = device.create_bind_group(&BindGroupDescriptor {
            layout: self.bind_group_layout,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    texture: tex.into(),
                    sampler: self.sampler,
                    layout: ImageLayout::ShaderReadOnly,
                },
            }],
            label: Some("quad_batch_tex_bg"),
        })?;
        self.texture_bind_groups.insert(tex, bg);
        Ok(bg)
    }

    fn prepare_draw(
        &mut self,
        device: &dyn GraphicsDevice,
        frame: FrameIndex,
        batch: &QuadBatch2d,
    ) -> Result<Option<(BufferHandle, BufferHandle)>, RhiError> {
        if batch.indices().is_empty() {
            return Ok(None);
        }
        let slot = frame.index();
        let vertices = self.vertices[slot].upload(
            device,
            BufferUsage::VERTEX,
            "quad_batch_vertices",
            vertex_bytes(batch.vertices()),
        )?;
        let indices = self.indices[slot].upload(
            device,
            BufferUsage::INDEX,
            "quad_batch_indices",
            index_bytes_of(batch.indices()),
        )?;
        // Pre-resolve bind groups for every unique texture in the batch before opening the render pass.
        for run in batch.draws() {
            self.bind_group_for_texture(device, run.texture)?;
        }
        Ok(Some((vertices, indices)))
    }

    fn record_draws(
        &self,
        recorder: &mut dyn CommandBufferRecorder,
        target_size: (u32, u32),
        canvas_size: (f32, f32),
        batch: &QuadBatch2d,
        vertices: BufferHandle,
        indices: BufferHandle,
    ) -> Result<(), RhiError> {
        let (width, height) = target_size;
        recorder.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        recorder.set_scissor(0, 0, width, height);
        recorder.bind_pipeline(self.pipeline);
        let decode_srgb = if self.color_format.is_srgb() {
            1.0
        } else {
            0.0
        };
        let canvas = [
            2.0 / canvas_size.0.max(f32::EPSILON),
            2.0 / canvas_size.1.max(f32::EPSILON),
            decode_srgb,
            0.0_f32,
        ];
        let mut push = [0_u8; CANVAS_PUSH_CONSTANTS_SIZE as usize];
        for (bytes, value) in push.as_chunks_mut::<4>().0.iter_mut().zip(canvas) {
            *bytes = value.to_ne_bytes();
        }
        recorder.push_constants(0, &push)?;
        recorder.bind_vertex_buffers(0, &[vertices], &[0]);
        recorder.bind_index_buffer(indices, 0, IndexType::Uint32);

        let fallback_run = [QuadBatchDraw {
            texture: None,
            first_index: 0,
            index_count: batch.indices().len() as u32,
        }];
        let runs = if batch.draws().is_empty() {
            &fallback_run[..]
        } else {
            batch.draws()
        };
        for run in runs {
            let bg = match run.texture {
                Some(tex) => self
                    .texture_bind_groups
                    .get(&tex)
                    .copied()
                    .unwrap_or(self.white_bind_group),
                None => self.white_bind_group,
            };
            recorder.set_bind_group(0, bg, &[]);
            recorder.draw_indexed(run.index_count, 1, run.first_index, 0, 0);
        }
        Ok(())
    }

    /// Uploads `batch` into the frame slot `frame` buffers and records its draw commands inside
    /// an already-open dynamic rendering pass of `target_size` pixels (such as a [`RenderGraph`](crate::graph::RenderGraph) graphics pass).
    pub fn record_in_pass(
        &mut self,
        device: &dyn GraphicsDevice,
        recorder: &mut dyn CommandBufferRecorder,
        frame: FrameIndex,
        target_size: (u32, u32),
        canvas_size: (f32, f32),
        batch: &QuadBatch2d,
    ) -> Result<(), RhiError> {
        if let Some((vertices, indices)) = self.prepare_draw(device, frame, batch)? {
            self.record_draws(recorder, target_size, canvas_size, batch, vertices, indices)?;
        }
        Ok(())
    }

    /// Records a pass over the whole `target` (`target_size` pixels, in the colour-attachment
    /// layout) that clears it to `clear_color` and draws `batch`.
    ///
    /// The batch's coordinates are canvas units with a top-left origin, `canvas_size` of them
    /// spanning the target (logical window pixels, for example). Colours, including
    /// `clear_color`, are taken as authored sRGB values: on an sRGB target they are decoded
    /// first, so the target shows them unchanged.
    /// The upload goes into the buffers of frame slot `frame`, which must not be in use by the GPU
    /// (call after `GraphicsDevice::begin_frame` returned that slot).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &dyn GraphicsDevice,
        recorder: &mut dyn CommandBufferRecorder,
        frame: FrameIndex,
        target: TextureHandle,
        target_size: (u32, u32),
        canvas_size: (f32, f32),
        clear_color: [f32; 4],
        batch: &QuadBatch2d,
    ) -> Result<(), RhiError> {
        let (width, height) = target_size;
        let clear_color = if self.color_format.is_srgb() {
            let [r, g, b, a] = clear_color;
            [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a]
        } else {
            clear_color
        };
        let draw = self.prepare_draw(device, frame, batch)?;

        recorder.begin_rendering(&RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width,
            height,
            color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                texture: target,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: Some(clear_color),
                resolve_target: None,
            }],
            depth_attachment: None,
        })?;
        if let Some((vertices, indices)) = draw {
            self.record_draws(recorder, target_size, canvas_size, batch, vertices, indices)?;
        }
        recorder.end_rendering()
    }

    /// Destroys the pipeline, bind groups, default texture, sampler, shaders and buffers.
    /// The GPU must be done with them.
    pub fn destroy(self, device: &dyn GraphicsDevice) -> Result<(), RhiError> {
        for buffer in self
            .vertices
            .iter()
            .chain(self.indices.iter())
            .filter_map(|b| b.buffer)
        {
            device.destroy_buffer(buffer)?;
        }
        for (_, bg) in self.texture_bind_groups {
            device.destroy_bind_group(bg)?;
        }
        device.destroy_bind_group(self.white_bind_group)?;
        device.destroy_texture(self.white_texture)?;
        device.destroy_sampler(self.sampler)?;
        device.destroy_pipeline(self.pipeline)?;
        device.destroy_bind_group_layout(self.bind_group_layout)?;
        device.destroy_shader(self.vertex_shader)?;
        device.destroy_shader(self.fragment_shader)
    }
}

/// The sRGB transfer function's inverse (IEC 61966-2-1), as in `quad_batch.vert`.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The bytes of a vertex slice, as the vertex layout above reads them.
fn vertex_bytes(vertices: &[QuadVertex2d]) -> &[u8] {
    korppi_core::pod::cast_slice(vertices)
}

/// The bytes of an index slice.
fn index_bytes_of(indices: &[u32]) -> &[u8] {
    korppi_core::pod::cast_slice(indices)
}
