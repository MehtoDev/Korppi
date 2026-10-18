//! Integration tests for RHI traits, descriptors, and hardware abstractions.

use korppi_rhi::{
    align_to_ubo_offset, AccessFlags, BlendState, BufferDescriptor, BufferHandle,
    BufferTextureCopy, BufferUsage, ColorTargetState, ColorWriteMask, CommandBuffer,
    CommandBufferRecorder, ComputePipelineDescriptor, CullMode, DepthAttachment, DepthStencilState,
    FrontFace, GraphicsDevice, GraphicsPipelineDescriptor, ImageLayout, ImageMemoryBarrier,
    IndexType, LoadOp, MemoryUsage, PipelineBarrier, PipelineHandle, PipelineStageFlags,
    PolygonMode, PrimitiveTopology, PushConstantRange, RasterizerState, RenderingAttachment,
    RenderingDescriptor, RhiError, SamplerDescriptor, SamplerHandle, ShaderDescriptor,
    ShaderHandle, ShaderSource, ShaderStage, StoreOp, TextureDescriptor, TextureDimension,
    TextureFormat, TextureHandle, TextureUsage, VertexAttribute, VertexBinding, VertexFormat,
    VertexLayoutDescriptor, VertexStepMode, MAX_PUSH_CONSTANTS_SIZE,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

#[test]
fn test_generational_handles_and_type_safety() {
    let b1 = BufferHandle::new(1, 1);
    let b2 = BufferHandle::new(1, 2);
    let t1 = TextureHandle::new(1, 1);

    assert_ne!(
        b1, b2,
        "Handles with different generations must not be equal"
    );
    assert_eq!(b1.id(), 1);
    assert_eq!(b1.generation(), 1);
    assert_eq!(b2.generation(), 2);
    assert_eq!(t1.id(), 1);
    assert!(b1.is_valid());
    assert!(!BufferHandle::null().is_valid());
}

#[test]
fn test_normative_alignment_helpers() {
    // Vulkan & OpenGL require dynamic UBO offsets to align to minUniformBufferOffsetAlignment / GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT (typically 256)
    assert_eq!(align_to_ubo_offset(0, 256), 0);
    assert_eq!(align_to_ubo_offset(1, 256), 256);
    assert_eq!(align_to_ubo_offset(255, 256), 256);
    assert_eq!(align_to_ubo_offset(256, 256), 256);
    assert_eq!(align_to_ubo_offset(300, 256), 512);

    // Guaranteed push constant size across all platforms is 128 bytes
    assert_eq!(MAX_PUSH_CONSTANTS_SIZE, 128);
}

#[test]
fn test_texture_format_properties() {
    assert_eq!(TextureFormat::Rgba8Unorm.bytes_per_pixel(), 4);
    assert_eq!(TextureFormat::Rgba16Float.bytes_per_pixel(), 8);
    assert_eq!(TextureFormat::Rgba32Float.bytes_per_pixel(), 16);
    assert_eq!(TextureFormat::Depth32Float.bytes_per_pixel(), 4);

    assert!(TextureFormat::Depth32Float.is_depth());
    assert!(!TextureFormat::Rgba8Unorm.is_depth());
    assert!(TextureFormat::Rgba8Srgb.is_srgb());
    assert!(!TextureFormat::Rgba8Unorm.is_srgb());
}

#[test]
fn test_vertex_layout_stride_and_offsets() {
    let layout = VertexLayoutDescriptor::new()
        .with_binding(VertexBinding {
            binding: 0,
            stride: 32,
            step_mode: VertexStepMode::Vertex,
        })
        .with_attribute(VertexAttribute {
            location: 0,
            binding: 0,
            format: VertexFormat::Float32x3, // Position: 12 bytes
            offset: 0,
        })
        .with_attribute(VertexAttribute {
            location: 1,
            binding: 0,
            format: VertexFormat::Float32x3, // Normal: 12 bytes
            offset: 12,
        })
        .with_attribute(VertexAttribute {
            location: 2,
            binding: 0,
            format: VertexFormat::Float32x2, // UV: 8 bytes
            offset: 24,
        });

    assert_eq!(layout.bindings.len(), 1);
    assert_eq!(layout.attributes.len(), 3);
    assert!(layout.validate().is_ok());

    // Invalid binding reference
    let invalid_layout = VertexLayoutDescriptor::new().with_attribute(VertexAttribute {
        location: 0,
        binding: 5, // Doesn't exist
        format: VertexFormat::Float32,
        offset: 0,
    });
    assert!(invalid_layout.validate().is_err());
}

#[test]
fn test_pipeline_push_constant_bounds() {
    let valid_pipeline = GraphicsPipelineDescriptor {
        vertex_shader: ShaderHandle::new(1, 1),
        fragment_shader: None,
        vertex_layout: VertexLayoutDescriptor::default(),
        topology: PrimitiveTopology::TriangleList,
        rasterizer: RasterizerState::default(),
        multisample: Default::default(),
        depth_stencil: None,
        color_targets: vec![],
        bind_group_layouts: vec![],
        push_constant_ranges: vec![PushConstantRange {
            stages: korppi_rhi::ShaderStages::VERTEX,
            offset: 0,
            size: 128,
        }],
        label: Some("test_pipeline"),
    };
    assert!(valid_pipeline.validate().is_ok());

    let invalid_pipeline = GraphicsPipelineDescriptor {
        vertex_shader: ShaderHandle::new(1, 1),
        fragment_shader: None,
        vertex_layout: VertexLayoutDescriptor::default(),
        topology: PrimitiveTopology::TriangleList,
        rasterizer: RasterizerState::default(),
        multisample: Default::default(),
        depth_stencil: None,
        color_targets: vec![],
        bind_group_layouts: vec![],
        push_constant_ranges: vec![PushConstantRange {
            stages: korppi_rhi::ShaderStages::VERTEX,
            offset: 64,
            size: 65, // 64 + 65 = 129 > 128!
        }],
        label: None,
    };
    assert!(invalid_pipeline.validate().is_err());
}

#[test]
fn test_dynamic_rendering_and_barrier_structures() {
    let color_tex = TextureHandle::new(1, 1);
    let depth_tex = TextureHandle::new(2, 1);

    let rendering_desc = RenderingDescriptor {
        offset_x: 0,
        offset_y: 0,
        width: 1920,
        height: 1080,
        color_attachments: korppi_rhi::smallvec![RenderingAttachment {
            texture: color_tex,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color: Some([0.1, 0.2, 0.3, 1.0]),
            resolve_target: None,
        }],
        depth_attachment: Some(DepthAttachment {
            texture: depth_tex,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_depth: Some(0.0), // Reverse-Z: 0.0 is Far
            clear_stencil: None,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
        }),
    };

    assert_eq!(rendering_desc.width, 1920);
    assert_eq!(rendering_desc.color_attachments.len(), 1);

    let barrier = PipelineBarrier::default().with_image_barrier(ImageMemoryBarrier {
        texture: color_tex,
        range: korppi_rhi::TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stages: PipelineStageFlags::FRAGMENT_SHADER,
        dst_access: AccessFlags::SHADER_READ,
        old_layout: ImageLayout::ColorAttachment,
        new_layout: ImageLayout::ShaderReadOnly,
    });

    assert_eq!(barrier.image_barriers.len(), 1);
    assert_eq!(
        barrier.image_barriers[0].new_layout,
        ImageLayout::ShaderReadOnly
    );
}

// ---------------------------------------------------------------------------
// Mock Device & Command Recorder Implementation
// ---------------------------------------------------------------------------

struct MockCommandBuffer;

impl CommandBuffer for MockCommandBuffer {}

struct MockCommandRecorder {
    commands: Vec<String>,
}

impl CommandBufferRecorder for MockCommandRecorder {
    fn begin(&mut self) -> Result<(), RhiError> {
        self.commands.push("begin".into());
        Ok(())
    }

    fn begin_rendering(&mut self, _desc: &RenderingDescriptor) -> Result<(), RhiError> {
        self.commands.push("begin_rendering".into());
        Ok(())
    }

    fn end_rendering(&mut self) -> Result<(), RhiError> {
        self.commands.push("end_rendering".into());
        Ok(())
    }

    fn set_viewport(
        &mut self,
        _x: f32,
        _y: f32,
        _width: f32,
        _height: f32,
        _min_depth: f32,
        _max_depth: f32,
    ) {
        self.commands.push("set_viewport".into());
    }

    fn set_scissor(&mut self, _x: i32, _y: i32, _width: u32, _height: u32) {
        self.commands.push("set_scissor".into());
    }

    fn set_stencil_reference(&mut self, _reference: u32) {
        self.commands.push("set_stencil_reference".into());
    }

    fn set_blend_constant(&mut self, _color: [f32; 4]) {
        self.commands.push("set_blend_constant".into());
    }

    fn bind_pipeline(&mut self, _pipeline: PipelineHandle) {
        self.commands.push("bind_pipeline".into());
    }

    fn bind_vertex_buffers(&mut self, _first: u32, _buffers: &[BufferHandle], _offsets: &[u64]) {
        self.commands.push("bind_vertex_buffers".into());
    }

    fn bind_index_buffer(&mut self, _buffer: BufferHandle, _offset: u64, _index_type: IndexType) {
        self.commands.push("bind_index_buffer".into());
    }

    fn set_bind_group(
        &mut self,
        _index: u32,
        _group: korppi_rhi::BindGroupHandle,
        _dynamic_offsets: &[u32],
    ) {
        self.commands.push("set_bind_group".into());
    }

    fn push_constants(&mut self, offset: u32, data: &[u8]) -> Result<(), RhiError> {
        if (offset as usize + data.len()) > MAX_PUSH_CONSTANTS_SIZE {
            return Err(RhiError::ValidationError(
                "Push constants out of bounds".into(),
            ));
        }
        self.commands.push("push_constants".into());
        Ok(())
    }

    fn draw(&mut self, _v_count: u32, _i_count: u32, _first_v: u32, _first_i: u32) {
        self.commands.push("draw".into());
    }

    fn draw_indexed(
        &mut self,
        _i_count: u32,
        _instance_count: u32,
        _first_i: u32,
        _v_offset: i32,
        _first_instance: u32,
    ) {
        self.commands.push("draw_indexed".into());
    }

    fn dispatch(&mut self, _x: u32, _y: u32, _z: u32) {
        self.commands.push("dispatch".into());
    }

    fn pipeline_barrier(&mut self, _barrier: &PipelineBarrier) {
        self.commands.push("pipeline_barrier".into());
    }

    fn copy_buffer_to_buffer(
        &mut self,
        _src: BufferHandle,
        _s_off: u64,
        _dst: BufferHandle,
        _d_off: u64,
        _sz: u64,
    ) {
        self.commands.push("copy_buffer_to_buffer".into());
    }

    fn fill_buffer(&mut self, _buffer: BufferHandle, _offset: u64, _size: u64, _value: u32) {
        self.commands.push("fill_buffer".into());
    }

    fn clear_texture(
        &mut self,
        _texture: TextureHandle,
        _range: &korppi_rhi::TextureSubresourceRange,
        _value: korppi_rhi::TextureClear,
    ) {
        self.commands.push("clear_texture".into());
    }

    fn copy_buffer_to_texture(
        &mut self,
        _src: BufferHandle,
        _dst: TextureHandle,
        _copy: &BufferTextureCopy,
    ) {
        self.commands.push("copy_buffer_to_texture".into());
    }

    fn copy_texture_to_buffer(
        &mut self,
        _src: TextureHandle,
        _dst: BufferHandle,
        _copy: &BufferTextureCopy,
    ) {
        self.commands.push("copy_texture_to_buffer".into());
    }

    fn copy_texture_to_texture(
        &mut self,
        _src: TextureHandle,
        _dst: TextureHandle,
        _copy: &korppi_rhi::TextureCopy,
    ) {
        self.commands.push("copy_texture_to_texture".into());
    }

    fn blit_texture(
        &mut self,
        _src: TextureHandle,
        _dst: TextureHandle,
        _blit: &korppi_rhi::TextureBlit,
    ) {
        self.commands.push("blit_texture".into());
    }

    fn generate_mipmaps(&mut self, _texture: TextureHandle) {
        self.commands.push("generate_mipmaps".into());
    }

    fn draw_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
        self.commands.push("draw_indirect".into());
    }

    fn draw_indexed_indirect(&mut self, _: BufferHandle, _: u64, _: u32, _: u32) {
        self.commands.push("draw_indexed_indirect".into());
    }

    fn dispatch_indirect(&mut self, _: BufferHandle, _: u64) {
        self.commands.push("dispatch_indirect".into());
    }

    fn write_timestamp(&mut self, _: korppi_rhi::QuerySetHandle, _: u32) {
        self.commands.push("write_timestamp".into());
    }

    fn resolve_query_set(
        &mut self,
        _: korppi_rhi::QuerySetHandle,
        _: u32,
        _: u32,
        _: BufferHandle,
        _: u64,
    ) {
        self.commands.push("resolve_query_set".into());
    }

    fn push_debug_group(&mut self, _: &str) {
        self.commands.push("push_debug_group".into());
    }

    fn pop_debug_group(&mut self) {
        self.commands.push("pop_debug_group".into());
    }

    fn insert_debug_marker(&mut self, _: &str) {
        self.commands.push("insert_debug_marker".into());
    }

    fn finish(self: Box<Self>) -> Result<Box<dyn CommandBuffer>, RhiError> {
        Ok(Box::new(MockCommandBuffer))
    }
}

struct MockDevice {
    counter: AtomicU32,
    buffers: Mutex<Vec<Vec<u8>>>,
    frames: korppi_rhi::FrameTracker,
}

impl MockDevice {
    fn new() -> Self {
        Self {
            counter: AtomicU32::new(1),
            buffers: Mutex::new(Vec::new()),
            frames: korppi_rhi::FrameTracker::new(),
        }
    }
}

impl GraphicsDevice for MockDevice {
    fn adapter_info(&self) -> korppi_rhi::AdapterInfo {
        korppi_rhi::AdapterInfo {
            api: korppi_rhi::GraphicsApi::Vulkan,
            name: "mock".into(),
            vendor_id: 0,
            device_id: 0,
            device_type: korppi_rhi::DeviceType::Other,
            driver: String::new(),
            api_version: (1, 3, 0),
        }
    }

    fn limits(&self) -> korppi_rhi::DeviceLimits {
        korppi_rhi::DeviceLimits {
            max_texture_dimension_1d: 4096,
            max_texture_dimension_2d: 4096,
            max_texture_dimension_3d: 256,
            max_texture_dimension_cube: 4096,
            max_texture_array_layers: 256,
            max_color_attachments: 4,
            max_vertex_buffers: 16,
            max_vertex_attributes: 16,
            max_vertex_buffer_stride: 2048,
            max_push_constants_size: 128,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_uniform_buffer_binding_size: 16384,
            max_storage_buffer_binding_size: 1 << 27,
            max_per_stage_uniform_buffers: 12,
            max_per_stage_storage_buffers: 4,
            max_per_stage_sampled_textures: 16,
            max_per_stage_storage_textures: 4,
            max_sampler_anisotropy: 1.0,
            max_compute_workgroup_size: [128, 128, 64],
            max_compute_invocations_per_workgroup: 128,
            max_compute_workgroup_count: [65535; 3],
            max_compute_shared_memory_size: 16384,
            timestamp_period_ns: 0.0,
        }
    }

    fn features(&self) -> korppi_rhi::DeviceFeatures {
        korppi_rhi::DeviceFeatures::empty()
    }

    fn format_capabilities(&self, _format: TextureFormat) -> korppi_rhi::FormatCapabilities {
        korppi_rhi::FormatCapabilities::all()
    }

    fn create_texture_view(
        &self,
        _desc: &korppi_rhi::TextureViewDescriptor,
    ) -> Result<korppi_rhi::TextureViewHandle, RhiError> {
        Ok(korppi_rhi::TextureViewHandle::new(1, 1))
    }

    fn destroy_texture_view(&self, _view: korppi_rhi::TextureViewHandle) -> Result<(), RhiError> {
        Ok(())
    }

    fn create_bind_group_layout(
        &self,
        desc: &korppi_rhi::BindGroupLayoutDescriptor,
    ) -> Result<korppi_rhi::BindGroupLayoutHandle, RhiError> {
        desc.validate()?;
        Ok(korppi_rhi::BindGroupLayoutHandle::new(1, 1))
    }

    fn destroy_bind_group_layout(
        &self,
        _layout: korppi_rhi::BindGroupLayoutHandle,
    ) -> Result<(), RhiError> {
        Ok(())
    }

    fn create_bind_group(
        &self,
        _desc: &korppi_rhi::BindGroupDescriptor,
    ) -> Result<korppi_rhi::BindGroupHandle, RhiError> {
        Ok(korppi_rhi::BindGroupHandle::new(1, 1))
    }

    fn destroy_bind_group(&self, _group: korppi_rhi::BindGroupHandle) -> Result<(), RhiError> {
        Ok(())
    }

    fn create_buffer(
        &self,
        desc: &BufferDescriptor,
        initial_data: Option<&[u8]>,
    ) -> Result<BufferHandle, RhiError> {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        let mut bufs = self.buffers.lock().unwrap();
        let mut data = vec![0u8; desc.size as usize];
        if let Some(init) = initial_data {
            data[..init.len()].copy_from_slice(init);
        }
        bufs.push(data);
        Ok(BufferHandle::new(id, 1))
    }

    fn destroy_buffer(&self, _handle: BufferHandle) -> Result<(), RhiError> {
        Ok(())
    }

    fn write_buffer(&self, handle: BufferHandle, offset: u64, data: &[u8]) -> Result<(), RhiError> {
        let mut bufs = self.buffers.lock().unwrap();
        let idx = (handle.id() - 1) as usize;
        if idx < bufs.len() {
            let buf = &mut bufs[idx];
            let end = offset as usize + data.len();
            if end <= buf.len() {
                buf[offset as usize..end].copy_from_slice(data);
                return Ok(());
            }
        }
        Err(RhiError::ValidationError("Buffer overflow".into()))
    }

    fn read_buffer(
        &self,
        handle: BufferHandle,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, RhiError> {
        let bufs = self.buffers.lock().unwrap();
        let idx = (handle.id() - 1) as usize;
        if idx < bufs.len() {
            let buf = &bufs[idx];
            let end = offset as usize + size as usize;
            if end <= buf.len() {
                return Ok(buf[offset as usize..end].to_vec());
            }
        }
        Err(RhiError::ValidationError("Buffer out of bounds".into()))
    }

    fn create_texture(
        &self,
        _desc: &TextureDescriptor,
        _init: Option<&[u8]>,
    ) -> Result<TextureHandle, RhiError> {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(TextureHandle::new(id, 1))
    }

    fn destroy_texture(&self, _handle: TextureHandle) -> Result<(), RhiError> {
        Ok(())
    }

    fn read_texture(&self, _handle: TextureHandle, _mip: u32) -> Result<Vec<u8>, RhiError> {
        Ok(vec![255, 0, 0, 255])
    }

    fn create_sampler(&self, _desc: &SamplerDescriptor) -> Result<SamplerHandle, RhiError> {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(SamplerHandle::new(id, 1))
    }

    fn destroy_sampler(&self, _handle: SamplerHandle) -> Result<(), RhiError> {
        Ok(())
    }

    fn create_shader(&self, _desc: &ShaderDescriptor) -> Result<ShaderHandle, RhiError> {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(ShaderHandle::new(id, 1))
    }

    fn destroy_shader(&self, _handle: ShaderHandle) -> Result<(), RhiError> {
        Ok(())
    }

    fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        desc.validate()?;
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(PipelineHandle::new(id, 1))
    }

    fn create_compute_pipeline(
        &self,
        _desc: &ComputePipelineDescriptor,
    ) -> Result<PipelineHandle, RhiError> {
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(PipelineHandle::new(id, 1))
    }

    fn destroy_pipeline(&self, _handle: PipelineHandle) -> Result<(), RhiError> {
        Ok(())
    }

    fn create_command_recorder(&self) -> Result<Box<dyn CommandBufferRecorder>, RhiError> {
        Ok(Box::new(MockCommandRecorder {
            commands: Vec::new(),
        }))
    }

    /// The mock executes nothing, so every submission is complete as soon as it is made.
    fn submit(&self, _cmd: Box<dyn CommandBuffer>) -> Result<korppi_rhi::SubmissionId, RhiError> {
        let id =
            korppi_rhi::SubmissionId::new(u64::from(self.counter.fetch_add(1, Ordering::SeqCst)));
        self.frames.record(id);
        Ok(id)
    }

    fn is_submission_complete(&self, _id: korppi_rhi::SubmissionId) -> Result<bool, RhiError> {
        Ok(true)
    }

    fn wait_for_submission(&self, _id: korppi_rhi::SubmissionId) -> Result<(), RhiError> {
        Ok(())
    }

    fn wait_idle(&self) -> Result<(), RhiError> {
        Ok(())
    }

    fn begin_frame(&self) -> Result<korppi_rhi::FrameIndex, RhiError> {
        Ok(self.frames.begin_frame().0)
    }
}

#[test]
fn test_mock_device_lifecycle_and_command_execution() {
    let device = MockDevice::new();

    // 1. Buffer creation, write, read
    let buf_desc = BufferDescriptor {
        size: 64,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
        memory: MemoryUsage::GpuOnly,
        label: Some("vertex_buffer"),
    };
    let buffer = device
        .create_buffer(&buf_desc, None)
        .expect("Buffer creation failed");
    let test_data = [42u8; 16];
    device
        .write_buffer(buffer, 0, &test_data)
        .expect("Buffer write failed");
    let read_back = device
        .read_buffer(buffer, 0, 16)
        .expect("Buffer read failed");
    assert_eq!(&read_back, &test_data);

    // 2. Texture & Sampler creation
    let tex_desc = TextureDescriptor {
        width: 256,
        height: 256,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: korppi_rhi::SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        dimension: TextureDimension::D2,
        label: Some("render_target"),
    };
    let texture = device
        .create_texture(&tex_desc, None)
        .expect("Texture creation failed");
    let sampler = device
        .create_sampler(&SamplerDescriptor::default())
        .expect("Sampler creation failed");

    // 3. Shader & Pipeline creation
    let vert_shader = device
        .create_shader(&ShaderDescriptor {
            stage: ShaderStage::Vertex,
            entry_point: "main",
            source: ShaderSource::Spirv(&[0x07230203]),
            label: Some("vert_shader"),
        })
        .expect("Vertex shader failed");

    let pipeline = device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: vert_shader,
            fragment_shader: None,
            vertex_layout: VertexLayoutDescriptor::new().with_binding(VertexBinding {
                binding: 0,
                stride: 16,
                step_mode: VertexStepMode::Vertex,
            }),
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::Clockwise,
                polygon_mode: PolygonMode::Fill,
                depth_bias: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_bias_clamp: 0.0,
                depth_clamp: false,
            },
            multisample: Default::default(),
            depth_stencil: Some(DepthStencilState::default()),
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![PushConstantRange {
                stages: korppi_rhi::ShaderStages::VERTEX,
                offset: 0,
                size: 64,
            }],
            label: Some("test_pso"),
        })
        .expect("Pipeline creation failed");

    // 4. Command Recording
    let mut recorder = device
        .create_command_recorder()
        .expect("Failed to create command recorder");
    recorder.begin().expect("Begin failed");
    recorder.set_viewport(0.0, 1080.0, 1920.0, -1080.0, 0.0, 1.0);
    recorder.set_scissor(0, 0, 1920, 1080);
    recorder.bind_pipeline(pipeline);
    recorder.bind_vertex_buffers(0, &[buffer], &[0]);
    let layout = device
        .create_bind_group_layout(&korppi_rhi::BindGroupLayoutDescriptor {
            entries: vec![korppi_rhi::BindGroupLayoutEntry {
                binding: 0,
                ty: korppi_rhi::BindingType::SampledTexture,
                stages: korppi_rhi::ShaderStages::FRAGMENT,
            }],
            label: None,
        })
        .expect("Bind group layout creation failed");
    let group = device
        .create_bind_group(&korppi_rhi::BindGroupDescriptor {
            layout,
            entries: vec![korppi_rhi::BindGroupEntry {
                binding: 0,
                resource: korppi_rhi::BindingResource::SampledTexture {
                    texture: texture.into(),
                    sampler,
                    layout: korppi_rhi::ImageLayout::ShaderReadOnly,
                },
            }],
            label: None,
        })
        .expect("Bind group creation failed");
    recorder.set_bind_group(0, group, &[]);
    recorder
        .push_constants(0, &[1, 2, 3, 4])
        .expect("Push constants failed");
    recorder
        .begin_rendering(&RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width: 1920,
            height: 1080,
            color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                texture,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: Some([0.0, 0.0, 0.0, 1.0]),
                resolve_target: None,
            }],
            depth_attachment: None,
        })
        .expect("Begin rendering failed");
    recorder.draw(3, 1, 0, 0);
    recorder.end_rendering().expect("End rendering failed");

    let cmd_buffer = recorder.finish().expect("Finish failed");
    device.submit(cmd_buffer).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");
}

#[test]
fn devices_without_presentation_reject_swapchain_calls() {
    let device = MockDevice::new();
    let unknown = korppi_rhi::SwapchainHandle::new(7, 1);
    assert!(matches!(
        device.acquire_next_image(unknown),
        Err(RhiError::InvalidHandle(_))
    ));
    assert!(matches!(
        device.resize_swapchain(unknown, 640, 480),
        Err(RhiError::InvalidHandle(_))
    ));
    assert!(matches!(
        device.destroy_swapchain(unknown),
        Err(RhiError::InvalidHandle(_))
    ));
}

#[test]
fn swapchain_descriptor_defaults_to_vsynced_srgb() {
    let desc = korppi_rhi::SwapchainDescriptor::new(1280, 720);
    assert_eq!((desc.width, desc.height), (1280, 720));
    assert_eq!(desc.present_mode, korppi_rhi::PresentMode::Fifo);
    assert!(desc.srgb);
    assert_eq!(desc.image_count, 0);
}
