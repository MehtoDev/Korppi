//! Pipeline and draw state must reach the GPU on every backend (findings RVK-03, RGL-01,
//! RGL-02, RGL-03).
//!
//! Each scenario renders into a 64x64 `Rgba8Unorm` target cleared to opaque blue and checks the
//! texels that the state under test controls.

use korppi_rhi::{
    BlendState, BufferDescriptor, BufferHandle, BufferUsage, ColorTargetState, ColorWriteMask,
    CullMode, FrontFace, GraphicsPipelineDescriptor, IndexType, LoadOp, MemoryUsage,
    PipelineHandle, PolygonMode, PrimitiveTopology, RasterizerState, RhiError, ShaderHandle,
    ShaderStage, TextureFormat, TextureUsage, VertexAttribute, VertexBinding, VertexFormat,
    VertexLayoutDescriptor, VertexStepMode,
};
use korppi_rhi_conformance::{
    approx_rgba, color_pass, color_target, undefined_to_color_attachment, Backend, Image,
    TestDevice,
};

const SIZE: u32 = 64;

/// Position from binding 0 plus a per-instance x offset from binding 1 (location 1).
const INSTANCED_POSITIONS: &str = r#"#version 450
layout(location = 0) in vec2 pos;
layout(location = 1) in float offset_x;
void main() { gl_Position = vec4(pos.x + offset_x, pos.y, 0.5, 1.0); }
"#;

/// Position from binding 0 only.
const POSITIONS: &str = r#"#version 450
layout(location = 0) in vec2 pos;
void main() { gl_Position = vec4(pos, 0.5, 1.0); }
"#;

const HALF_RED: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.0, 0.0, 0.5); }
"#;

struct Setup {
    vertex_shader: &'static str,
    layout: VertexLayoutDescriptor,
    topology: PrimitiveTopology,
    blend: Option<BlendState>,
    write_mask: ColorWriteMask,
}

fn float_buffer(device: &TestDevice, values: &[f32], usage: BufferUsage) -> BufferHandle {
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_ne_bytes()).collect();
    device
        .create_buffer(
            &BufferDescriptor {
                size: bytes.len() as u64,
                usage,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            Some(&bytes),
        )
        .expect("create buffer")
}

fn position_layout() -> VertexLayoutDescriptor {
    VertexLayoutDescriptor::new()
        .with_binding(VertexBinding {
            binding: 0,
            stride: 8,
            step_mode: VertexStepMode::Vertex,
        })
        .with_attribute(VertexAttribute {
            location: 0,
            binding: 0,
            format: VertexFormat::Float32x2,
            offset: 0,
        })
}

fn instanced_layout() -> VertexLayoutDescriptor {
    position_layout()
        .with_binding(VertexBinding {
            binding: 1,
            stride: 4,
            step_mode: VertexStepMode::Instance,
        })
        .with_attribute(VertexAttribute {
            location: 1,
            binding: 1,
            format: VertexFormat::Float32,
            offset: 0,
        })
}

fn pipeline(device: &TestDevice, setup: Setup) -> PipelineHandle {
    let vertex_shader = device.shader(ShaderStage::Vertex, setup.vertex_shader);
    let fragment_shader = device.shader(ShaderStage::Fragment, HALF_RED);
    device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout: setup.layout,
            topology: setup.topology,
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
                format: TextureFormat::Rgba8Unorm,
                blend: setup.blend,
                write_mask: setup.write_mask,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create_graphics_pipeline")
}

/// Clears to opaque blue, runs `draw` inside a full-target pass, reads the target back.
fn render(
    device: &TestDevice,
    draw: impl FnOnce(&mut dyn korppi_rhi::CommandBufferRecorder),
) -> Image {
    let target = color_target(&**device, SIZE, SIZE, TextureUsage::empty());
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    recorder
        .begin_rendering(&color_pass(
            target,
            SIZE,
            SIZE,
            LoadOp::Clear,
            [0.0, 0.0, 1.0, 1.0],
        ))
        .expect("begin_rendering");
    recorder.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
    recorder.set_scissor(0, 0, SIZE, SIZE);
    draw(recorder.as_mut());
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    Image::read(&**device, target, SIZE, SIZE)
}

fn drawn(p: [u8; 4]) -> bool {
    p[2] != 255
}

/// Texels touched in the left (x < 32) and right (x >= 32) halves of the image.
fn drawn_per_half(image: &Image) -> (usize, usize) {
    let mut left = 0;
    let mut right = 0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            if drawn(image.pixel(x, y)) {
                if x < SIZE / 2 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
    }
    (left, right)
}

// ---------------------------------------------------------------------------------------------

fn color_write_mask_is_applied(backend: Backend) {
    let device = TestDevice::open(backend);
    let positions = float_buffer(
        &device,
        &[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
        BufferUsage::VERTEX,
    );
    let pipeline = pipeline(
        &device,
        Setup {
            vertex_shader: POSITIONS,
            layout: position_layout(),
            topology: PrimitiveTopology::TriangleList,
            blend: None,
            write_mask: ColorWriteMask::RED,
        },
    );
    let image = render(&device, |r| {
        r.bind_pipeline(pipeline);
        r.bind_vertex_buffers(0, &[positions], &[0]);
        r.draw(3, 1, 0, 0);
    });
    // Only red is written: (1, 0, 0, 0.5) over (0, 0, 1, 1) leaves green, blue and alpha alone.
    let texel = image.pixel(SIZE / 2, SIZE / 2);
    assert!(
        approx_rgba(texel, [255, 0, 255, 255], 1),
        "{backend:?}: texel {texel:?}, expected [255, 0, 255, 255]"
    );
}

#[test]
fn color_write_mask_is_applied_vulkan() {
    color_write_mask_is_applied(Backend::Vulkan);
}

#[test]
fn color_write_mask_is_applied_opengl() {
    color_write_mask_is_applied(Backend::OpenGl);
}

fn per_instance_attributes_advance_per_instance(backend: Backend) {
    let device = TestDevice::open(backend);
    // A triangle confined to the left half (x in [-0.9, -0.1]) and two instance offsets.
    let positions = float_buffer(
        &device,
        &[-0.9, -0.5, -0.1, -0.5, -0.5, 0.5],
        BufferUsage::VERTEX,
    );
    let offsets = float_buffer(&device, &[0.0, 1.0], BufferUsage::VERTEX);
    let pipeline = pipeline(
        &device,
        Setup {
            vertex_shader: INSTANCED_POSITIONS,
            layout: instanced_layout(),
            topology: PrimitiveTopology::TriangleList,
            blend: None,
            write_mask: ColorWriteMask::ALL,
        },
    );
    let image = render(&device, |r| {
        r.bind_pipeline(pipeline);
        r.bind_vertex_buffers(0, &[positions, offsets], &[0, 0]);
        r.draw(3, 2, 0, 0);
    });
    let (left, right) = drawn_per_half(&image);
    assert!(
        left > 100 && right > 100,
        "{backend:?}: drawn texels left {left}, right {right}"
    );
    assert!(
        left.abs_diff(right) <= left / 10,
        "{backend:?}: instances differ: {left} vs {right}"
    );
}

#[test]
fn per_instance_attributes_advance_per_instance_vulkan() {
    per_instance_attributes_advance_per_instance(Backend::Vulkan);
}

#[test]
fn per_instance_attributes_advance_per_instance_opengl() {
    per_instance_attributes_advance_per_instance(Backend::OpenGl);
}

fn first_instance_selects_instance_data(backend: Backend) {
    let device = TestDevice::open(backend);
    let positions = float_buffer(
        &device,
        &[-0.9, -0.5, -0.1, -0.5, -0.5, 0.5],
        BufferUsage::VERTEX,
    );
    // Instance 0 would push the triangle off-screen; instance 1 moves it to the right half.
    let offsets = float_buffer(&device, &[10.0, 1.0], BufferUsage::VERTEX);
    let pipeline = pipeline(
        &device,
        Setup {
            vertex_shader: INSTANCED_POSITIONS,
            layout: instanced_layout(),
            topology: PrimitiveTopology::TriangleList,
            blend: None,
            write_mask: ColorWriteMask::ALL,
        },
    );
    let image = render(&device, |r| {
        r.bind_pipeline(pipeline);
        r.bind_vertex_buffers(0, &[positions, offsets], &[0, 0]);
        r.draw(3, 1, 0, 1);
    });
    let (left, right) = drawn_per_half(&image);
    assert!(
        left == 0 && right > 100,
        "{backend:?}: drawn texels left {left}, right {right}"
    );
}

#[test]
fn first_instance_selects_instance_data_vulkan() {
    first_instance_selects_instance_data(Backend::Vulkan);
}

#[test]
fn first_instance_selects_instance_data_opengl() {
    first_instance_selects_instance_data(Backend::OpenGl);
}

fn line_list_topology_draws_lines(backend: Backend) {
    let device = TestDevice::open(backend);
    // One horizontal line through the middle row of texel centres.
    let y = 1.0 / SIZE as f32;
    let positions = float_buffer(&device, &[-1.0, y, 1.0, y], BufferUsage::VERTEX);
    let pipeline = pipeline(
        &device,
        Setup {
            vertex_shader: POSITIONS,
            layout: position_layout(),
            topology: PrimitiveTopology::LineList,
            blend: None,
            write_mask: ColorWriteMask::ALL,
        },
    );
    let image = render(&device, |r| {
        r.bind_pipeline(pipeline);
        r.bind_vertex_buffers(0, &[positions], &[0]);
        r.draw(2, 1, 0, 0);
    });
    let count = image.count(drawn);
    let rows = image.rows_where(drawn);
    assert!(
        (SIZE as usize - 2..=SIZE as usize).contains(&count),
        "{backend:?}: a 64-texel line drew {count} texels"
    );
    assert!(
        matches!(rows, Some((a, b)) if a == b),
        "{backend:?}: line rows {rows:?}"
    );
}

#[test]
fn line_list_topology_draws_lines_vulkan() {
    line_list_topology_draws_lines(Backend::Vulkan);
}

#[test]
fn line_list_topology_draws_lines_opengl() {
    line_list_topology_draws_lines(Backend::OpenGl);
}

fn u16_indices_with_offset_and_base_vertex(backend: Backend) {
    let device = TestDevice::open(backend);
    // Three unused vertices, then the top-half triangle (-1,0) (3,0) (-1,4).
    let positions = float_buffer(
        &device,
        &[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, -1.0, 0.0, 3.0, 0.0, -1.0, 4.0],
        BufferUsage::VERTEX,
    );
    // Three padding indices (skipped by the 6-byte buffer offset), then 0, 1, 2.
    let indices: Vec<u8> = [7_u16, 7, 7, 0, 1, 2]
        .iter()
        .flat_map(|i| i.to_ne_bytes())
        .collect();
    let index_buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: indices.len() as u64,
                usage: BufferUsage::INDEX,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            Some(&indices),
        )
        .expect("create index buffer");
    let pipeline = pipeline(
        &device,
        Setup {
            vertex_shader: POSITIONS,
            layout: position_layout(),
            topology: PrimitiveTopology::TriangleList,
            blend: None,
            write_mask: ColorWriteMask::ALL,
        },
    );
    let image = render(&device, |r| {
        r.bind_pipeline(pipeline);
        r.bind_vertex_buffers(0, &[positions], &[0]);
        r.bind_index_buffer(index_buffer, 6, IndexType::Uint16);
        r.draw_indexed(3, 1, 0, 3, 0);
    });
    assert_eq!(
        image.rows_where(drawn),
        Some((0, SIZE / 2 - 1)),
        "{backend:?}: expected the top-half triangle"
    );
    assert_eq!(
        image.count(drawn),
        (SIZE * SIZE / 2) as usize,
        "{backend:?}"
    );
}

#[test]
fn u16_indices_with_offset_and_base_vertex_vulkan() {
    u16_indices_with_offset_and_base_vertex(Backend::Vulkan);
}

#[test]
fn u16_indices_with_offset_and_base_vertex_opengl() {
    u16_indices_with_offset_and_base_vertex(Backend::OpenGl);
}

fn unknown_fragment_shader_is_an_error(backend: Backend) {
    let device = TestDevice::open(backend);
    let vertex_shader = device.shader(ShaderStage::Vertex, POSITIONS);
    let result = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
        vertex_shader,
        fragment_shader: Some(ShaderHandle::new(9_999, 1)),
        vertex_layout: position_layout(),
        topology: PrimitiveTopology::TriangleList,
        rasterizer: RasterizerState::default(),
        multisample: Default::default(),
        depth_stencil: None,
        color_targets: vec![ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: ColorWriteMask::ALL,
        }],
        bind_group_layouts: vec![],
        push_constant_ranges: vec![],
        label: None,
    });
    assert!(
        matches!(result, Err(RhiError::InvalidHandle(_))),
        "{backend:?}: expected InvalidHandle, got {result:?}"
    );
}

#[test]
fn unknown_fragment_shader_is_an_error_vulkan() {
    unknown_fragment_shader_is_an_error(Backend::Vulkan);
}

#[test]
fn unknown_fragment_shader_is_an_error_opengl() {
    unknown_fragment_shader_is_an_error(Backend::OpenGl);
}

fn premultiplied_blending_is_applied(backend: Backend) {
    let device = TestDevice::open(backend);
    let positions = float_buffer(
        &device,
        &[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
        BufferUsage::VERTEX,
    );
    let pipeline = pipeline(
        &device,
        Setup {
            vertex_shader: POSITIONS,
            layout: position_layout(),
            topology: PrimitiveTopology::TriangleList,
            blend: Some(BlendState::PREMULTIPLIED),
            write_mask: ColorWriteMask::ALL,
        },
    );
    let image = render(&device, |r| {
        r.bind_pipeline(pipeline);
        r.bind_vertex_buffers(0, &[positions], &[0]);
        r.draw(3, 1, 0, 0);
    });
    // One * (1, 0, 0, 0.5) + (1 - 0.5) * (0, 0, 1, 1) = (1, 0, 0.5, 1).
    let texel = image.pixel(SIZE / 2, SIZE / 2);
    assert!(
        approx_rgba(texel, [255, 0, 128, 255], 2),
        "{backend:?}: texel {texel:?}, expected ~[255, 0, 128, 255]"
    );
}

#[test]
fn premultiplied_blending_is_applied_vulkan() {
    premultiplied_blending_is_applied(Backend::Vulkan);
}

#[test]
fn premultiplied_blending_is_applied_opengl() {
    premultiplied_blending_is_applied(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// LoadOp::Clear must ignore write masks left by earlier pipelines (RGL-09)
// ---------------------------------------------------------------------------------------------

fn clear_ignores_previous_color_write_mask(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let positions = float_buffer(
        &device,
        &[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
        BufferUsage::VERTEX,
    );
    let red_only = pipeline(
        &device,
        Setup {
            vertex_shader: POSITIONS,
            layout: position_layout(),
            topology: PrimitiveTopology::TriangleList,
            blend: None,
            write_mask: ColorWriteMask::RED,
        },
    );
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    for clear in [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]] {
        recorder
            .begin_rendering(&color_pass(target, SIZE, SIZE, LoadOp::Clear, clear))
            .expect("begin_rendering");
        recorder.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        recorder.set_scissor(0, 0, SIZE, SIZE);
        recorder.bind_pipeline(red_only);
        recorder.bind_vertex_buffers(0, &[positions], &[0]);
        if clear[2] == 1.0 {
            // First pass only: leave the red-only write mask behind.
            recorder.draw(3, 1, 0, 0);
        }
        recorder.end_rendering().expect("end_rendering");
    }
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let image = Image::read(&*device, target, SIZE, SIZE);
    assert_eq!(
        image.count(|p| p == [0, 255, 0, 255]),
        (SIZE * SIZE) as usize,
        "{backend:?}: the second pass must clear every channel to green"
    );
}

#[test]
fn clear_ignores_previous_color_write_mask_vulkan() {
    clear_ignores_previous_color_write_mask(Backend::Vulkan);
}

#[test]
fn clear_ignores_previous_color_write_mask_opengl() {
    clear_ignores_previous_color_write_mask(Backend::OpenGl);
}

fn depth_clear_ignores_previous_depth_write_state(backend: Backend) {
    use korppi_rhi::{
        AccessFlags, CompareFunction, DepthAttachment, DepthStencilState, ImageLayout,
        ImageMemoryBarrier, PipelineStageFlags, SampleCount, StoreOp, TextureDescriptor,
        TextureDimension,
    };
    let device = TestDevice::open(backend);
    let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
    let depth = device
        .create_texture(
            &TextureDescriptor {
                width: SIZE,
                height: SIZE,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("create depth texture");
    let positions = float_buffer(
        &device,
        &[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
        BufferUsage::VERTEX,
    );
    let depth_pipeline = |write: bool| {
        let vertex_shader = device.shader(ShaderStage::Vertex, POSITIONS);
        let fragment_shader = device.shader(ShaderStage::Fragment, HALF_RED);
        device
            .create_graphics_pipeline(&GraphicsPipelineDescriptor {
                vertex_shader,
                fragment_shader: Some(fragment_shader),
                vertex_layout: position_layout(),
                topology: PrimitiveTopology::TriangleList,
                rasterizer: RasterizerState {
                    cull_mode: CullMode::None,
                    ..RasterizerState::default()
                },
                multisample: Default::default(),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_test_enabled: true,
                    depth_write_enabled: write,
                    depth_compare: CompareFunction::GreaterEqual,
                    ..Default::default()
                }),
                color_targets: vec![ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWriteMask::ALL,
                }],
                bind_group_layouts: vec![],
                push_constant_ranges: vec![],
                label: None,
            })
            .expect("create_graphics_pipeline")
    };
    let read_only = depth_pipeline(false);
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    let mut barrier = undefined_to_color_attachment(target);
    barrier.image_barriers.push(ImageMemoryBarrier {
        texture: depth,
        range: korppi_rhi::TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | PipelineStageFlags::LATE_FRAGMENT_TESTS,
        dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::DepthStencilAttachment,
    });
    recorder.pipeline_barrier(&barrier);
    // Pass 1 clears depth to 1.0 (nearest in reverse-Z) and binds a pipeline with depth writes
    // disabled. Pass 2 clears depth to 0.0 and draws at z = 0.5 with GREATER_EQUAL: it is only
    // visible if the second clear really reset the depth buffer.
    for (clear_depth, draw) in [(1.0, false), (0.0, true)] {
        let mut pass = color_pass(target, SIZE, SIZE, LoadOp::Clear, [0.0, 0.0, 1.0, 1.0]);
        pass.depth_attachment = Some(DepthAttachment {
            texture: depth,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_depth: Some(clear_depth),
            clear_stencil: None,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
        });
        recorder.begin_rendering(&pass).expect("begin_rendering");
        recorder.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        recorder.set_scissor(0, 0, SIZE, SIZE);
        recorder.bind_pipeline(read_only);
        recorder.bind_vertex_buffers(0, &[positions], &[0]);
        if draw {
            recorder.draw(3, 1, 0, 0);
        }
        recorder.end_rendering().expect("end_rendering");
    }
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let image = Image::read(&*device, target, SIZE, SIZE);
    assert_eq!(
        image.count(drawn),
        (SIZE * SIZE) as usize,
        "{backend:?}: the depth clear to 0.0 must not be masked by a read-only depth pipeline"
    );
}

#[test]
fn depth_clear_ignores_previous_depth_write_state_vulkan() {
    depth_clear_ignores_previous_depth_write_state(Backend::Vulkan);
}

#[test]
fn depth_clear_ignores_previous_depth_write_state_opengl() {
    depth_clear_ignores_previous_depth_write_state(Backend::OpenGl);
}

/// Draws a full-screen triangle at z = 0.5 into a depth buffer cleared to 0.6 with
/// `GreaterEqual`, through a viewport mapping depth to [`min_depth`, `max_depth`].
fn draw_with_depth_range(device: &TestDevice, min_depth: f32, max_depth: f32) -> Image {
    use korppi_rhi::{
        AccessFlags, CompareFunction, DepthAttachment, DepthStencilState, ImageLayout,
        ImageMemoryBarrier, PipelineStageFlags, SampleCount, StoreOp, TextureDescriptor,
        TextureDimension,
    };
    let target = color_target(&**device, SIZE, SIZE, TextureUsage::empty());
    let depth = device
        .create_texture(
            &TextureDescriptor {
                width: SIZE,
                height: SIZE,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("create depth texture");
    let positions = float_buffer(
        device,
        &[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0],
        BufferUsage::VERTEX,
    );
    let pipeline = device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: device.shader(ShaderStage::Vertex, POSITIONS),
            fragment_shader: Some(device.shader(ShaderStage::Fragment, HALF_RED)),
            vertex_layout: position_layout(),
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::None,
                ..RasterizerState::default()
            },
            multisample: Default::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_test_enabled: true,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                ..Default::default()
            }),
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create_graphics_pipeline");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    let mut barrier = undefined_to_color_attachment(target);
    barrier.image_barriers.push(ImageMemoryBarrier {
        texture: depth,
        range: korppi_rhi::TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | PipelineStageFlags::LATE_FRAGMENT_TESTS,
        dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::DepthStencilAttachment,
    });
    recorder.pipeline_barrier(&barrier);
    let mut pass = color_pass(target, SIZE, SIZE, LoadOp::Clear, [0.0, 0.0, 1.0, 1.0]);
    pass.depth_attachment = Some(DepthAttachment {
        texture: depth,
        mip_level: 0,
        layer: 0,
        load_op: LoadOp::Clear,
        store_op: StoreOp::Store,
        clear_depth: Some(0.6),
        clear_stencil: None,
        stencil_load_op: LoadOp::DontCare,
        stencil_store_op: StoreOp::DontCare,
    });
    recorder.begin_rendering(&pass).expect("begin_rendering");
    recorder.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, min_depth, max_depth);
    recorder.set_scissor(0, 0, SIZE, SIZE);
    recorder.bind_pipeline(pipeline);
    recorder.bind_vertex_buffers(0, &[positions], &[0]);
    recorder.draw(3, 1, 0, 0);
    recorder.end_rendering().expect("end_rendering");
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");
    Image::read(&**device, target, SIZE, SIZE)
}

/// The viewport's depth range maps NDC depth to [min, max] on both backends (RGL-15).
fn viewport_depth_range_is_applied(backend: Backend) {
    let device = TestDevice::open(backend);
    let full = (SIZE * SIZE) as usize;
    assert_eq!(
        draw_with_depth_range(&device, 0.4, 1.0).count(drawn),
        full,
        "{backend:?}: z = 0.5 in [0.4, 1.0] is 0.7, in front of the cleared 0.6"
    );
    assert_eq!(
        draw_with_depth_range(&device, 0.0, 1.0).count(drawn),
        0,
        "{backend:?}: z = 0.5 in [0, 1] is behind the cleared 0.6"
    );
    assert_eq!(
        draw_with_depth_range(&device, 0.8, 0.2).count(drawn),
        0,
        "{backend:?}: an inverted range maps z = 0.5 to 0.5"
    );
    device.assert_no_api_errors();
}

#[test]
fn viewport_depth_range_is_applied_vulkan() {
    viewport_depth_range_is_applied(Backend::Vulkan);
}

#[test]
fn viewport_depth_range_is_applied_opengl() {
    viewport_depth_range_is_applied(Backend::OpenGl);
}
