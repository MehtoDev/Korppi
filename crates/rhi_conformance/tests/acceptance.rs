//! Milestone 7 acceptance scene: the "colored triangle" test from `research/rhi/04` §4.3 that was
//! never written, extended to the state a real frame uses.
//!
//! One pass renders, into a 64x64 colour target with a reverse-Z `Depth32Float` buffer:
//! 1. an opaque green quad in the centre (z = 0.25);
//! 2. a full-screen **clockwise** white quad (z = 0.9) that back-face culling must remove;
//! 3. an opaque yellow quad in the bottom-right quadrant **behind** the green one (z = 0.1),
//!    which the depth test must reject where they overlap;
//! 4. a half-transparent red quad in the top-left quadrant (z = 0.5), alpha-blended.
//!
//! All quads come from one vertex buffer and one `u16` index buffer, drawn with
//! `draw_indexed(.., vertex_offset, ..)`. Every texel away from region edges is compared with
//! the analytically expected colour under the RHI contract (row 0 = top).

use korppi_rhi::{
    AccessFlags, BlendState, BufferDescriptor, BufferUsage, ColorTargetState, ColorWriteMask,
    CompareFunction, CullMode, DepthAttachment, DepthStencilState, FrontFace,
    GraphicsPipelineDescriptor, ImageLayout, ImageMemoryBarrier, IndexType, LoadOp, MemoryUsage,
    PipelineStageFlags, PolygonMode, PrimitiveTopology, RasterizerState, SampleCount, ShaderStage,
    StoreOp, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage, VertexAttribute,
    VertexBinding, VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
use korppi_rhi_conformance::{
    approx_rgba, color_pass, color_target, to_rgba8, undefined_to_color_attachment, Backend, Image,
    TestDevice,
};

const SIZE: u32 = 64;
const CLEAR: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
const TRANSLUCENT_RED: [f32; 4] = [1.0, 0.0, 0.0, 0.5];

const COLORED_VERTEX: &str = r#"#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 col;
layout(location = 0) out vec4 v_col;
void main() { v_col = col; gl_Position = vec4(pos, 1.0); }
"#;

const PASS_THROUGH_COLOR: &str = r#"#version 450
layout(location = 0) in vec4 v_col;
layout(location = 0) out vec4 o;
void main() { o = v_col; }
"#;

/// Axis-aligned quad in NDC.
struct Quad {
    min: [f32; 2],
    max: [f32; 2],
    z: f32,
    color: [f32; 4],
    clockwise: bool,
}

const QUADS: [Quad; 4] = [
    Quad {
        min: [-0.5, -0.5],
        max: [0.5, 0.5],
        z: 0.25,
        color: GREEN,
        clockwise: false,
    },
    Quad {
        min: [-1.0, -1.0],
        max: [1.0, 1.0],
        z: 0.9,
        color: WHITE,
        clockwise: true,
    },
    Quad {
        min: [0.0, -1.0],
        max: [1.0, 0.0],
        z: 0.1,
        color: YELLOW,
        clockwise: false,
    },
    Quad {
        min: [-1.0, 0.0],
        max: [0.0, 1.0],
        z: 0.5,
        color: TRANSLUCENT_RED,
        clockwise: false,
    },
];

/// Vertex bytes: position (3 x f32) + colour (4 x f32) per vertex, 4 vertices per quad.
fn vertex_bytes() -> Vec<u8> {
    let mut floats = Vec::new();
    for quad in &QUADS {
        let [x0, y0] = quad.min;
        let [x1, y1] = quad.max;
        // Counter-clockwise in +Y-up NDC: bottom-left, bottom-right, top-right, top-left.
        let ccw = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        let corners = if quad.clockwise {
            [ccw[0], ccw[3], ccw[2], ccw[1]]
        } else {
            ccw
        };
        for [x, y] in corners {
            floats.extend_from_slice(&[x, y, quad.z]);
            floats.extend_from_slice(&quad.color);
        }
    }
    floats.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

fn pipeline(
    device: &TestDevice,
    depth_write: bool,
    blend: Option<BlendState>,
) -> korppi_rhi::PipelineHandle {
    let vertex_shader = device.shader(ShaderStage::Vertex, COLORED_VERTEX);
    let fragment_shader = device.shader(ShaderStage::Fragment, PASS_THROUGH_COLOR);
    device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout: VertexLayoutDescriptor::new()
                .with_binding(VertexBinding {
                    binding: 0,
                    stride: 28,
                    step_mode: VertexStepMode::Vertex,
                })
                .with_attribute(VertexAttribute {
                    location: 0,
                    binding: 0,
                    format: VertexFormat::Float32x3,
                    offset: 0,
                })
                .with_attribute(VertexAttribute {
                    location: 1,
                    binding: 0,
                    format: VertexFormat::Float32x4,
                    offset: 12,
                }),
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                polygon_mode: PolygonMode::Fill,
                depth_bias: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_bias_clamp: 0.0,
                depth_clamp: false,
            },
            multisample: Default::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_test_enabled: true,
                depth_write_enabled: depth_write,
                depth_compare: CompareFunction::GreaterEqual,
                ..Default::default()
            }),
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: Some("acceptance"),
        })
        .expect("create_graphics_pipeline")
}

fn render_scene(backend: Backend) -> Image {
    let device = TestDevice::open(backend);
    let color = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
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
                label: Some("acceptance_depth"),
            },
            None,
        )
        .expect("create depth texture");

    let vertices = vertex_bytes();
    let vertex_buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: vertices.len() as u64,
                usage: BufferUsage::VERTEX,
                memory: MemoryUsage::CpuToGpu,
                label: Some("acceptance_vertices"),
            },
            Some(&vertices),
        )
        .expect("create vertex buffer");
    let indices: Vec<u8> = [0_u16, 1, 2, 0, 2, 3]
        .iter()
        .flat_map(|i| i.to_ne_bytes())
        .collect();
    let index_buffer = device
        .create_buffer(
            &BufferDescriptor {
                size: indices.len() as u64,
                usage: BufferUsage::INDEX,
                memory: MemoryUsage::CpuToGpu,
                label: Some("acceptance_indices"),
            },
            Some(&indices),
        )
        .expect("create index buffer");

    let opaque = pipeline(&device, true, None);
    let translucent = pipeline(&device, false, Some(BlendState::ALPHA_BLENDING));

    let mut pass = color_pass(color, SIZE, SIZE, LoadOp::Clear, CLEAR);
    pass.depth_attachment = Some(DepthAttachment {
        texture: depth,
        mip_level: 0,
        layer: 0,
        load_op: LoadOp::Clear,
        store_op: StoreOp::DontCare,
        clear_depth: Some(0.0),
        clear_stencil: None,
        stencil_load_op: LoadOp::DontCare,
        stencil_store_op: StoreOp::DontCare,
    });

    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    let mut barrier = undefined_to_color_attachment(color);
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
    recorder.begin_rendering(&pass).expect("begin_rendering");
    recorder.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
    recorder.set_scissor(0, 0, SIZE, SIZE);
    recorder.bind_vertex_buffers(0, &[vertex_buffer], &[0]);
    recorder.bind_index_buffer(index_buffer, 0, IndexType::Uint16);
    recorder.bind_pipeline(opaque);
    for quad in 0..3 {
        recorder.draw_indexed(6, 1, 0, 4 * quad, 0);
    }
    recorder.bind_pipeline(translucent);
    recorder.draw_indexed(6, 1, 0, 12, 0);
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    Image::read(&*device, color, SIZE, SIZE)
}

/// Expected colour at the centre of texel (`x`, `y`), or `None` within 1.5 texels of a region
/// edge where rasterization rules may legitimately differ.
fn expected_texel(x: u32, y: u32) -> Option<[u8; 4]> {
    let half = SIZE as f32 / 2.0;
    let edges = [0.25 * SIZE as f32, half, 0.75 * SIZE as f32];
    let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
    if edges
        .iter()
        .any(|e| (cx - e).abs() < 1.5 || (cy - e).abs() < 1.5)
    {
        return None;
    }
    // Row 0 is the top of the image, so NDC y decreases as the row index grows.
    let (nx, ny) = (cx / half - 1.0, 1.0 - cy / half);
    let in_green = nx.abs() < 0.5 && ny.abs() < 0.5;
    let in_yellow = nx > 0.0 && ny < 0.0;
    let in_red = nx < 0.0 && ny > 0.0;
    let base = if in_green {
        GREEN
    } else if in_yellow {
        YELLOW
    } else {
        CLEAR
    };
    let color = if in_red {
        let a = TRANSLUCENT_RED[3];
        [
            a * TRANSLUCENT_RED[0] + (1.0 - a) * base[0],
            a * TRANSLUCENT_RED[1] + (1.0 - a) * base[1],
            a * TRANSLUCENT_RED[2] + (1.0 - a) * base[2],
            a + base[3] * (1.0 - a),
        ]
    } else {
        base
    };
    Some(to_rgba8(color))
}

fn acceptance_scene_matches_expected_image(backend: Backend) {
    let image = render_scene(backend);
    let mut mismatches = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            if let Some(expected) = expected_texel(x, y) {
                let actual = image.pixel(x, y);
                if !approx_rgba(actual, expected, 2) {
                    mismatches.push((x, y, actual, expected));
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{backend:?}: {} texels differ from the expected image; first: {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(4)]
    );
}

#[test]
fn acceptance_scene_matches_expected_image_vulkan() {
    acceptance_scene_matches_expected_image(Backend::Vulkan);
}

#[test]
fn acceptance_scene_matches_expected_image_opengl() {
    acceptance_scene_matches_expected_image(Backend::OpenGl);
}
