//! Vertex attribute formats are decoded identically by both backends (FIX-1.4; finding RHI-13).
//!
//! Each format feeds one attribute value, the same for all three vertices of a full-screen
//! triangle, through a flat varying into an `Rgba32Float` target; the target shows the decoded
//! value (missing components default to (0, 0, 0, 1)).

use korppi_rhi::{
    BufferDescriptor, BufferUsage, ColorTargetState, ColorWriteMask, CullMode, FrontFace,
    GraphicsPipelineDescriptor, LoadOp, MemoryUsage, PolygonMode, PrimitiveTopology,
    RasterizerState, SampleCount, ShaderStage, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsage, VertexAttribute, VertexBinding, VertexFormat, VertexLayoutDescriptor,
    VertexStepMode,
};
use korppi_rhi_conformance::{color_pass, undefined_to_color_attachment, Backend, TestDevice};

/// How the shader declares the attribute.
#[derive(Copy, Clone)]
enum Input {
    Float,
    Uint,
    Sint,
}

fn vertex_shader(input: Input) -> String {
    let ty = match input {
        Input::Float => "vec4",
        Input::Uint => "uvec4",
        Input::Sint => "ivec4",
    };
    format!(
        r#"#version 450
layout(location = 0) in vec2 pos;
layout(location = 1) in {ty} data;
layout(location = 0) flat out vec4 value;
void main() {{ gl_Position = vec4(pos, 0.5, 1.0); value = vec4(data); }}
"#
    )
}

const FRAGMENT: &str = r#"#version 450
layout(location = 0) flat in vec4 value;
layout(location = 0) out vec4 o;
void main() { o = value; }
"#;

fn halves(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn cases() -> Vec<(VertexFormat, Input, Vec<u8>, [f32; 4])> {
    let u16s = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let i16s = |v: &[i16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let i32s = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    vec![
        (
            VertexFormat::Float32,
            Input::Float,
            2.5f32.to_le_bytes().to_vec(),
            [2.5, 0.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Float16x2,
            Input::Float,
            halves(&[0x3800, 0xC000]),
            [0.5, -2.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Float16x4,
            Input::Float,
            halves(&[0x3C00, 0x3800, 0x0000, 0xC000]),
            [1.0, 0.5, 0.0, -2.0],
        ),
        (
            VertexFormat::Uint8x4Norm,
            Input::Float,
            vec![255, 128, 0, 64],
            [1.0, 128.0 / 255.0, 0.0, 64.0 / 255.0],
        ),
        (
            VertexFormat::Sint8x4Norm,
            Input::Float,
            vec![127, 0x81, 0, 64],
            [1.0, -1.0, 0.0, 64.0 / 127.0],
        ),
        (
            VertexFormat::Uint16x2Norm,
            Input::Float,
            u16s(&[65535, 32768]),
            [1.0, 32768.0 / 65535.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Uint16x4Norm,
            Input::Float,
            u16s(&[0, 65535, 16384, 49152]),
            [0.0, 1.0, 16384.0 / 65535.0, 49152.0 / 65535.0],
        ),
        (
            VertexFormat::Sint16x2Norm,
            Input::Float,
            i16s(&[32767, -32767]),
            [1.0, -1.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Sint16x4Norm,
            Input::Float,
            i16s(&[16384, -16384, 0, 32767]),
            [16384.0 / 32767.0, -16384.0 / 32767.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Uint8x4,
            Input::Uint,
            vec![1, 2, 3, 250],
            [1.0, 2.0, 3.0, 250.0],
        ),
        (
            VertexFormat::Uint16x2,
            Input::Uint,
            u16s(&[7, 65535]),
            [7.0, 65535.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Uint16x4,
            Input::Uint,
            u16s(&[1, 2, 3, 4]),
            [1.0, 2.0, 3.0, 4.0],
        ),
        (
            VertexFormat::Uint32x3,
            Input::Uint,
            u32s(&[9, 8, 7]),
            [9.0, 8.0, 7.0, 1.0],
        ),
        (
            VertexFormat::Sint32,
            Input::Sint,
            i32s(&[-5]),
            [-5.0, 0.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Sint32x2,
            Input::Sint,
            i32s(&[-1, 100]),
            [-1.0, 100.0, 0.0, 1.0],
        ),
        (
            VertexFormat::Sint32x4,
            Input::Sint,
            i32s(&[1, -2, 3, -4]),
            [1.0, -2.0, 3.0, -4.0],
        ),
    ]
}

/// Draws with `format` holding `value` and returns the decoded texel.
fn decode(device: &TestDevice, format: VertexFormat, input: Input, value: &[u8]) -> [f32; 4] {
    assert_eq!(
        value.len() as u64,
        format.size(),
        "{format:?} test value size"
    );
    let target = device
        .create_texture(
            &TextureDescriptor {
                width: 4,
                height: 4,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("target");
    let vertex_shader = device.shader(ShaderStage::Vertex, &vertex_shader(input));
    let fragment_shader = device.shader(ShaderStage::Fragment, FRAGMENT);
    let pipeline = device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout: VertexLayoutDescriptor::new()
                .with_binding(VertexBinding {
                    binding: 0,
                    stride: 8,
                    step_mode: VertexStepMode::Vertex,
                })
                .with_binding(VertexBinding {
                    binding: 1,
                    stride: format.size(),
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
                    binding: 1,
                    format,
                    offset: 0,
                }),
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
                format: TextureFormat::Rgba32Float,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: None,
        })
        .unwrap_or_else(|e| panic!("{format:?}: pipeline: {e}"));
    let positions: Vec<u8> = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let buffer = |data: &[u8]| {
        device
            .create_buffer(
                &BufferDescriptor {
                    size: data.len() as u64,
                    usage: BufferUsage::VERTEX,
                    memory: MemoryUsage::CpuToGpu,
                    label: None,
                },
                Some(data),
            )
            .expect("vertex buffer")
    };
    let position_buffer = buffer(&positions);
    let attribute_buffer = buffer(&value.repeat(3));

    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    recorder
        .begin_rendering(&color_pass(target, 4, 4, LoadOp::Clear, [9.0; 4]))
        .expect("begin_rendering");
    recorder.set_viewport(0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
    recorder.set_scissor(0, 0, 4, 4);
    recorder.bind_pipeline(pipeline);
    recorder.bind_vertex_buffers(0, &[position_buffer, attribute_buffer], &[0, 0]);
    recorder.draw(3, 1, 0, 0);
    recorder.end_rendering().expect("end_rendering");
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");
    let texels = device.read_texture(target, 0).expect("read_texture");
    let mut decoded = [0.0f32; 4];
    for (value, chunk) in decoded.iter_mut().zip(texels.as_chunks::<4>().0) {
        *value = f32::from_le_bytes(*chunk);
    }
    for buffer in [position_buffer, attribute_buffer] {
        device.destroy_buffer(buffer).expect("destroy buffer");
    }
    device.destroy_pipeline(pipeline).expect("destroy pipeline");
    device.destroy_texture(target).expect("destroy target");
    decoded
}

#[test]
fn vertex_formats_decode_identically_on_both_backends() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        for (format, input, value, expected) in cases() {
            let decoded = decode(&device, format, input, &value);
            assert!(
                decoded
                    .iter()
                    .zip(expected)
                    .all(|(d, e)| (d - e).abs() < 1e-4),
                "{backend:?} {format:?}: decoded {decoded:?}, expected {expected:?}"
            );
        }
    }
}
