//! OpenGL state cache (FIX-1.5; finding RGL-15):
//! - code that uses the context directly through `OpenGlDevice::with_gl` may leave any state
//!   behind; the device forgets what it cached and renders correctly afterwards;
//! - a resource created after another was destroyed (and possibly given its GL name) is bound
//!   even though the name matches what the cache last bound.

use korppi_rhi::{
    BufferDescriptor, BufferHandle, BufferUsage, ColorTargetState, ColorWriteMask, CullMode,
    GraphicsPipelineDescriptor, IndexType, LoadOp, MemoryUsage, PipelineHandle, PrimitiveTopology,
    RasterizerState, ShaderStage, ShaderStages, TextureFormat, TextureHandle, TextureUsage,
    VertexAttribute, VertexBinding, VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
use korppi_rhi_conformance::{
    buffer_range, color_pass, color_target, undefined_to_color_attachment, Backend, Image,
    TestDevice, UNIFORM,
};
use korppi_rhi_opengl::glow::{self, HasContext};

const SIZE: u32 = 16;
const GREEN: [u8; 4] = [0, 255, 0, 255];

const POSITIONS: &str = r"#version 450
layout(location = 0) in vec2 pos;
void main() { gl_Position = vec4(pos, 0.5, 1.0); }
";

const UNIFORM_COLOR: &str = r"#version 450
layout(set = 0, binding = 0) uniform Color { vec4 color; };
layout(location = 0) out vec4 o;
void main() { o = color; }
";

struct Scene {
    pipeline: PipelineHandle,
    layout: korppi_rhi::BindGroupLayoutHandle,
    color: BufferHandle,
    group: korppi_rhi::BindGroupHandle,
    vertices: BufferHandle,
    indices: BufferHandle,
    target: TextureHandle,
}

fn buffer(device: &TestDevice, usage: BufferUsage, data: &[u8]) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size: data.len() as u64,
                usage,
                memory: MemoryUsage::GpuOnly,
                label: None,
            },
            Some(data),
        )
        .expect("buffer")
}

fn scene(device: &TestDevice) -> Scene {
    let color = color_buffer(device, [0.0, 1.0, 0.0, 1.0]);
    let (layout, group) = device.bind_group(&[(
        0,
        UNIFORM,
        ShaderStages::FRAGMENT,
        buffer_range(color, 0, 16),
    )]);
    // A full-screen triangle, drawn indexed.
    let vertices = buffer(
        device,
        BufferUsage::VERTEX,
        bytemuck_floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]).as_slice(),
    );
    let indices = buffer(
        device,
        BufferUsage::INDEX,
        &[0u16, 1, 2]
            .iter()
            .flat_map(|i| i.to_ne_bytes())
            .collect::<Vec<_>>(),
    );
    let pipeline = device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: device.shader(ShaderStage::Vertex, POSITIONS),
            fragment_shader: Some(device.shader(ShaderStage::Fragment, UNIFORM_COLOR)),
            vertex_layout: VertexLayoutDescriptor {
                bindings: vec![VertexBinding {
                    binding: 0,
                    stride: 8,
                    step_mode: VertexStepMode::Vertex,
                }],
                attributes: vec![VertexAttribute {
                    location: 0,
                    binding: 0,
                    format: VertexFormat::Float32x2,
                    offset: 0,
                }],
            },
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::Back,
                ..RasterizerState::default()
            },
            multisample: Default::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("pipeline");
    Scene {
        pipeline,
        layout,
        color,
        group,
        vertices,
        indices,
        target: color_target(&**device, SIZE, SIZE, TextureUsage::empty()),
    }
}

fn bytemuck_floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn render(device: &TestDevice, scene: &Scene) -> Image {
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    r.pipeline_barrier(&undefined_to_color_attachment(scene.target));
    r.begin_rendering(&color_pass(
        scene.target,
        SIZE,
        SIZE,
        LoadOp::Clear,
        [0.0, 0.0, 1.0, 1.0],
    ))
    .expect("pass");
    r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
    r.set_scissor(0, 0, SIZE, SIZE);
    r.bind_pipeline(scene.pipeline);
    r.set_bind_group(0, scene.group, &[]);
    r.bind_vertex_buffers(0, &[scene.vertices], &[0]);
    r.bind_index_buffer(scene.indices, 0, IndexType::Uint16);
    r.draw_indexed(3, 1, 0, 0, 0);
    r.end_rendering().expect("end");
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");
    Image::read(&**device, scene.target, SIZE, SIZE)
}

#[test]
fn state_left_behind_by_with_gl_does_not_leak_into_rendering() {
    let device = TestDevice::open(Backend::OpenGl);
    let scene = scene(&device);
    let full = (SIZE * SIZE) as usize;
    assert_eq!(render(&device, &scene).count(|p| p == GREEN), full);

    let gl_device = device.opengl().expect("OpenGL device");
    gl_device
        .with_gl(|gl| {
            // SAFETY: `with_gl` runs `f` with the context current; only bindings and context
            // state change, no object is deleted.
            unsafe {
                // Detach the buffers of the VAO the last draw used.
                let vao = gl.get_parameter_i32(glow::VERTEX_ARRAY_BINDING);
                if let Some(name) = std::num::NonZeroU32::new(vao as u32) {
                    let vao = glow::NativeVertexArray(name);
                    gl.vertex_array_element_buffer(vao, None);
                    gl.vertex_array_vertex_buffer(vao, 0, None, 0, 0);
                }
                gl.bind_vertex_array(None);
                gl.use_program(None);
                gl.bind_buffer_base(glow::UNIFORM_BUFFER, 0, None);
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl.viewport(0, 0, 1, 1);
                gl.enable(glow::SCISSOR_TEST);
                gl.scissor(0, 0, 1, 1);
                gl.color_mask(false, false, false, false);
                gl.enable(glow::BLEND);
                gl.blend_func(glow::ZERO, glow::ZERO);
                gl.cull_face(glow::FRONT_AND_BACK);
                gl.polygon_mode(glow::FRONT_AND_BACK, glow::POINT);
                gl.enable(glow::RASTERIZER_DISCARD);
            }
        })
        .expect("with_gl");

    let image = render(&device, &scene);
    assert_eq!(
        image.count(|p| p == GREEN),
        full,
        "rendering after with_gl must not see its state; pixel (8, 8) is {:?}",
        image.pixel(8, 8)
    );
}

fn color_buffer(device: &TestDevice, color: [f32; 4]) -> BufferHandle {
    buffer(device, BufferUsage::UNIFORM, &bytemuck_floats(&color))
}

/// Destroys the scene's colour buffer and binds a new one in a new group.
fn replace_color(device: &TestDevice, scene: &mut Scene, color: [f32; 4]) {
    device
        .destroy_bind_group(scene.group)
        .expect("destroy group");
    device.destroy_buffer(scene.color).expect("destroy buffer");
    scene.color = color_buffer(device, color);
    scene.group = device.group(scene.layout, &[(0, buffer_range(scene.color, 0, 16))]);
}

fn buffers_replacing_destroyed_ones_are_bound(backend: Backend) {
    let device = TestDevice::open(backend);
    let mut scene = scene(&device);
    let full = (SIZE * SIZE) as usize;
    assert_eq!(render(&device, &scene).count(|p| p == GREEN), full);
    for (color, expected) in [
        ([1.0, 0.0, 0.0, 1.0], [255, 0, 0, 255]),
        ([0.0, 1.0, 0.0, 1.0], GREEN),
    ] {
        replace_color(&device, &mut scene, color);
        assert_eq!(
            render(&device, &scene).count(|p| p == expected),
            full,
            "{backend:?}: the new buffer's {expected:?}"
        );
    }
    // Index and vertex buffers too (VAO attachments).
    device
        .destroy_buffer(scene.indices)
        .expect("destroy indices");
    scene.indices = buffer(
        &device,
        BufferUsage::INDEX,
        &[0u16, 1, 2]
            .iter()
            .flat_map(|i| i.to_ne_bytes())
            .collect::<Vec<_>>(),
    );
    device
        .destroy_buffer(scene.vertices)
        .expect("destroy vertices");
    scene.vertices = buffer(
        &device,
        BufferUsage::VERTEX,
        &bytemuck_floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]),
    );
    assert_eq!(render(&device, &scene).count(|p| p == GREEN), full);
}

#[test]
fn buffers_replacing_destroyed_ones_are_bound_vulkan() {
    buffers_replacing_destroyed_ones_are_bound(Backend::Vulkan);
}

#[test]
fn buffers_replacing_destroyed_ones_are_bound_opengl() {
    buffers_replacing_destroyed_ones_are_bound(Backend::OpenGl);
}
