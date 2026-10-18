//! Vulkan pipelines share PSOs and layouts (FIX-1.5; finding RVK-16): pipelines that differ
//! only in state Vulkan 1.3 makes dynamic use one `VkPipeline`, set that state when bound, and
//! release it with the last of them.

use korppi_rhi::{
    BlendState, ColorTargetState, ColorWriteMask, CompareFunction, CullMode, DepthStencilState,
    FrontFace, GraphicsDevice, GraphicsPipelineDescriptor, LoadOp, PipelineHandle,
    PrimitiveTopology, RasterizerState, ShaderDescriptor, ShaderHandle, ShaderSource, ShaderStage,
    TextureFormat, TextureUsage, VertexLayoutDescriptor,
};
use korppi_rhi_conformance::{
    color_pass, color_target, compile_spirv, undefined_to_color_attachment, Image,
};
use korppi_rhi_vulkan::VulkanDevice;

/// A clockwise (back-facing) full-screen triangle.
const CLOCKWISE: &str = r"#version 450
void main() {
    vec2 p[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
    gl_Position = vec4(p[gl_VertexIndex], 0.5, 1.0);
}
";

const RED: &str = r"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.0, 0.0, 1.0); }
";

fn shader(device: &VulkanDevice, stage: ShaderStage, source: &str) -> ShaderHandle {
    let spirv = compile_spirv(source, stage);
    device
        .create_shader(&ShaderDescriptor {
            stage,
            entry_point: "main",
            source: ShaderSource::SpirvBytes(&spirv),
            label: None,
        })
        .expect("shader")
}

fn descriptor(
    vertex: ShaderHandle,
    fragment: ShaderHandle,
    cull_mode: CullMode,
    depth: Option<DepthStencilState>,
    blend: Option<BlendState>,
) -> GraphicsPipelineDescriptor {
    GraphicsPipelineDescriptor {
        vertex_shader: vertex,
        fragment_shader: Some(fragment),
        vertex_layout: VertexLayoutDescriptor::new(),
        topology: PrimitiveTopology::TriangleList,
        rasterizer: RasterizerState {
            cull_mode,
            front_face: FrontFace::CounterClockwise,
            ..RasterizerState::default()
        },
        multisample: Default::default(),
        depth_stencil: depth,
        color_targets: vec![ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend,
            write_mask: ColorWriteMask::ALL,
        }],
        bind_group_layouts: vec![],
        push_constant_ranges: vec![],
        label: None,
    }
}

fn draw(device: &VulkanDevice, pipeline: PipelineHandle) -> Image {
    let target = color_target(device, 16, 16, TextureUsage::empty());
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    r.pipeline_barrier(&undefined_to_color_attachment(target));
    r.begin_rendering(&color_pass(
        target,
        16,
        16,
        LoadOp::Clear,
        [0.0, 0.0, 1.0, 1.0],
    ))
    .expect("pass");
    r.set_viewport(0.0, 0.0, 16.0, 16.0, 0.0, 1.0);
    r.set_scissor(0, 0, 16, 16);
    r.bind_pipeline(pipeline);
    r.draw(3, 1, 0, 0);
    r.end_rendering().expect("end");
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");
    Image::read(device, target, 16, 16)
}

#[test]
fn pipelines_differing_in_dynamic_state_share_a_pso() {
    let device = VulkanDevice::create_headless().expect("device");
    let vertex = shader(&device, ShaderStage::Vertex, CLOCKWISE);
    let fragment = shader(&device, ShaderStage::Fragment, RED);

    let culled = device
        .create_graphics_pipeline(&descriptor(vertex, fragment, CullMode::Back, None, None))
        .expect("pipeline");
    let unculled = device
        .create_graphics_pipeline(&descriptor(vertex, fragment, CullMode::None, None, None))
        .expect("pipeline");
    let front_culled = device
        .create_graphics_pipeline(&descriptor(vertex, fragment, CullMode::Front, None, None))
        .expect("pipeline");
    assert_eq!(device.pipeline_object_count(), 1, "cull mode is dynamic");
    assert_eq!(device.pipeline_layout_count(), 1);

    // Each pipeline still draws with its own state.
    assert_eq!(
        draw(&device, culled).pixel(4, 4),
        [0, 0, 255, 255],
        "back faces culled"
    );
    assert_eq!(
        draw(&device, unculled).pixel(4, 4),
        [255, 0, 0, 255],
        "nothing culled"
    );
    assert_eq!(
        draw(&device, front_culled).pixel(4, 4),
        [255, 0, 0, 255],
        "front culled"
    );

    // Depth state is dynamic as well; the depth format is not, nor is blending.
    let depth = |compare| DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_compare: compare,
        ..DepthStencilState::default()
    };
    let less = device
        .create_graphics_pipeline(&descriptor(
            vertex,
            fragment,
            CullMode::None,
            Some(depth(CompareFunction::Less)),
            None,
        ))
        .expect("pipeline");
    let greater = device
        .create_graphics_pipeline(&descriptor(
            vertex,
            fragment,
            CullMode::None,
            Some(depth(CompareFunction::Greater)),
            None,
        ))
        .expect("pipeline");
    assert_eq!(
        device.pipeline_object_count(),
        2,
        "one more for the depth format"
    );
    let blended = device
        .create_graphics_pipeline(&descriptor(
            vertex,
            fragment,
            CullMode::None,
            None,
            Some(BlendState::PREMULTIPLIED),
        ))
        .expect("pipeline");
    assert_eq!(device.pipeline_object_count(), 3, "blending is baked in");

    // The shared objects live until their last pipeline is destroyed.
    for pipeline in [culled, unculled, less] {
        device.destroy_pipeline(pipeline).expect("destroy");
    }
    assert_eq!(device.pipeline_object_count(), 3);
    for pipeline in [front_culled, greater, blended] {
        device.destroy_pipeline(pipeline).expect("destroy");
    }
    device.wait_idle().expect("wait_idle");
    assert_eq!(device.pipeline_object_count(), 0);
    assert_eq!(device.pipeline_layout_count(), 0);
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == korppi_rhi::DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
}
