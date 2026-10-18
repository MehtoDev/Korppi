//! Fixed-function pipeline state (FIX-1.4; finding RHI-04): stencil test and reference, blend
//! constants, depth clamp, depth-bias clamp, and push-constant ranges shared by several stages.
//!
//! Every scenario renders into a 64x64 `Rgba8Unorm` target cleared to opaque blue; geometry
//! comes from `gl_VertexIndex`, so no vertex buffers are needed.

use korppi_rhi::{
    AccessFlags, BlendComponent, BlendFactor, BlendOp, BlendState, ColorTargetState,
    ColorWriteMask, CommandBufferRecorder, CompareFunction, CullMode, DepthAttachment,
    DepthStencilState, DeviceFeatures, FormatCapabilities, GraphicsPipelineDescriptor, ImageLayout,
    ImageMemoryBarrier, LoadOp, PipelineBarrier, PipelineHandle, PipelineStageFlags,
    PrimitiveTopology, PushConstantRange, RasterizerState, RenderingAttachment,
    RenderingDescriptor, SampleCount, ShaderStage, ShaderStages, StencilFaceState,
    StencilOperation, StoreOp, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle,
    TextureSubresourceRange, TextureUsage, VertexLayoutDescriptor,
};
use korppi_rhi_conformance::{approx_rgba, color_target, Backend, Image, TestDevice};

const SIZE: u32 = 64;

/// A rectangle from x0 to x1 (NDC) over the full height, at depth `z`, from push constants:
/// `vec4(x0, x1, z, unused)`.
const RECT: &str = r"#version 450
layout(push_constant) uniform Push { vec4 rect; } push;
void main() {
    vec2 corners[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 0), vec2(1, 1), vec2(0, 1));
    vec2 c = corners[gl_VertexIndex];
    float x = mix(push.rect.x, push.rect.y, c.x);
    gl_Position = vec4(x, c.y * 2.0 - 1.0, push.rect.z, 1.0);
}
";

/// Outputs the colour in push-constant bytes 16..32.
const PUSHED_COLOUR: &str = r"#version 450
layout(push_constant) uniform Push { layout(offset = 16) vec4 colour; } push;
layout(location = 0) out vec4 o;
void main() { o = push.colour; }
";

fn rect_push(x0: f32, x1: f32, z: f32, colour: [f32; 4]) -> Vec<u8> {
    [x0, x1, z, 0.0]
        .iter()
        .chain(colour.iter())
        .flat_map(|v| v.to_ne_bytes())
        .collect()
}

/// Push-constant ranges of `RECT` + `PUSHED_COLOUR`.
fn rect_ranges() -> Vec<PushConstantRange> {
    vec![
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            offset: 0,
            size: 16,
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            offset: 16,
            size: 16,
        },
    ]
}

struct Pipeline {
    blend: Option<BlendState>,
    write_mask: ColorWriteMask,
    depth_stencil: Option<DepthStencilState>,
    rasterizer: RasterizerState,
    push_constant_ranges: Vec<PushConstantRange>,
    vertex: &'static str,
    fragment: &'static str,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            blend: None,
            write_mask: ColorWriteMask::ALL,
            depth_stencil: None,
            rasterizer: RasterizerState {
                cull_mode: CullMode::None,
                ..RasterizerState::default()
            },
            push_constant_ranges: rect_ranges(),
            vertex: RECT,
            fragment: PUSHED_COLOUR,
        }
    }
}

fn pipeline(device: &TestDevice, setup: Pipeline) -> Result<PipelineHandle, korppi_rhi::RhiError> {
    let vertex_shader = device.shader(ShaderStage::Vertex, setup.vertex);
    let fragment_shader = device.shader(ShaderStage::Fragment, setup.fragment);
    device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
        vertex_shader,
        fragment_shader: Some(fragment_shader),
        vertex_layout: VertexLayoutDescriptor::new(),
        topology: PrimitiveTopology::TriangleList,
        rasterizer: setup.rasterizer,
        multisample: Default::default(),
        depth_stencil: setup.depth_stencil,
        color_targets: vec![ColorTargetState {
            format: TextureFormat::Rgba8Unorm,
            blend: setup.blend,
            write_mask: setup.write_mask,
        }],
        bind_group_layouts: vec![],
        push_constant_ranges: setup.push_constant_ranges,
        label: None,
    })
}

fn depth_texture(device: &TestDevice, format: TextureFormat) -> TextureHandle {
    device
        .create_texture(
            &TextureDescriptor {
                width: SIZE,
                height: SIZE,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format,
                // Depth24PlusStencil8 cannot be copied; Depth32Float is read back.
                usage: if format.is_stencil() {
                    TextureUsage::DEPTH_STENCIL_ATTACHMENT
                } else {
                    TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::COPY_SRC
                },
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("depth texture")
}

/// Renders into a blue-cleared colour target and, if given, a depth(/stencil) texture cleared
/// to depth 0 and stencil 0; returns the colour target.
fn render(
    device: &TestDevice,
    depth: Option<TextureHandle>,
    draw: impl FnOnce(&mut dyn CommandBufferRecorder),
) -> Image {
    let target = color_target(&**device, SIZE, SIZE, TextureUsage::empty());
    let mut barrier = PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
        texture: target,
        range: TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::ColorAttachment,
    });
    if let Some(depth) = depth {
        barrier = barrier.with_image_barrier(ImageMemoryBarrier {
            texture: depth,
            range: TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | PipelineStageFlags::LATE_FRAGMENT_TESTS,
            dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::DepthStencilAttachment,
        });
    }
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&barrier);
    recorder
        .begin_rendering(&RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width: SIZE,
            height: SIZE,
            color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                texture: target,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: Some([0.0, 0.0, 1.0, 1.0]),
                resolve_target: None,
            }],
            depth_attachment: depth.map(|texture| DepthAttachment {
                clear_stencil: Some(0),
                stencil_load_op: LoadOp::Clear,
                stencil_store_op: StoreOp::Store,
                ..DepthAttachment::cleared(texture, 0.0)
            }),
        })
        .expect("begin_rendering");
    recorder.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
    recorder.set_scissor(0, 0, SIZE, SIZE);
    draw(&mut *recorder);
    recorder.end_rendering().expect("end_rendering");
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");
    Image::read(&**device, target, SIZE, SIZE)
}

fn draw_rect(r: &mut dyn CommandBufferRecorder, x0: f32, x1: f32, z: f32, colour: [f32; 4]) {
    r.push_constants(0, &rect_push(x0, x1, z, colour))
        .expect("push_constants");
    r.draw(6, 1, 0, 0);
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE_U8: [u8; 4] = [0, 0, 255, 255];
const RED_U8: [u8; 4] = [255, 0, 0, 255];
const GREEN_U8: [u8; 4] = [0, 255, 0, 255];

fn stencil(compare: CompareFunction, pass_op: StencilOperation) -> StencilFaceState {
    StencilFaceState {
        compare,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op,
    }
}

#[test]
fn stencil_test_masks_later_draws_by_reference() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let format = TextureFormat::Depth24PlusStencil8;
        let stencil_only = |face: StencilFaceState| DepthStencilState {
            format,
            depth_test_enabled: false,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Always,
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 0xFF,
            stencil_write_mask: 0xFF,
        };
        // Writes the reference into the stencil buffer only.
        let mark = pipeline(
            &device,
            Pipeline {
                write_mask: ColorWriteMask::empty(),
                depth_stencil: Some(stencil_only(stencil(
                    CompareFunction::Always,
                    StencilOperation::Replace,
                ))),
                ..Pipeline::default()
            },
        )
        .expect("mark pipeline");
        // Draws where the stencil value equals the reference.
        let fill = pipeline(
            &device,
            Pipeline {
                depth_stencil: Some(stencil_only(stencil(
                    CompareFunction::Equal,
                    StencilOperation::Keep,
                ))),
                ..Pipeline::default()
            },
        )
        .expect("fill pipeline");
        let depth = depth_texture(&device, format);
        let image = render(&device, Some(depth), |r| {
            r.bind_pipeline(mark);
            r.set_stencil_reference(1);
            draw_rect(r, -1.0, 0.0, 0.5, RED); // left half: stencil 1
            r.bind_pipeline(fill);
            draw_rect(r, -1.0, 1.0, 0.5, RED); // only where stencil == 1
            r.set_stencil_reference(0);
            draw_rect(r, -1.0, 1.0, 0.5, GREEN); // only where stencil == 0
        });
        assert_eq!(image.pixel(8, 32), RED_U8, "{backend:?}: left half");
        assert_eq!(image.pixel(56, 32), GREEN_U8, "{backend:?}: right half");
    }
}

#[test]
fn blend_constants_feed_constant_blend_factors() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let constant = BlendComponent {
            src_factor: BlendFactor::Constant,
            dst_factor: BlendFactor::Zero,
            operation: BlendOp::Add,
        };
        let blended = pipeline(
            &device,
            Pipeline {
                blend: Some(BlendState {
                    color: constant,
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOp::Add,
                    },
                }),
                ..Pipeline::default()
            },
        )
        .expect("pipeline");
        let image = render(&device, None, |r| {
            r.bind_pipeline(blended);
            // The constant is zero until set.
            draw_rect(r, -1.0, 0.0, 0.5, [1.0; 4]);
            r.set_blend_constant([0.25, 0.5, 0.75, 1.0]);
            draw_rect(r, 0.0, 1.0, 0.5, [1.0; 4]);
        });
        assert_eq!(image.pixel(8, 32), [0, 0, 0, 255], "{backend:?}: default");
        assert!(
            approx_rgba(image.pixel(56, 32), [64, 128, 191, 255], 1),
            "{backend:?}: {:?} (left {:?}, top-right {:?})",
            image.pixel(56, 32),
            image.pixel(8, 32),
            image.pixel(60, 2)
        );
    }
}

fn read_depth(device: &TestDevice, texture: TextureHandle, x: u32, y: u32) -> f32 {
    let bytes = device.read_texture(texture, 0).expect("read depth");
    let i = ((y * SIZE + x) * 4) as usize;
    f32::from_ne_bytes(bytes[i..i + 4].try_into().unwrap())
}

fn depth_writer(
    device: &TestDevice,
    rasterizer: RasterizerState,
) -> Result<PipelineHandle, korppi_rhi::RhiError> {
    pipeline(
        device,
        Pipeline {
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_test_enabled: true,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                ..DepthStencilState::default()
            }),
            rasterizer,
            ..Pipeline::default()
        },
    )
}

#[test]
fn depth_clamp_keeps_geometry_beyond_the_far_plane() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let rasterizer = RasterizerState {
            cull_mode: CullMode::None,
            depth_clamp: true,
            ..RasterizerState::default()
        };
        if !device.features().contains(DeviceFeatures::DEPTH_CLAMP) {
            assert!(
                matches!(
                    depth_writer(&device, rasterizer.clone()),
                    Err(korppi_rhi::RhiError::Unsupported(_))
                ),
                "{backend:?}: depth clamp without the feature must be Unsupported"
            );
            continue;
        }
        let clamped = depth_writer(&device, rasterizer).expect("clamping pipeline");
        let clipped = depth_writer(
            &device,
            RasterizerState {
                cull_mode: CullMode::None,
                ..RasterizerState::default()
            },
        )
        .expect("clipping pipeline");
        let depth = depth_texture(&device, TextureFormat::Depth32Float);
        let image = render(&device, Some(depth), |r| {
            r.bind_pipeline(clipped);
            draw_rect(r, -1.0, 0.0, 1.5, RED); // beyond the far plane: clipped
            r.bind_pipeline(clamped);
            draw_rect(r, 0.0, 1.0, 1.5, GREEN); // clamped to depth 1
        });
        assert_eq!(image.pixel(8, 32), BLUE_U8, "{backend:?}: clipped half");
        assert_eq!(image.pixel(56, 32), GREEN_U8, "{backend:?}: clamped half");
        assert_eq!(read_depth(&device, depth, 56, 32), 1.0, "{backend:?}");
        assert_eq!(read_depth(&device, depth, 8, 32), 0.0, "{backend:?}");
    }
}

#[test]
fn depth_bias_is_limited_by_its_clamp() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let biased = |clamp| RasterizerState {
            cull_mode: CullMode::None,
            depth_bias: 1.0e8,
            depth_bias_clamp: clamp,
            ..RasterizerState::default()
        };
        if !device.features().contains(DeviceFeatures::DEPTH_BIAS_CLAMP) {
            assert!(
                matches!(
                    depth_writer(&device, biased(0.125)),
                    Err(korppi_rhi::RhiError::Unsupported(_))
                ),
                "{backend:?}: a bias clamp without the feature must be Unsupported"
            );
            continue;
        }
        let clamped = depth_writer(&device, biased(0.125)).expect("pipeline");
        let unclamped = depth_writer(&device, biased(0.0)).expect("pipeline");
        let depth = depth_texture(&device, TextureFormat::Depth32Float);
        render(&device, Some(depth), |r| {
            r.bind_pipeline(clamped);
            draw_rect(r, -1.0, 0.0, 0.5, RED);
            r.bind_pipeline(unclamped);
            draw_rect(r, 0.0, 1.0, 0.5, RED);
        });
        let left = read_depth(&device, depth, 8, 32);
        assert!(
            (left - 0.625).abs() < 1e-6,
            "{backend:?}: clamped bias gave {left}"
        );
        let right = read_depth(&device, depth, 56, 32);
        assert_eq!(
            right, 1.0,
            "{backend:?}: unclamped bias saturates at the far plane"
        );
    }
}

#[test]
fn push_constant_ranges_may_be_shared_by_stages() {
    // The fragment stage reads bytes 16..32; the vertex stage 0..16 and, through the shared
    // range, 16..32 too. On Vulkan a stage may appear in one range only, so the backend merges
    // the vertex stage's two ranges.
    const RECT_TINTED: &str = r"#version 450
layout(push_constant) uniform Push { vec4 rect; vec4 colour; } push;
layout(location = 0) out vec4 tint;
void main() {
    vec2 corners[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 0), vec2(1, 1), vec2(0, 1));
    vec2 c = corners[gl_VertexIndex];
    gl_Position = vec4(mix(push.rect.x, push.rect.y, c.x), c.y * 2.0 - 1.0, push.rect.z, 1.0);
    tint = push.colour;
}
";
    const TINT_TIMES_PUSHED: &str = r"#version 450
layout(push_constant) uniform Push { layout(offset = 16) vec4 colour; } push;
layout(location = 0) in vec4 tint;
layout(location = 0) out vec4 o;
void main() { o = tint * push.colour; }
";
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let shared = pipeline(
            &device,
            Pipeline {
                vertex: RECT_TINTED,
                fragment: TINT_TIMES_PUSHED,
                push_constant_ranges: vec![
                    PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 0,
                        size: 16,
                    },
                    PushConstantRange {
                        stages: ShaderStages::VERTEX_FRAGMENT,
                        offset: 16,
                        size: 16,
                    },
                ],
                ..Pipeline::default()
            },
        )
        .expect("pipeline");
        let image = render(&device, None, |r| {
            r.bind_pipeline(shared);
            draw_rect(r, -1.0, 0.0, 0.5, [0.0, 1.0, 0.0, 1.0]);
        });
        assert_eq!(image.pixel(8, 32), GREEN_U8, "{backend:?}");
        assert_eq!(image.pixel(56, 32), BLUE_U8, "{backend:?}");
    }
}

#[test]
fn stencil_state_needs_a_stencil_format() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        assert!(device
            .format_capabilities(TextureFormat::Depth24PlusStencil8)
            .contains(FormatCapabilities::DEPTH_STENCIL_ATTACHMENT));
        let result = pipeline(
            &device,
            Pipeline {
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    stencil_front: stencil(CompareFunction::Equal, StencilOperation::Keep),
                    ..DepthStencilState::default()
                }),
                ..Pipeline::default()
            },
        );
        assert!(
            matches!(result, Err(korppi_rhi::RhiError::ValidationError(_))),
            "{backend:?}: {result:?}"
        );
    }
}

/// Pipeline state persists across passes: a pass's clears do not disturb the write masks of a
/// pipeline bound in an earlier pass (OpenGL clears honour write masks, so the backend resets
/// and then restores them).
#[test]
fn bound_pipeline_state_survives_the_next_pass_clear() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let red_only = pipeline(
            &device,
            Pipeline {
                write_mask: ColorWriteMask::RED,
                ..Pipeline::default()
            },
        )
        .expect("pipeline");
        let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
        let pass = |load_op| RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width: SIZE,
            height: SIZE,
            color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                texture: target,
                mip_level: 0,
                layer: 0,
                load_op,
                store_op: StoreOp::Store,
                clear_color: Some([0.0, 0.0, 1.0, 1.0]),
                resolve_target: None,
            }],
            depth_attachment: None,
        };
        let mut r = device.create_command_recorder().expect("recorder");
        r.begin().expect("begin");
        r.pipeline_barrier(&korppi_rhi_conformance::undefined_to_color_attachment(
            target,
        ));
        r.begin_rendering(&pass(LoadOp::Clear)).expect("pass 1");
        r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        r.set_scissor(0, 0, SIZE, SIZE);
        r.bind_pipeline(red_only);
        r.end_rendering().expect("end pass 1");
        r.begin_rendering(&pass(LoadOp::Clear)).expect("pass 2");
        r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        r.set_scissor(0, 0, SIZE, SIZE);
        draw_rect(&mut *r, -1.0, 1.0, 0.5, [1.0, 1.0, 0.0, 0.5]);
        r.end_rendering().expect("end pass 2");
        device
            .submit_and_wait(r.finish().expect("finish"))
            .expect("submit");
        let image = Image::read(&*device, target, SIZE, SIZE);
        assert_eq!(
            image.pixel(32, 32),
            [255, 0, 255, 255],
            "{backend:?}: only red may be written over the blue clear"
        );
    }
}
