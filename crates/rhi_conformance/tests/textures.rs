//! The texture model (FIX-1.4; findings RHI-05, RVK-13, RGL-13, RGL-14): arrays, cube maps, 3D
//! textures, mip chains, per-mip/layer attachments, multisampling with resolve, and views.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferMemoryBarrier, BufferUsage, ColorTargetState,
    ColorWriteMask, ComputePipelineDescriptor, CullMode, DepthAttachment, FilterMode, FrontFace,
    GraphicsPipelineDescriptor, ImageLayout, ImageMemoryBarrier, LoadOp, MemoryUsage,
    MultisampleState, PipelineBarrier, PipelineStageFlags, PolygonMode, PrimitiveTopology,
    RasterizerState, RenderingAttachment, RenderingDescriptor, RhiError, SampleCount,
    SamplerDescriptor, ShaderStage, StoreOp, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureHandle, TextureUsage, TextureViewDescriptor, VertexLayoutDescriptor,
};
use korppi_rhi::{BindingResource, BindingType, ShaderStages, TextureBinding};
use korppi_rhi_conformance::{buffer_range, STORAGE};
use korppi_rhi_conformance::{Backend, TestDevice};

fn desc(
    dimension: TextureDimension,
    size: (u32, u32, u32),
    array_layers: u32,
    mip_levels: u32,
    format: TextureFormat,
    usage: TextureUsage,
) -> TextureDescriptor {
    TextureDescriptor {
        width: size.0,
        height: size.1,
        depth: size.2,
        array_layers,
        mip_levels,
        sample_count: SampleCount::Count1,
        format,
        usage,
        dimension,
        label: Some("texture_model_test"),
    }
}

fn to_attachment(texture: TextureHandle) -> PipelineBarrier {
    PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
        texture,
        range: korppi_rhi::TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::ColorAttachment,
    })
}

#[test]
fn inconsistent_texture_descriptors_are_rejected() {
    let usage = TextureUsage::SAMPLED;
    let bad = [
        desc(
            TextureDimension::D2,
            (8, 8, 1),
            2,
            1,
            TextureFormat::Rgba8Unorm,
            usage,
        ),
        desc(
            TextureDimension::Cube,
            (8, 8, 1),
            4,
            1,
            TextureFormat::Rgba8Unorm,
            usage,
        ),
        desc(
            TextureDimension::Cube,
            (8, 4, 1),
            6,
            1,
            TextureFormat::Rgba8Unorm,
            usage,
        ),
        desc(
            TextureDimension::CubeArray,
            (8, 8, 1),
            9,
            1,
            TextureFormat::Rgba8Unorm,
            usage,
        ),
        desc(
            TextureDimension::D3,
            (8, 8, 4),
            2,
            1,
            TextureFormat::Rgba8Unorm,
            usage,
        ),
        desc(
            TextureDimension::D2,
            (8, 8, 1),
            1,
            5,
            TextureFormat::Rgba8Unorm,
            usage,
        ),
        desc(
            TextureDimension::D2,
            (0, 8, 1),
            1,
            1,
            TextureFormat::Rgba8Unorm,
            usage,
        ),
        TextureDescriptor {
            sample_count: SampleCount::Count4,
            ..desc(
                TextureDimension::D2,
                (8, 8, 1),
                1,
                2,
                TextureFormat::Rgba8Unorm,
                usage,
            )
        },
    ];
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        for descriptor in &bad {
            let result = device.create_texture(descriptor, None);
            assert!(
                matches!(result, Err(RhiError::ValidationError(_))),
                "{backend:?}: {descriptor:?} gave {result:?}"
            );
        }
    }
}

/// Distinct colour of mip `mip`, layer `layer`.
fn colour(mip: u32, layer: u32) -> [u8; 4] {
    [(40 * mip + 10) as u8, (50 * layer + 5) as u8, 200, 255]
}

#[test]
fn every_mip_and_layer_of_an_array_can_be_rendered_and_read_back() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let (layers, mips) = (4, 3);
        let texture = device
            .create_texture(
                &desc(
                    TextureDimension::D2Array,
                    (16, 8, 1),
                    layers,
                    mips,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
                ),
                None,
            )
            .expect("array texture");
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.pipeline_barrier(&to_attachment(texture));
        for mip in 0..mips {
            for layer in 0..layers {
                let [r, g, b, a] = colour(mip, layer).map(|c| f32::from(c) / 255.0);
                recorder
                    .begin_rendering(&RenderingDescriptor {
                        offset_x: 0,
                        offset_y: 0,
                        width: 16 >> mip,
                        height: (8 >> mip).max(1),
                        color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                            texture,
                            mip_level: mip,
                            layer,
                            load_op: LoadOp::Clear,
                            store_op: StoreOp::Store,
                            clear_color: Some([r, g, b, a]),
                            resolve_target: None,
                        }],
                        depth_attachment: None,
                    })
                    .expect("begin_rendering");
                recorder.end_rendering().expect("end_rendering");
            }
        }
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");
        for mip in 0..mips {
            let texels = device.read_texture(texture, mip).expect("read_texture");
            let layer_bytes = ((16 >> mip) * (8u32 >> mip).max(1) * 4) as usize;
            assert_eq!(
                texels.len(),
                layer_bytes * layers as usize,
                "{backend:?} mip {mip}"
            );
            for layer in 0..layers {
                let start = layer as usize * layer_bytes;
                assert!(
                    texels[start..start + layer_bytes]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|t| *t == colour(mip, layer)),
                    "{backend:?}: mip {mip} layer {layer} starts with {:?}",
                    &texels[start..start + 4]
                );
            }
        }
    }
}

/// Samples the texture at binding 0 (`SAMPLER` declaration) at the `COORDS` of four probes into
/// the storage buffer at binding 1.
fn sampling_shader(sampler: &str, coords: [&str; 4]) -> String {
    format!(
        r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform {sampler} t;
layout(set = 0, binding = 1) buffer B {{ vec4 v[]; }};
void main() {{
    v[0] = {};
    v[1] = {};
    v[2] = {};
    v[3] = {};
}}
"#,
        coords[0], coords[1], coords[2], coords[3]
    )
}

/// Runs `shader` with `texture` (a texture or a view) at binding 0 and returns the four probes.
fn sample_probes(
    device: &TestDevice,
    texture: impl Into<TextureBinding>,
    shader: &str,
) -> [[f32; 4]; 4] {
    let sampler = device
        .create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..SamplerDescriptor::default()
        })
        .expect("sampler");
    let output = device
        .create_buffer(
            &BufferDescriptor {
                size: 64,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            None,
        )
        .expect("output");
    let (layout, group) = device.bind_group(&[
        (
            0,
            BindingType::SampledTexture,
            ShaderStages::COMPUTE,
            BindingResource::SampledTexture {
                texture: texture.into(),
                sampler,
                layout: ImageLayout::ShaderReadOnly,
            },
        ),
        (
            1,
            STORAGE,
            ShaderStages::COMPUTE,
            buffer_range(output, 0, 64),
        ),
    ]);
    let shader = device.shader(ShaderStage::Compute, shader);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("compute pipeline");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.bind_pipeline(pipeline);
    recorder.set_bind_group(0, group, &[]);
    recorder.dispatch(1, 1, 1);
    recorder.pipeline_barrier(
        &PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
            buffer: output,
            src_stages: PipelineStageFlags::COMPUTE_SHADER,
            src_access: AccessFlags::SHADER_WRITE,
            dst_stages: PipelineStageFlags::HOST,
            dst_access: AccessFlags::HOST_READ,
            offset: 0,
            size: 64,
        }),
    );
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");
    let bytes = device.read_buffer(output, 0, 64).expect("read_buffer");
    let floats: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    [0, 1, 2, 3].map(|i| {
        [
            floats[4 * i],
            floats[4 * i + 1],
            floats[4 * i + 2],
            floats[4 * i + 3],
        ]
    })
}

fn rgba(texel: [u8; 4]) -> [f32; 4] {
    texel.map(|c| f32::from(c) / 255.0)
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.01)
}

#[test]
fn cube_faces_are_sampled_by_direction() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        // Faces +X, -X, +Y, -Y, +Z, -Z, each one solid colour.
        let faces: Vec<[u8; 4]> = (0..6).map(|face| colour(0, face)).collect();
        let data: Vec<u8> = faces.iter().flat_map(|c| c.repeat(4 * 4)).collect();
        let texture = device
            .create_texture(
                &desc(
                    TextureDimension::Cube,
                    (4, 4, 1),
                    6,
                    1,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED,
                ),
                Some(&data),
            )
            .expect("cube texture");
        let probes = sample_probes(
            &device,
            texture,
            &sampling_shader(
                "samplerCube",
                [
                    "texture(t, vec3(1.0, 0.0, 0.0))",
                    "texture(t, vec3(0.0, -1.0, 0.0))",
                    "texture(t, vec3(0.0, 0.0, 1.0))",
                    "texture(t, vec3(0.0, 0.0, -1.0))",
                ],
            ),
        );
        for (probe, face) in probes.iter().zip([0, 3, 4, 5]) {
            assert!(
                close(*probe, rgba(faces[face])),
                "{backend:?}: face {face} sampled as {probe:?}"
            );
        }
    }
}

/// Views select what sampling sees (RHI-05, RHI-07): one layer of an array as a 2D texture,
/// a layer range as a smaller array, and six layers of an array as a cube.
#[test]
fn sampled_views_see_their_layers() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let layers: Vec<[u8; 4]> = (0..8).map(|layer| colour(0, layer)).collect();
        let data: Vec<u8> = layers.iter().flat_map(|c| c.repeat(4 * 4)).collect();
        let texture = device
            .create_texture(
                &desc(
                    TextureDimension::D2Array,
                    (4, 4, 1),
                    8,
                    1,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED,
                ),
                Some(&data),
            )
            .expect("array texture");
        let view = |dimension, base_array_layer, array_layer_count| {
            device
                .create_texture_view(&TextureViewDescriptor {
                    texture,
                    dimension,
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer,
                    array_layer_count,
                    aspect: TextureAspect::All,
                    label: None,
                })
                .expect("view")
        };
        let centre = "texture(t, vec2(0.5))";
        let layer3 = sample_probes(
            &device,
            view(TextureDimension::D2, 3, 1),
            &sampling_shader("sampler2D", [centre; 4]),
        );
        assert!(
            close(layer3[0], rgba(layers[3])),
            "{backend:?}: layer 3 {:?}",
            layer3[0]
        );

        let range = sample_probes(
            &device,
            view(TextureDimension::D2Array, 4, 3),
            &sampling_shader(
                "sampler2DArray",
                [
                    "texture(t, vec3(0.5, 0.5, 0.0))",
                    "texture(t, vec3(0.5, 0.5, 1.0))",
                    "texture(t, vec3(0.5, 0.5, 2.0))",
                    "texture(t, vec3(0.5, 0.5, 2.0))",
                ],
            ),
        );
        for (probe, layer) in range.iter().zip([4, 5, 6]) {
            assert!(
                close(*probe, rgba(layers[layer])),
                "{backend:?}: view layer {} is texture layer {layer}, sampled as {probe:?}",
                layer - 4
            );
        }

        let cube = sample_probes(
            &device,
            view(TextureDimension::Cube, 2, 6),
            &sampling_shader(
                "samplerCube",
                [
                    "texture(t, vec3(1.0, 0.0, 0.0))",
                    "texture(t, vec3(-1.0, 0.0, 0.0))",
                    "texture(t, vec3(0.0, 0.0, 1.0))",
                    "texture(t, vec3(0.0, 0.0, -1.0))",
                ],
            ),
        );
        for (probe, face) in cube.iter().zip([0, 1, 4, 5]) {
            assert!(
                close(*probe, rgba(layers[2 + face])),
                "{backend:?}: cube face {face} (layer {}) sampled as {probe:?}",
                2 + face
            );
        }
    }
}

#[test]
fn volume_slices_are_sampled_by_depth() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let slices: Vec<[u8; 4]> = (0..4).map(|slice| colour(1, slice)).collect();
        let data: Vec<u8> = slices.iter().flat_map(|c| c.repeat(2 * 2)).collect();
        let texture = device
            .create_texture(
                &desc(
                    TextureDimension::D3,
                    (2, 2, 4),
                    1,
                    1,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED,
                ),
                Some(&data),
            )
            .expect("3D texture");
        let probes = sample_probes(
            &device,
            texture,
            &sampling_shader(
                "sampler3D",
                [
                    "texture(t, vec3(0.5, 0.5, 0.125))",
                    "texture(t, vec3(0.5, 0.5, 0.375))",
                    "texture(t, vec3(0.5, 0.5, 0.625))",
                    "texture(t, vec3(0.5, 0.5, 0.875))",
                ],
            ),
        );
        for (slice, probe) in probes.iter().enumerate() {
            assert!(
                close(*probe, rgba(slices[slice])),
                "{backend:?}: slice {slice} sampled as {probe:?}"
            );
        }
    }
}

#[test]
fn a_full_mip_chain_uploads_and_samples_per_level() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        // 8x8, 4x4, 2x2, 1x1, each level one colour.
        let data: Vec<u8> = (0..4u32)
            .flat_map(|mip| {
                let side = 8 >> mip;
                colour(mip, 0).repeat((side * side) as usize)
            })
            .collect();
        let texture = device
            .create_texture(
                &desc(
                    TextureDimension::D2,
                    (8, 8, 1),
                    1,
                    4,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED | TextureUsage::COPY_SRC,
                ),
                Some(&data),
            )
            .expect("mipmapped texture");
        let probes = sample_probes(
            &device,
            texture,
            &sampling_shader(
                "sampler2D",
                [
                    "textureLod(t, vec2(0.5), 0.0)",
                    "textureLod(t, vec2(0.5), 1.0)",
                    "textureLod(t, vec2(0.5), 2.0)",
                    "textureLod(t, vec2(0.5), 3.0)",
                ],
            ),
        );
        for (mip, probe) in probes.iter().enumerate() {
            assert!(
                close(*probe, rgba(colour(mip as u32, 0))),
                "{backend:?}: mip {mip} sampled as {probe:?}"
            );
        }
        assert_eq!(
            device.read_texture(texture, 2).expect("read_texture"),
            colour(2, 0).repeat(4),
            "{backend:?}: mip 2 reads back"
        );
    }
}

const TRIANGLE: &str = r#"#version 450
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0));
void main() { gl_Position = vec4(P[gl_VertexIndex], 0.5, 1.0); }
"#;
const WHITE: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0); }
"#;

#[test]
fn multisampled_rendering_resolves_partial_coverage() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let size = 16;
        let mut msaa_desc = desc(
            TextureDimension::D2,
            (size, size, 1),
            1,
            1,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COLOR_ATTACHMENT,
        );
        msaa_desc.sample_count = SampleCount::Count4;
        let msaa = device
            .create_texture(&msaa_desc, None)
            .expect("MSAA target");
        let resolved = device
            .create_texture(
                &desc(
                    TextureDimension::D2,
                    (size, size, 1),
                    1,
                    1,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
                ),
                None,
            )
            .expect("resolve target");
        let vertex_shader = device.shader(ShaderStage::Vertex, TRIANGLE);
        let fragment_shader = device.shader(ShaderStage::Fragment, WHITE);
        let pipeline = device
            .create_graphics_pipeline(&GraphicsPipelineDescriptor {
                vertex_shader,
                fragment_shader: Some(fragment_shader),
                vertex_layout: VertexLayoutDescriptor::new(),
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
                multisample: MultisampleState {
                    count: SampleCount::Count4,
                    alpha_to_coverage: false,
                },
                depth_stencil: None,
                color_targets: vec![ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWriteMask::ALL,
                }],
                bind_group_layouts: vec![],
                push_constant_ranges: vec![],
                label: None,
            })
            .expect("MSAA pipeline");
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.pipeline_barrier(&to_attachment(msaa));
        recorder.pipeline_barrier(&to_attachment(resolved));
        recorder
            .begin_rendering(&RenderingDescriptor {
                offset_x: 0,
                offset_y: 0,
                width: size,
                height: size,
                color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                    texture: msaa,
                    mip_level: 0,
                    layer: 0,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::DontCare,
                    clear_color: Some([0.0, 0.0, 0.0, 1.0]),
                    resolve_target: Some(resolved),
                }],
                depth_attachment: None,
            })
            .expect("begin_rendering");
        recorder.set_viewport(0.0, 0.0, size as f32, size as f32, 0.0, 1.0);
        recorder.set_scissor(0, 0, size, size);
        recorder.bind_pipeline(pipeline);
        recorder.draw(3, 1, 0, 0);
        recorder.end_rendering().expect("end_rendering");
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");
        let texels = device.read_texture(resolved, 0).expect("read_texture");
        let reds: Vec<u8> = texels.as_chunks::<4>().0.iter().map(|t| t[0]).collect();
        let covered = reds.iter().filter(|&&r| r == 255).count();
        let partial = reds.iter().filter(|&&r| r > 0 && r < 255).count();
        // The diagonal edge cuts through pixels: 4x MSAA resolves them to intermediate values.
        assert!(covered > 50, "{backend:?}: {covered} fully covered pixels");
        assert!(
            partial >= size as usize / 2,
            "{backend:?}: {partial} edge pixels resolved"
        );
    }
}

#[test]
fn depth_renders_into_an_array_layer() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let depth = device
            .create_texture(
                &desc(
                    TextureDimension::D2Array,
                    (4, 4, 1),
                    3,
                    1,
                    TextureFormat::Depth32Float,
                    TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::COPY_SRC,
                ),
                None,
            )
            .expect("depth array");
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.pipeline_barrier(
            &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
                texture: depth,
                range: korppi_rhi::TextureSubresourceRange::ALL,
                src_stages: PipelineStageFlags::TOP_OF_PIPE,
                src_access: AccessFlags::empty(),
                dst_stages: PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                old_layout: ImageLayout::Undefined,
                new_layout: ImageLayout::DepthStencilAttachment,
            }),
        );
        for (layer, value) in [(0, 0.25), (1, 0.5), (2, 0.75)] {
            recorder
                .begin_rendering(&RenderingDescriptor {
                    offset_x: 0,
                    offset_y: 0,
                    width: 4,
                    height: 4,
                    color_attachments: korppi_rhi::smallvec![],
                    depth_attachment: Some(DepthAttachment {
                        layer,
                        ..DepthAttachment::cleared(depth, value)
                    }),
                })
                .expect("begin_rendering");
            recorder.end_rendering().expect("end_rendering");
        }
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");
        let bytes = device.read_texture(depth, 0).expect("read_texture");
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(values.len(), 48, "{backend:?}");
        for (layer, expected) in [0.25f32, 0.5, 0.75].iter().enumerate() {
            assert!(
                values[layer * 16..(layer + 1) * 16]
                    .iter()
                    .all(|v| (v - expected).abs() < 1e-6),
                "{backend:?}: layer {layer} holds {:?}",
                &values[layer * 16..layer * 16 + 2]
            );
        }
    }
}

#[test]
fn texture_views_check_their_range_and_die_with_their_texture() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let texture = device
            .create_texture(
                &desc(
                    TextureDimension::D2Array,
                    (8, 8, 1),
                    6,
                    2,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::SAMPLED,
                ),
                None,
            )
            .expect("texture");
        let view = |dimension, base_mip_level, base_array_layer, array_layer_count| {
            device.create_texture_view(&TextureViewDescriptor {
                texture,
                dimension,
                base_mip_level,
                mip_level_count: 1,
                base_array_layer,
                array_layer_count,
                aspect: TextureAspect::All,
                label: None,
            })
        };
        let single = view(TextureDimension::D2, 1, 3, 1).expect("single-layer view");
        let cube = view(TextureDimension::Cube, 0, 0, 6).expect("cube view of 6 layers");
        assert!(matches!(
            view(TextureDimension::D2, 2, 0, 1),
            Err(RhiError::ValidationError(_))
        ));
        assert!(matches!(
            view(TextureDimension::D2Array, 0, 4, 3),
            Err(RhiError::ValidationError(_))
        ));
        assert!(matches!(
            view(TextureDimension::Cube, 0, 0, 5),
            Err(RhiError::ValidationError(_))
        ));
        assert!(matches!(
            view(TextureDimension::D3, 0, 0, 1),
            Err(RhiError::ValidationError(_))
        ));
        device.destroy_texture_view(single).expect("destroy view");
        assert!(matches!(
            device.destroy_texture_view(single),
            Err(RhiError::InvalidHandle(_))
        ));
        device.destroy_texture(texture).expect("destroy texture");
        assert!(
            matches!(
                device.destroy_texture_view(cube),
                Err(RhiError::InvalidHandle(_))
            ),
            "{backend:?}: destroying a texture destroys its views"
        );
    }
}
