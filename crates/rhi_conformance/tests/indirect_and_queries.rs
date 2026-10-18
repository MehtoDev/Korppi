//! Indirect draws and dispatches, timestamp queries, debug groups and object labels
//! (FIX-1.4; audit missing-functionality matrix).

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferHandle, BufferMemoryBarrier, BufferUsage,
    ColorTargetState, ColorWriteMask, CommandBufferRecorder, ComputePipelineDescriptor, CullMode,
    DeviceFeatures, DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs,
    GraphicsPipelineDescriptor, IndexType, LoadOp, MemoryUsage, PipelineBarrier, PipelineHandle,
    PipelineStageFlags, PrimitiveTopology, QuerySetDescriptor, QueryType, RasterizerState,
    RhiError, ShaderStage, ShaderStages, TextureFormat, TextureUsage, VertexLayoutDescriptor,
};
use korppi_rhi_conformance::{
    buffer_range, color_pass, color_target, undefined_to_color_attachment, Backend, Image,
    TestDevice, STORAGE,
};

const SIZE: u32 = 64;

/// Vertices 0..6: the left half; 6..12: the right half (from `gl_VertexIndex`, which includes
/// the first vertex and base vertex on both backends).
const HALVES: &str = r"#version 450
void main() {
    vec2 corners[6] = vec2[](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 0), vec2(1, 1), vec2(0, 1));
    vec2 c = corners[gl_VertexIndex % 6];
    float x0 = gl_VertexIndex < 6 ? -1.0 : 0.0;
    gl_Position = vec4(x0 + c.x, c.y * 2.0 - 1.0, 0.5, 1.0);
}
";

const RED: &str = r"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.0, 0.0, 1.0); }
";

fn buffer(
    device: &TestDevice,
    usage: BufferUsage,
    data: &[u8],
    label: &'static str,
) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size: data.len() as u64,
                usage,
                memory: MemoryUsage::CpuToGpu,
                label: Some(label),
            },
            Some(data),
        )
        .expect("create_buffer")
}

fn red_halves(device: &TestDevice) -> PipelineHandle {
    device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: device.shader(ShaderStage::Vertex, HALVES),
            fragment_shader: Some(device.shader(ShaderStage::Fragment, RED)),
            vertex_layout: VertexLayoutDescriptor::new(),
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::None,
                ..RasterizerState::default()
            },
            multisample: Default::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: Some("red_halves"),
        })
        .expect("pipeline")
}

fn render(device: &TestDevice, draw: impl FnOnce(&mut dyn CommandBufferRecorder)) -> Image {
    let target = color_target(&**device, SIZE, SIZE, TextureUsage::empty());
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    r.pipeline_barrier(&undefined_to_color_attachment(target));
    r.begin_rendering(&color_pass(
        target,
        SIZE,
        SIZE,
        LoadOp::Clear,
        [0.0, 0.0, 1.0, 1.0],
    ))
    .expect("begin_rendering");
    r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
    r.set_scissor(0, 0, SIZE, SIZE);
    draw(&mut *r);
    r.end_rendering().expect("end_rendering");
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");
    Image::read(&**device, target, SIZE, SIZE)
}

const RED_U8: [u8; 4] = [255, 0, 0, 255];
const BLUE_U8: [u8; 4] = [0, 0, 255, 255];

#[test]
fn indirect_draws_read_their_arguments_from_buffers() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let pipeline = red_halves(&device);
        let records = [
            DrawIndirectArgs {
                vertex_count: 6,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            DrawIndirectArgs {
                vertex_count: 6,
                instance_count: 1,
                first_vertex: 6,
                first_instance: 0,
            },
        ];
        let args: Vec<u8> = records.iter().flat_map(|r| r.to_bytes()).collect();
        let args = buffer(&device, BufferUsage::INDIRECT, &args, "draw_args");
        let multi = device
            .features()
            .contains(DeviceFeatures::MULTI_DRAW_INDIRECT);
        let image = render(&device, |r| {
            r.bind_pipeline(pipeline);
            if multi {
                r.draw_indirect(args, 0, 2, DrawIndirectArgs::SIZE as u32);
            } else {
                r.draw_indirect(args, 0, 1, 0);
                r.draw_indirect(args, DrawIndirectArgs::SIZE, 1, 0);
            }
        });
        assert_eq!(image.pixel(8, 32), RED_U8, "{backend:?}: left");
        assert_eq!(image.pixel(56, 32), RED_U8, "{backend:?}: right");

        // Indexed: indices 0..6 shifted to the right half by the base vertex.
        let indices: Vec<u8> = (0u16..6).flat_map(|i| i.to_ne_bytes()).collect();
        let indices = buffer(&device, BufferUsage::INDEX, &indices, "indices");
        let indexed = DrawIndexedIndirectArgs {
            index_count: 6,
            instance_count: 1,
            first_index: 0,
            base_vertex: 6,
            first_instance: 0,
        };
        let indexed_args = buffer(
            &device,
            BufferUsage::INDIRECT,
            &indexed.to_bytes(),
            "indexed_args",
        );
        let image = render(&device, |r| {
            r.bind_pipeline(pipeline);
            r.bind_index_buffer(indices, 0, IndexType::Uint16);
            r.draw_indexed_indirect(indexed_args, 0, 1, 0);
        });
        assert_eq!(image.pixel(8, 32), BLUE_U8, "{backend:?}: indexed left");
        assert_eq!(image.pixel(56, 32), RED_U8, "{backend:?}: indexed right");
    }
}

fn compute(
    device: &TestDevice,
    source: &str,
    output: BufferHandle,
    size: u64,
) -> (PipelineHandle, korppi_rhi::BindGroupHandle) {
    let (layout, group) = device.bind_group(&[(
        0,
        STORAGE,
        ShaderStages::COMPUTE,
        buffer_range(output, 0, size),
    )]);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: device.shader(ShaderStage::Compute, source),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("pipeline");
    (pipeline, group)
}

#[test]
fn dispatches_read_arguments_the_gpu_wrote() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let args = buffer(
            &device,
            BufferUsage::STORAGE | BufferUsage::INDIRECT,
            &DispatchIndirectArgs::default().to_bytes(),
            "dispatch_args",
        );
        let output = buffer(&device, BufferUsage::STORAGE, &[0; 16], "dispatch_output");
        let (write_args, args_group) = compute(
            &device,
            r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer Args { uint x; uint y; uint z; };
void main() { x = 3u; y = 1u; z = 1u; }
",
            args,
            12,
        );
        let (count_groups, output_group) = compute(
            &device,
            r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer Out { uint ids[4]; };
void main() { ids[gl_WorkGroupID.x] = gl_WorkGroupID.x + 1u; }
",
            output,
            16,
        );
        let mut r = device.create_command_recorder().expect("recorder");
        r.begin().expect("begin");
        r.push_debug_group("gpu-driven dispatch");
        r.bind_pipeline(write_args);
        r.set_bind_group(0, args_group, &[]);
        r.dispatch(1, 1, 1);
        r.pipeline_barrier(
            &PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
                buffer: args,
                src_stages: PipelineStageFlags::COMPUTE_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_stages: PipelineStageFlags::DRAW_INDIRECT,
                dst_access: AccessFlags::INDIRECT_COMMAND_READ,
                offset: 0,
                size: 12,
            }),
        );
        r.bind_pipeline(count_groups);
        r.set_bind_group(0, output_group, &[]);
        r.dispatch_indirect(args, 0);
        r.pipeline_barrier(
            &PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
                buffer: output,
                src_stages: PipelineStageFlags::COMPUTE_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_stages: PipelineStageFlags::HOST,
                dst_access: AccessFlags::HOST_READ,
                offset: 0,
                size: 16,
            }),
        );
        r.pop_debug_group();
        device
            .submit_and_wait(r.finish().expect("finish"))
            .expect("submit");
        let ids: Vec<u32> = device
            .read_buffer(output, 0, 16)
            .expect("read")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_ne_bytes(*c))
            .collect();
        assert_eq!(ids, [1, 2, 3, 0], "{backend:?}");
    }
}

#[test]
fn timestamps_bracket_gpu_work() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let desc = QuerySetDescriptor {
            ty: QueryType::Timestamp,
            count: 2,
            label: Some("frame_timestamps"),
        };
        if !device.features().contains(DeviceFeatures::TIMESTAMP_QUERY) {
            assert!(matches!(
                device.create_query_set(&desc),
                Err(RhiError::Unsupported(_))
            ));
            continue;
        }
        let queries = device.create_query_set(&desc).expect("query set");
        let results = device
            .create_buffer(
                &BufferDescriptor {
                    size: 16,
                    usage: BufferUsage::COPY_DST,
                    memory: MemoryUsage::GpuToCpu,
                    label: Some("timestamp_results"),
                },
                None,
            )
            .expect("buffer");
        let pipeline = red_halves(&device);
        let period = f64::from(device.limits().timestamp_period_ns);
        for frame in 0..2 {
            // Queries are written again after each resolve.
            render(&device, |r| {
                r.write_timestamp(queries, 0);
                r.bind_pipeline(pipeline);
                r.draw(12, 1, 0, 0);
                r.write_timestamp(queries, 1);
            });
            let mut r = device.create_command_recorder().expect("recorder");
            r.begin().expect("begin");
            r.resolve_query_set(queries, 0, 2, results, 0);
            r.pipeline_barrier(
                &PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
                    buffer: results,
                    src_stages: PipelineStageFlags::TRANSFER,
                    src_access: AccessFlags::TRANSFER_WRITE,
                    dst_stages: PipelineStageFlags::HOST,
                    dst_access: AccessFlags::HOST_READ,
                    offset: 0,
                    size: 16,
                }),
            );
            device
                .submit_and_wait(r.finish().expect("finish"))
                .expect("submit");
            let bytes = device.read_buffer(results, 0, 16).expect("read");
            let start = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
            let end = u64::from_ne_bytes(bytes[8..].try_into().unwrap());
            assert!(
                start > 0 && end >= start,
                "{backend:?} frame {frame}: {start} {end}"
            );
            let elapsed_ms = (end - start) as f64 * period / 1.0e6;
            assert!(
                elapsed_ms < 10_000.0,
                "{backend:?} frame {frame}: {elapsed_ms} ms between timestamps"
            );
        }
        device.destroy_query_set(queries).expect("destroy");
    }
}

#[test]
fn debug_groups_must_balance() {
    if !korppi_rhi::api_validation_requested() {
        return;
    }
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let pipeline = red_halves(&device);
        // Balanced groups and markers, inside and outside a pass, record without errors.
        render(&device, |r| {
            r.push_debug_group("opaque");
            r.insert_debug_marker("draw halves");
            r.bind_pipeline(pipeline);
            r.draw(12, 1, 0, 0);
            r.pop_debug_group();
        });
        let mut r = device.create_command_recorder().expect("recorder");
        r.begin().expect("begin");
        r.push_debug_group("never closed");
        match r.finish() {
            Err(RhiError::ValidationError(message)) => {
                assert!(message.contains("debug group"), "{backend:?}: {message}")
            }
            other => panic!("{backend:?}: unbalanced groups accepted: {:?}", other.err()),
        }
        let indices = buffer(&device, BufferUsage::INDEX, &[0; 12], "indices");
        let args = buffer(
            &device,
            BufferUsage::INDIRECT,
            &DrawIndexedIndirectArgs::default().to_bytes(),
            "args",
        );
        // Indexed indirect draws need the index buffer at offset 0 on every backend.
        let target = color_target(&*device, SIZE, SIZE, TextureUsage::empty());
        let mut r = device.create_command_recorder().expect("recorder");
        r.begin().expect("begin");
        r.pipeline_barrier(&undefined_to_color_attachment(target));
        r.begin_rendering(&color_pass(target, SIZE, SIZE, LoadOp::Clear, [0.0; 4]))
            .expect("pass");
        r.set_viewport(0.0, 0.0, SIZE as f32, SIZE as f32, 0.0, 1.0);
        r.set_scissor(0, 0, SIZE, SIZE);
        r.bind_pipeline(pipeline);
        r.bind_index_buffer(indices, 2, IndexType::Uint16);
        r.draw_indexed_indirect(args, 0, 1, 0);
        r.end_rendering().expect("end");
        match r.finish() {
            Err(RhiError::ValidationError(message)) => {
                assert!(message.contains("offset 0"), "{backend:?}: {message}")
            }
            other => panic!(
                "{backend:?}: offset index buffer accepted: {:?}",
                other.err()
            ),
        }
    }
}
