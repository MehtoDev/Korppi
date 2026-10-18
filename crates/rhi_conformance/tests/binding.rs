//! Explicit resource binding (FIX-1.4; findings RHI-07, RVK-09): bind group layouts referenced
//! by pipelines, bind groups set per descriptor set, dynamic offsets, storage textures, several
//! sets, and groups staying bound across pipelines with compatible layouts.

use korppi_rhi::{
    AccessFlags, BindGroupDescriptor, BindGroupEntry, BindGroupHandle, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutHandle, BindingResource, BindingType, BufferDescriptor,
    BufferHandle, BufferMemoryBarrier, BufferUsage, ComputePipelineDescriptor, ImageLayout,
    ImageMemoryBarrier, MemoryUsage, PipelineBarrier, PipelineHandle, PipelineStageFlags, RhiError,
    SampleCount, SamplerDescriptor, ShaderStage, ShaderStages, StorageTextureAccess, TextureAspect,
    TextureBinding, TextureDescriptor, TextureDimension, TextureFormat, TextureSubresourceRange,
    TextureUsage, TextureViewDescriptor,
};
use korppi_rhi_conformance::{buffer_range, sampled, Backend, TestDevice, STORAGE, UNIFORM};

const COMPUTE: ShaderStages = ShaderStages::COMPUTE;

fn vec4_bytes(values: [f32; 4]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn read_vec4s(
    device: &TestDevice,
    buffer: BufferHandle,
    offset: u64,
    count: usize,
) -> Vec<[f32; 4]> {
    let bytes = device
        .read_buffer(buffer, offset, 16 * count as u64)
        .expect("read_buffer");
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|chunk| {
            let mut out = [0.0; 4];
            for (value, c) in out.iter_mut().zip(chunk.as_chunks::<4>().0) {
                *value = f32::from_ne_bytes(*c);
            }
            out
        })
        .collect()
}

fn buffer(device: &TestDevice, size: u64, usage: BufferUsage, data: Option<&[u8]>) -> BufferHandle {
    let memory = if usage.contains(BufferUsage::UNIFORM) {
        MemoryUsage::CpuToGpu
    } else {
        MemoryUsage::GpuToCpu
    };
    device
        .create_buffer(
            &BufferDescriptor {
                size,
                usage,
                memory,
                label: Some("binding_test"),
            },
            data,
        )
        .expect("create_buffer")
}

fn output_buffer(device: &TestDevice, size: u64) -> BufferHandle {
    buffer(
        device,
        size,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        None,
    )
}

fn compute_pipeline(
    device: &TestDevice,
    source: &str,
    layouts: &[BindGroupLayoutHandle],
) -> PipelineHandle {
    let compute_shader = device.shader(ShaderStage::Compute, source);
    device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader,
            bind_group_layouts: layouts.to_vec(),
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create_compute_pipeline")
}

fn compute_to_compute(buffer: BufferHandle, size: u64) -> PipelineBarrier {
    PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
        buffer,
        src_stages: PipelineStageFlags::COMPUTE_SHADER,
        src_access: AccessFlags::SHADER_WRITE,
        dst_stages: PipelineStageFlags::COMPUTE_SHADER,
        dst_access: AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
        offset: 0,
        size,
    })
}

fn compute_to_host(buffer: BufferHandle, size: u64) -> PipelineBarrier {
    PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
        buffer,
        src_stages: PipelineStageFlags::COMPUTE_SHADER,
        src_access: AccessFlags::SHADER_WRITE,
        dst_stages: PipelineStageFlags::HOST,
        dst_access: AccessFlags::HOST_READ,
        offset: 0,
        size,
    })
}

const DOUBLE_UNIFORM: &str = r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform Params { vec4 value; };
layout(set = 0, binding = 1) buffer Out { vec4 result; };
void main() { result = value * 2.0; }
";

#[test]
fn dynamic_offsets_select_the_data_of_each_dispatch() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let mut params = vec4_bytes([1.0, 2.0, 3.0, 4.0]);
        params.resize(256, 0);
        params.extend(vec4_bytes([5.0, 6.0, 7.0, 8.0]));
        let uniforms = buffer(&device, 512, BufferUsage::UNIFORM, Some(&params));
        let output = output_buffer(&device, 512);
        let (layout, group) = device.bind_group(&[
            (
                0,
                BindingType::UniformBuffer {
                    dynamic_offset: true,
                },
                COMPUTE,
                buffer_range(uniforms, 0, 16),
            ),
            (
                1,
                BindingType::StorageBuffer {
                    read_only: false,
                    dynamic_offset: true,
                },
                COMPUTE,
                buffer_range(output, 0, 16),
            ),
        ]);
        let pipeline = compute_pipeline(&device, DOUBLE_UNIFORM, &[layout]);

        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.bind_pipeline(pipeline);
        recorder.set_bind_group(0, group, &[0, 0]);
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&compute_to_compute(output, 512));
        recorder.set_bind_group(0, group, &[256, 256]);
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&compute_to_host(output, 512));
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");

        assert_eq!(
            read_vec4s(&device, output, 0, 1),
            [[2.0, 4.0, 6.0, 8.0]],
            "{backend:?}: offset 0"
        );
        assert_eq!(
            read_vec4s(&device, output, 256, 1),
            [[10.0, 12.0, 14.0, 16.0]],
            "{backend:?}: offset 256"
        );
    }
}

const SEVERAL_SETS: &str = r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 1) uniform A { vec4 a; };
layout(set = 1, binding = 0) uniform B { vec4 b; };
layout(set = 1, binding = 2) uniform sampler2D t;
layout(set = 2, binding = 0) buffer Out { vec4 result[2]; };
layout(set = 2, binding = 3) readonly buffer In { vec4 c; };
void main() {
    result[0] = a + b * 10.0;
    result[1] = textureLod(t, vec2(0.5), 0.0) + c;
}
";

#[test]
fn bindings_in_several_sets_reach_their_declarations() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let a = buffer(
            &device,
            16,
            BufferUsage::UNIFORM,
            Some(&vec4_bytes([1.0, 2.0, 3.0, 4.0])),
        );
        let b = buffer(
            &device,
            16,
            BufferUsage::UNIFORM,
            Some(&vec4_bytes([0.5, 0.0, 0.0, 0.0])),
        );
        let c = buffer(
            &device,
            16,
            BufferUsage::STORAGE,
            Some(&vec4_bytes([0.0, 1.0, 0.0, 0.0])),
        );
        let output = output_buffer(&device, 32);
        let red = device
            .create_texture(
                &TextureDescriptor {
                    width: 1,
                    height: 1,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                    dimension: TextureDimension::D2,
                    label: None,
                },
                Some(&[255, 0, 0, 255]),
            )
            .expect("texture");
        let sampler = device
            .create_sampler(&SamplerDescriptor::default())
            .expect("sampler");

        let (set0, group0) = device.bind_group(&[(1, UNIFORM, COMPUTE, buffer_range(a, 0, 16))]);
        let (set1, group1) = device.bind_group(&[
            (
                2,
                BindingType::SampledTexture,
                COMPUTE,
                sampled(red, sampler),
            ),
            (0, UNIFORM, COMPUTE, buffer_range(b, 0, 16)),
        ]);
        let (set2, group2) = device.bind_group(&[
            (0, STORAGE, COMPUTE, buffer_range(output, 0, 32)),
            (
                3,
                BindingType::StorageBuffer {
                    read_only: true,
                    dynamic_offset: false,
                },
                COMPUTE,
                buffer_range(c, 0, 16),
            ),
        ]);
        let pipeline = compute_pipeline(&device, SEVERAL_SETS, &[set0, set1, set2]);

        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        // Groups may be set before the pipeline.
        recorder.set_bind_group(2, group2, &[]);
        recorder.set_bind_group(0, group0, &[]);
        recorder.bind_pipeline(pipeline);
        recorder.set_bind_group(1, group1, &[]);
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&compute_to_host(output, 32));
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");

        assert_eq!(
            read_vec4s(&device, output, 0, 2),
            [[6.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 1.0]],
            "{backend:?}"
        );
    }
}

#[test]
fn groups_stay_bound_across_pipelines_with_compatible_layouts() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let uniforms = buffer(
            &device,
            16,
            BufferUsage::UNIFORM,
            Some(&vec4_bytes([1.0, 2.0, 3.0, 4.0])),
        );
        let output = output_buffer(&device, 32);
        let bindings = [
            (0, UNIFORM, COMPUTE, buffer_range(uniforms, 0, 16)),
            (1, STORAGE, COMPUTE, buffer_range(output, 0, 32)),
        ];
        let (layout, group) = device.bind_group(&bindings);
        // A second layout with the same entries is compatible with the first.
        let (twin_layout, _) = device.bind_group(&bindings);
        let first = compute_pipeline(
            &device,
            r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform Params { vec4 value; };
layout(set = 0, binding = 1) buffer Out { vec4 result[2]; };
void main() { result[0] = value; }
",
            &[layout],
        );
        let second = compute_pipeline(
            &device,
            r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform Params { vec4 value; };
layout(set = 0, binding = 1) buffer Out { vec4 result[2]; };
void main() { result[1] = value + 1.0; }
",
            &[twin_layout],
        );

        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.set_bind_group(0, group, &[]);
        recorder.bind_pipeline(first);
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&compute_to_compute(output, 32));
        recorder.bind_pipeline(second);
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&compute_to_host(output, 32));
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");

        assert_eq!(
            read_vec4s(&device, output, 0, 2),
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            "{backend:?}"
        );
    }
}

const WRITE_IMAGE: &str = r"#version 450
layout(local_size_x = 4, local_size_y = 4) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    imageStore(img, p, vec4(float(p.x) / 3.0, float(p.y) / 3.0, 0.0, 1.0));
}
";

#[test]
fn compute_writes_one_mip_of_a_storage_texture_through_a_view() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let texture = device
            .create_texture(
                &TextureDescriptor {
                    width: 8,
                    height: 8,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 2,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::STORAGE | TextureUsage::COPY_SRC,
                    dimension: TextureDimension::D2,
                    label: None,
                },
                None,
            )
            .expect("texture");
        let mip1 = device
            .create_texture_view(&TextureViewDescriptor {
                texture,
                dimension: TextureDimension::D2,
                base_mip_level: 1,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: TextureAspect::All,
                label: None,
            })
            .expect("view");
        let storage = BindingType::StorageTexture {
            format: TextureFormat::Rgba8Unorm,
            access: StorageTextureAccess::WriteOnly,
        };
        let (layout, group) = device.bind_group(&[(
            0,
            storage,
            COMPUTE,
            BindingResource::StorageTexture {
                texture: TextureBinding::View(mip1),
            },
        )]);
        // The whole two-mip texture cannot be a storage binding; one mip of it can.
        let too_many_mips = device.create_bind_group(&BindGroupDescriptor {
            layout,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::StorageTexture {
                    texture: texture.into(),
                },
            }],
            label: None,
        });
        assert!(
            matches!(too_many_mips, Err(RhiError::ValidationError(_))),
            "{backend:?}: {too_many_mips:?}"
        );
        let pipeline = compute_pipeline(&device, WRITE_IMAGE, &[layout]);

        let barrier = |old_layout, new_layout, src_access, dst_access, src_stages, dst_stages| {
            PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
                texture,
                range: TextureSubresourceRange::single(1, 0),
                src_stages,
                src_access,
                dst_stages,
                dst_access,
                old_layout,
                new_layout,
            })
        };
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.pipeline_barrier(&barrier(
            ImageLayout::Undefined,
            ImageLayout::General,
            AccessFlags::empty(),
            AccessFlags::SHADER_WRITE,
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::COMPUTE_SHADER,
        ));
        recorder.bind_pipeline(pipeline);
        recorder.set_bind_group(0, group, &[]);
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&barrier(
            ImageLayout::General,
            ImageLayout::TransferSrc,
            AccessFlags::SHADER_WRITE,
            AccessFlags::TRANSFER_READ,
            PipelineStageFlags::COMPUTE_SHADER,
            PipelineStageFlags::TRANSFER,
        ));
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");

        let texels = device.read_texture(texture, 1).expect("read_texture");
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 4) as usize;
                let expected = [(x * 255 / 3) as u8, (y * 255 / 3) as u8, 0, 255];
                let actual: [u8; 4] = texels[i..i + 4].try_into().unwrap();
                assert!(
                    actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 1),
                    "{backend:?}: texel ({x}, {y}) is {actual:?}, expected {expected:?}"
                );
            }
        }
    }
}

#[test]
fn bind_groups_are_checked_against_their_layout_and_resources() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let uniforms = buffer(&device, 512, BufferUsage::UNIFORM, None);
        let storage_only = output_buffer(&device, 512);
        let layout = device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    ty: UNIFORM,
                    stages: COMPUTE,
                }],
                label: None,
            })
            .expect("layout");
        let group = |resource| {
            device.create_bind_group(&BindGroupDescriptor {
                layout,
                entries: vec![BindGroupEntry {
                    binding: 0,
                    resource,
                }],
                label: None,
            })
        };
        let rejected = [
            (
                "missing UNIFORM usage",
                group(buffer_range(storage_only, 0, 16)),
            ),
            ("misaligned offset", group(buffer_range(uniforms, 4, 16))),
            (
                "range past the end",
                group(buffer_range(uniforms, 256, 512)),
            ),
        ];
        for (what, result) in rejected {
            assert!(
                matches!(result, Err(RhiError::ValidationError(_))),
                "{backend:?}: {what}: {result:?}"
            );
        }
        let unknown = group(buffer_range(BufferHandle::new(9999, 1), 0, 16));
        assert!(
            matches!(unknown, Err(RhiError::InvalidHandle(_))),
            "{backend:?}: {unknown:?}"
        );
        let duplicate = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    ty: UNIFORM,
                    stages: COMPUTE,
                },
                BindGroupLayoutEntry {
                    binding: 0,
                    ty: STORAGE,
                    stages: COMPUTE,
                },
            ],
            label: None,
        });
        assert!(duplicate.is_err(), "{backend:?}: duplicate binding");
        let five_sets = device.create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: device.shader(ShaderStage::Compute, DOUBLE_UNIFORM),
            bind_group_layouts: vec![layout; 5],
            push_constant_ranges: vec![],
            label: None,
        });
        assert!(
            matches!(five_sets, Err(RhiError::ValidationError(_))),
            "{backend:?}: {five_sets:?}"
        );
    }
}

#[test]
fn groups_whose_resources_were_destroyed_are_not_bound() {
    if !korppi_rhi::api_validation_requested() {
        return; // The check is part of debug-build recording validation.
    }
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let output = output_buffer(&device, 16);
        let uniforms = buffer(&device, 16, BufferUsage::UNIFORM, None);
        let (layout, group): (BindGroupLayoutHandle, BindGroupHandle) = device.bind_group(&[
            (0, UNIFORM, COMPUTE, buffer_range(uniforms, 0, 16)),
            (1, STORAGE, COMPUTE, buffer_range(output, 0, 16)),
        ]);
        let pipeline = compute_pipeline(
            &device,
            r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform Params { vec4 value; };
layout(set = 0, binding = 1) buffer Out { vec4 result; };
void main() { result = value; }
",
            &[layout],
        );
        device.destroy_buffer(uniforms).expect("destroy_buffer");

        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.bind_pipeline(pipeline);
        recorder.set_bind_group(0, group, &[]);
        recorder.dispatch(1, 1, 1);
        match recorder.finish() {
            Err(RhiError::ValidationError(message)) => assert!(
                message.contains("destroyed") && message.contains("no bind group is bound"),
                "{backend:?}: {message}"
            ),
            Err(other) => panic!("{backend:?}: unexpected error {other:?}"),
            Ok(_) => panic!("{backend:?}: a group with a destroyed buffer was accepted"),
        }
        device
            .destroy_bind_group(group)
            .expect("destroy_bind_group");
    }
}
