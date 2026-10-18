//! Sampler state and depth readback (findings RVK-05, RGL-13).

use korppi_rhi::{
    AccessFlags, BorderColor, BufferDescriptor, BufferHandle, BufferMemoryBarrier, BufferUsage,
    ComputePipelineDescriptor, DepthAttachment, FilterMode, ImageLayout, ImageMemoryBarrier,
    LoadOp, MemoryUsage, PipelineBarrier, PipelineStageFlags, SampleCount, SamplerAddressMode,
    SamplerDescriptor, ShaderStage, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsage,
};
use korppi_rhi::{BindingType, ShaderStages};
use korppi_rhi_conformance::{buffer_range, sampled, STORAGE};
use korppi_rhi_conformance::{color_pass, color_target, Backend, TestDevice};

/// Samples the texture at binding 0 outside [0, 1] into the storage buffer at binding 1.
const SAMPLE_OUTSIDE: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform sampler2D t;
layout(set = 0, binding = 1) buffer B { vec4 v[]; };
void main() { v[0] = textureLod(t, vec2(-0.5, -0.5), 0.0); }
"#;

const SAMPLE_CENTRE: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform sampler2D t;
layout(set = 0, binding = 1) buffer B { vec4 v[]; };
void main() { v[0] = texture(t, vec2(0.5, 0.5)); }
"#;

fn red_texture(device: &TestDevice) -> korppi_rhi::TextureHandle {
    device
        .create_texture(
            &TextureDescriptor {
                width: 2,
                height: 2,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                dimension: TextureDimension::D2,
                label: None,
            },
            Some(&[255_u8, 0, 0, 255].repeat(4)),
        )
        .expect("create texture")
}

fn output_buffer(device: &TestDevice) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size: 16,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            None,
        )
        .expect("create storage buffer")
}

/// Samples a red 2x2 texture with `sampler` using `shader` and returns the vec4 result.
fn sample(backend: Backend, sampler: SamplerDescriptor, shader: &'static str) -> [f32; 4] {
    let device = TestDevice::open(backend);
    let texture = red_texture(&device);
    let sampler = device.create_sampler(&sampler).expect("create sampler");
    let output = output_buffer(&device);
    let (layout, group) = device.bind_group(&[
        (
            0,
            BindingType::SampledTexture,
            ShaderStages::COMPUTE,
            sampled(texture, sampler),
        ),
        (
            1,
            STORAGE,
            ShaderStages::COMPUTE,
            buffer_range(output, 0, 16),
        ),
    ]);
    let compute_shader = device.shader(ShaderStage::Compute, shader);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create compute pipeline");
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
            size: 16,
        }),
    );
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let bytes = device.read_buffer(output, 0, 16).expect("read_buffer");
    let mut out = [0.0; 4];
    for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *value = f32::from_ne_bytes(*chunk);
    }
    out
}

fn border_color_is_applied(backend: Backend) {
    let clamp_to_white = SamplerDescriptor {
        min_filter: FilterMode::Nearest,
        mag_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        address_mode_u: SamplerAddressMode::ClampToBorder,
        address_mode_v: SamplerAddressMode::ClampToBorder,
        address_mode_w: SamplerAddressMode::ClampToBorder,
        border_color: BorderColor::OpaqueWhite,
        ..SamplerDescriptor::default()
    };
    assert_eq!(
        sample(backend, clamp_to_white.clone(), SAMPLE_OUTSIDE),
        [1.0, 1.0, 1.0, 1.0],
        "{backend:?}: opaque white border"
    );
    let clamp_to_default = SamplerDescriptor {
        border_color: BorderColor::default(),
        ..clamp_to_white
    };
    assert_eq!(
        sample(backend, clamp_to_default, SAMPLE_OUTSIDE),
        [0.0, 0.0, 0.0, 0.0],
        "{backend:?}: default (transparent black) border"
    );
}

#[test]
fn border_color_is_applied_vulkan() {
    border_color_is_applied(Backend::Vulkan);
}

#[test]
fn border_color_is_applied_opengl() {
    border_color_is_applied(Backend::OpenGl);
}

fn anisotropic_sampler_is_valid(backend: Backend) {
    let anisotropic = SamplerDescriptor {
        max_anisotropy: 16,
        ..SamplerDescriptor::default()
    };
    // TestDevice fails the test on any API error (e.g. VUID-VkSamplerCreateInfo-anisotropyEnable-01070).
    assert_eq!(
        sample(backend, anisotropic, SAMPLE_CENTRE),
        [1.0, 0.0, 0.0, 1.0],
        "{backend:?}"
    );
}

#[test]
fn anisotropic_sampler_is_valid_vulkan() {
    anisotropic_sampler_is_valid(Backend::Vulkan);
}

#[test]
fn anisotropic_sampler_is_valid_opengl() {
    anisotropic_sampler_is_valid(Backend::OpenGl);
}

fn depth32_texture_reads_back_as_f32(backend: Backend) {
    const SIZE: u32 = 8;
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
                usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::COPY_SRC,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("create depth texture");
    let mut pass = color_pass(color, SIZE, SIZE, LoadOp::Clear, [0.0; 4]);
    pass.depth_attachment = Some(DepthAttachment {
        texture: depth,
        mip_level: 0,
        layer: 0,
        load_op: LoadOp::Clear,
        store_op: StoreOp::Store,
        clear_depth: Some(0.25),
        clear_stencil: None,
        stencil_load_op: LoadOp::DontCare,
        stencil_store_op: StoreOp::DontCare,
    });
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    let mut barrier = korppi_rhi_conformance::undefined_to_color_attachment(color);
    barrier.image_barriers.push(ImageMemoryBarrier {
        texture: depth,
        range: korppi_rhi::TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | PipelineStageFlags::LATE_FRAGMENT_TESTS,
        dst_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::DepthStencilAttachment,
    });
    recorder.pipeline_barrier(&barrier);
    recorder.begin_rendering(&pass).expect("begin_rendering");
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let bytes = device.read_texture(depth, 0).expect("read_texture");
    assert_eq!(bytes.len(), (SIZE * SIZE * 4) as usize, "{backend:?}");
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect();
    assert!(
        values.iter().all(|&d| d == 0.25),
        "{backend:?}: depth readback {:?}",
        &values[..4]
    );
}

#[test]
fn depth32_texture_reads_back_as_f32_vulkan() {
    depth32_texture_reads_back_as_f32(Backend::Vulkan);
}

#[test]
fn depth32_texture_reads_back_as_f32_opengl() {
    depth32_texture_reads_back_as_f32(Backend::OpenGl);
}

/// A texture created after another was destroyed is the one sampled, even though OpenGL may
/// give it the destroyed texture's name, which the backend's state cache still had bound to the
/// unit (RGL-15, found by the FIX-1.4 format tests).
fn textures_replacing_destroyed_ones_are_sampled(backend: Backend) {
    let device = TestDevice::open(backend);
    let sampler = device
        .create_sampler(&SamplerDescriptor::default())
        .expect("sampler");
    let compute_shader = device.shader(ShaderStage::Compute, SAMPLE_CENTRE);
    let layout = device.layout(&[
        (0, BindingType::SampledTexture, ShaderStages::COMPUTE),
        (1, STORAGE, ShaderStages::COMPUTE),
    ]);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("compute pipeline");
    let mut results = Vec::new();
    for colour in [[255_u8, 0, 0, 255], [0, 0, 255, 255]] {
        let texture = device
            .create_texture(
                &TextureDescriptor {
                    width: 2,
                    height: 2,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                    dimension: TextureDimension::D2,
                    label: None,
                },
                Some(&colour.repeat(4)),
            )
            .expect("create texture");
        let output = output_buffer(&device);
        let group = device.group(
            layout,
            &[
                (0, sampled(texture, sampler)),
                (1, buffer_range(output, 0, 16)),
            ],
        );
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.bind_pipeline(pipeline);
        recorder.set_bind_group(0, group, &[]);
        recorder.dispatch(1, 1, 1);
        recorder.pipeline_barrier(&PipelineBarrier::new().with_buffer_barrier(
            BufferMemoryBarrier {
                buffer: output,
                src_stages: PipelineStageFlags::COMPUTE_SHADER,
                src_access: AccessFlags::SHADER_WRITE,
                dst_stages: PipelineStageFlags::HOST,
                dst_access: AccessFlags::HOST_READ,
                offset: 0,
                size: 16,
            },
        ));
        device
            .submit_and_wait(recorder.finish().expect("finish"))
            .expect("submit");
        let bytes = device.read_buffer(output, 0, 16).expect("read_buffer");
        results.push(f32::from_ne_bytes(bytes.as_chunks::<4>().0[2]));
        device.destroy_texture(texture).expect("destroy texture");
        device.destroy_buffer(output).expect("destroy buffer");
    }
    assert_eq!(
        results,
        [0.0, 1.0],
        "{backend:?}: blue channel of the first (red) and second (blue) texture"
    );
}

#[test]
fn textures_replacing_destroyed_ones_are_sampled_vulkan() {
    textures_replacing_destroyed_ones_are_sampled(Backend::Vulkan);
}

#[test]
fn textures_replacing_destroyed_ones_are_sampled_opengl() {
    textures_replacing_destroyed_ones_are_sampled(Backend::OpenGl);
}

/// `textureLod` at the LOD in the push constants.
const SAMPLE_AT_LOD: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform sampler2D t;
layout(set = 0, binding = 1) buffer B { vec4 v[]; };
layout(push_constant) uniform P { float lod; };
void main() { v[0] = textureLod(t, vec2(0.5, 0.5), lod); }
"#;

/// Samples a 3-mip texture (red, green, blue levels) at `lod` with `sampler`.
fn sample_mip_chain(device: &TestDevice, sampler: &SamplerDescriptor, lod: f32) -> [f32; 4] {
    use korppi_rhi::{PushConstantRange, TextureClear, TextureSubresourceRange};
    let texture = device
        .create_texture(
            &TextureDescriptor {
                width: 4,
                height: 4,
                depth: 1,
                array_layers: 1,
                mip_levels: 3,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("create texture");
    let sampler = device.create_sampler(sampler).expect("create sampler");
    let output = output_buffer(device);
    let (layout, group) = device.bind_group(&[
        (
            0,
            BindingType::SampledTexture,
            ShaderStages::COMPUTE,
            sampled(texture, sampler),
        ),
        (
            1,
            STORAGE,
            ShaderStages::COMPUTE,
            buffer_range(output, 0, 16),
        ),
    ]);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: device.shader(ShaderStage::Compute, SAMPLE_AT_LOD),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: 4,
            }],
            label: None,
        })
        .expect("create compute pipeline");
    let barrier = |old_layout, new_layout, src_access, dst_access, dst_stages| {
        PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture,
            range: TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TRANSFER,
            src_access,
            dst_stages,
            dst_access,
            old_layout,
            new_layout,
        })
    };
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    r.pipeline_barrier(&barrier(
        ImageLayout::Undefined,
        ImageLayout::TransferDst,
        AccessFlags::empty(),
        AccessFlags::TRANSFER_WRITE,
        PipelineStageFlags::TRANSFER,
    ));
    for (mip, color) in [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    {
        r.clear_texture(
            texture,
            &TextureSubresourceRange::single(mip as u32, 0),
            TextureClear::Color(color),
        );
    }
    r.pipeline_barrier(&barrier(
        ImageLayout::TransferDst,
        ImageLayout::ShaderReadOnly,
        AccessFlags::TRANSFER_WRITE,
        AccessFlags::SHADER_READ,
        PipelineStageFlags::COMPUTE_SHADER,
    ));
    r.bind_pipeline(pipeline);
    r.set_bind_group(0, group, &[]);
    r.push_constants(0, &lod.to_ne_bytes()).expect("push");
    r.dispatch(1, 1, 1);
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
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");
    let bytes = device.read_buffer(output, 0, 16).expect("read_buffer");
    let mut out = [0.0; 4];
    for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *value = f32::from_ne_bytes(*chunk);
    }
    out
}

/// LOD clamps and bias select mip levels on both backends (finding RVK-05).
fn lod_clamps_and_bias_select_mip_levels(backend: Backend) {
    let device = TestDevice::open(backend);
    let nearest = SamplerDescriptor {
        min_filter: FilterMode::Nearest,
        mag_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..SamplerDescriptor::default()
    };
    let (red, green, blue) = (
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    );
    let cases = [
        ("no clamp, LOD 0", nearest.clone(), 0.0, red),
        ("no clamp, LOD 2", nearest.clone(), 2.0, blue),
        (
            "minimum LOD 1",
            SamplerDescriptor {
                lod_min_clamp: 1.0,
                ..nearest.clone()
            },
            0.0,
            green,
        ),
        (
            "maximum LOD 1",
            SamplerDescriptor {
                lod_max_clamp: 1.0,
                ..nearest.clone()
            },
            2.0,
            green,
        ),
        (
            "LOD bias 1",
            SamplerDescriptor {
                mip_lod_bias: 1.0,
                ..nearest.clone()
            },
            0.0,
            green,
        ),
    ];
    for (what, sampler, lod, expected) in cases {
        assert_eq!(
            sample_mip_chain(&device, &sampler, lod),
            expected,
            "{backend:?}: {what}"
        );
    }
    let inverted = SamplerDescriptor {
        lod_min_clamp: 2.0,
        lod_max_clamp: 1.0,
        ..nearest
    };
    assert!(
        matches!(
            device.create_sampler(&inverted),
            Err(korppi_rhi::RhiError::ValidationError(_))
        ),
        "{backend:?}: a minimum LOD above the maximum is rejected"
    );
}

#[test]
fn lod_clamps_and_bias_select_mip_levels_vulkan() {
    lod_clamps_and_bias_select_mip_levels(Backend::Vulkan);
}

#[test]
fn lod_clamps_and_bias_select_mip_levels_opengl() {
    lod_clamps_and_bias_select_mip_levels(Backend::OpenGl);
}
