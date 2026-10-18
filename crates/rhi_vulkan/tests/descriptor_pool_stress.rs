//! Stress tests for descriptor pool growth and recycling. Every bind group owns a descriptor set
//! from the device's pools, which grow on demand; destroyed groups' sets are freed once the GPU
//! is done with them and reused (RHI-07, RVK-09).

use korppi_rhi_vulkan::VulkanDevice;

use korppi_rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferDescriptor, BufferUsage, ComputePipelineDescriptor,
    FilterMode, GraphicsDevice, ImageLayout, MemoryUsage, SamplerAddressMode, SamplerDescriptor,
    ShaderDescriptor, ShaderSource, ShaderStage, ShaderStages, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsage,
};
use std::io::Write;
use std::process::{Command, Stdio};

fn compile_glsl_to_spirv(glsl_src: &str, stage: &str) -> Vec<u8> {
    let mut child = Command::new("glslc")
        .arg(format!("-fshader-stage={stage}"))
        .arg("-")
        .arg("-o")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn glslc");

    child
        .stdin
        .as_mut()
        .expect("Failed to open stdin")
        .write_all(glsl_src.as_bytes())
        .expect("Failed to write to glslc stdin");

    let output = child.wait_with_output().expect("Failed to wait on glslc");
    assert!(
        output.status.success(),
        "glslc error: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn bind_groups_grow_and_reuse_descriptor_pools() {
    let device = VulkanDevice::create_headless().expect("Headless Vulkan device creation failed");

    let ubo_desc = BufferDescriptor {
        size: 256,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryUsage::CpuToGpu,
        label: Some("stress_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, None)
        .expect("Create UBO failed");

    let tex_desc = TextureDescriptor {
        width: 1,
        height: 1,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: korppi_rhi::SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        dimension: TextureDimension::D2,
        label: Some("stress_texture"),
    };
    let tex_bytes = [255u8, 0u8, 0u8, 255u8];
    let tex = device
        .create_texture(&tex_desc, Some(&tex_bytes))
        .expect("Create texture failed");

    let samp_desc = SamplerDescriptor {
        min_filter: FilterMode::Nearest,
        mag_filter: FilterMode::Nearest,
        address_mode_u: SamplerAddressMode::ClampToEdge,
        address_mode_v: SamplerAddressMode::ClampToEdge,
        ..Default::default()
    };
    let sampler = device
        .create_sampler(&samp_desc)
        .expect("Create sampler failed");

    let ssbo_desc = BufferDescriptor {
        size: 16,
        usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: MemoryUsage::GpuToCpu,
        label: Some("stress_ssbo"),
    };
    let ssbo = device
        .create_buffer(&ssbo_desc, None)
        .expect("Create SSBO failed");

    let glsl = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform FrameUbo { float u_Multiplier; };
layout(set = 0, binding = 1) uniform sampler2D u_Tex;
layout(set = 0, binding = 2) buffer DataBuffer { vec4 values[]; };
void main() {
    values[gl_GlobalInvocationID.x] = texture(u_Tex, vec2(0.5, 0.5)) * u_Multiplier;
}
"#;
    let spirv_bytes = compile_glsl_to_spirv(glsl, "compute");
    let shader = device
        .create_shader(&ShaderDescriptor {
            stage: ShaderStage::Compute,
            entry_point: "main",
            source: ShaderSource::SpirvBytes(&spirv_bytes),
            label: Some("stress_compute_shader"),
        })
        .expect("Create shader failed");
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        ty,
        stages: ShaderStages::COMPUTE,
    };
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![
                entry(
                    0,
                    BindingType::UniformBuffer {
                        dynamic_offset: false,
                    },
                ),
                entry(1, BindingType::SampledTexture),
                entry(
                    2,
                    BindingType::StorageBuffer {
                        read_only: false,
                        dynamic_offset: false,
                    },
                ),
            ],
            label: None,
        })
        .expect("Create layout failed");
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: Some("stress_compute_pipeline"),
        })
        .expect("Create compute pipeline failed");
    let group_desc = BindGroupDescriptor {
        layout,
        entries: vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: ubo,
                    offset: 0,
                    size: 256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::SampledTexture {
                    texture: tex.into(),
                    sampler,
                    layout: ImageLayout::ShaderReadOnly,
                },
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer {
                    buffer: ssbo,
                    offset: 0,
                    size: 16,
                },
            },
        ],
        label: None,
    };

    // 1,200 groups, far beyond one pool, each used by a dispatch.
    let run = || {
        let groups: Vec<_> = (0..1200)
            .map(|_| {
                device
                    .create_bind_group(&group_desc)
                    .expect("create_bind_group")
            })
            .collect();
        let mut recorder = device
            .create_command_recorder()
            .expect("Create command recorder failed");
        recorder.begin().expect("Begin failed");
        recorder.bind_pipeline(pipeline);
        for &group in &groups {
            recorder.set_bind_group(0, group, &[]);
            recorder.dispatch(1, 1, 1);
        }
        device
            .submit(recorder.finish().expect("Finish command recording failed"))
            .expect("Submit failed");
        // Destroyed while the submission may still run: freed once it completes.
        for group in groups {
            device
                .destroy_bind_group(group)
                .expect("destroy_bind_group");
        }
    };

    run();
    device.wait_idle().expect("Wait idle failed");
    let after_first = device.descriptor_pool_count();
    assert!(
        after_first >= 2,
        "1,200 groups must need more than one descriptor pool: {after_first}"
    );
    assert_eq!(
        device.pending_destruction_count(),
        0,
        "freed after wait_idle"
    );

    // Later rounds reuse the freed sets instead of adding pools.
    for _ in 0..4 {
        run();
        device.wait_idle().expect("Wait idle failed");
    }
    assert_eq!(
        device.descriptor_pool_count(),
        after_first,
        "descriptor sets of destroyed groups must be reused"
    );
    assert_no_api_errors(&device);
}

/// Fails the test if the backend's API validation (Vulkan validation layers, OpenGL debug
/// output) reported errors (FIX-1.0).
fn assert_no_api_errors(device: &dyn korppi_rhi::GraphicsDevice) {
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == korppi_rhi::DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "API errors reported: {errors:#?}");
}
