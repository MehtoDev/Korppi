//! Integration test verifying Vulkan UBO, SSBO, and Texture resource bindings via compute shader.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferUsage, ComputePipelineDescriptor, FilterMode,
    GraphicsDevice, ImageLayout, ImageMemoryBarrier, MemoryUsage, PipelineBarrier,
    PipelineStageFlags, SamplerAddressMode, SamplerDescriptor, ShaderDescriptor, ShaderSource,
    ShaderStage, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, ShaderStages,
};
use korppi_rhi_vulkan::VulkanDevice;
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
fn test_vulkan_compute_resource_binding_sampling_and_readback() {
    let device = VulkanDevice::create_headless().expect("Headless Vulkan device creation failed");

    // 1. Create Frame UBO: Set 0, binding 0 (float multiplier = 2.0)
    let ubo_desc = BufferDescriptor {
        size: 256,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        memory: MemoryUsage::CpuToGpu,
        label: Some("frame_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, None)
        .expect("Create UBO failed");
    let multiplier: f32 = 2.0;
    device
        .write_buffer(ubo, 0, &multiplier.to_ne_bytes())
        .expect("Write UBO failed");

    // 2. Create Texture & Sampler: Set 1, binding 0
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
        label: Some("input_texture"),
    };
    // Texture texel: RGBA(128, 64, 32, 255) -> normalized floats approx (0.502, 0.251, 0.125, 1.0)
    let tex_bytes = [128u8, 64u8, 32u8, 255u8];
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

    // 3. Create SSBO: Set 2, binding 0 (output buffer for vec4)
    let ssbo_desc = BufferDescriptor {
        size: 16, // vec4 = 16 bytes
        usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        memory: MemoryUsage::GpuToCpu,
        label: Some("output_ssbo"),
    };
    let ssbo = device
        .create_buffer(&ssbo_desc, None)
        .expect("Create SSBO failed");

    // 4. Compile compute shader that binds UBO (set 0), Texture (set 1), and SSBO (set 2)
    let glsl = r#"#version 450
layout(local_size_x = 1) in;

layout(set = 0, binding = 0) uniform FrameUbo {
    float u_Multiplier;
};

layout(set = 1, binding = 0) uniform sampler2D u_Tex;

layout(set = 2, binding = 0) buffer DataBuffer {
    vec4 values[];
};

void main() {
    vec4 tex_color = texture(u_Tex, vec2(0.5, 0.5));
    values[gl_GlobalInvocationID.x] = tex_color * u_Multiplier;
}
"#;
    let spirv_bytes = compile_glsl_to_spirv(glsl, "compute");

    let shader_desc = ShaderDescriptor {
        stage: ShaderStage::Compute,
        entry_point: "main",
        source: ShaderSource::SpirvBytes(&spirv_bytes),
        label: Some("compute_shader"),
    };
    let shader = device
        .create_shader(&shader_desc)
        .expect("Create shader failed");

    // One bind group per set: uniform buffer (0), texture (1), storage buffer (2).
    let set = |ty, resource| {
        let layout = device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    ty,
                    stages: ShaderStages::COMPUTE,
                }],
                label: None,
            })
            .expect("Create bind group layout failed");
        let group = device
            .create_bind_group(&BindGroupDescriptor {
                layout,
                entries: vec![BindGroupEntry {
                    binding: 0,
                    resource,
                }],
                label: None,
            })
            .expect("Create bind group failed");
        (layout, group)
    };
    let (ubo_layout, ubo_group) = set(
        BindingType::UniformBuffer {
            dynamic_offset: false,
        },
        BindingResource::Buffer {
            buffer: ubo,
            offset: 0,
            size: 256,
        },
    );
    let (tex_layout, tex_group) = set(
        BindingType::SampledTexture,
        BindingResource::SampledTexture {
            texture: tex.into(),
            sampler,
            layout: ImageLayout::ShaderReadOnly,
        },
    );
    let (ssbo_layout, ssbo_group) = set(
        BindingType::StorageBuffer {
            read_only: false,
            dynamic_offset: false,
        },
        BindingResource::Buffer {
            buffer: ssbo,
            offset: 0,
            size: 16,
        },
    );

    let pipe_desc = ComputePipelineDescriptor {
        compute_shader: shader,
        bind_group_layouts: vec![ubo_layout, tex_layout, ssbo_layout],
        push_constant_ranges: vec![],
        label: Some("compute_pipeline"),
    };
    let pipeline = device
        .create_compute_pipeline(&pipe_desc)
        .expect("Create compute pipeline failed");

    // 5. Record command buffer with pipeline & resource bindings
    let mut recorder = device
        .create_command_recorder()
        .expect("Create command recorder failed");
    recorder.begin().expect("Begin failed");

    recorder.pipeline_barrier(&PipelineBarrier {
        memory_barriers: korppi_rhi::smallvec![],
        buffer_barriers: korppi_rhi::smallvec![],
        image_barriers: korppi_rhi::smallvec![ImageMemoryBarrier {
            texture: tex,
            range: korppi_rhi::TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::COMPUTE_SHADER,
            dst_access: AccessFlags::SHADER_READ,
            old_layout: ImageLayout::ShaderReadOnly,
            new_layout: ImageLayout::ShaderReadOnly,
        }],
    });

    recorder.bind_pipeline(pipeline);
    recorder.set_bind_group(0, ubo_group, &[]);
    recorder.set_bind_group(1, tex_group, &[]);
    recorder.set_bind_group(2, ssbo_group, &[]);
    recorder.dispatch(1, 1, 1);

    let cmd = recorder.finish().expect("Finish command recording failed");
    device.submit(cmd).expect("Submit failed");
    device.wait_idle().expect("Wait idle failed");

    // 6. Read back SSBO and verify computed values
    let output_bytes = device.read_buffer(ssbo, 0, 16).expect("Read SSBO failed");
    let r = f32::from_ne_bytes(output_bytes[0..4].try_into().unwrap());
    let g = f32::from_ne_bytes(output_bytes[4..8].try_into().unwrap());
    let b = f32::from_ne_bytes(output_bytes[8..12].try_into().unwrap());
    let a = f32::from_ne_bytes(output_bytes[12..16].try_into().unwrap());

    // Expected: tex_color = [128/255, 64/255, 32/255, 1.0] * 2.0 = [1.004, 0.502, 0.251, 2.0]
    let expected_r = (128.0 / 255.0) * 2.0;
    let expected_g = (64.0 / 255.0) * 2.0;
    let expected_b = (32.0 / 255.0) * 2.0;
    let expected_a = 2.0;

    assert!(
        (r - expected_r).abs() < 0.05,
        "Red mismatch: {} vs {}",
        r,
        expected_r
    );
    assert!(
        (g - expected_g).abs() < 0.05,
        "Green mismatch: {} vs {}",
        g,
        expected_g
    );
    assert!(
        (b - expected_b).abs() < 0.05,
        "Blue mismatch: {} vs {}",
        b,
        expected_b
    );
    assert!(
        (a - expected_a).abs() < 0.05,
        "Alpha mismatch: {} vs {}",
        a,
        expected_a
    );

    device.destroy_buffer(ssbo).expect("Destroy SSBO failed");
    device.destroy_buffer(ubo).expect("Destroy UBO failed");
    device.destroy_texture(tex).expect("Destroy texture failed");
    device
        .destroy_sampler(sampler)
        .expect("Destroy sampler failed");
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
