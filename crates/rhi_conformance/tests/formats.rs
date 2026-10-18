//! Texture formats and device capabilities (FIX-1.4; findings RHI-09, RHI-13, RVK-13, RGL-13).

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferMemoryBarrier, BufferUsage, ComputePipelineDescriptor,
    DeviceFeatures, FormatCapabilities, GraphicsApi, MemoryUsage, PipelineBarrier,
    PipelineStageFlags, RhiError, SampleCount, SamplerDescriptor, ShaderStage, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi::{BindingType, ShaderStages};
use korppi_rhi_conformance::{buffer_range, sampled, STORAGE};
use korppi_rhi_conformance::{Backend, TestDevice};

fn texture_desc(
    format: TextureFormat,
    width: u32,
    height: u32,
    usage: TextureUsage,
) -> TextureDescriptor {
    TextureDescriptor {
        width,
        height,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        format,
        usage,
        dimension: TextureDimension::D2,
        label: Some("format_test"),
    }
}

#[test]
fn adapter_info_and_limits_meet_the_core_minimums() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let info = device.adapter_info();
        let expected_api = match backend {
            Backend::Vulkan => GraphicsApi::Vulkan,
            Backend::OpenGl => GraphicsApi::OpenGl,
        };
        assert_eq!(info.api, expected_api);
        assert!(!info.name.is_empty(), "{backend:?}: {info:?}");
        assert!(info.api_version >= (1, 3, 0) || backend == Backend::OpenGl);
        assert!(info.api_version >= (4, 5, 0) || backend == Backend::Vulkan);

        let limits = device.limits();
        // Vulkan core minimums (OpenGL 4.5's are at least as high for these).
        assert!(
            limits.max_texture_dimension_2d >= 4096,
            "{backend:?}: {limits:?}"
        );
        assert!(
            limits.max_texture_dimension_cube >= 4096,
            "{backend:?}: {limits:?}"
        );
        assert!(
            limits.max_texture_dimension_3d >= 256,
            "{backend:?}: {limits:?}"
        );
        assert!(
            limits.max_texture_array_layers >= 256,
            "{backend:?}: {limits:?}"
        );
        assert!(limits.max_color_attachments >= 4, "{backend:?}: {limits:?}");
        assert!(
            limits.max_vertex_attributes >= 16,
            "{backend:?}: {limits:?}"
        );
        assert!(limits.max_vertex_buffers >= 16, "{backend:?}: {limits:?}");
        assert_eq!(limits.max_push_constants_size, 128, "{backend:?}");
        for alignment in [
            limits.min_uniform_buffer_offset_alignment,
            limits.min_storage_buffer_offset_alignment,
        ] {
            assert!(
                alignment.is_power_of_two() && alignment <= 256,
                "{backend:?}: {limits:?}"
            );
        }
        assert!(
            limits.max_uniform_buffer_binding_size >= 16384,
            "{backend:?}"
        );
        assert!(
            limits.max_storage_buffer_binding_size >= 1 << 27,
            "{backend:?}"
        );
        assert!(limits.max_per_stage_uniform_buffers >= 12, "{backend:?}");
        assert!(limits.max_per_stage_storage_buffers >= 4, "{backend:?}");
        assert!(limits.max_per_stage_sampled_textures >= 16, "{backend:?}");
        assert!(limits.max_per_stage_storage_textures >= 4, "{backend:?}");
        assert!(limits.max_sampler_anisotropy >= 1.0, "{backend:?}");
        assert!(
            limits.max_compute_workgroup_size.iter().all(|&s| s >= 64),
            "{backend:?}"
        );
        assert!(
            limits.max_compute_invocations_per_workgroup >= 128,
            "{backend:?}"
        );
        assert!(
            limits
                .max_compute_workgroup_count
                .iter()
                .all(|&c| c >= 65535),
            "{backend:?}"
        );
        assert!(
            limits.max_compute_shared_memory_size >= 16384,
            "{backend:?}"
        );
        let features = device.features();
        assert_eq!(
            features.contains(DeviceFeatures::TIMESTAMP_QUERY),
            limits.timestamp_period_ns > 0.0,
            "{backend:?}: {features:?}"
        );
    }
}

#[test]
fn format_capabilities_are_consistent() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let features = device.features();
        for format in TextureFormat::ALL {
            let caps = device.format_capabilities(format);
            let context = format!("{backend:?} {format:?}: {caps:?}");
            if caps.contains(FormatCapabilities::FILTERABLE) {
                assert!(caps.contains(FormatCapabilities::SAMPLED), "{context}");
            }
            if caps.contains(FormatCapabilities::BLENDABLE) {
                assert!(
                    caps.contains(FormatCapabilities::COLOR_ATTACHMENT),
                    "{context}"
                );
            }
            if format.is_depth() {
                assert!(
                    !caps.intersects(
                        FormatCapabilities::COLOR_ATTACHMENT
                            | FormatCapabilities::BLENDABLE
                            | FormatCapabilities::STORAGE
                    ),
                    "{context}"
                );
                assert!(
                    caps.contains(FormatCapabilities::DEPTH_STENCIL_ATTACHMENT),
                    "{context}: every RHI depth format must be renderable (D24S8 falls back)"
                );
            } else {
                assert!(
                    !caps.contains(FormatCapabilities::DEPTH_STENCIL_ATTACHMENT),
                    "{context}"
                );
            }
            if format.is_integer() {
                assert!(
                    !caps
                        .intersects(FormatCapabilities::FILTERABLE | FormatCapabilities::BLENDABLE),
                    "{context}"
                );
            }
            if format.is_compressed() {
                assert!(
                    !caps.intersects(
                        FormatCapabilities::COLOR_ATTACHMENT
                            | FormatCapabilities::DEPTH_STENCIL_ATTACHMENT
                            | FormatCapabilities::STORAGE
                    ),
                    "{context}"
                );
                assert_eq!(
                    caps.contains(FormatCapabilities::SAMPLED),
                    features.contains(DeviceFeatures::TEXTURE_COMPRESSION_BC),
                    "{context}"
                );
            }
            if format == TextureFormat::Depth24PlusStencil8 {
                assert!(
                    !caps.intersects(FormatCapabilities::COPY_SRC | FormatCapabilities::COPY_DST),
                    "{context}: its depth layout is device-specific, so it is not copyable"
                );
            }
        }
        // Formats Vulkan requires full support for, and OpenGL 4.5 has as core formats.
        let rgba8 = device.format_capabilities(TextureFormat::Rgba8Unorm);
        assert!(
            rgba8.contains(
                FormatCapabilities::SAMPLED
                    | FormatCapabilities::FILTERABLE
                    | FormatCapabilities::STORAGE
                    | FormatCapabilities::COLOR_ATTACHMENT
                    | FormatCapabilities::BLENDABLE
                    | FormatCapabilities::COPY_SRC
                    | FormatCapabilities::COPY_DST
                    | FormatCapabilities::MULTISAMPLE_X4
            ),
            "{backend:?}: {rgba8:?}"
        );
        let depth = device.format_capabilities(TextureFormat::Depth32Float);
        assert!(
            depth.contains(
                FormatCapabilities::SAMPLED
                    | FormatCapabilities::DEPTH_STENCIL_ATTACHMENT
                    | FormatCapabilities::COPY_SRC
            ),
            "{backend:?}: {depth:?}"
        );
    }
}

#[test]
fn unsupported_format_usages_are_rejected() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        // Compressed formats can never be render targets.
        let result = device.create_texture(
            &texture_desc(
                TextureFormat::Bc1RgbaUnorm,
                4,
                4,
                TextureUsage::COLOR_ATTACHMENT,
            ),
            None,
        );
        assert!(
            matches!(result, Err(RhiError::Unsupported(_))),
            "{backend:?}: {result:?}"
        );
        let result = device.create_texture(
            &texture_desc(
                TextureFormat::Depth24PlusStencil8,
                4,
                4,
                TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::COPY_SRC,
            ),
            None,
        );
        assert!(
            matches!(result, Err(RhiError::Unsupported(_))),
            "{backend:?}: {result:?}"
        );
    }
}

/// Bit patterns that every format stores exactly: finite floats for float formats, depth in
/// [0, 1], arbitrary bytes otherwise.
fn pattern(format: TextureFormat, len: usize) -> Vec<u8> {
    let mut data = vec![0u8; len];
    match format {
        TextureFormat::R16Float | TextureFormat::Rg16Float | TextureFormat::Rgba16Float => {
            // Halves 0.0..2.0 in steps of 1/8 (exponents 0x00..0x40, small mantissas).
            for (i, half) in data.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                *half = (0x3000u16 + (i as u16 % 64) * 0x40).to_le_bytes();
            }
        }
        TextureFormat::R32Float
        | TextureFormat::Rg32Float
        | TextureFormat::Rgb32Float
        | TextureFormat::Rgba32Float => {
            for (i, float) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                *float = (i as f32 * 0.25 - 3.0).to_le_bytes();
            }
        }
        TextureFormat::Depth32Float | TextureFormat::Depth32FloatStencil8 => {
            for (i, float) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                *float = (i as f32 / 64.0).fract().to_le_bytes();
            }
        }
        TextureFormat::Rg11B10Float => {
            // Each component: exponent 15 (1.0) with a varying mantissa; the 10-bit blue
            // channel likewise.
            for (i, packed) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let m = i as u32 % 32;
                let r = (15 << 6) | m;
                let g = (15 << 6) | (m ^ 0x15);
                let b = (15 << 5) | (m & 0x1F);
                *packed = (r | (g << 11) | (b << 22)).to_le_bytes();
            }
        }
        _ => {
            for (i, byte) in data.iter_mut().enumerate() {
                *byte = (i * 37 + 11) as u8;
            }
        }
    }
    data
}

#[test]
fn copyable_formats_round_trip_through_textures_unchanged() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        for format in TextureFormat::ALL {
            let caps = device.format_capabilities(format);
            let needed = FormatCapabilities::COPY_SRC | FormatCapabilities::COPY_DST;
            if !caps.contains(needed) {
                continue;
            }
            let (width, height) = (8, 8);
            let data = pattern(format, format.image_bytes(width, height) as usize);
            let texture = device
                .create_texture(
                    &texture_desc(
                        format,
                        width,
                        height,
                        TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                    ),
                    Some(&data),
                )
                .unwrap_or_else(|e| panic!("{backend:?} {format:?}: create_texture: {e}"));
            let read = device
                .read_texture(texture, 0)
                .unwrap_or_else(|e| panic!("{backend:?} {format:?}: read_texture: {e}"));
            assert!(
                read == data,
                "{backend:?} {format:?}: read back {} bytes differing from the {} uploaded \
                 (first bytes {:?} vs {:?})",
                read.len(),
                data.len(),
                &read[..read.len().min(16)],
                &data[..16]
            );
            device.destroy_texture(texture).expect("destroy_texture");
        }
    }
}

/// Samples the texture at binding 0 at its centre into the storage buffer at binding 1.
const SAMPLE_CENTRE: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform sampler2D t;
layout(set = 0, binding = 1) buffer B { vec4 v[]; };
void main() { v[0] = textureLod(t, vec2(0.5, 0.5), 0.0); }
"#;

/// One 4x4 block of `format` that decodes to a single known colour.
fn solid_block(format: TextureFormat) -> (Vec<u8>, [f32; 4]) {
    // BC1: colour0 = pure red (RGB565 0xF800) > colour1 = blue, all indices 0 (colour0).
    let bc1_red = vec![0x00, 0xF8, 0x1F, 0x00, 0, 0, 0, 0];
    // BC4: red0 = 255 > red1 = 0, all indices 0 (red0).
    let bc4_one = vec![0xFF, 0x00, 0, 0, 0, 0, 0, 0];
    // BC4 alpha block of BC3: alpha0 = 0x80 > alpha1 = 0, indices 0.
    let bc3_alpha = vec![0x80, 0x00, 0, 0, 0, 0, 0, 0];
    match format {
        TextureFormat::Bc1RgbaUnorm => (bc1_red, [1.0, 0.0, 0.0, 1.0]),
        TextureFormat::Bc3RgbaUnorm => (
            [bc3_alpha, bc1_red].concat(),
            [1.0, 0.0, 0.0, 128.0 / 255.0],
        ),
        TextureFormat::Bc4RUnorm => (bc4_one, [1.0, 0.0, 0.0, 1.0]),
        TextureFormat::Bc5RgUnorm => ([bc4_one.clone(), bc4_one].concat(), [1.0, 1.0, 0.0, 1.0]),
        // An all-zero BC7 block uses no valid mode and decodes to transparent black; an
        // all-zero BC6H block (mode 0, zero endpoints) decodes to black.
        TextureFormat::Bc7RgbaUnorm => (vec![0; 16], [0.0, 0.0, 0.0, 0.0]),
        TextureFormat::Bc6hRgbUfloat => (vec![0; 16], [0.0, 0.0, 0.0, 1.0]),
        _ => unreachable!("no test block for {format:?}"),
    }
}

#[test]
fn compressed_textures_upload_read_back_and_decode() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        if !device
            .features()
            .contains(DeviceFeatures::TEXTURE_COMPRESSION_BC)
        {
            eprintln!("{backend:?}: BC compression unsupported; skipping");
            continue;
        }
        let shader = device.shader(ShaderStage::Compute, SAMPLE_CENTRE);
        let layout = device.layout(&[
            (0, BindingType::SampledTexture, ShaderStages::COMPUTE),
            (1, STORAGE, ShaderStages::COMPUTE),
        ]);
        let pipeline = device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                compute_shader: shader,
                bind_group_layouts: vec![layout],
                push_constant_ranges: vec![],
                label: None,
            })
            .expect("compute pipeline");
        let sampler = device
            .create_sampler(&SamplerDescriptor::default())
            .expect("sampler");
        for format in [
            TextureFormat::Bc1RgbaUnorm,
            TextureFormat::Bc3RgbaUnorm,
            TextureFormat::Bc4RUnorm,
            TextureFormat::Bc5RgUnorm,
            TextureFormat::Bc6hRgbUfloat,
            TextureFormat::Bc7RgbaUnorm,
        ] {
            let (block, expected) = solid_block(format);
            let texture = device
                .create_texture(
                    &texture_desc(
                        format,
                        4,
                        4,
                        TextureUsage::SAMPLED | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
                    ),
                    Some(&block),
                )
                .unwrap_or_else(|e| panic!("{backend:?} {format:?}: {e}"));
            assert_eq!(
                device.read_texture(texture, 0).expect("read_texture"),
                block,
                "{backend:?} {format:?}: compressed blocks read back unchanged"
            );
            let output = device
                .create_buffer(
                    &BufferDescriptor {
                        size: 16,
                        usage: BufferUsage::STORAGE,
                        memory: MemoryUsage::GpuToCpu,
                        label: None,
                    },
                    None,
                )
                .expect("output buffer");
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
            let mut sampled = [0.0f32; 4];
            for (value, chunk) in sampled.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *value = f32::from_le_bytes(*chunk);
            }
            assert!(
                sampled
                    .iter()
                    .zip(expected)
                    .all(|(s, e)| (s - e).abs() < 0.01),
                "{backend:?} {format:?}: sampled {sampled:?}, expected {expected:?}"
            );
            device.destroy_buffer(output).expect("destroy buffer");
            device.destroy_texture(texture).expect("destroy texture");
        }
    }
}
