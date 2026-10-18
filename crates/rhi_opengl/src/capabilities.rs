//! Device capabilities reported through the RHI (finding RHI-09), queried once on the render
//! thread: limits with `glGetIntegerv`, per-format support with `glGetInternalformativ`
//! (`ARB_internalformat_query2`, core in 4.3).

use crate::gl_fns::GlExtraFns;
use crate::types::to_gl_texture_format;
use glow::HasContext;
use korppi_rhi::{
    AdapterInfo, DeviceFeatures, DeviceLimits, DeviceType, FormatCapabilities, GraphicsApi,
    TextureFormat, MAX_PUSH_CONSTANTS_SIZE,
};

/// Everything the device reports through `GraphicsDevice::{adapter_info, limits, features,
/// format_capabilities}`.
#[derive(Clone, Debug)]
pub struct GlCapabilities {
    /// Device and driver identification.
    pub adapter: AdapterInfo,
    /// Numeric limits.
    pub limits: DeviceLimits,
    /// Supported optional features.
    pub features: DeviceFeatures,
    formats: Vec<FormatCapabilities>,
}

impl GlCapabilities {
    /// Capabilities of `format`.
    pub fn format(&self, format: TextureFormat) -> FormatCapabilities {
        self.formats[format.index()]
    }

    /// Queries the current context.
    ///
    /// # Safety
    /// The context must be current on the calling thread; `extra` must belong to it.
    pub(crate) unsafe fn query(
        gl: &glow::Context,
        extra: &GlExtraFns,
        max_anisotropy: f32,
        presentation: bool,
    ) -> Self {
        // SAFETY: the context is current (caller); every query below is a plain state query.
        unsafe {
            let int = |name| gl.get_parameter_i32(name).max(0) as u32;
            let indexed = |name, index| gl.get_parameter_indexed_i32(name, index).max(0) as u32;
            let min_of = |names: &[u32]| names.iter().map(|&n| int(n)).min().unwrap_or(0);
            let version = gl.version();
            let renderer = gl.get_parameter_string(glow::RENDERER);
            let vendor = gl.get_parameter_string(glow::VENDOR);
            let version_string = gl.get_parameter_string(glow::VERSION);
            let software = ["llvmpipe", "softpipe", "swrast", "SwiftShader"]
                .iter()
                .any(|s| renderer.contains(s));
            let adapter = AdapterInfo {
                api: GraphicsApi::OpenGl,
                name: renderer,
                vendor_id: 0,
                device_id: 0,
                device_type: if software {
                    DeviceType::Cpu
                } else {
                    DeviceType::Other
                },
                driver: format!("{vendor} {version_string}"),
                api_version: (version.major, version.minor, 0),
            };

            let extensions = gl.supported_extensions();
            let timestamp_bits = extra.get_queryiv.map_or(0, |get_queryiv| {
                let mut bits = 0;
                get_queryiv(glow::TIMESTAMP, glow::QUERY_COUNTER_BITS, &mut bits);
                bits
            });
            let mut features = DeviceFeatures::DEPTH_CLAMP
                | DeviceFeatures::POLYGON_MODE_LINE
                | DeviceFeatures::MULTI_DRAW_INDIRECT
                | DeviceFeatures::INDIRECT_FIRST_INSTANCE;
            if max_anisotropy > 0.0 {
                features |= DeviceFeatures::SAMPLER_ANISOTROPY;
            }
            // RGTC (BC4/5) and BPTC (BC6H/7) are core; S3TC (BC1-3) is an extension.
            if extensions.contains("GL_EXT_texture_compression_s3tc") {
                features |= DeviceFeatures::TEXTURE_COMPRESSION_BC;
            }
            if extra.polygon_offset_clamp.is_some() {
                features |= DeviceFeatures::DEPTH_BIAS_CLAMP;
            }
            if timestamp_bits > 0 {
                features |= DeviceFeatures::TIMESTAMP_QUERY;
            }
            if presentation {
                features |= DeviceFeatures::PRESENTATION;
            }

            let limits = DeviceLimits {
                max_texture_dimension_1d: int(glow::MAX_TEXTURE_SIZE),
                max_texture_dimension_2d: int(glow::MAX_TEXTURE_SIZE),
                max_texture_dimension_3d: int(glow::MAX_3D_TEXTURE_SIZE),
                max_texture_dimension_cube: int(glow::MAX_CUBE_MAP_TEXTURE_SIZE),
                max_texture_array_layers: int(glow::MAX_ARRAY_TEXTURE_LAYERS),
                max_color_attachments: min_of(&[
                    glow::MAX_COLOR_ATTACHMENTS,
                    glow::MAX_DRAW_BUFFERS,
                ]),
                max_vertex_buffers: int(glow::MAX_VERTEX_ATTRIB_BINDINGS),
                max_vertex_attributes: int(glow::MAX_VERTEX_ATTRIBS),
                max_vertex_buffer_stride: int(glow::MAX_VERTEX_ATTRIB_STRIDE),
                max_push_constants_size: MAX_PUSH_CONSTANTS_SIZE as u32,
                min_uniform_buffer_offset_alignment: u64::from(int(
                    glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
                )),
                min_storage_buffer_offset_alignment: u64::from(int(
                    glow::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT,
                )),
                max_uniform_buffer_binding_size: u64::from(int(glow::MAX_UNIFORM_BLOCK_SIZE)),
                max_storage_buffer_binding_size: gl
                    .get_parameter_i64(glow::MAX_SHADER_STORAGE_BLOCK_SIZE)
                    .max(0) as u64,
                max_per_stage_uniform_buffers: min_of(&[
                    glow::MAX_VERTEX_UNIFORM_BLOCKS,
                    glow::MAX_FRAGMENT_UNIFORM_BLOCKS,
                    glow::MAX_COMPUTE_UNIFORM_BLOCKS,
                ]),
                // The vertex stage may have none (its minimum is 0); fragment and compute
                // stages, where storage buffers are used, decide the limit.
                max_per_stage_storage_buffers: min_of(&[
                    glow::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
                    glow::MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
                ]),
                max_per_stage_sampled_textures: min_of(&[
                    glow::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
                    glow::MAX_TEXTURE_IMAGE_UNITS,
                    glow::MAX_COMPUTE_TEXTURE_IMAGE_UNITS,
                ]),
                max_per_stage_storage_textures: min_of(&[
                    glow::MAX_FRAGMENT_IMAGE_UNIFORMS,
                    glow::MAX_COMPUTE_IMAGE_UNIFORMS,
                ]),
                max_sampler_anisotropy: max_anisotropy.max(1.0),
                max_compute_workgroup_size: [0, 1, 2]
                    .map(|i| indexed(glow::MAX_COMPUTE_WORK_GROUP_SIZE, i)),
                max_compute_invocations_per_workgroup: int(
                    glow::MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
                ),
                max_compute_workgroup_count: [0, 1, 2]
                    .map(|i| indexed(glow::MAX_COMPUTE_WORK_GROUP_COUNT, i)),
                max_compute_shared_memory_size: int(glow::MAX_COMPUTE_SHARED_MEMORY_SIZE),
                timestamp_period_ns: if timestamp_bits > 0 { 1.0 } else { 0.0 },
            };

            let formats = TextureFormat::ALL
                .iter()
                .map(|&format| format_capabilities(gl, format, features))
                .collect();
            Self {
                adapter,
                limits,
                features,
                formats,
            }
        }
    }
}

/// One `glGetInternalformativ` value for a 2D texture of `internal_format`.
///
/// # Safety
/// The context must be current.
unsafe fn internal_format_query(
    gl: &glow::Context,
    target: u32,
    internal_format: u32,
    pname: u32,
) -> i32 {
    let mut value = [0];
    // SAFETY: forwarded from the caller; `value` holds the one integer these queries return.
    unsafe { gl.get_internal_format_i32_slice(target, internal_format, pname, &mut value) };
    value[0]
}

/// Capabilities of `format`.
///
/// # Safety
/// The context must be current.
unsafe fn format_capabilities(
    gl: &glow::Context,
    format: TextureFormat,
    features: DeviceFeatures,
) -> FormatCapabilities {
    let internal = to_gl_texture_format(format).internal_format;
    // SAFETY: forwarded from the caller.
    let query = |pname| unsafe { internal_format_query(gl, glow::TEXTURE_2D, internal, pname) };
    let supported =
        |value: i32| value == glow::FULL_SUPPORT as i32 || value == glow::CAVEAT_SUPPORT as i32;
    if query(glow::INTERNALFORMAT_SUPPORTED) != glow::TRUE as i32 {
        return FormatCapabilities::empty();
    }
    if format.is_compressed() && !features.contains(DeviceFeatures::TEXTURE_COMPRESSION_BC) {
        return FormatCapabilities::empty();
    }
    let mut caps = FormatCapabilities::COPY_SRC | FormatCapabilities::COPY_DST;
    if supported(query(glow::FRAGMENT_TEXTURE)) {
        caps |= FormatCapabilities::SAMPLED;
        // FILTER returns a support level (FULL_SUPPORT/CAVEAT_SUPPORT/NONE), not a boolean.
        if supported(query(glow::FILTER)) {
            caps |= FormatCapabilities::FILTERABLE;
        }
    }
    let renderable = query(glow::FRAMEBUFFER_RENDERABLE) == glow::FULL_SUPPORT as i32;
    if format.is_depth() {
        if renderable {
            caps |= FormatCapabilities::DEPTH_STENCIL_ATTACHMENT;
        }
    } else if renderable && !format.is_compressed() {
        caps |= FormatCapabilities::COLOR_ATTACHMENT;
        if supported(query(glow::FRAMEBUFFER_BLEND)) && !format.is_integer() {
            caps |= FormatCapabilities::BLENDABLE;
        }
    }
    if !format.is_depth() && !format.is_compressed() && supported(query(glow::SHADER_IMAGE_STORE)) {
        caps |= FormatCapabilities::STORAGE;
    }
    if caps.intersects(
        FormatCapabilities::COLOR_ATTACHMENT | FormatCapabilities::DEPTH_STENCIL_ATTACHMENT,
    ) {
        // SAFETY: forwarded from the caller.
        let count = unsafe {
            internal_format_query(
                gl,
                glow::TEXTURE_2D_MULTISAMPLE,
                internal,
                glow::NUM_SAMPLE_COUNTS,
            )
        };
        let mut samples = vec![0; count.max(0) as usize];
        if !samples.is_empty() {
            // SAFETY: forwarded from the caller; `samples` holds `NUM_SAMPLE_COUNTS` integers.
            unsafe {
                gl.get_internal_format_i32_slice(
                    glow::TEXTURE_2D_MULTISAMPLE,
                    internal,
                    glow::SAMPLES,
                    &mut samples,
                )
            };
        }
        for (count, cap) in [
            (2, FormatCapabilities::MULTISAMPLE_X2),
            (4, FormatCapabilities::MULTISAMPLE_X4),
            (8, FormatCapabilities::MULTISAMPLE_X8),
        ] {
            if samples.contains(&count) {
                caps |= cap;
            }
        }
    }
    // Blits go through framebuffers, so the format must be renderable.
    if caps.intersects(
        FormatCapabilities::COLOR_ATTACHMENT | FormatCapabilities::DEPTH_STENCIL_ATTACHMENT,
    ) {
        caps |= FormatCapabilities::BLIT;
    }
    // Its depth layout differs between implementations, so copies are not portable (the RHI's
    // contract, as on Vulkan).
    if format == TextureFormat::Depth24PlusStencil8 {
        caps -= FormatCapabilities::COPY_SRC | FormatCapabilities::COPY_DST;
    }
    caps
}
