//! OpenGL 4.5 resource definitions and thread-safe generational resource registry.

use korppi_rhi::{
    BufferHandle, BufferUsage, ColorTargetState, DepthStencilState, MemoryUsage, MultisampleState,
    PipelineHandle, PipelineSignature, RasterizerState, ResourceOracle, SamplerHandle, ShaderStage,
    TextureFormat, TextureHandle, TextureInfo,
};
use std::sync::Mutex;

/// Internal OpenGL buffer resource.
pub struct GlBuffer {
    /// Glow buffer handle.
    pub buffer: glow::Buffer,
    /// Total buffer size in bytes.
    pub size: u64,
    /// Usage bitflags.
    pub usage: BufferUsage,
    /// Memory placement.
    pub memory: MemoryUsage,
    /// Persistent, coherent mapping of `CpuToGpu` and `GpuToCpu` buffers (RHI-05).
    pub mapped: Option<MappedPtr>,
    /// Makes host access exclusive (`map_buffer`, `write_buffer`, `read_buffer`,
    /// `destroy_buffer`).
    pub host_access: korppi_rhi::HostAccessLock,
}

/// Start of a buffer's persistent mapping.
#[derive(Copy, Clone, Debug)]
pub struct MappedPtr(pub std::ptr::NonNull<u8>);

// SAFETY: a persistent mapping is process memory valid until the buffer is deleted, usable from
// any thread; host access to it is serialized by the buffer's `HostAccessLock`.
unsafe impl Send for MappedPtr {}
// SAFETY: as above.
unsafe impl Sync for MappedPtr {}

/// Internal OpenGL texture resource.
pub struct GlTexture {
    /// Glow texture handle.
    pub texture: glow::Texture,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Depth or layer count.
    pub depth: u32,
    /// Number of mipmap levels.
    pub mip_levels: u32,
    /// Texture pixel format.
    pub format: TextureFormat,
    /// ID of the swapchain that owns the texture, which destroys it; `None` for device textures.
    pub swapchain: Option<u32>,
    /// Texture target (`GL_TEXTURE_2D`, `GL_TEXTURE_2D_ARRAY`, `GL_TEXTURE_CUBE_MAP`, ...).
    pub target: u32,
    /// Dimension, mips, layers and samples.
    pub info: TextureInfo,
    /// Views of this texture, deleted with it.
    pub views: Vec<korppi_rhi::TextureViewHandle>,
}

impl GlTexture {
    /// True if attachments select a layer (`glNamedFramebufferTextureLayer`): arrays, cube
    /// maps and 3D textures.
    pub fn is_layered(&self) -> bool {
        matches!(
            self.target,
            glow::TEXTURE_2D_ARRAY
                | glow::TEXTURE_2D_MULTISAMPLE_ARRAY
                | glow::TEXTURE_CUBE_MAP
                | glow::TEXTURE_CUBE_MAP_ARRAY
                | glow::TEXTURE_3D
        )
    }
}

/// A texture view: a separate GL texture created with `glTextureView`.
pub struct GlTextureView {
    /// The view's texture name.
    pub texture: glow::Texture,
    /// The viewed texture.
    pub parent: korppi_rhi::TextureHandle,
    /// Target of the view.
    pub target: u32,
    /// The view's description.
    pub desc: korppi_rhi::TextureViewDescriptor,
}

/// Internal OpenGL sampler resource.
pub struct GlSampler {
    /// Glow sampler handle.
    pub sampler: glow::Sampler,
}

/// Internal OpenGL shader module resource.
pub struct GlShader {
    /// The shader's code.
    pub code: GlShaderCode,
    /// Shader execution stage.
    pub stage: ShaderStage,
}

/// Code of a GL shader.
pub enum GlShaderCode {
    /// GLSL compiled when the shader was created; its binding points are its own.
    Compiled {
        /// The shader object.
        shader: glow::Shader,
        /// Hash of its source (program-binary keys).
        source_hash: u64,
    },
    /// Parsed SPIR-V, written as GLSL with each pipeline's binding points (RHI-07).
    Spirv(std::sync::Arc<crate::spirv::SpirvShader>),
}

/// Vertex buffer bindings supported per pipeline (`GL_MAX_VERTEX_ATTRIB_BINDINGS` is at least 16).
pub const MAX_VERTEX_BINDINGS: usize = 16;

/// Internal OpenGL pipeline (linked GLSL program + fixed function state).
pub struct GlPipeline {
    /// Glow program handle.
    pub program: glow::Program,
    /// Vertex array object holding this pipeline's vertex input layout (graphics pipelines).
    pub vertex_array: Option<glow::VertexArray>,
    /// Stride in bytes of each vertex buffer binding the layout declares, indexed by binding.
    pub vertex_strides: [Option<i32>; MAX_VERTEX_BINDINGS],
    /// Primitive topology.
    pub topology: u32,
    /// Rasterizer state.
    pub rasterizer: RasterizerState,
    /// Optional depth stencil state.
    pub depth_stencil: Option<DepthStencilState>,
    /// Multisample state.
    pub multisample: MultisampleState,
    /// Blend state and write mask per colour target (empty for compute pipelines).
    pub color_targets: Vec<ColorTargetState>,
    /// Summary used by command validation.
    pub signature: PipelineSignature,
    /// GL binding points of the pipeline's bind group layouts.
    pub binding_table: crate::binding::GlBindingTable,
}

/// Thread-safe generational resource registry for OpenGL handles, of one device: its maps find
/// no handle of another device (RHI-15).
pub struct GlResourceRegistry {
    /// The device whose resources these are; every handle the registry hands out carries it.
    pub device: korppi_rhi::DeviceId,
    /// Active buffers.
    pub buffers: Mutex<korppi_rhi::SlotMap<GlBuffer>>,
    /// Active textures.
    pub textures: Mutex<korppi_rhi::SlotMap<GlTexture>>,
    /// Active texture views.
    pub texture_views: Mutex<korppi_rhi::SlotMap<GlTextureView>>,
    /// Active bind group layouts.
    pub bind_group_layouts: Mutex<korppi_rhi::SlotMap<crate::binding::GlBindGroupLayout>>,
    /// Active bind groups.
    pub bind_groups: Mutex<korppi_rhi::SlotMap<crate::binding::GlBindGroup>>,
    /// Active samplers.
    pub samplers: Mutex<korppi_rhi::SlotMap<GlSampler>>,
    /// Active shaders.
    pub shaders: Mutex<korppi_rhi::SlotMap<GlShader>>,
    /// Active pipelines.
    pub pipelines: Mutex<korppi_rhi::SlotMap<GlPipeline>>,
    /// Active query sets: one GL query per query.
    pub query_sets: Mutex<korppi_rhi::SlotMap<Vec<glow::Query>>>,
    /// The device's optional features, for recording validation; set at device creation.
    pub features: std::sync::OnceLock<korppi_rhi::DeviceFeatures>,
    /// Per-format capabilities of the device (indexed by `TextureFormat::index`), for recording
    /// validation; set once at device creation.
    pub format_capabilities: std::sync::OnceLock<Vec<korppi_rhi::FormatCapabilities>>,
}

impl GlResourceRegistry {
    /// An empty registry of `device`'s resources.
    pub fn for_device(device: korppi_rhi::DeviceId) -> Self {
        fn map<T>(device: korppi_rhi::DeviceId) -> Mutex<korppi_rhi::SlotMap<T>> {
            Mutex::new(korppi_rhi::SlotMap::for_device(device))
        }
        Self {
            device,
            buffers: map(device),
            textures: map(device),
            texture_views: map(device),
            bind_group_layouts: map(device),
            bind_groups: map(device),
            samplers: map(device),
            shaders: map(device),
            pipelines: map(device),
            query_sets: map(device),
            features: std::sync::OnceLock::new(),
            format_capabilities: std::sync::OnceLock::new(),
        }
    }
}

/// Locks a registry map (recovering from poisoning).
fn read<T>(
    map: &Mutex<korppi_rhi::SlotMap<T>>,
) -> std::sync::MutexGuard<'_, korppi_rhi::SlotMap<T>> {
    map.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl ResourceOracle for GlResourceRegistry {
    fn buffer_exists(&self, handle: BufferHandle) -> bool {
        read(&self.buffers).contains(handle)
    }

    fn buffer_summary(&self, handle: BufferHandle) -> Option<korppi_rhi::BufferSummary> {
        read(&self.buffers)
            .get(handle)
            .map(|(b, _)| korppi_rhi::BufferSummary {
                size: Some(b.size),
                usage: Some(b.usage),
            })
    }

    fn texture_info(&self, handle: TextureHandle) -> Option<TextureInfo> {
        read(&self.textures).get(handle).map(|(t, _)| t.info)
    }

    fn sampler_exists(&self, handle: SamplerHandle) -> bool {
        read(&self.samplers).contains(handle)
    }

    fn pipeline_signature(&self, handle: PipelineHandle) -> Option<PipelineSignature> {
        read(&self.pipelines).get(handle).map(|(p, _)| p.signature)
    }

    fn bind_group_info(
        &self,
        handle: korppi_rhi::BindGroupHandle,
    ) -> Option<korppi_rhi::BindGroupInfo> {
        read(&self.bind_groups).get(handle).map(|(g, _)| g.info)
    }

    fn query_set_count(&self, handle: korppi_rhi::QuerySetHandle) -> Option<u32> {
        read(&self.query_sets)
            .get(handle)
            .map(|(q, _)| q.len() as u32)
    }

    fn buffer_usage(&self, handle: BufferHandle) -> Option<BufferUsage> {
        read(&self.buffers).get(handle).map(|(b, _)| b.usage)
    }

    fn features(&self) -> korppi_rhi::DeviceFeatures {
        self.features
            .get()
            .copied()
            .unwrap_or(korppi_rhi::DeviceFeatures::all())
    }

    fn buffer_size(&self, handle: BufferHandle) -> Option<u64> {
        read(&self.buffers).get(handle).map(|(b, _)| b.size)
    }

    fn format_capabilities(
        &self,
        format: korppi_rhi::TextureFormat,
    ) -> korppi_rhi::FormatCapabilities {
        self.format_capabilities
            .get()
            .and_then(|caps| caps.get(format.index()).copied())
            .unwrap_or(korppi_rhi::FormatCapabilities::all())
    }
}
