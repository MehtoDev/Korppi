//! GPU-backed 3D Clustered Forward Rendering Pipeline (`FIX-6.4`, `RND-01`, `RND-11`).

use crate::clustered::{
    ClusterGridConfig, FroxelAabb, GpuCullingBuffers, GpuLightCuller, PointLight, ReverseZ,
    SpotLight,
};
use crate::error::RenderError;
use crate::graph::{
    PassContext, PassType, RenderGraph, RenderGraphTextureHandle, TextureAccess,
    TransientResourcePool, TransientTextureDesc,
};
use crate::pbr::dfg_lut_packed;
use crate::pipeline::scene::{ForwardScene, Vertex3d};
use crate::postprocess::{
    generate_ssao_kernel, BloomConfig, SsaoConfig, TonemapConfig, TonemapOperator,
};
use crate::shadows::{compute_cascades_into, Cascade, CascadeConfig};
use korppi_math::{Mat4, Vec3, Vec4};
use korppi_rhi::{
    AccessFlags, BindGroupDescriptor, BindGroupEntry, BindGroupHandle, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindGroupLayoutHandle, BindingResource, BindingType, BlendComponent,
    BlendFactor, BlendOp, BlendState, BufferDescriptor, BufferHandle, BufferMemoryBarrier,
    BufferUsage, ColorTargetState, ColorWriteMask, CommandBufferRecorder, CompareFunction,
    CullMode, DepthStencilState, FilterMode, FrontFace, GraphicsDevice, GraphicsPipelineDescriptor,
    ImageLayout, IndexType, LoadOp, MemoryUsage, MultisampleState, PipelineBarrier, PipelineHandle,
    PipelineStageFlags, PolygonMode, PrimitiveTopology, PushConstantRange, RasterizerState,
    RhiError, SampleCount, SamplerAddressMode, SamplerDescriptor, SamplerHandle, ShaderDescriptor,
    ShaderHandle, ShaderSource, ShaderStage, ShaderStages, StencilFaceState, StoreOp,
    TextureBinding, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle,
    TextureUsage, VertexAttribute, VertexBinding, VertexFormat, VertexLayoutDescriptor,
    VertexStepMode, FRAMES_IN_FLIGHT,
};

const TEX_UNDEFINED: TextureAccess = TextureAccess {
    stage: PipelineStageFlags::TOP_OF_PIPE,
    access: AccessFlags::empty(),
    layout: ImageLayout::Undefined,
};

const TEX_SHADER_READ: TextureAccess = TextureAccess {
    stage: PipelineStageFlags::FRAGMENT_SHADER,
    access: AccessFlags::SHADER_READ,
    layout: ImageLayout::ShaderReadOnly,
};

const TEX_SHADER_READ_DEPTH: TextureAccess = TextureAccess {
    stage: PipelineStageFlags::FRAGMENT_SHADER,
    access: AccessFlags::SHADER_READ,
    layout: ImageLayout::DepthStencilReadOnly,
};

const DEPTH_PREPASS_VERT_SPV: &[u8] = include_bytes!("shaders/depth_prepass.vert.spv");
const DEPTH_PREPASS_FRAG_SPV: &[u8] = include_bytes!("shaders/depth_prepass.frag.spv");
const FULLSCREEN_VERT_SPV: &[u8] = include_bytes!("shaders/fullscreen.vert.spv");
const OCTAHEDRAL_UNWRAP_FRAG_SPV: &[u8] = include_bytes!("shaders/octahedral_unwrap.frag.spv");
const SSAO_FRAG_SPV: &[u8] = include_bytes!("shaders/ssao.frag.spv");
const BLOOM_FRAG_SPV: &[u8] = include_bytes!("shaders/bloom.frag.spv");
const TONEMAP_FRAG_SPV: &[u8] = include_bytes!("shaders/tonemap.frag.spv");
const FORWARD_PBR_VERT_SPV: &[u8] = include_bytes!("shaders/forward_pbr.vert.spv");
const FORWARD_PBR_FRAG_SPV: &[u8] = include_bytes!("shaders/forward_pbr.frag.spv");

const CUBE_FACE_SIZE: u32 = 256;
const CUBE_STRIP_WIDTH: u32 = CUBE_FACE_SIZE * 6;
const CUBE_STRIP_HEIGHT: u32 = CUBE_FACE_SIZE;

const MAX_BLOOM_MIPS: usize = 6;

const BLOOM_MIP_TEX_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "bloom_mip_0",
    "bloom_mip_1",
    "bloom_mip_2",
    "bloom_mip_3",
    "bloom_mip_4",
    "bloom_mip_5",
];

const BLOOM_DOWN_PASS_NAMES: [&str; MAX_BLOOM_MIPS] = [
    "bloom_down_0",
    "bloom_down_1",
    "bloom_down_2",
    "bloom_down_3",
    "bloom_down_4",
    "bloom_down_5",
];

const BLOOM_UP_PASS_NAMES: [&str; MAX_BLOOM_MIPS - 1] = [
    "bloom_up_0",
    "bloom_up_1",
    "bloom_up_2",
    "bloom_up_3",
    "bloom_up_4",
];

const POINT_SHADOW_CUBE_NAMES: [&str; 4] = [
    "point_shadow_cube_0",
    "point_shadow_cube_1",
    "point_shadow_cube_2",
    "point_shadow_cube_3",
];

const POINT_SHADOW_UNWRAP_NAMES: [&str; 4] = [
    "point_shadow_unwrap_0",
    "point_shadow_unwrap_1",
    "point_shadow_unwrap_2",
    "point_shadow_unwrap_3",
];

const CUBE_FACE_DIRS: [(Vec3, Vec3); 6] = [
    // +X, -X
    (Vec3::X, Vec3::Y),
    (Vec3::NEG_X, Vec3::Y),
    // +Y, -Y
    (Vec3::Y, Vec3::NEG_Z),
    (Vec3::NEG_Y, Vec3::Z),
    // +Z, -Z
    (Vec3::Z, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y),
];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DepthPush {
    mvp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct OctUnwrapPush {
    near_far_offset: [f32; 4],
    inv_tile_size: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DrawPush {
    model: [[f32; 4]; 4],
    albedo_roughness: [f32; 4],
    emissive_metallic: [f32; 4],
    reflectance_clearcoat: [f32; 4],
    sheen: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BloomPush {
    texel_radius_mode: [f32; 4],
    threshold_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TonemapPush {
    params: [f32; 4],
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SsaoParamsGpu {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    params0: [f32; 4],
    params1: [f32; 4],
    samples: [[f32; 4]; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SceneUniformsGpu {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    ambient_light: [f32; 4],
    dir_light_dir: [f32; 4],
    dir_light_radiance: [f32; 4],
    cluster_dims: [u32; 4],
    cluster_depth_params: [f32; 4],
    viewport_and_shadow_counts: [f32; 4],
    csm_view_proj: [[[f32; 4]; 4]; 4],
    csm_atlas_uv: [[f32; 4]; 4],
    csm_split_depths: [f32; 4],
    spot_shadow_view_proj: [[[f32; 4]; 4]; 4],
    spot_shadow_atlas_uv: [[f32; 4]; 4],
    spot_shadow_light_indices: [u32; 4],
    point_shadow_pos_radius: [[f32; 4]; 4],
    point_shadow_atlas_uv: [[f32; 4]; 4],
    point_shadow_light_indices: [u32; 4],
}

gpu_pod!(
    DepthPush = 16 * 4,
    OctUnwrapPush = 8 * 4,
    DrawPush = 32 * 4,
    BloomPush = 8 * 4,
    TonemapPush = 8 * 4,
    SsaoParamsGpu = 296 * 4,
    SceneUniformsGpu = 280 * 4,
);

/// Configuration for [`ForwardPipeline`].
#[derive(Clone, Debug)]
pub struct ForwardPipelineConfig {
    /// Viewport width in pixels.
    pub width: u32,
    /// Viewport height in pixels.
    pub height: u32,
    /// Format of the output LDR color target.
    pub ldr_format: TextureFormat,
    /// Hardware MSAA sample count for the forward opaque pass (`Count1` or `Count4`).
    pub msaa_samples: SampleCount,
    /// Width and height of the square `Depth32Float` unified shadow atlas in pixels.
    pub shadow_atlas_size: u32,
    /// Tonemapping and exposure configuration.
    pub tonemap: TonemapConfig,
    /// Optional progressive bloom configuration.
    pub bloom: Option<BloomConfig>,
    /// Optional screen-space ambient occlusion configuration.
    pub ssao: Option<SsaoConfig>,
}

impl Default for ForwardPipelineConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            ldr_format: TextureFormat::Rgba8Unorm,
            msaa_samples: SampleCount::Count4,
            shadow_atlas_size: 2048,
            tonemap: TonemapConfig::default(),
            bloom: Some(BloomConfig::default()),
            ssao: Some(SsaoConfig::default()),
        }
    }
}

/// GPU-backed 3D Clustered Forward Renderer orchestrating shadow mapping, Reverse-Z depth prepass,
/// compute froxel light culling, SSAO, MSAA forward PBR shading, progressive bloom, and tonemapping.
pub struct ForwardPipeline {
    config: ForwardPipelineConfig,
    pool: TransientResourcePool,
    gpu_culler: GpuLightCuller,

    // Shaders
    sh_depth_vert: ShaderHandle,
    sh_depth_frag: ShaderHandle,
    sh_fullscreen_vert: ShaderHandle,
    sh_oct_unwrap_frag: ShaderHandle,
    sh_ssao_frag: ShaderHandle,
    sh_bloom_frag: ShaderHandle,
    sh_tonemap_frag: ShaderHandle,
    sh_forward_vert: ShaderHandle,
    sh_forward_frag: ShaderHandle,

    // Bind group layouts
    bgl_oct_unwrap: BindGroupLayoutHandle,
    bgl_ssao: BindGroupLayoutHandle,
    bgl_forward_scene: BindGroupLayoutHandle,
    bgl_forward_lights: BindGroupLayoutHandle,
    bgl_bloom: BindGroupLayoutHandle,
    bgl_tonemap: BindGroupLayoutHandle,

    // Pipelines
    pipe_depth_prepass: PipelineHandle,
    pipe_depth_prepass_msaa: Option<PipelineHandle>,
    pipe_shadow_depth: PipelineHandle,
    pipe_oct_unwrap: PipelineHandle,
    pipe_ssao: PipelineHandle,
    pipe_forward: PipelineHandle,
    pipe_bloom_down: PipelineHandle,
    pipe_bloom_up: PipelineHandle,
    pipe_tonemap: PipelineHandle,

    // Samplers & persistent textures/buffers
    sampler_linear: SamplerHandle,
    sampler_nearest: SamplerHandle,
    tex_white_1x1: TextureHandle,
    tex_black_1x1: TextureHandle,
    tex_depth_prepass: TextureHandle,
    depth_1x_written: bool,
    tex_shadow_atlas: TextureHandle,
    tex_cube_strip: TextureHandle,
    buf_dfg_lut: BufferHandle,
    buf_scene_ubo: [BufferHandle; FRAMES_IN_FLIGHT],
    buf_ssao_ubo: [BufferHandle; FRAMES_IN_FLIGHT],
    ssao_kernel: [[f32; 4]; 64],

    // Cached bind groups to guarantee 0 allocations in steady-state frame recording
    bg_oct_unwrap: BindGroupHandle,
    bg_ssao: [Option<(TextureHandle, BindGroupHandle)>; FRAMES_IN_FLIGHT],
    bg_forward_scene: [Option<(TextureHandle, BindGroupHandle)>; FRAMES_IN_FLIGHT],
    bg_forward_lights: [Option<([BufferHandle; 4], BindGroupHandle)>; FRAMES_IN_FLIGHT],
    bg_bloom_sources: Vec<(TextureHandle, BindGroupHandle)>,
    bg_tonemap_sources: Vec<((TextureHandle, TextureHandle), BindGroupHandle)>,

    // Reusable CPU scratch buffers for zero-allocation steady-state recording
    scratch_froxels: Vec<FroxelAabb>,
    scratch_point_lights: Vec<PointLight>,
    scratch_spot_lights: Vec<SpotLight>,
    scratch_cascades: Vec<Cascade>,
}

impl ForwardPipeline {
    /// Creates and initializes all shaders, pipelines, samplers, and persistent GPU resources.
    pub fn new(
        device: &dyn GraphicsDevice,
        config: ForwardPipelineConfig,
    ) -> Result<Self, RenderError> {
        let mut gpu_culler = GpuLightCuller::new();
        gpu_culler.init(device)?;

        let create_sh = |stage: ShaderStage, spv: &'static [u8], label: &'static str| {
            device
                .create_shader(&ShaderDescriptor {
                    stage,
                    entry_point: "main",
                    source: ShaderSource::SpirvBytes(spv),
                    label: Some(label),
                })
                .map_err(|e| {
                    RenderError::Rhi(RhiError::ShaderCompilationFailed(format!("{label}: {e}")))
                })
        };

        let sh_depth_vert = create_sh(
            ShaderStage::Vertex,
            DEPTH_PREPASS_VERT_SPV,
            "forward_depth_prepass_vert",
        )?;
        let sh_depth_frag = create_sh(
            ShaderStage::Fragment,
            DEPTH_PREPASS_FRAG_SPV,
            "forward_depth_prepass_frag",
        )?;
        let sh_fullscreen_vert = create_sh(
            ShaderStage::Vertex,
            FULLSCREEN_VERT_SPV,
            "forward_fullscreen_vert",
        )?;
        let sh_oct_unwrap_frag = create_sh(
            ShaderStage::Fragment,
            OCTAHEDRAL_UNWRAP_FRAG_SPV,
            "forward_oct_unwrap_frag",
        )?;
        let sh_ssao_frag = create_sh(ShaderStage::Fragment, SSAO_FRAG_SPV, "forward_ssao_frag")?;
        let sh_bloom_frag = create_sh(ShaderStage::Fragment, BLOOM_FRAG_SPV, "forward_bloom_frag")?;
        let sh_tonemap_frag = create_sh(
            ShaderStage::Fragment,
            TONEMAP_FRAG_SPV,
            "forward_tonemap_frag",
        )?;
        let sh_forward_vert = create_sh(
            ShaderStage::Vertex,
            FORWARD_PBR_VERT_SPV,
            "forward_pbr_vert",
        )?;
        let sh_forward_frag = create_sh(
            ShaderStage::Fragment,
            FORWARD_PBR_FRAG_SPV,
            "forward_pbr_frag",
        )?;

        // Bind group layouts
        let bgl_oct_unwrap = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                ty: BindingType::SampledTexture,
                stages: ShaderStages::FRAGMENT,
            }],
            label: Some("bgl_oct_unwrap"),
        })?;

        let bgl_ssao = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: false,
                    },
                    stages: ShaderStages::FRAGMENT,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    ty: BindingType::SampledTexture,
                    stages: ShaderStages::FRAGMENT,
                },
            ],
            label: Some("bgl_ssao"),
        })?;

        let bgl_forward_scene = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: false,
                    },
                    stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    ty: BindingType::SampledTexture,
                    stages: ShaderStages::FRAGMENT,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    ty: BindingType::SampledTexture,
                    stages: ShaderStages::FRAGMENT,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    ty: BindingType::StorageBuffer {
                        read_only: true,
                        dynamic_offset: false,
                    },
                    stages: ShaderStages::FRAGMENT,
                },
            ],
            label: Some("bgl_forward_scene"),
        })?;

        let bgl_forward_lights = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: (0..4)
                .map(|binding| BindGroupLayoutEntry {
                    binding,
                    ty: BindingType::StorageBuffer {
                        read_only: true,
                        dynamic_offset: false,
                    },
                    stages: ShaderStages::FRAGMENT,
                })
                .collect(),
            label: Some("bgl_forward_lights"),
        })?;

        let bgl_bloom = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                ty: BindingType::SampledTexture,
                stages: ShaderStages::FRAGMENT,
            }],
            label: Some("bgl_bloom"),
        })?;

        let bgl_tonemap = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    ty: BindingType::SampledTexture,
                    stages: ShaderStages::FRAGMENT,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    ty: BindingType::SampledTexture,
                    stages: ShaderStages::FRAGMENT,
                },
            ],
            label: Some("bgl_tonemap"),
        })?;

        let vertex_layout_3d = VertexLayoutDescriptor::new()
            .with_binding(VertexBinding {
                binding: 0,
                stride: size_of::<Vertex3d>() as u64,
                step_mode: VertexStepMode::Vertex,
            })
            .with_attribute(VertexAttribute {
                location: 0,
                binding: 0,
                format: VertexFormat::Float32x3,
                offset: 0,
            })
            .with_attribute(VertexAttribute {
                location: 1,
                binding: 0,
                format: VertexFormat::Float32x3,
                offset: 12,
            })
            .with_attribute(VertexAttribute {
                location: 2,
                binding: 0,
                format: VertexFormat::Float32x2,
                offset: 24,
            });

        // 1. Reverse-Z Depth Prepass pipeline (also used for 6-face point light cube strip)
        let pipe_depth_prepass = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_depth_vert,
            fragment_shader: Some(sh_depth_frag),
            vertex_layout: vertex_layout_3d.clone(),
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
                count: SampleCount::Count1,
                alpha_to_coverage: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_test_enabled: true,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil_front: StencilFaceState::IGNORE,
                stencil_back: StencilFaceState::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            color_targets: Vec::new(),
            bind_group_layouts: Vec::new(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::VERTEX,
                offset: 0,
                size: size_of::<DepthPush>() as u32,
            }],
            label: Some("pipe_depth_prepass_reverse_z"),
        })?;

        let pipe_depth_prepass_msaa = if config.msaa_samples != SampleCount::Count1 {
            Some(
                device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
                    vertex_shader: sh_depth_vert,
                    fragment_shader: Some(sh_depth_frag),
                    vertex_layout: vertex_layout_3d.clone(),
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
                        count: config.msaa_samples,
                        alpha_to_coverage: false,
                    },
                    depth_stencil: Some(DepthStencilState {
                        format: TextureFormat::Depth32Float,
                        depth_test_enabled: true,
                        depth_write_enabled: true,
                        depth_compare: CompareFunction::GreaterEqual,
                        stencil_front: StencilFaceState::IGNORE,
                        stencil_back: StencilFaceState::IGNORE,
                        stencil_read_mask: 0,
                        stencil_write_mask: 0,
                    }),
                    color_targets: Vec::new(),
                    bind_group_layouts: Vec::new(),
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        offset: 0,
                        size: size_of::<DepthPush>() as u32,
                    }],
                    label: Some("pipe_depth_prepass_reverse_z_msaa"),
                })?,
            )
        } else {
            None
        };

        // 2. Shadow Atlas Depth pipeline (standard [0, 1] LessEqual for CSM & spotlight shadow tiles)
        let pipe_shadow_depth = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_depth_vert,
            fragment_shader: Some(sh_depth_frag),
            vertex_layout: vertex_layout_3d.clone(),
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
                count: SampleCount::Count1,
                alpha_to_coverage: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_test_enabled: true,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil_front: StencilFaceState::IGNORE,
                stencil_back: StencilFaceState::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            color_targets: Vec::new(),
            bind_group_layouts: Vec::new(),
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::VERTEX,
                offset: 0,
                size: size_of::<DepthPush>() as u32,
            }],
            label: Some("pipe_shadow_depth_less_equal"),
        })?;

        // 3. Octahedral unwrap pipeline (writes radial distance to shadow atlas tile)
        let pipe_oct_unwrap = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_fullscreen_vert,
            fragment_shader: Some(sh_oct_unwrap_frag),
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
            multisample: MultisampleState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_test_enabled: true,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil_front: StencilFaceState::IGNORE,
                stencil_back: StencilFaceState::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            color_targets: Vec::new(),
            bind_group_layouts: vec![bgl_oct_unwrap],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                offset: 0,
                size: size_of::<OctUnwrapPush>() as u32,
            }],
            label: Some("pipe_oct_unwrap"),
        })?;

        // 4. SSAO pipeline
        let pipe_ssao = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_fullscreen_vert,
            fragment_shader: Some(sh_ssao_frag),
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
            multisample: MultisampleState::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![bgl_ssao],
            push_constant_ranges: Vec::new(),
            label: Some("pipe_ssao"),
        })?;

        // 5. Forward Opaque PBR pipeline (Reverse-Z GreaterEqual against prepass, MSAA configured)
        let pipe_forward = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_forward_vert,
            fragment_shader: Some(sh_forward_frag),
            vertex_layout: vertex_layout_3d,
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: CullMode::None,
                front_face: FrontFace::CounterClockwise,
                polygon_mode: PolygonMode::Fill,
                depth_bias: 4.0,
                slope_scaled_depth_bias: 1.0,
                depth_bias_clamp: 0.0,
                depth_clamp: false,
            },
            multisample: MultisampleState {
                count: config.msaa_samples,
                alpha_to_coverage: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_test_enabled: true,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil_front: StencilFaceState::IGNORE,
                stencil_back: StencilFaceState::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![bgl_forward_scene, bgl_forward_lights],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                offset: 0,
                size: size_of::<DrawPush>() as u32,
            }],
            label: Some("pipe_forward_pbr"),
        })?;

        // 6. Bloom downsample & upsample pipelines
        let pipe_bloom_down = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_fullscreen_vert,
            fragment_shader: Some(sh_bloom_frag),
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
            multisample: MultisampleState::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![bgl_bloom],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                offset: 0,
                size: size_of::<BloomPush>() as u32,
            }],
            label: Some("pipe_bloom_down"),
        })?;

        let pipe_bloom_up = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_fullscreen_vert,
            fragment_shader: Some(sh_bloom_frag),
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
            multisample: MultisampleState::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOp::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOp::Add,
                    },
                }),
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![bgl_bloom],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                offset: 0,
                size: size_of::<BloomPush>() as u32,
            }],
            label: Some("pipe_bloom_up"),
        })?;

        // 7. Tonemap composite pipeline
        let pipe_tonemap = device.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: sh_fullscreen_vert,
            fragment_shader: Some(sh_tonemap_frag),
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
            multisample: MultisampleState::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: config.ldr_format,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![bgl_tonemap],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                offset: 0,
                size: size_of::<TonemapPush>() as u32,
            }],
            label: Some("pipe_tonemap"),
        })?;

        // Samplers
        let sampler_linear = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            label: Some("forward_sampler_linear"),
            ..Default::default()
        })?;
        let sampler_nearest = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            label: Some("forward_sampler_nearest"),
            ..Default::default()
        })?;

        // Default 1x1 white and black textures
        let tex_white_1x1 = device.create_texture(
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
                label: Some("forward_white_1x1"),
            },
            Some(&[255, 255, 255, 255]),
        )?;
        let tex_black_1x1 = device.create_texture(
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
                label: Some("forward_black_1x1"),
            },
            Some(&[0, 0, 0, 255]),
        )?;

        let tex_depth_prepass = device.create_texture(
            &TextureDescriptor {
                width: config.width.max(1),
                height: config.height.max(1),
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT
                    | TextureUsage::SAMPLED
                    | TextureUsage::COPY_SRC,
                dimension: TextureDimension::D2,
                label: Some("forward_depth_prepass_1x"),
            },
            None,
        )?;

        let atlas_size = config.shadow_atlas_size.max(256);
        let tex_shadow_atlas = device.create_texture(
            &TextureDescriptor {
                width: atlas_size,
                height: atlas_size,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT
                    | TextureUsage::SAMPLED
                    | TextureUsage::COPY_SRC,
                dimension: TextureDimension::D2,
                label: Some("forward_shadow_atlas"),
            },
            None,
        )?;

        let tex_cube_strip = device.create_texture(
            &TextureDescriptor {
                width: CUBE_STRIP_WIDTH,
                height: CUBE_STRIP_HEIGHT,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Depth32Float,
                usage: TextureUsage::DEPTH_STENCIL_ATTACHMENT | TextureUsage::SAMPLED,
                dimension: TextureDimension::D2,
                label: Some("forward_point_shadow_cube_strip"),
            },
            None,
        )?;

        // Upload reference 21x21 DFG + Charlie sheen LUT
        let lut_packed = dfg_lut_packed();
        let lut_bytes: &[u8] = korppi_core::pod::cast_slice(&lut_packed);
        let buf_dfg_lut = device.create_buffer(
            &BufferDescriptor {
                size: lut_bytes.len() as u64,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuOnly,
                label: Some("forward_dfg_lut_ssbo"),
            },
            Some(lut_bytes),
        )?;

        let mut buf_scene_ubo = [BufferHandle::default(); FRAMES_IN_FLIGHT];
        let mut buf_ssao_ubo = [BufferHandle::default(); FRAMES_IN_FLIGHT];
        for slot in 0..FRAMES_IN_FLIGHT {
            buf_scene_ubo[slot] = device.create_buffer(
                &BufferDescriptor {
                    size: size_of::<SceneUniformsGpu>() as u64,
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("forward_scene_ubo"),
                },
                None,
            )?;
            buf_ssao_ubo[slot] = device.create_buffer(
                &BufferDescriptor {
                    size: size_of::<SsaoParamsGpu>() as u64,
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("forward_ssao_ubo"),
                },
                None,
            )?;
        }

        let raw_kernel = generate_ssao_kernel(64);
        let mut ssao_kernel = [[0.0f32; 4]; 64];
        for (i, s) in raw_kernel.into_iter().enumerate().take(64) {
            ssao_kernel[i] = [s.x, s.y, s.z, 0.0];
        }

        let bg_oct_unwrap = device.create_bind_group(&BindGroupDescriptor {
            layout: bgl_oct_unwrap,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    texture: TextureBinding::Texture(tex_cube_strip),
                    sampler: sampler_nearest,
                    layout: ImageLayout::DepthStencilReadOnly,
                },
            }],
            label: Some("bg_oct_unwrap"),
        })?;

        Ok(Self {
            config,
            pool: TransientResourcePool::new(),
            gpu_culler,
            sh_depth_vert,
            sh_depth_frag,
            sh_fullscreen_vert,
            sh_oct_unwrap_frag,
            sh_ssao_frag,
            sh_bloom_frag,
            sh_tonemap_frag,
            sh_forward_vert,
            sh_forward_frag,
            bgl_oct_unwrap,
            bgl_ssao,
            bgl_forward_scene,
            bgl_forward_lights,
            bgl_bloom,
            bgl_tonemap,
            pipe_depth_prepass,
            pipe_depth_prepass_msaa,
            pipe_shadow_depth,
            pipe_oct_unwrap,
            pipe_ssao,
            pipe_forward,
            pipe_bloom_down,
            pipe_bloom_up,
            pipe_tonemap,
            sampler_linear,
            sampler_nearest,
            tex_white_1x1,
            tex_black_1x1,
            tex_depth_prepass,
            depth_1x_written: false,
            tex_shadow_atlas,
            tex_cube_strip,
            buf_dfg_lut,
            buf_scene_ubo,
            buf_ssao_ubo,
            ssao_kernel,
            bg_oct_unwrap,
            bg_ssao: [None; FRAMES_IN_FLIGHT],
            bg_forward_scene: [None; FRAMES_IN_FLIGHT],
            bg_forward_lights: [None; FRAMES_IN_FLIGHT],
            bg_bloom_sources: Vec::with_capacity(16),
            bg_tonemap_sources: Vec::with_capacity(8),
            scratch_froxels: Vec::new(),
            scratch_point_lights: Vec::new(),
            scratch_spot_lights: Vec::new(),
            scratch_cascades: Vec::with_capacity(4),
        })
    }

    /// Reads back the 1× Reverse-Z depth prepass buffer as row-major `f32` depth values in `[0.0, 1.0]`.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    pub fn read_depth_prepass(&self, device: &dyn GraphicsDevice) -> Result<Vec<f32>, RenderError> {
        if !self.depth_1x_written {
            let count = (self.config.width.max(1) as usize) * (self.config.height.max(1) as usize);
            return Ok(vec![0.0; count]);
        }
        let bytes = device.read_texture(self.tex_depth_prepass, 0)?;
        let mut depths = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            depths.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(depths)
    }

    /// Records the full frame graph for `scene` into `recorder` and submits + waits on `device`.
    pub fn render_scene(
        &mut self,
        device: &dyn GraphicsDevice,
        frame_index: korppi_rhi::FrameIndex,
        scene: &ForwardScene,
        target: TextureHandle,
    ) -> Result<(), RenderError> {
        let mut recorder = device.create_command_recorder()?;
        recorder.begin()?;
        self.record_scene(device, recorder.as_mut(), frame_index, scene, target)?;
        let cmd = recorder.finish()?;
        device.submit_and_wait(cmd)?;
        Ok(())
    }

    /// Records the full Clustered Forward Pipeline for `scene` into `recorder` for `frame_index`
    /// with zero steady-state heap allocations (`RND-01`, `RND-11`).
    pub fn record_scene(
        &mut self,
        device: &dyn GraphicsDevice,
        recorder: &mut dyn CommandBufferRecorder,
        frame_index: korppi_rhi::FrameIndex,
        scene: &ForwardScene,
        target: TextureHandle,
    ) -> Result<(), RenderError> {
        let slot = frame_index.index() % FRAMES_IN_FLIGHT;
        let width = self.config.width.max(1);
        let height = self.config.height.max(1);
        let atlas_size = self.config.shadow_atlas_size.max(256);
        let tile_size = atlas_size / 4;
        let inv_atlas = 1.0 / (atlas_size as f32);

        let safe_near = if scene.camera.z_near.is_finite() && scene.camera.z_near > 1e-4 {
            scene.camera.z_near
        } else {
            0.1
        };
        let safe_far = if scene.camera.z_far.is_finite() && scene.camera.z_far > safe_near + 1e-3 {
            scene.camera.z_far
        } else {
            safe_near + 100.0
        };

        let view = scene.camera.view_matrix();
        let proj = scene.camera.projection_matrix();
        let view_proj = proj * view;
        let inv_proj = proj.inverse();

        // 1. Build froxels & view-space lights into reusable scratch buffers (0 steady-state allocations)
        let cluster_cfg = ClusterGridConfig::new(width, height, 64, 16, safe_near, safe_far);
        cluster_cfg.build_froxel_aabbs_into(inv_proj, &mut self.scratch_froxels);

        self.scratch_point_lights.clear();
        if self.scratch_point_lights.capacity() < scene.point_lights.len() {
            self.scratch_point_lights
                .reserve(scene.point_lights.len() - self.scratch_point_lights.capacity());
        }
        for pt in &scene.point_lights {
            let pos_vs4 = view * Vec4::new(pt.position.x, pt.position.y, pt.position.z, 1.0);
            self.scratch_point_lights.push(PointLight {
                position_vs: Vec3::new(pos_vs4.x, pos_vs4.y, pos_vs4.z),
                radius: pt.radius,
                color: pt.color,
                intensity: pt.intensity,
            });
        }

        self.scratch_spot_lights.clear();
        if self.scratch_spot_lights.capacity() < scene.spot_lights.len() {
            self.scratch_spot_lights
                .reserve(scene.spot_lights.len() - self.scratch_spot_lights.capacity());
        }
        let view_mat3 = korppi_math::Mat3::from_mat4(view);
        for sp in &scene.spot_lights {
            let pos_vs4 = view * Vec4::new(sp.position.x, sp.position.y, sp.position.z, 1.0);
            let dir_vs = (view_mat3 * sp.direction)
                .try_normalize()
                .unwrap_or(Vec3::NEG_Z);
            self.scratch_spot_lights.push(SpotLight {
                position_vs: Vec3::new(pos_vs4.x, pos_vs4.y, pos_vs4.z),
                direction_vs: dir_vs,
                range: sp.range,
                half_angle: sp.half_angle,
                color: sp.color,
                intensity: sp.intensity,
            });
        }

        // 2. Compute CSM cascades and spotlight/point shadow descriptors
        let mut csm_view_proj = [Mat4::IDENTITY.to_cols_array_2d(); 4];
        let mut csm_atlas_uv = [[0.0f32; 4]; 4];
        let mut csm_split_depths = [safe_far; 4];

        let has_dir_shadows = scene
            .directional_light
            .map(|d| d.casts_shadows)
            .unwrap_or(false);
        if let Some(dir_light) = scene.directional_light {
            if dir_light.casts_shadows {
                let csm_cfg = CascadeConfig {
                    cascade_count: 4,
                    split_lambda: 0.5,
                    near_plane: safe_near,
                    far_plane: safe_far,
                    shadow_map_resolution: tile_size,
                };
                let safe_dir = dir_light.direction.try_normalize().unwrap_or(Vec3::NEG_Y);
                compute_cascades_into(view, proj, safe_dir, &csm_cfg, &mut self.scratch_cascades);
                for (i, c) in self.scratch_cascades.iter().enumerate().take(4) {
                    csm_view_proj[i] = c.view_proj_matrix.to_cols_array_2d();
                    csm_split_depths[i] = c.split_depth;
                    let tx = (i as u32) * tile_size;
                    csm_atlas_uv[i] = [
                        (tx as f32) * inv_atlas,
                        0.0,
                        (tile_size as f32) * inv_atlas,
                        (tile_size as f32) * inv_atlas,
                    ];
                }
            }
        }

        let mut spot_shadow_view_proj = [Mat4::IDENTITY.to_cols_array_2d(); 4];
        let mut spot_shadow_atlas_uv = [[0.0f32; 4]; 4];
        let mut spot_shadow_light_indices = [u32::MAX; 4];
        let mut spot_shadow_count = 0usize;

        for (idx, sp) in scene.spot_lights.iter().enumerate() {
            if sp.casts_shadows && spot_shadow_count < 4 {
                let s = spot_shadow_count;
                let dir = sp.direction.try_normalize().unwrap_or(Vec3::NEG_Y);
                let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
                let sp_view = Mat4::look_at_rh(sp.position, sp.position + dir, up);
                let fov = (sp.half_angle * 2.0).clamp(0.1, 3.0);
                let sp_proj = Mat4::perspective_rh(fov, 1.0, 0.1, sp.range.max(0.2));
                spot_shadow_view_proj[s] = (sp_proj * sp_view).to_cols_array_2d();
                let tx = (s as u32) * tile_size;
                let ty = tile_size;
                spot_shadow_atlas_uv[s] = [
                    (tx as f32) * inv_atlas,
                    (ty as f32) * inv_atlas,
                    (tile_size as f32) * inv_atlas,
                    (tile_size as f32) * inv_atlas,
                ];
                spot_shadow_light_indices[s] = idx as u32;
                spot_shadow_count += 1;
            }
        }

        let mut point_shadow_pos_radius = [[0.0f32; 4]; 4];
        let mut point_shadow_atlas_uv = [[0.0f32; 4]; 4];
        let mut point_shadow_light_indices = [u32::MAX; 4];
        let mut point_shadow_count = 0usize;

        for (idx, pt) in scene.point_lights.iter().enumerate() {
            if pt.casts_shadows && point_shadow_count < 4 {
                let p = point_shadow_count;
                point_shadow_pos_radius[p] =
                    [pt.position.x, pt.position.y, pt.position.z, pt.radius];
                let tx = (p as u32) * tile_size;
                let ty = tile_size * 2;
                point_shadow_atlas_uv[p] = [
                    (tx as f32) * inv_atlas,
                    (ty as f32) * inv_atlas,
                    (tile_size as f32) * inv_atlas,
                    (tile_size as f32) * inv_atlas,
                ];
                point_shadow_light_indices[p] = idx as u32;
                point_shadow_count += 1;
            }
        }

        let (dir_vec, dir_rad) = if let Some(d) = scene.directional_light {
            let dn = d.direction.try_normalize().unwrap_or(Vec3::NEG_Y);
            let rad = d.color * d.illuminance;
            (
                [dn.x, dn.y, dn.z, 1.0],
                [rad.x, rad.y, rad.z, if d.casts_shadows { 1.0 } else { 0.0 }],
            )
        } else {
            ([0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        };

        let log_far_over_near = (safe_far / safe_near.max(1e-4)).ln().max(1e-5);
        let scene_ubo = SceneUniformsGpu {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [
                scene.camera.position.x,
                scene.camera.position.y,
                scene.camera.position.z,
                1.0,
            ],
            ambient_light: [
                scene.ambient_light.x,
                scene.ambient_light.y,
                scene.ambient_light.z,
                0.0,
            ],
            dir_light_dir: dir_vec,
            dir_light_radiance: dir_rad,
            cluster_dims: [
                cluster_cfg.dim_x(),
                cluster_cfg.dim_y(),
                cluster_cfg.dim_z(),
                cluster_cfg.tile_size(),
            ],
            cluster_depth_params: [safe_near, safe_far, 1.0 / log_far_over_near, inv_atlas],
            viewport_and_shadow_counts: [
                1.0 / (width as f32),
                1.0 / (height as f32),
                spot_shadow_count as f32,
                point_shadow_count as f32,
            ],
            csm_view_proj,
            csm_atlas_uv,
            csm_split_depths,
            spot_shadow_view_proj,
            spot_shadow_atlas_uv,
            spot_shadow_light_indices,
            point_shadow_pos_radius,
            point_shadow_atlas_uv,
            point_shadow_light_indices,
        };
        device.write_buffer(
            self.buf_scene_ubo[slot],
            0,
            korppi_core::pod::bytes_of(&scene_ubo),
        )?;

        if let Some(ssao_cfg) = &self.config.ssao {
            let ssao_ubo = SsaoParamsGpu {
                proj: proj.to_cols_array_2d(),
                inv_proj: inv_proj.to_cols_array_2d(),
                params0: [
                    ssao_cfg.sample_radius,
                    ssao_cfg.bias,
                    ssao_cfg.intensity,
                    ssao_cfg.power,
                ],
                params1: [
                    1.0 / (width as f32),
                    1.0 / (height as f32),
                    (ssao_cfg.kernel_size.clamp(1, 64)) as f32,
                    0.0,
                ],
                samples: self.ssao_kernel,
            };
            device.write_buffer(
                self.buf_ssao_ubo[slot],
                0,
                korppi_core::pod::bytes_of(&ssao_ubo),
            )?;
        }

        // 3. Dispatch GPU compute froxel light culling before graphics passes
        let culling_buffers = self.gpu_culler.dispatch_culling_for_frame(
            device,
            recorder,
            frame_index,
            &self.scratch_froxels,
            &self.scratch_point_lights,
            &self.scratch_spot_lights,
            &[],
            &[],
        )?;

        // Emit compute -> fragment SSBO barrier for culled light records & indices
        recorder.pipeline_barrier(
            &PipelineBarrier::new()
                .with_buffer_barrier(BufferMemoryBarrier {
                    buffer: culling_buffers.records_buffer,
                    offset: 0,
                    size: culling_buffers.records_size,
                    src_stages: PipelineStageFlags::COMPUTE_SHADER,
                    src_access: AccessFlags::SHADER_WRITE,
                    dst_stages: PipelineStageFlags::FRAGMENT_SHADER,
                    dst_access: AccessFlags::SHADER_READ,
                })
                .with_buffer_barrier(BufferMemoryBarrier {
                    buffer: culling_buffers.light_indices_buffer,
                    offset: 0,
                    size: culling_buffers.light_indices_size,
                    src_stages: PipelineStageFlags::COMPUTE_SHADER,
                    src_access: AccessFlags::SHADER_WRITE,
                    dst_stages: PipelineStageFlags::FRAGMENT_SHADER,
                    dst_access: AccessFlags::SHADER_READ,
                }),
        );

        // 4. Build and compile the Render Graph for all graphics passes
        self.pool.begin_frame(frame_index);
        let mut graph = RenderGraph::new();

        let rg_shadow_atlas =
            graph.import_texture(self.tex_shadow_atlas, TEX_UNDEFINED, atlas_size, atlas_size);
        let rg_cube_strip = if point_shadow_count > 0 {
            Some(graph.import_texture(
                self.tex_cube_strip,
                TEX_UNDEFINED,
                CUBE_STRIP_WIDTH,
                CUBE_STRIP_HEIGHT,
            ))
        } else {
            None
        };

        let need_1x_depth =
            self.config.msaa_samples == SampleCount::Count1 || self.config.ssao.is_some();
        self.depth_1x_written = need_1x_depth;
        let rg_depth_prepass = if need_1x_depth {
            Some(graph.import_texture_with_final_access(
                self.tex_depth_prepass,
                TEX_UNDEFINED,
                TEX_SHADER_READ_DEPTH,
                width,
                height,
            ))
        } else {
            None
        };
        let rg_target = graph.import_texture(target, TEX_UNDEFINED, width, height);

        let rg_hdr_1x = graph.create_transient_texture(TransientTextureDesc::new(
            "hdr_1x",
            width,
            height,
            TextureFormat::Rgba16Float,
            TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        ));

        let msaa_pair: Option<(RenderGraphTextureHandle, RenderGraphTextureHandle)> =
            if self.config.msaa_samples != SampleCount::Count1 {
                let hdr_msaa = graph.create_transient_texture(
                    TransientTextureDesc::new(
                        "hdr_msaa",
                        width,
                        height,
                        TextureFormat::Rgba16Float,
                        TextureUsage::COLOR_ATTACHMENT,
                    )
                    .with_samples(self.config.msaa_samples),
                );
                let depth_msaa = graph.create_transient_texture(
                    TransientTextureDesc::new(
                        "depth_msaa",
                        width,
                        height,
                        TextureFormat::Depth32Float,
                        TextureUsage::DEPTH_STENCIL_ATTACHMENT,
                    )
                    .with_samples(self.config.msaa_samples),
                );
                Some((hdr_msaa, depth_msaa))
            } else {
                None
            };

        let rg_ssao: Option<RenderGraphTextureHandle> = if self.config.ssao.is_some() {
            Some(graph.create_transient_texture(TransientTextureDesc::new(
                "ssao",
                width,
                height,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
            )))
        } else {
            None
        };

        let bloom_mip_count = self
            .config
            .bloom
            .as_ref()
            .map(|b| (b.mip_levels as usize).clamp(1, MAX_BLOOM_MIPS))
            .unwrap_or(0);
        let mut bloom_mips = [RenderGraphTextureHandle::default(); MAX_BLOOM_MIPS];
        for (i, slot_handle) in bloom_mips.iter_mut().enumerate().take(bloom_mip_count) {
            let mip_w = (width >> (i + 1)).max(1);
            let mip_h = (height >> (i + 1)).max(1);
            *slot_handle = graph.create_transient_texture(TransientTextureDesc::new(
                BLOOM_MIP_TEX_NAMES[i],
                mip_w,
                mip_h,
                TextureFormat::Rgba16Float,
                TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
            ));
        }

        // Pass 1: Shadow Atlas (CSM & Spotlights) + Declarative Point-Light Cube/Unwrap passes
        graph
            .add_pass("shadow_atlas", PassType::Graphics)
            .write_depth(rg_shadow_atlas, LoadOp::Clear, StoreOp::Store, 1.0)
            .mark_as_root();

        if let Some(cube_strip) = rg_cube_strip {
            for p_slot in 0..point_shadow_count {
                graph
                    .add_pass(POINT_SHADOW_CUBE_NAMES[p_slot], PassType::Graphics)
                    .write_depth(cube_strip, LoadOp::Clear, StoreOp::Store, 0.0);
                graph
                    .add_pass(POINT_SHADOW_UNWRAP_NAMES[p_slot], PassType::Graphics)
                    .read_texture(cube_strip, TEX_SHADER_READ_DEPTH)
                    .write_depth(rg_shadow_atlas, LoadOp::Load, StoreOp::Store, 1.0)
                    .mark_as_root();
            }
        }

        // Pass 2: Reverse-Z Depth Prepass matching forward pass sample count
        if let Some((_, depth_msaa)) = msaa_pair {
            graph
                .add_pass("depth_prepass", PassType::Graphics)
                .write_depth(depth_msaa, LoadOp::Clear, StoreOp::Store, 0.0);
            if let Some(depth_1x) = rg_depth_prepass {
                graph
                    .add_pass("depth_prepass_1x", PassType::Graphics)
                    .write_depth(depth_1x, LoadOp::Clear, StoreOp::Store, 0.0);
            }
        } else {
            let depth_1x = rg_depth_prepass.expect("1x depth prepass");
            graph
                .add_pass("depth_prepass", PassType::Graphics)
                .write_depth(depth_1x, LoadOp::Clear, StoreOp::Store, 0.0);
        }

        // Pass 3: Optional SSAO
        if let (Some(ssao_tex), Some(depth_1x)) = (rg_ssao, rg_depth_prepass) {
            graph
                .add_pass("ssao", PassType::Graphics)
                .read_texture(depth_1x, TEX_SHADER_READ_DEPTH)
                .write_color(
                    ssao_tex,
                    0,
                    LoadOp::Clear,
                    StoreOp::Store,
                    [1.0, 1.0, 1.0, 1.0],
                );
        }

        // Pass 4: Forward Opaque PBR Pass (loads prepass depth with read-only GreaterEqual)
        {
            let mut pass = graph
                .add_pass("forward_opaque", PassType::Graphics)
                .read_texture(rg_shadow_atlas, TEX_SHADER_READ_DEPTH);
            if let Some(ssao_tex) = rg_ssao {
                pass = pass.read_texture(ssao_tex, TEX_SHADER_READ);
            }
            if let Some((hdr_msaa, depth_msaa)) = msaa_pair {
                pass.write_color_msaa(
                    hdr_msaa,
                    rg_hdr_1x,
                    0,
                    LoadOp::Clear,
                    StoreOp::DontCare,
                    scene.clear_color,
                )
                .write_depth(depth_msaa, LoadOp::Load, StoreOp::DontCare, 0.0);
            } else {
                let depth_1x = rg_depth_prepass.expect("1x depth prepass");
                pass.write_color(
                    rg_hdr_1x,
                    0,
                    LoadOp::Clear,
                    StoreOp::Store,
                    scene.clear_color,
                )
                .write_depth(depth_1x, LoadOp::Load, StoreOp::Store, 0.0);
            }
        }

        // Pass 5: Optional Progressive Bloom Downsample & Upsample Pyramid
        for i in 0..bloom_mip_count {
            let src = if i == 0 { rg_hdr_1x } else { bloom_mips[i - 1] };
            let dst = bloom_mips[i];
            graph
                .add_pass(BLOOM_DOWN_PASS_NAMES[i], PassType::Graphics)
                .read_texture(src, TEX_SHADER_READ)
                .write_color(dst, 0, LoadOp::Clear, StoreOp::Store, [0.0, 0.0, 0.0, 1.0]);
        }
        if bloom_mip_count > 1 {
            for (step, &pass_name) in BLOOM_UP_PASS_NAMES
                .iter()
                .enumerate()
                .take(bloom_mip_count - 1)
            {
                let src_mip = bloom_mip_count - 1 - step;
                let dst_mip = src_mip - 1;
                graph
                    .add_pass(pass_name, PassType::Graphics)
                    .read_texture(bloom_mips[src_mip], TEX_SHADER_READ)
                    .write_color(
                        bloom_mips[dst_mip],
                        0,
                        LoadOp::Load,
                        StoreOp::Store,
                        [0.0, 0.0, 0.0, 1.0],
                    );
            }
        }

        // Pass 6: Final Tonemap & Composite into `target`
        {
            let mut pass = graph
                .add_pass("tonemap", PassType::Graphics)
                .read_texture(rg_hdr_1x, TEX_SHADER_READ);
            if bloom_mip_count > 0 {
                pass = pass.read_texture(bloom_mips[0], TEX_SHADER_READ);
            }
            pass.write_color(
                rg_target,
                0,
                LoadOp::Clear,
                StoreOp::Store,
                [0.0, 0.0, 0.0, 1.0],
            )
            .mark_as_root();
        }

        let mut point_light_indices_for_unwrap: [usize; 4] = [0; 4];
        for (p, &pt_idx) in point_shadow_light_indices
            .iter()
            .enumerate()
            .take(point_shadow_count)
        {
            point_light_indices_for_unwrap[p] = pt_idx as usize;
        }

        let mut compiled = graph.compile_with_device(device, &mut self.pool)?;

        // Resolve physical handles for bind group caching
        let phys_hdr_1x = compiled.resolve_texture(rg_hdr_1x);
        let phys_ssao = rg_ssao
            .map(|h| compiled.resolve_texture(h))
            .unwrap_or(self.tex_white_1x1);

        let mut phys_bloom_mips = [TextureHandle::default(); MAX_BLOOM_MIPS];
        for i in 0..bloom_mip_count {
            phys_bloom_mips[i] = compiled.resolve_texture(bloom_mips[i]);
        }

        let bg_ssao = if rg_ssao.is_some() {
            Some(self.ensure_ssao_bind_group(device, slot, self.tex_depth_prepass)?)
        } else {
            None
        };
        let bg_forward_scene = self.ensure_forward_scene_bind_group(device, slot, phys_ssao)?;
        let bg_forward_lights =
            self.ensure_forward_lights_bind_group(device, slot, &culling_buffers)?;

        let mut bg_bloom_down = [BindGroupHandle::default(); MAX_BLOOM_MIPS];
        for i in 0..bloom_mip_count {
            let src_phys = if i == 0 {
                phys_hdr_1x
            } else {
                phys_bloom_mips[i - 1]
            };
            bg_bloom_down[i] = self.ensure_bloom_bind_group(device, src_phys)?;
        }
        let mut bg_bloom_up = [BindGroupHandle::default(); MAX_BLOOM_MIPS - 1];
        if bloom_mip_count > 1 {
            for (step, bg_slot) in bg_bloom_up.iter_mut().enumerate().take(bloom_mip_count - 1) {
                let src_mip = bloom_mip_count - 1 - step;
                *bg_slot = self.ensure_bloom_bind_group(device, phys_bloom_mips[src_mip])?;
            }
        }

        let bg_tonemap = self.ensure_tonemap_bind_group(
            device,
            phys_hdr_1x,
            if bloom_mip_count > 0 {
                phys_bloom_mips[0]
            } else {
                self.tex_black_1x1
            },
        )?;

        compiled.execute_with(recorder, |pass_name, _ctx: &PassContext<'_>, rec| {
            match pass_name {
                "shadow_atlas" => {
                    // 1a. Render CSM cascades (tiles 0..4 on row 0)
                    if has_dir_shadows {
                        rec.bind_pipeline(self.pipe_shadow_depth);
                        for (c_idx, vp) in csm_view_proj.iter().enumerate() {
                            let tx = (c_idx as u32) * tile_size;
                            rec.set_viewport(
                                tx as f32,
                                0.0,
                                tile_size as f32,
                                tile_size as f32,
                                0.0,
                                1.0,
                            );
                            rec.set_scissor(tx as i32, 0, tile_size, tile_size);
                            let vp_mat = Mat4::from_cols_array_2d(vp);
                            Self::record_shadow_draws(rec, &scene.draws, vp_mat)?;
                        }
                    }

                    // 1b. Render spotlight perspective shadows (tiles 0..4 on row 1)
                    if spot_shadow_count > 0 {
                        rec.bind_pipeline(self.pipe_shadow_depth);
                        for (s_idx, vp) in spot_shadow_view_proj
                            .iter()
                            .enumerate()
                            .take(spot_shadow_count)
                        {
                            let tx = (s_idx as u32) * tile_size;
                            let ty = tile_size;
                            rec.set_viewport(
                                tx as f32,
                                ty as f32,
                                tile_size as f32,
                                tile_size as f32,
                                0.0,
                                1.0,
                            );
                            rec.set_scissor(tx as i32, ty as i32, tile_size, tile_size);
                            let vp_mat = Mat4::from_cols_array_2d(vp);
                            Self::record_shadow_draws(rec, &scene.draws, vp_mat)?;
                        }
                    }
                    Ok(())
                }
                "depth_prepass" | "depth_prepass_1x" => {
                    rec.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    rec.set_scissor(0, 0, width, height);
                    let pipe = if pass_name == "depth_prepass" {
                        self.pipe_depth_prepass_msaa
                            .unwrap_or(self.pipe_depth_prepass)
                    } else {
                        self.pipe_depth_prepass
                    };
                    rec.bind_pipeline(pipe);
                    for draw in &scene.draws {
                        if draw.mesh.index_count == 0 {
                            continue;
                        }
                        let mvp = view_proj * draw.transform;
                        let push = DepthPush {
                            mvp: mvp.to_cols_array_2d(),
                        };
                        rec.push_constants(0, korppi_core::pod::bytes_of(&push))?;
                        let vbs = [draw.mesh.vertex_buffer];
                        let offsets = [0u64];
                        rec.bind_vertex_buffers(0, &vbs, &offsets);
                        rec.bind_index_buffer(draw.mesh.index_buffer, 0, IndexType::Uint32);
                        rec.draw_indexed(draw.mesh.index_count, 1, 0, 0, 0);
                    }
                    Ok(())
                }
                "ssao" => {
                    rec.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    rec.set_scissor(0, 0, width, height);
                    rec.bind_pipeline(self.pipe_ssao);
                    rec.set_bind_group(0, bg_ssao.expect("ssao bind group"), &[]);
                    rec.draw(3, 1, 0, 0);
                    Ok(())
                }
                "forward_opaque" => {
                    rec.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    rec.set_scissor(0, 0, width, height);
                    rec.bind_pipeline(self.pipe_forward);
                    rec.set_bind_group(0, bg_forward_scene, &[]);
                    rec.set_bind_group(1, bg_forward_lights, &[]);

                    for draw in &scene.draws {
                        if draw.mesh.index_count == 0 {
                            continue;
                        }
                        let m = &draw.material;
                        let (cc_r, cc_w, has_cc) = match m.pbr.clear_coat {
                            Some(cc) => (cc.roughness, cc.weight, 1.0),
                            None => (0.0, 0.0, 0.0),
                        };
                        let sheen = match m.pbr.sheen {
                            Some(sh) => {
                                [sh.color.x, sh.color.y, sh.color.z, sh.roughness.max(0.045)]
                            }
                            None => [0.0, 0.0, 0.0, 0.0],
                        };
                        let push = DrawPush {
                            model: draw.transform.to_cols_array_2d(),
                            albedo_roughness: [m.albedo.x, m.albedo.y, m.albedo.z, m.pbr.roughness],
                            emissive_metallic: [
                                m.emissive.x,
                                m.emissive.y,
                                m.emissive.z,
                                m.pbr.metallic,
                            ],
                            reflectance_clearcoat: [m.pbr.reflectance, cc_r, cc_w, has_cc],
                            sheen,
                        };
                        rec.push_constants(0, korppi_core::pod::bytes_of(&push))?;
                        let vbs = [draw.mesh.vertex_buffer];
                        let offsets = [0u64];
                        rec.bind_vertex_buffers(0, &vbs, &offsets);
                        rec.bind_index_buffer(draw.mesh.index_buffer, 0, IndexType::Uint32);
                        rec.draw_indexed(draw.mesh.index_count, 1, 0, 0, 0);
                    }
                    Ok(())
                }
                "tonemap" => {
                    rec.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    rec.set_scissor(0, 0, width, height);
                    rec.bind_pipeline(self.pipe_tonemap);
                    rec.set_bind_group(0, bg_tonemap, &[]);
                    let op_code = match self.config.tonemap.operator {
                        TonemapOperator::AgX => 0u32,
                        TonemapOperator::Aces => 1u32,
                        TonemapOperator::Reinhard => 2u32,
                        TonemapOperator::Filmic => 3u32,
                    };
                    let bloom_intensity = self
                        .config
                        .bloom
                        .as_ref()
                        .map(|b| b.intensity)
                        .unwrap_or(0.0);
                    let srgb_target = u32::from(self.config.ldr_format.is_srgb());
                    let push = TonemapPush {
                        params: [
                            self.config.tonemap.exposure,
                            self.config.tonemap.gamma,
                            self.config.tonemap.white_point,
                            bloom_intensity,
                        ],
                        flags: [op_code, u32::from(bloom_mip_count > 0), srgb_target, 0],
                    };
                    rec.push_constants(0, korppi_core::pod::bytes_of(&push))?;
                    rec.draw(3, 1, 0, 0);
                    Ok(())
                }
                other => {
                    if let Some(p_slot) = POINT_SHADOW_CUBE_NAMES.iter().position(|&n| n == other) {
                        let light_idx = point_light_indices_for_unwrap[p_slot];
                        let pt = &scene.point_lights[light_idx];
                        let pt_near = 0.1f32;
                        let pt_far = pt.radius.max(0.2);
                        let face_proj = ReverseZ::perspective_rh(
                            std::f32::consts::FRAC_PI_2,
                            1.0,
                            pt_near,
                            pt_far,
                        );
                        rec.bind_pipeline(self.pipe_depth_prepass);
                        for (face_idx, &(forward, up)) in CUBE_FACE_DIRS.iter().enumerate() {
                            let fx = (face_idx as u32) * CUBE_FACE_SIZE;
                            rec.set_viewport(
                                fx as f32,
                                0.0,
                                CUBE_FACE_SIZE as f32,
                                CUBE_FACE_SIZE as f32,
                                0.0,
                                1.0,
                            );
                            rec.set_scissor(fx as i32, 0, CUBE_FACE_SIZE, CUBE_FACE_SIZE);
                            let face_view =
                                Mat4::look_at_rh(pt.position, pt.position + forward, up);
                            let face_vp = face_proj * face_view;
                            Self::record_shadow_draws(rec, &scene.draws, face_vp)?;
                        }
                        return Ok(());
                    }
                    if let Some(p_slot) = POINT_SHADOW_UNWRAP_NAMES.iter().position(|&n| n == other)
                    {
                        let light_idx = point_light_indices_for_unwrap[p_slot];
                        let pt = &scene.point_lights[light_idx];
                        let pt_near = 0.1f32;
                        let pt_far = pt.radius.max(0.2);
                        let tx = (p_slot as u32) * tile_size;
                        let ty = tile_size * 2;
                        rec.set_viewport(
                            tx as f32,
                            ty as f32,
                            tile_size as f32,
                            tile_size as f32,
                            0.0,
                            1.0,
                        );
                        rec.set_scissor(tx as i32, ty as i32, tile_size, tile_size);
                        rec.bind_pipeline(self.pipe_oct_unwrap);
                        rec.set_bind_group(0, self.bg_oct_unwrap, &[]);
                        let push = OctUnwrapPush {
                            near_far_offset: [pt_near, pt_far, tx as f32, ty as f32],
                            inv_tile_size: [
                                1.0 / (tile_size as f32),
                                1.0 / (tile_size as f32),
                                0.0,
                                0.0,
                            ],
                        };
                        rec.push_constants(0, korppi_core::pod::bytes_of(&push))?;
                        rec.draw(3, 1, 0, 0);
                        return Ok(());
                    }
                    if let Some(i) = BLOOM_DOWN_PASS_NAMES.iter().position(|&n| n == other) {
                        let bloom_cfg = self.config.bloom.as_ref().expect("bloom config");
                        let src_w = (width >> i).max(1);
                        let src_h = (height >> i).max(1);
                        let dst_w = (width >> (i + 1)).max(1);
                        let dst_h = (height >> (i + 1)).max(1);
                        rec.set_viewport(0.0, 0.0, dst_w as f32, dst_h as f32, 0.0, 1.0);
                        rec.set_scissor(0, 0, dst_w, dst_h);
                        rec.bind_pipeline(self.pipe_bloom_down);
                        rec.set_bind_group(0, bg_bloom_down[i], &[]);
                        let threshold = if i == 0 {
                            [bloom_cfg.threshold, bloom_cfg.soft_knee, 1.0, 0.0]
                        } else {
                            [0.0, 0.0, 0.0, 0.0]
                        };
                        let push = BloomPush {
                            texel_radius_mode: [
                                1.0 / (src_w as f32),
                                1.0 / (src_h as f32),
                                bloom_cfg.filter_radius,
                                0.0,
                            ],
                            threshold_params: threshold,
                        };
                        rec.push_constants(0, korppi_core::pod::bytes_of(&push))?;
                        rec.draw(3, 1, 0, 0);
                        return Ok(());
                    }
                    if let Some(step) = BLOOM_UP_PASS_NAMES.iter().position(|&n| n == other) {
                        let bloom_cfg = self.config.bloom.as_ref().expect("bloom config");
                        let src_mip = bloom_mip_count - 1 - step;
                        let dst_mip = src_mip - 1;
                        let src_w = (width >> (src_mip + 1)).max(1);
                        let src_h = (height >> (src_mip + 1)).max(1);
                        let dst_w = (width >> (dst_mip + 1)).max(1);
                        let dst_h = (height >> (dst_mip + 1)).max(1);
                        rec.set_viewport(0.0, 0.0, dst_w as f32, dst_h as f32, 0.0, 1.0);
                        rec.set_scissor(0, 0, dst_w, dst_h);
                        rec.bind_pipeline(self.pipe_bloom_up);
                        rec.set_bind_group(0, bg_bloom_up[step], &[]);
                        let push = BloomPush {
                            texel_radius_mode: [
                                1.0 / (src_w as f32),
                                1.0 / (src_h as f32),
                                bloom_cfg.filter_radius,
                                1.0,
                            ],
                            threshold_params: [0.0, 0.0, 0.0, 0.0],
                        };
                        rec.push_constants(0, korppi_core::pod::bytes_of(&push))?;
                        rec.draw(3, 1, 0, 0);
                        return Ok(());
                    }
                    Ok(())
                }
            }
        })?;

        Ok(())
    }

    fn record_shadow_draws(
        rec: &mut dyn CommandBufferRecorder,
        draws: &[crate::pipeline::scene::MeshDraw],
        view_proj: Mat4,
    ) -> Result<(), RhiError> {
        for draw in draws {
            if !draw.casts_shadows || draw.mesh.index_count == 0 {
                continue;
            }
            let mvp = view_proj * draw.transform;
            let push = DepthPush {
                mvp: mvp.to_cols_array_2d(),
            };
            rec.push_constants(0, korppi_core::pod::bytes_of(&push))?;
            let vbs = [draw.mesh.vertex_buffer];
            let offsets = [0u64];
            rec.bind_vertex_buffers(0, &vbs, &offsets);
            rec.bind_index_buffer(draw.mesh.index_buffer, 0, IndexType::Uint32);
            rec.draw_indexed(draw.mesh.index_count, 1, 0, 0, 0);
        }
        Ok(())
    }

    fn ensure_ssao_bind_group(
        &mut self,
        device: &dyn GraphicsDevice,
        slot: usize,
        depth_tex: TextureHandle,
    ) -> Result<BindGroupHandle, RhiError> {
        if let Some((cached_tex, bg)) = self.bg_ssao[slot] {
            if cached_tex == depth_tex {
                return Ok(bg);
            }
            device.destroy_bind_group(bg)?;
        }
        let bg = device.create_bind_group(&BindGroupDescriptor {
            layout: self.bgl_ssao,
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: self.buf_ssao_ubo[slot],
                        offset: 0,
                        size: size_of::<SsaoParamsGpu>() as u64,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        texture: TextureBinding::Texture(depth_tex),
                        sampler: self.sampler_nearest,
                        layout: ImageLayout::DepthStencilReadOnly,
                    },
                },
            ],
            label: Some("bg_ssao"),
        })?;
        self.bg_ssao[slot] = Some((depth_tex, bg));
        Ok(bg)
    }

    fn ensure_forward_scene_bind_group(
        &mut self,
        device: &dyn GraphicsDevice,
        slot: usize,
        ssao_tex: TextureHandle,
    ) -> Result<BindGroupHandle, RhiError> {
        if let Some((cached_ssao, bg)) = self.bg_forward_scene[slot] {
            if cached_ssao == ssao_tex {
                return Ok(bg);
            }
            device.destroy_bind_group(bg)?;
        }
        let bg = device.create_bind_group(&BindGroupDescriptor {
            layout: self.bgl_forward_scene,
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: self.buf_scene_ubo[slot],
                        offset: 0,
                        size: size_of::<SceneUniformsGpu>() as u64,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        texture: TextureBinding::Texture(self.tex_shadow_atlas),
                        sampler: self.sampler_nearest,
                        layout: ImageLayout::DepthStencilReadOnly,
                    },
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::SampledTexture {
                        texture: TextureBinding::Texture(ssao_tex),
                        sampler: self.sampler_linear,
                        layout: ImageLayout::ShaderReadOnly,
                    },
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer {
                        buffer: self.buf_dfg_lut,
                        offset: 0,
                        size: (441 * 16) as u64,
                    },
                },
            ],
            label: Some("bg_forward_scene"),
        })?;
        self.bg_forward_scene[slot] = Some((ssao_tex, bg));
        Ok(bg)
    }

    fn ensure_forward_lights_bind_group(
        &mut self,
        device: &dyn GraphicsDevice,
        slot: usize,
        culling: &GpuCullingBuffers,
    ) -> Result<BindGroupHandle, RhiError> {
        let key = [
            culling.points_buffer,
            culling.spots_buffer,
            culling.records_buffer,
            culling.light_indices_buffer,
        ];
        if let Some((cached_key, bg)) = self.bg_forward_lights[slot] {
            if cached_key == key {
                return Ok(bg);
            }
            device.destroy_bind_group(bg)?;
        }
        let bg = device.create_bind_group(&BindGroupDescriptor {
            layout: self.bgl_forward_lights,
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: culling.points_buffer,
                        offset: 0,
                        size: culling.points_size,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer {
                        buffer: culling.spots_buffer,
                        offset: 0,
                        size: culling.spots_size,
                    },
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer {
                        buffer: culling.records_buffer,
                        offset: 0,
                        size: culling.records_size,
                    },
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer {
                        buffer: culling.light_indices_buffer,
                        offset: 0,
                        size: culling.light_indices_size,
                    },
                },
            ],
            label: Some("bg_forward_lights"),
        })?;
        self.bg_forward_lights[slot] = Some((key, bg));
        Ok(bg)
    }

    fn ensure_bloom_bind_group(
        &mut self,
        device: &dyn GraphicsDevice,
        src_tex: TextureHandle,
    ) -> Result<BindGroupHandle, RhiError> {
        for &(cached_tex, bg) in &self.bg_bloom_sources {
            if cached_tex == src_tex {
                return Ok(bg);
            }
        }
        let bg = device.create_bind_group(&BindGroupDescriptor {
            layout: self.bgl_bloom,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    texture: TextureBinding::Texture(src_tex),
                    sampler: self.sampler_linear,
                    layout: ImageLayout::ShaderReadOnly,
                },
            }],
            label: Some("bg_bloom_src"),
        })?;
        self.bg_bloom_sources.push((src_tex, bg));
        Ok(bg)
    }

    fn ensure_tonemap_bind_group(
        &mut self,
        device: &dyn GraphicsDevice,
        hdr_tex: TextureHandle,
        bloom_tex: TextureHandle,
    ) -> Result<BindGroupHandle, RhiError> {
        let key = (hdr_tex, bloom_tex);
        for &(cached_key, bg) in &self.bg_tonemap_sources {
            if cached_key == key {
                return Ok(bg);
            }
        }
        let bg = device.create_bind_group(&BindGroupDescriptor {
            layout: self.bgl_tonemap,
            entries: vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        texture: TextureBinding::Texture(hdr_tex),
                        sampler: self.sampler_linear,
                        layout: ImageLayout::ShaderReadOnly,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        texture: TextureBinding::Texture(bloom_tex),
                        sampler: self.sampler_linear,
                        layout: ImageLayout::ShaderReadOnly,
                    },
                },
            ],
            label: Some("bg_tonemap"),
        })?;
        self.bg_tonemap_sources.push((key, bg));
        Ok(bg)
    }

    /// Destroys all GPU pipelines, shaders, bind groups, buffers, and pooled transient resources.
    pub fn destroy(&mut self, device: &dyn GraphicsDevice) -> Result<(), RenderError> {
        self.gpu_culler.destroy(device)?;
        self.pool.destroy(device)?;

        device.destroy_bind_group(self.bg_oct_unwrap)?;
        for slot in 0..FRAMES_IN_FLIGHT {
            if let Some((_, bg)) = self.bg_ssao[slot].take() {
                device.destroy_bind_group(bg)?;
            }
            if let Some((_, bg)) = self.bg_forward_scene[slot].take() {
                device.destroy_bind_group(bg)?;
            }
            if let Some((_, bg)) = self.bg_forward_lights[slot].take() {
                device.destroy_bind_group(bg)?;
            }
            device.destroy_buffer(self.buf_scene_ubo[slot])?;
            device.destroy_buffer(self.buf_ssao_ubo[slot])?;
        }
        for (_, bg) in self.bg_bloom_sources.drain(..) {
            device.destroy_bind_group(bg)?;
        }
        for (_, bg) in self.bg_tonemap_sources.drain(..) {
            device.destroy_bind_group(bg)?;
        }

        device.destroy_buffer(self.buf_dfg_lut)?;
        device.destroy_texture(self.tex_cube_strip)?;
        device.destroy_texture(self.tex_shadow_atlas)?;
        device.destroy_texture(self.tex_depth_prepass)?;
        device.destroy_texture(self.tex_black_1x1)?;
        device.destroy_texture(self.tex_white_1x1)?;
        device.destroy_sampler(self.sampler_nearest)?;
        device.destroy_sampler(self.sampler_linear)?;

        device.destroy_pipeline(self.pipe_tonemap)?;
        device.destroy_pipeline(self.pipe_bloom_up)?;
        device.destroy_pipeline(self.pipe_bloom_down)?;
        device.destroy_pipeline(self.pipe_forward)?;
        device.destroy_pipeline(self.pipe_ssao)?;
        device.destroy_pipeline(self.pipe_oct_unwrap)?;
        device.destroy_pipeline(self.pipe_shadow_depth)?;
        if let Some(p) = self.pipe_depth_prepass_msaa.take() {
            device.destroy_pipeline(p)?;
        }
        device.destroy_pipeline(self.pipe_depth_prepass)?;

        device.destroy_bind_group_layout(self.bgl_tonemap)?;
        device.destroy_bind_group_layout(self.bgl_bloom)?;
        device.destroy_bind_group_layout(self.bgl_forward_lights)?;
        device.destroy_bind_group_layout(self.bgl_forward_scene)?;
        device.destroy_bind_group_layout(self.bgl_ssao)?;
        device.destroy_bind_group_layout(self.bgl_oct_unwrap)?;

        device.destroy_shader(self.sh_forward_frag)?;
        device.destroy_shader(self.sh_forward_vert)?;
        device.destroy_shader(self.sh_tonemap_frag)?;
        device.destroy_shader(self.sh_bloom_frag)?;
        device.destroy_shader(self.sh_ssao_frag)?;
        device.destroy_shader(self.sh_oct_unwrap_frag)?;
        device.destroy_shader(self.sh_fullscreen_vert)?;
        device.destroy_shader(self.sh_depth_frag)?;
        device.destroy_shader(self.sh_depth_vert)?;

        Ok(())
    }
}
