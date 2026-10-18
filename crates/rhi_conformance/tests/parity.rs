//! Cross-backend pixel-parity tests ported from RHI audit probes R1–R8
//! (`docs/audits/2026-10-10-rhi-followup-review.md`).
//!
//! Each scenario runs once per backend. Tests that fail today because of a known finding are
//! `#[ignore]`d with the finding ID and the sub-milestone that fixes it; run them with
//! `cargo test -p korppi_rhi_conformance -- --ignored` to see the current failures. Remove the
//! `#[ignore]` in the commit that fixes the finding.

use korppi_rhi::{
    AccessFlags, BlendState, BufferDescriptor, BufferHandle, BufferMemoryBarrier,
    BufferTextureCopy, BufferUsage, ColorTargetState, ColorWriteMask, ComputePipelineDescriptor,
    CullMode, FilterMode, FrontFace, GraphicsPipelineDescriptor, ImageLayout, ImageMemoryBarrier,
    LoadOp, MemoryUsage, PipelineBarrier, PipelineHandle, PipelineStageFlags, PolygonMode,
    PrimitiveTopology, RasterizerState, SampleCount, SamplerAddressMode, SamplerDescriptor,
    ShaderStage, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage, VertexAttribute,
    VertexBinding, VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
use korppi_rhi::{BindingType, ShaderStages};
use korppi_rhi_conformance::{
    approx_rgba, color_pass, color_target, undefined_to_color_attachment, Backend, Image,
    TestDevice,
};
use korppi_rhi_conformance::{buffer_range, sampled, STORAGE};

const W: u32 = 64;
const H: u32 = 64;
/// Texels covered by a triangle clipped to exactly the top half of the target.
const HALF: usize = (W * H / 2) as usize;
const CLEAR_BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// CCW triangle in +Y-up NDC, (-1,0) -> (3,0) -> (-1,4): after clipping it covers exactly the
/// top half of the viewport.
const TOP_HALF_TRIANGLE: &str = r#"#version 450
const vec2 P[3] = vec2[3](vec2(-1.0, 0.0), vec2(3.0, 0.0), vec2(-1.0, 4.0));
void main() { gl_Position = vec4(P[gl_VertexIndex], 0.5, 1.0); }
"#;

/// CCW triangle covering the whole viewport: (-1,-1) -> (3,-1) -> (-1,3).
const FULLSCREEN_TRIANGLE: &str = r#"#version 450
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(P[gl_VertexIndex], 0.5, 1.0); }
"#;

/// Positions from vertex buffer binding 0, location 0.
const VERTEX_BUFFER_POSITIONS: &str = r#"#version 450
layout(location = 0) in vec2 pos;
void main() { gl_Position = vec4(pos, 0.5, 1.0); }
"#;

/// Constant half-transparent red.
const HALF_RED: &str = r#"#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.0, 0.0, 0.5); }
"#;

/// Writes vec4(1, 2, 3, 4) into the storage buffer at binding 0.
const WRITE_SSBO: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer B { vec4 v[]; };
void main() { v[0] = vec4(1.0, 2.0, 3.0, 4.0); }
"#;

/// Samples the texture at binding 0 at LOD 1 into the storage buffer at binding 1.
const SAMPLE_MIP1: &str = r#"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform sampler2D t;
layout(set = 0, binding = 1) buffer B { vec4 v[]; };
void main() { v[0] = textureLod(t, vec2(0.5, 0.5), 1.0); }
"#;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct DrawSetup {
    vertex_shader: &'static str,
    cull: CullMode,
    blend: Option<BlendState>,
    vertex_buffer: bool,
    viewport: [f32; 4],
    scissor: [u32; 4],
}

impl Default for DrawSetup {
    fn default() -> Self {
        Self {
            vertex_shader: TOP_HALF_TRIANGLE,
            cull: CullMode::None,
            blend: None,
            vertex_buffer: false,
            viewport: [0.0, 0.0, W as f32, H as f32],
            scissor: [0, 0, W, H],
        }
    }
}

fn triangle_pipeline(device: &TestDevice, setup: &DrawSetup) -> PipelineHandle {
    let vertex_shader = device.shader(ShaderStage::Vertex, setup.vertex_shader);
    let fragment_shader = device.shader(ShaderStage::Fragment, HALF_RED);
    let vertex_layout = if setup.vertex_buffer {
        VertexLayoutDescriptor::new()
            .with_binding(VertexBinding {
                binding: 0,
                stride: 8,
                step_mode: VertexStepMode::Vertex,
            })
            .with_attribute(VertexAttribute {
                location: 0,
                binding: 0,
                format: VertexFormat::Float32x2,
                offset: 0,
            })
    } else {
        VertexLayoutDescriptor::new()
    };
    device
        .create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader: Some(fragment_shader),
            vertex_layout,
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState {
                cull_mode: setup.cull,
                front_face: FrontFace::CounterClockwise,
                polygon_mode: PolygonMode::Fill,
                depth_bias: 0.0,
                slope_scaled_depth_bias: 0.0,
                depth_bias_clamp: 0.0,
                depth_clamp: false,
            },
            multisample: Default::default(),
            depth_stencil: None,
            color_targets: vec![ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: setup.blend,
                write_mask: ColorWriteMask::ALL,
            }],
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: Some("conformance_triangle"),
        })
        .expect("create_graphics_pipeline")
}

fn top_half_vertex_buffer(device: &TestDevice) -> BufferHandle {
    let positions: [f32; 6] = [-1.0, 0.0, 3.0, 0.0, -1.0, 4.0];
    let bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_ne_bytes()).collect();
    device
        .create_buffer(
            &BufferDescriptor {
                size: bytes.len() as u64,
                usage: BufferUsage::VERTEX,
                memory: MemoryUsage::CpuToGpu,
                label: Some("conformance_triangle_vertices"),
            },
            Some(&bytes),
        )
        .expect("create vertex buffer")
}

/// Clears a 64x64 target to opaque blue, draws one triangle, and reads the target back.
fn render(backend: Backend, setup: DrawSetup) -> Image {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, W, H, TextureUsage::empty());
    let pipeline = triangle_pipeline(&device, &setup);
    let vertex_buffer = setup.vertex_buffer.then(|| top_half_vertex_buffer(&device));

    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    recorder
        .begin_rendering(&color_pass(target, W, H, LoadOp::Clear, CLEAR_BLUE))
        .expect("begin_rendering");
    let [vx, vy, vw, vh] = setup.viewport;
    recorder.set_viewport(vx, vy, vw, vh, 0.0, 1.0);
    let [sx, sy, sw, sh] = setup.scissor;
    recorder.set_scissor(sx as i32, sy as i32, sw, sh);
    recorder.bind_pipeline(pipeline);
    if let Some(buffer) = vertex_buffer {
        recorder.bind_vertex_buffers(0, &[buffer], &[0]);
    }
    recorder.draw(3, 1, 0, 0);
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    Image::read(&*device, target, W, H)
}

/// The fragment shader leaves blue at 0, so any texel no longer fully blue was drawn.
fn drawn(p: [u8; 4]) -> bool {
    p[2] != 255
}

fn storage_vec4(device: &TestDevice) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size: 16,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
                memory: MemoryUsage::GpuToCpu,
                label: Some("conformance_ssbo"),
            },
            None,
        )
        .expect("create storage buffer")
}

fn read_vec4(device: &TestDevice, buffer: BufferHandle) -> [f32; 4] {
    let bytes = device.read_buffer(buffer, 0, 16).expect("read_buffer");
    let mut out = [0.0; 4];
    for (value, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *value = f32::from_ne_bytes(*chunk);
    }
    out
}

fn compute_to_host_barrier(buffer: BufferHandle) -> PipelineBarrier {
    PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
        buffer,
        src_stages: PipelineStageFlags::COMPUTE_SHADER,
        src_access: AccessFlags::SHADER_WRITE,
        dst_stages: PipelineStageFlags::HOST,
        dst_access: AccessFlags::HOST_READ,
        offset: 0,
        size: 16,
    })
}

// ---------------------------------------------------------------------------------------------
// R1: back-face culling keeps CCW triangles
// ---------------------------------------------------------------------------------------------

fn r1_ccw_triangle_survives_back_face_culling(backend: Backend) {
    let unculled = render(backend, DrawSetup::default());
    let culled = render(
        backend,
        DrawSetup {
            cull: CullMode::Back,
            ..DrawSetup::default()
        },
    );
    assert_eq!(unculled.count(drawn), HALF, "{backend:?}: CullMode::None");
    assert_eq!(
        culled.count(drawn),
        HALF,
        "{backend:?}: CullMode::Back removed a CCW (front-facing) triangle"
    );
}

#[test]
fn r1_ccw_triangle_survives_back_face_culling_vulkan() {
    r1_ccw_triangle_survives_back_face_culling(Backend::Vulkan);
}

#[test]
fn r1_ccw_triangle_survives_back_face_culling_opengl() {
    r1_ccw_triangle_survives_back_face_culling(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// R2: render-target row 0 is the top of the image; viewport and scissor origins are top-left
// ---------------------------------------------------------------------------------------------

fn r2_render_target_row_zero_is_top(backend: Backend) {
    let image = render(backend, DrawSetup::default());
    assert_eq!(
        image.rows_where(drawn),
        Some((0, H / 2 - 1)),
        "{backend:?}: a triangle covering the top half of NDC must land in rows 0..{}",
        H / 2 - 1
    );
}

#[test]
fn r2_render_target_row_zero_is_top_vulkan() {
    r2_render_target_row_zero_is_top(Backend::Vulkan);
}

#[test]
fn r2_render_target_row_zero_is_top_opengl() {
    r2_render_target_row_zero_is_top(Backend::OpenGl);
}

fn viewport_origin_is_top_left(backend: Backend) {
    let image = render(
        backend,
        DrawSetup {
            vertex_shader: FULLSCREEN_TRIANGLE,
            viewport: [0.0, 0.0, W as f32, (H / 2) as f32],
            ..DrawSetup::default()
        },
    );
    assert_eq!(image.rows_where(drawn), Some((0, H / 2 - 1)), "{backend:?}");
}

#[test]
fn viewport_origin_is_top_left_vulkan() {
    viewport_origin_is_top_left(Backend::Vulkan);
}

#[test]
fn viewport_origin_is_top_left_opengl() {
    viewport_origin_is_top_left(Backend::OpenGl);
}

fn scissor_origin_is_top_left(backend: Backend) {
    let image = render(
        backend,
        DrawSetup {
            vertex_shader: FULLSCREEN_TRIANGLE,
            scissor: [0, 0, W, H / 2],
            ..DrawSetup::default()
        },
    );
    assert_eq!(image.rows_where(drawn), Some((0, H / 2 - 1)), "{backend:?}");
}

#[test]
fn scissor_origin_is_top_left_vulkan() {
    scissor_origin_is_top_left(Backend::Vulkan);
}

#[test]
fn scissor_origin_is_top_left_opengl() {
    scissor_origin_is_top_left(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// R3: blend state is applied
// ---------------------------------------------------------------------------------------------

fn r3_alpha_blending_is_applied(backend: Backend) {
    let image = render(
        backend,
        DrawSetup {
            blend: Some(BlendState::ALPHA_BLENDING),
            ..DrawSetup::default()
        },
    );
    // (1,0,0,0.5) over (0,0,1,1): rgb = 0.5*src + 0.5*dst, alpha = 0.5 + 1*(1-0.5) = 1.
    // Orientation-independent: take any drawn texel (RHI-01 is tested separately by R2).
    let sample = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .map(|(x, y)| image.pixel(x, y))
        .find(|p| drawn(*p))
        .expect("the triangle drew nothing");
    assert!(
        approx_rgba(sample, [128, 0, 128, 255], 2),
        "{backend:?}: blended texel {sample:?}, expected ~[128, 0, 128, 255]"
    );
}

#[test]
fn r3_alpha_blending_is_applied_vulkan() {
    r3_alpha_blending_is_applied(Backend::Vulkan);
}

#[test]
fn r3_alpha_blending_is_applied_opengl() {
    r3_alpha_blending_is_applied(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// R4: vertex buffers feed vertex attributes
// ---------------------------------------------------------------------------------------------

fn r4_vertex_buffer_triangle_is_drawn(backend: Backend) {
    let image = render(
        backend,
        DrawSetup {
            vertex_shader: VERTEX_BUFFER_POSITIONS,
            vertex_buffer: true,
            ..DrawSetup::default()
        },
    );
    assert_eq!(image.count(drawn), HALF, "{backend:?}");
}

#[test]
fn r4_vertex_buffer_triangle_is_drawn_vulkan() {
    r4_vertex_buffer_triangle_is_drawn(Backend::Vulkan);
}

#[test]
fn r4_vertex_buffer_triangle_is_drawn_opengl() {
    r4_vertex_buffer_triangle_is_drawn(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// R5: LoadOp::Clear clears the whole render area regardless of earlier scissor state
// ---------------------------------------------------------------------------------------------

fn r5_clear_ignores_previous_scissor(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = color_target(&*device, W, H, TextureUsage::empty());
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    recorder
        .begin_rendering(&color_pass(
            target,
            W,
            H,
            LoadOp::Clear,
            [0.0, 0.0, 0.0, 1.0],
        ))
        .expect("first pass");
    recorder.set_viewport(0.0, 0.0, W as f32, H as f32, 0.0, 1.0);
    recorder.set_scissor(0, 0, 8, 8);
    recorder.end_rendering().expect("end first pass");
    recorder
        .begin_rendering(&color_pass(
            target,
            W,
            H,
            LoadOp::Clear,
            [1.0, 0.0, 0.0, 1.0],
        ))
        .expect("second pass");
    recorder.end_rendering().expect("end second pass");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let image = Image::read(&*device, target, W, H);
    assert_eq!(
        image.count(|p| p[0] > 200),
        (W * H) as usize,
        "{backend:?}: LoadOp::Clear must clear the whole render area"
    );
}

#[test]
fn r5_clear_ignores_previous_scissor_vulkan() {
    r5_clear_ignores_previous_scissor(Backend::Vulkan);
}

#[test]
fn r5_clear_ignores_previous_scissor_opengl() {
    r5_clear_ignores_previous_scissor(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// R6: buffer -> texture copies honour a byte row pitch
// ---------------------------------------------------------------------------------------------

fn r6_buffer_to_texture_copy_uses_byte_row_pitch(backend: Backend) {
    let device = TestDevice::open(backend);
    let texture = device
        .create_texture(
            &TextureDescriptor {
                width: 4,
                height: 4,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::COPY_DST
                    | TextureUsage::COPY_SRC
                    | TextureUsage::COLOR_ATTACHMENT,
                dimension: TextureDimension::D2,
                label: None,
            },
            None,
        )
        .expect("create texture");
    // Four 16-byte rows with R = 10, 20, 30, 40, followed by 0xEE filler a wrong stride would hit.
    let mut bytes = vec![0xEE_u8; 256];
    for row in 0..4_usize {
        for column in 0..4_usize {
            let offset = row * 16 + column * 4;
            bytes[offset..offset + 4].copy_from_slice(&[10 * (row as u8 + 1), 0, 0, 255]);
        }
    }
    let staging = device
        .create_buffer(
            &BufferDescriptor {
                size: 256,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            Some(&bytes),
        )
        .expect("create staging buffer");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(
        &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture,
            range: korppi_rhi::TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::TRANSFER,
            dst_access: AccessFlags::TRANSFER_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::TransferDst,
        }),
    );
    recorder.copy_buffer_to_texture(
        staging,
        texture,
        &BufferTextureCopy {
            buffer_offset: 0,
            buffer_row_pitch: 16,
            texture_mip_level: 0,
            texture_layer: 0,
            origin: [0, 0, 0],
            extent: [4, 4, 1],
        },
    );
    recorder.pipeline_barrier(
        &PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture,
            range: korppi_rhi::TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TRANSFER,
            src_access: AccessFlags::TRANSFER_WRITE,
            dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access: AccessFlags::COLOR_ATTACHMENT_READ,
            old_layout: ImageLayout::TransferDst,
            new_layout: ImageLayout::ColorAttachment,
        }),
    );
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let image = Image::read(&*device, texture, 4, 4);
    let rows: Vec<u8> = (0..4).map(|y| image.pixel(0, y)[0]).collect();
    assert_eq!(rows, vec![10, 20, 30, 40], "{backend:?}: row R values");
}

#[test]
fn r6_buffer_to_texture_copy_uses_byte_row_pitch_vulkan() {
    r6_buffer_to_texture_copy_uses_byte_row_pitch(Backend::Vulkan);
}

#[test]
fn r6_buffer_to_texture_copy_uses_byte_row_pitch_opengl() {
    r6_buffer_to_texture_copy_uses_byte_row_pitch(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// R7: read_buffer returns what the GPU wrote
// ---------------------------------------------------------------------------------------------

fn r7_gpu_written_buffer_reads_back(backend: Backend) {
    let device = TestDevice::open(backend);
    let output = storage_vec4(&device);
    let (layout, group) = device.bind_group(&[(
        0,
        STORAGE,
        ShaderStages::COMPUTE,
        buffer_range(output, 0, 16),
    )]);
    let compute_shader = device.shader(ShaderStage::Compute, WRITE_SSBO);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create_compute_pipeline");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.bind_pipeline(pipeline);
    recorder.set_bind_group(0, group, &[]);
    recorder.dispatch(1, 1, 1);
    recorder.pipeline_barrier(&compute_to_host_barrier(output));
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    assert_eq!(
        read_vec4(&device, output),
        [1.0, 2.0, 3.0, 4.0],
        "{backend:?}"
    );
}

#[test]
fn r7_gpu_written_buffer_reads_back_vulkan() {
    r7_gpu_written_buffer_reads_back(Backend::Vulkan);
}

#[test]
fn r7_gpu_written_buffer_reads_back_opengl() {
    r7_gpu_written_buffer_reads_back(Backend::OpenGl);
}

// ---------------------------------------------------------------------------------------------
// R8: samplers sample mip levels > 0
// ---------------------------------------------------------------------------------------------

fn r8_sampler_reads_requested_mip_level(backend: Backend) {
    let device = TestDevice::open(backend);
    let red_level0 = [255_u8, 0, 0, 255].repeat(4);
    let texture = device
        .create_texture(
            &TextureDescriptor {
                width: 2,
                height: 2,
                depth: 1,
                array_layers: 1,
                mip_levels: 2,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                dimension: TextureDimension::D2,
                label: None,
            },
            Some(&red_level0),
        )
        .expect("create texture");
    let green_level1 = device
        .create_buffer(
            &BufferDescriptor {
                size: 4,
                usage: BufferUsage::COPY_SRC,
                memory: MemoryUsage::CpuToGpu,
                label: None,
            },
            Some(&[0, 255, 0, 255]),
        )
        .expect("create staging buffer");
    // Textures created with initial data are in `ShaderReadOnly`; the copy into mip 1 needs
    // that one level in `TransferDst` (RHI-06: barriers address mip ranges).
    let mip1 = |old_layout, new_layout, src_access, dst_access| {
        PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
            texture,
            range: korppi_rhi::TextureSubresourceRange::single(1, 0),
            src_stages: PipelineStageFlags::ALL_COMMANDS,
            src_access,
            dst_stages: PipelineStageFlags::ALL_COMMANDS,
            dst_access,
            old_layout,
            new_layout,
        })
    };
    let mut upload = device.create_command_recorder().expect("recorder");
    upload.begin().expect("begin");
    upload.pipeline_barrier(&mip1(
        ImageLayout::ShaderReadOnly,
        ImageLayout::TransferDst,
        AccessFlags::empty(),
        AccessFlags::TRANSFER_WRITE,
    ));
    upload.copy_buffer_to_texture(
        green_level1,
        texture,
        &BufferTextureCopy {
            buffer_offset: 0,
            buffer_row_pitch: 0,
            texture_mip_level: 1,
            texture_layer: 0,
            origin: [0, 0, 0],
            extent: [1, 1, 1],
        },
    );
    upload.pipeline_barrier(&mip1(
        ImageLayout::TransferDst,
        ImageLayout::ShaderReadOnly,
        AccessFlags::TRANSFER_WRITE,
        AccessFlags::SHADER_READ,
    ));
    device
        .submit(upload.finish().expect("finish"))
        .expect("submit");

    let sampler = device
        .create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            ..SamplerDescriptor::default()
        })
        .expect("create sampler");
    let output = storage_vec4(&device);
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
    let compute_shader = device.shader(ShaderStage::Compute, SAMPLE_MIP1);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader,
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("create_compute_pipeline");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.bind_pipeline(pipeline);
    recorder.set_bind_group(0, group, &[]);
    recorder.dispatch(1, 1, 1);
    recorder.pipeline_barrier(&compute_to_host_barrier(output));
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    assert_eq!(
        read_vec4(&device, output),
        [0.0, 1.0, 0.0, 1.0],
        "{backend:?}: textureLod(.., 1.0) must return mip level 1 (green)"
    );
}

#[test]
fn r8_sampler_reads_requested_mip_level_vulkan() {
    r8_sampler_reads_requested_mip_level(Backend::Vulkan);
}

#[test]
fn r8_sampler_reads_requested_mip_level_opengl() {
    r8_sampler_reads_requested_mip_level(Backend::OpenGl);
}

/// Writes to sRGB render targets are encoded on both backends (FIX-1.3): Vulkan encodes
/// `*_SRGB` attachments, OpenGL only with `GL_FRAMEBUFFER_SRGB` enabled.
fn srgb_render_targets_encode_linear_colour(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = device
        .create_texture(
            &TextureDescriptor {
                width: 4,
                height: 4,
                depth: 1,
                array_layers: 1,
                mip_levels: 1,
                sample_count: SampleCount::Count1,
                format: TextureFormat::Rgba8Srgb,
                usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
                dimension: TextureDimension::D2,
                label: Some("srgb_target"),
            },
            None,
        )
        .expect("create sRGB target");
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    recorder.pipeline_barrier(&undefined_to_color_attachment(target));
    recorder
        .begin_rendering(&color_pass(
            target,
            4,
            4,
            LoadOp::Clear,
            [0.5, 0.0, 1.0, 1.0],
        ))
        .expect("begin_rendering");
    recorder.end_rendering().expect("end_rendering");
    device
        .submit(recorder.finish().expect("finish"))
        .expect("submit");
    device.wait_idle().expect("wait_idle");
    let image = Image::read(&*device, target, 4, 4);
    // Linear 0.5 encodes to sRGB 188; 0 and 1 are unchanged; alpha is linear.
    assert!(
        approx_rgba(image.pixel(1, 1), [188, 0, 255, 255], 1),
        "{backend:?}: linear 0.5 must be stored sRGB-encoded (188), got {:?}",
        image.pixel(1, 1)
    );
}

#[test]
fn srgb_render_targets_encode_linear_colour_vulkan() {
    srgb_render_targets_encode_linear_colour(Backend::Vulkan);
}

#[test]
fn srgb_render_targets_encode_linear_colour_opengl() {
    srgb_render_targets_encode_linear_colour(Backend::OpenGl);
}
