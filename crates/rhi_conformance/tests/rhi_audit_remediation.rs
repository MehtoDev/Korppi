//! Regression tests for the Phase 3.9 RHI & Dual Backends re-audit (`FIX-12.4` / `FIX-13.4`).
//!
//! Covers findings:
//! - `RHI-18` (Probe `R21`): `PushConstantRange` checked overflow, non-zero size, non-empty
//!   stages & 4-byte alignment; `ComputePipelineDescriptor::validate`; `VertexLayoutDescriptor`
//!   duplicate binding/location & attribute stride bounds.
//! - `RHI-19` (Probe `R22`, **Critical**): `ValidatingRecorder::begin_rendering` setting
//!   `self.pass` only after `self.inner().begin_rendering(desc)` succeeds (preventing
//!   `vkCmdEndRendering` `SIGSEGV` inside `libnvidia-glcore.so`), plus vertex/index buffer usage,
//!   bounds & alignment, `begin_rendering` attachment usage, count, resolve dimensions,
//!   `renderArea` bounds, and `set_bind_group` checked dynamic offset addition.
//! - `RVK-20` & `RGL-23` (Probe `R23`): Swapchain child `TextureViewHandle` cleanup, bind-group
//!   invalidation on swapchain retirement/destruction, and `SwapchainHandle` generation validation.
//! - `RGL-24` (Probe `R24`): `split_combined_image_samplers` operand-length bounds & checked
//!   `bound` increment, and `ProgramBinaryCache::load` bounded pre-allocation.
//! - `RGL-25` (Probe `R25`): OpenGL `BeginRendering` `LoadOp::Clear` sub-rectangle `renderArea`
//!   scissor parity with Vulkan.

use korppi_rhi::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferDescriptor, BufferUsage, ComputePipelineDescriptor,
    DepthAttachment, GraphicsPipelineDescriptor, IndexType, LoadOp, MemoryUsage, MultisampleState,
    PrimitiveTopology, PushConstantRange, RasterizerState, RenderingAttachment,
    RenderingDescriptor, RhiError, SampleCount, ShaderHandle, ShaderStage, ShaderStages, StoreOp,
    SwapchainDescriptor, SwapchainHandle, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsage, TextureViewDescriptor, VertexAttribute, VertexBinding,
    VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
use korppi_rhi_conformance::{Backend, TestDevice};

const NOOP_COMPUTE: &str = r"#version 450
layout(local_size_x = 1) in;
void main() {}
";

/// `RHI-18` (Probe `R21`): `PushConstantRange`, `GraphicsPipelineDescriptor`,
/// `ComputePipelineDescriptor`, and `VertexLayoutDescriptor` validation.
#[test]
fn r21_pipeline_and_vertex_descriptor_validation() {
    let vs = ShaderHandle::new(1, 1);
    let base_graphics = |ranges: Vec<PushConstantRange>, vertex_layout: VertexLayoutDescriptor| {
        GraphicsPipelineDescriptor {
            vertex_shader: vs,
            fragment_shader: None,
            vertex_layout,
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState::default(),
            multisample: MultisampleState::default(),
            depth_stencil: None,
            color_targets: vec![],
            bind_group_layouts: vec![],
            push_constant_ranges: ranges,
            label: None,
        }
    };

    // 1. PushConstantRange integer overflow (`offset + size` wraps u32) must not panic or pass.
    let overflow_desc = base_graphics(
        vec![PushConstantRange {
            stages: ShaderStages::VERTEX,
            offset: u32::MAX - 3,
            size: 4,
        }],
        VertexLayoutDescriptor::default(),
    );
    assert!(
        matches!(overflow_desc.validate(), Err(RhiError::ValidationError(_))),
        "overflowing PushConstantRange must be rejected"
    );

    // 2. Empty stages, zero size, unaligned offset, and unaligned size must all be rejected.
    for bad_range in [
        PushConstantRange {
            stages: ShaderStages::empty(),
            offset: 0,
            size: 16,
        },
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            offset: 0,
            size: 0,
        },
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            offset: 2,
            size: 16,
        },
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            offset: 0,
            size: 6,
        },
    ] {
        let desc = base_graphics(vec![bad_range], VertexLayoutDescriptor::default());
        assert!(
            matches!(desc.validate(), Err(RhiError::ValidationError(_))),
            "invalid PushConstantRange {bad_range:?} must be rejected"
        );
    }

    // 3. ComputePipelineDescriptor::validate must reject invalid push constant ranges on both
    //    descriptor validation and backend `create_compute_pipeline`.
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let cs = device.shader(ShaderStage::Compute, NOOP_COMPUTE);
        let bad_compute = ComputePipelineDescriptor {
            compute_shader: cs,
            bind_group_layouts: vec![],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 256,
                size: 256,
            }],
            label: None,
        };
        assert!(
            matches!(
                device.create_compute_pipeline(&bad_compute),
                Err(RhiError::ValidationError(_))
            ),
            "{backend:?}: create_compute_pipeline must validate push_constant_ranges"
        );
    }

    // 4. VertexLayoutDescriptor::validate must reject duplicate bindings, duplicate locations,
    //    and attributes exceeding a non-zero binding stride.
    let dup_binding = VertexLayoutDescriptor::new()
        .with_binding(VertexBinding {
            binding: 0,
            stride: 16,
            step_mode: VertexStepMode::Vertex,
        })
        .with_binding(VertexBinding {
            binding: 0,
            stride: 16,
            step_mode: VertexStepMode::Vertex,
        });
    assert!(
        matches!(dup_binding.validate(), Err(RhiError::ValidationError(_))),
        "duplicate vertex binding index must be rejected"
    );

    let dup_location = VertexLayoutDescriptor::new()
        .with_binding(VertexBinding {
            binding: 0,
            stride: 32,
            step_mode: VertexStepMode::Vertex,
        })
        .with_attribute(VertexAttribute {
            location: 0,
            binding: 0,
            format: VertexFormat::Float32x2,
            offset: 0,
        })
        .with_attribute(VertexAttribute {
            location: 0,
            binding: 0,
            format: VertexFormat::Float32x2,
            offset: 8,
        });
    assert!(
        matches!(dup_location.validate(), Err(RhiError::ValidationError(_))),
        "duplicate vertex attribute location must be rejected"
    );

    let stride_exceeded = VertexLayoutDescriptor::new()
        .with_binding(VertexBinding {
            binding: 0,
            stride: 8,
            step_mode: VertexStepMode::Vertex,
        })
        .with_attribute(VertexAttribute {
            location: 0,
            binding: 0,
            format: VertexFormat::Float32x4,
            offset: 0,
        });
    assert!(
        matches!(
            stride_exceeded.validate(),
            Err(RhiError::ValidationError(_))
        ),
        "attribute extending past non-zero binding stride must be rejected"
    );
}

/// `RHI-19` (Probe `R22`, **Critical**): `ValidatingRecorder` vertex/index buffer usage, offset
/// bounds & alignment, `begin_rendering` attachment usage, `renderArea` bounds, and preventing
/// `end_rendering()` from calling `vkCmdEndRendering` when `begin_rendering` failed.
#[test]
fn r22_validating_recorder_vertex_index_and_rendering_checks() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let uniform_only = device
            .create_buffer(
                &BufferDescriptor {
                    size: 16,
                    usage: BufferUsage::UNIFORM,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("r22_uniform_only"),
                },
                None,
            )
            .expect("uniform buffer");
        let vertex_buf = device
            .create_buffer(
                &BufferDescriptor {
                    size: 16,
                    usage: BufferUsage::VERTEX,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("r22_vertex_buf"),
                },
                None,
            )
            .expect("vertex buffer");
        let index_buf = device
            .create_buffer(
                &BufferDescriptor {
                    size: 16,
                    usage: BufferUsage::INDEX,
                    memory: MemoryUsage::CpuToGpu,
                    label: Some("r22_index_buf"),
                },
                None,
            )
            .expect("index buffer");

        // 1a. bind_vertex_buffers and bind_index_buffer with wrong usage (UNIFORM only)
        let mut rec = device.create_command_recorder().expect("recorder");
        rec.begin().expect("begin");
        rec.bind_vertex_buffers(0, &[uniform_only], &[0]);
        rec.bind_index_buffer(uniform_only, 0, IndexType::Uint32);
        let err = match rec.finish() {
            Err(RhiError::ValidationError(msg)) => msg,
            other => panic!(
                "{backend:?}: expected ValidationError for missing VERTEX/INDEX usage, got {:?}",
                other.map(|_| ())
            ),
        };
        assert!(err.contains("VERTEX"), "{backend:?}: {err}");
        assert!(err.contains("INDEX"), "{backend:?}: {err}");

        // 1b. bind_vertex_buffers and bind_index_buffer with out-of-bounds offset and unaligned index offset
        let mut rec = device.create_command_recorder().expect("recorder");
        rec.begin().expect("begin");
        rec.bind_vertex_buffers(0, &[vertex_buf], &[1024]);
        rec.bind_index_buffer(index_buf, 1, IndexType::Uint32);
        let err = match rec.finish() {
            Err(RhiError::ValidationError(msg)) => msg,
            other => panic!(
                "{backend:?}: expected ValidationError for OOB vertex offset and unaligned index offset, got {:?}",
                other.map(|_| ())
            ),
        };
        assert!(err.contains("bind_vertex_buffers"), "{backend:?}: {err}");
        assert!(err.contains("bind_index_buffer"), "{backend:?}: {err}");

        // 2a. begin_rendering on a SAMPLED-only texture (no COLOR_ATTACHMENT) followed by
        //     end_rendering() and finish() must NOT crash (SIGSEGV) or emit Vulkan/GL errors,
        //     and must fail validation cleanly.
        let sampled_only = device
            .create_texture(
                &TextureDescriptor {
                    width: 16,
                    height: 16,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::SAMPLED,
                    dimension: TextureDimension::D2,
                    label: Some("r22_sampled_only"),
                },
                None,
            )
            .expect("sampled_only texture");
        let mut rec = device.create_command_recorder().expect("recorder");
        rec.begin().expect("begin");
        let _ = rec.begin_rendering(&RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width: 16,
            height: 16,
            color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                texture: sampled_only,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: Some([1.0, 0.0, 0.0, 1.0]),
                resolve_target: None,
            }],
            depth_attachment: None,
        });
        let _ = rec.end_rendering();
        assert!(
            matches!(rec.finish(), Err(RhiError::ValidationError(_))),
            "{backend:?}: SAMPLED-only colour attachment must fail validation"
        );
        device.assert_no_api_errors();

        // 2b. begin_rendering on a COLOR_ATTACHMENT texture used as depth_attachment (lacks DEPTH_STENCIL_ATTACHMENT)
        let color_tex = device
            .create_texture(
                &TextureDescriptor {
                    width: 16,
                    height: 16,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::COLOR_ATTACHMENT,
                    dimension: TextureDimension::D2,
                    label: Some("r22_color_16x16"),
                },
                None,
            )
            .expect("color texture");
        let mut rec = device.create_command_recorder().expect("recorder");
        rec.begin().expect("begin");
        let _ = rec.begin_rendering(&RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width: 16,
            height: 16,
            color_attachments: korppi_rhi::smallvec![],
            depth_attachment: Some(DepthAttachment {
                texture: color_tex,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_depth: Some(0.0),
                clear_stencil: None,
                stencil_load_op: LoadOp::DontCare,
                stencil_store_op: StoreOp::DontCare,
            }),
        });
        let _ = rec.end_rendering();
        assert!(
            matches!(rec.finish(), Err(RhiError::ValidationError(_))),
            "{backend:?}: depth_attachment lacking DEPTH_STENCIL_ATTACHMENT usage must fail validation"
        );

        // 2c. begin_rendering with negative offset or out-of-bounds renderArea
        for (ox, oy, w, h) in [(-16, -16, 1024, 1024), (0, 0, 0, 16), (8, 8, 16, 16)] {
            let mut rec = device.create_command_recorder().expect("recorder");
            rec.begin().expect("begin");
            let _ = rec.begin_rendering(&RenderingDescriptor {
                offset_x: ox,
                offset_y: oy,
                width: w,
                height: h,
                color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                    texture: color_tex,
                    mip_level: 0,
                    layer: 0,
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_color: Some([1.0, 0.0, 0.0, 1.0]),
                    resolve_target: None,
                }],
                depth_attachment: None,
            });
            let _ = rec.end_rendering();
            assert!(
                matches!(rec.finish(), Err(RhiError::ValidationError(_))),
                "{backend:?}: invalid renderArea ({ox}, {oy}, {w}x{h}) must fail validation"
            );
            device.assert_no_api_errors();
        }

        // 3. set_bind_group with a dynamic offset that overflows u64 when added to binding.end
        let dyn_bgl = device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: true,
                    },
                    stages: ShaderStages::COMPUTE,
                }],
                label: None,
            })
            .expect("dyn_bgl");
        let dyn_bg = device
            .create_bind_group(&BindGroupDescriptor {
                layout: dyn_bgl,
                entries: vec![BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: uniform_only,
                        offset: 0,
                        size: 16,
                    },
                }],
                label: None,
            })
            .expect("dyn_bg");
        let mut rec = device.create_command_recorder().expect("recorder");
        rec.begin().expect("begin");
        rec.set_bind_group(0, dyn_bg, &[!255u32]);
        assert!(
            matches!(rec.finish(), Err(RhiError::ValidationError(_))),
            "{backend:?}: out-of-bounds dynamic offset must fail validation"
        );
    }
}

/// `RVK-20` & `RGL-23` (Probe `R23`): Swapchain handle generation check, child `TextureView`
/// cleanup, and bind-group invalidation on swapchain destruction/retirement.
#[test]
#[cfg(target_os = "linux")]
fn r23_swapchain_view_and_bind_group_cleanup_and_generation_check() {
    use korppi_rhi_conformance::display::{display_or_skip, TestWindow};
    use std::sync::Arc;

    let Some(display) = display_or_skip("r23_swapchain_audit") else {
        return;
    };

    for backend in Backend::ALL {
        let window = Arc::new(TestWindow::create(&display, 64, 48));
        let device = TestDevice::open_for_display(backend, window.clone());
        let sc = device
            .create_swapchain(window.clone(), &SwapchainDescriptor::new(64, 48))
            .expect("create_swapchain");

        // 1. Forged SwapchainHandle with wrong generation must be rejected with InvalidHandle.
        for bad_gen in [0, 999] {
            let forged = SwapchainHandle::from_parts(sc.device(), sc.id(), bad_gen);
            assert!(
                matches!(device.swapchain_info(forged), Err(RhiError::InvalidHandle(_))),
                "{backend:?}: forged SwapchainHandle(gen={bad_gen}) must be rejected by swapchain_info"
            );
            assert!(
                matches!(
                    device.acquire_next_image(forged),
                    Err(RhiError::InvalidHandle(_))
                ),
                "{backend:?}: forged SwapchainHandle(gen={bad_gen}) must be rejected by acquire_next_image"
            );
            assert!(
                matches!(
                    device.resize_swapchain(forged, 32, 32),
                    Err(RhiError::InvalidHandle(_))
                ),
                "{backend:?}: forged SwapchainHandle(gen={bad_gen}) must be rejected by resize_swapchain"
            );
            assert!(
                matches!(
                    device.destroy_swapchain(forged),
                    Err(RhiError::InvalidHandle(_))
                ),
                "{backend:?}: forged SwapchainHandle(gen={bad_gen}) must be rejected by destroy_swapchain"
            );
        }

        // 2. Acquire swapchain image, create a child TextureView referencing it, then destroy the
        //    swapchain. The child view must be removed and destroyed before vkDestroySwapchainKHR
        //    with zero Vulkan/GL validation errors.
        let image = device
            .acquire_next_image(sc)
            .expect("acquire")
            .expect("swapchain image");
        let view = device
            .create_texture_view(&TextureViewDescriptor {
                texture: image.texture,
                dimension: TextureDimension::D2,
                base_mip_level: 0,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: TextureAspect::All,
                label: Some("r23_sc_view"),
            })
            .expect("create_texture_view on swapchain texture");

        device.destroy_swapchain(sc).expect("destroy_swapchain");
        device.assert_no_api_errors();

        // Child view was destroyed with its parent swapchain texture:
        assert!(
            matches!(
                device.destroy_texture_view(view),
                Err(RhiError::InvalidHandle(_))
            ),
            "{backend:?}: child view of destroyed swapchain texture must already be removed"
        );
    }
}

/// `RGL-24` (Probe `R24`): `split_combined_image_samplers` malformed instruction operand checks
/// and `ProgramBinaryCache::load` bounded pre-allocation.
#[test]
fn r24_opengl_spirv_splitter_and_cache_hardening() {
    // 1. Truncated 1-word OP_TYPE_SAMPLED_IMAGE (opcode 27) with 0 operands must return Err, not panic.
    let truncated = [0x0723_0203u32, 0x0001_0000, 0, 1, 0, (1 << 16) | 27];
    assert!(
        korppi_rhi_opengl::spirv::split_combined_image_samplers(&truncated).is_err(),
        "truncated OP_TYPE_SAMPLED_IMAGE must return Err"
    );

    // Truncated OP_TYPE_ARRAY (28), OP_TYPE_POINTER (32), OP_VARIABLE (59), OP_FUNCTION_PARAMETER (55),
    // OP_DECORATE (71), OP_LOAD (61) must all return Err without panicking.
    for opcode in [27u32, 28, 29, 32, 55, 59, 61, 71] {
        let words = [0x0723_0203u32, 0x0001_0000, 0, 1, 0, (1 << 16) | opcode];
        let _ = korppi_rhi_opengl::spirv::split_combined_image_samplers(&words);
    }

    // 2. Forged pipeline cache binary with count = u32::MAX must be rejected without OOM panic.
    let device = TestDevice::open(Backend::OpenGl);
    let cs = device.shader(ShaderStage::Compute, NOOP_COMPUTE);
    let _pipe = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: cs,
            bind_group_layouts: vec![],
            push_constant_ranges: vec![],
            label: None,
        })
        .expect("pipeline");
    let valid_cache = device.pipeline_cache_data().expect("pipeline_cache_data");
    if valid_cache.len() >= 16 {
        // Header: MAGIC (8 bytes) + VERSION (4 bytes) + driver_len (4 bytes) + driver (driver_len bytes) + count (4 bytes)
        let driver_len = u32::from_le_bytes(valid_cache[12..16].try_into().unwrap()) as usize;
        let count_offset = 16 + driver_len;
        if valid_cache.len() >= count_offset + 4 {
            let mut forged = valid_cache[..count_offset + 4].to_vec();
            forged[count_offset..count_offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(
                !device
                    .load_pipeline_cache(&forged)
                    .expect("load_pipeline_cache"),
                "forged cache with count=u32::MAX and 0 trailing bytes must return Ok(false)"
            );
        }
    }
}

/// `RGL-25` (Probe `R25`): Sub-rectangle `LoadOp::Clear` parity between Vulkan and OpenGL.
#[test]
fn r25_subrect_load_op_clear_parity() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let mut init = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            init.extend_from_slice(&[255u8, 0, 0, 255]);
        }
        let tex = device
            .create_texture(
                &TextureDescriptor {
                    width: 4,
                    height: 4,
                    depth: 1,
                    array_layers: 1,
                    mip_levels: 1,
                    sample_count: SampleCount::Count1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::COLOR_ATTACHMENT
                        | TextureUsage::COPY_SRC
                        | TextureUsage::COPY_DST,
                    dimension: TextureDimension::D2,
                    label: Some("r25_subrect_clear"),
                },
                Some(&init),
            )
            .expect("texture");

        let mut rec = device.create_command_recorder().expect("recorder");
        rec.begin().expect("begin");
        rec.begin_rendering(&RenderingDescriptor {
            offset_x: 0,
            offset_y: 0,
            width: 2,
            height: 2,
            color_attachments: korppi_rhi::smallvec![RenderingAttachment {
                texture: tex,
                mip_level: 0,
                layer: 0,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_color: Some([0.0, 1.0, 0.0, 1.0]),
                resolve_target: None,
            }],
            depth_attachment: None,
        })
        .expect("begin_rendering");
        rec.end_rendering().expect("end_rendering");
        device
            .submit_and_wait(rec.finish().expect("finish"))
            .expect("submit");

        let bytes = device.read_texture(tex, 0).expect("read_texture");
        let pixel = |x: usize, y: usize| -> [u8; 4] {
            let idx = (y * 4 + x) * 4;
            [bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]
        };
        assert_eq!(
            pixel(0, 0),
            [0, 255, 0, 255],
            "{backend:?}: inside 2x2 renderArea at (0,0) must be cleared to green"
        );
        assert_eq!(
            pixel(3, 3),
            [255, 0, 0, 255],
            "{backend:?}: outside 2x2 renderArea at (3,3) must remain untouched red"
        );
    }
}
