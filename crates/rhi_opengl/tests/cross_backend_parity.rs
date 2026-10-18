//! Integration tests verifying cross-backend rendering parity between Vulkan and OpenGL backends.

use korppi_rhi::{
    AccessFlags, GraphicsDevice, ImageLayout, ImageMemoryBarrier, LoadOp, PipelineBarrier,
    PipelineStageFlags, RenderingAttachment, RenderingDescriptor, StoreOp, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_cross_backend_dynamic_rendering_clear_parity() {
    let clear_color = [0.2f32, 0.4f32, 0.8f32, 1.0f32];
    let width = 64u32;
    let height = 64u32;

    let expected_r = (clear_color[0] * 255.0).round() as u8;
    let expected_g = (clear_color[1] * 255.0).round() as u8;
    let expected_b = (clear_color[2] * 255.0).round() as u8;
    let expected_a = (clear_color[3] * 255.0).round() as u8;

    let tex_desc = TextureDescriptor {
        width,
        height,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: korppi_rhi::SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        label: Some("parity_offscreen_target"),
    };

    // ----------------------------------------------------
    // 1. Vulkan 1.3 Execution
    // ----------------------------------------------------
    let vk_device =
        VulkanDevice::create_headless().expect("Headless Vulkan device creation failed");
    let vk_tex = vk_device
        .create_texture(&tex_desc, None)
        .expect("Vulkan create_texture failed");

    let mut vk_recorder = vk_device
        .create_command_recorder()
        .expect("Vulkan create_command_recorder failed");
    vk_recorder.begin().expect("Vulkan begin failed");

    let vk_barrier = PipelineBarrier {
        memory_barriers: korppi_rhi::smallvec![],
        buffer_barriers: korppi_rhi::smallvec![],
        image_barriers: korppi_rhi::smallvec![ImageMemoryBarrier {
            texture: vk_tex,
            range: korppi_rhi::TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::ColorAttachment,
        }],
    };
    vk_recorder.pipeline_barrier(&vk_barrier);

    let render_desc_vk = RenderingDescriptor {
        width,
        height,
        offset_x: 0,
        offset_y: 0,
        color_attachments: korppi_rhi::smallvec![RenderingAttachment {
            texture: vk_tex,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color: Some(clear_color),
            resolve_target: None,
        }],
        depth_attachment: None,
    };

    vk_recorder
        .begin_rendering(&render_desc_vk)
        .expect("Vulkan begin_rendering failed");
    vk_recorder.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    vk_recorder.set_scissor(0, 0, width, height);
    vk_recorder
        .end_rendering()
        .expect("Vulkan end_rendering failed");

    let vk_cmd = vk_recorder.finish().expect("Vulkan finish failed");
    vk_device.submit(vk_cmd).expect("Vulkan submit failed");
    vk_device.wait_idle().expect("Vulkan wait_idle failed");

    let vk_pixels = vk_device
        .read_texture(vk_tex, 0)
        .expect("Vulkan read_texture failed");

    vk_device
        .destroy_texture(vk_tex)
        .expect("Vulkan destroy_texture failed");

    // ----------------------------------------------------
    // 2. OpenGL 4.5 Execution
    // ----------------------------------------------------
    let gl_device =
        OpenGlDevice::create_headless().expect("Headless OpenGL device creation failed");
    let gl_tex = gl_device
        .create_texture(&tex_desc, None)
        .expect("OpenGL create_texture failed");

    let mut gl_recorder = gl_device
        .create_command_recorder()
        .expect("OpenGL create_command_recorder failed");
    gl_recorder.begin().expect("OpenGL begin failed");

    let gl_barrier = PipelineBarrier {
        memory_barriers: korppi_rhi::smallvec![],
        buffer_barriers: korppi_rhi::smallvec![],
        image_barriers: korppi_rhi::smallvec![ImageMemoryBarrier {
            texture: gl_tex,
            range: korppi_rhi::TextureSubresourceRange::ALL,
            src_stages: PipelineStageFlags::TOP_OF_PIPE,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::ColorAttachment,
        }],
    };
    gl_recorder.pipeline_barrier(&gl_barrier);

    let render_desc_gl = RenderingDescriptor {
        width,
        height,
        offset_x: 0,
        offset_y: 0,
        color_attachments: korppi_rhi::smallvec![RenderingAttachment {
            texture: gl_tex,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color: Some(clear_color),
            resolve_target: None,
        }],
        depth_attachment: None,
    };

    gl_recorder
        .begin_rendering(&render_desc_gl)
        .expect("OpenGL begin_rendering failed");
    gl_recorder.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    gl_recorder.set_scissor(0, 0, width, height);
    gl_recorder
        .end_rendering()
        .expect("OpenGL end_rendering failed");

    let gl_cmd = gl_recorder.finish().expect("OpenGL finish failed");
    gl_device.submit(gl_cmd).expect("OpenGL submit failed");
    gl_device.wait_idle().expect("OpenGL wait_idle failed");

    let gl_pixels = gl_device
        .read_texture(gl_tex, 0)
        .expect("OpenGL read_texture failed");

    gl_device
        .destroy_texture(gl_tex)
        .expect("OpenGL destroy_texture failed");

    // ----------------------------------------------------
    // 3. Parity & Invariant Assertions
    // ----------------------------------------------------
    let expected_len = (width * height * 4) as usize;
    assert_eq!(
        vk_pixels.len(),
        expected_len,
        "Vulkan pixel buffer size mismatch"
    );
    assert_eq!(
        gl_pixels.len(),
        expected_len,
        "OpenGL pixel buffer size mismatch"
    );

    for idx in 0..(width * height) as usize {
        let base = idx * 4;
        let vk_r = vk_pixels[base];
        let vk_g = vk_pixels[base + 1];
        let vk_b = vk_pixels[base + 2];
        let vk_a = vk_pixels[base + 3];

        let gl_r = gl_pixels[base];
        let gl_g = gl_pixels[base + 1];
        let gl_b = gl_pixels[base + 2];
        let gl_a = gl_pixels[base + 3];

        // Assert against expected values
        assert!(
            (vk_r as i32 - expected_r as i32).abs() <= 1,
            "Pixel {idx} Vulkan Red mismatch: got {vk_r}, expected {expected_r}"
        );
        assert!(
            (vk_g as i32 - expected_g as i32).abs() <= 1,
            "Pixel {idx} Vulkan Green mismatch: got {vk_g}, expected {expected_g}"
        );
        assert!(
            (vk_b as i32 - expected_b as i32).abs() <= 1,
            "Pixel {idx} Vulkan Blue mismatch: got {vk_b}, expected {expected_b}"
        );
        assert_eq!(vk_a, expected_a, "Pixel {idx} Vulkan Alpha mismatch");

        assert!(
            (gl_r as i32 - expected_r as i32).abs() <= 1,
            "Pixel {idx} OpenGL Red mismatch: got {gl_r}, expected {expected_r}"
        );
        assert!(
            (gl_g as i32 - expected_g as i32).abs() <= 1,
            "Pixel {idx} OpenGL Green mismatch: got {gl_g}, expected {expected_g}"
        );
        assert!(
            (gl_b as i32 - expected_b as i32).abs() <= 1,
            "Pixel {idx} OpenGL Blue mismatch: got {gl_b}, expected {expected_b}"
        );
        assert_eq!(gl_a, expected_a, "Pixel {idx} OpenGL Alpha mismatch");

        // Cross-backend parity: Vulkan vs OpenGL difference <= 1 texel unit
        assert!(
            (vk_r as i32 - gl_r as i32).abs() <= 1,
            "Cross-backend Red parity failure at pixel {idx}: Vulkan={vk_r}, OpenGL={gl_r}"
        );
        assert!(
            (vk_g as i32 - gl_g as i32).abs() <= 1,
            "Cross-backend Green parity failure at pixel {idx}: Vulkan={vk_g}, OpenGL={gl_g}"
        );
        assert!(
            (vk_b as i32 - gl_b as i32).abs() <= 1,
            "Cross-backend Blue parity failure at pixel {idx}: Vulkan={vk_b}, OpenGL={gl_b}"
        );
        assert_eq!(
            vk_a, gl_a,
            "Cross-backend Alpha parity failure at pixel {idx}: Vulkan={vk_a}, OpenGL={gl_a}"
        );
    }
    assert_no_api_errors(&vk_device);
    assert_no_api_errors(&gl_device);
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
