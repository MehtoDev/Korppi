//! Benchmark and verification test verifying zero FBO churn in OpenGlDevice dynamic rendering.

use korppi_rhi::{
    GraphicsDevice, LoadOp, RenderingAttachment, RenderingDescriptor, StoreOp, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;

#[test]
fn test_opengl_fbo_caching_and_zero_churn() {
    let device = OpenGlDevice::create_headless().expect("Headless OpenGL device creation failed");

    let width = 64;
    let height = 64;
    let tex_desc = TextureDescriptor {
        width,
        height,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: korppi_rhi::SampleCount::Count1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::SAMPLED,
        dimension: TextureDimension::D2,
        label: Some("fbo_cache_test_tex"),
    };
    let tex = device
        .create_texture(&tex_desc, None)
        .expect("Create texture failed");

    let render_desc = RenderingDescriptor {
        width,
        height,
        offset_x: 0,
        offset_y: 0,
        color_attachments: korppi_rhi::smallvec![RenderingAttachment {
            texture: tex,
            mip_level: 0,
            layer: 0,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_color: Some([0.1, 0.2, 0.3, 1.0]),
            resolve_target: None,
        }],
        depth_attachment: None,
    };

    assert_eq!(device.fbo_cache_len(), 0, "Initial cache must be empty");

    // Execute 100 passes using the same attachment configuration
    for _ in 0..100 {
        let mut recorder = device
            .create_command_recorder()
            .expect("Create recorder failed");
        recorder.begin().expect("Begin failed");
        recorder
            .begin_rendering(&render_desc)
            .expect("Begin rendering failed");
        recorder.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        recorder.set_scissor(0, 0, width, height);
        recorder.end_rendering().expect("End rendering failed");
        let cmd = recorder.finish().expect("Finish failed");
        device.submit(cmd).expect("Submit failed");
    }

    device.wait_idle().expect("Wait idle failed");

    // The FBO must be retained in the cache rather than deleted and recreated 100 times!
    assert_eq!(
        device.fbo_cache_len(),
        1,
        "Exactly 1 FBO must be created and reused across all passes without churn"
    );

    device.destroy_texture(tex).expect("Destroy texture failed");
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
