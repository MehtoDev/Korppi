//! Unit tests for 2D Batched Rendering, Texture Atlasing, 9-Patch Slicing, and Canvas Scaling.

use korppi_math::Vec2;
use korppi_render::render2d::{
    compute_9patch_quads, compute_canvas_transform, CanvasScaleMode, Light2d, NinePatchMargins,
    Occluder2d, QuadBatch2d, Rect2d, TextureAtlas2d,
};
use korppi_render::{PassType, RenderGraph, TransientTextureDesc};
use korppi_rhi::{LoadOp, SampleCount, StoreOp, TextureDimension, TextureFormat, TextureUsage};

#[test]
fn test_quad_batch_builder_and_index_generation() {
    let mut batch = QuadBatch2d::new(100);

    // Add 3 quads
    for i in 0..3 {
        let min = Vec2::new(i as f32 * 10.0, 0.0);
        let max = Vec2::new((i + 1) as f32 * 10.0, 10.0);
        let uv_min = Vec2::ZERO;
        let uv_max = Vec2::ONE;
        let color = [1.0, 1.0, 1.0, 1.0];
        batch.push_quad(min, max, uv_min, uv_max, color);
    }

    assert_eq!(batch.quad_count(), 3);
    assert_eq!(batch.vertices().len(), 12);
    assert_eq!(batch.indices().len(), 18);

    // Verify index patterns: [0, 1, 2, 2, 3, 0], [4, 5, 6, 6, 7, 4], [8, 9, 10, 10, 11, 8]
    assert_eq!(&batch.indices()[0..6], &[0, 1, 2, 2, 3, 0]);
    assert_eq!(&batch.indices()[6..12], &[4, 5, 6, 6, 7, 4]);
    assert_eq!(&batch.indices()[12..18], &[8, 9, 10, 10, 11, 8]);

    // Clear batch
    batch.clear();
    assert_eq!(batch.quad_count(), 0);
    assert!(batch.vertices().is_empty());
    assert!(batch.indices().is_empty());
}

#[test]
fn test_texture_atlas_uv_mapping_and_9patch() {
    let mut atlas = TextureAtlas2d::new(1024, 1024);
    let region = Rect2d {
        x: 64.0,
        y: 128.0,
        width: 256.0,
        height: 128.0,
    };
    let id = atlas.add_region("button_normal", region);
    assert_eq!(id, 0);

    let uv = atlas.get_uv(id).expect("UV for region 0");
    // [64 / 1024, 128 / 1024, (64 + 256) / 1024, (128 + 128) / 1024]
    assert!((uv.x - 0.0625).abs() < 1e-5);
    assert!((uv.y - 0.125).abs() < 1e-5);
    assert!((uv.z - 0.3125).abs() < 1e-5);
    assert!((uv.w - 0.25).abs() < 1e-5);

    // 9-Patch slicing
    let margins = NinePatchMargins {
        left: 16.0,
        right: 16.0,
        top: 16.0,
        bottom: 16.0,
    };
    let dest_rect = Rect2d {
        x: 100.0,
        y: 100.0,
        width: 400.0,
        height: 200.0,
    };

    let quads = compute_9patch_quads(region, uv, margins, dest_rect);
    assert_eq!(quads.len(), 9, "9-patch must generate exactly 9 sub-quads");

    // Corner 0 (top-left) must have width 16 and height 16
    let top_left = &quads[0];
    assert!((top_left.dest.width - 16.0).abs() < 1e-4);
    assert!((top_left.dest.height - 16.0).abs() < 1e-4);

    // Center patch (index 4) must stretch over destination minus margins
    let center = &quads[4];
    assert!((center.dest.width - (400.0 - 32.0)).abs() < 1e-4);
    assert!((center.dest.height - (200.0 - 32.0)).abs() < 1e-4);
}

#[test]
fn test_canvas_scaling_modes() {
    let virtual_size = Vec2::new(800.0, 600.0);

    // 1. Stretch mode
    let window_size = Vec2::new(1920.0, 1080.0);
    let stretch = compute_canvas_transform(virtual_size, window_size, CanvasScaleMode::Stretch);
    assert_eq!(stretch.viewport_pos, Vec2::ZERO);
    assert_eq!(stretch.viewport_size, window_size);

    // 2. KeepAspect mode (Aspect 800/600 = 1.333, Window 1920/1080 = 1.777 -> pillarbox)
    let keep_aspect =
        compute_canvas_transform(virtual_size, window_size, CanvasScaleMode::KeepAspect);
    assert_eq!(keep_aspect.viewport_size.y, 1080.0);
    // scale = 1080 / 600 = 1.8. width = 800 * 1.8 = 1440.0
    assert!((keep_aspect.viewport_size.x - 1440.0).abs() < 1e-3);
    // pillarbox offset x = (1920 - 1440) / 2 = 240.0
    assert!((keep_aspect.viewport_pos.x - 240.0).abs() < 1e-3);
    assert_eq!(keep_aspect.viewport_pos.y, 0.0);

    // 3. IntegerRatio mode
    let int_scale =
        compute_canvas_transform(virtual_size, window_size, CanvasScaleMode::IntegerRatio);
    assert_eq!(int_scale.scale, 1.0); // 2x would be 1600x1200 > 1080, so 1x fits
    assert_eq!(int_scale.viewport_size, virtual_size);
    assert!((int_scale.viewport_pos.x - (1920.0 - 800.0) * 0.5).abs() < 1e-3);
    assert!((int_scale.viewport_pos.y - (1080.0 - 600.0) * 0.5).abs() < 1e-3);

    // Coordinate roundtrip: virtual center (400, 300) in KeepAspect maps to window center
    let screen_pos = keep_aspect.virtual_to_screen(Vec2::new(400.0, 300.0));
    assert!((screen_pos.x - 1920.0 * 0.5).abs() < 1e-3);
    assert!((screen_pos.y - 1080.0 * 0.5).abs() < 1e-3);

    let virtual_pos = keep_aspect.screen_to_virtual(screen_pos);
    assert!((virtual_pos.x - 400.0).abs() < 1e-3);
    assert!((virtual_pos.y - 300.0).abs() < 1e-3);
}

#[test]
fn test_2d_occluder_shadow_fin_extrusion() {
    let light = Light2d {
        position: Vec2::ZERO,
        radius: 100.0,
        color: [1.0, 1.0, 1.0, 1.0],
        energy: 1.0,
    };

    let occluder = Occluder2d {
        p0: Vec2::new(10.0, 0.0),
        p1: Vec2::new(20.0, 0.0),
    };

    let fin = occluder.extrude_shadow(&light);
    assert_eq!(fin.len(), 4, "Shadow fin must form a 4-vertex quad");
    assert_eq!(fin[0], occluder.p0);
    assert_eq!(fin[1], occluder.p1);
    // Extruded points must be at distance ~ light radius
    assert!((fin[2].length() - 100.0).abs() < 1e-3);
    assert!((fin[3].length() - 100.0).abs() < 1e-3);
}

#[test]
fn test_2d_render_graph_pipeline() {
    let mut graph = RenderGraph::new();

    let canvas_target = graph.create_transient_texture(TransientTextureDesc {
        name: "Canvas2dTarget".to_string(),
        width: 1280,
        height: 720,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        dimension: TextureDimension::D2,
        sample_count: SampleCount::Count1,
    });

    graph
        .add_pass("Batched2dSpritePass", PassType::Graphics)
        .write_color(
            canvas_target,
            0,
            LoadOp::Clear,
            StoreOp::Store,
            [0.1, 0.1, 0.1, 1.0],
        )
        .mark_as_root()
        .record(|_ctx, cmd| {
            cmd.set_viewport(0.0, 0.0, 1280.0, 720.0, 0.0, 1.0);
            cmd.set_scissor(0, 0, 1280, 720);
        });

    let compiled = graph.compile().expect("2D canvas graph must compile");
    assert_eq!(compiled.passes().len(), 1);
}
