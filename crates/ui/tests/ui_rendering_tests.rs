//! Unit tests for UI Canvas Rendering, Draw Command Generation, Scissor Clipping,
//! QuadBatch2d Compilation, and Render Graph UI Pass Scheduling.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_math::Vec2;
use korppi_render::graph::{RenderGraph, TransientTextureDesc};
use korppi_render::render2d::QuadBatch2d;
use korppi_rhi::{TextureFormat, TextureUsage};
use korppi_scene::hierarchy::attach_child;
use korppi_ui::control::{Control, LayoutPreset};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d};
use korppi_ui::render::{add_ui_render_pass, UiCanvasRenderer, UiDrawCommand};
use korppi_ui::theme::{StyleBox, StyleBoxFlat, ThemeOverride};
use korppi_ui::widgets::{Button, Label, ProgressBar};

#[test]
fn test_ui_canvas_renderer_command_generation() {
    let mut world = World::new();

    // 1. Root control
    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world.spawn((root_ctl,));

    // 2. Panel with StyleBoxFlat
    let mut panel_ctl = Control::default();
    panel_ctl.offset_left = 50.0;
    panel_ctl.offset_top = 50.0;
    panel_ctl.offset_right = 450.0;
    panel_ctl.offset_bottom = 350.0;
    let mut panel_override = ThemeOverride::default();
    let mut flat = StyleBoxFlat::default();
    flat.bg_color = [0.2, 0.2, 0.2, 1.0];
    flat.border_width = [2.0, 2.0, 2.0, 2.0];
    flat.border_color = [0.4, 0.4, 0.8, 1.0];
    panel_override.set_stylebox("panel", StyleBox::Flat(flat));
    let panel = world.spawn((panel_ctl, panel_override));
    attach_child(&mut world, root, panel).expect("Attach panel");

    // 3. Button
    let mut btn_ctl = Control::default();
    btn_ctl.offset_left = 20.0;
    btn_ctl.offset_top = 20.0;
    btn_ctl.offset_right = 160.0;
    btn_ctl.offset_bottom = 60.0;
    let btn = world.spawn((btn_ctl, Button::new("Click")));
    attach_child(&mut world, panel, btn).expect("Attach btn");

    // 4. Label
    let mut label_ctl = Control::default();
    label_ctl.offset_left = 20.0;
    label_ctl.offset_top = 80.0;
    label_ctl.offset_right = 200.0;
    label_ctl.offset_bottom = 110.0;
    let label = world.spawn((label_ctl, Label::new("Status: Ready")));
    attach_child(&mut world, panel, label).expect("Attach label");

    // 5. ProgressBar
    let mut pbar_ctl = Control::default();
    pbar_ctl.offset_left = 20.0;
    pbar_ctl.offset_top = 130.0;
    pbar_ctl.offset_right = 300.0;
    pbar_ctl.offset_bottom = 150.0;
    let mut pbar = ProgressBar::new(0.0, 100.0);
    pbar.set_value(60.0);
    let pbar_ent = world.spawn((pbar_ctl, pbar));
    attach_child(&mut world, panel, pbar_ent).expect("Attach pbar");

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1024.0,
        height: 768.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    let renderer = UiCanvasRenderer::new();
    let commands = renderer.collect_draw_commands(&world);

    assert!(!commands.is_empty());
    // Verify panel stylebox command exists
    assert!(commands
        .iter()
        .any(|cmd| matches!(cmd, UiDrawCommand::StyleBoxFlat { .. })));
    // Verify button command exists
    assert!(commands
        .iter()
        .any(|cmd| matches!(cmd, UiDrawCommand::Button { .. })));
    // Verify label command exists
    assert!(commands
        .iter()
        .any(|cmd| matches!(cmd, UiDrawCommand::Text { .. })));
    // Verify progress bar command exists
    assert!(commands
        .iter()
        .any(|cmd| matches!(cmd, UiDrawCommand::ProgressBar { .. })));
}

#[test]
fn test_ui_scissor_stack_clipping() {
    let mut world = World::new();

    // Scroll container with clip_contents = true
    let mut scroll_ctl = Control::default();
    scroll_ctl.offset_left = 100.0;
    scroll_ctl.offset_top = 100.0;
    scroll_ctl.offset_right = 300.0;
    scroll_ctl.offset_bottom = 300.0;
    scroll_ctl.clip_contents = true;
    let scroll = world.spawn((scroll_ctl,));

    // Child extending beyond scroll container bounds
    let mut child_ctl = Control::default();
    child_ctl.offset_left = 50.0;
    child_ctl.offset_top = 50.0;
    child_ctl.offset_right = 400.0;
    child_ctl.offset_bottom = 400.0;
    let child = world.spawn((child_ctl, Label::new("Overflow Content")));
    attach_child(&mut world, scroll, child).expect("Attach child");

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1024.0,
        height: 768.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    let renderer = UiCanvasRenderer::new();
    let commands = renderer.collect_draw_commands(&world);

    let child_cmd = commands
        .iter()
        .find(|cmd| matches!(cmd, UiDrawCommand::Text { text, .. } if text == "Overflow Content"))
        .expect("Child text command not found");

    match child_cmd {
        UiDrawCommand::Text { scissor, .. } => {
            assert!(scissor.is_some());
            let sc = scissor.unwrap();
            // Scissor must be clamped to scroll container (100, 100, 200, 200)
            assert_eq!(sc.x, 100.0);
            assert_eq!(sc.y, 100.0);
            assert_eq!(sc.width, 200.0);
            assert_eq!(sc.height, 200.0);
        }
        _ => panic!("Expected text command"),
    }
}

#[test]
fn test_compile_to_quad_batch() {
    let commands = vec![
        UiDrawCommand::Quad {
            rect: Rect2d::new(10.0, 10.0, 100.0, 50.0),
            uv_min: Vec2::ZERO,
            uv_max: Vec2::ONE,
            color: [1.0, 0.0, 0.0, 1.0],
            scissor: None,
        },
        UiDrawCommand::Quad {
            rect: Rect2d::new(120.0, 10.0, 100.0, 50.0),
            uv_min: Vec2::ZERO,
            uv_max: Vec2::ONE,
            color: [0.0, 1.0, 0.0, 1.0],
            scissor: None,
        },
    ];

    let renderer = UiCanvasRenderer::new();
    let batch = renderer.compile_to_quad_batch(&commands, 1000);

    assert_eq!(batch.quad_count(), 2);
    assert_eq!(batch.vertices().len(), 8);
    assert_eq!(batch.indices().len(), 12);
}

#[test]
fn test_render_graph_ui_pass_scheduling() {
    let mut graph = RenderGraph::new();

    // Create virtual color output target
    let target_desc = TransientTextureDesc::new(
        "color_target",
        1280,
        720,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let color_target = graph.create_transient_texture(target_desc);

    let batch = QuadBatch2d::new(512);
    add_ui_render_pass(&mut graph, color_target, batch);

    let compiled = graph.compile().expect("Compile RenderGraph with UI pass");
    assert!(!compiled.passes().is_empty());
    assert!(compiled
        .passes()
        .iter()
        .any(|p| p.name() == "ui_canvas_pass"));
}
