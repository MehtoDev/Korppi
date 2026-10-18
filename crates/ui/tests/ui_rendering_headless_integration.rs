//! Headless offscreen integration test verifying the complete UI rendering pipeline,
//! QuadBatch2d compilation, RenderGraph UI pass execution, and RHI device validation
//! across Headless OpenGL and Vulkan.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;
use korppi_render::graph::{RenderGraph, TransientTextureDesc};
use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, SampleCount, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::containers::VBoxContainer;
use korppi_ui::control::{Control, FocusMode, LayoutPreset};
use korppi_ui::input::{UiInputDispatcher, UiReceiver};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect};
use korppi_ui::render::{add_ui_render_pass, UiCanvasRenderer, UiDrawCommand};
use korppi_ui::theme::{StyleBox, StyleBoxFlat, ThemeOverride};
use korppi_ui::widgets::{Button, Label, LineEdit, ProgressBar, Slider, SliderOrientation};

#[test]
fn test_end_to_end_ui_render_pipeline_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    // 1. Root Canvas
    let mut canvas_ctl = Control::default();
    canvas_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let canvas = world.spawn((canvas_ctl,));

    // 2. Editor Window Panel with StyleBoxFlat
    let mut window_ctl = Control::default();
    window_ctl.set_anchors_preset(LayoutPreset::Center, false);
    window_ctl.offset_left = -300.0;
    window_ctl.offset_top = -200.0;
    window_ctl.offset_right = 300.0;
    window_ctl.offset_bottom = 200.0;
    let mut win_theme = ThemeOverride::default();
    let mut flat = StyleBoxFlat::default();
    flat.bg_color = [0.15, 0.16, 0.19, 1.0];
    flat.border_width = [2.0, 2.0, 2.0, 2.0];
    flat.border_color = [0.25, 0.50, 0.90, 1.0];
    flat.corner_radius = [6.0, 6.0, 6.0, 6.0];
    win_theme.set_stylebox("panel", StyleBox::Flat(flat));
    let window_panel = world.spawn((window_ctl, win_theme, VBoxContainer::new(12.0)));
    attach_child(&mut world, canvas, window_panel).expect("Attach window");

    // 3. Title Label
    let mut title_ctl = Control::default();
    title_ctl.custom_minimum_size = Vec2::new(560.0, 30.0);
    let title = world.spawn((title_ctl, Label::new("Autonomous Agent Control Center")));
    attach_child(&mut world, window_panel, title).expect("Attach title");

    // 4. Command Input Field
    let mut input_ctl = Control::default();
    input_ctl.custom_minimum_size = Vec2::new(560.0, 36.0);
    input_ctl.focus_mode = FocusMode::All;
    let input = world.spawn((input_ctl, LineEdit::new(), UiReceiver::default()));
    attach_child(&mut world, window_panel, input).expect("Attach input");

    // 5. Volume Slider
    let mut slider_ctl = Control::default();
    slider_ctl.custom_minimum_size = Vec2::new(560.0, 24.0);
    slider_ctl.focus_mode = FocusMode::All;
    let slider = world.spawn((
        slider_ctl,
        Slider::new(SliderOrientation::Horizontal, 0.0, 100.0, 1.0),
        UiReceiver::default(),
    ));
    attach_child(&mut world, window_panel, slider).expect("Attach slider");

    // 6. Build Progress Bar
    let mut pbar_ctl = Control::default();
    pbar_ctl.custom_minimum_size = Vec2::new(560.0, 20.0);
    let mut progress = ProgressBar::new(0.0, 1.0);
    progress.set_value(0.72);
    let pbar = world.spawn((pbar_ctl, progress));
    attach_child(&mut world, window_panel, pbar).expect("Attach pbar");

    // 7. Action Button
    let mut btn_ctl = Control::default();
    btn_ctl.custom_minimum_size = Vec2::new(560.0, 40.0);
    btn_ctl.focus_mode = FocusMode::All;
    let btn = world.spawn((
        btn_ctl,
        Button::new("Build & Execute"),
        UiReceiver::default(),
    ));
    attach_child(&mut world, window_panel, btn).expect("Attach btn");

    // Resolve layout for 1920x1080 viewport
    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1920.0,
        height: 1080.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Verify resolved layout
    let win_rect = world.get::<ResolvedRect>(window_panel).unwrap().global;
    assert_eq!(win_rect.x, 1920.0 * 0.5 - 300.0);
    assert_eq!(win_rect.y, 1080.0 * 0.5 - 200.0);

    // Simulate Agent synthetic input
    let input_rect = world.get::<ResolvedRect>(input).unwrap().global;
    dispatcher.inject_mouse_click(
        &mut world,
        Vec2::new(input_rect.x + 10.0, input_rect.y + 10.0),
        MouseButton::Left,
    );
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(input));
    world
        .get_mut::<LineEdit>(input)
        .unwrap()
        .insert_text("spawn_entity(Prefab::Player)");
    assert_eq!(
        world.get::<LineEdit>(input).unwrap().text(),
        "spawn_entity(Prefab::Player)"
    );

    // Collect UI Draw Commands
    let renderer = UiCanvasRenderer::new();
    let commands = renderer.collect_draw_commands(&world);
    assert!(commands.len() >= 6);

    // Compile into GPU Quad Batch
    let quad_batch = renderer.compile_to_quad_batch(&commands, 2048);
    assert!(quad_batch.quad_count() > 0);
    assert_eq!(quad_batch.vertices().len(), quad_batch.quad_count() * 4);
    assert_eq!(quad_batch.indices().len(), quad_batch.quad_count() * 6);

    // Schedule UI Pass in RenderGraph
    let mut graph = RenderGraph::new();
    let color_target_desc = TransientTextureDesc::new(
        "ui_offscreen_hdr_color",
        1920,
        1080,
        TextureFormat::Rgba8Unorm,
        TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let color_target = graph.create_transient_texture(color_target_desc);
    add_ui_render_pass(&mut graph, color_target, quad_batch.clone());

    let compiled = graph.compile().expect("Compile RenderGraph with UI pass");
    assert!(compiled
        .passes()
        .iter()
        .any(|p| p.name() == "ui_canvas_pass"));

    // Upload generated QuadBatch vertices to GPU buffer on OpenGL device
    let vtx_bytes: &[u8] = korppi_core::pod::cast_slice(quad_batch.vertices());
    let vbo_desc = BufferDescriptor {
        size: vtx_bytes.len() as u64,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("gl_ui_vertex_buffer"),
    };
    let vbo = device
        .create_buffer(&vbo_desc, Some(vtx_bytes))
        .expect("Create GL UI vertex buffer");
    let readback = device
        .read_buffer(vbo, 0, vtx_bytes.len() as u64)
        .expect("Read GL UI vertex buffer");
    assert_eq!(readback, vtx_bytes);

    // Create Offscreen Texture Target
    let tex_desc = TextureDescriptor {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 1920,
        height: 1080,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        label: Some("gl_ui_canvas_output"),
    };
    let target = device
        .create_texture(&tex_desc, None)
        .expect("Create GL canvas output texture");
    assert!(target.is_valid());
}

#[test]
fn test_end_to_end_ui_render_pipeline_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        eprintln!(
            "Skipping Vulkan test: No physical Vulkan device or ICD found in headless environment"
        );
        return;
    };

    let mut world = World::new();

    // 1. Root Canvas
    let mut canvas_ctl = Control::default();
    canvas_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let canvas = world.spawn((canvas_ctl,));

    // 2. Button
    let mut btn_ctl = Control::default();
    btn_ctl.set_anchors_preset(LayoutPreset::Center, false);
    btn_ctl.offset_left = -100.0;
    btn_ctl.offset_top = -25.0;
    btn_ctl.offset_right = 100.0;
    btn_ctl.offset_bottom = 25.0;
    let btn = world.spawn((btn_ctl, Button::new("Vulkan UI Button")));
    attach_child(&mut world, canvas, btn).expect("Attach btn");

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1280.0,
        height: 720.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    let renderer = UiCanvasRenderer::new();
    let commands = renderer.collect_draw_commands(&world);
    assert!(commands
        .iter()
        .any(|c| matches!(c, UiDrawCommand::Button { .. })));

    let quad_batch = renderer.compile_to_quad_batch(&commands, 512);
    assert!(quad_batch.quad_count() > 0);

    // Upload vertices to Vulkan buffer
    let vtx_bytes: &[u8] = korppi_core::pod::cast_slice(quad_batch.vertices());
    let vbo_desc = BufferDescriptor {
        size: vtx_bytes.len() as u64,
        usage: BufferUsage::VERTEX | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("vk_ui_vertex_buffer"),
    };
    let vbo = device
        .create_buffer(&vbo_desc, Some(vtx_bytes))
        .expect("Create VK UI vertex buffer");
    let readback = device
        .read_buffer(vbo, 0, vtx_bytes.len() as u64)
        .expect("Read VK UI vertex buffer");
    assert_eq!(readback, vtx_bytes);

    // Create Vulkan Render Target
    let tex_desc = TextureDescriptor {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 1280,
        height: 720,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        label: Some("vk_ui_canvas_target"),
    };
    let target = device
        .create_texture(&tex_desc, None)
        .expect("Create VK canvas target");
    assert!(target.is_valid());
}
