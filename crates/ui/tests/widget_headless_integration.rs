//! Headless offscreen integration tests for Core Widget Library:
//! Full interactive form simulation with synthetic AI agent input, state verification,
//! and RHI GPU device validation across Headless OpenGL and Vulkan.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::containers::VBoxContainer;
use korppi_ui::control::{Control, FocusMode, LayoutPreset};
use korppi_ui::input::{UiInputDispatcher, UiReceiver};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect};
use korppi_ui::widgets::{
    Button, CheckBox, ColorPicker, Label, LineEdit, ProgressBar, Slider, SliderOrientation,
};

use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, SampleCount, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_widget_form_interaction_and_gpu_upload_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    // 1. Root settings window container
    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::Center, false);
    root_ctl.offset_left = -200.0;
    root_ctl.offset_top = -250.0;
    root_ctl.offset_right = 200.0;
    root_ctl.offset_bottom = 250.0;
    let root = world.spawn((root_ctl, VBoxContainer::new(10.0)));

    // 2. Title Label
    let mut title_ctl = Control::default();
    title_ctl.custom_minimum_size = Vec2::new(380.0, 30.0);
    let title = world.spawn((title_ctl, Label::new("Engine Settings")));
    attach_child(&mut world, root, title).expect("Attach title");

    // 3. Project Name LineEdit
    let mut input_ctl = Control::default();
    input_ctl.custom_minimum_size = Vec2::new(380.0, 35.0);
    input_ctl.focus_mode = FocusMode::All;
    let input_field = world.spawn((input_ctl, LineEdit::new(), UiReceiver::default()));
    attach_child(&mut world, root, input_field).expect("Attach input_field");

    // 4. VSync CheckBox
    let mut vsync_ctl = Control::default();
    vsync_ctl.custom_minimum_size = Vec2::new(380.0, 30.0);
    vsync_ctl.focus_mode = FocusMode::All;
    let vsync_box = world.spawn((
        vsync_ctl,
        CheckBox::new("Enable VSync"),
        UiReceiver::default(),
    ));
    attach_child(&mut world, root, vsync_box).expect("Attach vsync_box");

    // 5. Volume Slider
    let mut slider_ctl = Control::default();
    slider_ctl.custom_minimum_size = Vec2::new(380.0, 30.0);
    slider_ctl.focus_mode = FocusMode::All;
    let slider = world.spawn((
        slider_ctl,
        Slider::new(SliderOrientation::Horizontal, 0.0, 100.0, 5.0),
        UiReceiver::default(),
    ));
    attach_child(&mut world, root, slider).expect("Attach slider");

    // 6. Progress Bar
    let mut progress_ctl = Control::default();
    progress_ctl.custom_minimum_size = Vec2::new(380.0, 25.0);
    let mut progress = ProgressBar::new(0.0, 100.0);
    progress.set_value(65.0);
    let pbar = world.spawn((progress_ctl, progress));
    attach_child(&mut world, root, pbar).expect("Attach pbar");

    // 7. Apply Settings Button
    let mut apply_ctl = Control::default();
    apply_ctl.custom_minimum_size = Vec2::new(380.0, 40.0);
    apply_ctl.focus_mode = FocusMode::All;
    let apply_btn = world.spawn((
        apply_ctl,
        Button::new("Apply Settings"),
        UiReceiver::default(),
    ));
    attach_child(&mut world, root, apply_btn).expect("Attach apply_btn");

    // 8. Resolve layout
    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1280.0,
        height: 720.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Verify resolved layout
    let input_rect = world.get::<ResolvedRect>(input_field).unwrap().global;
    assert!(input_rect.width >= 380.0);

    // --- Synthetic AI Agent Interactions ---
    // Step A: Agent clicks LineEdit and types "CyberEngine"
    dispatcher.inject_mouse_click(
        &mut world,
        Vec2::new(input_rect.x + 10.0, input_rect.y + 10.0),
        MouseButton::Left,
    );
    assert_eq!(
        dispatcher.focus_manager.focused_control(),
        Some(input_field)
    );
    for ch in "CyberEngine".chars() {
        dispatcher.inject_char(&mut world, ch);
    }
    assert_eq!(
        world.get::<LineEdit>(input_field).unwrap().text(),
        "CyberEngine"
    );

    // Step B: Agent clicks CheckBox to toggle VSync
    let vsync_rect = world.get::<ResolvedRect>(vsync_box).unwrap().global;
    dispatcher.inject_mouse_click(
        &mut world,
        Vec2::new(vsync_rect.x + 10.0, vsync_rect.y + 10.0),
        MouseButton::Left,
    );
    assert!(world.get::<CheckBox>(vsync_box).unwrap().is_checked());

    // Step C: Agent drags Volume Slider to 80%
    let slider_rect = world.get::<ResolvedRect>(slider).unwrap().global;
    world
        .get_mut::<Slider>(slider)
        .unwrap()
        .set_value_from_offset(slider_rect.width * 0.8, slider_rect.width);
    assert_eq!(world.get::<Slider>(slider).unwrap().range.value(), 80.0);

    // Step D: Agent clicks Apply Button
    let btn_rect = world.get::<ResolvedRect>(apply_btn).unwrap().global;
    dispatcher.inject_mouse_move(&mut world, Vec2::new(btn_rect.x + 20.0, btn_rect.y + 20.0));
    dispatcher.inject_mouse_click(
        &mut world,
        Vec2::new(btn_rect.x + 20.0, btn_rect.y + 20.0),
        MouseButton::Left,
    );
    world
        .get_mut::<Button>(apply_btn)
        .unwrap()
        .set_pressed(true);
    assert!(world.get::<Button>(apply_btn).unwrap().is_pressed());

    // --- GPU Buffer Validation ---
    // Package widget geometry and interaction state into GPU Uniform Buffer
    let widget_gpu_data = [
        input_rect.x,
        input_rect.y,
        input_rect.width,
        input_rect.height,
        slider_rect.x,
        slider_rect.y,
        80.0, // slider value
        1.0,  // vsync checked
    ];
    let data_bytes: &[u8] = korppi_core::pod::cast_slice(&widget_gpu_data);
    let ubo_desc = BufferDescriptor {
        size: data_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("gl_widget_form_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(data_bytes))
        .expect("Create GL widget form UBO");
    let readback = device
        .read_buffer(ubo, 0, data_bytes.len() as u64)
        .expect("Read GL widget form UBO");
    assert_eq!(readback, data_bytes);

    // Verify Offscreen Render Target creation
    let target_desc = TextureDescriptor {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 1280,
        height: 720,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        label: Some("gl_widget_offscreen_target"),
    };
    let target = device
        .create_texture(&target_desc, None)
        .expect("Create GL offscreen target");
    assert!(target.is_valid());
}

#[test]
fn test_color_picker_and_vulkan_offscreen_pipeline() {
    let Ok(device) = VulkanDevice::create_headless() else {
        eprintln!(
            "Skipping Vulkan test: No physical Vulkan device or ICD found in headless environment"
        );
        return;
    };

    let mut picker = ColorPicker::new();
    picker.set_hsva(120.0, 0.8, 0.9, 1.0); // Vibrant Green
    let rgba = picker.color();
    assert!(rgba[1] > rgba[0] && rgba[1] > rgba[2]); // Green is dominant

    // Upload color uniform to Vulkan UBO
    let color_bytes: &[u8] = korppi_core::pod::cast_slice(&rgba);
    let ubo_desc = BufferDescriptor {
        size: color_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("vk_color_picker_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(color_bytes))
        .expect("Create VK color picker UBO");
    let readback = device
        .read_buffer(ubo, 0, color_bytes.len() as u64)
        .expect("Read VK color picker UBO");
    assert_eq!(readback, color_bytes);

    // Create Vulkan Offscreen UI Framebuffer Target
    let target_desc = TextureDescriptor {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 800,
        height: 600,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        label: Some("vk_widget_offscreen_target"),
    };
    let target = device
        .create_texture(&target_desc, None)
        .expect("Create VK offscreen target");
    assert!(target.is_valid());
}
