//! Headless offscreen integration tests for UI Event Routing, Focus Management,
//! and Synthetic Agent Input Injection on Headless OpenGL and Vulkan devices.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_input::keyboard::KeyCode;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::control::{Control, FocusMode, LayoutPreset, MouseFilter};
use korppi_ui::input::{FocusDirection, FocusManager, UiEvent, UiInputDispatcher, UiReceiver};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect};

use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, SampleCount, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_agent_synthetic_input_workflow_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut world = World::new();
    let mut dispatcher = UiInputDispatcher::default();

    // 1. Root Dialog Window (0, 0, 400, 300)
    let mut dialog_ctl = Control::default();
    dialog_ctl.set_anchors_preset(LayoutPreset::Center, false);
    dialog_ctl.offset_left = -200.0;
    dialog_ctl.offset_top = -150.0;
    dialog_ctl.offset_right = 200.0;
    dialog_ctl.offset_bottom = 150.0;
    dialog_ctl.mouse_filter = MouseFilter::Stop;
    let dialog = world.spawn((dialog_ctl, UiReceiver::default()));

    // 2. LineEdit Input at top of dialog
    let mut input_ctl = Control::default();
    input_ctl.offset_left = 20.0;
    input_ctl.offset_top = 40.0;
    input_ctl.offset_right = 380.0;
    input_ctl.offset_bottom = 80.0;
    input_ctl.focus_mode = FocusMode::All;
    let input_field = world.spawn((input_ctl, UiReceiver::default()));
    attach_child(&mut world, dialog, input_field).expect("Attach input_field");

    // 3. OK Button
    let mut ok_ctl = Control::default();
    ok_ctl.offset_left = 60.0;
    ok_ctl.offset_top = 220.0;
    ok_ctl.offset_right = 160.0;
    ok_ctl.offset_bottom = 260.0;
    ok_ctl.focus_mode = FocusMode::All;
    let ok_btn = world.spawn((ok_ctl, UiReceiver::default()));
    attach_child(&mut world, dialog, ok_btn).expect("Attach ok_btn");

    // 4. Cancel Button
    let mut cancel_ctl = Control::default();
    cancel_ctl.offset_left = 240.0;
    cancel_ctl.offset_top = 220.0;
    cancel_ctl.offset_right = 340.0;
    cancel_ctl.offset_bottom = 260.0;
    cancel_ctl.focus_mode = FocusMode::All;
    let cancel_btn = world.spawn((cancel_ctl, UiReceiver::default()));
    attach_child(&mut world, dialog, cancel_btn).expect("Attach cancel_btn");

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 800.0,
        height: 600.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Dialog global origin: (800*0.5 - 200, 600*0.5 - 150) = (200, 150)
    let dialog_rect = world.get::<ResolvedRect>(dialog).unwrap().global;
    assert_eq!(dialog_rect.x, 200.0);
    assert_eq!(dialog_rect.y, 150.0);

    // Input field global rect: (220, 190, 360, 40)
    let input_rect = world.get::<ResolvedRect>(input_field).unwrap().global;
    assert_eq!(input_rect.x, 220.0);
    assert_eq!(input_rect.y, 190.0);

    // Step A: Autonomous AI Agent hovers over OK Button (global 260..360, 370..410)
    dispatcher.inject_mouse_move(&mut world, Vec2::new(300.0, 390.0));
    assert!(world
        .get::<UiReceiver>(ok_btn)
        .unwrap()
        .events
        .contains(&UiEvent::MouseEnter));

    // Step B: Agent clicks inside Input Field (global 250, 200) to focus it
    dispatcher.inject_mouse_click(&mut world, Vec2::new(250.0, 200.0), MouseButton::Left);
    assert_eq!(
        dispatcher.focus_manager.focused_control(),
        Some(input_field)
    );
    assert!(world
        .get::<UiReceiver>(input_field)
        .unwrap()
        .events
        .contains(&UiEvent::FocusGained));

    // Step C: Agent types text "Search" into input field
    for ch in "Search".chars() {
        dispatcher.inject_char(&mut world, ch);
    }
    let input_events = world.get::<UiReceiver>(input_field).unwrap().events.clone();
    let char_count = input_events
        .iter()
        .filter(|e| matches!(e, UiEvent::CharInput { .. }))
        .count();
    assert_eq!(char_count, 6);

    // Step D: Agent hits Tab to advance focus to OK button
    dispatcher.inject_tab(&mut world, false);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(ok_btn));

    // Step E: Agent hits Tab again to advance focus to Cancel button
    dispatcher.inject_tab(&mut world, false);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(cancel_btn));

    // Step F: Agent hits Shift+Tab to return focus to OK button
    dispatcher.inject_tab(&mut world, true);
    assert_eq!(dispatcher.focus_manager.focused_control(), Some(ok_btn));

    // Step G: Agent hits Enter key on OK button
    dispatcher.inject_key(&mut world, KeyCode::Enter, true);
    dispatcher.inject_key(&mut world, KeyCode::Enter, false);
    let ok_events = world.get::<UiReceiver>(ok_btn).unwrap().events.clone();
    assert!(ok_events.contains(&UiEvent::KeyDown {
        key: KeyCode::Enter
    }));
    assert!(ok_events.contains(&UiEvent::KeyUp {
        key: KeyCode::Enter
    }));

    // Step H: Offscreen RHI validation - upload interaction debug buffer to GPU
    let debug_data = [
        input_rect.x,
        input_rect.y,
        input_rect.width,
        input_rect.height,
        300.0, // Cursor X
        390.0, // Cursor Y
        1.0,   // Clicked flag
        1.0,   // Focused flag
    ];
    let debug_bytes: &[u8] = korppi_core::pod::cast_slice(&debug_data);
    let buf_desc = BufferDescriptor {
        size: debug_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("ui_interaction_debug_ubo"),
    };
    let ubo = device
        .create_buffer(&buf_desc, Some(debug_bytes))
        .expect("Create GL UI debug buffer");
    let readback = device
        .read_buffer(ubo, 0, debug_bytes.len() as u64)
        .expect("Read GL UI debug buffer");
    assert_eq!(readback, debug_bytes);
}

#[test]
fn test_agent_directional_navigation_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        eprintln!(
            "Skipping Vulkan test: No physical Vulkan device or ICD found in headless environment"
        );
        return;
    };

    let mut world = World::new();
    let mut focus_mgr = FocusManager::default();

    // Spawn 3x3 grid of focusable controls
    let mut grid_entities = Vec::new();
    for row in 0..3 {
        for col in 0..3 {
            let mut ctl = Control::default();
            ctl.offset_left = col as f32 * 100.0;
            ctl.offset_top = row as f32 * 80.0;
            ctl.offset_right = ctl.offset_left + 80.0;
            ctl.offset_bottom = ctl.offset_top + 60.0;
            ctl.focus_mode = FocusMode::All;
            let e = world.spawn((ctl, UiReceiver::default()));
            grid_entities.push(e);
        }
    }

    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1024.0,
        height: 768.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Initial focus on Center (1, 1) -> index 4
    let center_e = grid_entities[4];
    focus_mgr.set_focus(&mut world, Some(center_e));
    assert_eq!(focus_mgr.focused_control(), Some(center_e));

    // Navigate Left -> (1, 0) index 3
    focus_mgr.navigate_direction(&mut world, FocusDirection::Left);
    assert_eq!(focus_mgr.focused_control(), Some(grid_entities[3]));

    // Navigate Up -> (0, 0) index 0
    focus_mgr.navigate_direction(&mut world, FocusDirection::Up);
    assert_eq!(focus_mgr.focused_control(), Some(grid_entities[0]));

    // Navigate Right -> (0, 1) index 1
    focus_mgr.navigate_direction(&mut world, FocusDirection::Right);
    assert_eq!(focus_mgr.focused_control(), Some(grid_entities[1]));

    // Navigate Down -> (1, 1) index 4 (back to center)
    focus_mgr.navigate_direction(&mut world, FocusDirection::Down);
    assert_eq!(focus_mgr.focused_control(), Some(grid_entities[4]));

    // Verify Vulkan Offscreen target creation
    let target_desc = TextureDescriptor {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 1024,
        height: 768,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        label: Some("vk_ui_offscreen_target"),
    };
    let offscreen_tex = device
        .create_texture(&target_desc, None)
        .expect("Create VK offscreen UI texture");
    assert!(offscreen_tex.is_valid());
}
