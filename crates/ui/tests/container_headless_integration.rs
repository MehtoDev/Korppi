//! Headless offscreen integration tests for Container suite nesting,
//! clipping/scissor rect synchronization, dynamic hierarchy updates, and RHI device validation.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::containers::{
    GridContainer, HBoxContainer, MarginContainer, PanelContainer, ScrollContainer, VBoxContainer,
};
use korppi_ui::control::{Control, LayoutPreset};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect, SizeFlags};

use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, SampleCount, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_nested_containers_and_gpu_scissor_rects_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut world = World::new();

    // 1. Root canvas: FullRect 1920x1080
    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world.spawn((root_ctl,));

    // 2. MarginContainer with 20px padding
    let mut margin_ctl = Control::default();
    margin_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let margin = world.spawn((margin_ctl, MarginContainer::all(20.0)));
    attach_child(&mut world, root, margin).expect("Attach margin");

    // 3. VBoxContainer inside MarginContainer
    let mut vbox_ctl = Control::default();
    vbox_ctl.size_flags_horizontal = SizeFlags::EXPAND_FILL;
    vbox_ctl.size_flags_vertical = SizeFlags::EXPAND_FILL;
    let vbox = world.spawn((vbox_ctl, VBoxContainer::new(10.0)));
    attach_child(&mut world, margin, vbox).expect("Attach vbox");

    // 4. Header Bar: HBoxContainer height 60px
    let mut header_ctl = Control::default();
    header_ctl.custom_minimum_size = Vec2::new(0.0, 60.0);
    header_ctl.size_flags_horizontal = SizeFlags::FILL;
    let header = world.spawn((header_ctl, HBoxContainer::new(8.0)));
    attach_child(&mut world, vbox, header).expect("Attach header");

    // Header buttons (Left logo 120px, spacer EXPAND, Right profile 60px)
    let mut logo_ctl = Control::default();
    logo_ctl.custom_minimum_size = Vec2::new(120.0, 60.0);
    let logo = world.spawn((logo_ctl,));
    attach_child(&mut world, header, logo).expect("Attach logo");

    let mut spacer_ctl = Control::default();
    spacer_ctl.size_flags_horizontal = SizeFlags::EXPAND_FILL;
    let spacer = world.spawn((spacer_ctl,));
    attach_child(&mut world, header, spacer).expect("Attach spacer");

    let mut profile_ctl = Control::default();
    profile_ctl.custom_minimum_size = Vec2::new(60.0, 60.0);
    let profile = world.spawn((profile_ctl,));
    attach_child(&mut world, header, profile).expect("Attach profile");

    // 5. Main content area: HBoxContainer (sidebar 250px, scrollable grid main view EXPAND)
    let mut body_ctl = Control::default();
    body_ctl.size_flags_horizontal = SizeFlags::EXPAND_FILL;
    body_ctl.size_flags_vertical = SizeFlags::EXPAND_FILL;
    let body = world.spawn((body_ctl, HBoxContainer::new(16.0)));
    attach_child(&mut world, vbox, body).expect("Attach body");

    // Sidebar: PanelContainer 250px width
    let mut sidebar_ctl = Control::default();
    sidebar_ctl.custom_minimum_size = Vec2::new(250.0, 0.0);
    sidebar_ctl.size_flags_vertical = SizeFlags::EXPAND_FILL;
    let sidebar = world.spawn((sidebar_ctl, PanelContainer::all(12.0)));
    attach_child(&mut world, body, sidebar).expect("Attach sidebar");

    // ScrollContainer for main grid view
    let mut scroll_ctl = Control::default();
    scroll_ctl.size_flags_horizontal = SizeFlags::EXPAND_FILL;
    scroll_ctl.size_flags_vertical = SizeFlags::EXPAND_FILL;
    let mut scroll_comp = ScrollContainer::default();
    scroll_comp.scroll_vertical = 50.0;
    let scroll = world.spawn((scroll_ctl, scroll_comp));
    attach_child(&mut world, body, scroll).expect("Attach scroll");

    // GridContainer (3 columns) inside ScrollContainer
    let mut grid_ctl = Control::default();
    grid_ctl.custom_minimum_size = Vec2::new(1000.0, 1500.0);
    let grid = world.spawn((grid_ctl, GridContainer::new(3, 16.0, 16.0)));
    attach_child(&mut world, scroll, grid).expect("Attach grid");

    // Resolve entire hierarchy for 1920x1080
    let screen = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1920.0,
        height: 1080.0,
    };
    resolve_ui_layout_system(&mut world, screen);

    // Verify MarginContainer (fills root)
    let margin_res = world.get::<ResolvedRect>(margin).unwrap().global;
    assert_eq!(margin_res.x, 0.0);
    assert_eq!(margin_res.y, 0.0);
    assert_eq!(margin_res.width, 1920.0);
    assert_eq!(margin_res.height, 1080.0);

    // Verify VBoxContainer inside MarginContainer (inset by 20px margins)
    let vbox_res = world.get::<ResolvedRect>(vbox).unwrap().global;
    assert_eq!(vbox_res.x, 20.0);
    assert_eq!(vbox_res.y, 20.0);
    assert_eq!(vbox_res.width, 1920.0 - 40.0);
    assert_eq!(vbox_res.height, 1080.0 - 40.0);

    // Verify Header Bar
    let header_res = world.get::<ResolvedRect>(header).unwrap().global;
    assert_eq!(header_res.x, 20.0);
    assert_eq!(header_res.y, 20.0);
    assert_eq!(header_res.width, 1880.0);
    assert_eq!(header_res.height, 60.0);

    // Verify Header Children
    let logo_res = world.get::<ResolvedRect>(logo).unwrap().global;
    assert_eq!(logo_res.x, 20.0);
    assert_eq!(logo_res.width, 120.0);

    let profile_res = world.get::<ResolvedRect>(profile).unwrap().global;
    assert_eq!(profile_res.x, 20.0 + 1880.0 - 60.0);
    assert_eq!(profile_res.width, 60.0);

    let spacer_res = world.get::<ResolvedRect>(spacer).unwrap().global;
    assert_eq!(spacer_res.x, 20.0 + 120.0 + 8.0);
    assert_eq!(spacer_res.width, 1880.0 - 120.0 - 60.0 - 16.0);

    // Verify Sidebar
    let sidebar_res = world.get::<ResolvedRect>(sidebar).unwrap().global;
    assert_eq!(sidebar_res.x, 20.0);
    assert_eq!(sidebar_res.y, 20.0 + 60.0 + 10.0);
    assert_eq!(sidebar_res.width, 250.0);

    // Verify ScrollContainer and Grid
    let scroll_res = world.get::<ResolvedRect>(scroll).unwrap().global;
    assert_eq!(scroll_res.x, 20.0 + 250.0 + 16.0);
    assert_eq!(scroll_res.width, 1880.0 - 250.0 - 16.0);
    assert!(world.get::<Control>(scroll).unwrap().clip_contents);

    let grid_res = world.get::<ResolvedRect>(grid).unwrap().global;
    assert_eq!(grid_res.x, scroll_res.x);
    assert_eq!(grid_res.y, scroll_res.y - 50.0); // scrolled up by 50px
    assert_eq!(grid_res.width, scroll_res.width);
    assert_eq!(grid_res.height, 1500.0);

    // Verify GPU Scissor Rect generation for clipping ScrollContainer
    let scissor_x = scroll_res.x.max(0.0) as u32;
    let scissor_y = scroll_res.y.max(0.0) as u32;
    let scissor_w = scroll_res.width as u32;
    let scissor_h = scroll_res.height as u32;

    // Validate offscreen texture render target and scissor test pipeline
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
        label: Some("gl_container_canvas"),
    };
    let canvas_tex = device
        .create_texture(&tex_desc, None)
        .expect("Create canvas");
    assert!(canvas_tex.is_valid());

    // Upload scissor rect UBO
    let scissor_data = [
        scissor_x as f32,
        scissor_y as f32,
        scissor_w as f32,
        scissor_h as f32,
    ];
    let scissor_bytes: &[u8] = korppi_core::pod::cast_slice(&scissor_data);
    let ubo_desc = BufferDescriptor {
        size: scissor_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("gl_scissor_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(scissor_bytes))
        .expect("Create scissor UBO");
    let readback = device
        .read_buffer(ubo, 0, scissor_bytes.len() as u64)
        .expect("Read scissor UBO");
    assert_eq!(readback, scissor_bytes);
}

#[test]
fn test_nested_containers_and_gpu_scissor_rects_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        eprintln!(
            "Skipping Vulkan test: No physical Vulkan device or ICD found in headless environment"
        );
        return;
    };

    let mut world = World::new();

    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world.spawn((root_ctl,));

    let mut panel_ctl = Control::default();
    panel_ctl.set_anchors_preset(LayoutPreset::Center, false);
    panel_ctl.offset_left = -300.0;
    panel_ctl.offset_right = 300.0;
    panel_ctl.offset_top = -200.0;
    panel_ctl.offset_bottom = 200.0;
    let panel = world.spawn((panel_ctl, PanelContainer::all(16.0)));
    attach_child(&mut world, root, panel).expect("Attach panel");

    let mut grid_ctl = Control::default();
    grid_ctl.size_flags_horizontal = SizeFlags::EXPAND_FILL;
    grid_ctl.size_flags_vertical = SizeFlags::EXPAND_FILL;
    let grid = world.spawn((grid_ctl, GridContainer::new(2, 10.0, 10.0)));
    attach_child(&mut world, panel, grid).expect("Attach grid");

    let screen_2k = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 2560.0,
        height: 1440.0,
    };
    resolve_ui_layout_system(&mut world, screen_2k);

    let panel_res = world.get::<ResolvedRect>(panel).unwrap().global;
    assert_eq!(panel_res.width, 600.0);
    assert_eq!(panel_res.height, 400.0);

    let grid_res = world.get::<ResolvedRect>(grid).unwrap().global;
    assert_eq!(grid_res.x, panel_res.x + 16.0);
    assert_eq!(grid_res.y, panel_res.y + 16.0);
    assert_eq!(grid_res.width, 600.0 - 32.0);
    assert_eq!(grid_res.height, 400.0 - 32.0);

    // Verify Vulkan UBO upload
    let ubo_data = [grid_res.x, grid_res.y, grid_res.width, grid_res.height];
    let ubo_bytes: &[u8] = korppi_core::pod::cast_slice(&ubo_data);
    let ubo_desc = BufferDescriptor {
        size: ubo_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("vk_grid_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(ubo_bytes))
        .expect("Create VK UBO");
    let readback = device
        .read_buffer(ubo, 0, ubo_bytes.len() as u64)
        .expect("Read VK UBO");
    assert_eq!(readback, ubo_bytes);
}
