//! Headless offscreen integration tests for UI layout resolution, viewport resize handling,
//! coordinate mapping, and RHI device synchronization.

use korppi_ecs::World;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::control::{Control, LayoutPreset};
use korppi_ui::layout::{resolve_ui_layout_system, Rect2d, ResolvedRect};

use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_ui_layout_viewport_resize_and_coordinate_remapping_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut world = World::new();

    // 1. Root canvas control (FullRect)
    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let root = world.spawn((root_ctl,));

    // 2. Header panel: TopWide, height 80px
    let mut header_ctl = Control::default();
    header_ctl.set_anchors_preset(LayoutPreset::TopWide, false);
    header_ctl.offset_bottom = 80.0;
    let header = world.spawn((header_ctl,));
    attach_child(&mut world, root, header).expect("Attach header");

    // 3. Right Sidebar: RightWide, width 300px, starts below header (top offset 80px)
    let mut sidebar_ctl = Control::default();
    sidebar_ctl.set_anchors_preset(LayoutPreset::RightWide, false);
    sidebar_ctl.offset_left = -300.0;
    sidebar_ctl.offset_top = 80.0;
    let sidebar = world.spawn((sidebar_ctl,));
    attach_child(&mut world, root, sidebar).expect("Attach sidebar");

    // 4. Centered modal dialog: 400x300 centered in root
    let mut dialog_ctl = Control::default();
    dialog_ctl.set_anchors_preset(LayoutPreset::Center, false);
    dialog_ctl.offset_left = -200.0;
    dialog_ctl.offset_top = -150.0;
    dialog_ctl.offset_right = 200.0;
    dialog_ctl.offset_bottom = 150.0;
    let dialog = world.spawn((dialog_ctl,));
    attach_child(&mut world, root, dialog).expect("Attach dialog");

    // 5. Button inside dialog: BottomRight corner, 120x40 with 10px margin
    let mut btn_ctl = Control::default();
    btn_ctl.set_anchors_preset(LayoutPreset::BottomRight, false);
    btn_ctl.offset_left = -130.0;
    btn_ctl.offset_top = -50.0;
    btn_ctl.offset_right = -10.0;
    btn_ctl.offset_bottom = -10.0;
    let btn = world.spawn((btn_ctl,));
    attach_child(&mut world, dialog, btn).expect("Attach button");

    // Phase A: 1920x1080 Viewport
    let vp_1080p = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1920.0,
        height: 1080.0,
    };
    resolve_ui_layout_system(&mut world, vp_1080p);

    let header_res = *world.get::<ResolvedRect>(header).unwrap();
    assert_eq!(header_res.global.x, 0.0);
    assert_eq!(header_res.global.y, 0.0);
    assert_eq!(header_res.global.width, 1920.0);
    assert_eq!(header_res.global.height, 80.0);

    let sidebar_res = *world.get::<ResolvedRect>(sidebar).unwrap();
    assert_eq!(sidebar_res.global.x, 1920.0 - 300.0);
    assert_eq!(sidebar_res.global.y, 80.0);
    assert_eq!(sidebar_res.global.width, 300.0);
    assert_eq!(sidebar_res.global.height, 1000.0);

    let dialog_res = *world.get::<ResolvedRect>(dialog).unwrap();
    assert_eq!(dialog_res.global.x, 960.0 - 200.0);
    assert_eq!(dialog_res.global.y, 540.0 - 150.0);
    assert_eq!(dialog_res.global.width, 400.0);
    assert_eq!(dialog_res.global.height, 300.0);

    let btn_res = *world.get::<ResolvedRect>(btn).unwrap();
    // In dialog local coords: right edge is 400, bottom is 300
    // btn global x = dialog.x + (400 - 130) = 760 + 270 = 1030
    // btn global y = dialog.y + (300 - 50) = 390 + 250 = 640
    assert_eq!(btn_res.global.x, 760.0 + 270.0);
    assert_eq!(btn_res.global.y, 390.0 + 250.0);
    assert_eq!(btn_res.global.width, 120.0);
    assert_eq!(btn_res.global.height, 40.0);

    // Verify coordinate remapping: cursor inside button
    let cursor_screen = Vec2::new(1050.0, 650.0);
    assert!(btn_res.global.contains_point(cursor_screen));
    let cursor_local_to_dialog =
        cursor_screen - Vec2::new(dialog_res.global.x, dialog_res.global.y);
    assert_eq!(cursor_local_to_dialog, Vec2::new(290.0, 260.0));
    let cursor_local_to_btn = cursor_screen - Vec2::new(btn_res.global.x, btn_res.global.y);
    assert_eq!(cursor_local_to_btn, Vec2::new(20.0, 10.0));

    // Phase B: Dynamic Viewport Resize to 1280x720
    let vp_720p = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 1280.0,
        height: 720.0,
    };
    resolve_ui_layout_system(&mut world, vp_720p);

    let header_res_720 = *world.get::<ResolvedRect>(header).unwrap();
    assert_eq!(header_res_720.global.width, 1280.0);
    assert_eq!(header_res_720.global.height, 80.0);

    let sidebar_res_720 = *world.get::<ResolvedRect>(sidebar).unwrap();
    assert_eq!(sidebar_res_720.global.x, 1280.0 - 300.0);
    assert_eq!(sidebar_res_720.global.height, 640.0);

    let dialog_res_720 = *world.get::<ResolvedRect>(dialog).unwrap();
    assert_eq!(dialog_res_720.global.x, 640.0 - 200.0);
    assert_eq!(dialog_res_720.global.y, 360.0 - 150.0);

    // Verify GPU offscreen render target descriptor sync
    let target_desc = TextureDescriptor {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 1280,
        height: 720,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: korppi_rhi::SampleCount::Count1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        label: Some("ui_offscreen_target_720p"),
    };
    let target = device
        .create_texture(&target_desc, None)
        .expect("Create UI render target");
    assert!(target.is_valid());

    // Verify UI transform uniform buffer upload
    let ui_proj_matrix = [
        2.0 / 1280.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 / 720.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        1.0,
        0.0,
        1.0,
    ];
    let ubo_bytes: &[u8] = korppi_core::pod::cast_slice(&ui_proj_matrix);
    let ubo_desc = BufferDescriptor {
        size: ubo_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("ui_projection_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(ubo_bytes))
        .expect("Create UBO");
    let readback = device
        .read_buffer(ubo, 0, ubo_bytes.len() as u64)
        .expect("Read UBO");
    assert_eq!(readback, ubo_bytes);
}

#[test]
fn test_ui_layout_viewport_resize_and_coordinate_remapping_vulkan() {
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

    let mut center_box = Control::default();
    center_box.set_anchors_preset(LayoutPreset::Center, false);
    center_box.offset_left = -100.0;
    center_box.offset_top = -100.0;
    center_box.offset_right = 100.0;
    center_box.offset_bottom = 100.0;
    let box_ent = world.spawn((center_box,));
    attach_child(&mut world, root, box_ent).expect("Attach box");

    let vp_4k = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 3840.0,
        height: 2160.0,
    };
    resolve_ui_layout_system(&mut world, vp_4k);

    let box_res = *world.get::<ResolvedRect>(box_ent).unwrap();
    assert_eq!(box_res.global.x, 1920.0 - 100.0);
    assert_eq!(box_res.global.y, 1080.0 - 100.0);
    assert_eq!(box_res.global.width, 200.0);
    assert_eq!(box_res.global.height, 200.0);

    let ubo_data = [1.0f32, 2.0, 3.0, 4.0];
    let ubo_bytes: &[u8] = korppi_core::pod::cast_slice(&ubo_data);
    let ubo_desc = BufferDescriptor {
        size: ubo_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("vk_ui_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(ubo_bytes))
        .expect("Create VK UBO");
    let readback = device
        .read_buffer(ubo, 0, ubo_bytes.len() as u64)
        .expect("Read VK UBO");
    assert_eq!(readback, ubo_bytes);
}
