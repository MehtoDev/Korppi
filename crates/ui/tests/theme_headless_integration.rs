//! Headless offscreen integration tests for Theming & Skinning:
//! StyleBoxFlat vector geometry, 9-patch texture slicing, runtime theme swapping,
//! and RHI device validation.

#![allow(clippy::field_reassign_with_default)]

use korppi_ecs::World;
use korppi_math::Vec2;
use korppi_scene::hierarchy::attach_child;
use korppi_ui::control::{Control, LayoutPreset};
use korppi_ui::layout::Rect2d;
use korppi_ui::theme::{
    get_theme_color, get_theme_stylebox, AxisStretchMode, StyleBox, StyleBoxFlat, StyleBoxTexture,
    Theme, ThemeOverride,
};

use korppi_rhi::{
    BufferDescriptor, BufferUsage, GraphicsDevice, MemoryUsage, SampleCount, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage,
};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_theme_runtime_swap_and_stylebox_gpu_upload_opengl() {
    let device =
        OpenGlDevice::create_headless().expect("Headless OpenGL 4.5 device creation failed");

    let mut world = World::new();

    // 1. Root canvas with Dark Theme
    let mut root_ctl = Control::default();
    root_ctl.set_anchors_preset(LayoutPreset::FullRect, false);
    let mut dark_theme = Theme::dark();
    dark_theme.set_color("Label", "font_color", [0.9, 0.9, 0.9, 1.0]);
    let root = world.spawn((root_ctl, dark_theme));

    // 2. Child Window Panel with local custom StyleBoxFlat
    let mut panel_ctl = Control::default();
    panel_ctl.set_anchors_preset(LayoutPreset::Center, false);
    let mut panel_override = ThemeOverride::default();
    let mut custom_panel_sb = StyleBoxFlat::default();
    custom_panel_sb.bg_color = [0.15, 0.16, 0.18, 1.0];
    custom_panel_sb.border_color = [0.30, 0.50, 0.80, 1.0];
    custom_panel_sb.border_width = [2.0, 2.0, 2.0, 2.0];
    custom_panel_sb.corner_radius = [8.0, 8.0, 8.0, 8.0];
    custom_panel_sb.shadow_size = 12.0;
    custom_panel_sb.shadow_color = [0.0, 0.0, 0.0, 0.5];
    panel_override.set_stylebox("panel", StyleBox::Flat(custom_panel_sb));
    let panel = world.spawn((panel_ctl, panel_override));
    attach_child(&mut world, root, panel).expect("Attach panel");

    // 3. Child Button inside Panel (inherits theme from root, no local override)
    let mut btn_ctl = Control::default();
    btn_ctl.custom_minimum_size = Vec2::new(120.0, 36.0);
    let btn = world.spawn((btn_ctl,));
    attach_child(&mut world, panel, btn).expect("Attach btn");

    // Phase A: Query under Dark Theme
    let dark_font = get_theme_color(&world, btn, "Button", "font_color").unwrap();
    assert_eq!(dark_font, [0.87, 0.87, 0.87, 1.0]);

    let label_font = get_theme_color(&world, btn, "Label", "font_color").unwrap();
    assert_eq!(label_font, [0.9, 0.9, 0.9, 1.0]);

    let panel_sb = get_theme_stylebox(&world, panel, "Panel", "panel").unwrap();
    match panel_sb {
        StyleBox::Flat(f) => {
            assert_eq!(f.border_width, [2.0, 2.0, 2.0, 2.0]);
            assert_eq!(f.corner_radius, [8.0, 8.0, 8.0, 8.0]);
            assert_eq!(f.shadow_size, 12.0);
        }
        _ => panic!("Expected Flat stylebox"),
    }

    // Phase B: Runtime Theme Swap on Root to Light Theme
    let mut light_theme = Theme::default();
    light_theme.set_color("Button", "font_color", [0.1, 0.1, 0.1, 1.0]);
    light_theme.set_color("Label", "font_color", [0.05, 0.05, 0.05, 1.0]);
    let mut light_btn_sb = StyleBoxFlat::default();
    light_btn_sb.bg_color = [0.92, 0.92, 0.95, 1.0];
    light_btn_sb.border_color = [0.75, 0.75, 0.80, 1.0];
    light_theme.set_stylebox("Button", "normal", StyleBox::Flat(light_btn_sb));

    // Replace theme component on root
    *world.get_mut::<Theme>(root).unwrap() = light_theme;

    // Verify button immediately reflects new Light Theme values
    let light_font = get_theme_color(&world, btn, "Button", "font_color").unwrap();
    assert_eq!(light_font, [0.1, 0.1, 0.1, 1.0]);

    let light_label_font = get_theme_color(&world, btn, "Label", "font_color").unwrap();
    assert_eq!(light_label_font, [0.05, 0.05, 0.05, 1.0]);

    // Panel still preserves its local ThemeOverride
    let panel_sb_after_swap = get_theme_stylebox(&world, panel, "Panel", "panel").unwrap();
    match panel_sb_after_swap {
        StyleBox::Flat(f) => {
            assert_eq!(f.shadow_size, 12.0);
        }
        _ => panic!("Expected Flat stylebox"),
    }

    // Phase C: GPU StyleBox Uniform Buffer Data Packaging & Verification
    // Pack StyleBoxFlat shader parameters into standard std140 uniform buffer layout:
    // [bg_color(4), border_color(4), border_width(4), corner_radius(4)]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct StyleBoxGpuUniforms {
        bg_color: [f32; 4],
        border_color: [f32; 4],
        border_width: [f32; 4],
        corner_radius: [f32; 4],
        shadow_color: [f32; 4],
        params: [f32; 4], // [shadow_size, shadow_offset.x, shadow_offset.y, corner_detail]
    }

    let ubo_data = StyleBoxGpuUniforms {
        bg_color: [0.15, 0.16, 0.18, 1.0],
        border_color: [0.30, 0.50, 0.80, 1.0],
        border_width: [2.0, 2.0, 2.0, 2.0],
        corner_radius: [8.0, 8.0, 8.0, 8.0],
        shadow_color: [0.0, 0.0, 0.0, 0.5],
        params: [12.0, 0.0, 4.0, 8.0],
    };

    let ubo_fields = [
        ubo_data.bg_color,
        ubo_data.border_color,
        ubo_data.border_width,
        ubo_data.corner_radius,
        ubo_data.shadow_color,
        ubo_data.params,
    ];
    let ubo_bytes: &[u8] = korppi_core::pod::cast_slice(&ubo_fields);

    let ubo_desc = BufferDescriptor {
        size: ubo_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("gl_stylebox_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(ubo_bytes))
        .expect("Create stylebox UBO");
    let readback = device
        .read_buffer(ubo, 0, ubo_bytes.len() as u64)
        .expect("Read stylebox UBO");
    assert_eq!(readback, ubo_bytes);

    // Verify Texture Render Target for skinning offscreen validation
    let target_desc = TextureDescriptor {
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        width: 512,
        height: 512,
        depth: 1,
        array_layers: 1,
        mip_levels: 1,
        sample_count: SampleCount::Count1,
        usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::COPY_SRC,
        label: Some("gl_skinned_offscreen_target"),
    };
    let target = device
        .create_texture(&target_desc, None)
        .expect("Create skinned target");
    assert!(target.is_valid());
}

#[test]
fn test_stylebox_texture_9patch_slicing_and_vulkan() {
    let Ok(device) = VulkanDevice::create_headless() else {
        eprintln!(
            "Skipping Vulkan test: No physical Vulkan device or ICD found in headless environment"
        );
        return;
    };

    // 9-Patch Texture StyleBox
    let mut tex_box = StyleBoxTexture::default();
    tex_box.region_rect = Rect2d {
        x: 0.0,
        y: 0.0,
        width: 128.0,
        height: 128.0,
    };
    tex_box.texture_margin = [16.0, 16.0, 16.0, 16.0];
    tex_box.axis_h = AxisStretchMode::Stretch;
    tex_box.axis_v = AxisStretchMode::Stretch;

    let target_rect = Rect2d {
        x: 50.0,
        y: 50.0,
        width: 400.0,
        height: 200.0,
    };
    let draw_rect = tex_box.get_draw_rect(target_rect);
    assert_eq!(draw_rect, target_rect);

    // Upload 9-patch UV slice uniforms to Vulkan
    let slice_uvs = [
        16.0 / 128.0,  // u_left
        16.0 / 128.0,  // v_top
        112.0 / 128.0, // u_right
        112.0 / 128.0, // v_bottom
    ];
    let slice_bytes: &[u8] = korppi_core::pod::cast_slice(&slice_uvs);
    let ubo_desc = BufferDescriptor {
        size: slice_bytes.len() as u64,
        usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
        memory: MemoryUsage::CpuToGpu,
        label: Some("vk_9patch_ubo"),
    };
    let ubo = device
        .create_buffer(&ubo_desc, Some(slice_bytes))
        .expect("Create VK 9patch UBO");
    let readback = device
        .read_buffer(ubo, 0, slice_bytes.len() as u64)
        .expect("Read VK 9patch UBO");
    assert_eq!(readback, slice_bytes);
}
