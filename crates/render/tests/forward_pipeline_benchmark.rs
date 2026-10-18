//! Zero-allocation verification and 1440p / 1080p frame-time benchmark for [`ForwardPipeline`]
//! (`FIX-6.4`, `RND-01`, `RND-11`, `ISSUE-005`).
//!
//! - `forward_pipeline_steady_state_records_with_zero_heap_allocations` runs on every `cargo test`
//!   invocation (both Vulkan and OpenGL) and verifies that `ForwardPipeline::record_scene`
//!   performs 0 heap allocations after 3 warm-up frames.
//! - `forward_pipeline_1440p_and_1080p_frame_budget_benchmark` measures native 1440p (`2560x1440`)
//!   and 1080p (`1920x1080`) frame times with 4x MSAA, CSM shadows, clustered lights, SSAO,
//!   bloom, and AgX tonemapping. When `KORPPI_REQUIRE_PERF_BUDGET=1` is set on hardware (Ampere /
//!   RDNA2 or Turing / RDNA1), it asserts the 60 FPS (`<= 16.67 ms`) budget of Rule 4.

use korppi_math::{Mat4, Vec3};
use korppi_render::pbr::{ClearCoat, PbrMaterial};
use korppi_render::pipeline::{
    Camera3d, ForwardMaterial, ForwardPipeline, ForwardPipelineConfig, ForwardScene, GpuMesh,
    MeshData, MeshDraw, WorldDirectionalLight, WorldPointLight, WorldSpotLight,
};
use korppi_render::postprocess::{BloomConfig, SsaoConfig, TonemapConfig, TonemapOperator};
use korppi_rhi::{FrameIndex, GraphicsDevice, PresentMode, SampleCount, SwapchainDescriptor};
use korppi_rhi_conformance::display::{display_or_skip, TestWindow, VirtualDisplay};
use korppi_rhi_conformance::{Backend, Image, TestDevice};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct CountingAlloc;

thread_local! {
    static COUNTING_ENABLED: Cell<bool> = const { Cell::new(false) };
    static THREAD_ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards directly to the standard `System` allocator while incrementing a thread-local counter.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNTING_ENABLED.try_with(|enabled| {
            if enabled.get() {
                let _ = THREAD_ALLOC_COUNT.try_with(|c| c.set(c.get() + 1));
            }
        });
        // SAFETY: caller upholds `GlobalAlloc::alloc` preconditions.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller upholds `GlobalAlloc::dealloc` preconditions.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

fn allocations_in<R>(f: impl FnOnce() -> R) -> (usize, R) {
    THREAD_ALLOC_COUNT.with(|c| c.set(0));
    COUNTING_ENABLED.with(|e| e.set(true));
    let out = f();
    COUNTING_ENABLED.with(|e| e.set(false));
    let count = THREAD_ALLOC_COUNT.with(|c| c.get());
    (count, out)
}

fn wait_for_window(
    window: &TestWindow,
    device: &dyn GraphicsDevice,
    width: u32,
    height: u32,
    expected: impl Fn(&Image) -> bool,
) -> Image {
    device.wait_idle().expect("wait_idle");
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let image = window.capture(width, height);
        if expected(&image) || Instant::now() > deadline {
            return image;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn has_rendered_scene(image: &Image) -> bool {
    // Clear color is roughly [0.02, 0.03, 0.05] in linear (~[39, 47, 60] in sRGB).
    // The lit floor and 16 PBR meshes produce significantly brighter pixels in the lower two thirds.
    let mut lit_pixels = 0usize;
    let step_x = (image.width / 32).max(1);
    let step_y = (image.height / 32).max(1);
    for y in (image.height / 3..image.height).step_by(step_y as usize) {
        for x in (0..image.width).step_by(step_x as usize) {
            let px = image.pixel(x, y);
            if px[0] > 90 || px[1] > 90 || px[2] > 90 {
                lit_pixels += 1;
            }
        }
    }
    lit_pixels >= 16
}

fn build_benchmark_scene(
    floor_mesh: GpuMesh,
    cube_mesh: GpuMesh,
    sphere_mesh: GpuMesh,
    aspect_ratio: f32,
) -> ForwardScene {
    let mut draws = Vec::with_capacity(32);
    draws.push(MeshDraw {
        mesh: floor_mesh,
        transform: Mat4::from_translation(Vec3::new(0.0, -1.5, -8.0)),
        material: ForwardMaterial {
            albedo: Vec3::new(0.45, 0.47, 0.50),
            pbr: PbrMaterial {
                roughness: 0.7,
                metallic: 0.0,
                reflectance: 0.5,
                clear_coat: None,
                sheen: None,
            },
            ..Default::default()
        },
        casts_shadows: true,
    });

    for idx in 0..16 {
        let row = (idx / 4) as f32;
        let col = (idx % 4) as f32;
        let pos = Vec3::new(
            (col - 1.5) * 2.2,
            -0.5 + (idx % 2) as f32 * 0.6,
            -4.0 - row * 2.5,
        );
        let mesh = if idx % 2 == 0 { cube_mesh } else { sphere_mesh };
        draws.push(MeshDraw {
            mesh,
            transform: Mat4::from_translation(pos),
            material: ForwardMaterial {
                albedo: Vec3::new(0.2 + 0.05 * col, 0.3 + 0.15 * row, 0.75),
                pbr: PbrMaterial {
                    roughness: (0.15 + 0.2 * col).clamp(0.05, 1.0),
                    metallic: if idx % 3 == 0 { 1.0 } else { 0.0 },
                    reflectance: 0.5,
                    clear_coat: if idx % 4 == 0 {
                        Some(ClearCoat {
                            roughness: 0.1,
                            weight: 0.8,
                        })
                    } else {
                        None
                    },
                    sheen: None,
                },
                ..Default::default()
            },
            casts_shadows: true,
        });
    }

    let mut point_lights = Vec::with_capacity(24);
    for i in 0..16 {
        let x = ((i % 4) as f32 - 1.5) * 3.0;
        let z = -3.0 - (i / 4) as f32 * 2.5;
        point_lights.push(WorldPointLight {
            position: Vec3::new(x, 1.2, z),
            radius: 4.5,
            color: Vec3::new(
                0.5 + 0.5 * (f32::from(u8::from(i % 3 == 0))),
                0.5 + 0.5 * (f32::from(u8::from(i % 3 == 1))),
                0.5 + 0.5 * (f32::from(u8::from(i % 3 == 2))),
            ),
            intensity: 4.0,
            casts_shadows: i < 2,
        });
    }

    let spot_lights = vec![
        WorldSpotLight {
            position: Vec3::new(-2.0, 3.0, -3.0),
            direction: Vec3::new(0.2, -1.0, -0.5).normalize(),
            range: 10.0,
            half_angle: 0.55,
            color: Vec3::new(1.0, 0.9, 0.7),
            intensity: 12.0,
            casts_shadows: true,
        },
        WorldSpotLight {
            position: Vec3::new(2.0, 3.0, -5.0),
            direction: Vec3::new(-0.2, -1.0, -0.4).normalize(),
            range: 10.0,
            half_angle: 0.55,
            color: Vec3::new(0.7, 0.85, 1.0),
            intensity: 12.0,
            casts_shadows: false,
        },
    ];

    ForwardScene {
        camera: Camera3d {
            position: Vec3::new(0.0, 2.0, 2.0),
            target: Vec3::new(0.0, -0.5, -6.0),
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            aspect_ratio,
            z_near: 0.2,
            z_far: 40.0,
        },
        directional_light: Some(WorldDirectionalLight {
            direction: Vec3::new(-0.4, -0.8, -0.5).normalize(),
            color: Vec3::new(1.0, 0.98, 0.95),
            illuminance: 3.0,
            casts_shadows: true,
        }),
        ambient_light: Vec3::splat(0.06),
        clear_color: [0.02, 0.03, 0.05, 1.0],
        draws,
        point_lights,
        spot_lights,
    }
}

fn present_scene_frame(
    device: &dyn GraphicsDevice,
    swapchain: korppi_rhi::SwapchainHandle,
    pipeline: &mut ForwardPipeline,
    frame_index: FrameIndex,
    scene: &ForwardScene,
) {
    device.begin_frame().expect("begin_frame");
    let image = device
        .acquire_next_image(swapchain)
        .expect("acquire_next_image")
        .expect("swapchain image available");
    let mut rec = device.create_command_recorder().expect("recorder");
    rec.begin().expect("begin");
    pipeline
        .record_scene(device, rec.as_mut(), frame_index, scene, image.texture)
        .expect("record_scene");
    let cmd = rec.finish().expect("finish");
    let sub = device.submit(cmd).expect("submit");
    device.present(&image).expect("present");
    device
        .wait_for_submission(sub)
        .expect("wait_for_submission");
}

#[test]
fn forward_pipeline_steady_state_records_with_zero_heap_allocations() {
    let Some(display) =
        display_or_skip("forward_pipeline_steady_state_records_with_zero_heap_allocations")
    else {
        return;
    };
    const W: u32 = 128;
    const H: u32 = 72;

    for backend in [Backend::OpenGl, Backend::Vulkan] {
        let window = Arc::new(TestWindow::create(&display, W, H));
        let device = TestDevice::open_for_display(backend, window.clone());
        let swapchain = device
            .create_swapchain(
                window.clone(),
                &SwapchainDescriptor {
                    present_mode: PresentMode::Immediate,
                    ..SwapchainDescriptor::new(W, H)
                },
            )
            .expect("create_swapchain");
        let ldr_format = device
            .swapchain_info(swapchain)
            .expect("swapchain_info")
            .format;

        let config = ForwardPipelineConfig {
            width: W,
            height: H,
            ldr_format,
            msaa_samples: SampleCount::Count4,
            shadow_atlas_size: 512,
            tonemap: TonemapConfig {
                operator: TonemapOperator::AgX,
                exposure: 1.0,
                gamma: 2.2,
                white_point: 10.0,
            },
            bloom: Some(BloomConfig::default()),
            ssao: Some(SsaoConfig::default()),
        };
        let mut pipeline = ForwardPipeline::new(&*device, config).expect("ForwardPipeline");

        let floor_mesh =
            GpuMesh::upload(&*device, &MeshData::plane(10.0, 10.0), Some("bench_floor"))
                .expect("upload floor");
        let cube_mesh = GpuMesh::upload(&*device, &MeshData::unit_cube(), Some("bench_cube"))
            .expect("upload cube");
        let sphere_mesh = GpuMesh::upload(
            &*device,
            &MeshData::uv_sphere(0.5, 8, 12),
            Some("bench_sphere"),
        )
        .expect("upload sphere");

        let scene = build_benchmark_scene(floor_mesh, cube_mesh, sphere_mesh, W as f32 / H as f32);

        // Warm up 4 presented frames (covering both FrameIndex slots 0 and 1)
        for i in 0..4 {
            present_scene_frame(
                &*device,
                swapchain,
                &mut pipeline,
                FrameIndex::new(i % 2),
                &scene,
            );
        }
        device.assert_no_api_errors();

        // Measure allocations in steady-state `record_scene` into an acquired swapchain image
        device.begin_frame().expect("begin_frame");
        let image = device
            .acquire_next_image(swapchain)
            .expect("acquire_next_image")
            .expect("swapchain image");
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");

        let (allocs, res) = allocations_in(|| {
            pipeline.record_scene(
                &*device,
                recorder.as_mut(),
                FrameIndex::new(0),
                &scene,
                image.texture,
            )
        });
        res.expect("steady-state record_scene");

        let cmd = recorder.finish().expect("finish");
        let sub = device.submit(cmd).expect("submit");
        device.present(&image).expect("present");
        device.wait_for_submission(sub).expect("wait");

        let captured = wait_for_window(&window, &*device, W, H, has_rendered_scene);
        assert!(
            has_rendered_scene(&captured),
            "{backend:?}: presented window did not show rendered 3D scene"
        );
        device.assert_no_api_errors();
        assert_eq!(
            allocs, 0,
            "{backend:?}: steady-state ForwardPipeline::record_scene must perform 0 heap allocations, got {allocs}"
        );

        floor_mesh.destroy(&*device).expect("destroy floor");
        cube_mesh.destroy(&*device).expect("destroy cube");
        sphere_mesh.destroy(&*device).expect("destroy sphere");
        pipeline.destroy(&*device).expect("destroy pipeline");
        device
            .destroy_swapchain(swapchain)
            .expect("destroy swapchain");
    }
}

fn benchmark_backend_resolution(
    display: &VirtualDisplay,
    backend: Backend,
    (width, height, res_label): (u32, u32, &str),
    require_hardware_budget: bool,
    warmup_frames: usize,
    measured_frames: usize,
) -> Image {
    let window = Arc::new(TestWindow::create(display, width, height));
    let device = TestDevice::open_for_display(backend, window.clone());
    let swapchain = device
        .create_swapchain(
            window.clone(),
            &SwapchainDescriptor {
                present_mode: PresentMode::Immediate,
                ..SwapchainDescriptor::new(width, height)
            },
        )
        .expect("create_swapchain");
    let swap_info = device.swapchain_info(swapchain).expect("swapchain_info");

    let floor_mesh = GpuMesh::upload(&*device, &MeshData::plane(12.0, 12.0), Some("bench_floor"))
        .expect("upload floor");
    let cube_mesh =
        GpuMesh::upload(&*device, &MeshData::unit_cube(), Some("bench_cube")).expect("upload cube");
    let sphere_mesh = GpuMesh::upload(
        &*device,
        &MeshData::uv_sphere(0.5, 12, 16),
        Some("bench_sphere"),
    )
    .expect("upload sphere");

    let config = ForwardPipelineConfig {
        width,
        height,
        ldr_format: swap_info.format,
        msaa_samples: if require_hardware_budget {
            SampleCount::Count4
        } else {
            SampleCount::Count1
        },
        shadow_atlas_size: if require_hardware_budget { 2048 } else { 512 },
        tonemap: TonemapConfig {
            operator: TonemapOperator::AgX,
            exposure: 1.0,
            gamma: 2.2,
            white_point: 10.0,
        },
        bloom: Some(BloomConfig::default()),
        ssao: Some(SsaoConfig::default()),
    };
    let mut pipeline = ForwardPipeline::new(&*device, config).expect("ForwardPipeline");
    let scene = build_benchmark_scene(
        floor_mesh,
        cube_mesh,
        sphere_mesh,
        width as f32 / height as f32,
    );

    for i in 0..warmup_frames {
        present_scene_frame(
            &*device,
            swapchain,
            &mut pipeline,
            FrameIndex::new(i % 2),
            &scene,
        );
    }
    device.wait_idle().expect("wait_idle after warmup");

    let start = Instant::now();
    for i in 0..measured_frames {
        present_scene_frame(
            &*device,
            swapchain,
            &mut pipeline,
            FrameIndex::new(i % 2),
            &scene,
        );
    }
    device.wait_idle().expect("wait_idle after measured frames");
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / (measured_frames as f64);
    let fps = 1000.0 / avg_ms.max(1e-6);

    println!(
        "[ISSUE-005 benchmark] {backend:?} {res_label} ({width}x{height}, format={:?}, present={:?}): avg {avg_ms:.2} ms/frame ({fps:.1} FPS over {measured_frames} presented frames)",
        swap_info.format, swap_info.present_mode
    );

    let captured = wait_for_window(&window, &*device, width, height, has_rendered_scene);
    assert!(
        has_rendered_scene(&captured),
        "{backend:?} {res_label}: presented window did not show rendered 3D scene"
    );

    if require_hardware_budget {
        assert!(
            avg_ms <= 16.67,
            "{backend:?} {res_label}: frame time {avg_ms:.2} ms exceeded 16.67 ms (60 FPS) budget (Rule 4 / ISSUE-005)"
        );
    }

    floor_mesh.destroy(&*device).expect("destroy floor");
    cube_mesh.destroy(&*device).expect("destroy cube");
    sphere_mesh.destroy(&*device).expect("destroy sphere");
    pipeline.destroy(&*device).expect("destroy pipeline");
    device
        .destroy_swapchain(swapchain)
        .expect("destroy swapchain");

    captured
}

#[test]
fn forward_pipeline_1440p_and_1080p_frame_budget_benchmark() {
    let Some(display) = display_or_skip("forward_pipeline_1440p_and_1080p_frame_budget_benchmark")
    else {
        return;
    };
    let require_hardware_budget = std::env::var("KORPPI_REQUIRE_PERF_BUDGET")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let warmup_frames = if require_hardware_budget { 6 } else { 2 };
    let measured_frames = if require_hardware_budget { 30 } else { 2 };
    let resolutions: &[(u32, u32, &str)] = if require_hardware_budget {
        &[(1920, 1080, "1080p"), (2560, 1440, "1440p")]
    } else {
        &[(1920, 1080, "1080p")]
    };

    for &res in resolutions {
        let (width, height, res_label) = res;
        let vk_img = benchmark_backend_resolution(
            &display,
            Backend::Vulkan,
            res,
            require_hardware_budget,
            warmup_frames,
            measured_frames,
        );
        let gl_img = benchmark_backend_resolution(
            &display,
            Backend::OpenGl,
            res,
            require_hardware_budget,
            warmup_frames,
            measured_frames,
        );

        // Verify cross-backend parity on the actually drawn and captured window pixels
        let mut matching_samples = 0usize;
        let mut total_samples = 0usize;
        let step_x = (width / 32).max(1);
        let step_y = (height / 32).max(1);
        for y in (0..height).step_by(step_y as usize) {
            for x in (0..width).step_by(step_x as usize) {
                total_samples += 1;
                let vk_px = vk_img.pixel(x, y);
                let gl_px = gl_img.pixel(x, y);
                if vk_px[0].abs_diff(gl_px[0]) <= 8
                    && vk_px[1].abs_diff(gl_px[1]) <= 8
                    && vk_px[2].abs_diff(gl_px[2]) <= 8
                {
                    matching_samples += 1;
                }
            }
        }
        assert!(
            matching_samples * 100 >= total_samples * 95,
            "{res_label}: Vulkan and OpenGL captured window frames diverged ({matching_samples}/{total_samples} samples within tolerance)"
        );
    }
}
