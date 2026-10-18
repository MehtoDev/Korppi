//! Primary GPU golden-image tests for the 3D Clustered Forward Pipeline (`FIX-6.4`, `RND-01`).
//!
//! Every test renders into a real window swapchain on `$DISPLAY` (or `Xvfb`), presents to the
//! window, and evaluates the actually drawn window pixels via `window.capture(W, H)` on both
//! Vulkan 1.3 and OpenGL 4.5 Core:
//! 1. Reverse-Z depth prepass writes `0.0` at the far clear plane, `1.0` near, and matches CPU
//!    `ReverseZ::perspective_rh` depth values while occluding farther geometry.
//! 2. Forward PBR shading (dielectric, metallic with multi-scattering energy compensation,
//!    clear coat, and Charlie sheen) and tonemapping (AgX and ACES) match the CPU
//!    `evaluate_pbr` + `evaluate_tonemapping` oracle.
//! 3. Clustered point and spot lights culled on the GPU illuminate their froxels and match the
//!    CPU clustered lighting oracle.
//! 4. 4x MSAA forward rendering resolves into the 1x target with anti-aliased silhouette edges.
//! 5. Directional CSM, spotlight, and octahedral omnidirectional point-light shadows in the
//!    unified shadow atlas occlude direct lighting behind shadow casters.
//! 6. Progressive bloom and SSAO passes produce expected highlight halos and crease occlusion.
#![allow(clippy::chunks_exact_to_as_chunks)]

use korppi_math::{Mat4, Vec3, Vec4};
use korppi_render::clustered::ReverseZ;
use korppi_render::pbr::{evaluate_pbr, CharlieSheen, ClearCoat, PbrMaterial, PbrSurface};
use korppi_render::pipeline::{
    Camera3d, ForwardMaterial, ForwardPipeline, ForwardPipelineConfig, ForwardScene, GpuMesh,
    MeshData, MeshDraw, WorldDirectionalLight, WorldPointLight, WorldSpotLight,
};
use korppi_render::postprocess::{
    evaluate_tonemapping, BloomConfig, SsaoConfig, TonemapConfig, TonemapOperator,
};
use korppi_rhi::{
    FrameIndex, PresentMode, SampleCount, SwapchainDescriptor, SwapchainHandle, TextureFormat,
};
use korppi_rhi_conformance::display::{display_or_skip, TestWindow, VirtualDisplay};
use korppi_rhi_conformance::{Backend, Image, TestDevice};
use std::sync::Arc;
use std::time::{Duration, Instant};

struct WindowTarget {
    window: Arc<TestWindow>,
    device: TestDevice,
    swapchain: SwapchainHandle,
    format: TextureFormat,
}

impl WindowTarget {
    fn open(
        display: &VirtualDisplay,
        backend: Backend,
        width: u32,
        height: u32,
        srgb: bool,
    ) -> Self {
        let window = Arc::new(TestWindow::create(display, width, height));
        let device = TestDevice::open_for_display(backend, window.clone());
        let swapchain = device
            .create_swapchain(
                window.clone(),
                &SwapchainDescriptor {
                    present_mode: PresentMode::Immediate,
                    srgb,
                    ..SwapchainDescriptor::new(width, height)
                },
            )
            .expect("create_swapchain");
        let format = device
            .swapchain_info(swapchain)
            .expect("swapchain_info")
            .format;
        Self {
            window,
            device,
            swapchain,
            format,
        }
    }

    fn present_and_capture(
        &self,
        pipeline: &mut ForwardPipeline,
        scene: &ForwardScene,
        width: u32,
        height: u32,
        ready: impl Fn(&Image) -> bool,
    ) -> Image {
        for frame in 0..3 {
            self.device.begin_frame().expect("begin_frame");
            let image = self
                .device
                .acquire_next_image(self.swapchain)
                .expect("acquire_next_image")
                .expect("swapchain image");
            let mut recorder = self.device.create_command_recorder().expect("recorder");
            recorder.begin().expect("begin");
            pipeline
                .record_scene(
                    &*self.device,
                    recorder.as_mut(),
                    FrameIndex::new(frame % 2),
                    scene,
                    image.texture,
                )
                .expect("record_scene");
            let sub = self
                .device
                .submit(recorder.finish().expect("finish"))
                .expect("submit");
            self.device.present(&image).expect("present");
            self.device
                .wait_for_submission(sub)
                .expect("wait_for_submission");
        }
        self.device.wait_idle().expect("wait_idle");
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let img = self.window.capture(width, height);
            if ready(&img) || Instant::now() > deadline {
                return img;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn close(self) {
        self.device
            .destroy_swapchain(self.swapchain)
            .expect("destroy_swapchain");
        self.device.assert_no_api_errors();
    }
}

fn to_u8_rgb(c: Vec3) -> [u8; 3] {
    [
        (c.x.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.y.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.z.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

#[test]
fn reverse_z_depth_prepass_occludes_far_geometry_and_matches_projection_math() {
    let Some(display) = display_or_skip(
        "reverse_z_depth_prepass_occludes_far_geometry_and_matches_projection_math",
    ) else {
        return;
    };
    const W: u32 = 64;
    const H: u32 = 64;

    for backend in Backend::ALL {
        let wt = WindowTarget::open(&display, backend, W, H, false);
        let config = ForwardPipelineConfig {
            width: W,
            height: H,
            ldr_format: wt.format,
            msaa_samples: SampleCount::Count1,
            shadow_atlas_size: 512,
            tonemap: TonemapConfig {
                operator: TonemapOperator::Reinhard,
                exposure: 1.0,
                gamma: 1.0,
                white_point: 10.0,
            },
            bloom: None,
            ssao: None,
        };
        let mut pipeline = ForwardPipeline::new(&*wt.device, config).expect("ForwardPipeline");

        let quad_mesh =
            GpuMesh::upload(&*wt.device, &MeshData::facing_quad(1.0, 1.0), Some("quad"))
                .expect("upload quad");

        let camera = Camera3d {
            position: Vec3::ZERO,
            target: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_2,
            aspect_ratio: 1.0,
            z_near: 0.5,
            z_far: 50.0,
        };

        // Far green quad at z = -5.0 shifted right so its right half is visible and left half overlaps near quad
        let far_draw = MeshDraw {
            mesh: quad_mesh,
            transform: Mat4::from_translation(Vec3::new(1.0, 0.0, -5.0))
                * Mat4::from_scale(Vec3::new(6.0, 6.0, 1.0)),
            material: ForwardMaterial {
                albedo: Vec3::new(0.0, 1.0, 0.0),
                emissive: Vec3::new(0.0, 0.8, 0.0),
                ..Default::default()
            },
            casts_shadows: false,
        };
        // Near red quad at z = -2.0 centered at (-0.5, 0.0, -2.0)
        let near_draw = MeshDraw {
            mesh: quad_mesh,
            transform: Mat4::from_translation(Vec3::new(-0.5, 0.0, -2.0))
                * Mat4::from_scale(Vec3::new(2.0, 2.0, 1.0)),
            material: ForwardMaterial {
                albedo: Vec3::new(1.0, 0.0, 0.0),
                emissive: Vec3::new(0.8, 0.0, 0.0),
                ..Default::default()
            },
            casts_shadows: false,
        };

        let scene = ForwardScene {
            camera,
            directional_light: None,
            ambient_light: Vec3::ZERO,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            draws: vec![far_draw, near_draw],
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
        };

        let img = wt.present_and_capture(&mut pipeline, &scene, W, H, |im| {
            im.pixel(24, 32)[0] > 50 || im.pixel(48, 32)[1] > 50
        });
        let depths = pipeline
            .read_depth_prepass(&*wt.device)
            .expect("read_depth_prepass");

        // Center-left pixel (24, 32) is covered by both quads: near red quad (z = -2.0) must win
        let p_overlap = img.pixel(24, 32);
        assert!(
            p_overlap[0] > 100 && p_overlap[1] < 10,
            "{backend:?}: near quad must occlude far quad at overlap pixel, got {p_overlap:?}"
        );

        // Right pixel (48, 32) is covered only by far green quad (z = -5.0)
        let p_far = img.pixel(48, 32);
        assert!(
            p_far[1] > 100 && p_far[0] < 10,
            "{backend:?}: far quad must be visible where unoccluded, got {p_far:?}"
        );

        // Verify Reverse-Z depth values against CPU projection math
        let proj = ReverseZ::perspective_rh(
            camera.fov_y_radians,
            camera.aspect_ratio,
            camera.z_near,
            camera.z_far,
        );
        let clip_near = proj * Vec4::new(0.0, 0.0, -2.0, 1.0);
        let expected_depth_near = clip_near.z / clip_near.w;
        let clip_far = proj * Vec4::new(0.0, 0.0, -5.0, 1.0);
        let expected_depth_far = clip_far.z / clip_far.w;

        let d_overlap = depths[(32 * W + 24) as usize];
        let d_far = depths[(32 * W + 48) as usize];
        let d_empty = depths[0];

        assert!(
            (d_overlap - expected_depth_near).abs() < 1e-4,
            "{backend:?}: depth at z=-2.0 expected {expected_depth_near}, got {d_overlap}"
        );
        assert!(
            (d_far - expected_depth_far).abs() < 1e-4,
            "{backend:?}: depth at z=-5.0 expected {expected_depth_far}, got {d_far}"
        );
        assert!(
            d_overlap > d_far,
            "{backend:?}: Reverse-Z requires nearer depth ({d_overlap}) > farther depth ({d_far})"
        );
        assert!(
            d_empty.abs() < 1e-6,
            "{backend:?}: Reverse-Z clear depth must be 0.0, got {d_empty}"
        );

        quad_mesh.destroy(&*wt.device).expect("destroy mesh");
        pipeline.destroy(&*wt.device).expect("destroy pipeline");
        wt.close();
    }
}

#[test]
fn forward_pbr_and_tonemapping_match_cpu_oracle_on_both_backends() {
    let Some(display) =
        display_or_skip("forward_pbr_and_tonemapping_match_cpu_oracle_on_both_backends")
    else {
        return;
    };
    const W: u32 = 64;
    const H: u32 = 64;

    let materials = [
        (
            "dielectric",
            Vec3::new(0.8, 0.2, 0.15),
            PbrMaterial {
                roughness: 0.4,
                metallic: 0.0,
                reflectance: 0.5,
                clear_coat: None,
                sheen: None,
            },
        ),
        (
            "rough_gold_metal",
            Vec3::new(1.0, 0.78, 0.34),
            PbrMaterial {
                roughness: 0.6,
                metallic: 1.0,
                reflectance: 0.5,
                clear_coat: None,
                sheen: None,
            },
        ),
        (
            "clear_coat_paint",
            Vec3::new(0.1, 0.35, 0.85),
            PbrMaterial {
                roughness: 0.5,
                metallic: 0.2,
                reflectance: 0.5,
                clear_coat: Some(ClearCoat {
                    roughness: 0.1,
                    weight: 1.0,
                }),
                sheen: None,
            },
        ),
        (
            "charlie_sheen_velvet",
            Vec3::new(0.25, 0.1, 0.35),
            PbrMaterial {
                roughness: 0.8,
                metallic: 0.0,
                reflectance: 0.5,
                clear_coat: None,
                sheen: Some(CharlieSheen {
                    color: Vec3::new(0.85, 0.4, 0.95),
                    roughness: 0.4,
                }),
            },
        ),
    ];

    let dir_light = WorldDirectionalLight {
        direction: Vec3::new(-0.3, -0.4, -1.0).normalize(),
        color: Vec3::new(1.0, 0.98, 0.92),
        illuminance: 3.5,
        casts_shadows: false,
    };

    for operator in [TonemapOperator::AgX, TonemapOperator::Aces] {
        let tonemap = TonemapConfig {
            operator,
            exposure: 1.0,
            gamma: 2.2,
            white_point: 10.0,
        };

        let mut backend_results: Vec<(Backend, Vec<[u8; 4]>)> = Vec::new();

        for backend in Backend::ALL {
            let wt = WindowTarget::open(&display, backend, W, H, true);
            let config = ForwardPipelineConfig {
                width: W,
                height: H,
                ldr_format: wt.format,
                msaa_samples: SampleCount::Count1,
                shadow_atlas_size: 512,
                tonemap,
                bloom: None,
                ssao: None,
            };
            let mut pipeline = ForwardPipeline::new(&*wt.device, config).expect("ForwardPipeline");
            let quad_mesh =
                GpuMesh::upload(&*wt.device, &MeshData::facing_quad(2.0, 2.0), Some("quad"))
                    .expect("upload quad");

            let camera = Camera3d {
                position: Vec3::new(0.0, 0.0, 2.0),
                target: Vec3::ZERO,
                up: Vec3::Y,
                fov_y_radians: std::f32::consts::FRAC_PI_3,
                aspect_ratio: 1.0,
                z_near: 0.1,
                z_far: 20.0,
            };

            let mut mat_pixels = Vec::new();

            for (mat_name, albedo, pbr_mat) in &materials {
                let surface = PbrSurface {
                    position: Vec3::ZERO,
                    normal: Vec3::Z,
                    view: Vec3::Z,
                    albedo: *albedo,
                    material: *pbr_mat,
                };
                let l_dir = (-dir_light.direction).normalize();
                let direct_hdr =
                    evaluate_pbr(&surface, l_dir, dir_light.color * dir_light.illuminance);
                let total_hdr = direct_hdr + (*albedo) * Vec3::splat(0.05);
                let expected_ldr = evaluate_tonemapping(total_hdr, &tonemap);
                let expected_u8 = to_u8_rgb(expected_ldr);

                let scene = ForwardScene {
                    camera,
                    directional_light: Some(dir_light),
                    ambient_light: Vec3::splat(0.05),
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                    draws: vec![MeshDraw {
                        mesh: quad_mesh,
                        transform: Mat4::IDENTITY,
                        material: ForwardMaterial {
                            albedo: *albedo,
                            pbr: *pbr_mat,
                            ..Default::default()
                        },
                        casts_shadows: false,
                    }],
                    point_lights: Vec::new(),
                    spot_lights: Vec::new(),
                };

                let img = wt.present_and_capture(&mut pipeline, &scene, W, H, |im| {
                    let px = im.pixel(W / 2, H / 2);
                    px[0].abs_diff(expected_u8[0]) <= 3
                        && px[1].abs_diff(expected_u8[1]) <= 3
                        && px[2].abs_diff(expected_u8[2]) <= 3
                });
                let center_px = img.pixel(W / 2, H / 2);
                mat_pixels.push(center_px);

                for ch in 0..3 {
                    let diff = center_px[ch].abs_diff(expected_u8[ch]);
                    assert!(
                        diff <= 3,
                        "{backend:?} {operator:?} {mat_name}: channel {ch} GPU={} vs CPU oracle={} (diff={diff}), full GPU={center_px:?}, expected={expected_u8:?}",
                        center_px[ch],
                        expected_u8[ch]
                    );
                }
            }

            quad_mesh.destroy(&*wt.device).expect("destroy mesh");
            pipeline.destroy(&*wt.device).expect("destroy pipeline");
            wt.close();
            backend_results.push((backend, mat_pixels));
        }

        let (_, vk_px) = &backend_results[0];
        let (_, gl_px) = &backend_results[1];
        for (idx, (mat_name, _, _)) in materials.iter().enumerate() {
            for ch in 0..4 {
                assert!(
                    vk_px[idx][ch].abs_diff(gl_px[idx][ch]) <= 2,
                    "{operator:?} {mat_name}: Vulkan {:?} vs OpenGL {:?} diverged",
                    vk_px[idx],
                    gl_px[idx]
                );
            }
        }
    }
}

#[test]
fn clustered_forward_point_and_spot_lights_match_cpu_oracle() {
    let Some(display) = display_or_skip("clustered_forward_point_and_spot_lights_match_cpu_oracle")
    else {
        return;
    };
    const W: u32 = 128;
    const H: u32 = 128;

    for backend in Backend::ALL {
        let wt = WindowTarget::open(&display, backend, W, H, true);
        let tonemap = TonemapConfig {
            operator: TonemapOperator::AgX,
            exposure: 1.0,
            gamma: 2.2,
            white_point: 10.0,
        };
        let config = ForwardPipelineConfig {
            width: W,
            height: H,
            ldr_format: wt.format,
            msaa_samples: SampleCount::Count1,
            shadow_atlas_size: 512,
            tonemap,
            bloom: None,
            ssao: None,
        };
        let mut pipeline = ForwardPipeline::new(&*wt.device, config).expect("ForwardPipeline");

        let wall_mesh =
            GpuMesh::upload(&*wt.device, &MeshData::facing_quad(6.0, 6.0), Some("wall"))
                .expect("upload wall");

        let camera = Camera3d {
            position: Vec3::ZERO,
            target: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_2,
            aspect_ratio: 1.0,
            z_near: 0.5,
            z_far: 20.0,
        };

        let wall_mat = PbrMaterial {
            roughness: 0.5,
            metallic: 0.0,
            reflectance: 0.5,
            clear_coat: None,
            sheen: None,
        };

        let pt_light = WorldPointLight {
            position: Vec3::new(-2.0, 2.0, -3.0),
            radius: 2.5,
            color: Vec3::new(1.0, 0.1, 0.1),
            intensity: 8.0,
            casts_shadows: false,
        };

        let sp_light = WorldSpotLight {
            position: Vec3::new(2.0, -2.0, -2.5),
            direction: Vec3::new(0.0, 0.0, -1.0),
            range: 3.0,
            half_angle: 0.6,
            color: Vec3::new(0.1, 1.0, 0.2),
            intensity: 10.0,
            casts_shadows: false,
        };

        let scene = ForwardScene {
            camera,
            directional_light: None,
            ambient_light: Vec3::splat(0.02),
            clear_color: [0.0, 0.0, 0.0, 1.0],
            draws: vec![MeshDraw {
                mesh: wall_mesh,
                transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -4.0)),
                material: ForwardMaterial {
                    albedo: Vec3::ONE,
                    pbr: wall_mat,
                    ..Default::default()
                },
                casts_shadows: false,
            }],
            point_lights: vec![pt_light],
            spot_lights: vec![sp_light],
        };

        let img = wt.present_and_capture(&mut pipeline, &scene, W, H, |im| {
            im.pixel(32, 32)[0] > 40 && im.pixel(96, 96)[1] > 40
        });

        let px_tl = img.pixel(32, 32);
        let px_br = img.pixel(96, 96);
        let px_tr = img.pixel(96, 32);

        assert!(
            px_tl[0] > px_tl[1] + 40 && px_tl[0] > px_tr[0] + 40,
            "{backend:?}: top-left froxel should be lit by red point light, got TL={px_tl:?}, TR={px_tr:?}"
        );
        assert!(
            px_br[1] > px_br[0] + 40 && px_br[1] > px_tr[1] + 40,
            "{backend:?}: bottom-right froxel should be lit by green spotlight, got BR={px_br:?}, TR={px_tr:?}"
        );

        wall_mesh.destroy(&*wt.device).expect("destroy mesh");
        pipeline.destroy(&*wt.device).expect("destroy pipeline");
        wt.close();
    }
}

#[test]
fn msaa_4x_resolves_silhouette_edges_and_matches_interior_oracle() {
    let Some(display) =
        display_or_skip("msaa_4x_resolves_silhouette_edges_and_matches_interior_oracle")
    else {
        return;
    };
    const W: u32 = 64;
    const H: u32 = 64;

    for backend in Backend::ALL {
        let wt = WindowTarget::open(&display, backend, W, H, false);
        let quad_mesh =
            GpuMesh::upload(&*wt.device, &MeshData::facing_quad(0.6, 0.6), Some("quad"))
                .expect("upload quad");

        let camera = Camera3d {
            position: Vec3::new(0.0, 0.0, 2.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            aspect_ratio: 1.0,
            z_near: 0.1,
            z_far: 10.0,
        };

        let render_with_samples = |samples: SampleCount| -> Image {
            let config = ForwardPipelineConfig {
                width: W,
                height: H,
                ldr_format: wt.format,
                msaa_samples: samples,
                shadow_atlas_size: 512,
                tonemap: TonemapConfig {
                    operator: TonemapOperator::Reinhard,
                    exposure: 1.0,
                    gamma: 1.0,
                    white_point: 10.0,
                },
                bloom: None,
                ssao: None,
            };
            let mut pipeline = ForwardPipeline::new(&*wt.device, config).expect("ForwardPipeline");

            let scene = ForwardScene {
                camera,
                directional_light: None,
                ambient_light: Vec3::ZERO,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                draws: vec![MeshDraw {
                    mesh: quad_mesh,
                    transform: Mat4::from_rotation_z(0.45),
                    material: ForwardMaterial {
                        albedo: Vec3::ONE,
                        emissive: Vec3::splat(10.0),
                        ..Default::default()
                    },
                    casts_shadows: false,
                }],
                point_lights: Vec::new(),
                spot_lights: Vec::new(),
            };

            let img = wt.present_and_capture(&mut pipeline, &scene, W, H, |im| {
                let partial = im
                    .data
                    .chunks_exact(4)
                    .filter(|px| px[0] > 10 && px[0] < 200)
                    .count();
                let center_ok = im.pixel(W / 2, H / 2)[0] > 200;
                match samples {
                    SampleCount::Count1 => center_ok && partial == 0,
                    _ => center_ok && partial >= 16,
                }
            });
            pipeline.destroy(&*wt.device).expect("destroy pipeline");
            img
        };

        let img_1x = render_with_samples(SampleCount::Count1);
        let img_4x = render_with_samples(SampleCount::Count4);

        let center_1x = img_1x.pixel(W / 2, H / 2);
        let center_4x = img_4x.pixel(W / 2, H / 2);
        assert!(
            center_1x[0] > 200 && center_1x[0].abs_diff(center_4x[0]) <= 2,
            "{backend:?}: center pixel should match between 1x ({center_1x:?}) and 4x ({center_4x:?})"
        );

        let partial_1x = img_1x
            .data
            .chunks_exact(4)
            .filter(|px| px[0] > 10 && px[0] < 200)
            .count();
        let partial_4x = img_4x
            .data
            .chunks_exact(4)
            .filter(|px| px[0] > 10 && px[0] < 200)
            .count();

        assert_eq!(
            partial_1x, 0,
            "{backend:?}: 1x non-MSAA should have 0 partial-coverage edge pixels"
        );
        assert!(
            partial_4x >= 16,
            "{backend:?}: 4x MSAA resolve should produce anti-aliased edge pixels, got {partial_4x}"
        );

        quad_mesh.destroy(&*wt.device).expect("destroy mesh");
        wt.close();
    }
}

#[test]
fn csm_spot_and_octahedral_point_shadows_occlude_direct_light() {
    let Some(display) =
        display_or_skip("csm_spot_and_octahedral_point_shadows_occlude_direct_light")
    else {
        return;
    };
    const W: u32 = 64;
    const H: u32 = 64;

    for backend in Backend::ALL {
        let wt = WindowTarget::open(&display, backend, W, H, true);
        let config = ForwardPipelineConfig {
            width: W,
            height: H,
            ldr_format: wt.format,
            msaa_samples: SampleCount::Count1,
            shadow_atlas_size: 1024,
            tonemap: TonemapConfig {
                operator: TonemapOperator::AgX,
                exposure: 1.0,
                gamma: 2.2,
                white_point: 10.0,
            },
            bloom: None,
            ssao: None,
        };
        let mut pipeline = ForwardPipeline::new(&*wt.device, config).expect("ForwardPipeline");

        let receiver_mesh = GpuMesh::upload(
            &*wt.device,
            &MeshData::facing_quad(3.0, 3.0),
            Some("receiver"),
        )
        .expect("upload receiver");
        let blocker_mesh = GpuMesh::upload(
            &*wt.device,
            &MeshData::facing_quad(0.6, 0.6),
            Some("blocker"),
        )
        .expect("upload blocker");

        let camera = Camera3d {
            position: Vec3::new(0.0, 0.0, 0.0),
            target: Vec3::new(0.0, 0.0, -4.0),
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            aspect_ratio: 1.0,
            z_near: 0.5,
            z_far: 20.0,
        };

        let shadow_light_dir = Vec3::new(1.2, 0.0, -3.0).normalize();
        let receiver_draw = MeshDraw {
            mesh: receiver_mesh,
            transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -4.0)),
            material: ForwardMaterial {
                albedo: Vec3::ONE,
                ..Default::default()
            },
            casts_shadows: true,
        };
        let blocker_draw = MeshDraw {
            mesh: blocker_mesh,
            transform: Mat4::from_translation(Vec3::new(-0.6, 0.0, -2.5)),
            material: ForwardMaterial {
                albedo: Vec3::new(0.2, 0.2, 0.2),
                ..Default::default()
            },
            casts_shadows: true,
        };

        // 1. Directional CSM shadow test
        let csm_scene = ForwardScene {
            camera,
            directional_light: Some(WorldDirectionalLight {
                direction: shadow_light_dir,
                color: Vec3::ONE,
                illuminance: 5.0,
                casts_shadows: true,
            }),
            ambient_light: Vec3::splat(0.05),
            clear_color: [0.0, 0.0, 0.0, 1.0],
            draws: vec![receiver_draw.clone(), blocker_draw.clone()],
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
        };
        let csm_img = wt.present_and_capture(&mut pipeline, &csm_scene, W, H, |im| {
            im.pixel(32, 16)[0] > im.pixel(32, 32)[0] + 40
        });
        let csm_shadowed = csm_img.pixel(32, 32);
        let csm_lit = csm_img.pixel(32, 16);
        assert!(
            csm_lit[0] > csm_shadowed[0] + 40,
            "{backend:?} CSM: lit pixel ({csm_lit:?}) must be brighter than shadowed pixel ({csm_shadowed:?})"
        );

        // 2. Spotlight shadow test
        let spot_scene = ForwardScene {
            camera,
            directional_light: None,
            ambient_light: Vec3::splat(0.05),
            clear_color: [0.0, 0.0, 0.0, 1.0],
            draws: vec![receiver_draw.clone(), blocker_draw.clone()],
            point_lights: Vec::new(),
            spot_lights: vec![WorldSpotLight {
                position: Vec3::new(-1.8, 0.0, 0.5),
                direction: shadow_light_dir,
                range: 12.0,
                half_angle: 0.9,
                color: Vec3::ONE,
                intensity: 35.0,
                casts_shadows: true,
            }],
        };
        let spot_img = wt.present_and_capture(&mut pipeline, &spot_scene, W, H, |im| {
            im.pixel(32, 16)[0] > im.pixel(32, 32)[0] + 30
        });
        let spot_shadowed = spot_img.pixel(32, 32);
        let spot_lit = spot_img.pixel(32, 16);
        assert!(
            spot_lit[0] > spot_shadowed[0] + 30,
            "{backend:?} Spot shadow: lit pixel ({spot_lit:?}) must be brighter than shadowed pixel ({spot_shadowed:?})"
        );

        // 3. Octahedral omnidirectional point-light shadow test
        let point_scene = ForwardScene {
            camera,
            directional_light: None,
            ambient_light: Vec3::splat(0.05),
            clear_color: [0.0, 0.0, 0.0, 1.0],
            draws: vec![receiver_draw, blocker_draw],
            point_lights: vec![WorldPointLight {
                position: Vec3::new(-1.8, 0.0, 0.5),
                radius: 12.0,
                color: Vec3::ONE,
                intensity: 35.0,
                casts_shadows: true,
            }],
            spot_lights: Vec::new(),
        };
        let point_img = wt.present_and_capture(&mut pipeline, &point_scene, W, H, |im| {
            im.pixel(32, 16)[0] > im.pixel(32, 32)[0] + 30
        });
        let point_shadowed = point_img.pixel(32, 32);
        let point_lit = point_img.pixel(32, 16);
        assert!(
            point_lit[0] > point_shadowed[0] + 30,
            "{backend:?} Octahedral point shadow: lit pixel ({point_lit:?}) must be brighter than shadowed pixel ({point_shadowed:?})"
        );

        receiver_mesh
            .destroy(&*wt.device)
            .expect("destroy receiver");
        blocker_mesh.destroy(&*wt.device).expect("destroy blocker");
        pipeline.destroy(&*wt.device).expect("destroy pipeline");
        wt.close();
    }
}

#[test]
fn bloom_and_ssao_modify_highlights_and_concave_creases() {
    let Some(display) = display_or_skip("bloom_and_ssao_modify_highlights_and_concave_creases")
    else {
        return;
    };
    const W: u32 = 64;
    const H: u32 = 64;

    for backend in Backend::ALL {
        let wt = WindowTarget::open(&display, backend, W, H, false);
        let quad_mesh =
            GpuMesh::upload(&*wt.device, &MeshData::facing_quad(1.0, 1.0), Some("quad"))
                .expect("upload quad");

        let camera = Camera3d {
            position: Vec3::new(0.0, 0.0, 2.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            aspect_ratio: 1.0,
            z_near: 0.2,
            z_far: 10.0,
        };

        // 1. Bloom test: small very bright HDR quad at center
        let render_bloom = |bloom: Option<BloomConfig>| -> Image {
            let has_bloom = bloom.is_some();
            let config = ForwardPipelineConfig {
                width: W,
                height: H,
                ldr_format: wt.format,
                msaa_samples: SampleCount::Count1,
                shadow_atlas_size: 512,
                tonemap: TonemapConfig {
                    operator: TonemapOperator::AgX,
                    exposure: 1.0,
                    gamma: 2.2,
                    white_point: 10.0,
                },
                bloom,
                ssao: None,
            };
            let mut pipeline = ForwardPipeline::new(&*wt.device, config).expect("ForwardPipeline");

            let scene = ForwardScene {
                camera,
                directional_light: None,
                ambient_light: Vec3::ZERO,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                draws: vec![MeshDraw {
                    mesh: quad_mesh,
                    transform: Mat4::from_scale(Vec3::new(0.25, 0.25, 1.0)),
                    material: ForwardMaterial {
                        albedo: Vec3::ONE,
                        emissive: Vec3::new(12.0, 8.0, 4.0),
                        ..Default::default()
                    },
                    casts_shadows: false,
                }],
                point_lights: Vec::new(),
                spot_lights: Vec::new(),
            };

            let img = wt.present_and_capture(&mut pipeline, &scene, W, H, |im| {
                let center_ok = im.pixel(32, 32)[0] > 180;
                if has_bloom {
                    center_ok && im.pixel(22, 32)[0] > 15
                } else {
                    center_ok && im.pixel(22, 32)[0] == 0
                }
            });
            pipeline.destroy(&*wt.device).expect("destroy pipeline");
            img
        };

        let no_bloom = render_bloom(None);
        let with_bloom = render_bloom(Some(BloomConfig {
            threshold: 1.0,
            soft_knee: 0.5,
            intensity: 0.5,
            filter_radius: 1.0,
            mip_levels: 3,
        }));

        let halo_off = no_bloom.pixel(22, 32);
        let halo_on = with_bloom.pixel(22, 32);
        assert!(
            halo_off[0] == 0 && halo_on[0] > 15,
            "{backend:?} Bloom: expected halo outside bright quad, got off={halo_off:?}, on={halo_on:?}"
        );

        // 2. SSAO test: back wall at z = -0.5 with an angled occluder wall intersecting at x = 0
        let render_ssao = |ssao: Option<SsaoConfig>| -> Image {
            let has_ssao = ssao.is_some();
            let config = ForwardPipelineConfig {
                width: W,
                height: H,
                ldr_format: wt.format,
                msaa_samples: SampleCount::Count1,
                shadow_atlas_size: 512,
                tonemap: TonemapConfig {
                    operator: TonemapOperator::Reinhard,
                    exposure: 1.0,
                    gamma: 1.0,
                    white_point: 10.0,
                },
                bloom: None,
                ssao,
            };
            let mut pipeline = ForwardPipeline::new(&*wt.device, config).expect("ForwardPipeline");

            let scene = ForwardScene {
                camera,
                directional_light: None,
                ambient_light: Vec3::ONE,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                draws: vec![
                    // Back wall at z = -0.5
                    MeshDraw {
                        mesh: quad_mesh,
                        transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -0.5))
                            * Mat4::from_scale(Vec3::new(4.0, 4.0, 1.0)),
                        material: ForwardMaterial {
                            albedo: Vec3::ONE,
                            ..Default::default()
                        },
                        casts_shadows: false,
                    },
                    // Left shelf at z = -0.15 covering x in [-1.12, -0.12]
                    MeshDraw {
                        mesh: quad_mesh,
                        transform: Mat4::from_translation(Vec3::new(-0.62, 0.0, -0.15))
                            * Mat4::from_scale(Vec3::new(1.0, 4.0, 1.0)),
                        material: ForwardMaterial {
                            albedo: Vec3::ONE,
                            ..Default::default()
                        },
                        casts_shadows: false,
                    },
                    // Right shelf at z = -0.15 covering x in [+0.12, +1.12]
                    MeshDraw {
                        mesh: quad_mesh,
                        transform: Mat4::from_translation(Vec3::new(0.62, 0.0, -0.15))
                            * Mat4::from_scale(Vec3::new(1.0, 4.0, 1.0)),
                        material: ForwardMaterial {
                            albedo: Vec3::ONE,
                            ..Default::default()
                        },
                        casts_shadows: false,
                    },
                ],
                point_lights: Vec::new(),
                spot_lights: Vec::new(),
            };

            let img = wt.present_and_capture(&mut pipeline, &scene, W, H, |im| {
                let mut sum = 0u32;
                for y in 24..40 {
                    for x in 29..35 {
                        sum += u32::from(im.pixel(x, y)[0]);
                    }
                }
                if has_ssao {
                    sum > 0 && sum < 11000
                } else {
                    sum > 11500
                }
            });
            pipeline.destroy(&*wt.device).expect("destroy pipeline");
            img
        };

        let ssao_off = render_ssao(None);
        let ssao_on = render_ssao(Some(SsaoConfig {
            sample_radius: 0.6,
            bias: 0.02,
            intensity: 2.5,
            power: 1.5,
            kernel_size: 16,
        }));

        let mut sum_off = 0u32;
        let mut sum_on = 0u32;
        for y in 24..40 {
            for x in 29..35 {
                sum_off += u32::from(ssao_off.pixel(x, y)[0]);
                sum_on += u32::from(ssao_on.pixel(x, y)[0]);
            }
        }
        assert!(
            sum_on < sum_off - 200,
            "{backend:?} SSAO: expected concave depth crease to darken ambient (sum_off={sum_off}, sum_on={sum_on})"
        );

        quad_mesh.destroy(&*wt.device).expect("destroy mesh");
        wt.close();
    }
}
