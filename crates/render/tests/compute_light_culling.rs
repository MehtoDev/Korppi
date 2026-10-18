//! Verification test for Asynchronous GPU Compute Shader Clustered Froxel Culling (AUDIT-4.1 / Finding 3.1).
//!
//! Research reference: `research/render/clustered_shading_and_light_culling.md:60-120`
//! and `comprehensive_forward_renderer_synthesis.md:43-48`.
//! GPU compute shader dispatch culls point and spot lights into SSBOs, with CPU fallback.

use korppi_math::Vec3;
use korppi_render::clustered::{
    ClusterGridConfig, ClusteredLightCuller, GpuLightCuller, PointLight, ReverseZ, SpotLight,
    CLUSTERED_LIGHT_CULL_GLSL, CLUSTERED_LIGHT_CULL_SPIRV,
};
use korppi_rhi::GraphicsDevice;
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;

#[test]
fn test_compute_shader_artifacts_present() {
    assert!(
        !CLUSTERED_LIGHT_CULL_GLSL.is_empty(),
        "GLSL source code must be non-empty"
    );
    assert!(
        !CLUSTERED_LIGHT_CULL_SPIRV.is_empty(),
        "Compiled SPIR-V bytecode must be non-empty"
    );
    assert!(
        CLUSTERED_LIGHT_CULL_SPIRV.len().is_multiple_of(4),
        "SPIR-V bytecode must be 4-byte aligned"
    );
}

#[test]
fn test_cpu_fallback_when_device_unavailable() {
    let config = ClusterGridConfig::new(1280, 720, 64, 16, 0.1, 100.0);
    let inv_proj =
        ReverseZ::perspective_rh(60.0f32.to_radians(), 1280.0 / 720.0, 0.1, 100.0).inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);

    let point_lights = vec![PointLight {
        position_vs: Vec3::new(0.0, 0.0, -10.0),
        radius: 3.0,
        color: Vec3::ONE,
        intensity: 50.0,
    }];
    let spot_lights = vec![];

    let mut culler = ClusteredLightCuller::new(&config);
    // Explicit fallback: device is None, gpu_culler is None
    let output =
        culler.cull_lights_gpu_or_fallback(None, None, &froxels, &point_lights, &spot_lights);

    assert_eq!(
        output.cluster_records.len(),
        config.total_clusters() as usize
    );
    let visible_count = output
        .cluster_records
        .iter()
        .filter(|r| r.point_count > 0)
        .count();
    assert!(
        visible_count > 0,
        "CPU fallback must cull lights successfully"
    );
}

/// GPU culling matches the CPU reference. Runs the same SPIR-V on both backends (the OpenGL
/// backend translates it since FIX-1.3, RHI-08; before, OpenGL rejected the shader).
fn gpu_compute_light_culling_matches_cpu(device: &dyn GraphicsDevice) {
    let config = ClusterGridConfig::new(640, 360, 64, 8, 0.1, 50.0);
    let inv_proj =
        ReverseZ::perspective_rh(60.0f32.to_radians(), 640.0 / 360.0, 0.1, 50.0).inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);

    let point_lights = vec![
        PointLight {
            position_vs: Vec3::new(0.0, 0.0, -5.0),
            radius: 2.5,
            color: Vec3::ONE,
            intensity: 100.0,
        },
        PointLight {
            position_vs: Vec3::new(2.0, 1.0, -15.0),
            radius: 4.0,
            color: Vec3::new(1.0, 0.5, 0.2),
            intensity: 80.0,
        },
    ];

    let spot_lights = vec![SpotLight {
        position_vs: Vec3::new(0.0, 0.0, -2.0),
        direction_vs: Vec3::new(0.0, 0.0, -1.0),
        range: 10.0,
        half_angle: 30.0f32.to_radians(),
        color: Vec3::new(0.2, 0.5, 1.0),
        intensity: 120.0,
    }];

    // 1. Run CPU reference culler
    let mut cpu_culler = ClusteredLightCuller::new(&config);
    let cpu_output = cpu_culler.cull_lights(&froxels, &point_lights, &spot_lights);

    // 2. Run GPU compute culler
    let mut gpu_culler = GpuLightCuller::new();
    gpu_culler
        .init(device)
        .expect("Failed to initialize GPU light culler pipeline");

    let gpu_output = gpu_culler
        .cull_lights(device, &froxels, &point_lights, &spot_lights)
        .expect("GPU compute light culling dispatch failed");

    assert_eq!(
        gpu_output.cluster_records.len(),
        config.total_clusters() as usize
    );

    // Verify per-cluster light counts match CPU reference
    for (idx, (gpu_rec, cpu_rec)) in gpu_output
        .cluster_records
        .iter()
        .zip(cpu_output.cluster_records.iter())
        .enumerate()
    {
        assert_eq!(
            gpu_rec.point_count, cpu_rec.point_count,
            "Cluster {idx} point light count mismatch: GPU {} vs CPU {}",
            gpu_rec.point_count, cpu_rec.point_count
        );
        assert_eq!(
            gpu_rec.spot_count, cpu_rec.spot_count,
            "Cluster {idx} spot light count mismatch: GPU {} vs CPU {}",
            gpu_rec.spot_count, cpu_rec.spot_count
        );
    }

    // Clean up pipeline
    gpu_culler
        .destroy(device)
        .expect("Destroy GPU culler failed");
}

#[test]
fn test_gpu_compute_light_culling_vulkan_execution() {
    let device = match VulkanDevice::create_headless() {
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("Skipping Vulkan compute test: headless device creation failed: {e}");
            return;
        }
    };
    gpu_compute_light_culling_matches_cpu(&device);
}

#[test]
fn test_gpu_compute_light_culling_opengl_execution() {
    let device = OpenGlDevice::create_headless().expect("headless OpenGL device");
    gpu_compute_light_culling_matches_cpu(&device);
    let errors: Vec<_> = device
        .take_diagnostics()
        .into_iter()
        .filter(|d| d.severity == korppi_rhi::DiagnosticSeverity::Error)
        .collect();
    assert!(errors.is_empty(), "OpenGL API errors: {errors:#?}");
}

#[test]
fn test_scaling_256_lights() {
    let config = ClusterGridConfig::new(1280, 720, 64, 16, 0.1, 100.0);
    let inv_proj =
        ReverseZ::perspective_rh(60.0f32.to_radians(), 1280.0 / 720.0, 0.1, 100.0).inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);

    let mut point_lights = Vec::with_capacity(256);
    for i in 0..256 {
        let z = -((i as f32 % 50.0) + 1.0);
        let x = ((i % 10) as f32 - 5.0) * 0.5;
        let y = (((i / 10) % 10) as f32 - 5.0) * 0.5;
        point_lights.push(PointLight {
            position_vs: Vec3::new(x, y, z),
            radius: 2.0,
            color: Vec3::ONE,
            intensity: 10.0,
        });
    }

    let mut culler = ClusteredLightCuller::new(&config);
    let output = culler.cull_lights(&froxels, &point_lights, &[]);

    assert_eq!(
        output.cluster_records.len(),
        config.total_clusters() as usize
    );
    let total_indexed_points: u32 = output.cluster_records.iter().map(|r| r.point_count).sum();
    assert!(
        total_indexed_points > 0,
        "256 lights must be indexed across froxels"
    );
}

#[test]
fn test_gpu_compute_culling_persistent_buffers_no_churn() {
    let device = match VulkanDevice::create_headless() {
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("Skipping Vulkan compute test: {e}");
            return;
        }
    };
    let config = ClusterGridConfig::new(640, 360, 64, 8, 0.1, 50.0);
    let inv_proj =
        ReverseZ::perspective_rh(60.0f32.to_radians(), 640.0 / 360.0, 0.1, 50.0).inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);
    let point_lights = vec![PointLight {
        position_vs: Vec3::new(0.0, 0.0, -5.0),
        radius: 2.5,
        color: Vec3::ONE,
        intensity: 100.0,
    }];

    let mut gpu_culler = GpuLightCuller::new();
    gpu_culler.init(&device).expect("Failed to init GPU culler");

    // Frame 1
    gpu_culler
        .cull_lights(&device, &froxels, &point_lights, &[])
        .expect("Frame 1 cull failed");
    let rec_buf_1 = gpu_culler
        .records_buffer()
        .expect("Records buffer must exist");
    let ind_buf_1 = gpu_culler
        .light_indices_buffer()
        .expect("Indices buffer must exist");

    // Frame 2
    gpu_culler
        .cull_lights(&device, &froxels, &point_lights, &[])
        .expect("Frame 2 cull failed");
    let rec_buf_2 = gpu_culler
        .records_buffer()
        .expect("Records buffer must exist");
    let ind_buf_2 = gpu_culler
        .light_indices_buffer()
        .expect("Indices buffer must exist");

    // Must be identical handles (persistent allocation without churn)
    assert_eq!(
        rec_buf_1, rec_buf_2,
        "Records buffer handle must be retained across frames"
    );
    assert_eq!(
        ind_buf_1, ind_buf_2,
        "Indices buffer handle must be retained across frames"
    );

    gpu_culler.destroy(&device).expect("Destroy failed");
}

#[test]
fn test_gpu_compute_cull_decals_and_probes_parity() {
    let device = match VulkanDevice::create_headless() {
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("Skipping Vulkan compute test: {e}");
            return;
        }
    };
    let config = ClusterGridConfig::new(640, 360, 64, 8, 0.1, 50.0);
    let inv_proj =
        ReverseZ::perspective_rh(60.0f32.to_radians(), 640.0 / 360.0, 0.1, 50.0).inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);

    let point_lights = vec![PointLight {
        position_vs: Vec3::new(0.0, 0.0, -5.0),
        radius: 3.0,
        color: Vec3::ONE,
        intensity: 50.0,
    }];
    let spot_lights = vec![];

    let decals = vec![
        korppi_render::clustered::Decal::axis_aligned(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(1.0, 1.0, 1.0),
        ),
        korppi_render::clustered::Decal::new(
            Vec3::new(1.0, 1.0, -10.0),
            Vec3::new(2.0, 2.0, 2.0),
            korppi_math::Quat::from_rotation_y(0.5),
        ),
    ];

    let probes = vec![
        korppi_render::clustered::ReflectionProbe::sphere(Vec3::new(0.0, 0.0, -5.0), 3.0, 1.0),
        korppi_render::clustered::ReflectionProbe::box_projected(
            Vec3::new(-1.0, 0.0, -8.0),
            Vec3::new(2.0, 2.0, 2.0),
            1.0,
        ),
    ];

    let mut cpu_culler = ClusteredLightCuller::new(&config);
    let cpu_output =
        cpu_culler.cull_features(&froxels, &point_lights, &spot_lights, &decals, &probes);

    let mut gpu_culler = GpuLightCuller::new();
    gpu_culler.init(&device).expect("Failed to init GPU culler");

    let gpu_output = gpu_culler
        .cull_features(
            &device,
            &froxels,
            &point_lights,
            &spot_lights,
            &decals,
            &probes,
        )
        .expect("GPU cull_features failed");

    assert!(
        !gpu_output.global_decal_indices.is_empty(),
        "GPU must output culled decal indices"
    );
    assert!(
        !gpu_output.global_probe_indices.is_empty(),
        "GPU must output culled probe indices"
    );

    for (idx, (gpu_rec, cpu_rec)) in gpu_output
        .cluster_records
        .iter()
        .zip(cpu_output.cluster_records.iter())
        .enumerate()
    {
        assert_eq!(
            gpu_rec.decal_count, cpu_rec.decal_count,
            "Cluster {idx} decal count mismatch: GPU {} vs CPU {}",
            gpu_rec.decal_count, cpu_rec.decal_count
        );
        assert_eq!(
            gpu_rec.probe_count, cpu_rec.probe_count,
            "Cluster {idx} probe count mismatch: GPU {} vs CPU {}",
            gpu_rec.probe_count, cpu_rec.probe_count
        );
    }

    gpu_culler.destroy(&device).expect("Destroy failed");
}

#[test]
fn test_gpu_compute_dispatch_async_no_stalls() {
    let device = match VulkanDevice::create_headless() {
        Ok(dev) => dev,
        Err(e) => {
            eprintln!("Skipping Vulkan compute test: {e}");
            return;
        }
    };
    let config = ClusterGridConfig::new(640, 360, 64, 8, 0.1, 50.0);
    let inv_proj =
        ReverseZ::perspective_rh(60.0f32.to_radians(), 640.0 / 360.0, 0.1, 50.0).inverse();
    let froxels = config.build_froxel_aabbs(inv_proj);

    let mut gpu_culler = GpuLightCuller::new();
    gpu_culler.init(&device).expect("Failed to init GPU culler");

    let mut recorder = device
        .create_command_recorder()
        .expect("Failed to create recorder");
    recorder.begin().expect("Begin failed");

    let buffers = gpu_culler
        .dispatch_culling_async(&device, &mut *recorder, &froxels, &[], &[], &[], &[])
        .expect("dispatch_culling_async failed");

    assert_eq!(buffers.total_clusters, config.total_clusters());
    assert_eq!(buffers.records_buffer, gpu_culler.records_buffer().unwrap());

    let cmd = recorder.finish().expect("Finish recorder failed");
    device.submit(cmd).expect("Submit async compute failed");
    device.wait_idle().expect("Wait idle failed");

    gpu_culler.destroy(&device).expect("Destroy failed");
}
