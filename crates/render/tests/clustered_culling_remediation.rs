//! Regression tests for `FIX-6.5`: GPU light culler limits (`RND-12`, Probe `G11`) and
//! CPU/GPU zero steady-state frame allocations (`RND-11`, Probe `G14`).

use korppi_math::{Mat4, Vec3};
use korppi_render::clustered::{
    ClusterGridConfig, ClusteredLightCuller, CullingOutput, GpuLightCuller, PointLight, ReverseZ,
    SpotLight,
};
use korppi_render::shadows::{
    compute_cascades_into, CachedShadowAtlas, CascadeConfig, CasterType, FrameShadowPlan,
    ShadowCaster,
};
use korppi_rhi::{FrameIndex, GraphicsDevice};
use korppi_rhi_opengl::OpenGlDevice;
use korppi_rhi_vulkan::VulkanDevice;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;

thread_local! {
    static TRACKING: Cell<(bool, usize)> = const { Cell::new((false, 0)) };
}

// SAFETY: Delegates directly to the OS `System` allocator while incrementing a per-thread counter
// when tracking is enabled on the current thread.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = TRACKING.try_with(|cell| {
            let (active, count) = cell.get();
            if active {
                cell.set((true, count + 1));
            }
        });
        // SAFETY: Forwards `layout` unchanged to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Forwards `ptr` and `layout` unchanged to `System::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = TRACKING.try_with(|cell| {
            let (active, count) = cell.get();
            if active {
                cell.set((true, count + 1));
            }
        });
        // SAFETY: Forwards `ptr`, `layout`, and `new_size` unchanged to `System::realloc`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn count_allocations<R>(f: impl FnOnce() -> R) -> (usize, R) {
    TRACKING.with(|cell| cell.set((true, 0)));
    let out = f();
    let count = TRACKING.with(|cell| {
        let (_, c) = cell.get();
        cell.set((false, 0));
        c
    });
    (count, out)
}

fn devices() -> Vec<(&'static str, Box<dyn GraphicsDevice>)> {
    let vk = VulkanDevice::create_headless().expect("headless Vulkan device");
    let gl = OpenGlDevice::create_headless().expect("headless OpenGL device");
    vec![("Vulkan", Box::new(vk)), ("OpenGL", Box::new(gl))]
}

#[test]
fn test_g11_cpu_and_gpu_culler_match_with_100_lights_per_froxel() {
    let config = ClusterGridConfig::new(64, 64, 64, 1, 0.1, 100.0);
    let proj = ReverseZ::perspective_rh(60f32.to_radians(), 1.0, 0.1, 100.0);
    let froxels = config.build_froxel_aabbs(proj.inverse());

    let point_lights: Vec<PointLight> = (0..100)
        .map(|i| PointLight {
            position_vs: Vec3::new(0.0, 0.0, -5.0 - i as f32 * 0.01),
            radius: 100.0,
            color: Vec3::ONE,
            intensity: 1.0,
        })
        .collect();

    let spot_lights: Vec<SpotLight> = (0..80)
        .map(|i| SpotLight {
            position_vs: Vec3::new(0.0, 0.0, -1.0),
            direction_vs: Vec3::NEG_Z,
            range: 50.0 + i as f32 * 0.1,
            half_angle: 0.8,
            color: Vec3::ONE,
            intensity: 1.0,
        })
        .collect();

    let cpu = ClusteredLightCuller::new(&config).cull_lights(&froxels, &point_lights, &spot_lights);
    assert_eq!(cpu.cluster_records[0].point_count, 100);
    assert_eq!(cpu.cluster_records[0].spot_count, 80);

    for (backend, device) in devices() {
        let mut gpu_culler = GpuLightCuller::new();
        gpu_culler
            .init(device.as_ref())
            .expect("init GpuLightCuller");
        let gpu = gpu_culler
            .cull_lights(device.as_ref(), &froxels, &point_lights, &spot_lights)
            .expect("cull_lights on GPU");

        assert_eq!(
            gpu.cluster_records[0].point_count, cpu.cluster_records[0].point_count,
            "{backend} G11: GPU culler must keep all 100 point lights matching CPU culler"
        );
        assert_eq!(
            gpu.cluster_records[0].spot_count, cpu.cluster_records[0].spot_count,
            "{backend} G11: GPU culler must keep all 80 spot lights matching CPU culler"
        );
        assert_eq!(
            gpu.global_light_indices, cpu.global_light_indices,
            "{backend} G11: GPU and CPU global light index lists must match identically"
        );

        gpu_culler
            .destroy(device.as_ref())
            .expect("destroy GpuLightCuller");
        assert!(device
            .take_diagnostics()
            .into_iter()
            .all(|d| d.severity != korppi_rhi::DiagnosticSeverity::Error));
    }
}

#[test]
fn test_rnd12_gpu_culler_per_frame_in_flight_buffers_are_distinct() {
    let config = ClusterGridConfig::new(128, 128, 64, 4, 0.1, 100.0);
    let proj = ReverseZ::perspective_rh(60f32.to_radians(), 1.0, 0.1, 100.0);
    let froxels = config.build_froxel_aabbs(proj.inverse());
    let point_lights = [PointLight {
        position_vs: Vec3::new(0.0, 0.0, -5.0),
        radius: 10.0,
        color: Vec3::ONE,
        intensity: 1.0,
    }];

    for (backend, device) in devices() {
        let mut gpu_culler = GpuLightCuller::new();
        gpu_culler
            .init(device.as_ref())
            .expect("init GpuLightCuller");

        let mut rec0 = device.create_command_recorder().expect("rec0");
        rec0.begin().expect("begin0");
        let b0 = gpu_culler
            .dispatch_culling_for_frame(
                device.as_ref(),
                rec0.as_mut(),
                FrameIndex::new(0),
                &froxels,
                &point_lights,
                &[],
                &[],
                &[],
            )
            .expect("dispatch frame 0");
        let cmd0 = rec0.finish().expect("finish0");
        device.submit_and_wait(cmd0).expect("submit0");

        let mut rec1 = device.create_command_recorder().expect("rec1");
        rec1.begin().expect("begin1");
        let b1 = gpu_culler
            .dispatch_culling_for_frame(
                device.as_ref(),
                rec1.as_mut(),
                FrameIndex::new(1),
                &froxels,
                &point_lights,
                &[],
                &[],
                &[],
            )
            .expect("dispatch frame 1");
        let cmd1 = rec1.finish().expect("finish1");
        device.submit_and_wait(cmd1).expect("submit1");

        assert_ne!(
            b0.points_buffer, b1.points_buffer,
            "{backend} RND-12: frame 0 and frame 1 must use distinct point light input buffers"
        );
        assert_ne!(
            b0.spots_buffer, b1.spots_buffer,
            "{backend} RND-12: frame 0 and frame 1 must use distinct spot light input buffers"
        );

        gpu_culler
            .destroy(device.as_ref())
            .expect("destroy GpuLightCuller");
        assert!(device
            .take_diagnostics()
            .into_iter()
            .all(|d| d.severity != korppi_rhi::DiagnosticSeverity::Error));
    }
}

#[test]
fn test_g14_cpu_render_path_zero_heap_allocations_in_steady_state() {
    let config = ClusterGridConfig::new(1920, 1080, 64, 24, 0.1, 100.0);
    let proj = ReverseZ::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
    let inv_proj = proj.inverse();
    let mut froxels = Vec::new();
    config.build_froxel_aabbs_into(inv_proj, &mut froxels);

    let mut culler = ClusteredLightCuller::new(&config);
    let lights: Vec<PointLight> = (0..64)
        .map(|i| PointLight {
            position_vs: Vec3::new((i % 8) as f32 - 4.0, (i / 8) as f32 - 4.0, -10.0),
            radius: 2.0,
            color: Vec3::ONE,
            intensity: 1.0,
        })
        .collect();

    let mut cull_out = CullingOutput::default();
    culler.cull_lights_into(&froxels, &lights, &[], &mut cull_out);

    let view = Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y);
    let csm_cfg = CascadeConfig::default();
    let mut cascades = Vec::with_capacity(csm_cfg.cascade_count);
    compute_cascades_into(
        view,
        proj,
        Vec3::new(-0.3, -1.0, -0.2),
        &csm_cfg,
        &mut cascades,
    );

    let mut atlas = CachedShadowAtlas::new(4096);
    let lights_shadow: Vec<(u64, Vec3, f32)> = (0..16)
        .map(|i| (i, Vec3::new(i as f32, 0.0, 0.0), 5.0))
        .collect();
    let casters: Vec<ShadowCaster> = (0..64)
        .map(|i| ShadowCaster {
            id: i,
            caster_type: if i % 2 == 0 {
                CasterType::Static
            } else {
                CasterType::Dynamic
            },
            aabb_min: Vec3::new(i as f32 * 0.25, 0.0, 0.0),
            aabb_max: Vec3::new(i as f32 * 0.25 + 1.0, 1.0, 1.0),
        })
        .collect();

    let mut shadow_plan = FrameShadowPlan::default();
    // Frame 0 (cold) + Frame 1 (warm) populate and recycle caster vectors
    atlas.prepare_frame_into(&lights_shadow, &casters, &mut shadow_plan);
    atlas.prepare_frame_into(&lights_shadow, &casters, &mut shadow_plan);

    // Measure steady-state allocations on the `*_into` frame path
    let (froxel_allocs, ()) = count_allocations(|| {
        config.build_froxel_aabbs_into(inv_proj, &mut froxels);
    });
    let (cull_allocs, ()) = count_allocations(|| {
        culler.cull_lights_into(&froxels, &lights, &[], &mut cull_out);
    });
    let (csm_allocs, ()) = count_allocations(|| {
        compute_cascades_into(
            view,
            proj,
            Vec3::new(-0.3, -1.0, -0.2),
            &csm_cfg,
            &mut cascades,
        );
    });
    let (plan_into_allocs, ()) = count_allocations(|| {
        atlas.prepare_frame_into(&lights_shadow, &casters, &mut shadow_plan);
    });

    // Also verify `cull_lights` and `prepare_frame` by value reuse pooled vectors after warm-up
    let _ = culler.cull_lights(&froxels, &lights, &[]);
    let (cull_val_allocs, _) = count_allocations(|| culler.cull_lights(&froxels, &lights, &[]));
    let _ = atlas.prepare_frame(&lights_shadow, &casters);
    let (plan_val_allocs, _) = count_allocations(|| atlas.prepare_frame(&lights_shadow, &casters));

    assert_eq!(
        froxel_allocs, 0,
        "G14: build_froxel_aabbs_into must allocate 0 times"
    );
    assert_eq!(
        cull_allocs, 0,
        "G14: cull_lights_into must allocate 0 times"
    );
    assert_eq!(
        cull_val_allocs, 0,
        "G14: cull_lights must allocate 0 times after warm-up"
    );
    assert_eq!(
        csm_allocs, 0,
        "G14: compute_cascades_into must allocate 0 times"
    );
    assert_eq!(
        plan_into_allocs, 0,
        "G14: prepare_frame_into must allocate 0 times"
    );
    assert_eq!(
        plan_val_allocs, 0,
        "G14: prepare_frame must allocate 0 times after warm-up"
    );
}
