//! Pipeline-cache persistence (FIX-1.5; findings RVK-16, RGL-12): cache data of one device
//! seeds another of the same backend and driver, and foreign data is ignored.

use korppi_rhi::{
    BufferDescriptor, BufferUsage, ComputePipelineDescriptor, MemoryUsage, ShaderStage,
    ShaderStages,
};
use korppi_rhi_conformance::{buffer_range, Backend, TestDevice, STORAGE};

const WRITE: &str = r"#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer Out { uint value; };
void main() { value = 42u; }
";

/// Creates the pipeline on `device`, runs it and checks its result.
fn run(device: &TestDevice) {
    let output = device
        .create_buffer(
            &BufferDescriptor {
                size: 4,
                usage: BufferUsage::STORAGE,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            Some(&[0; 4]),
        )
        .expect("buffer");
    let (layout, group) = device.bind_group(&[(
        0,
        STORAGE,
        ShaderStages::COMPUTE,
        buffer_range(output, 0, 4),
    )]);
    let pipeline = device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            compute_shader: device.shader(ShaderStage::Compute, WRITE),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
            label: Some("cached"),
        })
        .expect("pipeline");
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    r.bind_pipeline(pipeline);
    r.set_bind_group(0, group, &[]);
    r.dispatch(1, 1, 1);
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");
    assert_eq!(
        device.read_buffer(output, 0, 4).expect("read"),
        42u32.to_ne_bytes()
    );
}

#[test]
fn cache_data_seeds_a_new_device_and_foreign_data_is_ignored() {
    let mut data = Vec::new();
    for backend in Backend::ALL {
        let first = TestDevice::open(backend);
        run(&first);
        let cached = first.pipeline_cache_data().expect("pipeline_cache_data");
        drop(first);

        let second = TestDevice::open(backend);
        assert!(
            !second.load_pipeline_cache(b"not a cache").expect("load"),
            "{backend:?}: garbage is ignored"
        );
        if cached.is_empty() {
            eprintln!("{backend:?}: the driver keeps no pipeline cache");
        } else {
            assert!(
                second.load_pipeline_cache(&cached).expect("load"),
                "{backend:?}: its own data is used"
            );
            let mut truncated = cached.clone();
            truncated.truncate(cached.len() / 2);
            let _ = second
                .load_pipeline_cache(&truncated)
                .expect("truncated data does not fail");
        }
        // Pipelines created after loading work as before.
        run(&second);
        data.push((backend, cached));
    }
    // Neither backend uses the other's data.
    let [(_, vulkan), (_, opengl)] = [data[0].clone(), data[1].clone()];
    let device = TestDevice::open(Backend::Vulkan);
    assert!(!device.load_pipeline_cache(&opengl).expect("load"));
    let device = TestDevice::open(Backend::OpenGl);
    assert!(!device.load_pipeline_cache(&vulkan).expect("load"));
}

#[test]
fn opengl_reuses_program_binaries_across_devices() {
    let first = korppi_rhi_opengl::OpenGlDevice::create_headless().expect("device");
    let device = TestDevice::open(Backend::OpenGl);
    drop(device);
    let wrapped = TestDevice::open(Backend::OpenGl);
    run(&wrapped);
    let data = wrapped.pipeline_cache_data().expect("data");
    if data.is_empty() {
        eprintln!("the driver offers no program binary formats");
        return;
    }
    use korppi_rhi::GraphicsDevice;
    assert_eq!(first.program_binary_count(), 0);
    assert!(first.load_pipeline_cache(&data).expect("load"));
    assert!(
        first.program_binary_count() > 0,
        "binaries are kept after loading"
    );
}
