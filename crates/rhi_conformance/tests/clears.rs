//! Clear commands (FIX-1.5 review; finding RHI-05): `fill_buffer` and `clear_texture` outside
//! render passes, on both backends.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferHandle, BufferUsage, ImageLayout, ImageMemoryBarrier,
    MemoryUsage, PipelineBarrier, PipelineStageFlags, RhiError, SampleCount, TextureClear,
    TextureDescriptor, TextureDimension, TextureFormat, TextureHandle, TextureSubresourceRange,
    TextureUsage,
};
use korppi_rhi_conformance::{Backend, TestDevice};

fn buffer(device: &TestDevice, size: u64, usage: BufferUsage) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size,
                usage,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            Some(&vec![0; size as usize]),
        )
        .expect("buffer")
}

fn texture(
    device: &TestDevice,
    format: TextureFormat,
    layers: u32,
    mips: u32,
    usage: TextureUsage,
) -> TextureHandle {
    device
        .create_texture(
            &TextureDescriptor {
                width: 4,
                height: 4,
                depth: 1,
                array_layers: layers,
                mip_levels: mips,
                sample_count: SampleCount::Count1,
                format,
                usage,
                dimension: if layers > 1 {
                    TextureDimension::D2Array
                } else {
                    TextureDimension::D2
                },
                label: None,
            },
            None,
        )
        .expect("texture")
}

fn to_transfer_dst(texture: TextureHandle) -> PipelineBarrier {
    let mut barrier = PipelineBarrier::default();
    barrier.image_barriers.push(ImageMemoryBarrier {
        texture,
        range: TextureSubresourceRange::ALL,
        src_stages: PipelineStageFlags::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStageFlags::TRANSFER,
        dst_access: AccessFlags::TRANSFER_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::TransferDst,
    });
    barrier
}

/// Orders two transfer writes to the same subresources.
fn transfer_write_after_write(texture: TextureHandle) -> PipelineBarrier {
    let mut barrier = to_transfer_dst(texture);
    let b = &mut barrier.image_barriers[0];
    b.src_stages = PipelineStageFlags::TRANSFER;
    b.src_access = AccessFlags::TRANSFER_WRITE;
    b.old_layout = ImageLayout::TransferDst;
    barrier
}

fn fill_buffer_writes_the_word_into_the_range(backend: Backend) {
    let device = TestDevice::open(backend);
    let target = buffer(&device, 64, BufferUsage::STORAGE | BufferUsage::COPY_DST);
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    r.fill_buffer(target, 16, 32, 0xDEAD_BEEF);
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");
    let bytes = device.read_buffer(target, 0, 64).expect("read");
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_ne_bytes(*w))
        .collect();
    let mut expected = [0_u32; 16];
    expected[4..12].fill(0xDEAD_BEEF);
    assert_eq!(words, expected, "{backend:?}");
}

#[test]
fn fill_buffer_writes_the_word_into_the_range_vulkan() {
    fill_buffer_writes_the_word_into_the_range(Backend::Vulkan);
}

#[test]
fn fill_buffer_writes_the_word_into_the_range_opengl() {
    fill_buffer_writes_the_word_into_the_range(Backend::OpenGl);
}

fn clear_texture_clears_the_range(backend: Backend) {
    let device = TestDevice::open(backend);
    let usage = TextureUsage::COPY_DST | TextureUsage::COPY_SRC | TextureUsage::SAMPLED;
    let color = texture(&device, TextureFormat::Rgba8Unorm, 2, 2, usage);
    let uint = texture(&device, TextureFormat::R32Uint, 1, 1, usage);
    let depth = texture(&device, TextureFormat::Depth32Float, 1, 1, usage);
    let mut r = device.create_command_recorder().expect("recorder");
    r.begin().expect("begin");
    let mut barrier = to_transfer_dst(color);
    barrier
        .image_barriers
        .extend(to_transfer_dst(uint).image_barriers);
    barrier
        .image_barriers
        .extend(to_transfer_dst(depth).image_barriers);
    r.pipeline_barrier(&barrier);
    // Everything red, then mip 1 of layer 1 green.
    r.clear_texture(
        color,
        &TextureSubresourceRange::ALL,
        TextureClear::Color([1.0, 0.0, 0.0, 1.0]),
    );
    r.pipeline_barrier(&transfer_write_after_write(color));
    r.clear_texture(
        color,
        &TextureSubresourceRange::single(1, 1),
        TextureClear::Color([0.0, 1.0, 0.0, 1.0]),
    );
    r.clear_texture(
        uint,
        &TextureSubresourceRange::ALL,
        TextureClear::Color([7.0, 0.0, 0.0, 0.0]),
    );
    r.clear_texture(
        depth,
        &TextureSubresourceRange::ALL,
        TextureClear::DepthStencil {
            depth: 0.25,
            stencil: 0,
        },
    );
    device
        .submit_and_wait(r.finish().expect("finish"))
        .expect("submit");

    let red = [255, 0, 0, 255];
    let green = [0, 255, 0, 255];
    let mip0 = device.read_texture(color, 0).expect("read mip 0");
    assert!(
        mip0.as_chunks::<4>().0.iter().all(|p| *p == red),
        "{backend:?}: mip 0 of both layers is red"
    );
    let mip1 = device.read_texture(color, 1).expect("read mip 1");
    let (layer0, layer1) = mip1.split_at(mip1.len() / 2);
    assert!(
        layer0.as_chunks::<4>().0.iter().all(|p| *p == red),
        "{backend:?}"
    );
    assert!(
        layer1.as_chunks::<4>().0.iter().all(|p| *p == green),
        "{backend:?}"
    );
    let uints = device.read_texture(uint, 0).expect("read uint");
    assert!(
        uints
            .as_chunks::<4>()
            .0
            .iter()
            .all(|w| u32::from_ne_bytes(*w) == 7),
        "{backend:?}: integer formats clear to integer values"
    );
    let depths = device.read_texture(depth, 0).expect("read depth");
    assert!(
        depths
            .as_chunks::<4>()
            .0
            .iter()
            .all(|w| f32::from_ne_bytes(*w) == 0.25),
        "{backend:?}: depth"
    );
}

#[test]
fn clear_texture_clears_the_range_vulkan() {
    clear_texture_clears_the_range(Backend::Vulkan);
}

#[test]
fn clear_texture_clears_the_range_opengl() {
    clear_texture_clears_the_range(Backend::OpenGl);
}

fn invalid_clears_are_rejected(backend: Backend) {
    let device = TestDevice::open(backend);
    let no_copy_dst = texture(
        &device,
        TextureFormat::Rgba8Unorm,
        1,
        1,
        TextureUsage::SAMPLED,
    );
    let depth = texture(
        &device,
        TextureFormat::Depth32Float,
        1,
        1,
        TextureUsage::COPY_DST,
    );
    let color = texture(
        &device,
        TextureFormat::Rgba8Unorm,
        1,
        1,
        TextureUsage::COPY_DST,
    );
    let target = buffer(&device, 64, BufferUsage::COPY_DST);
    let unfillable = buffer(&device, 64, BufferUsage::STORAGE);
    let red = TextureClear::Color([1.0, 0.0, 0.0, 1.0]);
    type Record = Box<dyn Fn(&mut dyn korppi_rhi::CommandBufferRecorder)>;
    let cases: Vec<(&str, Record)> = vec![
        (
            "no COPY_DST usage",
            Box::new(move |r| r.clear_texture(no_copy_dst, &TextureSubresourceRange::ALL, red)),
        ),
        (
            "colour value for a depth format",
            Box::new(move |r| r.clear_texture(depth, &TextureSubresourceRange::ALL, red)),
        ),
        (
            "depth value for a colour format",
            Box::new(move |r| {
                r.clear_texture(
                    color,
                    &TextureSubresourceRange::ALL,
                    TextureClear::DepthStencil {
                        depth: 1.0,
                        stencil: 0,
                    },
                )
            }),
        ),
        (
            "mip outside the texture",
            Box::new(move |r| r.clear_texture(color, &TextureSubresourceRange::single(1, 0), red)),
        ),
        (
            "unaligned fill",
            Box::new(move |r| r.fill_buffer(target, 2, 8, 0)),
        ),
        (
            "fill past the end",
            Box::new(move |r| r.fill_buffer(target, 32, 64, 0)),
        ),
        (
            "fill without COPY_DST",
            Box::new(move |r| r.fill_buffer(unfillable, 0, 16, 0)),
        ),
    ];
    for (what, record) in cases {
        let mut r = device.create_command_recorder().expect("recorder");
        r.begin().expect("begin");
        record(&mut *r);
        let result = r.finish().map(|_| ());
        assert!(
            matches!(result, Err(RhiError::ValidationError(_))),
            "{backend:?} {what}: {result:?}"
        );
    }
}

#[test]
fn invalid_clears_are_rejected_vulkan() {
    invalid_clears_are_rejected(Backend::Vulkan);
}

#[test]
fn invalid_clears_are_rejected_opengl() {
    invalid_clears_are_rejected(Backend::OpenGl);
}
