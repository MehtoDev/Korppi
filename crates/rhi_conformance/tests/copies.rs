//! Recorded transfers (FIX-1.4; finding RHI-05): texture-to-buffer and texture-to-texture
//! copies (layers, 3D slices, row pitches, compressed blocks), blits with scaling, filtering and
//! mirroring, and mipmap generation.

use korppi_rhi::{
    AccessFlags, BufferDescriptor, BufferHandle, BufferMemoryBarrier, BufferTextureCopy,
    BufferUsage, CommandBufferRecorder, DeviceFeatures, FilterMode, ImageLayout,
    ImageMemoryBarrier, MemoryUsage, PipelineBarrier, PipelineStageFlags, RhiError, SampleCount,
    TextureBlit, TextureCopy, TextureDescriptor, TextureDimension, TextureFormat, TextureHandle,
    TextureSubresourceRange, TextureUsage,
};
use korppi_rhi_conformance::{Backend, TestDevice};

const COPY: TextureUsage = TextureUsage::COPY_SRC.union(TextureUsage::COPY_DST);

fn texture(
    device: &TestDevice,
    dimension: TextureDimension,
    size: (u32, u32, u32),
    layers: u32,
    mips: u32,
    format: TextureFormat,
    data: Option<&[u8]>,
) -> TextureHandle {
    device
        .create_texture(
            &TextureDescriptor {
                width: size.0,
                height: size.1,
                depth: size.2,
                array_layers: layers,
                mip_levels: mips,
                sample_count: SampleCount::Count1,
                format,
                usage: COPY,
                dimension,
                label: None,
            },
            data,
        )
        .expect("create_texture")
}

fn readback_buffer(device: &TestDevice, size: u64) -> BufferHandle {
    device
        .create_buffer(
            &BufferDescriptor {
                size,
                usage: BufferUsage::COPY_DST,
                memory: MemoryUsage::GpuToCpu,
                label: None,
            },
            None,
        )
        .expect("create_buffer")
}

/// A layout transition of `range` of `texture` between transfer uses.
fn transition(
    texture: TextureHandle,
    range: TextureSubresourceRange,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
) -> PipelineBarrier {
    let access = |layout| match layout {
        ImageLayout::TransferSrc => AccessFlags::TRANSFER_READ,
        ImageLayout::TransferDst => AccessFlags::TRANSFER_WRITE,
        _ => AccessFlags::empty(),
    };
    PipelineBarrier::new().with_image_barrier(ImageMemoryBarrier {
        texture,
        range,
        src_stages: PipelineStageFlags::TRANSFER,
        src_access: access(old_layout) & AccessFlags::TRANSFER_WRITE,
        dst_stages: PipelineStageFlags::TRANSFER,
        dst_access: access(new_layout),
        old_layout,
        new_layout,
    })
}

fn to_host(buffer: BufferHandle, size: u64) -> PipelineBarrier {
    PipelineBarrier::new().with_buffer_barrier(BufferMemoryBarrier {
        buffer,
        src_stages: PipelineStageFlags::TRANSFER,
        src_access: AccessFlags::TRANSFER_WRITE,
        dst_stages: PipelineStageFlags::HOST,
        dst_access: AccessFlags::HOST_READ,
        offset: 0,
        size,
    })
}

fn record(device: &TestDevice, commands: impl FnOnce(&mut dyn CommandBufferRecorder)) {
    let mut recorder = device.create_command_recorder().expect("recorder");
    recorder.begin().expect("begin");
    commands(&mut *recorder);
    device
        .submit_and_wait(recorder.finish().expect("finish"))
        .expect("submit");
}

/// RGBA8 texel (x, y) of layer `layer`.
fn texel(x: u32, y: u32, layer: u32) -> [u8; 4] {
    [(x * 16) as u8, (y * 32) as u8, (layer * 50) as u8, 255]
}

fn rgba_image(width: u32, height: u32, layers: u32) -> Vec<u8> {
    (0..layers)
        .flat_map(|l| (0..height).flat_map(move |y| (0..width).flat_map(move |x| texel(x, y, l))))
        .collect()
}

#[test]
fn texture_regions_copy_into_buffers_with_a_row_pitch() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        // Layer 2 of an 8x4 array; created with data, the texture is in `TransferSrc`.
        let array = texture(
            &device,
            TextureDimension::D2Array,
            (8, 4, 1),
            3,
            1,
            TextureFormat::Rgba8Unorm,
            Some(&rgba_image(8, 4, 3)),
        );
        let buffer = readback_buffer(&device, 1024);
        record(&device, |r| {
            r.copy_texture_to_buffer(
                array,
                buffer,
                &BufferTextureCopy {
                    buffer_offset: 256,
                    buffer_row_pitch: 256,
                    texture_mip_level: 0,
                    texture_layer: 2,
                    origin: [2, 1, 0],
                    extent: [4, 2, 1],
                },
            );
            r.pipeline_barrier(&to_host(buffer, 1024));
        });
        let bytes = device.read_buffer(buffer, 0, 1024).expect("read_buffer");
        for row in 0..2u32 {
            let start = 256 + row as usize * 256;
            let expected: Vec<u8> = (2..6).flat_map(|x| texel(x, 1 + row, 2)).collect();
            assert_eq!(
                &bytes[start..start + 16],
                expected.as_slice(),
                "{backend:?}: row {row}"
            );
        }

        // Slices 1..3 of a 4x4x4 R8 volume, tightly packed.
        let volume_data: Vec<u8> = (0..64).map(|i| i as u8 * 3).collect();
        let volume = texture(
            &device,
            TextureDimension::D3,
            (4, 4, 4),
            1,
            1,
            TextureFormat::R8Unorm,
            Some(&volume_data),
        );
        let slices = readback_buffer(&device, 32);
        record(&device, |r| {
            r.copy_texture_to_buffer(
                volume,
                slices,
                &BufferTextureCopy {
                    buffer_offset: 0,
                    buffer_row_pitch: 0,
                    texture_mip_level: 0,
                    texture_layer: 0,
                    origin: [0, 0, 1],
                    extent: [4, 4, 2],
                },
            );
            r.pipeline_barrier(&to_host(slices, 32));
        });
        assert_eq!(
            device.read_buffer(slices, 0, 32).expect("read_buffer"),
            volume_data[16..48],
            "{backend:?}: volume slices"
        );
    }
}

#[test]
fn texture_regions_copy_between_textures_and_layers() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let src = texture(
            &device,
            TextureDimension::D2,
            (8, 8, 1),
            1,
            1,
            TextureFormat::Rgba8Unorm,
            Some(&rgba_image(8, 8, 1)),
        );
        let dst = texture(
            &device,
            TextureDimension::D2Array,
            (8, 8, 1),
            2,
            1,
            TextureFormat::Rgba8Unorm,
            Some(&vec![0; 8 * 8 * 4 * 2]),
        );
        record(&device, |r| {
            r.pipeline_barrier(&transition(
                dst,
                TextureSubresourceRange::ALL,
                ImageLayout::TransferSrc,
                ImageLayout::TransferDst,
            ));
            r.copy_texture_to_texture(
                src,
                dst,
                &TextureCopy {
                    src_mip_level: 0,
                    src_layer: 0,
                    src_origin: [4, 4, 0],
                    dst_mip_level: 0,
                    dst_layer: 1,
                    dst_origin: [1, 2, 0],
                    extent: [4, 4, 1],
                    layer_count: 1,
                },
            );
            r.pipeline_barrier(&transition(
                dst,
                TextureSubresourceRange::ALL,
                ImageLayout::TransferDst,
                ImageLayout::TransferSrc,
            ));
        });
        let layers = device.read_texture(dst, 0).expect("read_texture");
        let layer1 = &layers[8 * 8 * 4..];
        for y in 0..8u32 {
            for x in 0..8u32 {
                let i = ((y * 8 + x) * 4) as usize;
                let inside = (1..5).contains(&x) && (2..6).contains(&y);
                let expected = if inside {
                    texel(x - 1 + 4, y - 2 + 4, 0)
                } else {
                    [0; 4]
                };
                assert_eq!(&layer1[i..i + 4], &expected, "{backend:?}: ({x}, {y})");
            }
        }
        assert!(
            layers[..8 * 8 * 4].iter().all(|&b| b == 0),
            "{backend:?}: layer 0 untouched"
        );

        if device
            .features()
            .contains(DeviceFeatures::TEXTURE_COMPRESSION_BC)
        {
            // Four BC1 blocks, each a distinct byte pattern; copy block (1, 1) to block (0, 0).
            let blocks: Vec<u8> = (0..4u8).flat_map(|b| [b + 1; 8]).collect();
            let bc_src = texture(
                &device,
                TextureDimension::D2,
                (8, 8, 1),
                1,
                1,
                TextureFormat::Bc1RgbaUnorm,
                Some(&blocks),
            );
            let bc_dst = texture(
                &device,
                TextureDimension::D2,
                (8, 8, 1),
                1,
                1,
                TextureFormat::Bc1RgbaUnorm,
                Some(&[0; 32]),
            );
            record(&device, |r| {
                r.pipeline_barrier(&transition(
                    bc_dst,
                    TextureSubresourceRange::ALL,
                    ImageLayout::TransferSrc,
                    ImageLayout::TransferDst,
                ));
                r.copy_texture_to_texture(
                    bc_src,
                    bc_dst,
                    &TextureCopy {
                        src_mip_level: 0,
                        src_layer: 0,
                        src_origin: [4, 4, 0],
                        dst_mip_level: 0,
                        dst_layer: 0,
                        dst_origin: [0, 0, 0],
                        extent: [4, 4, 1],
                        layer_count: 1,
                    },
                );
                r.pipeline_barrier(&transition(
                    bc_dst,
                    TextureSubresourceRange::ALL,
                    ImageLayout::TransferDst,
                    ImageLayout::TransferSrc,
                ));
            });
            let copied = device.read_texture(bc_dst, 0).expect("read_texture");
            assert_eq!(&copied[..8], &[4; 8], "{backend:?}: BC1 block");
            assert!(
                copied[8..].iter().all(|&b| b == 0),
                "{backend:?}: other blocks"
            );
        }
    }
}

#[test]
fn blits_scale_filter_and_mirror() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        // 4x4 checkerboard of black and white.
        let checker: Vec<u8> = (0..16)
            .flat_map(|i| {
                let v = if (i % 4 + i / 4) % 2 == 0 { 255 } else { 0 };
                [v, v, v, 255]
            })
            .collect();
        let src = texture(
            &device,
            TextureDimension::D2,
            (4, 4, 1),
            1,
            1,
            TextureFormat::Rgba8Unorm,
            Some(&checker),
        );
        let dst = texture(
            &device,
            TextureDimension::D2,
            (4, 2, 1),
            1,
            1,
            TextureFormat::Rgba8Unorm,
            Some(&[0; 4 * 2 * 4]),
        );
        let blit = |dst_region, filter| TextureBlit {
            src_mip_level: 0,
            src_layer: 0,
            src_region: [0, 0, 4, 4],
            dst_mip_level: 0,
            dst_layer: 0,
            dst_region,
            layer_count: 1,
            filter,
        };
        record(&device, |r| {
            r.pipeline_barrier(&transition(
                dst,
                TextureSubresourceRange::ALL,
                ImageLayout::TransferSrc,
                ImageLayout::TransferDst,
            ));
            // Left 2x2: linear halves average every 2x2 block to grey.
            r.blit_texture(src, dst, &blit([0, 0, 2, 2], FilterMode::Linear));
            // Right 2x2: a mirrored nearest blit of the top-left 2x2 source quarter.
            r.blit_texture(
                src,
                dst,
                &TextureBlit {
                    src_region: [0, 0, 2, 2],
                    ..blit([4, 0, 2, 2], FilterMode::Nearest)
                },
            );
            r.pipeline_barrier(&transition(
                dst,
                TextureSubresourceRange::ALL,
                ImageLayout::TransferDst,
                ImageLayout::TransferSrc,
            ));
        });
        let out = device.read_texture(dst, 0).expect("read_texture");
        let px =
            |x: usize, y: usize| -> [u8; 4] { out[(y * 4 + x) * 4..][..4].try_into().unwrap() };
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let p = px(x, y);
            assert!(
                p[0].abs_diff(128) <= 1 && p[3] == 255,
                "{backend:?}: linear ({x}, {y}) = {p:?}"
            );
        }
        // Source quarter: (0,0) white, (1,0) black, (0,1) black, (1,1) white; mirrored in x.
        assert_eq!(px(2, 0), [0, 0, 0, 255], "{backend:?}: mirrored (2, 0)");
        assert_eq!(px(3, 0), [255; 4], "{backend:?}: mirrored (3, 0)");
        assert_eq!(px(2, 1), [255; 4], "{backend:?}: mirrored (2, 1)");
        assert_eq!(px(3, 1), [0, 0, 0, 255], "{backend:?}: mirrored (3, 1)");
    }
}

#[test]
fn mipmaps_are_generated_for_every_layer() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        // Layer 0: checkerboard; layer 1: uniform green.
        let checker: Vec<u8> = (0..16)
            .flat_map(|i| {
                let v = if (i % 4 + i / 4) % 2 == 0 { 255 } else { 0 };
                [v, v, v, 255]
            })
            .collect();
        let green: Vec<u8> = [0, 200, 0, 255].repeat(16);
        let data = [checker, green].concat();
        let tex = texture(
            &device,
            TextureDimension::D2Array,
            (4, 4, 1),
            2,
            3,
            TextureFormat::Rgba8Unorm,
            Some(&data),
        );
        record(&device, |r| {
            r.pipeline_barrier(&transition(
                tex,
                TextureSubresourceRange::single(0, 0),
                ImageLayout::TransferSrc,
                ImageLayout::TransferDst,
            ));
            r.pipeline_barrier(&transition(
                tex,
                TextureSubresourceRange {
                    base_mip_level: 0,
                    mip_level_count: 1,
                    base_array_layer: 1,
                    array_layer_count: 1,
                    ..TextureSubresourceRange::ALL
                },
                ImageLayout::TransferSrc,
                ImageLayout::TransferDst,
            ));
            r.pipeline_barrier(&transition(
                tex,
                TextureSubresourceRange {
                    base_mip_level: 1,
                    ..TextureSubresourceRange::ALL
                },
                ImageLayout::Undefined,
                ImageLayout::TransferDst,
            ));
            r.generate_mipmaps(tex);
        });
        for mip in 1..3u32 {
            let level = device.read_texture(tex, mip).expect("read_texture");
            let texels = (4 >> mip) * (4 >> mip);
            let (layer0, layer1) = level.split_at(texels * 4);
            for p in layer0.chunks(4) {
                assert!(
                    p[0].abs_diff(128) <= 2 && p[3] == 255,
                    "{backend:?}: mip {mip} layer 0 {p:?}"
                );
            }
            for p in layer1.chunks(4) {
                assert_eq!(p, [0, 200, 0, 255], "{backend:?}: mip {mip} layer 1");
            }
        }
    }
}

#[test]
fn invalid_transfers_are_reported() {
    if !korppi_rhi::api_validation_requested() {
        return;
    }
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let tex = texture(
            &device,
            TextureDimension::D2,
            (4, 4, 1),
            1,
            1,
            TextureFormat::Rgba8Unorm,
            None,
        );
        let other = texture(
            &device,
            TextureDimension::D2,
            (4, 4, 1),
            1,
            1,
            TextureFormat::R8Unorm,
            None,
        );
        let buffer = readback_buffer(&device, 32);
        let mut recorder = device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        // Past the end of a 32-byte buffer.
        recorder.copy_texture_to_buffer(
            tex,
            buffer,
            &BufferTextureCopy {
                buffer_offset: 0,
                buffer_row_pitch: 0,
                texture_mip_level: 0,
                texture_layer: 0,
                origin: [0, 0, 0],
                extent: [4, 4, 1],
            },
        );
        // Different formats.
        recorder.copy_texture_to_texture(
            tex,
            other,
            &TextureCopy {
                src_mip_level: 0,
                src_layer: 0,
                src_origin: [0, 0, 0],
                dst_mip_level: 0,
                dst_layer: 0,
                dst_origin: [0, 0, 0],
                extent: [1, 1, 1],
                layer_count: 1,
            },
        );
        // Beyond the texture.
        recorder.copy_texture_to_buffer(
            tex,
            buffer,
            &BufferTextureCopy {
                buffer_offset: 0,
                buffer_row_pitch: 0,
                texture_mip_level: 0,
                texture_layer: 0,
                origin: [3, 0, 0],
                extent: [2, 1, 1],
            },
        );
        match recorder.finish() {
            Err(RhiError::ValidationError(message)) => {
                for expected in ["of a 32-byte buffer", "formats differ", "exceeds mip 0"] {
                    assert!(message.contains(expected), "{backend:?}: {message}");
                }
            }
            other => panic!("{backend:?}: invalid transfers accepted: {:?}", other.err()),
        }
    }
}
