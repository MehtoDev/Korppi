//! Test suite for Texture Decoders (PNG, JPEG, KTX2).

use korppi_asset::loader::AssetLoader;
use korppi_asset::texture::{TextureFormat, TextureLoader};

#[test]
fn test_png_decoding() {
    // Generate an in-memory 2x2 RGBA PNG image
    let mut img = image::RgbaImage::new(2, 2);
    img.put_pixel(0, 0, image::Rgba([255, 0, 0, 255])); // Red
    img.put_pixel(1, 0, image::Rgba([0, 255, 0, 255])); // Green
    img.put_pixel(0, 1, image::Rgba([0, 0, 255, 255])); // Blue
    img.put_pixel(1, 1, image::Rgba([255, 255, 255, 255])); // White

    let mut png_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut png_bytes);
    img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();

    let loader = TextureLoader::new();
    let texture = loader.load(&png_bytes, "test.png").unwrap();

    assert_eq!(texture.width, 2);
    assert_eq!(texture.height, 2);
    assert_eq!(texture.format, TextureFormat::Rgba8);
    assert_eq!(texture.data.len(), 2 * 2 * 4);

    // Verify first pixel is Red
    assert_eq!(&texture.data[0..4], &[255, 0, 0, 255]);
    // Verify second pixel is Green
    assert_eq!(&texture.data[4..8], &[0, 255, 0, 255]);
}

#[test]
fn test_jpeg_decoding() {
    // Generate an in-memory 4x4 RGB JPEG image
    let mut img = image::RgbImage::new(4, 4);
    for pixel in img.pixels_mut() {
        *pixel = image::Rgb([128, 64, 32]);
    }

    let mut jpeg_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut jpeg_bytes);
    img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();

    let loader = TextureLoader::new();
    let texture = loader.load(&jpeg_bytes, "test.jpg").unwrap();

    assert_eq!(texture.width, 4);
    assert_eq!(texture.height, 4);
    assert_eq!(texture.format, TextureFormat::Rgba8);
    assert_eq!(texture.data.len(), 4 * 4 * 4);
}

#[test]
fn test_ktx2_container_decoding() {
    // Construct a valid KTX 2.0 container binary
    let mut ktx_bytes = Vec::new();

    // 1. Magic identifier: «KTX 20»\r\n\x1A\n
    ktx_bytes.extend_from_slice(&[
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ]);

    // 2. Header
    let vk_format = 37u32; // VK_FORMAT_R8G8B8A8_UNORM
    let type_size = 1u32;
    let pixel_width = 8u32;
    let pixel_height = 8u32;
    let pixel_depth = 0u32;
    let layer_count = 0u32;
    let face_count = 1u32;
    let level_count = 1u32;
    let supercompression = 0u32; // None

    ktx_bytes.extend_from_slice(&vk_format.to_le_bytes());
    ktx_bytes.extend_from_slice(&type_size.to_le_bytes());
    ktx_bytes.extend_from_slice(&pixel_width.to_le_bytes());
    ktx_bytes.extend_from_slice(&pixel_height.to_le_bytes());
    ktx_bytes.extend_from_slice(&pixel_depth.to_le_bytes());
    ktx_bytes.extend_from_slice(&layer_count.to_le_bytes());
    ktx_bytes.extend_from_slice(&face_count.to_le_bytes());
    ktx_bytes.extend_from_slice(&level_count.to_le_bytes());
    ktx_bytes.extend_from_slice(&supercompression.to_le_bytes());

    // dfdByteOffset, dfdByteLength, kvdByteOffset, kvdByteLength, sgdByteOffset, sgdByteLength
    let dfd_offset = 0u32;
    let dfd_length = 0u32;
    let kvd_offset = 0u32;
    let kvd_length = 0u32;
    let sgd_offset = 0u64;
    let sgd_length = 0u64;

    ktx_bytes.extend_from_slice(&dfd_offset.to_le_bytes());
    ktx_bytes.extend_from_slice(&dfd_length.to_le_bytes());
    ktx_bytes.extend_from_slice(&kvd_offset.to_le_bytes());
    ktx_bytes.extend_from_slice(&kvd_length.to_le_bytes());
    ktx_bytes.extend_from_slice(&sgd_offset.to_le_bytes());
    ktx_bytes.extend_from_slice(&sgd_length.to_le_bytes());

    // 3. Level Index (level_count = 1)
    let payload_size = (8 * 8 * 4) as u64;
    let header_size = 12 + 36 + 32 + 24; // 104 bytes
    let byte_offset = header_size as u64;
    let byte_length = payload_size;
    let uncompressed_length = payload_size;

    ktx_bytes.extend_from_slice(&byte_offset.to_le_bytes());
    ktx_bytes.extend_from_slice(&byte_length.to_le_bytes());
    ktx_bytes.extend_from_slice(&uncompressed_length.to_le_bytes());

    // 4. Mip payload
    ktx_bytes.resize(header_size + payload_size as usize, 0xFF);

    let loader = TextureLoader::new();
    let texture = loader.load(&ktx_bytes, "test.ktx2").unwrap();

    assert_eq!(texture.width, 8);
    assert_eq!(texture.height, 8);
    assert_eq!(texture.format, TextureFormat::Rgba8);
    assert_eq!(texture.data.len(), 8 * 8 * 4);
    assert_eq!(texture.mip_levels, 1);
}
