//! Texture asset types and decoders (PNG, JPEG, KTX2).

use crate::error::AssetError;
use crate::handle::Asset;
use crate::loader::AssetLoader;

/// GPU texture format representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// Single channel 8-bit unsigned normalized.
    R8,
    /// Dual channel 8-bit unsigned normalized.
    Rg8,
    /// Triple channel 8-bit unsigned normalized.
    Rgb8,
    /// Four channel 8-bit unsigned normalized.
    Rgba8,
    /// Four channel 8-bit sRGB.
    Rgba8Srgb,
    /// Single channel 16-bit floating point.
    R16F,
    /// Four channel 16-bit floating point.
    Rgba16F,
    /// Four channel 32-bit floating point.
    Rgba32F,
    /// Block compressed BC1 (DXT1) format.
    Bc1Rgba,
    /// Block compressed BC3 (DXT5) format.
    Bc3Rgba,
    /// Block compressed BC5 format.
    Bc5Rg,
    /// Block compressed BC7 format.
    Bc7Rgba,
}

/// In-memory CPU representation of a decoded texture.
#[derive(Debug, Clone)]
pub struct Texture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: TextureFormat,
    /// Raw uncompressed or block-compressed pixel buffer.
    pub data: Vec<u8>,
    /// Number of mipmap levels present.
    pub mip_levels: u32,
}

impl Asset for Texture {
    const TYPE_NAME: &'static str = "Texture";
}

/// Loader capable of decoding PNG, JPEG, and KTX2 texture formats.
#[derive(Debug, Default, Clone, Copy)]
pub struct TextureLoader;

impl TextureLoader {
    /// Creates a new `TextureLoader`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn decode_ktx2(&self, bytes: &[u8]) -> Result<Texture, AssetError> {
        const KTX2_MAGIC: [u8; 12] = [
            0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
        ];

        if bytes.len() < 80 || bytes[0..12] != KTX2_MAGIC {
            return Err(AssetError::Decode("Invalid KTX2 magic identifier".into()));
        }

        let vk_format = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let _type_size = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let pixel_width = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let pixel_height = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        let _pixel_depth = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
        let _layer_count = u32::from_le_bytes(bytes[32..36].try_into().unwrap());
        let _face_count = u32::from_le_bytes(bytes[36..40].try_into().unwrap());
        let level_count = u32::from_le_bytes(bytes[40..44].try_into().unwrap()).max(1);
        let supercompression = u32::from_le_bytes(bytes[44..48].try_into().unwrap());

        if pixel_width == 0 || pixel_height == 0 {
            return Err(AssetError::Decode(
                "KTX2 pixel_width and pixel_height must be non-zero".into(),
            ));
        }

        if supercompression != 0 {
            return Err(AssetError::UnsupportedFormat(format!(
                "KTX2 supercompression scheme {supercompression} is not supported"
            )));
        }

        // Map vk_format
        let format = match vk_format {
            9 => TextureFormat::R8,         // VK_FORMAT_R8_UNORM
            16 => TextureFormat::Rg8,       // VK_FORMAT_R8G8_UNORM
            23 => TextureFormat::Rgb8,      // VK_FORMAT_R8G8B8_UNORM
            37 => TextureFormat::Rgba8,     // VK_FORMAT_R8G8B8A8_UNORM
            43 => TextureFormat::Rgba8Srgb, // VK_FORMAT_R8G8B8A8_SRGB
            76 => TextureFormat::R16F,      // VK_FORMAT_R16_SFLOAT
            97 => TextureFormat::Rgba16F,   // VK_FORMAT_R16G16B16A16_SFLOAT
            109 => TextureFormat::Rgba32F,  // VK_FORMAT_R32G32B32A32_SFLOAT
            131 => TextureFormat::Bc1Rgba,  // VK_FORMAT_BC1_RGBA_UNORM_BLOCK
            137 => TextureFormat::Bc3Rgba,  // VK_FORMAT_BC3_UNORM_BLOCK
            141 => TextureFormat::Bc5Rg,    // VK_FORMAT_BC5_UNORM_BLOCK
            145 => TextureFormat::Bc7Rgba,  // VK_FORMAT_BC7_UNORM_BLOCK
            other => {
                return Err(AssetError::UnsupportedFormat(format!(
                    "Unsupported KTX2 VkFormat {other}"
                )))
            }
        };

        // Level index begins at offset 80 (12 magic + 68 header fields)
        let level_index_offset: usize = 80;
        let mut level_cursor = level_index_offset;

        let index_end = (level_count as usize)
            .checked_mul(24)
            .and_then(|sz| level_index_offset.checked_add(sz))
            .ok_or_else(|| AssetError::Decode("KTX2 level index size overflow".into()))?;
        if index_end > bytes.len() {
            return Err(AssetError::Decode("Truncated KTX2 level index".into()));
        }

        let mut data = Vec::new();
        for _ in 0..level_count {
            let offset_u64 =
                u64::from_le_bytes(bytes[level_cursor..level_cursor + 8].try_into().unwrap());
            level_cursor += 8;
            let length_u64 =
                u64::from_le_bytes(bytes[level_cursor..level_cursor + 8].try_into().unwrap());
            level_cursor += 8;
            let _uncompressed_length =
                u64::from_le_bytes(bytes[level_cursor..level_cursor + 8].try_into().unwrap());
            level_cursor += 8;

            let byte_offset = usize::try_from(offset_u64)
                .map_err(|_| AssetError::Decode("KTX2 mip level offset overflow".into()))?;
            let byte_length = usize::try_from(length_u64)
                .map_err(|_| AssetError::Decode("KTX2 mip level length overflow".into()))?;
            if byte_length == 0 {
                return Err(AssetError::Decode("KTX2 mip level has zero length".into()));
            }
            let end = byte_offset
                .checked_add(byte_length)
                .ok_or_else(|| AssetError::Decode("KTX2 mip level range overflow".into()))?;
            if end > bytes.len() {
                return Err(AssetError::Decode(
                    "KTX2 mip level extends past file end".into(),
                ));
            }

            data.extend_from_slice(&bytes[byte_offset..end]);
        }

        Ok(Texture {
            width: pixel_width,
            height: pixel_height,
            format,
            data,
            mip_levels: level_count,
        })
    }
}

impl AssetLoader for TextureLoader {
    type Asset = Texture;

    fn extensions(&self) -> &[&str] {
        &["png", "jpg", "jpeg", "ktx2"]
    }

    fn load(&self, bytes: &[u8], path: &str) -> Result<Self::Asset, AssetError> {
        let ext = path
            .rsplit_once('.')
            .map(|(_, e)| e.to_ascii_lowercase())
            .unwrap_or_default();

        if ext == "ktx2" {
            return self.decode_ktx2(bytes);
        }
        if !matches!(ext.as_str(), "png" | "jpg" | "jpeg") {
            return Err(AssetError::UnsupportedFormat(ext));
        }

        let dynamic_img = image::load_from_memory(bytes)
            .map_err(|e| AssetError::Decode(format!("Image decode error: {e}")))?;

        let rgba = dynamic_img.to_rgba8();
        let (width, height) = rgba.dimensions();

        Ok(Texture {
            width,
            height,
            format: TextureFormat::Rgba8,
            data: rgba.into_raw(),
            mip_levels: 1,
        })
    }
}
