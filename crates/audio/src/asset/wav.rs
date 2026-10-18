//! WAV file asset loader integrating with the engine asset pipeline.

use super::clip::AudioClip;
use korppi_asset::{AssetError, AssetLoader};

/// Loader decoding `.wav` and `.wave` audio files into in-memory [`AudioClip`] assets.
#[derive(Debug, Default, Clone, Copy)]
pub struct WavAudioLoader;

impl AssetLoader for WavAudioLoader {
    type Asset = AudioClip;

    fn extensions(&self) -> &[&str] {
        &["wav", "wave"]
    }

    fn load(&self, bytes: &[u8], path: &str) -> Result<Self::Asset, AssetError> {
        AudioClip::from_wav_bytes(bytes)
            .map_err(|err| AssetError::Decode(format!("Failed to decode WAV at '{path}': {err}")))
    }
}
