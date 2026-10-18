//! Ogg Vorbis decoding (FIX-7.4; finding AUD-09): `lewton`, a pure Rust decoder, chosen by the
//! owner under rule 2. A file decodes whole into an [`AudioClip`] when it loads; the mixer plays
//! it like any other clip.
//!
//! [`AudioClipLoader`] takes WAV and Ogg Vorbis files alike, choosing by their first bytes: the
//! asset server keeps one loader per asset type, so it is the loader to register there.

use super::clip::AudioClip;
use crate::error::{AudioError, AudioResult};
use korppi_asset::{AssetError, AssetLoader};
use lewton::inside_ogg::OggStreamReader;
use lewton::samples::InterleavedSamples;
use std::io::Cursor;
use std::panic::{catch_unwind, AssertUnwindSafe};

impl AudioClip {
    /// Decodes an audio clip from Ogg Vorbis file bytes (the first logical stream).
    ///
    /// # Errors
    /// [`AudioError::AssetError`] if the bytes are not an Ogg Vorbis stream, or the stream is
    /// malformed. A decoder panic on malformed data is caught and reported as an error, since
    /// asset files can come from anywhere.
    pub fn from_ogg_vorbis_bytes(bytes: &[u8]) -> AudioResult<Self> {
        let error = |message: String| AudioError::AssetError(message);
        let decoded = catch_unwind(AssertUnwindSafe(|| -> AudioResult<Self> {
            let mut reader = OggStreamReader::new(Cursor::new(bytes))
                .map_err(|e| error(format!("not an Ogg Vorbis stream: {e}")))?;
            let channels = u16::from(reader.ident_hdr.audio_channels);
            let sample_rate = reader.ident_hdr.audio_sample_rate;
            if channels == 0 || sample_rate == 0 {
                return Err(error(
                    "an Ogg Vorbis stream without channels or rate".into(),
                ));
            }
            let mut samples = Vec::new();
            while let Some(packet) = reader
                .read_dec_packet_generic::<InterleavedSamples<f32>>()
                .map_err(|e| error(format!("malformed Ogg Vorbis audio: {e}")))?
            {
                samples.extend_from_slice(&packet.samples);
            }
            // The last page's granule position is the stream's length in frames (Vorbis I, A.2).
            // `lewton`'s output starts on the stream's first frame but can run past its end (by
            // 704 frames for the 1 s test file: its trim of the last packet relies on a running
            // count that comes out too high), so keep that many frames, or a clip and its loops
            // would end late.
            if let Some(length) = reader.get_last_absgp() {
                let length = usize::try_from(length).unwrap_or(usize::MAX);
                samples.truncate(length.saturating_mul(usize::from(channels)));
            }
            Ok(Self::new(channels, sample_rate, samples))
        }));
        decoded.unwrap_or_else(|_| {
            Err(error(
                "the Ogg Vorbis decoder failed on malformed data".into(),
            ))
        })
    }
}

/// Loader decoding `.ogg` and `.oga` Ogg Vorbis files into [`AudioClip`] assets.
#[derive(Debug, Default, Clone, Copy)]
pub struct OggVorbisLoader;

impl AssetLoader for OggVorbisLoader {
    type Asset = AudioClip;

    fn extensions(&self) -> &[&str] {
        &["ogg", "oga"]
    }

    fn load(&self, bytes: &[u8], path: &str) -> Result<Self::Asset, AssetError> {
        AudioClip::from_ogg_vorbis_bytes(bytes).map_err(|err| {
            AssetError::Decode(format!("Failed to decode Ogg Vorbis at '{path}': {err}"))
        })
    }
}

/// Loader decoding WAV and Ogg Vorbis files into [`AudioClip`] assets, by their first bytes
/// (`RIFF` or `OggS`): the loader to register with the asset server.
#[derive(Debug, Default, Clone, Copy)]
pub struct AudioClipLoader;

impl AssetLoader for AudioClipLoader {
    type Asset = AudioClip;

    fn extensions(&self) -> &[&str] {
        &["wav", "wave", "ogg", "oga"]
    }

    fn load(&self, bytes: &[u8], path: &str) -> Result<Self::Asset, AssetError> {
        match bytes.get(..4) {
            Some(b"OggS") => OggVorbisLoader.load(bytes, path),
            _ => super::wav::WavAudioLoader.load(bytes, path),
        }
    }
}
