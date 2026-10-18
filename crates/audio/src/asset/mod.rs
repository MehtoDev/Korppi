//! Audio asset types and decoders.

pub mod clip;
pub mod vorbis;
pub mod wav;

pub use clip::AudioClip;
pub use vorbis::{AudioClipLoader, OggVorbisLoader};
pub use wav::WavAudioLoader;
