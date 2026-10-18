//! Audio clip asset containing multi-channel PCM audio buffer and metadata.

use crate::error::{AudioError, AudioResult};
use korppi_asset::Asset;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// In-memory audio clip containing uncompressed floating-point PCM audio data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioClip {
    /// Number of audio channels (1 = Mono, 2 = Stereo).
    pub channels: u16,
    /// Sampling rate in Hz (e.g. 44100, 48000).
    pub sample_rate: u32,
    /// Interleaved audio sample buffer normalized in [-1.0, 1.0].
    pub samples: Vec<f32>,
}

impl Asset for AudioClip {
    const TYPE_NAME: &'static str = "AudioClip";
}

impl AudioClip {
    /// Creates a new audio clip from raw interleaved PCM samples.
    #[must_use]
    pub fn new(channels: u16, sample_rate: u32, samples: Vec<f32>) -> Self {
        Self {
            channels: channels.max(1),
            sample_rate: sample_rate.max(1),
            samples,
        }
    }

    /// Computes the playback duration in seconds.
    #[must_use]
    pub fn duration_sec(&self) -> f32 {
        if self.channels == 0 || self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / (self.channels as f32 * self.sample_rate as f32)
    }

    /// Returns the total number of multi-channel audio frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    /// Reads a sample at the given frame and channel index.
    #[must_use]
    pub fn sample_at(&self, frame: usize, channel: usize) -> f32 {
        if channel >= self.channels as usize {
            return 0.0;
        }
        let index = frame * (self.channels as usize) + channel;
        self.samples.get(index).copied().unwrap_or(0.0)
    }

    /// Generates a synthetic mono sine wave audio clip.
    #[must_use]
    pub fn sine(frequency_hz: f32, duration_sec: f32, sample_rate: u32) -> Self {
        let total_frames = ((duration_sec * sample_rate as f32).round() as usize).max(1);
        let mut samples = Vec::with_capacity(total_frames);

        for n in 0..total_frames {
            let t = n as f32 / sample_rate as f32;
            let sample = (2.0 * PI * frequency_hz * t).sin();
            samples.push(sample);
        }

        Self::new(1, sample_rate, samples)
    }

    /// Generates a discrete unit impulse audio clip.
    #[must_use]
    pub fn impulse(sample_rate: u32) -> Self {
        let total_frames = 256;
        let mut samples = vec![0.0f32; total_frames];
        samples[0] = 1.0;
        Self::new(1, sample_rate, samples)
    }

    /// Generates a white noise audio clip.
    #[must_use]
    pub fn noise(duration_sec: f32, sample_rate: u32) -> Self {
        let total_frames = ((duration_sec * sample_rate as f32).round() as usize).max(1);
        let mut samples = Vec::with_capacity(total_frames);
        let mut seed = 0x1234_5678u32;

        for _ in 0..total_frames {
            // Simple Xorshift RNG
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let norm = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
            samples.push(norm * 0.5);
        }

        Self::new(1, sample_rate, samples)
    }

    /// Encodes this audio clip into 16-bit PCM RIFF/WAVE file bytes.
    #[must_use]
    pub fn to_wav_bytes(&self) -> Vec<u8> {
        let num_channels = self.channels;
        let sample_rate = self.sample_rate;
        let bits_per_sample: u16 = 16;
        let block_align = num_channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * block_align as u32;
        let data_byte_count = (self.samples.len() * 2) as u32;
        let total_file_size = 36 + data_byte_count;

        let mut bytes = Vec::with_capacity((44 + data_byte_count) as usize);

        // RIFF Header
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&total_file_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        // fmt chunk
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size (16 for PCM)
        bytes.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (1 for PCM)
        bytes.extend_from_slice(&num_channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());

        // data chunk
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_byte_count.to_le_bytes());

        for &sample in &self.samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let pcm_16 = (clamped * 32767.0).round() as i16;
            bytes.extend_from_slice(&pcm_16.to_le_bytes());
        }

        bytes
    }

    /// Decodes an audio clip from RIFF/WAVE file bytes: integer PCM of 8, 16, 24 or 32 bits and
    /// IEEE float of 32 or 64 bits, plain or `WAVE_FORMAT_EXTENSIBLE` (FIX-7.4, AUD-09). Chunks
    /// of odd size are followed by a pad byte; a `data` chunk cut short by the end of the file
    /// decodes the whole frames it holds.
    ///
    /// # Errors
    /// [`AudioError::AssetError`] if the bytes are not a RIFF/WAVE file, a chunk before `data`
    /// runs past the end, `fmt ` is missing or comes after `data`, there are no channels, or the
    /// format is a compressed one.
    pub fn from_wav_bytes(bytes: &[u8]) -> AudioResult<Self> {
        let error = |message: &str| AudioError::AssetError(message.to_string());
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(error("not a RIFF/WAVE file"));
        }
        let mut format = None;
        let mut cursor = 12;
        while cursor + 8 <= bytes.len() {
            let id = &bytes[cursor..cursor + 4];
            let size = read_u32(bytes, cursor + 4) as usize;
            let start = cursor + 8;
            if id == b"data" {
                let format: WavFormat =
                    format.ok_or_else(|| error("the data chunk comes before fmt"))?;
                let end = start.saturating_add(size).min(bytes.len());
                return Ok(format.decode(&bytes[start..end]));
            }
            let end = start
                .checked_add(size)
                .filter(|&end| end <= bytes.len())
                .ok_or_else(|| error("a chunk runs past the end of the file"))?;
            if id == b"fmt " {
                format = Some(WavFormat::parse(&bytes[start..end])?);
            }
            // RIFF pads a chunk of odd size to an even boundary.
            cursor = end + (size & 1);
        }
        Err(error("no data chunk"))
    }
}

/// A WAV file's sample format.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Encoding {
    /// Unsigned 8-bit integers centred on 128.
    Unsigned8,
    /// Signed little-endian integers of this many bytes (2, 3 or 4).
    Signed(usize),
    Float32,
    Float64,
}

impl Encoding {
    fn bytes(self) -> usize {
        match self {
            Encoding::Unsigned8 => 1,
            Encoding::Signed(bytes) => bytes,
            Encoding::Float32 => 4,
            Encoding::Float64 => 8,
        }
    }
}

/// What a `fmt ` chunk says.
#[derive(Copy, Clone, Debug)]
struct WavFormat {
    channels: u16,
    sample_rate: u32,
    encoding: Encoding,
}

impl WavFormat {
    /// Reads a `fmt ` chunk body: `WAVEFORMATEX`, or `WAVE_FORMAT_EXTENSIBLE`, whose format tag is
    /// the first two bytes of its sub-format GUID. Samples come in containers of `bits` bits;
    /// fewer valid bits (24 in 32) are the container's high bits, so they decode as the container.
    fn parse(body: &[u8]) -> AudioResult<Self> {
        let error = |message: String| AudioError::AssetError(message);
        if body.len() < 16 {
            return Err(error("the fmt chunk is too short".into()));
        }
        let mut tag = read_u16(body, 0);
        let channels = read_u16(body, 2);
        let sample_rate = read_u32(body, 4);
        let bits = read_u16(body, 14);
        if tag == 0xFFFE {
            if body.len() < 40 {
                return Err(error(
                    "the WAVE_FORMAT_EXTENSIBLE fmt chunk is too short".into(),
                ));
            }
            tag = read_u16(body, 24);
        }
        if channels == 0 || sample_rate == 0 {
            return Err(error(format!(
                "a WAV file needs channels and a sample rate ({channels} channels, {sample_rate} Hz)"
            )));
        }
        let encoding = match (tag, bits) {
            (1, 8) => Encoding::Unsigned8,
            (1, 16 | 24 | 32) => Encoding::Signed(usize::from(bits / 8)),
            (3, 32) => Encoding::Float32,
            (3, 64) => Encoding::Float64,
            _ => {
                return Err(error(format!(
                    "unsupported WAV format {tag:#06x} with {bits}-bit samples"
                )))
            }
        };
        Ok(Self {
            channels,
            sample_rate,
            encoding,
        })
    }

    /// The whole frames in `data`, as an interleaved clip.
    fn decode(self, data: &[u8]) -> AudioClip {
        let size = self.encoding.bytes();
        let frame = size * usize::from(self.channels);
        let data = &data[..data.len() / frame * frame];
        let samples = data
            .chunks_exact(size)
            .map(|sample| match self.encoding {
                Encoding::Unsigned8 => (f32::from(sample[0]) - 128.0) / 128.0,
                Encoding::Signed(2) => {
                    f32::from(i16::from_le_bytes([sample[0], sample[1]])) / 32_768.0
                }
                Encoding::Signed(3) => {
                    // Sign-extended through the top of an i32.
                    let value = i32::from_le_bytes([0, sample[0], sample[1], sample[2]]) >> 8;
                    value as f32 / 8_388_608.0
                }
                Encoding::Signed(_) => {
                    let value = i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
                    (f64::from(value) / 2_147_483_648.0) as f32
                }
                Encoding::Float32 => {
                    f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
                }
                Encoding::Float64 => {
                    let mut bytes = [0; 8];
                    bytes.copy_from_slice(sample);
                    f64::from_le_bytes(bytes) as f32
                }
            })
            .collect();
        AudioClip::new(self.channels, self.sample_rate, samples)
    }
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}
