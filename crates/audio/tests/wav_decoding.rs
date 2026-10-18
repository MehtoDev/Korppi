//! WAV files as encoders and audio tools write them (FIX-7.4; finding AUD-09, probe A7).
//!
//! Before FIX-7.4 the decoder skipped odd-sized chunks without their pad byte (a `LIST` chunk of
//! odd size, which many encoders write, made it miss the `data` chunk) and refused 24-bit PCM and
//! `WAVE_FORMAT_EXTENSIBLE`, the format DAWs export 24-bit and multichannel audio in. Each file
//! here is built byte by byte and decoded through `WavAudioLoader`.

use korppi_asset::AssetLoader;
use korppi_audio::{AudioClip, WavAudioLoader};

/// A chunk: its ID, size and data, and the pad byte RIFF adds after an odd size.
fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut bytes = id.to_vec();
    bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
    bytes.extend_from_slice(data);
    if data.len() % 2 == 1 {
        bytes.push(0);
    }
    bytes
}

/// A plain `fmt ` chunk body (`WAVEFORMAT` with bits per sample).
fn fmt(format: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
    let block = channels * bits / 8;
    let mut body = Vec::new();
    body.extend_from_slice(&format.to_le_bytes());
    body.extend_from_slice(&channels.to_le_bytes());
    body.extend_from_slice(&rate.to_le_bytes());
    body.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    body.extend_from_slice(&block.to_le_bytes());
    body.extend_from_slice(&bits.to_le_bytes());
    body
}

/// A `WAVE_FORMAT_EXTENSIBLE` `fmt ` chunk body whose sub-format is `sub_format` (1 PCM, 3
/// float), with `valid` valid bits in `bits`-bit containers.
fn fmt_extensible(sub_format: u16, channels: u16, rate: u32, bits: u16, valid: u16) -> Vec<u8> {
    let mut body = fmt(0xFFFE, channels, rate, bits);
    body.extend_from_slice(&22u16.to_le_bytes()); // cbSize
    body.extend_from_slice(&valid.to_le_bytes());
    body.extend_from_slice(&3u32.to_le_bytes()); // front left and right
                                                 // The sub-format GUID: the format tag, then the fixed KSDATAFORMAT suffix.
    body.extend_from_slice(&sub_format.to_le_bytes());
    body.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
    ]);
    body
}

/// A RIFF/WAVE file of `chunks`.
fn wave(chunks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = chunks.concat();
    let mut bytes = b"RIFF".to_vec();
    bytes.extend_from_slice(&(4 + body.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(&body);
    bytes
}

fn decode(bytes: &[u8]) -> Result<AudioClip, String> {
    WavAudioLoader
        .load(bytes, "test.wav")
        .map_err(|error| error.to_string())
}

/// 100 frames of a mono ramp from -1 towards 1.
fn ramp() -> Vec<f32> {
    (0..100).map(|i| -1.0 + i as f32 / 50.0).collect()
}

fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&s| ((s * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes())
        .collect()
}

fn pcm24(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&s| {
            let v = (s * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
            let b = v.to_le_bytes();
            [b[0], b[1], b[2]]
        })
        .collect()
}

fn assert_close(clip: &AudioClip, expected: &[f32], tolerance: f32) {
    assert_eq!(clip.samples.len(), expected.len());
    for (i, (&got, &want)) in clip.samples.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "sample {i}: {got}, not {want}"
        );
    }
}

/// An odd-sized chunk before `data` is followed by a pad byte, which the decoder skips.
#[test]
fn odd_sized_chunks_are_padded() {
    let bytes = wave(&[
        chunk(b"fmt ", &fmt(1, 1, 48_000, 16)),
        chunk(b"LIST", b"INFOISFT\x07\x00\x00\x00Korppi\x00"), // 21 bytes: odd
        chunk(b"data", &pcm16(&ramp())),
    ]);
    let clip = decode(&bytes).expect("decodes past the padded LIST chunk");
    assert_eq!((clip.channels, clip.sample_rate), (1, 48_000));
    assert_close(&clip, &ramp(), 1.0 / 32768.0);
}

#[test]
fn twenty_four_bit_pcm_decodes() {
    let bytes = wave(&[
        chunk(b"fmt ", &fmt(1, 1, 44_100, 24)),
        chunk(b"data", &pcm24(&ramp())),
    ]);
    let clip = decode(&bytes).expect("24-bit PCM");
    assert_eq!(clip.sample_rate, 44_100);
    assert_close(&clip, &ramp(), 1.0 / 8_388_608.0);
}

#[test]
fn thirty_two_bit_formats_decode() {
    let ints: Vec<u8> = ramp()
        .iter()
        .flat_map(|&s| ((f64::from(s) * 2_147_483_648.0) as i32).to_le_bytes())
        .collect();
    let clip = decode(&wave(&[
        chunk(b"fmt ", &fmt(1, 1, 48_000, 32)),
        chunk(b"data", &ints),
    ]))
    .expect("32-bit integer PCM");
    assert_close(&clip, &ramp(), 1e-6);

    let doubles: Vec<u8> = ramp()
        .iter()
        .flat_map(|&s| f64::from(s).to_le_bytes())
        .collect();
    let clip = decode(&wave(&[
        chunk(b"fmt ", &fmt(3, 1, 48_000, 64)),
        chunk(b"data", &doubles),
    ]))
    .expect("64-bit float");
    assert_close(&clip, &ramp(), 1e-7);
}

/// `WAVE_FORMAT_EXTENSIBLE` takes its format from the sub-format GUID: PCM at 16 and 24 bits
/// (stereo, as DAWs export it), and float.
#[test]
fn wave_format_extensible_decodes() {
    let stereo: Vec<f32> = ramp().iter().flat_map(|&s| [s, -s]).collect();
    let clip = decode(&wave(&[
        chunk(b"fmt ", &fmt_extensible(1, 2, 48_000, 16, 16)),
        chunk(b"data", &pcm16(&stereo)),
    ]))
    .expect("extensible 16-bit PCM");
    assert_eq!(clip.channels, 2);
    assert_close(&clip, &stereo, 1.0 / 32768.0);

    let clip = decode(&wave(&[
        chunk(b"fmt ", &fmt_extensible(1, 2, 96_000, 24, 24)),
        chunk(b"data", &pcm24(&stereo)),
    ]))
    .expect("extensible 24-bit PCM");
    assert_eq!(clip.sample_rate, 96_000);
    assert_close(&clip, &stereo, 1.0 / 8_388_608.0);

    let floats: Vec<u8> = stereo.iter().flat_map(|s| s.to_le_bytes()).collect();
    let clip = decode(&wave(&[
        chunk(b"fmt ", &fmt_extensible(3, 2, 48_000, 32, 32)),
        chunk(b"data", &floats),
    ]))
    .expect("extensible float");
    assert_close(&clip, &stereo, 0.0);
}

/// Malformed files are errors, never panics: every truncation of a valid file, a `data` chunk
/// before `fmt `, no channels, a compressed format, and chunk sizes past the end.
#[test]
fn malformed_files_are_errors() {
    let valid = wave(&[
        chunk(b"fmt ", &fmt(1, 2, 48_000, 16)),
        chunk(b"LIST", b"odd"),
        chunk(b"data", &pcm16(&ramp())),
    ]);
    for end in 0..valid.len() {
        // Every truncation decodes or fails; none panics.
        let _ = decode(&valid[..end]);
    }
    assert!(
        decode(&wave(&[chunk(b"data", &pcm16(&ramp()))])).is_err(),
        "no fmt chunk"
    );
    assert!(
        decode(&wave(&[
            chunk(b"fmt ", &fmt(1, 0, 48_000, 16)),
            chunk(b"data", &pcm16(&ramp())),
        ]))
        .is_err(),
        "no channels"
    );
    assert!(
        decode(&wave(&[
            chunk(b"fmt ", &fmt(2, 1, 48_000, 4)),
            chunk(b"data", &[0; 64]),
        ]))
        .is_err(),
        "ADPCM is not decoded"
    );
    let mut huge = wave(&[chunk(b"fmt ", &fmt(1, 1, 48_000, 16))]);
    huge.extend_from_slice(b"JUNK\xff\xff\xff\xff");
    assert!(decode(&huge).is_err(), "a chunk past the end");
}

/// The clips the engine writes still decode exactly as before.
#[test]
fn the_engines_own_files_round_trip() {
    let clip = AudioClip::new(2, 48_000, vec![0.5, -0.25, 0.0, 0.75]);
    let decoded = decode(&clip.to_wav_bytes()).unwrap();
    assert_eq!((decoded.channels, decoded.sample_rate), (2, 48_000));
    assert_close(&decoded, &clip.samples, 1.0 / 32767.0);
}
