//! Ogg Vorbis clips (FIX-7.4; finding AUD-09, the owner's decision): decoded by `lewton` through
//! `OggVorbisLoader` into clips the mixer plays. The test files are tones encoded by `oggenc`
//! (`tests/data/README.md`); Vorbis is lossy, so they are checked by frequency and level.

mod common;

use common::{channel, frequency, rms_db, Game};
use korppi_asset::AssetLoader;
use korppi_audio::player::{AudioPlayer, AudioSource};
use korppi_audio::{AudioClip, OggVorbisLoader};

const MONO: &[u8] = include_bytes!("data/tone_440_mono_48k.ogg");
const STEREO: &[u8] = include_bytes!("data/tones_440_660_stereo_44k.ogg");

fn decode(bytes: &[u8]) -> Result<AudioClip, String> {
    OggVorbisLoader
        .load(bytes, "test.ogg")
        .map_err(|error| error.to_string())
}

/// One channel of a clip.
fn clip_channel(clip: &AudioClip, index: usize) -> Vec<f32> {
    clip.samples
        .iter()
        .skip(index)
        .step_by(usize::from(clip.channels))
        .copied()
        .collect()
}

/// The frequency of a tone sampled at `rate`, from its rising zero crossings.
fn tone_hz(samples: &[f32], rate: u32) -> f32 {
    frequency(samples) * rate as f32 / common::RATE as f32
}

#[test]
fn a_mono_file_decodes_to_its_tone() {
    let clip = decode(MONO).expect("decodes");
    assert_eq!((clip.channels, clip.sample_rate), (1, 48_000));
    assert_eq!(clip.frame_count(), 48_000, "one second, as encoded");
    let samples = clip_channel(&clip, 0);
    // The middle of the clip, away from the encoder's edges.
    let middle = &samples[4_800..43_200];
    let hz = tone_hz(middle, 48_000);
    assert!((hz - 440.0).abs() < 0.5, "a 440 Hz tone, not {hz} Hz");
    // In phase with the encoded sine: the encoder's leading frames are dropped, not trailing
    // ones (a shift of them would put the tone out of phase).
    let worst = (4_800..43_200)
        .map(|n| {
            (samples[n] - 0.5 * (std::f32::consts::TAU * 440.0 * n as f32 / 48_000.0).sin()).abs()
        })
        .fold(0.0f32, f32::max);
    assert!(
        worst < 0.02,
        "the decoded tone is off the encoded one by up to {worst}"
    );
    let level = rms_db(middle);
    let expected = 20.0 * (0.5f32 / 2f32.sqrt()).log10();
    assert!(
        (level - expected).abs() < 0.5,
        "{level} dB, not {expected} dB (amplitude 0.5)"
    );
}

#[test]
fn a_stereo_file_keeps_its_channels() {
    let clip = decode(STEREO).expect("decodes");
    assert_eq!((clip.channels, clip.sample_rate), (2, 44_100));
    assert_eq!(clip.frame_count(), 44_100);
    let (left, right) = (clip_channel(&clip, 0), clip_channel(&clip, 1));
    let (left, right) = (&left[4_410..39_690], &right[4_410..39_690]);
    assert!(
        (tone_hz(left, 44_100) - 440.0).abs() < 0.5,
        "left {}",
        tone_hz(left, 44_100)
    );
    assert!(
        (tone_hz(right, 44_100) - 660.0).abs() < 0.5,
        "right {}",
        tone_hz(right, 44_100)
    );
}

/// An Ogg Vorbis file loaded by the asset server (`AudioClipLoader`) plays through a player,
/// resampled from 44.1 kHz to the device's 48 kHz.
#[test]
fn a_vorbis_clip_plays_through_a_player() {
    let mut game = Game::new();
    game.file("music.ogg", STEREO.to_vec());
    let music = game.start().remove(0);
    game.world
        .spawn((AudioPlayer::new(AudioSource::clip(music)).autoplay(),));
    let out = game.ticks(30);
    let (left, right) = (channel(&out, 0), channel(&out, 1));
    let (hz_left, hz_right) = (
        frequency(&left[4_800..19_200]),
        frequency(&right[4_800..19_200]),
    );
    assert!(
        (hz_left - 440.0).abs() < 1.0,
        "left plays 440 Hz, not {hz_left}"
    );
    assert!(
        (hz_right - 660.0).abs() < 1.0,
        "right plays 660 Hz, not {hz_right}"
    );
}

/// The loader takes `.ogg` and `.oga` files.
#[test]
fn the_loader_names_its_extensions() {
    assert!(OggVorbisLoader.extensions().contains(&"ogg"));
    assert!(OggVorbisLoader.extensions().contains(&"oga"));
}

/// Malformed input is an error, never a panic: every truncation of a valid file, its bytes with
/// a flipped bit at each of many positions, a WAV file and noise.
#[test]
fn malformed_files_are_errors() {
    for end in (0..MONO.len()).step_by(37) {
        let _ = decode(&MONO[..end]);
    }
    for at in (0..MONO.len()).step_by(53) {
        let mut corrupt = MONO.to_vec();
        corrupt[at] ^= 0x10;
        let _ = decode(&corrupt);
    }
    assert!(decode(&AudioClip::sine(440.0, 0.1, 48_000).to_wav_bytes()).is_err());
    let noise: Vec<u8> = (0..4096u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    assert!(decode(&noise).is_err());
    assert!(decode(&[]).is_err());
}
