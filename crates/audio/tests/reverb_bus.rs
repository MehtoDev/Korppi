//! Reverb on a send bus of the mixer (FIX-7.5, ADR-051; findings AUD-08 and AUD-10).
//!
//! Voices send to a bus whose effect is a reverb: the feedback delay network in raytraced mode,
//! with a decay time per band, or Freeverb's room in standard mode. Before FIX-7.5 the FDN took
//! its loop gains from the mid band alone and Freeverb was reached only by its own tests. Decay
//! times are measured here as acousticians do: each band of the impulse response is filtered
//! out, Schroeder-integrated, and its slope from -5 to -25 dB extrapolated to 60 dB (T20).

mod common;

use common::{channel, Game, RATE};
use korppi_audio::mixer::{
    AudioMixer, BusEffect, ClipData, MixerConfig, MixerControl, RoomReverb, VoiceParams, MASTER,
    REVERB_BANDS,
};
use korppi_audio::player::{AudioPlayer, AudioSource};
use korppi_audio::{
    AudioClip, AudioDevice, AudioListener, AudioSettings, AudioStream, AudioStreamConfig,
    BiquadFilter, BiquadType, BufferSize, ChannelCount, HeadlessDummyDevice,
};
use korppi_math::Vec3;
use korppi_scene::GlobalTransform;
use std::sync::Arc;

fn open() -> (MixerControl, Box<dyn AudioStream>) {
    let config = AudioStreamConfig {
        sample_rate: RATE,
        channels: ChannelCount::Stereo,
        buffer_size: BufferSize::Fixed(256),
    };
    let (control, mixer) = AudioMixer::new(MixerConfig::for_stream(&config));
    let mut stream = HeadlessDummyDevice::new()
        .create_output_stream(config, Box::new(mixer))
        .expect("stream");
    stream.play().expect("play");
    (control, stream)
}

/// The left channel of `effect`'s response to an impulse sent at `send`, rendered for
/// `seconds`. The voice plays on a muted bus, so only its send is heard.
fn impulse_response(effect: BusEffect, send: f32, seconds: f32) -> Vec<f32> {
    let (mut control, mut stream) = open();
    let dry = control.add_bus("Dry", MASTER).unwrap();
    control.set_bus_muted(dry, true).unwrap();
    let reverb = control.add_bus("Reverb", MASTER).unwrap();
    control.set_bus_effect(reverb, Some(effect)).unwrap();
    let mut samples = vec![0.0; 64];
    samples[0] = 1.0;
    let impulse = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(1, RATE, samples))))
        .unwrap();
    let voice = control
        .start(
            impulse,
            VoiceParams {
                bus: dry,
                ..VoiceParams::default()
            },
        )
        .unwrap();
    control.set_send(voice, reverb, send, 0).unwrap();
    let frames = (seconds * RATE as f32) as usize;
    channel(&stream.render_frames(frames), 0)
}

/// The decay time of `response` in the octave band around `hz` (T20 extrapolated to 60 dB).
fn decay_time(response: &[f32], hz: f32) -> f32 {
    let mut band = BiquadFilter::new(BiquadType::BandPass, hz, 1.41, RATE as f32);
    let filtered: Vec<f32> = response.iter().map(|&s| band.process_sample(s)).collect();
    // Schroeder's backward integration of the energy.
    let mut energy: Vec<f64> = filtered.iter().map(|&s| f64::from(s * s)).collect();
    for i in (0..energy.len() - 1).rev() {
        energy[i] += energy[i + 1];
    }
    let total = energy[0];
    let level = |i: usize| 10.0 * (energy[i] / total).log10();
    let at = |db: f64| {
        (0..energy.len())
            .find(|&i| level(i) <= db)
            .expect("decays that far")
    };
    let (start, end) = (at(-5.0), at(-25.0));
    let slope = (level(end) - level(start)) / ((end - start) as f64 / f64::from(RATE));
    (-60.0 / slope) as f32
}

/// The late reverb decays at the RT60 it is given in each band.
#[test]
fn the_late_reverb_decays_at_its_rt60_per_band() {
    let rt60 = [1.2, 0.9, 0.5];
    let response = impulse_response(BusEffect::late(rt60, RATE), 1.0, 2.5);
    for (band, (&hz, &target)) in REVERB_BANDS.iter().zip(&rt60).enumerate() {
        let measured = decay_time(&response, hz);
        eprintln!("band {band} ({hz} Hz): RT60 {measured:.3} s, set {target} s");
        assert!(
            (measured / target - 1.0).abs() < 0.15,
            "band {band} ({hz} Hz) decays in {measured} s, not {target} s"
        );
    }
}

/// The room reverb's decay grows with its size.
#[test]
fn the_room_reverb_grows_with_its_size() {
    let decay = |size: f32| {
        let room = RoomReverb { size, damping: 0.3 };
        decay_time(
            &impulse_response(BusEffect::room(room, RATE), 1.0, 4.0),
            1000.0,
        )
    };
    let (small, large) = (decay(0.3), decay(0.85));
    eprintln!("room 0.3: {small:.3} s; room 0.85: {large:.3} s");
    assert!(small > 0.1, "a small room still reverberates: {small} s");
    assert!(
        large > small * 1.3,
        "a larger room decays longer: {small} s, {large} s"
    );
}

/// The send level scales the reverb: half the send is a quarter of the energy (-6 dB), and no
/// send is silence.
#[test]
fn the_send_level_scales_the_reverb() {
    let energy = |send: f32| -> f64 {
        impulse_response(BusEffect::late([1.0; 3], RATE), send, 0.5)
            .iter()
            .map(|&s| f64::from(s * s))
            .sum()
    };
    let (full, half, none) = (energy(1.0), energy(0.5), energy(0.0));
    let ratio = 10.0 * (half / full).log10();
    assert!((ratio + 6.02).abs() < 0.1, "half the send: {ratio} dB");
    assert_eq!(none, 0.0, "no send, no reverb");
}

/// A voice that does not send is untouched by a reverb bus, sample for sample.
#[test]
fn a_voice_without_a_send_is_untouched() {
    let (mut control, mut stream) = open();
    let reverb = control.add_bus("Reverb", MASTER).unwrap();
    control
        .set_bus_effect(reverb, Some(BusEffect::late([1.0; 3], RATE)))
        .unwrap();
    let samples: Vec<f32> = (0..4800).map(|n| ((n % 97) as f32 / 97.0) - 0.5).collect();
    let clip = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            RATE,
            samples.clone(),
        ))))
        .unwrap();
    control.start(clip, VoiceParams::default()).unwrap();
    let out = channel(&stream.render_frames(4800), 0);
    assert_eq!(out, samples);
}

/// A spatialized player sends to the room reverb the settings describe: after a short clip ends,
/// its tail goes on, and with the room's send turned off there is none.
#[test]
fn spatialized_players_send_to_the_room_reverb() {
    let tail = |wet: f32| -> f32 {
        let mut game = Game::new();
        game.world
            .resource_mut::<AudioSettings>()
            .unwrap()
            .room_reverb
            .wet = wet;
        game.clip("knock", AudioClip::sine(500.0, 0.05, RATE));
        let knock = game.start().remove(0);
        game.world.spawn((
            AudioListener::default(),
            GlobalTransform::from_translation(Vec3::ZERO),
        ));
        game.world.spawn((
            AudioPlayer::new(AudioSource::clip(knock)).autoplay(),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -3.0)),
        ));
        // The clip lasts 2400 frames; 0.2 s to 0.4 s later, only a reverb can still sound.
        let out = channel(&game.ticks(24), 0);
        out[12_000..19_200]
            .iter()
            .map(|s| s.abs())
            .fold(0.0, f32::max)
    };
    let (with, without) = (tail(1.0), tail(0.0));
    assert!(with > 1e-3, "a tail after the clip: peak {with}");
    assert_eq!(without, 0.0, "no room send, no tail");
}

/// In raytraced mode a player's send follows its distance tick by tick: an emitter that moves
/// from 2 m to 14 m while it sounds leaves a louder tail than one that stays (the send grows
/// from about 0.3 to 0.9; it is taken before distance attenuation, as a diffuse field's level).
#[test]
fn raytraced_sends_follow_the_distance() {
    let tail = |moves: bool| -> f32 {
        let mut game = Game::new();
        game.world
            .resource_mut::<AudioSettings>()
            .unwrap()
            .spatializer_mode = korppi_audio::spatial::SpatializerMode::Raytraced;
        game.clip("hum", AudioClip::sine(500.0, 1.0, RATE));
        let hum = game.start().remove(0);
        game.world.spawn((
            AudioListener::default(),
            GlobalTransform::from_translation(Vec3::ZERO),
        ));
        let emitter = game.world.spawn((
            AudioPlayer::new(AudioSource::clip(hum).looping()).autoplay(),
            GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -2.0)),
        ));
        game.ticks(3);
        if moves {
            *game.world.get_mut::<GlobalTransform>(emitter).unwrap() =
                GlobalTransform::from_translation(Vec3::new(0.0, 0.0, -14.0));
        }
        game.ticks(6);
        game.world
            .get_mut::<AudioPlayer>(emitter)
            .unwrap()
            .stop(0.0);
        game.ticks(2);
        let out = channel(&game.ticks(6), 0);
        out.iter().map(|s| s.abs()).fold(0.0, f32::max)
    };
    let (stays, moves) = (tail(false), tail(true));
    eprintln!("tail peak: {stays} staying at 2 m, {moves} moved to 14 m");
    assert!(
        moves > stays * 2.0,
        "the tail follows the send: {stays}, {moves}"
    );
}
