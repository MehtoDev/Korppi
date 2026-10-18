//! Verification test for voice playback continuity across game ticks (AUDIT-5.4 / Finding 4.4).
//!
//! A playing voice must keep its place in its clip from one game tick to the next instead of
//! restarting. Moved to the player path in FIX-7.2 (the owner's decision): before, it checked
//! `AudioCoordinator::update_ecs` with `AudioEmitter`; now an `AudioPlayer` plays through the
//! mixer, and the output over many ticks must be the clip itself, sample for sample, then silence.

mod common;

use common::{channel, Game, RATE, TICK};
use korppi_audio::player::{AudioPlayer, AudioSource, PlayerState};
use korppi_audio::AudioClip;

#[test]
fn test_voice_playback_continues_across_ticks() {
    let mut game = Game::new();
    // A 1-second sine wave clip, as the asset server decodes it.
    let clip = AudioClip::sine(440.0, 1.0, RATE);
    let decoded = AudioClip::from_wav_bytes(&clip.to_wav_bytes()).unwrap();
    game.clip("tone", clip);
    let tone = game.start().remove(0);
    let player = game
        .world
        .spawn((AudioPlayer::new(AudioSource::clip(tone)).autoplay(),));

    // 61 ticks of 800 frames: the whole clip, then 800 frames past its end.
    let out = channel(&game.ticks(61), 0);
    assert_eq!(out.len(), 61 * TICK);
    let restarts = (TICK..48_000)
        .step_by(TICK)
        .filter(|&tick| out[tick] != decoded.samples[tick])
        .count();
    assert_eq!(
        restarts, 0,
        "a tick restarted the voice instead of continuing it"
    );
    assert_eq!(
        &out[..48_000],
        &decoded.samples[..],
        "the ticks render the clip's consecutive samples"
    );

    // A non-looping clip ends in silence, and the player reports it finished.
    assert!(out[48_000..].iter().all(|&s| s == 0.0));
    game.tick();
    assert_eq!(game.player(player).state(), PlayerState::Finished);
}
