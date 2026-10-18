//! The gameplay API: `AudioPlayer` components played through the mixer (FIX-7.2, ADR-050;
//! finding AUD-07).
//!
//! Before FIX-7.2 an entity had one voice and no play call; a finished clip kept reading as
//! playing (probe A9); `max_voices`, `bus_name`, `play_on_spawn` and the listener's volume had no
//! effect (probe A10); and clips had an ID space of their own beside the asset server's. Here every
//! player names clips loaded by the asset server, starts and stops on calls, reports its state
//! from the mixer, and is spatialized from its entity's `GlobalTransform`, also as a child entity.

mod common;

use common::{channel, constant, Game};
use korppi_audio::player::{AudioPlayer, AudioSource, PlayerState};
use korppi_audio::{AudioClip, AudioListener, AudioSettings};
use korppi_ecs::ChildOf;
use korppi_math::Vec3;
use korppi_scene::LocalTransform;

const RATE: u32 = common::RATE;

#[test]
fn a_one_shot_plays_once_finishes_and_plays_again() {
    let mut game = Game::new();
    game.clip("shot", constant(0.5, 1000));
    let shot = game.start().remove(0);
    let player = game
        .world
        .spawn((AudioPlayer::new(AudioSource::clip(shot)).autoplay(),));
    let out = channel(&game.ticks(2), 0);
    assert!(out[..1000].iter().all(|&s| s == 0.5), "the clip, at once");
    assert!(out[1000..].iter().all(|&s| s == 0.0), "then silence");

    game.tick();
    assert_eq!(game.player(player).state(), PlayerState::Finished);
    assert!(
        !game.player(player).is_playing(),
        "a finished clip no longer reads as playing (probe A9)"
    );

    game.player(player).play();
    let out = channel(&game.tick(), 0);
    assert!(out.iter().all(|&s| s == 0.5), "played again");
    assert_eq!(game.player(player).state(), PlayerState::Playing);
}

#[test]
fn instances_overlap_up_to_the_limit() {
    let mut game = Game::new();
    game.clip("hum", constant(0.25, 48_000));
    let hum = game.start().remove(0);
    let player = game
        .world
        .spawn((AudioPlayer::new(AudioSource::clip(hum)).max_instances(2),));
    let mut levels = Vec::new();
    for _ in 0..3 {
        game.player(player).play();
        let out = channel(&game.tick(), 0);
        levels.push(out[out.len() - 1]);
    }
    assert_eq!(
        levels,
        [0.25, 0.5, 0.5],
        "the third play replaces the oldest instance"
    );
}

#[test]
fn an_intro_plays_once_then_its_loop_repeats() {
    let mut game = Game::new();
    let samples: Vec<f32> = [0.25, 0.5, 0.75]
        .into_iter()
        .flat_map(|v| std::iter::repeat_n(v, 100))
        .collect();
    game.clip("music", AudioClip::new(1, RATE, samples));
    let music = game.start().remove(0);
    game.world.spawn((AudioPlayer::new(
        AudioSource::clip(music)
            .looping()
            .loop_region(100, Some(200)),
    )
    .autoplay(),));
    let out = channel(&game.tick(), 0);
    assert!(out[..100].iter().all(|&s| s == 0.25), "the intro");
    assert!(out[100..].iter().all(|&s| s == 0.5), "the loop region");
}

/// A playlist starts its next item exactly when the current one has `crossfade` left, and the
/// equal-power crossfade keeps the power constant.
#[test]
fn a_playlist_crossfades_on_the_sample() {
    let mut game = Game::new();
    let left: Vec<f32> = (0..4800).flat_map(|_| [0.5, 0.0]).collect();
    let right: Vec<f32> = (0..4800).flat_map(|_| [0.0, 0.5]).collect();
    game.clip("a", AudioClip::new(2, RATE, left));
    game.clip("b", AudioClip::new(2, RATE, right));
    let items = game.start();
    let player = game
        .world
        .spawn((AudioPlayer::new(AudioSource::playlist(items, 0.02)).autoplay(),));
    let out = game.ticks(12);
    let (l, r) = (channel(&out, 0), channel(&out, 1));
    // 0.02 s is 960 frames: the crossfade runs from frame 3840 to 4800.
    assert!(l[..3840].iter().all(|&s| s == 0.5) && r[..3840].iter().all(|&s| s == 0.0));
    for frame in 3840..4800 {
        let power = l[frame] * l[frame] + r[frame] * r[frame];
        assert!(
            (power - 0.25).abs() < 1e-4,
            "frame {frame}: power {power} ({}, {})",
            l[frame],
            r[frame]
        );
    }
    assert!(
        r[3840] < 1e-3 && l[3840] > 0.499,
        "the crossfade starts at frame 3840"
    );
    assert!(l[4800..8640].iter().all(|&s| s == 0.0));
    assert!(r[4800..8640].iter().all(|&s| s == 0.5), "the second item");
    assert!(
        r[8640..].iter().all(|&s| s == 0.0),
        "and the end of the list"
    );
    assert_eq!(game.player(player).state(), PlayerState::Finished);
}

/// Stems start on the same frame and stay in sync; each has its own volume.
#[test]
fn stems_play_in_sync_with_their_own_volumes() {
    let mut game = Game::new();
    let ramp = |left: bool| -> Vec<f32> {
        (0..48_000)
            .flat_map(|i| {
                let v = (i % 1000) as f32 / 2000.0;
                if left {
                    [v, 0.0]
                } else {
                    [0.0, v]
                }
            })
            .collect()
    };
    game.clip("drums", AudioClip::new(2, RATE, ramp(true)));
    game.clip("bass", AudioClip::new(2, RATE, ramp(false)));
    let stems = game.start();
    let player = game
        .world
        .spawn((AudioPlayer::new(AudioSource::stems(stems)).autoplay(),));
    let out = game.ticks(2);
    assert_eq!(channel(&out, 0), channel(&out, 1), "the stems are in sync");

    game.player(player).set_stem_volume(1, 0.0, 0.0);
    let out = game.ticks(2);
    let (l, r) = (channel(&out, 0), channel(&out, 1));
    assert!(r[480..].iter().all(|&s| s == 0.0), "the bass is muted");
    assert!(l.iter().any(|&s| s > 0.4), "the drums play on");
}

#[test]
fn stop_pause_and_volume_follow_their_ramps() {
    let mut game = Game::new();
    game.clip("hum", constant(0.5, 48_000));
    let hum = game.start().remove(0);
    let player = game
        .world
        .spawn((AudioPlayer::new(AudioSource::clip(hum).looping()).autoplay(),));
    game.tick();

    game.player(player).set_volume(0.5, 0.01);
    let out = channel(&game.tick(), 0);
    assert!((out[0] - 0.5).abs() < 1e-3 && (out[480] - 0.25).abs() < 1e-6);
    assert!(
        (out[240] - 0.375).abs() < 1e-3,
        "halfway through the 10 ms ramp"
    );

    game.player(player).pause();
    let out = channel(&game.tick(), 0);
    assert!(out[240..].iter().all(|&s| s == 0.0));
    game.tick();
    assert_eq!(game.player(player).state(), PlayerState::Paused);
    game.player(player).resume();
    let out = channel(&game.tick(), 0);
    assert!(out[240..].iter().all(|&s| (s - 0.25).abs() < 1e-6));
    assert_eq!(game.player(player).state(), PlayerState::Playing);

    game.player(player).stop(0.01);
    let out = channel(&game.tick(), 0);
    assert!(
        (out[240] - 0.125).abs() < 1e-3,
        "halfway through the 10 ms fade"
    );
    assert!(out[480..].iter().all(|&s| s == 0.0));
    game.tick();
    assert_eq!(game.player(player).state(), PlayerState::Stopped);
}

#[test]
fn a_finished_one_shot_can_despawn_its_entity() {
    let mut game = Game::new();
    game.clip("shot", constant(0.5, 1000));
    let shot = game.start().remove(0);
    let player = game.world.spawn((AudioPlayer::new(AudioSource::clip(shot))
        .autoplay()
        .despawn_on_finish(),));
    game.ticks(3);
    assert!(
        !game.world.is_alive(player),
        "despawned when the clip ended"
    );
}

/// A player on an entity with a transform is spatialized against the listener: a child of an
/// entity 10 m to the listener's right is quieter than a 2D player and louder on the right from
/// its first frame, and follows its parent when the parent moves to the left.
#[test]
fn players_on_child_entities_are_spatialized() {
    let mut flat = Game::new();
    flat.clip("hum", constant(0.5, 48_000));
    let hum = flat.start().remove(0);
    flat.world
        .spawn((AudioListener::default(), LocalTransform::default()));
    flat.world
        .spawn((AudioPlayer::new(AudioSource::clip(hum).looping()).autoplay(),));
    let out = flat.tick();
    assert!(
        out.iter().all(|&s| s == 0.5),
        "a 2D player is not spatialized"
    );

    let mut game = Game::new();
    game.clip("hum", constant(0.5, 48_000));
    let hum = game.start().remove(0);
    game.world
        .spawn((AudioListener::default(), LocalTransform::default()));
    let parent = game.world.spawn((LocalTransform {
        translation: Vec3::new(10.0, 0.0, 0.0),
        ..LocalTransform::default()
    },));
    let child = game.world.spawn((
        AudioPlayer::new(AudioSource::clip(hum).looping()).autoplay(),
        LocalTransform::default(),
    ));
    game.world.add_pair::<ChildOf>(child, parent);
    let out = game.tick();
    let (left, right) = (out[0], out[1]);
    assert!(
        right > left,
        "louder on the right from the first frame: {left} {right}"
    );
    assert!(
        left * left + right * right < 0.5 * 0.5,
        "attenuated over 10 m: {left} {right}"
    );
    let (end_left, end_right) = (out[out.len() - 2], out[out.len() - 1]);
    assert!(
        (end_left - left).abs() < 1e-6 && (end_right - right).abs() < 1e-6,
        "it started in place, without a ramp"
    );

    game.world
        .get_mut::<LocalTransform>(parent)
        .unwrap()
        .translation = Vec3::new(-10.0, 0.0, 0.0);
    let out = game.ticks(2);
    let (left, right) = (out[out.len() - 2], out[out.len() - 1]);
    assert!(
        left > right,
        "louder on the left once its parent moved: {left} {right}"
    );
}

/// The settings' master volume and the listener's volume scale every player.
#[test]
fn master_and_listener_volume_scale_the_output() {
    let mut game = Game::new();
    game.clip("hum", constant(0.5, 48_000));
    let hum = game.start().remove(0);
    game.world
        .spawn((AudioPlayer::new(AudioSource::clip(hum).looping()).autoplay(),));
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .master_volume = 0.5;
    game.world.spawn((
        AudioListener {
            volume: 0.5,
            ..AudioListener::default()
        },
        LocalTransform::default(),
    ));
    let out = channel(&game.ticks(2), 0);
    assert!(
        (out[out.len() - 1] - 0.125).abs() < 1e-6,
        "0.5 × master 0.5 × listener 0.5: {}",
        out[out.len() - 1]
    );
}
