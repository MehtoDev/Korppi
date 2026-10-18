//! Spatial correctness through the players (FIX-7.3; findings AUD-04 and AUD-05).
//!
//! Before FIX-7.3 the spatializers took +Z as forward, against the coordinate contract's −Z, so
//! front and back were swapped and emitter cones were inverted (probe A4), and the interaural time
//! differences they computed never reached the mixer, nor did any head shadow (probe A5). Here a
//! listener faces −Z, or a direction it is turned to, and what reaches each channel is measured:
//! which side is louder, when each channel starts, and how much of the highs the far ear loses.

mod common;

use common::{channel, rms_db, Game, RATE, TICK};
use korppi_audio::player::{AudioPlayer, AudioSource};
use korppi_audio::spatial::EmitterCone;
use korppi_audio::{AudioClip, AudioListener};
use korppi_math::{Quat, Vec3};
use korppi_scene::LocalTransform;
use std::f32::consts::{FRAC_PI_2, PI};

/// Woodworth's ITD for a source at `azimuth` from the front, with the spatializer's head.
fn woodworth_itd(azimuth: f32) -> f32 {
    0.0875 / 343.0 * (azimuth + azimuth.sin())
}

/// A game with a listener at the origin turned by `listener_yaw` about +Y, and `clip` looping on
/// an entity at `at` turned by `emitter_yaw`, with `cone` if any. Returns the stereo output of
/// `ticks` ticks.
fn play(
    clip: AudioClip,
    listener_yaw: f32,
    at: Vec3,
    emitter_yaw: f32,
    cone: Option<EmitterCone>,
    ticks: usize,
) -> Vec<f32> {
    let mut game = Game::new();
    game.clip("clip", clip);
    let handle = game.start().remove(0);
    game.world.spawn((
        AudioListener::default(),
        LocalTransform {
            rotation: Quat::from_rotation_y(listener_yaw),
            ..LocalTransform::default()
        },
    ));
    let mut player = AudioPlayer::new(AudioSource::clip(handle).looping()).autoplay();
    player.spatial.cone = cone;
    game.world.spawn((
        player,
        LocalTransform {
            translation: at,
            rotation: Quat::from_rotation_y(emitter_yaw),
            ..LocalTransform::default()
        },
    ));
    game.ticks(ticks)
}

fn tone(hz: f32) -> AudioClip {
    AudioClip::sine(hz, 1.0, RATE)
}

/// The levels of the left and right channels over the last half of `out`, in dB.
fn levels(out: &[f32]) -> (f32, f32) {
    let (left, right) = (channel(out, 0), channel(out, 1));
    let half = left.len() / 2;
    (rms_db(&left[half..]), rms_db(&right[half..]))
}

/// A listener facing −Z (no rotation) hears a source at +X on its right; turned to face +X, it
/// hears a source at +Z on its right and one at +X in front of it, centred.
#[test]
fn right_is_forward_cross_up() {
    let (left, right) = levels(&play(
        tone(440.0),
        0.0,
        Vec3::new(5.0, 0.0, 0.0),
        0.0,
        None,
        6,
    ));
    assert!(
        right > left + 6.0,
        "+X is on the right of a listener facing −Z: {left} {right} dB"
    );

    // Turned −90° about +Y, the listener's −Z forward points along +X, and its right along +Z.
    let turned = -FRAC_PI_2;
    let (left, right) = levels(&play(
        tone(440.0),
        turned,
        Vec3::new(0.0, 0.0, 5.0),
        0.0,
        None,
        6,
    ));
    assert!(right > left + 6.0, "+Z is on its right: {left} {right} dB");
    let (left, right) = levels(&play(
        tone(440.0),
        turned,
        Vec3::new(5.0, 0.0, 0.0),
        0.0,
        None,
        6,
    ));
    assert!(
        (right - left).abs() < 0.5,
        "+X is in front of it, centred: {left} {right} dB"
    );
}

/// An emitter's cone faces along its −Z: one turned to face the listener plays at full gain, one
/// facing away at the cone's outer gain (−20 dB). The audit found these inverted.
#[test]
fn emitter_cones_face_their_forward() {
    let cone = EmitterCone {
        inner_angle_rad: PI / 3.0,
        outer_angle_rad: PI / 2.0,
        outer_gain: 0.1,
        outer_lpf_cutoff: 22_000.0,
    };
    // The emitter is 5 m in front of the listener (at −Z). Turned 180°, it faces +Z: the listener.
    let at = Vec3::new(0.0, 0.0, -5.0);
    let facing = levels(&play(tone(440.0), 0.0, at, PI, Some(cone), 6)).0;
    let away = levels(&play(tone(440.0), 0.0, at, 0.0, Some(cone), 6)).0;
    let open = levels(&play(tone(440.0), 0.0, at, 0.0, None, 6)).0;
    assert!(
        (facing - open).abs() < 0.5,
        "facing the listener: full gain ({facing} dB, {open} dB without a cone)"
    );
    assert!(
        (away - (open - 20.0)).abs() < 0.5,
        "facing away: the outer gain, −20 dB ({away} dB)"
    );
}

/// The lag of `late` behind `early` at which their cross-correlation peaks, in frames.
fn lag(early: &[f32], late: &[f32], max: usize) -> usize {
    (0..max)
        .max_by(|&a, &b| {
            let score = |lag: usize| -> f32 {
                early[..early.len() - max]
                    .iter()
                    .zip(&late[lag..])
                    .map(|(e, l)| e * l)
                    .sum()
            };
            score(a).total_cmp(&score(b))
        })
        .unwrap()
}

/// The far ear hears a source later by Woodworth's ITD: at 30° and 45° to the right, the left
/// channel lags the right by 0.261 ms and 0.381 ms (12.5 and 18.3 frames), to within 2 frames
/// (the far ear's head-shadow filter adds about a frame of group delay of its own).
#[test]
fn the_far_ear_hears_the_source_later() {
    // A band-limited click train: a Hann-windowed 2 kHz burst every 2400 frames.
    let burst: Vec<f32> = (0..2400)
        .map(|n| {
            if n < 96 {
                let window = 0.5 - 0.5 * (2.0 * PI * n as f32 / 96.0).cos();
                window * (2.0 * PI * 2000.0 * n as f32 / RATE as f32).sin()
            } else {
                0.0
            }
        })
        .collect();
    for degrees in [30.0f32, 45.0] {
        let azimuth = degrees.to_radians();
        // At `azimuth` to the right of a listener facing −Z.
        let at = Vec3::new(azimuth.sin(), 0.0, -azimuth.cos()) * 5.0;
        let out = play(
            AudioClip::new(1, RATE, burst.clone()),
            0.0,
            at,
            0.0,
            None,
            8,
        );
        let (left, right) = (channel(&out, 0), channel(&out, 1));
        let measured = lag(&right[4 * TICK..], &left[4 * TICK..], 64) as f32;
        let expected = woodworth_itd(azimuth) * RATE as f32;
        eprintln!("{degrees}°: the left ear lags {measured} frames; Woodworth {expected:.1}");
        assert!(
            (measured - expected).abs() <= 2.0,
            "{degrees}°: the left ear lags by {measured} frames, not {expected:.1}"
        );
    }
}

/// The head shadows the far ear: for a source 60° to the right, the right ear's lead over the
/// left is at least 6 dB larger at 8 kHz than at 200 Hz.
#[test]
fn the_head_shadows_the_far_ear() {
    let at = Vec3::new(60f32.to_radians().sin(), 0.0, -60f32.to_radians().cos()) * 5.0;
    let lead = |hz: f32| {
        let (left, right) = levels(&play(tone(hz), 0.0, at, 0.0, None, 8));
        right - left
    };
    let (low, high) = (lead(200.0), lead(8_000.0));
    eprintln!("the right ear leads by {low:.2} dB at 200 Hz and {high:.2} dB at 8 kHz");
    assert!(
        high > low + 6.0,
        "the right ear leads by {high} dB at 8 kHz and {low} dB at 200 Hz"
    );
}

/// A source straight ahead reaches both ears together and alike: no delay, no shadow.
#[test]
fn a_source_ahead_reaches_both_ears_alike() {
    let out = play(tone(8_000.0), 0.0, Vec3::new(0.0, 0.0, -5.0), 0.0, None, 6);
    assert_eq!(channel(&out, 0), channel(&out, 1));
}
