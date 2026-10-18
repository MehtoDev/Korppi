//! Per-voice early reflections (FIX-7.5, ADR-051; finding AUD-08).
//!
//! Before FIX-7.5 the acoustics worker merged every emitter's reflections into one set of taps
//! and the coordinator fed the mono sum of all voices through it. Now each of the loudest
//! raytraced voices gets its own taps (delay after its direct sound, gain per ear), rendered in
//! one of the mixer's reflection slots, from the paths that reach its emitter: the rays find the
//! chains of surfaces a specular reflection takes, and each chain's image source gives the exact
//! path to the emitter, checked against the geometry.

mod common;

use common::{channel, Game, RATE};
use korppi_audio::mixer::{
    AudioMixer, ClipData, ClipId, MixerConfig, MixerControl, ReflectionTap, Reflections,
    VoiceParams,
};
use korppi_audio::player::{AudioPlayer, AudioRoom, AudioSource};
use korppi_audio::raytrace::{
    AcousticGeometry, AcousticMaterial, ReflectingPlane, RoomInput, RoomWorker, SpecularChain,
    AIR_ABSORPTION_COEFFS, SPEED_OF_SOUND,
};
use korppi_audio::spatial::SpatializerMode;
use korppi_audio::{
    AudioClip, AudioDevice, AudioListener, AudioSettings, AudioStream, AudioStreamConfig,
    BufferSize, ChannelCount, HeadlessDummyDevice,
};
use korppi_math::{Quat, Vec3};
use korppi_physics::{BodyType, Collider, PhysicsWorld, RigidBody, Shape};
use korppi_scene::GlobalTransform;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

fn impulse(control: &mut MixerControl) -> ClipId {
    let mut samples = vec![0.0; 2048];
    samples[0] = 1.0;
    control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(1, RATE, samples))))
        .unwrap()
}

fn taps(list: &[(f32, f32, f32)]) -> Reflections {
    let mut reflections = Reflections::default();
    for &(delay, gain_left, gain_right) in list {
        reflections.taps[reflections.count] = ReflectionTap {
            delay,
            gain_left,
            gain_right,
        };
        reflections.count += 1;
    }
    reflections
}

/// A voice's taps repeat it at their delays after its direct sound, with their gains per ear,
/// to the sample.
#[test]
fn taps_repeat_the_voice() {
    let (mut control, mut stream) = open();
    let clip = impulse(&mut control);
    let voice = control.start(clip, VoiceParams::default()).unwrap();
    control
        .set_reflections(voice, taps(&[(0.010, 0.5, 0.0), (0.0205, 0.25, 0.25)]), 0)
        .unwrap();
    let out = stream.render_frames(2048);
    let (left, right) = (channel(&out, 0), channel(&out, 1));
    assert_eq!((left[0], right[0]), (1.0, 1.0), "the direct sound");
    assert_eq!(
        (left[480], right[480]),
        (0.5, 0.0),
        "10 ms later, on the left"
    );
    assert_eq!(
        (left[984], right[984]),
        (0.25, 0.25),
        "20.5 ms later, on both"
    );
    let others = (1..2048).filter(|&n| n != 480 && n != 984);
    assert!(
        others.clone().all(|n| left[n] == 0.0 && right[n] == 0.0),
        "nothing else"
    );
}

/// A voice's reflections outlive it: a click shorter than its echo's delay is still heard
/// 10 ms later, after the voice has ended.
#[test]
fn reflections_outlive_their_voice() {
    let (mut control, mut stream) = open();
    let mut samples = vec![0.0; 64];
    samples[0] = 1.0;
    let click = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(1, RATE, samples))))
        .unwrap();
    let voice = control.start(click, VoiceParams::default()).unwrap();
    control
        .set_reflections(voice, taps(&[(0.010, 0.5, 0.25)]), 0)
        .unwrap();
    let out = stream.render_frames(1024);
    let (left, right) = (channel(&out, 0), channel(&out, 1));
    assert_eq!(
        (left[480], right[480]),
        (0.5, 0.25),
        "the echo after the voice"
    );
}

/// The mixer renders reflections for a fixed number of voices (16): a 17th voice's taps wait
/// until a slot is free, which a voice's end frees.
#[test]
fn reflection_slots_are_fixed_and_reused() {
    let (mut control, mut stream) = open();
    let clip = impulse(&mut control);
    let echo = taps(&[(0.010, 0.5, 0.5)]);
    let long = control
        .register_clip(Arc::new(ClipData::from(AudioClip::new(
            1,
            RATE,
            vec![0.0; 48_000],
        ))))
        .unwrap();
    // Sixteen silent voices hold every slot.
    let holders: Vec<_> = (0..16)
        .map(|_| {
            let voice = control.start(long, VoiceParams::default()).unwrap();
            control.set_reflections(voice, echo, 0).unwrap();
            voice
        })
        .collect();
    let late = control.start(clip, VoiceParams::default()).unwrap();
    control.set_reflections(late, echo, 0).unwrap();
    let out = channel(&stream.render_frames(1024), 0);
    assert_eq!(out[480], 0.0, "no slot for the 17th voice");

    control.stop(holders[0]).unwrap();
    stream.render_frames(1024);
    let next = control.start(clip, VoiceParams::default()).unwrap();
    control.set_reflections(next, echo, 0).unwrap();
    let out = channel(&stream.render_frames(1024), 0);
    assert_eq!(out[480], 0.5, "the freed slot serves the next voice");
}

/// The click's peak, and the echo's amplitude and lateness in frames after it, searched `window`
/// frames after the peak. A tap at a fractional delay spreads the click over a few frames by
/// interpolation, whose weights sum to one and whose centroid is the delay: the echo's
/// amplitude is its frames' sum, its lateness their centroid.
fn echo(out: &[f32], window: std::ops::Range<usize>) -> (f32, f32, f32) {
    let loudest = |samples: &[f32]| {
        samples
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(n, &s)| (n, s))
            .unwrap()
    };
    let (at, direct) = loudest(out);
    let (peak, _) = loudest(&out[at + window.start..at + window.end]);
    let around = at + window.start + peak - 4..at + window.start + peak + 5;
    let amplitude: f32 = out[around.clone()].iter().sum();
    let centroid = around.map(|n| (n - at) as f32 * out[n]).sum::<f32>() / amplitude;
    (direct, amplitude, centroid)
}

/// In raytraced mode each player hears the floor's reflection from where it stands, with the
/// default 128 rays: the image source's extra path after the direct sound, at its amplitude
/// (the default inverse-distance attenuation over the longer path, the floor's reflectance, the
/// air its share over the extra path). Player A is 4 m to the right (paths of 4 m and 5 m:
/// 140 frames later at 4/5 of the direct sound before the floor's 0.8), player B 2 m ahead (2 m
/// and 3.61 m: 225 frames later at 0.555 before the floor's 0.8). Both reflect at once, each
/// with its own taps, among sixteen quieter players.
#[test]
fn each_player_hears_the_floor_from_where_it_stands() {
    let mut world = PhysicsWorld::new();
    let body = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, -0.5, 0.0),
        rotation: Quat::IDENTITY,
        ..RigidBody::default()
    });
    world.create_collider(
        body,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        }),
    );
    // Absorbing 36 %, the floor keeps 0.8 of the amplitude.
    let floor = AcousticMaterial {
        absorption: [0.36; 3],
        scattering: 0.0,
        transmission: [0.0; 3],
    };
    let floor = Arc::new(AcousticGeometry::from_physics(&world, |_| Some(floor)));

    let mut game = Game::new();
    game.world
        .resource_mut::<AudioSettings>()
        .unwrap()
        .spatializer_mode = SpatializerMode::Raytraced;
    game.world.resource_mut::<AudioRoom>().unwrap().geometry = Some(floor);
    let mut samples = vec![0.0; 9600];
    samples[0] = 1.0;
    game.clip("click", AudioClip::new(1, RATE, samples));
    let click = game.start().remove(0);
    let listener = Vec3::new(0.0, 1.5, 0.0);
    game.world.spawn((
        AudioListener::default(),
        GlobalTransform::from_translation(listener),
    ));
    let at = |offset: Vec3| GlobalTransform::from_translation(listener + offset);
    // Sixteen quiet players far away, first in the query: the worker traces the 16 loudest.
    for index in 0..16 {
        game.world.spawn((
            AudioPlayer::new(AudioSource::clip(click.clone())),
            at(Vec3::new(index as f32 - 8.0, 0.0, 60.0)),
        ));
    }
    let a = game.world.spawn((
        AudioPlayer::new(AudioSource::clip(click.clone())),
        at(Vec3::new(4.0, 0.0, 0.0)),
    ));
    let b = game.world.spawn((
        AudioPlayer::new(AudioSource::clip(click)),
        at(Vec3::new(0.0, 0.0, -2.0)),
    ));
    // The worker traces on its own thread, in real time.
    let start = Instant::now();
    while game.world.resource::<AudioRoom>().unwrap().traces() < 3 {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "the worker does not trace"
        );
        game.tick();
        std::thread::sleep(Duration::from_millis(5));
    }
    let frames = |metres: f32| metres / SPEED_OF_SOUND * RATE as f32;
    for (player, extra, ratio) in [
        (a, 1.0, 0.8 * 0.8),
        (b, 13f32.sqrt() - 2.0, 2.0 / 13f32.sqrt() * 0.8),
    ] {
        game.player(player).play();
        let out = channel(&game.ticks(4), 1);
        let (direct, reflected, late) = echo(&out, 100..300);
        let (expected, ratio) = (
            frames(extra),
            ratio * (-AIR_ABSORPTION_COEFFS[1] * extra / 2.0).exp(),
        );
        eprintln!(
            "direct {direct:.4}, echo {reflected:.4} ({:.4}, expected {ratio:.4}) {late:.2} \
             frames later ({expected:.2})",
            reflected / direct
        );
        assert!(
            (late - expected).abs() < 0.05,
            "the echo comes {late} frames after the direct sound, not {expected}"
        );
        assert!(
            (reflected / direct / ratio - 1.0).abs() < 0.005,
            "the echo is {} of the direct sound, not {ratio}",
            reflected / direct
        );
        // The click's late reverb dies away before the next.
        game.ticks(120);
    }
}

/// A floor at y = 0, a ceiling at y = 3, and whatever `extra` boxes, of `material`.
fn slabs(material: AcousticMaterial, extra: &[(Vec3, Vec3)]) -> AcousticGeometry {
    let mut world = PhysicsWorld::new();
    let slab = Vec3::new(50.0, 0.5, 50.0);
    let slabs = [
        (Vec3::new(0.0, -0.5, 0.0), slab),
        (Vec3::new(0.0, 3.5, 0.0), slab),
    ];
    for &(center, half_extents) in slabs.iter().chain(extra) {
        let body = world.create_body(RigidBody {
            body_type: BodyType::Static,
            position: center,
            rotation: Quat::IDENTITY,
            ..RigidBody::default()
        });
        world.create_collider(body, Collider::new(Shape::Box { half_extents }));
    }
    AcousticGeometry::from_physics(&world, |_| Some(material))
}

/// An image path follows its chain's planes in order, only where their surfaces are, and only
/// with every leg clear: off the floor then the ceiling, the emitter's image is mirrored across
/// the ceiling first; a chain whose mirror point misses its surface, lands on another surface
/// (a platform above the floor) or whose leg a box blocks has no path.
#[test]
fn image_paths_follow_their_surfaces() {
    let material = AcousticMaterial {
        absorption: [0.1, 0.2, 0.3],
        scattering: 0.5,
        transmission: [0.0; 3],
    };
    let (listener, emitter) = (Vec3::new(0.0, 1.5, 0.0), Vec3::new(4.0, 1.5, 0.0));
    let floor = ReflectingPlane::through(Vec3::ZERO, Vec3::Y);
    let ceiling = ReflectingPlane::through(Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y);
    let open = slabs(material, &[]);
    let path = SpecularChain::new(&[floor, ceiling])
        .image_path(&open, listener, emitter)
        .expect("floor, then ceiling");
    // The image is 4 m across and 6 m down: down to the floor first.
    assert!((path.length - 52f32.sqrt()).abs() < 1e-4, "{}", path.length);
    assert!(path.arrival.y < 0.0, "{:?}", path.arrival);
    let expected = [0.1f32, 0.2, 0.3].map(|alpha| ((1.0 - alpha) * 0.5f32).powi(2));
    for (got, want) in path.reflectance.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "{:?}", path.reflectance);
    }
    assert_eq!(path.order, 2);

    // A floor 1 m wide under the listener: the mirror point, 2 m out, misses it.
    let mut world = PhysicsWorld::new();
    let body = world.create_body(RigidBody {
        body_type: BodyType::Static,
        position: Vec3::new(0.0, -0.5, 0.0),
        rotation: Quat::IDENTITY,
        ..RigidBody::default()
    });
    world.create_collider(
        body,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(0.5, 0.5, 0.5),
        }),
    );
    let patch = AcousticGeometry::from_physics(&world, |_| Some(material));
    let first = SpecularChain::new(&[floor]);
    assert_eq!(first.image_path(&patch, listener, emitter), None);
    // A box on the floor between the mirror point (x = 2) and the emitter blocks the last leg.
    let blocked = slabs(
        material,
        &[(Vec3::new(3.0, 0.5, 0.0), Vec3::new(0.2, 0.5, 1.0))],
    );
    assert_eq!(first.image_path(&blocked, listener, emitter), None);
    // A platform 0.3 m above the floor around the mirror point: the leg reflects off it, not off
    // the floor.
    let raised = slabs(
        material,
        &[(Vec3::new(2.0, 0.15, 0.0), Vec3::new(1.0, 0.15, 1.0))],
    );
    assert_eq!(first.image_path(&raised, listener, emitter), None);
    assert!(first.image_path(&open, listener, emitter).is_some());
}

/// The worker keeps an emitter's chains from trace to trace: with one ray per trace, which
/// finds at most three chains, an emitter between a floor, a ceiling and a wall gathers eight
/// paths (of the more than eight chains of up to three reflections there; between the floor and
/// the ceiling alone there are six).
#[test]
fn an_emitter_keeps_its_chains_from_trace_to_trace() {
    let material = AcousticMaterial {
        absorption: [0.1; 3],
        scattering: 0.0,
        transmission: [0.0; 3],
    };
    let wall = (Vec3::new(6.5, 1.5, 0.0), Vec3::new(0.5, 1.5, 50.0));
    let mut worker = RoomWorker::spawn(Arc::new(slabs(material, &[wall])));
    let mut input = RoomInput {
        listener: Vec3::new(0.0, 1.5, 0.0),
        receiver_count: 1,
        rays: 1,
        max_bounces: 3,
        ..RoomInput::default()
    };
    input.receivers[0] = Vec3::new(4.0, 1.0, 1.0);
    input.ids[0] = 7;
    worker.update(input);
    let start = Instant::now();
    loop {
        let output = worker.output();
        let paths = output
            .reflections_of(7)
            .map_or(0, |reflections| reflections.count);
        if paths == 8 {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "{paths} paths after {} traces",
            output.traces
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}
