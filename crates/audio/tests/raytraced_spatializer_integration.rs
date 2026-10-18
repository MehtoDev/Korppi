//! Verification test for RaytracedSpatializer integration with AcousticRaytracer (AUDIT-5.1 / Finding 4.1).
//!
//! Research reference: `research/audio/03_raytraced_audio_and_geometric_acoustics.md:1-150`.
//! RaytracedSpatializer must invoke geometric acoustic raytracing, evaluating multi-band material
//! transmission and multi-path reflection energy rather than relying on heuristic formulas.

use korppi_audio::raytrace::{AcousticBox, AcousticMaterial, AcousticScene, PathTraceConfig};
use korppi_audio::spatial::{
    EmitterState, ListenerState, RaytracedSpatializer, SpatialAudioModel, VoiceSpatialParams,
};
use korppi_math::Vec3;

#[test]
fn test_raytraced_spatializer_evaluates_material_transmission_spectrum() {
    let mut spatializer = RaytracedSpatializer::new(48_000.0);

    let listener = ListenerState {
        position: Vec3::ZERO,
        forward: Vec3::Z,
        up: Vec3::Y,
        velocity: Vec3::ZERO,
    };

    let emitter = EmitterState {
        position: Vec3::new(0.0, 0.0, 10.0),
        forward: -Vec3::Z,
        velocity: Vec3::ZERO,
        ..Default::default()
    };

    // 1. Free field (no obstacles)
    let mut params_free = VoiceSpatialParams::default();
    spatializer.spatialize_emitter(&listener, &emitter, None, &mut params_free);

    // 2. Concrete wall obstacle at z = 5.0 (heavy absorption & low transmission)
    let mut concrete_scene = AcousticScene::new();
    concrete_scene.add_box(AcousticBox::new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(10.0, 10.0, 0.5),
        AcousticMaterial::concrete(),
    ));
    spatializer.set_acoustic_scene(concrete_scene);

    let mut params_concrete = VoiceSpatialParams::default();
    spatializer.spatialize_emitter(&listener, &emitter, None, &mut params_concrete);

    // Concrete must attenuate gain and muffle high frequencies
    assert!(
        params_concrete.gain_left < params_free.gain_left * 0.5,
        "Concrete barrier must attenuate direct sound significantly: concrete={}, free={}",
        params_concrete.gain_left,
        params_free.gain_left
    );
    assert!(
        params_concrete.low_pass_cutoff < 5_000.0,
        "Concrete barrier must muffle low-pass cutoff below 5 kHz, got {}",
        params_concrete.low_pass_cutoff
    );

    // 3. Glass wall obstacle at z = 5.0 (higher high-frequency transmission than concrete)
    let mut glass_scene = AcousticScene::new();
    glass_scene.add_box(AcousticBox::new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(10.0, 10.0, 0.5),
        AcousticMaterial::glass(),
    ));
    spatializer.set_acoustic_scene(glass_scene);

    let mut params_glass = VoiceSpatialParams::default();
    spatializer.spatialize_emitter(&listener, &emitter, None, &mut params_glass);

    assert!(
        params_glass.gain_left > params_concrete.gain_left,
        "Glass must transmit more sound than concrete: glass={}, concrete={}",
        params_glass.gain_left,
        params_concrete.gain_left
    );
    assert!(
        params_glass.low_pass_cutoff > params_concrete.low_pass_cutoff,
        "Glass must have higher cutoff than concrete: glass={}, concrete={}",
        params_glass.low_pass_cutoff,
        params_concrete.low_pass_cutoff
    );
}

#[test]
fn test_raytraced_spatializer_reflection_energy_drives_reverb_send() {
    let mut spatializer = RaytracedSpatializer::new(48_000.0);
    spatializer.set_path_trace_config(PathTraceConfig {
        ray_count: 64,
        max_bounces: 3,
        energy_threshold: 1e-4,
    });

    let listener = ListenerState {
        position: Vec3::new(0.0, 1.0, 0.0),
        forward: Vec3::Z,
        up: Vec3::Y,
        velocity: Vec3::ZERO,
    };

    let emitter = EmitterState {
        position: Vec3::new(2.0, 1.0, 2.0),
        forward: -Vec3::Z,
        velocity: Vec3::ZERO,
        ..Default::default()
    };

    // 1. Free field (no reflecting surfaces)
    spatializer.set_acoustic_scene(AcousticScene::new());
    let mut params_free = VoiceSpatialParams::default();
    spatializer.spatialize_emitter(&listener, &emitter, None, &mut params_free);

    // 2. Enclosed concrete room (10m x 10m x 10m box)
    let mut room_scene = AcousticScene::new();
    // 6 walls around origin
    room_scene.add_box(AcousticBox::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(10.0, 10.0, 10.0),
        AcousticMaterial::concrete(),
    ));
    spatializer.set_acoustic_scene(room_scene);

    let mut params_room = VoiceSpatialParams::default();
    spatializer.spatialize_emitter(&listener, &emitter, None, &mut params_room);

    // In enclosed reflective room, raytraced reflection hits must increase reverb send above free-field!
    assert!(
        params_room.reverb_send > params_free.reverb_send,
        "Reflective room reverb send ({}) must exceed free field ({})",
        params_room.reverb_send,
        params_free.reverb_send
    );
}
