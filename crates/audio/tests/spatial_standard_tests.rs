//! Unit and integration tests for Sub-Milestone 10.3:
//! Standard Spatial Audio & Psychoacoustic Panning (Default Mode).

use korppi_audio::spatial::attenuation::{AttenuationModel, DistanceAttenuation};
use korppi_audio::spatial::binaural::WoodworthBinaural;
use korppi_audio::spatial::directivity::EmitterCone;
use korppi_audio::spatial::doppler::DopplerPitch;
use korppi_audio::spatial::panning::EqualPowerPanner;
use korppi_audio::spatial::standard::StandardSpatializer;
use korppi_audio::spatial::{
    AcousticRaycaster, EmitterState, ListenerState, SpatialAudioModel, SpatializerMode,
    VoiceSpatialParams,
};
use korppi_math::Vec3;
use std::f32::consts::PI;

#[test]
fn test_spatializer_mode_defaults_to_standard() {
    assert_eq!(SpatializerMode::default(), SpatializerMode::Standard);
}

#[test]
fn test_distance_attenuation_models() {
    // 1. Inverse Distance Clamped
    let inv = DistanceAttenuation::InverseClamped {
        min_distance: 1.0,
        max_distance: 100.0,
        rolloff: 1.0,
    };
    assert_eq!(
        inv.calculate_gain(0.5),
        1.0,
        "Distances <= min_distance must have unity gain"
    );
    assert_eq!(inv.calculate_gain(1.0), 1.0);
    // At d = 2.0: 1.0 / (1.0 + 1.0 * (2.0 - 1.0)) = 0.5
    assert!((inv.calculate_gain(2.0) - 0.5).abs() < 1e-5);
    // At d = 3.0: 1.0 / (1.0 + 1.0 * (3.0 - 1.0)) = 0.3333
    assert!((inv.calculate_gain(3.0) - 0.333333).abs() < 1e-4);
    assert_eq!(
        inv.calculate_gain(100.0),
        0.0,
        "At max_distance, gain must be 0.0"
    );
    assert_eq!(inv.calculate_gain(150.0), 0.0);

    // 2. Linear Attenuation
    let lin = DistanceAttenuation::Linear {
        min_distance: 10.0,
        max_distance: 50.0,
    };
    assert_eq!(lin.calculate_gain(5.0), 1.0);
    assert_eq!(lin.calculate_gain(10.0), 1.0);
    assert!((lin.calculate_gain(30.0) - 0.5).abs() < 1e-5);
    assert_eq!(lin.calculate_gain(50.0), 0.0);
    assert_eq!(lin.calculate_gain(60.0), 0.0);

    // 3. Exponential Attenuation
    let exp = DistanceAttenuation::Exponential {
        min_distance: 2.0,
        max_distance: 100.0,
        rolloff: 2.0,
    };
    assert_eq!(exp.calculate_gain(1.0), 1.0);
    // At d = 4.0: (4.0 / 2.0)^(-2) = 2^(-2) = 0.25
    assert!((exp.calculate_gain(4.0) - 0.25).abs() < 1e-5);
    assert_eq!(exp.calculate_gain(100.0), 0.0);
}

#[test]
fn test_equal_power_stereo_panning_conservation() {
    let panner = EqualPowerPanner::new();

    // Test equal-power conservation across all angles
    for deg in (-180..=180).step_by(10) {
        let rad = (deg as f32) * PI / 180.0;
        let (gl, gr) = panner.calculate_gains(rad);
        let power = gl * gl + gr * gr;
        assert!(
            (power - 1.0).abs() < 1e-4,
            "Total acoustic power must be 1.0 at angle {deg} deg (got {power})"
        );
    }

    // Hard left (azimuth -PI/2)
    let (gl_left, gr_left) = panner.calculate_gains(-PI * 0.5);
    assert!((gl_left - 1.0).abs() < 1e-4);
    assert!(gr_left.abs() < 1e-4);

    // Hard right (azimuth +PI/2)
    let (gl_right, gr_right) = panner.calculate_gains(PI * 0.5);
    assert!(gl_right.abs() < 1e-4);
    assert!((gr_right - 1.0).abs() < 1e-4);

    // Dead center (azimuth 0.0)
    let (gl_center, gr_center) = panner.calculate_gains(0.0);
    assert!((gl_center - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
    assert!((gr_center - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-4);
}

#[test]
fn test_woodworth_itd_binaural_delay() {
    let binaural = WoodworthBinaural::default();

    // Center (0 rad): no delay between ears
    let (delay_l_center, delay_r_center) = binaural.calculate_itd_seconds(0.0);
    assert_eq!(delay_l_center, 0.0);
    assert_eq!(delay_r_center, 0.0);

    // Source to the right (+PI/2 rad): Left ear is contralateral (receives delay)
    let (delay_l_right, delay_r_right) = binaural.calculate_itd_seconds(PI * 0.5);
    assert_eq!(
        delay_r_right, 0.0,
        "Ipsilateral ear must have zero ITD delay"
    );
    // Standard human head ITD delay is approx 0.000655 seconds (0.655 ms)
    assert!(
        (delay_l_right - 0.000655).abs() < 0.00005,
        "Contralateral ear delay expected ~0.655 ms, got {}",
        delay_l_right * 1000.0
    );

    // Source to the left (-PI/2 rad): Right ear is contralateral
    let (delay_l_left, delay_r_left) = binaural.calculate_itd_seconds(-PI * 0.5);
    assert_eq!(delay_l_left, 0.0);
    assert!((delay_r_left - 0.000655).abs() < 0.00005);
}

#[test]
fn test_emitter_cone_directivity() {
    let cone = EmitterCone {
        inner_angle_rad: PI * 0.25, // 45 degrees
        outer_angle_rad: PI * 0.5,  // 90 degrees
        outer_gain: 0.2,            // -14 dB
        outer_lpf_cutoff: 2000.0,
    };

    // Forward facing directly at listener: angle 0.0 rad
    let (gain_on_axis, lpf_on_axis) = cone.calculate_attenuation(0.0);
    assert_eq!(gain_on_axis, 1.0);
    assert_eq!(lpf_on_axis, 22_000.0); // Full bandwidth

    // Inside inner cone: angle 10 degrees
    let (gain_inner, _) = cone.calculate_attenuation(10.0 * PI / 180.0);
    assert_eq!(gain_inner, 1.0);

    // Halfway between inner (22.5 deg half-angle) and outer (45 deg half-angle): angle 33.75 deg
    let (gain_mid, _) = cone.calculate_attenuation((33.75 * PI / 180.0).clamp(0.0, PI));
    assert!(
        gain_mid > 0.2 && gain_mid < 1.0,
        "Transition cone gain must be between outer_gain and 1.0 (got {gain_mid})"
    );

    // Outside outer cone: angle 120 degrees
    let (gain_behind, lpf_behind) = cone.calculate_attenuation(120.0 * PI / 180.0);
    assert_eq!(gain_behind, 0.2);
    assert_eq!(lpf_behind, 2000.0);
}

#[test]
fn test_doppler_radial_velocity_pitch_shift() {
    let doppler = DopplerPitch::new();
    let listener_pos = Vec3::ZERO;
    let emitter_pos = Vec3::new(10.0, 0.0, 0.0); // Emitter is on the +X axis

    // 1. Static: pitch is 1.0
    let pitch_static = doppler.calculate_pitch(listener_pos, Vec3::ZERO, emitter_pos, Vec3::ZERO);
    assert!((pitch_static - 1.0).abs() < 1e-5);

    // 2. Emitter approaching listener at 34.3 m/s (10% speed of sound)
    // Moving in -X direction towards listener
    let pitch_approaching = doppler.calculate_pitch(
        listener_pos,
        Vec3::ZERO,
        emitter_pos,
        Vec3::new(-34.3, 0.0, 0.0),
    );
    assert!(
        pitch_approaching > 1.10 && pitch_approaching < 1.12,
        "Approaching at 0.1c should pitch shift to ~1.11 (got {pitch_approaching})"
    );

    // 3. Emitter receding from listener at 34.3 m/s
    let pitch_receding = doppler.calculate_pitch(
        listener_pos,
        Vec3::ZERO,
        emitter_pos,
        Vec3::new(34.3, 0.0, 0.0),
    );
    assert!(
        pitch_receding < 0.92 && pitch_receding > 0.90,
        "Receding at 0.1c should pitch shift to ~0.91 (got {pitch_receding})"
    );

    // 4. Extreme supersonic velocity: pitch must be clamped safely without NaN
    let pitch_extreme = doppler.calculate_pitch(
        listener_pos,
        Vec3::ZERO,
        emitter_pos,
        Vec3::new(-1000.0, 0.0, 0.0),
    );
    assert!((0.2..=5.0).contains(&pitch_extreme));
    assert!(!pitch_extreme.is_nan());
}

struct MockWallRaycaster;
impl AcousticRaycaster for MockWallRaycaster {
    fn raycast_occlusion(&self, _from: Vec3, _to: Vec3) -> Option<f32> {
        // Wall blocks line of sight with 100% occlusion
        Some(1.0)
    }
}

#[test]
fn test_standard_spatializer_end_to_end_with_occlusion() {
    let mut spatializer = StandardSpatializer::new(48_000.0);

    let listener = ListenerState {
        position: Vec3::ZERO,
        forward: Vec3::NEG_Z, // Facing -Z, the coordinate contract's forward (FIX-7.3, AUD-04; owner-approved)
        up: Vec3::Y,          // +Y Up, so +X is Right
        velocity: Vec3::ZERO,
    };

    // Emitter positioned 10 meters to the right: (+10, 0, 0)
    let emitter = EmitterState {
        position: Vec3::new(10.0, 0.0, 0.0),
        forward: Vec3::NEG_X, // Pointing at listener
        velocity: Vec3::ZERO,
        attenuation: AttenuationModel::InverseClamped {
            min_distance: 1.0,
            max_distance: 100.0,
            rolloff: 1.0,
        },
        cone: None,
        doppler_factor: 1.0,
    };

    let mut params_clear = VoiceSpatialParams::default();
    spatializer.spatialize_emitter(&listener, &emitter, None, &mut params_clear);

    // Panning: source is at +X (Right), so Right channel gain must exceed Left channel gain
    assert!(
        params_clear.gain_right > params_clear.gain_left,
        "Sound to the right must have louder right ear gain (left {} vs right {})",
        params_clear.gain_left,
        params_clear.gain_right
    );

    // ITD: Left ear (contralateral) must have positive arrival delay
    assert!(params_clear.delay_left_sec > 0.0004);
    assert_eq!(params_clear.delay_right_sec, 0.0);

    // Distance attenuation: at 10 meters, gain is approx 1 / (1 + 1 * 9) = 0.1
    let total_gain = (params_clear.gain_left * params_clear.gain_left
        + params_clear.gain_right * params_clear.gain_right)
        .sqrt();
    assert!(
        (total_gain - 0.1).abs() < 0.02,
        "Total distance gain at 10m expected ~0.1, got {total_gain}"
    );

    // Now test with Mock Wall Raycaster (100% occlusion)
    let raycaster = MockWallRaycaster;
    let mut params_occluded = VoiceSpatialParams::default();
    spatializer.spatialize_emitter(&listener, &emitter, Some(&raycaster), &mut params_occluded);

    // Occluded sound must be significantly quieter and have a lower cutoff frequency
    let total_gain_occluded = (params_occluded.gain_left * params_occluded.gain_left
        + params_occluded.gain_right * params_occluded.gain_right)
        .sqrt();
    assert!(
        total_gain_occluded < total_gain * 0.6,
        "Occluded sound must be attenuated (clear {total_gain} vs occluded {total_gain_occluded})"
    );
    assert!(
        params_occluded.low_pass_cutoff < 2000.0,
        "Occluded sound must have low-pass cutoff muffling (got {})",
        params_occluded.low_pass_cutoff
    );
}

#[test]
fn test_zero_distance_emitter_uses_equal_power_center_gains() {
    use korppi_audio::spatial::raytraced::RaytracedSpatializer;

    let mut standard = StandardSpatializer::new(48_000.0);
    let mut raytraced = RaytracedSpatializer::new(48_000.0);

    let listener = ListenerState {
        position: Vec3::new(1.0, 2.0, -3.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        velocity: Vec3::ZERO,
    };

    let emitter_coincident = EmitterState {
        position: listener.position,
        forward: Vec3::NEG_Z,
        velocity: Vec3::ZERO,
        attenuation: AttenuationModel::InverseClamped {
            min_distance: 1.0,
            max_distance: 100.0,
            rolloff: 1.0,
        },
        cone: None,
        doppler_factor: 1.0,
    };

    let emitter_just_ahead = EmitterState {
        position: listener.position + Vec3::new(0.0, 0.0, -1e-3),
        ..emitter_coincident
    };

    for model in [
        &mut standard as &mut dyn SpatialAudioModel,
        &mut raytraced as &mut dyn SpatialAudioModel,
    ] {
        let mut at_zero = VoiceSpatialParams::default();
        let mut ahead = VoiceSpatialParams::default();
        model.spatialize_emitter(&listener, &emitter_coincident, None, &mut at_zero);
        model.spatialize_emitter(&listener, &emitter_just_ahead, None, &mut ahead);

        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (at_zero.gain_left - expected).abs() < 1e-5
                && (at_zero.gain_right - expected).abs() < 1e-5,
            "Coincident emitter must get equal-power center gains ({expected}, {expected}), got ({}, {})",
            at_zero.gain_left,
            at_zero.gain_right
        );
        let power_zero = at_zero.gain_left.powi(2) + at_zero.gain_right.powi(2);
        let power_ahead = ahead.gain_left.powi(2) + ahead.gain_right.powi(2);
        assert!(
            (power_zero - 1.0).abs() < 1e-4 && (power_zero - power_ahead).abs() < 1e-4,
            "Zero-distance power ({power_zero}) must match centered equal-power ({power_ahead}) without a +3 dB step"
        );
    }
}
