//! Raytraced spatializer implementing the `SpatialAudioModel` interface.

use std::sync::Arc;

use crate::raytrace::{AcousticRay, AcousticScene, PathTraceConfig, RaytracedAcoustics};

use super::binaural::WoodworthBinaural;
use super::doppler::DopplerPitch;
use super::panning::EqualPowerPanner;
use super::{
    AcousticRaycaster, EmitterState, ListenerState, ReverbEnvironmentParams, SpatialAudioModel,
    VoiceSpatialParams,
};

/// Physically simulated spatializer computing acoustic ray propagation, diffraction, and reflections.
#[derive(Debug)]
pub struct RaytracedSpatializer {
    sample_rate: f32,
    panner: EqualPowerPanner,
    binaural: WoodworthBinaural,
    doppler: DopplerPitch,
    raytracer: RaytracedAcoustics,
    acoustic_scene: Option<Arc<AcousticScene>>,
}

impl RaytracedSpatializer {
    /// Creates a new raytraced spatializer initialized with the target sample rate.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            panner: EqualPowerPanner::new(),
            binaural: WoodworthBinaural::default(),
            doppler: DopplerPitch::new(),
            raytracer: RaytracedAcoustics::new(PathTraceConfig::default()),
            acoustic_scene: None,
        }
    }

    /// Sets the acoustic scene for geometric raytracing, multi-wall transmission, and reflections.
    pub fn set_acoustic_scene(&mut self, scene: AcousticScene) {
        self.set_acoustic_scene_arc(Arc::new(scene));
    }

    /// Sets the acoustic scene via an [`Arc`] reference to prevent deep geometric clones.
    pub fn set_acoustic_scene_arc(&mut self, scene: Arc<AcousticScene>) {
        self.acoustic_scene = Some(scene);
    }

    /// Returns a reference to the active acoustic scene, if present.
    #[must_use]
    pub fn acoustic_scene(&self) -> Option<&AcousticScene> {
        self.acoustic_scene.as_deref()
    }

    /// Sets the path tracer configuration (ray count, bounce depth, energy threshold).
    pub fn set_path_trace_config(&mut self, config: PathTraceConfig) {
        self.raytracer.config = config;
    }

    /// Returns the current path tracer configuration.
    #[must_use]
    pub fn path_trace_config(&self) -> &PathTraceConfig {
        &self.raytracer.config
    }

    /// Returns the active sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

impl SpatialAudioModel for RaytracedSpatializer {
    fn spatialize_emitter(
        &mut self,
        listener: &ListenerState,
        emitter: &EmitterState,
        occlusion_query: Option<&dyn AcousticRaycaster>,
        output_params: &mut VoiceSpatialParams,
    ) {
        let delta = emitter.position - listener.position;
        let distance = delta.length();

        if distance < 1e-4 {
            // Coincident with the listener: centred equal-power gains (1/√2 per ear, total
            // power 1.0) rather than 1.0 per ear, avoiding a +3 dB step (AUD-12).
            let center_gain =
                emitter.attenuation.calculate_gain(distance) * std::f32::consts::FRAC_1_SQRT_2;
            output_params.gain_left = center_gain;
            output_params.gain_right = center_gain;
            output_params.delay_left_sec = 0.0;
            output_params.delay_right_sec = 0.0;
            output_params.pitch_scale = 1.0;
            output_params.low_pass_cutoff = 22_000.0;
            output_params.head_shadow_left = 22_000.0;
            output_params.head_shadow_right = 22_000.0;
            output_params.reverb_send = 0.2;
            return;
        }

        let direction = delta / distance;

        // 1. Distance Attenuation
        let mut dist_gain = emitter.attenuation.calculate_gain(distance);
        if dist_gain <= 0.0 {
            output_params.gain_left = 0.0;
            output_params.gain_right = 0.0;
            return;
        }

        // 2. Local Azimuth Computation
        let forward = listener.forward.normalize_or_zero();
        let up = listener.up.normalize_or_zero();
        // The coordinate contract's right: forward × up (−Z forward, +Y up, +X right; AUD-04).
        let right = forward.cross(up).normalize_or_zero();

        let x_local = direction.dot(right);
        let z_local = direction.dot(forward);
        let azimuth = x_local.atan2(z_local);

        // 3. Stereo Panning
        let (pan_l, pan_r) = self.panner.calculate_gains(azimuth);

        // 4. Binaural ITD Delay
        let (itd_l, itd_r) = self.binaural.calculate_itd_seconds(azimuth);

        // 5. Distance High-Frequency Roll-Off Heuristic (linear 22 kHz − 25 Hz/m down to 1 kHz; AUD-12)
        let mut lpf_cutoff = (22_000.0 - distance * 25.0).clamp(1000.0, 22_000.0);

        // 6. Directional Cone
        if let Some(cone) = &emitter.cone {
            let emitter_fwd = emitter.forward.normalize_or_zero();
            let cos_angle = emitter_fwd.dot(-direction).clamp(-1.0, 1.0);
            let off_axis_angle = cos_angle.acos();
            let (cone_gain, cone_lpf) = cone.calculate_attenuation(off_axis_angle);
            dist_gain *= cone_gain;
            lpf_cutoff = lpf_cutoff.min(cone_lpf);
        }

        // 7. Classical Doppler Pitch Shift
        let raw_pitch = self.doppler.calculate_pitch(
            listener.position,
            listener.velocity,
            emitter.position,
            emitter.velocity,
        );
        let pitch = 1.0 + emitter.doppler_factor * (raw_pitch - 1.0);

        // 8. Line-of-sight Occlusion & Transmission (when no AcousticGeometry is active, falls back to heuristic; AUD-12)
        if let Some(scene) = &self.acoustic_scene {
            let direct_ray = AcousticRay::new(listener.position, direction);
            let trans = scene.raycast_transmission(&direct_ray, distance);
            if trans[0] < 0.999 || trans[1] < 0.999 || trans[2] < 0.999 {
                let avg_trans = (trans[0] * 0.3 + trans[1] * 0.5 + trans[2] * 0.2).clamp(1e-8, 1.0);
                dist_gain *= avg_trans;
                let high_factor = trans[2].clamp(1e-8, 1.0).powf(0.25);
                lpf_cutoff = (800.0 + (lpf_cutoff - 800.0) * high_factor).min(lpf_cutoff);
            }
        } else if let Some(raycaster) = occlusion_query {
            if let Some(occ_fraction) =
                raycaster.raycast_occlusion(listener.position, emitter.position)
            {
                let occ = occ_fraction.clamp(0.0, 1.0);
                dist_gain *= (1.0 - 0.7 * occ).clamp(0.05, 1.0);
                lpf_cutoff = lpf_cutoff.min(22_000.0 - occ * 19_000.0);
            }
        }

        output_params.gain_left = dist_gain * pan_l;
        output_params.gain_right = dist_gain * pan_r;
        output_params.delay_left_sec = itd_l;
        output_params.delay_right_sec = itd_r;
        output_params.pitch_scale = pitch;
        output_params.low_pass_cutoff = lpf_cutoff;
        let (shadow_left, shadow_right) = self.binaural.head_shadow_cutoffs(azimuth);
        output_params.head_shadow_left = shadow_left;
        output_params.head_shadow_right = shadow_right;

        // 9. Physical Reverb Send derived from Distance (and AcousticScene when attached directly)
        let mut reverb_send = (0.2 + (distance * 0.05)).clamp(0.1, 0.85);
        if let Some(scene) = &self.acoustic_scene {
            if scene.triangle_count() > 0 {
                let reflections = self.raytracer.trace_early_reflections(
                    scene,
                    emitter.position,
                    listener.position,
                );
                let total_energy: f32 = reflections
                    .iter()
                    .map(|r| r.energy[0] + r.energy[1] + r.energy[2])
                    .sum();
                if !reflections.is_empty() || total_energy > 1e-6 {
                    reverb_send = (reverb_send + 0.05 + total_energy * 20.0).clamp(0.15, 0.95);
                }
            }
        }
        output_params.reverb_send = reverb_send;
    }

    fn update_environment_acoustics(
        &mut self,
        _listener: &ListenerState,
        _raycaster: Option<&dyn AcousticRaycaster>,
        reverb_params: &mut ReverbEnvironmentParams,
    ) {
        if let Some(scene) = &self.acoustic_scene {
            if scene.triangle_count() > 0 {
                let rt60 = scene.estimate_rt60_eyring();
                let mid_rt60 = rt60[1];
                reverb_params.room_size = (mid_rt60 / 2.0).clamp(0.1, 0.95);
                reverb_params.damping = (1.0 - (rt60[2] / mid_rt60.max(0.01))).clamp(0.05, 0.9);
                reverb_params.wet_mix = (0.2 + mid_rt60 * 0.1).clamp(0.2, 0.8);
                return;
            }
        }

        // In raytraced mode, dynamic reverberation is parameterized physically
        reverb_params.room_size = 0.7;
        reverb_params.damping = 0.4;
        reverb_params.wet_mix = 0.4;
    }
}
