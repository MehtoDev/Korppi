//! Standard mathematical spatial audio implementation (Default Engine Mode).

use crate::spatial::binaural::WoodworthBinaural;
use crate::spatial::doppler::DopplerPitch;
use crate::spatial::panning::EqualPowerPanner;
use crate::spatial::{
    AcousticRaycaster, EmitterState, ListenerState, ReverbEnvironmentParams, SpatialAudioModel,
    VoiceSpatialParams,
};

/// Default spatializer implementing analytical distance attenuation,
/// equal-power stereo panning, Woodworth binaural ITD delays, emitter directivity,
/// classical Doppler pitch shifts, and heuristic distance/occlusion low-pass filtering (AUD-12).
#[derive(Debug, Clone)]
pub struct StandardSpatializer {
    sample_rate: f32,
    panner: EqualPowerPanner,
    binaural: WoodworthBinaural,
    doppler: DopplerPitch,
}

impl StandardSpatializer {
    /// Constructs a new standard spatializer.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            panner: EqualPowerPanner::new(),
            binaural: WoodworthBinaural::default(),
            doppler: DopplerPitch::new(),
        }
    }

    /// Returns the active sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

impl SpatialAudioModel for StandardSpatializer {
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
            output_params.reverb_send = 0.1;
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
        let forward = listener.forward.normalize();
        let up = listener.up.normalize();
        // The coordinate contract's right: forward × up (−Z forward, +Y up, +X right; AUD-04).
        let right = forward.cross(up).normalize();

        let x_local = direction.dot(right);
        let z_local = direction.dot(forward);
        let azimuth = x_local.atan2(z_local);

        // 3. Stereo Panning
        let (pan_l, pan_r) = self.panner.calculate_gains(azimuth);

        // 4. Binaural ITD Delay
        let (itd_l, itd_r) = self.binaural.calculate_itd_seconds(azimuth);

        // 5. Distance High-Frequency Roll-Off Heuristic (linear 22 kHz − 25 Hz/m down to 1 kHz; AUD-12)
        let mut lpf_cutoff = (22_000.0 - distance * 25.0).clamp(1000.0, 22_000.0);

        // 6. Emitter Directional Cone
        if let Some(cone) = &emitter.cone {
            let emitter_fwd = emitter.forward.normalize();
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

        // 8. Line-of-Sight Occlusion Heuristic (up to −6 dB gain and 1.2 kHz low-pass when blocked; AUD-12)
        if let Some(raycaster) = occlusion_query {
            if let Some(occ_fraction) =
                raycaster.raycast_occlusion(listener.position, emitter.position)
            {
                let occ = occ_fraction.clamp(0.0, 1.0);
                dist_gain *= 1.0 - 0.5 * occ;
                lpf_cutoff = lpf_cutoff.min(22_000.0 - occ * 20_800.0);
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
        output_params.reverb_send = 0.1;
    }

    fn update_environment_acoustics(
        &mut self,
        _listener: &ListenerState,
        _raycaster: Option<&dyn AcousticRaycaster>,
        reverb_params: &mut ReverbEnvironmentParams,
    ) {
        // Standard spatializer preserves static room presets
        reverb_params.room_size = 0.5;
        reverb_params.damping = 0.5;
        reverb_params.wet_mix = 0.25;
    }
}
