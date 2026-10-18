//! Peak limiter dynamics processor with lookahead preventing digital clipping (AUD-12).

use crate::frame::AudioFrame;

/// Default lookahead window in milliseconds (ADR-051, AUD-12).
const DEFAULT_LOOKAHEAD_MS: f32 = 2.0;

/// Dynamic range peak limiter with a lookahead delay line and attack/release envelope detector.
#[derive(Debug, Clone, PartialEq)]
pub struct PeakLimiter {
    sample_rate: f32,
    threshold_linear: f32,
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
    delay_line: Vec<AudioFrame>,
    write_pos: usize,
}

impl PeakLimiter {
    /// Constructs a new peak limiter with -0.1 dBFS threshold, 2 ms lookahead, 1 ms attack,
    /// and 50 ms release.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let sample_rate = sample_rate.max(1.0);
        let lookahead_frames = (DEFAULT_LOOKAHEAD_MS * 0.001 * sample_rate)
            .round()
            .max(1.0) as usize;
        let mut limiter = Self {
            sample_rate,
            threshold_linear: 10.0f32.powf(-0.1 / 20.0),
            envelope: 0.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            delay_line: vec![AudioFrame::SILENT; lookahead_frames],
            write_pos: 0,
        };
        limiter.set_attack_ms(1.0);
        limiter.set_release_ms(50.0);
        limiter
    }

    /// Returns the lookahead delay length in audio frames.
    #[must_use]
    pub fn lookahead_frames(&self) -> usize {
        self.delay_line.len()
    }

    /// Sets the lookahead window in milliseconds and resets the delay line.
    pub fn set_lookahead_ms(&mut self, lookahead_ms: f32) {
        let frames = (lookahead_ms.max(0.0) * 0.001 * self.sample_rate).round() as usize;
        self.delay_line.clear();
        self.delay_line.resize(frames, AudioFrame::SILENT);
        self.write_pos = 0;
    }

    /// Sets the peak limiting threshold in decibels relative to full scale (dBFS).
    pub fn set_threshold_db(&mut self, threshold_db: f32) {
        self.threshold_linear = 10.0f32.powf(threshold_db / 20.0);
    }

    /// Sets the attack time constant in milliseconds.
    pub fn set_attack_ms(&mut self, attack_ms: f32) {
        let tau = (attack_ms.max(0.1) * 0.001) * self.sample_rate;
        self.attack_coeff = 1.0 - (-1.0 / tau).exp();
    }

    /// Sets the release time constant in milliseconds.
    pub fn set_release_ms(&mut self, release_ms: f32) {
        let tau = (release_ms.max(1.0) * 0.001) * self.sample_rate;
        self.release_coeff = 1.0 - (-1.0 / tau).exp();
    }

    /// Resets the internal envelope follower and lookahead delay line.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.delay_line.fill(AudioFrame::SILENT);
        self.write_pos = 0;
    }

    /// Processes an audio frame slice in-place with lookahead, preventing attack overshoot and
    /// digital clipping without heap allocation.
    pub fn process(&mut self, frames: &mut [AudioFrame]) {
        let len = self.delay_line.len();
        if len == 0 {
            for frame in frames.iter_mut() {
                let peak = frame.left.abs().max(frame.right.abs());
                if peak > self.envelope {
                    self.envelope = peak;
                } else {
                    self.envelope += self.release_coeff * (peak - self.envelope);
                }
                let gain = if self.envelope > self.threshold_linear {
                    self.threshold_linear / self.envelope
                } else {
                    1.0
                };
                frame.left *= gain;
                frame.right *= gain;
                *frame = frame.clamp(-1.0, 1.0);
            }
            return;
        }

        for frame in frames.iter_mut() {
            // Find the steepest linear slope required to reach every peak in the lookahead
            // delay line by the frame it exits (`d = 1` for the sample exiting now, up to
            // `d = len` for the newest sample currently in the delay line).
            let mut max_step = 0.0f32;
            for offset in 0..len {
                let idx = (self.write_pos + offset) % len;
                let upcoming = self.delay_line[idx];
                let peak = upcoming.left.abs().max(upcoming.right.abs());
                if peak > self.envelope {
                    let step = (peak - self.envelope) / (offset + 1) as f32;
                    if step > max_step {
                        max_step = step;
                    }
                }
            }

            let delayed = self.delay_line[self.write_pos];
            self.delay_line[self.write_pos] = *frame;
            self.write_pos = (self.write_pos + 1) % len;

            let delayed_peak = delayed.left.abs().max(delayed.right.abs());
            if max_step > 0.0 {
                self.envelope += max_step;
            } else {
                self.envelope += self.release_coeff * (delayed_peak - self.envelope);
            }

            let gain = if self.envelope > self.threshold_linear {
                self.threshold_linear / self.envelope
            } else {
                1.0
            };

            frame.left = delayed.left * gain;
            frame.right = delayed.right * gain;

            // Final safety clamp for floating-point rounding at 0 dBFS.
            *frame = frame.clamp(-1.0, 1.0);
        }
    }
}
