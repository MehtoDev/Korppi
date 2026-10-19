//! Second-Order Infinite Impulse Response (Biquad IIR) filter implementation.
//!
//! Uses Direct Form II Transposed (DF2T) structure for optimal numerical stability
//! in single-precision floating point, with formulas based on Robert Bristow-Johnson's
//! Audio EQ Cookbook.

use crate::frame::AudioFrame;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Biquad filter characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiquadType {
    /// Low-pass filter (passes frequencies below cutoff).
    LowPass,
    /// High-pass filter (passes frequencies above cutoff).
    HighPass,
    /// Band-pass filter (passes frequencies in a narrow band around center).
    BandPass,
    /// Notch / Band-stop filter (attenuates frequencies around center).
    Notch,
    /// Peaking / Bell equalizer (boosts or cuts around center frequency).
    Peaking,
    /// Low-shelf equalizer (boosts or cuts frequencies below corner).
    LowShelf,
    /// High-shelf equalizer (boosts or cuts frequencies above corner).
    HighShelf,
}

/// Direct Form II Transposed Biquad IIR filter.
#[derive(Debug, Clone, PartialEq)]
pub struct BiquadFilter {
    filter_type: BiquadType,
    frequency: f32,
    q: f32,
    gain_db: f32,
    sample_rate: f32,

    // Normalized filter coefficients (a0 normalized to 1.0)
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,

    // DF2T state registers for stereo channels
    z1_left: f32,
    z2_left: f32,
    z1_right: f32,
    z2_right: f32,
}

impl BiquadFilter {
    /// Constructs a new biquad filter.
    #[must_use]
    pub fn new(filter_type: BiquadType, frequency: f32, q: f32, sample_rate: f32) -> Self {
        let mut filter = Self {
            filter_type,
            frequency,
            q: q.max(0.001),
            gain_db: 0.0,
            sample_rate,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1_left: 0.0,
            z2_left: 0.0,
            z1_right: 0.0,
            z2_right: 0.0,
        };
        filter.recalculate_coefficients();
        filter
    }

    /// Constructs a new peaking or shelving filter with specified gain in decibels.
    #[must_use]
    pub fn with_gain_db(
        filter_type: BiquadType,
        frequency: f32,
        q: f32,
        gain_db: f32,
        sample_rate: f32,
    ) -> Self {
        let mut filter = Self {
            filter_type,
            frequency,
            q: q.max(0.001),
            gain_db,
            sample_rate,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1_left: 0.0,
            z2_left: 0.0,
            z1_right: 0.0,
            z2_right: 0.0,
        };
        filter.recalculate_coefficients();
        filter
    }

    /// Updates the filter's cutoff frequency in Hz.
    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.clamp(10.0, self.sample_rate * 0.499);
        self.recalculate_coefficients();
    }

    /// Updates the quality factor Q.
    pub fn set_q(&mut self, q: f32) {
        self.q = q.max(0.001);
        self.recalculate_coefficients();
    }

    /// Updates the gain in decibels (for peaking and shelving filters).
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.gain_db = gain_db;
        self.recalculate_coefficients();
    }

    /// Resets the internal delay state registers to zero.
    pub fn reset(&mut self) {
        self.z1_left = 0.0;
        self.z2_left = 0.0;
        self.z1_right = 0.0;
        self.z2_right = 0.0;
    }

    /// Processes a single scalar mono sample.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let out = self.b0 * input + self.z1_left;
        self.z1_left = self.b1 * input - self.a1 * out + self.z2_left;
        self.z2_left = self.b2 * input - self.a2 * out;
        out
    }

    /// Processes a single stereo audio frame.
    #[inline]
    pub fn process_frame(&mut self, frame: AudioFrame) -> AudioFrame {
        let out_left = self.b0 * frame.left + self.z1_left;
        self.z1_left = self.b1 * frame.left - self.a1 * out_left + self.z2_left;
        self.z2_left = self.b2 * frame.left - self.a2 * out_left;

        let out_right = self.b0 * frame.right + self.z1_right;
        self.z1_right = self.b1 * frame.right - self.a1 * out_right + self.z2_right;
        self.z2_right = self.b2 * frame.right - self.a2 * out_right;

        AudioFrame::new(out_left, out_right)
    }

    /// Recalculates filter transfer function coefficients.
    fn recalculate_coefficients(&mut self) {
        let sample_rate = if self.sample_rate.is_finite() {
            self.sample_rate.max(25.0)
        } else {
            48_000.0
        };
        let nyquist = sample_rate * 0.5;
        let max_freq = (nyquist * 0.999).max(10.0);
        let clamped_freq = if self.frequency.is_finite() {
            self.frequency.clamp(10.0, max_freq)
        } else {
            1000.0f32.min(max_freq)
        };
        let q = if self.q.is_finite() {
            self.q.max(0.01)
        } else {
            0.707
        };
        let omega = 2.0 * PI * clamped_freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * q);
        let a_gain = 10.0f32.powf(self.gain_db / 40.0);

        let (b0, b1, b2, a0, a1, a2) = match self.filter_type {
            BiquadType::LowPass => {
                let b0 = (1.0 - cos_omega) * 0.5;
                let b1 = 1.0 - cos_omega;
                let b2 = (1.0 - cos_omega) * 0.5;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighPass => {
                let b0 = (1.0 + cos_omega) * 0.5;
                let b1 = -(1.0 + cos_omega);
                let b2 = (1.0 + cos_omega) * 0.5;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::BandPass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_omega;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::Peaking => {
                let b0 = 1.0 + alpha * a_gain;
                let b1 = -2.0 * cos_omega;
                let b2 = 1.0 - alpha * a_gain;
                let a0 = 1.0 + alpha / a_gain;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha / a_gain;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowShelf => {
                let sqrt_a = a_gain.sqrt();
                let b0 =
                    a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha);
                let b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * cos_omega);
                let b2 =
                    a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha);
                let a0 = (a_gain + 1.0) + (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha;
                let a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * cos_omega);
                let a2 = (a_gain + 1.0) + (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighShelf => {
                let sqrt_a = a_gain.sqrt();
                let b0 =
                    a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha);
                let b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * cos_omega);
                let b2 =
                    a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha);
                let a0 = (a_gain + 1.0) - (a_gain - 1.0) * cos_omega + 2.0 * sqrt_a * alpha;
                let a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * cos_omega);
                let a2 = (a_gain + 1.0) - (a_gain - 1.0) * cos_omega - 2.0 * sqrt_a * alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        let inv_a0 = 1.0 / a0;
        self.b0 = b0 * inv_a0;
        self.b1 = b1 * inv_a0;
        self.b2 = b2 * inv_a0;
        self.a1 = a1 * inv_a0;
        self.a2 = a2 * inv_a0;
    }
}
