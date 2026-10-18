//! Multi-tap circular delay line with fractional interpolation and raised-cosine tap cross-fading.

use std::f32::consts::FRAC_PI_2;

/// Discrete early reflection tap parameter descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarlyReflectionTap {
    /// Delay time in fractional samples.
    pub delay_samples: f32,
    /// Linear gain for the left stereo channel.
    pub gain_left: f32,
    /// Linear gain for the right stereo channel.
    pub gain_right: f32,
    /// High-frequency absorption damping factor (0.0 = bright, 1.0 = heavy absorption).
    pub damping: f32,
}

impl Default for EarlyReflectionTap {
    fn default() -> Self {
        Self {
            delay_samples: 0.0,
            gain_left: 0.0,
            gain_right: 0.0,
            damping: 0.0,
        }
    }
}

/// Internal runtime state for an active early reflection tap with crossfade support.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ActiveTapState {
    current_delay: f32,
    target_delay: f32,
    gain_left: f32,
    gain_right: f32,
    damping: f32,
    filter_state_l: f32,
    filter_state_r: f32,
    crossfade_progress: usize,
    crossfade_total: usize,
}

impl Default for ActiveTapState {
    fn default() -> Self {
        Self {
            current_delay: 0.0,
            target_delay: 0.0,
            gain_left: 0.0,
            gain_right: 0.0,
            damping: 0.0,
            filter_state_l: 0.0,
            filter_state_r: 0.0,
            crossfade_progress: 0,
            crossfade_total: 0,
        }
    }
}

/// Multi-tap circular delay line for rendering early reflection ray arrivals.
#[derive(Debug, Clone)]
pub struct TappedDelayLine {
    buffer: Vec<f32>,
    write_cursor: usize,
    capacity: usize,
    taps: Vec<ActiveTapState>,
    sample_rate: f32,
    crossfade_samples: usize,
}

impl TappedDelayLine {
    /// Creates a new tapped delay line with the specified maximum capacity in samples.
    #[must_use]
    pub fn new(capacity: usize, sample_rate: f32) -> Self {
        let cap = capacity.max(64);
        let crossfade_samples = ((sample_rate * 0.010).round() as usize).max(32); // ~10ms crossfade

        Self {
            buffer: vec![0.0; cap],
            write_cursor: 0,
            capacity: cap,
            taps: Vec::new(),
            sample_rate,
            crossfade_samples,
        }
    }

    /// Returns the sample rate configured for this delay line.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Updates the active reflection taps.
    ///
    /// If an existing tap's delay shifts by more than 2 samples, a smooth raised-cosine
    /// crossfade is initiated to eliminate clicks and zipper noise.
    pub fn set_taps(&mut self, new_taps: &[EarlyReflectionTap]) {
        self.taps
            .resize_with(new_taps.len(), ActiveTapState::default);

        for (slot, tap) in self.taps.iter_mut().zip(new_taps.iter()) {
            let clamped_delay = tap.delay_samples.clamp(0.0, (self.capacity - 2) as f32);

            if (slot.current_delay - clamped_delay).abs() > 2.0 && slot.current_delay > 0.0 {
                // Trigger crossfade from current_delay to clamped_delay
                slot.target_delay = clamped_delay;
                slot.crossfade_progress = 0;
                slot.crossfade_total = self.crossfade_samples;
            } else {
                slot.current_delay = clamped_delay;
                slot.target_delay = clamped_delay;
                slot.crossfade_total = 0;
            }

            slot.gain_left = tap.gain_left;
            slot.gain_right = tap.gain_right;
            slot.damping = tap.damping.clamp(0.0, 0.95);
        }
    }

    /// Pushes an input sample and computes the wet early reflections stereo output: `(left, right)`.
    pub fn process_sample(&mut self, input: f32) -> (f32, f32) {
        // Write incoming sample to circular delay buffer
        self.buffer[self.write_cursor] = input;

        let buffer = &self.buffer;
        let write_cursor = self.write_cursor;
        let capacity = self.capacity;

        let mut out_l = 0.0f32;
        let mut out_r = 0.0f32;

        for tap in &mut self.taps {
            if tap.gain_left.abs() < 1e-6 && tap.gain_right.abs() < 1e-6 {
                continue;
            }

            let sample_val =
                if tap.crossfade_total > 0 && tap.crossfade_progress < tap.crossfade_total {
                    // Raised-cosine crossfade:
                    // w_out = cos(pi/2 * progress / total)
                    // w_in  = sin(pi/2 * progress / total)
                    let frac = (tap.crossfade_progress as f32) / (tap.crossfade_total as f32);
                    let angle = FRAC_PI_2 * frac;
                    let w_out = angle.cos();
                    let w_in = angle.sin();

                    let s_old =
                        Self::read_interpolated(buffer, write_cursor, capacity, tap.current_delay);
                    let s_new =
                        Self::read_interpolated(buffer, write_cursor, capacity, tap.target_delay);

                    tap.crossfade_progress += 1;
                    if tap.crossfade_progress >= tap.crossfade_total {
                        tap.current_delay = tap.target_delay;
                        tap.crossfade_total = 0;
                    }

                    s_old * w_out + s_new * w_in
                } else {
                    Self::read_interpolated(buffer, write_cursor, capacity, tap.current_delay)
                };

            // Single-pole high-frequency damping: y[n] = (1 - d) x[n] + d y[n-1]
            let d = tap.damping;
            tap.filter_state_l = (1.0 - d) * sample_val + d * tap.filter_state_l;
            tap.filter_state_r = (1.0 - d) * sample_val + d * tap.filter_state_r;

            out_l += tap.filter_state_l * tap.gain_left;
            out_r += tap.filter_state_r * tap.gain_right;
        }

        // Advance write pointer
        self.write_cursor = (self.write_cursor + 1) % self.capacity;

        (out_l, out_r)
    }

    /// Reads a sample at fractional delay using linear interpolation.
    fn read_interpolated(
        buffer: &[f32],
        write_cursor: usize,
        capacity: usize,
        delay_samples: f32,
    ) -> f32 {
        let int_delay = delay_samples.floor() as usize;
        let frac = delay_samples - (int_delay as f32);

        let idx0 = if write_cursor >= int_delay {
            write_cursor - int_delay
        } else {
            capacity + write_cursor - int_delay
        };

        let idx1 = if idx0 == 0 { capacity - 1 } else { idx0 - 1 };

        let s0 = buffer[idx0];
        let s1 = buffer[idx1];

        s0 * (1.0 - frac) + s1 * frac
    }

    /// Clears the delay buffer and internal filter states.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_cursor = 0;
        for t in &mut self.taps {
            t.filter_state_l = 0.0;
            t.filter_state_r = 0.0;
            t.crossfade_total = 0;
        }
    }
}
