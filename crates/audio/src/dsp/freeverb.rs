//! Freeverb / Schroeder algorithmic reverberator.
//!
//! Implements the classic Jezar / Schroeder reverberator combining 8 parallel
//! lowpass feedback comb filters (LBCF) and 4 series allpass filters (APF) per channel.

use crate::frame::AudioFrame;

const COMB_TUNING_L: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASS_TUNING_L: [usize; 4] = [556, 441, 341, 225];
const STEREO_SPREAD: usize = 23;

#[derive(Debug, Clone)]
struct CombFilter {
    buffer: Vec<f32>,
    buf_size: usize,
    index: usize,
    filter_store: f32,
    damp: f32,
    feedback: f32,
}

impl CombFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            buf_size: size,
            index: 0,
            filter_store: 0.0,
            damp: 0.2,
            feedback: 0.5,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.index];
        self.filter_store = output * (1.0 - self.damp) + self.filter_store * self.damp;
        self.buffer[self.index] = input + self.filter_store * self.feedback;

        self.index += 1;
        if self.index >= self.buf_size {
            self.index = 0;
        }

        output
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.filter_store = 0.0;
        self.index = 0;
    }
}

#[derive(Debug, Clone)]
struct AllpassFilter {
    buffer: Vec<f32>,
    buf_size: usize,
    index: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(size: usize) -> Self {
        Self {
            buffer: vec![0.0; size],
            buf_size: size,
            index: 0,
            feedback: 0.5,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let bufout = self.buffer[self.index];
        let output = -input + bufout;
        self.buffer[self.index] = input + bufout * self.feedback;

        self.index += 1;
        if self.index >= self.buf_size {
            self.index = 0;
        }

        output
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
    }
}

/// Algorithmic reverberator simulating environmental reflections.
#[derive(Debug, Clone)]
pub struct Freeverb {
    combs_l: Vec<CombFilter>,
    combs_r: Vec<CombFilter>,
    allpasses_l: Vec<AllpassFilter>,
    allpasses_r: Vec<AllpassFilter>,

    room_size: f32,
    damp: f32,
    wet: f32,
    dry: f32,
    width: f32,
}

impl Freeverb {
    /// Constructs a new Freeverb instance tuned for the given sample rate.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let scale = sample_rate / 44_100.0;

        let combs_l = COMB_TUNING_L
            .iter()
            .map(|&s| CombFilter::new((s as f32 * scale).round() as usize))
            .collect();

        let combs_r = COMB_TUNING_L
            .iter()
            .map(|&s| CombFilter::new(((s + STEREO_SPREAD) as f32 * scale).round() as usize))
            .collect();

        let allpasses_l = ALLPASS_TUNING_L
            .iter()
            .map(|&s| AllpassFilter::new((s as f32 * scale).round() as usize))
            .collect();

        let allpasses_r = ALLPASS_TUNING_L
            .iter()
            .map(|&s| AllpassFilter::new(((s + STEREO_SPREAD) as f32 * scale).round() as usize))
            .collect();

        let mut reverb = Self {
            combs_l,
            combs_r,
            allpasses_l,
            allpasses_r,
            room_size: 0.5,
            damp: 0.5,
            wet: 0.33,
            dry: 0.67,
            width: 1.0,
        };

        reverb.update_parameters();
        reverb
    }

    /// Sets the room size parameter (0.0 to 1.0).
    pub fn set_room_size(&mut self, room_size: f32) {
        self.room_size = room_size.clamp(0.0, 1.0);
        self.update_parameters();
    }

    /// Sets the high-frequency damping factor (0.0 to 1.0).
    pub fn set_damp(&mut self, damp: f32) {
        self.damp = damp.clamp(0.0, 1.0);
        self.update_parameters();
    }

    /// Sets the wet gain (reverberant signal volume).
    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet.max(0.0);
    }

    /// Sets the dry gain (unprocessed signal volume).
    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry.max(0.0);
    }

    /// Sets the stereo width factor (0.0 = mono, 1.0 = wide stereo).
    pub fn set_width(&mut self, width: f32) {
        self.width = width.clamp(0.0, 1.0);
    }

    /// Resets all internal delay buffers and filter states to silence.
    pub fn reset(&mut self) {
        for comb in &mut self.combs_l {
            comb.reset();
        }
        for comb in &mut self.combs_r {
            comb.reset();
        }
        for ap in &mut self.allpasses_l {
            ap.reset();
        }
        for ap in &mut self.allpasses_r {
            ap.reset();
        }
    }

    /// Reverberates one mono sample and returns the wet stereo output alone, scaled by the wet
    /// gain and spread by the width: the form a send bus uses.
    #[inline]
    pub fn process_mono(&mut self, input: f32) -> (f32, f32) {
        let input = input * 0.15;
        let mut out_l = 0.0f32;
        for comb in &mut self.combs_l {
            out_l += comb.process(input);
        }
        let mut out_r = 0.0f32;
        for comb in &mut self.combs_r {
            out_r += comb.process(input);
        }
        for ap in &mut self.allpasses_l {
            out_l = ap.process(out_l);
        }
        for ap in &mut self.allpasses_r {
            out_r = ap.process(out_r);
        }
        let wet1 = self.wet * (self.width * 0.5 + 0.5);
        let wet2 = self.wet * ((1.0 - self.width) * 0.5);
        (out_l * wet1 + out_r * wet2, out_r * wet1 + out_l * wet2)
    }

    /// Processes an audio frame slice in-place.
    pub fn process(&mut self, frames: &mut [AudioFrame]) {
        let wet1 = self.wet * (self.width * 0.5 + 0.5);
        let wet2 = self.wet * ((1.0 - self.width) * 0.5);
        let dry = self.dry;

        for frame in frames.iter_mut() {
            let input = (frame.left + frame.right) * 0.15;

            // Parallel comb filters
            let mut out_l = 0.0f32;
            for comb in &mut self.combs_l {
                out_l += comb.process(input);
            }

            let mut out_r = 0.0f32;
            for comb in &mut self.combs_r {
                out_r += comb.process(input);
            }

            // Series allpass filters
            for ap in &mut self.allpasses_l {
                out_l = ap.process(out_l);
            }
            for ap in &mut self.allpasses_r {
                out_r = ap.process(out_r);
            }

            // Wet/dry mix with stereo cross-coupling
            frame.left = out_l * wet1 + out_r * wet2 + frame.left * dry;
            frame.right = out_r * wet1 + out_l * wet2 + frame.right * dry;
        }
    }

    fn update_parameters(&mut self) {
        let feedback = self.room_size * 0.28 + 0.7;
        let damp = self.damp * 0.4;

        for comb in &mut self.combs_l {
            comb.feedback = feedback;
            comb.damp = damp;
        }
        for comb in &mut self.combs_r {
            comb.feedback = feedback;
            comb.damp = damp;
        }
    }
}
