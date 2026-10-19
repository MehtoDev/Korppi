//! Parametric Feedback Delay Network (FDN) with $O(N)$ Householder matrix and $RT_{60}$ calibration.

/// Computes the $O(N)$ Householder reflection matrix multiplication for $N = 8$:
/// $$\mathbf{A} = \mathbf{I} - \frac{2}{8} \mathbf{1} \mathbf{1}^T = \mathbf{I} - \frac{1}{4} \mathbf{1} \mathbf{1}^T$$
///
/// This matrix is provably unitary ($\mathbf{A}^T \mathbf{A} = \mathbf{I}$), lossless,
/// and provides maximum diffusion across all 8 delay channels in just 8 subtractions and 1 sum.
#[must_use]
pub fn householder_diffuse_8(u: [f32; 8]) -> [f32; 8] {
    let sum: f32 = u[0] + u[1] + u[2] + u[3] + u[4] + u[5] + u[6] + u[7];
    let mu = sum * 0.25;

    [
        u[0] - mu,
        u[1] - mu,
        u[2] - mu,
        u[3] - mu,
        u[4] - mu,
        u[5] - mu,
        u[6] - mu,
        u[7] - mu,
    ]
}

/// Configuration descriptor for the 8-line Feedback Delay Network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdnConfig {
    /// Mutually prime delay line lengths in samples to eliminate metallic flutter echoes.
    pub delay_lengths: [usize; 8],
    /// Default dry signal pass-through mix (0.0 to 1.0).
    pub dry_mix: f32,
    /// Default reverberant wet signal mix (0.0 to 1.0).
    pub wet_mix: f32,
}

impl Default for FdnConfig {
    fn default() -> Self {
        Self {
            // Empirically selected mutually prime numbers for 48 kHz
            delay_lengths: [1013, 1279, 1583, 1973, 2411, 2963, 3593, 4397],
            dry_mix: 0.0,
            wet_mix: 1.0,
        }
    }
}

/// The centre frequencies, in Hz, of the three bands a [`FeedbackDelayNetwork`] takes its decay
/// times for: low, mid and high (the raytracer's material bands).
pub const RT60_BANDS: [f32; 3] = [250.0, 1000.0, 4000.0];

/// The corners, in Hz, of the low and high shelves of each line's absorption filter: halfway, in
/// octaves, between the bands.
const SHELF_CORNERS: [f32; 2] = [500.0, 2000.0];

/// One line's absorption: a gain and first-order low and high shelves, whose response at each of
/// [`RT60_BANDS`] is the attenuation the line's delay needs for that band's decay time.
#[derive(Debug, Clone, Copy, Default)]
struct Absorption {
    gain: f32,
    /// Low shelf: `b0`, `b1`, `a1` (normalized) and its state.
    low: [f32; 3],
    low_state: [f32; 2],
    /// High shelf, likewise.
    high: [f32; 3],
    high_state: [f32; 2],
}

/// A first-order shelf's coefficients (`b0`, `b1`, `a1`) for a corner at `corner` Hz: gain `g` below
/// it (`low`) or above it, and 1 on the other side; by the bilinear transform with prewarping.
fn shelf(g: f32, corner: f32, sample_rate: f32, low: bool) -> [f32; 3] {
    let sample_rate = if sample_rate.is_finite() {
        sample_rate.max(100.0)
    } else {
        48_000.0
    };
    let corner = corner.clamp(10.0, sample_rate * 0.45);
    let g = if g.is_finite() {
        g.clamp(1e-3, 1e3)
    } else {
        1.0
    };
    let k = (std::f32::consts::PI * corner / sample_rate).tan();
    let (b0, b1, a0, a1) = if low {
        (k * g + 1.0, k * g - 1.0, k + 1.0, k - 1.0)
    } else {
        (g + k, k - g, 1.0 + k, k - 1.0)
    };
    [b0 / a0, b1 / a0, a1 / a0]
}

/// The magnitude of a first-order filter `[b0, b1, a1]` at `hz`.
fn magnitude(coefficients: [f32; 3], hz: f32, sample_rate: f32) -> f32 {
    let w = (2.0 * std::f32::consts::PI * hz / sample_rate).clamp(0.0, std::f32::consts::PI);
    let (c, s) = (w.cos(), w.sin());
    let [b0, b1, a1] = coefficients;
    let numerator = ((b0 + b1 * c).powi(2) + (b1 * s).powi(2)).sqrt();
    let denominator = ((1.0 + a1 * c).powi(2) + (a1 * s).powi(2))
        .sqrt()
        .max(1e-12);
    numerator / denominator
}

impl Absorption {
    /// The absorption of a line of `length` frames for decay times `rt60` in [`RT60_BANDS`].
    ///
    /// A pass through the line must lose `60 dB × length / (rate × RT60)` in each band. The
    /// shelves overlap the bands a little, so their gains are found by a few fixed-point steps on
    /// the exact digital response, each moving a gain by its band's error.
    fn new(length: usize, rt60: [f32; 3], sample_rate: f32) -> Self {
        let sample_rate = if sample_rate.is_finite() {
            sample_rate.max(100.0)
        } else {
            48_000.0
        };
        let seconds = length as f32 / sample_rate;
        let target = rt60.map(|t| {
            let t = if t.is_finite() {
                t.clamp(0.05, 30.0)
            } else {
                1.0
            };
            -60.0 * seconds / t
        });
        let db = |gain: f32| 20.0 * gain.max(1e-12).log10();
        let amp = |db: f32| 10f32.powf(db.clamp(-120.0, 24.0) / 20.0);
        // Gains in dB: overall (mid), low shelf, high shelf.
        let (mut mid, mut low, mut high) = if SHELF_CORNERS[1] < sample_rate * 0.45 {
            (
                target[1].clamp(-120.0, 0.0),
                (target[0] - target[1]).clamp(-18.0, 18.0),
                (target[2] - target[1]).clamp(-18.0, 18.0),
            )
        } else {
            (target[1].clamp(-120.0, 0.0), 0.0, 0.0)
        };
        if SHELF_CORNERS[1] < sample_rate * 0.45 {
            for _ in 0..8 {
                let low_shelf = shelf(amp(low), SHELF_CORNERS[0], sample_rate, true);
                let high_shelf = shelf(amp(high), SHELF_CORNERS[1], sample_rate, false);
                let response = RT60_BANDS.map(|hz| {
                    mid + db(magnitude(low_shelf, hz, sample_rate))
                        + db(magnitude(high_shelf, hz, sample_rate))
                });
                mid = (mid + target[1] - response[1]).clamp(-120.0, 0.0);
                low = (low + target[0] - response[0]).clamp(-18.0, 18.0);
                high = (high + target[2] - response[2]).clamp(-18.0, 18.0);
            }
        }
        let low_shelf = shelf(amp(low), SHELF_CORNERS[0], sample_rate, true);
        let high_shelf = shelf(amp(high), SHELF_CORNERS[1], sample_rate, false);
        let check_freqs = [
            0.0,
            RT60_BANDS[0],
            SHELF_CORNERS[0],
            RT60_BANDS[1],
            SHELF_CORNERS[1],
            RT60_BANDS[2],
            sample_rate * 0.5,
        ];
        let peak_filter_mag = check_freqs
            .into_iter()
            .map(|hz| {
                magnitude(low_shelf, hz, sample_rate) * magnitude(high_shelf, hz, sample_rate)
            })
            .fold(1e-6f32, f32::max);
        let max_gain = (0.9995 / peak_filter_mag).min(0.9995);
        Self {
            gain: amp(mid).clamp(0.0, max_gain),
            low: low_shelf,
            low_state: [0.0; 2],
            high: high_shelf,
            high_state: [0.0; 2],
        }
    }

    /// Takes the line's new absorption, keeping its filter states.
    fn retune(&mut self, other: Self) {
        self.gain = other.gain;
        self.low = other.low;
        self.high = other.high;
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let [b0, b1, a1] = self.low;
        let [x1, y1] = self.low_state;
        let low = b0 * input + b1 * x1 - a1 * y1;
        self.low_state = [input, low];
        let [b0, b1, a1] = self.high;
        let [x1, y1] = self.high_state;
        let high = b0 * low + b1 * x1 - a1 * y1;
        self.high_state = [low, high];
        high * self.gain
    }
}

/// 8-channel Feedback Delay Network reverberator parameterized by physical raytraced $RT_{60}$.
///
/// Eight delay lines of mutually prime lengths feed back through the Householder matrix; each
/// line's [`Absorption`] makes every band of [`RT60_BANDS`] decay at its own RT60 (Jot's method;
/// FIX-7.5 replaced a calibration from the mid band alone). Processing allocates nothing.
#[derive(Debug, Clone)]
pub struct FeedbackDelayNetwork {
    config: FdnConfig,
    buffers: [Vec<f32>; 8],
    write_cursors: [usize; 8],
    absorption: [Absorption; 8],
}

impl FeedbackDelayNetwork {
    /// Creates a network of `config`'s delay lines for `sample_rate`, decaying in 1.5, 1.5 and
    /// 1.2 s.
    #[must_use]
    pub fn new(config: FdnConfig, sample_rate: f32) -> Self {
        let buffers = config.delay_lengths.map(|length| vec![0.0; length.max(1)]);
        let mut fdn = Self {
            config,
            buffers,
            write_cursors: [0; 8],
            absorption: [Absorption::default(); 8],
        };
        fdn.set_rt60([1.5, 1.5, 1.2], sample_rate);
        fdn
    }

    /// Sets the decay times (seconds to fall 60 dB) of the low, mid and high bands
    /// ([`RT60_BANDS`]), keeping the network's state.
    pub fn set_rt60(&mut self, rt60: [f32; 3], sample_rate: f32) {
        for (absorption, &length) in self.absorption.iter_mut().zip(&self.config.delay_lengths) {
            absorption.retune(Absorption::new(length, rt60, sample_rate));
        }
    }

    /// Processes one input sample and returns the left and right reverberated output.
    pub fn process_sample(&mut self, input: f32) -> (f32, f32) {
        let mut u = [0.0f32; 8];
        for (i, slot) in u.iter_mut().enumerate() {
            *slot = self.buffers[i][self.write_cursors[i]];
        }

        let out_l = (u[0] - u[2] + u[4] - u[6]) * 0.5;
        let out_r = (u[1] - u[3] + u[5] - u[7]) * 0.5;

        let mut y = [0.0f32; 8];
        for i in 0..8 {
            y[i] = self.absorption[i].process(u[i]);
        }
        let v = householder_diffuse_8(y);

        const INPUT_SCALE: f32 = 0.353_553_38; // 1 / sqrt(8)
        for (i, &v_val) in v.iter().enumerate() {
            let cursor = self.write_cursors[i];
            self.buffers[i][cursor] = input * INPUT_SCALE + v_val;
            self.write_cursors[i] = (cursor + 1) % self.buffers[i].len();
        }

        (out_l, out_r)
    }

    /// Processes a contiguous block of audio samples into stereo buffers.
    pub fn process_block(&mut self, input: &[f32], output_l: &mut [f32], output_r: &mut [f32]) {
        let count = input.len().min(output_l.len()).min(output_r.len());
        for i in 0..count {
            let (l, r) = self.process_sample(input[i]);
            output_l[i] = l;
            output_r[i] = r;
        }
    }

    /// Clears internal delay buffers and filter memories.
    pub fn reset(&mut self) {
        for b in &mut self.buffers {
            b.fill(0.0);
        }
        self.write_cursors = [0; 8];
        for absorption in &mut self.absorption {
            absorption.low_state = [0.0; 2];
            absorption.high_state = [0.0; 2];
        }
    }
}
