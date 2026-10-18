//! Schroeder backward integration, Energy Decay Curves (EDC), and Sabine/Eyring validation.

/// Calculates room reverberation time $RT_{60}$ in seconds using the Sabine formula.
///
/// Valid for live, diffuse acoustic spaces with low average absorption ($\bar{\alpha} < 0.2$):
/// $$RT_{60} \approx \frac{0.161 \cdot V}{S \cdot \bar{\alpha}}$$
#[must_use]
pub fn sabine_rt60(volume: f32, surface_area: f32, avg_absorption: f32) -> f32 {
    let absorption_clamped = avg_absorption.clamp(0.001, 1.0);
    (0.161 * volume) / (surface_area * absorption_clamped)
}

/// Calculates room reverberation time $RT_{60}$ in seconds using the Eyring-Norris formula.
///
/// Accurately models rooms across all absorption ranges, including dead/anechoic spaces ($\bar{\alpha} \ge 0.2$):
/// $$RT_{60} \approx \frac{0.161 \cdot V}{-S \cdot \ln(1.0 - \bar{\alpha})}$$
#[must_use]
pub fn eyring_rt60(volume: f32, surface_area: f32, avg_absorption: f32) -> f32 {
    let alpha_clamped = avg_absorption.clamp(0.001, 0.999);
    let ln_term = -(1.0 - alpha_clamped).ln();
    (0.161 * volume) / (surface_area * ln_term)
}

/// Temporal histogram accumulating received acoustic ray energy for Schroeder decay curve analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyHistogram {
    /// Duration of each discrete histogram bin in seconds (e.g. 0.01s = 10ms).
    pub bin_duration_sec: f32,
    /// Total number of temporal bins.
    pub bin_count: usize,
    /// Accumulated energy per bin across 3 octave frequency bands: [Low, Mid, High].
    pub bins: Vec<[f32; 3]>,
}

impl EnergyHistogram {
    /// Creates a new energy histogram with the given bin duration and total bin count.
    #[must_use]
    pub fn new(bin_duration_sec: f32, bin_count: usize) -> Self {
        Self {
            bin_duration_sec,
            bin_count,
            bins: vec![[0.0; 3]; bin_count],
        }
    }

    /// Clears all accumulated energy in the histogram bins.
    pub fn clear(&mut self) {
        self.bins.fill([0.0; 3]);
    }

    /// Accumulates received energy into the appropriate temporal bin.
    pub fn add_energy(&mut self, time_sec: f32, energy: [f32; 3]) {
        if time_sec < 0.0 {
            return;
        }

        let bin_idx = (time_sec / self.bin_duration_sec) as usize;
        if bin_idx < self.bin_count {
            self.bins[bin_idx][0] += energy[0];
            self.bins[bin_idx][1] += energy[1];
            self.bins[bin_idx][2] += energy[2];
        }
    }

    /// Computes the Schroeder Energy Decay Curve (EDC) via backward temporal integration:
    /// $$\text{EDC}(t, b) = \int_t^\infty \text{ETC}(\tau, b) \, d\tau$$
    #[must_use]
    pub fn schroeder_curve(&self) -> Vec<[f32; 3]> {
        let mut edc = vec![[0.0; 3]; self.bin_count];
        let mut accum = [0.0; 3];

        for i in (0..self.bin_count).rev() {
            accum[0] += self.bins[i][0];
            accum[1] += self.bins[i][1];
            accum[2] += self.bins[i][2];
            edc[i] = accum;
        }

        edc
    }

    /// Extracts the reverberation time $RT_{60}$ in seconds across all 3 frequency bands
    /// using standard $T_{30}$ decay slope linear regression ($RT_{60} = 2 \times T_{30}$).
    #[must_use]
    pub fn extract_rt60(&self) -> [f32; 3] {
        let edc = self.schroeder_curve();
        let mut rt60 = [0.0f32; 3];

        for band in 0..3 {
            let e0 = edc.first().map_or(0.0, |b| b[band]);
            if e0 <= 1e-12 {
                rt60[band] = 0.0;
                continue;
            }

            // -5 dB threshold = e0 * 10^(-0.5) ≈ 0.316227 * e0
            let target_5db = e0 * 0.316_227_76;
            // -35 dB threshold = e0 * 10^(-3.5) ≈ 0.000316227 * e0
            let target_35db = e0 * 0.000_316_227_76;

            let t_5db = self.find_time_at_energy(&edc, band, target_5db);
            let t_35db = self.find_time_at_energy(&edc, band, target_35db);

            if let (Some(t5), Some(t35)) = (t_5db, t_35db) {
                if t35 > t5 {
                    let t30 = t35 - t5;
                    rt60[band] = t30 * 2.0;
                    continue;
                }
            }

            // If -35 dB is not reached, fall back to linear decay slope from t5 to minimum recorded energy
            if let Some(t5) = t_5db {
                let last_idx = self.bin_count.saturating_sub(1);
                let e_last = edc[last_idx][band].max(1e-12);
                let t_last = last_idx as f32 * self.bin_duration_sec;

                if t_last > t5 && e_last < target_5db {
                    let delta_db = 10.0 * (e_last / target_5db).log10();
                    let delta_t = t_last - t5;
                    let decay_rate_db_per_sec = (delta_db / delta_t).abs().max(1e-4);
                    rt60[band] = 60.0 / decay_rate_db_per_sec;
                    continue;
                }
            }

            // Fallback default for flat/empty signal
            rt60[band] = 0.5;
        }

        rt60
    }

    /// Finds the interpolated time in seconds where the EDC crosses the target energy level.
    fn find_time_at_energy(
        &self,
        edc: &[[f32; 3]],
        band: usize,
        target_energy: f32,
    ) -> Option<f32> {
        for (i, win) in edc.windows(2).enumerate() {
            let e1 = win[0][band];
            let e2 = win[1][band];

            if e1 >= target_energy && e2 <= target_energy {
                let delta = e1 - e2;
                let fraction = if delta.abs() > 1e-12 {
                    (e1 - target_energy) / delta
                } else {
                    0.0
                };
                let t = (i as f32 + fraction) * self.bin_duration_sec;
                return Some(t);
            }
        }

        None
    }
}
