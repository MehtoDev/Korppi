//! Fixed-timestep clock for deterministic simulation updates and variable frame interpolation.

use std::time::Duration;

/// Manages deterministic simulation ticks decoupled from variable frame rates.
///
/// Converts elapsed real/wall time into discrete fixed-duration simulation steps.
/// Tracks interpolation factor (`alpha`) for smooth visual rendering between ticks.
#[derive(Debug, Clone)]
pub struct FixedClock {
    tick_rate: u32,
    timestep: Duration,
    accumulator: Duration,
    max_sub_steps: u32,
    current_tick: u64,
}

impl FixedClock {
    /// Creates a new `FixedClock` with the target tick rate (e.g., 60 for 60 Hz).
    ///
    /// # Panics
    /// Panics if `tick_rate` is `0` or exceeds `1_000_000_000` Hz (1 ns timestep).
    #[must_use]
    pub fn new(tick_rate: u32) -> Self {
        assert!(tick_rate > 0, "tick_rate must be greater than zero");
        assert!(
            tick_rate <= 1_000_000_000,
            "tick_rate ({tick_rate}) must not exceed 1_000_000_000 Hz (1 ns minimum timestep)"
        );
        let nanos_per_tick = 1_000_000_000 / u64::from(tick_rate);
        let timestep = Duration::from_nanos(nanos_per_tick);

        Self {
            tick_rate,
            timestep,
            accumulator: Duration::ZERO,
            max_sub_steps: 8,
            current_tick: 0,
        }
    }

    /// Returns the target tick rate in Hertz (ticks per second).
    #[must_use]
    pub const fn tick_rate(&self) -> u32 {
        self.tick_rate
    }

    /// Returns the exact duration of a single simulation tick.
    #[must_use]
    pub const fn timestep(&self) -> Duration {
        self.timestep
    }

    /// Returns the total number of fixed ticks completed since creation.
    #[must_use]
    pub const fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Sets the maximum allowable sub-steps per frame to prevent simulation death spirals.
    pub fn set_max_sub_steps(&mut self, max: u32) {
        assert!(max > 0, "max_sub_steps must be greater than zero");
        self.max_sub_steps = max;
    }

    /// Advances the clock accumulator by the elapsed delta time.
    /// Clamps accumulator if it exceeds `max_sub_steps * timestep`.
    pub fn advance(&mut self, delta: Duration) {
        self.accumulator = self.accumulator.saturating_add(delta);
        let max_accum = self.timestep.saturating_mul(self.max_sub_steps);
        if self.accumulator > max_accum {
            self.accumulator = max_accum;
        }
    }

    /// Returns `true` if enough time has accumulated to execute a simulation tick.
    #[must_use]
    pub fn should_tick(&self) -> bool {
        self.accumulator >= self.timestep
    }

    /// Consumes one fixed timestep from the accumulator and increments the tick counter.
    ///
    /// # Panics
    /// Panics if called when `should_tick()` is false.
    pub fn tick(&mut self) {
        assert!(
            self.should_tick(),
            "Cannot tick when accumulator is less than timestep"
        );
        self.accumulator -= self.timestep;
        self.current_tick += 1;
    }

    /// Returns the normalized fraction `[0.0, 1.0)` of time remaining in the accumulator.
    /// Used for rendering interpolation between the previous and current physics states.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        const MAX_ALPHA_BELOW_ONE: f32 = f32::from_bits(1.0f32.to_bits() - 1);
        let accum_secs = self.accumulator.as_secs_f64();
        let step_secs = self.timestep.as_secs_f64();
        let ratio = (accum_secs / step_secs).clamp(0.0, 1.0) as f32;
        ratio.min(MAX_ALPHA_BELOW_ONE)
    }
}
