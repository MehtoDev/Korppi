//! Change ticks: when a component was added or last changed, relative to query runs.
//!
//! The world counts runs with a wrapping `u32` change tick. Each run of a [`QueryState`] (or,
//! later, of a system) takes the current tick as its `this_run` and advances the world's tick;
//! changes made afterwards carry a newer tick. A component counts as added or changed for a run
//! when its tick is newer than the run's `last_run` (the `this_run` of the previous run).
//!
//! Comparisons are made relative to `this_run`, so they survive the counter wrapping around as
//! long as no tick is older than [`MAX_CHANGE_AGE`]: the world clamps older ticks every
//! [`CHECK_TICK_THRESHOLD`] ticks, and a query state clamps its `last_run`. Both values are
//! Bevy's.
//!
//! [`QueryState`]: crate::QueryState

/// How far the world's tick advances between two clamping passes.
pub const CHECK_TICK_THRESHOLD: u32 = 518_400_000;

/// The oldest age a tick keeps; older ticks are clamped to this age.
pub const MAX_CHANGE_AGE: u32 = u32::MAX - (2 * CHECK_TICK_THRESHOLD - 1);

/// A point in the world's change history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tick(u32);

impl Tick {
    /// Creates a tick with the raw value `value`.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Returns `true` if this tick is newer than `last_run`, as seen from `this_run`.
    #[inline]
    #[must_use]
    pub const fn is_newer_than(self, last_run: Self, this_run: Self) -> bool {
        let since_change = this_run.0.wrapping_sub(self.0);
        let since_run = this_run.0.wrapping_sub(last_run.0);
        since_run > since_change
    }

    /// Clamps this tick to at most [`MAX_CHANGE_AGE`] before `now`.
    #[inline]
    pub fn clamp_age(&mut self, now: Self) {
        if now.0.wrapping_sub(self.0) > MAX_CHANGE_AGE {
            self.0 = now.0.wrapping_sub(MAX_CHANGE_AGE);
        }
    }
}

/// When one component value was added and last changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComponentTicks {
    /// The tick the value was added at.
    pub added: Tick,
    /// The tick the value was last changed at through its own row (the column can also record a
    /// newer change of every row at once).
    pub changed: Tick,
}

impl ComponentTicks {
    /// Ticks of a value added (and so changed) at `tick`.
    #[must_use]
    pub const fn new(tick: Tick) -> Self {
        Self {
            added: tick,
            changed: tick,
        }
    }
}

/// The ticks a query run compares with: its previous run and its current one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChangeTicks {
    /// `this_run` of the previous run; changes after it are reported.
    pub last_run: Tick,
    /// The tick of the current run; mutable access marks rows with it.
    pub this_run: Tick,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_than_compares_across_wrap_around() {
        let last = Tick::new(u32::MAX - 1);
        let now = Tick::new(3);
        assert!(Tick::new(1).is_newer_than(last, now));
        assert!(!Tick::new(u32::MAX - 5).is_newer_than(last, now));
        assert!(!last.is_newer_than(last, now));
    }

    #[test]
    fn clamp_age_keeps_recent_ticks() {
        let now = Tick::new(10);
        let mut recent = Tick::new(5);
        recent.clamp_age(now);
        assert_eq!(recent, Tick::new(5));
        let mut old = Tick::new(now.get().wrapping_sub(MAX_CHANGE_AGE + 7));
        old.clamp_age(now);
        assert_eq!(old, Tick::new(now.get().wrapping_sub(MAX_CHANGE_AGE)));
    }
}
