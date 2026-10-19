//! Numerical value range, slider, and progress bar widgets.

use korppi_ecs::Component;

/// Base numerical value range container with step snapping and normalized ratio calculations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    /// Minimum allowed value.
    pub min_value: f64,
    /// Maximum allowed value.
    pub max_value: f64,
    /// Step increment for snapping (0.0 for continuous).
    pub step: f64,
    /// Current numerical value.
    value: f64,
}

impl Default for Range {
    fn default() -> Self {
        Self {
            min_value: 0.0,
            max_value: 100.0,
            step: 1.0,
            value: 0.0,
        }
    }
}

impl Component for Range {}

impl Range {
    fn sanitize_bounds(&mut self) {
        if !self.min_value.is_finite() {
            self.min_value = 0.0;
        }
        if !self.max_value.is_finite() {
            self.max_value = self.min_value.max(100.0);
        }
        if self.min_value > self.max_value {
            std::mem::swap(&mut self.min_value, &mut self.max_value);
        }
        if !self.step.is_finite() || self.step < 0.0 {
            self.step = 0.0;
        }
    }

    /// Creates a new `Range` with given min, max, and step snapping.
    pub fn new(min_value: f64, max_value: f64, step: f64) -> Self {
        let mut r = Self {
            min_value,
            max_value,
            step,
            value: 0.0,
        };
        r.sanitize_bounds();
        let initial = r.min_value;
        r.set_value(initial);
        r
    }

    /// Returns current numerical value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Sets numerical value, applying step snapping and clamping within `[min_value, max_value]`.
    pub fn set_value(&mut self, val: f64) {
        self.sanitize_bounds();
        let val = if val.is_finite() { val } else { self.min_value };
        let clamped = val.clamp(self.min_value, self.max_value);
        if self.step > 0.0 {
            let steps = ((clamped - self.min_value) / self.step).round();
            self.value = (self.min_value + steps * self.step).clamp(self.min_value, self.max_value);
        } else {
            self.value = clamped;
        }
    }

    /// Returns normalized ratio in $[0.0, 1.0]$.
    pub fn ratio(&self) -> f64 {
        let min = if self.min_value.is_finite() {
            self.min_value
        } else {
            0.0
        };
        let max = if self.max_value.is_finite() {
            self.max_value
        } else {
            min
        };
        let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
        let span = hi - lo;
        if span.abs() < 1e-9 || !self.value.is_finite() {
            0.0
        } else {
            ((self.value - lo) / span).clamp(0.0, 1.0)
        }
    }

    /// Sets value by normalized ratio $[0.0, 1.0]$.
    pub fn set_ratio(&mut self, ratio: f64) {
        self.sanitize_bounds();
        let ratio = if ratio.is_finite() { ratio } else { 0.0 };
        let clamped_ratio = ratio.clamp(0.0, 1.0);
        let span = self.max_value - self.min_value;
        self.set_value(self.min_value + clamped_ratio * span);
    }
}

/// Slider direction orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SliderOrientation {
    /// Horizontal track slider (left-to-right).
    #[default]
    Horizontal,
    /// Vertical track slider (bottom-to-top).
    Vertical,
}

/// Slider widget component allowing continuous or stepped value adjustment.
#[derive(Clone, Debug, PartialEq)]
pub struct Slider {
    /// Underlying numerical range.
    pub range: Range,
    /// Slider track orientation.
    pub orientation: SliderOrientation,
    /// Whether the slider can be adjusted.
    pub editable: bool,
    /// Whether mouse wheel can adjust the slider.
    pub scrollable: bool,
}

impl Component for Slider {}

impl Slider {
    /// Creates a new slider with specified orientation and range bounds.
    pub fn new(orientation: SliderOrientation, min: f64, max: f64, step: f64) -> Self {
        Self {
            range: Range::new(min, max, step),
            orientation,
            editable: true,
            scrollable: true,
        }
    }

    /// Updates value given an offset in pixels along the slider length.
    pub fn set_value_from_offset(&mut self, offset: f32, total_length: f32) {
        if !self.editable || total_length <= 0.0 {
            return;
        }
        let ratio = (offset / total_length) as f64;
        self.range.set_ratio(ratio);
    }
}

/// Progress bar visual fill direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ProgressFillMode {
    /// Fills horizontally from left to right.
    #[default]
    LeftToRight,
    /// Fills horizontally from right to left.
    RightToLeft,
    /// Fills vertically from bottom to top.
    BottomToTop,
    /// Fills vertically from top to bottom.
    TopToBottom,
}

/// Progress indicator widget component.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgressBar {
    /// Underlying numerical range.
    pub range: Range,
    /// Visual fill direction mode.
    pub fill_mode: ProgressFillMode,
    /// Whether to render textual percentage string readout.
    pub show_percentage: bool,
}

impl Component for ProgressBar {}

impl ProgressBar {
    /// Creates a new progress bar with bounds `[min, max]`.
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            range: Range::new(min, max, 0.0),
            fill_mode: ProgressFillMode::LeftToRight,
            show_percentage: true,
        }
    }

    /// Returns current numerical value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.range.value()
    }

    /// Sets progress value.
    #[inline]
    pub fn set_value(&mut self, val: f64) {
        self.range.set_value(val);
    }

    /// Returns normalized fill ratio $[0.0, 1.0]$.
    #[inline]
    pub fn ratio(&self) -> f64 {
        self.range.ratio()
    }

    /// Returns formatted percentage string (e.g. `"75%"`).
    pub fn percentage_string(&self) -> String {
        let pct = (self.ratio() * 100.0).round() as i64;
        format!("{}%", pct)
    }
}
