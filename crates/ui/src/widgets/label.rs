//! Label widget for text rendering, multi-line wrapping, and alignments.

use crate::widgets::alignment::{HorizontalAlignment, VerticalAlignment};
use korppi_ecs::Component;

/// Text label widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Label {
    /// Source raw text content.
    text: String,
    /// Cached split lines of the text.
    lines: Vec<String>,
    /// Horizontal text alignment.
    pub horizontal_alignment: HorizontalAlignment,
    /// Vertical text alignment.
    pub vertical_alignment: VerticalAlignment,
    /// Whether long lines automatically wrap to the next line.
    pub autowrap: bool,
    /// Whether to force all text to uppercase presentation.
    pub uppercase: bool,
    /// Number of visible characters to display (useful for typewriter dialog effects).
    pub visible_characters: Option<usize>,
}

impl Component for Label {}

impl Label {
    /// Creates a new `Label` with the given text.
    pub fn new(text: impl Into<String>) -> Self {
        let mut label = Self {
            text: String::with_capacity(128),
            lines: Vec::with_capacity(4),
            horizontal_alignment: HorizontalAlignment::Left,
            vertical_alignment: VerticalAlignment::Top,
            autowrap: false,
            uppercase: false,
            visible_characters: None,
        };
        label.set_text(text);
        label
    }

    /// Returns the underlying raw text.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Updates the label text and updates cached lines.
    pub fn set_text(&mut self, text: impl Into<String>) {
        let incoming = text.into();
        self.text.clear();
        self.text.push_str(&incoming);
        self.rebuild_lines_in_place();
    }

    /// Updates the label text in-place by clearing the underlying buffer and invoking `fill`,
    /// reusing existing `String` and line capacities without heap allocations once warmed up.
    pub fn update_text_with(&mut self, fill: impl FnOnce(&mut String)) {
        self.text.clear();
        fill(&mut self.text);
        self.rebuild_lines_in_place();
    }

    fn rebuild_lines_in_place(&mut self) {
        let mut count = 0usize;
        for part in self.text.split('\n') {
            if count < self.lines.len() {
                self.lines[count].clear();
                self.lines[count].push_str(part);
            } else {
                let mut s = String::with_capacity(part.len().max(128));
                s.push_str(part);
                self.lines.push(s);
            }
            count += 1;
        }
        self.lines.truncate(count);
    }

    /// Returns the total number of lines separated by newlines.
    #[inline]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Returns the line string at the specified 0-based line index.
    pub fn get_line(&self, line: usize) -> Option<&str> {
        self.lines.get(line).map(|s| s.as_str())
    }

    /// Writes the formatted display string into `out` without allocating when `out` has capacity.
    pub fn write_display_text(&self, out: &mut String) {
        out.clear();
        let limit = self.visible_characters.unwrap_or(usize::MAX);
        if self.uppercase {
            for ch in self.text.chars().take(limit) {
                for upper in ch.to_uppercase() {
                    out.push(upper);
                }
            }
        } else if self.visible_characters.is_none() {
            out.push_str(&self.text);
        } else {
            out.extend(self.text.chars().take(limit));
        }
    }

    /// Computes the formatted display string taking `uppercase` and `visible_characters` into account.
    pub fn display_text(&self) -> String {
        let mut out = String::with_capacity(self.text.len());
        self.write_display_text(&mut out);
        out
    }
}
