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
            text: String::new(),
            lines: Vec::new(),
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
        self.text = text.into();
        self.lines = self.text.split('\n').map(|s| s.to_string()).collect();
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

    /// Computes the formatted display string taking `uppercase` and `visible_characters` into account.
    pub fn display_text(&self) -> String {
        let base = if self.uppercase {
            self.text.to_uppercase()
        } else {
            self.text.clone()
        };

        if let Some(limit) = self.visible_characters {
            base.chars().take(limit).collect()
        } else {
            base
        }
    }
}
