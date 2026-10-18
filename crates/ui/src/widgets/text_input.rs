//! Single-line (`LineEdit`) and multi-line (`TextEdit`) editable text input widgets.

use korppi_ecs::Component;

/// Single-line editable text input widget component.
#[derive(Clone, Debug, PartialEq)]
pub struct LineEdit {
    /// Stored text string.
    text: String,
    /// Placeholder text shown when `text` is empty.
    pub placeholder_text: String,
    /// Current cursor/caret column position in UTF-8 character indices.
    caret_column: usize,
    /// Start and end character indices of the current selection.
    pub selection: Option<(usize, usize)>,
    /// Maximum character length constraint (0 = unlimited).
    pub max_length: usize,
    /// Whether user text editing is enabled.
    pub editable: bool,
    /// Whether text characters are masked with `secret_character`.
    pub secret: bool,
    /// Character used to mask secret text (default `'*'`).
    pub secret_character: char,
}

impl Default for LineEdit {
    fn default() -> Self {
        Self {
            text: String::new(),
            placeholder_text: String::new(),
            caret_column: 0,
            selection: None,
            max_length: 0,
            editable: true,
            secret: false,
            secret_character: '*',
        }
    }
}

impl Component for LineEdit {}

impl LineEdit {
    /// Creates a new empty `LineEdit`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the text content.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Replaces the text content and clamps caret column.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        let len = self.text.chars().count();
        if self.caret_column > len {
            self.caret_column = len;
        }
    }

    /// Returns the caret column position.
    #[inline]
    pub fn caret_column(&self) -> usize {
        self.caret_column
    }

    /// Sets the caret column position, clamping to valid bounds.
    pub fn set_caret_column(&mut self, col: usize) {
        let len = self.text.chars().count();
        self.caret_column = col.min(len);
    }

    /// Inserts text at the current caret column and advances the caret.
    pub fn insert_text(&mut self, input: &str) {
        if !self.editable {
            return;
        }

        let mut chars: Vec<char> = self.text.chars().collect();
        let col = self.caret_column.min(chars.len());

        let input_chars: Vec<char> = input.chars().collect();
        for (i, &ch) in input_chars.iter().enumerate() {
            if self.max_length > 0 && chars.len() >= self.max_length {
                break;
            }
            chars.insert(col + i, ch);
            self.caret_column += 1;
        }

        self.text = chars.into_iter().collect();
    }

    /// Deletes the character immediately before the caret (Backspace).
    pub fn delete_char_before_caret(&mut self) {
        if !self.editable || self.caret_column == 0 {
            return;
        }

        let mut chars: Vec<char> = self.text.chars().collect();
        if self.caret_column <= chars.len() {
            chars.remove(self.caret_column - 1);
            self.caret_column -= 1;
            self.text = chars.into_iter().collect();
        }
    }

    /// Deletes the character immediately at the caret (Delete key).
    pub fn delete_char_at_caret(&mut self) {
        if !self.editable {
            return;
        }

        let mut chars: Vec<char> = self.text.chars().collect();
        if self.caret_column < chars.len() {
            chars.remove(self.caret_column);
            self.text = chars.into_iter().collect();
        }
    }

    /// Selects all characters.
    pub fn select_all(&mut self) {
        let len = self.text.chars().count();
        if len > 0 {
            self.selection = Some((0, len));
        }
    }

    /// Deselects any active text selection.
    pub fn deselect(&mut self) {
        self.selection = None;
    }

    /// Returns the currently selected substring slice, if any.
    pub fn selected_text(&self) -> Option<&str> {
        let (start, end) = self.selection?;
        let (lo, hi) = (start.min(end), start.max(end));

        let mut byte_start = None;
        let mut byte_end = None;

        for (idx, (byte_idx, _)) in self.text.char_indices().enumerate() {
            if idx == lo {
                byte_start = Some(byte_idx);
            }
            if idx == hi {
                byte_end = Some(byte_idx);
            }
        }

        if byte_start.is_none() && lo == self.text.chars().count() {
            byte_start = Some(self.text.len());
        }
        if byte_end.is_none() && hi >= self.text.chars().count() {
            byte_end = Some(self.text.len());
        }

        match (byte_start, byte_end) {
            (Some(s), Some(e)) => Some(&self.text[s..e]),
            _ => None,
        }
    }

    /// Formats the display string, substituting secret characters if `secret` is true.
    pub fn display_text(&self) -> String {
        if self.secret {
            let count = self.text.chars().count();
            std::iter::repeat_n(self.secret_character, count).collect()
        } else {
            self.text.clone()
        }
    }
}

/// Multi-line editable text widget component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextEdit {
    /// Combined raw text string.
    text: String,
    /// Cached lines of text.
    lines: Vec<String>,
    /// Caret line row index.
    caret_line: usize,
    /// Caret column character index within the current line.
    caret_column: usize,
    /// Whether user text editing is enabled.
    pub editable: bool,
    /// Whether long lines automatically wrap to the next line.
    pub wrap_lines: bool,
}

impl Component for TextEdit {}

impl TextEdit {
    /// Creates a new empty `TextEdit`.
    pub fn new() -> Self {
        let mut edit = Self {
            text: String::new(),
            lines: Vec::new(),
            caret_line: 0,
            caret_column: 0,
            editable: true,
            wrap_lines: false,
        };
        edit.set_text("");
        edit
    }

    /// Returns the underlying text.
    #[inline]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Replaces the text content and splits lines.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.lines = self.text.split('\n').map(|s| s.to_string()).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.clamp_caret();
    }

    /// Returns total line count.
    #[inline]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Returns the line string at the specified 0-based index.
    pub fn get_line(&self, line: usize) -> Option<&str> {
        self.lines.get(line).map(|s| s.as_str())
    }

    /// Returns current caret line index.
    #[inline]
    pub fn caret_line(&self) -> usize {
        self.caret_line
    }

    /// Returns current caret column index.
    #[inline]
    pub fn caret_column(&self) -> usize {
        self.caret_column
    }

    /// Sets caret line and column.
    pub fn set_caret(&mut self, line: usize, col: usize) {
        self.caret_line = line;
        self.caret_column = col;
        self.clamp_caret();
    }

    /// Inserts text at the current caret position.
    pub fn insert_text(&mut self, input: &str) {
        if !self.editable {
            return;
        }

        let input_lines: Vec<&str> = input.split('\n').collect();
        if input_lines.is_empty() {
            return;
        }

        self.clamp_caret();
        let cur_line = &self.lines[self.caret_line];
        let chars: Vec<char> = cur_line.chars().collect();
        let prefix: String = chars[..self.caret_column].iter().collect();
        let suffix: String = chars[self.caret_column..].iter().collect();

        if input_lines.len() == 1 {
            // Single line insert
            let new_line = format!("{}{}{}", prefix, input_lines[0], suffix);
            self.caret_column += input_lines[0].chars().count();
            self.lines[self.caret_line] = new_line;
        } else {
            // Multi-line insert
            let mut new_lines = Vec::new();
            for (i, &line_slice) in input_lines.iter().enumerate() {
                if i == 0 {
                    new_lines.push(format!("{}{}", prefix, line_slice));
                } else if i == input_lines.len() - 1 {
                    self.caret_line += input_lines.len() - 1;
                    self.caret_column = line_slice.chars().count();
                    new_lines.push(format!("{}{}", line_slice, suffix));
                } else {
                    new_lines.push(line_slice.to_string());
                }
            }

            self.lines.splice(
                self.caret_line - (input_lines.len() - 1)
                    ..=self.caret_line - (input_lines.len() - 1),
                new_lines,
            );
        }

        self.text = self.lines.join("\n");
        self.clamp_caret();
    }

    fn clamp_caret(&mut self) {
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        if self.caret_line >= self.lines.len() {
            self.caret_line = self.lines.len() - 1;
        }
        let line_len = self.lines[self.caret_line].chars().count();
        if self.caret_column > line_len {
            self.caret_column = line_len;
        }
    }
}
