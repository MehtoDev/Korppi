//! Keyboard key representations and zero-allocation bitset state tracking.

use serde::{Deserialize, Serialize};

/// Cross-platform physical keyboard scan codes.
///
/// Discriminants match `winit::keyboard::KeyCode` (`as u32`) so `WindowEvent::KeyboardInput`
/// codes round-trip without translation overhead (`INP-01`).
#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyCode {
    // Letters
    KeyA = 19,
    KeyB = 20,
    KeyC = 21,
    KeyD = 22,
    KeyE = 23,
    KeyF = 24,
    KeyG = 25,
    KeyH = 26,
    KeyI = 27,
    KeyJ = 28,
    KeyK = 29,
    KeyL = 30,
    KeyM = 31,
    KeyN = 32,
    KeyO = 33,
    KeyP = 34,
    KeyQ = 35,
    KeyR = 36,
    KeyS = 37,
    KeyT = 38,
    KeyU = 39,
    KeyV = 40,
    KeyW = 41,
    KeyX = 42,
    KeyY = 43,
    KeyZ = 44,

    // Digits
    Digit0 = 5,
    Digit1 = 6,
    Digit2 = 7,
    Digit3 = 8,
    Digit4 = 9,
    Digit5 = 10,
    Digit6 = 11,
    Digit7 = 12,
    Digit8 = 13,
    Digit9 = 14,

    // Modifiers
    ShiftLeft = 60,
    ShiftRight = 61,
    ControlLeft = 55,
    ControlRight = 56,
    AltLeft = 50,
    AltRight = 51,
    SuperLeft = 58,
    SuperRight = 59,

    // Controls & Navigation
    Space = 62,
    Enter = 57,
    Escape = 114,
    Tab = 63,
    Backspace = 52,
    Delete = 72,
    Insert = 76,
    Home = 75,
    End = 73,
    PageUp = 78,
    PageDown = 77,
    ArrowUp = 82,
    ArrowDown = 79,
    ArrowLeft = 80,
    ArrowRight = 81,

    // Function keys
    F1 = 159,
    F2 = 160,
    F3 = 161,
    F4 = 162,
    F5 = 163,
    F6 = 164,
    F7 = 165,
    F8 = 166,
    F9 = 167,
    F10 = 168,
    F11 = 169,
    F12 = 170,
}

impl KeyCode {
    /// All 69 `KeyCode` variants in dense bitset order.
    pub const ALL: &'static [Self] = &[
        Self::KeyA,
        Self::KeyB,
        Self::KeyC,
        Self::KeyD,
        Self::KeyE,
        Self::KeyF,
        Self::KeyG,
        Self::KeyH,
        Self::KeyI,
        Self::KeyJ,
        Self::KeyK,
        Self::KeyL,
        Self::KeyM,
        Self::KeyN,
        Self::KeyO,
        Self::KeyP,
        Self::KeyQ,
        Self::KeyR,
        Self::KeyS,
        Self::KeyT,
        Self::KeyU,
        Self::KeyV,
        Self::KeyW,
        Self::KeyX,
        Self::KeyY,
        Self::KeyZ,
        Self::Digit0,
        Self::Digit1,
        Self::Digit2,
        Self::Digit3,
        Self::Digit4,
        Self::Digit5,
        Self::Digit6,
        Self::Digit7,
        Self::Digit8,
        Self::Digit9,
        Self::ShiftLeft,
        Self::ShiftRight,
        Self::ControlLeft,
        Self::ControlRight,
        Self::AltLeft,
        Self::AltRight,
        Self::SuperLeft,
        Self::SuperRight,
        Self::Space,
        Self::Enter,
        Self::Escape,
        Self::Tab,
        Self::Backspace,
        Self::Delete,
        Self::Insert,
        Self::Home,
        Self::End,
        Self::PageUp,
        Self::PageDown,
        Self::ArrowUp,
        Self::ArrowDown,
        Self::ArrowLeft,
        Self::ArrowRight,
        Self::F1,
        Self::F2,
        Self::F3,
        Self::F4,
        Self::F5,
        Self::F6,
        Self::F7,
        Self::F8,
        Self::F9,
        Self::F10,
        Self::F11,
        Self::F12,
    ];

    /// Returns the numeric discriminant matching `WindowEvent::KeyboardInput` / `winit::keyboard::KeyCode`.
    #[inline]
    #[must_use]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }

    /// Converts a raw `u32` key code (from `WindowEvent::KeyboardInput`) into a [`KeyCode`].
    #[must_use]
    pub const fn from_u32(code: u32) -> Option<Self> {
        match code {
            19 => Some(Self::KeyA),
            20 => Some(Self::KeyB),
            21 => Some(Self::KeyC),
            22 => Some(Self::KeyD),
            23 => Some(Self::KeyE),
            24 => Some(Self::KeyF),
            25 => Some(Self::KeyG),
            26 => Some(Self::KeyH),
            27 => Some(Self::KeyI),
            28 => Some(Self::KeyJ),
            29 => Some(Self::KeyK),
            30 => Some(Self::KeyL),
            31 => Some(Self::KeyM),
            32 => Some(Self::KeyN),
            33 => Some(Self::KeyO),
            34 => Some(Self::KeyP),
            35 => Some(Self::KeyQ),
            36 => Some(Self::KeyR),
            37 => Some(Self::KeyS),
            38 => Some(Self::KeyT),
            39 => Some(Self::KeyU),
            40 => Some(Self::KeyV),
            41 => Some(Self::KeyW),
            42 => Some(Self::KeyX),
            43 => Some(Self::KeyY),
            44 => Some(Self::KeyZ),
            5 => Some(Self::Digit0),
            6 => Some(Self::Digit1),
            7 => Some(Self::Digit2),
            8 => Some(Self::Digit3),
            9 => Some(Self::Digit4),
            10 => Some(Self::Digit5),
            11 => Some(Self::Digit6),
            12 => Some(Self::Digit7),
            13 => Some(Self::Digit8),
            14 => Some(Self::Digit9),
            60 => Some(Self::ShiftLeft),
            61 => Some(Self::ShiftRight),
            55 => Some(Self::ControlLeft),
            56 => Some(Self::ControlRight),
            50 => Some(Self::AltLeft),
            51 => Some(Self::AltRight),
            58 => Some(Self::SuperLeft),
            59 => Some(Self::SuperRight),
            62 => Some(Self::Space),
            57 => Some(Self::Enter),
            114 => Some(Self::Escape),
            63 => Some(Self::Tab),
            52 => Some(Self::Backspace),
            72 => Some(Self::Delete),
            76 => Some(Self::Insert),
            75 => Some(Self::Home),
            73 => Some(Self::End),
            78 => Some(Self::PageUp),
            77 => Some(Self::PageDown),
            82 => Some(Self::ArrowUp),
            79 => Some(Self::ArrowDown),
            80 => Some(Self::ArrowLeft),
            81 => Some(Self::ArrowRight),
            159 => Some(Self::F1),
            160 => Some(Self::F2),
            161 => Some(Self::F3),
            162 => Some(Self::F4),
            163 => Some(Self::F5),
            164 => Some(Self::F6),
            165 => Some(Self::F7),
            166 => Some(Self::F8),
            167 => Some(Self::F9),
            168 => Some(Self::F10),
            169 => Some(Self::F11),
            170 => Some(Self::F12),
            _ => None,
        }
    }

    /// Converts a `winit::keyboard::KeyCode as u32` discriminant into a [`KeyCode`].
    #[inline]
    #[must_use]
    pub const fn from_winit_code(code: u32) -> Option<Self> {
        Self::from_u32(code)
    }

    #[inline]
    const fn bit_index(self) -> usize {
        match self {
            Self::KeyA => 0,
            Self::KeyB => 1,
            Self::KeyC => 2,
            Self::KeyD => 3,
            Self::KeyE => 4,
            Self::KeyF => 5,
            Self::KeyG => 6,
            Self::KeyH => 7,
            Self::KeyI => 8,
            Self::KeyJ => 9,
            Self::KeyK => 10,
            Self::KeyL => 11,
            Self::KeyM => 12,
            Self::KeyN => 13,
            Self::KeyO => 14,
            Self::KeyP => 15,
            Self::KeyQ => 16,
            Self::KeyR => 17,
            Self::KeyS => 18,
            Self::KeyT => 19,
            Self::KeyU => 20,
            Self::KeyV => 21,
            Self::KeyW => 22,
            Self::KeyX => 23,
            Self::KeyY => 24,
            Self::KeyZ => 25,
            Self::Digit0 => 26,
            Self::Digit1 => 27,
            Self::Digit2 => 28,
            Self::Digit3 => 29,
            Self::Digit4 => 30,
            Self::Digit5 => 31,
            Self::Digit6 => 32,
            Self::Digit7 => 33,
            Self::Digit8 => 34,
            Self::Digit9 => 35,
            Self::ShiftLeft => 36,
            Self::ShiftRight => 37,
            Self::ControlLeft => 38,
            Self::ControlRight => 39,
            Self::AltLeft => 40,
            Self::AltRight => 41,
            Self::SuperLeft => 42,
            Self::SuperRight => 43,
            Self::Space => 44,
            Self::Enter => 45,
            Self::Escape => 46,
            Self::Tab => 47,
            Self::Backspace => 48,
            Self::Delete => 49,
            Self::Insert => 50,
            Self::Home => 51,
            Self::End => 52,
            Self::PageUp => 53,
            Self::PageDown => 54,
            Self::ArrowUp => 55,
            Self::ArrowDown => 56,
            Self::ArrowLeft => 57,
            Self::ArrowRight => 58,
            Self::F1 => 59,
            Self::F2 => 60,
            Self::F3 => 61,
            Self::F4 => 62,
            Self::F5 => 63,
            Self::F6 => 64,
            Self::F7 => 65,
            Self::F8 => 66,
            Self::F9 => 67,
            Self::F10 => 68,
            Self::F11 => 69,
            Self::F12 => 70,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct KeyBitset([u64; 2]);

impl KeyBitset {
    #[inline]
    fn insert(&mut self, key: KeyCode) -> bool {
        let idx = key.bit_index();
        let word = idx >> 6;
        let mask = 1u64 << (idx & 63);
        let was_unset = (self.0[word] & mask) == 0;
        self.0[word] |= mask;
        was_unset
    }

    #[inline]
    fn remove(&mut self, key: KeyCode) -> bool {
        let idx = key.bit_index();
        let word = idx >> 6;
        let mask = 1u64 << (idx & 63);
        let was_set = (self.0[word] & mask) != 0;
        self.0[word] &= !mask;
        was_set
    }

    #[inline]
    fn contains(&self, key: KeyCode) -> bool {
        let idx = key.bit_index();
        let word = idx >> 6;
        let mask = 1u64 << (idx & 63);
        (self.0[word] & mask) != 0
    }

    #[inline]
    fn clear(&mut self) {
        self.0 = [0, 0];
    }

    #[inline]
    fn union_with(&mut self, other: &Self) {
        self.0[0] |= other.0[0];
        self.0[1] |= other.0[1];
    }
}

/// Tracks current, just-pressed, and just-released keyboard state across frame boundaries
/// using fixed-size stack bitsets (`0` heap allocations).
#[derive(Debug, Clone, Default)]
pub struct KeyboardState {
    pressed: KeyBitset,
    just_pressed: KeyBitset,
    just_released: KeyBitset,
}

impl KeyboardState {
    /// Creates a new, cleared `KeyboardState`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a key press event.
    #[inline]
    pub fn press(&mut self, key: KeyCode) {
        if self.pressed.insert(key) {
            self.just_pressed.insert(key);
        }
    }

    /// Registers a key release event.
    #[inline]
    pub fn release(&mut self, key: KeyCode) {
        if self.pressed.remove(key) {
            self.just_released.insert(key);
        }
    }

    /// Releases all currently held keys and marks them as `just_released` (e.g. on window focus loss).
    pub fn on_focus_lost(&mut self) {
        self.just_released.union_with(&self.pressed);
        self.pressed.clear();
    }

    /// Returns `true` if the key is currently held down.
    #[inline]
    #[must_use]
    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(key)
    }

    /// Returns `true` if the key was pressed during the current frame.
    #[inline]
    #[must_use]
    pub fn is_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed.contains(key)
    }

    /// Returns `true` if the key was released during the current frame.
    #[inline]
    #[must_use]
    pub fn is_just_released(&self, key: KeyCode) -> bool {
        self.just_released.contains(key)
    }

    /// Returns `true` if any of the provided keys are currently held down.
    #[must_use]
    pub fn is_any_pressed(&self, keys: &[KeyCode]) -> bool {
        keys.iter().any(|&k| self.is_pressed(k))
    }

    /// Returns `true` if either Shift key is currently held down.
    #[must_use]
    pub fn is_shift_pressed(&self) -> bool {
        self.is_pressed(KeyCode::ShiftLeft) || self.is_pressed(KeyCode::ShiftRight)
    }

    /// Returns `true` if either Control key is currently held down.
    #[must_use]
    pub fn is_ctrl_pressed(&self) -> bool {
        self.is_pressed(KeyCode::ControlLeft) || self.is_pressed(KeyCode::ControlRight)
    }

    /// Returns `true` if either Alt key is currently held down.
    #[must_use]
    pub fn is_alt_pressed(&self) -> bool {
        self.is_pressed(KeyCode::AltLeft) || self.is_pressed(KeyCode::AltRight)
    }

    /// Clears frame-transient states (`just_pressed` and `just_released`).
    pub fn tick(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Clears all keyboard state.
    pub fn reset(&mut self) {
        self.pressed.clear();
        self.just_pressed.clear();
        self.just_released.clear();
    }
}
