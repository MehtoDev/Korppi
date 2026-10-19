//! Gamepad buttons, analog sticks, triggers, and deadzone processing.

use korppi_math::Vec2;
use serde::{Deserialize, Serialize};

/// Maximum number of gamepads stored inline on the stack without heap allocation.
pub const MAX_INLINE_GAMEPADS: usize = 8;

/// Unique identifier for a connected gamepad device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GamepadId(pub usize);

/// Standardized gamepad buttons (matching modern controller layouts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GamepadButton {
    /// Bottom face button (A on Xbox, Cross on PlayStation, B on Nintendo).
    South,
    /// Right face button (B on Xbox, Circle on PlayStation, A on Nintendo).
    East,
    /// Left face button (X on Xbox, Square on PlayStation, Y on Nintendo).
    West,
    /// Top face button (Y on Xbox, Triangle on PlayStation, X on Nintendo).
    North,
    /// D-pad Up.
    DPadUp,
    /// D-pad Down.
    DPadDown,
    /// D-pad Left.
    DPadLeft,
    /// D-pad Right.
    DPadRight,
    /// Left shoulder bumper button (LB / L1).
    LeftBumper,
    /// Right shoulder bumper button (RB / R1).
    RightBumper,
    /// Left trigger digital click or threshold (LT / L2).
    LeftTrigger,
    /// Right trigger digital click or threshold (RT / R2).
    RightTrigger,
    /// Left analog stick click (L3).
    LeftThumb,
    /// Right analog stick click (R3).
    RightThumb,
    /// Select / Back / View button.
    Select,
    /// Start / Menu button.
    Start,
    /// Guide / Home / PS button.
    Mode,
}

impl GamepadButton {
    #[inline]
    const fn bit_mask(self) -> u32 {
        1u32 << (self as u32)
    }
}

/// Standardized analog axes on gamepads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GamepadAxis {
    /// Left analog stick horizontal axis (-1.0 Left to +1.0 Right).
    LeftStickX,
    /// Left analog stick vertical axis (-1.0 Down to +1.0 Up).
    LeftStickY,
    /// Right analog stick horizontal axis (-1.0 Left to +1.0 Right).
    RightStickX,
    /// Right analog stick vertical axis (-1.0 Down to +1.0 Up).
    RightStickY,
    /// Left analog trigger (0.0 Released to +1.0 Fully Pressed).
    LeftTrigger,
    /// Right analog trigger (0.0 Released to +1.0 Fully Pressed).
    RightTrigger,
}

impl GamepadAxis {
    #[inline]
    const fn index(self) -> usize {
        match self {
            Self::LeftStickX => 0,
            Self::LeftStickY => 1,
            Self::RightStickX => 2,
            Self::RightStickY => 3,
            Self::LeftTrigger => 4,
            Self::RightTrigger => 5,
        }
    }

    #[inline]
    const fn is_trigger(self) -> bool {
        matches!(self, Self::LeftTrigger | Self::RightTrigger)
    }
}

/// Internal state tracking for a single connected gamepad using fixed-size bitmasks and arrays.
#[derive(Debug, Clone, Copy)]
struct GamepadDevice {
    id: GamepadId,
    connected: bool,
    pressed: u32,
    just_pressed: u32,
    just_released: u32,
    axes: [f32; 6],
    raw_axes: [f32; 6],
    deadzone: f32,
}

impl Default for GamepadDevice {
    fn default() -> Self {
        Self {
            id: GamepadId(0),
            connected: false,
            pressed: 0,
            just_pressed: 0,
            just_released: 0,
            axes: [0.0; 6],
            raw_axes: [0.0; 6],
            deadzone: 0.15,
        }
    }
}

impl GamepadDevice {
    fn filter_1d(raw_val: f32, deadzone: f32, is_trigger: bool) -> f32 {
        if !raw_val.is_finite() || !deadzone.is_finite() || deadzone >= 1.0 {
            return 0.0;
        }
        let dz = deadzone.max(0.0);
        let clamped = if is_trigger {
            raw_val.clamp(0.0, 1.0)
        } else {
            raw_val.clamp(-1.0, 1.0)
        };
        if clamped.abs() <= dz {
            0.0
        } else {
            let sign = clamped.signum();
            let scaled = (clamped.abs() - dz) / (1.0 - dz);
            sign * scaled.clamp(0.0, 1.0)
        }
    }

    fn filter_radial_2d(raw_x: f32, raw_y: f32, deadzone: f32) -> Vec2 {
        let x = if raw_x.is_finite() {
            raw_x.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let y = if raw_y.is_finite() {
            raw_y.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        if !deadzone.is_finite() || deadzone >= 1.0 {
            return Vec2::ZERO;
        }
        let dz = deadzone.max(0.0);
        let raw = Vec2::new(x, y);
        let len = raw.length();
        if len <= dz || len <= f32::EPSILON {
            Vec2::ZERO
        } else {
            let scaled = ((len.min(1.0) - dz) / (1.0 - dz)).clamp(0.0, 1.0);
            (raw / len) * scaled
        }
    }
}

/// Manages multiple gamepads, button transitions, and analog axis deadzone calibration
/// with inline storage (`0` heap allocations for up to [`MAX_INLINE_GAMEPADS`] controllers).
#[derive(Debug, Clone)]
pub struct GamepadState {
    inline_devices: [GamepadDevice; MAX_INLINE_GAMEPADS],
    overflow_devices: Vec<GamepadDevice>,
}

impl Default for GamepadState {
    fn default() -> Self {
        Self {
            inline_devices: [GamepadDevice::default(); MAX_INLINE_GAMEPADS],
            overflow_devices: Vec::new(),
        }
    }
}

impl GamepadState {
    /// Creates an empty `GamepadState`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn get_device(&self, id: GamepadId) -> Option<&GamepadDevice> {
        if id.0 < MAX_INLINE_GAMEPADS {
            let dev = &self.inline_devices[id.0];
            if dev.connected && dev.id == id {
                return Some(dev);
            }
        }
        self.inline_devices
            .iter()
            .find(|d| d.connected && d.id == id)
            .or_else(|| {
                self.overflow_devices
                    .iter()
                    .find(|d| d.connected && d.id == id)
            })
    }

    fn get_device_mut(&mut self, id: GamepadId) -> Option<&mut GamepadDevice> {
        if id.0 < MAX_INLINE_GAMEPADS
            && self.inline_devices[id.0].connected
            && self.inline_devices[id.0].id == id
        {
            return Some(&mut self.inline_devices[id.0]);
        }
        if let Some(idx) = self
            .inline_devices
            .iter()
            .position(|d| d.connected && d.id == id)
        {
            return Some(&mut self.inline_devices[idx]);
        }
        self.overflow_devices
            .iter_mut()
            .find(|d| d.connected && d.id == id)
    }

    fn connected_devices(&self) -> impl Iterator<Item = &GamepadDevice> {
        self.inline_devices
            .iter()
            .chain(self.overflow_devices.iter())
            .filter(|d| d.connected)
    }

    /// Returns an iterator over the IDs of all currently connected gamepads.
    pub fn connected_ids(&self) -> impl Iterator<Item = GamepadId> + '_ {
        self.connected_devices().map(|d| d.id)
    }

    /// Registers a gamepad connection.
    pub fn connect(&mut self, id: GamepadId) {
        if self.get_device(id).is_some() {
            return;
        }
        if id.0 < MAX_INLINE_GAMEPADS && !self.inline_devices[id.0].connected {
            self.inline_devices[id.0] = GamepadDevice {
                id,
                connected: true,
                ..GamepadDevice::default()
            };
            return;
        }
        if let Some(slot) = self.inline_devices.iter_mut().find(|d| !d.connected) {
            *slot = GamepadDevice {
                id,
                connected: true,
                ..GamepadDevice::default()
            };
            return;
        }
        self.overflow_devices.push(GamepadDevice {
            id,
            connected: true,
            ..GamepadDevice::default()
        });
    }

    /// Unregisters a disconnected gamepad.
    pub fn disconnect(&mut self, id: GamepadId) {
        if let Some(dev) = self.get_device_mut(id) {
            *dev = GamepadDevice::default();
        }
    }

    /// Returns `true` if the gamepad is currently connected.
    #[must_use]
    pub fn is_connected(&self, id: GamepadId) -> bool {
        self.get_device(id).is_some()
    }

    /// Sets the axis deadzone for the gamepad (default is 0.15, clamped to `[0.0, 1.0]`).
    pub fn set_deadzone(&mut self, id: GamepadId, deadzone: f32) {
        if !deadzone.is_finite() {
            return;
        }
        if let Some(dev) = self.get_device_mut(id) {
            dev.deadzone = deadzone.clamp(0.0, 1.0);
        }
    }

    /// Sets raw axis value, applying deadzone calibration and range clamping (`[-1.0, 1.0]` for
    /// sticks, `[0.0, 1.0]` for triggers).
    pub fn set_axis(&mut self, id: GamepadId, axis: GamepadAxis, raw_val: f32) {
        if let Some(dev) = self.get_device_mut(id) {
            let idx = axis.index();
            dev.raw_axes[idx] = if raw_val.is_finite() { raw_val } else { 0.0 };
            dev.axes[idx] = GamepadDevice::filter_1d(raw_val, dev.deadzone, axis.is_trigger());
        }
    }

    /// Sets a 2D analog stick pair (`x_axis`, `y_axis`) with radial deadzone calibration.
    pub fn set_stick(
        &mut self,
        id: GamepadId,
        x_axis: GamepadAxis,
        y_axis: GamepadAxis,
        raw_x: f32,
        raw_y: f32,
    ) {
        if let Some(dev) = self.get_device_mut(id) {
            let xi = x_axis.index();
            let yi = y_axis.index();
            dev.raw_axes[xi] = if raw_x.is_finite() { raw_x } else { 0.0 };
            dev.raw_axes[yi] = if raw_y.is_finite() { raw_y } else { 0.0 };
            let calibrated = GamepadDevice::filter_radial_2d(raw_x, raw_y, dev.deadzone);
            dev.axes[xi] = calibrated.x;
            dev.axes[yi] = calibrated.y;
        }
    }

    /// Reads current calibrated value for `axis`.
    #[must_use]
    pub fn axis(&self, id: GamepadId, axis: GamepadAxis) -> f32 {
        self.get_device(id)
            .map_or(0.0, |dev| dev.axes[axis.index()])
    }

    /// Reads the radially deadzone-calibrated 2D vector for the left analog stick.
    #[must_use]
    pub fn left_stick(&self, id: GamepadId) -> Vec2 {
        self.get_device(id).map_or(Vec2::ZERO, |dev| {
            GamepadDevice::filter_radial_2d(
                dev.raw_axes[GamepadAxis::LeftStickX.index()],
                dev.raw_axes[GamepadAxis::LeftStickY.index()],
                dev.deadzone,
            )
        })
    }

    /// Reads the radially deadzone-calibrated 2D vector for the right analog stick.
    #[must_use]
    pub fn right_stick(&self, id: GamepadId) -> Vec2 {
        self.get_device(id).map_or(Vec2::ZERO, |dev| {
            GamepadDevice::filter_radial_2d(
                dev.raw_axes[GamepadAxis::RightStickX.index()],
                dev.raw_axes[GamepadAxis::RightStickY.index()],
                dev.deadzone,
            )
        })
    }

    /// Registers a gamepad button press.
    pub fn press(&mut self, id: GamepadId, button: GamepadButton) {
        if let Some(dev) = self.get_device_mut(id) {
            let mask = button.bit_mask();
            if (dev.pressed & mask) == 0 {
                dev.pressed |= mask;
                dev.just_pressed |= mask;
            }
        }
    }

    /// Registers a gamepad button release.
    pub fn release(&mut self, id: GamepadId, button: GamepadButton) {
        if let Some(dev) = self.get_device_mut(id) {
            let mask = button.bit_mask();
            if (dev.pressed & mask) != 0 {
                dev.pressed &= !mask;
                dev.just_released |= mask;
            }
        }
    }

    /// Returns `true` if the button is held down on `id`.
    #[must_use]
    pub fn is_pressed(&self, id: GamepadId, button: GamepadButton) -> bool {
        let mask = button.bit_mask();
        self.get_device(id)
            .is_some_and(|dev| (dev.pressed & mask) != 0)
    }

    /// Returns `true` if the button was pressed in the current frame on `id`.
    #[must_use]
    pub fn is_just_pressed(&self, id: GamepadId, button: GamepadButton) -> bool {
        let mask = button.bit_mask();
        self.get_device(id)
            .is_some_and(|dev| (dev.just_pressed & mask) != 0)
    }

    /// Returns `true` if the button was released in the current frame on `id`.
    #[must_use]
    pub fn is_just_released(&self, id: GamepadId, button: GamepadButton) -> bool {
        let mask = button.bit_mask();
        self.get_device(id)
            .is_some_and(|dev| (dev.just_released & mask) != 0)
    }

    /// Advances frame tick, clearing frame-transient transitions.
    pub fn tick(&mut self) {
        for dev in self
            .inline_devices
            .iter_mut()
            .chain(self.overflow_devices.iter_mut())
        {
            dev.just_pressed = 0;
            dev.just_released = 0;
        }
    }
}
