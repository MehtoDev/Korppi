//! UI input event definitions and event receiver component.

use korppi_ecs::Component;
use korppi_input::keyboard::KeyCode;
use korppi_input::mouse::MouseButton;
use korppi_math::Vec2;

/// UI input event dispatched to control nodes.
#[derive(Clone, Debug, PartialEq)]
pub enum UiEvent {
    /// Cursor moved to a new screen coordinate.
    MouseMove {
        /// New cursor screen coordinate.
        position: Vec2,
    },
    /// Mouse button pressed or released.
    MouseButton {
        /// Cursor screen coordinate when event occurred.
        position: Vec2,
        /// Mouse button pressed or released.
        button: MouseButton,
        /// Whether the button is down (`true`) or up (`false`).
        pressed: bool,
    },
    /// Mouse wheel scrolled.
    MouseWheel {
        /// Cursor screen coordinate when scroll occurred.
        position: Vec2,
        /// Scroll delta vector (X and Y offsets).
        delta: Vec2,
    },
    /// Cursor entered the control's bounding box.
    MouseEnter,
    /// Cursor exited the control's bounding box.
    MouseExit,
    /// Control gained keyboard/gamepad focus.
    FocusGained,
    /// Control lost keyboard/gamepad focus.
    FocusLost,
    /// Keyboard key pressed.
    KeyDown {
        /// Key code of the pressed key.
        key: KeyCode,
    },
    /// Keyboard key released.
    KeyUp {
        /// Key code of the released key.
        key: KeyCode,
    },
    /// Printable text character entered.
    CharInput {
        /// Character input.
        character: char,
    },
}

/// Component buffering UI events dispatched to a Control node.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UiReceiver {
    /// Queue of pending UI events.
    pub events: Vec<UiEvent>,
}

impl Component for UiReceiver {}

impl UiReceiver {
    /// Creates a new empty `UiReceiver`.
    #[inline]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Clears all buffered events.
    #[inline]
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Appends a new event to the buffer.
    #[inline]
    pub fn push(&mut self, event: UiEvent) {
        self.events.push(event);
    }
}
