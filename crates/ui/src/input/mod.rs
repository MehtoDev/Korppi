//! UI event routing, mouse filtering, focus management, and synthetic input dispatcher.

pub mod dispatcher;
pub mod event;
pub mod focus;
pub mod routing;

pub use dispatcher::UiInputDispatcher;
pub use event::{UiEvent, UiReceiver};
pub use focus::{FocusDirection, FocusManager};
pub use routing::{collect_front_to_back_order, hit_test_controls};
