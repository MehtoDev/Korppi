//! Window instance management, event buffering, and raw handle export.

use crate::config::WindowConfig;
use crate::event::WindowEvent;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use std::collections::{vec_deque::Drain, VecDeque};
use std::fmt;
use std::sync::Arc;

/// Error types occurring during window creation or handle acquisition.
#[derive(Debug)]
pub enum WindowError {
    /// Failed to create native OS window.
    OsError(String),

    /// Native window or display handle is unavailable.
    HandleUnavailable(String),
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OsError(msg) => write!(f, "Failed to create native OS window: {msg}"),
            Self::HandleUnavailable(msg) => {
                write!(f, "Window or display handle is unavailable: {msg}")
            }
        }
    }
}

impl std::error::Error for WindowError {}

/// Cross-platform window representation managing display metrics, events, and raw OS handles.
pub struct Window {
    config: WindowConfig,
    width: u32,
    height: u32,
    scale_factor: f64,
    focused: bool,
    should_close: bool,
    event_queue: VecDeque<WindowEvent>,
    /// Shared so that swapchains can keep the OS window alive while they present to it.
    winit_window: Option<Arc<winit::window::Window>>,
}

impl Window {
    /// Creates a headless window instance without requiring an active OS display server.
    ///
    /// Ideal for automated integration testing, CI pipelines, and server-side rendering.
    pub fn new_headless(config: WindowConfig) -> Result<Self, WindowError> {
        let width = config.width;
        let height = config.height;

        Ok(Self {
            config,
            width,
            height,
            scale_factor: 1.0,
            focused: true,
            should_close: false,
            event_queue: VecDeque::new(),
            winit_window: None,
        })
    }

    /// Wraps a native `winit::window::Window` with engine configuration.
    #[must_use]
    pub fn from_winit(window: winit::window::Window, config: WindowConfig) -> Self {
        let inner_size = window.inner_size();
        let scale_factor = window.scale_factor();
        let width = (inner_size.width as f64 / scale_factor) as u32;
        let height = (inner_size.height as f64 / scale_factor) as u32;

        Self {
            config,
            width,
            height,
            scale_factor,
            focused: window.has_focus(),
            should_close: false,
            event_queue: VecDeque::new(),
            winit_window: Some(Arc::new(window)),
        }
    }

    /// Returns the window configuration.
    #[must_use]
    pub fn config(&self) -> &WindowConfig {
        &self.config
    }

    /// Returns the logical width of the window client area.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the logical height of the window client area.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the physical pixel width of the window client area.
    #[must_use]
    pub fn physical_width(&self) -> u32 {
        (self.width as f64 * self.scale_factor).round() as u32
    }

    /// Returns the physical pixel height of the window client area.
    #[must_use]
    pub fn physical_height(&self) -> u32 {
        (self.height as f64 * self.scale_factor).round() as u32
    }

    /// Returns the aspect ratio (width / height) of the window client area.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Returns the monitor DPI scale factor.
    #[must_use]
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Returns `true` if the window currently has keyboard/input focus.
    #[must_use]
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Returns `true` if a close request has been received.
    #[must_use]
    pub fn should_close(&self) -> bool {
        self.should_close
    }

    /// Signals the window that it should close.
    pub fn request_close(&mut self) {
        self.should_close = true;
    }

    /// Updates window dimensions and enqueues a `Resized` event.
    pub fn on_resized(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.push_event(WindowEvent::Resized { width, height });
    }

    /// Updates the monitor scale factor and enqueues a `ScaleFactorChanged` event.
    pub fn on_scale_factor_changed(&mut self, scale_factor: f64) {
        self.scale_factor = scale_factor;
        self.push_event(WindowEvent::ScaleFactorChanged { scale_factor });
    }

    /// Pushes a window event into the FIFO event queue.
    pub fn push_event(&mut self, event: WindowEvent) {
        if let WindowEvent::CloseRequested = event {
            self.should_close = true;
        } else if let WindowEvent::Focused(f) = event {
            self.focused = f;
        }
        self.event_queue.push_back(event);
    }

    /// Drains all buffered window events since the previous frame.
    pub fn drain_events(&mut self) -> Drain<'_, WindowEvent> {
        self.event_queue.drain(..)
    }

    /// Returns an immutable reference to the underlying `winit::window::Window`, if available.
    #[must_use]
    pub fn winit_window(&self) -> Option<&winit::window::Window> {
        self.winit_window.as_deref()
    }

    /// Returns a shared handle to the OS window (`None` for headless windows), for presentation:
    /// `Arc<winit::window::Window>` coerces to `Arc<dyn korppi_rhi::PresentTarget>`.
    #[must_use]
    pub fn shared_winit_window(&self) -> Option<Arc<winit::window::Window>> {
        self.winit_window.clone()
    }
}

impl HasWindowHandle for Window {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        if let Some(winit) = &self.winit_window {
            winit.window_handle()
        } else {
            Err(HandleError::Unavailable)
        }
    }
}

impl HasDisplayHandle for Window {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        if let Some(winit) = &self.winit_window {
            winit.display_handle()
        } else {
            Err(HandleError::Unavailable)
        }
    }
}
