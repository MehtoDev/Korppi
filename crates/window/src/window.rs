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
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    focused: bool,
    should_close: bool,
    event_queue: VecDeque<WindowEvent>,
    /// Shared so that swapchains can keep the OS window alive while they present to it.
    winit_window: Option<Arc<winit::window::Window>>,
}

impl Window {
    /// Maximum number of events buffered in `event_queue` when `drain_events` is not called.
    pub const MAX_EVENT_QUEUE_CAPACITY: usize = 1024;

    /// Logical pixels per scroll line used to normalize `MouseScrollDelta::PixelDelta`.
    pub const SCROLL_LINE_PIXELS: f64 = 40.0;

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
            physical_width: width,
            physical_height: height,
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
        window.set_cursor_visible(config.cursor_visible);
        Self::apply_cursor_grab(&window, config.cursor_grabbed);

        let inner_size = window.inner_size();
        let raw_scale = window.scale_factor();
        let scale_factor = if raw_scale.is_finite() && raw_scale > 0.0 {
            raw_scale
        } else {
            1.0
        };
        let physical_width = inner_size.width;
        let physical_height = inner_size.height;
        let width = (physical_width as f64 / scale_factor).round() as u32;
        let height = (physical_height as f64 / scale_factor).round() as u32;

        Self {
            config,
            width,
            height,
            physical_width,
            physical_height,
            scale_factor,
            focused: window.has_focus(),
            should_close: false,
            event_queue: VecDeque::new(),
            winit_window: Some(Arc::new(window)),
        }
    }

    fn apply_cursor_grab(window: &winit::window::Window, grabbed: bool) {
        if grabbed {
            if window
                .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                .is_err()
            {
                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Locked);
            }
        } else {
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
        }
    }

    /// Normalizes physical pixel scroll deltas into line units using `scale_factor`.
    #[must_use]
    pub fn normalize_pixel_scroll(pixel_x: f64, pixel_y: f64, scale_factor: f64) -> (f32, f32) {
        let valid_scale = if scale_factor.is_finite() && scale_factor > 0.0 {
            scale_factor
        } else {
            1.0
        };
        let divisor = valid_scale * Self::SCROLL_LINE_PIXELS;
        ((pixel_x / divisor) as f32, (pixel_y / divisor) as f32)
    }

    /// Returns the window configuration.
    #[must_use]
    pub fn config(&self) -> &WindowConfig {
        &self.config
    }

    /// Returns the current window title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.config.title
    }

    /// Sets the window title and updates the underlying OS window if present.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.config.title = title.into();
        if let Some(winit) = &self.winit_window {
            winit.set_title(&self.config.title);
        }
    }

    /// Returns whether the mouse cursor is currently configured to be visible.
    #[must_use]
    pub fn cursor_visible(&self) -> bool {
        self.config.cursor_visible
    }

    /// Sets mouse cursor visibility and updates the underlying OS window if present.
    pub fn set_cursor_visible(&mut self, visible: bool) {
        self.config.cursor_visible = visible;
        if let Some(winit) = &self.winit_window {
            winit.set_cursor_visible(visible);
        }
    }

    /// Returns whether the mouse cursor is currently grabbed/confined.
    #[must_use]
    pub fn cursor_grabbed(&self) -> bool {
        self.config.cursor_grabbed
    }

    /// Sets mouse cursor grab/confinement state and updates the underlying OS window if present.
    pub fn set_cursor_grabbed(&mut self, grabbed: bool) {
        self.config.cursor_grabbed = grabbed;
        if let Some(winit) = &self.winit_window {
            Self::apply_cursor_grab(winit, grabbed);
        }
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

    /// Returns the authoritative physical pixel width of the window client area.
    #[must_use]
    pub fn physical_width(&self) -> u32 {
        self.physical_width
    }

    /// Returns the authoritative physical pixel height of the window client area.
    #[must_use]
    pub fn physical_height(&self) -> u32 {
        self.physical_height
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

    /// Updates focus state without buffering an event.
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
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

    /// Updates logical window dimensions and recomputes physical dimensions without buffering an event.
    pub fn set_logical_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.physical_width = (width as f64 * self.scale_factor).round() as u32;
        self.physical_height = (height as f64 * self.scale_factor).round() as u32;
    }

    /// Updates authoritative physical pixel dimensions and recomputes logical dimensions without buffering an event.
    pub fn set_physical_size(&mut self, physical_width: u32, physical_height: u32) {
        self.physical_width = physical_width;
        self.physical_height = physical_height;
        self.width = (physical_width as f64 / self.scale_factor).round() as u32;
        self.height = (physical_height as f64 / self.scale_factor).round() as u32;
    }

    /// Updates the monitor scale factor and recomputes physical dimensions from logical size without buffering an event.
    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        if scale_factor.is_finite() && scale_factor > 0.0 {
            self.scale_factor = scale_factor;
            self.physical_width = (self.width as f64 * self.scale_factor).round() as u32;
            self.physical_height = (self.height as f64 * self.scale_factor).round() as u32;
        }
    }

    /// Updates logical window dimensions and enqueues a `Resized` event.
    pub fn on_resized(&mut self, width: u32, height: u32) {
        self.set_logical_size(width, height);
        self.push_event(WindowEvent::Resized { width, height });
    }

    /// Updates authoritative physical pixel dimensions and enqueues a `Resized` event.
    pub fn on_physical_resized(&mut self, physical_width: u32, physical_height: u32) {
        self.set_physical_size(physical_width, physical_height);
        self.push_event(WindowEvent::Resized {
            width: self.width,
            height: self.height,
        });
    }

    /// Updates the monitor scale factor and enqueues a `ScaleFactorChanged` event.
    pub fn on_scale_factor_changed(&mut self, scale_factor: f64) {
        self.set_scale_factor(scale_factor);
        self.push_event(WindowEvent::ScaleFactorChanged {
            scale_factor: self.scale_factor,
        });
    }

    /// Pushes a window event into the bounded FIFO event queue.
    pub fn push_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.should_close = true,
            WindowEvent::Focused(f) => self.focused = f,
            WindowEvent::Resized { width, height }
                if self.width != width || self.height != height =>
            {
                self.set_logical_size(width, height);
            }
            WindowEvent::ScaleFactorChanged { scale_factor }
                if (self.scale_factor - scale_factor).abs() > f64::EPSILON =>
            {
                self.set_scale_factor(scale_factor);
            }
            _ => {}
        }
        if self.event_queue.len() >= Self::MAX_EVENT_QUEUE_CAPACITY {
            self.event_queue.pop_front();
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
