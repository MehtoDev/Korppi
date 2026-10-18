//! Engine application trait and Winit event loop runner.

use crate::config::WindowConfig;
use crate::event::WindowEvent;
use crate::window::{Window, WindowError};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent as WinitWindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

/// Trait defining lifecycle and event callbacks for Korppi applications.
pub trait KorppiApp {
    /// Called when the application starts and the window is initialized.
    fn on_start(&mut self, _window: &mut Window) {}

    /// Called on every frame tick.
    fn on_update(&mut self, _window: &mut Window, _delta_time: f32) {}

    /// Called when a window or OS input event occurs.
    fn on_event(&mut self, _window: &mut Window, _event: &WindowEvent) {}

    /// Called when the application is stopping.
    fn on_stop(&mut self) {}
}

/// Backwards-compatible alias for [`KorppiApp`].
pub use KorppiApp as EngineApp;

/// Bridges a [`KorppiApp`] to Winit's cross-platform event loop.
pub struct AppRunner<A: KorppiApp> {
    app: A,
    config: WindowConfig,
    window: Option<Window>,
    last_frame_time: Option<Instant>,
}

impl<A: KorppiApp> AppRunner<A> {
    /// Creates a new application runner with the given application and window configuration.
    pub fn new(app: A, config: WindowConfig) -> Self {
        Self {
            app,
            config,
            window: None,
            last_frame_time: None,
        }
    }

    /// Starts and runs the platform event loop blocking the main thread.
    ///
    /// The loop sleeps until the OS has events (`ControlFlow::Wait`) and frames are driven by
    /// redraw requests: every `RedrawRequested` calls [`KorppiApp::on_update`] and requests the
    /// next redraw, so rendering runs continuously at the pace presentation allows (vsync)
    /// instead of busy-polling (ISSUE-001).
    ///
    /// Returns the app once the loop has exited (after [`KorppiApp::on_stop`]), so that the
    /// caller can inspect its final state.
    pub fn run(mut self) -> Result<A, WindowError> {
        let event_loop = EventLoop::new().map_err(|e| WindowError::OsError(e.to_string()))?;
        event_loop.set_control_flow(ControlFlow::Wait);
        event_loop
            .run_app(&mut self)
            .map_err(|e| WindowError::OsError(e.to_string()))?;
        self.app.on_stop();
        // Drop the window after the app has released everything that presents to it.
        self.window = None;
        Ok(self.app)
    }
}

impl<A: KorppiApp> ApplicationHandler for AppRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let mut attributes = WindowAttributes::default()
            .with_title(&self.config.title)
            .with_inner_size(LogicalSize::new(self.config.width, self.config.height))
            .with_resizable(self.config.resizable);

        match self.config.mode {
            crate::config::WindowMode::Fullscreen => {
                attributes =
                    attributes.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            }
            crate::config::WindowMode::BorderlessFullscreen => {
                attributes =
                    attributes.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            }
            crate::config::WindowMode::Windowed => {}
        }

        let winit_win = match event_loop.create_window(attributes) {
            Ok(w) => w,
            Err(e) => {
                tracing::error!("Failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let mut win = Window::from_winit(winit_win, self.config.clone());
        self.app.on_start(&mut win);
        // The first frame; each later one is requested after the previous frame's update.
        if let Some(w) = win.winit_window() {
            w.request_redraw();
        }
        self.window = Some(win);
        self.last_frame_time = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WinitWindowEvent,
    ) {
        let Some(window) = self.window.as_mut() else {
            return;
        };

        match event {
            WinitWindowEvent::CloseRequested => {
                window.request_close();
                let ev = WindowEvent::CloseRequested;
                self.app.on_event(window, &ev);
                event_loop.exit();
            }
            WinitWindowEvent::Resized(phys_size) => {
                let scale = window.scale_factor();
                let log_w = (phys_size.width as f64 / scale).round() as u32;
                let log_h = (phys_size.height as f64 / scale).round() as u32;
                window.on_resized(log_w, log_h);
                let ev = WindowEvent::Resized {
                    width: log_w,
                    height: log_h,
                };
                self.app.on_event(window, &ev);
            }
            WinitWindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                window.on_scale_factor_changed(scale_factor);
                let ev = WindowEvent::ScaleFactorChanged { scale_factor };
                self.app.on_event(window, &ev);
            }
            WinitWindowEvent::Focused(focused) => {
                let ev = WindowEvent::Focused(focused);
                window.push_event(ev.clone());
                self.app.on_event(window, &ev);
            }
            WinitWindowEvent::CursorMoved { position, .. } => {
                let scale = window.scale_factor();
                let ev = WindowEvent::CursorMoved {
                    x: position.x / scale,
                    y: position.y / scale,
                };
                window.push_event(ev.clone());
                self.app.on_event(window, &ev);
            }
            WinitWindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    winit::event::MouseButton::Left => 0,
                    winit::event::MouseButton::Right => 1,
                    winit::event::MouseButton::Middle => 2,
                    winit::event::MouseButton::Back => 3,
                    winit::event::MouseButton::Forward => 4,
                    winit::event::MouseButton::Other(o) => 5 + o as u32,
                };
                let ev = WindowEvent::MouseInput {
                    button: btn,
                    pressed: state.is_pressed(),
                };
                window.push_event(ev.clone());
                self.app.on_event(window, &ev);
            }
            WinitWindowEvent::MouseWheel { delta, .. } => {
                let (delta_x, delta_y) = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (x, y),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                };
                let ev = WindowEvent::MouseWheel { delta_x, delta_y };
                window.push_event(ev.clone());
                self.app.on_event(window, &ev);
            }
            WinitWindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                let key_code = match key_event.physical_key {
                    winit::keyboard::PhysicalKey::Code(code) => Some(code as u32),
                    winit::keyboard::PhysicalKey::Unidentified(_) => None,
                };
                let ev = WindowEvent::KeyboardInput {
                    key_code,
                    pressed: key_event.state.is_pressed(),
                };
                window.push_event(ev.clone());
                self.app.on_event(window, &ev);
            }
            WinitWindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_frame_time
                    .map_or(1.0 / 60.0, |last| now.duration_since(last).as_secs_f32());
                self.last_frame_time = Some(now);

                self.app.on_update(window, dt);

                if let Some(w) = window.winit_window() {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.as_ref().is_some_and(Window::should_close) {
            event_loop.exit();
        }
    }
}
