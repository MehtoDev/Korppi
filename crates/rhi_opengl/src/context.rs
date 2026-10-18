//! OpenGL contexts: creation on a platform context API (EGL or WGL), the OpenGL loader and the
//! context-wide state the RHI relies on, plus window surfaces for presentation.

use crate::gl_fns::GlExtraFns;
use glow::HasContext;
use korppi_rhi::{api_validation_requested, DiagnosticLog, DiagnosticSeverity, RhiError};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::c_void;
use std::sync::Arc;

#[cfg(not(windows))]
use crate::egl::{EglContext as NativeContext, EglWindowSurface as NativeWindowSurface};
#[cfg(windows)]
use crate::wgl::{WglContext as NativeContext, WglWindowSurface as NativeWindowSurface};

/// Concrete OpenGL context: the platform context (EGL or WGL) and Glow function pointers.
///
/// The context is made current on the thread that creates it and must stay on that thread:
/// `GlContext` is neither `Send` nor `Sync`. `OpenGlDevice` creates, uses and drops it on its
/// render thread (RGL-08).
pub struct GlContext {
    /// Glow OpenGL function table.
    pub gl: glow::Context,
    /// Whether `KHR_debug` output is enabled and feeds [`GlContext::diagnostics`].
    pub debug_output_enabled: bool,
    /// Messages reported through the OpenGL debug output callback.
    pub diagnostics: Arc<DiagnosticLog>,
    /// Entry points `glow` does not wrap.
    pub extra: GlExtraFns,
    native: NativeContext,
    platform: GlPlatform,
    /// Raw pointers make the type `!Send + !Sync`; see the type documentation.
    _not_send: std::marker::PhantomData<*const ()>,
}

/// The window system an OpenGL context is created for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GlPlatform {
    /// No window system: offscreen rendering only.
    Surfaceless,
    /// Xlib: `display` is a `Display*`.
    Xlib {
        /// The Xlib `Display*`.
        display: *mut c_void,
        /// Screen number.
        screen: i32,
    },
    /// XCB: `connection` is an `xcb_connection_t*`.
    Xcb {
        /// The `xcb_connection_t*`.
        connection: *mut c_void,
        /// Screen number.
        screen: i32,
    },
    /// Wayland: `display` is a `wl_display*`.
    Wayland {
        /// The `wl_display*`.
        display: *mut c_void,
    },
    /// Win32 (WGL): windows of this process.
    Win32,
}

impl GlPlatform {
    /// The platform of a window system's display handle.
    ///
    /// # Errors
    /// `RhiError::Unsupported` for window systems without a context API in this backend (EGL
    /// for X11/XCB/Wayland, WGL for Win32).
    pub fn from_display_handle(handle: RawDisplayHandle) -> Result<Self, RhiError> {
        match handle {
            RawDisplayHandle::Xlib(h) => Ok(Self::Xlib {
                display: h
                    .display
                    .ok_or_else(|| {
                        RhiError::Unsupported("Xlib display handle without a Display".into())
                    })?
                    .as_ptr(),
                screen: h.screen,
            }),
            RawDisplayHandle::Xcb(h) => Ok(Self::Xcb {
                connection: h
                    .connection
                    .ok_or_else(|| {
                        RhiError::Unsupported("XCB display handle without a connection".into())
                    })?
                    .as_ptr(),
                screen: h.screen,
            }),
            RawDisplayHandle::Wayland(h) => Ok(Self::Wayland {
                display: h.display.as_ptr(),
            }),
            RawDisplayHandle::Windows(_) => Ok(Self::Win32),
            other => Err(RhiError::Unsupported(format!(
                "the OpenGL backend cannot present on this window system ({other:?}); only \
                 X11 and Wayland (EGL) and Win32 (WGL) are implemented"
            ))),
        }
    }
}

/// A window surface of a [`GlContext`].
pub struct GlWindowSurface(NativeWindowSurface);

impl GlContext {
    /// Creates a headless context (EGL surfaceless platform or default display; a hidden window
    /// with WGL) with OpenGL 4.5 Core Profile.
    pub fn create_headless() -> Result<Self, RhiError> {
        // SAFETY: the surfaceless platform has no native display.
        unsafe { Self::create_for_platform(GlPlatform::Surfaceless) }
    }

    /// Creates a context for `platform`'s windows and makes it current on this thread.
    ///
    /// # Safety
    /// The native display in `platform` must stay valid until the context is dropped.
    pub unsafe fn create_for_platform(platform: GlPlatform) -> Result<Self, RhiError> {
        // Context attributes: OpenGL 4.5 Core Profile, with a debug context when API validation
        // is requested (falls back to a non-debug context if the driver refuses one).
        let validation_requested = api_validation_requested();
        // SAFETY: forwarded from the caller.
        let native = unsafe { NativeContext::create(platform, validation_requested) }?;
        // Load Glow OpenGL function table
        // SAFETY: the context is current on this thread; the loader returns addresses of this
        // context's functions (or null), as `from_loader_function` requires.
        let mut gl =
            unsafe { glow::Context::from_loader_function(|name| native.get_proc_address(name)) };

        // Validate OpenGL version and DSA support
        // SAFETY: the context is current on this thread.
        let version_str = unsafe { gl.get_parameter_string(glow::VERSION) };
        tracing::info!("Loaded OpenGL version: {}", version_str);

        // Debug output first, so later initialization errors are reported too (RGL-17).
        let diagnostics = Arc::new(DiagnosticLog::new());
        let debug_output_enabled = validation_requested && gl.supports_debug();
        if debug_output_enabled {
            let log = Arc::clone(&diagnostics);
            // SAFETY: the context was made current on this thread above. The callback only
            // captures an `Arc<DiagnosticLog>` (`Send + Sync`) and never panics.
            unsafe {
                gl.enable(glow::DEBUG_OUTPUT);
                gl.enable(glow::DEBUG_OUTPUT_SYNCHRONOUS);
                gl.debug_message_callback(move |_source, gl_type, id, severity, message| {
                    record_debug_message(&log, gl_type, id, severity, message);
                });
            }
        } else if validation_requested {
            tracing::warn!("OpenGL API validation requested but KHR_debug is unavailable");
        }

        // SAFETY: the context is current on this thread and the platform loader returns this
        // context's functions.
        let extra = unsafe { GlExtraFns::load(|name| native.get_proc_address(name)) };

        if extra.clip_control.is_none() {
            let message = "glClipControl is unavailable: render targets are stored bottom-up, \
                           depth range stays [-1, 1] and reverse-Z depth is wrong";
            tracing::error!("{message}");
            diagnostics.push(DiagnosticSeverity::Error, message);
        }

        let context = Self {
            gl,
            debug_output_enabled,
            diagnostics,
            extra,
            native,
            platform,
            _not_send: std::marker::PhantomData,
        };
        context.apply_context_state();
        Ok(context)
    }

    /// Applies the context-wide state the backend relies on and never changes afterwards; again
    /// after code outside the backend used the context (`OpenGlDevice::with_gl`).
    ///
    /// RHI coordinate contract (ADR-013, RHI-01): row 0 of every render target is the top of
    /// the image, viewport/scissor origins are top-left, CCW is front-facing, depth is [0, 1].
    /// `GL_UPPER_LEFT` negates NDC y (so +Y-up content lands top-first in memory, like Vulkan
    /// with its negative-height viewport) and also negates the polygon-area sign (GL 4.6
    /// §13.8, §14.6.1 eq. 14.8), so `glFrontFace(GL_CCW)` keeps CCW triangles front-facing.
    /// Presenting to a window's default framebuffer therefore flips vertically (FIX-1.3).
    pub fn apply_context_state(&self) {
        let gl = &self.gl;
        // SAFETY: a `GlContext` is current on the thread that owns it (it is neither `Send` nor
        // `Sync`); `clip_control` was loaded from this context.
        unsafe {
            if let Some(clip_control) = self.extra.clip_control {
                clip_control(glow::UPPER_LEFT, glow::ZERO_TO_ONE);
            }
            // Byte-aligned pixel transfers from client memory.
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            gl.enable(glow::TEXTURE_CUBE_MAP_SEAMLESS);
            // Encode writes to sRGB attachments, as Vulkan does for `*_SRGB` formats. Only
            // attachments with sRGB encoding are affected.
            gl.enable(glow::FRAMEBUFFER_SRGB);
            gl.disable(glow::RASTERIZER_DISCARD);
            gl.disable(glow::PRIMITIVE_RESTART);
            gl.disable(glow::PRIMITIVE_RESTART_FIXED_INDEX);
        }
    }

    /// The window system this context was created for.
    pub fn platform(&self) -> GlPlatform {
        self.platform
    }

    /// Creates a window surface for `window`, which must belong to this context's display.
    ///
    /// # Safety
    /// `window` must stay valid until the surface is destroyed with
    /// [`GlContext::destroy_window_surface`].
    pub unsafe fn create_window_surface(
        &mut self,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<GlWindowSurface, RhiError> {
        // SAFETY: forwarded from the caller.
        unsafe { self.native.create_window_surface(window, width, height) }.map(GlWindowSurface)
    }

    /// Makes `surface` (or, with `None`, the offscreen surface) the context's draw surface.
    pub fn make_current(&mut self, surface: Option<&GlWindowSurface>) -> Result<(), RhiError> {
        self.native.make_current(surface.map(|s| &s.0))
    }

    /// Sets the swap interval of the current window surface (1 = vsync, 0 = immediate).
    pub fn set_swap_interval(&self, interval: i32) -> Result<(), RhiError> {
        self.native.set_swap_interval(interval)
    }

    /// Size of a window surface in pixels, as the window system reports it.
    pub fn surface_size(&self, surface: &GlWindowSurface) -> Option<(u32, u32)> {
        self.native.surface_size(&surface.0)
    }

    /// Resizes a window surface's buffers (needed on Wayland; elsewhere surfaces follow the
    /// window).
    pub fn resize_window_surface(&self, surface: &GlWindowSurface, width: u32, height: u32) {
        self.native.resize_window_surface(&surface.0, width, height);
    }

    /// Posts the back buffer of `surface`, which must be the current draw surface.
    pub fn swap_buffers(&self, surface: &GlWindowSurface) -> Result<(), RhiError> {
        self.native.swap_buffers(&surface.0)
    }

    /// Destroys a window surface, making the offscreen surface current first if it was current.
    pub fn destroy_window_surface(&mut self, surface: GlWindowSurface) {
        self.native.destroy_window_surface(surface.0);
    }
}

/// Maps an OpenGL debug message type and severity to an RHI diagnostic severity; `None` means
/// the message is informational only (logged, not recorded).
fn classify_debug_message(gl_type: u32, severity: u32) -> Option<DiagnosticSeverity> {
    match gl_type {
        glow::DEBUG_TYPE_ERROR | glow::DEBUG_TYPE_UNDEFINED_BEHAVIOR => {
            Some(DiagnosticSeverity::Error)
        }
        glow::DEBUG_TYPE_DEPRECATED_BEHAVIOR
        | glow::DEBUG_TYPE_PORTABILITY
        | glow::DEBUG_TYPE_PERFORMANCE => Some(DiagnosticSeverity::Warning),
        glow::DEBUG_TYPE_MARKER | glow::DEBUG_TYPE_PUSH_GROUP | glow::DEBUG_TYPE_POP_GROUP => None,
        _ => match severity {
            glow::DEBUG_SEVERITY_HIGH | glow::DEBUG_SEVERITY_MEDIUM => {
                Some(DiagnosticSeverity::Warning)
            }
            _ => None,
        },
    }
}

/// OpenGL debug output callback body: logs through `tracing` and records warnings and errors.
fn record_debug_message(log: &DiagnosticLog, gl_type: u32, id: u32, severity: u32, message: &str) {
    match classify_debug_message(gl_type, severity) {
        Some(DiagnosticSeverity::Error) => {
            tracing::error!(target: "korppi_rhi_opengl::debug", id, "{message}");
            log.push(
                DiagnosticSeverity::Error,
                format!("GL debug id {id}: {message}"),
            );
        }
        Some(level) => {
            tracing::warn!(target: "korppi_rhi_opengl::debug", id, "{message}");
            log.push(level, format!("GL debug id {id}: {message}"));
        }
        None => tracing::debug!(target: "korppi_rhi_opengl::debug", id, "{message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_message_classification() {
        assert_eq!(
            classify_debug_message(glow::DEBUG_TYPE_ERROR, glow::DEBUG_SEVERITY_HIGH),
            Some(DiagnosticSeverity::Error)
        );
        assert_eq!(
            classify_debug_message(
                glow::DEBUG_TYPE_UNDEFINED_BEHAVIOR,
                glow::DEBUG_SEVERITY_LOW
            ),
            Some(DiagnosticSeverity::Error)
        );
        assert_eq!(
            classify_debug_message(glow::DEBUG_TYPE_PERFORMANCE, glow::DEBUG_SEVERITY_MEDIUM),
            Some(DiagnosticSeverity::Warning)
        );
        assert_eq!(
            classify_debug_message(glow::DEBUG_TYPE_OTHER, glow::DEBUG_SEVERITY_NOTIFICATION),
            None
        );
        assert_eq!(
            classify_debug_message(glow::DEBUG_TYPE_PUSH_GROUP, glow::DEBUG_SEVERITY_HIGH),
            None
        );
    }

    #[test]
    fn recorded_errors_reach_the_log() {
        let log = DiagnosticLog::new();
        record_debug_message(
            &log,
            glow::DEBUG_TYPE_ERROR,
            7,
            glow::DEBUG_SEVERITY_HIGH,
            "GL_INVALID_ENUM",
        );
        record_debug_message(
            &log,
            glow::DEBUG_TYPE_OTHER,
            8,
            glow::DEBUG_SEVERITY_NOTIFICATION,
            "buffer detailed info",
        );
        let entries = log.take();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].severity, DiagnosticSeverity::Error);
        assert!(entries[0].message.contains("GL_INVALID_ENUM"));
    }

    #[test]
    fn test_egl_headless_context_creation() {
        let ctx = GlContext::create_headless().expect("Headless EGL context creation failed");
        // SAFETY: the context is current on this thread.
        let version = unsafe { ctx.gl.get_parameter_string(glow::VERSION) };
        println!("EGL Headless GL Version: {}", version);
        assert!(!version.is_empty());
    }
}
