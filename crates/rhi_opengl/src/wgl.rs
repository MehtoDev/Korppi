//! WGL contexts and window surfaces (Windows; finding RGL-07).
//!
//! WGL needs a window to create a context, so the context is created on a hidden window that the
//! render thread owns, and is current with that window's device context while no swapchain
//! window is. The standard bootstrap is used: a legacy context on the hidden window loads
//! `wglCreateContextAttribsARB` (`WGL_ARB_create_context`), which creates the OpenGL 4.5 core
//! context; the legacy context is then deleted. Every swapchain window gets the hidden window's
//! pixel format so the context can be current with it, and presents with `SwapBuffers`.
//!
//! This module is type-checked on Windows targets (`cargo check --target x86_64-pc-windows-gnu`)
//! but has not been run: the development machine has no Windows installation
//! (`docs/project/completed/fix-1-rhi-remediation.md`, FIX-1.3).

use crate::context::GlPlatform;
use korppi_rhi::RhiError;
use raw_window_handle::RawWindowHandle;
use std::ffi::{c_void, CString};
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE, HWND, RECT};
use windows_sys::Win32::Graphics::Gdi::{GetDC, ReleaseDC, HDC};
use windows_sys::Win32::Graphics::OpenGL::{
    wglCreateContext, wglDeleteContext, wglGetCurrentContext, wglGetProcAddress, wglMakeCurrent,
    ChoosePixelFormat, GetPixelFormat, SetPixelFormat, SwapBuffers, HGLRC, PFD_DOUBLEBUFFER,
    PFD_DRAW_TO_WINDOW, PFD_MAIN_PLANE, PFD_SUPPORT_OPENGL, PFD_TYPE_RGBA, PIXELFORMATDESCRIPTOR,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryA};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, GetClientRect, RegisterClassExW, CS_OWNDC,
    WNDCLASSEXW, WS_OVERLAPPEDWINDOW,
};

const WGL_CONTEXT_MAJOR_VERSION_ARB: i32 = 0x2091;
const WGL_CONTEXT_MINOR_VERSION_ARB: i32 = 0x2092;
const WGL_CONTEXT_FLAGS_ARB: i32 = 0x2094;
const WGL_CONTEXT_PROFILE_MASK_ARB: i32 = 0x9126;
const WGL_CONTEXT_CORE_PROFILE_BIT_ARB: i32 = 0x0001;
const WGL_CONTEXT_DEBUG_BIT_ARB: i32 = 0x0001;

type CreateContextAttribsArb =
    unsafe extern "system" fn(hdc: HDC, share: HGLRC, attribs: *const i32) -> HGLRC;
type SwapIntervalExt = unsafe extern "system" fn(interval: i32) -> i32;

/// UTF-16, NUL-terminated name of the hidden window's class.
fn class_name() -> &'static [u16] {
    static NAME: OnceLock<Vec<u16>> = OnceLock::new();
    NAME.get_or_init(|| "KorppiGlContextWindow\0".encode_utf16().collect())
}

/// The pixel format every window the context renders to uses: 32-bit RGBA, double-buffered.
fn pixel_format_descriptor() -> PIXELFORMATDESCRIPTOR {
    PIXELFORMATDESCRIPTOR {
        nSize: std::mem::size_of::<PIXELFORMATDESCRIPTOR>() as u16,
        nVersion: 1,
        dwFlags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        iPixelType: PFD_TYPE_RGBA,
        cColorBits: 32,
        cRedBits: 0,
        cRedShift: 0,
        cGreenBits: 0,
        cGreenShift: 0,
        cBlueBits: 0,
        cBlueShift: 0,
        cAlphaBits: 8,
        cAlphaShift: 0,
        cAccumBits: 0,
        cAccumRedBits: 0,
        cAccumGreenBits: 0,
        cAccumBlueBits: 0,
        cAccumAlphaBits: 0,
        cDepthBits: 0,
        cStencilBits: 0,
        cAuxBuffers: 0,
        iLayerType: PFD_MAIN_PLANE as u8,
        bReserved: 0,
        dwLayerMask: 0,
        dwVisibleMask: 0,
        dwDamageMask: 0,
    }
}

/// A WGL context, current on the thread that created it, with its hidden window.
pub(crate) struct WglContext {
    opengl32: HMODULE,
    window: HWND,
    window_dc: HDC,
    context: HGLRC,
    current_dc: HDC,
    pixel_format: i32,
    swap_interval: Option<SwapIntervalExt>,
}

/// A swapchain window's device context.
pub(crate) struct WglWindowSurface {
    hwnd: HWND,
    hdc: HDC,
}

fn error(what: &str) -> RhiError {
    // SAFETY: `GetLastError` has no preconditions.
    let code = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    RhiError::InitializationFailed(format!("{what} failed (Win32 error {code})"))
}

impl WglContext {
    /// Creates an OpenGL 4.5 core context (a debug context if `debug` and the driver allows it)
    /// and makes it current on this thread.
    ///
    /// # Safety
    /// None beyond the platform being `Surfaceless` or `Win32` (other platforms are rejected).
    pub(crate) unsafe fn create(platform: GlPlatform, debug: bool) -> Result<Self, RhiError> {
        if !matches!(platform, GlPlatform::Surfaceless | GlPlatform::Win32) {
            return Err(RhiError::Unsupported(format!(
                "WGL contexts present to Win32 windows, not {platform:?}"
            )));
        }
        // SAFETY: plain Win32 calls with valid arguments; every object created here is owned by
        // the returned `WglContext` (or released on the error paths below).
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            let class = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_OWNDC,
                lpfnWndProc: Some(DefWindowProcW),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: instance,
                hIcon: std::ptr::null_mut(),
                hCursor: std::ptr::null_mut(),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class_name().as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            // Fails harmlessly with ERROR_CLASS_ALREADY_EXISTS for later devices.
            RegisterClassExW(&class);
            let window = CreateWindowExW(
                0,
                class_name().as_ptr(),
                class_name().as_ptr(),
                WS_OVERLAPPEDWINDOW,
                0,
                0,
                1,
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                instance,
                std::ptr::null(),
            );
            if window.is_null() {
                return Err(error("CreateWindowExW"));
            }
            let window_dc = GetDC(window);
            let descriptor = pixel_format_descriptor();
            let pixel_format = ChoosePixelFormat(window_dc, &descriptor);
            if pixel_format == 0 || SetPixelFormat(window_dc, pixel_format, &descriptor) == 0 {
                let err = error("choosing the context window's pixel format");
                ReleaseDC(window, window_dc);
                DestroyWindow(window);
                return Err(err);
            }
            let cleanup = |context: HGLRC| {
                if !context.is_null() {
                    wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
                    wglDeleteContext(context);
                }
                ReleaseDC(window, window_dc);
                DestroyWindow(window);
            };

            // A legacy context loads wglCreateContextAttribsARB.
            let legacy = wglCreateContext(window_dc);
            if legacy.is_null() || wglMakeCurrent(window_dc, legacy) == 0 {
                let err = error("wglCreateContext");
                cleanup(legacy);
                return Err(err);
            }
            let create_attribs = wglGetProcAddress(c"wglCreateContextAttribsARB".as_ptr().cast());
            let Some(create_attribs) = create_attribs else {
                cleanup(legacy);
                return Err(RhiError::InitializationFailed(
                    "WGL_ARB_create_context is unavailable; OpenGL 4.5 core needs it".into(),
                ));
            };
            // SAFETY: the address belongs to `wglCreateContextAttribsARB`, whose signature is
            // `HGLRC (HDC, HGLRC, const int*)`.
            let create_attribs: CreateContextAttribsArb = std::mem::transmute(create_attribs);
            let attribs = |flags: i32| {
                [
                    WGL_CONTEXT_MAJOR_VERSION_ARB,
                    4,
                    WGL_CONTEXT_MINOR_VERSION_ARB,
                    5,
                    WGL_CONTEXT_PROFILE_MASK_ARB,
                    WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                    WGL_CONTEXT_FLAGS_ARB,
                    flags,
                    0,
                ]
            };
            let mut context = std::ptr::null_mut();
            if debug {
                context = create_attribs(
                    window_dc,
                    std::ptr::null_mut(),
                    attribs(WGL_CONTEXT_DEBUG_BIT_ARB).as_ptr(),
                );
                if context.is_null() {
                    tracing::warn!("creating a debug WGL context failed; using a regular one");
                }
            }
            if context.is_null() {
                context = create_attribs(window_dc, std::ptr::null_mut(), attribs(0).as_ptr());
            }
            if context.is_null() || wglMakeCurrent(window_dc, context) == 0 {
                let err = error("wglCreateContextAttribsARB(OpenGL 4.5 core)");
                wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
                if !context.is_null() {
                    wglDeleteContext(context);
                }
                cleanup(legacy);
                return Err(err);
            }
            wglDeleteContext(legacy);

            let swap_interval = wglGetProcAddress(c"wglSwapIntervalEXT".as_ptr().cast())
                // SAFETY: the address belongs to `wglSwapIntervalEXT`, `BOOL (int)`.
                .map(|f| std::mem::transmute::<_, SwapIntervalExt>(f));
            let opengl32 = LoadLibraryA(c"opengl32.dll".as_ptr().cast());
            Ok(Self {
                opengl32,
                window,
                window_dc,
                context,
                current_dc: window_dc,
                pixel_format,
                swap_interval,
            })
        }
    }

    /// The address of an OpenGL function, or null: `wglGetProcAddress` for everything newer than
    /// OpenGL 1.1, `opengl32.dll` exports for 1.1 functions.
    pub(crate) fn get_proc_address(&self, name: &str) -> *const c_void {
        let Ok(name) = CString::new(name) else {
            return std::ptr::null();
        };
        // SAFETY: the context is current on this thread; `name` is NUL-terminated.
        let wgl = unsafe { wglGetProcAddress(name.as_ptr().cast()) };
        // wglGetProcAddress returns small sentinel values (1, 2, 3, -1) on some drivers instead
        // of null.
        match wgl.map(|f| f as usize) {
            Some(address) if !matches!(address, 1 | 2 | 3 | usize::MAX) => address as *const c_void,
            _ if !self.opengl32.is_null() => {
                // SAFETY: `opengl32` is a loaded module; `name` is NUL-terminated.
                unsafe { GetProcAddress(self.opengl32, name.as_ptr().cast()) }
                    .map_or(std::ptr::null(), |f| f as *const c_void)
            }
            _ => std::ptr::null(),
        }
    }

    /// Gives `window` the context's pixel format (if it has none yet) and returns its device
    /// context.
    ///
    /// # Safety
    /// `window` must stay valid until the surface is destroyed.
    pub(crate) unsafe fn create_window_surface(
        &mut self,
        window: RawWindowHandle,
        _width: u32,
        _height: u32,
    ) -> Result<WglWindowSurface, RhiError> {
        let RawWindowHandle::Win32(handle) = window else {
            return Err(RhiError::Unsupported(format!(
                "window {window:?} is not a Win32 window"
            )));
        };
        let hwnd = handle.hwnd.get() as HWND;
        // SAFETY: the caller guarantees `hwnd` is a live window; the device context is released
        // in `destroy_window_surface`.
        unsafe {
            let hdc = GetDC(hwnd);
            if hdc.is_null() {
                return Err(error("GetDC"));
            }
            let current = GetPixelFormat(hdc);
            if current == 0 {
                let descriptor = pixel_format_descriptor();
                if SetPixelFormat(hdc, self.pixel_format, &descriptor) == 0 {
                    let err = error("SetPixelFormat on the swapchain window");
                    ReleaseDC(hwnd, hdc);
                    return Err(err);
                }
            } else if current != self.pixel_format {
                ReleaseDC(hwnd, hdc);
                return Err(RhiError::SwapchainError(format!(
                    "the window already has pixel format {current}, which this context ({}) \
                     cannot render to",
                    self.pixel_format
                )));
            }
            Ok(WglWindowSurface { hwnd, hdc })
        }
    }

    /// Makes `surface` (or, with `None`, the hidden window) the context's draw surface.
    pub(crate) fn make_current(
        &mut self,
        surface: Option<&WglWindowSurface>,
    ) -> Result<(), RhiError> {
        let dc = surface.map_or(self.window_dc, |s| s.hdc);
        if dc == self.current_dc {
            return Ok(());
        }
        // SAFETY: `dc` has the context's pixel format; the context belongs to this thread.
        if unsafe { wglMakeCurrent(dc, self.context) } == 0 {
            return Err(error("wglMakeCurrent"));
        }
        self.current_dc = dc;
        Ok(())
    }

    /// Sets the swap interval of the current window (`WGL_EXT_swap_control`).
    pub(crate) fn set_swap_interval(&self, interval: i32) -> Result<(), RhiError> {
        let Some(swap_interval) = self.swap_interval else {
            return Err(RhiError::Unsupported(
                "WGL_EXT_swap_control is unavailable".into(),
            ));
        };
        // SAFETY: a context is current on this thread.
        if unsafe { swap_interval(interval) } == 0 {
            return Err(error("wglSwapIntervalEXT"));
        }
        Ok(())
    }

    /// The window's client-area size in pixels.
    pub(crate) fn surface_size(&self, surface: &WglWindowSurface) -> Option<(u32, u32)> {
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        // SAFETY: the window is live while the surface exists; `rect` is a valid out pointer.
        if unsafe { GetClientRect(surface.hwnd, &mut rect) } == 0 {
            return None;
        }
        Some((
            (rect.right - rect.left).max(0) as u32,
            (rect.bottom - rect.top).max(0) as u32,
        ))
    }

    /// Win32 window surfaces follow the window's size; nothing to do.
    pub(crate) fn resize_window_surface(
        &self,
        _surface: &WglWindowSurface,
        _width: u32,
        _height: u32,
    ) {
    }

    /// Presents the window's back buffer.
    pub(crate) fn swap_buffers(&self, surface: &WglWindowSurface) -> Result<(), RhiError> {
        // SAFETY: the device context is live while the surface exists.
        if unsafe { SwapBuffers(surface.hdc) } == 0 {
            return Err(RhiError::SwapchainError(format!(
                "SwapBuffers failed: {:?}",
                error("SwapBuffers")
            )));
        }
        Ok(())
    }

    /// Releases the window's device context, making the hidden window current first if needed.
    pub(crate) fn destroy_window_surface(&mut self, surface: WglWindowSurface) {
        if self.current_dc == surface.hdc {
            if let Err(error) = self.make_current(None) {
                tracing::error!("releasing a window surface: {error}");
            }
        }
        // SAFETY: the device context was obtained with `GetDC` for this window.
        unsafe { ReleaseDC(surface.hwnd, surface.hdc) };
    }
}

impl Drop for WglContext {
    fn drop(&mut self) {
        // SAFETY: the context and window belong to this thread (the render thread) and are
        // released exactly once.
        unsafe {
            if wglGetCurrentContext() == self.context {
                wglMakeCurrent(std::ptr::null_mut(), std::ptr::null_mut());
            }
            wglDeleteContext(self.context);
            ReleaseDC(self.window, self.window_dc);
            DestroyWindow(self.window);
            if !self.opengl32.is_null() {
                FreeLibrary(self.opengl32);
            }
        }
    }
}
