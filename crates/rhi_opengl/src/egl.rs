//! EGL contexts and window surfaces (Linux and other EGL platforms): headless (surfaceless
//! platform or the default display) or on an X11, XCB or Wayland display.

use crate::context::GlPlatform;
use khronos_egl as egl;
use korppi_rhi::RhiError;
use raw_window_handle::RawWindowHandle;
use std::ffi::{c_int, c_ulong, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

/// An EGL context on one display, with its offscreen surface.
pub(crate) struct EglContext {
    egl: Arc<egl::DynamicInstance<egl::EGL1_5>>,
    context: egl::Context,
    /// Config of the context; `None` for an `EGL_KHR_no_config_context` context.
    config: Option<egl::Config>,
    /// Offscreen surface current while no window surface is (`None` if the context can be
    /// current without a surface).
    surface: Option<egl::Surface>,
    /// Draw surface currently bound on the render thread.
    current_surface: Option<egl::Surface>,
    platform: GlPlatform,
    /// Declared last: the display reference is released after the context is destroyed.
    display: DisplayRef,
}

/// Initialized EGL displays and how many live contexts use each, keyed by `EGLDisplay` address.
///
/// `eglGetPlatformDisplay` returns the same `EGLDisplay` for every device in the process, and
/// `eglTerminate` invalidates it for all of them, so the display is terminated only when its last
/// user goes away (RGL-08, probe R10). Initialization and termination happen under this lock, so
/// a display cannot be terminated between another device's `eglInitialize` and its registration.
static DISPLAY_USERS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// One context's share of an initialized EGL display (see [`DISPLAY_USERS`]).
struct DisplayRef {
    egl: Arc<egl::DynamicInstance<egl::EGL1_5>>,
    display: egl::Display,
}

impl DisplayRef {
    /// Initializes `display` (a no-op if another context already did) and registers a user.
    fn acquire(
        egl: &Arc<egl::DynamicInstance<egl::EGL1_5>>,
        display: egl::Display,
    ) -> Result<(Self, (egl::Int, egl::Int)), RhiError> {
        let mut users = DISPLAY_USERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let version = egl
            .initialize(display)
            .map_err(|e| RhiError::InitializationFailed(format!("eglInitialize failed: {e}")))?;
        let key = display.as_ptr() as usize;
        match users.iter_mut().find(|(k, _)| *k == key) {
            Some((_, count)) => *count += 1,
            None => users.push((key, 1)),
        }
        Ok((
            Self {
                egl: Arc::clone(egl),
                display,
            },
            version,
        ))
    }
}

impl Drop for DisplayRef {
    fn drop(&mut self) {
        let mut users = DISPLAY_USERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let key = self.display.as_ptr() as usize;
        if let Some(index) = users.iter().position(|(k, _)| *k == key) {
            users[index].1 -= 1;
            if users[index].1 == 0 {
                users.swap_remove(index);
                let _ = self.egl.terminate(self.display);
            }
        }
    }
}

/// `EGL_PLATFORM_X11_KHR` (`EGL_KHR_platform_x11`): native display is an Xlib `Display*`.
const EGL_PLATFORM_X11_KHR: egl::Enum = 0x31D5;
/// `EGL_PLATFORM_X11_SCREEN_KHR`.
const EGL_PLATFORM_X11_SCREEN_KHR: egl::Attrib = 0x31D6;
/// `EGL_PLATFORM_XCB_EXT` (`EGL_EXT_platform_xcb`): native display is an `xcb_connection_t*`.
const EGL_PLATFORM_XCB_EXT: egl::Enum = 0x31DC;
/// `EGL_PLATFORM_XCB_SCREEN_EXT`.
const EGL_PLATFORM_XCB_SCREEN_EXT: egl::Attrib = 0x31DE;
/// `EGL_PLATFORM_WAYLAND_KHR` (`EGL_KHR_platform_wayland`): native display is a `wl_display*`.
const EGL_PLATFORM_WAYLAND_KHR: egl::Enum = 0x31D8;
/// `EGL_PLATFORM_SURFACELESS_MESA`: no window system.
const EGL_PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

/// `libwayland-egl` entry points, loaded on first use.
struct WaylandEgl {
    _library: libloading::Library,
    create: unsafe extern "C" fn(*mut c_void, c_int, c_int) -> *mut c_void,
    destroy: unsafe extern "C" fn(*mut c_void),
    resize: unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int, c_int),
}

impl WaylandEgl {
    fn get() -> Result<&'static Self, RhiError> {
        static WAYLAND_EGL: OnceLock<Result<WaylandEgl, String>> = OnceLock::new();
        WAYLAND_EGL
            .get_or_init(|| {
                // SAFETY: libwayland-egl has no library initializers with preconditions; the
                // symbols are looked up with their C signatures from `wayland-egl-core.h`, and
                // the library is kept loaded (in the static) as long as they are used.
                unsafe {
                    let library = libloading::Library::new("libwayland-egl.so.1")
                        .map_err(|e| format!("libwayland-egl.so.1: {e}"))?;
                    let create = *library
                        .get(b"wl_egl_window_create\0")
                        .map_err(|e| e.to_string())?;
                    let destroy = *library
                        .get(b"wl_egl_window_destroy\0")
                        .map_err(|e| e.to_string())?;
                    let resize = *library
                        .get(b"wl_egl_window_resize\0")
                        .map_err(|e| e.to_string())?;
                    Ok(WaylandEgl {
                        _library: library,
                        create,
                        destroy,
                        resize,
                    })
                }
            })
            .as_ref()
            .map_err(|e| RhiError::Unsupported(format!("Wayland EGL is unavailable: {e}")))
    }
}

/// An EGL window surface (plus its `wl_egl_window` on Wayland).
pub(crate) struct EglWindowSurface {
    surface: egl::Surface,
    wayland_window: Option<NonNull<c_void>>,
}

impl EglContext {
    /// Creates an OpenGL 4.5 core context on `platform`'s display and makes it current
    /// (offscreen) on this thread; a debug context if `debug` and the driver allows it.
    ///
    /// # Safety
    /// The native display in `platform` must stay valid until the context is dropped.
    pub(crate) unsafe fn create(platform: GlPlatform, debug: bool) -> Result<Self, RhiError> {
        // SAFETY: loading libEGL has no preconditions beyond the library being well-formed.
        let egl = unsafe {
            match egl::DynamicInstance::<egl::EGL1_5>::load_required() {
                Ok(inst) => Arc::new(inst),
                Err(e) => {
                    return Err(RhiError::InitializationFailed(format!(
                        "Failed to load libEGL.so: {e}"
                    )))
                }
            }
        };

        let display = match platform {
            GlPlatform::Surfaceless => {
                // SAFETY: the surfaceless platform takes no native display.
                let surfaceless = unsafe {
                    egl.get_platform_display(
                        EGL_PLATFORM_SURFACELESS_MESA,
                        egl::DEFAULT_DISPLAY,
                        &[egl::ATTRIB_NONE],
                    )
                };
                match surfaceless {
                    Ok(display) => display,
                    Err(_) => {
                        // SAFETY: EGL_DEFAULT_DISPLAY is always a valid argument.
                        unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }.ok_or_else(|| {
                            RhiError::InitializationFailed(
                                "eglGetDisplay(EGL_DEFAULT_DISPLAY) returned None".into(),
                            )
                        })?
                    }
                }
            }
            GlPlatform::Xlib { display, screen } => {
                // SAFETY: the caller guarantees `display` is a live Xlib `Display*`.
                unsafe {
                    egl.get_platform_display(
                        EGL_PLATFORM_X11_KHR,
                        display,
                        &[
                            EGL_PLATFORM_X11_SCREEN_KHR,
                            screen as egl::Attrib,
                            egl::ATTRIB_NONE,
                        ],
                    )
                }
                .map_err(|e| {
                    RhiError::InitializationFailed(format!("eglGetPlatformDisplay(X11): {e}"))
                })?
            }
            GlPlatform::Xcb { connection, screen } => {
                // SAFETY: the caller guarantees `connection` is a live `xcb_connection_t*`.
                unsafe {
                    egl.get_platform_display(
                        EGL_PLATFORM_XCB_EXT,
                        connection,
                        &[
                            EGL_PLATFORM_XCB_SCREEN_EXT,
                            screen as egl::Attrib,
                            egl::ATTRIB_NONE,
                        ],
                    )
                }
                .map_err(|e| {
                    RhiError::InitializationFailed(format!("eglGetPlatformDisplay(XCB): {e}"))
                })?
            }
            GlPlatform::Win32 => {
                return Err(RhiError::Unsupported(
                    "Win32 windows need a WGL context, which this build does not include".into(),
                ))
            }
            GlPlatform::Wayland { display } => {
                // SAFETY: the caller guarantees `display` is a live `wl_display*`.
                unsafe {
                    egl.get_platform_display(EGL_PLATFORM_WAYLAND_KHR, display, &[egl::ATTRIB_NONE])
                }
                .map_err(|e| {
                    RhiError::InitializationFailed(format!("eglGetPlatformDisplay(Wayland): {e}"))
                })?
            }
        };

        let (display_ref, (major, minor)) = DisplayRef::acquire(&egl, display)?;
        tracing::info!("Initialized EGL version {}.{}", major, minor);

        // Bind OpenGL API (desktop GL)
        egl.bind_api(egl::OPENGL_API).map_err(|e| {
            RhiError::InitializationFailed(format!("eglBindAPI(OPENGL_API) failed: {e}"))
        })?;

        let extensions = egl
            .query_string(Some(display), egl::EXTENSIONS)
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default();
        let has_extension = |name: &str| extensions.split_ascii_whitespace().any(|e| e == name);
        let surfaceless_current = has_extension("EGL_KHR_surfaceless_context");
        // A context without a config can be current with window surfaces of any config, so each
        // window gets a config matching its visual (EGL_KHR_no_config_context).
        let windowed = platform != GlPlatform::Surfaceless;
        let no_config = windowed && has_extension("EGL_KHR_no_config_context");

        let config = if no_config {
            None
        } else {
            let surface_type = if windowed {
                egl::WINDOW_BIT | egl::PBUFFER_BIT
            } else {
                egl::PBUFFER_BIT
            };
            Some(choose_config(&egl, display, surface_type)?)
        };

        // Context attributes: OpenGL 4.5 Core Profile, with a debug context when API validation
        // is requested (falls back to a non-debug context if the driver refuses one).
        let core_attribs = [
            egl::CONTEXT_MAJOR_VERSION,
            4,
            egl::CONTEXT_MINOR_VERSION,
            5,
            egl::CONTEXT_OPENGL_PROFILE_MASK,
            egl::CONTEXT_OPENGL_CORE_PROFILE_BIT,
        ];
        let mut debug_attribs = core_attribs.to_vec();
        debug_attribs.extend([egl::CONTEXT_OPENGL_DEBUG, 1, egl::NONE]);
        let mut plain_attribs = core_attribs.to_vec();
        plain_attribs.push(egl::NONE);

        // SAFETY: a null config is EGL_NO_CONFIG_KHR, valid with EGL_KHR_no_config_context.
        let context_config =
            config.unwrap_or_else(|| unsafe { egl::Config::from_ptr(std::ptr::null_mut()) });
        let debug_context = if debug {
            egl.create_context(display, context_config, None, &debug_attribs)
                .map_err(|e| tracing::warn!("eglCreateContext with a debug context failed: {e}"))
                .ok()
        } else {
            None
        };
        let context = match debug_context {
            Some(context) => context,
            None => egl
                .create_context(display, context_config, None, &plain_attribs)
                .map_err(|e| {
                    RhiError::InitializationFailed(format!(
                        "eglCreateContext(OpenGL 4.5 Core) failed: {e}"
                    ))
                })?,
        };

        // Without a window surface current, render offscreen: surfaceless if supported,
        // otherwise into a 1x1 pbuffer.
        let pbuffer = if surfaceless_current {
            None
        } else {
            let pbuffer_config = match config {
                Some(config) => config,
                None => choose_config(&egl, display, egl::PBUFFER_BIT)?,
            };
            egl.create_pbuffer_surface(
                display,
                pbuffer_config,
                &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE],
            )
            .ok()
        };

        egl.make_current(display, pbuffer, pbuffer, Some(context))
            .map_err(|e| RhiError::InitializationFailed(format!("eglMakeCurrent failed: {e}")))?;

        Ok(Self {
            egl,
            context,
            config,
            surface: pbuffer,
            current_surface: pbuffer,
            platform,
            display: display_ref,
        })
    }

    /// The address of an OpenGL or EGL function, or null.
    pub(crate) fn get_proc_address(&self, name: &str) -> *const c_void {
        self.egl
            .get_proc_address(name)
            .map_or(std::ptr::null(), |p| p as *const c_void)
    }

    /// Creates a window surface for `window`, which must belong to this context's display.
    ///
    /// # Safety
    /// `window` must stay valid until the surface is destroyed with
    /// [`EglContext::destroy_window_surface`].
    pub(crate) unsafe fn create_window_surface(
        &mut self,
        window: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<EglWindowSurface, RhiError> {
        let display = self.display.display;
        // EGL_KHR_platform_x11 and EGL_EXT_platform_xcb take a pointer to the window ID;
        // EGL_KHR_platform_wayland takes a `wl_egl_window*`.
        let mut xlib_window: c_ulong = 0;
        let mut xcb_window: u32 = 0;
        let mut wayland_window = None;
        let (native, visual_id): (*mut c_void, u64) = match (self.platform, window) {
            (GlPlatform::Xlib { .. }, RawWindowHandle::Xlib(h)) => {
                xlib_window = h.window;
                // `VisualID` is a C `unsigned long`: 32 bits on 32-bit targets.
                #[allow(clippy::useless_conversion)]
                let visual_id = u64::from(h.visual_id);
                ((&raw mut xlib_window).cast(), visual_id)
            }
            (GlPlatform::Xcb { .. }, RawWindowHandle::Xcb(h)) => {
                xcb_window = h.window.get();
                (
                    (&raw mut xcb_window).cast(),
                    h.visual_id.map_or(0, |v| u64::from(v.get())),
                )
            }
            (GlPlatform::Wayland { .. }, RawWindowHandle::Wayland(h)) => {
                let wayland = WaylandEgl::get()?;
                // SAFETY: the caller guarantees `h.surface` is a live `wl_surface*`.
                let window = unsafe {
                    (wayland.create)(
                        h.surface.as_ptr(),
                        width.max(1) as c_int,
                        height.max(1) as c_int,
                    )
                };
                let window = NonNull::new(window).ok_or_else(|| {
                    RhiError::SwapchainError("wl_egl_window_create failed".into())
                })?;
                wayland_window = Some(window);
                (window.as_ptr(), 0)
            }
            (platform, window) => {
                return Err(RhiError::Unsupported(format!(
                    "window {window:?} does not belong to this device's display ({platform:?})"
                )))
            }
        };
        let destroy_wayland_window = |window: Option<NonNull<c_void>>| {
            if let Some(window) = window {
                if let Ok(wayland) = WaylandEgl::get() {
                    // SAFETY: the window was created above and has no EGL surface.
                    unsafe { (wayland.destroy)(window.as_ptr()) };
                }
            }
        };
        let config = match self.config {
            Some(config) => config,
            None => match choose_window_config(&self.egl, display, visual_id) {
                Ok(config) => config,
                Err(error) => {
                    destroy_wayland_window(wayland_window);
                    return Err(error);
                }
            },
        };
        // SAFETY: `native` points to a window of this display's platform, in the form its EGL
        // platform extension requires; the caller keeps the window alive.
        let surface = unsafe {
            self.egl
                .create_platform_window_surface(display, config, native, &[egl::ATTRIB_NONE])
        };
        let _ = (xlib_window, xcb_window);
        match surface {
            Ok(surface) => Ok(EglWindowSurface {
                surface,
                wayland_window,
            }),
            Err(error) => {
                destroy_wayland_window(wayland_window);
                Err(RhiError::SwapchainError(format!(
                    "eglCreatePlatformWindowSurface failed: {error}"
                )))
            }
        }
    }

    /// Makes `surface` (or, with `None`, the offscreen surface) the context's draw surface.
    pub(crate) fn make_current(
        &mut self,
        surface: Option<&EglWindowSurface>,
    ) -> Result<(), RhiError> {
        let target = surface.map(|s| s.surface).or(self.surface);
        if self.current_surface == target {
            return Ok(());
        }
        self.egl
            .make_current(self.display.display, target, target, Some(self.context))
            .map_err(|e| RhiError::SwapchainError(format!("eglMakeCurrent failed: {e}")))?;
        self.current_surface = target;
        Ok(())
    }

    /// Sets the swap interval of the current window surface (1 = vsync, 0 = immediate).
    pub(crate) fn set_swap_interval(&self, interval: i32) -> Result<(), RhiError> {
        self.egl
            .swap_interval(self.display.display, interval)
            .map_err(|e| RhiError::SwapchainError(format!("eglSwapInterval failed: {e}")))
    }

    /// Size of a window surface in pixels, as EGL reports it.
    pub(crate) fn surface_size(&self, surface: &EglWindowSurface) -> Option<(u32, u32)> {
        let width = self
            .egl
            .query_surface(self.display.display, surface.surface, egl::WIDTH)
            .ok()?;
        let height = self
            .egl
            .query_surface(self.display.display, surface.surface, egl::HEIGHT)
            .ok()?;
        Some((width.max(0) as u32, height.max(0) as u32))
    }

    /// Resizes a window surface's buffers (needed on Wayland; X11 surfaces follow the window).
    pub(crate) fn resize_window_surface(
        &self,
        surface: &EglWindowSurface,
        width: u32,
        height: u32,
    ) {
        if let (Some(window), Ok(wayland)) = (surface.wayland_window, WaylandEgl::get()) {
            // SAFETY: the `wl_egl_window` is live until the surface is destroyed.
            unsafe {
                (wayland.resize)(
                    window.as_ptr(),
                    width.max(1) as c_int,
                    height.max(1) as c_int,
                    0,
                    0,
                )
            };
        }
    }

    /// Posts the back buffer of `surface`, which must be the current draw surface.
    pub(crate) fn swap_buffers(&self, surface: &EglWindowSurface) -> Result<(), RhiError> {
        self.egl
            .swap_buffers(self.display.display, surface.surface)
            .map_err(|e| RhiError::SwapchainError(format!("eglSwapBuffers failed: {e}")))
    }

    /// Destroys a window surface, making the offscreen surface current first if it was current.
    pub(crate) fn destroy_window_surface(&mut self, surface: EglWindowSurface) {
        if self.current_surface == Some(surface.surface) {
            if let Err(error) = self.make_current(None) {
                tracing::error!("releasing a window surface: {error}");
            }
        }
        let _ = self
            .egl
            .destroy_surface(self.display.display, surface.surface);
        if let (Some(window), Ok(wayland)) = (surface.wayland_window, WaylandEgl::get()) {
            // SAFETY: its EGL surface was destroyed above.
            unsafe { (wayland.destroy)(window.as_ptr()) };
        }
    }
}

/// An RGB8 config with `surface_type` and desktop OpenGL support.
fn choose_config(
    egl: &egl::DynamicInstance<egl::EGL1_5>,
    display: egl::Display,
    surface_type: egl::Int,
) -> Result<egl::Config, RhiError> {
    let attribs = [
        egl::SURFACE_TYPE,
        surface_type,
        egl::RENDERABLE_TYPE,
        egl::OPENGL_BIT,
        egl::RED_SIZE,
        8,
        egl::GREEN_SIZE,
        8,
        egl::BLUE_SIZE,
        8,
        egl::NONE,
    ];
    egl.choose_first_config(display, &attribs)
        .map_err(|e| RhiError::InitializationFailed(format!("eglChooseConfig failed: {e}")))?
        .ok_or_else(|| RhiError::InitializationFailed("No compatible EGL configs found".into()))
}

/// A window config for a window with native visual `visual_id` (0 if unknown), preferring one
/// without alpha so that compositors treat the window as opaque.
fn choose_window_config(
    egl: &egl::DynamicInstance<egl::EGL1_5>,
    display: egl::Display,
    visual_id: u64,
) -> Result<egl::Config, RhiError> {
    let attribs = [
        egl::SURFACE_TYPE,
        egl::WINDOW_BIT,
        egl::RENDERABLE_TYPE,
        egl::OPENGL_BIT,
        egl::RED_SIZE,
        8,
        egl::GREEN_SIZE,
        8,
        egl::BLUE_SIZE,
        8,
        egl::NONE,
    ];
    let mut configs = Vec::with_capacity(64);
    egl.choose_config(display, &attribs, &mut configs)
        .map_err(|e| RhiError::SwapchainError(format!("eglChooseConfig failed: {e}")))?;
    let attrib =
        |config: egl::Config, name| egl.get_config_attrib(display, config, name).unwrap_or(0);
    let matches_visual = |config: egl::Config| {
        visual_id == 0 || attrib(config, egl::NATIVE_VISUAL_ID) as u64 == visual_id
    };
    configs
        .iter()
        .copied()
        .filter(|&c| matches_visual(c))
        .min_by_key(|&c| attrib(c, egl::ALPHA_SIZE))
        .ok_or_else(|| {
            RhiError::SwapchainError(format!(
                "no EGL window config matches the window's visual {visual_id:#x}"
            ))
        })
}

impl Drop for EglContext {
    /// Releases the context only if it is current on this thread (never another device's), then
    /// destroys it. The display is terminated by `DisplayRef` once no context uses it.
    fn drop(&mut self) {
        let display = self.display.display;
        if self.egl.get_current_context() == Some(self.context) {
            let _ = self.egl.make_current(display, None, None, None);
        }
        if let Some(surface) = self.surface {
            let _ = self.egl.destroy_surface(display, surface);
        }
        let _ = self.egl.destroy_context(display, self.context);
        // Frees EGL's per-thread state, unless another context is current on this thread.
        if self.egl.get_current_context().is_none() {
            let _ = self.egl.release_thread();
        }
    }
}
