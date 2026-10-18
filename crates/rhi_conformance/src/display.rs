//! X11 windows for presentation tests, on the user's display or a virtual one (Xvfb).
//!
//! [`VirtualDisplay::open`] uses `$DISPLAY` when it is set and otherwise starts `Xvfb`, so the
//! presentation tests run on headless machines too. When neither is available the tests skip
//! themselves (see [`display_or_skip`]); set `KORPPI_REQUIRE_DISPLAY=1` to make that a failure.
//!
//! [`TestWindow`] is a plain Xlib window (the same handle types `winit` produces on X11) that can
//! read back what was presented into it with `XGetImage`. It has a fixed size (equal minimum and
//! maximum size hints), so a tiling window manager or Wayland compositor on the user's display
//! floats it at that size instead of tiling it (ISSUE-011). X errors fail the test that caused
//! them, instead of ending the process as Xlib's default error handler does.
//!
//! [`WaylandCompositor`] does the same for Wayland: `$WAYLAND_DISPLAY`, or a headless Weston.
//! [`WaylandTestWindow`] is a fixed-size `xdg_toplevel` on it, without readback.

use crate::Image;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WindowHandle, XlibDisplayHandle, XlibWindowHandle,
};
use std::cell::Cell;
use std::ffi::{c_int, CString};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::ptr::NonNull;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use x11_dl::xlib;

mod wayland_window;
pub use wayland_window::WaylandTestWindow;

/// Environment variable that turns a missing display into a test failure instead of a skip.
pub const REQUIRE_DISPLAY_ENV_VAR: &str = "KORPPI_REQUIRE_DISPLAY";

/// An X display to open test windows on: `$DISPLAY`, or an `Xvfb` server owned by this value.
pub struct VirtualDisplay {
    name: String,
    server: Option<Child>,
}

impl VirtualDisplay {
    /// Returns `$DISPLAY` if it is set, otherwise starts `Xvfb` on a free display number.
    /// Returns `None` if there is no display and `Xvfb` cannot be started.
    pub fn open() -> Option<Self> {
        if let Some(name) = std::env::var("DISPLAY").ok().filter(|d| !d.is_empty()) {
            return Some(Self { name, server: None });
        }
        let mut server = Command::new("Xvfb")
            .args([
                "-displayfd",
                "1",
                "-screen",
                "0",
                "2560x1440x24",
                "-nolisten",
                "tcp",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        // Xvfb writes the display number to the `-displayfd` descriptor once it accepts
        // connections.
        let mut line = String::new();
        let read = server
            .stdout
            .take()
            .map(|stdout| BufReader::new(stdout).read_line(&mut line));
        match (read, line.trim().parse::<u32>()) {
            (Some(Ok(_)), Ok(number)) => Some(Self {
                name: format!(":{number}"),
                server: Some(server),
            }),
            _ => {
                let _ = server.kill();
                let _ = server.wait();
                None
            }
        }
    }

    /// The display name, such as `:1`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether this display is an `Xvfb` server started for the test.
    pub fn is_virtual(&self) -> bool {
        self.server.is_some()
    }
}

impl Drop for VirtualDisplay {
    fn drop(&mut self) {
        if let Some(server) = &mut self.server {
            // SIGTERM lets Xvfb remove its lock file and socket; SIGKILL left them behind for
            // the next server to trip over. Fall back to SIGKILL if `kill` is unavailable.
            let terminated = Command::new("kill")
                .args(["-TERM", &server.id().to_string()])
                .status()
                .is_ok_and(|status| status.success());
            if !terminated {
                let _ = server.kill();
            }
            let _ = server.wait();
        }
    }
}

/// Opens a display for `test`, or returns `None` (after printing why) when the test should be
/// skipped.
///
/// # Panics
/// Panics instead of skipping when `KORPPI_REQUIRE_DISPLAY` is set.
pub fn display_or_skip(test: &str) -> Option<VirtualDisplay> {
    let display = VirtualDisplay::open();
    if display.is_none() {
        let required = std::env::var_os(REQUIRE_DISPLAY_ENV_VAR).is_some_and(|v| v != "0");
        assert!(
            !required,
            "{test}: no X display and Xvfb could not be started ({REQUIRE_DISPLAY_ENV_VAR} is set)"
        );
        eprintln!("SKIPPED {test}: no X display ($DISPLAY unset) and Xvfb is not installed");
    }
    display
}

/// A Wayland compositor to run clients on: `$WAYLAND_DISPLAY`, or a headless Weston owned by
/// this value.
pub struct WaylandCompositor {
    socket: String,
    server: Option<Child>,
}

impl WaylandCompositor {
    /// Returns `$WAYLAND_DISPLAY` if it is set, otherwise starts `weston --backend=headless` on
    /// a new socket in `$XDG_RUNTIME_DIR`. Returns `None` if neither works.
    pub fn open() -> Option<Self> {
        if let Some(socket) = std::env::var("WAYLAND_DISPLAY")
            .ok()
            .filter(|d| !d.is_empty())
        {
            return Some(Self {
                socket,
                server: None,
            });
        }
        let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR")?;
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let socket = format!(
            "korppi-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let mut server = Command::new("weston")
            .args([
                "--backend=headless",
                "--renderer=pixman",
                "--shell=kiosk",
                "--no-config",
                "--width=1024",
                "--height=768",
                &format!("--socket={socket}"),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        // Weston creates the socket once it accepts clients.
        let path = std::path::Path::new(&runtime_dir).join(&socket);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !path.exists() {
            if Instant::now() > deadline || server.try_wait().ok().flatten().is_some() {
                let _ = server.kill();
                let _ = server.wait();
                return None;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Some(Self {
            socket,
            server: Some(server),
        })
    }

    /// The socket name to put in `WAYLAND_DISPLAY`.
    pub fn socket(&self) -> &str {
        &self.socket
    }
}

impl Drop for WaylandCompositor {
    fn drop(&mut self) {
        if let Some(server) = &mut self.server {
            let terminated = Command::new("kill")
                .args(["-TERM", &server.id().to_string()])
                .status()
                .is_ok_and(|status| status.success());
            if !terminated {
                let _ = server.kill();
            }
            let _ = server.wait();
        }
    }
}

/// Opens a Wayland compositor for `test`, or returns `None` (after printing why) when the test
/// should be skipped.
///
/// # Panics
/// Panics instead of skipping when `KORPPI_REQUIRE_DISPLAY` is set.
pub fn wayland_or_skip(test: &str) -> Option<WaylandCompositor> {
    let compositor = WaylandCompositor::open();
    if compositor.is_none() {
        let required = std::env::var_os(REQUIRE_DISPLAY_ENV_VAR).is_some_and(|v| v != "0");
        assert!(
            !required,
            "{test}: no Wayland compositor and Weston could not be started \
             ({REQUIRE_DISPLAY_ENV_VAR} is set)"
        );
        eprintln!("SKIPPED {test}: no Wayland compositor ($WAYLAND_DISPLAY unset, no weston)");
    }
    compositor
}

/// Xlib, loaded once and initialized for multi-threaded use before any other Xlib call, with
/// [`record_x_error`] as the error handler.
fn xlib() -> &'static xlib::Xlib {
    static XLIB: OnceLock<xlib::Xlib> = OnceLock::new();
    XLIB.get_or_init(|| {
        let xlib = xlib::Xlib::open().expect("libX11 must be installed to open test windows");
        // SAFETY: `XInitThreads` has no preconditions; running it before any other Xlib call of
        // this process (guaranteed by the `OnceLock`) makes every later call thread-safe, which
        // the render threads of the backends rely on. The error handler is a plain function that
        // never unwinds.
        unsafe {
            (xlib.XInitThreads)();
            (xlib.XSetErrorHandler)(Some(record_x_error));
        }
        xlib
    })
}

thread_local! {
    /// The error code of the last X error reported on this thread, if any.
    static LAST_X_ERROR: Cell<Option<u8>> = const { Cell::new(None) };
}

/// X error handler: reports the error on stderr and records it for the thread whose request
/// failed (Xlib calls the handler on that thread), which then fails its test. Xlib's default
/// handler would end the process, under the other tests' threads.
unsafe extern "C" fn record_x_error(
    _display: *mut xlib::Display,
    event: *mut xlib::XErrorEvent,
) -> c_int {
    // SAFETY: Xlib passes a valid error event for the duration of the call.
    if let Some(event) = unsafe { event.as_ref() } {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "X error {} ({}) for request {}.{}",
            event.error_code,
            x_error_name(event.error_code),
            event.request_code,
            event.minor_code
        );
        LAST_X_ERROR.set(Some(event.error_code));
    }
    0
}

/// The protocol name of core X error `code`.
fn x_error_name(code: u8) -> &'static str {
    const NAMES: [&str; 18] = [
        "Success",
        "BadRequest",
        "BadValue",
        "BadWindow",
        "BadPixmap",
        "BadAtom",
        "BadCursor",
        "BadFont",
        "BadMatch",
        "BadDrawable",
        "BadAccess",
        "BadAlloc",
        "BadColor",
        "BadGC",
        "BadIDChoice",
        "BadName",
        "BadLength",
        "BadImplementation",
    ];
    NAMES
        .get(code as usize)
        .copied()
        .unwrap_or("extension error")
}

/// A mapped X11 window with its own display connection.
///
/// `Send + Sync`: Xlib was initialized with `XInitThreads`, so the connection may be used from
/// any thread (the OpenGL backend's render thread presents through it).
pub struct TestWindow {
    display: NonNull<xlib::Display>,
    screen: i32,
    window: xlib::Window,
    visual_id: xlib::VisualID,
}

// SAFETY: see the type documentation; every Xlib call goes through a connection that was opened
// after `XInitThreads`.
unsafe impl Send for TestWindow {}
// SAFETY: as above.
unsafe impl Sync for TestWindow {}

impl TestWindow {
    /// Opens a connection to `display` and maps a `width` x `height` window on it.
    ///
    /// # Panics
    /// Panics if the display cannot be opened.
    pub fn create(display: &VirtualDisplay, width: u32, height: u32) -> Self {
        let xlib = xlib();
        let name = CString::new(display.name()).expect("display name without NUL bytes");
        // A server that has just started may still be setting up; retry for a moment.
        let deadline = Instant::now() + Duration::from_secs(2);
        let connection = loop {
            // SAFETY: `name` is a valid C string; Xlib was initialized by `xlib()`.
            let connection = NonNull::new(unsafe { (xlib.XOpenDisplay)(name.as_ptr()) });
            match connection {
                Some(connection) => break connection,
                None if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                None => panic!("XOpenDisplay({}) failed", display.name()),
            }
        };
        let dpy = connection.as_ptr();
        // SAFETY: `dpy` is an open connection; the window is created on its default screen and
        // destroyed in `Drop`. The size hints are set before mapping, so that a window manager
        // sees them when it decides how to place the window.
        let (screen, window, visual_id) = unsafe {
            let screen = (xlib.XDefaultScreen)(dpy);
            let root = (xlib.XRootWindow)(dpy, screen);
            let black = (xlib.XBlackPixel)(dpy, screen);
            let window =
                (xlib.XCreateSimpleWindow)(dpy, root, 0, 0, width, height, 0, black, black);
            let visual_id = (xlib.XVisualIDFromVisual)((xlib.XDefaultVisual)(dpy, screen));
            set_fixed_size(dpy, window, width, height);
            (xlib.XMapWindow)(dpy, window);
            (xlib.XSync)(dpy, xlib::False);
            (screen, window, visual_id)
        };
        let test_window = Self {
            display: connection,
            screen,
            window,
            visual_id,
        };
        test_window.wait_for_geometry(width, height);
        test_window
    }

    /// Resizes the window (as a user would) and waits until it keeps the new size.
    ///
    /// A window manager may still apply a configure for the old size after the resize, or not act
    /// on the request (niri through xwayland-satellite, under load: ISSUE-011). So the new size
    /// has to hold for a moment, and the resize is repeated every half second until it does.
    ///
    /// # Panics
    /// Panics with the window's state if it does not keep the size within five seconds.
    pub fn resize(&self, width: u32, height: u32) {
        const SETTLE: Duration = Duration::from_millis(250);
        const RETRY: Duration = Duration::from_millis(500);
        let xlib = xlib();
        let target = (true, (width as i32, height as i32));
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            // SAFETY: the connection and window are live until `Drop`.
            unsafe {
                set_fixed_size(self.display.as_ptr(), self.window, width, height);
                (xlib.XResizeWindow)(self.display.as_ptr(), self.window, width, height);
                (xlib.XSync)(self.display.as_ptr(), xlib::False);
            }
            let retry = (Instant::now() + RETRY).min(deadline);
            if self.reaches(target, retry) && !self.leaves(target, SETTLE) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the test window does not keep {width}x{height} after 5 s: {}",
                self.placement()
            );
        }
    }

    /// Waits until the window is viewable at `width` x `height`. Without a window manager the
    /// server applies both at once; a window manager maps and configures the window when it gets
    /// to it.
    ///
    /// # Panics
    /// Panics with the window's state if it does not get there within five seconds.
    fn wait_for_geometry(&self, width: u32, height: u32) {
        let deadline = Instant::now() + Duration::from_secs(5);
        assert!(
            self.reaches((true, (width as i32, height as i32)), deadline),
            "the test window is not viewable at {width}x{height} after 5 s: {}",
            self.placement()
        );
    }

    /// Whether the window is viewable, and its size.
    fn geometry(&self) -> (bool, (i32, i32)) {
        let xlib = xlib();
        // SAFETY: the connection and window are live; `attributes` is a plain C struct that
        // `XGetWindowAttributes` fills in.
        unsafe {
            let mut attributes: xlib::XWindowAttributes = std::mem::zeroed();
            (xlib.XGetWindowAttributes)(self.display.as_ptr(), self.window, &mut attributes);
            (
                attributes.map_state == xlib::IsViewable,
                (attributes.width, attributes.height),
            )
        }
    }

    /// Whether the window's geometry becomes `target` before `deadline`.
    fn reaches(&self, target: (bool, (i32, i32)), deadline: Instant) -> bool {
        loop {
            if self.geometry() == target {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Whether the window's geometry stops being `target` within `time`.
    fn leaves(&self, target: (bool, (i32, i32)), time: Duration) -> bool {
        let end = Instant::now() + time;
        while Instant::now() < end {
            if self.geometry() != target {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    /// Where the window is on its screen, for error messages: `XGetImage` can only read the
    /// part of a window that lies inside the screen.
    fn placement(&self) -> String {
        let xlib = xlib();
        let dpy = self.display.as_ptr();
        // SAFETY: the connection and window are live; the attribute structs are plain C structs
        // that Xlib fills in.
        unsafe {
            let root = (xlib.XRootWindow)(dpy, self.screen);
            let (mut x, mut y, mut child) = (0, 0, 0);
            (xlib.XTranslateCoordinates)(dpy, self.window, root, 0, 0, &mut x, &mut y, &mut child);
            let mut window: xlib::XWindowAttributes = std::mem::zeroed();
            (xlib.XGetWindowAttributes)(dpy, self.window, &mut window);
            let mut screen: xlib::XWindowAttributes = std::mem::zeroed();
            (xlib.XGetWindowAttributes)(dpy, root, &mut screen);
            format!(
                "window {}x{} at ({x}, {y}) on a {}x{} screen",
                window.width, window.height, screen.width, screen.height
            )
        }
    }

    /// Reads the window's current contents as RGBA8 (alpha 255), top row first.
    ///
    /// # Panics
    /// Panics with the X error if `XGetImage` fails (for example when the window is not mapped,
    /// or smaller than `width` x `height`).
    pub fn capture(&self, width: u32, height: u32) -> Image {
        let xlib = xlib();
        let dpy = self.display.as_ptr();
        // SAFETY: the connection and window are live; `XSync` first so that every request sent
        // by other connections (the backends present through their own) is processed. The image
        // is read with `XGetPixel` inside its bounds and freed with `XDestroyImage`.
        unsafe {
            (xlib.XSync)(dpy, xlib::False);
            LAST_X_ERROR.set(None);
            let image = (xlib.XGetImage)(
                dpy,
                self.window,
                0,
                0,
                width,
                height,
                !0, // AllPlanes
                xlib::ZPixmap,
            );
            assert!(
                !image.is_null(),
                "XGetImage failed: {} ({})",
                LAST_X_ERROR
                    .get()
                    .map_or("no X error reported", x_error_name),
                self.placement()
            );
            let masks = [(*image).red_mask, (*image).green_mask, (*image).blue_mask];
            let mut data = Vec::with_capacity((width * height * 4) as usize);
            for y in 0..height {
                for x in 0..width {
                    let pixel = (xlib.XGetPixel)(image, x as i32, y as i32);
                    for mask in masks {
                        data.push(((pixel & mask) >> mask.trailing_zeros()) as u8);
                    }
                    data.push(255);
                }
            }
            (xlib.XDestroyImage)(image);
            Image {
                width,
                height,
                data,
            }
        }
    }
}

/// Sets `window`'s minimum and maximum size hints to `width` x `height`: a fixed size, which
/// window managers and compositors take as a request to float the window at that size.
///
/// # Safety
/// `dpy` must be an open connection and `window` a live window on it.
unsafe fn set_fixed_size(dpy: *mut xlib::Display, window: xlib::Window, width: u32, height: u32) {
    let xlib = xlib();
    let (width, height) = (width as i32, height as i32);
    // SAFETY: `XSizeHints` is a plain C struct, valid when zeroed; Xlib reads only the fields
    // `flags` names. The caller guarantees the connection and window.
    unsafe {
        let mut hints: xlib::XSizeHints = std::mem::zeroed();
        hints.flags = xlib::PSize | xlib::PMinSize | xlib::PMaxSize;
        (hints.width, hints.height) = (width, height);
        (hints.min_width, hints.min_height) = (width, height);
        (hints.max_width, hints.max_height) = (width, height);
        (xlib.XSetWMNormalHints)(dpy, window, &mut hints);
    }
}

impl HasWindowHandle for TestWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let mut handle = XlibWindowHandle::new(self.window);
        handle.visual_id = self.visual_id;
        // SAFETY: the window lives as long as `self`.
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Xlib(handle)) })
    }
}

impl HasDisplayHandle for TestWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        let handle = XlibDisplayHandle::new(Some(self.display.cast()), self.screen);
        // SAFETY: the connection stays open as long as `self`.
        Ok(unsafe { DisplayHandle::borrow_raw(RawDisplayHandle::Xlib(handle)) })
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        let xlib = xlib();
        // SAFETY: the window and connection are destroyed exactly once; nothing presents to the
        // window any more (swapchains hold an `Arc<TestWindow>` until they are destroyed).
        unsafe {
            (xlib.XDestroyWindow)(self.display.as_ptr(), self.window);
            (xlib.XCloseDisplay)(self.display.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window's `WM_NORMAL_HINTS` minimum and maximum sizes, if both are set.
    fn fixed_size_hints(window: &TestWindow) -> Option<((i32, i32), (i32, i32))> {
        let xlib = xlib();
        // SAFETY: the connection and window are live; `hints` is a plain C struct that
        // `XGetWMNormalHints` fills in.
        unsafe {
            let mut hints: xlib::XSizeHints = std::mem::zeroed();
            let mut supplied = 0;
            let found = (xlib.XGetWMNormalHints)(
                window.display.as_ptr(),
                window.window,
                &mut hints,
                &mut supplied,
            );
            let both = xlib::PMinSize | xlib::PMaxSize;
            (found != 0 && hints.flags & both == both).then_some((
                (hints.min_width, hints.min_height),
                (hints.max_width, hints.max_height),
            ))
        }
    }

    /// ISSUE-011: test windows ask to float at their size. Tiling window managers and Wayland
    /// compositors (niri through xwayland-satellite) float a window whose minimum and maximum
    /// sizes are equal, and leave its size alone.
    #[test]
    fn test_windows_have_a_fixed_size_that_follows_their_resizes() {
        let Some(display) = display_or_skip("test_window_size_hints") else {
            return;
        };
        let window = TestWindow::create(&display, 64, 48);
        assert_eq!(fixed_size_hints(&window), Some(((64, 48), (64, 48))));
        window.resize(96, 72);
        assert_eq!(fixed_size_hints(&window), Some(((96, 72), (96, 72))));
    }

    /// A failed X request fails the test that made it. Xlib's default error handler ends the
    /// whole process instead, while other tests' threads are still inside the graphics drivers.
    #[test]
    fn a_failed_capture_panics_with_the_x_error() {
        let Some(display) = display_or_skip("test_window_capture_error") else {
            return;
        };
        let window = TestWindow::create(&display, 32, 32);
        // A region larger than the window: `XGetImage` fails with `BadMatch`.
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            window.capture(64, 64);
        }))
        .expect_err("capturing outside the window must fail");
        let message = panic.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(
            message.contains("BadMatch"),
            "unexpected panic: {message:?}"
        );
    }
}
