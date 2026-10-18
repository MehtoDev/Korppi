//! A Wayland window for presentation tests, on a [`WaylandCompositor`].
//!
//! Presentation behaves differently on Wayland: the client picks the surface size (the swapchain
//! extent is not dictated by the window), and Mesa's Wayland driver enforces rules its X11 driver
//! does not, such as one non-retired swapchain per surface (ISSUE-012). The window is an
//! `xdg_toplevel` with equal minimum and maximum sizes, which tiling compositors such as niri
//! open floating at that size (ISSUE-011). It cannot read back what was presented: Wayland gives
//! clients no access to the composited image.

use super::WaylandCompositor;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle, WindowHandle,
};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Mutex;
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::{wl_compositor, wl_registry, wl_surface};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols::xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base};

/// Event state of a [`WaylandTestWindow`].
#[derive(Default)]
struct WindowState {
    /// Whether the compositor has sent the first `xdg_surface.configure` (acknowledged at once).
    configured: bool,
}

/// An `xdg_toplevel` window on its own Wayland connection.
///
/// `Send + Sync`: `wayland-client` connections and proxies may be used from any thread, and the
/// event queue is behind a lock.
pub struct WaylandTestWindow {
    connection: Connection,
    queue: Mutex<(EventQueue<WindowState>, WindowState)>,
    surface: wl_surface::WlSurface,
    xdg_surface: xdg_surface::XdgSurface,
    toplevel: xdg_toplevel::XdgToplevel,
}

impl WaylandTestWindow {
    /// Connects to `compositor` and maps a toplevel window with a fixed `width` x `height`
    /// size, waiting until the compositor has configured it.
    ///
    /// # Panics
    /// Panics if the compositor cannot be reached or lacks `wl_compositor` or `xdg_wm_base`.
    pub fn create(compositor: &WaylandCompositor, width: u32, height: u32) -> Self {
        let stream = UnixStream::connect(socket_path(compositor.socket()))
            .unwrap_or_else(|e| panic!("connect to Wayland socket {}: {e}", compositor.socket()));
        let connection = Connection::from_socket(stream).expect("Wayland connection");
        let (globals, mut queue) =
            registry_queue_init::<WindowState>(&connection).expect("Wayland registry");
        let qh = queue.handle();
        let wl_compositor: wl_compositor::WlCompositor = globals
            .bind(&qh, 4..=6, ())
            .expect("the compositor offers wl_compositor");
        let wm_base: xdg_wm_base::XdgWmBase = globals
            .bind(&qh, 1..=6, ())
            .expect("the compositor offers xdg_wm_base");

        let surface = wl_compositor.create_surface(&qh, ());
        let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
        let toplevel = xdg_surface.get_toplevel(&qh, ());
        toplevel.set_title("Korppi RHI conformance".into());
        toplevel.set_app_id("korppi_rhi_conformance".into());
        set_fixed_size(&toplevel, width, height);
        // The first commit carries no buffer; the compositor answers with a configure that must
        // be acknowledged before a buffer (the swapchain's first image) is attached.
        surface.commit();
        let mut state = WindowState::default();
        while !state.configured {
            queue
                .blocking_dispatch(&mut state)
                .expect("Wayland event dispatch");
        }
        Self {
            connection,
            queue: Mutex::new((queue, state)),
            surface,
            xdg_surface,
            toplevel,
        }
    }

    /// Changes the window's fixed size to `width` x `height`. On Wayland the client sizes the
    /// surface, so the size takes effect with the next presented image (`resize_swapchain`).
    pub fn resize(&self, width: u32, height: u32) {
        set_fixed_size(&self.toplevel, width, height);
        self.dispatch_events();
    }

    /// Answers the compositor's events (configures, pings) after a round trip, as an
    /// application's event loop does; call it once per frame.
    pub fn dispatch_events(&self) {
        let mut guard = self
            .queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (queue, state) = &mut *guard;
        queue.roundtrip(state).expect("Wayland roundtrip");
    }
}

/// The path of a compositor socket: `name` itself if it is absolute (`WAYLAND_DISPLAY` may be),
/// otherwise in `$XDG_RUNTIME_DIR`.
fn socket_path(name: &str) -> PathBuf {
    let path = PathBuf::from(name);
    if path.is_absolute() {
        return path;
    }
    let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR is set");
    PathBuf::from(runtime_dir).join(path)
}

/// Sets equal minimum and maximum sizes: a fixed-size window, which compositors open floating.
fn set_fixed_size(toplevel: &xdg_toplevel::XdgToplevel, width: u32, height: u32) {
    let (width, height) = (width as i32, height as i32);
    toplevel.set_min_size(width, height);
    toplevel.set_max_size(width, height);
}

impl HasWindowHandle for WaylandTestWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let surface =
            NonNull::new(self.surface.id().as_ptr().cast()).ok_or(HandleError::Unavailable)?;
        // SAFETY: the surface lives as long as `self`.
        Ok(unsafe {
            WindowHandle::borrow_raw(RawWindowHandle::Wayland(WaylandWindowHandle::new(surface)))
        })
    }
}

impl HasDisplayHandle for WaylandTestWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        let display = NonNull::new(self.connection.backend().display_ptr().cast())
            .ok_or(HandleError::Unavailable)?;
        // SAFETY: the connection stays open as long as `self`.
        Ok(unsafe {
            DisplayHandle::borrow_raw(RawDisplayHandle::Wayland(WaylandDisplayHandle::new(
                display,
            )))
        })
    }
}

impl Drop for WaylandTestWindow {
    fn drop(&mut self) {
        // Nothing presents to the surface any more (swapchains hold an `Arc` of the window until
        // they are destroyed).
        self.toplevel.destroy();
        self.xdg_surface.destroy();
        self.surface.destroy();
        let _ = self.connection.flush();
    }
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for WindowState {
    fn event(
        _: &mut Self,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_compositor::WlCompositor, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &wl_compositor::WlCompositor,
        _: wl_compositor::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<wl_surface::WlSurface, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &wl_surface::WlSurface,
        _: wl_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for WindowState {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for WindowState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            state.configured = true;
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for WindowState {
    fn event(
        _: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        _: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The suggested size is ignored: the window keeps its fixed size.
    }
}
