//! Presentation to Wayland windows (ISSUE-011, ISSUE-012).
//!
//! Every test presents into a fixed-size `xdg_toplevel` on `$WAYLAND_DISPLAY` or on a headless
//! Weston it starts, with one backend each (a `_vulkan` or `_opengl` suffix); without a
//! compositor the tests are skipped (see `korppi_rhi_conformance::display`).
//! Wayland gives clients no readback, so these tests check the swapchain's behaviour, and
//! `tests/present.rs` checks presented content on X11. Mesa's Wayland driver enforces swapchain
//! rules its X11 driver does not, such as one non-retired swapchain per surface.
#![cfg(target_os = "linux")]

use korppi_rhi::{LoadOp, SwapchainDescriptor, SwapchainHandle, SwapchainImage};
use korppi_rhi_conformance::display::{wayland_or_skip, WaylandCompositor, WaylandTestWindow};
use korppi_rhi_conformance::{color_pass, undefined_to_color_attachment, Backend, TestDevice};
use std::sync::Arc;

const W: u32 = 64;
const H: u32 = 48;

/// A window, a device that can present to it, and a swapchain.
struct Presentation {
    window: Arc<WaylandTestWindow>,
    device: TestDevice,
    swapchain: SwapchainHandle,
}

impl Presentation {
    fn open(backend: Backend, compositor: &WaylandCompositor) -> Self {
        let window = Arc::new(WaylandTestWindow::create(compositor, W, H));
        let device = TestDevice::open_for_display(backend, window.clone());
        let swapchain = device
            .create_swapchain(window.clone(), &SwapchainDescriptor::new(W, H))
            .unwrap_or_else(|e| panic!("{backend:?}: create_swapchain failed: {e}"));
        Self {
            window,
            device,
            swapchain,
        }
    }

    /// Clears an acquired image and presents it.
    fn frame(&self) -> SwapchainImage {
        self.window.dispatch_events();
        self.device.begin_frame().expect("begin_frame");
        let backend = self.device.backend();
        let image = self
            .device
            .acquire_next_image(self.swapchain)
            .expect("acquire_next_image")
            .unwrap_or_else(|| panic!("{backend:?}: the swapchain has an area, but no image"));
        let mut recorder = self.device.create_command_recorder().expect("recorder");
        recorder.begin().expect("begin");
        recorder.pipeline_barrier(&undefined_to_color_attachment(image.texture));
        recorder
            .begin_rendering(&color_pass(
                image.texture,
                image.width,
                image.height,
                LoadOp::Clear,
                [0.0, 0.0, 0.5, 1.0],
            ))
            .expect("begin_rendering");
        recorder.end_rendering().expect("end_rendering");
        self.device
            .submit(recorder.finish().expect("finish"))
            .expect("submit");
        self.device.present(&image).expect("present");
        image
    }
}

impl Drop for Presentation {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.device
                .destroy_swapchain(self.swapchain)
                .expect("destroy_swapchain");
        }
    }
}

/// ISSUE-012: shrinking to no area and back must recreate the swapchain from the parked one;
/// creating a second one for the surface fails with `VK_ERROR_NATIVE_WINDOW_IN_USE_KHR`.
fn acquisition_pauses_while_the_swapchain_has_no_area(backend: Backend) {
    let Some(compositor) = wayland_or_skip("present_wayland_zero_area") else {
        return;
    };
    let presentation = Presentation::open(backend, &compositor);
    let device = &presentation.device;
    presentation.frame();
    device
        .resize_swapchain(presentation.swapchain, 0, 0)
        .expect("resize to zero");
    assert_eq!(
        device.acquire_next_image(presentation.swapchain),
        Ok(None),
        "{backend:?}: a zero-area swapchain has no image to acquire"
    );
    device
        .resize_swapchain(presentation.swapchain, W, H)
        .expect("resize back");
    let image = presentation.frame();
    assert_eq!((image.width, image.height), (W, H), "{backend:?}");
}

#[test]
fn acquisition_pauses_while_the_swapchain_has_no_area_on_wayland_vulkan() {
    acquisition_pauses_while_the_swapchain_has_no_area(Backend::Vulkan);
}

#[test]
fn acquisition_pauses_while_the_swapchain_has_no_area_on_wayland_opengl() {
    acquisition_pauses_while_the_swapchain_has_no_area(Backend::OpenGl);
}

fn resizing_recreates_the_swapchain_at_the_new_size(backend: Backend) {
    let Some(compositor) = wayland_or_skip("present_wayland_resize") else {
        return;
    };
    let presentation = Presentation::open(backend, &compositor);
    presentation.frame();
    let (w, h) = (96, 72);
    presentation.window.resize(w, h);
    presentation
        .device
        .resize_swapchain(presentation.swapchain, w, h)
        .expect("resize_swapchain");
    let image = presentation.frame();
    assert_eq!((image.width, image.height), (w, h), "{backend:?}");
    let info = presentation
        .device
        .swapchain_info(presentation.swapchain)
        .expect("swapchain_info");
    assert_eq!((info.width, info.height), (w, h), "{backend:?}");
}

#[test]
fn resizing_recreates_the_swapchain_at_the_new_size_on_wayland_vulkan() {
    resizing_recreates_the_swapchain_at_the_new_size(Backend::Vulkan);
}

#[test]
fn resizing_recreates_the_swapchain_at_the_new_size_on_wayland_opengl() {
    resizing_recreates_the_swapchain_at_the_new_size(Backend::OpenGl);
}

fn many_frames_reuse_synchronization_objects_without_api_errors(backend: Backend) {
    let Some(compositor) = wayland_or_skip("present_wayland_many_frames") else {
        return;
    };
    let presentation = Presentation::open(backend, &compositor);
    for _ in 0..30 {
        presentation.frame();
    }
    presentation.device.wait_idle().expect("wait_idle");
    presentation.device.assert_no_api_errors();
}

#[test]
fn many_frames_reuse_synchronization_objects_without_api_errors_on_wayland_vulkan() {
    many_frames_reuse_synchronization_objects_without_api_errors(Backend::Vulkan);
}

#[test]
fn many_frames_reuse_synchronization_objects_without_api_errors_on_wayland_opengl() {
    many_frames_reuse_synchronization_objects_without_api_errors(Backend::OpenGl);
}
