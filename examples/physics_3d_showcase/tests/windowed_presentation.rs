//! The showcase opens a window and presents frames on both backends, on X11 and on Wayland
//! (FIX-1.3, ISSUE-001).
//!
//! Runs the binary on `$DISPLAY` or a self-started Xvfb, and on `$WAYLAND_DISPLAY` or a
//! self-started headless Weston (see `korppi_rhi_conformance::display`); each case is skipped
//! when its window system is unavailable.
#![cfg(target_os = "linux")]

use korppi_rhi_conformance::display::{display_or_skip, wayland_or_skip};
use std::process::Command;

/// Runs the showcase for five frames with each backend under `env` and checks it presented them.
fn run_with_each_backend(window_system: &str, env: &[(&str, Option<&str>)]) {
    for backend in ["vulkan", "opengl"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_physics_3d_showcase"));
        command
            .args(["--backend", backend, "--frames", "5"])
            .env("RUST_LOG", "warn");
        for (name, value) in env {
            match value {
                Some(value) => command.env(name, value),
                None => command.env_remove(name),
            };
        }
        let output = command.output().expect("run the showcase binary");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "{window_system} {backend}: exit status {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        assert!(
            stdout.contains("Presented 5 frames"),
            "{window_system} {backend}: stdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }
}

#[test]
fn presents_frames_on_x11_with_each_backend_and_exits() {
    let Some(display) = display_or_skip("physics_3d_showcase on X11") else {
        return;
    };
    // winit prefers Wayland when both are available.
    run_with_each_backend(
        "X11",
        &[("DISPLAY", Some(display.name())), ("WAYLAND_DISPLAY", None)],
    );
}

#[test]
fn presents_frames_on_wayland_with_each_backend_and_exits() {
    let Some(compositor) = wayland_or_skip("physics_3d_showcase on Wayland") else {
        return;
    };
    run_with_each_backend(
        "Wayland",
        &[
            ("WAYLAND_DISPLAY", Some(compositor.socket())),
            ("DISPLAY", None),
        ],
    );
}
