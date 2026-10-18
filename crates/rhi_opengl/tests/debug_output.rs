//! Integration tests for OpenGL `KHR_debug` output capture on the headless EGL device.

use glow::HasContext;
use korppi_rhi::{api_validation_requested, DiagnosticSeverity, GraphicsDevice};
use korppi_rhi_opengl::OpenGlDevice;

#[test]
fn debug_output_is_enabled_exactly_when_validation_is_requested() {
    let device = OpenGlDevice::create_headless().expect("OpenGL device");
    assert_eq!(device.api_validation_enabled(), api_validation_requested());
}

#[test]
fn gl_errors_are_reported_as_error_diagnostics() {
    let device = OpenGlDevice::create_headless().expect("OpenGL device");
    if !device.api_validation_enabled() {
        eprintln!("skipped: API validation is disabled for this process");
        return;
    }
    let _ = device.take_diagnostics();
    device
        // SAFETY: `with_gl` runs on the render thread, where the context is current. 0xDEAD is
        // not a buffer target, so the driver must raise GL_INVALID_ENUM and change no state.
        .with_gl(|gl| unsafe { gl.bind_buffer(0xDEAD, None) })
        .expect("render thread");
    let diagnostics = device.take_diagnostics();
    assert!(
        diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error),
        "GL_INVALID_ENUM was not reported: {diagnostics:?}"
    );
}
