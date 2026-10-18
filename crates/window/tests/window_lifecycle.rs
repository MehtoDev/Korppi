//! Integration tests for WindowConfig, Window lifecycle, and event buffering.

use korppi_window::config::{WindowConfig, WindowMode};
use korppi_window::event::WindowEvent;
use korppi_window::window::Window;

#[test]
fn window_config_builder_constructs_desired_settings() {
    let config = WindowConfig::default()
        .with_title("Custom Engine Window")
        .with_dimensions(1920, 1080)
        .with_vsync(false)
        .with_resizable(false)
        .with_mode(WindowMode::BorderlessFullscreen);

    assert_eq!(config.title, "Custom Engine Window");
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert!(!config.vsync);
    assert!(!config.resizable);
    assert_eq!(config.mode, WindowMode::BorderlessFullscreen);
}

#[test]
fn headless_window_lifecycle_and_resize() {
    let config = WindowConfig::default()
        .with_title("Test Window")
        .with_dimensions(800, 600);

    let mut window = Window::new_headless(config).unwrap();
    assert_eq!(window.width(), 800);
    assert_eq!(window.height(), 600);
    assert!((window.aspect_ratio() - 800.0 / 600.0).abs() < f32::EPSILON);
    assert_eq!(window.scale_factor(), 1.0);
    assert!(!window.should_close());

    // Request resize
    window.on_resized(1024, 768);
    assert_eq!(window.width(), 1024);
    assert_eq!(window.height(), 768);
    assert!((window.aspect_ratio() - 1024.0 / 768.0).abs() < f32::EPSILON);
}

#[test]
fn window_event_queue_buffering_and_drain() {
    let config = WindowConfig::default();
    let mut window = Window::new_headless(config).unwrap();

    window.push_event(WindowEvent::Focused(true));
    window.push_event(WindowEvent::Resized {
        width: 1280,
        height: 720,
    });
    window.push_event(WindowEvent::CloseRequested);

    let events: Vec<WindowEvent> = window.drain_events().collect();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0], WindowEvent::Focused(true));
    assert_eq!(
        events[1],
        WindowEvent::Resized {
            width: 1280,
            height: 720
        }
    );
    assert_eq!(events[2], WindowEvent::CloseRequested);

    assert!(window.should_close());
}

#[test]
fn window_dpi_scale_factor_changes_physical_resolution() {
    let config = WindowConfig::default().with_dimensions(1000, 500);
    let mut window = Window::new_headless(config).unwrap();

    assert_eq!(window.physical_width(), 1000);
    assert_eq!(window.physical_height(), 500);

    window.on_scale_factor_changed(2.0);
    assert_eq!(window.scale_factor(), 2.0);
    assert_eq!(window.physical_width(), 2000);
    assert_eq!(window.physical_height(), 1000);
    assert_eq!(window.width(), 1000); // logical width remains 1000
}
