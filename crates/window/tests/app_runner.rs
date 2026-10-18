//! Integration test for KorppiApp and event dispatching.

use korppi_window::app::{EngineApp, KorppiApp};
use korppi_window::config::WindowConfig;
use korppi_window::event::WindowEvent;
use korppi_window::window::Window;

struct TestApp {
    started: bool,
    update_count: usize,
    resized: bool,
}

impl KorppiApp for TestApp {
    fn on_start(&mut self, _window: &mut Window) {
        self.started = true;
    }

    fn on_update(&mut self, _window: &mut Window, _dt: f32) {
        self.update_count += 1;
    }

    fn on_event(&mut self, _window: &mut Window, event: &WindowEvent) {
        if let WindowEvent::Resized { .. } = event {
            self.resized = true;
        }
    }
}

#[test]
fn korppi_app_lifecycle_callbacks_fire_correctly() {
    // Verify backward compatibility alias works
    fn _assert_alias<T: EngineApp>() {}
    _assert_alias::<TestApp>();
    let mut app = TestApp {
        started: false,
        update_count: 0,
        resized: false,
    };

    let mut window = Window::new_headless(WindowConfig::default()).unwrap();

    app.on_start(&mut window);
    assert!(app.started);

    app.on_update(&mut window, 1.0 / 60.0);
    app.on_update(&mut window, 1.0 / 60.0);
    assert_eq!(app.update_count, 2);

    window.push_event(WindowEvent::Resized {
        width: 1920,
        height: 1080,
    });
    let events: Vec<WindowEvent> = window.drain_events().collect();
    for event in &events {
        app.on_event(&mut window, event);
    }
    assert!(app.resized);
}
