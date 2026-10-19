//! Regression tests for `FIX-12.3` / `FIX-13.3` window audit findings (`WIN-01`–`WIN-03`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use korppi_window::{KorppiApp, Window, WindowConfig, WindowEvent, WindowMode};

struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: delegates all allocation operations directly to the system allocator while counting calls.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: caller upholds `GlobalAlloc::alloc` layout invariants.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller upholds `GlobalAlloc::dealloc` pointer and layout invariants.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: caller upholds `GlobalAlloc::realloc` pointer, layout, and size invariants.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocations_in(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

/// WIN-01 (W1): State updates must not allocate or grow `event_queue` unboundedly, and headless
/// `push_event` must bound its queue capacity so undrained queues never leak memory across frames.
#[test]
fn w1_window_state_updates_are_zero_alloc_and_event_queue_is_bounded() {
    let mut win = Window::new_headless(WindowConfig::default()).unwrap();

    // Direct state mutators used by AppRunner must not push to event_queue or allocate on the heap:
    let allocs = allocations_in(|| {
        for i in 0..10_000u32 {
            win.set_physical_size(1920 + (i & 1), 1080 + (i & 1));
            win.set_scale_factor(1.5);
            win.set_focused((i & 1) == 0);
        }
    });
    assert_eq!(
        allocs, 0,
        "Window state updates must not allocate on the heap"
    );
    assert_eq!(
        win.drain_events().count(),
        0,
        "Direct state updates must not buffer duplicate events in event_queue"
    );

    // Even if a headless caller pushes 10,000 events without calling `drain_events()`,
    // the internal queue must be bounded to prevent unbounded heap growth.
    for i in 0..10_000 {
        win.push_event(WindowEvent::CursorMoved {
            x: i as f64,
            y: i as f64,
        });
    }
    let queued = win.drain_events().count();
    assert!(
        queued <= Window::MAX_EVENT_QUEUE_CAPACITY,
        "Window::event_queue grew to {queued} (expected <= {})",
        Window::MAX_EVENT_QUEUE_CAPACITY
    );
}

/// WIN-02 (W2): `Window` must store authoritative physical pixel dimensions so fractional HiDPI
/// scale factors (`1.25`, `1.5`, `1.75`) never suffer +/-1 px rounding or truncation drift.
#[test]
fn w2_authoritative_physical_dimensions_under_fractional_dpi() {
    for (phys_w, phys_h, scale) in [
        (2560u32, 1440u32, 1.5f64),
        (1000, 1000, 1.5),
        (1920, 1080, 1.25),
        (3840, 2160, 1.75),
    ] {
        let mut win = Window::new_headless(WindowConfig::default()).unwrap();
        win.set_scale_factor(scale);
        win.on_physical_resized(phys_w, phys_h);

        assert_eq!(
            win.physical_width(),
            phys_w,
            "physical_width drifted at {phys_w}x{phys_h} @ {scale}x"
        );
        assert_eq!(
            win.physical_height(),
            phys_h,
            "physical_height drifted at {phys_w}x{phys_h} @ {scale}x"
        );
        assert_eq!(win.width(), (phys_w as f64 / scale).round() as u32);
        assert_eq!(win.height(), (phys_h as f64 / scale).round() as u32);
    }
}

/// WIN-03 (W3): `Window` must expose runtime cursor & title mutators and `AppRunner` helpers must
/// normalize scroll deltas and ignore key repeat.
#[test]
fn w3_window_cursor_mutators_and_scroll_normalization() {
    let cfg = WindowConfig::default()
        .with_cursor_visible(false)
        .with_cursor_grabbed(true)
        .with_mode(WindowMode::Fullscreen);
    let mut win = Window::new_headless(cfg).unwrap();

    assert!(!win.cursor_visible());
    assert!(win.cursor_grabbed());
    assert_eq!(win.config().mode, WindowMode::Fullscreen);

    win.set_cursor_visible(true);
    win.set_cursor_grabbed(false);
    win.set_title("Updated Korppi Title");

    assert!(win.cursor_visible());
    assert!(!win.cursor_grabbed());
    assert_eq!(win.title(), "Updated Korppi Title");
    assert_eq!(win.config().title, "Updated Korppi Title");

    // Verify PixelDelta scroll normalization accounts for scale factor and line height:
    let (dx, dy) = Window::normalize_pixel_scroll(0.0, 60.0, 1.5);
    assert_eq!(dx, 0.0);
    assert!(
        (dy - 1.0).abs() < 1e-5,
        "60 physical pixels at 1.5x scale (40 logical px = 1 line) should normalize to 1.0 line, got {dy}"
    );

    // Ensure KorppiApp is still object-safe and usable with the new methods:
    struct DummyApp;
    impl KorppiApp for DummyApp {}
    let mut app = DummyApp;
    app.on_start(&mut win);
}
