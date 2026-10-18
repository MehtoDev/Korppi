//! Zero steady-state heap allocations during `EngineApp::update` simulation ticks
//! (`ENG-02`, Rule 4).

use korppi_engine::EngineApp;
use korppi_math::{Vec2, Vec3};
use korppi_physics::{
    body::{BodyType as BodyType3d, RigidBody},
    collider::Collider,
    shape::Shape,
};
use korppi_physics2d::{
    body::BodyType as BodyType2d,
    collider::Collider2d,
    shape::{Circle2d, Polygon2d, Shape2d},
    RigidBody2d,
};
use korppi_scene::{GlobalTransform, LocalTransform};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

struct TrackingAlloc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static TRACKING_ENABLED: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Delegates directly to `std::alloc::System` while atomically counting allocations
// when `TRACKING_ENABLED` is non-zero.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING_ENABLED.load(Ordering::Relaxed) != 0 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Forwards exact `layout` to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Forwards exact `ptr` and `layout` allocated by `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING_ENABLED.load(Ordering::Relaxed) != 0 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: Forwards exact `ptr`, `layout`, and `new_size` to `System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

const DT: Duration = Duration::from_nanos(1_000_000_000 / 60);

#[test]
fn steady_state_simulation_ticks_do_not_allocate_on_the_heap() {
    let mut app = EngineApp::new().expect("engine");

    // Static ground in 3D and 2D so dynamic bodies collide and rest.
    app.world.spawn((
        RigidBody::new(BodyType3d::Static, Vec3::new(50.0, -0.5, 0.0)),
        Collider::new(Shape::Box {
            half_extents: Vec3::new(200.0, 0.5, 20.0),
        }),
        LocalTransform::default(),
        GlobalTransform::default(),
    ));
    app.world.spawn((
        RigidBody2d::new(BodyType2d::Static).with_position(Vec2::new(50.0, -0.5)),
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(200.0, 0.5))),
        LocalTransform::default(),
        GlobalTransform::default(),
    ));

    let mut entity_3d = None;
    let mut entity_2d = None;

    for i in 0..100 {
        let e3 = app.world.spawn((
            RigidBody::new(BodyType3d::Dynamic, Vec3::new(i as f32 * 1.5, 0.5, 0.0)),
            Collider::new(Shape::Sphere { radius: 0.5 }),
            LocalTransform::default(),
            GlobalTransform::default(),
        ));
        let e2 = app.world.spawn((
            RigidBody2d::new(BodyType2d::Dynamic).with_position(Vec2::new(i as f32 * 1.5, 0.5)),
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))),
            LocalTransform::default(),
            GlobalTransform::default(),
        ));
        if i == 0 {
            entity_3d = Some(e3);
            entity_2d = Some(e2);
        }
    }

    // Warm up simulation and pre-warm mutation scratch buffers once.
    for _ in 0..60 {
        app.update(DT);
    }
    let e3 = entity_3d.unwrap();
    let e2 = entity_2d.unwrap();
    app.world.get_mut::<RigidBody>(e3).unwrap().linear_velocity = Vec3::new(1.0, 0.0, 0.0);
    app.world
        .get_mut::<RigidBody2d>(e2)
        .unwrap()
        .linear_velocity = Vec2::new(1.0, 0.0);
    app.update(DT);

    // Measure 60 steady-state ticks (including active ECS velocity mutations).
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    TRACKING_ENABLED.store(1, Ordering::SeqCst);

    for tick in 0..60 {
        let vx = if tick % 2 == 0 { 0.5 } else { -0.5 };
        app.world
            .get_mut::<RigidBody>(e3)
            .unwrap()
            .linear_velocity
            .x = vx;
        app.world
            .get_mut::<RigidBody2d>(e2)
            .unwrap()
            .linear_velocity
            .x = vx;
        app.update(DT);
    }

    TRACKING_ENABLED.store(0, Ordering::SeqCst);
    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    let bytes = ALLOC_BYTES.load(Ordering::SeqCst);

    assert_eq!(
        allocs, 0,
        "Rule 4 violation: steady-state EngineApp::update allocated {allocs} times ({bytes} bytes) over 60 ticks"
    );
}
