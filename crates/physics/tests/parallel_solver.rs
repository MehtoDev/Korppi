//! The 3D world steps in parallel, with results independent of the worker count (P3D-26).
//!
//! As Box3D v0.1 (`solver.c`, `contact_solver_wide.inl`), a world given a task executor spreads
//! its step over the executor's workers: the narrowphase runs as a parallel loop over awake
//! contacts, and the solver runs its stages as blocks that workers claim, graph color by graph
//! color, with the colored convex contacts solved four at a time by the wide SIMD contact solver.
//! Constraints of one color share no dynamic body, and everything workers produce is merged in a
//! fixed order, so a world steps bit for bit the same on any number of workers.

use korppi_core::{TaskExecutor, WorkerPool};
use korppi_math::{Quat, Vec3};
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::{Collider, Filter3d};
use korppi_physics::joint::{Joint, RevoluteJoint, WeldJoint};
use korppi_physics::shape::Shape;
use korppi_physics::world::{BodyId3d, PhysicsWorld};
use korppi_physics::{Compound3d, CompoundChild, Mesh3d};
use std::sync::{Arc, Mutex, PoisonError};

const DT: f32 = 1.0 / 60.0;
const WORKER_COUNTS: [usize; 5] = [1, 2, 3, 4, 8];

/// Worker-pool tests take turns so multi-worker runs do not contend with one another.
static SERIAL: Mutex<()> = Mutex::new(());

fn add_box(world: &mut PhysicsWorld, x: f32, y: f32, z: f32) -> BodyId3d {
    let id = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(x, y, z)));
    world.create_collider(
        id,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.6),
    );
    id
}

/// A ten-row pyramid of unit boxes on a ground box, sleep off.
fn pyramid() -> (PhysicsWorld, BodyId3d) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -10.0, 0.0));
    world.enable_sleep(false);
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.6),
    );
    let mut top = None;
    for row in 0..10 {
        for i in 0..10 - row {
            top = Some(add_box(
                &mut world,
                -4.5 + 0.5 * row as f32 + i as f32,
                0.5 + row as f32,
                0.0,
            ));
        }
    }
    (world, top.unwrap())
}

/// A busy 3D scene with sleep on: stacks and a pyramid, spheres/capsules/convex hulls with
/// rolling resistance and restitution raining down, fast bullets, hinged and welded chains with
/// a motor, a kinematic paddle, a triangle mesh ramp, a compound platform, and sensors.
fn busy() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -10.0, 0.0));

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(30.0, 0.5, 30.0),
        })
        .with_friction(0.6),
    );

    // Triangle mesh ramp
    let ramp_verts = vec![
        Vec3::new(-8.0, 0.0, -4.0),
        Vec3::new(-2.0, 0.0, -4.0),
        Vec3::new(-2.0, 0.0, 4.0),
        Vec3::new(-8.0, 2.0, 4.0),
    ];
    let ramp_tris = vec![[0, 2, 1], [0, 3, 2]];
    let mesh = Mesh3d::new(&ramp_verts, &ramp_tris).unwrap();
    let ramp_body = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ramp_body,
        Collider::new(Shape::TriangleMesh(mesh)).with_friction(0.5),
    );

    // Compound static platform
    let c1 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.5, 0.25, 1.5),
        },
        Vec3::new(8.0, 0.25, -2.0),
        Quat::IDENTITY,
    );
    let c2 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.5, 0.25, 1.5),
        },
        Vec3::new(8.0, 0.25, 2.0),
        Quat::IDENTITY,
    );
    let comp = Compound3d::new(vec![c1, c2]).unwrap();
    let comp_body = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        comp_body,
        Collider::new(Shape::Compound(comp)).with_friction(0.6),
    );

    // 4 stacks of 6 boxes (some far away so they settle and fall asleep)
    for column in 0..4 {
        for row in 0..6 {
            add_box(
                &mut world,
                -18.0 + 3.0 * column as f32,
                0.5 + row as f32,
                -8.0,
            );
        }
    }

    // 6-row pyramid
    for row in 0..6 {
        for i in 0..6 - row {
            add_box(
                &mut world,
                2.0 + 0.5 * row as f32 + i as f32,
                0.5 + row as f32,
                0.0,
            );
        }
    }

    // Spheres, capsules, and convex hulls raining down
    let hull_pts = vec![
        Vec3::new(-0.3, -0.3, -0.3),
        Vec3::new(0.3, -0.3, -0.3),
        Vec3::new(0.3, 0.3, -0.3),
        Vec3::new(-0.3, 0.3, -0.3),
        Vec3::new(-0.3, -0.3, 0.3),
        Vec3::new(0.3, -0.3, 0.3),
        Vec3::new(0.3, 0.3, 0.3),
        Vec3::new(-0.3, 0.3, 0.3),
    ];
    let hull_shape = Shape::convex_hull(&hull_pts).unwrap();
    for i in 0..24 {
        let is_bullet = i % 8 == 0;
        let body = world.create_body(
            RigidBody::new(
                BodyType::Dynamic,
                Vec3::new(
                    -8.0 + 0.7 * i as f32,
                    10.0 + (i % 4) as f32,
                    (i % 3) as f32 - 1.0,
                ),
            )
            .with_angular_velocity(Vec3::new(0.2 * (i % 5) as f32, -0.4, 0.3))
            .with_bullet(is_bullet)
            .with_linear_velocity(Vec3::new(
                0.0,
                if is_bullet { -45.0 } else { 0.0 },
                0.0,
            )),
        );
        let shape = match i % 3 {
            0 => Shape::Sphere { radius: 0.3 },
            1 => Shape::Capsule {
                radius: 0.2,
                half_height: 0.25,
            },
            _ => hull_shape.clone(),
        };
        world.create_collider(
            body,
            Collider::new(shape)
                .with_restitution(0.3)
                .with_rolling_resistance(0.05)
                .with_contact_events(true)
                .with_hit_events(true),
        );
    }

    // Hinged and welded chain with motor
    let anchor = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(12.0, 10.0, 0.0)));
    let mut previous = anchor;
    for i in 0..8 {
        let link = world.create_body(RigidBody::new(
            BodyType::Dynamic,
            Vec3::new(12.5 + i as f32, 10.0, 0.0),
        ));
        world.create_collider(
            link,
            Collider::new(Shape::Box {
                half_extents: Vec3::new(0.5, 0.1, 0.1),
            })
            .with_filter(Filter3d {
                group_index: -1,
                ..Filter3d::DEFAULT
            }),
        );
        let anchor_a = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.5, 0.0, 0.0)
        };
        let joint = if i % 3 == 2 {
            Joint::Weld(WeldJoint::new(
                previous,
                link,
                anchor_a,
                Vec3::new(-0.5, 0.0, 0.0),
            ))
        } else {
            let mut hinge =
                RevoluteJoint::new(previous, link, anchor_a, Vec3::new(-0.5, 0.0, 0.0), Vec3::Z);
            if i == 0 {
                hinge = hinge.with_motor(1.0, 50.0);
            }
            Joint::Revolute(hinge)
        };
        world.create_joint(joint);
        previous = link;
    }

    // Kinematic paddle
    let paddle = world.create_body(
        RigidBody::new(BodyType::Kinematic, Vec3::new(-4.0, 4.0, 0.0))
            .with_angular_velocity(Vec3::new(0.0, 1.5, 0.0)),
    );
    world.create_collider(
        paddle,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(2.0, 0.1, 0.5),
        }),
    );

    // Sensor volume
    let sensor = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 3.0, 0.0)));
    world.create_collider(
        sensor,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(4.0, 1.0, 4.0),
        })
        .with_sensor(true)
        .with_sensor_events(true),
    );

    world
}

fn executor(workers: usize) -> Arc<dyn TaskExecutor> {
    Arc::new(WorkerPool::new(workers))
}

#[test]
fn a_pyramid_steps_bit_for_bit_on_any_number_of_workers() {
    let _turn = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut serial_world, serial_top) = pyramid();
    for _ in 0..120 {
        serial_world.step(DT);
    }
    let serial_hash = serial_world.state_hash();
    let serial_body = serial_world.body(serial_top).unwrap();
    let expected_bits = [
        serial_body.position.x,
        serial_body.position.y,
        serial_body.position.z,
        serial_body.rotation.x,
        serial_body.rotation.y,
        serial_body.rotation.z,
        serial_body.rotation.w,
    ]
    .map(f32::to_bits);

    for workers in WORKER_COUNTS {
        let (mut world, top) = pyramid();
        world.set_executor(Some(executor(workers)));
        for _ in 0..120 {
            world.step(DT);
        }
        assert_eq!(world.state_hash(), serial_hash, "{workers} workers hash");
        let body = world.body(top).unwrap();
        let bits = [
            body.position.x,
            body.position.y,
            body.position.z,
            body.rotation.x,
            body.rotation.y,
            body.rotation.z,
            body.rotation.w,
        ]
        .map(f32::to_bits);
        assert_eq!(bits, expected_bits, "{workers} workers top pose");
    }
}

#[test]
fn a_busy_world_steps_the_same_on_any_number_of_workers() {
    let _turn = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let mut serial = busy();
    let hashes: Vec<u64> = (0..180)
        .map(|_| {
            serial.step(DT);
            serial.state_hash()
        })
        .collect();
    assert!(
        serial.awake_body_count() < serial.body_count(),
        "some bodies fell asleep, so sleep is exercised"
    );
    for workers in WORKER_COUNTS {
        let mut world = busy();
        world.set_executor(Some(executor(workers)));
        for (step, &hash) in hashes.iter().enumerate() {
            world.step(DT);
            assert_eq!(world.state_hash(), hash, "{workers} workers, step {step}");
        }
    }
}

#[test]
fn an_executor_is_host_wiring() {
    let _turn = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let (mut world, top) = pyramid();
    world.set_executor(Some(executor(4)));
    world.step(DT);
    let snapshot = world.create_snapshot();
    world.restore_snapshot(&snapshot);
    assert_eq!(world.executor().map(|e| e.worker_count()), Some(4));
    assert!(PhysicsWorld::from_snapshot(&snapshot).executor().is_none());
    world.set_executor(None);
    world.step(DT);
    assert!(world.body(top).is_some());
}
