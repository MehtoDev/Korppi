//! Joints are Box2D v3.2's (P2D-22).
//!
//! Every joint joins two local frames, one on each body (Box2D `localFrameA`, `localFrameB`):
//! the frames' rotations replace the reference angle and the slider axis. As the contacts, the
//! joints are prepared once per step (frames relative to the bodies' centres of mass, effective
//! masses, softness) and solved per substep on the bodies' motion since the step started, so
//! that a joint and a contact see the same body. Before FIX-2.6 the joints recomputed their
//! anchors from the bodies' poses in every pass, distance joints had no limits, motor or spring
//! force range, welds one stiffness for both rows, the motor joint was v3.1's, and there were no
//! mover, pogo or filter joints or joint queries. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_6_joints.c`, with rotations from the exact sine and
//! cosine as `Rot2::from_angle`.

use korppi_math::{Rot2, Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::{Collider2d, Filter2d};
use korppi_physics2d::joint::{
    DistanceJoint2d, Joint2d, MotorJoint2d, MoverJoint2d, PogoJoint2d, PrismaticJoint2d,
    RevoluteJoint2d, WeldJoint2d, WheelJoint2d,
};
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, JointId2d, PhysicsWorld2d, WorldConfig2d};
use std::f32::consts::FRAC_PI_2;

const DT: f32 = 1.0 / 60.0;

fn world(sleep: bool) -> PhysicsWorld2d {
    PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(WorldConfig2d {
            enable_sleep: sleep,
            ..WorldConfig2d::default()
        })
}

fn run(world: &mut PhysicsWorld2d, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

fn frame(x: f32, y: f32, angle: f32) -> Transform2d {
    Transform2d::from_translation_rotation(Vec2::new(x, y), Rot2::from_angle(angle))
}

fn ground(world: &mut PhysicsWorld2d) -> BodyId2d {
    world.create_body(RigidBody2d::new(BodyType::Static))
}

/// A dynamic box of density 1 that collides with none of the other boxes.
fn add_box(world: &mut PhysicsWorld2d, x: f32, y: f32, hx: f32, hy: f32) -> BodyId2d {
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, y)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(hx, hy))).with_filter(Filter2d {
            group_index: -1,
            ..Filter2d::DEFAULT
        }),
    );
    id
}

fn assert_near(actual: f32, expected: f32, tolerance: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual}, Box2D {expected}"
    );
}

/// Asserts body `id`'s pose and velocities against Box2D's (x, y, angle, vx, vy, w).
fn assert_body(world: &PhysicsWorld2d, id: BodyId2d, box2d: [f32; 6], tolerance: f32, name: &str) {
    let body = world.body(id).unwrap();
    let actual = [
        body.position.x,
        body.position.y,
        body.rotation.angle(),
        body.linear_velocity.x,
        body.linear_velocity.y,
        body.angular_velocity,
    ];
    for ((a, e), what) in actual
        .into_iter()
        .zip(box2d)
        .zip(["x", "y", "angle", "vx", "vy", "w"])
    {
        assert_near(a, e, tolerance, &format!("{name} {what}"));
    }
}

fn joint(world: &PhysicsWorld2d, id: JointId2d) -> Joint2d {
    world.joint(id).unwrap().joint
}

#[test]
fn a_revolute_chain_swings_like_box2d() {
    // Ten links hanging sideways from a hinge, released: Box2D v3.2 after 2 s.
    let mut world = world(false);
    let mut previous = ground(&mut world);
    for i in 0..10 {
        let link = add_box(&mut world, 0.5 + i as f32, 10.0, 0.5, 0.125);
        let frame_a = if i == 0 {
            frame(0.0, 10.0, 0.0)
        } else {
            frame(0.5, 0.0, 0.0)
        };
        world.create_joint(
            previous,
            link,
            Joint2d::Revolute(
                RevoluteJoint2d::default().with_local_frames(frame_a, frame(-0.5, 0.0, 0.0)),
            ),
        );
        previous = link;
    }
    run(&mut world, 120);
    assert_body(
        &world,
        previous,
        [
            -7.272_90, 4.330_39, -2.716_81, -7.585_63, 5.832_15, 0.231_19,
        ],
        2e-4,
        "chain end",
    );
}

#[test]
fn revolute_springs_limits_and_motors_act_like_box2d() {
    // A 2 m bar hinged at its left end, level, with the hinge frame turned so that the joint
    // angle starts at 0.3. Box2D v3.2 after 1.5 s: (joint angle, constraint torque, bar pose).
    let cases = [
        (
            "spring",
            RevoluteJoint2d::default()
                .with_spring(2.0, 0.5)
                .with_target_angle(0.5),
            0.312_64,
            0.0,
            [0.999_92, 5.012_35, 0.012_63, -0.000_03, 0.002_13, 0.002_13],
        ),
        (
            "limit",
            RevoluteJoint2d::default().with_limits(-0.2, 0.6),
            -0.200_18,
            3.51,
            [0.877_41, 4.520_19, -0.500_16, 0.0, 0.0, 0.0],
        ),
        (
            "motor",
            RevoluteJoint2d::default()
                .with_motor(1.0, 100.0)
                .with_limits(-1.0, 1.0),
            1.005_07,
            3.046_3,
            [0.761_68, 5.647_88, 0.705_07, 0.0, 0.0, 0.0],
        ),
    ];
    for (name, revolute, angle, torque, pose) in cases {
        let mut world = world(false);
        let hinge = ground(&mut world);
        let bar = add_box(&mut world, 1.0, 5.0, 1.0, 0.1);
        let revolute = revolute.with_local_frames(frame(0.0, 5.0, -0.3), frame(-1.0, 0.0, 0.0));
        let id = world.create_joint(hinge, bar, Joint2d::Revolute(revolute));
        run(&mut world, 90);
        let Joint2d::Revolute(revolute) = joint(&world, id) else {
            unreachable!()
        };
        let (a, b) = (world.body(hinge).unwrap(), world.body(bar).unwrap());
        assert_near(revolute.angle(a, b), angle, 1e-4, &format!("{name} angle"));
        let actual_torque = world.joint_constraint_torque(id).unwrap();
        assert_near(actual_torque, torque, 1e-3, &format!("{name} torque"));
        assert_body(&world, bar, pose, 1e-4, name);
    }
}

#[test]
fn distance_joints_have_springs_force_ranges_limits_and_motors() {
    // A bob hung 2 m from a point 2 m to its left, released. Box2D v3.2 after 1.5 s: (current
    // length, constraint force, linear separation, bob pose).
    let base = DistanceJoint2d::default()
        .with_local_frames(frame(0.0, 5.0, 0.0), frame(0.0, 0.25, 0.0))
        .with_length(2.0);
    let cases = [
        (
            "rod",
            base,
            2.000_13,
            (0.373_2, 0.014_8),
            0.000_13,
            [
                -2.003_50, 4.670_91, -0.019_73, -0.104_11, 2.558_09, -0.011_34,
            ],
        ),
        (
            "spring held by limits",
            base.with_spring(1.0, 0.1).with_limits(1.5, 2.5),
            2.453_84,
            (2.427_2, 1.586_9),
            0.0,
            [
                -2.163_08, 3.432_36, -0.452_18, -2.025_83, 3.204_01, -4.533_33,
            ],
        ),
        (
            "spring with a force range",
            base.with_spring(5.0, 0.7)
                .with_spring_force_range(-5.0, 5.0),
            2.780_88,
            (3.365_6, 3.697_6),
            0.0,
            [
                -2.121_75, 2.935_25, -1.537_95, -0.901_02, 2.779_64, -7.002_32,
            ],
        ),
        (
            "winch motor",
            base.with_spring(0.0, 0.0)
                .with_motor(-0.5, 20.0)
                .with_limits(0.5, 3.0),
            1.329_73,
            (2.177_6, -0.985_4),
            0.0,
            [
                -1.403_99, 5.707_68, -2.262_65, 1.752_97, 1.430_18, -7.625_71,
            ],
        ),
    ];
    for (name, distance, length, force, separation, pose) in cases {
        let mut world = world(false);
        let anchor = ground(&mut world);
        let bob = add_box(&mut world, 2.0, 5.0, 0.25, 0.25);
        let id = world.create_joint(anchor, bob, Joint2d::Distance(distance));
        run(&mut world, 90);
        let Joint2d::Distance(distance) = joint(&world, id) else {
            unreachable!()
        };
        let (a, b) = (world.body(anchor).unwrap(), world.body(bob).unwrap());
        assert_near(
            distance.current_length(a, b),
            length,
            1e-4,
            &format!("{name} length"),
        );
        let f = world.joint_constraint_force(id).unwrap();
        assert_near(f.x, force.0, 1e-3, &format!("{name} force x"));
        assert_near(f.y, force.1, 1e-3, &format!("{name} force y"));
        let s = world.joint_linear_separation(id).unwrap();
        assert_near(s, separation, 1e-4, &format!("{name} separation"));
        assert_body(&world, bob, pose, 1e-3, name);
    }
}

#[test]
fn prismatic_springs_limits_and_motors_act_like_box2d() {
    // A box on a slider turned 30° up from the x-axis. Box2D v3.2 after 1.5 s: (translation,
    // speed, constraint force, box pose).
    let axis = std::f32::consts::FRAC_PI_6;
    let base =
        PrismaticJoint2d::default().with_local_frames(frame(0.0, 2.0, axis), frame(0.0, 0.0, axis));
    let cases = [
        (
            "spring",
            base.with_spring(2.0, 0.3).with_target_translation(1.0),
            0.968_46,
            -0.018_28,
            (-2.165_1, 3.75),
            [0.838_74, 2.484_18, 0.0, -0.015_83, -0.009_14, 0.0],
        ),
        (
            "limit",
            base.with_limits(-0.75, 1.5),
            -0.750_04,
            0.0,
            (0.0, 5.0),
            [-0.649_52, 1.624_93, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "motor",
            base.with_motor(2.0, 50.0).with_limits(-1.0, 1.25),
            1.250_67,
            0.0,
            (0.0, 5.0),
            [1.083_14, 2.625_28, 0.0, 0.0, 0.0, 0.0],
        ),
    ];
    for (name, prismatic, translation, speed, force, pose) in cases {
        let mut world = world(false);
        let rail = ground(&mut world);
        let slider = add_box(&mut world, 0.0, 2.0, 0.5, 0.25);
        let id = world.create_joint(rail, slider, Joint2d::Prismatic(prismatic));
        run(&mut world, 90);
        let Joint2d::Prismatic(prismatic) = joint(&world, id) else {
            unreachable!()
        };
        let (a, b) = (world.body(rail).unwrap(), world.body(slider).unwrap());
        assert_near(
            prismatic.translation(a, b),
            translation,
            1e-4,
            &format!("{name} translation"),
        );
        assert_near(prismatic.speed(a, b), speed, 1e-4, &format!("{name} speed"));
        let f = world.joint_constraint_force(id).unwrap();
        assert_near(f.x, force.0, 1e-3, &format!("{name} force x"));
        assert_near(f.y, force.1, 1e-3, &format!("{name} force y"));
        assert_body(&world, slider, pose, 1e-4, name);
    }
}

#[test]
fn a_car_on_wheel_joints_drives_like_box2d() {
    // A chassis on two wheels with soft, limited suspension and a driven rear wheel. Box2D
    // v3.2 after 2 s: the chassis at x = 6.6767 doing 3.986 m/s.
    let mut world = world(false);
    let floor = world.create_body(RigidBody2d::new(BodyType::Static));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::offset_box(
            50.0,
            0.5,
            Vec2::new(0.0, -0.5),
            Rot2::IDENTITY,
        )))
        .with_friction(0.9),
    );
    let chassis =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 1.0)));
    world.create_collider(
        chassis,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(1.5, 0.25))),
    );
    let mut rear = None;
    for (i, x) in [-1.0f32, 1.0].into_iter().enumerate() {
        let wheel =
            world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(x, 0.4)));
        world.create_collider(
            wheel,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.4))).with_friction(0.9),
        );
        let mut suspension = WheelJoint2d::default()
            .with_local_frames(frame(x, -0.6, FRAC_PI_2), frame(0.0, 0.0, FRAC_PI_2))
            .with_spring(5.0, 0.7)
            .with_limits(-0.25, 0.25);
        if i == 0 {
            suspension = suspension.with_motor(-10.0, 20.0);
            rear = Some(wheel);
        }
        world.create_joint(chassis, wheel, Joint2d::Wheel(suspension));
    }
    run(&mut world, 120);
    assert_body(
        &world,
        chassis,
        [6.676_70, 0.969_85, 0.0, 3.986_33, 0.000_33, -0.000_11],
        2e-4,
        "chassis",
    );
    assert_body(
        &world,
        rear.unwrap(),
        [
            5.676_73, 0.396_89, -0.880_69, 3.985_89, -0.196_89, -9.996_67,
        ],
        2e-4,
        "rear wheel",
    );
}

#[test]
fn welds_are_rigid_or_have_separate_linear_and_angular_springs() {
    // A cantilever of five boxes welded end to end to a wall, rigid then soft (5 Hz linear,
    // 2 Hz angular). Box2D v3.2 after 1.5 s: (root force, root torque, root linear and angular
    // separation, tip pose).
    let cases = [
        (
            "rigid",
            WeldJoint2d::default(),
            (0.079_5, 14.840_5),
            39.618_1,
            0.001_610,
            -0.012_711,
            [4.497_11, 4.845_27, -0.044_39, 0.004_94, 0.132_98, 0.040_25],
        ),
        (
            "soft",
            WeldJoint2d::default()
                .with_linear_spring(5.0, 0.7)
                .with_angular_spring(2.0, 0.5),
            (10.692_6, 16.080_3),
            6.917_1,
            0.0,
            0.0,
            [
                -2.633_88, 0.806_01, -2.167_20, -5.912_69, 4.228_06, -0.154_70,
            ],
        ),
    ];
    for (name, weld, force, torque, linear, angular, tip) in cases {
        let mut world = world(false);
        let mut previous = ground(&mut world);
        let mut root = None;
        for i in 0..5 {
            let link = add_box(&mut world, 0.5 + i as f32, 5.0, 0.5, 0.125);
            let frame_a = if i == 0 {
                frame(0.0, 5.0, 0.0)
            } else {
                frame(0.5, 0.0, 0.0)
            };
            let id = world.create_joint(
                previous,
                link,
                Joint2d::Weld(weld.with_local_frames(frame_a, frame(-0.5, 0.0, 0.0))),
            );
            root.get_or_insert(id);
            previous = link;
        }
        run(&mut world, 90);
        let root = root.unwrap();
        let f = world.joint_constraint_force(root).unwrap();
        assert_near(f.x, force.0, 2e-3, &format!("{name} force x"));
        assert_near(f.y, force.1, 2e-3, &format!("{name} force y"));
        let t = world.joint_constraint_torque(root).unwrap();
        assert_near(t, torque, 2e-3, &format!("{name} torque"));
        let s = world.joint_linear_separation(root).unwrap();
        assert_near(s, linear, 1e-5, &format!("{name} linear separation"));
        let s = world.joint_angular_separation(root).unwrap();
        assert_near(s, angular, 1e-5, &format!("{name} angular separation"));
        assert_body(&world, previous, tip, 1e-3, name);
    }
}

#[test]
fn the_motor_joint_drives_with_springs_and_velocities_like_box2d() {
    // Springs pull a box from (0, 2) to a frame at (3, 4) turned 1 rad: Box2D v3.2 has it
    // there within 0.5 s, then holding its weight. Velocities drive a second box at
    // (1, 0.5) m/s and 2 rad/s against gravity, with at most 15 N and 0.5 N m.
    let mut world = world(false);
    let anchor = ground(&mut world);
    let body = add_box(&mut world, 0.0, 2.0, 0.5, 0.5);
    let id = world.create_joint(
        anchor,
        body,
        Joint2d::Motor(
            MotorJoint2d::default()
                .with_local_frames(frame(3.0, 4.0, 1.0), Transform2d::IDENTITY)
                .with_linear_spring(4.0, 0.7, 500.0)
                .with_angular_spring(4.0, 0.7, 50.0),
        ),
    );
    run(&mut world, 30);
    assert_body(
        &world,
        body,
        [2.999_36, 3.983_74, 0.999_88, 0.013_18, 0.008_76, 0.003_57],
        1e-4,
        "spring after 0.5 s",
    );
    run(&mut world, 150);
    let f = world.joint_constraint_force(id).unwrap();
    assert_near(f.x, 0.0, 1e-3, "spring force x");
    assert_near(f.y, 10.0, 1e-3, "spring force y");
    assert_near(
        world.joint_constraint_torque(id).unwrap(),
        0.0,
        1e-3,
        "spring torque",
    );
    assert_body(
        &world,
        body,
        [3.0, 3.984_17, 1.000_01, 0.0, 0.0, 0.0],
        1e-4,
        "spring after 3 s",
    );

    let mut world = self::world(false);
    let anchor = ground(&mut world);
    let body = add_box(&mut world, 0.0, 2.0, 0.5, 0.5);
    world.create_joint(
        anchor,
        body,
        Joint2d::Motor(
            MotorJoint2d::default()
                .with_linear_velocity(Vec2::new(1.0, 0.5), 15.0)
                .with_angular_velocity(2.0, 0.5),
        ),
    );
    run(&mut world, 60);
    assert_body(
        &world,
        body,
        [0.953_29, 2.438_59, 1.337_47, 1.0, 0.5, 2.0],
        1e-4,
        "velocity",
    );
}

#[test]
fn mover_pogo_and_filter_joints_act_like_box2d() {
    // A mover pushes a box sideways at 2 m/s with at most 3 N, gravity free: Box2D v3.2 has it
    // at x = 1.3375 after 1 s, falling at 10 m/s.
    let mut world = world(false);
    let anchor = ground(&mut world);
    let body = add_box(&mut world, 0.0, 5.0, 0.5, 0.5);
    world.create_joint(
        anchor,
        body,
        Joint2d::Mover(MoverJoint2d::new(Vec2::new(2.0, 0.0), Vec2::new(3.0, 0.0))),
    );
    run(&mut world, 60);
    assert_body(
        &world,
        body,
        [1.337_50, -0.020_83, 0.0, 2.0, -10.0, 0.0],
        1e-4,
        "mover",
    );

    // A pogo of rest length 1 pushes a box up with at most 30 N and never pulls: it settles at
    // 1.41667 m, carrying the 10 N weight.
    let mut world = self::world(false);
    let anchor = ground(&mut world);
    let body = add_box(&mut world, 0.0, 1.5, 0.5, 0.5);
    let id = world.create_joint(
        anchor,
        body,
        Joint2d::Pogo(PogoJoint2d::new(Vec2::Y, 1.0, 3.0, 0.7).with_force_limits(0.0, 30.0)),
    );
    run(&mut world, 120);
    let Joint2d::Pogo(pogo) = joint(&world, id) else {
        unreachable!()
    };
    let (a, b) = (world.body(anchor).unwrap(), world.body(body).unwrap());
    assert_near(pogo.length(a, b), 1.416_67, 1e-4, "pogo length");
    let f = world.joint_constraint_force(id).unwrap();
    assert_near(f.y, 10.0, 1e-3, "pogo force");
    assert_body(
        &world,
        body,
        [0.0, 1.416_67, 0.0, 0.0, 0.0, 0.0],
        1e-4,
        "pogo",
    );

    // A filter joint lets a box fall through the ground box it starts on.
    let mut world = self::world(false);
    let floor =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        floor,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(5.0, 0.5))),
    );
    let body =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 0.5)));
    world.create_collider(
        body,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5))),
    );
    world.create_joint(floor, body, Joint2d::Filter);
    run(&mut world, 30);
    assert_body(
        &world,
        body,
        [0.0, -0.760_42, 0.0, 0.0, -5.0, 0.0],
        1e-4,
        "filtered",
    );
}

#[test]
fn joint_queries_report_separations_and_wake_bodies() {
    // A revolute limited to ±0.2 rad, its bar teleported to -0.5 rad: Box2D v3.2 reports an
    // angular separation of 0.29998 and no linear separation; once asleep, waking the joint's
    // bodies wakes the bar.
    let mut world = world(true);
    let hinge = ground(&mut world);
    let bar = add_box(&mut world, 1.0, 5.0, 1.0, 0.1);
    let id = world.create_joint(
        hinge,
        bar,
        Joint2d::Revolute(
            RevoluteJoint2d::default()
                .with_local_frames(frame(0.0, 5.0, 0.0), frame(-1.0, 0.0, 0.0))
                .with_limits(-0.2, 0.2),
        ),
    );
    world.set_transform(
        bar,
        Vec2::new(0.877_582_6, 4.520_574_5),
        Rot2::from_angle(-0.5),
    );
    assert_near(
        world.joint_angular_separation(id).unwrap(),
        0.299_98,
        1e-5,
        "angular",
    );
    assert_near(
        world.joint_linear_separation(id).unwrap(),
        0.0,
        1e-5,
        "linear",
    );
    run(&mut world, 120);
    assert!(world.body(bar).unwrap().is_sleeping, "the bar fell asleep");
    assert!(world.wake_joint_bodies(id));
    assert!(
        !world.body(bar).unwrap().is_sleeping,
        "waking the joint wakes the bar"
    );
}
