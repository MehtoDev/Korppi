//! Bodies integrate gravity, forces, damping and their spin as Box3D's bodies do (P3D-17).
//!
//! Box3D integrates velocities every substep (gravity times the body's gravity scale, forces,
//! Padé damping `1 / (1 + h c)`, and an implicit gyroscopic Newton step on the full inertia
//! tensor), then integrates positions with the speed caps and motion locks (`solver.c`). Before
//! FIX-3.2 Korppi integrated gravity and forces once per step with explicit damping, held the
//! angular velocity constant without torques (so asymmetric bodies never precessed or flipped),
//! had only a diagonal inertia, and lacked gravity scale, speed caps and motion locks.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_2_integration.c`):
//! 60 Hz, 4 substeps, 1 kg bodies without shapes, no damping unless a scene sets it.

use korppi_math::{Mat3, Quat, Vec3};
use korppi_physics::{BodyId3d, BodyType, MotionLocks3d, PhysicsWorld, RigidBody, MAX_ROTATION};

const DT: f32 = 1.0 / 60.0;

fn world(gravity: f32) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, gravity, 0.0));
    world
}

fn free_body(world: &mut PhysicsWorld, body: RigidBody) -> BodyId3d {
    world.create_body(body.with_enable_sleep(false))
}

fn dynamic() -> RigidBody {
    RigidBody::new(BodyType::Dynamic, Vec3::ZERO)
}

fn run(world: &mut PhysicsWorld, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
    }
}

fn close(value: f32, expected: f32, tolerance: f32) -> bool {
    (value - expected).abs() <= tolerance
}

#[test]
fn forces_and_gravity_are_integrated_every_substep() {
    let mut w = world(0.0);
    let pushed = free_body(&mut w, dynamic());
    w.body_mut(pushed)
        .unwrap()
        .apply_force(Vec3::new(10.0, 0.0, 0.0));
    w.step(DT);
    let body = w.body(pushed).unwrap();
    // Box3D: x = 0.00174 and vx = 0.16667 after one step, the force applied each substep.
    // Integrated once per step it went 0.00278.
    assert!(
        close(body.position.x, 0.001_736, 2e-6),
        "a 10 N force over one step should move 1 kg by 0.00174 m, moved {}",
        body.position.x
    );
    assert!(close(body.linear_velocity.x, 0.166_67, 1e-5));
    // The force lasts one step.
    w.step(DT);
    assert!(close(
        w.body(pushed).unwrap().linear_velocity.x,
        0.166_67,
        1e-5
    ));
    for _ in 0..58 {
        w.body_mut(pushed)
            .unwrap()
            .apply_force(Vec3::new(10.0, 0.0, 0.0));
        w.step(DT);
    }
    // Box3D: 4.85798 m after 60 steps.
    let x = w.body(pushed).unwrap().position.x;
    assert!(
        close(x, 4.857_98, 1e-4),
        "pushed 60 steps, the body is at {x}"
    );

    let mut w = world(-9.81);
    let falling = free_body(&mut w, dynamic());
    w.step(DT);
    // Box3D: y = -0.00170 after one step, -4.92544 with vy = -9.81001 after 1 s.
    assert!(close(w.body(falling).unwrap().position.y, -0.001_703, 2e-6));
    run(&mut w, 59);
    let body = w.body(falling).unwrap();
    assert!(
        close(body.position.y, -4.925_44, 1e-4),
        "a body falls 4.92544 m in 1 s in Box3D, fell {}",
        -body.position.y
    );
    assert!(close(body.linear_velocity.y, -9.81, 1e-4));
}

#[test]
fn damping_is_pade_and_off_by_default() {
    let defaults = RigidBody::default();
    assert_eq!(
        (defaults.linear_damping, defaults.angular_damping),
        (0.0, 0.0)
    );
    assert_eq!(defaults.gravity_scale, 1.0);

    let mut w = world(0.0);
    let mut body = dynamic();
    body.linear_damping = 2.0;
    body.angular_damping = 2.0;
    body.linear_velocity = Vec3::new(10.0, 0.0, 0.0);
    body.angular_velocity = Vec3::new(0.0, 0.0, 10.0);
    let damped = free_body(&mut w, body);
    run(&mut w, 60);

    // Box3D: 10 / (1 + 2 h)^240 = 1.36461 m/s and rad/s, after travelling 4.31768 m.
    let body = w.body(damped).unwrap();
    assert!(
        close(body.linear_velocity.x, 1.364_61, 1e-4),
        "{}",
        body.linear_velocity.x
    );
    assert!(
        close(body.angular_velocity.z, 1.364_61, 1e-4),
        "{}",
        body.angular_velocity.z
    );
    assert!(
        close(body.position.x, 4.317_68, 1e-3),
        "{}",
        body.position.x
    );
}

#[test]
fn gravity_scale_scales_the_fall() {
    // Box3D after 1 s: -2.46272 m at scale 0.5, 0 at 0, +4.92544 m at -1.
    for (scale, expected) in [(0.5, -2.462_72), (0.0, 0.0), (-1.0, 4.925_44)] {
        let mut w = world(-9.81);
        let mut body = dynamic();
        body.gravity_scale = scale;
        let id = free_body(&mut w, body);
        run(&mut w, 60);
        let y = w.body(id).unwrap().position.y;
        assert!(
            close(y, expected, 1e-4),
            "gravity scale {scale}: y = {y}, Box3D {expected}"
        );
    }
}

#[test]
fn speeds_are_capped_unless_fast_rotation_is_allowed() {
    let mut w = world(0.0);
    let mut body = dynamic();
    body.linear_velocity = Vec3::new(1000.0, 0.0, 0.0);
    body.angular_velocity = Vec3::new(0.0, 0.0, 100.0);
    let fast = free_body(&mut w, body);
    let mut wheel = dynamic().with_position(Vec3::new(0.0, 10.0, 0.0));
    wheel.angular_velocity = Vec3::new(0.0, 0.0, 100.0);
    wheel.allow_fast_rotation = true;
    let wheel = free_body(&mut w, wheel);
    assert_eq!(w.solver_config().max_linear_speed, 400.0);

    w.step(DT);

    // Box3D: 400 m/s (6.66667 m in the step) and a quarter turn per step, 47.12389 rad/s; a
    // body allowed fast rotation keeps 100 rad/s.
    let body = w.body(fast).unwrap();
    assert!(
        close(body.linear_velocity.x, 400.0, 1e-3),
        "{}",
        body.linear_velocity.x
    );
    assert!(
        close(body.position.x, 6.666_67, 1e-4),
        "{}",
        body.position.x
    );
    let cap = MAX_ROTATION / DT;
    assert!(close(cap, 47.123_89, 1e-3));
    assert!(
        close(body.angular_velocity.z, cap, 1e-3),
        "{}",
        body.angular_velocity.z
    );
    assert!(close(
        w.body(wheel).unwrap().angular_velocity.z,
        100.0,
        1e-3
    ));
}

#[test]
fn motion_locks_hold_their_axes() {
    let mut w = world(-9.81);
    let mut body = dynamic();
    body.motion_locks = MotionLocks3d {
        linear_y: true,
        angular_x: true,
        angular_z: true,
        ..MotionLocks3d::default()
    };
    body.linear_velocity = Vec3::X;
    let locked = free_body(&mut w, body);
    for _ in 0..60 {
        w.body_mut(locked)
            .unwrap()
            .apply_torque(Vec3::new(5.0, 1.0, 5.0));
        w.step(DT);
    }

    // Box3D: gravity cannot move it along y, the torque spins it only about y (0.99999 rad/s).
    let body = w.body(locked).unwrap();
    assert_eq!(body.position.y, 0.0);
    assert!(close(body.position.x, 1.0, 1e-3));
    assert!(
        close(body.angular_velocity.y, 1.0, 1e-3),
        "{}",
        body.angular_velocity.y
    );
    assert_eq!(
        (body.angular_velocity.x, body.angular_velocity.z),
        (0.0, 0.0)
    );
}

/// Spins a free body of principal inertia (1, 2, 3) at `spin` for 10 s. Returns the first step
/// whose body-frame spin about y is negative (none: `None`), the range of the angular
/// momentum's magnitude, and the smallest share of the spin about body z.
fn spinning_body(spin: Vec3) -> (Option<usize>, f32, f32, f32) {
    let mut w = world(0.0);
    let body = dynamic().with_inertia(Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0)));
    let id = free_body(&mut w, body);
    w.body_mut(id).unwrap().angular_velocity = spin;
    let (mut flip, mut lowest, mut highest, mut share_z) = (None, f32::MAX, 0.0f32, 1.0f32);
    for step in 1..=600 {
        w.step(DT);
        let body = w.body(id).unwrap();
        let local = body.rotation.conjugate() * body.angular_velocity;
        let momentum = (body.rotation * Vec3::new(local.x, 2.0 * local.y, 3.0 * local.z)).length();
        lowest = lowest.min(momentum);
        highest = highest.max(momentum);
        share_z = share_z.min(local.z / local.length());
        if flip.is_none() && local.y < 0.0 {
            flip = Some(step);
        }
    }
    (flip, lowest, highest, share_z)
}

#[test]
fn spin_about_the_intermediate_axis_flips() {
    let (flip, lowest, highest, _) = spinning_body(Vec3::new(0.01, 5.0, 0.0));

    // Box3D: the tennis-racket flip reverses the spin at step 158, while the implicit
    // gyroscopic step keeps |L| between 9.76969 and 10.00002. Before FIX-3.2 the angular
    // velocity stayed constant and never flipped.
    let flip = flip.expect("a body spun about its intermediate axis should flip");
    assert!(
        (150..=166).contains(&flip),
        "flipped at step {flip}, Box3D at 158"
    );
    assert!(
        lowest >= 9.7 && highest <= 10.01,
        "angular momentum should be nearly conserved, |L| ranged {lowest}-{highest}"
    );
}

#[test]
fn spin_about_the_major_axis_is_stable() {
    let (_, _, _, share_z) = spinning_body(Vec3::new(0.01, 0.0, 5.0));
    // Box3D: the spin stays about z (share 1.00000).
    assert!(
        share_z >= 0.99,
        "the spin wandered off the major axis to {share_z}"
    );
}

#[test]
fn full_inertia_tensor_couples_the_axes() {
    let (s, c) = (std::f32::consts::FRAC_PI_6).sin_cos();
    let rotation = Mat3::from_quat(Quat::from_rotation_z(std::f32::consts::FRAC_PI_6));
    let inertia = rotation * Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0)) * rotation.transpose();
    assert!(
        close(inertia.x_axis.y, c * s * (1.0 - 2.0), 1e-6),
        "off-diagonal term"
    );

    let mut w = world(0.0);
    let id = free_body(&mut w, dynamic().with_inertia(inertia));
    w.body_mut(id)
        .unwrap()
        .apply_torque(Vec3::new(6.0, 0.0, 0.0));
    w.step(DT);

    // Box3D: (0.0875000, 0.0216506, 0.0000056): the torque about x also turns the body about y.
    let spin = w.body(id).unwrap().angular_velocity;
    assert!(close(spin.x, 0.0875, 1e-6), "{spin:?}");
    assert!(close(spin.y, 0.021_650_6, 1e-6), "{spin:?}");
    assert!(close(spin.z, 0.000_005_6, 1e-6), "{spin:?}");
}

#[test]
fn forces_and_impulses_at_points_turn_the_body() {
    let mut w = world(0.0);
    let id = free_body(&mut w, dynamic());
    let body = w.body_mut(id).unwrap();
    body.apply_force_at_point(Vec3::new(0.0, 6.0, 0.0), Vec3::X);
    body.apply_impulse_at_point(Vec3::Z, Vec3::new(0.0, 2.0, 0.0));
    body.apply_angular_impulse(Vec3::new(0.0, 0.5, 0.0));
    w.step(DT);

    // Box3D: linear velocity (0, 0.1, 1), angular velocity (2, 0.5, 0.1).
    let body = w.body(id).unwrap();
    assert!((body.linear_velocity - Vec3::new(0.0, 0.1, 1.0)).length() < 1e-5);
    assert!(
        (body.angular_velocity - Vec3::new(2.0, 0.5, 0.1)).length() < 1e-5,
        "{:?}",
        body.angular_velocity
    );
}
