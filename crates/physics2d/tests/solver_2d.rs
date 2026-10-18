//! Integration tests for 2D rigid body dynamics and soft-step solver.

use korppi_math::{Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::narrowphase::collide_shapes;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::solver::{solve_contacts_2d, SolverConfig2d};

#[test]
fn static_body_remains_immobile_while_dynamic_body_integrates() {
    let mut static_body = RigidBody2d::new(BodyType::Static);
    static_body.position = Vec2::new(0.0, 0.0);
    static_body.linear_velocity = Vec2::new(10.0, 10.0); // should not move
    static_body.integrate_velocities(Vec2::new(0.0, -9.81), 0.016);
    static_body.integrate_positions(0.016);
    assert_eq!(static_body.position, Vec2::ZERO);

    let mut dynamic_body = RigidBody2d::new(BodyType::Dynamic);
    dynamic_body.set_mass(1.0, 0.5);
    dynamic_body.position = Vec2::new(0.0, 5.0);
    dynamic_body.integrate_velocities(Vec2::new(0.0, -10.0), 0.1);
    dynamic_body.integrate_positions(0.1);

    // After 0.1s under -10m/s^2 gravity, velocity = (0, -1.0)
    assert!((dynamic_body.linear_velocity - Vec2::new(0.0, -1.0)).length() < 1e-4);
    // Position = 5.0 - 0.1 = 4.9
    assert!((dynamic_body.position.y - 4.9).abs() < 1e-4);
}

#[test]
fn circle_settles_on_ground_using_soft_step_solver() {
    let mut ground = RigidBody2d::new(BodyType::Static);
    ground.position = Vec2::new(0.0, 0.0);
    let ground_shape = Shape2d::Polygon(Polygon2d::from_box(10.0, 1.0));

    let mut ball = RigidBody2d::new(BodyType::Dynamic);
    ball.set_mass(1.0, 0.5);
    ball.position = Vec2::new(0.0, 2.0); // radius 1.0, ground top is at y = 1.0
    let ball_shape = Shape2d::Circle(Circle2d::new(Vec2::ZERO, 1.0));

    let gravity = Vec2::new(0.0, -9.81);
    let dt = 1.0 / 60.0;
    let config = SolverConfig2d {
        sub_steps: 4,
        hertz: 60.0,
        damping_ratio: 1.0,
        friction: 0.3,
        restitution: 0.0,
    };

    // Simulate 60 steps (1 second)
    for _ in 0..60 {
        ball.integrate_velocities(gravity, dt);

        let sub_dt = dt / config.sub_steps as f32;
        for _ in 0..config.sub_steps {
            ball.integrate_positions(sub_dt);

            let xf_ground =
                Transform2d::from_translation_rotation(ground.position, ground.rotation);
            let xf_ball = Transform2d::from_translation_rotation(ball.position, ball.rotation);

            if let Some(mut manifold) =
                collide_shapes(&ground_shape, &xf_ground, &ball_shape, &xf_ball)
            {
                solve_contacts_2d(&mut ground, &mut ball, &mut manifold, sub_dt, &config);
            }
        }
    }

    // Ball should settle at y ≈ 2.0 (ground half_height 1.0 + ball radius 1.0)
    assert!(
        (ball.position.y - 2.0).abs() < 0.05,
        "Ball position y={:.3} should be close to 2.0",
        ball.position.y
    );
    assert!(
        ball.linear_velocity.length() < 0.1,
        "Ball velocity {:.3} should settle close to zero",
        ball.linear_velocity.length()
    );
}

#[test]
fn friction_decelerates_sliding_box_to_rest() {
    let mut ground = RigidBody2d::new(BodyType::Static);
    let ground_shape = Shape2d::Polygon(Polygon2d::from_box(100.0, 1.0));

    let mut slider = RigidBody2d::new(BodyType::Dynamic);
    slider.set_mass(1.0, 0.5);
    slider.position = Vec2::new(0.0, 1.5);
    slider.linear_velocity = Vec2::new(5.0, 0.0); // sliding right at 5 m/s
    let slider_shape = Shape2d::Polygon(Polygon2d::from_box(0.5, 0.5));

    let gravity = Vec2::new(0.0, -9.81);
    let dt = 1.0 / 60.0;
    let config = SolverConfig2d {
        sub_steps: 4,
        hertz: 60.0,
        damping_ratio: 1.0,
        friction: 0.5,
        restitution: 0.0,
    };

    // Simulate 120 steps (2 seconds)
    for _ in 0..120 {
        slider.integrate_velocities(gravity, dt);

        let sub_dt = dt / config.sub_steps as f32;
        for _ in 0..config.sub_steps {
            slider.integrate_positions(sub_dt);

            let xf_ground =
                Transform2d::from_translation_rotation(ground.position, ground.rotation);
            let xf_slider =
                Transform2d::from_translation_rotation(slider.position, slider.rotation);

            if let Some(mut manifold) =
                collide_shapes(&ground_shape, &xf_ground, &slider_shape, &xf_slider)
            {
                solve_contacts_2d(&mut ground, &mut slider, &mut manifold, sub_dt, &config);
            }
        }
    }

    // Box should have decelerated almost to zero velocity along X
    assert!(
        slider.linear_velocity.x < 0.1,
        "Slider x velocity {:.3} should be decelerated by friction",
        slider.linear_velocity.x
    );
}
