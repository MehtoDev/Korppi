//! Automated headless verification suite for the 2D Physics Showcase.

use korppi_math::Vec2;
use korppi_physics2d::body::BodyType;
use korppi_physics2d::joint::Joint2d;
use korppi_physics2d::world::{JointInstance2d, PhysicsWorld2d};
use physics_2d_showcase_lib::app::Physics2dShowcaseApp;

#[test]
fn test_all_galleries_run_300_ticks_stably() {
    let mut app = Physics2dShowcaseApp::new();
    let dt = 1.0 / 60.0;

    for gallery_idx in 0..app.gallery_count() {
        app.load_gallery(gallery_idx);

        for tick in 0..300 {
            app.step_simulation(dt);

            for (id, body) in app.world().bodies() {
                assert!(
                    body.position.x.is_finite() && body.position.y.is_finite(),
                    "Gallery {gallery_idx} tick {tick} body {id:?} position is non-finite: {:?}",
                    body.position
                );
                assert!(
                    body.linear_velocity.x.is_finite() && body.linear_velocity.y.is_finite(),
                    "Gallery {gallery_idx} tick {tick} body {id:?} velocity is non-finite: {:?}",
                    body.linear_velocity
                );
                assert!(
                    body.angular_velocity.is_finite(),
                    "Gallery {gallery_idx} tick {tick} body {id:?} angular velocity is non-finite"
                );
            }
        }
    }
}

#[test]
fn test_stacking_gallery_settles_and_bodies_sleep() {
    let mut app = Physics2dShowcaseApp::new();
    app.load_gallery(0); // StackingGallery

    let dt = 1.0 / 60.0;
    // Step for 360 ticks (6.0 seconds of simulation)
    for _ in 0..360 {
        app.step_simulation(dt);
    }

    let sleeping_bodies = app
        .world()
        .bodies()
        .filter(|(_, b)| b.body_type == BodyType::Dynamic && b.is_sleeping)
        .count();

    assert!(
        sleeping_bodies > 0,
        "Stacking bodies must settle and enter sleep state under resting gravity equilibrium, found {sleeping_bodies} sleeping"
    );
}

#[test]
fn test_ccd_bullet_does_not_tunnel_thin_wall() {
    let mut app = Physics2dShowcaseApp::new();
    app.load_gallery(4); // ContactVisualizerGallery

    let dt = 1.0 / 60.0;
    for tick in 0..60 {
        app.step_simulation(dt);

        // Find the high-speed bullet (started at x=0, y=4 moving at +350 m/s)
        for (_, body) in app.world().bodies() {
            if body.is_bullet {
                assert!(
                    body.position.x <= 9.0 + 0.05,
                    "Bullet tunneled through thin wall at tick {tick}! Pos: {:?}",
                    body.position
                );
            }
        }
    }
}

#[test]
fn test_kinetic_newtons_cradle_propagates_motion() {
    let mut app = Physics2dShowcaseApp::new();
    app.load_gallery(3); // KineticGallery

    let dt = 1.0 / 60.0;
    let mut rightmost_moved = false;

    for _ in 0..120 {
        app.step_simulation(dt);

        // In Newton's cradle, the right-most dynamic ball is body index 5 (rail is 1, balls 2..6)
        if let Some((_, b)) = app.world().bodies().nth(5) {
            if b.linear_velocity.x > 0.5 {
                rightmost_moved = true;
                break;
            }
        }
    }

    assert!(
        rightmost_moved,
        "Newton's cradle must transfer kinetic energy through the chain to the right-most ball"
    );
}

#[test]
fn test_ragdoll_revolute_joint_angle_limits() {
    let mut app = Physics2dShowcaseApp::new();
    app.load_gallery(2); // RagdollGallery

    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        app.step_simulation(dt);

        for (_, j_inst) in app.world().joints() {
            if let Joint2d::Revolute(ref rev) = j_inst.joint {
                if rev.enable_limit {
                    if let (Some(ba), Some(bb)) = (
                        app.world().body(j_inst.body_a),
                        app.world().body(j_inst.body_b),
                    ) {
                        let rel_angle = rev.angle(ba, bb);
                        assert!(
                            rel_angle >= rev.lower_angle - 0.15
                                && rel_angle <= rev.upper_angle + 0.15,
                            "Joint limit violated: angle {:.2} not in [{:.2}, {:.2}]",
                            rel_angle,
                            rev.lower_angle,
                            rev.upper_angle
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_mouse_picking_and_spring_dragging() {
    let mut app = Physics2dShowcaseApp::new();
    app.load_gallery(0); // Stacking

    // Query point on the top-most box of the vertical stack (x = -8.0, y = 10.0)
    let pick_pt = Vec2::new(-8.0, 9.9);
    app.pointer_down(pick_pt);

    assert_eq!(app.world().joint_count(), 1, "MouseJoint must be created");

    // Drag to new target (x = -3.0, y = 12.0)
    let new_target = Vec2::new(-3.0, 12.0);
    app.pointer_move(new_target);

    let dt = 1.0 / 60.0;
    for _ in 0..30 {
        app.step_simulation(dt);
    }

    // Release mouse joint
    app.pointer_up();
    assert_eq!(
        app.world().joint_count(),
        0,
        "the drag joint must be cleanly removed"
    );
}

/// How far a joint's rigid rows are from being satisfied, in meters; `None` for joints that are
/// springs or drives rather than rigid connections.
fn rigid_joint_error(world: &PhysicsWorld2d, joint: &JointInstance2d) -> Option<f32> {
    let a = world.body(joint.body_a)?;
    let b = world.body(joint.body_b)?;
    let anchor_gap = |local_a: Vec2, local_b: Vec2| {
        (b.position + b.rotation.rotate(local_b)) - (a.position + a.rotation.rotate(local_a))
    };
    let off_axis = |local_a: Vec2, local_b: Vec2, local_axis: Vec2| {
        let axis = a.rotation.rotate(local_axis);
        anchor_gap(local_a, local_b)
            .dot(Vec2::new(-axis.y, axis.x))
            .abs()
    };
    let (frame_a, frame_b) = joint.joint.local_frames();
    let (fa, fb, axis) = (
        frame_a.translation,
        frame_b.translation,
        frame_a.rotation.x_axis(),
    );
    match joint.joint {
        Joint2d::Revolute(_) => Some(anchor_gap(fa, fb).length()),
        Joint2d::Weld(j) if j.linear_hertz == 0.0 => Some(anchor_gap(fa, fb).length()),
        Joint2d::Prismatic(_) | Joint2d::Wheel(_) => Some(off_axis(fa, fb, axis)),
        Joint2d::Distance(j) if !j.enable_spring => {
            Some((anchor_gap(fa, fb).length() - j.length).abs())
        }
        _ => None,
    }
}

#[test]
fn test_rigid_joints_stay_attached_in_every_gallery() {
    let mut app = Physics2dShowcaseApp::new();
    let dt = 1.0 / 60.0;

    for gallery_idx in 0..app.gallery_count() {
        app.load_gallery(gallery_idx);
        for (joint_idx, (_, joint)) in app.world().joints().enumerate() {
            if let Some(error) = rigid_joint_error(app.world(), joint) {
                assert!(
                    error < 1e-4,
                    "Gallery {gallery_idx}: joint {joint_idx} starts {error:.4} m apart: {:?}",
                    joint.joint
                );
            }
        }
        // Ragdoll impacts stretch joints by up to ~6 cm; Box2D v3.2 stretches the same scenes
        // by 6.4 cm (ragdolls) and 0.9 cm (teeter). A joint with no effect drifted metres
        // (P2D-01).
        for tick in 0..300 {
            app.step_simulation(dt);
            for (joint_idx, (_, joint)) in app.world().joints().enumerate() {
                if let Some(error) = rigid_joint_error(app.world(), joint) {
                    assert!(
                        error < 0.1,
                        "Gallery {gallery_idx} tick {tick}: joint {joint_idx} came apart by {error:.4} m: {:?}",
                        joint.joint
                    );
                }
            }
        }
    }
}
