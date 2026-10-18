//! Automated headless verification suite for the 3D Physics Showcase.

use korppi_math::{Aabb, Ray, Vec3};
use korppi_physics::body::BodyType;
use physics_3d_showcase_lib::app::Physics3dShowcaseApp;

#[test]
fn test_all_6_galleries_run_300_ticks_stably() {
    let mut app = Physics3dShowcaseApp::new();
    let dt = 1.0 / 60.0;

    for gallery_idx in 0..app.gallery_count() {
        app.load_gallery(gallery_idx);

        for tick in 0..300 {
            app.step_simulation(dt);

            for id in 0..app.world().body_count() as u32 {
                if let Some(body) = app.world().body(id) {
                    assert!(
                        body.position.x.is_finite()
                            && body.position.y.is_finite()
                            && body.position.z.is_finite(),
                        "Gallery {gallery_idx} tick {tick} body {id} position is non-finite: {:?}",
                        body.position
                    );
                    assert!(
                        body.linear_velocity.x.is_finite()
                            && body.linear_velocity.y.is_finite()
                            && body.linear_velocity.z.is_finite(),
                        "Gallery {gallery_idx} tick {tick} body {id} velocity is non-finite: {:?}",
                        body.linear_velocity
                    );
                    assert!(
                        body.angular_velocity.x.is_finite()
                            && body.angular_velocity.y.is_finite()
                            && body.angular_velocity.z.is_finite(),
                        "Gallery {gallery_idx} tick {tick} body {id} angular velocity is non-finite: {:?}",
                        body.angular_velocity
                    );
                }
            }
        }
    }
}

#[test]
fn test_heavy_stacking_100_bodies_settle() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(0); // HeavyStackingGallery

    assert!(
        app.world().body_count() >= 120,
        "Heavy stacking gallery must contain at least 120 bodies, found {}",
        app.world().body_count()
    );

    let dt = 1.0 / 60.0;
    // Step for 180 ticks (3.0 seconds) to allow settling
    for _ in 0..180 {
        app.step_simulation(dt);
    }

    let sleeping_bodies = (0..app.world().body_count() as u32)
        .filter(|&id| {
            app.world()
                .body(id)
                .is_some_and(|b| b.body_type == BodyType::Dynamic && b.is_sleeping)
        })
        .count();

    assert!(
        sleeping_bodies >= 30,
        "Stacking bodies must settle and sleep under resting equilibrium, found {sleeping_bodies} sleeping"
    );
    assert!(
        app.world().contact_count() > 100,
        "Heavy stacking must generate robust contact points, found {}",
        app.world().contact_count()
    );
}

#[test]
fn test_restitution_bounce_height_scaling() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(2); // FrictionRestitutionGallery

    let dt = 1.0 / 60.0;
    // Low restitution sphere is at x = 8.0, z = -4.0 (e = 0.0)
    // High restitution sphere is at x = 8.0, z = 4.0 (e = 1.0)
    // They hit ground around tick 84 (t = 1.4s). Track post-impact rebound heights in ticks 95..140.
    let mut max_y_high_restitution = 0.0f32;
    let mut max_y_low_restitution = 0.0f32;

    for tick in 0..140 {
        app.step_simulation(dt);

        if tick >= 85 {
            for id in 0..app.world().body_count() as u32 {
                if let Some(body) = app.world().body(id) {
                    if (body.position.x - 8.0).abs() < 1.0 {
                        if (body.position.z - (-4.0)).abs() < 1.0 {
                            max_y_low_restitution = max_y_low_restitution.max(body.position.y);
                        } else if (body.position.z - 4.0).abs() < 1.0 {
                            max_y_high_restitution = max_y_high_restitution.max(body.position.y);
                        }
                    }
                }
            }
        }
    }

    assert!(
        max_y_high_restitution > max_y_low_restitution + 2.0,
        "High restitution sphere must bounce significantly higher than low restitution sphere (high: {max_y_high_restitution}, low: {max_y_low_restitution})"
    );
}

#[test]
fn test_ragdoll_joints_maintain_constraints() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(3); // Ragdoll3dGallery

    assert!(
        app.world().joint_count() >= 10,
        "Ragdoll gallery must contain articulated joints, found {}",
        app.world().joint_count()
    );

    let dt = 1.0 / 60.0;
    for tick in 0..120 {
        app.step_simulation(dt);

        for (joint_idx, joint) in app.world().joints().iter().enumerate() {
            let (body_a_id, body_b_id, anchor_a, anchor_b) = (
                joint.body_a(),
                joint.body_b(),
                joint.local_anchor_a(),
                joint.local_anchor_b(),
            );

            let ba = app.world().body(body_a_id).unwrap();
            let bb = app.world().body(body_b_id).unwrap();

            let world_anchor_a = ba.position + (ba.rotation * anchor_a);
            let world_anchor_b = bb.position + (bb.rotation * anchor_b);

            let separation = (world_anchor_a - world_anchor_b).length();
            assert!(
                separation < 0.25,
                "Joint {joint_idx} anchor constraint violated at tick {tick}: separation {separation:.4}m exceeds tolerance",
            );
        }
    }
}

#[test]
fn test_spatial_queries_and_raycast_hit() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(4); // SpatialQueriesGallery

    // Raycast straight at the target sphere at origin (0.0, 1.5, 0.0) from (0.0, 1.5, 10.0)
    let ray = Ray::new(Vec3::new(0.0, 1.5, 10.0), Vec3::new(0.0, 0.0, -1.0));
    let hit = app.world().ray_cast(&ray, 20.0);

    assert!(
        hit.is_some(),
        "Raycast must hit the target shape along Z axis"
    );
    let hit_info = hit.unwrap();
    assert!(
        hit_info.point.z > 0.0 && hit_info.point.z < 10.0,
        "Hit point Z must be between origin and 10.0, got {:?}",
        hit_info.point
    );
    assert!(
        hit_info.normal.z > 0.5,
        "Hit surface normal should face back toward the ray origin along +Z, got {:?}",
        hit_info.normal
    );

    // Query AABB volume around origin
    let query_box = Aabb {
        min: Vec3::new(-3.0, 0.0, -3.0),
        max: Vec3::new(3.0, 6.0, 3.0),
    };
    let overlapping_bodies = app.world().query_aabb(&query_box);

    assert!(
        !overlapping_bodies.is_empty(),
        "Query AABB around origin must find overlapping test shapes"
    );
}

#[test]
fn test_snapshot_rollback_bitwise_determinism() {
    let mut app = Physics3dShowcaseApp::new();
    app.load_gallery(5); // RollbackDeterminismGallery

    let dt = 1.0 / 60.0;

    // 1. Advance simulation 60 ticks
    for _ in 0..60 {
        app.step_simulation(dt);
    }

    // 2. Take reference snapshot S
    let snapshot = app.world().create_snapshot();

    // 3. Advance simulation 60 more ticks
    for _ in 0..60 {
        app.step_simulation(dt);
    }

    // Record expected state
    let mut expected_positions = Vec::new();
    let mut expected_rotations = Vec::new();
    let mut expected_velocities = Vec::new();
    for id in 0..app.world().body_count() as u32 {
        if let Some(b) = app.world().body(id) {
            expected_positions.push(b.position);
            expected_rotations.push(b.rotation);
            expected_velocities.push(b.linear_velocity);
        }
    }

    // 4. Restore snapshot S
    app.world_mut().restore_snapshot(&snapshot);

    // 5. Advance simulation 60 ticks again from restored snapshot
    for _ in 0..60 {
        app.step_simulation(dt);
    }

    // 6. Assert bit-exact determinism across all bodies
    for id in 0..app.world().body_count() as u32 {
        if let Some(b) = app.world().body(id) {
            let idx = id as usize;
            assert_eq!(
                b.position, expected_positions[idx],
                "Body {id} position mismatch after snapshot restore and replay"
            );
            assert_eq!(
                b.rotation, expected_rotations[idx],
                "Body {id} rotation mismatch after snapshot restore and replay"
            );
            assert_eq!(
                b.linear_velocity, expected_velocities[idx],
                "Body {id} velocity mismatch after snapshot restore and replay"
            );
        }
    }
}
