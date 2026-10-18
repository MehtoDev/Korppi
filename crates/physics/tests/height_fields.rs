//! Height fields are Box3D's: a regular grid of heights with edge flags, one-sided
//! triangles, hole masking, and contacts that hold manifolds per normal cluster (P3D-12, P3D-10).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_5_heightfields.c`):
//! 60 Hz, 4 substeps, no sleep, density 1000 kg/m³, static height fields.

use korppi_math::{Quat, Vec3};
use korppi_physics::{
    BodyType, Collider, HeightField3d, HeightFieldOptions, MeshEdge, PhysicsWorld, RigidBody, Shape,
};

const DT: f32 = 1.0 / 60.0;

fn edge_counts(hf: &HeightField3d) -> (usize, usize, usize) {
    let mut counts = (0, 0, 0);
    for t in 0..hf.triangle_count() {
        for e in 0..3 {
            match hf.triangle(t).edge(e) {
                MeshEdge::Flat => counts.0 += 1,
                MeshEdge::Concave => counts.1 += 1,
                MeshEdge::Convex => counts.2 += 1,
                MeshEdge::Open => {}
            }
        }
    }
    counts
}

#[test]
fn height_field_construction_and_bounds() {
    // 4x4 flat grid with scale (1, 1, 1)
    let grid = HeightField3d::grid(4, 4, Vec3::ONE).expect("grid builds");
    assert_eq!(grid.triangle_count(), 18);
    let (min, max) = grid.bounds();
    assert_eq!(min, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(max, Vec3::new(3.0, 0.0, 3.0));
    assert_eq!(edge_counts(&grid), (42, 0, 0));
    assert_eq!(grid.triangle(0).flags, 34);
    assert_eq!(grid.triangle(grid.triangle_count() - 1).flags, 34);

    // 5x5 wave with scale (0.5, 0.4, 0.5)
    let wave =
        HeightField3d::wave(5, 5, Vec3::new(0.5, 0.4, 0.5), 0.25, 0.25).expect("wave builds");
    assert_eq!(wave.triangle_count(), 32);
    let (min, max) = wave.bounds();
    assert!((min - Vec3::new(0.0, -0.4, 0.0)).length() <= 1e-4);
    assert!((max - Vec3::new(2.0, 0.4, 2.0)).length() <= 1e-4);
    assert_eq!(edge_counts(&wave), (0, 40, 40));
    assert_eq!(wave.triangle(0).flags, 2);
    assert_eq!(wave.triangle(wave.triangle_count() - 1).flags, 2);
}

#[test]
fn rays_hit_heightfield_fronts() {
    let heights = vec![0.0f32; 16];
    let hf = HeightField3d::new(4, 4, Vec3::ONE, &heights).expect("heightfield builds");

    // Ray down
    let hit = hf
        .ray_cast(Vec3::new(1.25, 10.0, 1.25), Vec3::new(0.0, -20.0, 0.0), 1.0)
        .expect("ray hits front face");
    assert!((hit.fraction - 0.5).abs() <= 1e-4);
    assert!((hit.point - Vec3::new(1.25, 0.0, 1.25)).length() <= 1e-3);
    assert!((hit.normal - Vec3::Y).length() <= 1e-4);

    // Ray from below (facing up) misses
    assert!(hf
        .ray_cast(Vec3::new(1.25, -10.0, 1.25), Vec3::new(0.0, 20.0, 0.0), 1.0)
        .is_none());
}

#[test]
fn shapes_rest_on_heightfield() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let heights = vec![0.0f32; 121];
    let hf = HeightField3d::new(11, 11, Vec3::ONE, &heights).expect("hf builds");
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::HeightField(hf)).with_friction(0.6),
    );

    let capsule = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(3.0, 0.3, 3.0))
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
            .with_enable_sleep(false),
    );
    world.create_collider(
        capsule,
        Collider::new(Shape::Capsule {
            radius: 0.1,
            half_height: 0.3,
        })
        .with_friction(0.6),
    );

    let cube = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(5.0, 0.7, 5.0)).with_enable_sleep(false),
    );
    world.create_collider(
        cube,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.6),
    );

    let ball = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(7.0, 0.4, 7.0)).with_enable_sleep(false),
    );
    world.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.2 }).with_friction(0.6),
    );

    for _ in 0..120 {
        world.step(DT);
    }

    let y_cap = world.body(capsule).unwrap().position.y;
    let y_cube = world.body(cube).unwrap().position.y;
    let y_ball = world.body(ball).unwrap().position.y;

    assert!((y_cap - 0.104_95).abs() <= 5e-4, "capsule y: {y_cap}");
    assert!((y_cube - 0.504_92).abs() <= 5e-4, "cube y: {y_cube}");
    assert!((y_ball - 0.204_92).abs() <= 5e-4, "ball y: {y_ball}");
}

#[test]
fn holes_let_bodies_fall_through() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let heights = vec![0.0f32; 25];
    let mut materials = vec![0u8; 16];
    materials[10] = 0xFF; // hole at cell (2, 2)

    let hf = HeightField3d::with_options(
        5,
        5,
        Vec3::ONE,
        &heights,
        HeightFieldOptions {
            materials: Some(materials),
            clockwise: false,
        },
    )
    .expect("hf builds");

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(ground, Collider::new(Shape::HeightField(hf)));

    let solid_ball = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(1.5, 1.0, 1.5)).with_enable_sleep(false),
    );
    world.create_collider(solid_ball, Collider::new(Shape::Sphere { radius: 0.2 }));

    let hole_ball = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(2.5, 1.0, 2.5)).with_enable_sleep(false),
    );
    world.create_collider(hole_ball, Collider::new(Shape::Sphere { radius: 0.2 }));

    for _ in 0..60 {
        world.step(DT);
    }

    let y_solid = world.body(solid_ball).unwrap().position.y;
    let y_hole = world.body(hole_ball).unwrap().position.y;

    assert!(
        (y_solid - 0.204_92).abs() <= 5e-4,
        "solid ball y: {y_solid}"
    );
    assert!(y_hole < -0.5, "hole ball should fall through: {y_hole}");
}

#[test]
fn boxes_slide_over_internal_edges() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let heights = vec![0.0f32; 1681];
    let hf = HeightField3d::new(41, 41, Vec3::ONE, &heights).expect("hf builds");

    let ground = world.create_body(RigidBody::new(
        BodyType::Static,
        Vec3::new(-20.0, 0.0, -20.0),
    ));
    world.create_collider(
        ground,
        Collider::new(Shape::HeightField(hf)).with_friction(0.0),
    );

    let mut body =
        RigidBody::new(BodyType::Dynamic, Vec3::new(-8.0, 0.505, -3.0)).with_enable_sleep(false);
    body.linear_velocity = Vec3::new(6.0, 0.0, 4.0);
    let box_id = world.create_body(body);
    world.create_collider(
        box_id,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.0),
    );

    let (mut max_vy, mut max_w) = (0.0f32, 0.0f32);
    for step in 0..120 {
        world.step(DT);
        let b = world.body(box_id).unwrap();
        if step >= 10 {
            max_vy = max_vy.max(b.linear_velocity.y.abs());
            max_w = max_w.max(b.angular_velocity.length());
        }
    }
    assert!(max_vy <= 0.002, "a ghost bump: |vy| reached {max_vy}");
    assert!(max_w <= 0.002, "a ghost bump: |w| reached {max_w}");
    let p = world.body(box_id).unwrap().position;
    assert!(
        (p - Vec3::new(3.999_80, 0.504_94, 4.999_18)).length() <= 1e-3,
        "{p:?}"
    );
}

#[test]
fn balls_roll_over_internal_edges() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    let heights = vec![0.0f32; 1681];
    let hf = HeightField3d::new(41, 41, Vec3::ONE, &heights).expect("hf builds");

    let ground = world.create_body(RigidBody::new(
        BodyType::Static,
        Vec3::new(-20.0, 0.0, -20.0),
    ));
    world.create_collider(
        ground,
        Collider::new(Shape::HeightField(hf)).with_friction(0.6),
    );

    let r = 0.25;
    let mut body =
        RigidBody::new(BodyType::Dynamic, Vec3::new(-8.0, r + 0.005, 2.0)).with_enable_sleep(false);
    let v0 = Vec3::new(5.0, 0.0, 1.0);
    body.linear_velocity = v0;
    body.angular_velocity = Vec3::new(v0.z / r, 0.0, -v0.x / r);
    let ball_id = world.create_body(body);
    world.create_collider(
        ball_id,
        Collider::new(Shape::Sphere { radius: r }).with_friction(0.6),
    );

    let (mut heights_range, mut vys_range) = ((f32::MAX, f32::MIN), (f32::MAX, f32::MIN));
    for step in 0..120 {
        world.step(DT);
        let b = world.body(ball_id).unwrap();
        if step >= 20 {
            heights_range = (
                heights_range.0.min(b.position.y),
                heights_range.1.max(b.position.y),
            );
            vys_range = (
                vys_range.0.min(b.linear_velocity.y),
                vys_range.1.max(b.linear_velocity.y),
            );
        }
    }
    assert!(
        heights_range.1 - heights_range.0 <= 1e-4,
        "a ghost bump: {heights_range:?}"
    );
    assert!(
        vys_range.1 - vys_range.0 <= 1e-3,
        "a ghost bump: {vys_range:?}"
    );
    let p = world.body(ball_id).unwrap().position;
    assert!(
        (p - Vec3::new(2.005_47, 0.250_88, 4.001_09)).length() <= 1e-3,
        "{p:?}"
    );
}
