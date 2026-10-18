//! Compound shapes are Box3D's baked compound shapes (b3CompoundData): static-only
//! aggregate shapes containing multiple child shapes with an internal BVH, multi-manifold
//! contacts across child shapes, and front-to-back ray casting (P3D-13, P3D-10).
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_5_compounds.c`):
//! 60 Hz, 4 substeps, no sleep, density 1000 kg/m³, static compounds.

use korppi_math::{Quat, Vec3};
use korppi_physics::{
    BodyType, Collider, Compound3d, CompoundChild, Hull3d, PhysicsWorld, RigidBody, Shape,
};

const DT: f32 = 1.0 / 60.0;

#[test]
fn compound_creation_and_bounds() {
    let capsule = CompoundChild::new(
        Shape::Capsule {
            radius: 0.25,
            half_height: 1.0,
        },
        Vec3::ZERO,
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
    );
    let box_hull = CompoundChild::new(
        Shape::ConvexHull(Hull3d::box_hull(0.5, 0.5, 0.5)),
        Vec3::ZERO,
        Quat::IDENTITY,
    );
    let sphere1 = CompoundChild::new(
        Shape::Sphere { radius: 0.5 },
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let sphere2 = CompoundChild::new(
        Shape::Sphere { radius: 0.5 },
        Vec3::new(-5.0, 0.0, 0.0),
        Quat::IDENTITY,
    );

    let compound =
        Compound3d::new(vec![capsule, box_hull, sphere1, sphere2]).expect("compound builds");

    assert_eq!(compound.child_count(), 4);
    let (min, max) = compound.bounds();
    assert!(
        (min - Vec3::new(-5.5, -0.5, -0.5)).length() <= 1e-4,
        "min: {min:?}"
    );
    assert!(
        (max - Vec3::new(5.5, 0.5, 0.5)).length() <= 1e-4,
        "max: {max:?}"
    );
}

#[test]
fn compound_rays() {
    // 1. Two spheres along +X
    let s1 = CompoundChild::new(
        Shape::Sphere { radius: 1.0 },
        Vec3::new(5.0, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let s2 = CompoundChild::new(
        Shape::Sphere { radius: 1.0 },
        Vec3::new(10.0, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let comp_spheres = Compound3d::new(vec![s1, s2]).expect("compound builds");

    let hit = comp_spheres
        .ray_cast(Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0), 1.0)
        .expect("ray hits first sphere");
    assert!(
        (hit.fraction - 0.2).abs() <= 1e-4,
        "fraction: {}",
        hit.fraction
    );
    assert!(
        (hit.normal - Vec3::new(-1.0, 0.0, 0.0)).length() <= 1e-4,
        "normal: {:?}",
        hit.normal
    );
    assert_eq!(hit.child_index, 0);

    // 2. Rotated box hull: 90 degrees about Z placed at x=5
    let rotated_box = CompoundChild::new(
        Shape::ConvexHull(Hull3d::box_hull(1.0, 1.0, 1.0)),
        Vec3::new(5.0, 0.0, 0.0),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
    );
    let comp_box = Compound3d::new(vec![rotated_box]).expect("compound builds");

    let hit_box = comp_box
        .ray_cast(Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0), 1.0)
        .expect("ray hits rotated box");
    assert!(
        (hit_box.fraction - 0.2).abs() <= 1e-4,
        "fraction: {}",
        hit_box.fraction
    );
    assert!(
        (hit_box.normal - Vec3::new(-1.0, 0.0, 0.0)).length() <= 1e-3,
        "normal: {:?}",
        hit_box.normal
    );
    assert_eq!(hit_box.child_index, 0);
}

#[test]
fn shapes_rest_on_compound() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Ground compound made of 2 boxes:
    // Box 1 at (-2, 0, 0) half extents (1.5, 0.5, 2.0) -> top at y=0.5
    // Box 2 at (2, 0, 0) half extents (1.5, 0.5, 2.0) -> top at y=0.5
    let b1 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.5, 0.5, 2.0),
        },
        Vec3::new(-2.0, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let b2 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.5, 0.5, 2.0),
        },
        Vec3::new(2.0, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let compound = Compound3d::new(vec![b1, b2]).expect("compound builds");

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::Compound(compound)).with_friction(0.6),
    );

    // Drop cube on child 0 (-2, 0, 0)
    let cube = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(-2.0, 1.5, 0.0)).with_enable_sleep(false),
    );
    world.create_collider(
        cube,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_friction(0.6),
    );

    // Drop ball on child 1 (2, 0, 0)
    let ball = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 1.2, 0.0)).with_enable_sleep(false),
    );
    world.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.2 }).with_friction(0.6),
    );

    for _ in 0..120 {
        world.step(DT);
    }

    let y_cube = world.body(cube).unwrap().position.y;
    let y_ball = world.body(ball).unwrap().position.y;

    assert!((y_cube - 0.999_93).abs() <= 1e-3, "cube y: {y_cube}");
    assert!((y_ball - 0.699_93).abs() <= 1e-3, "ball y: {y_ball}");
}

#[test]
fn multi_manifold_spanning_compound() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Ground compound made of 2 boxes separated by a tiny gap of 0.1:
    // Box 1 at (-1.05, 0, 0) half extents (1.0, 0.5, 1.0)
    // Box 2 at (1.05, 0, 0) half extents (1.0, 0.5, 1.0)
    let b1 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        },
        Vec3::new(-1.05, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let b2 = CompoundChild::new(
        Shape::Box {
            half_extents: Vec3::new(1.0, 0.5, 1.0),
        },
        Vec3::new(1.05, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let compound = Compound3d::new(vec![b1, b2]).expect("compound builds");

    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
    world.create_collider(
        ground,
        Collider::new(Shape::Compound(compound)).with_friction(0.6),
    );

    // Wide dynamic plank spanning both children: half extents (1.2, 0.2, 0.5)
    let plank = world.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 1.2, 0.0)).with_enable_sleep(false),
    );
    world.create_collider(
        plank,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(1.2, 0.2, 0.5),
        })
        .with_friction(0.6),
    );

    for _ in 0..120 {
        world.step(DT);
    }

    let y_plank = world.body(plank).unwrap().position.y;
    assert!(
        (y_plank - 0.699_96).abs() <= 1e-3,
        "plank resting y: {y_plank}"
    );

    let contacts: Vec<_> = world
        .contacts()
        .filter(|c| (c.body_a == plank || c.body_b == plank) && c.is_touching)
        .collect();

    let total_manifolds: usize = contacts.iter().map(|c| c.manifolds().len()).sum();
    let total_points: usize = contacts
        .iter()
        .flat_map(|c| c.manifolds())
        .map(|m| m.points.len())
        .sum();

    assert_eq!(
        total_manifolds, 2,
        "spanning plank touches both child boxes (2 manifolds)"
    );
    assert_eq!(total_points, 8, "4 points on each child box");
}
