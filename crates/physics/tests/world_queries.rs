//! Integration tests for 3D world queries and character mover (P3D-21).
//!
//! Validates ray casts (closest, collector, filtering), shape casts (conservative advancement),
//! AABB and shape overlaps, and character mover (cast_mover, collide_mover, solve_planes, clip_vector)
//! against Erin Catto's Box3D reference probe `fix_3_6_queries.c`.

use korppi_math::{Aabb, Vec3};
use korppi_physics::body::{BodyType, RigidBody};
use korppi_physics::collider::{Collider, Filter3d, QueryFilter3d};
use korppi_physics::distance::ShapeProxy;
use korppi_physics::math::Transform;
use korppi_physics::mover::{
    clip_vector, solve_planes, CapsuleMover, CollisionPlane3d, Plane3d, PlaneResult3d,
};
use korppi_physics::shape::Shape;
use korppi_physics::world::ExplosionDef3d;
use korppi_physics::{ColliderId3d, PhysicsWorld};

fn scene() -> (PhysicsWorld, ColliderId3d, ColliderId3d, ColliderId3d) {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::new(0.0, -9.81, 0.0));

    // Ground box at y = -0.5, half extents (10, 0.5, 10) -> top face at y = 0
    let ground_b = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    let ground = world.create_collider(
        ground_b,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        }),
    );

    // Sphere at (0, 5, 0), radius 1
    let sphere_b = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 5.0, 0.0)));
    let sphere = world.create_collider(
        sphere_b,
        Collider::new(Shape::Sphere { radius: 1.0 }).with_filter(Filter3d::new(2, u64::MAX)),
    );

    // Box at (5, 5, 0), half extents (1, 1, 1)
    let box_b = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(5.0, 5.0, 0.0)));
    let box_col = world.create_collider(
        box_b,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(1.0),
        }),
    );

    (world, ground, sphere, box_col)
}

#[test]
fn solve_planes_and_clip_vector_match_box3d() {
    let mut planes = [
        CollisionPlane3d::new(Plane3d::new(Vec3::Z, 0.5), f32::MAX, true),
        CollisionPlane3d::new(Plane3d::new(Vec3::Z, 1.0), f32::MAX, true),
    ];

    let target = Vec3::ZERO;
    let result = solve_planes(target, &mut planes);

    assert_eq!(result.iteration_count, 2, "iterations");
    assert!(
        (result.delta.z - 0.995).abs() < 0.01,
        "delta z {}",
        result.delta.z
    );

    let velocity = Vec3::new(1.0, 2.0, -3.0);
    let clipped = clip_vector(velocity, &planes);
    assert_eq!(clipped.x, 1.0);
    assert_eq!(clipped.y, 2.0);
    assert_eq!(clipped.z, 0.0);
}

#[test]
fn rays_report_fractions_and_filtering_like_box3d() {
    let (world, ground, sphere, _box_col) = scene();
    let all = QueryFilter3d::DEFAULT;

    // 1. Ray cast closest down towards sphere at (0, 5, 0)
    let hit = world
        .cast_ray_closest(Vec3::new(0.0, 10.0, 0.0), Vec3::new(0.0, -15.0, 0.0), &all)
        .expect("hit sphere");

    assert_eq!(hit.collider_id, sphere);
    assert!(
        (hit.fraction - 0.26667).abs() < 1e-3,
        "fraction {}",
        hit.fraction
    );
    assert!((hit.point.y - 6.0).abs() < 0.02, "point y {}", hit.point.y);
    assert!(
        (hit.normal.y - 1.0).abs() < 1e-3,
        "normal y {}",
        hit.normal.y
    );

    // 2. Ray cast with filter excluding sphere category (bit 2)
    let no_sphere = QueryFilter3d::new(u64::MAX, !2);
    let hit_ground = world
        .cast_ray_closest(
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, -15.0, 0.0),
            &no_sphere,
        )
        .expect("hit ground through sphere");
    assert_eq!(hit_ground.collider_id, ground);
    // Ground top is at y = 0 -> distance is 10 out of 15 -> fraction = 10/15 = 0.66667
    assert!(
        (hit_ground.fraction - 0.66667).abs() < 1e-3,
        "fraction {}",
        hit_ground.fraction
    );

    // 3. Ray cast collector callback
    let mut hits = Vec::new();
    world.cast_ray(
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(0.0, -15.0, 0.0),
        &all,
        |h| {
            hits.push(*h);
            1.0 // continue and see all hits
        },
    );
    assert_eq!(hits.len(), 2, "hits both sphere and ground");
    assert_eq!(hits[0].collider_id, sphere);
    assert_eq!(hits[1].collider_id, ground);

    // 4. Ray cast callback returning 0.0 stops immediately
    let mut stopped_count = 0;
    world.cast_ray(
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(0.0, -15.0, 0.0),
        &all,
        |_| {
            stopped_count += 1;
            0.0 // stop at first hit
        },
    );
    assert_eq!(stopped_count, 1);

    // 5. Ray missing everything
    let miss = world.cast_ray_closest(Vec3::new(50.0, 10.0, 0.0), Vec3::new(0.0, -15.0, 0.0), &all);
    assert!(miss.is_none());
}

#[test]
fn shapes_cast_along_a_translation_like_box3d() {
    let (world, ground, sphere, _box_col) = scene();
    let all = QueryFilter3d::DEFAULT;

    // Cast a sphere of radius 0.5 downwards from (0, 10, 0) by (0, -15, 0)
    let center = Vec3::new(0.0, 10.0, 0.0);
    let proxy = ShapeProxy {
        points: std::slice::from_ref(&center),
        radius: 0.5,
    };

    // Box3D `b3World_CastShape` visits the leaf child (`ground`) before descending into the
    // internal sibling (`sphere + box`), calling back twice while clipping the cast to the
    // nearer sphere.
    let mut hits = Vec::new();
    world.cast_shape(&proxy, Vec3::new(0.0, -15.0, 0.0), &all, |h| {
        hits.push(*h);
        h.fraction
    });
    assert_eq!(hits.len(), 2, "cast_shape visits ground leaf then sphere");
    assert_eq!(hits[0].collider_id, ground);
    assert_eq!(hits[1].collider_id, sphere);
    assert!(
        (hits[1].fraction - 0.23367).abs() < 1e-3,
        "fraction {}",
        hits[1].fraction
    );
    assert!(
        (hits[1].point.y - 6.0).abs() < 0.02,
        "point y {}",
        hits[1].point.y
    );
    assert!(
        (hits[1].normal.y - 1.0).abs() < 1e-3,
        "normal y {}",
        hits[1].normal.y
    );

    let hit = world
        .cast_shape_closest(&proxy, Vec3::new(0.0, -15.0, 0.0), &all)
        .expect("hit sphere");

    assert_eq!(hit.collider_id, sphere);
    assert!(
        (hit.fraction - 0.23367).abs() < 1e-3,
        "fraction {}",
        hit.fraction
    );
    assert!((hit.point.y - 6.0).abs() < 0.02, "point y {}", hit.point.y);
    assert!(
        (hit.normal.y - 1.0).abs() < 1e-3,
        "normal y {}",
        hit.normal.y
    );
}

#[test]
fn overlaps_report_bounds_and_shapes_like_box3d() {
    let (world, _ground, sphere, _box_col) = scene();
    let all = QueryFilter3d::DEFAULT;

    // AABB covering sphere at (0, 5, 0)
    let mut aabb_hits = Vec::new();
    world.overlap_aabb(
        &Aabb {
            min: Vec3::new(-1.5, 4.0, -1.5),
            max: Vec3::new(1.5, 6.0, 1.5),
        },
        &all,
        |id| {
            aabb_hits.push(id);
            true
        },
    );
    assert_eq!(aabb_hits.len(), 1);
    assert_eq!(aabb_hits[0], sphere);

    // Shape overlap: sphere proxy of radius 0.5 at (0, 5.8, 0)
    let center = Vec3::new(0.0, 5.8, 0.0);
    let proxy = ShapeProxy {
        points: std::slice::from_ref(&center),
        radius: 0.5,
    };
    let mut shape_hits = Vec::new();
    world.overlap_shape(&proxy, &Transform::IDENTITY, &all, |id| {
        shape_hits.push(id);
        true
    });
    assert_eq!(shape_hits.len(), 1);
    assert_eq!(shape_hits[0], sphere);

    // Shape far away does not overlap
    let far_center = Vec3::new(0.0, 8.0, 0.0);
    let far_proxy = ShapeProxy {
        points: std::slice::from_ref(&far_center),
        radius: 0.5,
    };
    let mut far_hits = Vec::new();
    world.overlap_shape(&far_proxy, &Transform::IDENTITY, &all, |id| {
        far_hits.push(id);
        true
    });
    assert!(far_hits.is_empty());
}

#[test]
fn character_mover_casts_and_collides_like_box3d() {
    let (world, ground, _sphere, _box_col) = scene();
    let all = QueryFilter3d::DEFAULT;

    // 1. Cast mover capsule downwards
    let mover = CapsuleMover::new(Vec3::new(-2.0, 5.0, 0.0), Vec3::new(-2.0, 6.0, 0.0), 0.5);
    let fraction = world.cast_mover(&mover, Vec3::new(0.0, -10.0, 0.0), &all);
    assert!(
        (fraction - 0.45050).abs() < 1e-3,
        "mover cast fraction {fraction}"
    );

    // 2. Collide mover sunk 0.1 into ground
    let sunk = CapsuleMover::new(Vec3::new(-2.0, 0.4, 0.0), Vec3::new(-2.0, 1.4, 0.0), 0.5);
    let mut planes: Vec<(ColliderId3d, PlaneResult3d)> = Vec::new();
    world.collide_mover(&sunk, &all, |id, result| {
        planes.push((id, *result));
        true
    });
    assert_eq!(planes.len(), 1);
    assert_eq!(planes[0].0, ground);
    assert!(
        (planes[0].1.plane.normal.y - 1.0).abs() < 1e-3,
        "normal {:?}",
        planes[0].1.plane.normal
    );
    assert!(
        (planes[0].1.plane.offset - 0.1).abs() < 1e-3,
        "offset {}",
        planes[0].1.plane.offset
    );
    assert!(
        planes[0].1.point.y.abs() < 0.02,
        "point y {}",
        planes[0].1.point.y
    );
}

#[test]
fn explosion_pushes_dynamic_bodies_radially() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO);

    let b1 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(2.0, 0.0, 0.0)));
    world.create_collider(
        b1,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    let b2 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 4.0, 0.0)));
    world.create_collider(
        b2,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    // Distant body outside radius
    let b3 = world.create_body(RigidBody::new(BodyType::Dynamic, Vec3::new(50.0, 0.0, 0.0)));
    world.create_collider(
        b3,
        Collider::new(Shape::Sphere { radius: 0.5 }).with_mass(1.0),
    );

    world.explode(&ExplosionDef3d {
        mask_bits: u64::MAX,
        position: Vec3::ZERO,
        radius: 2.0,
        falloff: 8.0,
        impulse_per_area: 50.0,
    });

    let v1 = world.body(b1).unwrap().linear_velocity;
    let v2 = world.body(b2).unwrap().linear_velocity;
    let v3 = world.body(b3).unwrap().linear_velocity;

    // b1 is pushed along +X
    assert!(v1.x > 0.0, "v1.x = {}", v1.x);
    assert_eq!(v1.y, 0.0);
    // b2 is pushed along +Y
    assert!(v2.y > 0.0, "v2.y = {}", v2.y);
    assert_eq!(v2.x, 0.0);
    // b1 closer than b2 -> higher velocity
    assert!(v1.x > v2.y, "v1.x {} > v2.y {}", v1.x, v2.y);
    // b3 outside radius -> zero velocity
    assert_eq!(v3, Vec3::ZERO);
}
