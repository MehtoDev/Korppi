//! World queries are Box2D v3.2's (P2D-23).
//!
//! Rays and shapes are cast along a translation, and hits report the fraction of it (Box2D
//! `b2World_CastRay`, `b2World_CastRayClosest`, `b2World_CastShape`); a callback decides per hit
//! whether to skip it, stop, clip the cast or go on. Overlap queries report colliders whose fat
//! boxes, or whose shapes, overlap (`b2World_OverlapAABB`, `b2World_OverlapShape`) without
//! allocating. Character movers cast and collide a capsule (`b2World_CastMover`,
//! `b2World_CollideMover`) and resolve the resulting planes (`b2SolvePlanes`, `b2ClipVector`).
//! Explosions, wind and per-collider closest points, point tests and ray casts complete the set.
//! Before FIX-2.6 there was only a closest-hit ray cast whose `fraction` was a distance, and an
//! AABB query that tested tight boxes into a `Vec`. Expected values are Box2D's, from
//! `docs/audits/probes/box2d_reference/fix_2_6_queries.c`.

use korppi_math::{Aabb2d, Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::chain::ChainDef2d;
use korppi_physics2d::collider::{Collider2d, Filter2d};
use korppi_physics2d::distance::ShapeProxy2d;
use korppi_physics2d::mover::{clip_vector, solve_planes, CollisionPlane2d, Plane2d};
use korppi_physics2d::shape::{Capsule2d, Circle2d, Polygon2d, Segment2d, Shape2d};
use korppi_physics2d::world::{
    ColliderId2d, ExplosionDef2d, PhysicsWorld2d, QueryFilter2d, RayHit2d, WorldConfig2d,
};

const NAMES: [&str; 7] = [
    "ground", "circle", "capsule", "box", "segment", "chain0", "chain1",
];

/// The probe's scene, without gravity, and its colliders in [`NAMES`] order.
fn scene() -> (PhysicsWorld2d, [ColliderId2d; 7]) {
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::ZERO)
        .with_config(WorldConfig2d {
            enable_sleep: false,
            ..WorldConfig2d::default()
        });
    let mut body = |kind, x: f32, y: f32, angle: f32| {
        world.create_body(
            RigidBody2d::new(kind)
                .with_position(Vec2::new(x, y))
                .with_rotation(Rot2::from_angle(angle)),
        )
    };
    let ground = body(BodyType::Static, 0.0, 0.0, 0.0);
    let circle = body(BodyType::Static, 3.0, 2.0, 0.0);
    let capsule = body(BodyType::Static, -3.0, 2.0, 0.3);
    let rounded = body(BodyType::Dynamic, 0.0, 3.0, 0.2);
    let segment = body(BodyType::Static, 0.0, 0.0, 0.0);
    let chain = body(BodyType::Static, 0.0, 0.0, 0.0);
    let ground = world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::offset_box(
            5.0,
            0.5,
            Vec2::new(0.0, -0.5),
            Rot2::IDENTITY,
        ))),
    );
    let circle = world.create_collider(
        circle,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.5))).with_filter(Filter2d {
            category_bits: 2,
            ..Filter2d::DEFAULT
        }),
    );
    let capsule = world.create_collider(
        capsule,
        Collider2d::new(Shape2d::Capsule(Capsule2d::new(
            Vec2::new(-0.5, 0.0),
            Vec2::new(0.5, 0.0),
            0.25,
        ))),
    );
    let rounded = world.create_collider(
        rounded,
        Collider2d::new(Shape2d::Polygon(Polygon2d::rounded_box(0.5, 0.5, 0.1))),
    );
    let segment = world.create_collider(
        segment,
        Collider2d::new(Shape2d::Segment(Segment2d::new(
            Vec2::new(-2.0, 5.0),
            Vec2::new(2.0, 5.0),
        ))),
    );
    let points = vec![
        Vec2::new(2.0, 7.0),
        Vec2::new(0.0, 7.0),
        Vec2::new(-2.0, 7.0),
    ];
    let chain = world.create_chain(
        chain,
        &ChainDef2d::open(points, Vec2::new(3.0, 7.0), Vec2::new(-3.0, 7.0)),
    );
    let chain_segments = world.chain(chain).unwrap().segments().to_vec();
    (
        world,
        [
            ground,
            circle,
            capsule,
            rounded,
            segment,
            chain_segments[0],
            chain_segments[1],
        ],
    )
}

fn name(ids: &[ColliderId2d; 7], id: ColliderId2d) -> &'static str {
    NAMES[ids.iter().position(|&c| c == id).expect("a scene collider")]
}

fn near(a: Vec2, b: (f32, f32), tolerance: f32) -> bool {
    (a.x - b.0).abs() <= tolerance && (a.y - b.1).abs() <= tolerance
}

/// Asserts a hit against Box2D's (collider, fraction, point, normal).
fn assert_hit(
    ids: &[ColliderId2d; 7],
    hit: &RayHit2d,
    box2d: (&str, f32, (f32, f32), (f32, f32)),
    what: &str,
) {
    let (collider, fraction, point, normal) = box2d;
    assert_eq!(name(ids, hit.collider_id), collider, "{what}: collider");
    assert!(
        (hit.fraction - fraction).abs() < 2e-6
            && near(hit.point, point, 2e-5)
            && near(hit.normal, normal, 2e-5),
        "{what}: {hit:?}, Box2D {box2d:?}"
    );
}

/// Every hit of a cast whose callback returns `answer` (or the hit's fraction, to clip), sorted
/// by fraction and then by scene order as the probe prints them.
fn hits(
    ids: &[ColliderId2d; 7],
    cast: impl FnOnce(&mut dyn FnMut(&RayHit2d) -> f32),
) -> Vec<RayHit2d> {
    let mut hits = Vec::new();
    cast(&mut |hit: &RayHit2d| {
        hits.push(*hit);
        hit.fraction
    });
    hits.sort_by(|a, b| {
        a.fraction
            .total_cmp(&b.fraction)
            .then(name(ids, a.collider_id).cmp(name(ids, b.collider_id)))
    });
    hits
}

#[test]
fn rays_report_fractions_of_their_translation_like_box2d() {
    let (world, ids) = scene();
    let all = QueryFilter2d::default();
    let closest = |origin: (f32, f32), translation: (f32, f32), filter| {
        world.cast_ray_closest(
            Vec2::new(origin.0, origin.1),
            Vec2::new(translation.0, translation.1),
            filter,
        )
    };
    // Box2D v3.2 `b2World_CastRayClosest`.
    for (what, origin, translation, box2d) in [
        (
            "down onto the chain",
            (0.0, 10.0),
            (0.0, -12.0),
            ("chain1", 0.25, (0.0, 7.0), (0.0, 1.0)),
        ),
        (
            "down onto the segment",
            (0.0, 6.0),
            (0.0, -8.0),
            ("segment", 0.125, (0.0, 5.0), (0.0, 1.0)),
        ),
        (
            "across to the capsule",
            (-6.0, 2.1),
            (12.0, 0.0),
            (
                "capsule",
                0.206_442,
                (-3.522_69, 2.1),
                (-0.295_52, 0.955_34),
            ),
        ),
        // A ray starting inside the box skips it.
        (
            "from inside the box",
            (0.0, 3.0),
            (0.0, -4.0),
            ("ground", 0.75, (0.0, 0.0), (0.0, 1.0)),
        ),
        (
            "to the circle",
            (6.0, 2.0),
            (-4.0, 0.0),
            ("circle", 0.625, (3.5, 2.0), (1.0, 0.0)),
        ),
    ] {
        let hit = closest(origin, translation, all).unwrap_or_else(|| panic!("{what}: a hit"));
        assert_hit(&ids, &hit, box2d, what);
    }
    // The chain is one-sided, and a filter can leave the circle out.
    assert!(closest((1.0, 6.0), (0.0, 2.0), all).is_none());
    let no_circle = QueryFilter2d::new(QueryFilter2d::default().category_bits, !2);
    assert!(closest((6.0, 2.0), (-4.0, 0.0), no_circle).is_none());

    // `b2World_CastRay` returning 1 sees every hit, returning 0 stops at the first, returning
    // -1 skips every hit, without shortening the cast, and sees both chain segments at their
    // shared vertex.
    let every = hits(&ids, |record| {
        world.cast_ray(Vec2::new(-6.0, 2.0), Vec2::new(12.0, 0.0), all, |hit| {
            record(hit);
            1.0
        });
    });
    assert_eq!(every.len(), 2);
    assert_hit(
        &ids,
        &every[0],
        (
            "capsule",
            0.193_389,
            (-3.679_33, 2.0),
            (-0.806_64, 0.591_04),
        ),
        "every hit",
    );
    assert_hit(
        &ids,
        &every[1],
        ("circle", 0.708_333, (2.5, 2.0), (-1.0, 0.0)),
        "every hit",
    );
    let mut count = 0;
    world.cast_ray(Vec2::new(-6.0, 2.0), Vec2::new(12.0, 0.0), all, |_| {
        count += 1;
        0.0
    });
    assert_eq!(count, 1);
    let skipped = hits(&ids, |record| {
        world.cast_ray(Vec2::new(0.0, 10.0), Vec2::new(0.0, -12.0), all, |hit| {
            record(hit);
            -1.0
        });
    });
    let names: Vec<_> = skipped.iter().map(|h| name(&ids, h.collider_id)).collect();
    assert_eq!(names, ["chain0", "chain1", "segment", "box", "ground"]);
    assert_hit(
        &ids,
        &skipped[2],
        ("segment", 0.416_667, (0.0, 5.0), (0.0, 1.0)),
        "skip",
    );
    assert_hit(
        &ids,
        &skipped[3],
        ("box", 0.532_741, (-0.000_99, 3.612), (-0.198_67, 0.980_07)),
        "skip",
    );
    assert_hit(
        &ids,
        &skipped[4],
        ("ground", 0.833_333, (0.0, 0.0), (0.0, 1.0)),
        "skip",
    );
}

#[test]
fn shapes_cast_along_a_translation_like_box2d() {
    let (world, ids) = scene();
    let all = QueryFilter2d::default();
    // A ball of radius 0.25 cast across, clipping at each hit: Box2D v3.2 reports the capsule,
    // then the circle (seen before the cast was clipped).
    let ball = ShapeProxy2d::new(&[Vec2::new(-6.0, 2.0)], 0.25);
    let cast = hits(&ids, |record| {
        world.cast_shape(&ball, Vec2::new(12.0, 0.0), all, |hit| {
            record(hit);
            hit.fraction
        });
    });
    assert_eq!(cast.len(), 2);
    assert_hit(
        &ids,
        &cast[0],
        (
            "capsule",
            0.170_823,
            (-3.716_27, 1.926_86),
            (-0.954_41, 0.298_49),
        ),
        "ball cast",
    );
    assert_hit(
        &ids,
        &cast[1],
        ("circle", 0.687_917, (2.5, 2.0), (-1.0, 0.0)),
        "ball cast",
    );

    // A box starting inside the circle: an initial overlap at fraction 0 without a normal,
    // then the ground.
    let corners = [(2.6, 1.6), (3.4, 1.6), (3.4, 2.4), (2.6, 2.4)].map(|(x, y)| Vec2::new(x, y));
    let inside = ShapeProxy2d::new(&corners, 0.0);
    let cast = hits(&ids, |record| {
        world.cast_shape(&inside, Vec2::new(0.0, -5.0), all, |hit| {
            record(hit);
            1.0
        });
    });
    assert_eq!(cast.len(), 2);
    assert_hit(
        &ids,
        &cast[0],
        ("circle", 0.0, (3.0, 2.0), (0.0, 0.0)),
        "overlap",
    );
    assert_hit(
        &ids,
        &cast[1],
        ("ground", 0.319, (2.777_78, 0.0), (0.0, 1.0)),
        "overlap",
    );
}

/// An overlap query's callback.
type OverlapCallback<'a> = dyn FnMut(ColliderId2d) -> bool + 'a;

#[test]
fn overlaps_report_fat_boxes_or_shapes_like_box2d() {
    let (world, ids) = scene();
    let all = QueryFilter2d::default();
    let collect = |query: &mut dyn FnMut(&mut OverlapCallback)| {
        let mut found = Vec::new();
        query(&mut |id| {
            found.push(name(&ids, id));
            true
        });
        found.sort_by_key(|n| NAMES.iter().position(|m| m == n));
        found
    };
    // Box2D v3.2: the box's fat box stays clear of y = 2.2.
    let area = Aabb2d::new(Vec2::new(-4.0, 1.5), Vec2::new(4.0, 2.2));
    assert_eq!(
        collect(&mut |f| world.overlap_aabb(&area, all, f)),
        ["circle", "capsule"]
    );
    let probe = ShapeProxy2d::new(&[Vec2::new(2.3, 2.0)], 0.25);
    assert_eq!(
        collect(&mut |f| world.overlap_shape(&probe, all, f)),
        ["circle"]
    );
    let beside = ShapeProxy2d::new(&[Vec2::new(2.1, 2.0)], 0.25);
    assert!(collect(&mut |f| world.overlap_shape(&beside, all, f)).is_empty());
}

#[test]
fn character_movers_cast_collide_and_solve_planes_like_box2d() {
    let (world, ids) = scene();
    let all = QueryFilter2d::default();
    // Box2D v3.2: a capsule mover 0.7 m above the ground, cast 2 m down, stops at 0.3525; one
    // sunk 0.1 m into the ground gets a plane of normal (0, 1) and offset 0.1.
    let mover = Capsule2d::new(Vec2::new(-1.0, 1.0), Vec2::new(-1.0, 1.5), 0.3);
    let fraction = world.cast_mover(&mover, Vec2::new(0.0, -2.0), all);
    assert!((fraction - 0.3525).abs() < 1e-6, "cast fraction {fraction}");
    let sunk = Capsule2d::new(Vec2::new(-1.0, 0.2), Vec2::new(-1.0, 0.7), 0.3);
    let mut planes = Vec::new();
    world.collide_mover(&sunk, all, |id, result| {
        planes.push((name(&ids, id), *result));
        true
    });
    assert_eq!(planes.len(), 1);
    let (collider, result) = planes[0];
    assert_eq!(collider, "ground");
    assert!(near(result.plane.normal, (0.0, 1.0), 1e-6), "{result:?}");
    assert!((result.plane.offset - 0.1).abs() < 1e-6, "{result:?}");
    assert!(near(result.point, (-1.0, 0.0), 1e-5), "{result:?}");

    // `b2SolvePlanes` and `b2ClipVector`: a rigid floor and a soft wall limited to 2 cm.
    let mut planes = [
        CollisionPlane2d::new(Plane2d::new(Vec2::Y, 0.1), f32::MAX, true),
        CollisionPlane2d::new(Plane2d::new(Vec2::new(-1.0, 0.0), 0.05), 0.02, false),
    ];
    let solved = solve_planes(Vec2::new(1.0, -0.5), &mut planes);
    assert!(near(solved.delta, (0.98, 0.095), 1e-6), "{solved:?}");
    assert_eq!(solved.iteration_count, 1);
    assert!((planes[0].push - 0.595).abs() < 1e-6 && (planes[1].push - 0.02).abs() < 1e-6);
    assert!(near(
        clip_vector(Vec2::new(1.0, -1.0), &planes),
        (1.0, 0.0),
        1e-6
    ));
}

#[test]
fn explosions_and_wind_push_bodies_like_box2d() {
    // Four bodies around an explosion of radius 1.5 and falloff 2 at the origin, 5 N s per
    // metre of facing perimeter: Box2D v3.2's velocities, the farthest body untouched.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let shapes = [
        Shape2d::Polygon(Polygon2d::from_box(0.5, 0.25)),
        Shape2d::Circle(Circle2d::new(Vec2::ZERO, 0.4)),
        Shape2d::Capsule(Capsule2d::new(
            Vec2::new(-0.4, 0.0),
            Vec2::new(0.4, 0.0),
            0.2,
        )),
        Shape2d::Polygon(Polygon2d::from_box(0.5, 0.25)),
    ];
    let places = [(1.5, 0.0), (-2.5, 0.5), (0.0, 3.2), (6.0, 0.0)];
    let bodies: Vec<_> = shapes
        .into_iter()
        .zip(places)
        .enumerate()
        .map(|(i, (shape, (x, y)))| {
            let id = world.create_body(
                RigidBody2d::new(BodyType::Dynamic)
                    .with_position(Vec2::new(x, y))
                    .with_rotation(Rot2::from_angle(0.4 * i as f32)),
            );
            world.create_collider(id, Collider2d::new(shape));
            id
        })
        .collect();
    world.explode(&ExplosionDef2d {
        position: Vec2::ZERO,
        radius: 1.5,
        falloff: 2.0,
        impulse_per_length: 5.0,
        ..ExplosionDef2d::default()
    });
    for (body, (v, w)) in bodies.iter().zip([
        ((5.0, 0.0), 0.0),
        ((-5.269_08, 1.053_82), 0.0),
        ((-0.417_21, 4.361_08), -11.336_21),
        ((0.0, 0.0), 0.0),
    ]) {
        let body = world.body(*body).unwrap();
        assert!(
            near(body.linear_velocity, v, 1e-4) && (body.angular_velocity - w).abs() < 1e-4,
            "{:?} {}, Box2D {v:?} {w}",
            body.linear_velocity,
            body.angular_velocity
        );
    }

    // Wind of (10, 2) m/s with drag 1 and lift 0.5 on a circle, a capsule and a moving offset
    // box, for one step: Box2D v3.2's velocities.
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::ZERO)
        .with_config(WorldConfig2d {
            enable_sleep: false,
            ..WorldConfig2d::default()
        });
    let shapes = [
        Shape2d::Circle(Circle2d::new(Vec2::new(0.1, 0.0), 0.5)),
        Shape2d::Capsule(Capsule2d::new(
            Vec2::new(-0.5, 0.0),
            Vec2::new(0.5, 0.1),
            0.2,
        )),
        Shape2d::Polygon(Polygon2d::offset_box(
            1.0,
            0.1,
            Vec2::new(0.2, 0.0),
            Rot2::from_angle(0.1),
        )),
    ];
    let mut bodies = Vec::new();
    for (i, shape) in shapes.into_iter().enumerate() {
        let id = world.create_body(
            RigidBody2d::new(BodyType::Dynamic)
                .with_position(Vec2::new(0.0, 3.0 * i as f32))
                .with_rotation(Rot2::from_angle([0.0, 0.5, 0.3][i])),
        );
        let collider = world.create_collider(id, Collider2d::new(shape));
        if i == 2 {
            // Set once the collider has moved the centre of mass, as in the probe.
            let body = world.body_mut(id).unwrap();
            body.linear_velocity = Vec2::new(1.0, 0.0);
            body.angular_velocity = 0.5;
        }
        bodies.push((id, collider));
    }
    for &(_, collider) in &bodies {
        assert!(world.apply_wind(collider, Vec2::new(10.0, 2.0), 1.0, 0.5, true));
    }
    world.step(1.0 / 60.0);
    for (&(body, _), (v, w)) in bodies.iter().zip([
        ((1.325_506, 0.265_101), 0.0),
        ((1.709_523, -0.407_159), -1.481_525),
        ((2.255_482, -0.075_763), 0.408_192),
    ]) {
        let body = world.body(body).unwrap();
        assert!(
            near(body.linear_velocity, v, 2e-5) && (body.angular_velocity - w).abs() < 2e-5,
            "{:?} {}, Box2D {v:?} {w}",
            body.linear_velocity,
            body.angular_velocity
        );
    }
}

#[test]
fn colliders_answer_closest_points_point_tests_and_ray_casts_like_box2d() {
    let (world, ids) = scene();
    let [_, circle, capsule, rounded, ..] = ids;
    for (collider, target, box2d) in [
        (circle, (6.0, 6.0), (3.3, 2.4)),
        (rounded, (2.0, 5.0), (0.465_90, 3.655_28)),
        // Inside a shape, the closest point is the target itself.
        (capsule, (-3.0, 2.1), (-3.045_65, 2.247_57)),
    ] {
        let p = world
            .collider_closest_point(collider, Vec2::new(target.0, target.1))
            .unwrap();
        assert!(near(p, box2d, 1e-5), "closest point {p:?}, Box2D {box2d:?}");
    }
    for (collider, point) in [
        (rounded, (0.0, 3.0)),
        (rounded, (0.55, 3.0)),
        (rounded, (0.4, 3.6)),
        (circle, (3.4, 2.2)),
        (capsule, (-3.6, 1.8)),
    ] {
        assert_eq!(
            world.collider_test_point(collider, Vec2::new(point.0, point.1)),
            Some(true),
            "{point:?}"
        );
    }
    assert_eq!(
        world.collider_test_point(rounded, Vec2::new(0.7, 3.7)),
        Some(false)
    );
    let cast = world
        .collider_ray_cast(rounded, Vec2::new(2.0, 3.0), Vec2::new(-4.0, 0.0))
        .unwrap();
    assert!(cast.hit);
    assert!((cast.fraction - 0.348_225).abs() < 2e-6, "{cast:?}");
    assert!(near(cast.point, (0.612, 3.000_99), 2e-5), "{cast:?}");
    assert!(near(cast.normal, (0.980_07, 0.198_67), 2e-5), "{cast:?}");
}
