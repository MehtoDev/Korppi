//! Convex hulls are built from point clouds with Box3D's quickhull, with exact mass and Box3D's
//! hull collisions (P3D-11).
//!
//! `b3CreateHull` builds a half-edge hull from a point cloud, merging coplanar faces and
//! dropping interior points, and computes its volume, centroid, inertia tensor and inner radius
//! exactly (`hull.c`). Hulls collide like boxes, by the separating axis test and face clipping,
//! with spheres and capsules by GJK, and rays clip against their planes. Before FIX-3.5 a hull
//! was a bare vertex list: its mass was half its bounding box's, every pair went through GJK and
//! EPA for one contact point, and a ray tested every triple of vertices.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_5_hulls.c`):
//! 60 Hz, 4 substeps, no sleep, density 1000 kg/m³, friction 0.6, a static ground box with its
//! top at y = 0.

use korppi_math::{Mat3, Quat, Ray, Vec3};
use korppi_physics::{BodyType, Collider, Hull3d, PhysicsWorld, RigidBody, Shape};

const DT: f32 = 1.0 / 60.0;

/// A cube of side 1 from its corners, its face centres and its centre.
fn cube_cloud() -> Vec<Vec3> {
    let mut points = Vec::new();
    for z in [-0.5, 0.5] {
        for (x, y) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            points.push(Vec3::new(x, y, z));
        }
    }
    points.extend([
        Vec3::ZERO,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(0.0, 0.0, 0.5),
        Vec3::new(0.0, 0.0, -0.5),
    ]);
    points
}

/// A wedge: a 1 x 0.4 x 0.8 box whose top slopes down to one edge, off the origin, with an
/// interior point.
fn wedge_cloud() -> Vec<Vec3> {
    vec![
        Vec3::new(-0.4, 0.0, -0.3),
        Vec3::new(0.6, 0.0, -0.3),
        Vec3::new(0.6, 0.0, 0.5),
        Vec3::new(-0.4, 0.0, 0.5),
        Vec3::new(-0.4, 0.4, -0.3),
        Vec3::new(-0.4, 0.4, 0.5),
        Vec3::new(0.1, 0.2, 0.1),
    ]
}

/// An irregular rock: 20 points on a squashed, wavy sphere.
fn rock_cloud() -> Vec<Vec3> {
    let count = 20;
    (0..count)
        .map(|i| {
            let t = i as f32 / count as f32;
            let phi = (1.0 - 2.0 * (i as f32 + 0.5) / count as f32).acos();
            let theta = 2.399_963_2 * i as f32;
            let r = 0.5 + 0.1 * (7.0 * t).sin();
            Vec3::new(
                r * phi.sin() * theta.cos(),
                0.7 * r * phi.cos(),
                r * phi.sin() * theta.sin(),
            )
        })
        .collect()
}

/// A hull and Box3D's numbers for it: name, points, vertex limit, vertex, face and half-edge
/// counts, volume, inner radius, centre of mass and inertia at 1000 kg/m³.
struct HullCase(
    &'static str,
    Vec<Vec3>,
    usize,
    (usize, usize, usize),
    f32,
    f32,
    Vec3,
    [[f32; 3]; 3],
);

#[test]
fn point_clouds_make_box3d_hulls() {
    // Box3D: vertex, face and half-edge counts, volume and inner radius; mass, centre and
    // inertia at 1000 kg/m³.
    #[rustfmt::skip]
    let cases: [HullCase; 4] = [
        HullCase("cube", cube_cloud(), 128, (8, 6, 24), 1.0, 0.5, Vec3::ZERO,
            [[166.666_7, 0.0, 0.0], [0.0, 166.666_7, 0.0], [0.0, 0.0, 166.666_7]]),
        HullCase("wedge", wedge_cloud(), 128, (6, 5, 18), 0.16, 0.123_797, Vec3::new(-0.066_667, 0.133_333, 0.1),
            [[9.955_56, 1.777_78, 0.0], [1.777_78, 17.422_23, 0.0], [0.0, 0.0, 10.311_11]]),
        HullCase("rock", rock_cloud(), 128, (20, 36, 108), 0.275_881, 0.277_49, Vec3::new(0.002_139, 0.041_666, 0.001_889),
            [[17.441_44, 0.073_1, 0.072_32], [0.073_1, 23.215_52, 0.094_93], [0.072_32, 0.094_93, 17.417_69]]),
        HullCase("rock of 8", rock_cloud(), 8, (8, 12, 36), 0.171_439, 0.175_906, Vec3::new(-0.000_656, 0.011_387, 0.002_652),
            [[8.306_36, 0.721_67, 0.732_9], [0.721_67, 12.958_91, 0.351_47], [0.732_9, 0.351_47, 9.259_88]]),
    ];
    for HullCase(what, points, max_vertices, counts, volume, inner_radius, center, inertia) in cases
    {
        let hull = Hull3d::with_max_vertices(&points, max_vertices)
            .unwrap_or_else(|| panic!("{what}: Box3D builds a hull"));
        assert_eq!(
            (
                hull.vertex_count(),
                hull.face_count(),
                hull.half_edge_count()
            ),
            counts,
            "{what}: Box3D's vertex, face and half-edge counts"
        );
        assert!(
            (hull.volume() - volume).abs() <= 1e-5,
            "{what}: volume {}",
            hull.volume()
        );
        assert!(
            (hull.inner_radius() - inner_radius).abs() <= 1e-5,
            "{what}: inner radius {}",
            hull.inner_radius()
        );
        let mass = Collider::new(Shape::ConvexHull(hull)).mass_data();
        assert!(
            (mass.mass - 1000.0 * volume).abs() <= 1e-2,
            "{what}: mass {}",
            mass.mass
        );
        assert!(
            (mass.center - center).length() <= 1e-5,
            "{what}: centre {:?}",
            mass.center
        );
        let expected = Mat3::from_cols_array_2d(&inertia).transpose();
        assert!(
            mass.inertia.abs_diff_eq(expected, 1e-3),
            "{what}: Box3D's inertia is {expected:?}, Korppi's {:?}",
            mass.inertia
        );
    }
    // Box3D: four coplanar points make no hull.
    let flat = [Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::new(1.0, 0.0, 1.0)];
    assert!(Hull3d::new(&flat).is_none());
}

fn world_with_ground() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    let ground = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, -0.5, 0.0)));
    world.create_collider(
        ground,
        Collider::new(Shape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        })
        .with_friction(0.6),
    );
    world
}

fn hull_body(world: &mut PhysicsWorld, body: RigidBody, points: &[Vec3]) -> u32 {
    let id = world.create_body(body.with_enable_sleep(false));
    world.create_collider(
        id,
        Collider::new(Shape::convex_hull(points).unwrap()).with_friction(0.6),
    );
    id.into()
}

fn point_count(world: &PhysicsWorld, id: impl Into<korppi_physics::BodyId3d>) -> usize {
    let id = id.into().index();
    world
        .manifolds()
        .iter()
        .filter(|m| m.body_a == id || m.body_b == id)
        .map(|m| m.points.len())
        .sum()
}

#[test]
fn asymmetric_hulls_rest_without_creeping() {
    // Box3D: the wedge dropped 0.3 m rests on its bottom face on 4 points, its origin at
    // (-0.00199, -0.00012, -0.00004), and does not creep or tilt over 4 s.
    let mut w = world_with_ground();
    let id = hull_body(
        &mut w,
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.3, 0.0)),
        &wedge_cloud(),
    );
    for _ in 0..60 {
        w.step(DT);
    }
    let settled = w.body(id).unwrap().position;
    for _ in 0..240 {
        w.step(DT);
    }
    let b = w.body(id).unwrap();
    assert!(
        (b.position - settled).length() <= 1e-3,
        "the wedge should not creep, it moved {} m",
        (b.position - settled).length()
    );
    let tilt = 2.0 * Vec3::new(b.rotation.x, 0.0, b.rotation.z).length();
    assert!(tilt <= 1e-3, "{tilt}");
    assert!((-0.006..=0.001).contains(&b.position.y), "{}", b.position.y);
    assert_eq!(point_count(&w, id), 4, "Box3D rests the wedge on 4 points");
}

#[test]
fn hull_stacks_rest() {
    // Box3D: two cube hulls stacked, the upper one turned 0.3 rad, rest after 2 s at 0.49986 and
    // 1.49939, 4 points between them.
    let mut w = world_with_ground();
    let lower = hull_body(
        &mut w,
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.0, 0.5, 0.0)),
        &cube_cloud(),
    );
    let upper = hull_body(
        &mut w,
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.1, 1.5, 0.0))
            .with_rotation(Quat::from_rotation_y(0.3)),
        &cube_cloud(),
    );
    for _ in 0..120 {
        w.step(DT);
    }
    for (id, height) in [(lower, 0.5), (upper, 1.5)] {
        let b = w.body(id).unwrap();
        assert!(
            b.linear_velocity.length() <= 0.01,
            "{:?}",
            b.linear_velocity
        );
        assert!(
            (height - 0.012..=height + 0.001).contains(&b.position.y),
            "{}",
            b.position.y
        );
    }
    assert_eq!(point_count(&w, upper), 4);
}

#[test]
fn balls_and_capsules_rest_on_hulls() {
    // Box3D: a 0.2 m ball and a lying capsule (radius 0.1) dropped onto static cube hulls rest
    // on one and two points, at 1.19993 and 1.09990.
    let mut w = world_with_ground();
    for x in [0.0, 3.0] {
        hull_body(
            &mut w,
            RigidBody::new(BodyType::Static, Vec3::new(x, 0.5, 0.0)),
            &cube_cloud(),
        );
    }
    let ball = w.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(0.1, 2.0, 0.0)).with_enable_sleep(false),
    );
    w.create_collider(
        ball,
        Collider::new(Shape::Sphere { radius: 0.2 }).with_friction(0.6),
    );
    let capsule = w.create_body(
        RigidBody::new(BodyType::Dynamic, Vec3::new(3.0, 2.0, 0.0))
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2))
            .with_enable_sleep(false),
    );
    w.create_collider(
        capsule,
        Collider::new(Shape::Capsule {
            radius: 0.1,
            half_height: 0.3,
        })
        .with_friction(0.6),
    );
    for _ in 0..120 {
        w.step(DT);
    }
    let y = w.body(ball).unwrap().position.y;
    assert!(
        (1.194..=1.201).contains(&y),
        "the ball rests at {y}, Box3D at 1.19993"
    );
    let y = w.body(capsule).unwrap().position.y;
    assert!(
        (1.094..=1.101).contains(&y),
        "the capsule rests at {y}, Box3D at 1.09990"
    );
    assert_eq!(point_count(&w, ball), 1);
    assert_eq!(point_count(&w, capsule), 2);
}

#[test]
fn rays_hit_hull_faces() {
    // Box3D: a ray down onto the wedge's slope hits at (0, 0.24, 0.1) with normal
    // (0.37139, 0.92848, 0); one along x hits its upright face at x = -0.4; one above misses.
    let wedge = Shape::convex_hull(&wedge_cloud()).unwrap();
    let down = Ray::new(Vec3::new(0.0, 2.0, 0.1), Vec3::NEG_Y);
    let (distance, normal) = wedge.ray_cast(&down, 4.0).expect("the ray hits the slope");
    assert!((distance - 1.76).abs() <= 1e-5, "{distance}");
    assert!(
        (normal - Vec3::new(0.371_39, 0.928_48, 0.0)).length() <= 1e-4,
        "{normal:?}"
    );
    let side = Ray::new(Vec3::new(-2.0, 0.1, 0.0), Vec3::X);
    let (distance, normal) = wedge.ray_cast(&side, 4.0).expect("the ray hits the face");
    assert!((distance - 1.6).abs() <= 1e-5, "{distance}");
    assert!((normal - Vec3::NEG_X).length() <= 1e-6);
    let high = Ray::new(Vec3::new(-2.0, 1.0, 0.0), Vec3::X);
    assert!(wedge.ray_cast(&high, 4.0).is_none());
}
