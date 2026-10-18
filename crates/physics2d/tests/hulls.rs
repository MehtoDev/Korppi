//! Polygons are built from Box2D v3.2 hulls (P2D-19).
//!
//! Before FIX-2.6 `Polygon2d::from_vertices` trusted its input: clockwise or concave points gave
//! inward normals, interior and duplicate points stayed, more than 8 vertices were accepted and
//! fewer than 3 panicked. Box2D computes a quickhull (`b2ComputeHull`) that welds points closer
//! than half a linear slop, drops interior and collinear points, keeps at most 8 and starts from
//! the point farthest from the centre of the bounds; polygons are made from hulls and keep their
//! centroid, and boxes can be offset and rounded. The vertex order decides contact feature ids,
//! so a polygon from the same points must be Box2D's to behave like it. Expected values are
//! Box2D's, from `docs/audits/probes/box2d_reference/fix_2_6_hulls.c`, with rotations from the
//! exact sine and cosine as `Rot2::from_angle` (Box2D's `b2MakeRot` approximates them).

use korppi_math::{Rot2, Transform2d, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::hull::Hull2d;
use korppi_physics2d::shape::{Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d, WorldConfig2d};

const DT: f32 = 1.0 / 60.0;

/// A unit hexagon given clockwise from mid-way, with an interior point and a point 1 mm from a
/// vertex.
const SCRAMBLED_HEXAGON: [Vec2; 8] = [
    Vec2::new(0.5, -0.866_025),
    Vec2::new(-0.5, -0.866_025),
    Vec2::new(0.1, 0.2),
    Vec2::new(-1.0, 0.0),
    Vec2::new(-0.5, 0.866_025),
    Vec2::new(0.5, 0.866_025),
    Vec2::new(1.0, 0.0),
    Vec2::new(1.0, 0.001),
];

fn assert_points(actual: &[Vec2], expected: &[(f32, f32)], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: {actual:?}");
    for (a, &(x, y)) in actual.iter().zip(expected) {
        assert!(
            (a.x - x).abs() < 2e-6 && (a.y - y).abs() < 2e-6,
            "{what}: {actual:?}, Box2D {expected:?}"
        );
    }
}

fn assert_near(actual: f32, expected: f32, tolerance: f32, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: {actual}, Box2D {expected}"
    );
}

#[test]
fn hulls_are_box2d_quickhulls() {
    // Box2D v3.2: the hexagon's six vertices counter-clockwise from (-1, 0), without the
    // interior or the welded point.
    let hull = Hull2d::compute(&SCRAMBLED_HEXAGON).expect("a hexagon");
    assert_points(
        hull.points(),
        &[
            (-1.0, 0.0),
            (-0.5, -0.866_025),
            (0.5, -0.866_025),
            (1.0, 0.0),
            (0.5, 0.866_025),
            (-0.5, 0.866_025),
        ],
        "hexagon",
    );

    // The midpoints of a square's edges are collinear and go.
    let midpoints = [
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
    ]
    .map(|(x, y)| Vec2::new(x, y));
    let hull = Hull2d::compute(&midpoints).expect("a square");
    assert_points(
        hull.points(),
        &[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        "midpoints",
    );

    // A point 4 mm outside an edge is within two linear slops of it and goes too.
    let bump = [(0.0, 0.0), (2.0, 0.0), (1.0, -0.004), (1.0, 2.0)].map(|(x, y)| Vec2::new(x, y));
    let hull = Hull2d::compute(&bump).expect("a triangle");
    assert_points(hull.points(), &[(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)], "bump");

    // Box2D fails on nine points, two points, collinear points and points within a weld.
    let nine: Vec<Vec2> = (0..9)
        .map(|i| {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / 9.0;
            Vec2::new(angle.cos(), angle.sin())
        })
        .collect();
    assert!(Hull2d::compute(&nine).is_none());
    assert!(Hull2d::compute(&SCRAMBLED_HEXAGON[..2]).is_none());
    let line = [Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)];
    assert!(Hull2d::compute(&line).is_none());
    let dot = [Vec2::ZERO, Vec2::new(0.001, 0.0), Vec2::new(0.0, 0.001)];
    assert!(Hull2d::compute(&dot).is_none());
    assert!(Polygon2d::from_vertices(&line, 0.0).is_none());
}

#[test]
fn hulls_are_validated_like_box2d() {
    // Box2D v3.2 `b2ValidateHull`: only a convex, counter-clockwise hull of 3 to 8 points
    // without collinear points.
    let points = |p: &[(f32, f32)]| p.iter().map(|&(x, y)| Vec2::new(x, y)).collect::<Vec<_>>();
    let ccw = points(&[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]);
    let cw = points(&[(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)]);
    let collinear = points(&[
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ]);
    let concave = points(&[
        (-1.0, -1.0),
        (1.0, -1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ]);
    let two = points(&[(-1.0, -1.0), (1.0, -1.0)]);
    assert!(Hull2d::validate(&ccw));
    for (name, hull) in [
        ("cw", &cw),
        ("collinear", &collinear),
        ("concave", &concave),
        ("two", &two),
    ] {
        assert!(!Hull2d::validate(hull), "{name} is not a hull");
        assert!(Hull2d::new(hull).is_none(), "{name} is not a hull");
    }
    // A valid hull keeps its points in the given order.
    assert_eq!(Hull2d::new(&ccw).unwrap().points(), &ccw[..]);
}

#[test]
fn polygons_keep_box2d_normals_and_centroids() {
    let hull = Hull2d::compute(&SCRAMBLED_HEXAGON).unwrap();

    // Box2D v3.2 `b2MakePolygon`: centroid (1e-7, 0), outward normals from the first edge, and
    // density 1 gives 2.598075 kg and 1.082530 kg m² about the centroid.
    let hexagon = Polygon2d::from_hull(&hull, 0.0);
    assert_points(&[hexagon.centroid()], &[(0.000_000_1, 0.0)], "centroid");
    assert_points(
        &[hexagon.normals()[0], hexagon.normals()[5]],
        &[(-0.866_025, -0.5), (-0.866_025, 0.5)],
        "normals",
    );
    assert_eq!(
        Polygon2d::from_vertices(&SCRAMBLED_HEXAGON, 0.0),
        Some(hexagon)
    );
    let mass = hexagon.compute_mass_properties(1.0);
    assert_near(mass.mass, 2.598_075, 1e-5, "hexagon mass");
    assert_near(mass.rotational_inertia, 1.082_53, 1e-5, "hexagon inertia");

    // `b2MakeOffsetPolygon` at (2, 1), turned 0.5 rad: the centroid is the offset.
    let offset =
        Polygon2d::from_offset_hull(&hull, Vec2::new(2.0, 1.0), Rot2::from_angle(0.5), 0.0);
    assert_points(
        &[offset.vertices()[0], offset.normals()[0], offset.centroid()],
        &[(1.122_417, 0.520_574), (-0.520_296, -0.853_986), (2.0, 1.0)],
        "offset polygon",
    );

    // `b2MakeOffsetBox(0.5, 0.25, (1, 0.5), 0.3 rad)`; at density 2 it weighs 1 kg with
    // 0.104167 kg m² about its centre, and rounded by 0.1 m, 1.678810 kg and 0.269842 kg m².
    let offset_box = Polygon2d::offset_box(0.5, 0.25, Vec2::new(1.0, 0.5), Rot2::from_angle(0.3));
    assert_points(
        &[
            offset_box.vertices()[0],
            offset_box.vertices()[2],
            offset_box.normals()[1],
            offset_box.centroid(),
        ],
        &[
            (0.596_212, 0.113_406),
            (1.403_788, 0.886_594),
            (0.955_337, 0.295_520),
            (1.0, 0.5),
        ],
        "offset box",
    );
    let mass = offset_box.compute_mass_properties(2.0);
    assert_near(mass.mass, 1.0, 1e-5, "box mass");
    assert_points(&[mass.center], &[(1.0, 0.5)], "box mass centre");
    assert_near(mass.rotational_inertia, 0.104_167, 1e-5, "box inertia");
    let rounded =
        Polygon2d::offset_rounded_box(0.5, 0.25, Vec2::new(1.0, 0.5), Rot2::from_angle(0.3), 0.1);
    assert_eq!(rounded.radius, 0.1);
    let mass = rounded.compute_mass_properties(2.0);
    assert_near(mass.mass, 1.678_81, 1e-5, "rounded box mass");
    assert_near(
        mass.rotational_inertia,
        0.269_842,
        1e-5,
        "rounded box inertia",
    );
    assert_eq!(Polygon2d::rounded_box(0.5, 0.25, 0.1).radius, 0.1);

    // `b2TransformPolygon` of a half-metre square by (3, -1) and 1 rad.
    let square = Polygon2d::square(0.5).transformed(&Transform2d::from_translation_rotation(
        Vec2::new(3.0, -1.0),
        Rot2::from_angle(1.0),
    ));
    assert_points(
        &[square.vertices()[1], square.normals()[1], square.centroid()],
        &[(3.690_886, -0.849_416), (0.540_302, 0.841_471), (3.0, -1.0)],
        "transformed square",
    );
}

fn world_with_ground() -> PhysicsWorld2d {
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(WorldConfig2d {
            enable_sleep: false,
            ..WorldConfig2d::default()
        });
    let ground =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(0.6),
    );
    world
}

fn add_polygon(world: &mut PhysicsWorld2d, body: BodyId2d, polygon: Polygon2d) {
    world.create_collider(
        body,
        Collider2d::new(Shape2d::Polygon(polygon)).with_friction(0.6),
    );
}

#[test]
fn a_hammer_of_offset_boxes_lands_like_box2d() {
    // A square head and a handle offset by 1 m and turned 0.5 rad, on one body dropped from
    // y = 2. Box2D v3.2: 0.65 kg, centre of mass (0.615385, 0), 0.298930 kg m²; after 3 s it
    // lies at x = 0.35091, y = 0.57933, angle -0.49999.
    let mut world = world_with_ground();
    let hammer =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 2.0)));
    add_polygon(&mut world, hammer, Polygon2d::square(0.25));
    add_polygon(
        &mut world,
        hammer,
        Polygon2d::offset_box(1.0, 0.1, Vec2::new(1.0, 0.0), Rot2::from_angle(0.5)),
    );
    let body = world.body(hammer).unwrap();
    assert_near(body.mass, 0.65, 1e-6, "mass");
    assert_points(&[body.local_center], &[(0.615_385, 0.0)], "centre of mass");
    assert_near(body.rotational_inertia, 0.298_930, 1e-5, "inertia");
    for _ in 0..180 {
        world.step(DT);
    }
    let body = world.body(hammer).unwrap();
    assert_near(body.position.x, 0.350_91, 1e-3, "x");
    assert_near(body.position.y, 0.579_33, 1e-3, "y");
    assert_near(body.rotation.angle(), -0.499_99, 1e-3, "angle");
}

#[test]
fn a_polygon_from_scrambled_points_rolls_like_box2d() {
    // The scrambled hexagon thrown at 2 m/s spinning at -3 rad/s: Box2D v3.2 has it roll and
    // settle at x = 1.59575, y = 0.87515, angle -2.07648 after 2 s.
    let mut world = world_with_ground();
    let body = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(0.0, 2.0))
            .with_linear_velocity(Vec2::new(2.0, 0.0))
            .with_angular_velocity(-3.0),
    );
    add_polygon(
        &mut world,
        body,
        Polygon2d::from_vertices(&SCRAMBLED_HEXAGON, 0.0).unwrap(),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    let hexagon = world.body(body).unwrap();
    assert_near(hexagon.position.x, 1.595_75, 1e-3, "x");
    assert_near(hexagon.position.y, 0.875_15, 1e-3, "y");
    assert_near(hexagon.rotation.angle(), -2.076_48, 1e-3, "angle");
}

#[test]
fn polygons_deserialize_only_as_hulls() {
    // A polygon round-trips exactly; vertices that are not a hull, or more than eight, are
    // rejected instead of making a polygon the narrowphase cannot use.
    let polygon = Polygon2d::offset_box(0.5, 0.25, Vec2::new(1.0, 0.5), Rot2::from_angle(0.3));
    let json = serde_json::to_string(&polygon).unwrap();
    assert_eq!(serde_json::from_str::<Polygon2d>(&json).unwrap(), polygon);

    let clockwise = r#"{"vertices":[[-1,-1],[-1,1],[1,1],[1,-1]],
        "normals":[[-1,0],[0,1],[1,0],[0,-1]],"centroid":[0,0],"radius":0}"#;
    assert!(serde_json::from_str::<Polygon2d>(clockwise).is_err());
    let nine: Vec<Vec2> = (0..9)
        .map(|i| {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / 9.0;
            Vec2::new(angle.cos(), angle.sin())
        })
        .collect();
    let json = serde_json::json!({
        "vertices": nine.iter().map(|v| [v.x, v.y]).collect::<Vec<_>>(),
        "normals": nine.iter().map(|v| [v.x, v.y]).collect::<Vec<_>>(),
        "centroid": [0.0, 0.0],
        "radius": 0.0,
    });
    assert!(serde_json::from_value::<Polygon2d>(json).is_err());
}
