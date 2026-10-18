//! Collision filters decide which colliders collide and which ones queries see (P3D-14).
//!
//! Box3D's `b3Filter` gives each shape 64 category bits, 64 mask bits and a group index: shapes
//! sharing a non-zero group always collide if it is positive and never if it is negative;
//! otherwise each one's mask must accept the other's category. Queries take a `b3QueryFilter`,
//! and sensors filter their visitors the same way. Before FIX-3.1 every collider collided with
//! every other one and ray casts saw everything.
//!
//! Expectations come from Box3D v0.1 (`docs/audits/probes/box3d_reference/fix_3_1_filters.c`):
//! 60 Hz, 4 substeps, 1 kg boxes, friction 0.6, no damping and no sleep.

use korppi_math::{Aabb, Ray, Vec3};
use korppi_physics::{
    BodyId3d, BodyType, Collider, Filter3d, PhysicsWorld, QueryFilter3d, RigidBody,
    SensorBeginEvent3d, Shape,
};

const DT: f32 = 1.0 / 60.0;
const GROUND: u64 = 0x1;
const BOX: u64 = 0x2;
const DEBRIS: u64 = 0x4;

fn add_box(
    world: &mut PhysicsWorld,
    body_type: BodyType,
    position: Vec3,
    half_extents: Vec3,
    filter: Filter3d,
) -> BodyId3d {
    let mut body = RigidBody::new(body_type, position).with_enable_sleep(false);
    body.linear_damping = 0.0;
    body.angular_damping = 0.0;
    let id = world.create_body(body);
    world.create_collider(
        id,
        Collider::new(Shape::Box { half_extents })
            .with_friction(0.6)
            .with_filter(filter)
            .with_mass(1.0),
    );
    id
}

fn ground(world: &mut PhysicsWorld, filter: Filter3d) -> BodyId3d {
    add_box(
        world,
        BodyType::Static,
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(5.0, 0.5, 5.0),
        filter,
    )
}

fn run(world: &mut PhysicsWorld, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

/// A 0.5 m box dropped from y = 2 onto the ground; its height after 1 s.
fn drop_on_ground(ground_filter: Filter3d, box_filter: Filter3d) -> f32 {
    let mut world = PhysicsWorld::new();
    ground(&mut world, ground_filter);
    let dropped = add_box(
        &mut world,
        BodyType::Dynamic,
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::splat(0.5),
        box_filter,
    );
    run(&mut world, 1.0);
    world.body(dropped).unwrap().position.y
}

/// Box B dropped onto box A, which rests on the ground; B's height after 2 s.
fn stack(filter_a: Filter3d, filter_b: Filter3d) -> f32 {
    let mut world = PhysicsWorld::new();
    ground(&mut world, Filter3d::DEFAULT);
    add_box(
        &mut world,
        BodyType::Dynamic,
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        filter_a,
    );
    let top = add_box(
        &mut world,
        BodyType::Dynamic,
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::splat(0.5),
        filter_b,
    );
    run(&mut world, 2.0);
    world.body(top).unwrap().position.y
}

const RESTING: std::ops::RangeInclusive<f32> = 0.49..=0.51;
const STACKED: std::ops::RangeInclusive<f32> = 1.49..=1.51;

#[test]
fn default_filter_collides_with_everything() {
    // Box3D: the default filter has every category and mask bit set; the box rests at 0.49993.
    assert_eq!(Filter3d::default(), Filter3d::DEFAULT);
    assert_eq!(
        (Filter3d::DEFAULT.category_bits, Filter3d::DEFAULT.mask_bits),
        (u64::MAX, u64::MAX)
    );
    let y = drop_on_ground(Filter3d::DEFAULT, Filter3d::DEFAULT);
    assert!(
        RESTING.contains(&y),
        "the box should rest on the ground, it is at {y}"
    );
}

#[test]
fn a_mask_without_the_other_category_does_not_collide() {
    // Box3D: the box falls through to y = -2.92544, whichever side excludes the other.
    let ground_filter = Filter3d::new(GROUND, u64::MAX);
    let y = drop_on_ground(ground_filter, Filter3d::new(BOX, !GROUND));
    assert!(
        y < -1.0,
        "a box whose mask excludes the ground should fall through, it is at {y}"
    );
    let y = drop_on_ground(Filter3d::new(GROUND, !BOX), Filter3d::new(BOX, u64::MAX));
    assert!(
        y < -1.0,
        "a ground whose mask excludes the box should let it through, it is at {y}"
    );
}

#[test]
fn groups_override_the_masks() {
    // Box3D: boxes in different groups stack (1.49931); a shared negative group falls through
    // (0.49993, on the ground); a shared positive group stacks although the masks exclude each
    // other (1.49931); different positive groups leave it to the masks (0.49993).
    let boxes = |mask, group| Filter3d::new(BOX, mask).with_group(group);
    let cases = [
        (
            "different negative groups",
            boxes(u64::MAX, -1),
            boxes(u64::MAX, -2),
            STACKED,
        ),
        (
            "the same negative group",
            boxes(u64::MAX, -1),
            boxes(u64::MAX, -1),
            RESTING,
        ),
        (
            "the same positive group",
            boxes(!BOX, 3),
            boxes(!BOX, 3),
            STACKED,
        ),
        (
            "different positive groups",
            boxes(!BOX, 3),
            boxes(!BOX, 4),
            RESTING,
        ),
    ];
    for (what, a, b, expected) in cases {
        let y = stack(a, b);
        assert!(
            expected.contains(&y),
            "boxes in {what}: the top box should end at {expected:?}, it is at {y}"
        );
    }
}

#[test]
fn query_filter_skips_excluded_categories() {
    let mut world = PhysicsWorld::new();
    let floor = ground(&mut world, Filter3d::new(GROUND, u64::MAX));
    let debris = world.create_body(RigidBody::new(BodyType::Static, Vec3::new(0.0, 2.0, 0.0)));
    world.create_collider(
        debris,
        Collider::new(Shape::Box {
            half_extents: Vec3::splat(0.5),
        })
        .with_sensor(true)
        .with_filter(Filter3d::new(DEBRIS, u64::MAX))
        .with_mass(1.0),
    );
    let down = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y);
    let solid = QueryFilter3d::new(u64::MAX, GROUND);

    // Box3D: the default query hits the sensor's top at y = 2.5, a query for the ground's
    // category hits the ground at y = 0.
    let hit = world
        .ray_cast(&down, 10.0)
        .expect("the ray hits the debris");
    assert_eq!(hit.body_id, debris);
    assert!((hit.point.y - 2.5).abs() < 0.01);
    let hit = world
        .ray_cast_filtered(&down, 10.0, &solid)
        .expect("the filtered ray hits the ground");
    assert_eq!(hit.body_id, floor);
    assert!(hit.point.y.abs() < 0.01);

    let around = Aabb {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 3.0, 1.0),
    };
    let sorted = |mut hits: Vec<BodyId3d>| {
        hits.sort_unstable();
        hits
    };
    assert_eq!(sorted(world.query_aabb(&around)), [floor, debris]);
    assert_eq!(world.query_aabb_filtered(&around, &solid), [floor]);
    // A query must also be in a category the collider's mask accepts (Box3D
    // `b3ShouldQueryCollide`).
    let ground_only = Filter3d::new(GROUND, GROUND);
    assert!(ground_only.should_query(&QueryFilter3d::new(GROUND, u64::MAX)));
    assert!(!ground_only.should_query(&QueryFilter3d::new(BOX, u64::MAX)));
    assert!(!ground_only.should_query(&QueryFilter3d::new(u64::MAX, BOX)));
}

#[test]
fn sensors_filter_their_visitors() {
    // Box3D: a sensor reports a box its mask accepts and skips one it excludes.
    for (mask, expected) in [(u64::MAX, 1), (!BOX, 0)] {
        let mut world = PhysicsWorld::new();
        let sensor = world.create_body(RigidBody::new(BodyType::Static, Vec3::ZERO));
        let sensor_col = world.create_collider(
            sensor,
            Collider::new(Shape::Box {
                half_extents: Vec3::ONE,
            })
            .with_sensor(true)
            .with_sensor_events(true)
            .with_filter(Filter3d::new(GROUND, mask))
            .with_mass(1.0),
        );
        let visitor = world.create_body(RigidBody::new(
            BodyType::Kinematic,
            Vec3::new(0.5, 0.0, 0.0),
        ));
        let visitor_col = world.create_collider(
            visitor,
            Collider::new(Shape::Box {
                half_extents: Vec3::splat(0.25),
            })
            .with_sensor_events(true)
            .with_filter(Filter3d::new(BOX, u64::MAX))
            .with_mass(1.0),
        );
        world.step(DT);
        let begins = &world.sensor_events().begin;
        assert_eq!(begins.len(), expected, "sensor mask {mask:#x}: {begins:?}");
        if expected == 1 {
            assert_eq!(
                begins[0],
                SensorBeginEvent3d {
                    sensor: sensor_col,
                    visitor: visitor_col,
                }
            );
        }
    }
}
