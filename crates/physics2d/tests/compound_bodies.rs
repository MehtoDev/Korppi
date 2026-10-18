//! A body can hold several colliders (P2D-08).
//!
//! Before FIX-2.4 a body held one collider. As in Box2D v3.2, each collider now has its own
//! broadphase proxy and contacts, colliders of one body never collide with each other, and the
//! body's mass, centre of mass and inertia come from all of its colliders. Expected values are
//! Box2D's, from `docs/audits/probes/box2d_reference/fix_2_4_compound.c`.

use korppi_math::{Rot2, Vec2};
use korppi_physics2d::body::{BodyType, RigidBody2d};
use korppi_physics2d::collider::Collider2d;
use korppi_physics2d::shape::{Circle2d, Polygon2d, Shape2d};
use korppi_physics2d::world::{BodyId2d, PhysicsWorld2d};

const DT: f32 = 1.0 / 60.0;

fn run(world: &mut PhysicsWorld2d, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

/// A box with half extents `hx`, `hy` centred on `center` in body coordinates.
fn offset_box(hx: f32, hy: f32, center: Vec2) -> Shape2d {
    Shape2d::Polygon(Polygon2d::offset_box(hx, hy, center, Rot2::IDENTITY))
}

fn add_ground(world: &mut PhysicsWorld2d, friction: f32) -> BodyId2d {
    let id =
        world.create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(0.0, -0.5)));
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(friction),
    );
    id
}

#[test]
fn mass_comes_from_every_collider() {
    // Box2D v3.2: an L of a 2 x 0.5 bar and a 0.5 x 1 post (density 2) with a circular knob
    // (r 0.25, density 2): 2.392699 kg, centre (0.149330, 0.395515), inertia 1.658828 about it.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic));
    world.create_collider(id, Collider2d::new(offset_box(1.0, 0.25, Vec2::ZERO)));
    world.create_collider(
        id,
        Collider2d::new(offset_box(0.25, 0.5, Vec2::new(0.75, 0.75))).with_density(2.0),
    );
    world.create_collider(
        id,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::new(-1.0, 0.5), 0.25)))
            .with_density(2.0),
    );
    assert_eq!(world.colliders_of(id).count(), 3);
    let body = world.body(id).unwrap();
    assert!((body.mass - 2.392_699).abs() < 1e-4, "mass {}", body.mass);
    assert!(
        (body.local_center - Vec2::new(0.149_330, 0.395_515)).length() < 1e-4,
        "centre {:?}",
        body.local_center
    );
    assert!(
        (body.rotational_inertia - 1.658_828).abs() < 1e-4,
        "inertia {}",
        body.rotational_inertia
    );
}

#[test]
fn a_table_stands_on_its_two_legs() {
    // A top and two legs on one body, dropped onto the ground. Box2D v3.2 after 2 s: level at
    // y = 0.09996, touching the ground with two contacts, one per leg.
    let mut world = PhysicsWorld2d::new();
    let floor = add_ground(&mut world, 0.6);
    let table =
        world.create_body(RigidBody2d::new(BodyType::Dynamic).with_position(Vec2::new(0.0, 1.2)));
    for (hx, hy, center) in [
        (1.5, 0.1, Vec2::new(0.0, 0.9)),
        (0.1, 0.45, Vec2::new(-1.3, 0.35)),
        (0.1, 0.45, Vec2::new(1.3, 0.35)),
    ] {
        world.create_collider(
            table,
            Collider2d::new(offset_box(hx, hy, center)).with_friction(0.6),
        );
    }
    run(&mut world, 2.0);
    let body = world.body(table).unwrap();
    assert!(
        (body.position.y - 0.09996).abs() < 0.002,
        "the table rests at y = {:.5}",
        body.position.y
    );
    assert!(
        body.rotation.angle().abs() < 0.001,
        "the table tilted to {:.5}",
        body.rotation.angle()
    );
    let touching = world
        .manifolds()
        .iter()
        .filter(|(a, b, m)| {
            m.point_count > 0 && ([*a, *b] == [floor, table] || [*b, *a] == [floor, table])
        })
        .count();
    assert_eq!(touching, 2, "one contact per leg");
}

#[test]
fn a_dumbbell_slides_down_a_ramp() {
    // Two circles on one body on a 20° ramp with friction 0.2. Box2D v3.2: 3.0552 m down the
    // ramp in 2 s, without leaving it.
    let slope = 20.0_f32.to_radians();
    let q = Rot2::from_angle(slope);
    let mut world = PhysicsWorld2d::new();
    let ramp = world.create_body(RigidBody2d::new(BodyType::Static).with_rotation(q));
    world.create_collider(
        ramp,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(0.2),
    );
    let start = q.rotate(Vec2::new(5.0, 0.8));
    let bell = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(start)
            .with_rotation(q),
    );
    for x in [-0.6, 0.6] {
        world.create_collider(
            bell,
            Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::new(x, 0.0), 0.3)))
                .with_friction(0.2),
        );
    }
    run(&mut world, 2.0);
    let along = q.inv_rotate(world.body(bell).unwrap().position - start);
    assert!(
        (along.x + 3.0552).abs() < 0.03,
        "the dumbbell slid {:.4} m",
        along.x
    );
    assert!(
        along.y.abs() < 0.01,
        "the dumbbell left the ramp by {:.4} m",
        along.y
    );
}

#[test]
fn colliders_of_one_body_do_not_collide() {
    // Box2D v3.2: two overlapping boxes on one body stay at rest, with no contact.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic));
    world.create_collider(id, Collider2d::new(offset_box(0.5, 0.5, Vec2::ZERO)));
    world.create_collider(
        id,
        Collider2d::new(offset_box(0.5, 0.5, Vec2::new(0.3, 0.2))),
    );
    run(&mut world, 1.0);
    assert_eq!(world.body(id).unwrap().linear_velocity, Vec2::ZERO);
    assert!(world.manifolds().is_empty());
}

#[test]
fn destroying_one_collider_keeps_the_other() {
    // Box2D v3.2: two unit boxes at x = -1 and 1 weigh 2 kg; without the first, 1 kg centred at
    // x = 1 with inertia 1/6.
    let mut world = PhysicsWorld2d::new().with_gravity(Vec2::ZERO);
    let id = world.create_body(RigidBody2d::new(BodyType::Dynamic));
    let left = world.create_collider(
        id,
        Collider2d::new(offset_box(0.5, 0.5, Vec2::new(-1.0, 0.0))),
    );
    let right = world.create_collider(
        id,
        Collider2d::new(offset_box(0.5, 0.5, Vec2::new(1.0, 0.0))),
    );
    assert_eq!(world.body(id).unwrap().mass, 2.0);
    assert!(world.destroy_collider(left));
    assert_eq!(world.colliders_of(id).collect::<Vec<_>>(), vec![right]);
    let body = world.body(id).unwrap();
    assert_eq!(body.mass, 1.0);
    assert!((body.local_center - Vec2::new(1.0, 0.0)).length() < 1e-6);
    assert!((body.rotational_inertia - 1.0 / 6.0).abs() < 1e-5);
}

#[test]
fn a_body_with_its_centre_of_mass_off_its_origin_follows_box2d_bit_for_bit() {
    // As Box2D's `b2BodySim`, the world keeps each body's centre of mass apart from its origin
    // and moves the centre by the step's motion, rebuilding the origin from it, so rounding does
    // not drift the centre (`fix_2_review.c`): an offset, turned box and an offset circle,
    // spinning as they fall 1234.5 m from the world origin, after 3 s. Far from the origin,
    // rebuilding the centre from the origin each step would round it.
    let mut world = PhysicsWorld2d::new()
        .with_gravity(Vec2::new(0.0, -10.0))
        .with_config(korppi_physics2d::world::WorldConfig2d {
            enable_sleep: false,
            ..Default::default()
        });
    let ground = world
        .create_body(RigidBody2d::new(BodyType::Static).with_position(Vec2::new(1234.5, -0.5)));
    world.create_collider(
        ground,
        Collider2d::new(Shape2d::Polygon(Polygon2d::from_box(20.0, 0.5))).with_friction(0.6),
    );
    let body = world.create_body(
        RigidBody2d::new(BodyType::Dynamic)
            .with_position(Vec2::new(1234.5, 2.0))
            .with_rotation(Rot2::from_angle(0.4))
            .with_angular_velocity(2.0),
    );
    world.create_collider(
        body,
        Collider2d::new(Shape2d::Polygon(Polygon2d::offset_box(
            0.5,
            0.25,
            Vec2::new(0.3, 0.1),
            Rot2::from_angle(0.2),
        )))
        .with_friction(0.6),
    );
    world.create_collider(
        body,
        Collider2d::new(Shape2d::Circle(Circle2d::new(Vec2::new(-0.4, 0.0), 0.2)))
            .with_friction(0.6),
    );
    let center = world.body(body).unwrap().local_center;
    assert_eq!((center.x, center.y), (0.159_405_95, 0.079_915_16));
    for _ in 0..180 {
        world.step(1.0 / 60.0);
    }
    let body = world.body(body).unwrap();
    let bits = [
        body.position.x,
        body.position.y,
        body.rotation.c,
        body.rotation.s,
    ]
    .map(f32::to_bits);
    assert_eq!(bits, [0x449a_39aa, 0x3e93_9f4c, 0xbf7a_e5bf, 0x3e4b_6e10]);
}
